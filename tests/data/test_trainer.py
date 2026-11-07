import json
from dataclasses import replace
from pathlib import Path

import pytest

from nuzline.data.trainer import (
    AreaCondition,
    IntermediateData,
    PartnerVariant,
    build_trainer_party,
    default_moves,
    difficulty_to_ivs,
    lcrng_next,
    load_trainers,
    trainer_pid,
)
from tests.engine.helpers import roark_trainer

INTER = Path(__file__).resolve().parents[2] / "data" / "intermediate"
needs_inter = pytest.mark.skipif(not INTER.exists(), reason="intermediate data not extracted")


def test_difficulty_to_ivs_exact_integer():
    assert (difficulty_to_ivs(255), difficulty_to_ivs(0), difficulty_to_ivs(128)) == (31, 0, 15)


def test_lcrng_matches_decomp():
    # LCRNG_Next (src/math_util.c): state = state * 1103515245 + 24691 (u32); returns state >> 16.
    assert lcrng_next(0) == (24691, 0)
    assert lcrng_next(1) == (1103539936, 16838)
    state, _ = lcrng_next(0xFFFFFFFF)
    assert 0 <= state < 2**32


def test_trainer_pid_follows_party_build():
    # TrainerData_BuildParty (src/trainer_data.c): seed = iv_scale + level + species +
    # trainer id; LCRNG_Next once per unit of the class id; PID = (rnd << 8) + 120 for a
    # female class, else 136.
    state = rnd = 50 + 12 + 74 + 246
    for _ in range(62):
        state, rnd = lcrng_next(state)
    assert trainer_pid(50, 12, 74, 246, 62, female_class=False) == (rnd << 8) + 136
    # Class id 0 draws nothing, so the seed itself is shifted.
    assert trainer_pid(1, 2, 3, 4, 0, female_class=True) == (10 << 8) + 120


def _move_ids():
    moves = json.loads((INTER / "moves.json").read_text(encoding="utf-8"))
    return {m["name"]: i for i, m in enumerate(moves)}


def _species_id(inter, name):
    return next(i for i, s in enumerate(inter.species) if s["name"] == name)


@needs_inter
def test_default_moves_skip_known_and_push_out_oldest():
    # BoxPokemon_SetDefaultMoves (src/pokemon.c:3819): a move already known is skipped;
    # with four slots full the oldest is pushed out. Monferno's L1 and L7 Ember count
    # once, so at L14 Mach Punch pushes out Scratch.
    inter, m = IntermediateData(INTER), _move_ids()
    monferno = inter.learnsets[_species_id(inter, "MONFERNO")]
    assert default_moves(monferno, 14) == tuple(
        m[x] for x in ("Leer", "Ember", "Taunt", "Mach Punch")
    )
    starly = inter.learnsets[_species_id(inter, "STARLY")]
    assert default_moves(starly, 5) == (m["Tackle"], m["Growl"], m["Quick Attack"], 0)


def _roark():
    return next(t for t in load_trainers(INTER) if t.name.lower() == "roark")


@needs_inter
def test_party_ivs_follow_difficulty():
    inter = IntermediateData(INTER)
    roark = _roark()
    party = build_trainer_party(roark, inter)
    for mon, raw in zip(party, roark.party, strict=True):
        assert mon.ivs == (raw.difficulty * 31 // 255,) * 6
        assert mon.evs == (0,) * 6


@needs_inter
def test_roark_listed_moves_fill_all_four_slots():
    # The plan asserted four non-empty moves per mon. The decomp disagrees:
    # TrainerData_BuildParty (src/trainer_data.c, TRDATATYPE_WITH_MOVES) writes all four
    # listed slots over the default moves, so Geodude keeps MOVE_NONE in slots 3 and 4.
    inter = IntermediateData(INTER)
    roark = _roark()
    party = build_trainer_party(roark, inter)
    assert [mon.moves for mon in party] == [raw.moves for raw in roark.party]
    assert party[0].moves[2:] == (0, 0)


@needs_inter
def test_unlisted_moves_use_the_default_moveset():
    inter = IntermediateData(INTER)
    tristan = load_trainers(INTER)[1]
    raw = tristan.party[0]
    assert raw.moves is None
    (mon,) = build_trainer_party(tristan, inter)
    assert mon.moves == default_moves(inter.learnsets[raw.species], raw.level)
    assert all(mv != 0 for mv in mon.moves[:2])


@needs_inter
def test_roark_party_identity():
    # TRAINER_CLASS_LEADER_ROARK (id 62) is a male class, so every PID's low byte is 136:
    # all three mons are male (gender ratios 127, 127, 31 are all <= 136) and PID bit 0 is
    # 0, so each takes ability slot 0. The nature is pid % 25 (Pokemon_GetNatureOf).
    inter = IntermediateData(INTER)
    roark = _roark()
    assert (roark.trainer_class, roark.class_id, roark.class_gender) == ("Leader", 62, 0)
    party = build_trainer_party(roark, inter)
    assert [mon.gender for mon in party] == [0, 0, 0]
    assert [mon.ability_slot for mon in party] == [0, 0, 0]
    # Pinned: Geodude Lax (9), Onix Bold (5), Cranidos Jolly (13).
    assert [mon.nature for mon in party] == [9, 5, 13]
    for mon in party:
        pid = trainer_pid(50, mon.level, mon.species, 246, 62, female_class=False)
        assert pid & 0xFF == 136
        assert mon.nature == pid % 25
        assert mon.friendship == inter.species[mon.species]["base_friendship"]
        assert (mon.item, mon.form, mon.current_hp, mon.status) == (0, 0, None, 0)


def test_trainers_without_a_format_are_refused(tmp_path):
    # An intermediate extracted before the battle format was recorded.
    (tmp_path / "trainers.json").write_text(json.dumps([{"id": 0}]), encoding="utf-8")
    with pytest.raises(ValueError, match="no battle format"):
        load_trainers(tmp_path)


def test_trainers_without_a_battle_structure_are_refused(tmp_path):
    # An intermediate extracted before two-trainer, tag and partner battles were recorded.
    row = {"id": 0, "format": "single", "opponent_2": None}
    (tmp_path / "trainers.json").write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="has no battle structure: re-run"):
        load_trainers(tmp_path)


@needs_inter
def test_partner_variants_load():
    grunt = next(t for t in load_trainers(INTER) if t.format == "multi" and len(t.partner) == 6)
    assert isinstance(grunt.partner[0], PartnerVariant) and grunt.opponent_2 is not None
    assert {p.player_gender for p in grunt.partner} == {0, 1}


@needs_inter
def test_trainer_form_uses_row(names):
    from dataclasses import replace

    inter = IntermediateData(INTER)
    roark = _roark()
    t = replace(roark, party=(replace(roark.party[0], species=487, form=1),))
    (mon,) = build_trainer_party(t, inter)
    assert mon.form == 1
    assert names.species_abilities(487, 1)[mon.ability_slot] == names.ability("Levitate")
    assert mon.friendship == names.species_base_friendship(487, 1)


@needs_inter
def test_trainer_form_default_moves_use_the_row_learnset():
    # Wormadam-Sandy's level-up learnset (row 499) parts from Plant Cloak's at level 26.
    from dataclasses import replace

    inter = IntermediateData(INTER)
    roark = _roark()
    raw = replace(roark.party[0], species=413, form=1, level=30, moves=None)
    (mon,) = build_trainer_party(replace(roark, party=(raw,)), inter)
    assert mon.moves == default_moves(inter.learnsets[499], 30)
    assert mon.moves != default_moves(inter.learnsets[413], 30)
    with pytest.raises(ValueError, match="species 413 has no form 3"):
        build_trainer_party(replace(roark, party=(replace(raw, form=3),)), inter)


# --- Kaizo trainer fields (K2b Task 14) ----------------------------------------------


def test_vanilla_trainers_load_with_defaults(trainers):  # data/intermediate trainers.json
    assert all(
        t.label is None
        and t.pid_routine == "vanilla"
        and not t.optional
        and t.gauntlet is None
        and t.sources == {}
        and t.gauntlet_rules == {}
        and t.player_starter is None
        and not t.left_side
        and t.split is None
        and t.split_order is None
        for t in trainers
    )
    assert all(
        m.nature is None
        and m.gender is None
        and m.ability_slot is None
        and m.ability_override == 0
        and m.friendship is None
        for t in trainers
        for m in t.party
    )
    assert trainers[246].area_condition == AreaCondition()  # vanilla Roark starts in no weather


def test_vanilla_area_condition_follows_start_condition(trainers):
    for t in trainers:
        area = t.area_condition
        if t.start_condition == "none":
            assert area == AreaCondition()
        elif t.start_condition == "fog":
            assert area == AreaCondition(fog=True)
        else:
            assert area == AreaCondition(weather=t.start_condition)
    assert {t.area_condition.weather for t in trainers} == {None, "rain", "hail", "sandstorm"}


def test_unknown_start_condition_is_not_a_weather():
    # Kaizo trainers whose start condition is a user call (318, 325, ... 747).
    assert AreaCondition.from_start_condition("unknown") == AreaCondition(weather="unknown")
    assert AreaCondition.from_start_condition("sun") == AreaCondition(weather="sun")
    with pytest.raises(ValueError, match="unknown start condition 'frost'"):
        AreaCondition.from_start_condition("frost")


def test_area_condition_kinds():
    # The sheet grammar: PERMANENT RAIN|SUN|SANDSTORM|HAIL, FOG, TRICK ROOM, GRAVITY.
    for w in ("rain", "sun", "sandstorm", "hail"):
        assert AreaCondition(weather=w).permanent
    assert AreaCondition(trick_room=True, gravity=True, fog=True).permanent
    with pytest.raises(ValueError, match="area weather 'fog'"):
        AreaCondition(weather="fog")


def _kaizo_row(**fields):
    mon = {"species": 74, "level": 12, "difficulty": 0, "moves": None, "item": 0, "form": 0}
    row = {
        "id": 7,
        "name": "Worker C",
        "class": "Worker",
        "class_id": 3,
        "class_gender": 0,
        "ai_flags": 7,
        "items": [0, 0, 0, 0],
        "party": [mon],
        "start_condition": "none",
        "format": "single",
        "opponent_2": None,
        "partner": [],
    }
    return {**row, **fields}


def _write(tmp_path, rows):
    (tmp_path / "trainers.json").write_text(json.dumps(rows), encoding="utf-8")
    return load_trainers(tmp_path)


def test_kaizo_fields_load(tmp_path):
    mon = {
        "species": 74,
        "level": 12,
        "difficulty": 0,
        "moves": None,
        "item": 0,
        "form": 0,
        "nature": 3,
        "gender": 1,
        "ability_slot": 1,
        "ability_override": 16,
        "friendship": 0,
    }
    row = _kaizo_row(
        party=[mon],
        label="Worker C (Oreburgh Gym)",
        pid_routine=None,
        player_starter=390,
        gauntlet=["Oreburgh Gym", 2],
        gauntlet_rules={"heal_between": True},
        optional=True,
        left_side=True,
        area_condition={
            "weather": "rain",
            "fog": False,
            "trick_room": True,
            "gravity": False,
            "permanent": True,
        },
        split="Roark Split",
        split_order=[1, 4],
        sources={"area_condition": "overrides.toml:12"},
    )
    (t,) = _write(tmp_path, [row])
    assert (t.label, t.pid_routine, t.player_starter) == ("Worker C (Oreburgh Gym)", None, 390)
    assert t.gauntlet == ("Oreburgh Gym", 2) and t.gauntlet_rules == {"heal_between": True}
    assert t.optional and t.left_side and t.split == "Roark Split" and t.split_order == (1, 4)
    assert t.area_condition == AreaCondition(weather="rain", trick_room=True)
    assert t.sources == {"area_condition": "overrides.toml:12"}
    m = t.party[0]
    assert (m.nature, m.gender, m.ability_slot, m.ability_override, m.friendship) == (
        3,
        1,
        1,
        16,
        0,
    )


def test_area_condition_from_start_condition_when_absent(tmp_path):
    (t,) = _write(tmp_path, [_kaizo_row(start_condition="unknown")])
    assert t.area_condition == AreaCondition(weather="unknown")


def test_explicit_nature_wins(inter):
    roark = roark_trainer()
    explicit = replace(roark.party[0], nature=3, gender=1, ability_slot=1)
    built = build_trainer_party(replace(roark, party=(explicit,)), inter)[0]
    assert (built.nature, built.gender, built.ability_slot) == (3, 1, 1)
    vanilla = build_trainer_party(roark, inter)[0]
    assert (vanilla.nature, vanilla.gender, vanilla.ability_slot) == (9, 0, 0)  # Lax Geodude


def test_each_explicit_value_wins_alone(inter):
    roark = roark_trainer()
    only_gender = replace(roark.party[0], gender=1)
    built = build_trainer_party(replace(roark, party=(only_gender,)), inter)[0]
    assert (built.nature, built.gender, built.ability_slot) == (9, 1, 0)


def test_explicit_friendship_wins(inter):
    # Kaizo's party builder sets 255, or 0 for a Pokemon with Frustration.
    roark = roark_trainer()
    for value in (0, 255):
        mon = replace(roark.party[0], friendship=value)
        assert build_trainer_party(replace(roark, party=(mon,)), inter)[0].friendship == value
    base = inter.species[roark.party[0].species]["base_friendship"]
    assert build_trainer_party(roark, inter)[0].friendship == base


def test_no_routine_without_data_raises(inter):
    t = replace(roark_trainer(), pid_routine=None)
    with pytest.raises(ValueError, match="trainer 246 slot 0: no nature/gender/ability data"):
        build_trainer_party(t, inter)
    partial = replace(t.party[0], nature=3, gender=0)
    with pytest.raises(ValueError, match="trainer 246 slot 0: no nature/gender/ability data"):
        build_trainer_party(replace(t, party=(partial,)), inter)


def test_no_routine_with_data_builds(inter):
    t = replace(roark_trainer(), pid_routine=None)
    party = tuple(replace(m, nature=3, gender=1, ability_slot=0) for m in t.party)
    built = build_trainer_party(replace(t, party=party), inter)
    assert [(m.nature, m.gender, m.ability_slot) for m in built] == [(3, 1, 0)] * len(party)


def test_ability_override_without_set_data_raises(inter):  # review I1
    roark = roark_trainer()
    assert roark.pid_routine == "vanilla"
    override = replace(roark.party[0], ability_override=16, nature=3, gender=0)
    with pytest.raises(ValueError, match="trainer 246 slot 0: no nature/gender/ability data"):
        build_trainer_party(replace(roark, party=(override,)), inter)
    full = replace(override, ability_slot=1)
    built = build_trainer_party(replace(roark, party=(full,)), inter)[0]
    assert (built.nature, built.gender, built.ability_slot) == (3, 0, 1)


@pytest.mark.parametrize("key", ["gauntlet", "split_order"])
def test_pair_fields_must_be_pairs(tmp_path, key):
    with pytest.raises(ValueError, match=f"trainer 7: {key} .* is not a pair"):
        _write(tmp_path, [_kaizo_row(**{key: ["Oreburgh Gym", 2, 3]})])


def test_vanilla_area_conditions_pinned(trainers):
    # Map weather (tools/extract_decomp.py _start_condition): Juan in rain, Keegan in a
    # sandstorm, Blake in hail, Ernest in fog.
    assert trainers[91].area_condition == AreaCondition(weather="rain")
    assert trainers[559].area_condition == AreaCondition(weather="sandstorm")
    assert trainers[132].area_condition == AreaCondition(weather="hail")
    assert trainers[66].area_condition == AreaCondition(fog=True)


def test_null_partner_loads_as_none(tmp_path):  # Kaizo: a partner present, not evaluated
    (t,) = _write(tmp_path, [_kaizo_row(format="multi", partner=None)])
    assert t.format == "multi" and t.partner is None
