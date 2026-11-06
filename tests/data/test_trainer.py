import json
from pathlib import Path

import pytest

from nuzline.data.trainer import (
    IntermediateData,
    PartnerVariant,
    build_trainer_party,
    default_moves,
    difficulty_to_ivs,
    lcrng_next,
    load_trainers,
    trainer_pid,
)

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
