"""The coverage registry refuses a fight that uses unimplemented mechanics and names
every one of them (spec §3)."""

from dataclasses import replace

import numpy as np
import pytest

import nuzline.engine.coverage as coverage_module
from nuzline.data.forms import personal_row
from nuzline.data.games import KAIZO
from nuzline.data.gauntlet import GauntletFacts
from nuzline.data.tables import Names
from nuzline.data.trainer import (
    TERRAIN_NAMES,
    TERRAIN_UNKNOWN,
    build_trainer_party,
    load_trainers,
)
from nuzline.engine.coverage import (
    IMPLEMENTED_ABILITIES,
    IMPLEMENTED_AI_FLAGS,
    IMPLEMENTED_EFFECTS,
    IMPLEMENTED_ITEMS,
    IMPLEMENTED_WEATHER,
    Support,
    UnsupportedMechanics,
    check_coverage,
    check_opponent,
    require_coverage,
    rom_change_refusals,
    start_weather,
    terrain_refusal,
    unknown_start,
)
from nuzline.engine.ids import AbilityId, ItemId, Weather
from tests.data.test_gauntlet_settings import facts_dict
from tests.engine.helpers import (
    INTERMEDIATE,
    _inter,
    gravity_of,
    kaizo_names_with,
    roark_party,
    roark_trainer,
    sample_party,
    trick_room_of,
)


def test_roark_vs_sample_party_is_fully_covered(tables, names):
    r = check_coverage(tables, names, sample_party(names), roark_trainer(), roark_party(names), 0)
    assert r.missing == [] and r.level == Support.EXACT


@pytest.mark.fullset
def test_all_missing_reported_at_once(tables, names, metronome_refused):
    # The held item is picked from the coverage table at run time: the first one still mapped
    # to ItemId.UNSUPPORTED.
    effects = np.asarray(tables.item_effect)
    refused = next(i for i in range(1, len(effects)) if effects[i] == ItemId.UNSUPPORTED)
    item = names.item_names[refused]
    party = sample_party(
        names,
        override_moves={0: ["Metronome", "Scratch", "Leer", "Ember"]},
        override_item={1: item},
    )
    with pytest.raises(UnsupportedMechanics) as e:
        require_coverage(tables, names, party, roark_trainer(), roark_party(names), 0)
    msg = str(e.value)
    assert "Metronome" in msg and item in msg
    assert e.value.report.level == Support.NONE


@pytest.mark.fullset
def test_every_weather_is_supported():
    # E2 batch 3: sun, rain and hail join sandstorm (coverage docstring); E1 Batch C: deep fog
    # (G-44).
    for w in Weather:
        assert IMPLEMENTED_WEATHER[w] == Support.EXACT


def test_hail_route_trainer_weather_is_supported(tables, names):
    hail = next(t for t in load_trainers(INTERMEDIATE) if t.start_condition == "hail")
    assert start_weather(hail) == Weather.HAIL
    opp = build_trainer_party(hail, _inter())
    r = check_coverage(tables, names, sample_party(names), hail, opp, Weather.HAIL)
    assert not any("Battle start: Hail" in m for m in r.missing)
    r = check_coverage(tables, names, sample_party(names), hail, opp, Weather.FOG)
    assert not any("Battle start" in m for m in r.missing)  # fog is modelled (G-44)


@pytest.mark.fullset
def test_double_battle_trainer_is_refused(tables, names):
    # TRAINER_TWINS_EMMA_AND_LIL (res/trainers/data/twins_emma_and_lil.json) is a
    # double battle; the engine models singles only.
    emma = next(t for t in load_trainers(INTERMEDIATE) if t.name == "Emma & Lil")
    assert emma.format == "double"
    opp = build_trainer_party(emma, _inter())
    with pytest.raises(UnsupportedMechanics) as e:
        require_coverage(tables, names, sample_party(names), emma, opp, 0)
    assert "Emma & Lil: a double battle (battle format UNSUPPORTED)" in e.value.report.missing
    theirs = check_opponent(tables, names, emma, opp, 0)
    assert "Emma & Lil: a double battle (battle format UNSUPPORTED)" in theirs.missing


def test_roark_is_a_single_battle(tables, names):
    assert roark_trainer().format == "single"
    r = check_opponent(tables, names, roark_trainer(), roark_party(names), 0)
    assert not any("battle format" in m for m in r.missing)
    unknown = replace(roark_trainer(), format="triple")
    r = check_opponent(tables, names, unknown, roark_party(names), 0)
    assert "Roark: a triple battle (battle format UNSUPPORTED)" in r.missing


def test_unknown_battle_format_is_refused_explicitly(tables, names):
    # Kaizo's Spear Pillar 521/527 at the ROM level (data/kaizo/battle_formats.json): an explicit
    # entry (the user's answer in overrides.toml then sets their format).
    from nuzline.engine.coverage import IMPLEMENTED_FORMATS

    assert IMPLEMENTED_FORMATS["unknown"] == Support.NONE
    t = replace(roark_trainer(), format="unknown")
    r = check_opponent(tables, names, t, roark_party(names), 0)
    msg = "Roark: battle format unknown (depends on game state) (battle format UNSUPPORTED)"
    assert msg in r.missing
    assert r.level == Support.NONE


@pytest.mark.parametrize(
    ("name", "fmt"), [("Mars", "multi"), ("Ismael", "double_two_trainer"), ("Jack", "multi")]
)
@pytest.mark.fullset
def test_multi_and_two_trainer_battles_are_refused(tables, names, name, fmt):
    # Mars @ Spear Pillar (tag), Artist Ismael (restaurant pair), Bug Catcher Jack (Eterna
    # Forest, Cheryl)
    t = next(t for t in load_trainers(INTERMEDIATE) if t.name == name and t.format == fmt)
    r = check_opponent(tables, names, t, build_trainer_party(t, _inter()), 0)
    assert f"{name}: a {fmt} battle (battle format UNSUPPORTED)" in r.missing
    assert r.level == Support.NONE


def test_partner_trainer_is_not_refused_for_its_format(tables, names):
    # Cheryl walks with the player (Ruling 3); her own row stays a single battle.
    cheryls = [t for t in load_trainers(INTERMEDIATE) if t.name == "Cheryl"]
    assert cheryls
    for cheryl in cheryls:
        assert (cheryl.format, cheryl.opponent_2, cheryl.partner) == ("single", None, ())
        r = check_opponent(tables, names, cheryl, build_trainer_party(cheryl, _inter()), 0)
        assert not any("battle format" in m for m in r.missing)


def test_trainer_items_are_modelled_and_a_mixed_bag_is_refused(tables, names):
    # E1 rulings G-43, G-62, F-15: every bag item acts by its party params (the friendship wrap
    # included, so even a Poke Ball is modelled); a bag of different items is refused.
    items = [names.item(n) for n in ("Super Potion", "Guard Spec.", "Poké Ball")]
    trainer = replace(roark_trainer(), items=tuple(items))
    r = check_coverage(tables, names, sample_party(names), trainer, roark_party(names), 0)
    for name in ("Super Potion", "Guard Spec.", "Poké Ball"):
        assert not any(f"{name} " in m and "trainer item" in m for m in r.missing)
    assert any("mixed items" in m for m in r.missing)


@pytest.mark.fullset
def test_ai_flags_outside_the_policy_are_refused(tables, names):
    # Vanilla's policy runs BASIC, EVAL_ATTACK and EXPERT only: a vanilla 0b10111 trainer is
    # still refused by its RISKY flag (K3c Task 6 registers RISKY for Kaizo only).
    risky = replace(roark_trainer(), ai_flags=0b10111)  # + AI_FLAG_RISKY
    r = check_coverage(tables, names, sample_party(names), risky, roark_party(names), 0)
    assert "Roark: AI_FLAG_RISKY (AI flag UNSUPPORTED)" in r.missing
    basic_only = replace(roark_trainer(), ai_flags=0b1)
    r = check_coverage(tables, names, sample_party(names), basic_only, roark_party(names), 0)
    assert any("AI flag set" in m for m in r.missing)


@pytest.mark.fullset
def test_ai_refusals_are_included(tables, names):
    """An opponent move whose AI script paths are not transcribed (ai.policy)."""
    opp = roark_party(names)
    opp[0] = replace(opp[0], moves=(names.move("Ember"), 0, 0, 0))
    r = check_coverage(tables, names, sample_party(names), roark_trainer(), opp, 0)
    assert any("Ember" in m and "trainer AI" in m for m in r.missing)


@pytest.mark.fullset
def test_sampling_only_is_unimplemented():
    # The spec's sampling-only grade (section 4.2) has no registry entry and no engine
    # path yet (coverage.py docstring): a registry that assigns it must come with that
    # path, so this guard fails first.
    registries = (
        IMPLEMENTED_EFFECTS,
        IMPLEMENTED_ABILITIES,
        IMPLEMENTED_ITEMS,
        *IMPLEMENTED_AI_FLAGS.values(),
        IMPLEMENTED_WEATHER,
    )
    assert {support for r in registries for support in r.values()} <= {Support.EXACT, Support.NONE}


@pytest.mark.parametrize("trainer_id", [246, 858])
def test_check_opponent_matches_check_coverage(tables, names, monkeypatch, trainer_id):
    trainer = next(t for t in load_trainers(INTERMEDIATE) if t.id == trainer_id)
    opp = build_trainer_party(trainer, _inter())
    full = check_coverage(tables, names, sample_party(names), trainer, opp, 0)
    expected = [
        m
        for m in full.missing
        if not m.startswith("Player ") and not m.endswith(" (trainer AI UNSUPPORTED)")
    ]

    def no_state(*a, **kw):
        raise AssertionError("check_opponent must not build a state")

    monkeypatch.setattr("nuzline.engine.coverage.make_state", no_state)
    theirs = check_opponent(tables, names, trainer, opp, 0)
    assert theirs.missing == expected


@pytest.mark.parametrize(
    ("species", "form", "item", "line"),
    [
        (
            487,
            1,
            "no item",
            "Giratina (slot 0): Origin Forme without the Griseous Orb (form change UNSUPPORTED)",
        ),
    ],
)
@pytest.mark.fullset
def test_in_battle_form_changes_are_refused(tables, names, species, form, item, line):  # Ruling 8
    # The brief writes the species title-cased; ``where`` names it as the data does
    # (``names.species_names``, upper case), so the line is compared with that name.
    line = line.replace(names.species_names[species].title(), names.species_names[species], 1)
    mon = replace(
        sample_party(names)[0],
        species=species,
        form=form,
        item=0 if item == "no item" else names.item(item),
    )
    r = check_coverage(tables, names, [mon], roark_trainer(), roark_party(names), Weather.NONE)
    assert any(m.endswith(line) for m in r.missing)
    # Either side: the same form on the trainer's side is refused too.
    r = check_coverage(
        tables, names, sample_party(names), roark_trainer(), [mon, *roark_party(names)[1:]], 0
    )
    assert any(m.endswith(line) for m in r.missing)
    ok = replace(mon, species=487, form=1, item=names.item("Griseous Orb"))
    r = check_coverage(tables, names, [ok], roark_trainer(), roark_party(names), Weather.NONE)
    assert not any("form change" in m for m in r.missing)


def test_coverage_reads_the_form_row_ability(tables, names):
    origin = replace(sample_party(names)[0], species=487, form=1, item=names.item("Griseous Orb"))
    # Origin Forme's row has Levitate, Altered Forme's Pressure: coverage reads the form's row.
    assert coverage_module._party_abilities(tables, [origin]) == {AbilityId.LEVITATE}
    altered = replace(origin, form=0)
    assert coverage_module._party_abilities(tables, [altered]) == {AbilityId.PRESSURE}
    check_coverage(tables, names, [origin], roark_trainer(), roark_party(names), 0)
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        check_coverage(
            tables,
            names,
            [replace(origin, species=479, form=6)],
            roark_trainer(),
            roark_party(names),
            0,
        )


def test_trainer_pokemon_in_a_data_form_is_not_refused(tables, names):
    # G-41: Devon (491) has Wormadam-Sandy and -Trash in slots 1 and 2, built from the base
    # species (data.trainer.build_trainer_party); coverage has no form line for them.
    devon = next(t for t in load_trainers(INTERMEDIATE) if t.id == 491)
    opp = build_trainer_party(devon, _inter())
    assert [m.form for m in opp] == [0, 1, 2]
    r = check_opponent(tables, names, devon, opp, 0)
    assert not any("form" in m for m in r.missing)


def test_player_rotom_wash_is_not_refused_for_its_form(tables, names):
    wash = replace(sample_party(names)[0], species=479, form=2, ability_slot=0)
    r = check_coverage(tables, names, [wash], roark_trainer(), roark_party(names), 0)
    assert not any("form" in m for m in r.missing)


NO_POLICY = "Roark: no transcribed AI policy for this game (spec 5.10) (AI policy UNSUPPORTED)"


def test_trainer_of_a_game_without_ai_policy_refused(tables, names, vanilla_manifest, monkeypatch):
    # Vanilla's data under the Kaizo game id, with the Kaizo policy removed (K3c Task 2
    # gave Kaizo one): a game without a transcribed AI policy is refused.
    monkeypatch.setitem(coverage_module.POLICY, KAIZO, None)
    kaizo = Names({**vanilla_manifest, "game": KAIZO})
    report = check_opponent(tables, kaizo, roark_trainer(), roark_party(names), Weather.NONE)
    assert NO_POLICY in report.missing and report.level == Support.NONE
    vanilla = check_opponent(tables, names, roark_trainer(), roark_party(names), Weather.NONE)
    assert vanilla.missing == []


def test_kaizo_roark_ai_flags_exact(kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party):
    # Kaizo has an AI policy since K3c Task 2, and its RISKY flag since Task 6: Kaizo Roark's
    # flags 23 (0b10111) are exactly the Kaizo policy's, so no flag line is refused, and none
    # of his AI is (every Expert label of his moves is transcribed). Reports name a labelled
    # trainer by its label (Task 18 fix round 1).
    report = check_opponent(kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party, Weather.NONE)
    assert kaizo_roark.label == "Leader Roark" and f"Leader {NO_POLICY}" not in report.missing
    assert kaizo_roark.ai_flags == 0b10111
    assert {0, 1, 2, 4} <= set(IMPLEMENTED_AI_FLAGS["kaizo"])
    assert not any("AI flag" in m for m in report.missing), report.missing
    player = [kaizo_roark_party[2]]  # Roark's Gible
    full = check_coverage(kaizo_tables, kaizo_names, player, kaizo_roark, kaizo_roark_party, 0)
    assert not any("trainer AI" in m or "AI flag" in m for m in full.missing), full.missing


@pytest.mark.parametrize("flags", [0x1, 0x5, 0x7, 0x17, 0xD, 0xF, 0x1F, 0x207, 0x20F, 0x27, 0x10F])
def test_kaizo_flag_set_has_no_flag_line(
    kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party, flags
):
    # E1 Ruling R2 (replaces plan K3c Ruling 10's exact set): the Kaizo policy runs each
    # trainer's own flag set, every flag of it transcribed (layout.KAIZO_FLAG_BITS): the
    # singles sets of the Roark and Gardenia splits give no "AI flag" line.
    trainer = replace(kaizo_roark, ai_flags=flags)
    report = check_opponent(kaizo_tables, kaizo_names, trainer, kaizo_roark_party, Weather.NONE)
    assert not [m for m in report.missing if "AI flag" in m]


@pytest.mark.fullset
def test_kaizo_implemented_ai_flags_are_the_transcribed_bits():
    from nuzline.ai.layout import KAIZO_FLAG_BITS

    assert set(IMPLEMENTED_AI_FLAGS["kaizo"]) == KAIZO_FLAG_BITS


# --- Kaizo trainer fields (K2b Task 14) ----------------------------------------------


def with_area(trainer, **fields):
    return replace(trainer, area_condition=replace(trainer.area_condition, **fields))


def test_area_trick_room_is_modelled(tables, names):
    """E3-2-4: a Trick Room area (map weather 18) is no longer refused; the fight starts with it
    up for ``TRICK_ROOM_TURNS`` (``start_trick_room``), which Kaizo's tables re-arm for ever."""
    from nuzline.engine.coverage import start_trick_room
    from nuzline.engine.ids import TRICK_ROOM_TURNS
    from nuzline.engine.state import make_state

    trainer = with_area(roark_trainer(), trick_room=True)
    report = check_opponent(tables, names, trainer, roark_party(names), 0)
    assert not any("Trick Room" in line for line in report.missing)
    assert start_trick_room(trainer) == TRICK_ROOM_TURNS and start_trick_room(roark_trainer()) == 0
    party = roark_party(names)
    state = make_state(tables, party, party, trick_room=start_trick_room(trainer))
    assert int(trick_room_of(state)) == TRICK_ROOM_TURNS
    assert int(trick_room_of(make_state(tables, party, party))) == 0


def test_area_gravity_is_modelled(tables, names):
    """E1 trial batch (G-67): a Gravity area is no longer refused; the fight starts with Gravity
    up for ``GRAVITY_TURNS`` (``start_gravity``), which Kaizo's tables re-arm for ever."""
    from nuzline.engine.coverage import start_gravity
    from nuzline.engine.ids import GRAVITY_TURNS
    from nuzline.engine.state import make_state

    trainer = with_area(roark_trainer(), gravity=True)
    report = check_opponent(tables, names, trainer, roark_party(names), 0)
    assert not any("Gravity" in line for line in report.missing)
    assert start_gravity(trainer) == GRAVITY_TURNS and start_gravity(roark_trainer()) == 0
    party = roark_party(names)
    state = make_state(tables, party, party, gravity=start_gravity(trainer))
    assert int(gravity_of(state)) == GRAVITY_TURNS
    assert int(gravity_of(make_state(tables, party, party))) == 0


@pytest.mark.fullset
def test_gauntlet_refused_facts_not_recorded(tables, names):
    t = replace(roark_trainer(), gauntlet=("Oreburgh Gym", 2))
    report = check_opponent(tables, names, t, roark_party(names), 0)
    line = "Roark: gauntlet Oreburgh Gym (fight 2; facts not recorded) (battle format UNSUPPORTED)"
    assert line in report.missing


@pytest.mark.fullset
def test_gauntlet_refused_with_facts(tables, names):
    t = replace(
        roark_trainer(),
        gauntlet=("Oreburgh Gym", 1),
        gauntlet_facts=GauntletFacts.from_dict(facts_dict()),
    )
    report = check_opponent(tables, names, t, roark_party(names), 0)
    line = "Roark: gauntlet Oreburgh Gym (fight 1; facts recorded) (battle format UNSUPPORTED)"
    assert line in report.missing


def test_optional_not_refused(tables, names):
    plain = check_opponent(tables, names, roark_trainer(), roark_party(names), 0)
    optional = replace(roark_trainer(), optional=True)
    assert check_opponent(tables, names, optional, roark_party(names), 0).missing == plain.missing


def test_missing_pkcalc_set_refused(tables, names):  # preflight ruling #34
    t = replace(roark_trainer(), pid_routine=None)
    report = check_opponent(tables, names, t, roark_party(names), 0)
    for k in range(len(t.party)):
        line = (
            f"Roark slot {k}: nature/gender/ability unknown (no PKCalc set) "
            "(trainer data UNSUPPORTED)"
        )
        assert line in report.missing
    full = replace(t.party[0], nature=3, gender=0, ability_slot=0)
    t = replace(t, party=(full, *t.party[1:]))
    report = check_opponent(tables, names, t, roark_party(names), 0)
    assert not any(m.startswith("Roark slot 0:") for m in report.missing)
    assert any(m.startswith("Roark slot 1:") for m in report.missing)


def test_unclassified_kaizo_rom_changes_refused(tables, names, vanilla_manifest):
    line = (
        "Kaizo ROM: 3 code changes unclassified (docs/kaizo/rom_diff.md) (ROM change UNSUPPORTED)"
    )
    kaizo = Names({**vanilla_manifest, "game": KAIZO, "unclassified_rom_changes": 3})
    report = check_opponent(tables, kaizo, roark_trainer(), roark_party(names), 0)
    assert line in report.missing
    clean = Names({**vanilla_manifest, "game": KAIZO, "unclassified_rom_changes": 0})
    report = check_opponent(tables, clean, roark_trainer(), roark_party(names), 0)
    assert not any("ROM change" in m for m in report.missing)
    # Vanilla never carries the line, whatever its manifest says.
    other = Names({**vanilla_manifest, "unclassified_rom_changes": 3})
    assert check_opponent(tables, other, roark_trainer(), roark_party(names), 0).missing == []


def test_unknown_start_condition_refused_not_read_as_weather(tables, names):
    # Kaizo trainer 676 Corbin and the unplaced trainers the user left unanswered.
    line = "Battle start: condition unknown (user call) (start condition UNSUPPORTED)"
    t = replace(roark_trainer(), start_condition="unknown")
    report = check_opponent(tables, names, t, roark_party(names), 0)
    assert line in report.missing
    assert not any("(weather UNSUPPORTED)" in m for m in report.missing)
    with pytest.raises(ValueError, match="pending user call, not a weather; coverage refuses"):
        start_weather(t)
    area_only = with_area(roark_trainer(), weather="unknown")
    assert line in check_opponent(tables, names, area_only, roark_party(names), 0).missing


def test_area_weather_and_fog_are_modelled(tables, names):
    # E2 batch 3: sandstorm, hail, sun, rain; E1 Batch C: fog (G-44).
    for cond in ("sandstorm", "hail", "sun", "rain", "fog"):
        t = replace(roark_trainer(), start_condition=cond)
        assert not check_opponent(tables, names, t, roark_party(names), 0).missing


def test_ability_override_without_set_data_refused(tables, names):  # review I1
    roark = roark_trainer()
    assert roark.pid_routine == "vanilla"
    t = replace(roark, party=(replace(roark.party[0], ability_override=16), *roark.party[1:]))
    report = check_opponent(tables, names, t, roark_party(names), 0)
    line = "Roark slot 0: nature/gender/ability unknown (no PKCalc set) (trainer data UNSUPPORTED)"
    assert line in report.missing
    assert not any(m.startswith("Roark slot 1:") for m in report.missing)
    full = replace(t.party[0], nature=3, gender=0, ability_slot=1)
    t = replace(t, party=(full, *t.party[1:]))
    report = check_opponent(tables, names, t, roark_party(names), 0)
    assert not any("no PKCalc set" in m for m in report.missing)


@pytest.mark.fullset
def test_report_names_a_labelled_trainer_by_its_label(tables, names, metronome_refused):
    # Task 18 fix round 1: a Kaizo label is what a report shows; the ROM name stays data.
    t = replace(roark_trainer(), label="Leader Roark", format="double")
    opp = roark_party(names)
    opp[0] = replace(opp[0], moves=(names.move("Metronome"), 0, 0, 0))
    report = check_opponent(tables, names, t, opp, 0)
    assert "Leader Roark: a double battle (battle format UNSUPPORTED)" in report.missing
    line = "Leader Roark GEODUDE (slot 0): Metronome (move effect UNSUPPORTED)"
    assert line.lower() in [m.lower() for m in report.missing]
    assert not any(m.startswith("Roark") for m in report.missing + report.sampling_only)
    assert t.name == "Roark"
    assert unknown_start(replace(t, start_condition="unknown"))
    assert not unknown_start(t)


# Gen IV move index of Rock Throw (names.move("Rock Throw") in the vanilla manifest; it is on
# Roark's Geodude and Onix, tests/engine/helpers.roark_party).
MOVE_ROCK_THROW = 88


@pytest.mark.parametrize(
    "cond, reached",
    [
        ({"formats": ["double"]}, False),  # Roark is single
        ({"formats": ["single"]}, True),
        ({"areas": ["gravity"]}, False),
        ({"moves": [MOVE_ROCK_THROW]}, True),  # a vanilla Roark move
        ({"trainer_items": True}, True),  # vanilla Roark has Potions
        ({"trainers": [246]}, True),
        ({"trainers": [247]}, False),
    ],
)
def test_rom_change_refusals(tables, names, vanilla_manifest, cond, reached):
    kaizo = kaizo_names_with(vanilla_manifest, [cond])
    roark, mons = roark_trainer(), sample_party(names) + roark_party(names)
    assert rom_change_refusals(tables, kaizo, roark, mons) == (["c"] if reached else [])


def test_area_ability_and_item_terms(tables, names, vanilla_manifest):
    roark, mons = roark_trainer(), sample_party(names)
    gravity = with_area(roark, gravity=True)
    kaizo = kaizo_names_with(vanilla_manifest, [{"areas": ["gravity"]}])
    assert rom_change_refusals(tables, kaizo, gravity, mons) == ["c"]
    assert rom_change_refusals(tables, kaizo, with_area(roark, trick_room=True), mons) == []
    leftovers = names.item("Leftovers")
    held = [replace(mons[0], item=leftovers), *mons[1:]]
    kaizo = kaizo_names_with(vanilla_manifest, [{"items": [leftovers]}])
    assert rom_change_refusals(tables, kaizo, roark, held) == ["c"]
    assert rom_change_refusals(tables, kaizo, roark, mons) == []
    spec = mons[0]
    row = personal_row(tables, spec.species, spec.form)
    ability = int(np.asarray(tables.species_abilities_decomp)[row][spec.ability_slot])
    kaizo = kaizo_names_with(vanilla_manifest, [{"abilities": [ability]}])
    assert rom_change_refusals(tables, kaizo, roark, [spec]) == ["c"]
    assert rom_change_refusals(tables, kaizo, roark, []) == []


def test_extra_moves_reach_a_condition(tables, names, vanilla_manifest):
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_ROCK_THROW]}])
    assert rom_change_refusals(tables, kaizo, roark_trainer(), []) == []
    assert rom_change_refusals(tables, kaizo, roark_trainer(), [], [MOVE_ROCK_THROW]) == ["c"]


def test_unknown_format_matches_format_conditions(tables, names, vanilla_manifest):
    kaizo = kaizo_names_with(vanilla_manifest, [{"formats": ["multi"]}])
    # Final review M1: the data's unknown format is the string "unknown"
    # (tools/battle_scripts.py); None is matched the same way.
    no_format = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_ROCK_THROW]}])
    for fmt in ("unknown", None):
        roark = replace(roark_trainer(), format=fmt)
        assert rom_change_refusals(tables, kaizo, roark, sample_party(names)) == ["c"]
        assert rom_change_refusals(tables, no_format, roark, sample_party(names)) == []


def test_vanilla_has_no_rom_lines(tables, names, vanilla_manifest):
    roark, mons = roark_trainer(), sample_party(names) + roark_party(names)
    assert rom_change_refusals(tables, names, roark, mons) == []
    plain = kaizo_names_with(vanilla_manifest, [])
    assert rom_change_refusals(tables, plain, roark, mons) == []


def test_only_a_reached_condition_refuses_the_fight(tables, names, vanilla_manifest):
    line = (
        "Kaizo ROM: c is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) "
        "(ROM change UNSUPPORTED)"
    )
    roark, opp, player = roark_trainer(), roark_party(names), sample_party(names)
    hit = kaizo_names_with(vanilla_manifest, [{"moves": [MOVE_ROCK_THROW]}])
    assert line in check_opponent(tables, hit, roark, opp, 0).missing
    assert line in check_coverage(tables, hit, player, roark, opp, 0).missing
    miss = kaizo_names_with(vanilla_manifest, [{"formats": ["double"]}])
    assert not any(
        "ROM change" in m for m in check_coverage(tables, miss, player, roark, opp, 0).missing
    )
    # A move only the player has reaches the condition through check_coverage, not check_opponent.
    mine = replace(player[0], moves=(MOVE_ROCK_THROW, 0, 0, 0))
    only_player = kaizo_names_with(vanilla_manifest, [{"moves": [mine.moves[0]]}])
    no_rock = [
        replace(m, moves=tuple(0 if x == MOVE_ROCK_THROW else x for x in m.moves)) for m in opp
    ]
    assert line not in check_opponent(tables, only_player, roark, no_rock, 0).missing
    assert line in check_opponent(tables, only_player, roark, no_rock, 0, [mine]).missing


# --- battle terrain (K3b2 Task 9) ----------------------------------------------------------


def test_terrain_effects_are_the_decomp_terrain_moves(tables, names):
    moves = [names.move(m) for m in ("Nature Power", "Secret Power", "Camouflage")]
    assert [int(tables.move_ai_effect[m]) for m in moves] == [173, 197, 213]


def test_terrain_moves_refused_only_for_an_unknown_terrain(tables, names):
    nature_power, tackle = names.move("Nature Power"), names.move("Tackle")
    roark = roark_trainer()
    assert TERRAIN_NAMES[roark.terrain] == "TERRAIN_BUILDING"  # tools/terrain.py
    assert terrain_refusal(tables, names, roark, [nature_power]) is None
    mixed = replace(roark, terrain=TERRAIN_UNKNOWN, terrains=(0, 2, 7))
    line = terrain_refusal(tables, names, mixed, [tackle, nature_power, 0])
    assert "start tiles give TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER (ROM; the calc" in line
    assert line.endswith("and Nature Power would read it: name it with --terrain")
    assert terrain_refusal(tables, names, mixed, [tackle]) is None
    unplaced = replace(roark, terrain=TERRAIN_UNKNOWN, terrains=())
    assert "no map places the trainer" in terrain_refusal(tables, names, unplaced, [nature_power])
    unread = replace(roark, terrain=TERRAIN_UNKNOWN, terrains=(TERRAIN_UNKNOWN, 0))
    assert "TERRAIN_PLAIN and a map whose tiles cannot be read" in terrain_refusal(
        tables, names, unread, [nature_power]
    )


def test_fight_with_a_terrain_move_names_the_terrains(tables, names):
    party = sample_party(names, override_moves={0: ["Camouflage", "Tackle"]})
    mixed = replace(roark_trainer(), terrain=TERRAIN_UNKNOWN, terrains=(2, 6))
    report = check_coverage(tables, names, party, mixed, roark_party(names), 0)
    terrain = [m for m in report.missing if "battle terrain" in m]
    want = (
        "Roark: the battle terrain is a run input: its start tiles give TERRAIN_GRASS, "
        "TERRAIN_SNOW (ROM; the calc repo and the docs sheet record none), and Camouflage would "
        "read it: name it with --terrain (battle terrain UNSUPPORTED)"
    )
    assert terrain == [want]
    known = check_coverage(tables, names, party, roark_trainer(), roark_party(names), 0)
    assert not [m for m in known.missing if "battle terrain" in m]


def test_pool_candidate_with_a_terrain_move_is_refused(tables, names):
    from nuzline.prep.pool import Candidate, check_pool

    spec = sample_party(names, override_moves={0: ["Tackle"]})[0]
    cand = Candidate(("party", 0, 0), "party slot 0", spec, 0, (names.move("Secret Power"),), ())
    mixed = replace(roark_trainer(), terrain=TERRAIN_UNKNOWN, terrains=(0, 2))
    report = check_pool(tables, names, [cand], roark_party(names), mixed)
    # G-21: an unnamed mixed terrain stays refused, a run input (G-18).
    assert any(
        "Secret Power would read it: name it with --terrain (battle terrain" in m
        for m in report.missing
    )
    report = check_pool(tables, names, [cand], roark_party(names), roark_trainer())
    assert not any("battle terrain" in m for m in report.missing)


def test_make_state_takes_the_trainers_terrain(tables, names):
    from nuzline.engine.state import make_state

    roark = roark_trainer()
    state = make_state(tables, sample_party(names), roark_party(names), terrain=roark.terrain)
    assert int(state.terrain) == roark.terrain
    assert int(make_state(tables, sample_party(names), roark_party(names)).terrain) == -1
    with pytest.raises(ValueError, match="terrain 25"):
        make_state(tables, sample_party(names), roark_party(names), terrain=25)
