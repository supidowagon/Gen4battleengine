"""The coverage registry refuses a fight that uses unimplemented mechanics and names
every one of them (spec §3)."""

from dataclasses import replace

import pytest

from nuzline.data.trainer import build_trainer_party, load_trainers
from nuzline.engine.coverage import (
    IMPLEMENTED_ABILITIES,
    IMPLEMENTED_AI_FLAGS,
    IMPLEMENTED_EFFECTS,
    IMPLEMENTED_ITEMS,
    IMPLEMENTED_TRAINER_ITEMS,
    IMPLEMENTED_WEATHER,
    Support,
    UnsupportedMechanics,
    check_coverage,
    check_opponent,
    require_coverage,
    start_weather,
)
from nuzline.engine.ids import Weather
from tests.engine.helpers import INTERMEDIATE, _inter, roark_party, roark_trainer, sample_party


def test_roark_vs_sample_party_is_fully_covered(tables, names):
    r = check_coverage(tables, names, sample_party(names), roark_trainer(), roark_party(names), 0)
    assert r.missing == [] and r.level == Support.EXACT


def test_all_missing_reported_at_once(tables, names):
    party = sample_party(
        names,
        override_moves={0: ["Metronome", "Scratch", "Leer", "Ember"]},
        override_item={1: "Leftovers"},
    )
    with pytest.raises(UnsupportedMechanics) as e:
        require_coverage(tables, names, party, roark_trainer(), roark_party(names), 0)
    msg = str(e.value)
    assert "Metronome" in msg and "Leftovers" in msg
    assert e.value.report.level == Support.NONE


def test_only_no_weather_is_supported():
    assert IMPLEMENTED_WEATHER[Weather.NONE] == Support.EXACT
    for w in Weather:
        if w != Weather.NONE:
            assert IMPLEMENTED_WEATHER[w] == Support.NONE


def test_hail_route_trainer_is_refused(tables, names):
    hail = next(t for t in load_trainers(INTERMEDIATE) if t.start_condition == "hail")
    assert start_weather(hail) == Weather.HAIL
    opp = build_trainer_party(hail, _inter())
    with pytest.raises(UnsupportedMechanics) as e:
        require_coverage(tables, names, sample_party(names), hail, opp, Weather.HAIL)
    assert "Hail" in str(e.value)


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


@pytest.mark.parametrize(
    ("name", "fmt"), [("Mars", "multi"), ("Ismael", "double_two_trainer"), ("Jack", "multi")]
)
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


def test_start_condition_argument_is_checked(tables, names):
    r = check_coverage(
        tables, names, sample_party(names), roark_trainer(), roark_party(names), Weather.RAIN
    )
    assert any("Rain" in m for m in r.missing)


def test_unsupported_trainer_item_and_mixed_bag(tables, names):
    potion, super_potion = names.item("Potion"), names.item("Super Potion")
    trainer = replace(roark_trainer(), items=(potion, super_potion))
    r = check_coverage(tables, names, sample_party(names), trainer, roark_party(names), 0)
    assert any("Super Potion (trainer item UNSUPPORTED)" in m for m in r.missing)
    assert any("mixed items" in m for m in r.missing)


def test_ai_flags_outside_the_policy_are_refused(tables, names):
    risky = replace(roark_trainer(), ai_flags=0b10111)  # + AI_FLAG_RISKY
    r = check_coverage(tables, names, sample_party(names), risky, roark_party(names), 0)
    assert any("AI_FLAG_RISKY" in m for m in r.missing)
    basic_only = replace(roark_trainer(), ai_flags=0b1)
    r = check_coverage(tables, names, sample_party(names), basic_only, roark_party(names), 0)
    assert any("AI flag set" in m for m in r.missing)


def test_ai_refusals_are_included(tables, names):
    """An opponent move whose AI script paths are not transcribed (ai.policy)."""
    opp = roark_party(names)
    opp[0] = replace(opp[0], moves=(names.move("Ember"), 0, 0, 0))
    r = check_coverage(tables, names, sample_party(names), roark_trainer(), opp, 0)
    assert any("Ember" in m and "trainer AI" in m for m in r.missing)


def test_sampling_only_is_unimplemented():
    # The spec's sampling-only grade (section 4.2) has no registry entry and no engine
    # path yet (coverage.py docstring): a registry that assigns it must come with that
    # path, so this guard fails first.
    registries = (
        IMPLEMENTED_EFFECTS,
        IMPLEMENTED_ABILITIES,
        IMPLEMENTED_ITEMS,
        IMPLEMENTED_TRAINER_ITEMS,
        IMPLEMENTED_AI_FLAGS,
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
        (
            492,
            1,
            "Griseous Orb",
            "Shaymin (slot 0): Sky Forme reverts when frozen (form change UNSUPPORTED)",
        ),
    ],
)
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
    r = check_coverage(tables, names, [origin], roark_trainer(), roark_party(names), 0)
    assert any(m.endswith("GIRATINA (slot 0): Levitate (ability UNSUPPORTED)") for m in r.missing)
    assert not any("Pressure" in m for m in r.missing)  # Altered Forme's ability
    with pytest.raises(ValueError, match="species 479 has no form 6"):
        check_coverage(
            tables,
            names,
            [replace(origin, species=479, form=6)],
            roark_trainer(),
            roark_party(names),
            0,
        )


DATA_FORM_LINE = (
    "trainer Pokemon in a data form keeps its base form's stats, ability and moves "
    "(form UNSUPPORTED)"
)


def test_trainer_pokemon_in_a_data_form_is_refused(tables, names):
    # Controller ruling (K2a Task 6): TrainerData_BuildParty sets the form after building
    # from the base form (src/trainer_data.c:216-218). Devon (491) has Wormadam-Sandy and
    # -Trash in slots 1 and 2; his Plant Cloak Wormadam in slot 0 passes.
    devon = next(t for t in load_trainers(INTERMEDIATE) if t.id == 491)
    opp = build_trainer_party(devon, _inter())
    assert [m.form for m in opp] == [0, 1, 2]
    r = check_opponent(tables, names, devon, opp, 0)
    hits = [m for m in r.missing if m.endswith(DATA_FORM_LINE)]
    assert hits == [f"{devon.name} WORMADAM (slot {s}): {DATA_FORM_LINE}" for s in (1, 2)]
    assert r.level == Support.NONE


@pytest.mark.parametrize(("species", "form"), [("Shellos", 1), ("Unown", 5), ("Arceus", 3)])
def test_trainer_pokemon_in_a_form_without_data_passes(tables, names, species, form):
    mon = replace(roark_party(names)[0], species=names.species(species), form=form)
    r = check_opponent(tables, names, roark_trainer(), [mon], 0)
    assert not any("data form" in m for m in r.missing)


def test_player_rotom_wash_is_not_refused_for_its_form(tables, names):
    wash = replace(sample_party(names)[0], species=479, form=2, ability_slot=0)
    r = check_coverage(tables, names, [wash], roark_trainer(), roark_party(names), 0)
    assert not any("form" in m for m in r.missing)
