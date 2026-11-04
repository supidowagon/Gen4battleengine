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
