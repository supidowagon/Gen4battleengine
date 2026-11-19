"""E1 batch G (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): the per-party ability override
(F-O2, G-1..G-3): a Transform user's copy of a frozen Sky Shaymin reverts to Land Forme and
Land's ability is written into the user's own party Pokemon (subscript_check_shaymin_form.s:5-30,
battle_script.c:3401-3407, battle_io_command.c:597-604). Each case is checked against the
reference engine."""

from __future__ import annotations

import dataclasses

import pytest

from nuzline.ai.switching import switching_refusals
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import decomp_abilities
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Stat, Status, Type
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()
SHAYMIN = 492


def _acts(names, s, d, mine, theirs):
    """Move names as actions: side ``d`` uses ``mine``, the other side ``theirs``."""
    acts = [0, 0]
    acts[d] = names.move_slot(s, d, mine)
    acts[1 - d] = names.move_slot(s, 1 - d, theirs)
    return acts


def _fight(tables, names, d, *, personality_slot=0):
    """Turn 1 of a Transform: side ``d`` (the Transform user, a level-50 Ditto and a bench
    Geodude) copies the other side's level-5 Sky Shaymin (form 1); the Ditto acts first. Returns the
    plain (untransformed) start state and the state after the turn."""
    ditto = dataclasses.replace(
        _mon(names, "Ditto", ["Transform"], 50), ability_slot=personality_slot
    )
    sky = dataclasses.replace(_mon(names, "Shaymin", ["Air Slash"], 5), form=1)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    parties = {d: [ditto, geo], 1 - d: [sky, geo]}
    s = make_state(tables, parties[0], parties[1])
    plain = s
    s = s._replace(stats=s.stats.at[d, 0, Stat.SPE].set(500))
    out = assert_matches_reference(tables, s, _acts(names, s, d, "Transform", "Air Slash"), O)
    return plain, out


def _revert(tables, names, d, *, status=Status.FREEZE, substitute=0, personality_slot=0):
    """Turn 2: the copy has ``status`` and the Shaymin's Air Slash hits it first."""
    plain, s = _fight(tables, names, d, personality_slot=personality_slot)
    assert int(s.transformed[d]) == 1 and int(s.species[d, 0]) == SHAYMIN and int(s.form[d, 0]) == 1
    s = s.with_status(d, 0, status)
    s = s._replace(
        substitute_hp=s.substitute_hp.at[d].set(substitute),
        stats=s.stats.at[d, 0, Stat.SPE].set(1).at[1 - d, 0, Stat.SPE].set(500),
    )
    acts = _acts(names, s, d, "Air Slash", "Air Slash")
    return plain, s, assert_matches_reference(tables, s, acts, O)


@pytest.mark.parametrize("d", [0, 1], ids=["player Ditto", "AI Ditto"])
def test_a_transformed_copy_of_a_frozen_sky_shaymin_reverts_into_its_party_pokemon(
    tables, names, d
):
    """shay:5-9 pass for the copy; :10-30 and BtlIOCmd_UpdatePartyMon (io:597-604): the copy is
    form 0 with Natural Cure and Land's types, the party Pokemon holds Natural Cure (override)
    and form 0, and RefreshMonData gives the copy the Ditto's OWN stats (lib:169-193)."""
    plain, _, out = _revert(tables, names, d)
    assert int(out.form[d, 0]) == 0 and int(out.species[d, 0]) == SHAYMIN
    assert int(out.transformed[d]) == 1 and int(out.shadow_form[d]) == 0
    assert int(out.ability[d, 0]) == int(out.party_ability[d, 0]) == AbilityId.NATURAL_CURE
    assert out.types[d, 0].tolist() == [Type.GRASS, Type.GRASS]
    assert out.stats[d, 0].tolist() == plain.stats[d, 0].tolist() == out.shadow_stats[d].tolist()
    assert int(out.max_hp[d, 0]) == int(plain.max_hp[d, 0])
    if d == 0:  # the AI learns the written ability (battle_script.c:3401-3403)
        assert int(out.ai_ability_known[0]) != 0


@pytest.mark.parametrize("case", ["not frozen", "substitute"])
def test_the_copy_is_left_alone_without_a_freeze_or_with_a_substitute(tables, names, case):
    """shay:8-9: a Substitute (one the hit does not break), or a status other than Freeze, ends
    the script at :32."""
    _, before, out = _revert(
        tables,
        names,
        0,
        status=Status.BURN if case == "not frozen" else Status.FREEZE,
        substitute=200 if case == "substitute" else 0,
    )
    assert int(out.form[0, 0]) == 1 and int(out.party_ability[0, 0]) == 0
    assert int(out.ability[0, 0]) == int(before.ability[0, 0])


def test_land_slot_one_falls_back_to_slot_zero(tables, names):
    """shay:12-19: the personality bit picks Land's ability 2, which is empty (decomp id 0, whose
    AbilityId is NONE, not 0): ability 1 is taken."""
    _, _, out = _revert(tables, names, 0, personality_slot=1)
    assert int(out.ability[0, 0]) == int(out.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    assert int(out.ability_slot[0, 0]) == 0


def test_the_written_ability_outlives_the_copy_and_cures_at_the_next_switch_out(tables, names):
    """lib:124 (InitBattleMon reads MON_DATA_ABILITY): after the switch-out the Ditto is itself
    with Natural Cure (the bench slot reads it, ``decomp_abilities``); sent out again and
    burned, its next switch-out cures the status (the party override, not a copy, scr:9146-9154)."""
    plain, _, s = _revert(tables, names, 0)
    leave = MOVE_ACTIONS + 1
    foe = names.move_slot(s, 1, "Air Slash")
    s = assert_matches_reference(tables, s, [leave, foe], O)
    assert int(s.species[0, 0]) == int(plain.species[0, 0]) and int(s.transformed[0]) == 0
    assert s.types[0, 0].tolist() == plain.types[0, 0].tolist()
    assert s.stats[0, 0].tolist() == plain.stats[0, 0].tolist()
    assert int(s.ability[0, 0]) == int(s.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    assert int(decomp_abilities(tables, s)[0, 0]) == ABILITY_DECOMP_IDS[AbilityId.NATURAL_CURE]
    s = assert_matches_reference(tables, s, [MOVE_ACTIONS, foe], O)  # the Ditto comes back in
    assert int(s.active[0]) == 0 and int(s.transformed[0]) == 0
    s = s.with_status(0, 0, Status.BURN)
    s = assert_matches_reference(tables, s, [leave, foe], O)
    assert int(s.status[0, 0]) == 0


def test_a_new_transform_copies_over_the_override_and_ending_it_restores_the_override(
    tables, names
):
    """scr:5346-5348: the copy takes the target's ability; the party's stays (lib:124)."""
    ditto = _mon(names, "Ditto", ["Transform"], 50)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    s = make_state(tables, [ditto, geo], [geo, geo])
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.NATURAL_CURE).at[1, 0].set(AbilityId.INTIMIDATE),
        party_ability=s.party_ability.at[0, 0].set(AbilityId.NATURAL_CURE),
        stats=s.stats.at[0, 0, Stat.SPE].set(500),
    )
    foe = names.move_slot(s, 1, "Tackle")
    s = assert_matches_reference(tables, s, [names.move_slot(s, 0, "Transform"), foe], O)
    assert int(s.ability[0, 0]) == AbilityId.INTIMIDATE
    assert int(s.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    s = assert_matches_reference(tables, s, [MOVE_ACTIONS + 1, foe], O)
    assert int(s.ability[0, 0]) == AbilityId.NATURAL_CURE


def test_the_ais_bench_ditto_with_the_override_is_seen_as_natural_cure(tables, names):
    """trainer_ai.c:3451-3528 read the AI's own party abilities: ``decomp_abilities`` gives the
    override, and ``switching_refusals`` names AI_IsAsleepWithNaturalCure."""
    ditto = _mon(names, "Ditto", ["Transform"], 50)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    s = make_state(tables, [geo, geo], [geo, ditto])
    assert not any("Natural Cure" in line for line in switching_refusals(tables, s))
    s = s._replace(
        ability=s.ability.at[1, 1].set(AbilityId.NATURAL_CURE),
        party_ability=s.party_ability.at[1, 1].set(AbilityId.NATURAL_CURE),
    )
    assert int(decomp_abilities(tables, s)[1, 1]) == ABILITY_DECOMP_IDS[AbilityId.NATURAL_CURE]
    assert any("Natural Cure" in line for line in switching_refusals(tables, s))
