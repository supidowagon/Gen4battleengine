"""The secondary chance is read at the hit (``outcome.secondary_procs``).

``BattleSystem_TriggerSecondaryEffect`` (battle_lib.c:1552-1556, :1571-1575) and
``BtlCmd_CheckEffectActivation`` (battle_script.c:8369-8378) double ``effectChance`` when the
attacker's ``Battler_Ability`` is Serene Grace at that moment, so a Gastro Acid
(subscript_suppress_target_ability.s:10), Worry Seed (subscript_give_target_insomnia.s:11), Skill
Swap (subscript_exchange_abilities.s:21-24) or an armed Trace's copy before the hit changes it.
``Outcome.secondary`` holds the roll ``RandNext() % 100`` as its class over the chances the hit
can read (``outcome.secondary_thresholds``); a third battler's ability writer at B = 4 also
reaches the target's armor (``outcome.armored``) and the threshold grid (``hit_whole``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.enumerate import outcome_laws, secondary_class_law
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId, Status
from nuzline.engine.outcome import Outcome, empty_outcome
from tests.engine.d2_states import four, party_of, with_moves
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_enumerate import _assert_mask_sound

_laws = jax.jit(outcome_laws)


def _laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, jnp.asarray(actions, jnp.int32)))


def _grace(tables, names, opp_move="Leer"):
    """A Serene Grace Togepi with Ember (10 % burn) against a faster Geodude using ``opp_move``."""
    s = roark_state(
        tables,
        names,
        player_species="Togepi",
        player_moves=("Ember",),
        player_ability="Serene Grace",
        opp_moves=(opp_move,),
        opp_faster=True,
    )
    return s, A("Ember", opp_move, s, names)


def _with_roll(value: int) -> Outcome:
    out = empty_outcome()
    return out._replace(secondary=out.secondary.at[0, 0].set(value))


def _burned(state) -> bool:
    return int(state.status[1, 0]) == Status.BURN


@pytest.mark.slow  # both engines' step programs (about 8.5 GB peak locally)
def test_the_doubled_chance_is_the_live_ability_s(tables, names):
    # Roll class 11 (r = 10): it passes 20 (Serene Grace) and fails 10.
    s, acts = _grace(tables, names)
    assert _burned(assert_matches_reference(tables, s, acts, _with_roll(11)))
    plain = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    assert not _burned(assert_matches_reference(tables, plain, acts, _with_roll(11)))
    assert _burned(assert_matches_reference(tables, plain, acts, _with_roll(1)))


@pytest.mark.slow  # both engines' step programs (about 8.5 GB peak locally)
def test_a_faster_gastro_acid_takes_the_double_before_the_hit(tables, names):
    s, acts = _grace(tables, names, "Gastro Acid")
    assert not _burned(assert_matches_reference(tables, s, acts, _with_roll(11)))
    assert _burned(assert_matches_reference(tables, s, acts, _with_roll(1)))


def test_the_laws_class_the_roll_over_both_chances_when_the_ability_can_change(tables, names):
    s, acts = _grace(tables, names)
    calm = _laws_of(tables, s, acts)
    assert calm.secondary_thresholds[0].tolist() == [10, 20]
    assert calm.secondary_possible[0].tolist() == [False, True]
    s, acts = _grace(tables, names, "Gastro Acid")
    laws = _laws_of(tables, s, acts)
    assert laws.secondary_possible[0].tolist() == [True, True]
    assert secondary_class_law([10, 20]) == ([0, 1, 11], [80, 10, 10], 100)
    assert secondary_class_law([10]) == ([0, 1], [90, 10], 100)


@pytest.mark.slow  # the mask check's step program (several GB)
def test_a_gastro_acid_board_steps_alike_whatever_it_leaves_undrawn(tables, names):
    s, acts = _grace(tables, names, "Gastro Acid")
    empty = empty_outcome()
    ones = Outcome(*(np.ones_like(np.asarray(f)) for f in empty))
    for arbitrary in (_with_roll(11), ones):
        _assert_mask_sound(tables, s, np.asarray(acts), arbitrary, 7)


@pytest.mark.slow  # the B = 4 laws' compile
def test_b4_a_third_battlers_worry_seed_opens_the_armor_and_the_threshold(tables, names):
    # Battler 0's Tackle meets battler 1's Shell Armor; battler 2 (battler 0's partner) can
    # Worry Seed it first (Insomnia), so the crit is drawn and the accuracy roll kept whole.
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ("Worry Seed",) if b == 2 else ("Tackle",))
    p1 = party_of(s, 1)
    s = s._replace(ability=s.ability.at[p1, int(s.active[1])].set(AbilityId.SHELL_ARMOR))
    laws = _laws_of(tables, s, np.zeros(4, np.int32))
    assert not bool(laws.no_crit[0]) and bool(laws.hit_whole[0])
    calm = with_moves(s, names, 2, ("Tackle",))
    laws = _laws_of(tables, calm, np.zeros(4, np.int32))
    assert bool(laws.no_crit[0]) and not bool(laws.hit_whole[0])
    assert not bool(laws.relevant.contact_proc[1, 0])
    # Battler 2 holds Static and Skill Swaps it to battler 1 (subscript_exchange_abilities.s:
    # 21-24) before battler 0's contact hit: its roll is read (outcome.doubles_ability_change).
    p2 = party_of(s, 2)
    swap = with_moves(calm, names, 2, ("Skill Swap",))
    swap = swap._replace(ability=swap.ability.at[p2, int(swap.active[2])].set(AbilityId.STATIC))
    laws = _laws_of(tables, swap, np.zeros(4, np.int32))
    assert bool(laws.relevant.contact_proc[1, 0]) and not bool(laws.no_crit[0])


def test_a_klutz_king_s_rock_rolls_at_its_param_when_a_worry_seed_can_take_the_klutz(tables, names):
    # Klutz hides the King's Rock (Battler_HeldItem, battle_lib.c:5352-5356); a faster Worry Seed
    # (subscript_give_target_insomnia.s:11) can take the Klutz before the hit, which then reads
    # the item's param (bcp.c:4633-4639).
    s = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Worry Seed",))
    s = s._replace(
        item=s.item.at[0, 0].set(names.item("King’s Rock")),
        ability=s.ability.at[0, 0].set(AbilityId.KLUTZ),
    )
    laws = _laws_of(tables, s, A("Tackle", "Worry Seed", s, names))
    assert int(laws.flinch_item_chance[0]) == 10
    calm = roark_state(tables, names, player_moves=("Tackle",), opp_moves=("Leer",))
    calm = calm._replace(item=s.item, ability=s.ability)
    laws = _laws_of(tables, calm, A("Tackle", "Leer", calm, names))
    assert int(laws.flinch_item_chance[0]) == 0
