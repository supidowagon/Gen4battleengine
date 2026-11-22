"""E1 batch I.

I-9: ``hooks.order_swaps`` needs no dead-battler line of its own (the sort's short-circuit sits in
``order.compare_speed``).
I-6: the cures of ``BattleSystem_RecoverStatusByAbility`` reveal no ability.
I-3: Sketch (move effect 95), the player side: ``moveSketched`` and ``BtlCmd_TrySketch``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.engine.hooks import ALL_SWAPS, REVEALS_ON, STATUS_CURE, order_swaps
from nuzline.engine.ids import (
    AbilityId,
    Status,
)
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import A, assert_matches_reference, roark_state


def _down(s, side):
    return s._replace(hp=s.hp.at[side, s.active[side]].set(0))


@pytest.mark.parametrize("equal_speed", [False, True])
@pytest.mark.parametrize("opp_faster", [False, True])
def test_order_swaps_puts_a_fainted_battler_last(tables, names, equal_speed, opp_faster):
    """``BattleSystem_CompareBattlerSpeed`` (battle_lib.c:1204-1211): a fainted battler is SLOWER
    than a living one for every draw (tie bit, Quick Claw bits), whatever the speeds."""
    s = roark_state(tables, names, equal_speed=equal_speed, opp_faster=opp_faster)
    assert int(order_swaps(tables, _down(s, 0))) == ALL_SWAPS  # the player's second, every draw
    assert int(order_swaps(tables, _down(s, 1))) == 0  # the player's first, every draw
    assert order_swaps(tables, s).dtype == jnp.int32


# --- I-6: the cures of BattleSystem_RecoverStatusByAbility reveal nothing -------------------


def test_status_cure_reveals_are_shed_skin_and_hydration_alone():
    """``BattleMessage_AbilityTag`` returns ``msgAbilityTemp`` for ``BTLSCR_MSG_TEMP`` without
    ``BattleAI_SetAbility`` (battle_script.c:11801-11802), and ``RecoverStatusByAbility``'s
    script tags that (subscript_ability_forbids_status.s:6); Shed Skin's and Hydration's script
    tags a battler (subscript_ability_restore_status.s:8: reveal)."""
    holders = {a for a, kinds in REVEALS_ON.items() if STATUS_CURE in kinds}
    assert holders == {AbilityId.SHED_SKIN, AbilityId.HYDRATION}


@pytest.mark.parametrize(
    "species, ability, status",
    [
        ("Spinarak", "Insomnia", Status.SLEEP),
        ("Mankey", "Vital Spirit", Status.SLEEP),
        ("Persian", "Limber", Status.PARALYSIS),
    ],
)
def test_a_cure_at_move_end_leaves_the_ai_in_the_dark(tables, names, species, ability, status):
    """The statused holder is cured at the MoveEnd (battle_lib.c:4397-4410) and the AI's
    ``battlerAbilities`` stays unwritten (I-6): engine == reference."""
    s = roark_state(
        tables,
        names,
        player_species=species,
        player_ability=ability,
        player_moves=("Leer",),
        opp_moves=("Leer",),
    )
    s = s._replace(
        status=s.status.at[0, 0].set(status),
        status_counter=s.status_counter.at[0, 0].set(3 if status == Status.SLEEP else 0),
    )
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0


def test_shed_skin_still_reveals_its_cure(tables, names):
    """subscript_ability_restore_status.s:8 tags ``BTLSCR_MSG_BATTLER_TEMP``: the AI learns it."""
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_ability="Shed Skin",
        player_moves=("Growl",),
        opp_moves=("Leer",),
    )
    s = s.with_status(0, 0, Status.POISON)
    shed = empty_outcome()._replace(shed_skin=jnp.array([1, 0], jnp.int8))
    out = assert_matches_reference(tables, s, A("Growl", "Leer", s, names), shed)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0
