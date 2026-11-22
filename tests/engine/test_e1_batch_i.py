"""E1 batch I.

I-9: ``hooks.order_swaps`` needs no dead-battler line of its own (the sort's short-circuit sits in
``order.compare_speed``).
I-6: the cures of ``BattleSystem_RecoverStatusByAbility`` reveal no ability.
I-3: Sketch (move effect 95), the player side: ``moveSketched`` and ``BtlCmd_TrySketch``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.engine.hooks import ALL_SWAPS, order_swaps
from tests.engine.helpers import roark_state


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
