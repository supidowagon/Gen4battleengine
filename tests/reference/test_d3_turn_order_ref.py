"""D3-3: the reference's CalcTurnOrder over four (``reference.engine._four_turn_order``) against the
engine's ``action_order`` / ``claw_flags`` / ``custap_flags``, for the cases of
tests/engine/test_d3_turn_order.py on one B = 4 state each, over all 64 values of ``order_tie``; and
the action phase of a turn on both engines (``play``: the whole state field by field) for a tie
row, a claw holder and an absent battler.

The reference computes the claw / Custap flags inside its compare loop (lib.c:1282-1298), the
engine as a closed form (``order._compared_fighters``): this test is where they meet.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _choose, _command, _four_turn_order
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.test_d3_action_loop import play
from tests.engine.test_d3_turn_order import ALL_BITS, item_of, run, speeds


def _reference_rows(tables, state, actions):
    """``(orders, claw, custap)`` of the reference for every row of ``ALL_BITS``."""
    np_tables, rs = to_numpy(tables), to_ref(state)
    acts = [int(a) for a in actions]
    commands = [_command(a) for a in acts]
    choices = [_choose(np_tables, rs, b, acts[b]) for b in range(4)]
    orders, claws, custaps = [], [], []
    for bits in ALL_BITS:
        outcome = outcome_to_py(empty_outcome(DOUBLES)._replace(order_tie=jnp.asarray(bits)))
        order, custap, claw = _four_turn_order(np_tables, rs, commands, choices, outcome)
        orders.append(order)
        claws.append([bool(c) for c in claw])
        custaps.append([bool(c) for c in custap])
    return orders, claws, custaps


def _agree(tables, state, actions):
    orders, claws, custaps = _reference_rows(tables, state, actions)
    e_orders, _, e_claw, e_custap = run(tables, state, actions)
    assert e_orders.tolist() == orders
    # Every row writes the same flags (they do not depend on the bits).
    assert all(c == claws[0] for c in claws) and all(c == custaps[0] for c in custaps)
    assert e_claw.tolist() == claws[0]
    assert e_custap.tolist() == custaps[0]
    return orders, claws[0], custaps[0]


def test_groups_swap_loop_and_priority(tables, names):
    base = four(tables, names)
    for b in range(4):
        base = with_moves(base, names, b, ["Tackle"])
    s = speeds(base, (100, 90, 300, 200))
    _agree(tables, s, [0, 10, 5, 0])  # item and switch before the fighters
    _agree(tables, s, [0, 0, 0, 0])  # four fighters by speed
    q = with_moves(s, names, 0, ["Quick Attack"])
    orders, _, _ = _agree(tables, q, [0, 0, 0, 0])  # priority beats speed
    assert orders[0][0] == 0
    _agree(tables, q, [4 + 3, 4 + 3, 0, 0])  # two switchers, priority skipped


def test_equal_speeds_every_bit_pattern(tables, names):
    s = speeds(four(tables, names), (150, 150, 150, 150))
    _agree(tables, s, [0, 0, 0, 0])
    _agree(tables, s, [0, 5, 0, 0])
    _agree(tables, s, [10, 0, 5, 0])


def test_absent_battlers(tables, names):
    s = speeds(four(tables, names), (150, 150, 400, 150))
    _agree(tables, with_hp(s, 2, 0), [0, 0, 0, 0])
    _agree(tables, with_hp(with_hp(s, 1, 0), 2, 0), [0, 5, 0, 0])
    _agree(tables, with_hp(with_hp(with_hp(s, 0, 0), 1, 0), 2, 0), [0, 0, 0, 0])


def test_claw_and_custap_flags(tables, names):
    claw, custap = names.item("Quick Claw"), names.item("Custap Berry")
    s = speeds(four(tables, names), (150, 150, 150, 150))
    q = item_of(s, 3, claw)._replace(qc_roll=jnp.asarray([0, 0, 0, 1], jnp.int8))
    _, flags, _ = _agree(tables, q, [0, 0, 0, 0])
    assert flags == [False, False, False, True]
    _, flags, _ = _agree(tables, q, [0, 0, 0, 5])  # switching: no flag
    assert flags == [False] * 4
    alone = with_hp(with_hp(with_hp(q, 0, 0), 1, 0), 2, 0)
    _, flags, _ = _agree(tables, alone, [0, 0, 0, 0])  # only absent partners: no flag
    assert flags == [False] * 4
    c = with_hp(item_of(s, 2, custap), 2, 1)
    _, _, flags = _agree(tables, c, [0, 0, 0, 0])
    assert flags == [False, False, True, False]
    _, _, flags = _agree(tables, c, [0, 0, 5, 0])
    assert flags == [False] * 4


def test_action_phase_matches_on_both_engines(tables, names):
    """The whole state after the actions, for a tie row that reorders the battlers, a Quick Claw
    holder (the carried flag stays set on a holder that does not act first) and an absent
    battler."""
    base = four(tables, names)
    for b in range(4):
        base = with_moves(base, names, b, ["Tackle"])
    s = speeds(base, (150, 150, 150, 150))
    outcome = empty_outcome(DOUBLES)._replace(order_tie=jnp.asarray([1, 0, 1, 1, 0, 1], jnp.int8))
    play(tables, s, [0, 0, 0, 0], None, (1, 0, 3, 2), outcome)
    claw = item_of(s, 3, names.item("Quick Claw"))._replace(
        qc_roll=jnp.asarray([0, 0, 0, 1], jnp.int8)
    )
    play(tables, claw, [0, 0, 0, 0], None, (1, 0, 3, 2), outcome)
    absent = with_hp(s, 2, 0)
    assert int(absent.hp[party_of(absent, 2), int(absent.active[2])]) == 0
    play(tables, absent, [0, 0, 0, 0], None, (1, 0, 3, 2), outcome)
