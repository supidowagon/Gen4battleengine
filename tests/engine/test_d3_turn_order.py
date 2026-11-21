"""D3-3: ``BattleControllerPlayer_CalcTurnOrder`` over four battlers (sheet D3-3 rules 1-9, 4.1-4.9).

Decomp: bcp.c:707-791 (the two command groups, :754-770, and the position-pair swap loop,
:773-786), lib.c:1188-1454 (``CompareBattlerSpeed``: the dead short-circuit :1204-1211, the claw
flags :1278-1300 / :1344-1366, priority :1375-1399, the ladder and the tie draw :1401-1448),
subscript_check_quick_claw.s (the display). The DOUBLES states are ``d2_states.four``: every battler
a Bulbasaur with the same speed, so the speeds of each case are set here (``speeds``).

``model`` below is a plain-Python transcription of the loop and of the compare, used as the
oracle for the order, the read bits and the claw flags over all 64 values of ``order_tie``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.format import DOUBLES, SINGLES
from nuzline.engine.ids import Stat
from nuzline.engine.order import (
    TIE_PAIRS,
    action_order,
    action_order_reads,
    claw_flags,
    custap_flags,
    effective_speed,
    is_item,
    is_switch,
    player_second,
    quick_claw_active,
    custap_active,
    selected_move,
)
from nuzline.engine.outcome import empty_outcome, sample_outcome
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.strategies import any_outcome, legal_action_pair, supported_state

QUICK_ATTACK = "Quick Attack"
FIGHT, ITEM, PARTY = "F", "I", "P"
PAIR_ORDER = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
ALL_BITS = np.asarray(list(itertools.product((0, 1), repeat=6)), np.int8)  # 64 rows


@jax.jit
def _everything(tables, state, actions, bits):
    """``(order, reads, claw, custap)`` for each row of ``bits`` (int8[N, 6])."""

    def one(row):
        o = empty_outcome(DOUBLES)._replace(order_tie=row)
        return (
            action_order(tables, state, actions, o),
            action_order_reads(tables, state, actions, o),
        )

    order, reads = jax.vmap(one)(bits)
    return order, reads, claw_flags(tables, state, actions), custap_flags(tables, state, actions)


def speeds(state, values):
    """Each battler's Speed stat set to ``values[b]``."""
    for b, v in enumerate(values):
        p, slot = party_of(state, b), int(state.active[b])
        state = state._replace(stats=state.stats.at[p, slot, Stat.SPE - 1].set(v))
    return state


def item_of(state, b, item):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(item=state.item.at[p, slot].set(item))


def kinds_of(actions):
    return [ITEM if a >= 10 else PARTY if a >= 4 else FIGHT for a in actions]


def model(bits, kinds, alive, speed, claw=None, priority=None):
    """CalcTurnOrder, transcribed: ``(order, reads, quick_claw_flags)``. ``claw[b]``: the
    battler's Quick Claw / Custap is active; ``priority[b]``: its move's priority."""
    claw = claw or [False] * 4
    priority = priority or [0] * 4
    kind = [k if alive[b] else FIGHT for b, k in enumerate(kinds)]  # rule 7: absent = FIGHT
    order = [b for b in range(4) if kind[b] != FIGHT] + [b for b in range(4) if kind[b] == FIGHT]
    reads = [False] * 6
    flags = [False] * 4
    p = 0
    for i in range(3):
        for j in range(i + 1, 4):
            b1, b2 = order[i], order[j]
            if kind[b1] == kind[b2]:
                fight = kind[b1] == FIGHT
                swap = None
                if alive[b1] and not alive[b2]:
                    swap = False
                elif not alive[b1] and alive[b2]:
                    swap = True
                else:
                    if fight and alive[b1] and alive[b2]:
                        for b in (b1, b2):
                            flags[b] = flags[b] or claw[b]
                    p1, p2 = (priority[b1], priority[b2]) if fight else (0, 0)
                    if p1 != p2:
                        swap = p1 < p2
                    elif claw[b1] != claw[b2]:
                        swap = claw[b2]
                    else:
                        if speed[b1] == speed[b2]:
                            swap = bool(bits[p])
                            reads[p] = alive[b1] and alive[b2]
                        else:
                            swap = speed[b1] < speed[b2]
                if swap:
                    order[i], order[j] = b2, b1
            p += 1
    return order, reads, flags


def run(tables, state, actions, bits=ALL_BITS):
    """The engine's rows, as ``(orders, reads, claw, custap)`` of numpy / lists."""
    acts = jnp.asarray(actions, jnp.int32)
    order, reads, claw, custap = _everything(tables, state, acts, jnp.asarray(bits, jnp.int8))
    return np.asarray(order), np.asarray(reads), np.asarray(claw), np.asarray(custap)


def alive_of(state):
    return [hp(state, b) > 0 for b in range(4)]


def speed_of(tables, state):
    return [int(effective_speed(tables, state, b)) for b in range(4)]


def check_against_model(tables, state, actions, claw=None, priority=None):
    """Every row of ``order_tie`` through the engine equals ``model``; returns the engine rows."""
    orders, reads, claw_f, _ = run(tables, state, actions)
    kinds, alive, speed = kinds_of(actions), alive_of(state), speed_of(tables, state)
    for row, bits in enumerate(ALL_BITS):
        order, rd, flags = model(bits, kinds, alive, speed, claw, priority)
        assert orders[row].tolist() == order, (bits.tolist(), orders[row].tolist(), order)
        assert reads[row].tolist() == rd, bits.tolist()
    return orders, reads, claw_f


def test_tie_layout():
    assert TIE_PAIRS == {2: 0, 4: 6}
    for fmt, shapes in ((SINGLES, ((0,), (3, 0))), (DOUBLES, ((6,), (5, 6)))):
        o = empty_outcome(fmt)
        assert (o.order_tie.shape, o.sort_tie.shape) == shapes
        assert o.order_tie.dtype == jnp.int8 and o.sort_tie.dtype == jnp.int8


def test_groups_item_and_party_first_in_battler_order(tables, names):
    """Battler 1 uses a bag item, battler 2 switches, battlers 0 and 3 fight: [1, 2, then 0 and 3
    by speed] (rule 2); the item and the switch are never compared (the faster switcher stays
    behind the item)."""
    s = speeds(four(tables, names), (100, 90, 300, 200))
    acts = [0, 10, 5, 0]
    orders, reads, _ = check_against_model(tables, s, acts)
    assert {tuple(o) for o in orders.tolist()} <= {(1, 2, 3, 0), (1, 2, 0, 3)}
    assert orders[0].tolist() == [1, 2, 3, 0]  # distinct speeds: no bit decides
    assert not reads.any()
    # The other way round: battler 2 uses the item, battler 1 switches: battler order again.
    assert run(tables, s, [0, 5, 10, 0])[0][0].tolist() == [1, 2, 3, 0]


def test_swap_loop_orders_four_fighters_by_speed(tables, names):
    """Every speed order of four distinct speeds comes out fastest first (the swap loop)."""
    base = four(tables, names)
    for fastest_first in itertools.permutations(range(4)):
        values = [0] * 4
        for rank, b in enumerate(fastest_first):
            values[b] = 400 - 40 * rank
        s = speeds(base, values)
        order = run(tables, s, [0, 0, 0, 0], ALL_BITS[:1])[0][0].tolist()
        assert order == list(fastest_first), fastest_first


def test_priority_beats_speed_between_fighters_only(tables, names):
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Tackle"])
    s = with_moves(s, names, 0, [QUICK_ATTACK])  # slowest, +1 priority
    s = speeds(s, (50, 300, 200, 100))
    order = run(tables, s, [0, 0, 0, 0], ALL_BITS[:1])[0][0].tolist()
    assert order == [0, 1, 2, 3]
    # Two switchers compare by speed with the priority skipped (:779-780): battler 1 is faster.
    two_switches = run(tables, s, [4 + 3, 4 + 3, 0, 0], ALL_BITS[:1])[0][0].tolist()
    assert two_switches == [1, 0, 2, 3]
    # The priorities in the model, for a mixed turn (priority only in the FIGHT pair).
    check_against_model(tables, s, [0, 7, 0, 0], priority=[1, 0, 0, 0])


def test_all_equal_speed_all_64_tie_rows(tables, names):
    """Four fighters at equal speed: each value of ``order_tie`` gives the order of the loop of
    bcp.c:773-786 (a nonzero bit swaps), every bit read."""
    s = speeds(four(tables, names), (150, 150, 150, 150))
    orders, reads, _ = check_against_model(tables, s, [0, 0, 0, 0])
    assert reads.all()
    for bits, order in zip(ALL_BITS.tolist(), orders.tolist(), strict=True):
        expected, p = [0, 1, 2, 3], 0
        for i in range(3):
            for j in range(i + 1, 4):
                if bits[p]:
                    expected[i], expected[j] = expected[j], expected[i]
                p += 1
        assert order == expected, bits


def test_unread_bits_change_nothing(tables, names):
    """A pair with different commands, or one that meets a dead battler, draws no bit: flipping
    any unread bit gives the same order (equal states stay equal)."""
    base = speeds(four(tables, names), (150, 150, 150, 150))
    cases = [
        (base, [0, 5, 0, 0]),  # battler 1 switches: pairs (0,1) (1,2) (1,3)-like meetings unread
        (base, [10, 0, 5, 0]),  # an item, a fighter, a switch, a fighter
        (with_hp(base, 3, 0), [0, 0, 0, 0]),  # battler 3 absent
        (with_hp(with_hp(base, 1, 0), 2, 0), [0, 0, 0, 0]),  # two absent
    ]
    for state, acts in cases:
        orders, reads, _ = check_against_model(tables, state, acts)
        assert not reads.all()
        index = {tuple(b): i for i, b in enumerate(ALL_BITS.tolist())}
        alive = alive_of(state)  # the order between two absent entries is unobservable (rule 7)
        for row, bits in enumerate(ALL_BITS.tolist()):
            for p in range(6):
                if not reads[row][p]:
                    flipped = list(bits)
                    flipped[p] ^= 1
                    other = orders[index[tuple(flipped)]].tolist()
                    assert [b for b in other if alive[b]] == [
                        b for b in orders[row].tolist() if alive[b]
                    ]


def test_absent_battler_never_precedes_a_living_one(tables, names):
    """Battler 2 at 0 HP and the fastest: it goes last of the fighters in every row and no bit of
    a pair it is in is read."""
    s = speeds(with_hp(four(tables, names), 2, 0), (100, 200, 400, 300))
    orders, reads, _ = check_against_model(tables, s, [0, 0, 0, 0])
    assert (orders[:, 3] == 2).all()
    assert orders[0].tolist() == [3, 1, 0, 2]
    assert not reads.any()  # distinct speeds among the living, so no draw at all
    # Equal speeds among the living: the dead battler's pairs are the unread ones.
    t = speeds(s, (150, 150, 400, 150))
    orders, reads, _ = check_against_model(tables, t, [0, 0, 0, 0])
    assert (orders[:, 3] == 2).all()


def test_absent_battler_is_a_fight_entry_whatever_its_stale_command(tables, names):
    """The group of an absent battler is FIGHT (rule 7): with battler 2 absent, a switch of
    battler 1 still goes first and battler 2 is last."""
    s = speeds(with_hp(four(tables, names), 2, 0), (100, 200, 400, 300))
    assert run(tables, s, [0, 5, 0, 0], ALL_BITS[:1])[0][0].tolist() == [1, 3, 0, 2]
    assert run(tables, s, [0, 5, 10, 0], ALL_BITS[:1])[0][0].tolist() == [1, 3, 0, 2]


def test_quick_claw_holder_goes_first_and_is_flagged(tables, names):
    claw = names.item("Quick Claw")
    s = speeds(four(tables, names), (150, 150, 150, 150))
    s = item_of(s, 3, claw)._replace(qc_roll=jnp.asarray([0, 0, 0, 1], jnp.int8))
    acts = [0, 0, 0, 0]
    orders, reads, flags = check_against_model(tables, s, acts, claw=[False, False, False, True])
    assert (orders[:, 0] == 3).all()
    assert flags.tolist() == [False, False, False, True]
    # The same holder switching: no FIGHT compare, no flag (and it goes first as a switch).
    switching = run(tables, s, [0, 0, 0, 5], ALL_BITS[:1])
    assert switching[0][0][0] == 3 and not switching[2].any()
    # A holder whose only FIGHT partners are absent: its compares draw nothing, no flag.
    lone = with_hp(with_hp(with_hp(s, 0, 0), 1, 0), 2, 0)
    assert not run(tables, lone, acts, ALL_BITS[:1])[2].any()
    # A qc_roll of 0 is no activation.
    cold = s._replace(qc_roll=jnp.zeros(4, jnp.int8))
    assert not run(tables, cold, acts, ALL_BITS[:1])[2].any()


def test_two_claw_holders_compare_plain_speed(tables, names):
    claw = names.item("Quick Claw")
    s = speeds(four(tables, names), (100, 300, 200, 150))
    s = item_of(item_of(s, 0, claw), 2, claw)._replace(qc_roll=jnp.asarray([1, 0, 1, 0], jnp.int8))
    orders, _, flags = check_against_model(tables, s, [0, 0, 0, 0], claw=[True, False, True, False])
    assert orders[0].tolist() == [2, 0, 1, 3]  # the two holders first, the faster of them first
    assert flags.tolist() == [True, False, True, False]


def test_custap_in_its_pinch_is_flagged(tables, names):
    custap = names.item("Custap Berry")
    s = speeds(four(tables, names), (150, 150, 150, 150))
    s = with_hp(item_of(s, 2, custap), 2, 1)
    acts = [0, 0, 0, 0]
    orders, _, _ = check_against_model(tables, s, acts, claw=[False, False, True, False])
    assert (orders[:, 0] == 2).all()
    _, _, claw_f, custap_f = run(tables, s, acts, ALL_BITS[:1])
    assert custap_f.tolist() == [False, False, True, False]
    assert not claw_f.any()
    # Switching: no compare of FIGHT commands, no flag; not in its pinch: none.
    assert not run(tables, s, [0, 0, 5, 0], ALL_BITS[:1])[3].any()
    healthy = with_hp(s, 2, int(s.max_hp[party_of(s, 2), int(s.active[2])]))
    assert not run(tables, healthy, acts, ALL_BITS[:1])[3].any()


def test_quick_claw_display_counts_the_battlers_still_to_act(tables, names):
    """The display needs the count of not-MOVE_END battlers > 1 (bcp.c:881-886, the actor
    included; subscript_check_quick_claw.s:5-19): a Quick Claw flag shows its animation unless the
    holder acts last of the living, a Custap flag always shows its animation and its message only
    when others still wait. The flag is the one a former turn left standing (SF2 (k))."""
    claw, custap = names.item("Quick Claw"), names.item("Custap Berry")
    base = four(tables, names)
    for b in range(4):
        base = with_moves(base, names, b, ["Tackle"])
    base = speeds(base, (400, 300, 200, 100))  # battler 3 acts last
    for item, lone_bit, others_bit in ((claw, 0, 1), (custap, 1, 3)):
        s = item_of(base, 3, item)._replace(claw_flag=jnp.asarray([0, 0, 0, 1], jnp.int8))
        if item == custap:
            s = s._replace(
                claw_flag=jnp.zeros(4, jnp.int8), custap_flag=jnp.asarray([0, 0, 0, 1], jnp.int8)
            )
        _, shown = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=(1, 0, 3, 2))
        assert int(shown["quick_claw"][3]) == lone_bit, item  # acts last: waitingBattlers == 1
        # Battler 3 fastest: three battlers still wait when it starts.
        faster = speeds(s, (100, 200, 300, 400))
        _, shown = play(tables, faster, [0, 0, 0, 0], (3, 2, 1, 0), targets=(1, 0, 3, 2))
        assert int(shown["quick_claw"][3]) == others_bit, item
        # With every other battler absent the holder is the last to act: nothing shown.
        alone = with_hp(with_hp(with_hp(s, 0, 0), 1, 0), 2, 0)
        _, shown = play(tables, alone, [0, 0, 0, 0], (3, 0, 1, 2), targets=(3, 0, 1, 2))
        assert int(shown["quick_claw"][3]) == lone_bit, item


# --- B = 2 unchanged ------------------------------------------------------------------------


def _old_order(tables, state, actions, outcome):
    """The B = 2 ``action_order`` before D3-3 (the base's body)."""
    actions = jnp.asarray(actions, jnp.int32)
    switch, item = is_switch(actions), is_item(actions)
    early = switch | item
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in range(2)])
    both_moves = ~early[0] & ~early[1]
    priority = jnp.where(both_moves, tables.move_priority[moves].astype(jnp.int32), 0)
    second = player_second(tables, state, outcome.speed_tie, state.qc_roll)
    player_second_ = (priority[0] < priority[1]) | ((priority[0] == priority[1]) & second)
    same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
    player_second_ = jnp.where(same_kind, player_second_, early[1] & ~early[0])
    first = player_second_.astype(jnp.int8)
    return jnp.stack([first, 1 - first]).astype(jnp.int8)


def _old_flags(tables, state, actions):
    actions = jnp.asarray(actions, jnp.int32)
    fight = ~is_switch(actions) & ~is_item(actions)
    custap = custap_active(tables, state) & fight[0] & fight[1]
    others = jnp.sum(fight.astype(jnp.int32)) - fight.astype(jnp.int32)
    claw = quick_claw_active(tables, state, state.qc_roll) & fight & (others > 0)
    return claw, custap


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_two_battler_order_and_flags_are_unchanged(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome(SINGLES))
    assert o.order_tie.shape == (0,) and o.sort_tie.shape == (3, 0)
    new = action_order(tables, s, acts, o)
    assert new.dtype == jnp.int8 and new.shape == (2,)
    assert new.tolist() == _old_order(tables, s, acts, o).tolist()
    claw, custap = _old_flags(tables, s, acts)
    assert claw_flags(tables, s, acts).tolist() == claw.tolist()
    assert custap_flags(tables, s, acts).tolist() == custap.tolist()
    assert action_order_reads(tables, s, acts, o).shape == (0,)


def test_sampler_draws_bits_only_at_four(tables, names):
    """The tie arrays hold bits at B = 4 and have no element at B = 2 (no key consumed there:
    ``speed_tie`` keeps its draw); ``speed_tie`` / ``end_speed_tie`` are 0 at B = 4."""
    s4 = four(tables, names)
    o = sample_outcome(jax.random.PRNGKey(7), tables, s4, jnp.zeros(4, jnp.int32), replacement=False)
    assert o.order_tie.shape == (6,) and o.sort_tie.shape == (5, 6)
    assert set(np.asarray(o.order_tie).tolist()) <= {0, 1}
    assert int(o.speed_tie) == 0 and int(o.end_speed_tie) == 0
    seen = {
        tuple(
            np.asarray(
                sample_outcome(jax.random.PRNGKey(k), tables, s4, jnp.zeros(4, jnp.int32), replacement=False).order_tie
            ).tolist()
        )
        for k in range(12)
    }
    assert len(seen) > 4
