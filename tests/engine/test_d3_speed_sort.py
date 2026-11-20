"""D3-4: ``BattleSystem_SortMonSpeedOrder`` before each action (sheet D3-4 rules 1-6, section 5).

Decomp: lib.c:3364-3383 (the sort: ``monSpeedOrder[i] = i``, then every position pair compared
with ``ignoreQuickClaw`` TRUE), lib.c:1188-1454 (the compare), bcp.c:888 (the call before every
entry of ``battlerActionOrder``). ``sort_mon_speed`` is checked against ``model``, a plain-Python
transcription of the 20 lines of lib.c:3364-3383 over a transcription of the compare (priorities
0, the claw bits, the dead short-circuit). The step cases read the sort through its consumers (the
spread first defender, D2-8; the redirection holder, D2-12). Every test that pins a ``sort_tie``
row indexes it by the acting battler, ``order[k]`` (ledger D3-4).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.format import DOUBLES, SINGLES
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.order import PAIRS, effective_speed, sort_mon_speed
from nuzline.engine.outcome import empty_outcome
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_turn_order import ALL_BITS, item_of, speeds
from tests.strategies import any_outcome, legal_action_pair, supported_state

NO_CLAW = jnp.zeros(4, jnp.int8)


@jax.jit
def _sort_rows(tables, state, qc, bits):
    """``(order, reads)`` of ``sort_mon_speed`` for each row of ``bits`` (int8[N, 6])."""
    return jax.vmap(lambda row: sort_mon_speed(tables, state, qc, row))(bits)


def run(tables, state, qc=NO_CLAW, bits=ALL_BITS):
    order, reads = _sort_rows(tables, state, jnp.asarray(qc, jnp.int8), jnp.asarray(bits, jnp.int8))
    return np.asarray(order), np.asarray(reads)


def model(bits, alive, speed, claw=None, lag=None, trick_room=False):
    """lib.c:3364-3383 transcribed over lib.c:1188-1454 (``ignoreQuickClaw``: priorities 0, 0):
    ``(order, reads)``. ``claw[b]``: Quick Claw / Custap active; ``lag[b]``: Lagging Tail."""
    claw = claw or [False] * 4
    lag = lag or [False] * 4

    def compare(b1, b2, bit):
        """0 FASTER, 1 SLOWER, 2 TIE, and whether the bit was drawn."""
        if not alive[b1] and alive[b2]:
            return 1, False
        if alive[b1] and not alive[b2]:
            return 0, False
        s1, s2 = speed[b1], speed[b2]
        if claw[b1] and claw[b2]:
            slower = s1 < s2
        elif claw[b1] != claw[b2]:
            return (1 if claw[b2] else 0), False
        elif lag[b1] and lag[b2]:
            slower = s1 > s2
        elif lag[b1] != lag[b2]:
            return (1 if lag[b1] else 0), False
        elif trick_room:
            slower = s1 > s2
        else:
            slower = s1 < s2
        if slower:
            return 1, False
        tie = s1 == s2
        return (2 if tie and bit else 0), tie

    order = list(range(4))
    reads = []
    p = 0
    for j in range(3):
        for k in range(j + 1, 4):
            r, read = compare(order[j], order[k], bits[p])
            reads.append(read)
            if r:
                order[j], order[k] = order[k], order[j]
            p += 1
    return order, reads


def base(tables, names):
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Tackle"])
    return s


def alive_of(state):
    return [hp(state, b) > 0 for b in range(4)]


def speed_of(tables, state):
    return [int(effective_speed(tables, state, b)) for b in range(4)]


def check_model(tables, state, qc=NO_CLAW, **kw):
    orders, reads = run(tables, state, qc)
    for row, bits in enumerate(ALL_BITS):
        order, rd = model(bits, alive_of(state), speed_of(tables, state), **kw)
        assert orders[row].tolist() == order, bits.tolist()
        assert reads[row].tolist() == rd, bits.tolist()
    return orders, reads


def test_distinct_speeds_sort_descending_and_read_no_bit(tables, names):
    s = speeds(base(tables, names), (100, 300, 200, 250))
    orders, reads = run(tables, s)
    assert (orders == [1, 3, 2, 0]).all()
    assert not reads.any()


def test_zero_hp_goes_last_and_reads_no_bit(tables, names):
    """Rule 2: a 0-HP battler goes after a living one with no draw, even if faster; the one
    equal-speed living pair still draws."""
    s = speeds(base(tables, names), (300, 100, 100, 200))
    s = with_hp(s, 0, 0)
    orders, reads = check_model(tables, s)
    assert (orders[:, 3] == 0).all()
    assert reads.any()  # the equal-speed living pair of battlers 1 and 2 draws


def test_all_equal_speeds_all_64_rows_match_the_transcription(tables, names):
    s = speeds(base(tables, names), (150, 150, 150, 150))
    orders, _ = check_model(tables, s)
    assert len({tuple(o) for o in orders.tolist()}) > 4  # a tie bit does change the order


def test_mixed_speeds_and_deaths_match_the_transcription(tables, names):
    for speed in ((100, 100, 200, 200), (90, 120, 90, 120), (80, 80, 80, 300)):
        for dead in ((), (1,), (0, 3), (2,)):
            s = speeds(base(tables, names), speed)
            for b in dead:
                s = with_hp(s, b, 0)
            check_model(tables, s)


def test_quick_claw_holder_goes_first_without_trick_room(tables, names):
    """lib.c:1408-1411: the lone activated Quick Claw is FASTER; ``qc`` is the turn's draw."""
    s = speeds(base(tables, names), (300, 200, 100, 250))
    s = item_of(s, 2, names.item("Quick Claw"))
    orders, _ = check_model(tables, s, qc=[0, 0, 1, 0], claw=[False, False, True, False])
    assert (orders == [2, 0, 3, 1]).all()
    orders, _ = run(tables, s, qc=[0, 0, 0, 0])  # not drawn: plain speed
    assert (orders == [0, 3, 1, 2]).all()
    # under Trick Room too (lib.c:1408-1411 sits above the reversal)
    orders, _ = run(tables, s._replace(trick_room=jnp.int8(3)), qc=[0, 0, 1, 0])
    assert (orders[:, 0] == 2).all()


def test_custap_in_its_pinch_goes_first(tables, names):
    """lib.c:1288-1300: Custap Berry at <= 1/4 HP counts as the flag, on the current HP."""
    s = speeds(base(tables, names), (300, 200, 100, 250))
    s = item_of(s, 2, names.item("Custap Berry"))
    healthy, _ = run(tables, s)
    assert (healthy == [0, 3, 1, 2]).all()
    pinch = with_hp(s, 2, int(s.max_hp[party_of(s, 2), int(s.active[2])]) // 4)
    orders, _ = check_model(tables, pinch, claw=[False, False, True, False])
    assert (orders[:, 0] == 2).all()


def test_lagging_tail_holder_goes_last(tables, names):
    s = speeds(base(tables, names), (300, 200, 100, 250))
    s = item_of(s, 0, names.item("Lagging Tail"))
    orders, _ = check_model(tables, s, lag=[True, False, False, False])
    assert (orders == [3, 1, 2, 0]).all()


def test_trick_room_reverses_the_sort(tables, names):
    s = speeds(base(tables, names), (100, 300, 200, 250))._replace(trick_room=jnp.int8(3))
    orders, _ = check_model(tables, s, trick_room=True)
    assert (orders == [0, 2, 3, 1]).all()


def test_sort_reads_the_current_speeds(tables, names):
    """Rule 3: a Speed stage change between two calls changes the next sort."""
    s = speeds(base(tables, names), (300, 200, 150, 120))
    before, _ = run(tables, s)
    assert (before == [0, 1, 2, 3]).all()
    dropped = s._replace(stages=s.stages.at[1, Stat.SPE].set(-2))  # 200 -> 100
    after, _ = run(tables, dropped)
    assert (after == [0, 2, 3, 1]).all()


# --- the step reads the sort before each action ---------------------------------------------


def _moves(tables, names, assignments):
    s = base(tables, names)
    for b, move in assignments.items():
        s = with_moves(s, names, b, [move])
    return s


def test_rock_slide_first_defender_follows_the_sort_after_a_speed_drop(tables, names):
    """bcp.c:888 / lib.c:1617-1639: battler 0 (300) Scary Faces battler 3 (200 -> 100), below
    battler 1 (150). Battler 2 (120) then uses Rock Slide: its first defender is battler 1 (the
    sort at its action), where the turn-start order would give battler 3 (D2-8)."""
    s = speeds(
        _moves(tables, names, {0: "Scary Face", 1: "Swords Dance", 2: "Rock Slide", 3: "Swords Dance"}),
        (300, 150, 120, 200),
    )
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(3, 0, 1, 0))
    assert int(out.stages[3, Stat.SPE]) == -2
    assert hp(out, 1) < hp(s, 1)
    assert hp(out, 3) == hp(s, 3)


def test_rock_slide_first_defender_without_the_drop_is_the_faster_foe(tables, names):
    """The same turn with Swords Dance in place of Scary Face: battler 3 is still faster than 1."""
    s = speeds(
        _moves(tables, names, {0: "Swords Dance", 1: "Swords Dance", 2: "Rock Slide", 3: "Swords Dance"}),
        (300, 150, 120, 200),
    )
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(3, 0, 1, 0))
    assert hp(out, 3) < hp(s, 3)
    assert hp(out, 1) == hp(s, 1)


def test_lightning_rod_holder_is_the_faster_one_at_the_action(tables, names):
    """D2-12: battlers 2 and 3 hold Lightning Rod. Battler 1 (300) Scary Faces battler 2
    (250 -> 125), below battler 3 (200); then battler 0 (100) Thunderbolts battler 1: the faster
    holder at that action, battler 3, takes it (Platinum's Lightning Rod only redirects: the hit
    damages the holder); the turn-start order gave battler 2."""
    s = speeds(
        _moves(tables, names, {0: "Thunderbolt", 1: "Scary Face", 2: "Swords Dance", 3: "Swords Dance"}),
        (100, 300, 250, 200),
    )
    s = with_ability(with_ability(s, 2, AbilityId.LIGHTNINGROD), 3, AbilityId.LIGHTNINGROD)
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(1, 2, 0, 0))
    assert int(out.stages[2, Stat.SPE]) == -2
    assert hp(out, 3) < hp(s, 3) and hp(out, 2) == hp(s, 2)
    assert hp(out, 1) == hp(s, 1)


def test_the_row_is_the_acting_battlers(tables, names):
    """Ledger D3-4: row b of ``sort_tie`` is the sort before battler b's action. Four equal
    speeds with ``order_tie[5] = 1``: the turn order is 0, 1, 3, 2, so battler 2 acts last, at
    position 3. Its Rock Slide picks its first defender from the sort before its action, whose
    pair (1, 3) bit is ``sort_tie[2, 4]`` (its own row), not ``sort_tie[3, 4]`` (its position)."""
    s = speeds(
        _moves(tables, names, {0: "Swords Dance", 1: "Swords Dance", 2: "Rock Slide", 3: "Swords Dance"}),
        (150, 150, 150, 150),
    )
    results = {}
    for row2, row3 in itertools.product(range(2), repeat=2):
        o = empty_outcome(DOUBLES)
        o = o._replace(order_tie=o.order_tie.at[5].set(1))
        o = o._replace(sort_tie=o.sort_tie.at[2, 4].set(row2).at[3, 4].set(row3))
        out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(1, 0, 1, 0), outcome=o)
        results[(row2, row3)] = (hp(out, 1) < hp(s, 1), hp(out, 3) < hp(s, 3))
    for row2 in range(2):
        assert results[(row2, 0)] == results[(row2, 1)]  # row 3 is read by no battler's action
    assert results[(0, 0)] != results[(1, 0)]  # row 2 picks the first defender


# --- B = 2: unchanged -----------------------------------------------------------------------


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_b2_has_no_sort_and_steps_as_the_reference(tables, names, data):
    """B = 2: ``speed_order`` stays ``arange(2)`` (no call, no ``sort_tie`` element); the engine
    equals the reference, whose ``speed_order`` stays None, on 20 drawn states."""
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome(SINGLES))
    assert o.sort_tie.shape == (3, 0)
    assert_matches_reference(tables, s, acts, o)


def test_pairs_are_the_lexicographic_six():
    assert PAIRS == tuple(itertools.combinations(range(4), 2))
