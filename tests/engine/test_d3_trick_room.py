"""D3-18: Trick Room's MoveEnd re-sort over four (sheet D3-18 rules 1-10, section 5).

Decomp: eff/effect_script_0259.s:18 and script.c:7859-7866 (the trigger), bcp.c:3989-3998 (the
entry's MoveEnd: ``SortMonActionOrder`` :3991, ``turnOrderCounter = 0`` :3993), lib.c:7276-7300 (the
re-sort: position pairs, equal SELECTED_COMMAND only, ``ignoreQuickClaw = command != FIGHT``),
lib.c:1188-1454 (the compare), bcp.c:889-893 (the walk from position 0 again, MOVE_END entries run
MoveEnd only), subscript_check_quick_claw.s / subscript_knock_off.s:8 / script.c:8125 (the flags
the re-sort sets again).

The cases are DOUBLES (``d2_states.four``). Battler 0 calls Trick Room through Metronome (the
``Outcome.call`` rank of Trick Room among its candidates; priority 0), so battlers 1-3 are still to
act. Each battler has its own self-targeting move, so the last actor shows in ``move_prev`` and the
battlers that acted in their Defense stage. ``play`` is the action phase on both engines (every
field equal); ``full_step`` is the whole turn on both.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, SINGLES
from nuzline.engine.ids import TRICK_ROOM_TURNS, AbilityId, Stat
from nuzline.engine.order import effective_speed, metronome_candidates, resort_action_order
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import step, turn_events
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_d3_action_loop import hp, play, speed_ordered
from tests.engine.test_d3_pause import resume, same_state
from tests.engine.test_d3_turn_order import ALL_BITS, item_of, speeds
from tests.strategies import any_outcome, legal_action_pair, supported_state

DEF = Stat.DEF
NO_TARGETS = (0, 0, 0, 0)
FIGHTS = [0, 0, 0, 0]
_jit_step = jax.jit(step)


def lineup(tables, names, moves):
    """DOUBLES, battler ``b`` knows ``moves[b]`` (names); Speed 400, 360, 320, 280 (0 fastest)."""
    s = four(tables, names, DOUBLES)
    for b, known in enumerate(moves):
        s = with_moves(s, names, b, known)
    return speed_ordered(s, [0, 1, 2, 3])


def trick_call(tables, names, state, b=0, **fields):
    """An outcome in which battler ``b``'s Metronome calls Trick Room (its rank among the
    candidates, ``order.invoked_move``), with ``fields`` replaced."""
    cands = np.asarray(metronome_candidates(tables, state, b))
    move = names.move("Trick Room")
    assert cands[move]
    outcome = empty_outcome(DOUBLES)
    return outcome._replace(call=outcome.call.at[b].set(int(cands[:move].sum())), **fields)


def full_step(tables, state, actions, outcome, targets=NO_TARGETS):
    """The whole turn on both engines, every field asserted equal; returns the engine state."""
    acts = jnp.asarray(actions, jnp.int32)
    out = _jit_step(tables, state, acts, outcome, jnp.asarray(targets, jnp.int8))
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in actions),
            outcome_to_py(outcome),
            list(targets),
        )
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def item_id(state, b):
    return int(state.item[party_of(state, b), int(state.active[b])])


STATUS_MOVES = [["Metronome"], ["Harden"], ["Defense Curl"], ["Withdraw"]]


# --- the outcome field ----------------------------------------------------------------------


def test_resort_tie_has_one_row_per_battler_and_none_at_b2():
    assert empty_outcome(DOUBLES).resort_tie.shape == (4, 6)
    assert empty_outcome(SINGLES).resort_tie.shape == (2, 0)


# --- the restart ----------------------------------------------------------------------------


def test_a_called_trick_room_restarts_the_walk_in_the_new_order(tables, names):
    """Speeds 0 > 1 > 2 > 3, nothing up. Battler 0's Metronome calls Trick Room (the call draw set
    to it): the re-sort puts the slowest first (3, 2, 1, 0), the walk restarts at position 0 and
    battler 0 (MOVE_END) is skipped: the actions run 0, 3, 2, 1 and the last actor is battler 1.
    Each of 1, 2 and 3 acted once. With Splash the order stays 0, 1, 2, 3."""
    s = lineup(tables, names, STATUS_MOVES)
    out, _ = play(tables, s, FIGHTS, None, outcome=trick_call(tables, names, s))
    assert int(out.trick_room) == TRICK_ROOM_TURNS
    assert [int(out.stages[b, DEF]) for b in (1, 2, 3)] == [1, 1, 1]
    assert int(out.move_prev) == names.move("Harden")  # battler 1 acted last
    calm = lineup(tables, names, [["Splash"], *STATUS_MOVES[1:]])
    out, _ = play(tables, calm, FIGHTS, None)
    assert int(out.trick_room) == 0
    assert int(out.move_prev) == names.move("Withdraw")  # battler 3 acted last


def test_a_second_activation_re_sorts_again_with_its_own_row(tables, names):
    """Battlers 2 and 3 select Trick Room, battler 0 calls it: three toggles (0 on, 3 off, 2 on).
    After battler 0's re-sort the order is 1, 0, 3, 2 (priority first, then slowest first under
    Trick Room); battler 3's re-sort (Trick Room off) leaves battler 2 to act. The re-sort after
    battler 2's Trick Room reads row 2 of ``resort_tie``, and nobody is left to act: flipping it
    changes nothing."""
    s = lineup(tables, names, [["Metronome"], ["Harden"], ["Trick Room"], ["Trick Room"]])
    base = trick_call(tables, names, s)
    out, _ = play(tables, s, FIGHTS, None, outcome=base)
    assert int(out.trick_room) == TRICK_ROOM_TURNS  # the last toggle left it on
    assert int(out.stages[1, DEF]) == 1
    assert int(out.move_prev) == names.move("Trick Room")
    flipped = base._replace(resort_tie=base.resort_tie.at[2].set(1))
    other, _ = play(tables, s, FIGHTS, None, outcome=flipped)
    same_state(out, other)


def test_the_tie_row_of_the_first_re_sort_decides_the_order(tables, names):
    """Four equal Speeds: the turn order is 0, 1, 2, 3 (``order_tie`` zero). After battler 0's
    called Trick Room every pair is a tie: all bits 0 leave 0, 1, 2, 3 (battler 3 acts last); all
    bits 1 swap every pair (3, 2, 1, 0: battler 1 acts last). Row 0 is read, no other row."""
    s = speeds(lineup(tables, names, STATUS_MOVES), (150, 150, 150, 150))
    base = trick_call(tables, names, s)
    last = {}
    for bit in (0, 1):
        o = base._replace(resort_tie=base.resort_tie.at[0].set(bit))
        out, _ = play(tables, s, FIGHTS, None, outcome=o)
        last[bit] = int(out.move_prev)
    assert last[0] == names.move("Withdraw") and last[1] == names.move("Harden")
    wrong_row = base._replace(resort_tie=base.resort_tie.at[1:].set(1))
    out, _ = play(tables, s, FIGHTS, None, outcome=wrong_row)
    assert int(out.move_prev) == names.move("Withdraw")  # rows 1-3 are the other battlers'


def test_the_sort_before_an_action_reads_the_acting_battlers_row(tables, names):
    """Equal Speeds; battler 0's called Trick Room re-sorts to 3, 2, 1, 0 (row 0 all ones), so
    battler 2 acts at position 1. Its Rock Slide takes its first defender from the sort before its
    action, whose pair (1, 3) bit is ``sort_tie[2, 4]`` (its row), not ``sort_tie[1, 4]`` (its
    position after the restart)."""
    moves = [["Metronome"], ["Swords Dance"], ["Rock Slide"], ["Swords Dance"]]
    s = speeds(lineup(tables, names, moves), (150, 150, 150, 150))
    base = trick_call(tables, names, s)
    base = base._replace(resort_tie=base.resort_tie.at[0].set(1))
    results = {}
    for row2, row1 in itertools.product(range(2), repeat=2):
        o = base._replace(sort_tie=base.sort_tie.at[2, 4].set(row2).at[1, 4].set(row1))
        out, _ = play(tables, s, FIGHTS, None, targets=(0, 0, 1, 0), outcome=o)
        results[(row2, row1)] = (hp(out, 1) < hp(s, 1), hp(out, 3) < hp(s, 3))
    for row2 in range(2):
        assert results[(row2, 0)] == results[(row2, 1)]
    assert results[(0, 0)] != results[(1, 0)]


# --- the compare (lib.c:7276-7300) against a transcription ----------------------------------


@jax.jit
def _resorts(tables, state, order, real, qc, bits):
    """``(order, claw_set, custap_set, read)`` of ``resort_action_order`` for each row of ``bits``."""
    return jax.vmap(lambda row: resort_action_order(tables, state, order, real, qc, row))(bits)


def model_resort(order, bits, alive, speed, trick_room):
    """lib.c:7276-7300 over lib.c:1188-1454 for four FIGHT battlers, no claws, priorities equal:
    ``(order, reads)``."""
    order = list(order)
    reads = []
    p = 0
    for i in range(3):
        for j in range(i + 1, 4):
            b1, b2 = order[i], order[j]
            if alive[b1] != alive[b2]:
                swap, read = alive[b2], False
            else:
                s1, s2 = speed[b1], speed[b2]
                slower = s1 > s2 if trick_room else s1 < s2
                tie = s1 == s2
                swap, read = slower or (tie and bool(bits[p])), tie and alive[b1] and alive[b2]
            reads.append(read)
            if swap:
                order[i], order[j] = b2, b1
            p += 1
    return order, reads


def _check_resort(tables, state, start, trick_room):
    s = state._replace(trick_room=jnp.asarray(5 if trick_room else 0, state.trick_room.dtype))
    orders, cs, us, reads = _resorts(
        tables,
        s,
        jnp.asarray(start, jnp.int32),
        jnp.zeros(4, jnp.int32),
        jnp.zeros(4, jnp.int8),
        jnp.asarray(ALL_BITS, jnp.int8),
    )
    alive = [hp(s, b) > 0 for b in range(4)]
    speed = [int(effective_speed(tables, s, b)) for b in range(4)]
    for row, bits in enumerate(ALL_BITS):
        order, rd = model_resort(start, bits, alive, speed, trick_room)
        assert np.asarray(orders[row]).tolist() == order, (start, bits.tolist())
        assert np.asarray(reads[row]).tolist() == rd, (start, bits.tolist())
    assert not np.asarray(cs).any() and not np.asarray(us).any()  # no holder: no flag


def test_resort_action_order_matches_the_transcription_over_all_64_rows(tables, names):
    """The pairs run over the current order (here not the identity), every battler takes part; a
    pair with a 0-HP battler reads no bit (the dead one goes after a living one, lib.c:1204-1211)."""
    base = lineup(tables, names, [["Tackle"]] * 4)
    for speed in ((150, 150, 150, 150), (100, 100, 200, 200), (90, 120, 90, 120)):
        for dead in ((), (1,), (0, 3)):
            s = speeds(base, speed)
            for b in dead:
                s = with_hp(s, b, 0)
            for start in ((0, 1, 2, 3), (2, 0, 3, 1)):
                for trick_room in (False, True):
                    _check_resort(tables, s, start, trick_room)


# --- the flags the re-sort sets again -------------------------------------------------------


def test_quick_claw_is_set_again_and_knock_off_on_the_holder_fails(tables, names):
    """Battler 0 holds a Quick Claw (``qc_roll[0] = 0``: no flag at the turn start) and its
    ``Outcome.quick_claw[0] = 1`` (this turn's speedRand). After its called Trick Room the re-sort
    sets ``quickClaw`` again (lib.c:1283, :1349); battler 2's Knock Off on it fails
    (subscript_knock_off.s:8) and the flag stands into the next turn. With ``quick_claw[0] = 0``
    the Knock Off removes the item and no flag is set."""
    moves = [["Metronome"], ["Harden"], ["Knock Off"], ["Withdraw"]]
    s = item_of(lineup(tables, names, moves), 0, names.item("Quick Claw"))
    claw = names.item("Quick Claw")
    base = trick_call(tables, names, s)
    on = base._replace(quick_claw=base.quick_claw.at[0].set(1))
    out, _ = play(tables, s, FIGHTS, None, outcome=on)
    assert item_id(out, 0) == claw and int(out.claw_flag[0]) == 1
    out, _ = play(tables, s, FIGHTS, None, outcome=base)
    assert item_id(out, 0) == 0 and int(out.claw_flag[0]) == 0


def custap_line(tables, names, rival_moves):
    """Battler 1 holds a Custap Berry with Gluttony (the pinch is half its HP) and sits just above
    it; battler 0 Tackles it into the pinch; battler 3 Plucks it. Battler 1's slot 0 is Metronome."""
    moves = [["Tackle"], rival_moves, ["Swords Dance"], ["Pluck"]]
    s = lineup(tables, names, moves)
    s = item_of(with_ability(s, 1, AbilityId.GLUTTONY), 1, names.item("Custap Berry"))
    p, slot = party_of(s, 1), int(s.active[1])
    return with_hp(s, 1, int(s.max_hp[p, slot]) // 2 + 5)


def test_custap_is_set_again_and_pluck_on_the_holder_fails(tables, names):
    """Battler 0 Tackles battler 1 (Custap, into its pinch) at position 0; battler 1 calls Trick
    Room at position 1; the re-sort sets ``custapBerry`` (lib.c:1297, :1363) and battler 3's Pluck
    fails (script.c:8125): the berry stays, ``custap_flag[1] == 1``. With Swords Dance in place of
    the call the flag is never set and the Pluck eats the berry."""
    s = custap_line(tables, names, ["Metronome", "Swords Dance"])
    custap = names.item("Custap Berry")
    p, slot = party_of(s, 1), int(s.active[1])
    targets = (1, 0, 0, 1)
    out, _ = play(tables, s, FIGHTS, None, targets=targets, outcome=trick_call(tables, names, s, 1))
    assert 0 < hp(out, 1) <= int(s.max_hp[p, slot]) // 2  # Tackled into the pinch
    assert item_id(out, 1) == custap and int(out.custap_flag[1]) == 1
    out, _ = play(tables, s, [0, 1, 0, 0], None, targets=targets)
    assert item_id(out, 1) == 0 and int(out.custap_flag[1]) == 0


def test_the_custap_flag_is_eaten_at_the_holders_next_move(tables, names):
    """The next turn (battler 1's HP back above the pinch, battler 0 switching): battler 1's move
    start removes the berry and shows the Custap event (subscript_check_quick_claw.s:5, :12-19);
    the flag is 0 after."""
    s = custap_line(tables, names, ["Metronome", "Swords Dance"])
    custap = names.item("Custap Berry")
    p, slot = party_of(s, 1), int(s.active[1])
    after = full_step(tables, s, FIGHTS, trick_call(tables, names, s, 1), (1, 0, 0, 1))
    assert item_id(after, 1) == custap and int(after.custap_flag[1]) == 1
    nxt = with_hp(after, 1, int(s.max_hp[p, slot]))
    nxt = nxt._replace(qc_roll=jnp.zeros_like(nxt.qc_roll))
    acts = [4 + 2, 1, 0, 0]  # battler 0 switches to slot 2; battler 1 Swords Dance
    out = full_step(tables, nxt, acts, empty_outcome(DOUBLES))
    assert item_id(out, 1) == 0 and int(out.custap_flag[1]) == 0
    events = jax.jit(turn_events)(
        tables, nxt, jnp.asarray(acts, jnp.int32), empty_outcome(DOUBLES), jnp.zeros(4, jnp.int8)
    )
    assert int(events["quick_claw"][1]) & 1


def test_no_flag_without_another_fighter_alive(tables, names):
    """Rule 5: the flag needs another battler with FIGHT and HP at the re-sort. Battlers 1 and 3
    switch (PARTY, they act first) and battler 2 is at 0 HP: battler 0's Quick Claw sets none.
    With battler 2 alive (Withdraw) the pair (0, 2) sets it."""
    moves = [["Metronome"], ["Harden"], ["Withdraw"], ["Harden"]]
    s = item_of(lineup(tables, names, moves), 0, names.item("Quick Claw"))
    base = trick_call(tables, names, s)
    o = base._replace(quick_claw=base.quick_claw.at[0].set(1))
    acts = [0, 4 + 2, 0, 4 + 3]
    out, _ = play(tables, with_hp(s, 2, 0), acts, None, outcome=o)
    assert int(out.claw_flag[0]) == 0
    out, _ = play(tables, s, acts, None, outcome=o)
    assert int(out.claw_flag[0]) == 1


def test_the_carried_flag_is_cleared_by_a_switch_out_and_by_a_faint(tables, names):
    """A flag carried into the next turn (see the Quick Claw case) is cleared when its holder
    switches out (the newcomer's flags are 0, lib.c:2100-2104) or faints (lib.c:2185-2188)."""
    moves = [["Metronome", "Swords Dance"], ["Tackle"], ["Withdraw"], ["Withdraw"]]
    s = item_of(lineup(tables, names, moves), 0, names.item("Quick Claw"))
    base = trick_call(tables, names, s)
    o = base._replace(quick_claw=base.quick_claw.at[0].set(1))
    carried = full_step(tables, s, FIGHTS, o)
    assert int(carried.claw_flag[0]) == 1
    carried = carried._replace(qc_roll=jnp.zeros_like(carried.qc_roll))
    out = full_step(tables, carried, [4 + 2, 0, 0, 0], empty_outcome(DOUBLES))
    assert int(out.claw_flag[0]) == 0  # switched out
    # Trick Room is still up: battler 1 acts before battler 0 and its Tackle finishes it.
    weak = with_hp(carried, 0, 1)
    out = full_step(tables, weak, [1, 0, 0, 0], empty_outcome(DOUBLES), targets=(0, 0, 0, 0))
    assert hp(out, 0) == 0 and int(out.claw_flag[0]) == 0  # fainted


# --- a pause after a restart ----------------------------------------------------------------


def test_a_pause_after_a_restart_keeps_the_new_order_and_who_acted(tables, names):
    """Battler 0's called Trick Room re-sorts to 3, 2, 1, 0; battler 3's U-turn pauses at position
    0: ``ts_order`` is the re-sorted order, ``ts_cursor`` 0, ``ts_acted`` holds battler 0 (it sits
    after the cursor). The resume runs battlers 2 and 1 only, each once."""
    moves = [["Metronome"], ["Harden"], ["Defense Curl"], ["U-turn"]]
    s = lineup(tables, names, moves)
    paused, _ = play(tables, s, FIGHTS, None, outcome=trick_call(tables, names, s))
    assert int(paused.pivot_battler) == 4
    assert int(paused.ts_cursor) == 0
    assert np.asarray(paused.ts_order).tolist() == [3, 2, 1, 0]
    assert np.asarray(paused.ts_acted).tolist() == [1, 0, 0, 0]
    out = resume(tables, paused)
    assert [int(out.stages[b, DEF]) for b in (1, 2)] == [1, 1]
    assert int(out.pivot_battler) == 0


# --- Kaizo's re-arm -------------------------------------------------------------------------


def test_the_kaizo_re_arm_runs_no_move_end_and_no_re_sort(tables, names, kaizo_tables, kaizo_names):
    """Rule 10: with the counter at 1 and nobody using Trick Room, the turn-end pass re-arms it
    (``Tables.trick_room_rearm``) with no MoveEnd after it: the turn's order is the one
    CalcTurnOrder gave (reversed Speeds, Trick Room up), and the field is up again; the next turn
    starts from a fresh CalcTurnOrder (the engine equals the reference on both)."""
    s = lineup(kaizo_tables, kaizo_names, [["Swords Dance"]] * 4)
    s = s._replace(trick_room=jnp.asarray(1, s.trick_room.dtype))
    out = full_step(kaizo_tables, s, FIGHTS, empty_outcome(DOUBLES))
    assert int(out.trick_room) == TRICK_ROOM_TURNS
    assert [int(out.stages[b, Stat.ATK]) for b in range(4)] == [2, 2, 2, 2]
    again = full_step(kaizo_tables, out, FIGHTS, empty_outcome(DOUBLES))
    assert int(again.trick_room) == TRICK_ROOM_TURNS


# --- B = 2 ----------------------------------------------------------------------------------


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_b2_with_trick_room_in_a_slot_steps_as_the_reference(tables, names, data):
    """B = 2: the order is not re-sorted (``resort_tie`` has no element) and the flags are SF2
    (k)'s; the engine equals the reference on 20 drawn states with Trick Room in slot 0."""
    s = data.draw(supported_state(tables, names))
    tr = names.move("Trick Room")
    for side in (0, 1):
        mon = int(s.active[side])
        s = s._replace(moves=s.moves.at[side, mon, 0].set(tr), pp=s.pp.at[side, mon, 0].set(5))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome(SINGLES))
    assert_matches_reference(tables, s, acts, o)
