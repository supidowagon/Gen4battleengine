"""D3-14: the turn's start and reset over four battlers (sheet D3-14 rules 1-3, section 4.4).

``BattleSystem_SetupNextTurn`` (battle_lib.c:2228-2253) clears, for each of ``MAX_BATTLERS``, the
flinch (:2234), a sleeping battler's MOVE_LOCKED / THRASH lock (:2240-2247) and both sides'
``followMe`` (:2251-2252); ``CheckPreMoveActions`` (battle_controller_player.c:853-858) draws
``speedRand`` per battler, kept as ``Outcome.quick_claw`` [B] where a Quick Claw can read it
(``step.qc_kept``). No engine code changed: each case plays a whole DOUBLES turn through ``step``
and the reference's ``ref_step`` and asserts every field equal.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import Stat, Status
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.d2_states import four, party_of, with_moves
from tests.engine.helpers import jit_step
from tests.engine.test_d3_action_loop import _all_moves, _set_item

TARGETS = (1, 0, 3, 2)


def turn(tables, state, outcome=None, actions=(0, 0, 0, 0), targets=TARGETS):
    """A whole turn on the engine, asserted equal to the reference's in every field."""
    outcome = empty_outcome(DOUBLES) if outcome is None else outcome
    acts = jnp.asarray(actions, jnp.int32)
    tg = jnp.asarray(targets, jnp.int8)
    out = jit_step(tables, state, acts, outcome, tg)
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in actions),
            outcome_to_py(outcome),
            [int(t) for t in targets],
        )
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def test_flinch_of_battlers_one_and_three_is_cleared_at_the_turn_end(tables, names):
    """Fake Out by battlers 0 and 2 (+3, first) flinches 1 and 3: neither acts (no Swords Dance
    stage), and both ``flinched`` are 0 after the turn (lib.c:2234)."""
    s = _all_moves(four(tables, names), names, "Swords Dance")
    for b in (0, 2):
        s = with_moves(s, names, b, ["Fake Out"])
    s = s._replace(fake_out_turn=s.fake_out_turn.at[0].set(s.turn).at[2].set(s.turn))
    o = empty_outcome(DOUBLES)
    o = o._replace(secondary=o.secondary.at[0, 0].set(1).at[2, 0].set(1))
    out = turn(tables, s, o)
    for b in (1, 3):
        assert int(out.stages[b, Stat.ATK]) == 0, b  # flinched: its Swords Dance never ran
        assert int(out.flinched[b]) == 0, b
    for b in (0, 2):
        assert int(out.flinched[b]) == 0, b


def test_rollout_lock_of_a_sleeper_is_cleared_an_awake_one_is_kept(tables, names):
    """Battler 3 (asleep) and battler 2 (awake) are locked into Rollout: after the turn the
    sleeper's ``locked_move`` is 0 (lib.c:2240-2243), the awake battler's stands."""
    rollout = names.move("Rollout")
    s = _all_moves(four(tables, names), names, "Swords Dance")
    for b in (2, 3):
        s = with_moves(s, names, b, ["Rollout"])
    p3, slot3 = party_of(s, 3), int(s.active[3])
    s = s._replace(
        locked_move=s.locked_move.at[2].set(rollout).at[3].set(rollout),
        rollout_count=s.rollout_count.at[2].set(2).at[3].set(2),
        chosen_target=s.chosen_target.at[2].set(1).at[3].set(0),
        status=s.status.at[p3, slot3].set(Status.SLEEP),
        status_counter=s.status_counter.at[p3, slot3].set(5),
    )
    out = turn(tables, s, targets=(1, 0, 1, 0))
    assert int(out.status[p3, slot3]) == Status.SLEEP  # still asleep: the unlock is the sleeper's
    assert int(out.locked_move[3]) == 0
    assert int(out.locked_move[2]) == rollout


def test_follow_me_of_battler_one_is_cleared(tables, names):
    """Follow Me by battler 1 writes ``ts_follow_me[1]`` (D2-11); it is 0 after the turn
    (lib.c:2251-2252)."""
    s = _all_moves(four(tables, names), names, "Swords Dance")
    s = with_moves(s, names, 1, ["Follow Me"])
    out = turn(tables, s)
    assert np.asarray(out.ts_follow_me).tolist() == [0, 0]


def test_quick_claw_draw_is_kept_per_battler(tables, names):
    """``quick_claw = [1, 0, 1, 1]`` with Quick Claw holders on 0, 2 and 3: ``qc_roll`` keeps it
    per battler (``qc_kept``); a draw of 1 for battler 1, which holds none, is not kept."""
    claw = names.item("Quick Claw")
    s = _all_moves(four(tables, names), names, "Swords Dance")
    for b in (0, 2, 3):
        s = _set_item(s, b, claw)
    o = empty_outcome(DOUBLES)
    kept = turn(tables, s, o._replace(quick_claw=jnp.asarray([1, 0, 1, 1], o.quick_claw.dtype)))
    assert np.asarray(kept.qc_roll).tolist() == [1, 0, 1, 1]
    every = turn(tables, s, o._replace(quick_claw=jnp.asarray([1, 1, 1, 1], o.quick_claw.dtype)))
    assert np.asarray(every.qc_roll).tolist() == [1, 0, 1, 1]
