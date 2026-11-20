"""D3-18: the reference's Trick Room re-sort (``reference.engine._resort``, the restarting
``_actions``) against the engine, through ``ref_step`` and ``step`` on one B = 4 state each: the
restart, the Quick Claw re-set and a pause after a restart. ``full_step`` and ``play`` assert every
field equal."""

from __future__ import annotations

import numpy as np

from nuzline.engine.format import DOUBLES
from nuzline.engine.outcome import empty_outcome
from tests.engine.test_d3_action_loop import play
from tests.engine.test_d3_pause import resume
from tests.engine.test_d3_trick_room import (
    FIGHTS,
    STATUS_MOVES,
    full_step,
    item_id,
    lineup,
    trick_call,
)
from tests.engine.test_d3_turn_order import item_of


def test_the_restart_matches_on_both_engines(tables, names):
    s = lineup(tables, names, STATUS_MOVES)
    out = full_step(tables, s, FIGHTS, trick_call(tables, names, s))
    assert [int(out.stages[b, 2]) for b in (1, 2, 3)] == [1, 1, 1]
    assert int(out.move_prev) == names.move("Harden")


def test_the_quick_claw_re_set_matches_on_both_engines(tables, names):
    moves = [["Metronome"], ["Harden"], ["Knock Off"], ["Withdraw"]]
    s = item_of(lineup(tables, names, moves), 0, names.item("Quick Claw"))
    base = trick_call(tables, names, s)
    o = base._replace(quick_claw=base.quick_claw.at[0].set(1))
    out = full_step(tables, s, FIGHTS, o)
    assert item_id(out, 0) == names.item("Quick Claw") and int(out.claw_flag[0]) == 1
    out = full_step(tables, s, FIGHTS, empty_outcome(DOUBLES)._replace(call=base.call))
    assert int(out.claw_flag[0]) == 0


def test_a_pause_after_a_restart_matches_on_both_engines(tables, names):
    moves = [["Metronome"], ["Harden"], ["Defense Curl"], ["U-turn"]]
    s = lineup(tables, names, moves)
    paused, _ = play(tables, s, FIGHTS, None, outcome=trick_call(tables, names, s))
    assert np.asarray(paused.ts_order).tolist() == [3, 2, 1, 0]
    assert np.asarray(paused.ts_acted).tolist() == [1, 0, 0, 0]
    out = resume(tables, paused)
    assert int(out.pivot_battler) == 0
