"""D3-16 (owes D2-17 a): RANGE_OPPONENT_SIDE through the four-actor step. Stealth Rock's defender is
``BattleSystem_RandomOpponent`` (battle_lib.c:1661-1664; one ``RandNext() & 1``, :4177-4198) and the
script sets the hazard on the defender's side: either coin lands it on the user's opposing side.
Real CalcTurnOrder through ``play`` (engine equal to the reference, field for field)."""

from __future__ import annotations

import numpy as np

from nuzline.engine.format import DOUBLES
from nuzline.engine.outcome import empty_outcome
from tests.engine.d2_states import four, with_moves
from tests.engine.test_d3_action_loop import play

ORDER = (0, 1, 2, 3)


def idle(tables, names):
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    return s


def test_stealth_rock_lands_on_the_other_side_whatever_the_coin(tables, names):
    for user, want in ((0, [False, True]), (1, [True, False])):
        s = with_moves(idle(tables, names), names, user, ["Stealth Rock"])
        for coin in (0, 1):
            o = empty_outcome(DOUBLES)
            o = o._replace(target_draw=o.target_draw.at[user].set(coin))
            out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 1, 0), outcome=o)
            assert (np.asarray(out.stealth_rock) != 0).tolist() == want, (user, coin)
