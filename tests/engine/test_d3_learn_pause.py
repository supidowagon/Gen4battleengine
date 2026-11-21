"""D3 finding 8 x E3-B2-21: a partner's KO levels battler 0 up mid-turn at B = 4.

The payout runs inside the KO's MoveEnd (battle_controller_player.c:3968) and its learn prompt
pauses the turn there (``learn_site`` = ``LEARN_MOVE_END``); the answer is ``levelup.learn`` and
the next ``step`` resumes. A FIGHT still to come reads ``moves[moveSlot]`` when its action runs
(FightCommand, bcp.c:1889-1895), so battler 0's pending move is the answered slot's. Every case
runs the whole turn through ``step`` and ``ref_step`` (every field, ``step_both``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nuzline.engine.format import DOUBLES
from nuzline.engine.levelup import GIVE_UP, LEARN_MOVE_END, learn, needs_learn
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import BattleState, initial_arrays
from tests.engine.d2_states import mon, with_moves
from tests.engine.test_d3_owed_follow_me import ordered, step_both
from tests.engine.test_levelup import _base, _mon

IDLE = "Swords Dance"
WATER_GUN_SLOT = 3


def _partner_ko_state(tables, names):
    """DOUBLES: battler 0 a Prinplup one exp short of 19 (BubbleBeam at 19, vanilla learnset)
    with four moves, battler 2 a Bulbasaur with Tackle, battler 1 a 1-HP Geodude, battler 3 a
    Bulbasaur; the opponents idle."""
    prinplup = _mon(
        names,
        "Prinplup",
        18,
        moves=("Bubble", "Roar", "Tackle", "Water Gun"),
        ivs=(3,) * 6,
        exp=_base(tables, names, "Prinplup", 19) - 1,
    )
    player = [prinplup, mon(names, ("Tackle",))] + [mon(names) for _ in range(2)]
    opponent = [_mon(names, "Geodude", 12, moves=(IDLE,)), mon(names, (IDLE,))]
    opponent += [mon(names) for _ in range(2)]
    a = initial_arrays(DOUBLES, tables, [player, opponent], opp2_items=())
    s = BattleState(**{k: jnp.asarray(v) for k, v in a.items()})
    s = with_moves(s, names, 3, [IDLE])
    return s._replace(hp=s.hp.at[1, 0].set(1))  # battler 1: party 1 slot 0


@pytest.mark.parametrize("answer", ["give up", "replace Water Gun"])
def test_partner_ko_pauses_and_battler_0_rereads_the_answered_slot(tables, names, answer):
    """Battler 2 (faster) KOs battler 1; the payout levels battler 0 to 19 with its Water Gun
    (slot 3) still to come: the turn pauses at battler 2's MoveEnd. After the answer battler 0's
    pending FIGHT re-reads slot 3 (bcp.c:1889-1895): BubbleBeam after a replace, which spends its
    PP; Water Gun after giving up. Both engines agree at the pause and after the resume."""
    s = _partner_ko_state(tables, names)
    acts = [WATER_GUN_SLOT, 0, 0, 0]
    targets = (1, 0, 1, 0)
    outcome = empty_outcome(DOUBLES)
    s = ordered(tables, s, acts, outcome, (2, 0, 3, 1))
    paused = step_both(tables, s, acts, outcome, targets)
    assert int(paused.learn_site) == LEARN_MOVE_END and bool(needs_learn(paused))
    assert int(paused.hp[1, 0]) == 0 and int(paused.level[0, 0]) == 19
    assert int(paused.ts_cursor) == 0  # battler 2's position: its MoveEnd paused
    assert int(paused.ts_actions[0]) == WATER_GUN_SLOT
    option = GIVE_UP if answer == "give up" else WATER_GUN_SLOT
    answered = jax.jit(learn)(tables, paused, jnp.int32(option))
    done = step_both(tables, answered, [0, 0, 0, 0], outcome, (0, 0, 0, 0))
    assert int(done.learn_site) == 0 and int(done.turn) == int(s.turn) + 1
    if answer == "give up":
        assert int(done.moves[0, 0, WATER_GUN_SLOT]) == names.move("Water Gun")
    else:
        assert int(done.moves[0, 0, WATER_GUN_SLOT]) == names.move("BubbleBeam")
    assert int(done.pp[0, 0, WATER_GUN_SLOT]) < int(answered.pp[0, 0, WATER_GUN_SLOT])
