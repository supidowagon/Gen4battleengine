"""D3-15 x F-O3 (batch G): a Starf Berry eaten inside an action is a read of the turn's queue
entry 0 (``enumerate._starf_reads``; battle_lib.c:5286-5288), in both formats.

A lead at its pinch cannot start a turn with its Starf uneaten (ShowBattleMon's pass,
battle_controller_player.c:254-267, battle_lib.c:4151-4165, ``step._turn_start_items``), so each
board reaches the pinch during the turn:

- B = 2, at the foe's action: the holder starts one HP above its pinch (curHP <= maxHP / 4,
  battle_lib.c:4850-4855); the faster foe's Tackle lands it there and the holder, the defender,
  eats in that action's AfterMoveEffects (bcp.c:3672, :3691), before its own move; its own
  ATTACKER_ITEM then finds no berry.
- B = 4, at its own MoveEnd: battler 0's confusion self-hit (bcp.c:2629-2639) lands it at the
  pinch and skips AfterMoveEffects, so the berry is eaten by MoveEnd's TriggerEffectOnSwitch
  (bcp.c:3973; SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS over monSpeedOrder, lib.c:4151-4165; D3-15).
  A battler that is neither attacker nor defender has no such board: every action that reaches
  AfterMoveEffects checks every battler's item there (bcp.c:3672, :3730-3756), and the paths that
  skip it (a self-hit, a crash, a failed move) change only the attacker's HP.

Every run is checked against the reference engine."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.damage import confusion_damage
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import Stat
from nuzline.engine.outcome import empty_outcome
from tests.engine.d2_states import four, with_hp
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_d3_action_loop import _all_moves, _set_item, play
from tests.engine.test_e1_batch_g import _acts, _boosts, _consumed_starf, _draws, _starf_fight


@pytest.mark.slow
def test_a_starf_eaten_at_the_foes_action_is_read_by_the_law(tables, names):
    s = _starf_fight(names, tables, ["Swords Dance"], ["Tackle"])  # lead="hit"
    acts = jnp.asarray(_acts(names, s, 0, "Swords Dance", "Tackle"))
    out = assert_matches_reference(tables, s, acts, _draws(3))
    # Rank 3 of the five stats below +6 (Sp. Atk, lib.c:5288), then the Abra's own Swords Dance.
    assert int(out.item[0, 0]) == 0 and _boosts(out) == {"ATK": 2, "SPA": 2}
    laws = jax.tree.map(np.asarray, outcome_laws(tables, s, acts))
    relevant = laws.relevant.starf[0].tolist()
    consumed = _consumed_starf(tables, s, acts)
    assert consumed == [True, False, False, False]
    assert all(r for r, c in zip(relevant, consumed, strict=True) if c)


@pytest.mark.slow
def test_a_b4_self_hit_starf_is_eaten_at_its_move_end_with_entry_0(tables, names):
    """DOUBLES, everyone Swords Dance; battler 0 (Starf Berry, confused) acts last, so no later
    action's AfterMoveEffects (bcp.c:3672) can check its item, and the phase ends with its action
    (``play`` stops before the turn end). It hurts itself onto its pinch and eats the berry at its
    MoveEnd: entry 0 of its queue picks the stat (draw 0: Attack;
    draw 3: rank 3 of the five, Sp. Atk, lib.c:5288), the other entries are unread, and the law
    reads entry 0 only. Its Swords Dance never ran (the self-hit ends the action)."""
    s = _all_moves(four(tables, names, DOUBLES), names, "Swords Dance")
    s = _set_item(s, 0, names.item("Starf Berry"))
    s = s._replace(confusion=s.confusion.at[0].set(3))
    base = empty_outcome(DOUBLES)
    base = base._replace(confusion_hit=base.confusion_hit.at[0].set(1))
    d = int(confusion_damage(tables, s, 0, int(base.roll[0, 0])))
    s = with_hp(s, 0, int(s.max_hp[0, 0]) // 4 + d)

    def stages(col, draw):
        o = base._replace(starf=base.starf.at[0, col].set(draw))
        out, _ = play(tables, s, [0, 0, 0, 0], (1, 2, 3, 0), outcome=o)
        assert int(out.item[0, 0]) == 0 and int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 4
        return {Stat(i).name: int(v) for i, v in enumerate(np.asarray(out.stages[0])) if v}

    assert stages(0, 0) == {"ATK": 2} and stages(0, 3) == {"SPA": 2}
    assert all(stages(col, 3) == {"ATK": 2} for col in (1, 2, 3))
    laws = jax.tree.map(np.asarray, outcome_laws(tables, s, jnp.zeros(4, jnp.int32)))
    assert laws.relevant.starf.tolist() == [[True, False, False, False]] + [[False] * 4] * 3
