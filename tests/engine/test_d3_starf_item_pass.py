"""D3-15 x F-O3 (batch G): a Starf Berry eaten inside an action is a read of the turn's queue
entry 0 (``enumerate._starf_reads``; battle_lib.c:5286-5288), in both formats.

ShowBattleMon's pass at every turn start (battle_controller_player.c:254-267, battle_lib.c:
4151-4165, ``step._turn_start_items``) eats a Starf whose holder is at its pinch with a stat below
+6, so a board either reaches the pinch during the turn or starts with all five stats at +6, where
the Starf has no stat to raise and is kept (battle_lib.c:5278-5286):

- B = 2, at the foe's action: the holder starts one HP above its pinch (curHP <= maxHP / 4,
  battle_lib.c:4850-4855); the faster foe's Tackle lands it there and the holder, the defender,
  eats in that action's AfterMoveEffects (bcp.c:3672, :3691), before its own move; its own
  ATTACKER_ITEM then finds no berry.
- B = 4, at its own MoveEnd: battler 0 acts last; its confusion self-hit (bcp.c:2629-2639) lands it
  at the pinch and skips AfterMoveEffects, so the berry is eaten by MoveEnd's TriggerEffectOnSwitch
  (bcp.c:3973; SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS over monSpeedOrder, lib.c:4151-4165,
  TriggerHeldItemOnStatus's Starf case lib.c:5278; D3-15).
- B = 4, at another battler's action, as neither attacker nor defender: battler 0 starts at its
  pinch with all five stats at +6; battler 3's Haze (RANGE_FIELD: its defender is its user,
  lib.c:1665-1669), the turn's last action, zeroes every stage and battler 0's berry is eaten in
  that action. The game eats it at Haze's AfterMoveEffects (bcp.c:3672 -> lib.c:4151-4165,
  TriggerHeldItemOnStatus lib.c:5278); MoveEnd's check (bcp.c:3973, ``step._move_end_tail``) would
  find the same berry, and Haze cannot KO, so nothing sits between them: the test does not separate
  the two.

Every run is checked against the reference engine."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import confusion_damage
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.format import DOUBLES, MULTI
from nuzline.engine.ids import Stat
from nuzline.engine.outcome import empty_outcome
from tests.engine.d2_states import four, with_hp, with_moves
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
    draw 3: rank 3 of the five, Sp. Atk, lib.c:5288), Outcome.starf columns 1-3 are unread, and the law
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


@pytest.mark.slow
def test_a_b4_haze_lets_a_non_participant_eat_its_starf_with_entry_0(tables, names):
    """DOUBLES: battler 0 (Starf Berry) at its pinch with Attack to Speed at +6 keeps the berry
    (lib.c:5278-5286). ``play`` gives the order (0, 1, 2, 3) by raw speeds and asserts it
    (tests/engine/test_d3_action_loop.py:73-78, :106), so battler 3's Haze is the last action and
    the only one after which battler 0 can eat; its user is its only defender (lib.c:1665-1669), so
    battler 0 is neither. Haze zeroes every stage and the berry is eaten in that action: entry 0 of
    battler 0's queue picks the stat (draw 0: Attack; draw 3: Sp. Atk); Outcome.starf columns 1-3
    are unread (column 1 is the faint-replacement draw; the queue is columns 0, 2, 3, ids.py:1230);
    the law reads battler 0's column 0 only. Battler 2's column 1 stays unmarked: its replacement
    choices exclude battler 0's active holder of the shared party (ReplaceFainted, bcp.c:4096-4107;
    ``actions.replacement_bench``)."""
    s = _all_moves(four(tables, names, DOUBLES), names, "Swords Dance")
    s = with_moves(s, names, 3, ["Haze"])
    s = _set_item(s, 0, names.item("Starf Berry"))
    s = s._replace(stages=s.stages.at[0, Stat.ATK : Stat.SPE + 1].set(6))
    s = with_hp(s, 0, int(s.max_hp[0, 0]) // 4)
    base = empty_outcome(DOUBLES)

    def stages(col, draw):
        o = base._replace(starf=base.starf.at[0, col].set(draw))
        out, _ = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), outcome=o)
        assert int(out.item[0, 0]) == 0 and not np.any(np.asarray(out.stages[3]))
        return {Stat(i).name: int(v) for i, v in enumerate(np.asarray(out.stages[0])) if v}

    assert stages(0, 0) == {"ATK": 2} and stages(0, 3) == {"SPA": 2}
    assert all(stages(col, 3) == {"ATK": 2} for col in (1, 2, 3))
    laws = jax.tree.map(np.asarray, outcome_laws(tables, s, jnp.zeros(4, jnp.int32)))
    relevant = laws.relevant.starf.tolist()
    assert relevant[0] == [True, False, False, False]
    assert relevant[1] == relevant[2] == relevant[3] == [False] * 4


def _switching_holder(tables, names, fmt):
    """``fmt``: battler 0's active Pokemon holds a Starf Berry at its pinch with all five Starf
    stats at +6, so ShowBattleMon's TriggerEffectOnSwitch did not eat it (no stat to raise,
    battle_lib.c:5278-5286; else ``step._turn_start_items`` eats it before commands, bcp.c:256-268,
    battle_lib.c:4151-4165), and it switches out (its stages reset, so a replacement sends it in
    fresh at its pinch); battler 2 (1 HP) can fall to battler 1's Tackle."""
    s = _all_moves(four(tables, names, fmt), names, "Swords Dance")
    s = with_moves(s, names, 1, ["Tackle"])
    s = _set_item(s, 0, names.item("Starf Berry"))
    s = s._replace(stages=s.stages.at[0, Stat.ATK : Stat.SPE + 1].set(6))
    s = with_hp(s, 0, int(s.max_hp[0, 0]) // 4)
    s = with_hp(s, 2, 1)
    actions = jnp.asarray([MOVE_ACTIONS + 2, 0, 0, 0], jnp.int32)
    return jax.tree.map(np.asarray, outcome_laws(tables, s, actions)).relevant.starf


def test_a_partner_holder_that_switches_out_is_on_the_replacement_bench(tables, names):
    """DOUBLES, shared party (``_switching_holder``): at the end-of-turn replacement ReplaceFainted
    excludes the partner's slot as it stands then (``selectedPartySlot[i ^ 2]``, bcp.c:4096-4107),
    so battler 2 can send in battler 0's switched-out holder, which reads battler 2's replacement
    draw (column 1): ``relevant.starf[2][1]``. The Haze test above keeps row 2 all False."""
    assert bool(_switching_holder(tables, names, DOUBLES)[2][1])


def test_a_partner_holder_of_another_party_is_not_on_the_replacement_bench(tables, names):
    """MULTI, separate parties (format.py MULTI: battler 2 draws from party 2): the same board's
    holder belongs to battler 0's party only, so battler 2's replacement never sends it in
    (ReplaceFainted reads the battler's own party, bcp.c:4081-4107): ``starf[2][1]`` stays False."""
    assert not bool(_switching_holder(tables, names, MULTI)[2][1])
