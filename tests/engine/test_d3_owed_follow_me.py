"""D3-16 (owes D2-17 b, D2-32): Follow Me through the four-actor step.

``ts_follow_me`` (1 + followMeUser; BtlCmd_FollowMe, battle_script.c:6261-6270) is read by
``BattleSystem_Defender`` for every single-target move of the other side (battle_lib.c:1694-1697,
:1714-1717), kept over a U-turn pause (turn scratch) and cleared by SetupNextTurn (:2251-2252).
Each case runs the real CalcTurnOrder: the battlers get the speeds that give the order a test names
(``ordered``), the engine side is the action phase and the reference side is ``ref_step(...,
actions_only=True)``.

The action-phase seam compares every field but two that the end of the turn clears
(``SEAM``): ``flinched`` (SetupNextTurn, lib.c:2234) and ``ts_follow_me`` (lib.c:2251-2252). When no
pause stops the action phase the engine writes every ``ts_*`` field as 0 (``step._execute_actions``
zeroes the scratch of a finished turn) while the reference keeps the per-side ``ts_follow_me`` it
wrote at the move (reference ``_fight``) until ``_end_turn``; ``actions_only`` returns before it.
That is the seam, not a rule difference: the slow tests below compare the whole turn through
``step`` / ``ref_step`` with no seam, ``ts_follow_me`` included.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.levelup import LEARN_MOVE_END
from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import BattleState, initial_arrays
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.d2_states import four, mon, with_ability, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference, jit_step
from tests.engine.test_d3_action_loop import _engine, hp, speed_ordered
from tests.engine.test_d3_pause import _engine_resume
from tests.engine.test_levelup import _base
from tests.engine.test_levelup import _mon as _lv_mon
from tests.engine.test_singles_followup2 import O as O_B2
from tests.engine.test_singles_followup2 import _resume_slot1, _u_turn_duel

IDLE = "Swords Dance"
ATK = Stat.ATK
SEAM = ("flinched", "ts_follow_me")
SWITCH_2 = MOVE_ACTIONS + 2  # a pivot's pick: party slot 2


def idle(tables, names):
    """DOUBLES, every battler knows only Swords Dance."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, [IDLE])
    return s


def ordered(tables, state, actions, outcome, order):
    """``state`` with the speeds that make CalcTurnOrder give ``order`` (fastest first, an absent
    battler last), asserted through ``order.action_order``."""
    present = [hp(state, b) > 0 for b in order]
    wanted = [b for b, p in zip(order, present, strict=True) if p]
    wanted += [b for b, p in zip(order, present, strict=True) if not p]
    state = speed_ordered(state, wanted)
    got = action_order(tables, state, jnp.asarray(actions, jnp.int32), outcome)
    assert [int(b) for b in got] == wanted
    return state


def same(out, ref, skip=()):
    for name, x, y in zip(out._fields, out, ref, strict=True):
        if name not in skip:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def play_turn(tables, state, actions, order, targets, outcome=None):
    """The action phase on the engine against ``ref_step(..., actions_only=True)`` (every field
    but ``SEAM``): ``(state, out)``, the state with the speeds of ``order``."""
    outcome = empty_outcome(DOUBLES) if outcome is None else outcome
    state = ordered(tables, state, actions, outcome, order)
    out, _ = _engine(
        tables,
        state,
        jnp.asarray(actions, jnp.int32),
        outcome,
        jnp.asarray(targets, jnp.int8),
    )
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in actions),
            outcome_to_py(outcome),
            [int(t) for t in targets],
            actions_only=True,
        )
    )
    same(out, ref, SEAM)
    return state, out


def resume_turn(tables, paused, targets, slot=2):
    """The paused turn resumed with the pivot's switch to ``slot`` (the action phase, engine
    against reference, every field but ``SEAM``): ``(state, the actions)``."""
    outcome = empty_outcome(DOUBLES)
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[int(paused.pivot_battler) - 1] = MOVE_ACTIONS + slot
    out = _engine_resume(
        tables, paused, jnp.asarray(acts, jnp.int32), outcome, jnp.asarray(targets, jnp.int8)
    )
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(paused),
            tuple(acts),
            outcome_to_py(outcome),
            [int(t) for t in targets],
            actions_only=True,
        )
    )
    same(out, ref, SEAM)
    return out, acts


def step_both(tables, state, actions, outcome, targets):
    """A whole ``step`` against ``ref_step``, every field (the turn's end included)."""
    out = jit_step(
        tables, state, jnp.asarray(actions, jnp.int32), outcome, jnp.asarray(targets, jnp.int8)
    )
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in actions),
            outcome_to_py(outcome),
            [int(t) for t in targets],
        )
    )
    same(out, ref)
    return out


def with_follow_me_and_tackle(tables, names):
    """Battler 3 Follow Me (+3, first), battler 0 Tackle at battler 1."""
    s = with_moves(idle(tables, names), names, 3, ["Follow Me"])
    return with_moves(s, names, 0, ["Tackle"])


# --- D2-17 b: a foe's single-target move goes to the Follow Me user ---------------------------


def test_a_tackle_at_battler_one_hits_the_follow_me_user(tables, names):
    s = with_follow_me_and_tackle(tables, names)
    _, out = play_turn(tables, s, [0, 0, 0, 0], (3, 0, 1, 2), (1, 0, 0, 0))
    assert hp(out, 3) < hp(s, 3)  # BattleSystem_Defender gives battler 3 (lib.c:1714-1717)
    assert hp(out, 1) == hp(s, 1)
    assert hp(out, 0) == hp(s, 0) and hp(out, 2) == hp(s, 2)


def test_the_follow_me_user_knocked_out_earlier_in_the_turn_no_longer_redirects(tables, names):
    """Battler 3 (1 HP) sets Follow Me, then battler 2's Fake Out (+3, slower than it) knocks it
    out; battler 0's Tackle at battler 1 then hits battler 1: the reader needs the user's HP
    (lib.c:1714-1717), the foe party has a bench so the battle goes on."""
    s = with_hp(with_follow_me_and_tackle(tables, names), 3, 1)
    s = with_moves(s, names, 2, ["Fake Out"])
    s = s._replace(fake_out_turn=s.fake_out_turn.at[2].set(s.turn))  # Fake Out works this turn
    _, out = play_turn(tables, s, [0, 0, 0, 0], (3, 2, 0, 1), (1, 0, 3, 0))
    assert hp(out, 3) == 0
    assert hp(out, 1) < hp(s, 1)  # the chosen target, not the fainted user


# --- D2-32: over a U-turn pause --------------------------------------------------------------


PAUSE_TARGETS = (1, 0, 1, 0)  # battler 2's U-turn and battler 0's Tackle are chosen at battler 1
PAUSE_ORDER = (3, 2, 0, 1)


def pausing_state(tables, names):
    """Battler 3 Follow Me, then battler 2's U-turn (chosen at battler 1: it goes to battler 3)
    stops at its party list; battler 0's Tackle at battler 1 is still to come."""
    return with_moves(with_follow_me_and_tackle(tables, names), names, 2, ["U-turn"])


def paused_after_follow_me(tables, names):
    s = pausing_state(tables, names)
    _, paused = play_turn(tables, s, [0, 0, 0, 0], PAUSE_ORDER, PAUSE_TARGETS)
    return s, paused, PAUSE_TARGETS


def test_follow_me_is_kept_over_a_pause_and_the_later_actor_reads_it(tables, names):
    s, paused, targets = paused_after_follow_me(tables, names)
    assert int(paused.pivot_battler) == 3  # 1 + battler 2
    assert np.asarray(paused.ts_follow_me).tolist() == [0, 4]  # 1 + its user, battler 3
    assert hp(paused, 3) < hp(s, 3) and hp(paused, 1) == hp(s, 1)  # the U-turn went to 3
    out, _ = resume_turn(tables, paused, targets)
    assert hp(out, 3) < hp(paused, 3)  # battler 0's Tackle at battler 1 hit battler 3
    assert hp(out, 1) == hp(s, 1)


def test_the_resumed_u_turn_records_its_move_for_its_kept_defender(tables, names):
    """In the case above the resumed U-turn's UpdateFlagsWhenHit writes moveCopied and
    moveCopiedHit for battler 3, its kept defender (bcp.c:1900, :3956 -> :4480-4486), never for
    battler 0 (``step._act``'s resumed ``target``): battler 3 holds U-turn from battler 2 and
    battler 0's later Tackle overwrites its moveCopied. Both engines agree."""
    _, paused, targets = paused_after_follow_me(tables, names)
    out, _ = resume_turn(tables, paused, targets)
    tackle, u_turn = names.move("Tackle"), names.move("U-turn")
    assert np.asarray(out.move_copied).tolist() == [0, 0, 0, tackle]
    assert np.asarray(out.move_copied_hit)[3].tolist() == [tackle, 0, u_turn, 0]


def test_a_battle_that_ends_at_the_resumed_u_turn_reaches_its_fainted_defender(tables, names):
    """B = 2, a drawn state (Tyranitar given Unburden): the player's U-turn leaves the foe's last
    Pokemon at 1 HP, its Pursuit's Life Orb recoil fells it in the paused pass, and the battle ends
    at the resumed U-turn's MoveEnd (SF2 (a2)). battleCtx->defender is still the foe FightCommand
    named (bcp.c:1900; the list rewrites none), so MoveEnd's RecoverStatusByAbility runs for it
    (bcp.c:3966-3967) before CheckBattleOver (:3969). Its Unburden case has no HP test (battle_lib.c:
    4433-4437): the fainted holder of the Life Orb gets canUnburden again, which its faint cleared
    (moveEffectsData zeroed, battle_lib.c:2186-2189). The status cures need HP in both engines (a
    fainted Pokemon's party status is 0, battle_controller.c:1484-1486), so this is the only write.
    Both engines agree."""
    s = _u_turn_duel(tables, names)
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.UNBURDEN))
    paused = assert_matches_reference(tables, s, (0, 0), O_B2)
    assert int(paused.pivot_battler) == 1 and int(paused.hp[1, 0]) == 0
    assert int(paused.can_unburden[1]) == 0  # the faint cleanup
    out = _resume_slot1(tables, paused)
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.can_unburden[1]) == 1


def test_the_resumed_u_turns_move_end_cures_its_kept_defender_before_the_payout(tables, names):
    """B = 4: battler 2's U-turn hits battler 3 (Limber, paralyzed: a drawn state); battler 1's
    Pursuit (1 HP, Life Orb) hits battler 2 as it leaves and falls to its recoil. The resumed
    U-turn's MoveEnd cures its kept defender, battler 3 (bcp.c:3966; battleCtx->defender from
    FightCommand, :1900), then pays battler 1's exp (:3968): battler 0 levels to 19 and the
    learn prompt pauses the turn (``LEARN_MOVE_END``) before TriggerEffectOnSwitch's
    FORBIDDEN_STATUS pass (:3973). So the paused state shows battler 3 cured. Both engines agree."""
    prinplup = _lv_mon(
        names,
        "Prinplup",
        18,
        moves=("Bubble", "Roar", "Tackle", "Water Gun"),
        ivs=(3,) * 6,
        exp=_base(tables, names, "Prinplup", 19) - 1,
    )
    player = [prinplup, mon(names, ("U-turn",))] + [mon(names) for _ in range(2)]
    pursuer = _lv_mon(names, "Geodude", 12, moves=("Pursuit",), item=names.item("Life Orb"))
    opponent = [pursuer, mon(names, (IDLE,))] + [mon(names) for _ in range(2)]
    a = initial_arrays(DOUBLES, tables, [player, opponent], opp2_items=())
    s = BattleState(**{k: jnp.asarray(v) for k, v in a.items()})
    s = with_ability(with_hp(s, 1, 1), 3, AbilityId.LIMBER)
    s = s._replace(status=s.status.at[1, 1].set(Status.PARALYSIS))
    targets = (1, 2, 3, 0)
    # Battler 3 is paralyzed: its speed is quartered (battle_lib.c:1259-1261), so it goes last.
    _, paused = play_turn(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), targets)
    assert int(paused.pivot_battler) == 3 and int(paused.hp[1, 0]) == 0
    assert int(paused.status[1, 1]) == Status.PARALYSIS
    out, _ = resume_turn(tables, paused, targets)
    assert int(out.learn_site) == LEARN_MOVE_END and int(out.level[0, 0]) == 19
    assert int(out.status[1, 1]) == Status.NONE


# --- the whole turn: ts_follow_me is 0 after it (lib.c:2251-2252) ------------------------------


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_follow_me_is_clear_after_the_turn(tables, names):
    s = ordered(
        tables,
        with_follow_me_and_tackle(tables, names),
        [0, 0, 0, 0],
        empty_outcome(DOUBLES),
        (3, 0, 1, 2),
    )
    out = step_both(tables, s, [0, 0, 0, 0], empty_outcome(DOUBLES), (1, 0, 0, 0))
    assert hp(out, 3) < hp(s, 3) and hp(out, 1) == hp(s, 1)
    assert np.asarray(out.ts_follow_me).tolist() == [0, 0]


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_follow_me_over_a_pause_is_clear_after_the_turn(tables, names):
    targets = PAUSE_TARGETS
    s = ordered(
        tables, pausing_state(tables, names), [0, 0, 0, 0], empty_outcome(DOUBLES), PAUSE_ORDER
    )
    first = step_both(tables, s, [0, 0, 0, 0], empty_outcome(DOUBLES), targets)
    assert int(first.pivot_battler) == 3
    assert np.asarray(first.ts_follow_me).tolist() == [0, 4]
    acts = np.asarray(first.ts_actions).astype(int).tolist()
    acts[2] = SWITCH_2
    out = step_both(tables, first, acts, empty_outcome(DOUBLES), targets)
    assert int(out.pivot_battler) == 0
    assert hp(out, 3) < hp(first, 3) and hp(out, 1) == hp(s, 1)
    assert np.asarray(out.ts_follow_me).tolist() == [0, 0]


# --- the D3-3 case: Fake Out after a mover, with Follow Me through the real order -------------


def fake_out_against_follow_me(tables, names, follow_me=True):
    """Battler 0 Fake Out (works this turn; the flinch roll procs), battler 3 Follow Me, the
    others Swords Dance. Follow Me has priority 3 (res/moves/follow_me/data.json:19), Fake Out 1
    (res/moves/fake_out/data.json:20), and CalcTurnOrder compares priority first (battle_lib.c:
    1396-1400): Follow Me always goes first. ``follow_me=False``: battler 3 Swords Dance."""
    s = with_moves(idle(tables, names), names, 0, ["Fake Out"])
    if follow_me:
        s = with_moves(s, names, 3, ["Follow Me"])
    s = s._replace(fake_out_turn=s.fake_out_turn.at[0].set(s.turn))
    o = empty_outcome(DOUBLES)
    return s, o._replace(secondary=o.secondary.at[0, 0].set(1))


def test_a_fake_out_after_follow_me_reaches_the_user_and_flinches_nobody(tables, names):
    """The real order (3, 0, 1, 2): Follow Me (+3, before Fake Out's +1) is set when battler 0's Fake Out, chosen
    at battler 1, runs; it goes to battler 3, which has moved (subscript_flinch_mon.s:5, a defender
    that has not moved), and battler 1 is untouched and acts."""
    s, o = fake_out_against_follow_me(tables, names)
    _, out = play_turn(tables, s, [0, 0, 0, 0], (3, 0, 1, 2), (1, 0, 0, 0), o)
    assert hp(out, 3) < hp(s, 3) and hp(out, 1) == hp(s, 1)
    assert int(out.last_move[3]) == names.move("Follow Me")
    assert int(out.stages[1, ATK]) == 2  # battler 1 was not flinched: its Swords Dance ran


def test_a_fake_out_without_follow_me_flinches_its_defender(tables, names):
    """The control, the real order (0, 3, 1, 2): with no Follow Me (battler 3 Swords Dance) Fake
    Out (+1) runs first, hits battler 1, its chosen target, and flinches it (it has not moved): no
    Swords Dance for battler 1. A Fake Out can never run before a Follow Me (+3)."""
    s, o = fake_out_against_follow_me(tables, names, follow_me=False)
    _, out = play_turn(tables, s, [0, 0, 0, 0], (0, 3, 1, 2), (1, 0, 0, 0), o)
    assert hp(out, 1) < hp(s, 1) and hp(out, 3) == hp(s, 3)
    assert int(out.stages[1, ATK]) == 0  # flinched
    assert int(out.stages[3, ATK]) == 2  # battler 3 still moved


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_the_fake_out_follow_me_turn_through_step(tables, names):
    s, o = fake_out_against_follow_me(tables, names)
    s = ordered(tables, s, [0, 0, 0, 0], o, (3, 0, 1, 2))
    out = step_both(tables, s, [0, 0, 0, 0], o, (1, 0, 0, 0))
    assert hp(out, 3) < hp(s, 3) and hp(out, 1) == hp(s, 1)
    assert np.asarray(out.ts_follow_me).tolist() == [0, 0]
