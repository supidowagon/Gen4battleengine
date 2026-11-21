"""D3 review fix A: B = 4 reads of "the target" that read ``_foe(side)`` (battler ``b ^ 1``) where the
decomp reads ``battleCtx->defender``, the stored Future Sight battler, or a side's battler set.

Each case is a DOUBLES state where the action's target is battler 3 while battler 1 (``0 ^ 1``)
differs, so a ``_foe`` read shows. ``play`` asserts the engine's action phase equal to the
reference's in every field; ``both`` and ``future_both`` do the same for the end of the turn. Where
both engines read the wrong battler before the fix, an expected value is asserted too.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import _future_sight, end_of_turn_order
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _future_sight as _ref_future_sight
from nuzline.reference.engine import _speed_order, _Turn
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_end_of_turn import _future, both

ORDER = (0, 1, 2, 3)
AT_3 = (3, 0, 0, 0)  # battler 0 aims at battler 3


def idle(tables, names):
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    return s


def status_of(state, b):
    return int(state.status[party_of(state, b), int(state.active[b])])


def with_status(state, b, status, counter=0):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(
        status=state.status.at[p, slot].set(status),
        status_counter=state.status_counter.at[p, slot].set(counter),
    )


def with_stat(state, b, stat, value):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(stats=state.stats.at[p, slot, stat - 1].set(value))


# --- 1. Future Sight reads its stored battler ---------------------------------------------------


def future_both(tables, state, protecting, outcome=None):
    """The Future Sight pass alone, engine against reference, with ``protecting`` the battlers'
    turnFlags.protecting of the turn (bcp.c:3007)."""
    outcome = empty_outcome(DOUBLES) if outcome is None else outcome
    sides = end_of_turn_order(tables, state, outcome)
    out = _future_sight(tables, state, sides, outcome, jnp.asarray(protecting, jnp.int32))
    np_tables, rs, o = to_numpy(tables), to_ref(state), outcome_to_py(outcome)
    turn = _Turn(np_tables, o, entry_turn=rs.turn + 1, battler_count=len(rs.battlers))
    turn.protecting = [bool(p) for p in protecting]
    _ref_future_sight(turn, rs, _speed_order(np_tables, rs, o))
    ref = from_ref(rs)
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def test_future_sight_on_battler_3_ignores_battler_1s_protect_and_evasion(tables, names):
    """Battler 0's Future Sight lands on battler 3: CheckMoveHit of the stored attacker against
    msgBattlerTemp (subscript_future_sight_damage.s:8, bcp.c:217-221, :3001-3021), never battler
    1, which protects (first case) or sits at +6 evasion (second case)."""
    move = names.move("Future Sight")
    assert int(tables.move_accuracy[move]) >= 50
    s = _future(idle(tables, names), 3, 0)
    s = s._replace(future_move=s.future_move.at[3].set(move))
    out = future_both(tables, s, [0, 1, 0, 0])
    assert hp(out, 3) == hp(s, 3) - 5 and hp(out, 1) == hp(s, 1)
    evasive = s._replace(stages=s.stages.at[1, Stat.EVA].set(6))
    o = empty_outcome(DOUBLES)._replace(future_hit=jnp.asarray([50, 50, 50, 50], jnp.int8))
    out = future_both(tables, evasive, [0, 0, 0, 0], o)
    assert hp(out, 3) == hp(s, 3) - 5


def test_future_sight_stores_the_damage_against_its_defender(tables, names):
    """BtlCmd_TryFutureSight computes CalcMoveDamage against battleCtx->defender
    (battle_script.c:6079-6089): battler 3's Sp. Def, not battler 1's."""
    s = with_moves(idle(tables, names), names, 0, ["Future Sight"])
    s = with_stat(with_stat(s, 1, Stat.SPD, 400), 3, Stat.SPD, 40)
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    soft = int(out.future_damage[3])
    hard, _ = play(tables, with_stat(s, 3, Stat.SPD, 400), [0, 0, 0, 0], ORDER, targets=AT_3)
    assert int(out.future_timer[3]) != 0 and int(out.future_timer[1]) == 0
    assert soft > int(hard.future_damage[3])


# --- 2. U-turn's contact ability is its defender's ---------------------------------------------


def test_u_turn_meets_battler_3s_static_not_battler_1s_rough_skin(tables, names):
    """subscript_attack_then_switch_out.s:7: TriggerAbilityOnHit of the defender (lib.c:4200):
    battler 3's Static paralyses battler 0; battler 1's Rough Skin does nothing."""
    s = with_moves(idle(tables, names), names, 0, ["U-turn"])
    s = with_ability(with_ability(s, 3, AbilityId.STATIC), 1, AbilityId.ROUGH_SKIN)
    o = empty_outcome(DOUBLES)
    o = o._replace(contact_proc=o.contact_proc.at[3, 0].set(1).at[1, 0].set(1))
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3, outcome=o)
    assert status_of(out, 0) == Status.PARALYSIS
    assert hp(out, 0) == hp(s, 0)
    assert hp(out, 3) < hp(s, 3)


# --- 3. Gyro Ball reads its defender's speed ---------------------------------------------------


def test_gyro_ball_power_reads_battler_3s_speed(tables, names):
    """BtlCmd_CalcGyroBallPower (battle_script.c:7127): monSpeedValues[defender]. Battler 1's
    Speed changes nothing; battler 3's does."""
    s = with_moves(idle(tables, names), names, 0, ["Gyro Ball"])
    s = with_stat(with_stat(with_stat(s, 0, Stat.SPE, 20), 2, Stat.SPE, 60), 3, Stat.SPE, 100)
    lost = []
    for one, three in ((30, 100), (90, 100), (30, 80)):
        t = with_stat(with_stat(s, 1, Stat.SPE, one), 3, Stat.SPE, three)
        out, _ = play(tables, t, [0, 0, 0, 0], None, targets=AT_3)
        lost.append(hp(t, 3) - hp(out, 3))
    assert lost[0] == lost[1] and lost[2] < lost[0]


# --- 4. Dream Eater, Embargo, Endeavor read their defender -------------------------------------


def test_dream_eater_works_on_sleeping_battler_3_next_to_awake_battler_1(tables, names):
    """effect_script_0008.s:5-6: CheckSubstitute and the sleep test on BTLSCR_DEFENDER."""
    s = with_moves(idle(tables, names), names, 0, ["Dream Eater"])
    s = with_status(s, 3, Status.SLEEP, 4)
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    assert hp(out, 3) < hp(s, 3)


def test_embargo_ignores_battler_1s_substitute(tables, names):
    """effect_script_0232.s:5: CheckSubstitute BTLSCR_DEFENDER."""
    s = with_moves(idle(tables, names), names, 0, ["Embargo"])
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(20))
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    assert int(out.embargo[3]) != 0 and int(out.embargo[1]) == 0


def test_endeavor_reads_battler_3s_hp(tables, names):
    """effect_script_0189.s:5-7: the defender's curHP against the attacker's. Battler 1 has fewer
    HP than the user, battler 3 more: the move works and leaves battler 3 at the user's HP."""
    s = with_moves(idle(tables, names), names, 0, ["Endeavor"])
    s = with_hp(with_hp(s, 0, 10), 1, 5)
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    assert hp(out, 3) == 10 and hp(out, 1) == 5


# --- 5. Imprison: both opponents, never the partner --------------------------------------------


def test_imprison_marks_both_opponents_only(tables, names):
    """BtlCmd_TryImprison (battle_script.c:6686-6690): MOVE_EFFECT_IMPRISONED on the two enemy
    slots; the shared-move test loops the other side's battlers (:6697-6699)."""
    s = with_moves(idle(tables, names), names, 0, ["Imprison", "Swords Dance"])
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER)
    assert np.asarray(out.imprisoned).tolist() == [0, 1, 0, 1]
    assert int(out.imprison[0]) == 1


def test_imprison_fails_on_a_move_only_the_partner_knows(tables, names):
    """The user shares Ember with its partner (battler 2) alone: no battler of the other side
    knows one of its moves, so TryImprison fails (:6719-6720); the foes are marked all the same."""
    s = with_moves(idle(tables, names), names, 0, ["Imprison", "Ember"])
    s = with_moves(s, names, 2, ["Ember"])
    for b in (1, 3):
        s = with_moves(s, names, b, ["Tackle"])
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 1, 0))
    assert int(out.imprison[0]) == 0
    assert np.asarray(out.imprisoned).tolist() == [0, 1, 0, 1]
    shared = with_moves(s, names, 3, ["Ember"])
    out, _ = play(tables, shared, [0, 0, 0, 0], ORDER, targets=(1, 0, 1, 0))
    assert int(out.imprison[0]) == 1


def test_a_partners_imprison_seals_nothing(tables, names):
    """Move_Imprisoned (battle_lib.c:3323-3324) reads the other side only: battler 2 imprisoning
    with Tackle leaves its partner's Tackle usable."""
    s = with_moves(idle(tables, names), names, 0, ["Tackle"])
    s = with_moves(s, names, 2, ["Tackle"])
    s = s._replace(imprison=s.imprison.at[2].set(1))
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    assert hp(out, 3) < hp(s, 3)


# --- 6. Bad Dreams: the holders of the other side --------------------------------------------


def test_bad_dreams_hurts_the_other_side_and_never_its_ally(tables, names):
    """MON_COND_CHECK_STATE_BAD_DREAMS (bcp.c:1485): COUNT_ALIVE_BATTLERS_THEIR_SIDE_FLAG
    (battle_lib.c:2915-2922). Battler 3 holds Bad Dreams; battler 0 asleep loses Divide(maxHP, 8),
    battler 1 asleep (its ally) loses nothing."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Tackle"])
    s = with_ability(s, 3, AbilityId.BAD_DREAMS)
    s = with_status(with_status(s, 0, Status.SLEEP, 4), 1, Status.SLEEP, 4)
    out = both(tables, s)
    eighth = max(int(s.max_hp[party_of(s, 0), int(s.active[0])]) // 8, 1)
    assert hp(out, 0) == hp(s, 0) - eighth
    assert hp(out, 1) == hp(s, 1)


# --- 7. Fixed damage reads its defender --------------------------------------------------------


def test_super_fang_halves_battler_3s_hp(tables, names):
    """effect_script_0040.s: Divide(curHP, 2) of BTLSCR_DEFENDER: battler 3 at 100 loses 50 while
    battler 1 sits at 40."""
    s = with_moves(idle(tables, names), names, 0, ["Super Fang"])
    s = with_hp(with_hp(s, 3, 100), 1, 40)
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3)
    assert hp(out, 3) == 50 and hp(out, 1) == 40


def test_present_heal_reads_its_defenders_hp(tables, names):
    """subscript_present_heal.s:9 on BTLSCR_DEFENDER: battler 3 below full HP is healed, so no
    MOVE_STATUS_NO_MORE_WORK and AfterMoveEffects run (bcp.c:3313-3315): its Oran Berry, still
    under half HP, is eaten. Battler 1 at full HP plays no part (effects.recovers_at_full_hp)."""
    s = with_moves(idle(tables, names), names, 0, ["Present"])
    s = with_hp(s, 3, 5)
    p, slot = party_of(s, 3), int(s.active[3])
    s = s._replace(item=s.item.at[p, slot].set(names.item("Oran Berry")))
    o = empty_outcome(DOUBLES)
    o = o._replace(present=o.present.at[0].set(3))  # PRESENT_HEAL
    out, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=AT_3, outcome=o)
    assert hp(out, 3) > 5
    assert int(out.item[p, slot]) == names.item("")
