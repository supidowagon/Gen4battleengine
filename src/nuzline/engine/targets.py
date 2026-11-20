"""The fight command's defender: ``BattleSystem_Defender`` (battle_lib.c:1600-1730), the one
resolver of ledger groups D2-4 .. D2-8 (rule sheets ``_scratch/e2/rules/d2/D2-4.md`` section R).

``resolve`` evaluates the ten branches of the decomp's range switch in its order and returns the
defender, the battler the hit reads as its target, the spread loop's ``battlerCounter`` and
whether the call consumed its one ``RandNext()``. At ``B = 2`` (a static fact of the state's
shapes) no branch reads ``bit`` and Follow Me and the stored target act as in the decomp; the doubles-only
arms sit under a Python ``if doubles:`` (docs/doubles-state-design.md section 5, rule 1).

Battler geometry (D2-4 R2): side ``b & 1``, partner ``b ^ 2``, opponent slots
``BattleSystem_GetEnemyInSlot`` (``opponents``). "Alive" is ``curHP != 0``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine import field as fcm
from nuzline.engine.damage import (
    hidden_power,
    judgment_type,
    natural_gift_params,
    trace_copy,
    weather_ball_params,
    weather_suppressed,
)
from nuzline.engine.format import format_of, party, side_of
from nuzline.engine.ids import (
    MOVE_HIDDEN_POWER,
    MOVE_JUDGMENT,
    MOVE_NATURAL_GIFT,
    MOVE_WEATHER_BALL,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    MoveTarget,
    Type,
)
from nuzline.engine.state import BattleState, battler_ability, held_item

_I32 = jnp.int32

NO_DEFENDER = -1  # BATTLER_NONE (include/constants/battle.h:99)


class Resolved(NamedTuple):
    defender: jnp.ndarray  # int32: BattleSystem_Defender's battler, NO_DEFENDER for BATTLER_NONE
    target_side: jnp.ndarray  # int32: the battler the hit reads as its target (D2-4 R6)
    counter: jnp.ndarray  # int32: battlerCounter after branches 1-2 (D4's loop); 0 elsewhere
    drew: jnp.ndarray  # bool: this call consumed RandNext (D2-4 R4)


def alive(state: BattleState, battler) -> jnp.ndarray:
    """``bool``: ``battleMons[battler].curHP != 0``. A battler left empty keeps its fainted
    Pokemon (bcp.c:4108-4110), so it reads as not alive (D2-4 R2.6)."""
    fmt = format_of(state)
    battler = jnp.asarray(battler, _I32)
    return state.hp[party(fmt, battler), state.active[battler]] > 0


def opponents(battler) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(BattleSystem_GetEnemyInSlot(b, ENEMY_IN_SLOT_RIGHT), ..._LEFT)`` at B = 4 (sys.c:
    419-433; D2-4 R2.4): battler ``b`` of the other side whose type has bit 1 clear / set, i.e.
    ``(b ^ 1) | 2`` and ``(b ^ 1) & 1``. Table: 0 -> (3, 1), 1 -> (2, 0), 2 -> (3, 1),
    3 -> (2, 0). Only read at B = 4 (outside doubles the opponent is ``b ^ 1``)."""
    other = jnp.asarray(battler, _I32) ^ 1
    return other | 2, other & 1


def follow_me_user(state: BattleState, side) -> jnp.ndarray:
    """``int32``: the living Follow Me user of side ``side`` (a side index, ``side_of``):
    ``ts_follow_me[side] - 1`` when that field is set and the Pokemon now in the user's battler
    slot has HP (``sideConditions[side].followMe && FOLLOW_ME_MON(side).curHP``), else
    ``NO_DEFENDER`` (D2-11 rule 4: nothing clears ``followMe`` on a faint or switch; the readers
    test the slot's ``curHP``)."""
    fmt = format_of(state)
    value = state.ts_follow_me[jnp.asarray(side, _I32)].astype(_I32)
    user = jnp.clip(value - 1, 0, fmt.B - 1)
    return jnp.where((value != 0) & alive(state, user), user, NO_DEFENDER)


def follow_me_target(state: BattleState, attacker) -> jnp.ndarray:
    """``int32``: the living Follow Me user on the other side of ``attacker`` (the readers of
    ``BattleSystem_Defender``, lib.c:1694-1697, :1714-1717; ``follow_me_user``)."""
    return follow_me_user(state, side_of(jnp.asarray(attacker, _I32)) ^ 1)


def magic_coat_defender(state: BattleState, mover) -> jnp.ndarray:
    """``int32``: the defender of a reflected move's holder pass, ``BtlCmd_MagicCoat``
    (battle_script.c:6487-6507; E3-B2-6, review b2 #9): the living Follow Me user of the
    MOVER's side (``target`` is ``GetBattlerSide(attacker)`` read while the mover is still the
    attacker, :6490), else the mover. The range branch (:6497-6498, adjacent opponents / all
    adjacent) gives the mover; so does the last branch (:6500-6505): ``BattleSystem_Defender``
    reads the move's own range (lib.c:1604-1606, ``inRange`` only for move 0) and its battler is
    an opponent of the holder, while the one Lightning Rod / Storm Drain flag of the action
    (``selfTurnFlags``, cleared by BattleContext_Init at each action's end) sits on the redirect
    target, which is the holder itself (the reflection reads the DEFENDER's magicCoat, bcp.c:3069).
    At B = 2 it is the mover (its side's Follow Me is its own, and it is acting now)."""
    mover = jnp.asarray(mover, _I32)
    fm = follow_me_user(state, side_of(mover))
    return jnp.where(fm != NO_DEFENDER, fm, mover).astype(_I32)


def random_opponent(state: BattleState, attacker, bit):
    """``BattleSystem_RandomOpponent`` (lib.c:4177-4198): ``(battler, drew)``. B = 4: ``bit``
    picks ``opponents[bit]``; a dead pick moves to the other opponent, whatever its HP; always
    one draw. B = 2: ``attacker ^ 1`` at any HP, no draw."""
    attacker = jnp.asarray(attacker, _I32)
    if format_of(state).B != 4:
        return attacker ^ 1, jnp.asarray(False)
    first, second = opponents(attacker)
    pick = jnp.where(jnp.asarray(bit, _I32) & 1 == 0, first, second)
    other = jnp.where(pick == first, second, first)
    return jnp.where(alive(state, pick), pick, other), jnp.asarray(True)


def fight_randomizes(state: BattleState, side, struggling, reselected=False) -> jnp.ndarray:
    """``bool``: the fight command's ``randomizeTarget`` (bcp.c:1861-1903, D2-9 rule 1).
    ``struggling`` (:1866-1868), or an Encore on the attacker at the action's start (:1869-1884):
    both Encore branches randomize, whether or not the slot still holds the encored move. The
    lock branch (:1885-1888) and the plain slot (:1894-1896) do not; a locked battler is never
    struggling. Branch 1.5 (:1889-1893) randomizes when the selected move left its slot:
    ``reselected``, a learn prompt answered between the selection and the action replaced the
    slot (``step._execute_actions``, E3-B2-21)."""
    return (
        jnp.asarray(struggling, bool)
        | (state.encore_move[side] != 0)
        | jnp.asarray(reselected, bool)
    )


def multi_turn(tables: Tables, move) -> jnp.ndarray:
    """``Move_IsMultiTurn`` (lib.c:2975-2992): Bide and the nine charge effects (the ``CHARGE``
    flag marks exactly those, step.py's charge block)."""
    move = jnp.asarray(move, _I32)
    return ((tables.move_flags[move].astype(_I32) & MoveFlag.CHARGE) != 0) | (
        tables.move_effect[move] == EffectId.BIDE
    )


def no_target(tables: Tables, state: BattleState, battler, move, defender, last_of_multi):
    """``NO_TARGET`` (include/battle/common.h:56-61, D2-10 rule 1): ``defender`` is BATTLER_NONE and
    the move is not multi-turn, or it is and the battler is ``MOVE_LOCKED`` or
    ``SYSCTL_LAST_OF_MULTI_TURN`` (``last_of_multi``) is set."""
    locked = state.locked_move[battler] != 0
    none = jnp.asarray(defender, _I32) == NO_DEFENDER
    return none & (~multi_turn(tables, move) | locked | jnp.asarray(last_of_multi, bool))


def charge_keeps_user(tables: Tables, state: BattleState, battler, move, defender) -> jnp.ndarray:
    """``BattleControllerPlayer_HasNoTarget``'s second part (bcp.c:2319-2334, D2-10 rule 2): no
    defender, a multi-turn move the check did not fire on, not SolarBeam in unsuppressed sun, no
    Power Herb (``HOLD_EFFECT_CHARGE_SKIP``) and not ``MOVE_LOCKED``: the defender becomes the
    attacker, so a charge turn with no foe charges."""
    sun_now = (
        ((tables.move_flags[move].astype(_I32) & MoveFlag.SUN_SKIP) != 0)
        & fcm.sun(fcm.weather_bits(state))
        & ~weather_suppressed(tables, state)
    )
    herb = tables.item_effect[held_item(state, battler)] == ItemId.POWER_HERB
    keeps = (jnp.asarray(defender, _I32) == NO_DEFENDER) & multi_turn(tables, move)
    return keeps & ~no_target(tables, state, battler, move, defender, False) & ~sun_now & ~herb


def _ability(state: BattleState, battler) -> jnp.ndarray:
    """``int32``: ``Battler_Ability`` of ``battler`` (lib.c:3088-3093; D2-12 rule 11) through the
    shared reader ``state.battler_ability`` (none under Gastro Acid's suppression, E3-2-6a). The
    one read of a battler's ability in this module."""
    return battler_ability(state, jnp.asarray(battler, _I32)).astype(_I32)


def redirect_type(tables: Tables, state: BattleState, attacker, move) -> jnp.ndarray:
    """``CalcMoveType(attacker, move)`` with the Normal -> data type fallback of
    ``BattleSystem_CheckRedirectionAbilities`` (lib.c:1747-1750, :7815-7917; D2-12 rule 3). Weather
    Ball with no weather reads an uninitialised type in the C: Normal, then the data type (ledger
    D2-6)."""
    fmt = format_of(state)
    attacker = jnp.asarray(attacker, _I32)
    move = jnp.asarray(move, _I32)
    item = held_item(state, attacker)
    hp_type = hidden_power(state.ivs[party(fmt, attacker), state.active[attacker]])[0]
    kind = jnp.asarray(Type.NORMAL, _I32)
    kind = jnp.where(move == MOVE_HIDDEN_POWER, hp_type, kind)
    kind = jnp.where(move == MOVE_WEATHER_BALL, weather_ball_params(tables, state, move)[2], kind)
    kind = jnp.where(move == MOVE_NATURAL_GIFT, natural_gift_params(tables, item)[1], kind)
    kind = jnp.where(move == MOVE_JUDGMENT, judgment_type(tables, item, Type.NORMAL), kind)
    return jnp.where(kind == Type.NORMAL, tables.move_type[move].astype(_I32), kind).astype(_I32)


def redirect(tables: Tables, state: BattleState, attacker, move, defender, speed_order):
    """``BattleSystem_CheckRedirectionAbilities`` (lib.c:1732-1790, D2-12 rules 2-6): ``(defender,
    redirected)``. A single-target or random-opponent move of Electric (Water) type goes to the
    first living Lightning Rod (Storm Drain) holder other than the attacker in ``speed_order``
    (``monSpeedOrder``, D3's input; ``jnp.arange(2)`` at B = 2). No change for a BATTLER_NONE
    defender, a Mold Breaker attacker (Normalize: owed to E1) or a living Follow Me user on the
    other side; a holder that is already the defender sets no flag."""
    fmt = format_of(state)
    attacker = jnp.asarray(attacker, _I32)
    move = jnp.asarray(move, _I32)
    defender = jnp.asarray(defender, _I32)
    order = jnp.asarray(speed_order, _I32)
    rng = tables.move_target[move]
    ranged = (rng == MoveTarget.RANGE_SINGLE_TARGET) | (rng == MoveTarget.RANGE_RANDOM_OPPONENT)
    exits = (
        (defender == NO_DEFENDER)
        | (_ability(state, attacker) == AbilityId.MOLD_BREAKER)
        | (follow_me_target(state, attacker) != NO_DEFENDER)
    )
    kind = redirect_type(tables, state, attacker, move)
    abilities = jnp.stack([_ability(state, order[k]) for k in range(fmt.B)])
    living = jnp.stack([alive(state, order[k]) for k in range(fmt.B)])
    wanted = jnp.where(
        kind == Type.ELECTRIC,
        jnp.asarray(AbilityId.LIGHTNINGROD, _I32),
        jnp.where(kind == Type.WATER, jnp.asarray(AbilityId.STORM_DRAIN, _I32), -1),
    )
    cands = (abilities == wanted) & living & (order != attacker)
    found = jnp.any(cands)
    holder = order[jnp.argmax(cands)]
    redirected = ~exits & ranged & found & (holder != defender)
    return jnp.where(redirected, holder, defender).astype(_I32), redirected


def _ghost(state: BattleState, battler) -> jnp.ndarray:
    fmt = format_of(state)
    types = state.types[party(fmt, battler), state.active[battler]]
    return jnp.any(types == Type.GHOST)


def _pick(branches, default):
    """The first ``(condition, value)`` pair whose condition holds, else ``default`` (the
    decomp's ``if`` / ``else if`` chain)."""
    out = default
    for cond, value in reversed(branches):
        out = jnp.where(cond, value, out)
    return out


def resolve(
    tables: Tables,
    state: BattleState,
    attacker,
    move,
    randomize,
    chosen,
    speed_order,
    bit,
) -> Resolved:
    """``BattleSystem_Defender(attacker, move, randomize)`` (lib.c:1600-1730; D2-4 R3).

    ``chosen``: ``battlerActions[attacker][BATTLE_ACTION_CHOOSE_TARGET]`` (``chosen_target``).
    ``speed_order`` (int32[B]): ``monSpeedOrder``, D3's input (``jnp.arange(2)`` at B = 2).
    ``bit``: the low bit of the one ``RandNext()`` the call may draw (``Outcome.target_draw``);
    not read at B = 2.

    At B = 2 branch 9 gives the living foe (lib.c:1706-1707) and branch 10 the living Follow Me
    user, else ``chosen`` while it has HP, else the foe when alive (lib.c:1714-1725). ``chosen``
    is the user after a RANGE_MY_SIDE or non-Ghost Curse selection (ruling D2-33), so a locked
    single-target move reads the user. ``target_side`` is the defender whenever the move aims at
    it, also when that is the user (a ``self_defender`` move keeps the foe there)."""
    fmt = format_of(state)
    doubles = fmt.B == 4
    attacker = jnp.asarray(attacker, _I32)
    move = jnp.asarray(move, _I32)
    randomize = jnp.asarray(randomize, bool)
    chosen = jnp.asarray(chosen, _I32)
    order = jnp.asarray(speed_order, _I32)
    rng = tables.move_target[move].astype(_I32)
    none = jnp.asarray(NO_DEFENDER, _I32)
    foe = attacker ^ 1
    partner = attacker ^ 2

    # Branches 1-2 (D2-8): the first living candidate in monSpeedOrder.
    adjacent = jnp.stack(
        [alive(state, order[k]) & (side_of(order[k]) != side_of(attacker)) for k in range(fmt.B)]
    )
    all_adjacent = jnp.stack(
        [alive(state, order[k]) & (order[k] != attacker) for k in range(fmt.B)]
    )

    def first_of(cands):
        found = jnp.any(cands)
        k = jnp.argmax(cands)
        return (
            jnp.where(found, order[k], none),
            jnp.where(found, k + 1, fmt.B).astype(_I32),
        )

    adj_def, adj_ctr = first_of(adjacent)
    all_def, all_ctr = first_of(all_adjacent)

    # Branches 4-5 (D2-6).
    ro, ro_drew = random_opponent(state, attacker, bit)

    # Branches 3, 7, 8 (D2-5), 9 (D2-7), 10 (D2-4).
    if doubles:
        bit1 = jnp.asarray(bit, _I32) & 1
        uoa_random = jnp.where(
            bit1 == 0, jnp.where(alive(state, partner), partner, attacker), attacker
        )
        ally = partner
        uoa = jnp.where(alive(state, chosen), chosen, attacker)
        o0, o1 = opponents(attacker)
        fm = follow_me_target(state, attacker)
        has_fm = fm != none
        both = alive(state, o0) & alive(state, o1)
        random_def = jnp.where(
            has_fm,
            fm,
            jnp.where(
                both,
                jnp.where(bit1 == 0, o0, o1),
                jnp.where(alive(state, o0), o0, jnp.where(alive(state, o1), o1, none)),
            ),
        )
        random_drew = ~has_fm & both
        chosen_alive = alive(state, chosen)
        single_def = jnp.where(
            has_fm,
            fm,
            jnp.where(chosen_alive, chosen, jnp.where(alive(state, ro), ro, none)),
        )
        single_drew = ~has_fm & ~chosen_alive
        uoa_random_drew = jnp.asarray(True)
    else:
        uoa_random = attacker
        ally = attacker
        uoa = attacker
        # Branch 9 (randomized): the living foe (lib.c:1706-1707, no Follow Me arm outside doubles);
        # branch 10: Follow Me's user, else the stored target while it has HP, else
        # BattleSystem_RandomOpponent (lib.c:1714-1725, which is the foe at B = 2). The stored
        # target is the user after a RANGE_MY_SIDE or non-Ghost Curse selection (bcp.c:4363-4367),
        # so a battler locked into a single-target move reads itself (singles follow-up (ii)).
        live_foe = jnp.where(alive(state, foe), foe, none)
        fm = follow_me_target(state, attacker)
        random_def = live_foe
        single_def = jnp.where(fm != none, fm, jnp.where(alive(state, chosen), chosen, live_foe))
        random_drew = single_drew = uoa_random_drew = jnp.asarray(False)

    is_ = lambda *ranges: jnp.isin(rng, jnp.asarray([int(r) for r in ranges], _I32))
    no_draw = jnp.asarray(False)
    ro_def = ro
    branches = [
        (is_(MoveTarget.RANGE_ADJACENT_OPPONENTS), adj_def, adj_ctr, no_draw),
        (is_(MoveTarget.RANGE_ALL_ADJACENT), all_def, all_ctr, no_draw),
        (
            is_(MoveTarget.RANGE_USER_OR_ALLY) & randomize,
            uoa_random,
            jnp.zeros((), _I32),
            uoa_random_drew,
        ),
        (
            is_(MoveTarget.RANGE_SINGLE_TARGET_ME_FIRST) & randomize,
            ro_def,
            jnp.zeros((), _I32),
            ro_drew,
        ),
        (is_(MoveTarget.RANGE_OPPONENT_SIDE), ro_def, jnp.zeros((), _I32), ro_drew),
        (
            is_(
                MoveTarget.RANGE_USER,
                MoveTarget.RANGE_USER_SIDE,
                MoveTarget.RANGE_SINGLE_TARGET_SPECIAL,
                MoveTarget.RANGE_FIELD,
            ),
            attacker,
            jnp.zeros((), _I32),
            no_draw,
        ),
        (is_(MoveTarget.RANGE_ALLY), ally, jnp.zeros((), _I32), no_draw),
        (is_(MoveTarget.RANGE_USER_OR_ALLY), uoa, jnp.zeros((), _I32), no_draw),
        (
            is_(MoveTarget.RANGE_RANDOM_OPPONENT) | randomize,
            random_def,
            jnp.zeros((), _I32),
            random_drew,
        ),
    ]
    defender = _pick([(c, d) for c, d, _, _ in branches], single_def).astype(_I32)
    counter = _pick([(c, k) for c, _, k, _ in branches], jnp.zeros((), _I32)).astype(_I32)
    drew = _pick([(c, w) for c, _, _, w in branches], single_drew)
    # A self-range move or a non-Ghost Curse names the user as its defender but its gates read the
    # foe (``target_side``); every other move that names the user (a lock reading the stored
    # target) aims at it.
    from nuzline.engine.effects import self_defender

    aims = ~self_defender(tables, move, _ghost(state, attacker))
    other = (defender != none) & ((defender != attacker) | aims)
    target_side = jnp.where(other, defender, foe).astype(_I32)
    return Resolved(defender, target_side, counter, jnp.asarray(drew, bool))


def opposing(fmt, battler) -> tuple:
    """The battlers a switch-in ability of ``battler`` works on (D2-16): ``opponents(battler)`` at
    B = 4, the foe alone at B = 2 (``fmt``'s static count). A static-length tuple, so callers loop
    over it."""
    if fmt.B == 4:
        return opponents(battler)
    return (jnp.asarray(battler, _I32) ^ 1,)


def trace_target(tables: Tables, state: BattleState, holder, coin):
    """``(target, found)``: the battler ``SWITCH_IN_CHECK_STATE_TRACE`` copies for ``holder``
    (battle_lib.c:3712-3737, D2-16 rules 2-3). ``ChooseTraceTarget`` (:7750-7781) takes
    ``GetEnemyInSlot`` RIGHT / LEFT (``opponents``): both eligible, ``coin`` 1 gives the LEFT one
    (:7754-7767); else the eligible one, RIGHT first; none: not ``found``. ``trace_copy`` is the
    holder's test (HP, Trace, no Griseous Orb) and the target's eligibility (HP, ability not
    Forecast / Trace / Multitype by its raw field) together: the holder part is common to both
    slots. At B = 2 both slots are the foe, so the pick is the foe whatever ``coin``."""
    fmt = format_of(state)
    holder = jnp.asarray(holder, _I32)
    if fmt.B == 4:
        first, second = opponents(holder)
    else:
        first = second = holder ^ 1
    e1 = trace_copy(tables, state, holder, first)
    e2 = trace_copy(tables, state, holder, second)
    pick = jnp.where(jnp.asarray(coin, _I32) & 1 == 1, second, first)
    target = jnp.where(e1 & e2, pick, jnp.where(e1, first, second))
    return target.astype(_I32), e1 | e2


def called_defender(tables: Tables, state: BattleState, battler, move, speed_order, bit):
    """``(defender, redirected)`` of ``BtlCmd_GoToMoveScript`` for the called ``move`` (script.c:
    2111-2134, D2-13 A3-A4; also SetMirrorMove's resolution, :3511): ``BattleSystem_Defender(
    battler, move, randomize TRUE)`` by the called move's own range (``resolve`` with the stored
    target, which a randomized single-target call never reads), then
    ``BattleSystem_CheckRedirectionAbilities`` with the same move (:2122; D2-12). ``bit`` is
    ``Outcome.call_target_draw``. ``NO_DEFENDER`` takes the script to ``subscript_no_target.s``.
    At B = 2 the redirection never fires (its holder would be the living foe, already the
    defender), so the singles step skips it."""
    battler = jnp.asarray(battler, _I32)
    res = resolve(
        tables, state, battler, move, True, state.chosen_target[battler], speed_order, bit
    )
    if format_of(state).B != 4:
        return res.defender, jnp.asarray(False)
    return redirect(tables, state, battler, move, res.defender, speed_order)


def counter_defender(state: BattleState, battler, last_attacker) -> jnp.ndarray:
    """``int32``: the defender of Counter, Mirror Coat and Metal Burst (script.c:4628-4632,
    :4681-4685, :7165-7169, D2-14 rules 4-6): the living Follow Me user of ``last_attacker``'s
    side, else ``last_attacker``. The commands' success test has already required ``last_attacker``
    on the other side with HP, so the other side of ``battler`` is its side; the random-opponent
    fallback is unreachable (rule 7)."""
    follow = follow_me_target(state, battler)
    return jnp.where(follow != NO_DEFENDER, follow, jnp.asarray(last_attacker, _I32)).astype(_I32)


def bide_defender(state: BattleState, battler, bit) -> jnp.ndarray:
    """``int32``: the defender of Bide's release (bcp.c:2704-2716, D2-14 B2-B4):
    ``lastHitByBattler[battler]`` while that battler has HP, else ``BattleSystem_RandomOpponent``
    (one coin ``bit``, ``Outcome.bide_target_draw``) when its pick has HP, else ``NO_DEFENDER``
    (``subscript_bide_no_target.s``). ``last_hit_by`` is written at B = 4 only; at B = 2 it is
    always 0 and the opponent is the foe, the one battler that ever hit the user."""
    fmt = format_of(state)
    hit = state.last_hit_by[battler].astype(_I32) - 1
    hit_alive = (hit >= 0) & alive(state, jnp.clip(hit, 0, fmt.B - 1))
    pick, _ = random_opponent(state, battler, bit)
    other = jnp.where(alive(state, pick), pick, NO_DEFENDER)
    return jnp.where(hit_alive, hit, other).astype(_I32)
