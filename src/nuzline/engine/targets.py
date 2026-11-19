"""The fight command's defender: ``BattleSystem_Defender`` (battle_lib.c:1600-1730), the one
resolver of ledger groups D2-4 .. D2-8 (rule sheets ``_scratch/e2/rules/d2/D2-4.md`` section R).

``resolve`` evaluates the ten branches of the decomp's range switch in its order and returns the
defender, the battler the hit reads as its target, the spread loop's ``battlerCounter`` and
whether the call consumed its one ``RandNext()``. At ``B = 2`` (a static fact of the state's
shapes) every branch gives today's singles value and no branch reads ``bit``; the doubles-only
arms sit under a Python ``if doubles:`` (docs/doubles-state-design.md section 5, rule 1).

Battler geometry (D2-4 R2): side ``b & 1``, partner ``b ^ 2``, opponent slots
``BattleSystem_GetEnemyInSlot`` (``opponents``). "Alive" is ``curHP != 0``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import (
    hidden_power,
    judgment_type,
    natural_gift_params,
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
    Weather,
)
from nuzline.engine.state import BattleState, held_item

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


def follow_me_target(state: BattleState, attacker) -> jnp.ndarray:
    """``int32``: the living Follow Me user on the other side of ``attacker`` (the readers of
    ``BattleSystem_Defender``, lib.c:1694-1697, :1714-1717): ``ts_follow_me[side ^ 1] - 1`` when
    that field is set and the Pokemon now in the user's battler slot has HP, else ``NO_DEFENDER``
    (D2-11 rule 4: nothing clears ``followMe`` on a faint or switch; the readers test the slot's
    ``curHP``)."""
    fmt = format_of(state)
    attacker = jnp.asarray(attacker, _I32)
    value = state.ts_follow_me[side_of(attacker) ^ 1].astype(_I32)
    user = jnp.clip(value - 1, 0, fmt.B - 1)
    return jnp.where((value != 0) & alive(state, user), user, NO_DEFENDER)


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


def fight_randomizes(state: BattleState, side, struggling) -> jnp.ndarray:
    """``bool``: the fight command's ``randomizeTarget`` (bcp.c:1861-1903, D2-9 rule 1).
    ``struggling`` (:1866-1868), or an Encore on the attacker at the action's start (:1869-1884):
    both Encore branches randomize, whether or not the slot still holds the encored move. The
    lock branch (:1885-1888) and the plain slot (:1894-1896) do not; a locked battler is never
    struggling. Branch 1.5 (:1889-1893, the selected move left its slot) cannot occur: the move is
    read from the slot at the action (``levelup.learn_pin``)."""
    return jnp.asarray(struggling, bool) | (state.encore_move[side] != 0)


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
        & (state.weather == Weather.SUN)
        & ~weather_suppressed(tables, state)
    )
    herb = tables.item_effect[held_item(state, battler)] == ItemId.POWER_HERB
    keeps = (jnp.asarray(defender, _I32) == NO_DEFENDER) & multi_turn(tables, move)
    return keeps & ~no_target(tables, state, battler, move, defender, False) & ~sun_now & ~herb


def _ability(state: BattleState, battler) -> jnp.ndarray:
    """``int32``: ``Battler_Ability`` of ``battler`` (lib.c:3088-3106; D2-12 rule 11). The one read
    of a battler's ability in this module: E3's suppressed-ability function (Gastro Acid) replaces
    this body when it lands in hooks.py."""
    fmt = format_of(state)
    battler = jnp.asarray(battler, _I32)
    return state.ability[party(fmt, battler), state.active[battler]].astype(_I32)


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

    B = 2 keeps today's singles values (ledger D2-4 pending rulings (i), (ii)): ``chosen`` is
    read only through a non-Ghost Curse (the user), every other single-target or randomized move
    gives the foe when alive, else ``NO_DEFENDER``; Follow Me names the foe there (its user) so
    its arm changes nothing."""
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
        # A non-Ghost Curse selected the user as its target (bcp.c:4344-4346, :4363-4364): today's
        # self defender, kept for a randomized call too (ledger ruling D2-12).
        curse_self = (tables.move_effect[move] == EffectId.CURSE) & ~_ghost(state, attacker)
        live_foe = jnp.where(alive(state, foe), foe, none)
        random_def = jnp.where(curse_self, attacker, live_foe)
        single_def = random_def
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
    other = (defender != none) & (defender != attacker)
    target_side = jnp.where(other, defender, foe).astype(_I32)
    return Resolved(defender, target_side, counter, jnp.asarray(drew, bool))
