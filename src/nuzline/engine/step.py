"""One battle turn: ``step(tables, state, actions, outcome) -> state``.

Pure, jit- and vmap-able, fixed shapes; every random event comes from
``outcome`` (``step`` draws nothing). The turn follows spec §4.2's six phases:

1. The opponent's choice is made outside ``step`` (``actions`` is an input).
2. Order: ``order.action_order``.
3. ``execute_actions``: the first actor, a faint check, the second actor, a faint
   check (``BattleControllerPlayer_CheckAnyFainted`` after each action). A switch
   facing Pursuit is preceded by the Pursuit hit and its own faint check.
4. ``field_conditions``: ``BattleControllerPlayer_CheckFieldConditions``
   (battle_controller_player.c:932-1254): the side-condition countdowns, the weather
   countdown and the sandstorm / hail chip with one faint check after it, then Gravity.
   ``end_of_turn``: the checks of ``END_OF_TURN_ORDER`` for each battler in
   ``end_of_turn_order``, each followed by a faint check (the decomp re-enters
   ``BattleControllerPlayer_CheckMonConditions``, battle_controller_player.c:1289,
   through ``AnyFainted`` / ``CheckBattleOver`` after every check that ran a script).
5. Faint replacement is a separate decision point outside ``step``:
   ``needs_replacement`` and ``replace``.
6. The battle ends at the first faint check that finds no player Pokemon with HP
   (a loss, a simultaneous last opponent faint included) or no opponent left (a win)
   (``_faint_check``); later checks of the turn do nothing. A player faint otherwise
   only records a death.

Move locks (Bide, Rollout, Ice Ball; ``actions.locked_slot``): a locked battler's action is
its locked move and spends no PP (battle_controller_player.c:335); every script that
disrupts a move unlocks it (``_unlock``), and a sleeping battler is unlocked when the
next turn is set up (``_unlock_sleepers``, battle_lib.c:2240).

Implemented so far: the opponent's bag items (Potion, ``_use_trainer_item``, with the
decomp's multi-slot consumption, ``_consume_trainer_items``), the AI inputs
``fake_out_turn`` (``_entered``) and ``move_hit`` (``_use_move``, ``_move_end``), the
AI's knowledge of the player (``last_move``: ``_use_move``, ``_pursuit_hit``;
``ai_seen_moves``: ``_record_last_move``, ``_forget``; ``ai_ability_known``:
``hooks.reveal_ability``, ``_forget``),
switching (with Natural Cure) and the switch-in hooks
(``hooks.on_switch_in``: Stealth Rock, Intimidate, berries), damaging moves, the
accuracy check, PP, Struggle, the move effects of ``effects.py``, the berry check
after every hit (``hooks.after_damage_items``), Pursuit's pre-switch hit, the sleep,
freeze, flinch, Taunt, confusion, paralysis and attract checks before a move, the
held-item, Leech Seed, poison, toxic, burn, Bind and Taunt end-of-turn checks, the field-condition
pass (``field_conditions``), faints and the end of the battle. A player faint is a death, not the end (spec §4): the battle ends
when a side has no Pokemon with HP left. A state that is already over is returned unchanged.

``step_events`` plays the same turn and also returns what each action showed (the move
used and the result of each status check, hits and crits: ``ACT_*``, ``EVENT_FLAGS``),
recorded where ``_use_move`` and ``execute_actions`` run it; ``engine.observe`` builds
the player's observation from them, ``step`` drops them.

Not implemented: Wish's heal in the field pass (no supported move sets ``wish_timer``) and
the Trick Room countdown of the turn end's side-condition pass
(battle_controller_player.c:1823-1826): a nonzero ``trick_room`` never counts down.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION, locked_slot
from nuzline.engine.damage import (
    PRESENT_HEAL,
    confusion_damage,
    decomp_abilities,
    weather_suppressed,
)
from nuzline.engine.effects import (
    apply_move_effect,
    big_root,
    change_stage,
    consume_item,
    defender_always,
    focus_sash_damage,
    hit_count,
    holds,
    play_hit,
    protect_blocks,
    recovers_at_full_hp,
    resist_berry,
    resist_berry_damage,
    self_defender,
    unlock_move,
)
from nuzline.engine.hooks import (
    STATUS_CURE,
    after_damage_items,
    held_item_check,
    on_switch_in,
    recover_status_by_ability,
    reveal_ability,
)
from nuzline.engine.ids import (
    POTION_HP_RESTORED,
    PROTECT_CHAIN_MOVES,
    QUICK_CLAW_MODULUS,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveTarget,
    Stat,
    Status,
    TrainerItemId,
    Type,
    Weather,
)
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    is_item,
    is_switch,
    player_second,
    quick_claw_active,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome, live_hit_threshold
from nuzline.engine.state import BAG_SLOTS, BattleState

_I32 = jnp.int32
_unlock = unlock_move  # Battler_UnlockMoveChoice (battle_lib.c:1834): (state, side, cond)
WINNER_OPPONENT, WINNER_PLAYER = 1, 2
PURSUIT_POWER_MUL = 20  # BtlCmd_TryPursuit (battle_script.c:6884): powerMul = 20
NORMAL_POWER_MUL = 10  # BattleContext_Init (battle_lib.c:1980): powerMul = 10
# MON_CONDITION_TOXIC_COUNTER is 4 bits (include/constants/battle/condition.h:23).
TOXIC_COUNTER_MAX = 15

# Per-side fields of the active Pokemon that leave with it: the incoming battler is
# rebuilt from its party mon by ``BattleSystem_InitBattleMon`` (battle_lib.c:70), stat
# stages reset to neutral (:97, no Baton Pass in M1). Hazards and screens are per side
# and stay. ``move_hit`` is cleared by ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:
# 2121) and ``BattleSystem_CleanupFaintedMon`` (:2199); ``fake_out_turn`` is set there
# instead (``_entered``).
_VOLATILE = (
    "confusion",
    "leech_seed",
    "substitute_hp",
    "taunt",
    "trapped",
    "trap_move",
    "flinched",
    "choice_lock",
    "last_move",
    "protect_count",
    "rollout_count",
    "charge_turn",
    "recharge",
    "pursuit_target",
    "water_sport",
    "move_hit",
    "foresight",
    "disabled_move",
    "disable_turns",
    "locked_move",
    "bide_turns",
    "bide_damage",
    "curled",
    "mud_sport",
    "aqua_ring",
    "infatuated",
)
# BtlCmd_TryProtection caps protectSuccessTurns at NELEMS(sProtectSuccessRate) - 1
# (battle_script.c:5182-5184; outcome.PROTECT_RATES).
PROTECT_CHAIN_MAX = 3

# The turn's events (``step_events``, ``turn_events``): what each side's action showed,
# recorded where ``execute_actions`` runs it. ``engine.observe`` reads them; ``step``
# drops them. ``act_kind`` values:
ACT_NONE, ACT_MOVE, ACT_SWITCH, ACT_ITEM, ACT_TAUNTED, ACT_DISABLED = 0, 1, 2, 3, 4, 5
# Per-side event flags, in the observation's order (engine/observe.py documents each).
EVENT_FLAGS = (
    "missed",
    "no_target",
    "hits",
    "full_para",
    "flinched",
    "asleep",
    "woke",
    "frozen",
    "thawed",
    "thawed_by_hit",
    "confused",
    "snapped_out",
    "self_hit",
    "contact_ability",
    "protected",
    "protect_ok",
    "protect_failed",
    "in_love",
    "immobilized",
    "quick_claw",
)


def _where(cond, a: BattleState, b: BattleState) -> BattleState:
    """Field-wise ``jnp.where(cond, a, b)``."""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def _release_bind(state: BattleState, side, cond) -> BattleState:
    """End the Bind on ``side`` where ``cond``."""
    return state._replace(
        trapped=state.trapped.at[side].set(jnp.where(cond, 0, state.trapped[side])),
        trap_move=state.trap_move.at[side].set(jnp.where(cond, 0, state.trap_move[side])),
    )


def _source_leaves(state: BattleState, side, cond) -> BattleState:
    """``side``'s active battler leaves the field where ``cond`` (a switch, a faint, and Roar's
    drag): every effect it is the source of on the other side ends.
    ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2088-2098) and
    ``BattleSystem_CleanupFaintedMon`` (:2168-2186) clear the ATTRACT bit sourced from it
    (``FlagIndex(battler) << VOLATILE_CONDITION_ATTRACT_SHIFT``) and the Bind it put on a foe.
    In singles the only possible source of the foe's ``infatuated`` is this battler."""
    state = _release_bind(state, 1 - side, cond)
    keep = state.infatuated[1 - side]
    value = jnp.where(cond, 0, keep).astype(keep.dtype)
    return state._replace(infatuated=state.infatuated.at[1 - side].set(value))


def _set_mon(state: BattleState, field: str, side, slot, value, cond) -> BattleState:
    """``state.<field>[side, slot] = value`` where ``cond``."""
    arr = getattr(state, field)
    new = jnp.where(cond, value, arr[side, slot]).astype(arr.dtype)
    return state._replace(**{field: arr.at[side, slot].set(new)})


def _natural_cure(state: BattleState, side) -> BattleState:
    """``BtlCmd_TryRestoreStatusOnSwitch`` (battle_script.c:9139) clears the status of a
    switching battler with HP left and Natural Cure. ``subscript_switch_pokemon.s:5-6``
    runs it before Pursuit hits (:11), so ``execute_actions`` applies it before the
    Pursuit hit; repeating it in ``_send_out`` changes nothing."""
    mon = state.active[side]
    cure = (state.ability[side, mon] == AbilityId.NATURAL_CURE) & (state.hp[side, mon] > 0)
    state = _set_mon(state, "status", side, mon, Status.NONE, cure)
    return _set_mon(state, "status_counter", side, mon, 0, cure)


def _switch_out(state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon leaves by a switch action (``subscript_switch_pokemon.s``):
    Natural Cure (``_natural_cure``), and the toxic counter restarts: battle_controller.c:
    1482 writes the status back to the party without ``MON_CONDITION_TOXIC_COUNTER``, and
    a returning battler reads it from there. A sleep counter is part of the status and
    stays."""
    state = _natural_cure(state, side)
    mon = state.active[side]
    toxic = state.status[side, mon] == Status.TOXIC
    return _set_mon(state, "status_counter", side, mon, 0, toxic)


def _reset_battler(state: BattleState, side, cond) -> BattleState:
    """Stages and ``_VOLATILE`` fields of ``side``'s active battler back to neutral
    where ``cond``."""
    fields = {
        name: getattr(state, name).at[side].set(jnp.where(cond, 0, getattr(state, name)[side]))
        for name in (*_VOLATILE, "stages")
    }
    return state._replace(**fields)


def _entered(state: BattleState, side, entry_turn, cond) -> BattleState:
    """``fakeOutTurnNumber = totalTurns + 1`` for ``side``'s battler where ``cond``:
    ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2117) and
    ``BattleSystem_CleanupFaintedMon`` (:2195). ``entry_turn`` is that
    ``totalTurns + 1``: ``turn + 1`` inside ``step``, ``turn`` in ``replace`` (which runs
    after ``step`` has counted the turn; ``state`` module docstring)."""
    value = jnp.where(cond, entry_turn, state.fake_out_turn[side])
    fake_out_turn = state.fake_out_turn.at[side].set(value.astype(state.fake_out_turn.dtype))
    return state._replace(fake_out_turn=fake_out_turn)


def _set_last_move(state: BattleState, side, move, cond) -> BattleState:
    """``movePrevByBattler[side] = move`` where ``cond`` (``BattleState.last_move``)."""
    value = jnp.where(cond, move, state.last_move[side]).astype(state.last_move.dtype)
    return state._replace(last_move=state.last_move.at[side].set(value))


def _record_last_move(tables: Tables, state: BattleState, actions) -> BattleState:
    """``TrainerAI_RecordLastMove`` (trainer_ai.c:2705-2717), at the start of the turn:
    the AI adds the player's ``movePrevByBattler`` to its list of the player's moves.
    It runs only on the AI's FIGHT path (``TrainerAI_MainSingles``, :295, after
    ``TrainerAI_PickCommand`` chose neither a switch nor an item, :3989), and not when
    the opponent must Struggle: the move menu, and with it ``TrainerAI_Main``, is
    skipped for ``STRUGGLING_ALL`` (battle_controller_player.c:433) and for a locked
    opponent (:335). The list holds
    the moves as slots of the player's active Pokemon (``ai_seen_moves``); Struggle is
    in no slot and is not recorded (docs/discrepancies.md)."""
    mon = state.active[0]
    pending = state.last_move[0]
    _, struggling = selected_move(tables, state, 1, actions[1])
    locked = locked_slot(state, 1) >= 0  # a locked battler picks no command (:335)
    fight = (actions[1] < MOVE_ACTIONS) & ~struggling & ~locked
    match = (state.moves[0, mon] == pending) & (pending != 0) & fight
    seen = state.ai_seen_moves
    row = jnp.where(match, 1, seen[mon]).astype(seen.dtype)
    return state._replace(ai_seen_moves=seen.at[mon].set(row))


def _forget(state: BattleState, side, slot) -> BattleState:
    """The AI forgets what it knew about the player's party ``slot`` when it enters
    (``BattleSystem_UpdateAfterSwitch`` clears the battler's known moves and ability,
    battle_lib.c:2151-2152); nothing for the opponent's side."""
    on = jnp.asarray(side, _I32) == 0
    return state._replace(
        ai_seen_moves=state.ai_seen_moves.at[slot].set(
            jnp.where(on, 0, state.ai_seen_moves[slot]).astype(state.ai_seen_moves.dtype)
        ),
        ai_ability_known=state.ai_ability_known.at[slot].set(
            jnp.where(on, 0, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
        ),
    )


def _switch_in(state: BattleState, side, slot, entry_turn) -> BattleState:
    """``side`` sends out party ``slot``: stages and volatiles reset, the effects the
    outgoing Pokemon is the source of end (``_source_leaves``: the Bind it put on the other
    side, the other side's infatuation), ``fake_out_turn`` becomes ``entry_turn`` and the AI
    forgets the newcomer (``_forget``; ``BattleSystem_UpdateAfterSwitch``, battle_lib.c:2094)."""
    state = _reset_battler(state, side, True)
    state = _forget(state, side, slot)
    state = _entered(state, side, entry_turn, True)
    state = state._replace(active=state.active.at[side].set(jnp.asarray(slot).astype(jnp.int8)))
    return _source_leaves(state, side, True)


def _send_out(tables: Tables, state: BattleState, side, slot) -> BattleState:
    """A switch action: ``_switch_out``, ``_switch_in``, then ``hooks.on_switch_in``
    (entry hazards, Intimidate, berries)."""
    entered = _switch_in(_switch_out(state, side), side, slot, state.turn + 1)
    return on_switch_in(tables, entered, side)


def _empty_events() -> dict:
    """The turn's events before any action ran: every per-side field 0."""
    side = jnp.zeros((2,), _I32)
    events = {
        name: side
        for name in ("act_kind", "act_id", "shown", "magnitude", "present_heal", *EVENT_FLAGS)
    }
    events["crit"] = jnp.zeros((2, MAX_HITS), _I32)
    return events


def _put(events: dict, side, values: dict, cond) -> dict:
    """``events[name][side] = values[name]`` where ``cond``."""
    out = dict(events)
    for name, value in values.items():
        arr = events[name]
        new = jnp.where(cond, jnp.asarray(value).astype(arr.dtype), arr[side])
        out[name] = arr.at[side].set(new)
    return out


def _hit_events(before: BattleState, after: BattleState, side, landed, crit) -> dict:
    """What one hit of ``side``'s move showed, from the states around it: a crit on a
    landed hit that dealt damage (the target's HP drop: only the hit itself lowers it,
    effects.py:441-444; a crit into a substitute is not seen, a documented coarsening in
    ``observe``'s docstring), the defender's contact ability (the only change of the
    attacker's own status, and Cute Charm the only change of its ``infatuated``, inside a
    hit: effects._contact_ability) and a Fire hit's thaw (the only FREEZE -> NONE change of
    the target, effects._thaw_target)."""
    foe = 1 - side
    user, target = before.active[side], before.active[foe]
    dealt = before.hp[foe, target] > after.hp[foe, target]
    contact = (before.status[side, user] == Status.NONE) & (after.status[side, user] != Status.NONE)
    contact = contact | ((before.infatuated[side] == 0) & (after.infatuated[side] != 0))
    thaw = (before.status[foe, target] == Status.FREEZE) & (
        after.status[foe, target] == Status.NONE
    )
    return {
        "crit": landed & dealt & crit,
        "contact": contact,
        "thaw": thaw,
        "dealt": landed & dealt,
    }


def _use_move(
    tables: Tables,
    state: BattleState,
    side,
    action,
    choice,
    outcome: Outcome,
    target_moved,
    foe_protecting=False,
) -> tuple[BattleState, dict, dict]:
    """``side``'s active Pokemon uses the move in slot ``action`` (or Struggle).
    ``choice``: its ``(move, struggling, locked)`` from ``selected_move`` and
    ``locked_slot`` at the start of the turn, when the decomp fixes Struggle
    (battle_controller_player.c:433), so a Taunt landing earlier this turn makes a chosen
    power-0 move fail instead of Struggling, and a battler locked then spends no PP
    (``ppDecremented`` is set at command selection, :335).
    ``target_moved``: the other side has already acted this turn. ``foe_protecting``: its
    Protect succeeded earlier this turn (the decomp's ``turnFlags.protecting``, a per-turn
    flag cleared at the turn end, battle_lib.c:2231, and on a faint, :2190-2193).

    Returns ``(state, own, foe)``: the state after the move, the user's events (each
    status check's result, effect-gated: set only when the check was reached and had an
    effect) and the foe's (``contact_ability``, ``thawed_by_hit``)."""
    move, struggling, locked = choice
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    target_side = 1 - side
    target = state.active[target_side]
    ready = (state.over == 0) & (state.hp[side, user] > 0)
    # BattleControllerPlayer_FightCommand (battle_controller_player.c:1900) fixes the
    # defender before the status checks: the foe when it is alive (any HP for
    # RANGE_OPPONENT_SIDE: BattleSystem_RandomOpponent), the user for a self-defender range
    # and a non-Ghost Curse (effects.self_defender).
    move_range = tables.move_target[move]
    foe_defender = ~self_defender(tables, move) & (
        (state.hp[target_side, target] > 0) | (move_range == MoveTarget.RANGE_OPPONENT_SIDE)
    )

    # BattleControllerPlayer_CheckStatusDisruption (battle_controller_player.c:2417) runs
    # before the PP decrement; the first check that disrupts the move stops the rest.
    # CHECK_STATUS_STATE_SLEEP (:2430): the counter drops by 1 (Early Bird is not
    # implemented); still asleep disrupts the move, while waking up (subscript_wake_up)
    # clears the status and the checks go on.
    status = state.status[side, user]
    asleep = ready & (status == Status.SLEEP)
    sleep_left = jnp.maximum(state.status_counter[side, user].astype(_I32) - 1, 0)
    sleeping = asleep & (sleep_left > 0)
    state = _set_mon(state, "status_counter", side, user, sleep_left, asleep)
    state = _set_mon(state, "status", side, user, Status.NONE, asleep & ~sleeping)
    # CHECK_STATUS_STATE_FREEZE (:2478): RandNext() % 5 == 0 (outcome.thaw) thaws and the
    # move goes on; otherwise the move is disrupted unless it is a thawing move (Flame
    # Wheel, :2481), whose user thaws at CHECK_STATUS_STATE_SELF_THAW (:2728) below.
    thaw_move = (tables.move_effect[move] == EffectId.SECONDARY_STATUS) & (
        tables.move_param[move, 2] == 1
    )
    frozen = ready & (status == Status.FREEZE)
    thawed = frozen & (outcome.thaw[side] != 0)
    still_frozen = frozen & ~thawed
    state = _set_mon(state, "status", side, user, Status.NONE, thawed)
    go = ready & ~sleeping & ~(still_frozen & ~thaw_move)
    # CHECK_STATUS_STATE_FLINCH (:2529); subscript_flinched.s:10-14 gives Steadfast +1 Speed.
    flinched = go & (state.flinched[side] != 0)
    steadfast = flinched & (state.ability[side, user] == AbilityId.STEADFAST)
    state = change_stage(state, side, Stat.SPE, 1, steadfast)
    state = _unlock(state, side, flinched)  # subscript_flinched.s:9
    go = go & ~flinched
    # CHECK_STATUS_STATE_DISABLE (:2543): the move chosen is the disabled one (no PP spent).
    disabled = go & (state.disabled_move[side] != 0) & (state.disabled_move[side] == move)
    state = _unlock(state, side, disabled)  # subscript_move_is_disabled.s
    go = go & ~disabled
    # CHECK_STATUS_STATE_TAUNT (:2557): a taunted battler's power-0 move fails.
    taunted = go & (state.taunt[side] > 0) & (tables.move_power[move] == 0)
    state = _unlock(state, side, taunted)  # subscript_move_fail_taunted.s
    go = go & ~taunted
    # CHECK_STATUS_STATE_CONFUSION (:2614): the counter drops first; at 0 the battler snaps
    # out and moves, otherwise RandNext() & 1 (outcome.confusion_hit) makes it hit itself
    # (:2633, damage.confusion_damage) instead of moving. subscript_hurt_self_in_confusion.s
    # runs CheckHoldOnWith1HP on the user (script.c:9098-9124): a Focus Sash at full HP
    # leaves 1 HP, and the followup message removes it (the user is the defender, :2629).
    # Its BATTLE_SUBSCRIPT_UPDATE_HP then runs the resist-berry check on the user
    # (effects.resist_berry): BattleContext_Init left moveType Normal and moveStatusFlags 0
    # (battle_lib.c:1981-1982), so calcTemp is the table type of the move the battler
    # chose (moveCur, bcp.c:1898; Hidden Power's too) and nothing is super-effective: a
    # Chilan Berry halves the self-hit when that move is Normal-type, a Rindo Berry never.
    confused = go & (state.confusion[side] > 0)
    confusion = state.confusion[side] - confused.astype(state.confusion.dtype)
    still_confused = confused & (confusion > 0)
    self_hit = still_confused & (outcome.confusion_hit[side] != 0)
    user_hp = state.hp[side, user].astype(_I32)
    sash = holds(tables, state, side, ItemId.FOCUS_SASH)
    full = user_hp == state.max_hp[side, user].astype(_I32)
    raw = confusion_damage(tables, state, side, outcome.roll[side, 0])
    hurt, endured = focus_sash_damage(state, side, raw, sash & full)
    resisted = self_hit & resist_berry(tables, state, side, tables.move_type[move], False)
    taken = jnp.where(endured, user_hp - 1, raw)
    hurt = resist_berry_damage(state, side, taken, resisted, hurt)
    user_hp = user_hp - jnp.where(self_hit, hurt, 0)
    state = state._replace(
        confusion=state.confusion.at[side].set(confusion),
        hp=state.hp.at[side, user].set(user_hp.astype(state.hp.dtype)),
    )
    state = consume_item(state, side, (self_hit & endured) | resisted)
    state = _unlock(state, side, self_hit)  # subscript_hurt_self_in_confusion.s
    go = go & ~self_hit
    # CHECK_STATUS_STATE_PARALYSIS (:2652): RandNext() % 4 == 0 (outcome.full_para) fully
    # paralyzes (Magic Guard is not implemented).
    # Gated on reaching the check, so the event is set only when it had an effect.
    full_para = go & (state.status[side, user] == Status.PARALYSIS) & (outcome.full_para[side] != 0)
    state = _unlock(state, side, full_para)  # subscript_fully_paralyzed.s
    go = go & ~full_para
    # CHECK_STATUS_STATE_ATTRACT (:2669-2692): a battler in love (infatuated now: a faster
    # Attract this turn counts) shows "in love" (subscript_infatuated.s); RandNext() & 1 == 0
    # (outcome.infatuation) immobilizes it: subscript_immobilized_by_love.s unlocks it and
    # goes to UPDATE_MOVE_BUFFERS with no attack message (no PP, last_move 0, the Protect chain
    # resets, moveHit MOVE_NONE), like full paralysis.
    in_love = go & (state.infatuated[side] != 0)
    immobilized = in_love & (outcome.infatuation[side] != 0)
    state = _unlock(state, side, immobilized)
    can_act = go & ~immobilized
    # CHECK_STATUS_STATE_BIDE (:2694): the Bide counter drops. On reaching 0 the unleash
    # deals 2 * storedDamage (damage.fixed_damage) at the last hitter, i.e. the foe's
    # active Pokemon, a replacement included. Nothing stored (subscript_bide_end.s:
    # "But it failed!") or the foe fainted (subscript_bide_no_target.s) ends the move
    # here; either way the lock is released after the move (below).
    biding = can_act & (state.bide_turns[side] > 0)
    bide_left = jnp.maximum(state.bide_turns[side].astype(_I32) - 1, 0)
    state = state._replace(
        bide_turns=state.bide_turns.at[side].set(
            jnp.where(biding, bide_left, state.bide_turns[side]).astype(state.bide_turns.dtype)
        )
    )
    bide_ends = biding & (bide_left == 0)
    bide_stop = bide_ends & ((state.bide_damage[side] <= 0) | (state.hp[target_side, target] <= 0))
    can_act = can_act & ~bide_stop
    # CHECK_STATUS_STATE_SELF_THAW (:2728): a still-frozen user of a thawing move thaws.
    self_thaw = can_act & still_frozen
    state = _set_mon(state, "status", side, user, Status.NONE, self_thaw)

    # BattleControllerPlayer_DecrementPP (battle_controller_player.c:3182) runs before the
    # no-target check (:3190); Struggle has no PP to spend.
    spend = can_act & ~struggling & ~locked
    pp = state.pp.at[side, user, slot].add(jnp.where(spend, -1, 0).astype(state.pp.dtype))
    state = state._replace(pp=pp)

    # Accuracy 0 never misses (CheckMoveHitAccuracy returns before rolling, :2901). A
    # multi-hit move checks accuracy on its first hit only (effects.py).
    # The no-target check (:3190) fails a move whose defender has fainted; a move whose
    # defender exists whatever the foe's HP (effects.defender_always) never fails there.
    never_misses = tables.move_accuracy[move] == 0
    has_target = (state.hp[target_side, target] > 0) | defender_always(tables, move)
    # SYSCTL_MOVE_SUCCEEDED (battle_controller_player.c:3212) once the move got past the
    # status checks, the PP decrement and the no-target check, before the accuracy check
    # (a miss counts): UpdateMoveBuffers (:3945-3949) writes the move (Struggle's own id
    # when struggling) to movePrevByBattler, else MOVE_NONE. Every path of a battler
    # that acts reaches it, the confusion self-hit too (LOOP_FAINTED, :2639).
    succeeded = can_act & has_target
    state = _set_last_move(state, side, jnp.where(succeeded, move, 0), ready)
    # The move script runs next (BATTLE_CONTROL_EXEC_SCRIPT, :3214): Fake Out's
    # (effect_script_0158.s) sets MOVE_STATUS_FAILED unless fakeOutTurnNumber == totalTurns.
    # The accuracy roll still happens, but subscript_missed.s prints "But it failed!" (:97)
    # ahead of the miss message, so only an attempt that did not fail can show a miss.
    failed = (tables.move_effect[move] == EffectId.FAKE_OUT) & (
        state.fake_out_turn[side].astype(_I32) != state.turn.astype(_I32)
    )
    # Selfdestruct / Explosion (effect_script_0007.s): the script sets the user's HP to 0
    # before TryMove, so the user faints on a miss or into a Ghost too, and its own Cloud
    # Nine no longer counts for the hit (COUNT_ALIVE_BATTLERS). The faint itself is
    # BATTLE_CONTROL_FAINT_AFTER_SELFDESTRUCT (bcp.c:3889), within this action, so the
    # action's faint check sees it with the target's. Damp (the script's other branch) is
    # an unsupported ability.
    self_destruct = can_act & has_target & (tables.move_effect[move] == EffectId.SELF_DESTRUCT)
    state = _set_mon(state, "hp", side, user, 0, self_destruct)
    # CheckMoveHitAccuracy reads the threshold now, with this moment's stages (bcp.c:
    # 2873-2874): a faster Sand-Attack or Double Team this turn counts. outcome.hit[side, 0]
    # is the roll's class (outcome.py): it hits when 0 < class <= threshold (:2976).
    roll = outcome.hit[side, 0].astype(_I32)
    threshold = live_hit_threshold(tables, state, side, move)
    hit = never_misses | ((roll > 0) & (roll <= threshold))
    # CheckMoveHitOverrides (bcp.c:3001-3014) runs after the accuracy roll, hit or miss: a
    # protecting foe stops the move (effects.protect_blocks), MOVE_STATUS_PROTECTED, and
    # Battler_UnlockMoveChoice (the unlock below). Selfdestruct's user already fainted.
    blocked = jnp.asarray(foe_protecting) & protect_blocks(tables, move, bide_ends)
    connected = can_act & has_target & hit & ~failed & ~blocked
    # Protect / Detect (effect_script_0111.s, a direct TO_ATTACKER effect: the user has HP)
    # -> subscript_protect.s -> BtlCmd_TryProtection (battle_script.c:5149-5190). The
    # chain reset of :5154-5158 is folded into protect_count (below), so its value here is
    # protectSuccessTurns. Success iff sProtectSuccessRate[n] >= RandNext()
    # (outcome.protect; n = 0 never fails) and the user is not the last battler to act
    # (waitingBattlers != 1, bcp.c:881-886: in singles, it acts first): turnFlags.protecting
    # and the count up to PROTECT_CHAIN_MAX; else the count drops to 0 ("But it failed!").
    count = state.protect_count[side].astype(_I32)
    tried = connected & (tables.move_effect[move] == EffectId.PROTECT)
    protect_ok = tried & ~target_moved & ((count == 0) | (outcome.protect[side] != 0))
    protect_failed = tried & ~protect_ok
    count = jnp.where(
        protect_ok, jnp.minimum(count + 1, PROTECT_CHAIN_MAX), jnp.where(protect_failed, 0, count)
    )
    # UpdateMoveBuffers (bcp.c:3936-3942): moveProtect = the move when the attack message
    # was shown (the move got past the status checks: can_act), else MOVE_NONE, on every
    # path of a battler that acts. The next TryProtection resets the chain unless
    # moveProtect is Protect, Detect or Endure (script.c:5154-5158); the engine folds that
    # reset in here, and AICmd_LoadProtectChain (trainer_ai.c:2546-2552) reads the folded
    # value. A switch, a bag item and Pursuit's pre-switch hit (TryPursuit ends the action
    # without UpdateMoveBuffers, script.c:6931) leave it alone.
    chain = jnp.asarray(PROTECT_CHAIN_MOVES, _I32)
    keep = can_act & jnp.isin(jnp.asarray(move, _I32), chain)
    count = jnp.where(ready & ~keep, 0, count)
    state = state._replace(
        protect_count=state.protect_count.at[side].set(count.astype(state.protect_count.dtype))
    )
    # subscript_no_target.s and subscript_missed.s unlock the attacker (a Rollout that
    # missed; a first use never locked, so this is then a no-op).
    state = _unlock(state, side, can_act & ~connected)
    n_hits = hit_count(tables, move, outcome, side)
    # Present's heal path (effects._present): subscript_recover_hp.s prints "regained health"
    # (1), or "HP is full!" (2) for a target at full HP; the target's HP is unchanged
    # between here and the hit.
    present_heal = (
        connected
        & (tables.move_effect[move] == EffectId.PRESENT)
        & (outcome.present[side] == PRESENT_HEAL)
    )
    target_full = state.hp[target_side, target] == state.max_hp[target_side, target]
    present_heal = jnp.where(present_heal, jnp.where(target_full, 2, 1), 0)
    # A Recover / Present heal on a full-HP recipient (single-hit moves): MOVE_STATUS_NO_MORE_WORK
    # skips the move's AfterMoveEffects (effects.recovers_at_full_hp), so no item check runs in
    # any loop pass below (the passes after the hit would otherwise repeat it).
    no_more_work = connected & recovers_at_full_hp(tables, state, side, move, outcome)

    # A loop, not unrolled: one copy of the effect code in the program, not MAX_HITS
    # (unrolled, it dominated trace and compile time). It also counts the hits that
    # landed ("Hit N times", battle_controller_player.c:3777-3812) and their crits.
    def one_hit(hit_index, carry):
        state, hits, crits, contact, thaw, dealt, armed = carry
        # BattleControllerPlayer_LoopMultiHit (bcp.c:3780) stops once a battler fainted
        # (faintedMon): the target, or the attacker by Rough Skin (Selfdestruct's user is
        # at 0 HP before its only hit, so the first hit is exempt).
        going = connected & ((hit_index == 0) | (state.hp[side, user] > 0))
        landed = (
            going
            & (hit_index < n_hits)
            & ((state.hp[target_side, target] > 0) | defender_always(tables, move))
        )
        hit_state, info = play_hit(
            tables,
            state,
            side,
            move,
            outcome,
            hit_index,
            connected=going,
            target_moved=target_moved,
            struggling=struggling,
            focus_armed=armed,
        )
        # A LEVEL_DAMAGE move never runs CalcCrit (effect_script_0087.s): no crit message.
        # (nor does Bide's unleash: its script has no CalcCrit).
        crit = (outcome.crit[side, hit_index] != 0) & (
            (tables.move_effect[move] != EffectId.LEVEL_DAMAGE)
            & (tables.move_effect[move] != EffectId.BIDE)
        )
        seen = _hit_events(state, hit_state, side, landed, crit)
        # BattleControllerPlayer_AfterMoveEffects (battle_controller_player.c:3638) runs
        # after every hit of a move that connected: AFTER_MOVE_EFFECT_ATTACKER_ITEM
        # (:3684), then _DEFENDER_ITEM (:3691). A disrupted move, a confusion self-hit,
        # a miss (MOVE_STATUS_DID_NOT_HIT, :3325) or a move with no target skips it. A
        # Pluck / Bug Bite that left SELF_TURN_FLAG_PLUCK_BERRY set (info.pluck_flag) keeps
        # the next item that triggers (its PLUCK_CHECK clears the flag instead of removing
        # the item): the attacker's, as the defender's berry is the one just eaten. Then
        # AFTER_MOVE_EFFECT_HELD_ITEM_STATUS (:3730) re-checks both battlers. Nothing in
        # between changes either (the defender's thaw cures no item's condition), so only
        # an attacker's item kept above can trigger again, and is removed this time. A
        # later hit that did not happen changes no HP, so its check finds what the
        # previous one found.
        items = connected & ~no_more_work & ~protect_failed  # a failed Protect: subscript_missed
        state, used = held_item_check(tables, hit_state, side, items, info.pluck_flag)
        state = after_damage_items(tables, state, 1 - side, items)
        state = after_damage_items(tables, state, side, items & used & info.pluck_flag)
        return (
            state,
            hits + landed.astype(_I32),
            crits.at[hit_index].set(seen["crit"]),
            contact | seen["contact"],
            thaw | seen["thaw"],
            dealt | seen["dealt"],
            info.focus_armed,
        )

    init = (
        state,
        _I32(0),
        jnp.zeros((MAX_HITS,), bool),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
    )
    state, hits, crits, contact, thaw, dealt, _ = jax.lax.fori_loop(0, MAX_HITS, one_hit, init)
    # The unleash script clears the lock (the Bide counter is spent); the move above still
    # saw it locked, so effect_script_0026.s did not restart Bide.
    state = _unlock(state, side, bide_ends)
    # BattleControllerPlayer_UpdateFlagsWhenHit (battle_controller_player.c:4499-4507),
    # reached on every path once the move was chosen (UPDATE_MOVE_BUFFERS, :3956, or
    # LOOP_SPREAD_MOVES, :3837): moveHit[defender] = the move when the attack message
    # was shown (the move got past the status checks, hit or not; the message sets
    # SYSCTL_ATTACK_MESSAGE_SHOWN, battle_script.c:1432), else MOVE_NONE. A confusion
    # self-hit makes the user the defender (battle_controller_player.c:2629), so the
    # foe's record is untouched. ``_move_end`` keeps this write past the faint check, as
    # the decomp writes it after LOOP_FAINTED.
    record = ready & ~self_hit & (foe_defender | (bide_ends & can_act))
    move_hit = jnp.where(record, jnp.where(can_act, move, 0), state.move_hit[target_side])
    state = state._replace(
        move_hit=state.move_hit.at[target_side].set(move_hit.astype(state.move_hit.dtype))
    )
    own = {
        "act_kind": jnp.where(
            can_act,
            ACT_MOVE,
            jnp.where(taunted, ACT_TAUNTED, jnp.where(disabled, ACT_DISABLED, ACT_NONE)),
        ),
        "act_id": jnp.where(can_act | taunted | disabled, move, 0),
        "shown": ready,
        # subscript_missed.s tests PROTECTED before MISSED (:18, :157): a miss into a
        # Protect shows "protected itself"; a failed Fake Out prints "But it failed!" (:58).
        "missed": can_act & has_target & ~hit & ~failed & ~blocked,
        "protected": can_act & has_target & blocked & ~failed,
        "protect_ok": protect_ok,
        "protect_failed": protect_failed,
        "no_target": can_act & ~has_target,
        "hits": hits,
        # subscript_magnitude.s prints "Magnitude N!" through the ON_HIT primary effect
        # (BattleSystem_TriggerPrimaryEffect, battle_lib.c:1481), not run for a move with
        # MOVE_STATUS_DID_NOT_HIT (an immune target): shown when the hit dealt damage.
        "magnitude": jnp.where(
            dealt & (tables.move_effect[move] == EffectId.MAGNITUDE), outcome.magnitude[side], 0
        ),
        "present_heal": present_heal,
        "crit": crits,
        "full_para": full_para,
        "flinched": flinched,
        "asleep": sleeping,
        "woke": asleep & ~sleeping,
        "frozen": still_frozen & ~thaw_move,
        "thawed": thawed | self_thaw,
        "confused": still_confused,
        "snapped_out": confused & ~still_confused,
        "self_hit": self_hit,
        "in_love": in_love,
        "immobilized": immobilized,
    }
    return state, own, {"contact_ability": contact, "thawed_by_hit": thaw}


def _bag_slot(action) -> jnp.ndarray:
    return jnp.clip(action - FIRST_ITEM_ACTION, 0, BAG_SLOTS - 1)


def _consume_trainer_items(state: BattleState, actions) -> BattleState:
    """Each side choosing an item action loses that bag slot when it chooses:
    ``TrainerAI_ShouldUseItem`` (trainer_ai.c:4056) zeroes ``trainerItems[i]`` at command
    selection (:4193), so the item is spent even if its action never runs. Its bag loop
    has no ``break`` (:4091-4194): after the use, every later slot ``i`` it considers,
    ``aliveMons <= trainerItemCounts - i + 1`` (:4092), is zeroed too (an empty one stays
    empty). ``aliveMons`` counts the side's party members with HP left, the active one
    included (:4081-4089), at command selection. The item used stays slot ``k``'s: a
    later zeroed slot holding a different item would become ``usedItem`` in the decomp
    (docs/discrepancies.md)."""
    slots = jnp.arange(BAG_SLOTS)
    for side in (0, 1):
        k = _bag_slot(actions[side])
        alive = jnp.sum(((state.species[side] != 0) & (state.hp[side] > 0)).astype(_I32))
        count = state.trainer_item_count[side].astype(_I32)
        considered = (slots > k) & (alive <= count - slots + 1)
        spent = is_item(actions[side]) & ((slots == k) | considered)
        bag = jnp.where(spent, 0, state.trainer_items[side]).astype(state.trainer_items.dtype)
        state = state._replace(trainer_items=state.trainer_items.at[side].set(bag))
    return state


def _use_trainer_item(tables: Tables, state: BattleState, side, item) -> BattleState:
    """``BattleControllerPlayer_ItemCommand`` (battle_controller_player.c:1905): a
    Potion runs ``subscript_use_potion.s``. ``BattleSystem_UseBagItem``
    (battle_system.c:435) restores ``POTION_HP_RESTORED`` to the active Pokemon when it
    has HP left and is not at full HP; ``BATTLE_SUBSCRIPT_UPDATE_HP`` caps it at max HP."""
    mon = state.active[side]
    hp = state.hp[side, mon].astype(_I32)
    potion = (tables.trainer_item_effect[item] == TrainerItemId.POTION) & (hp > 0)
    healed = jnp.minimum(hp + POTION_HP_RESTORED, state.max_hp[side, mon].astype(_I32))
    return _set_mon(state, "hp", side, mon, healed, potion)


def _act(
    tables: Tables,
    state: BattleState,
    side,
    actions,
    choices,
    items,
    outcome: Outcome,
    target_moved,
    foe_protecting,
) -> tuple[BattleState, dict, dict]:
    """``side``'s action: ``(state, own, foe)`` with the events of ``_use_move``. A
    switch shows the party slot (``ACT_SWITCH``), an item its item id (``ACT_ITEM``)."""
    action = actions[side]
    switched = _send_out(tables, state, side, jnp.clip(action - MOVE_ACTIONS, 0, 5))
    used = _use_trainer_item(tables, state, side, items[side])
    choice = (choices[0][side], choices[1][side], choices[2][side])
    moved, own, foe = _use_move(
        tables, state, side, action, choice, outcome, target_moved, foe_protecting
    )
    live = state.over == 0
    switch, item = is_switch(action) & live, is_item(action) & live
    acted = _where(item, used, moved)
    acted = _where(switch, switched, acted)
    is_move = ~is_switch(action) & ~is_item(action)
    kind = jnp.where(switch, ACT_SWITCH, jnp.where(item, ACT_ITEM, own["act_kind"]))
    act_id = jnp.where(switch, action - MOVE_ACTIONS, jnp.where(item, items[side], own["act_id"]))
    own = {name: jnp.where(is_move, value, 0) for name, value in own.items()}
    own.update(act_kind=kind, act_id=act_id, shown=switch | item | (own["shown"] != 0))
    return acted, own, {name: value & is_move for name, value in foe.items()}


def pursuit_fires(tables: Tables, state: BattleState, side, actions, choices) -> jnp.ndarray:
    """``side`` switches and the other side, which has not acted yet, chose Pursuit:
    ``BtlCmd_TryPursuit`` (battle_script.c:6884, called from
    ``subscript_switch_pokemon.s:11`` before the switch) needs the pursuer alive, not
    asleep or frozen, and PP left in the move (Truant is not implemented)."""
    pursuer = 1 - side
    move, struggling = choices[0][pursuer], choices[1][pursuer]
    mon = state.active[pursuer]
    status = state.status[pursuer, mon]
    return (
        (state.over == 0)
        & is_switch(actions[side])
        & ~is_switch(actions[pursuer])
        & ~is_item(actions[pursuer])
        & ~struggling
        & (tables.move_effect[move] == EffectId.PURSUIT)
        & (state.hp[pursuer, mon] > 0)
        & (status != Status.SLEEP)
        & (status != Status.FREEZE)
    )


def replacement_power_mul(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32`` ``battleCtx->powerMul`` the opponent's forced replacement after the turn
    sees (``post_ko_switch_in``'s ``power_mul``): ``PURSUIT_POWER_MUL`` when the player's
    switch met Pursuit (``pursuit_fires`` on the turn-start ``state``, with each side's
    move fixed when chosen), else ``NORMAL_POWER_MUL``. The one source for playouts
    (``playout._play``) and exact expansion (``expand._turn``), which must agree."""
    picks = [selected_move(tables, state, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    pursued = pursuit_fires(tables, state, 0, actions, choices)
    return jnp.where(pursued, PURSUIT_POWER_MUL, NORMAL_POWER_MUL).astype(_I32)


def _pursuit_hit(
    tables: Tables, state: BattleState, pursuer, actions, choices, outcome: Outcome
) -> BattleState:
    """The pursuer hits the outgoing Pokemon: ``BtlCmd_TryPursuit`` spends the PP and
    sets ``powerMul = 20``; ``subscript_pursuit.s`` then runs crit, damage and the type
    chart with no accuracy check. The pursuer's own action is over; it wrote
    ``movePrevByBattler[pursuer] = move`` (battle_script.c:6930, ``last_move``)."""
    action = actions[pursuer]
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    mon = state.active[pursuer]
    state = state._replace(pp=state.pp.at[pursuer, mon, slot].add(jnp.int8(-1)))
    move = choices[0][pursuer]
    state = _set_last_move(state, pursuer, move, True)
    return apply_move_effect(
        tables,
        state,
        pursuer,
        move,
        outcome,
        0,
        connected=True,
        target_moved=False,
        power_mul=PURSUIT_POWER_MUL,
    )


def _faint_check(before: BattleState, after: BattleState, entry_turn=None) -> BattleState:
    """Count new player faints (``player_faints``) and record their party slots
    (``deaths``); a player faint ends the battle only when no player Pokemon has HP left.
    The battle is over when either side is out: both out at once is
    ``BATTLE_RESULT_WIN | BATTLE_RESULT_LOSE`` = a draw (``include/constants/battle.h:78``,
    ``BattleControllerPlayer_CheckBattleOver``, battle_controller_player.c:4181-4262),
    which ``HandleResult`` plays as a loss (:4015-4016): ``winner = 1`` (ruling 2). A faint
    check runs wherever the decomp's does, so within a turn the side whose check runs
    first loses first. A fainted active battler is cleaned up as
    ``BattleSystem_CleanupFaintedMon`` (battle_lib.c:2156, from ``PlayFaintAnimation``,
    battle_script.c:1803) does: its stat stages, ``statusVolatile``, ``moveEffectsMask``
    and ``moveEffectsData`` are cleared (``_reset_battler``, which also clears
    ``move_hit``, :2199), ``fake_out_turn`` becomes ``entry_turn`` (:2195; by default
    ``turn + 1``, the value inside ``step``), and the effects it is the source of end
    (``_source_leaves``: the Bind on the other side, :2179, and its infatuation,
    :2174-2176). Every newly fainted Pokemon also loses its status condition and counter: the
    decomp writes ``status = 0`` for a Pokemon with no HP (battle_controller.c:1482-1486)."""
    entry_turn = after.turn + 1 if entry_turn is None else entry_turn
    fainted = (before.hp > 0) & (after.hp == 0)
    after = after._replace(
        status=jnp.where(fainted, 0, after.status).astype(after.status.dtype),
        status_counter=jnp.where(fainted, 0, after.status_counter).astype(
            after.status_counter.dtype
        ),
    )
    player_fainted = jnp.sum(fainted[0].astype(_I32))
    player_out = jnp.all((after.species[0] == 0) | (after.hp[0] == 0))
    opp_out = jnp.all((after.species[1] == 0) | (after.hp[1] == 0))
    ended = (after.over == 0) & (player_out | opp_out)
    winner = jnp.where(player_out, WINNER_OPPONENT, WINNER_PLAYER)
    after = after._replace(
        player_faints=(after.player_faints + player_fainted).astype(after.player_faints.dtype),
        deaths=(after.deaths | fainted[0]).astype(after.deaths.dtype),
        over=jnp.where(ended, 1, after.over).astype(after.over.dtype),
        winner=jnp.where(ended, winner, after.winner).astype(after.winner.dtype),
    )
    for side in (0, 1):
        active_fainted = fainted[side, after.active[side]]
        after = _reset_battler(after, side, active_fainted)
        after = _entered(after, side, entry_turn, active_fainted)
        after = _source_leaves(after, side, active_fainted)
    return after


def _move_end(state: BattleState, acted: BattleState, side, ran) -> BattleState:
    """After ``side``'s action and its faint check. The foe keeps the ``move_hit`` that
    ``_use_move`` wrote: ``UpdateFlagsWhenHit`` runs after a KO'd defender's faint
    (LOOP_FAINTED before LOOP_SPREAD_MOVES, battle_controller_player.c:3820-3837), so a
    fainted defender holds the record until its replacement clears it. Then
    ``BattleControllerPlayer_MoveEnd`` clears the actor's own (``ClearFlags``, :4389, via
    :3986) where ``ran``: the action ran (a move, switch or item; neither a fainted
    battler's action nor a pursuer's, which ``TryPursuit`` ends, battle_script.c:6931,
    reaches it) and the battle goes on (``CheckBattleOver`` returns first, :3969)."""
    foe = 1 - side
    move_hit = state.move_hit.at[foe].set(acted.move_hit[foe])
    clear = ran & (state.over == 0)
    move_hit = move_hit.at[side].set(jnp.where(clear, 0, move_hit[side]))
    state = state._replace(move_hit=move_hit.astype(state.move_hit.dtype))
    # RecoverStatusByAbility for the attacker, then the defender (bcp.c:3965-3967); a
    # battle that CheckBattleOver ended returns first.
    state = recover_status_by_ability(state, side, clear)
    return recover_status_by_ability(state, foe, clear)


def execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 3: both actors in order, each followed by a faint check. A switch facing
    Pursuit is hit first (with its own faint check), and the pursuer does not act again.
    An item action's bag slot is spent first (``_consume_trainer_items``)."""
    return _execute_actions(tables, state, actions, outcome, order)[0]


def _execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> tuple[BattleState, dict]:
    """``execute_actions`` with the turn's events (``_act``; ``pursuit``: the Pursuit hit
    on the switching Pokemon ran, which shows the pursuer's move)."""
    first = order[0].astype(_I32)
    # Each side's move and bag item are fixed when chosen, from the turn's starting state.
    picks = [selected_move(tables, state, side, actions[side]) for side in (0, 1)]
    locked = jnp.stack([locked_slot(state, side) >= 0 for side in (0, 1)])
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]), locked)
    items = jnp.stack(
        [state.trainer_items[side, _bag_slot(actions[side])] for side in (0, 1)]
    ).astype(_I32)
    state = _consume_trainer_items(state, actions)
    pursued = pursuit_fires(tables, state, first, actions, choices)
    state = _where(
        is_switch(actions[first]) & (state.over == 0), _natural_cure(state, first), state
    )
    before = state
    pursuer = 1 - first
    hit = _pursuit_hit(tables, state, pursuer, actions, choices, outcome)
    seen = _hit_events(before, hit, pursuer, True, outcome.crit[pursuer, 0] != 0)
    events = _empty_events()
    pursuer_events = {
        "act_kind": ACT_MOVE,
        "act_id": choices[0][pursuer],
        "hits": 1,
        "crit": jnp.zeros((MAX_HITS,), _I32).at[0].set(seen["crit"]),
    }
    events = _put(events, pursuer, pursuer_events, pursued)
    target_events = {"contact_ability": seen["contact"], "thawed_by_hit": seen["thaw"]}
    events = _put(events, first, target_events, pursued)
    state = _faint_check(before, _where(pursued, hit, state))

    # Both actors through one loop body (a loop, not unrolled: one copy of ``_act``).
    def act(position, carry):
        state, events = carry
        before = state
        side = order[position].astype(_I32)
        second = position == 1  # the first actor's target has not moved yet
        # The foe's Protect this turn (turnFlags.protecting): only a first actor's succeeds.
        foe_protecting = events["protect_ok"][1 - side] != 0
        acted, own, foe_events = _act(
            tables, state, side, actions, choices, items, outcome, second, foe_protecting
        )
        ran = (state.over == 0) & (state.hp[side, state.active[side]] > 0)
        skip = second & pursued  # the pursuer does not act again
        acted = _where(skip, state, acted)
        ran = ran & ~skip
        events = _put(events, side, own, ~skip)
        foe = 1 - side
        foe_events = {name: v | (events[name][foe] != 0) for name, v in foe_events.items()}
        events = _put(events, foe, foe_events, ~skip)
        return _move_end(_faint_check(before, acted), acted, side, ran), events

    state, events = jax.lax.fori_loop(0, 2, act, (state, events))
    return state, {**events, "pursuit": pursued.astype(_I32)}


def _hp_change(state: BattleState, side, mon, delta) -> BattleState:
    hp = jnp.clip(state.hp[side, mon].astype(_I32) + delta, 0, state.max_hp[side, mon])
    return state._replace(hp=state.hp.at[side, mon].set(hp.astype(state.hp.dtype)))


def _leech_seed_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s seeded active Pokemon with HP left loses ``BattleSystem_Divide(maxHP, 8)``
    (capped at its HP) and the other side's active Pokemon, if alive, gains what it lost
    (``hitDamage``), times ``(100 + 30) / 100`` with a Big Root (``effects.big_root``;
    ``subscript_leech_seed_effect.s``)."""
    other = 1 - side
    mon, recipient = state.active[side], state.active[other]
    hp = state.hp[side, mon].astype(_I32)
    on = (state.leech_seed[side] != 0) & (hp > 0) & (state.hp[other, recipient] > 0)
    drained = jnp.minimum(jnp.maximum(state.max_hp[side, mon].astype(_I32) // 8, 1), hp)
    drained = jnp.where(on, drained, 0)
    gained = big_root(tables, state, other, drained)
    return _hp_change(_hp_change(state, side, mon, -drained), other, recipient, gained)


def _aqua_ring_check(tables: Tables, state: BattleState, side) -> BattleState:
    """A ringed ``side`` active Pokemon alive and below full HP heals
    ``BattleSystem_Divide(maxHP, 16)``, times ``(100 + 30) / 100`` with a Big Root
    (``effects.big_root``; ``subscript_aqua_ring_heal.s``; Heal Block is unsupported)."""
    mon = state.active[side]
    hp, full = state.hp[side, mon].astype(_I32), state.max_hp[side, mon].astype(_I32)
    on = (state.aqua_ring[side] != 0) & (hp > 0) & (hp != full)
    heal = jnp.where(on, big_root(tables, state, side, jnp.maximum(full // 16, 1)), 0)
    return _hp_change(state, side, mon, heal)


def _status_damage(state: BattleState, side, status, damage) -> BattleState:
    """``side``'s active Pokemon with ``status`` and HP left loses ``damage`` (capped at its
    HP, ``BATTLE_SUBSCRIPT_UPDATE_HP``)."""
    mon = state.active[side]
    on = (state.status[side, mon] == status) & (state.hp[side, mon] > 0)
    return _hp_change(state, side, mon, -jnp.where(on, damage, 0))


def _eighth(state: BattleState, side) -> jnp.ndarray:
    """``BattleSystem_Divide(maxHP, 8)`` of ``side``'s active Pokemon: at least 1."""
    return jnp.maximum(state.max_hp[side, state.active[side]].astype(_I32) // 8, 1)


def _poison_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``BattleSystem_Divide(maxHP * -1, 8)``, ``subscript_poison_damage.s`` (Magic Guard
    and Poison Heal are not implemented)."""
    return _status_damage(state, side, Status.POISON, _eighth(state, side))


def _toxic_check(tables: Tables, state: BattleState, side) -> BattleState:
    """The counter rises first (up to ``TOXIC_COUNTER_MAX``), then the Pokemon loses
    ``BattleSystem_Divide(maxHP, 16) * counter``."""
    mon = state.active[side]
    on = (state.status[side, mon] == Status.TOXIC) & (state.hp[side, mon] > 0)
    counter = jnp.minimum(state.status_counter[side, mon].astype(_I32) + 1, TOXIC_COUNTER_MAX)
    state = _set_mon(state, "status_counter", side, mon, counter, on)
    sixteenth = jnp.maximum(state.max_hp[side, mon].astype(_I32) // 16, 1)
    return _status_damage(state, side, Status.TOXIC, sixteenth * counter)


def _burn_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``subscript_burn_damage.s``: ``DivideVarByValue`` maxHP by 8, which is
    ``BattleSystem_Divide`` (battle_script.c:4413; Magic Guard and Heatproof are not
    implemented)."""
    return _status_damage(state, side, Status.BURN, _eighth(state, side))


def _bind_check(tables: Tables, state: BattleState, side) -> BattleState:
    """A bound active Pokemon with HP left: the counter drops, and while it stays above
    0 the Pokemon loses ``BattleSystem_Divide(maxHP, 16)``; at 0 Bind ends without damage."""
    mon = state.active[side]
    on = (state.trapped[side] > 0) & (state.hp[side, mon] > 0)
    trapped = state.trapped[side] - on.astype(state.trapped.dtype)
    damage = jnp.maximum(state.max_hp[side, mon].astype(_I32) // 16, 1)
    state = _hp_change(state, side, mon, -jnp.where(on & (trapped > 0), damage, 0))
    return state._replace(
        trapped=state.trapped.at[side].set(trapped),
        trap_move=state.trap_move.at[side].set(jnp.where(trapped > 0, state.trap_move[side], 0)),
    )


def _disable_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_DISABLE`` (battle_controller_player.c:1565): with a move
    disabled, a counter reset to 0 when the Pokemon no longer knows the move, then the
    counter drops while non-zero and, at 0, the move is cleared (``subscript_disable_end.s``).
    No HP condition in the decomp."""
    mon = state.active[side]
    disabled = state.disabled_move[side]
    on = disabled != 0
    known = jnp.any(state.moves[side, mon] == disabled)
    turns = jnp.where(on & ~known, 0, state.disable_turns[side]).astype(_I32)
    expire = on & (turns == 0)
    turns = jnp.where(on & (turns > 0), turns - 1, turns)
    return state._replace(
        disabled_move=state.disabled_move.at[side].set(
            jnp.where(expire, 0, disabled).astype(state.disabled_move.dtype)
        ),
        disable_turns=state.disable_turns.at[side].set(turns.astype(state.disable_turns.dtype)),
    )


def _taunt_check(tables: Tables, state: BattleState, side) -> BattleState:
    """The Taunt counter drops by 1 (no HP condition in the decomp)."""
    taunt = state.taunt[side] - (state.taunt[side] > 0).astype(state.taunt.dtype)
    return state._replace(taunt=state.taunt.at[side].set(taunt))


def _turn_end_ability(tables: Tables, state: BattleState, side, outcome: Outcome) -> BattleState:
    """``MON_COND_CHECK_STATE_ABILITY`` (battle_controller_player.c:1355), the first
    end-of-turn check: Shed Skin (``BattleSystem_TriggerAbilityOnTurnEnd``, battle_lib.c:
    3567-3590) cures any status of a holder with HP left on ``RandNext() % 10 < 3``
    (``outcome.shed_skin[side]``); the sleep counter and the poison counter reset, and
    ``subscript_ability_restore_status.s:8`` tags the ability."""
    mon = state.active[side]
    cured = (
        (state.ability[side, mon] == AbilityId.SHED_SKIN)
        & (state.status[side, mon] != Status.NONE)
        & (state.hp[side, mon] > 0)
        & (outcome.shed_skin[side] != 0)
    )
    state = state._replace(
        status=state.status.at[side, mon].set(
            jnp.where(cured, Status.NONE, state.status[side, mon]).astype(state.status.dtype)
        ),
        status_counter=state.status_counter.at[side, mon].set(
            jnp.where(cured, 0, state.status_counter[side, mon]).astype(state.status_counter.dtype)
        ),
    )
    return reveal_ability(state, side, mon, STATUS_CURE, cured)


# The implemented checks of ``enum MonCondCheckState`` (battle_controller_player.c:1256),
# in its order.
END_OF_TURN_ORDER = (
    # MON_COND_CHECK_STATE_AQUA_RING, battle_controller_player.c:1332
    ("aqua_ring", _aqua_ring_check),
    # MON_COND_CHECK_STATE_ABILITY, battle_controller_player.c:1355 (the only check that
    # reads the turn's ``Outcome``, see ``end_of_turn``)
    ("ability", _turn_end_ability),
    # MON_COND_CHECK_STATE_USE_ITEM, battle_controller_player.c:1362
    ("use_item", after_damage_items),
    # MON_COND_CHECK_STATE_LEECH_SEED, battle_controller_player.c:1378
    ("leech_seed", _leech_seed_check),
    # MON_COND_CHECK_STATE_POISON, battle_controller_player.c:1393
    ("poison", _poison_check),
    # MON_COND_CHECK_STATE_TOXIC, battle_controller_player.c:1405
    ("toxic", _toxic_check),
    # MON_COND_CHECK_STATE_BURN, battle_controller_player.c:1425
    ("burn", _burn_check),
    # MON_COND_CHECK_STATE_BIND, battle_controller_player.c:1462
    ("bind", _bind_check),
    # MON_COND_CHECK_STATE_DISABLE, battle_controller_player.c:1565
    ("disable", _disable_check),
    # MON_COND_CHECK_STATE_TAUNT, battle_controller_player.c:1635
    ("taunt", _taunt_check),
    # MON_COND_CHECK_STATE_ITEM_CONDITION, battle_controller_player.c:1699: the same berry
    # check, after the residual damage above.
    ("item_condition", after_damage_items),
)


def end_of_turn_order(tables: Tables, state: BattleState, outcome: Outcome) -> jnp.ndarray:
    """``int32[2]``: the sides in ``monSpeedOrder``, which
    ``BattleSystem_SortMonSpeedOrder`` (battle_lib.c:3364) sets after the last action
    (battle_controller_player.c:887). It compares with ``ignoreQuickClaw = TRUE``:
    no priority, so ``order.player_second`` with Quick Claw (the flag is still computed,
    battle_lib.c:1278-1286, from this turn's ``speedRand``, drawn at bcp.c:853-857 before
    every sort of the turn: ``outcome.quick_claw``), Lagging Tail, Trick Room and speed; a
    tie swaps on ``RandNext() & 1`` (``outcome.end_speed_tie``, 1 = opponent first)."""
    first = player_second(tables, state, outcome.end_speed_tie, outcome.quick_claw)
    first = first.astype(_I32)
    return jnp.stack([first, 1 - first])


# The side conditions FIELD_COND_CHECK_STATE_REFLECT .. _LUCKY_CHANT count down, in this
# order (battle_controller_player.c:947-1083).
FIELD_SIDE_TIMERS = ("reflect", "light_screen", "mist", "safeguard", "tailwind", "lucky_chant")
# The weathers with a countdown (FIELD_COND_CHECK_STATE_RAINING .. _HAILING, :1113-1211);
# deep fog (:1213-1225) only prints a message.
_TIMED_WEATHERS = (Weather.RAIN, Weather.SANDSTORM, Weather.SUN, Weather.HAIL)


def _count_down(x: jnp.ndarray) -> jnp.ndarray:
    return x - (x > 0).astype(x.dtype)


def _weather_effect(state: BattleState, side, ability) -> BattleState:
    """``BtlCmd_EndOfTurnWeatherEffect`` (battle_script.c:5561-5647) for ``side``'s active
    Pokemon with HP left, then ``subscript_weather_continues.s``'s HP update. Sandstorm:
    ``BattleSystem_Divide(maxHP * -1, 16)`` unless a type is Rock, Steel or Ground or the
    ability is Sand Veil (:5573-5583). Hail: Ice Body heals ``Divide(maxHP, 16)`` below full
    HP, else the same chip unless a type is Ice or the ability is Snow Cloak. The script
    skips a chip (not a heal) for Magic Guard (``CheckAbility ... ABILITY_MAGIC_GUARD``).
    Sand Veil and Snow Cloak are ``AbilityId`` members (``state.ability``); Ice Body and
    Magic Guard, which no table maps, are read from ``ability``
    (``damage.decomp_abilities``). The caller applies ``NO_CLOUD_NINE``.
    Rain and sun act only through abilities no table maps (Dry Skin, Solar Power, Rain
    Dish, Hydration: ``AbilityId.UNSUPPORTED``, refused by coverage), so nothing here.
    ``MOVE_EFFECT_NO_WEATHER_DAMAGE`` (Dig / Dive) cannot be set: no charge move is
    supported."""
    mon = state.active[side]
    hp, max_hp = state.hp[side, mon].astype(_I32), state.max_hp[side, mon].astype(_I32)
    types = state.types[side, mon]
    ab = ability[side, mon]
    own = state.ability[side, mon]

    def has(t):
        return jnp.any(types == t)

    sixteenth = jnp.maximum(max_hp // 16, 1)
    sand = (
        (state.weather == Weather.SANDSTORM)
        & ~has(Type.ROCK)
        & ~has(Type.STEEL)
        & ~has(Type.GROUND)
        & (own != AbilityId.SAND_VEIL)
    )
    hail = state.weather == Weather.HAIL
    ice_body = hail & (ab == WEATHER_ABILITY_IDS["Ice Body"])
    hail_chip = hail & ~ice_body & ~has(Type.ICE) & (own != AbilityId.SNOW_CLOAK)
    chip = (sand | hail_chip) & (ab != WEATHER_ABILITY_IDS["Magic Guard"])
    heal = ice_body & (hp < max_hp)
    delta = jnp.where(chip, -sixteenth, jnp.where(heal, sixteenth, 0))
    return _hp_change(state, side, mon, jnp.where(hp > 0, delta, 0))


def field_conditions(tables: Tables, state: BattleState, sides) -> BattleState:
    """``BattleControllerPlayer_CheckFieldConditions`` (battle_controller_player.c:
    932-1254), between the actions and ``end_of_turn``; nothing runs once the battle is
    over (``CheckBattleOver`` before every step, :939-944). In order:

    1. ``FIELD_SIDE_TIMERS``, side 0 then side 1: a nonzero count drops by 1 and the
       condition ends at 0 (``--reflectTurns == 0``, :947-1083).
    2. Wish (:1085-1111) is not implemented: no supported move sets ``wish_timer``.
    3. The weather: a temporary one (``weather_timer != 0``) drops its count and ends at 0
       with no effect that turn (``subscript_sandstorm_end.s`` and the like); a permanent
       one (``*_PERM``, ``weather_timer == 0``) never ends. A weather that goes on runs
       ``_weather_effect`` for each battler in ``sides`` (``monSpeedOrder``, sorted after
       the last action, :887) unless ``damage.weather_suppressed``. Deep fog never counts
       down (:1213-1225). The faints the chip causes are checked once, after both
       battlers (``BATTLE_SUBSCRIPT_FAINT_MON`` inside the subscript; the battle end only
       when the loop resumes, :939-944): two last Pokemon down at once is a draw, a loss.
    4. Gravity (:1227-1238) drops by 1.

    Trick Room counts down in the turn end's side-condition pass (:1823-1826), not here,
    and is not implemented."""
    live = state.over == 0
    counted = state._replace(
        **{name: _count_down(getattr(state, name)) for name in FIELD_SIDE_TIMERS}
    )
    weather = state.weather
    timed = jnp.isin(weather, jnp.array(_TIMED_WEATHERS, weather.dtype))
    countdown = timed & (state.weather_timer != 0)
    timer = jnp.where(countdown, state.weather_timer - 1, state.weather_timer)
    ended = countdown & (timer == 0)
    counted = counted._replace(
        weather=jnp.where(ended, Weather.NONE, weather).astype(weather.dtype),
        weather_timer=timer.astype(state.weather_timer.dtype),
    )
    state = _where(live, counted, state)

    before = state
    ability = decomp_abilities(tables, state)
    effects = live & ~ended & ~weather_suppressed(tables, state)
    for position in (0, 1):
        side = sides[position]
        state = _where(effects, _weather_effect(state, side, ability), state)
    state = _faint_check(before, state)

    gravity = _count_down(state.gravity)
    return state._replace(gravity=jnp.where(state.over == 0, gravity, state.gravity))


def end_of_turn(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order, sides
) -> BattleState:
    """Phase 4: ``BattleControllerPlayer_CheckMonConditions`` (battle_controller_player.c:
    1289) runs every check of ``END_OF_TURN_ORDER`` for one battler, then for the next,
    in ``sides`` (``end_of_turn_order`` of the state after the actions: ``monSpeedOrder``
    is sorted once, before ``field_conditions``, so a Tailwind that ended there still
    counts); a faint check follows each, and nothing runs once the battle is over.
    Flinch lasts only for the turn it is set in."""
    for position in (0, 1):
        side = sides[position]
        for name, check in END_OF_TURN_ORDER:
            before = state
            args = (outcome,) if name == "ability" else ()
            after = check(tables, state, side, *args)
            state = _faint_check(before, _where(state.over == 0, after, state))
    return state._replace(flinched=jnp.zeros_like(state.flinched))


def _unlock_sleepers(state: BattleState) -> BattleState:
    """``BattleSystem_SetupNextTurn`` (battle_lib.c:2240-2243): a locked battler that is
    asleep when the next turn is set up is unlocked."""
    for side in (0, 1):
        asleep = state.status[side, state.active[side]] == Status.SLEEP
        state = _unlock(state, side, asleep)
    return state


def _quick_claw_shown(tables: Tables, state: BattleState, actions, order) -> jnp.ndarray:
    """``int32[2]``: the side's Quick Claw animation played (``subscript_check_quick_claw.s``,
    the first step of ``BeforeMove``, bcp.c:3140-3143). ``CalcTurnOrder`` sets the battler's
    ``moveEffectsData.quickClaw`` only when two FIGHT commands are compared
    (``ignoreQuickClaw = FALSE``, bcp.c:772-786; battle_lib.c:1281-1284), whatever the
    priorities, from the turn's starting ``speedRand`` (``state.qc_roll``); the script plays
    the held-item animation (no text) only when the battler is not the last to act
    (``WAITING_BATTLERS != 1``): in singles, when it acts first."""
    fight = ~is_switch(actions) & ~is_item(actions)
    claw = quick_claw_active(tables, state, state.qc_roll)
    first = jnp.arange(2) == order[0]
    return (fight[0] & fight[1] & claw & first).astype(_I32)


def _action_phase(
    tables: Tables, state: BattleState, actions, outcome: Outcome
) -> tuple[BattleState, dict, jnp.ndarray]:
    """Phases 2-3 with the turn's events: ``(state, events, order)``."""
    s = _record_last_move(tables, state, actions)
    order = action_order(tables, s, actions, outcome)
    claw = _quick_claw_shown(tables, s, actions, order)
    s, events = _execute_actions(tables, s, actions, outcome, order)
    return s, {**events, "order": order.astype(_I32), "quick_claw": claw}, order


def turn_events(tables: Tables, state: BattleState, actions, outcome: Outcome) -> dict:
    """The events of ``step(tables, state, actions, outcome)``'s action phase, without
    playing the end of the turn (``step_events``)."""
    return _action_phase(tables, state, jnp.asarray(actions, _I32), outcome)[1]


def step_events(
    tables: Tables, state: BattleState, actions, outcome: Outcome
) -> tuple[BattleState, dict]:
    """``(step(...), its events)``: one implementation of the turn serves both ``step``
    (which drops the events) and ``engine.observe`` (which reads them). The events, a
    dict of ``int32`` arrays: per side (``[2]``) ``act_kind`` (``ACT_*``), ``act_id``,
    ``shown`` (the side's turn showed a message), ``crit`` (``[2, MAX_HITS]``) and
    ``EVENT_FLAGS``; ``order`` (``action_order``) and ``pursuit``."""
    actions = jnp.asarray(actions, _I32)
    s, events, order = _action_phase(tables, state, actions, outcome)
    sides = end_of_turn_order(tables, s, outcome)
    s = field_conditions(tables, s, sides)
    s = end_of_turn(tables, s, actions, outcome, order, sides)
    s = _unlock_sleepers(s)
    s = s._replace(turn=(s.turn + 1).astype(s.turn.dtype))
    # The turn's speedRand draw, kept for the next turn's readers (qc_kept).
    s = _keep_quick_claw(tables, s, outcome.quick_claw)
    return _where(state.over != 0, state, s), events


def step(tables: Tables, state: BattleState, actions, outcome: Outcome) -> BattleState:
    """Play one turn: ``actions`` is ``int32[2]`` (player, opponent; see ``order``)."""
    return step_events(tables, state, actions, outcome)[0]


def qc_kept(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[2]``: the side's ``speedRand`` can be read as a Quick Claw before it is redrawn
    (the tight storage rule, docs/kaizo/k3b2-rules.md §5b). Between turns its readers are the
    next command selection's AI ``IfSpeedCompare*`` (trainer_ai.c:1102-1124) and
    ``LoadBattlerSpeedRank`` (trainer_ai.c:2081, a ``SortMonSpeedOrder``-like sort), the next turn's
    ``CalcTurnOrder`` (bcp.c:772-786) and the start-of-turn sorts that order the switch-in
    effects (bcp.c:251, :263; order-free in singles, ``hooks``), all with the battler on the
    field then, and ``CompareBattlerSpeed`` tests the item of that battler
    (battle_lib.c:1278). So the draw is read when a living Quick Claw holder is active, or
    when the side's active fainted and its replacement (``needs_replacement``) can be a
    living Quick Claw holder. Elsewhere nothing reads it before the pre-move redraw
    (bcp.c:853-857): a Pokemon that switches in during a turn is first read by that turn's
    end-of-turn sort, which reads that turn's draw."""
    holds = tables.item_effect[state.item] == ItemId.QUICK_CLAW
    alive = (state.species != 0) & (state.hp > 0)
    on_field = jnp.arange(state.hp.shape[1])[None, :] == state.active[:, None]
    active = jnp.any(holds & alive & on_field, axis=1)
    bench = jnp.any(holds & alive & ~on_field, axis=1)
    return (state.over == 0) & (active | ((needs_replacement(state) != 0) & bench))


def _keep_quick_claw(tables: Tables, state: BattleState, qc) -> BattleState:
    """``state`` with ``qc_roll = qc`` where ``qc_kept``, else 0 (an unread value, so states
    that differ only in it are one state)."""
    kept = qc_kept(tables, state)
    return state._replace(
        qc_roll=jnp.where(kept, jnp.asarray(qc) != 0, False).astype(state.qc_roll.dtype)
    )


def sample_root(key, tables: Tables, state: BattleState) -> BattleState:
    """``state`` with its hidden ``qc_roll`` drawn (Stage A's root chance node): the
    battle-start ``speedRand`` (``BattleContext_InitCounters``, battle_lib.c:2019-2025), or at
    a later root the previous turn's draw, which the player never sees. ``RandNext() % 5 ==
    0`` (``QUICK_CLAW_MODULUS``), treated as uniform (1/5, the project's ``% n`` convention;
    exact 13108/65536), for the sides ``qc_kept`` reads; 0 elsewhere."""
    draw = jax.random.randint(key, (2,), 0, QUICK_CLAW_MODULUS) == 0
    return _keep_quick_claw(tables, state, draw)


def root_belief(tables: Tables, state: BattleState) -> list[tuple[BattleState, float]]:
    """Stage B's root chance node (host): ``(state, probability)`` for every value of the
    hidden ``qc_roll`` under ``sample_root``'s law, one state per combination over the
    ``qc_kept`` sides (1 at 1/5, 0 at 4/5), the other sides 0. They differ only in hidden
    state, so they are one information set. Without a kept side: ``[(state, 1.0)]``."""
    kept = [bool(k) for k in jax.device_get(qc_kept(tables, state))]
    n = QUICK_CLAW_MODULUS
    out = []
    for bits in itertools.product(*((0, 1) if k else (0,) for k in kept)):
        prob = 1.0
        for k, b in zip(kept, bits, strict=True):
            if k:
                prob *= 1 / n if b else (n - 1) / n
        out.append((state._replace(qc_roll=jnp.asarray(bits, state.qc_roll.dtype)), prob))
    return out


def needs_replacement(state: BattleState) -> jnp.ndarray:
    """``int8[2]``: 1 where a side's active Pokemon has fainted and a healthy one is
    left to send in, while the battle goes on. A player faint is not terminal (the battle
    ends at a whiteout, ``_faint_check``), so both sides are served."""
    slots = jnp.arange(state.hp.shape[1])
    active_down = state.hp[jnp.arange(2), state.active] == 0
    bench = (state.species != 0) & (state.hp > 0) & (slots[None, :] != state.active[:, None])
    return (active_down & jnp.any(bench, axis=1) & (state.over == 0)).astype(jnp.int8)


def replace(tables: Tables, state: BattleState, side, slot) -> BattleState:
    """Send in party ``slot`` for ``side``'s fainted active Pokemon, through the same
    switch-in path as a switch action (``_switch_in``, then ``hooks.on_switch_in``) and a
    faint check.

    The decomp: ``BattleControllerPlayer_TurnEnd`` calls ``ReplaceFainted``
    (battle_controller_player.c:1848), which loads ``subscript_show_party_list.s``
    (``subscript_replace_fainted.s`` in shift style): the entry hazards and a faint check
    (``subscript_show_party_list.s:22-24``). ``TurnEnd`` then checks for the end of the
    battle (``CheckBattleOver``, :1847), and the next turn's ``ShowBattleMon`` (:256) runs
    ``BattleSystem_TriggerEffectOnSwitch``: Intimidate of a battler that has not announced
    it yet (battle_lib.c:3795, so an Intimidate already on the field does not fire again)
    and the held-item check of every battler (:4151). Both need HP left, so a replacement
    that Stealth Rock faints triggers neither.

    ``ReplaceFainted`` runs before ``totalTurns++`` (:1852), while ``step`` has already
    added 1 to ``turn``: the decomp's ``totalTurns + 1`` is ``turn`` here, for the
    newcomer's ``fake_out_turn`` and for a faint on entry."""
    entered = _switch_in(state, side, slot, state.turn)
    out = _faint_check(entered, on_switch_in(tables, entered, side), entry_turn=state.turn)
    # The slot's speedRand stays; it is unread unless the newcomer (or, after an entry
    # faint, the next replacement) can be a Quick Claw holder (qc_kept).
    return _keep_quick_claw(tables, out, out.qc_roll)
