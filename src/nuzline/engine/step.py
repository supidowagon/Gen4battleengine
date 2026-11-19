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

Move locks (Bide, Rollout, Ice Ball, Uproar; ``actions.locked_slot``): a locked battler's action
is its locked move and spends no PP (battle_controller_player.c:335); every script that
disrupts a move unlocks it (``_unlock``; an uproar stays, ruling B5-5), and a sleeping battler
is unlocked when the next turn is set up (``_unlock_sleepers``, battle_lib.c:2240). An uproar
(``BattleState.uproar``) ends at the turn end's check (``_uproar_check``).

The Choice lock (``choice_lock``, ``moveEffectsData.choiceLockedMove``): ``_choice_lock``
sets and clears it where ``UpdateMoveBuffers`` and ``UpdateFlagsWhenHit`` do, and
``_pursuit_hit`` where ``BtlCmd_TryPursuit`` does; ``actions.usable_moves`` reads it.
Shell Bell's heal and Life Orb's recoil follow the move's hits (``effects.shell_bell_heal``,
``effects.life_orb_recoil``); after every hit, ``AfterMoveEffects`` checks the held items,
moves a Sticky Barb and thaws the defender, in that order (``_use_move``). A Custap Berry the
turn order activated (``order.custap_flags``) is eaten when its holder's move starts. The
end of the turn runs Leftovers / Black Sludge and the status orbs / Sticky Barb in their
``MonCondCheckState`` slots (``END_OF_TURN_ORDER``).

Implemented so far: the opponent's bag items (``engine.bag``, ``_use_trainer_item``, with the
decomp's multi-slot consumption, ``_consume_trainer_items``), the AI inputs
``fake_out_turn`` (``_entered``) and ``move_hit`` (``_use_move``, ``_move_end``), the
AI's knowledge of the player (``last_move``: ``_use_move``, ``_pursuit_hit``;
``ai_seen_moves``: ``_record_last_move``, ``_forget``; ``ai_ability_known``:
``hooks.reveal_ability``, ``_forget``),
switching (with Natural Cure; Roar's drag takes the same path, ``_use_move``) and the switch-in hooks
(``hooks.on_switch_in``: Stealth Rock, Intimidate, berries), damaging moves, the
accuracy check, PP, Struggle, the move effects of ``effects.py``, the berry check
after every hit (``hooks.after_damage_items``), Pursuit's pre-switch hit, the sleep,
freeze, flinch, Taunt, confusion, paralysis and attract checks before a move, the
held-item, Leftovers, Leech Seed, poison, toxic, burn, Bind, Taunt and detrimental-item end-of-turn checks, the field-condition
pass (``field_conditions``), faints and the end of the battle. A player faint is a death, not the end (spec §4): the battle ends
when a side has no Pokemon with HP left. A state that is already over is returned unchanged.

The mid-turn switch decision (E2 batch 6, U-turn and Baton Pass; rulings B6-1..B6-9): a turn
that reaches a party list (subscript_attack_then_switch_out.s:71, subscript_baton_pass.s:17)
stops there, ``BattleState.pivot_battler`` naming the battler that picks and the ``ts_*``
scratch holding the turn (``_execute_actions``); ``step`` on such a state, with
``actions[pivot]`` the switch to the pick, resumes it: the switch, the hazards and a faint check,
the rest of the paused action on the newcomer (``_use_move``'s ``MODE_RESUME``), the later
actions and the end of the turn. Each half reads its own ``Outcome``.

``step_events`` plays the same turn and also returns what each action showed (the move
used and the result of each status check, hits and crits: ``ACT_*``, ``EVENT_FLAGS``),
recorded where ``_use_move`` and ``execute_actions`` run it; ``engine.observe`` builds
the player's observation from them, ``step`` drops them.

The Trick Room countdown is the turn end's side-condition pass's last state
(battle_controller_player.c:1822-1830, ``_trick_room_check``).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine import targets
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    chosen_targets,
    gravity_banned,
    imprisoning_foe,
    locked_slot,
    move_imprisoned,
)
from nuzline.engine.bag import use_bag_item
from nuzline.engine.damage import (
    PHYSICAL as PHYSICAL_CATEGORY,
)
from nuzline.engine.damage import (
    PRESENT_HEAL,
    battler_decomp_ability,
    calc_move_damage,
    chart_types,
    confusion_damage,
    damage_for_roll,
    divide,
    effectiveness,
    ground_immune,
    gyro_ball_speeds_power,
    hits_semi,
    identified,
    ignorable,
    move_type_of,
    natural_gift_params,
    trump_card_power,
    weather_suppressed,
    wonder_guarded,
)
from nuzline.engine.damage import (
    SPECIAL as SPECIAL_CATEGORY,
)
from nuzline.engine.damage import (
    STATUS as STATUS_CATEGORY,
)
from nuzline.engine.effects import (
    LOCK_ON_TURNS,
    NO_DEFENDER,
    big_root,
    can_have_status,
    captivate_fails,
    change_stage,
    consume_item,
    copy_stat_stages,
    crit_blocked,
    defender_always,
    entry_hazards,
    focus_sash_damage,
    forget_item,
    ghost_user,
    gravity_ground,
    hit_count,
    holds,
    hp_update,
    imprison_fails,
    ingrain_anchors,
    kings_rock_flinch,
    know_ability,
    life_orb_recoil,
    locked_onto,
    magnet_rise_fails,
    metronome_update,
    metronome_verify,
    on_hit_berry,
    pivot_item,
    play_hit,
    pressure_holder,
    pressure_pp_cost,
    primary_effect_fails,
    protect_blocks,
    psycho_shift_fails,
    recovers_at_full_hp,
    resist_berry,
    resist_berry_damage,
    reveal_item,
    self_defender,
    shell_bell_heal,
    sticky_barb_transfer,
    sync_landed,
    synchronize,
    targets_foe_side,
    targets_user,
    thaw_target,
    unlock_move,
    uproar_on,
    weather_fails,
    yawn_blocks,
    yawn_sleep,
)
from nuzline.engine.format import Format, format_of, party, side_of
from nuzline.engine.format import foe as _foe
from nuzline.engine.hooks import (
    ABSORB,
    DAMAGE_AVOIDED,
    OHKO_BLOCKED,
    REDIRECTED,
    STATUS_CURE,
    STATUS_HP,
    TORMENT,
    WEATHER_HP,
    after_damage_items,
    held_item_check,
    move_end_switch_in,
    recover_status_by_ability,
    reveal_ability,
    swap_war,
    switch_in_effects,
)
from nuzline.engine.ids import (
    CHOICE_ITEMS,
    CONFUSION_PENDING,
    COUNTER_PHYSICAL,
    COUNTER_SPECIAL,
    FUTURE_SIGHT_TURNS,
    GRAVITY_TURNS,
    ITEM_ABILITY_IDS,
    MOVE_BATON_PASS,
    MOVE_SLEEP_TALK,
    MOVE_SNORE,
    MOVE_STRUGGLE,
    MOVE_U_TURN,
    PROTECT_CHAIN_MOVES,
    QUICK_CLAW_MODULUS,
    SEMI_SET_FLAGS,
    SNORE_PARAM,
    TRICK_ROOM_TURNS,
    UNROLLED_EFFECTS,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    SemiKind,
    Stat,
    Status,
    Type,
    Weather,
    sleepless,
)
from nuzline.engine.levelup import pay_exp, pay_exp_leveled
from nuzline.engine.order import (
    CALL_ME_FIRST,
    CALL_NONE,
    CALL_SLEEP_TALK,
    MOVE_ACTIONS,
    action_order,
    call_kind,
    called_move,
    claw_flags,
    custap_flags,
    invoked_move,
    is_item,
    is_switch,
    me_first_copy,
    player_second,
    quick_claw_active,
    selected_move,
    truant_loafs,
)
from nuzline.engine.outcome import (
    MAX_HITS,
    Outcome,
    crit_lands,
    future_threshold,
    live_hit_threshold,
    roar_bench,
    roar_slot,
)
from nuzline.engine.state import (
    BAG_SLOTS,
    BATON_PASSED_FIELDS,
    MOVES,
    STAGE_MAX,
    TURN_SCRATCH,
    VOLATILE_FIELDS,
    BattleState,
    battler_ability,
    field_index,
    held_item,
    record_seen_move,
)

_I32 = jnp.int32
_unlock = unlock_move  # Battler_UnlockMoveChoice (battle_lib.c:1834): (state, side, cond)
WINNER_OPPONENT, WINNER_PLAYER = 1, 2
# BattleState.future_damage's int16 cap (every HP is far below it).
FUTURE_DAMAGE_MAX = 32767
PURSUIT_POWER_MUL = 20  # BtlCmd_TryPursuit (battle_script.c:6884): powerMul = 20
NORMAL_POWER_MUL = 10  # BattleContext_Init (battle_lib.c:1980): powerMul = 10
# BattleState.exp_refused: a payout inside the opponent's replacement loop left learn work
# (design §3.3(c)).
EXP_REFUSED_WINDOW_C = 2
# MON_CONDITION_TOXIC_COUNTER is 4 bits (include/constants/battle/condition.h:23).
TOXIC_COUNTER_MAX = 15

# The per-battler fields a switch resets (``_VOLATILE``) and what Baton Pass keeps of them
# (``_BATON_PASSED``) live in ``engine.state`` (the laws' ``outcome.pivot_view`` reads them too).
_VOLATILE, _BATON_PASSED = VOLATILE_FIELDS, BATON_PASSED_FIELDS
# ``_use_move``'s ``mode`` (E2 batch 6): the action from its start; its tail after the U-turn
# user survived the Pursuit of subscript_attack_then_switch_out.s:54 (or died to it); the tail
# after the party list, on the newcomer (``step``'s resume).
MODE_FULL, MODE_PURSUIT, MODE_RESUME = 0, 1, 2

# BtlCmd_TryProtection caps protectSuccessTurns at NELEMS(sProtectSuccessRate) - 1
# (battle_script.c:5182-5184; outcome.PROTECT_RATES).
PROTECT_CHAIN_MAX = 3

# The turn's events (``step_events``, ``turn_events``): what each side's action showed,
# recorded where ``execute_actions`` runs it. ``engine.observe`` reads them; ``step``
# drops them. ``act_kind`` values:
ACT_NONE, ACT_MOVE, ACT_SWITCH, ACT_ITEM, ACT_TAUNTED, ACT_DISABLED = 0, 1, 2, 3, 4, 5
ACT_IMPRISONED = 6  # "{0} can't use the sealed {1}!" (subscript_move_is_imprisoned.s), E2 batch 7
ACT_GRAVITY = 7  # "{0} can't use {1} because of gravity!" (subscript_move_fail_gravity.s), G-67
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


def _source_leaves(state: BattleState, side, cond, baton=False, faint=False) -> BattleState:
    """``side``'s active battler leaves the field where ``cond`` (a switch, a faint, and Roar's
    drag): every effect it is the source of on the other side ends.
    ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2088-2098) and
    ``BattleSystem_CleanupFaintedMon`` (:2168-2186) clear the ATTRACT bit sourced from it
    (``FlagIndex(battler) << VOLATILE_CONDITION_ATTRACT_SHIFT``) and the Bind it put on a foe.
    In singles the only possible source of the foe's ``infatuated`` is this battler. Both also
    clear the Mean Look of every battler whose ``meanLookTarget`` is it (:2056-2061, :2170-2172),
    except a switch by Baton Pass (``baton``: the test sits under ``SYSCTL_BATON_PASS == FALSE``,
    :2055), whose newcomer keeps the foe trapped (ruling B7-n)."""
    state = _release_bind(state, _foe(side), cond)
    keep = state.infatuated[_foe(side)]
    value = jnp.where(cond, 0, keep).astype(keep.dtype)
    state = state._replace(infatuated=state.infatuated.at[_foe(side)].set(value))
    clear = jnp.asarray(cond) & ~jnp.asarray(baton)
    # The Lock-On of every battler it locked onto (lockOnTarget == the leaving one): a switch
    # clears it (battle_lib.c:2063-2067), a Baton Pass restarts its count at
    # MOVE_EFFECT_LOCK_ON_INITIAL_DURATION (:2077-2085, moves.h:55); a faint leaves it
    # (CleanupFaintedMon, :2156-2205, has no such loop) until the slot's replacement comes in,
    # whose UpdateAfterSwitch (battle_script.c:2522) clears it like a switch (B8-7).
    if not faint:
        for battler in range(state.active.shape[0]):
            mine = (state.lock_on[battler] != 0) & (state.lock_on_target[battler] == 1 + side)
            state = _set_side(state, "lock_on", battler, 0, mine & clear)
            state = _set_side(state, "lock_on_target", battler, 0, mine & clear)
            refresh = mine & jnp.asarray(cond) & jnp.asarray(baton)
            state = _set_side(state, "lock_on", battler, LOCK_ON_TURNS, refresh)
    for battler in range(state.active.shape[0]):
        ends = (
            clear & (state.mean_look[battler] != 0) & (state.mean_look_target[battler] == 1 + side)
        )
        state = _set_side(state, "mean_look", battler, 0, ends)
    return state


def _set_mon(state: BattleState, field: str, side, slot, value, cond) -> BattleState:
    """``state.<field>[side, slot] = value`` where ``cond``."""
    arr = getattr(state, field)
    index = field_index(state, field, (side, slot))
    new = jnp.where(cond, value, arr[index]).astype(arr.dtype)
    return state._replace(**{field: arr.at[index].set(new)})


def _set_side(state: BattleState, field: str, side, value, cond) -> BattleState:
    """``state.<field>[side] = value`` where ``cond`` (a per-side or per-battler field)."""
    arr = getattr(state, field)
    index = field_index(state, field, side)
    new = jnp.where(cond, value, arr[index]).astype(arr.dtype)
    return state._replace(**{field: arr.at[index].set(new)})


def _natural_cure(state: BattleState, side, cond=True) -> BattleState:
    """``BtlCmd_TryRestoreStatusOnSwitch`` (battle_script.c:9139) clears the status of a
    switching battler with HP left and Natural Cure. ``subscript_switch_pokemon.s:5-6``
    runs it before Pursuit hits (:11), so ``execute_actions`` applies it before the
    Pursuit hit; repeating it in ``_send_out`` changes nothing."""
    fmt = format_of(state)
    mon = state.active[side]
    # Raw BattleMon.ability (battle_script.c:9151), not Battler_Ability: Gastro Acid leaves it.
    cure = (state.ability[party(fmt, side), mon] == AbilityId.NATURAL_CURE) & (
        state.hp[party(fmt, side), mon] > 0
    )
    cure = cure & jnp.asarray(cond)
    state = _set_mon(state, "status", side, mon, Status.NONE, cure)
    return _set_mon(state, "status_counter", side, mon, 0, cure)


def _switch_out(state: BattleState, side) -> BattleState:
    """``side``'s active Pokemon leaves by a switch action (``subscript_switch_pokemon.s``):
    Natural Cure (``_natural_cure``), and the toxic counter restarts: battle_controller.c:
    1482 writes the status back to the party without ``MON_CONDITION_TOXIC_COUNTER``, and
    a returning battler reads it from there. A sleep counter is part of the status and
    stays."""
    fmt = format_of(state)
    state = _natural_cure(state, side)
    mon = state.active[side]
    toxic = state.status[party(fmt, side), mon] == Status.TOXIC
    return _set_mon(state, "status_counter", side, mon, 0, toxic)


def _reset_battler(state: BattleState, side, cond, baton=False) -> BattleState:
    """Stages and ``_VOLATILE`` fields of ``side``'s active battler back to neutral
    where ``cond``; ``baton`` (a Baton Pass switch) keeps the stages and ``_BATON_PASSED``."""
    baton = jnp.asarray(baton)

    def reset(name):
        clear = cond & ~baton if name in (*_BATON_PASSED, "stages") else cond
        return getattr(state, name).at[side].set(jnp.where(clear, 0, getattr(state, name)[side]))

    return state._replace(**{name: reset(name) for name in (*_VOLATILE, "stages")})


def _entered(state: BattleState, side, entry_turn, cond) -> BattleState:
    """``fakeOutTurnNumber = totalTurns + 1`` for ``side``'s battler where ``cond``:
    ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2117) and
    ``BattleSystem_CleanupFaintedMon`` (:2195). ``entry_turn`` is that
    ``totalTurns + 1``: ``turn + 1`` inside ``step``, ``turn`` in ``replace`` (which runs
    after ``step`` has counted the turn; ``state`` module docstring)."""
    value = jnp.where(cond, entry_turn, state.fake_out_turn[side])
    fake_out_turn = state.fake_out_turn.at[side].set(value.astype(state.fake_out_turn.dtype))
    # ``moveEffectsData.truant = (totalTurns + 1) & 1`` at the same two sites (:2119, :2197).
    parity = jnp.where(cond, jnp.asarray(entry_turn, _I32) & 1, state.truant[side])
    truant = state.truant.at[side].set(parity.astype(state.truant.dtype))
    return state._replace(fake_out_turn=fake_out_turn, truant=truant)


def _holds_choice(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``side``'s active Pokemon holds a Choice item (``HOLD_EFFECT_CHOICE_ATK``, ``_SPEED``,
    ``_SPATK``: ``Battler_HeldItemEffect``, ``state.held_item``, B13-29)."""
    item = tables.item_effect[held_item(state, side)]
    return jnp.isin(item, jnp.asarray(CHOICE_ITEMS, item.dtype))


def _set_choice_lock(state: BattleState, side, move, cond) -> BattleState:
    lock = jnp.where(cond, move, state.choice_lock[side]).astype(state.choice_lock.dtype)
    return state._replace(choice_lock=state.choice_lock.at[side].set(lock))


def _choice_lock(
    tables: Tables, state: BattleState, side, defender, move, shown, record, exempt=False
) -> BattleState:
    """The Choice lock after ``side``'s move, where its attack message was ``shown``
    (``SYSCTL_ATTACK_MESSAGE_SHOWN``; ``SYSCTL_FIRST_OF_MULTI_TURN`` comes only from the
    charge-turn scripts, unsupported moves). ``UpdateFlagsWhenHit``
    (battle_controller_player.c:4489-4496) first clears the defender's lock unless it holds
    a Choice item, where ``record`` (the foe is the defender, the same test as its
    ``moveHit``); then ``UpdateMoveBuffers`` (:3921-3934) locks a Choice holder into
    ``moveTemp`` (``move``: the chosen move, Nature Power itself), except Struggle and ``exempt``
    (U-turn under ``SYSCTL_UTURN_ACTIVE``, Baton Pass under ``SYSCTL_MOVE_SUCCEEDED``, bcp.c:
    3927-3928, by ``moveTemp``), and clears a non-holder's.
    ``defender``: the defender's side (``DEFENDING_MON``, bcp.c:4488-4496)."""
    state = _set_choice_lock(
        state, defender, 0, record & shown & ~_holds_choice(tables, state, defender)
    )
    holder = _holds_choice(tables, state, side)
    move = jnp.asarray(move, _I32)
    lock = shown & holder & (move != MOVE_STRUGGLE) & ~jnp.asarray(exempt)
    state = _set_choice_lock(state, side, move, lock)
    return _set_choice_lock(state, side, 0, shown & ~holder)


def _set_last_move(state: BattleState, side, move, cond) -> BattleState:
    """``movePrevByBattler[side] = move`` where ``cond`` (``BattleState.last_move``)."""
    value = jnp.where(cond, move, state.last_move[side]).astype(state.last_move.dtype)
    return state._replace(last_move=state.last_move.at[side].set(value))


def _record_last_move(tables: Tables, state: BattleState, actions) -> BattleState:
    """``TrainerAI_RecordLastMove`` (trainer_ai.c:2705-2717), at the start of the turn:
    the AI adds the player's ``movePrevByBattler`` to its list of the player's moves
    (``state.record_seen_move``). It runs only on the AI's FIGHT path (``TrainerAI_MainSingles``,
    :295, after ``TrainerAI_PickCommand`` chose neither a switch nor an item, :3989), and not
    when the opponent must Struggle: the move menu, and with it ``TrainerAI_Main``, is
    skipped for ``STRUGGLING_ALL`` (battle_controller_player.c:433) and for a locked
    opponent (:335). The list holds the moves by id (``ai_seen_moves``, G-68), so a Mimic copy, a
    level-up's replacement and the player's own Struggle (``MOVE_STRUGGLE``) are recorded as the
    decomp does."""
    mon = state.active[0]
    pending = state.last_move[0]
    _, struggling = selected_move(tables, state, 1, actions[1])
    locked = locked_slot(state, 1) >= 0  # a locked battler picks no command (:335)
    fight = (actions[1] < MOVE_ACTIONS) & ~struggling & ~locked
    seen = state.ai_seen_moves
    row = jnp.where(fight, record_seen_move(seen[mon], pending), seen[mon]).astype(seen.dtype)
    return state._replace(ai_seen_moves=seen.at[mon].set(row))


def _forget(state: BattleState, side, slot) -> BattleState:
    """The AI forgets what it knew about the player's party ``slot`` when it enters
    (``BattleSystem_UpdateAfterSwitch`` clears the battler's known moves, ability and held
    item, battle_lib.c:2151-2153); nothing for the opponent's side."""
    on = jnp.asarray(side, _I32) == 0
    return state._replace(
        ai_seen_moves=state.ai_seen_moves.at[slot].set(
            jnp.where(on, 0, state.ai_seen_moves[slot]).astype(state.ai_seen_moves.dtype)
        ),
        ai_ability_known=state.ai_ability_known.at[slot].set(
            jnp.where(on, 0, state.ai_ability_known[slot]).astype(state.ai_ability_known.dtype)
        ),
        ai_item_known=state.ai_item_known.at[slot].set(
            jnp.where(on, 0, state.ai_item_known[slot]).astype(state.ai_item_known.dtype)
        ),
    )


def _restore_transform(state: BattleState, side, cond) -> BattleState:
    """``side``'s active battler leaves the field where ``cond``: a Transformed one gets its own
    species, form, stats, IVs and ability slot / form back from the ``shadow_*`` and the shadow
    clears (``BattleSystem_InitBattleMon`` loads every field from the party Pokemon, battle_lib.c:
    60-125; the volatile bit goes with ``statusVolatile``, :2100-2104). Run before
    ``_restore_types`` / ``_restore_ability`` / ``_restore_moves``, which read the restored
    species, form, ability slot and mask; a fainted battler keeps the copy until its replacement
    enters (the exp payout reads its copied species, battle_script.c:2216), G-69."""
    fmt = format_of(state)
    pm = party(fmt, side)
    mon = state.active[side]
    on = jnp.asarray(cond) & (state.transformed[side] != 0)

    def back(arr, shadow):
        return arr.at[pm, mon].set(jnp.where(on, shadow[side], arr[pm, mon]).astype(arr.dtype))

    def clear(arr):
        return arr.at[side].set(jnp.where(on, 0, arr[side]).astype(arr.dtype))

    return state._replace(
        species=back(state.species, state.shadow_species),
        form=back(state.form, state.shadow_form),
        stats=back(state.stats, state.shadow_stats),
        ivs=back(state.ivs, state.shadow_ivs),
        ability_slot=back(state.ability_slot, state.shadow_ability_slot),
        ability_form=back(state.ability_form, state.shadow_ability_form),
        transformed=clear(state.transformed),
        shadow_species=clear(state.shadow_species),
        shadow_form=clear(state.shadow_form),
        shadow_stats=clear(state.shadow_stats),
        shadow_ivs=clear(state.shadow_ivs),
        shadow_ability_slot=clear(state.shadow_ability_slot),
        shadow_ability_form=clear(state.shadow_ability_form),
    )


def _restore_moves(state: BattleState, side, cond) -> BattleState:
    """``side``'s active battler leaves the field where ``cond``: the move slots Mimic changed
    (``mimic_slots``) get the party's move and PP back from ``shadow_moves`` / ``shadow_pp``, and
    the shadow and the mask clear. The decomp's party Pokemon never saw the copy
    (``BattleMon_CopyToParty``, battle_lib.c:1819-1825, sends ``BtlIOCmd_UpdatePartyMon``,
    which skips the flagged slots, battle_io_command.c:581-587) and
    ``BattleSystem_InitBattleMon`` loads moves and PP from it on the next switch-in
    (:80-85); ``moveEffectsData`` (the mask) is cleared when the battler leaves (:2100-2104) and
    Baton Pass keeps none of it (:2107-2114). Run when the battler is switched or dragged out, and
    at the battle's end (``_faint_check``); a fainted battler keeps its copy until its
    replacement enters, as ``_restore_ability``'s."""
    fmt = format_of(state)
    pm = party(fmt, side)
    mon = state.active[side]
    masked = ((state.mimic_slots[side].astype(_I32) >> jnp.arange(MOVES)) & 1) == 1
    use = jnp.asarray(cond) & masked
    moves = jnp.where(use, state.shadow_moves[side], state.moves[pm, mon])
    pp = jnp.where(use, state.shadow_pp[side], state.pp[pm, mon])
    zero = jnp.asarray(cond)
    return state._replace(
        moves=state.moves.at[pm, mon].set(moves.astype(state.moves.dtype)),
        pp=state.pp.at[pm, mon].set(pp.astype(state.pp.dtype)),
        mimic_slots=state.mimic_slots.at[side].set(
            jnp.where(zero, 0, state.mimic_slots[side]).astype(state.mimic_slots.dtype)
        ),
        shadow_moves=state.shadow_moves.at[side].set(
            jnp.where(zero, 0, state.shadow_moves[side]).astype(state.shadow_moves.dtype)
        ),
        shadow_pp=state.shadow_pp.at[side].set(
            jnp.where(zero, 0, state.shadow_pp[side]).astype(state.shadow_pp.dtype)
        ),
    )


def _restore_types(tables: Tables, state: BattleState, side, cond) -> BattleState:
    """``side``'s active battler leaves the field where ``cond``: its ``types`` go back to its
    personal row's (``tables.species_types`` at ``species_form_row``, as ``make_state`` builds
    them). Only Camouflage (``effects._camouflage``) writes ``types``; the decomp writes the
    battler's ``BattleMon.type1/type2`` and ``BattleSystem_InitBattleMon`` re-reads the party
    Pokemon's on its next switch-in (battle_lib.c:114-115), so the party never sees it."""
    fmt = format_of(state)
    mon = state.active[side]
    row = tables.species_form_row[
        state.species[party(fmt, side), mon], state.form[party(fmt, side), mon]
    ]
    own_types = tables.species_types[row].astype(state.types.dtype)
    value = jnp.where(cond, own_types, state.types[party(fmt, side), mon])
    return state._replace(types=state.types.at[party(fmt, side), mon].set(value))


def _restore_ability(tables: Tables, state: BattleState, side, cond) -> BattleState:
    """``side``'s active battler leaves the field where ``cond``: its ``ability`` goes back to
    its party Pokemon's (``tables.species_abilities`` at its ``ability_form``'s row and
    ability slot, as ``make_state`` builds it). Only Trace
    (``hooks.trace``) writes it, to the battler's ``BattleMon.ability``;
    ``BattleSystem_InitBattleMon`` re-reads the party Pokemon's on the next switch-in
    (battle_lib.c:124). Run when the battler is switched or dragged out, and when a
    replacement takes a fainted one's place (``_switch_in``): a fainted battler keeps the
    copied ability until then, as its ``BattleMon`` does, and no Pokemon comes back from 0 HP."""
    fmt = format_of(state)
    mon = state.active[side]
    row = tables.species_form_row[
        state.species[party(fmt, side), mon], state.ability_form[party(fmt, side), mon]
    ]
    own_ability = tables.species_abilities[row, state.ability_slot[party(fmt, side), mon]]
    # The raw BattleMon.ability write-back (battle_lib.c:124).
    value = jnp.where(cond, own_ability, state.ability[party(fmt, side), mon]).astype(
        state.ability.dtype
    )
    return state._replace(ability=state.ability.at[party(fmt, side), mon].set(value))


def _switch_in(
    tables: Tables, state: BattleState, side, slot, entry_turn, baton=False
) -> BattleState:
    """``side`` sends out party ``slot``: the outgoing battler's types are restored
    (``_restore_types``), stages and volatiles reset, the effects the
    outgoing Pokemon is the source of end (``_source_leaves``: the Bind it put on the other
    side, the other side's infatuation), ``fake_out_turn`` becomes ``entry_turn`` and the AI
    forgets the newcomer (``_forget``; ``BattleSystem_UpdateAfterSwitch``, battle_lib.c:2094).
    Then the send-out updates the participation mask (``_flag_exp``). ``baton``: a Baton Pass
    switch (``_reset_battler``)."""
    state = _restore_transform(state, side, True)
    state = _restore_types(tables, state, side, True)
    state = _restore_ability(tables, state, side, True)
    state = _restore_moves(state, side, True)
    state = _reset_battler(state, side, True, baton)
    state = _forget(state, side, slot)
    state = _entered(state, side, entry_turn, True)
    state = state._replace(active=state.active.at[side].set(jnp.asarray(slot).astype(jnp.int8)))
    return _flag_exp(_source_leaves(state, side, True, baton), side)


def _flag_exp(state: BattleState, side) -> BattleState:
    """``exp_mask`` (``sideGetExpMask[0]``) after ``side`` sent its newcomer out
    (``BtlCmd_PokemonSendOut``, battle_script.c:543-635, which every switch, drag and
    replacement subscript runs before its hazards check, e.g. subscript_show_party_list.s:18,
    :22). A player send-out flags the enemy (:571-572, :592-593, :607-608, :622-623); an
    enemy send-out clears the mask first (:580-581, :595-596, :610-611, :625-626).
    ``BattleSystem_FlagBattlerExpGain`` (battle_lib.c:1461-1479) ORs in the player's active
    slot (``selectedPartySlot``) when the enemy's ``curHP`` is nonzero; in singles only
    ``PLAYER_1`` against ``ENEMY_1`` counts (``PLAYER_2`` / ``ENEMY_2`` are in
    ``battlersSwitchingMask`` from the start, battle_lib.c:2031-2033). The player's own HP is
    not read."""
    bit = jnp.left_shift(jnp.int32(1), state.active[0].astype(jnp.int32))
    foe_has_hp = state.hp[1, state.active[1]] > 0
    kept = jnp.where(side == 0, state.exp_mask.astype(jnp.int32), 0)  # ClearSideExpGain
    mask = kept | jnp.where(foe_has_hp, bit, 0)
    return state._replace(exp_mask=mask.astype(state.exp_mask.dtype))


def _send_out(
    tables: Tables, state: BattleState, side, slot, effects=True, baton=False, trace_pick=None
) -> BattleState:
    """A switch action, or Roar's drag (``_use_move``): ``_switch_out``, ``_switch_in``, the
    entry hazards, then (``effects``, static ``False`` or a traced flag) the rest of
    ``hooks.on_switch_in`` (Trace, weather, Intimidate, berries). The party list of a U-turn or
    Baton Pass runs only the hazards (subscript_show_party_list.s:22; the rest comes at
    ``_use_move``'s ``TRIGGER_SWITCH_IN_EFFECTS``, as Roar's drag does)."""
    entered = _switch_in(tables, _switch_out(state, side), side, slot, state.turn + 1, baton)
    hazards = entry_hazards(tables, entered, side)
    if effects is False:
        return hazards
    return _where(effects, switch_in_effects(tables, hazards, side, trace_pick), hazards)


def _alive_mask(state: BattleState) -> jnp.ndarray:
    """``int8[P]``: per party the bitmask of HP > 0 (``BattleState.ts_alive``)."""
    bits = jnp.left_shift(jnp.int32(1), jnp.arange(state.hp.shape[1], dtype=_I32))
    return jnp.sum(jnp.where(state.hp > 0, bits[None, :], 0), axis=1).astype(jnp.int8)


def _with_alive(state: BattleState, mask) -> BattleState:
    """``state`` with ``hp`` 1 where ``mask`` (``int8[P]``, ``_alive_mask``) has the slot, else 0:
    the ``before`` of a faint check or a payout, which read only ``before.hp > 0``
    (``_faint_check``; ``levelup.pay_exp_leveled``)."""
    slots = jnp.arange(state.hp.shape[1], dtype=_I32)
    on = (jnp.right_shift(jnp.asarray(mask, _I32)[:, None], slots[None, :]) & 1) == 1
    return state._replace(hp=on.astype(state.hp.dtype))


def _has_bench(state: BattleState, side) -> jnp.ndarray:
    """``BattleSystem_AnyReplacementMons`` (battle_lib.c:3125-3180) in singles: a living,
    non-empty party slot of ``side`` other than the active one (TryReplaceFaintedMon,
    battle_script.c:5867-5877)."""
    fmt = format_of(state)
    slots = jnp.arange(state.hp.shape[1])
    bench = (
        (state.species[party(fmt, side)] != 0)
        & (state.hp[party(fmt, side)] > 0)
        & (slots != state.active[side])
    )
    return jnp.any(bench)


def _blacked_out(state: BattleState, side) -> jnp.ndarray:
    """``BtlCmd_CheckBlackOut`` (battle_script.c:8846-8920) in singles: no party Pokemon of
    ``side`` has HP."""
    fmt = format_of(state)
    return ~jnp.any((state.species[party(fmt, side)] != 0) & (state.hp[party(fmt, side)] > 0))


def _empty_events(fmt: Format) -> dict:
    """The turn's events before any action ran: every per-battler field 0 (``[B]``, ``crit``
    ``[B, MAX_HITS]``)."""
    side = jnp.zeros((fmt.B,), _I32)
    events = {
        name: side
        for name in (
            "act_kind",
            "act_id",
            "shown",
            "magnitude",
            "present_heal",
            "called",
            "move_cur",
            "dragged",
            "roosted",
            "endure_ok",
            "phys_taken",
            "spec_taken",
            "last_taken",
            "phys_from",
            "spec_from",
            "last_from",
            "move_failed",
            "disabled_at_move",
            "sealed_at_move",
            "encored_at_move",
            "gravity_at_move",
            *EVENT_FLAGS,
        )
    }
    events["crit"] = jnp.zeros((fmt.B, MAX_HITS), _I32)
    return events


def _put(events: dict, side, values: dict, cond) -> dict:
    """``events[name][side] = values[name]`` where ``cond``."""
    out = dict(events)
    for name, value in values.items():
        arr = events[name]
        new = jnp.where(cond, jnp.asarray(value).astype(arr.dtype), arr[side])
        out[name] = arr.at[side].set(new)
    return out


def _hit_events(
    before: BattleState, after: BattleState, side, landed, crit, target_side=None
) -> dict:
    """What one hit of ``side``'s move showed, from the states around it: a crit on a
    landed hit that dealt damage (the target's HP drop: only the hit itself lowers it,
    effects.py:441-444; a crit into a substitute is not seen, a documented coarsening in
    ``observe``'s docstring), the defender's contact ability (the only change of the
    attacker's own status, and Cute Charm the only change of its ``infatuated``, inside a
    hit: effects._contact_ability). A Fire hit's thaw comes later, with the held items
    (``effects.thaw_target`` in ``_use_move``; Pursuit's hit has none)."""
    fmt = format_of(before)
    # The defender's battler (``targets.resolve``'s ``target_side``; default the foe): at B = 2
    # the user itself when a lock reads the stored target (singles follow-up (ii)).
    foe = _foe(side) if target_side is None else jnp.asarray(target_side, _I32)
    user, target = before.active[side], before.active[foe]
    dealt = before.hp[party(fmt, foe), target] > after.hp[party(fmt, foe), target]
    contact = (before.status[party(fmt, side), user] == Status.NONE) & (
        after.status[party(fmt, side), user] != Status.NONE
    )
    contact = contact | ((before.infatuated[side] == 0) & (after.infatuated[side] != 0))
    return {
        "crit": landed & dealt & crit,
        "contact": contact,
        "dealt": landed & dealt,
    }


def _redefend(resolved, side, defender, cond):
    """``resolved`` with ``defender`` (a battler or ``NO_DEFENDER``) as the action's defender where
    ``cond``: ``target_side`` is the defender when it is another battler, else the foe (as
    ``targets.resolve``). The commands that pick a new defender mid-action (GoToMoveScript, Counter,
    the Bide release) rebind it this way (D2-13, D2-14)."""
    defender = jnp.asarray(defender, _I32)
    other = (defender != NO_DEFENDER) & (defender != side)
    target = jnp.where(other, defender, jnp.asarray(side, _I32) ^ 1)
    return resolved._replace(
        defender=jnp.where(cond, defender, resolved.defender).astype(_I32),
        target_side=jnp.where(cond, target, resolved.target_side).astype(_I32),
    )


def _use_move(
    tables: Tables,
    state: BattleState,
    side,
    action,
    choice,
    outcome: Outcome,
    target_moved,
    foe_protecting=False,
    custap=(False, False),
    foe_roosting=False,
    foe_taken=(0, 0, 0, 0, 0, 0),
    mode=MODE_FULL,
    pursuit=None,
    claw=(False, False),
    entry_poisoned=False,
    entry_status=None,
    pursuit_sync=None,
    foe_enduring=False,
    speed_order=None,
) -> tuple[BattleState, dict, dict, dict]:
    """``side``'s active Pokemon uses the move in slot ``action`` (or Struggle).
    ``choice``: its ``(move, struggling, locked)`` from ``selected_move`` and
    ``locked_slot`` at the start of the turn, when the decomp fixes Struggle
    (battle_controller_player.c:433), so a Taunt landing earlier this turn makes a chosen
    power-0 move fail instead of Struggling, and a battler locked then spends no PP
    (``ppDecremented`` is set at command selection, :335).
    ``target_moved``: the other side has already acted this turn. ``foe_protecting``: its
    Protect succeeded earlier this turn (the decomp's ``turnFlags.protecting``, a per-turn
    flag cleared at the turn end, battle_lib.c:2231, and on a faint, :2190-2193). ``custap``
    (``bool[B]``): each side's ``moveEffectsData.custapBerry`` from the turn order
    (``order.custap_flags``). ``foe_enduring``: the foe's Endure succeeded earlier this turn
    (``turnFlags.enduring``, cleared with ``protecting``): ``effects.play_hit``'s UpdateHP read
    (G-65). ``foe_roosting``: the foe's Roost succeeded earlier this turn
    (``turnFlags.roosting``, cleared at the turn end, battle_lib.c:2231): the type chart of the
    move skips its Flying rows (``damage.chart_types``). ``foe_taken``: the
    physical, special and last damage the user took this turn and each one's attacker as
    ``1 + battler`` (its ``turnFlags``, bcp.c:3413-3434: the ``phys_taken`` / ``spec_taken`` /
    ``last_taken`` and ``phys_from`` / ``spec_from`` / ``last_from`` events), which Counter,
    Mirror Coat and Metal Burst read.

    E2 batch 6, the pivots (rulings B6-n). U-turn: its ON_HIT secondary effect
    (subscript_attack_then_switch_out.s) runs after the hit's contact ability; past the defender
    side's black-out check and the user's replacement check (:5-6) it runs the pivot item (:11,
    ``effects.pivot_item``) and Destiny Bond (:15-34); a user still alive (:48) then meets the foe's
    Pursuit (:54: ``pursuit``, ``(action, move, struggling)`` of the foe at selection), which
    ``_execute_actions`` plays (``ctl["pursue"]``), else its Natural Cure (:56-57) and the party
    list (:71): the action stops there (``ctl["pause"]``). Baton Pass (subscript_baton_pass.s): a
    user with a replacement (:5; else MOVE_STATUS_FAILED, :19-20) gets its Natural Cure (:7-8) and
    the list (:17). ``mode`` ``MODE_PURSUIT`` (after the Pursuit) and ``MODE_RESUME`` (on the
    newcomer, after the list's switch-in, hazards and faint check) run only the action's tail:
    the head's state is dropped and its values come from ``BattleState.ts_used`` /
    ``ts_damaged`` / ``ts_dealt`` (written where the action stops). ``entry_poisoned``
    (``MODE_RESUME``): the list's Toxic Spikes poisoned the newcomer (``_act``), which set
    ``SYSCTL_TRY_SYNCHRONIZE_STATUS`` after U-turn cleared it (rulings G-12, G-13), and
    ``entry_status`` the newcomer's status before the list's faint check (its
    ``BattleMon.status``, which the Synchronize step reads; ruling G-24). ``pursuit_sync``
    (``MODE_PURSUIT``): ``(flag, sideEffectMon, held)`` after the Pursuit, the head's
    Synchronize flag and side-effect battler as the Pursuit's contact ability left them and each
    battler's ``BattleMon.status`` before the Pursuit's faint check (``_execute_actions``).

    Returns ``(state, own, foe, ctl)``: the state after the move, the user's events (each
    status check's result, effect-gated: set only when the check was reached and had an
    effect), the foe's (``contact_ability``, ``thawed_by_hit``, ``dragged``: the party
    slot + 1 Roar dragged in, else 0) and ``ctl`` (``pause``, ``pursue``: where the action
    stopped)."""
    fmt = format_of(state)
    state_in = state
    mode = jnp.asarray(mode, _I32)
    head_on = mode == MODE_FULL
    resumed = mode == MODE_RESUME
    after_pursuit = mode == MODE_PURSUIT
    move, struggling, locked = choice
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    # BattleControllerPlayer_FightCommand (bcp.c:1866-1883), B10-9: after Struggle, an encored
    # battler whose encored slot holds the encored move uses it (and that slot's PP: the no-PP
    # failure below), whatever it chose (a foe's faster Encore this turn); a slot that no longer
    # holds it ends the encore (below).
    encore_slot = state.encore_slot[side].astype(_I32)
    encore_holds = (state.encore_move[side] != 0) & (
        state.moves[party(fmt, side), user, encore_slot] == state.encore_move[side]
    )
    encored_now = encore_holds & ~jnp.asarray(struggling, bool)
    move = jnp.where(encored_now, state.encore_move[side].astype(_I32), move)
    slot = jnp.where(encored_now, encore_slot, slot)
    # BattleControllerPlayer_FightCommand (bcp.c:1900) fixes the defender with the move after the
    # Encore takeover (``moveTemp``) and the stored target (D2-1) at the action's start: the
    # resolver (``targets.resolve``, D2-4). ``speed_order``: monSpeedOrder, D3's input (B = 2: the
    # identity, order-free).
    if speed_order is None:
        speed_order = jnp.arange(fmt.B)
    resolved = targets.resolve(
        tables,
        state_in,
        side,
        move,
        targets.fight_randomizes(state_in, side, choice[1]),
        state_in.chosen_target[side],
        speed_order,
        outcome.target_draw[side],
    )
    target_side = resolved.target_side
    target = state.active[target_side]
    ready = (state.over == 0) & (state.hp[party(fmt, side), user] > 0)
    # A move of the battler is disabled when its action starts (a faster foe's Disable this turn
    # included): the event the Sleep Talk call variants split on (enumerate.call_consistent).
    disabled_at_move = ready & (state.disabled_move[side] != 0)
    # Likewise a battler of the other side imprisoning (the Sleep Talk set's other change within
    # the turn, ruling B7-13).
    sealed_at_move = ready & imprisoning_foe(state, side)
    # Gravity is up when the action starts: a faster foe's Gravity this turn changes the
    # invokers' sets (G-67), the event their Gravity call variants split on.
    gravity_at_move = ready & (state.gravity != 0)
    # The encored move replaced the chosen one when the action started (FightCommand above): the
    # event the takeover variants split on (enumerate.call_consistent, ruling B11-1).
    encored_at_move = ready & encored_now & (state.encore_move[side] != jnp.asarray(choice[0]))
    # BattleControllerPlayer_FightCommand (battle_controller_player.c:1900) fixes the
    # defender before the status checks: the foe when it is alive (any HP for
    # RANGE_OPPONENT_SIDE: BattleSystem_RandomOpponent), the user for a self-defender range
    # and a non-Ghost Curse (effects.self_defender).
    foe_defender = (resolved.defender != NO_DEFENDER) & (resolved.defender != side)
    custap = jnp.asarray(custap, bool)
    # BEFORE_MOVE_STATE_QUICK_CLAW (bcp.c:3140-3143), before the status checks:
    # subscript_check_quick_claw.s:26-28 clears an activated Custap flag and removes the item
    # (RemoveItem BTLSCR_ATTACKER, kept in item_consumed), whether or not the battler acts last.
    state = consume_item(state, side, ready & custap[side])
    # BattleControllerPlayer_FightCommand (bcp.c:1874-1883), B10-9: an encored battler whose
    # encored slot no longer holds the encored move uses the slot's move and its encore ends
    # (move, slot and count to 0); Struggle comes first (:1866-1868).
    slot_move = state.moves[party(fmt, side), user, state.encore_slot[side].astype(_I32)]
    unheld = ready & ~jnp.asarray(choice[1], bool) & (state.encore_move[side] != 0)
    unheld = unheld & (slot_move != state.encore_move[side])
    for name in ("encore_move", "encore_slot", "encore_turns"):
        state = _set_side(state, name, side, 0, unheld)
    # CHECK_STATUS_START (bcp.c:2425): the move attempt clears the user's Destiny Bond, whatever
    # the checks below find.
    bond = jnp.where(ready, 0, state.destiny_bond[side]).astype(state.destiny_bond.dtype)
    state = state._replace(destiny_bond=state.destiny_bond.at[side].set(bond))

    # BattleControllerPlayer_CheckStatusDisruption (battle_controller_player.c:2417) runs
    # before the PP decrement; the first check that disrupts the move stops the rest.
    # CHECK_STATUS_STATE_SLEEP (:2430): the counter drops by 1 (Early Bird is not
    # implemented); still asleep disrupts the move, while waking up (subscript_wake_up)
    # clears the status and the checks go on. While FIELD_CONDITION_UPROAR is set (an uproar,
    # a faster one of this turn included) a sleeper without Soundproof (unsupported) wakes
    # instead, with no counter drop (:2432-2439, subscript_wake_up.s:19 clears every sleep bit).
    status = state.status[party(fmt, side), user]
    asleep = ready & (status == Status.SLEEP)
    roused = asleep & uproar_on(state)
    sleep_left = jnp.maximum(state.status_counter[party(fmt, side), user].astype(_I32) - 1, 0)
    sleep_left = jnp.where(roused, 0, sleep_left)
    sleeping = asleep & (sleep_left > 0)
    state = _set_mon(state, "status_counter", side, user, sleep_left, asleep)
    state = _set_mon(state, "status", side, user, Status.NONE, asleep & ~sleeping)
    # CHECK_STATUS_STATE_FREEZE (:2478): RandNext() % 5 == 0 (outcome.thaw) thaws and the
    # move goes on; otherwise the move is disrupted unless it is a thawing move (Flame
    # Wheel, :2481), whose user thaws at CHECK_STATUS_STATE_SELF_THAW (:2728) below.
    # Flare Blitz's family (RECOIL_STATUS with thaws_user, B10-1) is the other effect tested.
    thaw_move = (
        (tables.move_effect[move] == EffectId.SECONDARY_STATUS)
        | (tables.move_effect[move] == EffectId.RECOIL_STATUS)
    ) & (tables.move_param[move, 2] == 1)
    frozen = ready & (status == Status.FREEZE)
    thawed = frozen & (outcome.thaw[side] != 0)
    still_frozen = frozen & ~thawed
    state = _set_mon(state, "status", side, user, Status.NONE, thawed)
    # :2455: a sleeping battler still uses Snore (moveCur) or Sleep Talk (moveTemp).
    sleep_move = (move == MOVE_SNORE) | (move == MOVE_SLEEP_TALK)
    go = ready & ~(sleeping & ~sleep_move) & ~(still_frozen & ~thaw_move)
    # CHECK_STATUS_STATE_TRUANT (:2502-2510): a loafing Truant battler's move is disrupted;
    # subscript_loafing_around.s unlocks it (its message tags only the nickname).
    loafing = go & truant_loafs(state, side)
    state = _unlock(state, side, loafing)
    go = go & ~loafing
    # CHECK_STATUS_STATE_FLINCH (:2529); subscript_flinched.s:10-14 gives Steadfast +1 Speed.
    flinched = go & (state.flinched[side] != 0)
    steadfast = flinched & (battler_ability(state, side, user) == AbilityId.STEADFAST)
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
    # CHECK_STATUS_STATE_IMPRISON (:2572-2583): Move_Imprisoned (battle_lib.c:3316-3338) on
    # moveCur, the move chosen: a battler of the other side with MOVE_EFFECT_IMPRISON knows it (a
    # faster foe's Imprison this turn included). subscript_move_is_imprisoned.s unlocks the user.
    imprisoned = go & move_imprisoned(state, side, move)
    state = _unlock(state, side, imprisoned)
    go = go & ~imprisoned
    # CHECK_STATUS_STATE_GRAVITY (:2586-2595), right after Imprison (enum CheckStatusState): a
    # move chosen before Gravity went up (a faster foe's this turn) that Move_FailsInHighGravity
    # bans: moveFailFlags.gravity and subscript_move_fail_gravity.s, a message only (no unlock,
    # unlike Imprison's). It tests moveCur: Sleep Talk itself, not the move it calls (G-67).
    gravity_ban = go & gravity_banned(state, move)
    go = go & ~gravity_ban
    # CHECK_STATUS_STATE_CONFUSION (:2614): the counter drops first; at 0 the battler snaps
    # out and moves, otherwise RandNext() & 1 (outcome.confusion_hit) makes it hit itself
    # (:2633, damage.confusion_damage) instead of moving. subscript_hurt_self_in_confusion.s
    # runs CheckHoldOnWith1HP on the user (script.c:9098-9124): a Focus Sash at full HP, or a
    # Focus Band whose roll passes, leaves 1 HP, and the followup message removes a sash (the
    # user is the defender, :2629).
    # Its BATTLE_SUBSCRIPT_UPDATE_HP then runs the resist-berry check on the user
    # (effects.resist_berry): BattleContext_Init left moveType Normal and moveStatusFlags 0
    # (battle_lib.c:1981-1982), so calcTemp is the table type of the move the battler
    # chose (moveCur, bcp.c:1898; Hidden Power's too) and nothing is super-effective: a
    # Chilan Berry halves the self-hit when that move is Normal-type, a Rindo Berry never.
    # A flavor berry's pending counter (ids.CONFUSION_PENDING, hooks.berry_confuse) is drawn
    # at this first read: its Random 3, 2 (subscript_confuse.s:34), Outcome.berry_confusion
    # (ruling G-5).
    counter = jnp.where(
        state.confusion[side] == CONFUSION_PENDING,
        outcome.berry_confusion[side],
        state.confusion[side],
    ).astype(state.confusion.dtype)
    confused = go & (counter > 0)
    confusion = jnp.where(go, counter - confused.astype(counter.dtype), state.confusion[side])
    still_confused = confused & (confusion > 0)
    self_hit = still_confused & (outcome.confusion_hit[side] != 0)
    user_hp = state.hp[party(fmt, side), user].astype(_I32)
    sash = holds(tables, state, side, ItemId.FOCUS_SASH)
    full = user_hp == state.max_hp[party(fmt, side), user].astype(_I32)
    # A Focus Band's roll (script.c:9109-9112, ruling G-1): Outcome.focus_band[side, 0], the
    # user's own (its move does not run after a self-hit).
    band = holds(tables, state, side, ItemId.FOCUS_BAND) & (outcome.focus_band[side, 0] != 0)
    raw = confusion_damage(tables, state, side, outcome.roll[side, 0])
    hurt, endured = focus_sash_damage(state, side, raw, (sash & full) | band)
    resisted = self_hit & resist_berry(tables, state, side, tables.move_type[move], False)
    taken = jnp.where(endured, user_hp - 1, raw)
    hurt = resist_berry_damage(state, side, taken, resisted, hurt)
    state = state._replace(confusion=state.confusion.at[side].set(confusion))
    # subscript_hurt_self_in_confusion.s's UPDATE_HP: hpCalcTemp = -the capped damage (B13-17);
    # ``hurt`` has its sign.
    state = hp_update(state, side, user, -hurt, self_hit)
    # subscript_move_followup_message.s:16 names the endured item (the user is the defender).
    state = reveal_item(state, side, user, self_hit & endured)
    state = consume_item(state, side, (self_hit & endured & sash) | resisted)
    state = _unlock(state, side, self_hit)  # subscript_hurt_self_in_confusion.s
    go = go & ~self_hit
    # CHECK_STATUS_STATE_PARALYSIS (:2652-2654): RandNext() % 4 == 0 (outcome.full_para) fully
    # paralyzes, never a Magic Guard holder (Battler_Ability, tested before the draw).
    # Gated on reaching the check, so the event is set only when it had an effect.
    guarded = battler_ability(state, side, user) == AbilityId.MAGIC_GUARD
    full_para = (
        go
        & (state.status[party(fmt, side), user] == Status.PARALYSIS)
        & ~guarded
        & (outcome.full_para[side] != 0)
    )
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
    # D2-14: the release's defender is the last battler that hit the user (``last_hit_by``) while
    # it has HP, else a random opponent with HP (``targets.bide_defender``, bcp.c:2704-2716); none
    # ends the move in subscript_bide_no_target.s. At B = 2 it is the foe while it lives.
    bide_def = targets.bide_defender(state, side, outcome.bide_target_draw[side])
    bide_stop = bide_ends & ((state.bide_damage[side] <= 0) | (bide_def == NO_DEFENDER))
    can_act = can_act & ~bide_stop
    resolved = _redefend(resolved, side, bide_def, bide_ends)
    target_side = resolved.target_side
    target = state.active[target_side]
    # CHECK_STATUS_STATE_SELF_THAW (:2728): a still-frozen user of a thawing move thaws.
    self_thaw = can_act & still_frozen
    state = _set_mon(state, "status", side, user, Status.NONE, self_thaw)

    # BattleControllerPlayer_DecrementPP (battle_controller_player.c:3182) runs before the
    # no-target check (:3190); Struggle has no PP to spend. It takes 1 plus the Pressure cost of
    # moveTemp (``move``, an invoker's own range) against the fight command's defender
    # (effects.pressure_pp_cost), at most the PP left (:2276-2280), ruling G-8.
    spend = can_act & ~struggling & ~locked
    # No PP left in the slot (bcp.c:2275-2285): MOVE_STATUS_NO_PP, the move fails after its attack
    # message. Only an encored move taken over this turn can reach it (B10-9; the selection
    # offers no 0-PP slot).
    no_pp = spend & (state.pp[party(fmt, side), user, slot] == 0)
    spend = spend & ~no_pp
    cost = pressure_pp_cost(tables, state, side, resolved.defender, move)
    left = jnp.maximum(state.pp[party(fmt, side), user, slot].astype(_I32) - cost, 0)
    pp = state.pp.at[party(fmt, side), user, slot].set(
        jnp.where(spend, left, state.pp[party(fmt, side), user, slot]).astype(state.pp.dtype)
    )
    state = state._replace(pp=pp)
    # D2-10: HasNoTarget (bcp.c:3189-3194, :2306-2337) on the fight command's defender
    # (``targets.no_target``): a charge turn with no foe keeps the user as its defender; then D2-12:
    # REDIRECT_TARGET (bcp.c:3204-3206) moves a single-target Electric / Water move to a Lightning
    # Rod / Storm Drain holder (``targets.redirect``), the holder being the action's defender from
    # here on. The PP above took the fight command's defender (rule 9). At B = 2 it never fires
    # (the holder must be the living foe, then already the defender), so the singles step skips it.
    nt_before = targets.no_target(tables, state, side, move, resolved.defender, False)
    keeps_user = targets.charge_keeps_user(tables, state, side, move, resolved.defender)
    if fmt.B == 4:
        redirect_to, redirected = targets.redirect(
            tables, state, side, move, jnp.where(keeps_user, side, resolved.defender), speed_order
        )
        resolved = resolved._replace(
            defender=jnp.where(redirected, redirect_to, resolved.defender),
            target_side=jnp.where(redirected, redirect_to, resolved.target_side),
        )
    else:
        redirected = False
    target_side = resolved.target_side
    target = state.active[target_side]

    # Nature Power (effect_script_0173.s): its script turns it into the terrain's move
    # (order.called_move: GetTerrainMove, then GoToMoveScript sets moveCur and re-picks the
    # defender), so from here on the move that runs is ``used``; the status checks above, the
    # PP, priority and movePrevByBattler (moveTemp) stay Nature Power's.
    nature_power = tables.move_effect[move] == EffectId.NATURE_POWER
    # The invokers (E2 batch 5, B5-1..): Sleep Talk, Metronome and Copycat run the move their
    # script picks (order.invoked_move: BtlCmd_TrySleepTalk, _Metronome, _TryCopycat) through
    # GoToMoveScript (battle_script.c:2111-2134) as Nature Power does, read now, so a faster
    # foe's move this turn counts (Copycat's movePrev). Their own defender is the user
    # (RANGE_SINGLE_TARGET_SPECIAL, battle_lib.c:1665-1669), so they pass the no-target check
    # (bcp.c:3190) even with no living foe and set SYSCTL_MOVE_SUCCEEDED, ruling B12-30; a
    # call that finds no move, or a Sleep Talk whose user is not asleep now (effect_script_0097.s:5: one that woke this turn), is MOVE_STATUS_FAILED
    # ("But it failed!", subscript_missed.s). A called move uses no PP; the status checks above,
    # the PP, priority, movePrevByBattler, movePrev and the Choice lock stay the invoker's
    # (moveTemp).
    kind = call_kind(tables, move)
    invoker = kind != CALL_NONE
    # Me First (E3-2-3; BtlCmd_TryMeFirst, battle_script.c:7286-7312) copies the foe's chosen move
    # while the foe has not acted (``target_moved``): ``pursuit`` carries its action, selected move
    # and struggling flag (``_act``); with none given the call finds nothing.
    me_first = jnp.asarray(0, _I32)
    if pursuit is not None:
        foe_fights = ~is_switch(pursuit[0]) & ~is_item(pursuit[0])
        copied = me_first_copy(tables, state, _foe(side), pursuit[1], foe_fights)
        me_first = jnp.where(jnp.asarray(target_moved, bool), 0, copied)
    invoked = invoked_move(tables, state, side, move, outcome.call[side], me_first=me_first)
    call_failed = invoker & (
        (invoked == 0)
        | ((kind == CALL_SLEEP_TALK) & (state.status[party(fmt, side), user] != Status.SLEEP))
    )
    used = jnp.where(
        invoker & ~call_failed,
        called_move(tables, state, invoked),
        called_move(tables, state, move),
    )
    # TryMeFirst's success sets meFirst and meFirstTurnNumber = meFirstTurnOrder (:7303-7304).
    first_ok = head_on & can_act & (kind == CALL_ME_FIRST) & ~call_failed
    state = state._replace(
        me_first=state.me_first.at[side].set(
            jnp.where(first_ok, 1, state.me_first[side]).astype(state.me_first.dtype)
        ),
        me_first_turn=state.me_first_turn.at[side].set(
            jnp.where(first_ok, state.turn.astype(_I32) + 1, state.me_first_turn[side]).astype(
                state.me_first_turn.dtype
            )
        ),
    )
    # E2 batch 10 family 7, the charge turn (B10-12..). A charge move's script (MoveFlag.CHARGE:
    # effect_script_0039.s, _0075.s, _0145.s, _0151.s, _0155.s, _0255.s, _0256.s, _0263.s, _0272.s)
    # attacks when its user is VOLATILE_CONDITION_MOVE_LOCKED (line 5: the second turn,
    # ``locked_move``, PP spent on the first: bcp.c:335), when SolarBeam meets sun without Cloud
    # Nine / Air Lock (effect_script_0151.s:5-6) or with a Power Herb (HOLD_EFFECT_CHARGE_SKIP,
    # line 6-7: subscript_item_skip_charge_turn.s removes it, :20). Otherwise it charges: the
    # direct TO_ATTACKER subscript_vanish_on_charge_turn.s locks the user into moveCur (:10,
    # Battler_LockMoveChoice, battle_lib.c:1828-1832), the script sets its semi-invulnerable bit
    # and SYSCTL_FIRST_OF_MULTI_TURN: no accuracy roll (bcp.c:2905-2908), no Protect, Lock-On or
    # semi-invulnerable test (:3003-3005), no type chart (:2354) and no damage. A charge with no
    # living foe keeps the user as its defender (HasNoTarget, bcp.c:2327-2334), so it charges.
    used_flags = tables.move_flags[used].astype(_I32)
    charge_move = (used_flags & MoveFlag.CHARGE) != 0
    second_turn = charge_move & (state.locked_move[side] != 0)
    sun_now = (
        ((used_flags & MoveFlag.SUN_SKIP) != 0)
        & (state.weather == Weather.SUN)
        & ~weather_suppressed(tables, state)
    )
    # D2-13: GoToMoveScript (script.c:2111-2134) picks the called move's defender by its own range
    # with randomize TRUE and runs the redirection check (``targets.called_defender``; the
    # invoker's call draw ``Outcome.call_target_draw``), writing it to the stored target (:2123,
    # both formats: ruling D2-33). From here the called move's hit, accuracy, Protect and
    # records read it.
    calls = (invoker & ~call_failed) | nature_power
    called_def, called_redirected = targets.called_defender(
        tables, state, side, used, speed_order, outcome.call_target_draw[side]
    )
    resolved = _redefend(resolved, side, called_def, calls)
    if fmt.B == 4:
        redirected = redirected | (calls & called_redirected)
    target_side = resolved.target_side
    target = state.active[target_side]
    write = can_act & ~no_pp & calls & (called_def != NO_DEFENDER)
    state = state._replace(
        chosen_target=state.chosen_target.at[side].set(
            jnp.where(write, called_def, state.chosen_target[side]).astype(
                state.chosen_target.dtype
            )
        )
    )
    # The script runs with a defender: the foe, or for the user's own first turn the user
    # (HasNoTarget, bcp.c:2327-2334); a called move without a defender goes to subscript_no_target.s
    # from GoToMoveScript (battle_script.c:2126-2128) before its script.
    script_runs = jnp.where(invoker | nature_power, called_def != NO_DEFENDER, ~nt_before)
    herb = (
        can_act
        & charge_move
        & ~second_turn
        & ~sun_now
        & script_runs
        & holds(tables, state, side, ItemId.POWER_HERB)
    )
    charging = can_act & charge_move & ~second_turn & ~sun_now & ~herb & script_runs
    # Skull Bash charges with the indirect TO_ATTACKER Defense +1 (effect_script_0145.s:8, chance
    # 100); with a Power Herb subscript_power_herb_skull_bash.s:14-17 raises it before the hit.
    def_up = (used_flags & MoveFlag.CHARGE_DEF_UP) != 0
    # Accuracy 0 never misses (CheckMoveHitAccuracy returns before rolling, :2901). A
    # multi-hit move checks accuracy on its first hit only (effects.py).
    # The no-target check (:3190) fails a move whose defender has fainted; a move whose
    # defender exists whatever the foe's HP (effects.defender_always) never fails there.
    # Nature Power's own defender is its user (RANGE_SINGLE_TARGET_SPECIAL,
    # BattleSystem_Defender battle_lib.c:1665-1669), so it passes that check; its
    # GoToMoveScript (battle_script.c:2121-2131) re-picks the defender for the called move
    # (``randomize``: in singles the living foe, else BATTLER_NONE, lib.c:1683-1705) and goes
    # to subscript_no_target without one: ``has_target`` is the called move's.
    never_misses = (tables.move_accuracy[used] == 0) | (
        tables.move_effect[used] == EffectId.FUTURE_SIGHT
    )
    # D2-10: the BeforeMove check is ``~nt_before`` (the resolver's defender; at B = 2 today's "the
    # foe has HP or the move has a defender whatever its HP"); a called move's defender is
    # GoToMoveScript's (D2-13), still the foe test. TryMove checks again after the script
    # (bcp.c:3240-3245): a charge move that skipped its charge (Power Herb, sun) with no foe has
    # set SYSCTL_LAST_OF_MULTI_TURN (subscript_charge_move_cleanup.s:7) and ends in no target.
    called_target = called_def != NO_DEFENDER
    bm_target = ~nt_before
    last_of_multi = can_act & bm_target & (second_turn | sun_now | herb)
    nt_try = targets.no_target(
        tables, state, side, used, jnp.where(keeps_user, side, resolved.defender), last_of_multi
    )
    has_target = jnp.where(invoker | nature_power, called_target, bm_target & ~nt_try)
    # Counter, Mirror Coat, Metal Burst (battle_script.c:4615-4705, :7153-7184): the command
    # reads the user's turnFlags damage and its last attacker (physical / special /
    # lastAttacker, stored as 1 + battler) and fails (MOVE_STATUS_FAILED) on 0 taken, an
    # attacker of the user's own side or an attacker with no HP (:4619-4625, :4672-4678, :7157-7162); a
    # RANGE_SINGLE_TARGET_SPECIAL one is its own defender until then (battle_lib.c:1665-1669),
    # so it passes the no-target check and the command re-picks the foe.
    counter = tables.move_effect[used] == EffectId.COUNTER
    source = tables.move_param[used, 0].astype(_I32)
    foe_taken = jnp.asarray(foe_taken, _I32)
    pick = jnp.where(source == COUNTER_PHYSICAL, 0, jnp.where(source == COUNTER_SPECIAL, 1, 2))
    taken_by = foe_taken[pick]
    attacker = jnp.clip(foe_taken[3 + pick] - 1, 0, fmt.B - 1)
    counter_damage = taken_by * tables.move_param[used, 1].astype(_I32) // 10
    counter_ok = (
        (taken_by != 0)
        & (foe_taken[3 + pick] != 0)
        & (side_of(attacker) != side_of(side))  # D2-14: a partner's hit fails it too
        & (state.hp[party(fmt, attacker), state.active[attacker]] > 0)
    )
    # D2-14: the command's defender is the living Follow Me user of the last attacker's side, else
    # the last attacker (``targets.counter_defender``); the accuracy roll, Protect, the damage and
    # the records read it from here, and so does Pressure below.
    counter_def = targets.counter_defender(state, side, attacker)
    resolved = _redefend(resolved, side, counter_def, counter & counter_ok)
    target_side = resolved.target_side
    target = state.active[target_side]
    special_counter = counter & self_defender(tables, used, ghost_user(state, side))
    # Me First's defender is its target, not its user: it needs a living one (RANGE_SINGLE_TARGET_
    # ME_FIRST, battle_lib.c:1599-1727), unlike the self-targeting invokers.
    self_invoker = invoker & (kind != CALL_ME_FIRST)
    has_target = has_target | special_counter | (self_invoker & call_failed) | charging
    # Singles follow-up (ii): a lock move (Rollout, Ice Ball, a charge move's strike) reads the
    # stored target (lib.c:1711, :1718-1719), the user after a RANGE_MY_SIDE or non-Ghost Curse
    # selection, so the defender is the user (``targets.resolve``: ``target_side == side``). The
    # checks below read the defender, not the foe: the user's own stages, items and abilities
    # (bcp.c:2865-2981), no Protect / Endure / Roost of the defender (its flags are set by its own
    # action, which is this one), and ``Battler_MovedThisTurn(defender)`` (lib.c:5365-5368) is false
    # until MoveEnd (bcp.c:3982): no Zoom Lens, no flinch immunity, and no Quick Claw / Custap
    # flag of the defender (both cleared when its action started).
    self_target = target_side == side
    acc_target = target_side
    tgt_moved = jnp.asarray(target_moved, bool) & ~self_target
    foe_protecting = jnp.asarray(foe_protecting, bool) & ~self_target
    foe_roosting = jnp.asarray(foe_roosting, bool) & ~self_target
    foe_enduring = jnp.asarray(foe_enduring, bool) & ~self_target
    claw_target = jnp.asarray(claw, bool)[target_side] & ~self_target
    custap_target = custap[target_side] & ~self_target
    # A Counter-family command that passes (its script's first command, effect_script_0089.s,
    # _0144.s, _0227.s) runs BattleSystem_DecPPForPressure on its new defender, the last
    # attacker (battle_script.c:4646, :4699, :7180; battle_lib.c:7390-7397): one more PP from
    # the user's chosen slot (moveSlot: an invoker's when the invoker called it) while that slot
    # has PP left and the defender has Pressure, ruling G-9.
    counter_pressure = (
        can_act & has_target & counter & counter_ok & pressure_holder(state, counter_def)
    ) & (state.pp[party(fmt, side), user, slot] > 0)
    state = state._replace(
        pp=state.pp.at[party(fmt, side), user, slot].add(
            jnp.where(counter_pressure, -1, 0).astype(state.pp.dtype)
        )
    )
    # SYSCTL_MOVE_SUCCEEDED (battle_controller_player.c:3212) once the move got past the
    # status checks, the PP decrement and the no-target check, before the accuracy check
    # (a miss counts): UpdateMoveBuffers (:3945-3949) writes the move (Struggle's own id
    # when struggling; Nature Power, moveTemp, for Nature Power, even when its call finds no
    # target) to movePrevByBattler, else MOVE_NONE. Every path of a battler that acts
    # reaches it, the confusion self-hit too (LOOP_FAINTED, :2639).
    # A charge move's own first use passes BeforeMove's no-target check whatever the foe's HP
    # (HasNoTarget, bcp.c:2327-2334; its hit may still find none after a Power Herb), B10-12.
    first_charge = charge_move & ~second_turn & ~invoker & ~nature_power
    succeeded = can_act & (has_target | nature_power | self_invoker | first_charge)
    state = _set_last_move(state, side, jnp.where(succeeded, move, 0), ready)
    # BattleSystem_UpdateMetronomeCount at the end of BeforeMove (bcp.c:3221): reached once the
    # status checks pass and HasNoTarget finds a target (bcp.c:3175-3179; DecrementPP never
    # returns TRUE): where SYSCTL_MOVE_SUCCEEDED is set (``succeeded``), before the move's
    # script (bcp.c:3214) and its damage (G-60). So it reads the state before the script: a
    # charge move's second turn is still VOLATILE_CONDITION_MOVE_LOCKED (the script's
    # subscript_charge_move_cleanup.s:6 clears it later) and does not count; a Recycle (or a
    # Natural Gift, a Power Herb) changes the held item only after it (B13-34).
    state = metronome_update(tables, state, side, move, head_on & succeeded)
    # The move script runs next (BATTLE_CONTROL_EXEC_SCRIPT, :3214): Fake Out's
    # (effect_script_0158.s) sets MOVE_STATUS_FAILED unless fakeOutTurnNumber == totalTurns.
    # The accuracy roll still happens, but subscript_missed.s prints "But it failed!" (:97)
    # ahead of the miss message, so only an attempt that did not fail can show a miss.
    failed = (tables.move_effect[used] == EffectId.FAKE_OUT) & (
        state.fake_out_turn[side].astype(_I32) != state.turn.astype(_I32)
    )
    # Snore (effect_script_0092.s:5, :15-17): MOVE_STATUS_FAILED unless its user is asleep.
    snore = (tables.move_effect[used] == EffectId.FLINCH) & (
        tables.move_param[used, 2] == SNORE_PARAM
    )
    failed = failed | (snore & (state.status[party(fmt, side), user] != Status.SLEEP))
    failed = failed | (counter & ~counter_ok) | call_failed
    # Ingrain and Aqua Ring (effect_script_0181.s:5, _0251.s:5) fail a user that has the effect
    # already: MOVE_STATUS_FAILED from the script, as Fake Out's (rulings B7-n).
    effect_used = tables.move_effect[used]
    failed = failed | ((effect_used == EffectId.INGRAIN) & (state.ingrain[side] != 0))
    failed = failed | ((effect_used == EffectId.AQUA_RING) & (state.aqua_ring[side] != 0))
    # Magnet Rise (effect_script_0252.s:5-7, :16): the turns, Levitate or Ingrain (G-66).
    failed = failed | ((effect_used == EffectId.MAGNET_RISE) & magnet_rise_fails(state, side))
    # Gravity (effect_script_0215.s:3, :9-10): MOVE_STATUS_FAILED with Gravity already up (G-67).
    failed = failed | ((effect_used == EffectId.GRAVITY) & (state.gravity != 0))
    # Helping Hand (effect_script_0176.s:5, :15-17): BtlCmd_TryHelpingHand fails outside a double
    # battle (battle_script.c:6309-6311), so always in singles (B9-5).
    failed = failed | (effect_used == EffectId.HELPING_HAND)
    # Teleport (effect_script_0153.s:5, :9-10): MOVE_STATUS_FAILED in a trainer battle, which every
    # Nuzline battle is (E3-1-4); it still counts as used (bcp.c:3212, :3945-3949).
    failed = failed | (effect_used == EffectId.FLEE)
    # Dream Eater (effect_script_0008.s:5-6, :12-20): MOVE_STATUS_NO_MORE_WORK when the defender
    # is behind a substitute (:5, CheckSubstitute; no hit of this action came first) or is not
    # asleep (B9-12, B10-4).
    dream = (effect_used == EffectId.DRAIN) & (tables.move_param[used, 1] == 1)
    awake = state.status[party(fmt, target_side), state.active[target_side]] != Status.SLEEP
    failed = failed | (dream & (awake | (state.substitute_hp[target_side] != 0)))
    # Endeavor (effect_script_0189.s:5-7, :13-14): MOVE_STATUS_FAILED unless the defender's curHP
    # exceeds the user's (B10-5).
    gap = state.hp[party(fmt, target_side), state.active[target_side]].astype(_I32) - state.hp[
        party(fmt, side), state.active[side]
    ].astype(_I32)
    failed = failed | ((effect_used == EffectId.ENDEAVOR) & (gap <= 0))
    failed = failed | no_pp  # B10-9: the encored move had no PP
    # Focus Punch (effect_script_0170.s:5-6, :12-16): MOVE_STATUS_LOST_FOCUS, a DID_NOT_HIT status
    # (subscript_missed.s), when the user's physical / specialDamageAttackerMask is set: a hit on
    # its HP this turn (``foe_taken``'s attackers; a substitute hit sets none), B10-8.
    taken_mask = jnp.asarray(foe_taken, _I32)
    lost_focus = (taken_mask[3] != 0) | (taken_mask[4] != 0)
    failed = failed | ((effect_used == EffectId.FOCUS_PUNCH) & lost_focus)
    # E2 batch 13, the held-item moves. Their scripts run once the move passed the PP and
    # no-target checks (bcp.c:3205-3216: MOVE_STATUS_NO_PP is a NO_EFFECTS status, so no script).
    script_on = can_act & has_target & ~no_pp
    held_now = state.item[party(fmt, side), user]
    # Recycle (effect_script_0184.s): MOVE_STATUS_FAILED when the user holds an item (:5) or its
    # battler's recycleItem is empty (BtlCmd_TryRecycle, battle_script.c:9000-9013); else the
    # record is cleared and the user holds the item again (:7, UpdateMonDataFromVar: a
    # CopyToParty of a held item, no AI record change; the message names msgItemTemp, no
    # reveal). The move end's held-item check (AfterMoveEffects) then sees it (B13-19).
    recycle = effect_used == EffectId.RECYCLE
    recycle_fails = (held_now != 0) | (state.recycle_item[side] == 0)
    failed = failed | (recycle & recycle_fails)
    recycled = script_on & recycle & ~recycle_fails
    state = _set_mon(state, "item", side, user, state.recycle_item[side], recycled)
    state = _set_side(state, "recycle_item", side, 0, recycled)
    # Natural Gift (effect_script_0222.s): BtlCmd_CalcNaturalGiftParams fails at naturalGiftPower
    # 0 (battle_script.c:8094-8099; no item included); else CalcCrit, CalcDamage, then
    # RemoveItem BTLSCR_ATTACKER (:8) before TryMove: a miss or Protect still eats the berry.
    # The hit reads its power and type through damage.gift_item, given the removed berry
    # (``gift_removed``; 0 for every other action), B13-20, B13-37.
    gift = effect_used == EffectId.NATURAL_GIFT
    # Battler_NaturalGiftPower reads Battler_HeldItem (battle_lib.c:5465-5469: none under Klutz
    # or Embargo, B13-29); Recycle's test above reads BATTLEMON_HELD_ITEM itself.
    gift_fails = natural_gift_params(tables, held_item(state, side, user))[0] == 0
    failed = failed | (gift & gift_fails)
    gift_on = script_on & gift & ~gift_fails
    gift_removed = jnp.where(gift_on, held_item(state, side, user).astype(jnp.int32), 0)
    state = consume_item(state, side, gift_on)
    # Embargo (effect_script_0232.s:5, :7-8): CheckSubstitute on the defender, MOVE_STATUS_FAILED
    # before TryMove (its subscript's own failures come at the hit: effects._embargo), B13-31.
    failed = failed | ((effect_used == EffectId.EMBARGO) & (state.substitute_hp[target_side] != 0))
    # Imprison (effect_script_0192.s): BtlCmd_TryImprison first sets MOVE_EFFECT_IMPRISONED on
    # the user's foes (battle_script.c:6686-6690: both enemy slots, which in singles name the
    # one foe battler, whatever its HP), then fails (:10) a user already imprisoning or sharing
    # no move with them (effects.imprison_fails).
    imprison_runs = can_act & has_target & (effect_used == EffectId.IMPRISON)
    for battler in range(state.active.shape[0]):  # battler b is on side b in singles
        state = _set_side(state, "imprisoned", battler, 1, imprison_runs & (side != battler))
    failed = failed | ((effect_used == EffectId.IMPRISON) & imprison_fails(state, side))
    # Follow Me (effect_script_0172.s:5-8): BtlCmd_FollowMe (battle_script.c:6261-6270) sets the
    # user's side's followMe and followMeUser, on every run of the script (no failure branch; a
    # called Follow Me counts: ``effect_used`` is the called move's). One per-side field holds
    # both (``ts_follow_me``, D2-11): 1 + the user battler, read by ``targets.follow_me_target``.
    # Written in both formats (ruling D2-33): at B = 2 the other side's non-Ghost Curse reads it.
    follow = script_on & (effect_used == EffectId.FOLLOW_ME)
    state = _set_side(state, "ts_follow_me", side, side + 1, follow)
    # Mud Sport and Water Sport fail the same way (effect_script_0201.s:5, _0210.s:5).
    failed = failed | ((effect_used == EffectId.MUD_SPORT) & (state.mud_sport[side] != 0))
    failed = failed | ((effect_used == EffectId.WATER_SPORT) & (state.water_sport[side] != 0))
    # Future Sight / Doom Desire (effect_script_0148.s:5): BtlCmd_TryFutureSight
    # (battle_script.c:6065-6100) runs once the move passed the no-target check. With nothing
    # pending on the defender's battler (futureSightTurns == 0) it sets the defender side's
    # SIDE_CONDITION_FUTURE_SIGHT, 3 turns, the move (moveCur), the attacker (1 + battler) and the
    # damage of this moment: BattleSystem_CalcMoveDamage (criticalMul 1, the defender side's and
    # the field's conditions of now, :6079-6089; no type chart, no STAB, no Life Orb) times -1,
    # then BattleSystem_CalcDamageVariance (:6090; battle_lib.c:7081-7094: RandNext() % 16,
    # Outcome.roll[side, 0]; CalcMoveDamage's + 2 keeps it nonzero, so the "at least 1" never turns
    # its sign); Helping Hand (x15/10, :6092-6094) is unsupported. The script then sets SYSCTL_NONSTANDARD_ACC_CHECK and
    # SYSCTL_IGNORE_IMMUNITIES (:7): no accuracy roll (bcp.c:2911-2913), no type chart (:2354),
    # and its ON_HIT effect only prints. With one pending it prints "But it failed!"
    # (:10-14, subscript_but_it_failed.s: MOVE_STATUS_NO_MORE_WORK, no AfterMoveEffects, no miss
    # shown).
    future = can_act & has_target & (tables.move_effect[used] == EffectId.FUTURE_SIGHT)
    pending = state.future_timer[target_side] != 0
    failed = failed | (future & pending)
    sets = future & ~pending
    rolled = 85 + outcome.roll[side, 0].astype(_I32)
    stored = jnp.maximum(calc_move_damage(tables, state, side, used) * rolled // 100, 1)
    # futureSightDamage is an s32; the hit reads it only through min(damage, curHP) and
    # damage >= curHP, so it is kept capped to the int16 storage (above any HP).
    stored = jnp.minimum(stored, FUTURE_DAMAGE_MAX)
    state = _set_side(state, "future_sight", target_side, 1, sets)
    state = _set_side(state, "future_timer", target_side, FUTURE_SIGHT_TURNS, sets)
    state = _set_side(state, "future_move", target_side, used, sets)
    state = _set_side(state, "future_attacker", target_side, side + 1, sets)
    state = _set_side(state, "future_damage", target_side, stored, sets)
    # The Power Herb (line 6-7 of the charge scripts): subscript_item_skip_charge_turn.s:20 /
    # subscript_power_herb_skull_bash.s:24 RemoveItem, then the hit (a defender of BATTLER_NONE
    # skips it and TryMove's HasNoTarget, with SYSCTL_LAST_OF_MULTI_TURN set by
    # subscript_charge_move_cleanup.s:7, ends in subscript_no_target.s).
    state = consume_item(state, side, herb)
    state = change_stage(state, side, Stat.DEF, 1, herb & def_up)
    # The second turn's script clears the semi-invulnerable bit (effect_script_0155.s:24 and the
    # others) and subscript_charge_move_cleanup.s:6 VOLATILE_CONDITION_MOVE_LOCKED before TryMove.
    cleanup = can_act & has_target & second_turn
    state = _set_side(state, "semi_invulnerable", side, 0, cleanup)
    state = _set_side(state, "locked_move", side, 0, cleanup)
    # Selfdestruct / Explosion (effect_script_0007.s): the script sets the user's HP to 0
    # before TryMove, so the user faints on a miss or into a Ghost too, and its own Cloud
    # Nine no longer counts for the hit (COUNT_ALIVE_BATTLERS). The faint itself is
    # BATTLE_CONTROL_FAINT_AFTER_SELFDESTRUCT (bcp.c:3889), within this action, so the
    # action's faint check sees it with the target's. Damp (the script's other branch) is
    # an unsupported ability.
    self_destruct = can_act & has_target & (tables.move_effect[used] == EffectId.SELF_DESTRUCT)
    state = _set_mon(state, "hp", side, user, 0, self_destruct)
    # CheckMoveHitAccuracy reads the threshold now, with this moment's stages (bcp.c:
    # 2873-2874): a faster Sand-Attack or Double Team this turn counts. outcome.hit[side, 0]
    # is the roll's class (outcome.py): it hits when 0 < class <= threshold (:2976).
    # TryMove's TriggerRedirectionAbilities (bcp.c:3247-3252): the holder's ability message
    # (subscript_lightning_rod_redirected.s:9) unless the move failed (D2-12 rule 10).
    if fmt.B == 4:
        holder = jnp.clip(resolved.defender, 0, fmt.B - 1)
        state = reveal_ability(
            state,
            holder,
            state.active[holder],
            REDIRECTED,
            redirected & can_act & has_target & ~failed,
        )
    roll = outcome.hit[side, 0].astype(_I32)
    threshold = live_hit_threshold(tables, state, side, used, tgt_moved, target=acc_target)
    hit = never_misses | ((roll > 0) & (roll <= threshold)) | charging
    # bcp.c:2967-2970: the check clears the attacker's Micle Berry flag once it gets past its
    # early returns (accuracy 0, SYSCTL_FIRST_OF_MULTI_TURN, SYSCTL_NONSTANDARD_ACC_CHECK of
    # Future Sight and an OHKO move), G-59.
    accuracy_run = (
        can_act
        & has_target
        & ~charging
        & ~never_misses
        & (tables.move_effect[used] != EffectId.OHKO)
    )
    state = _set_side(state, "micle", side, 0, accuracy_run)
    repeated = (tables.move_effect[used] == EffectId.ROLLOUT) & (state.rollout_count[side] != 0)
    # CheckMoveHitOverrides (bcp.c:3016-3040): past Protect and the Lock-On / No Guard return
    # (either battler's No Guard, Battler_Ability; a Lock-On onto the defender; an OHKO move's
    # own bypass below, B10-15), a move whose defender is the foe, not RANGE_OPPONENT_SIDE, misses
    # (MOVE_STATUS_SEMI_INVULNERABLE) a foe with a semi-invulnerable bit its script does not hit
    # (MoveFlag.HIT_*: SYSCTL_HIT_DURING_*), B10-12. A special-range Counter's command and Bide's
    # unleash have made the foe the defender by then.
    foe_semi = state.semi_invulnerable[target_side]
    # A non-Ghost Curse whose defender is the foe (Follow Me, or a randomized call / Encore pick:
    # BattleSystem_Defender, lib.c:1706-1707, :1714-1717) is tested like any move aimed at it.
    curse_at_foe = (
        (effect_used == EffectId.CURSE)
        & ~ghost_user(state, side)
        & (resolved.defender != NO_DEFENDER)
        & (resolved.defender != side)
    )
    aims_foe = (
        ~self_defender(tables, used, ghost_user(state, side))
        | special_counter
        | bide_ends
        | curse_at_foe
    )
    no_guard = (battler_ability(state, side, user) == AbilityId.NO_GUARD) | (
        battler_ability(state, target_side, target) == AbilityId.NO_GUARD
    )
    # An OHKO move: BtlCmd_TryOHKOMove sets SYSCTL_NONSTANDARD_ACC_CHECK first (battle_script.c:
    # 4338), which skips the Lock-On / No Guard return (bcp.c:3016); only its
    # MOVE_STATUS_BYPASSED_ACCURACY skips the semi test (:3035), set on the non-Sturdy branch
    # when the defender has MOVE_EFFECT_LOCK_ON or either battler has No Guard (script.c:
    # 4342-4372). Against an ignorable Sturdy (:4340-4341) the semi-invulnerable foe is missed,
    # and subscript_missed.s:11-12 shows that miss before the Sturdy message (B10-15).
    ohko = tables.move_effect[used] == EffectId.OHKO
    ohko_bypass = (ignorable(state, side, target_side) != AbilityId.STURDY) & (
        (state.lock_on[target_side] != 0) | no_guard
    )
    bypass = jnp.where(ohko, ohko_bypass, no_guard | locked_onto(state, side, target_side))
    semi_miss = (
        can_act
        & has_target
        & ~charging
        & aims_foe
        & (foe_semi != 0)
        & ~hits_semi(tables, used, foe_semi)
        & ~targets_foe_side(tables, used)
        & ~bypass
    )
    hit = hit & ~semi_miss
    # CheckMoveHitOverrides (bcp.c:3001-3014) runs after the accuracy roll, hit or miss: a
    # protecting foe stops the move (effects.protect_blocks), MOVE_STATUS_PROTECTED, and
    # Battler_UnlockMoveChoice (the unlock below). Selfdestruct's user already fainted.
    blocked = foe_protecting & protect_blocks(tables, used, bide_ends) & ~charging
    # An OHKO move against an ignorable Sturdy (BtlCmd_TryOHKOMove, battle_script.c:4340-4341:
    # MOVE_STATUS_STURDY, no roll): subscript_missed.s:67-69 tags the defender's ability
    # ("{0} was protected by {1}!", BattleAI_SetAbility, battle_script.c:11798-11806), unless a
    # flag it tests first holds: protected (:18), Levitate (:34) or ineffective (:59).
    ohko_type = move_type_of(tables, state, side, used, gift_removed)
    sturdy_shown = (
        can_act
        & has_target
        & ~blocked
        & (tables.move_effect[used] == EffectId.OHKO)
        & (ignorable(state, side, target_side) == AbilityId.STURDY)
        & (
            effectiveness(
                tables,
                ohko_type,
                state.types[party(fmt, target_side), target],
                identified(tables, state, side, target_side),
            )
            != 0
        )
        & ~ground_immune(tables, state, side, target_side, ohko_type)
        # subscript_missed.s:13 shows the miss of MOVE_STATUS_SEMI_INVULNERABLE first.
        & ~semi_miss
    )
    state = reveal_ability(state, target_side, target, OHKO_BLOCKED, sturdy_shown)
    # Yawn is a direct CHECK_HP effect (effect_script_0187.s:5), whose subscript still runs on
    # MOVE_STATUS_GENERAL_MISS (battle_lib.c:1500-1503) for a living defender: subscript_yawn.s:
    # 5-9 test the blocking abilities before its miss test (:21) and print "made it ineffective"
    # with TAG_NICKNAME_ABILITY (:31-37 -> BattleAI_SetAbility, battle_script.c:11798-11806).
    # Protect returns before the semi test (bcp.c:3007-3013). The hit path is effects._yawn.
    yawn_shown = (
        semi_miss
        & ~blocked
        & (tables.move_effect[used] == EffectId.YAWN)
        & (state.hp[party(fmt, target_side), target] > 0)
        & yawn_blocks(tables, state, side, target_side)[1]
    )
    state = know_ability(state, target_side, target, yawn_shown)
    connected = can_act & has_target & hit & ~failed & ~blocked & ~charging
    # The effect script runs before TryMove's accuracy roll and CheckMoveHitOverrides (bcp.c:
    # 3211-3213, then :3256, :3264), so its own commands act on a move that then misses. Psych
    # Up's CopyStatStages (effect_script_0143.s:5) copies on a semi-invulnerable miss (accuracy 0
    # and no MOVE_FLAG_CAN_PROTECT, res/moves/psych_up/data.json, leave it no other miss), B12-6;
    # Captivate's CheckIgnorableAbility (effect_script_0265.s:5) tags a blocking Oblivious on a
    # miss or a Protect too (:48-53), B12-8. The paths of a move that connected are
    # effects._psych_up and effects._captivate.
    scripted = can_act & has_target & ~failed & ~charging & ~connected
    state = copy_stat_stages(
        state, side, target_side, scripted & semi_miss & (effect_used == EffectId.PSYCH_UP)
    )
    captivate_shown = (
        scripted
        & (effect_used == EffectId.CAPTIVATE)
        & (state.hp[party(fmt, target_side), target] > 0)
        & captivate_fails(tables, state, side, target_side)[1]
    )
    state = know_ability(state, target_side, target, captivate_shown)
    # Protect / Detect (effect_script_0111.s, a direct TO_ATTACKER effect: the user has HP)
    # -> subscript_protect.s -> BtlCmd_TryProtection (battle_script.c:5149-5190). The
    # chain reset of :5154-5158 is folded into protect_count (below), so its value here is
    # protectSuccessTurns. Success iff sProtectSuccessRate[n] >= RandNext()
    # (outcome.protect; n = 0 never fails) and the user is not the last battler to act
    # (waitingBattlers != 1, bcp.c:881-886: in singles, it acts first): turnFlags.protecting
    # and the count up to PROTECT_CHAIN_MAX; else the count drops to 0 ("But it failed!").
    # Endure (effect_script_0116.s:3, G-65) is the same script, roll and chain: a success sets
    # turnFlags.enduring instead of protecting (battle_script.c:5174-5177).
    count = state.protect_count[side].astype(_I32)
    tried_protect = connected & (tables.move_effect[used] == EffectId.PROTECT)
    tried_endure = connected & (tables.move_effect[used] == EffectId.ENDURE)
    tried = tried_protect | tried_endure
    tried_ok = tried & ~target_moved & ((count == 0) | (outcome.protect[side] != 0))
    protect_ok = tried_protect & tried_ok
    endure_ok = tried_endure & tried_ok
    protect_failed = tried & ~tried_ok
    count = jnp.where(
        tried_ok, jnp.minimum(count + 1, PROTECT_CHAIN_MAX), jnp.where(protect_failed, 0, count)
    )
    # UpdateMoveBuffers (bcp.c:3936-3942): moveProtect = moveCur (Nature Power's called move)
    # when the attack message was shown (the move got past the status checks: can_act), else
    # MOVE_NONE, on every path of a battler that acts. The next TryProtection resets the
    # chain unless moveProtect is Protect, Detect or Endure (script.c:5154-5158); the engine
    # folds that
    # reset in here, and AICmd_LoadProtectChain (trainer_ai.c:2546-2552) reads the folded
    # value. A switch, a bag item and Pursuit's pre-switch hit (TryPursuit ends the action
    # without UpdateMoveBuffers, script.c:6931) leave it alone.
    chain = jnp.asarray(PROTECT_CHAIN_MOVES, _I32)
    keep = can_act & jnp.isin(jnp.asarray(used, _I32), chain)
    count = jnp.where(ready & ~keep, 0, count)
    state = state._replace(
        protect_count=state.protect_count.at[side].set(count.astype(state.protect_count.dtype))
    )
    # Roar / Whirlwind (effect_script_0028.s: a DIRECT ON_HIT effect, run only for a move
    # that connected: no miss, no Protect, battle_lib.c:1485-1490; Soundproof, Suction Cups,
    # Ingrain and the semi-invulnerable test are unsupported, refused by coverage) ->
    # subscript_force_target_to_switch_or_flee.s -> BtlCmd_TryWhirlwind (battle_script.c:
    # 5241-5320, the trainer-battle singles branch): with no eligible bench Pokemon
    # (outcome.roar_bench) it fails before any draw (:5299-5300); else
    # BattleSystem_CanWhirlwind (battle_lib.c:3072-3086) passes for a user of at least the
    # target's level, otherwise on outcome.roar_level; then the rejection loop picks the
    # slot (outcome.roar_target, its rank among the eligible, :5303-5311). A failure is
    # MOVE_STATUS_FAILED: "But it failed!" (subscript_missed.s), no AfterMoveEffects.
    # Success: the target's Natural Cure (TryRestoreStatusOnSwitch, :11 of the subscript),
    # SwitchAndUpdateMon (InitBattleMon, UpdateAfterSwitch: battle_lib.c:2040-2155), the
    # hazards (subscript_hazards_check.s), and the switch-in effects of AfterMoveEffects
    # (bcp.c:3672-3680): ``_send_out``, a switch action's own path. There is no Pursuit
    # (the subscript has none) and no faint call: a hazard faint is caught by this action's
    # faint check. UpdateAfterSwitch ends the newcomer's turn (battlerActions = MOVE_END,
    # battle_lib.c:2053): ``_execute_actions`` skips its action.
    # subscript_force_target_to_switch_or_flee.s:7 compares the defender's moveEffectsMask with
    # OPCODE_EQU: only a mask that is exactly MOVE_EFFECT_INGRAIN anchors it ("{0} anchored
    # itself with its roots!", :86, MOVE_STATUS_FAILED); Ingrain beside any other bit of the mask
    # lets Roar through (ruling B7-n; ``ingrain_anchors``).
    roar = connected & (tables.move_effect[used] == EffectId.ROAR)
    anchored = roar & ingrain_anchors(tables, state, target_side)
    roar = roar & ~anchored
    eligible = roar_bench(state, target_side, target)
    level_ok = (
        state.level[party(fmt, side), user] >= state.level[party(fmt, target_side), target]
    ) | (outcome.roar_level[side] != 0)
    dragged = roar & jnp.any(eligible) & level_ok
    roar_failed = (roar & ~dragged) | anchored
    drag_slot = roar_slot(eligible, outcome.roar_target[side])
    pre_drag = state
    state = _where(dragged, _send_out(tables, state, target_side, drag_slot, effects=False), state)
    # The drag's Toxic Spikes poisoned the newcomer (subscript_hazards_check.s:14-28): its status
    # subscript set SYSCTL_TRY_SYNCHRONIZE_STATUS with the newcomer, the new defender, as
    # sideEffectMon (BtlCmd_CheckToxicSpikes, battle_script.c:7640-7641; ruling G-13).
    newcomer_status = state.status[party(fmt, target_side), drag_slot]
    roar_poisoned = (
        dragged
        & (pre_drag.status[party(fmt, target_side), drag_slot] == Status.NONE)
        & ((newcomer_status == Status.POISON) | (newcomer_status == Status.TOXIC))
    )
    # subscript_no_target.s and subscript_missed.s unlock the attacker (a Rollout that
    # missed; a first use never locked, so this is then a no-op).
    state = _unlock(state, side, can_act & ~connected & ~charging)
    # The charge (B10-12): the lock into moveCur (subscript_vanish_on_charge_turn.s:10), the
    # semi-invulnerable bit (effect_script_0155.s:7, _0255.s:7, _0256.s:7, _0263.s:7, _0272.s:8)
    # and Skull Bash's Defense +1 (effect_script_0145.s:8).
    state = _set_side(state, "locked_move", side, used, charging)
    semi_kind = jnp.select(
        [(used_flags & int(SEMI_SET_FLAGS[k])) != 0 for k in sorted(SEMI_SET_FLAGS)],
        [jnp.int32(int(k)) for k in sorted(SEMI_SET_FLAGS)],
        jnp.int32(0),
    )
    state = _set_side(state, "semi_invulnerable", side, semi_kind, charging)
    state = change_stage(state, side, Stat.DEF, 1, charging & def_up)
    n_hits = hit_count(tables, used, outcome, side, state)
    # effect_script_0122.s _004: Present's heal path sets SYSCTL_IGNORE_IMMUNITIES (:13) in the
    # move script (bcp.c:3214), before TryMove's chart (:3272), hit or miss: CheckTypeChart skips
    # the chart (bcp.c:2350-2356), so no Wonder Guard flag and no noEffect (ruling G-30).
    heal_path = (tables.move_effect[used] == EffectId.PRESENT) & (
        outcome.present[side] == PRESENT_HEAL
    )
    # Present's heal path (effects._present): subscript_recover_hp.s prints "regained health"
    # (1), or "HP is full!" (2) for a target at full HP; the target's HP is unchanged
    # between here and the hit.
    present_heal = (
        connected
        & (tables.move_effect[used] == EffectId.PRESENT)
        & (outcome.present[side] == PRESENT_HEAL)
    )
    target_full = (
        state.hp[party(fmt, target_side), target] == state.max_hp[party(fmt, target_side), target]
    )
    present_heal = jnp.where(present_heal, jnp.where(target_full, 2, 1), 0)
    # A Recover / Present heal on a full-HP recipient (single-hit moves): MOVE_STATUS_NO_MORE_WORK
    # skips the move's AfterMoveEffects (effects.recovers_at_full_hp), so no item check runs in
    # any loop pass below (the passes after the hit would otherwise repeat it).
    no_more_work = connected & recovers_at_full_hp(tables, state, side, used, outcome)
    # A direct effect that fails (Swagger) or does nothing (Dragon Dance at +6 / +6):
    # MOVE_STATUS_FAILED / _NO_MORE_WORK skip AfterMoveEffects (effects.primary_effect_fails).
    no_more_work = no_more_work | (
        connected
        & primary_effect_fails(
            tables,
            state,
            side,
            target_side,
            resolved.defender,
            used,
            # play_hit's flags (below): the defender's quickClaw and custapBerry until it moved.
            foe_claw=claw_target & ~tgt_moved,
            foe_custap=custap_target & ~tgt_moved,
        )
    )
    # TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES (bcp.c:3281-3286, :2782-2810): a move that hit
    # (no MOVE_STATUS_DID_NOT_HIT: missed, protected, immune) meets the defender's Dry Skin
    # (BattleSystem_TriggerImmunityAbility, battle_lib.c:3538-3544: Battler_IgnorableAbility, a
    # Water move by CalcMoveType, Hidden Power's included, with table power, not a charge turn):
    # subscript_ability_restores_hp.s heals Divide(maxHP, 4) through UPDATE_HP (capped), or
    # says the move was useless at full HP, and tags the holder's ability either way; then
    # MOVE_STATUS_NO_MORE_WORK skips the hits and AfterMoveEffects (LOOP_SPREAD_MOVES).
    # A foe that roosted earlier this turn: the move's type chart skips its Flying rows
    # (battle_lib.c:2540-2543). Every reader of the target's types inside the move is the chart's
    # or a non-Flying test, so its types read as damage.chart_types until the move ends.
    target_types = state.types[party(fmt, target_side), target]
    # (a foe the move's Roar dragged out is no longer the target: it keeps its party types).
    roosted_foe = foe_roosting & (state.active[target_side] == target)
    roost_types = chart_types(state, target_side, target, roosted_foe).astype(state.types.dtype)
    state = state._replace(types=state.types.at[party(fmt, target_side), target].set(roost_types))
    move_type = move_type_of(tables, state, side, used, gift_removed)
    chart = effectiveness(
        tables,
        move_type,
        state.types[party(fmt, target_side), target],
        identified(tables, state, side, target_side),
    )
    # TRY_MOVE_STATE_CHECK_TYPE_CHART (bcp.c:2350-2370, run hit, miss or Protect alike once the
    # move passed the no-target check, for a move with power that is not RANGE_USER /
    # RANGE_USER_SIDE and not under SYSCTL_IGNORE_IMMUNITIES): Wonder Guard's
    # MOVE_STATUS_WONDER_GUARD (damage.wonder_guarded, ruling G-18). ``guard_tested``: the chart
    # ran; ``guarded``: on a move that connected, the hits deal nothing (damage_for_roll) and the
    # move ends in subscript_missed.s (bcp.c:3324-3327): no AfterMoveEffects (``items_full``),
    # Jump Kick crashes (below). It is not MOVE_STATUS_INEFFECTIVE: no noEffect (below).
    guard_tested = (
        can_act
        & has_target
        & ~(invoker & call_failed)
        & ~targets_user(tables, used, ghost_user(state, side))
        & (tables.move_effect[used] != EffectId.FUTURE_SIGHT)  # SYSCTL_IGNORE_IMMUNITIES
        & ~heal_path  # SYSCTL_IGNORE_IMMUNITIES (G-30)
        & (tables.move_effect[used] != EffectId.BEAT_UP)  # effect_script_0154.s:5 (B13-10)
        # A charge turn skips the chart (SYSCTL_FIRST_OF_MULTI_TURN, bcp.c:2356); a charge
        # move's strike runs it under SYSCTL_LAST_OF_MULTI_TURN (damage.charge_strike), G-27.
        & ~charging
        & wonder_guarded(tables, state, side, used, move_type, target_side, charge_move)
    )
    guarded = connected & guard_tested
    # subscript_missed.s:107-110 "{0} avoided damage by using {1}!" (TAG_NICKNAME_ABILITY,
    # BTLSCR_DEFENDER: BattleAI_SetAbility, battle_script.c:11798-11806) unless a flag it tests
    # first holds: lost focus (:5), protected (:18) or failed (:97) (``failed`` carries the
    # first and the script failures); a miss reaches it (ruling G-20), a semi-invulnerable miss
    # does not (:11 -> :157, the miss message), ruling G-28.
    state = reveal_ability(
        state, target_side, target, DAMAGE_AVOIDED, guard_tested & ~blocked & ~failed & ~semi_miss
    )
    # Water Absorb (:3507-3512) is Dry Skin's test; Volt Absorb (:3500-3505) takes an Electric
    # move whatever its power (Thunder Wave too), attacker != defender (B9-6).
    seen = ignorable(state, side, target_side)
    water = (
        ((seen == AbilityId.DRY_SKIN) | (seen == AbilityId.WATER_ABSORB))
        & (move_type == Type.WATER)
        & (tables.move_power[used] != 0)
    )
    volt = (seen == AbilityId.VOLT_ABSORB) & (move_type == Type.ELECTRIC)
    # Volt Absorb tests ``attacker != defender`` (:3500-3502), Water Absorb and Dry Skin do not
    # (:3507-3512, :3538-3544): a Dive that hit its own user can heal it.
    absorbed = (
        connected
        & ~self_defender(tables, used, ghost_user(state, side))
        & (water | (volt & ~self_target))
        & (chart > 0)
    )
    target_hp = state.hp[party(fmt, target_side), target].astype(_I32)
    target_max = state.max_hp[party(fmt, target_side), target].astype(_I32)
    healed = jnp.minimum(target_hp + jnp.maximum(target_max // 4, 1), target_max)
    state = _set_mon(state, "hp", target_side, target, healed, absorbed & (target_hp != target_max))
    state = reveal_ability(state, target_side, target, ABSORB, absorbed)
    no_more_work = no_more_work | absorbed
    # Jump Kick / Hi Jump Kick (EffectId.CRASH, B10-2): effect_script_0045.s set
    # SYSCTL_CRASH_DAMAGE before CalcCrit / CalcDamage, so a move that ran its script and did
    # not connect (missed, protected) ends in subscript_missed.s, whose last call (:98) is
    # subscript_crash_on_miss.s: without Magic Guard (CheckAbility) the user loses through
    # UPDATE_HP (SYSCTL_SKIP_SPRITE_BLINK: no resist berry) Divide(damage, 2), capped at
    # Divide(the defender's maxHP, 2) (:11-19; battleCtx->damage is negative there, so the cap
    # is a lower bound). battleCtx->damage is CalcDamage's with hit 0's crit and roll, through
    # the type chart of TRY_MOVE_STATE_CHECK_TYPE_CHART (run hit or miss, bcp.c:3271-3277: an
    # immune target gives 0, which loses nothing). An immune target that connected deals 0
    # damage and crashes for 0 too.
    # A Wonder Guard block ends there too (MOVE_STATUS_WONDER_GUARD), with battleCtx->damage as
    # ApplyTypeChart leaves it: the chart applied, not zeroed (ruling G-19).
    crash = can_act & has_target & (~connected | guarded)
    crash = crash & (tables.move_effect[used] == EffectId.CRASH)
    crash = crash & (battler_ability(state, side, user) != AbilityId.MAGIC_GUARD)
    crash_crit = crit_lands(tables, state, side, used, outcome.crit[side, 0]) & ~crit_blocked(
        state, side, target_side
    )
    would = damage_for_roll(
        tables,
        state,
        side,
        used,
        crash_crit,
        outcome.roll[side, 0],
        wonder_guard=False,
        def_side=target_side,
    )
    cap = divide(state.max_hp[party(fmt, target_side), target].astype(_I32), 2)
    crash_loss = jnp.where(would > 0, jnp.minimum(divide(would, 2), cap), 0)
    state = hp_update(state, side, user, -crash_loss, crash)  # :19 UPDATE_HP (B13-17)
    # A direct sleep (a STATUS_FOE sleep move) on a living target that is not asleep, has no
    # Insomnia as the attacker sees it and meets FIELD_CONDITION_UPROAR: subscript_fall_asleep.s
    # :43-45 -> :_292 prints "...kept awake" and sets MOVE_STATUS_NO_MORE_WORK (:_330): no
    # AfterMoveEffects (effects._inflict blocks the sleep).
    # Psycho Shift's sleep (subscript_give_target_own_status.s:5-6 calls the same subscript as a
    # direct effect; CheckCanShareStatus passed: the defender has no status nor substitute), B12-17.
    shift_sleep = (
        (tables.move_effect[used] == EffectId.PSYCHO_SHIFT)
        & (state.status[party(fmt, side), user] == Status.SLEEP)
        & ~psycho_shift_fails(state, side, target_side)
    )
    kept_awake = (
        connected
        & (
            (
                (tables.move_effect[used] == EffectId.STATUS_FOE)
                & (tables.move_param[used, 0] == Status.SLEEP)
            )
            | shift_sleep
        )
        & (state.hp[party(fmt, target_side), target] > 0)
        & ~sleepless(ignorable(state, side, target_side))  # :23-24 Vital Spirit too (G-47)
        & (state.status[party(fmt, target_side), target] != Status.SLEEP)
        # :40 CheckSubstitute comes first: behind a substitute it fails instead (B10-4).
        & (state.substitute_hp[target_side] == 0)
        & uproar_on(state)
    )
    no_more_work = no_more_work | kept_awake
    # moveFailFlags (BattleContext_MoveFailed, battle_lib.c:2808-2820), which only the Uproar check
    # of the turn's end reads (bcp.c:1525), set by this action: flinch, Disable, Taunt, the
    # confusion self-hit, full paralysis, love (bcp.c:2531, :2545, :2560, :2628, :2656, :2681) and
    # noEffect: CheckTypeChart (bcp.c:2350-2370) found the move ineffective, run once the move
    # passed the no-target check (TRY_MOVE_STATE_CHECK_TYPE_CHART, bcp.c:3271-3277, hit, miss or
    # Protect alike) for a move with power that is not RANGE_USER / RANGE_USER_SIDE. Sleep,
    # freeze and Truant set none.
    no_effect = (
        can_act
        & has_target
        & ~(invoker & call_failed)
        & (tables.move_power[used] != 0)
        & ~targets_user(tables, used, ghost_user(state, side))
        & (tables.move_effect[used] != EffectId.FUTURE_SIGHT)  # SYSCTL_IGNORE_IMMUNITIES
        & ~heal_path  # SYSCTL_IGNORE_IMMUNITIES (G-30)
        & (tables.move_effect[used] != EffectId.BEAT_UP)  # effect_script_0154.s:5 (B13-10)
        & ~charging  # SYSCTL_FIRST_OF_MULTI_TURN (bcp.c:2356), B10-12
        & (chart == 0)
    )
    move_failed = (
        flinched
        | disabled
        | taunted
        | imprisoned
        | gravity_ban
        | self_hit
        | full_para
        | immobilized
        | no_effect
    )
    # A weather move that failed (effects.weather_fails, on the field before its script):
    # MOVE_STATUS_FAILED, subscript_missed.s, no AfterMoveEffects.
    weather_failed = (
        connected
        & (tables.move_effect[used] == EffectId.WEATHER)
        & weather_fails(state, tables.move_param[used].astype(_I32))
    )

    # Baton Pass (effect_script_0127.s: a direct ON_HIT effect at BATTLE_CONTROL_PRIMARY_EFFECT,
    # bcp.c:3298-3308, for a move that connected): subscript_baton_pass.s:5 fails without a
    # replacement (MOVE_STATUS_FAILED, :19-20: subscript_missed.s, no AfterMoveEffects); else
    # Natural Cure (:7-8) and the party list (:17): the action stops (``ctl["pause"]``).
    baton = connected & (tables.move_effect[used] == EffectId.BATON_PASS)
    baton_failed = baton & ~_has_bench(state, side)
    baton_go = baton & ~baton_failed & head_on
    # The head ends here in the tail modes: their state is the one given (MODE_RESUME: after the
    # list's switch-in, hazards and faint check), their inputs the scratch.
    state = _where(head_on, state, state_in)
    used_t = jnp.where(head_on, used, state_in.ts_used.astype(_I32))
    # MODE_PURSUIT: the U-turn user survived the Pursuit (:55): Natural Cure and the list; else the
    # action goes on without a switch.
    pursuit_pause = after_pursuit & (state_in.hp[party(fmt, side), user] > 0) & (state_in.over == 0)
    stops = baton_go | pursuit_pause
    state = _natural_cure(state, side, stops)
    # MODE_RESUME: the rest of AfterMoveMessage on the newcomer (bcp.c:3497-3530): King's Rock with
    # its item (CheckExtraFlinch, bcp.c:4630-4650, no U-turn guard; U-turn has
    # MOVE_FLAG_TRIGGERS_KINGS_ROCK, res/moves/u_turn/data.json:25; flinch_item[side, 0]), then
    # AfterMoveEffects' TRIGGER_SWITCH_IN_EFFECTS (bcp.c:3672-3682): the newcomer's switch-in
    # effects, at the site Roar's drag shares.
    rock = resumed & (state_in.ts_damaged != 0) & (outcome.flinch_item[side, 0] != 0)
    state = kings_rock_flinch(tables, state, side, target_side, used_t, rock & ~tgt_moved)
    items_full = connected & ~no_more_work & ~protect_failed & ~roar_failed & ~weather_failed
    items_full = items_full & ~baton_failed & ~guarded
    # AFTER_MOVE_EFFECT_SYNCHRONIZE_STATUS (bcp.c:3665-3670) comes before the switch-in effects
    # (:3672): a Roar's AfterMoveEffects, or the resumed tail's. The Synchronize flag is the
    # newcomer's Toxic Spikes poison (the U-turn subscript cleared it before the list,
    # subscript_attack_then_switch_out.s:69), its sideEffectMon the newcomer; on the resumed
    # tail CheckExtraFlinch's King's Rock then makes the defender the sideEffectMon
    # (bcp.c:4643, before subscript_flinch_mon.s's own tests). The defender is the user for a
    # self-defender move (Baton Pass: the pass then aims at the newcomer itself, already
    # poisoned). Rulings G-12, G-13.
    sync_defender = jnp.where(
        self_defender(tables, used_t, ghost_user(state, side)), side, target_side
    )
    resumed_rock = (
        rock
        & (tables.item_effect[held_item(state, side)] == ItemId.KINGS_ROCK)
        & ((tables.move_flags[used_t] & MoveFlag.KINGS_ROCK) != 0)
        & (state.hp[party(fmt, target_side), state.active[target_side]] > 0)
    )
    # MODE_PURSUIT with the user fainted (ruling G-24): the U-turn script ended at :55 before
    # its flag clear (:69), so AfterMoveEffects' Synchronize step reads the head's flag as the
    # Pursuit's contact ability toggled it, after CheckExtraFlinch made the defender the
    # sideEffectMon (bcp.c:3524, :4636-4643: the fainted user's King's Rock, its roll, the
    # flag, a damaged defender with HP). A fainted passer passes its BattleMon.status (``held``).
    if pursuit_sync is None:
        pursuit_sync = (False, target_side, None)
    p_flag, p_sem, p_held = pursuit_sync
    tail_fainted = after_pursuit & ~pursuit_pause
    pursuit_rock = (
        tail_fainted
        & (state_in.ts_damaged != 0)
        & (outcome.flinch_item[side, 0] != 0)
        & (tables.item_effect[held_item(state, side)] == ItemId.KINGS_ROCK)
        & ((tables.move_flags[used_t] & MoveFlag.KINGS_ROCK) != 0)
        & (state.hp[party(fmt, target_side), state.active[target_side]] > 0)
    )
    p_sem = jnp.where(pursuit_rock, target_side, jnp.asarray(p_sem, _I32))
    sync_sem = jnp.where(
        resumed,
        jnp.where(resumed_rock, target_side, side),
        jnp.where(after_pursuit, p_sem, target_side),
    ).astype(_I32)
    sync_flag = jnp.where(
        resumed,
        jnp.asarray(entry_poisoned, bool),
        jnp.where(after_pursuit, jnp.asarray(p_flag, bool), roar_poisoned),
    )
    held = state.status[party(fmt, jnp.arange(fmt.B)), state.active].astype(_I32)
    if entry_status is not None:
        held = jnp.where(resumed, held.at[side].set(jnp.asarray(entry_status, _I32)), held)
    if p_held is not None:
        held = jnp.where(after_pursuit, jnp.asarray(p_held, _I32), held)
    state, _, _ = synchronize(
        tables,
        state,
        side,
        sync_defender,
        sync_flag,
        sync_sem,
        (dragged & head_on & items_full) | resumed | tail_fainted,
        held,
    )
    fx_side = jnp.where(resumed, side, target_side)
    entering = (dragged & head_on) | resumed
    state = _where(entering, switch_in_effects(tables, state, fx_side, outcome.trace_pick), state)
    # Transform (:5350-5362 cleared the announce flags): AfterMoveEffects' TRIGGER_SWITCH_IN_EFFECTS
    # runs the transformed battler's copied ability again once the copy is made (the effect runs in
    # the hits below), G-69.
    transform_ok = (
        connected
        & head_on
        & (tables.move_effect[used] == EffectId.TRANSFORM)
        & (state.hp[party(fmt, target_side), target] > 0)
        & (state.semi_invulnerable[target_side] == 0)
        & (state.transformed[target_side] == 0)
    )
    # U-turn's Pursuit (:54): the foe has not moved and can Pursuit now (``_pursuer_ready``).
    if pursuit is None:
        pursuit = (jnp.int32(0), jnp.int32(0), jnp.bool_(True))
    foe_action, foe_move, foe_struggling = pursuit
    is_uturn = tables.move_effect[used] == EffectId.U_TURN
    # The tail modes' items (AfterMoveEffects): on the newcomer, or a U-turn user that died to the
    # Pursuit (its continuation without a switch).
    tail_items = resumed | (after_pursuit & ~pursuit_pause)

    # The script powers read from the action, not the hit's state (CalcMoveDamage's inPower):
    # Gyro Ball's monSpeedValues come from BranchActions' speed sort at the action's start
    # (bcp.c:888, battle_lib.c:1372-1373), before the status checks (a Quick Feet user's wake-up
    # or thaw) and the Custap Berry (Unburden), B13-14; Trump Card reads ppCur of moveSlot (the
    # invoker's slot for a called move) after the PP decrement above (battle_script.c:7237),
    # B13-15.
    used_effect = tables.move_effect[used]
    in_power = jnp.where(
        used_effect == EffectId.GYRO_BALL,
        gyro_ball_speeds_power(tables, state_in, side, target_side),
        jnp.where(
            used_effect == EffectId.TRUMP_CARD,
            trump_card_power(state.pp[party(fmt, side), user, slot]),
            0,
        ),
    )

    # A loop, not unrolled: one copy of the effect code in the program, not MAX_HITS
    # (unrolled, it dominated trace and compile time). It also counts the hits that
    # landed ("Hit N times", battle_controller_player.c:3777-3812) and their crits.
    script_calc = head_on & can_act & has_target & ~failed & ~charging

    def one_hit(hit_index, carry):
        state, hits, crits, contact, thaw, dealt, armed, struck, shell, barbed, record = carry[:11]
        # sub_hit: the defender's SELF_TURN_FLAG_SUBSTITUTE_HIT, set by an earlier hit of this
        # action (MoveHit.sub_hit).
        paused, pursue, damaged0, dealt0, sub_hit = carry[11:16]
        # SYSCTL_TRY_SYNCHRONIZE_STATUS and sideEffectMon through the action's hits (cleared by
        # BattleContext_Init at the action's end, battle_lib.c:2009); ``disrupted``: a later
        # Triple Kick hit missed (MOVE_STATUS_MULTI_HIT_DISRUPTED).
        sync_flag, sync_sem, disrupted = carry[16:]
        stopped = paused | pursue
        # BattleControllerPlayer_LoopMultiHit (bcp.c:3780) stops once a battler fainted
        # (faintedMon): the target, or the attacker by Rough Skin (Selfdestruct's user is
        # at 0 HP before its only hit, so the first hit is exempt).
        going = connected & ~absorbed & ((hit_index == 0) | (state.hp[party(fmt, side), user] > 0))
        going = going & head_on & ~stopped & ~disrupted
        # Triple Kick (B13-6): SYSCTL_TRIPLE_KICK has no SYSCTL_SKIP_ACCURACY_CHECK, so every
        # later hit's TryMove rolls CheckMoveHitAccuracy again with this moment's state
        # (bcp.c:3255-3259, its roll outcome.hit[side, hit]) but skips CheckMoveHitOverrides
        # (_SKIP_ACCURACY_OVERRIDES, :3264-3268: no No Guard, Lock-On or weather sure hit). A
        # miss is MOVE_STATUS_MULTI_HIT_DISRUPTED (CheckMoveFailure, :3320-3323): no hit, no
        # miss message, AfterMoveMessage and AfterMoveEffects as after a hit (the items below,
        # whose repeat finds what the previous pass found), then LoopMultiHit ends (:3782).
        # The disrupted hit's own CalcCrit can still print "A critical hit!" (MULTI_HIT_CRITICAL,
        # :3538-3541; subscript_critical_hit.s:5): not observed (``_hit_events``' coarsening).
        later_roll = outcome.hit[side, hit_index].astype(_I32)
        later_ok = (later_roll > 0) & (
            later_roll
            <= live_hit_threshold(
                tables, state, side, used, tgt_moved, overrides=False, target=acc_target
            )
        )
        kick_check = (
            (tables.move_effect[used] == EffectId.TRIPLE_KICK)
            & (hit_index > 0)
            & (hit_index < n_hits)
            & (state.hp[party(fmt, target_side), target] > 0)
        )
        kick_miss = going & kick_check & ~later_ok
        # That check also clears the attacker's Micle flag (bcp.c:2967-2970, G-59): a berry
        # eaten since the first hit's check boosts this hit's roll only.
        state = _set_side(state, "micle", side, 0, going & kick_check)
        going = going & ~kick_miss
        landed = (
            going
            & (hit_index < n_hits)
            & (
                (state.hp[party(fmt, target_side), target] > 0)
                | defender_always(tables, used, ghost_user(state, side))
            )
        )
        hit_state, info = play_hit(
            tables,
            state,
            side,
            used,
            outcome,
            hit_index,
            connected=going,
            target_moved=tgt_moved,
            struggling=struggling,
            focus_armed=armed,
            # moveEffectsData.custapBerry, cleared when its holder's move starts
            # (subscript_check_quick_claw.s:26-27, above), B13-31.
            foe_custap=custap_target & ~tgt_moved,
            # moveEffectsData.quickClaw (order.claw_flags), cleared once its holder moved.
            foe_claw=claw_target & ~tgt_moved,
            counter_damage=counter_damage,
            extra_flinch=~is_uturn,  # U-turn's check comes after its secondary effect (below)
            sub_hit=sub_hit,
            foe_enduring=foe_enduring,
            in_power=in_power,
            gift=gift_removed,
            target_side=target_side,
            defender=resolved.defender,
            # The script ran (its CalcDamage before TryMove's checks, bcp.c:3214-3216): Me First's
            # counter update for hit 0 hit or miss (battle_script.c:1356-1363).
            script_calc=script_calc,
        )
        first = hit_index == 0
        # U-turn's secondary effect (ONE_HIT_TRIGGER_SECONDARY, bcp.c:3484-3494: an ON_HIT effect
        # of a hit without MOVE_STATUS_DID_NOT_HIT / NO_EFFECTS, battle_lib.c:1524-1529):
        # subscript_attack_then_switch_out.s:5-6, then the contact ability (:7, play_hit's), the
        # pivot item (:11) and Destiny Bond (:15-34: the defender fainted holding it takes the
        # attacker down, before the post-hit check of LOOP_FAINTED).
        tries = (
            going
            & first
            & is_uturn
            & info.effective
            & ~_blacked_out(hit_state, target_side)
            & _has_bench(hit_state, side)
        )
        hit_state = pivot_item(
            tables,
            hit_state,
            side,
            target_side,
            used,
            info.shell_bell,
            tries & info.update_hp,
            tries & info.damaged,
        )
        fell = (state.hp[party(fmt, target_side), target] > 0) & (
            hit_state.hp[party(fmt, target_side), target] == 0
        )
        bonded = (
            tries
            & fell
            & (hit_state.destiny_bond[target_side] != 0)
            & (hit_state.hp[party(fmt, side), user] > 0)
        )
        hit_state = _set_mon(hit_state, "hp", side, user, 0, bonded)
        # :48: a user that fainted does not switch; :54 its foe's Pursuit, else the list.
        goes = tries & (hit_state.hp[party(fmt, side), user] > 0)
        foe = target_side
        chases = (
            goes
            & ~tgt_moved
            & _pursuer_ready(tables, hit_state, foe, foe_action, foe_move, foe_struggling)
        )
        lists = goes & ~chases
        hit_state = _natural_cure(hit_state, side, lists)
        # A U-turn that does not switch makes its King's Rock check after the secondary effect
        # (ONE_HIT_EXTRA_FLINCH, bcp.c:3524).
        rock_now = going & first & is_uturn & ~goes & landed & info.damaged
        rock_now = rock_now & (outcome.flinch_item[side, 0] != 0) & ~tgt_moved
        hit_state = kings_rock_flinch(tables, hit_state, side, target_side, used, rock_now)
        stop_now = lists | chases
        # A move whose script runs no CalcCrit (ids.UNROLLED_EFFECTS): no crit message.
        unrolled = jnp.isin(tables.move_effect[used], jnp.asarray(UNROLLED_EFFECTS, _I32))
        # The crit draw read with the odds at this hit (outcome.crit_lands, battle_lib.c:7105-7132).
        crit = crit_lands(tables, state, side, used, outcome.crit[side, hit_index])
        crit = crit & ~crit_blocked(state, side, target_side) & ~unrolled
        seen = _hit_events(state, hit_state, side, landed, crit, target_side)
        # BattleControllerPlayer_AfterMoveEffects (battle_controller_player.c:3638) runs
        # after every hit of a move that connected: AFTER_MOVE_EFFECT_ATTACKER_ITEM
        # (:3684), then _DEFENDER_ITEM (:3691). A disrupted move, a confusion self-hit,
        # a miss (MOVE_STATUS_DID_NOT_HIT, :3325) or a move with no target skips it. A
        # Pluck / Bug Bite that left SELF_TURN_FLAG_PLUCK_BERRY set (info.pluck_flag) keeps
        # the next item that triggers (its PLUCK_CHECK clears the flag instead of removing
        # the item): the attacker's, as the defender's berry is the one just eaten. Then
        # AFTER_MOVE_EFFECT_HELD_ITEM_STATUS (:3730) re-checks both battlers. What runs in
        # between (the Sticky Barb transfer and the defender's thaw, below) changes no item's
        # trigger: the barb moves only to an attacker holding nothing, and
        # TriggerHeldItemOnStatus has no case for it; the thaw cures no item's condition. So
        # only an attacker's item kept above can trigger again, and is removed this time. A
        # later hit that did not happen changes no HP, so its check finds what the
        # previous one found.
        # A failed Protect or Roar: subscript_missed.s, no AfterMoveEffects.
        # AFTER_MOVE_EFFECT_TRIGGER_ITEMS_ON_HIT (:3698) then _THAW_DEFENDER (:3708) come
        # between the defender's item and _HELD_ITEM_STATUS: the Sticky Barb transfer
        # (effects.sticky_barb_transfer) on a hit that happened, once the defender took damage
        # this action (struck), and the thaw of a frozen defender a Fire hit damaged
        # (effects.thaw_target; an Aspear / Lum Berry it holds has cured the freeze first).
        items = jnp.where(head_on, items_full & ~stopped & ~stop_now, tail_items & ~stops)
        # The hit's status subscripts toggled SYSCTL_TRY_SYNCHRONIZE_STATUS: the secondary
        # effect's on the defender (AfterMoveMessage's TRIGGER_SECONDARY, bcp.c:3484), then the
        # contact ability's on the attacker (TRIGGER_ABILITY, :3514; BattleSystem_
        # TriggerAbilityOnHit sets sideEffectMon to the attacker, battle_lib.c:4225), each the
        # new sideEffectMon; CheckExtraFlinch (:3524) makes the defender the sideEffectMon once
        # its own tests pass (bcp.c:4636-4643: the attacker's King's Rock, the roll, the move's
        # flag, a damaged defender with HP), before subscript_flinch_mon.s's. Then
        # AfterMoveEffects' Synchronize step (``effects.synchronize``), before the item checks
        # (rulings G-12, G-13). A Roar's and the tails' run above.
        def_landed = sync_landed(state, hit_state, target_side)
        att_landed = sync_landed(state, hit_state, side)
        sync_flag = sync_flag ^ def_landed ^ att_landed
        sync_sem = jnp.where(att_landed, side, jnp.where(def_landed, target_side, sync_sem))
        attacker_rock = tables.item_effect[held_item(hit_state, side)] == ItemId.KINGS_ROCK
        rock_test = (
            attacker_rock
            & (outcome.flinch_item[side, hit_index] != 0)
            & ((tables.move_flags[used] & MoveFlag.KINGS_ROCK) != 0)
            & (hit_state.hp[party(fmt, target_side), target] > 0)
        )
        rock_sem = (landed & info.damaged & ~is_uturn) | (
            going & first & is_uturn & ~goes & landed & info.damaged
        )
        sync_sem = jnp.where(rock_sem & rock_test, target_side, sync_sem).astype(_I32)
        hit_state, sync_flag, sync_sem = synchronize(
            tables,
            hit_state,
            side,
            sync_defender,
            sync_flag,
            sync_sem,
            items & landed & head_on & ~dragged,
        )
        state, item_used = held_item_check(tables, hit_state, side, items, info.pluck_flag)
        state = after_damage_items(tables, state, acc_target, items)
        struck = struck | info.update_hp
        # The transfer reads the defender's physical / specialDamageTaken (MoveHit.damaged).
        barbed = barbed | info.damaged
        # TriggerHeldItemOnHit returns early once the substitute was hit this action
        # (battle_lib.c:5378), a later hit on real HP included (B10-4).
        state = sticky_barb_transfer(
            tables, state, side, target_side, used, items & landed & barbed & ~info.sub_hit
        )
        # The same TriggerHeldItemOnHit's berry cases (effects.on_hit_berry, G-54, G-55): the
        # defender's physical / specialDamageTaken are this action's damage of the move's class.
        state, recoiled = on_hit_berry(
            tables,
            state,
            side,
            target_side,
            barbed & (category == PHYSICAL_CATEGORY),
            barbed & (category == SPECIAL_CATEGORY),
            info.super_effective & info.effective,
            items & landed & ~info.sub_hit,
        )
        hit_dealt = jnp.where(head_on, info.dealt, jnp.where(first, state_in.ts_dealt, 0))
        state, thawed = thaw_target(
            tables, state, side, target_side, used_t, hit_dealt, items, gift_removed
        )
        # AFTER_MOVE_EFFECT_HELD_ITEM_STATUS (bcp.c:3730): the attacker's item kept above, or
        # the attacker's item after a Jaboca / Rowap recoil (its HP fell), G-54.
        state = after_damage_items(
            tables, state, side, items & ((item_used & info.pluck_flag) | recoiled)
        )
        return (
            state,
            hits + landed.astype(_I32),
            crits.at[hit_index].set(seen["crit"]),
            contact | seen["contact"],
            thaw | thawed,
            dealt | seen["dealt"],
            info.focus_armed,
            struck,
            shell + info.shell_bell,
            barbed,
            # The defender's turnFlags (bcp.c:3413-3434): each UpdateHP assigns the damage to the
            # move class's field and to lastDamageTaken, and the attacker (1 + battler) to the
            # matching last-attacker field.
            jnp.where(
                info.update_hp,
                jnp.stack(
                    [
                        jnp.where(category == PHYSICAL_CATEGORY, info.taken, record[0]),
                        jnp.where(category == SPECIAL_CATEGORY, info.taken, record[1]),
                        info.taken,
                        jnp.where(category == PHYSICAL_CATEGORY, side + 1, record[3]),
                        jnp.where(category == SPECIAL_CATEGORY, side + 1, record[4]),
                        side + 1,
                    ]
                ),
                record,
            ),
            paused | lists,
            pursue | chases,
            jnp.where(first, info.damaged, damaged0),
            jnp.where(first, info.dealt, dealt0),
            info.sub_hit,
            sync_flag,
            sync_sem,
            disrupted | kick_miss,
        )

    init = (
        state,
        _I32(0),
        jnp.zeros((MAX_HITS,), bool),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        jnp.bool_(False),
        _I32(0),
        jnp.bool_(False),
        jnp.zeros((6,), _I32),
        stops,
        jnp.bool_(False),
        jnp.bool_(False),
        _I32(0),
        jnp.bool_(False),
        jnp.bool_(False),  # SYSCTL_TRY_SYNCHRONIZE_STATUS (BattleContext_Init clears it)
        jnp.asarray(target_side, _I32),  # sideEffectMon (no side effect yet)
        jnp.bool_(False),  # MOVE_STATUS_MULTI_HIT_DISRUPTED (B13-6)
    )
    category = tables.move_category[used]  # CURRENT_MOVE_DATA.class (moveCur)
    target_hp0 = state.hp[party(fmt, target_side), target]
    looped = jax.lax.fori_loop(0, MAX_HITS, one_hit, init)
    state, hits, crits, contact, thaw, dealt, _, struck, shell, _, damage_taken = looped[:11]
    paused, pursue, damaged0, dealt0 = looped[11:15]
    # Where the action stops (a pause at the list, or U-turn's Pursuit next), the tail waits:
    # the paused action's inputs go to the scratch (``ts_used``, ``ts_damaged``, ``ts_dealt``).
    stopped = paused | pursue
    tail = ~stopped
    note = stopped & head_on
    state = state._replace(
        ts_used=jnp.where(note, used, state.ts_used).astype(state.ts_used.dtype),
        ts_damaged=jnp.where(note, damaged0, state.ts_damaged).astype(state.ts_damaged.dtype),
        ts_dealt=jnp.where(note, dealt0, state.ts_dealt).astype(state.ts_dealt.dtype),
        pivot_battler=jnp.where(paused, side + 1, state.pivot_battler).astype(
            state.pivot_battler.dtype
        ),
    )
    state = _where(transform_ok, switch_in_effects(tables, state, side, outcome.trace_pick), state)
    # The roosting chart ends with the move: the target's own types again (if it is still the
    # active battler: a drag restored the outgoing one's from its personal row).
    still = state.active[target_side] == target
    back = jnp.where(still, target_types, state.types[party(fmt, target_side), target])
    state = state._replace(types=state.types.at[party(fmt, target_side), target].set(back))
    # BATTLE_CONTROL_LOOP_FAINTED (bcp.c:3820-3823, :4425-4426) after the hits:
    # subscript_faint_check_destiny_bond.s takes a living attacker of the other side down with a
    # defender that fainted holding Destiny Bond (:5-7, its HP to 0 through UPDATE_HP), before
    # TRIGGER_AFTER_HIT_EFFECTS (Shell Bell, Life Orb need HP).
    fell = (target_hp0 > 0) & (state.hp[party(fmt, target_side), target] == 0)
    bonded = fell & (state.destiny_bond[target_side] != 0) & (state.hp[party(fmt, side), user] > 0)
    state = _set_mon(state, "hp", side, user, 0, bonded)
    # The unleash script clears the lock (the Bide counter is spent); the move above still
    # saw it locked, so effect_script_0026.s did not restart Bide.
    state = _unlock(state, side, bide_ends & head_on)
    # BattleControllerPlayer_UpdateFlagsWhenHit (battle_controller_player.c:4499-4507),
    # reached on every path once the move was chosen (UPDATE_MOVE_BUFFERS, :3956, or
    # LOOP_SPREAD_MOVES, :3837): moveHit[defender] = the move when the attack message
    # was shown (the move got past the status checks, hit or not; the message sets
    # SYSCTL_ATTACK_MESSAGE_SHOWN, battle_script.c:1432), else MOVE_NONE. A confusion
    # self-hit makes the user the defender (battle_controller_player.c:2629), so the
    # foe's record is untouched. ``_move_end`` keeps this write past the faint check, as
    # the decomp writes it after LOOP_FAINTED.
    # Nature Power's defender is its user until its script re-picks the foe (``has_target``):
    # the foe gets moveCur, the called move, only then (no write when the move was disrupted
    # first or its call found no target: the defender is the user or BATTLER_NONE).
    record = ready & ~self_hit & (foe_defender | (bide_ends & can_act))
    record = jnp.where(nature_power, can_act & has_target, record)
    # An invoker's GoToMoveScript re-picks the defender for the called move: the foe gets moveCur
    # when the called move's defender is the foe (not a self-defender range) and it has one.
    record = jnp.where(
        invoker,
        can_act & ~call_failed & has_target & (called_def != NO_DEFENDER) & (called_def != side),
        record,
    )
    # A special-range Counter's defender is the foe only once its command succeeded.
    record = jnp.where(special_counter, can_act & counter_ok, record)
    # The tail modes: the paused move showed its attack message and its defender is the foe
    # unless it is a self-defender (Baton Pass).
    record = jnp.where(head_on, record, ~self_defender(tables, used_t, ghost_user(state, side)))
    # The charge turn shows no attack message (SYSCTL_SKIP_ATTACK_MESSAGE, and
    # subscript_use_move.s:5 skips subscript_attack_message_and_animation.s after a direct side
    # effect), so SYSCTL_ATTACK_MESSAGE_SHOWN stays clear: moveHit, movePrev and moveProtect get
    # MOVE_NONE (bcp.c:3936-3942, :4499-4508). Skull Bash's indirect Defense +1 runs
    # subscript_update_stat_stage.s, whose PrintAttackMessage (:6-7, shown or not) sets it
    # (battle_script.c:1431-1432), B10-12.
    shown_t = jnp.where(head_on, can_act & ~(charging & ~def_up), True)
    hit_record = jnp.where(record, jnp.where(shown_t, used_t, 0), state.move_hit[target_side])
    hit_record = jnp.where(tail, hit_record, state.move_hit[target_side])
    state = state._replace(
        move_hit=state.move_hit.at[target_side].set(hit_record.astype(state.move_hit.dtype))
    )
    # TRIGGER_AFTER_HIT_EFFECTS (bcp.c:3897-3915, after LOOP_SPREAD_MOVES and
    # FAINT_AFTER_SELFDESTRUCT): AFTER_MOVE_HIT_STATE_SHELL_BELL (:4736-4752) and
    # AFTER_MOVE_HIT_STATE_LIFE_ORB (:4758-4772) need SYSCTL_MOVE_HIT (a hit's UpdateHP,
    # :3441; never the confusion self-hit); Life Orb also a non-status moveCur (``used``).
    # In the tail modes they are guarded (SYSCTL_UTURN_ACTIVE, bcp.c:4739, :4761) or find a
    # fainted holder.
    struck = struck & head_on & tail
    state = shell_bell_heal(tables, state, side, shell, struck)
    life_orb = struck & (tables.move_category[used] != STATUS_CATEGORY)
    state = life_orb_recoil(tables, state, side, life_orb)
    # UPDATE_MOVE_BUFFERS (:3917) after it: the Choice lock, on the battler in the slot (the
    # newcomer after a switch), not for U-turn under SYSCTL_UTURN_ACTIVE (set at the list,
    # subscript_attack_then_switch_out.s:68, so on a resumed U-turn) nor a Baton Pass whose move
    # succeeded (bcp.c:3927-3928, by moveTemp).
    ready_t = jnp.where(head_on, ready, True)
    exempt = ((move == MOVE_U_TURN) & resumed) | (
        (move == MOVE_BATON_PASS) & jnp.where(head_on, succeeded, True)
    )
    # SYSCTL_FIRST_OF_MULTI_TURN updates the Choice locks as the attack message does
    # (bcp.c:3921-3922, :4490-4491), B10-12.
    state = _choice_lock(
        tables,
        state,
        side,
        target_side,
        move,
        ready_t & (shown_t | (charging & head_on)) & tail,
        record,
        exempt,
    )
    # UpdateMoveBuffers writes movePrevByBattler / moveProtect onto the battler in the slot
    # (bcp.c:3934-3944): the newcomer, whose switch-in cleared them (battle_lib.c:2119, :2123).
    # U-turn and Baton Pass are not Protect-chain moves, so its chain stays 0.
    state = _set_last_move(state, side, move, resumed)
    # UpdateMoveBuffers' BattleSystem_VerifyMetronomeCount (bcp.c:3957), G-60: on the head's
    # battler (a tail mode's newcomer had its count and move cleared by the switch). The move's
    # MOVE_STATUS_NO_EFFECTS: CheckMoveFailure (bcp.c:3312-3331) did not reach USE_MOVE (a miss,
    # Protect, a semi-invulnerable foe, a failure, an immunity: no AfterMoveEffects) once the
    # move got past BeforeMove; a status-check disruption or no target sets none of those bits.
    worked = items_full & ~no_effect
    no_effects = succeeded & ~charging & ~worked
    # SYSCTL_LAST_OF_MULTI_TURN: every charge move's strike runs subscript_charge_move_cleanup.s:7
    # (the second turn, a Power Herb, SolarBeam in sun), and subscript_bide_end.s.
    last_of_multi = can_act & ((charge_move & ~charging) | bide_ends)
    state = metronome_verify(
        tables, state, side, move, no_effects, repeated, last_of_multi, head_on & ready
    )
    # UpdateMoveBuffers (bcp.c:3936-3942) and UpdateFlagsWhenHit (:4499-4508), on every path of a
    # battler that acts: the battle's movePrev is moveTemp (``move``: the invoker, Nature Power,
    # Struggle, not the called move) when the attack message showed (``can_act``, as the Choice
    # lock's ``shown``), else MOVE_NONE. Copycat reads it (order.copycat_call). Pursuit's
    # pre-switch hit ends without either (battle_script.c:6931-6935), a switch and a bag item
    # never reach them.
    prev = jnp.where(ready_t, jnp.where(shown_t, move, 0), state.move_prev)
    prev = jnp.where(tail, prev, state.move_prev)
    state = state._replace(move_prev=prev.astype(state.move_prev.dtype))
    own = {
        "act_kind": jnp.where(
            can_act,
            ACT_MOVE,
            jnp.where(
                taunted,
                ACT_TAUNTED,
                jnp.where(
                    disabled,
                    ACT_DISABLED,
                    jnp.where(
                        imprisoned,
                        ACT_IMPRISONED,
                        jnp.where(gravity_ban, ACT_GRAVITY, ACT_NONE),
                    ),
                ),
            ),
        ),
        "act_id": jnp.where(can_act | taunted | disabled | imprisoned | gravity_ban, move, 0),
        "shown": ready,
        # subscript_check_quick_claw.s:5 -> :12-13: a Custap flag plays the held-item animation
        # (bit 0) always, and prints "let it move first!" (:16-19, bit 1) unless
        # waitingBattlers == 1 (:15; bcp.c:881-886, itself counted): at B = 2 the foe has acted
        # (``target_moved``). The same term as the removal above; Quick Claw's bit 0 comes from
        # ``_quick_claw_shown`` (one battler holds one item).
        "quick_claw": jnp.where(
            ready & custap[side],
            1 + 2 * (~jnp.asarray(target_moved, bool)).astype(_I32),
            0,
        ),
        # subscript_missed.s tests PROTECTED before MISSED (:18, :157): a miss into a
        # Protect shows "protected itself"; a failed Fake Out prints "But it failed!" (:58).
        "missed": can_act & has_target & ~hit & ~failed & ~blocked,
        "protected": can_act & has_target & blocked & ~failed,
        "protect_ok": protect_ok,
        "endure_ok": endure_ok,
        "protect_failed": protect_failed,
        "no_target": can_act & ~has_target,
        "hits": hits,
        # subscript_magnitude.s prints "Magnitude N!" through the ON_HIT primary effect
        # (BattleSystem_TriggerPrimaryEffect, battle_lib.c:1481), not run for a move with
        # MOVE_STATUS_DID_NOT_HIT (an immune target): shown when the hit dealt damage.
        "magnitude": jnp.where(
            dealt & (tables.move_effect[used] == EffectId.MAGNITUDE), outcome.magnitude[side], 0
        ),
        "present_heal": present_heal,
        # effect_script_0173.s: "Nature Power turned into MOVE!" (msgMoveTemp) once its script
        # runs: the called move's id, else 0.
        # The invokers' called move likewise (the attack message of the called move).
        "called": jnp.where(can_act & (nature_power | (invoker & ~call_failed)), used, 0),
        # The action's last moveCur: FightCommand's moveTemp (bcp.c:1898), then the called move
        # once GoToMoveScript runs (battle_script.c:2118). The Future Sight hit of the turn's end
        # reads its range (``_future_sight``, B10-14). A tail mode's is the paused action's
        # (``ts_used``).
        "move_cur": jnp.where(
            head_on,
            jnp.where(can_act & (nature_power | (invoker & ~call_failed)), used, move),
            used_t,
        ),
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
        "move_failed": move_failed,
        "disabled_at_move": disabled_at_move,
        "sealed_at_move": sealed_at_move,
        "encored_at_move": encored_at_move,
        "gravity_at_move": gravity_at_move,
    }
    foe = {
        "contact_ability": contact,
        "thawed_by_hit": thaw,
        # subscript_force_target_to_switch_or_flee.s: "{0} was dragged out!"
        "dragged": jnp.where(dragged, drag_slot + 1, 0),
        "phys_taken": damage_taken[0],
        "spec_taken": damage_taken[1],
        "last_taken": damage_taken[2],
        "phys_from": damage_taken[3],
        "spec_from": damage_taken[4],
        "last_from": damage_taken[5],
    }
    # turnFlags.roosting (subscript_roost.s:11-12): set unless "HP is full!" (NO_MORE_WORK).
    own["roosted"] = connected & (tables.move_effect[used] == EffectId.ROOST) & ~no_more_work
    # The head's Synchronize flag and sideEffectMon where it stopped for U-turn's Pursuit: the
    # MODE_PURSUIT tail's Synchronize step reads them (``pursuit_sync``, ruling G-24).
    ctl = {
        "pause": paused,
        "pursue": pursue,
        "sync_flag": looped[16],
        "sync_sem": looped[17],
        "target": jnp.asarray(target_side, _I32),
    }
    return state, own, foe, ctl


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
    fmt = format_of(state)
    slots = jnp.arange(BAG_SLOTS)
    for side in range(fmt.B):
        k = _bag_slot(actions[side])
        alive = jnp.sum(
            ((state.species[party(fmt, side)] != 0) & (state.hp[party(fmt, side)] > 0)).astype(_I32)
        )
        count = state.trainer_item_count[side].astype(_I32)
        considered = (slots > k) & (alive <= count - slots + 1)
        spent = is_item(actions[side]) & ((slots == k) | considered)
        bag = jnp.where(spent, 0, state.trainer_items[side]).astype(state.trainer_items.dtype)
        state = state._replace(trainer_items=state.trainer_items.at[side].set(bag))
    return state


def _use_trainer_item(tables: Tables, state: BattleState, side, item) -> BattleState:
    """``BattleControllerPlayer_ItemCommand`` (battle_controller_player.c:1905-1944): the
    AI's item subscript around ``BattleSystem_UseBagItem`` (``bag.use_bag_item``, E1 ruling
    G-43).

    Rule (recomputed choice): the decomp picks the subscript from ``usedItemType``, set by
    ``TrainerAI_ShouldUseItem`` at command selection; ``use_bag_item`` recomputes the item's
    branch here instead. They agree because ``BattleControllerPlayer_CalcTurnOrder``
    (battle_controller_player.c:707, :756-766) runs every item and switch command before any
    move, so only the other battlers' switches and items run between selection and this
    action, and none of those changes what the branch reads: the user's HP, status,
    confusion, ``fakeOutTurnNumber``, ``totalTurns`` or its side's Mist (the subscript then
    reads the item's params, fixed data)."""
    return use_bag_item(tables, state, side, item, jnp.asarray(item) != 0)


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
    custap,
    foe_roosting=False,
    taken=(0, 0, 0, 0, 0, 0),
    mode=MODE_FULL,
    base=None,
    foe=None,
    claw=(False, False),
    pursuit_sync=None,
    foe_enduring=False,
) -> tuple[BattleState, dict, dict, dict]:
    """``side``'s action: ``(state, own, foe, ctl)`` with the events of ``_use_move``. A
    switch shows the party slot (``ACT_SWITCH``), an item its item id (``ACT_ITEM``).

    ``mode`` ``MODE_RESUME`` (E2 batch 6): the paused U-turn / Baton Pass resumes with
    ``actions[side]`` the switch to the picked slot. The switch branch sends it in
    (subscript_show_party_list.s:9-24: ``SwitchAndUpdateMon``, a Baton Pass keeping its effects,
    the hazards, then a faint check against ``base``, the party bitmask of the faints already
    handled, ``BattleState.ts_alive``), and that state feeds the move branch, which runs the
    action's tail on the newcomer. ``ctl["fbase"]``: the bitmask the action's own faint check
    starts from (after the list's). ``pursuit_sync``: ``_use_move``'s, for ``MODE_PURSUIT``."""
    fmt = format_of(state)
    action = actions[side]
    mode = jnp.asarray(mode, _I32)
    resumed = mode == MODE_RESUME
    baton = resumed & (tables.move_effect[state.ts_used] == EffectId.BATON_PASS)
    switched = _send_out(
        tables,
        state,
        side,
        jnp.clip(action - MOVE_ACTIONS, 0, 5),
        effects=~resumed,
        baton=baton,
        trace_pick=outcome.trace_pick,
    )
    base = _alive_mask(state) if base is None else base
    listed = _faint_check(tables, _with_alive(state, base), switched)
    # The list's Toxic Spikes poisoned the newcomer (subscript_show_party_list.s:22, before its
    # faint check): the resumed tail's Synchronize flag (rulings G-12, G-13).
    newcomer = jnp.clip(action - MOVE_ACTIONS, 0, 5)
    landed = switched.status[party(fmt, side), newcomer]
    entry_poisoned = (
        resumed
        & (state.status[party(fmt, side), newcomer] == Status.NONE)
        & ((landed == Status.POISON) | (landed == Status.TOXIC))
    )
    used = _use_trainer_item(tables, state, side, items[side])
    choice = (choices[0][side], choices[1][side], choices[2][side])
    # U-turn's Pursuit reads the battler it hits (convention 1: passed in by the caller).
    pursuit = (actions[foe], choices[0][foe], choices[1][foe])
    moved, own, foe_events, ctl = _use_move(
        tables,
        _where(resumed, listed, state),
        side,
        action,
        choice,
        outcome,
        target_moved,
        foe_protecting,
        custap,
        foe_roosting,
        taken,
        mode,
        pursuit,
        claw,
        entry_poisoned,
        landed,  # the newcomer's BattleMon.status before the list's faint check (G-24)
        pursuit_sync,
        foe_enduring,
    )
    live = state.over == 0
    switch, item = is_switch(action) & live, is_item(action) & live
    acted = _where(item, used, moved)
    acted = _where(switch & ~resumed, switched, acted)
    is_move = ~is_switch(action) & ~is_item(action)
    kind = jnp.where(switch, ACT_SWITCH, jnp.where(item, ACT_ITEM, own["act_kind"]))
    act_id = jnp.where(switch, action - MOVE_ACTIONS, jnp.where(item, items[side], own["act_id"]))
    own = {name: jnp.where(is_move, value, 0) for name, value in own.items()}
    own.update(act_kind=kind, act_id=act_id, shown=switch | item | (own["shown"] != 0))
    foe_events = {name: jnp.where(is_move, value, 0) for name, value in foe_events.items()}
    # The move branch is evaluated for every action; only a move (or a resumed one) can stop at a
    # party list or a Pursuit. A switch or a bag item never runs a move script (the turn branches on
    # the command picked, bcp.c:893), so the move its slot would hold (action 4+ clipped to slot 3: a Baton Pass or
    # U-turn there) must not pause it: that replayed the switch on the newcomer (ruling B7-12).
    runs_move = (is_move & live) | resumed
    ctl = {
        **ctl,
        "pause": ctl["pause"] & runs_move,
        "pursue": ctl["pursue"] & runs_move,
        "fbase": jnp.where(resumed, _alive_mask(listed), base),
    }
    return acted, own, foe_events, ctl


def _pursuer_ready(tables: Tables, state: BattleState, pursuer, action, move, struggling):
    """``BtlCmd_TryPursuit``'s test of a battler that has not acted yet (battle_script.c:
    6899-6918): ``pursuer`` chose a move (``action``) whose effect is Pursuit (``move``, fixed at
    selection; not Struggle) and is alive, not asleep or frozen and not loafing
    (``truant_loafs``, :6903) now, as the switch or U-turn finds it."""
    fmt = format_of(state)
    mon = state.active[pursuer]
    status = state.status[party(fmt, pursuer), mon]
    return (
        ~truant_loafs(state, pursuer)
        & (state.over == 0)
        & ~is_switch(action)
        & ~is_item(action)
        & ~jnp.asarray(struggling)
        & (tables.move_effect[move] == EffectId.PURSUIT)
        & (state.hp[party(fmt, pursuer), mon] > 0)
        & (status != Status.SLEEP)
        & (status != Status.FREEZE)
    )


def pursuit_fires(tables: Tables, state: BattleState, side, actions, choices) -> jnp.ndarray:
    """``side`` switches and the other side, which has not acted yet, chose Pursuit:
    ``BtlCmd_TryPursuit`` (battle_script.c:6884, called from
    ``subscript_switch_pokemon.s:11`` before the switch; ``_pursuer_ready``)."""
    pursuer = _foe(side)
    ready = _pursuer_ready(
        tables, state, pursuer, actions[pursuer], choices[0][pursuer], choices[1][pursuer]
    )
    return is_switch(actions[side]) & ready


def replacement_power_mul(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int32`` ``battleCtx->powerMul`` the opponent's forced replacement after the turn
    sees (``post_ko_switch_in``'s ``power_mul``): ``PURSUIT_POWER_MUL`` when the player's
    switch met Pursuit (``pursuit_fires`` on the turn-start ``state``, with each side's
    move fixed when chosen), else ``NORMAL_POWER_MUL``. The one source for playouts
    (``playout._play``) and exact expansion (``expand._turn``), which must agree."""
    fmt = format_of(state)
    picks = [selected_move(tables, state, side, actions[side]) for side in range(fmt.B)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    pursued = pursuit_fires(tables, state, 0, actions, choices)
    return jnp.where(pursued, PURSUIT_POWER_MUL, NORMAL_POWER_MUL).astype(_I32)


def _active_hp(fmt: Format, state: BattleState, battler):
    """The HP of ``battler``'s active Pokemon."""
    return state.hp[party(fmt, battler), state.active[battler]]


def _pursuit_hit(
    tables: Tables, state: BattleState, pursuer, switcher, actions, choices, outcome: Outcome
) -> BattleState:
    """The pursuer hits the outgoing Pokemon: ``BtlCmd_TryPursuit`` spends the PP and
    sets ``powerMul = 20``; ``subscript_pursuit.s`` then runs crit, damage and the type
    chart with no accuracy check. The pursuer's own action is over; it wrote
    ``movePrevByBattler[pursuer] = move`` (battle_script.c:6930, ``last_move``) and, holding a
    Choice item, ``choiceLockedMove = move`` (:6943-6950). After the hit and the contact
    ability, the script sets ``SYSCTL_MOVE_HIT`` and runs ``TriggerHeldItemOnPivotMove``
    (subscript_pursuit.s:48-49, :94-95 when the target fainted; skipped for a hit that did
    not hit, :17), after ``TriggerHeldItemOnHit`` (:44, :90: the Sticky Barb transfer,
    ``effects.sticky_barb_transfer``, the defender's physicalDamageTaken being the script's
    damage, :33): Shell Bell's heal of the script's ``shellBellDamageDealt`` (:21-29,
    ``effects.shell_bell_heal``) or Life Orb's recoil (``effects.life_orb_recoil``; one
    holder holds one of them). The switching side chose no FIGHT, so no Custap flag is set.
    ``switcher``: the switching battler (``switchedMon``)."""
    fmt = format_of(state)
    action = actions[pursuer]
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    mon = state.active[pursuer]
    state = state._replace(pp=state.pp.at[party(fmt, pursuer), mon, slot].add(jnp.int8(-1)))
    # battle_script.c:6919-6922: one more while PP is left when the switching battler has
    # Pressure (Battler_Ability, its own: the switch has not happened yet), ruling G-9.
    extra = pressure_holder(state, switcher) & (state.pp[party(fmt, pursuer), mon, slot] > 0)
    state = state._replace(
        pp=state.pp.at[party(fmt, pursuer), mon, slot].add(
            jnp.where(extra, -1, 0).astype(state.pp.dtype)
        )
    )
    move = choices[0][pursuer]
    state = _set_last_move(state, pursuer, move, True)
    state = _set_choice_lock(state, pursuer, move, _holds_choice(tables, state, pursuer))
    # subscript_pursuit.s:11 ApplyTypeEffectiveness: a Wonder Guard block
    # (MOVE_STATUS_WONDER_GUARD, damage.wonder_guarded) goes to subscript_missed.s (:17, :154), whose :107-110 tags the
    # switcher's ability (ruling G-20); the hit then deals nothing (play_hit).
    guarded = wonder_guarded(
        tables, state, pursuer, move, move_type_of(tables, state, pursuer, move), switcher
    )
    state = reveal_ability(state, switcher, state.active[switcher], DAMAGE_AVOIDED, guarded)
    state, info = play_hit(
        tables,
        state,
        pursuer,
        move,
        outcome,
        0,
        connected=True,
        target_moved=False,
        power_mul=PURSUIT_POWER_MUL,
        extra_flinch=False,  # subscript_pursuit.s has no CheckExtraFlinch (bcp.c:3527, :3593)
        pursuit_switch=True,  # its substitute path (:20, :56-73), B10-4
        # subscript_pursuit.s ends in BtlCmd_CheckHoldOnWith1HP, which ignores Endure (G-65).
        target_side=switcher,
        defender=switcher,
    )
    # subscript_pursuit.s:85's Destiny Bond take-down runs inside play_hit (``MoveHit.bonded``),
    # before the contact ability (:86-87) and the items (:90, :94-95).
    state = sticky_barb_transfer(tables, state, pursuer, _foe(pursuer), move, info.damaged)
    # subscript_pursuit.s:32 writes the script's damage (before CheckHoldOnWith1HP) to the
    # defender's physicalDamageTaken whatever the class; its specialDamageTaken stays 0 (G-54).
    # The Jaboca Berry's TriggerHeldItemOnPivotMove case (:48-49) then finds no berry: OnHit
    # removed it, or failed on the same tests.
    state, _ = on_hit_berry(
        tables,
        state,
        pursuer,
        _foe(pursuer),
        info.update_hp,
        False,
        info.super_effective & info.effective,
        info.update_hp,  # the script's normal path (:30-49): not missed, no substitute
    )
    state = shell_bell_heal(tables, state, pursuer, info.pursuit_shell_bell, info.update_hp)
    return life_orb_recoil(tables, state, pursuer, info.update_hp)


def _faint_check(
    tables: Tables, before: BattleState, after: BattleState, entry_turn=None, decide: bool = True
) -> BattleState:
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
    :2174-2176); its Camouflage types are restored (``_restore_types``). Every newly fainted Pokemon also loses its status condition and counter: the
    decomp writes ``status = 0`` for a Pokemon with no HP (battle_controller.c:1482-1486)."""
    fmt = format_of(before)
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
    # ``decide`` False: the faint alone (AnyFainted), no CheckBattleOver after it (the side-
    # condition pass, bcp.c:1749-1751).
    ended = ended & decide
    after = after._replace(
        player_faints=(after.player_faints + player_fainted).astype(after.player_faints.dtype),
        deaths=(after.deaths | fainted[0]).astype(after.deaths.dtype),
        over=jnp.where(ended, 1, after.over).astype(after.over.dtype),
        winner=jnp.where(ended, winner, after.winner).astype(after.winner.dtype),
    )
    # The battle's end: the player's party is what the next battle carries, and it never saw a
    # Mimic copy (``_restore_moves``): every battler of party 0 (ruling D1-3). The opponent's
    # active keeps its battler until the payout has read it (the copied species of a Transform,
    # G-69).
    for b in (b for b in range(fmt.B) if fmt.party_of[b] == 0):
        after = _restore_transform(after, b, ended)
        after = _restore_types(tables, after, b, ended)
        after = _restore_ability(tables, after, b, ended)
        after = _restore_moves(after, b, ended)
    for side in range(fmt.B):
        active_fainted = fainted[party(fmt, side), after.active[side]]
        after = _restore_types(tables, after, side, active_fainted)
        after = _reset_battler(after, side, active_fainted)
        after = _entered(after, side, entry_turn, active_fainted)
        after = _source_leaves(after, side, active_fainted, faint=True)
        # CleanupFaintedMon zeroes the battler's turnFlags (battle_lib.c:2190-2193), B13-17.
        after = after._replace(
            ts_assurance=after.ts_assurance.at[side].set(
                jnp.where(active_fainted, 0, after.ts_assurance[side]).astype(jnp.int8)
            )
        )
    return after


def _move_end(
    tables: Tables,
    state: BattleState,
    acted: BattleState,
    side,
    ran,
    defender,
    fought,
    trace_pick=None,
) -> BattleState:
    """After ``side``'s action and its faint check. The foe keeps the ``move_hit`` that
    ``_use_move`` wrote: ``UpdateFlagsWhenHit`` runs after a KO'd defender's faint
    (LOOP_FAINTED before LOOP_SPREAD_MOVES, battle_controller_player.c:3820-3837), so a
    fainted defender holds the record until its replacement clears it. Then
    ``BattleControllerPlayer_MoveEnd`` clears the actor's own (``ClearFlags``, :4389, via
    :3986) where ``ran``: the action ran (a move, switch or item; neither a fainted
    battler's action nor a pursuer's, which ``TryPursuit`` ends, battle_script.c:6931,
    reaches it) and the battle goes on (``CheckBattleOver`` returns first, :3969). ``trace_pick``:
    ``Outcome.trace_pick``, the coins of a Trace that acts at the move's end. ``defender``: the
    action's defender (``ctl["target"]``); ``fought``: the action was a FIGHT (SF2 (a))."""
    foe = _foe(side)
    move_hit = state.move_hit.at[foe].set(acted.move_hit[foe])
    clear = ran & (state.over == 0)
    move_hit = move_hit.at[side].set(jnp.where(clear, 0, move_hit[side]))
    state = state._replace(move_hit=move_hit.astype(state.move_hit.dtype))
    # SF2 (a): RecoverStatusByAbility for the attacker, then the defender, both before
    # CheckBattleOver (bcp.c:3965-3969), so a battle that ends here cures them too (the cure is
    # UpdateMonData, which copies to the party, battle_script.c:3077-3078). The attacker under
    # ``ran``; while the battle goes on the foe call stands for TriggerEffectOnSwitch's
    # FORBIDDEN_STATUS pass over every battler (bcp.c:3973, lib.c:4135-4148); at the battle's
    # end only the FIGHT's defender (:3966-3967). The switch-MoveEnd case is SF2 (a2).
    ended = ran & (state.over != 0)
    state = recover_status_by_ability(state, side, ran)
    second = jnp.where(ended, defender, foe)
    state = recover_status_by_ability(state, second, clear | (ended & fought))
    # BattleSystem_TriggerEffectOnSwitch over every battler (bcp.c:3973): an entry ability gained
    # mid-battle (Skill Swap, E3-2-5) acts now; a no-op while every flag is set or no holder.
    return move_end_switch_in(tables, state, clear, trace_pick)


def execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 3: both actors in order, each followed by a faint check. A switch facing
    Pursuit is hit first (with its own faint check), and the pursuer does not act again.
    An item action's bag slot is spent first (``_consume_trainer_items``)."""
    actions = jnp.asarray(actions, _I32)
    return _execute_actions(tables, state, actions, outcome, order, jnp.bool_(False), _I32(10))[0]


# The phases of an action in ``_execute_actions``' loop: from its start, after U-turn's Pursuit
# (``_use_move``'s ``MODE_PURSUIT``), resumed after the party list (``MODE_RESUME``).
_START, _PURSUIT, _RESUME = MODE_FULL, MODE_PURSUIT, MODE_RESUME
# The turn-flag events later actors read (``_execute_actions``), kept in the scratch at a pause.
_TAKEN = ("phys_taken", "spec_taken", "last_taken", "phys_from", "spec_from", "last_from")


def _execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order, paused, power_mul
) -> tuple[BattleState, dict, dict]:
    """``execute_actions`` with the turn's events: ``(state, read, shown)``. ``read``: every
    event of the turn so far (later actors read the turn flags in it; on a resume it starts from
    the scratch); ``shown``: this call's own (the observation's; ``pursuit``: a Pursuit hit ran,
    which shows the pursuer's move).

    E2 batch 6: the actions run in a loop over (position, phase). A U-turn that reaches its
    foe's Pursuit (subscript_attack_then_switch_out.s:54) stops after its head; the next pass
    plays the Pursuit at the one Pursuit site (``_pursuit_hit``, a switch's pre-switch hit too),
    then the action's tail (``MODE_PURSUIT``). A pause at the party list ends the loop with
    ``pivot_battler`` set and the scratch written (``ts_*``: the order, the cursor, the picks,
    the turn flags, the bitmask of the faints handled, ``power_mul`` (the search's turn
    ``powerMul``), the AI switch, the list's ``powerMul``). ``paused`` (the state is such a
    pause): the picks, the order and the turn flags come from the scratch, ``actions[pivot]`` is
    the switch to the picked slot, and the loop resumes at the cursor (``MODE_RESUME``)."""
    fmt = format_of(state)
    real = jnp.where(paused, state.ts_actions.astype(_I32), actions)
    pivot = jnp.clip(state.pivot_battler.astype(_I32) - 1, 0, fmt.B - 1)
    acts = jnp.where(paused & (jnp.arange(fmt.B) == pivot), actions, real)
    # Each side's move and bag item are fixed when chosen, from the turn's starting state.
    picks = [selected_move(tables, state, side, real[side]) for side in range(fmt.B)]
    locked = jnp.stack([locked_slot(state, side) >= 0 for side in range(fmt.B)])
    choices = (
        jnp.where(paused, state.ts_move.astype(_I32), jnp.stack([p[0] for p in picks])),
        # A locked battler picks no command, so it never Struggles (bcp.c:335, :433).
        jnp.where(paused, state.ts_struggling != 0, jnp.stack([p[1] for p in picks]) & ~locked),
        jnp.where(paused, state.ts_locked != 0, locked),
    )
    items = jnp.stack([state.trainer_items[side, _bag_slot(real[side])] for side in range(fmt.B)])
    items = jnp.where(is_item(real), items, 0)
    items = jnp.where(paused, state.ts_item, items).astype(_I32)
    # CalcTurnOrder's moveEffectsData.custapBerry, from the state the order was fixed in.
    custap = jnp.where(paused, state.ts_custap != 0, custap_flags(tables, state, real))
    # CalcTurnOrder's moveEffectsData.quickClaw (B10-6), likewise.
    claw = jnp.where(paused, state.ts_claw != 0, claw_flags(tables, state, real))
    consumed = _consume_trainer_items(state, real).trainer_items
    state = state._replace(trainer_items=jnp.where(paused, state.trainer_items, consumed))
    state = state._replace(pivot_battler=jnp.zeros_like(state.pivot_battler))
    read = _empty_events(format_of(state))
    for name, field in (
        ("protect_ok", "ts_protect_ok"),
        ("endure_ok", "ts_endure_ok"),
        ("roosted", "roosting"),
    ):
        read[name] = jnp.where(paused, getattr(state, field).astype(_I32), 0)
    for k, name in enumerate(_TAKEN):
        read[name] = jnp.where(paused, state.ts_taken[:, k].astype(_I32), 0)
    read["dragged"] = jnp.where(paused, state.ts_dragged.astype(_I32), 0)
    read["move_failed"] = jnp.where(paused, state.ts_move_failed.astype(_I32), 0)
    shown = _empty_events(format_of(state))

    def put(events_pair, side, values, cond):
        a, b = events_pair
        return _put(a, side, values, cond), _put(b, side, values, cond)

    def cond(carry):
        state, _, _, position = carry[:4]
        return (position < 2) & (state.pivot_battler == 0)

    def body(carry):
        state, read, shown, position, phase, pursued, base, pick_mul, fired, sync = carry
        side = order[position].astype(_I32)
        second = position == 1  # the first actor's target has not moved yet
        start, resumed, ap = phase == _START, phase == _RESUME, phase == _PURSUIT
        # subscript_switch_pokemon.s:5-6: a first actor's switch gets its Natural Cure before
        # the Pursuit (:11).
        first_switch = start & (position == 0) & is_switch(acts[side]) & (state.over == 0)
        state = _natural_cure(state, side, first_switch)
        # The one Pursuit site: the hit before a first actor's switch, or the one U-turn's
        # head stopped for (``ap``); the pursuer has not acted yet, so it is the second actor.
        pursuer = _foe(side)
        fire = (start & (position == 0) & pursuit_fires(tables, state, side, acts, choices)) | ap
        pre = state
        hit = _pursuit_hit(tables, pre, pursuer, side, acts, choices, outcome)
        crit = crit_lands(tables, pre, pursuer, choices[0][pursuer], outcome.crit[pursuer, 0])
        crit = crit & ~crit_blocked(pre, pursuer, side)
        seen = _hit_events(pre, hit, pursuer, True, crit)
        pursuer_events = {
            "act_kind": ACT_MOVE,
            "act_id": choices[0][pursuer],
            "hits": 1,
            "crit": jnp.zeros((MAX_HITS,), _I32).at[0].set(seen["crit"]),
        }
        read, shown = put((read, shown), pursuer, pursuer_events, fire)
        read, shown = put((read, shown), side, {"contact_ability": seen["contact"]}, fire)
        struck = _where(fire, hit, pre)
        state = _faint_check(tables, pre, struck)
        # U-turn's head stopped with its Synchronize flag and sideEffectMon (``sync``); the
        # Pursuit's contact ability (subscript_pursuit.s:40, :86, TriggerAbilityOnHit: the
        # pursuer becomes the sideEffectMon) toggles the flag, and its faint check zeroes only
        # the party copy of a fainted battler's status (battle_controller.c:1481-1486): the
        # MODE_PURSUIT tail's Synchronize step reads both battlers' BattleMon.status (G-24).
        chased = ap & sync_landed(pre, struck, pursuer)
        sides = jnp.arange(fmt.B)
        pursuit_sync = (
            sync[0] ^ chased,
            jnp.where(chased, pursuer, sync[1]),
            struck.status[party(fmt, sides), struck.active].astype(_I32),
        )
        # subscript_pursuit.s:106-108: Pursuit's own payout (no pending move: the pursuer's
        # action is over, and a switcher's or a U-turn's reads no move list).
        state = pay_exp(tables, pre, state)
        # subscript_pursuit.s:100, :115: the KO of the switcher runs ``fieldConditionsMask -=
        # faintedMon`` (SUB_TO_ZERO, battle_script.c:3282-3287), which the engine cannot hold:
        # ``acid_rain`` records the operand and the state is refused (docs/kaizo FCM.md). The
        # operand is the switcher, unless the pursuer then fell to a recoil (Rough Skin,
        # Aftermath, Jaboca Berry, its Life Orb: battle_lib.c:4257-4269, :4358-4371, :5400-5409,
        # :7357-7374), and not after Destiny Bond, which restores ``faintedMon`` to the switcher
        # (subscript_faint_check_destiny_bond.s:10, :25) and runs first (:85) on the switcher's
        # own flag (not ``_pursuit_hit``'s ``bonded``). A Destiny Bond faint clears the
        # pursuer's Uproar bit before :115 (battle_lib.c:2213), a recoil faint keeps it; the
        # switcher's bit is gone too.
        ko = fire & (_active_hp(fmt, pre, side) > 0) & (_active_hp(fmt, hit, side) == 0)
        bonded_ko = ko & (pre.destiny_bond[side] != 0)
        operand = jnp.where(
            (_active_hp(fmt, hit, pursuer) == 0) & ~bonded_ko, pursuer, side
        ).astype(_I32)
        mask_up = (
            (hit.weather != Weather.NONE)
            | (hit.gravity != 0)
            | (hit.trick_room != 0)
            | jnp.any(
                jnp.stack(
                    [
                        (hit.uproar[b] != 0) & (b != side) & ~(bonded_ko & (b == pursuer))
                        for b in range(fmt.B)
                    ]
                )
            )
        )
        acid = ko & (operand != 0) & mask_up
        state = state._replace(
            acid_rain=jnp.where(acid, operand, state.acid_rain).astype(state.acid_rain.dtype)
        )
        pursued = jnp.where(fire, 1 + pursuer, pursued)
        fired = fired | fire
        # BtlCmd_TryPursuit's powerMul = 20 (battle_script.c:6928) stands at U-turn's list.
        pick_mul = jnp.where(ap, PURSUIT_POWER_MUL, pick_mul)
        # The faints handled so far: the action's faint check and payout start here.
        base = jnp.where(resumed, base, _alive_mask(state))
        # The foe's Protect this turn (turnFlags.protecting): only a first actor's succeeds.
        foe_protecting = read["protect_ok"][_foe(side)] != 0
        foe_enduring = read["endure_ok"][_foe(side)] != 0  # turnFlags.enduring (G-65)
        foe_roosting = read["roosted"][_foe(side)] != 0  # turnFlags.roosting
        # The side's turnFlags damage taken and its attackers (Counter, Metal Burst).
        taken = jnp.stack([read[n][side] for n in _TAKEN])
        acted, own, foe_events, ctl = _act(
            tables,
            state,
            side,
            acts,
            choices,
            items,
            outcome,
            second,
            foe_protecting,
            custap,
            foe_roosting,
            taken,
            phase,
            base,
            pursuer,
            claw=claw,
            pursuit_sync=pursuit_sync,
            foe_enduring=foe_enduring,
        )
        # The action ran: its battler was alive at its start (a resumed or pursued one was).
        ran = (state.over == 0) & ((state.hp[party(fmt, side), state.active[side]] > 0) | ~start)
        # The pursuer does not act again; a Pokemon the first actor's Roar dragged in does not
        # act this turn (UpdateAfterSwitch, battle_lib.c:2053: its action is MOVE_END, so
        # BranchActions runs only MoveEnd for it, bcp.c:893, whose checks repeat the first
        # actor's and change nothing).
        skip = second & start & ((pursued == 1 + side) | (read["dragged"][side] != 0))
        acted = _where(skip, state, acted)
        ran = ran & ~skip
        # B10-6: the player's FIGHT attempt or an HP update clears a knocked item's record.
        fight = ~is_switch(acts[side]) & ~is_item(acts[side])
        acted = _stale_known_item(state, acted, (jnp.arange(fmt.B) == side) & ran & fight)
        # The events of an action are its start's (a U-turn's head) or its resume's; the pass
        # after the Pursuit adds none of its own.
        keep = ~skip & ~ap
        # Only a FIGHT that ran reached FightCommand's moveCur (bcp.c:1898).
        own = {**own, "move_cur": jnp.where(ran, own["move_cur"], 0)}
        read, shown = put((read, shown), side, own, keep)
        # The defender's events: the action's defender (``ctl["target"]``), at B = 2 the foe or
        # the user itself when a lock hit itself.
        foe = ctl["target"]
        merged = [
            {name: jnp.where(v != 0, v, ev[name][foe]) for name, v in foe_events.items()}
            for ev in (read, shown)
        ]
        read = _put(read, foe, merged[0], keep)
        shown = _put(shown, foe, merged[1], keep)
        stop = ctl["pause"] | ctl["pursue"]
        done = _move_end(
            tables,
            _faint_check(tables, _with_alive(acted, ctl["fbase"]), acted),
            acted,
            side,
            ran,
            ctl["target"],
            ~is_switch(real[side]) & ~is_item(real[side]),
            outcome.trace_pick,
        )
        # MoveEnd's AnyExpPayout, after RecoverStatusByAbility and before CheckBattleOver
        # (bcp.c:3965-3969). The player's move is still to run when the opponent acted first
        # and the player chose a move (design §3.3(a), ``learn_pin``).
        pending = (position == 0) & (side == 1) & (real[0] < MOVE_ACTIONS)
        done = pay_exp(tables, _with_alive(acted, base), done, pending, real[0])
        state = _where(stop, acted, done)
        position = jnp.where(stop, position, position + 1)
        phase = jnp.where(ctl["pursue"], _PURSUIT, _START).astype(_I32)
        sync = (
            jnp.where(ctl["pursue"], ctl["sync_flag"], sync[0]),
            jnp.where(ctl["pursue"], ctl["sync_sem"], sync[1]).astype(_I32),
        )
        return state, read, shown, position, phase, pursued, base, pick_mul, fired, sync

    start = (
        state,
        read,
        shown,
        jnp.where(paused, state.ts_cursor.astype(_I32), 0),
        jnp.where(paused, _RESUME, _START).astype(_I32),
        jnp.where(paused, state.ts_pursued.astype(_I32), 0),
        jnp.where(paused, state.ts_alive, _alive_mask(state)),
        jnp.where(paused, state.ts_pick_mul.astype(_I32), NORMAL_POWER_MUL),
        jnp.bool_(False),
        (jnp.bool_(False), _I32(0)),  # the stopped head's Synchronize flag and sideEffectMon
    )
    out = jax.lax.while_loop(cond, body, start)
    state, read, shown, position, _, pursued, base, pick_mul, fired, _ = out
    # At a pause the scratch holds the turn; else every ts_* is 0 (states merge between turns).
    stopped = state.pivot_battler != 0
    # Each battler whose action is still to come (after the paused one in ``order``): doubles
    # section 5 rule 5, a fact about the named battler, not "the second actor" (B10-17).
    to_come = jnp.stack([jnp.any((order == b) & (jnp.arange(2) > position)) for b in range(fmt.B)])
    scratch = {
        "ts_cursor": position,
        # Battler_MovedThisTurn (battle_lib.c:5365-5368) of each battler: the actions before the
        # paused one are over (MoveEnd, bcp.c:3989).
        "ts_acted": jnp.stack(
            [jnp.any((order == b) & (jnp.arange(2) < position)) for b in range(fmt.B)]
        ),
        "ts_order": order,
        "ts_actions": real,
        "ts_move": choices[0],
        "ts_struggling": choices[1],
        "ts_locked": choices[2],
        "ts_item": items,
        # Only an actor still to come keeps its flag: an earlier one ate its Custap Berry when
        # its move started (_use_move).
        "ts_custap": custap & to_come,
        # Likewise: an earlier actor's own move start cleared its quickClaw.
        "ts_claw": claw & to_come,
        "ts_pursued": pursued,
        "ts_protect_ok": read["protect_ok"],
        "ts_endure_ok": read["endure_ok"],
        "ts_taken": jnp.stack([read[n] for n in _TAKEN], axis=1),
        "ts_dragged": read["dragged"],
        "ts_move_failed": read["move_failed"],
        "ts_used": state.ts_used,
        "ts_damaged": state.ts_damaged,
        "ts_dealt": state.ts_dealt,
        "ts_alive": base,
        "ts_power_mul": jnp.where(paused, state.ts_power_mul, power_mul),
        "ts_switched": is_switch(real),
        "ts_pick_mul": pick_mul,
        # turnFlags.assuranceDamageMask (B13-17): the turn's, kept at a pause.
        "ts_assurance": state.ts_assurance,
        # SideConditions.followMe / followMeUser (D2-11): cleared by SetupNextTurn at the turn end
        # (lib.c:2251-2252), kept at a pause.
        "ts_follow_me": state.ts_follow_me,
        # Trace fires of the step (D2-16): 0 once the turn completes, kept at a pause.
        "ts_trace_fires": state.ts_trace_fires,
    }
    if set(scratch) != set(TURN_SCRATCH):
        raise RuntimeError("the pause must write every ts_* field")
    state = state._replace(
        **{
            name: jnp.where(stopped, jnp.asarray(value), 0).astype(getattr(state, name).dtype)
            for name, value in scratch.items()
        },
        # Read when the order was fixed, read again only at the next turn (the resumed half's).
        qc_roll=jnp.where(stopped, 0, state.qc_roll).astype(state.qc_roll.dtype),
        roosting=jnp.where(stopped, roosting_now(state, read["roosted"]), 0).astype(
            state.roosting.dtype
        ),
    )
    return state, read, {**shown, "pursuit": fired.astype(_I32)}


def _hp_change(state: BattleState, side, mon, delta) -> BattleState:
    fmt = format_of(state)
    hp = jnp.clip(
        state.hp[party(fmt, side), mon].astype(_I32) + delta, 0, state.max_hp[party(fmt, side), mon]
    )
    return state._replace(hp=state.hp.at[party(fmt, side), mon].set(hp.astype(state.hp.dtype)))


def _leech_seed_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s seeded active Pokemon with HP left loses ``BattleSystem_Divide(maxHP, 8)``
    (capped at its HP) and the other side's active Pokemon, if alive, gains what it lost
    (``hitDamage``), times ``(100 + 30) / 100`` with a Big Root (``effects.big_root``;
    ``subscript_leech_seed_effect.s``)."""
    fmt = format_of(state)
    other = _foe(side)
    mon, recipient = state.active[side], state.active[other]
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD  # bcp.c:1381
    on = (
        (state.leech_seed[side] != 0)
        & (hp > 0)
        & (state.hp[party(fmt, other), recipient] > 0)
        & ~guarded
    )
    drained = jnp.minimum(jnp.maximum(state.max_hp[party(fmt, side), mon].astype(_I32) // 8, 1), hp)
    drained = jnp.where(on, drained, 0)
    gained = big_root(tables, state, other, drained)
    return _hp_change(_hp_change(state, side, mon, -drained), other, recipient, gained)


def _aqua_ring_check(tables: Tables, state: BattleState, side, field="aqua_ring") -> BattleState:
    """A ringed ``side`` active Pokemon alive and below full HP heals
    ``BattleSystem_Divide(maxHP, 16)``, times ``(100 + 30) / 100`` with a Big Root
    (``effects.big_root``; ``subscript_aqua_ring_heal.s``; Heal Block is unsupported).
    ``field`` ``"ingrain"``: ``MON_COND_CHECK_STATE_INGRAIN`` (bcp.c:1311-1326), the same heal
    through subscript_ingrain_heal.s."""
    fmt = format_of(state)
    mon = state.active[side]
    hp, full = (
        state.hp[party(fmt, side), mon].astype(_I32),
        state.max_hp[party(fmt, side), mon].astype(_I32),
    )
    on = (getattr(state, field)[field_index(state, field, side)] != 0) & (hp > 0) & (hp != full)
    heal = jnp.where(on, big_root(tables, state, side, jnp.maximum(full // 16, 1)), 0)
    return _hp_change(state, side, mon, heal)


def _ingrain_then_aqua_ring(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_INGRAIN`` then ``_AQUA_RING`` (bcp.c:1311, :1332): two heals, so one
    faint check serves both (the one between them finds nothing new)."""
    return _aqua_ring_check(tables, _aqua_ring_check(tables, state, side, "ingrain"), side)


def _status_damage(state: BattleState, side, status, damage) -> BattleState:
    """``side``'s active Pokemon with ``status`` and HP left loses ``damage`` (capped at its
    HP, ``BATTLE_SUBSCRIPT_UPDATE_HP``)."""
    fmt = format_of(state)
    mon = state.active[side]
    # subscript_poison_damage.s:5 / subscript_burn_damage.s:5 skip a Magic Guard holder
    # (CheckAbility); the toxic counter has already risen (bcp.c:1405-1415).
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD
    on = (
        (state.status[party(fmt, side), mon] == status)
        & (state.hp[party(fmt, side), mon] > 0)
        & ~guarded
    )
    # subscript_poison_damage.s:4-12 (poison and toxic): a Poison Heal holder (CheckAbility)
    # below full HP heals Divide(maxHP, 8) instead and tags its ability; at full HP nothing (B9-8).
    poisoned = (status == Status.POISON) | (status == Status.TOXIC)
    max_hp = state.max_hp[party(fmt, side), mon].astype(_I32)
    healer = on & poisoned & (battler_ability(state, side, mon) == AbilityId.POISON_HEAL)
    heals = healer & (state.hp[party(fmt, side), mon].astype(_I32) != max_hp)
    delta = jnp.where(heals, jnp.maximum(max_hp // 8, 1), jnp.where(on & ~healer, -damage, 0))
    state = _hp_change(state, side, mon, delta)
    return reveal_ability(state, side, mon, STATUS_HP, heals)


def _eighth(state: BattleState, side) -> jnp.ndarray:
    """``BattleSystem_Divide(maxHP, 8)`` of ``side``'s active Pokemon: at least 1."""
    fmt = format_of(state)
    return jnp.maximum(state.max_hp[party(fmt, side), state.active[side]].astype(_I32) // 8, 1)


def _poison_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``BattleSystem_Divide(maxHP * -1, 8)``, ``subscript_poison_damage.s`` (Magic Guard
    and Poison Heal are not implemented)."""
    return _status_damage(state, side, Status.POISON, _eighth(state, side))


def _toxic_check(tables: Tables, state: BattleState, side) -> BattleState:
    """The counter rises first (up to ``TOXIC_COUNTER_MAX``), then the Pokemon loses
    ``BattleSystem_Divide(maxHP, 16) * counter``."""
    fmt = format_of(state)
    mon = state.active[side]
    on = (state.status[party(fmt, side), mon] == Status.TOXIC) & (
        state.hp[party(fmt, side), mon] > 0
    )
    counter = jnp.minimum(
        state.status_counter[party(fmt, side), mon].astype(_I32) + 1, TOXIC_COUNTER_MAX
    )
    state = _set_mon(state, "status_counter", side, mon, counter, on)
    sixteenth = jnp.maximum(state.max_hp[party(fmt, side), mon].astype(_I32) // 16, 1)
    return _status_damage(state, side, Status.TOXIC, sixteenth * counter)


def _burn_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``subscript_burn_damage.s``: ``DivideVarByValue`` maxHP by 8, which is
    ``BattleSystem_Divide`` (battle_script.c:4413; Magic Guard and Heatproof are not
    implemented)."""
    return _status_damage(state, side, Status.BURN, _eighth(state, side))


def _curse_check(state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_CURSE`` (bcp.c:1451-1459): a cursed battler with HP left runs
    subscript_curse_damage.s: nothing for its own Magic Guard (``CheckAbility``, :2), else
    ``Divide(maxHP, 4)`` through UPDATE_HP (:5-13, capped at its HP; it may faint)."""
    fmt = format_of(state)
    mon = state.active[side]
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD
    on = (state.curse[side] != 0) & (state.hp[party(fmt, side), mon] > 0) & ~guarded
    loss = jnp.maximum(state.max_hp[party(fmt, side), mon].astype(_I32) // 4, 1)
    return _hp_change(state, side, mon, -jnp.where(on, loss, 0))


def _burn_then_curse(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_BURN`` then (``_NIGHTMARE``, unsupported, between) ``_CURSE``: one
    faint check serves both, as a battler the burn fainted has no HP for the curse and the
    battle-over check between finds what the one after them finds."""
    return _curse_check(_burn_check(tables, state, side), side)


def _bind_check(tables: Tables, state: BattleState, side) -> BattleState:
    """A bound active Pokemon with HP left: the counter drops, and while it stays above
    0 the Pokemon loses ``BattleSystem_Divide(maxHP, 16)``; at 0 Bind ends without damage."""
    fmt = format_of(state)
    mon = state.active[side]
    on = (state.trapped[side] > 0) & (state.hp[party(fmt, side), mon] > 0)
    trapped = state.trapped[side] - on.astype(state.trapped.dtype)
    damage = jnp.maximum(state.max_hp[party(fmt, side), mon].astype(_I32) // 16, 1)
    # subscript_bind_effect.s:5: no damage to a Magic Guard holder (CheckAbility).
    guarded = battler_ability(state, side, mon) == AbilityId.MAGIC_GUARD
    state = _hp_change(state, side, mon, -jnp.where(on & (trapped > 0) & ~guarded, damage, 0))
    return state._replace(
        trapped=state.trapped.at[side].set(trapped),
        trap_move=state.trap_move.at[side].set(jnp.where(trapped > 0, state.trap_move[side], 0)),
    )


def _bad_dreams_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_BAD_DREAMS`` (bcp.c:1484-1500): an asleep battler with HP and no
    Magic Guard (Battler_Ability) facing a living Bad Dreams holder of the other side
    (COUNT_ALIVE_BATTLERS_THEIR_SIDE_FLAG, battle_lib.c:2915-2922) runs subscript_bad_dreams.s:
    per holder Divide(maxHP, 8) through UPDATE_HP, stopping once it faints, each message tagging
    the holder's ability (B9-8). A loop over the foe battlers (in singles the one)."""
    fmt = format_of(state)
    mon = state.active[side]
    max_hp = state.max_hp[party(fmt, side), mon].astype(_I32)
    for foe in range(state.active.shape[0]):
        holder = state.active[foe]
        on = (
            (jnp.asarray(side) != foe)
            & (battler_ability(state, foe, holder) == AbilityId.BAD_DREAMS)
            & (state.hp[party(fmt, foe), holder] > 0)
            & (state.status[party(fmt, side), mon] == Status.SLEEP)
            & (battler_ability(state, side, mon) != AbilityId.MAGIC_GUARD)
            & (state.hp[party(fmt, side), mon] > 0)
        )
        state = _hp_change(state, side, mon, -jnp.where(on, jnp.maximum(max_hp // 8, 1), 0))
        state = reveal_ability(state, foe, holder, TORMENT, on)
    return state


def _bind_then_bad_dreams(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_BIND`` then ``_BAD_DREAMS`` (bcp.c:1462, :1484): one faint check, as a
    battler the bind fainted has no HP for Bad Dreams."""
    return _bad_dreams_check(tables, _bind_check(tables, state, side), side)


def _disable_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_DISABLE`` (battle_controller_player.c:1565): with a move
    disabled, a counter reset to 0 when the Pokemon no longer knows the move, then the
    counter drops while non-zero and, at 0, the move is cleared (``subscript_disable_end.s``).
    No HP condition in the decomp."""
    fmt = format_of(state)
    mon = state.active[side]
    disabled = state.disabled_move[side]
    on = disabled != 0
    known = jnp.any(state.moves[party(fmt, side), mon] == disabled)
    turns = jnp.where(on & ~known, 0, state.disable_turns[side]).astype(_I32)
    expire = on & (turns == 0)
    turns = jnp.where(on & (turns > 0), turns - 1, turns)
    return state._replace(
        disabled_move=state.disabled_move.at[side].set(
            jnp.where(expire, 0, disabled).astype(state.disabled_move.dtype)
        ),
        disable_turns=state.disable_turns.at[side].set(turns.astype(state.disable_turns.dtype)),
    )


def _uproar_check(tables: Tables, state: BattleState, side, failed) -> BattleState:
    """``MON_COND_CHECK_STATE_UPROAR`` (battle_controller_player.c:1505-1541) for ``side`` in an
    uproar: every battler, in battler order, that is asleep with HP left (Soundproof, which is
    spared, is unsupported) wakes (subscript_wake_up.s: every sleep bit cleared, :19), the check
    repeating until none sleeps (``STATE_DONE``, :1518-1521); then the count drops by 1 and the
    uproar ends (subscript_uproar_end.s; the volatile bits and the battler's
    ``FIELD_CONDITION_UPROAR`` bit cleared, :1525-1535) when this turn's ``moveFailFlags`` are
    set (``failed``, ``BattleContext_MoveFailed``, battle_lib.c:2808-2820) or it reached 0. The
    battler is then no longer locked: it was never ``VOLATILE_CONDITION_MOVE_LOCKED``
    (subscript_uproar.s sets only ``moveLockedInto``), so ``Battler_CanPickCommand`` is TRUE
    again and ``locked_move`` goes to 0."""
    fmt = format_of(state)
    on = state.uproar[side] != 0
    for battler in range(fmt.B):
        mon = state.active[battler]
        wake = (
            on
            & (state.status[party(fmt, battler), mon] == Status.SLEEP)
            & (state.hp[party(fmt, battler), mon] > 0)
        )
        state = _set_mon(state, "status", battler, mon, Status.NONE, wake)
        state = _set_mon(state, "status_counter", battler, mon, 0, wake)
    left = state.uproar[side].astype(_I32) - 1
    ends = on & ((jnp.asarray(failed)[side] != 0) | (left <= 0))
    uproar = jnp.where(on, jnp.where(ends, 0, left), state.uproar[side])
    locked = jnp.where(ends, 0, state.locked_move[side])
    return state._replace(
        uproar=state.uproar.at[side].set(uproar.astype(state.uproar.dtype)),
        locked_move=state.locked_move.at[side].set(locked.astype(state.locked_move.dtype)),
    )


def _lock_on_check(state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_LOCK_ON`` (bcp.c:1618-1624): the MOVE_EFFECT_LOCK_ON counter drops
    by 1; lockOnTarget stays (nothing clears it but the battler's own volatiles)."""
    on = state.lock_on[side] != 0
    return _set_side(state, "lock_on", side, state.lock_on[side] - 1, on)


def _stale_known_item(before: BattleState, after: BattleState, fought) -> BattleState:
    """B10-6: Knock Off's BtlCmd_TryKnockOff runs no CopyToParty, so the AI's record of the
    player's knocked item (``ai_item_known``) stays until the battler's next
    ``BattleMon_CopyToParty`` (battle_lib.c:1819-1823), which clears it for a battler holding
    nothing: its next HP-bar update (BtlCmd_UpdateHealthBarValue, battle_script.c:1720) or its
    next move attempt's status checks (bcp.c:2748). A knocked-off slot never holds an item again
    (the mask stops the Sticky Barb, and Thief / Trick are unsupported), so clearing at the end of
    the segment (an action, the end of the turn) in which such an event came is exact for the
    AI, which reads the record only at its next choice. ``before``: the segment's start; the slot
    must have been knocked off by then (the knock-off's own hit updates the HP first). ``fought``
    (``bool[B]``): each battler made a FIGHT attempt in the segment. Every battler is checked
    (doubles design section 5 rule 4); ``forget_item`` keeps the record of the player's only (the
    AI's own items are not in it)."""
    fmt = format_of(before)
    fought = jnp.broadcast_to(jnp.asarray(fought, bool), (fmt.B,))
    for side in range(fmt.B):
        mon = before.active[side]
        stale = (before.knocked_off[party(fmt, side), mon] != 0) & (
            after.item[party(fmt, side), mon] == 0
        )
        event = (after.hp[party(fmt, side), mon] != before.hp[party(fmt, side), mon]) | fought[side]
        after = forget_item(after, side, mon, stale & event)
    return after


def _encore_check(state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_ENCORE`` (bcp.c:1591-1612), B10-9: an encored battler whose moves no
    longer hold the encored move, or whose PP for it is 0, has its count set to 0; a count above 0
    drops by 1, and at 0 the encore ends (encoredMove = 0; subscript_encore_end.s). The slot and
    the count are left as they are (only encoredMove is cleared)."""
    fmt = format_of(state)
    mon = state.active[side]
    move = state.encore_move[side]
    on = move != 0
    held = state.moves[party(fmt, side), mon] == move
    spent = ~jnp.any(held) | jnp.any(held & (state.pp[party(fmt, side), mon] == 0))
    turns = jnp.where(on & spent, 0, state.encore_turns[side]).astype(_I32)
    ends = on & (turns == 0)
    state = _set_side(
        state, "encore_turns", side, jnp.where(on & (turns > 0), turns - 1, turns), on
    )
    return _set_side(state, "encore_move", side, 0, ends)


def _uproar_then_disable(tables: Tables, state: BattleState, side, failed) -> BattleState:
    """``MON_COND_CHECK_STATE_UPROAR``, then (``_THRASH``, unsupported, between)
    ``MON_COND_CHECK_STATE_DISABLE``, ``_ENCORE`` (B10-9) and ``_LOCK_ON``: none changes HP, so
    they share one faint check."""
    state = _disable_check(tables, _uproar_check(tables, state, side, failed), side)
    return _lock_on_check(_encore_check(state, side), side)


def _taunt_check(tables: Tables, state: BattleState, side) -> BattleState:
    """The Taunt counter drops by 1 (no HP condition in the decomp), then
    ``MON_COND_CHECK_STATE_MAGNET_RISE`` (bcp.c:1647-1656, between TAUNT and HEAL_BLOCK): a
    nonzero ``magnetRiseTurns`` drops by 1 and ``subscript_magnet_rise_end.s`` clears the mask
    bit at 0. Turns that Gravity zeroed never reach it: the bit stays (G-66)."""
    taunt = state.taunt[side] - (state.taunt[side] > 0).astype(state.taunt.dtype)
    turns = state.magnet_rise_turns[side]
    left = turns - (turns > 0).astype(turns.dtype)
    ended = (turns > 0) & (left == 0)
    bit = jnp.where(ended, 0, state.magnet_rise[side]).astype(state.magnet_rise.dtype)
    return state._replace(
        taunt=state.taunt.at[side].set(taunt),
        magnet_rise_turns=state.magnet_rise_turns.at[side].set(left),
        magnet_rise=state.magnet_rise.at[side].set(bit),
    )


def _embargo_check(state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_EMBARGO`` (battle_controller_player.c:1671-1677): a nonzero
    embargoTurns drops by 1; at 0 subscript_embargo_end.s clears the MOVE_EFFECT_EMBARGO bit
    (:9) and names the battler only (no HP condition, no AI record), B13-31."""
    embargo = state.embargo[side] - (state.embargo[side] > 0).astype(state.embargo.dtype)
    return state._replace(embargo=state.embargo.at[side].set(embargo))


def _taunt_then_yawn(tables: Tables, state: BattleState, side, outcome: Outcome) -> BattleState:
    """``MON_COND_CHECK_STATE_TAUNT`` with ``_MAGNET_RISE`` (``_taunt_check``), ``_EMBARGO``
    (bcp.c:1671, after the unsupported Heal Block) then ``_YAWN`` (``effects.yawn_sleep``): only
    Yawn's sleep changes a battler, so one faint check."""
    state = _embargo_check(_taunt_check(tables, state, side), side)
    return yawn_sleep(tables, state, side, outcome)


def _leftovers_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_LEFTOVERS`` (battle_controller_player.c:1369):
    ``BattleSystem_TriggerLeftovers`` (battle_lib.c:4922-4965) on a holder with HP left.
    Leftovers (``HOLD_EFFECT_HP_RESTORE_GRADUAL``), and Black Sludge
    (``HOLD_EFFECT_HP_RESTORE_PSN_TYPE``) on a Poison-type holder (``MON_HAS_TYPE``: the
    battler's types), restore ``BattleSystem_Divide(maxHP, 16)`` below full HP
    (``subscript_restore_a_little_hp.s``, capped); Black Sludge on any other holder without
    Magic Guard (read by decomp ability id) takes ``BattleSystem_Divide(maxHP * -1, 8)``
    (``subscript_lose_hp_from_item_with_message.s``: ``SYSCTL_SKIP_SPRITE_BLINK``, capped at
    its HP; it may faint). Neither item is used up."""
    fmt = format_of(state)
    mon = state.active[side]
    item = held_item(state, side, mon)  # Battler_HeldItemEffect (battle_lib.c:4927, B13-29)
    effect = tables.item_effect[item]
    hp, max_hp = (
        state.hp[party(fmt, side), mon].astype(_I32),
        state.max_hp[party(fmt, side), mon].astype(_I32),
    )
    poison = jnp.any(state.types[party(fmt, side), mon] == Type.POISON)
    magic_guard = (
        battler_decomp_ability(tables, state, side, mon) == WEATHER_ABILITY_IDS["Magic Guard"]
    )
    sludge = effect == ItemId.BLACK_SLUDGE
    heal = ((effect == ItemId.LEFTOVERS) | (sludge & poison)) & (hp < max_hp)
    hurt = sludge & ~poison & ~magic_guard
    delta = jnp.where(heal, jnp.maximum(max_hp // 16, 1), 0) - jnp.where(
        hurt, jnp.maximum(max_hp // 8, 1), 0
    )
    # Both subscripts name the item (BTLSCR_MSG_BATTLER_TEMP = the holder, battle_lib.c:4955):
    # the AI learns it (effects.reveal_item).
    state = reveal_item(state, side, mon, (hp > 0) & (heal | hurt))
    return _hp_change(state, side, mon, jnp.where(hp > 0, delta, 0))


def _detrimental_item_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT`` (battle_controller_player.c:1710), the
    last check: ``BattleSystem_TriggerDetrimentalHeldItem`` (battle_lib.c:5307-5350) on a
    holder with HP left.

    * Toxic Orb (``HOLD_EFFECT_PSN_USER``): ``subscript_badly_poison.s`` as a held-item side
      effect (:13-28): nothing for an Immunity holder (its own ability, :14), a Leaf Guard
      holder in sun unless a weather lock (``CheckIgnoreWeather``, :15-17), a holder with any
      status, or a Poison or Steel type (the battler's types, :20-26); else it is badly
      poisoned (counter 0), unless its side's Safeguard is up (:27, B10-7).
    * Flame Orb (``HOLD_EFFECT_BRN_USER``): ``subscript_burn.s`` (:5-17): the same with Water
      Veil for Immunity and Fire for Poison / Steel, and Safeguard (:16); else it is burned.
    * Sticky Barb (``HOLD_EFFECT_DMG_USER_CONTACT_XFR``): a holder without Magic Guard takes
      ``BattleSystem_Divide(maxHP * -1, param)``
      (``subscript_lose_hp_from_item_with_message.s``, capped at its HP; it may faint).

    Immunity, Water Veil, Leaf Guard and Magic Guard are read by decomp ability id
    (``ids.ITEM_ABILITY_IDS``, ``WEATHER_ABILITY_IDS``)."""
    fmt = format_of(state)
    mon = state.active[side]
    item = held_item(state, side, mon)  # Battler_HeldItemEffect (battle_lib.c:4971, B13-29)
    effect = tables.item_effect[item]
    hp, max_hp = (
        state.hp[party(fmt, side), mon].astype(_I32),
        state.max_hp[party(fmt, side), mon].astype(_I32),
    )
    alive = hp > 0
    ability = battler_decomp_ability(tables, state, side, mon)
    sun = (state.weather == Weather.SUN) & ~weather_suppressed(tables, state)
    leaf_guard = sun & (ability == ITEM_ABILITY_IDS["Leaf Guard"])
    toxic = (
        alive
        & (effect == ItemId.TOXIC_ORB)
        & (ability != ITEM_ABILITY_IDS["Immunity"])
        & ~leaf_guard
        & can_have_status(tables, state, side, mon, Status.TOXIC)
        & (state.safeguard[side_of(side)] == 0)  # subscript_badly_poison.s:27 (B10-7)
    )
    burn = (
        alive
        & (effect == ItemId.FLAME_ORB)
        & (ability != ITEM_ABILITY_IDS["Water Veil"])
        & ~leaf_guard
        & can_have_status(tables, state, side, mon, Status.BURN)
        & (state.safeguard[side_of(side)] == 0)  # subscript_burn.s:16 (B10-7)
    )
    status = jnp.where(toxic, Status.TOXIC, Status.BURN)
    state = _set_mon(state, "status", side, mon, status, toxic | burn)
    state = _set_mon(state, "status_counter", side, mon, 0, toxic | burn)
    magic_guard = ability == WEATHER_ABILITY_IDS["Magic Guard"]
    barb = alive & (effect == ItemId.STICKY_BARB) & ~magic_guard
    loss = jnp.maximum(max_hp // tables.item_param[item, 0].astype(_I32), 1)
    # "{0} is hurt by its {1}!" names the barb (BTLSCR_MSG_BATTLER_TEMP, battle_lib.c:5340); the
    # orbs' messages name msgItemTemp (BTLSCR_MSG_TEMP: subscript_badly_poison.s:77,
    # subscript_burn.s:73), which the AI does not record.
    state = reveal_item(state, side, mon, barb)
    return _hp_change(state, side, mon, jnp.where(barb, -loss, 0))


# Two consecutive checks share one faint check where the first can only heal or cure (a held
# item's trigger, ``hooks.held_item_check``): the decomp's faint check between them finds
# nothing new, so one copy of ``_faint_check`` per pair (each copy adds to step's compile time).
def _use_item_then_leftovers(tables: Tables, state: BattleState, side) -> BattleState:
    return _leftovers_check(tables, after_damage_items(tables, state, side), side)


def _item_condition_then_detrimental(tables: Tables, state: BattleState, side) -> BattleState:
    return _detrimental_item_check(tables, after_damage_items(tables, state, side), side)


def _turn_end_ability(tables: Tables, state: BattleState, side, outcome: Outcome) -> BattleState:
    """``MON_COND_CHECK_STATE_ABILITY`` (battle_controller_player.c:1355), the first
    end-of-turn check, ``BattleSystem_TriggerTurnEndAbility`` (battle_lib.c:3550-3599): Speed
    Boost (:3555-3564, ruling G-48) raises the Speed of a holder with HP left, below +6
    (``MAX_STAT_STAGE``), that did not enter this turn (``fakeOutTurnNumber != totalTurns + 1``:
    ``fake_out_turn``, ``turn``) by 1 through ``subscript_update_stat_stage.s`` as an ability
    side effect (its "raised its" message is built in C, ``SetupNicknameAbilityStatMsg``,
    battle_script.c:2854: no reveal); Shed Skin (:3567-3590) cures any status of a holder with
    HP left on ``RandNext() % 10 < 3`` (``outcome.shed_skin[side]``); the sleep counter and the
    poison counter reset, and ``subscript_ability_restore_status.s:8`` tags the ability."""
    fmt = format_of(state)
    mon = state.active[side]
    boost = (
        (battler_ability(state, side, mon) == AbilityId.SPEED_BOOST)
        & (state.hp[party(fmt, side), mon] > 0)
        & (state.stages[side, Stat.SPE] < STAGE_MAX)
        & (state.fake_out_turn[side].astype(_I32) != state.turn.astype(_I32) + 1)
    )
    state = change_stage(state, side, Stat.SPE, 1, boost)
    cured = (
        (battler_ability(state, side, mon) == AbilityId.SHED_SKIN)
        & (state.status[party(fmt, side), mon] != Status.NONE)
        & (state.hp[party(fmt, side), mon] > 0)
        & (outcome.shed_skin[side] != 0)
    )
    state = state._replace(
        status=state.status.at[party(fmt, side), mon].set(
            jnp.where(cured, Status.NONE, state.status[party(fmt, side), mon]).astype(
                state.status.dtype
            )
        ),
        status_counter=state.status_counter.at[party(fmt, side), mon].set(
            jnp.where(cured, 0, state.status_counter[party(fmt, side), mon]).astype(
                state.status_counter.dtype
            )
        ),
    )
    return reveal_ability(state, side, mon, STATUS_CURE, cured)


# The implemented checks of ``enum MonCondCheckState`` (battle_controller_player.c:1256),
# in its order.
END_OF_TURN_ORDER = (
    # MON_COND_CHECK_STATE_INGRAIN, battle_controller_player.c:1311, then
    # MON_COND_CHECK_STATE_AQUA_RING, :1332 (``_ingrain_then_aqua_ring``)
    ("ingrain_aqua_ring", _ingrain_then_aqua_ring),
    # MON_COND_CHECK_STATE_ABILITY, battle_controller_player.c:1355 (the only check that
    # reads the turn's ``Outcome``, see ``end_of_turn``)
    ("ability", _turn_end_ability),
    # MON_COND_CHECK_STATE_USE_ITEM, battle_controller_player.c:1362, then
    # MON_COND_CHECK_STATE_LEFTOVERS, :1369 (``_use_item_then_leftovers``)
    ("use_item", _use_item_then_leftovers),
    # MON_COND_CHECK_STATE_LEECH_SEED, battle_controller_player.c:1378
    ("leech_seed", _leech_seed_check),
    # MON_COND_CHECK_STATE_POISON, battle_controller_player.c:1393
    ("poison", _poison_check),
    # MON_COND_CHECK_STATE_TOXIC, battle_controller_player.c:1405
    ("toxic", _toxic_check),
    # MON_COND_CHECK_STATE_BURN, battle_controller_player.c:1425, then _CURSE, :1451
    # (``_burn_then_curse``)
    ("burn_curse", _burn_then_curse),
    # MON_COND_CHECK_STATE_BIND, battle_controller_player.c:1462, then _BAD_DREAMS, :1484
    # (``_bind_then_bad_dreams``)
    ("bind", _bind_then_bad_dreams),
    # MON_COND_CHECK_STATE_UPROAR, battle_controller_player.c:1505, then
    # MON_COND_CHECK_STATE_DISABLE, :1565, then _LOCK_ON, :1618 (``_uproar_then_disable``; the
    # only check that reads the turn's moveFailFlags)
    ("uproar_disable", _uproar_then_disable),
    # MON_COND_CHECK_STATE_TAUNT, battle_controller_player.c:1635, then MAGNET_RISE (:1647, in
    # ``_taunt_check``), (Heal Block, unsupported, between) _EMBARGO, :1671 (B13-31), then _YAWN,
    # :1684 (``_taunt_then_yawn``: none changes HP; the Yawn reads the turn's ``Outcome``)
    ("taunt_yawn", _taunt_then_yawn),
    # MON_COND_CHECK_STATE_ITEM_CONDITION, battle_controller_player.c:1699: the same berry
    # check, after the residual damage above; then MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT,
    # :1710 (``_item_condition_then_detrimental``)
    ("item_condition", _item_condition_then_detrimental),
)


def end_of_turn_order(tables: Tables, state: BattleState, outcome: Outcome) -> jnp.ndarray:
    """``int32[B]``: the sides in ``monSpeedOrder``, which
    ``BattleSystem_SortMonSpeedOrder`` (battle_lib.c:3364) sets after the last action
    (battle_controller_player.c:887). It compares with ``ignoreQuickClaw = TRUE``:
    no priority, so ``order.player_second`` with Quick Claw (the flag is still computed,
    battle_lib.c:1278-1286, from this turn's ``speedRand``, drawn at bcp.c:853-857 before
    every sort of the turn: ``outcome.quick_claw``), Lagging Tail, Trick Room and speed; a
    tie swaps on ``RandNext() & 1`` (``outcome.end_speed_tie``, 1 = opponent first)."""
    first = player_second(tables, state, outcome.end_speed_tie, outcome.quick_claw)
    first = first.astype(_I32)
    return jnp.stack([first, _foe(first)])


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
    Sun (:5585-5596): a Dry Skin or Solar Power holder loses ``Divide(maxHP, 8)``
    (``subscript_weather_continues.s`` :_045 / :_057, after its Magic Guard skip, tag the
    ability). Rain (:5613-5625): a Dry Skin holder below full HP regains ``Divide(maxHP, 8)``
    (:_076 tags it, as it tags Rain Dish's and Ice Body's heal, G-45); Hydration is unsupported.
    Sand Veil, Snow Cloak, Ice Body, Dry Skin and Solar Power are read through
    ``state.battler_ability``; Magic Guard by decomp id from ``ability`` (``int32[B]``,
    ``damage.battler_decomp_ability`` of each battler). The caller applies ``NO_CLOUD_NINE``.
    The sandstorm, sun and hail branches skip a battler with ``MOVE_EFFECT_NO_WEATHER_DAMAGE``
    (moves.h:50: Dig's UNDERGROUND, Dive's UNDERWATER; :5580, :5587, :5600), rain does not
    (B10-12)."""
    fmt = format_of(state)
    mon = state.active[side]
    hp, max_hp = (
        state.hp[party(fmt, side), mon].astype(_I32),
        state.max_hp[party(fmt, side), mon].astype(_I32),
    )
    types = state.types[party(fmt, side), mon]
    ab = ability[side]
    own = battler_ability(state, side)

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
    hidden = (state.semi_invulnerable[side] == SemiKind.UNDERGROUND) | (
        state.semi_invulnerable[side] == SemiKind.UNDERWATER
    )
    sand = sand & ~hidden
    hail = (state.weather == Weather.HAIL) & ~hidden
    ice_body = hail & (own == AbilityId.ICE_BODY)  # battle_script.c:5601 (G-45)
    hail_chip = hail & ~ice_body & ~has(Type.ICE) & (own != AbilityId.SNOW_CLOAK)
    eighth = jnp.maximum(max_hp // 8, 1)
    sun = (
        (state.weather == Weather.SUN)
        & ~hidden
        & ((own == AbilityId.DRY_SKIN) | (own == AbilityId.SOLAR_POWER))
    )
    rain = (state.weather == Weather.RAIN) & (own == AbilityId.DRY_SKIN) & (hp < max_hp)
    # battle_script.c:5614-5618 Rain Dish: Divide(maxHP, 16) below full HP in rain (B9-6).
    rain_dish = (state.weather == Weather.RAIN) & (own == AbilityId.RAIN_DISH) & (hp < max_hp)
    magic_guard = ab == WEATHER_ABILITY_IDS["Magic Guard"]
    chip = (sand | hail_chip) & ~magic_guard
    sun_chip = sun & ~magic_guard
    heal = (ice_body | rain_dish) & (hp < max_hp)
    delta = jnp.where(chip, -sixteenth, jnp.where(heal, sixteenth, 0))
    delta = jnp.where(sun_chip, -eighth, jnp.where(rain, eighth, delta))
    alive = hp > 0
    # :_076 tags every heal's battler, Ice Body's included (G-45).
    state = reveal_ability(state, side, mon, WEATHER_HP, alive & (sun_chip | rain | heal))
    return _hp_change(state, side, mon, jnp.where(alive, delta, 0))


def _wish(state: BattleState, side) -> BattleState:
    """FIELD_COND_CHECK_STATE_WISH (bcp.c:1085-1111) for ``side``'s battler: ``wishTurns`` drops
    when nonzero; at 0 the battler, if it has HP, regains ``Divide(maxHP, 2)``, its own max HP,
    unless at full HP (subscript_wish_heal.s:9-11), B12-14."""
    fmt = format_of(state)
    timer = state.wish_timer[side].astype(_I32)
    on = timer != 0
    left = jnp.where(on, timer - 1, timer)
    state = state._replace(
        wish_timer=state.wish_timer.at[side].set(left.astype(state.wish_timer.dtype))
    )
    mon = state.active[side]
    hp = state.hp[party(fmt, side), mon].astype(_I32)
    max_hp = state.max_hp[party(fmt, side), mon].astype(_I32)
    heals = on & (left == 0) & (hp > 0) & (hp != max_hp)
    healed = jnp.minimum(hp + divide(max_hp, 2), max_hp)
    return _set_mon(state, "hp", side, mon, healed, heals)


def field_conditions(tables: Tables, state: BattleState, sides) -> BattleState:
    """``BattleControllerPlayer_CheckFieldConditions`` (battle_controller_player.c:
    932-1254), between the actions and ``end_of_turn``; nothing runs once the battle is
    over (``CheckBattleOver`` before every step, :939-944). In order:

    1. ``FIELD_SIDE_TIMERS``, side 0 then side 1: a nonzero count drops by 1 and the
       condition ends at 0 (``--reflectTurns == 0``, :947-1083).
    2. Wish (:1085-1111, E2 batch 12, B12-14): each battler in ``sides`` (``monSpeedOrder``) with
       a nonzero ``wish_timer`` counts it down, and at 0 a battler with HP regains
       ``Divide(maxHP, 2)`` of its own (subscript_wish_heal.s: "HP is full!" at full HP, else
       UPDATE_HP, capped; Heal Block is unsupported) (``_wish``).
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
    for position in (0, 1):
        state = _where(live, _wish(state, sides[position]), state)

    before = state
    ability = battler_decomp_ability(tables, state, jnp.arange(state.active.shape[0]))
    effects = live & ~ended & ~weather_suppressed(tables, state)
    for position in (0, 1):
        side = sides[position]
        state = _where(effects, _weather_effect(state, side, ability), state)
    state = _faint_check(tables, before, state)
    # The loop head's AnyExpPayout (bcp.c:938-944) pays for the chip's faint.
    state = pay_exp(tables, before, state)

    # FIELD_COND_CHECK_STATE_GRAVITY (bcp.c:1227-1238): one turn off the counter; at 0 vanilla's
    # subscript_gravity_end only prints, Kaizo's (sub_seq 238, ``Tables.gravity_rearm``) sets the
    # counter to GRAVITY_TURNS again and runs the grounding loop (``gravity_ground``): Gravity
    # never ends (G-67).
    counted = _count_down(state.gravity)
    rearm = (tables.gravity_rearm != 0) & (state.gravity > 0) & (counted == 0) & (state.over == 0)
    gravity = jnp.where(rearm, GRAVITY_TURNS, counted)
    state = state._replace(gravity=jnp.where(state.over == 0, gravity, state.gravity))
    return gravity_ground(state, rearm)


def end_of_turn(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order, sides, failed=(0, 0)
) -> BattleState:
    """Phase 4: ``BattleControllerPlayer_CheckMonConditions`` (battle_controller_player.c:
    1289) runs every check of ``END_OF_TURN_ORDER`` for one battler, then for the next,
    in ``sides`` (``end_of_turn_order`` of the state after the actions: ``monSpeedOrder``
    is sorted once, before ``field_conditions``, so a Tailwind that ended there still
    counts); a faint check follows each, and nothing runs once the battle is over.
    Flinch lasts only for the turn it is set in. ``failed`` (``int[2]``): each side's
    ``moveFailFlags`` of the turn (the ``move_failed`` event), read by the Uproar check."""
    for position in (0, 1):
        side = sides[position]
        start = state
        for name, check in END_OF_TURN_ORDER:
            before = state
            args = (
                (outcome,)
                if name in ("ability", "taunt_yawn")
                else (failed,)
                if name == "uproar_disable"
                else ()
            )
            after = check(tables, state, side, *args)
            state = _faint_check(tables, before, _where(state.over == 0, after, state))
        # The payout of CheckMonConditions' head (AnyFainted, then AnyExpPayout, bcp.c:
        # 1295-1299) runs after the faint check of the check that fainted the opponent. It is
        # paid once per battler here, after its last check (one inline copy per battler instead
        # of one per check: 22 copies made the step program too slow to compile). Exact only
        # while the opponent can faint in its own checks alone: once it has fainted its later
        # checks do nothing, and they only heal the player, so nothing in between reads what
        # the payout writes or changes the player's living set it reads. The decomp has one
        # exception: a seeded Liquid Ooze holder damages the SEEDER during the seeded battler's
        # check (subscript_leech_seed_effect.s:_096), which can faint the opponent during the
        # player's checks, whose later checks read the max HP the payout changes. Liquid Ooze
        # has no AbilityId member, so coverage refuses it (IMPLEMENTED_ABILITIES,
        # coverage.py:164-166, covers every AbilityId but UNSUPPORTED; effects.py:212 lists
        # it), and it cannot happen here; whoever adds Liquid Ooze must pay after every check
        # again. The reference engine pays after every check, as the decomp does; the
        # differential tests compare the two.
        state = pay_exp(tables, start, state)
    return state._replace(flinched=jnp.zeros_like(state.flinched))


def _turn_move_cur(read: dict, order) -> jnp.ndarray:
    """``battleCtx->moveCur`` at the turn's end: the last action in ``order`` whose FIGHT ran
    (``read["move_cur"]``, 0 for none). Nothing else sets it after the actions, and
    ``BattleContext_Init`` keeps it (battle_lib.c:1974-2017). A turn with no FIGHT leaves an older
    moveCur, but then no battler is semi-invulnerable at the turn's end (the bit is set by a
    charge turn and its holder is locked into a FIGHT the next turn), so its range is never read."""
    first, last = (read["move_cur"][order[k]] for k in (0, 1))
    return jnp.where(last != 0, last, first).astype(_I32)


def _future_sight(
    tables: Tables, state: BattleState, sides, outcome: Outcome, protecting, move_cur=0
) -> BattleState:
    """``SIDE_COND_CHECK_STATE_FUTURE_SIGHT`` of ``BattleControllerPlayer_CheckSideConditions``
    (battle_controller_player.c:1758-1783), after ``CheckMonConditions`` (:1733): for each battler
    in ``sides`` (``monSpeedOrder``) with a count, the count drops, and at 0 a battler with HP
    clears its side's ``SIDE_CONDITION_FUTURE_SIGHT`` and runs subscript_future_sight_damage.s (a
    battler with no HP lets the count reach 0 and keeps the bit). The subscript: ``CheckMoveHit``
    (:8; ``outcome.future_threshold`` with ``Outcome.future_hit[battler]``, then
    ``CheckMoveHitOverrides``: a protecting defender (``protecting``: this turn's turnFlags) stops a
    ``MOVE_FLAG_CAN_PROTECT`` move and unlocks the battler in the attacker's slot, bcp.c:3005-3011;
    either battler's No Guard (or the attacker's battler locked onto by the defender) clears
    ``MOVE_STATUS_MISSED`` and returns, :3016-3021; else a semi-invulnerable defender sets
    ``MOVE_STATUS_SEMI_INVULNERABLE``, :3034-3040, B10-14: ``BattleContext_Init`` cleared every
    ``SYSCTL_HIT_DURING_*`` bit at the last MoveEnd, battle_lib.c:2009, system_control.h:31-38,
    and ``BYPASSED_ACCURACY`` with ``moveStatusFlags``, :1983; the range test reads
    ``CURRENT_MOVE_DATA`` (common.h:24), the turn's last ``moveCur`` (``move_cur``), so a last
    Spikes, Toxic Spikes or Stealth Rock, ``RANGE_OPPONENT_SIDE``, skips it); on
    ``MOVE_STATUS_DID_NOT_HIT`` "But it failed!" (:53-59). Else a defender behind a substitute
    (its bit, :13) has the damage taken off the substitute's HP, 0 when ``substituteHP <=
    damage`` (:15-21, B10-4; no flag, no HP change), then goes to :92; otherwise (Rage, :30-37,
    is unsupported) ``CheckHoldOnWith1HP`` (:28; battle_script.c:9098-9124: a Focus Sash at full HP,
    or a Focus Band whose roll ``Outcome.future_band[battler]`` passes, leaves 1 HP,
    ``MOVE_STATUS_ENDURED_ITEM``; ruling G-1) and ``UPDATE_HP`` (:29; ``waitingBattlers`` is 0
    after the actions, bcp.c:881-886, so no resist berry, subscript_update_hp.s:9-10); then, under
    ``MOVE_STATUS_ENDURED_ITEM``, a ``HOLD_EFFECT_ENDURE`` item is removed (:40-48).

    ``moveStatusFlags`` is cleared only by ``BattleContext_Init`` (battle_lib.c:1983, the last
    MoveEnd, bcp.c:3998), so the second hit of a pass reads the first one's flags: a miss or a
    Protect makes it fail too (unless No Guard clears the miss), and an endured first hit removes
    the second defender's Focus Sash. Between the two, ``AnyFainted`` faints the first defender
    (:1749, no ``CheckBattleOver``: the battle can end on both faints, a draw, which is a loss);
    the battle-over check and the exp payout are ``TurnEnd``'s (:1844-1847)."""
    fmt = format_of(state)
    live = state.over == 0
    start = state
    missed, shielded, endured = jnp.bool_(False), jnp.bool_(False), jnp.bool_(False)
    semi = jnp.bool_(False)
    protecting = jnp.asarray(protecting)
    semi_tested = ~targets_foe_side(tables, jnp.asarray(move_cur, _I32))
    mid = state
    for position in (0, 1):
        d = sides[position]
        mon = state.active[d]
        timer = state.future_timer[d].astype(_I32)
        counting = live & (timer > 0)
        left = jnp.where(counting, timer - 1, timer)
        fires = counting & (left == 0) & (state.hp[party(fmt, d), mon] > 0)
        state = _set_side(state, "future_timer", d, left, True)
        state = _set_side(state, "future_sight", d, 0, fires)
        attacker = jnp.clip(state.future_attacker[d].astype(_I32) - 1, 0, fmt.B - 1)
        move = state.future_move[d].astype(_I32)
        roll = outcome.future_hit[d].astype(_I32)
        missed_now = ~((roll > 0) & (roll <= future_threshold(tables, state, d)))
        # The hit's ordinary accuracy check clears the attacker slot's Micle flag (G-59).
        acc_run = fires & (tables.move_accuracy[move] != 0)
        state = _set_side(state, "micle", attacker, 0, acc_run)
        no_guard = (battler_ability(state, attacker) == AbilityId.NO_GUARD) | (
            battler_ability(state, d, mon) == AbilityId.NO_GUARD
        )
        # MON_IS_LOCKED_ONTO of the stored attacker's battler slot (bcp.c:3016-3018).
        no_guard = no_guard | locked_onto(state, attacker, d)
        blocked = (protecting[d] != 0) & protect_blocks(tables, move, False)
        miss_flag = jnp.where(~blocked & no_guard, False, missed | missed_now)
        shield_flag = shielded | blocked
        # MOVE_STATUS_SEMI_INVULNERABLE (bcp.c:3034-3040) past both returns; No Guard does not
        # clear one the first hit of the pass set.
        semi_now = ~blocked & ~no_guard & semi_tested & (state.semi_invulnerable[d] != 0)
        semi_flag = semi | semi_now
        state = _unlock(state, attacker, fires & blocked)
        hit = fires & ~miss_flag & ~shield_flag & ~semi_flag
        hp = state.hp[party(fmt, d), mon].astype(_I32)
        sash = holds(tables, state, d, ItemId.FOCUS_SASH)
        damage = state.future_damage[d].astype(_I32)
        full = hp == state.max_hp[party(fmt, d), mon].astype(_I32)
        sub_hp = state.substitute_hp[d].astype(_I32)
        behind = sub_hp != 0
        state = _set_side(state, "substitute_hp", d, jnp.maximum(sub_hp - damage, 0), hit & behind)
        # A Focus Band's roll (Outcome.future_band[d], ruling G-1) or a full-HP Focus Sash.
        band = holds(tables, state, d, ItemId.FOCUS_BAND) & (outcome.future_band[d] != 0)
        endured_now = hit & ~behind & ((sash & full) | band) & (damage >= hp)
        dealt = jnp.where(endured_now, hp - 1, jnp.minimum(damage, hp))
        state = _set_mon(state, "hp", d, mon, hp - dealt, hit & ~behind)
        endure_flag = endured | endured_now
        # :39-44: under MOVE_STATUS_ENDURED_ITEM the message tags msgBattlerTemp's item (bcp.c:
        # 1776; BattleMessage_ItemTag, battle_script.c:11741-11744), then removes a sash.
        state = reveal_item(state, d, mon, hit & endure_flag)
        state = consume_item(state, d, hit & endure_flag & sash)
        missed = jnp.where(fires, miss_flag, missed)
        shielded = jnp.where(fires, shield_flag, shielded)
        semi = jnp.where(fires, semi_flag, semi)
        endured = endured | (fires & endured_now)
        state = _faint_check(tables, mid, state, decide=False)
        mid = state
    state = _perish_song_check(tables, state, sides, live)
    state = _trick_room_check(tables, state, live)
    # TurnEnd's CheckBattleOver (bcp.c:1845) after the whole side-condition pass.
    state = _faint_check(tables, state, state)
    return pay_exp(tables, start, state)


def _trick_room_check(tables: Tables, state: BattleState, live) -> BattleState:
    """``SIDE_COND_CHECK_STATE_TRICK_ROOM`` (bcp.c:1822-1830), after Perish Song's: a Trick Room
    counter above 0 drops by 1; at 0 ``subscript_trick_room_end`` runs: vanilla's only prints, so
    Trick Room ends (started at 5 on turn T it is up through T + 4); Kaizo's (sub_seq 251, byte
    for byte the Trick Room move's script, ``Tables.trick_room_rearm``) finds the bits clear and
    sets ``FIELD_CONDITION_TRICK_ROOM_INIT`` again: it never ends. E3-2-4."""
    counted = _count_down(state.trick_room)
    running = live & (state.trick_room != 0)
    rearm = running & (counted == 0) & (tables.trick_room_rearm != 0)
    value = jnp.where(rearm, TRICK_ROOM_TURNS, counted)
    return state._replace(
        trick_room=jnp.where(running, value, state.trick_room).astype(state.trick_room.dtype)
    )


def _perish_song_check(tables: Tables, state: BattleState, sides, live) -> BattleState:
    """``SIDE_COND_CHECK_STATE_PERISH_SONG`` (bcp.c:1791-1820), after Future Sight's: each
    battler in ``sides`` (``monSpeedOrder``) with MOVE_EFFECT_PERISH_SONG either, at a count of 0,
    loses the effect and all its HP (subscript_continue_perish_song.s:6-7, UPDATE_HP of -curHP:
    no Focus Sash, no Magic Guard), or has its count drop by 1. Each re-entry of the pass runs
    ``AnyFainted`` first (:1749), with no battle-over check until TurnEnd."""
    for position in (0, 1):
        d = sides[position]
        mon = state.active[d]
        on = live & (state.perish_song[d] != 0)
        turns = state.perish_song_turns[d].astype(_I32)
        dies = on & (turns == 0)
        before = state
        state = _set_side(state, "perish_song", d, 0, dies)
        state = _set_side(state, "perish_song_turns", d, turns - 1, on & ~dies)
        state = _set_mon(state, "hp", d, mon, 0, dies)
        state = _faint_check(tables, before, state, decide=False)
    return state


def _turn_start_items(tables: Tables, state: BattleState) -> BattleState:
    """The next turn's ``ShowBattleMon`` (battle_controller_player.c:256-268) runs
    ``BattleSystem_TriggerEffectOnSwitch``, whose ``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS``
    (battle_lib.c:4151-4165) checks every battler's held item before the commands are chosen
    (``hooks.after_damage_items``). Its other states find nothing new at a turn start, and before
    a Future Sight hit (``_future_sight``, after the last item check of the turn's end,
    ``MON_COND_CHECK_STATE_ITEM_CONDITION``) the item check found nothing new either; a berry the
    hit brings into its range is eaten here. With a replacement pending, ``replace`` runs the
    whole switch-in check (``hooks.on_switch_in``) at that point instead, after the newcomer's
    hazards (``TurnEnd``'s ``ReplaceFainted``, bcp.c:1848, comes before the next ``ShowBattleMon``)."""
    fmt = format_of(state)
    run = (state.over == 0) & ~jnp.any(needs_replacement(state) != 0)
    for side in range(fmt.B):
        state = after_damage_items(tables, state, side, run)
    return state


def _unlock_sleepers(state: BattleState) -> BattleState:
    """``BattleSystem_SetupNextTurn`` (battle_lib.c:2240-2243): a locked battler that is
    asleep when the next turn is set up is unlocked."""
    fmt = format_of(state)
    for side in range(fmt.B):
        asleep = state.status[party(fmt, side), state.active[side]] == Status.SLEEP
        state = _unlock(state, side, asleep)
    return state


def _quick_claw_shown(tables: Tables, state: BattleState, actions, order) -> jnp.ndarray:
    """``int32[B]``: the side's Quick Claw animation played (``subscript_check_quick_claw.s``,
    the first step of ``BeforeMove``, bcp.c:3140-3143). ``CalcTurnOrder`` sets the battler's
    ``moveEffectsData.quickClaw`` only when two FIGHT commands are compared
    (``ignoreQuickClaw = FALSE``, bcp.c:772-786; battle_lib.c:1281-1284), whatever the
    priorities, from the turn's starting ``speedRand`` (``state.qc_roll``); the script plays
    the held-item animation (no text) only when the battler is not the last to act
    (``WAITING_BATTLERS != 1``): in singles, when it acts first. Bit 0 of the ``quick_claw``
    event; the Custap part comes from ``_use_move``."""
    fmt = format_of(state)
    fight = ~is_switch(actions) & ~is_item(actions)
    claw = quick_claw_active(tables, state, state.qc_roll)
    first = jnp.arange(fmt.B) == order[0]
    return (fight[0] & fight[1] & claw & first).astype(_I32)


def _action_phase(
    tables: Tables, state: BattleState, actions, outcome: Outcome, targets=None
) -> tuple[BattleState, dict, dict, jnp.ndarray]:
    """Phases 2-3 with the turn's events: ``(state, shown, read, order)``. ``learn_pin`` is the
    turn's own (design §3.3(a)): it starts each turn at 0. A resumed pause (``pivot_battler``)
    skips the turn's prologue (the pin, the AI's record of the player's move, the order and its
    Quick Claw, the bag items and a switch's Pursuit) and reads the order from the scratch.
    ``targets`` (``int8[B]``, the menu's picks) is required at B = 4 and ignored at B = 2, where
    the range decides (D2-1); the selection writes ``chosen_target`` before the turn order in both
    formats (ruling D2-33: a locked battler's defender reads it at B = 2 too, lib.c:1711)."""
    if targets is None:
        if state.active.shape[0] != 2:
            raise ValueError("a four-battler turn needs the chosen targets (D2-1)")
        targets = jnp.zeros(state.active.shape[0], jnp.int8)
    paused = state.pivot_battler != 0
    fresh = state._replace(learn_pin=jnp.zeros_like(state.learn_pin))
    fresh = fresh._replace(chosen_target=chosen_targets(tables, state, actions, targets))
    fresh = _record_last_move(tables, fresh, actions)
    order = action_order(tables, fresh, actions, outcome)
    claw = jnp.where(paused, 0, _quick_claw_shown(tables, fresh, actions, order))
    order = jnp.where(paused, state.ts_order, order).astype(_I32)
    power_mul = replacement_power_mul(tables, state, actions)
    s = _where(paused, state, fresh)
    s, read, shown = _execute_actions(tables, s, actions, outcome, order, paused, power_mul)
    return s, {**shown, "order": order, "quick_claw": claw | shown["quick_claw"]}, read, order


def turn_events(
    tables: Tables, state: BattleState, actions, outcome: Outcome, targets=None
) -> dict:
    """The events of ``step(tables, state, actions, outcome)``'s action phase, without
    playing the end of the turn (``step_events``)."""
    return _action_phase(tables, state, jnp.asarray(actions, _I32), outcome, targets)[1]


def step_events(
    tables: Tables, state: BattleState, actions, outcome: Outcome, targets=None
) -> tuple[BattleState, dict]:
    """``(step(...), its events)``: one implementation of the turn serves both ``step``
    (which drops the events) and ``engine.observe`` (which reads them). The events, a
    dict of ``int32`` arrays: per side (``[2]``) ``act_kind`` (``ACT_*``), ``act_id``,
    ``shown`` (the side's turn showed a message), ``crit`` (``[2, MAX_HITS]``) and
    ``EVENT_FLAGS``; ``order`` (``action_order``) and ``pursuit``. A call that pauses (E2 batch
    6) returns the first half's events and stops before the end of the turn; the resuming call
    returns the second half's."""
    actions = jnp.asarray(actions, _I32)
    s, events, read, order = _action_phase(tables, state, actions, outcome, targets)
    halted = s.pivot_battler != 0
    sides = end_of_turn_order(tables, s, outcome)
    e = field_conditions(tables, s, sides)
    e = end_of_turn(tables, e, actions, outcome, order, sides, read["move_failed"])
    e = _future_sight(tables, e, sides, outcome, read["protect_ok"], _turn_move_cur(read, order))
    e = _stale_known_item(s, e, False)  # B10-6: the end of the turn's HP updates
    e = _unlock_sleepers(e)
    e = e._replace(turn=(e.turn + 1).astype(e.turn.dtype))
    e = _turn_start_items(tables, e)
    # The turn's speedRand draw, kept for the next turn's readers (qc_kept).
    e = _keep_quick_claw(tables, e, outcome.quick_claw)
    # turnFlags.roosting lives until BattleSystem_SetupNextTurn (battle_lib.c:2229-2231), after
    # the faint replacements (bcp.c:1843-1856): kept while one is pending (``roosting_now``).
    pending = jnp.any(needs_replacement(e) != 0)
    e = e._replace(
        roosting=jnp.where(pending, roosting_now(e, read["roosted"]), 0).astype(e.roosting.dtype)
    )
    e = canonical_target(e)
    s = _where(halted, s, e)
    return _where(state.over != 0, state, s), events


def canonical_target(state: BattleState) -> BattleState:
    """At B = 2, ``chosen_target[b] = b ^ 1`` unless battler ``b`` is locked (``locked_move``)
    or encored (``encore_move``), at the end of a turn (ruling D2-35). At B = 2 the stored target
    is read only by ``BattleSystem_Defender``'s branch 10 on an action that does not randomize
    (lib.c:1709-1719): a locked battler, which picks no command (bcp.c:334-338, :1885-1888), so
    the selection does not overwrite it. An encored one keeps it too: it can lock under the
    Encore and read the old value once the Encore ends. Every other battler's next fight command
    overwrites it at selection (bcp.c:534-542, :4363-4367) before any read; lib.c:1678 and the
    other readers are doubles only. A paused turn (U-turn, Baton Pass) keeps it: the actions
    still to run read this turn's selection. So states that differ only in an unread stored
    target are one state (``search/rows.row_bytes``)."""
    if format_of(state).B != 2:
        return state
    keep = (state.locked_move != 0) | (state.encore_move != 0)
    canon = jnp.where(keep, state.chosen_target, jnp.arange(2) ^ 1)
    return state._replace(chosen_target=canon.astype(state.chosen_target.dtype))


def roosting_now(state: BattleState, roosted) -> jnp.ndarray:
    """``int8[B]``: ``turnFlags.roosting`` per battler from the turn's ``roosted`` events: a
    battler with no HP left has none (``BattleSystem_CleanupFaintedMon`` clears its turn flags,
    battle_lib.c:2189-2192). Per battler slot: a Pokemon Roar dragged in after its side's Roost
    keeps it, as the decomp's ``turnFlags[battler]`` does."""
    fmt = format_of(state)
    sides = jnp.arange(state.active.shape[0])
    alive = state.hp[party(fmt, sides), state.active] > 0
    return ((jnp.asarray(roosted) != 0) & alive).astype(jnp.int8)


def step(
    tables: Tables, state: BattleState, actions, outcome: Outcome, targets=None
) -> BattleState:
    """Play one turn: ``actions`` is ``int32[B]`` (singles: player, opponent; see ``order``),
    ``targets`` the ``int8[B]`` target each battler chose (D2-1; required at B = 4, ignored
    at B = 2, where the range decides)."""
    return step_events(tables, state, actions, outcome, targets)[0]


def qc_kept(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[B]``: the side's ``speedRand`` can be read as a Quick Claw before it is redrawn
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
    fmt = format_of(state)
    # The raw item (a Klutz holder or an embargoed one keeps the draw: over-keeping only splits
    # states, B13-29).
    rows = party(fmt, jnp.arange(fmt.B))  # each battler's party
    holds = tables.item_effect[state.item[rows]] == ItemId.QUICK_CLAW
    alive = (state.species[rows] != 0) & (state.hp[rows] > 0)
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
    exact 13108/65536), for the sides ``qc_kept`` reads; 0 elsewhere. Then a weather war the
    battle-start sort left open (``BattleState.war_swaps``) is settled on that ``speedRand``
    and a tie bit from ``fold_in(key, 1)`` (``resolve_war``, ruling G-26)."""
    draw = jax.random.randint(key, (format_of(state).B,), 0, QUICK_CLAW_MODULUS) == 0
    state = _keep_quick_claw(tables, jax.tree.map(jnp.asarray, state), draw)
    return sample_war(jax.random.fold_in(key, 1), state)


def root_belief(tables: Tables, state: BattleState) -> list[tuple[BattleState, float]]:
    """Stage B's root chance node (host): ``(state, probability)`` for every value of the
    hidden ``qc_roll`` under ``sample_root``'s law, one state per combination over the
    ``qc_kept`` sides (1 at 1/5, 0 at 4/5), the other sides 0. They differ only in hidden
    state, so they are one information set. Without a kept side: ``[(state, 1.0)]``. A weather
    war left open (``resolve_war``, ruling G-26) is settled per combination, split into its
    two tie values (1/2 each) where the tie decides it: those states differ in the weather,
    which the player sees before acting, so Stage B makes the root a chance node over the
    visible outcomes, each its own belief node (``search.stage_b``, ruling G-34). A host
    state (NumPy leaves, as ``prep`` and the CLI build it) is read as JAX arrays first: the
    war's settling writes with ``.at`` (``hooks.reveal_ability``)."""
    from nuzline.engine.coverage import check_acid_rain  # coverage imports the engine

    check_acid_rain(state)  # a refused state is never played on (docs/kaizo FCM.md section 5)
    state = jax.tree.map(jnp.asarray, state)
    kept = [bool(k) for k in jax.device_get(qc_kept(tables, state))]
    n = QUICK_CLAW_MODULUS
    out = []
    for bits in itertools.product(*((0, 1) if k else (0,) for k in kept)):
        prob = 1.0
        for k, b in zip(kept, bits, strict=True):
            if k:
                prob *= 1 / n if b else (n - 1) / n
        drawn = state._replace(qc_roll=jnp.asarray(bits, state.qc_roll.dtype))
        if bool(war_split(drawn)):
            out.extend((resolve_war(drawn, tie), prob / 2) for tie in (0, 1))
        else:
            out.append((resolve_war(drawn, 0), prob))
    return out


def war_split(state: BattleState) -> jnp.ndarray:
    """``bool``: ``state`` holds an open weather war (``BattleState.war_swaps``) whose order
    the tie bit decides at its ``speedRand`` (``qc_roll``)."""
    swaps = state.war_swaps.astype(_I32)
    qc = state.qc_roll.astype(_I32)
    at = 2 * qc[0] + 4 * qc[1]
    return (swaps != 0) & (((swaps >> at) & 1) != ((swaps >> (at + 1)) & 1))


def resolve_war(state: BattleState, tie) -> BattleState:
    """``state`` with its open weather war settled (ruling G-26): ``monSpeedOrder`` is the
    sort's at the state's ``speedRand`` (``qc_roll``) and the tie's ``RandNext() & 1``
    ``tie`` (bit ``tie + 2 * qc[0] + 4 * qc[1]`` of ``war_swaps``, ``hooks.order_swaps``);
    where it put the player second, the player's weather stays (``hooks.swap_war``).
    ``war_swaps`` is then 0. Without an open war: ``state`` (``war_swaps`` is 0)."""
    swaps = state.war_swaps.astype(_I32)
    qc = state.qc_roll.astype(_I32)
    bit = (swaps >> (jnp.asarray(tie, _I32) + 2 * qc[0] + 4 * qc[1])) & 1
    state = swap_war(state, (swaps != 0) & (bit == 1))
    return state._replace(war_swaps=jnp.zeros_like(state.war_swaps))


def sample_war(key, state: BattleState) -> BattleState:
    """``resolve_war`` with the tie drawn from ``key`` (``RandNext() & 1``, uniform): the
    draw a playout makes before the next decision (``search.playout``)."""
    return resolve_war(state, jax.random.randint(key, (), 0, 2))


def needs_replacement(state: BattleState) -> jnp.ndarray:
    """``int8[B]``: 1 where a side's active Pokemon has fainted and a healthy one is
    left to send in, while the battle goes on. A player faint is not terminal (the battle
    ends at a whiteout, ``_faint_check``), so both sides are served."""
    fmt = format_of(state)
    slots = jnp.arange(state.hp.shape[1])
    active_down = state.hp[party(fmt, jnp.arange(fmt.B)), state.active] == 0
    rows = party(fmt, jnp.arange(fmt.B))  # each battler's party
    bench = (
        (state.species[rows] != 0)
        & (state.hp[rows] > 0)
        & (slots[None, :] != state.active[:, None])
    )
    return (active_down & jnp.any(bench, axis=1) & (state.over == 0)).astype(jnp.int8)


def replace(
    tables: Tables, state: BattleState, side, slot, effects=True, trace_pick=None
) -> BattleState:
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
    newcomer's ``fake_out_turn`` and for a faint on entry.

    Where ``effects`` (``bool``, traceable) is false only the list runs (the hazards and the
    faint check): the newcomer's switch-in states wait for ``ShowBattleMon`` while the other
    side sends in again first, ``entry_pending[side]`` is set for a newcomer with HP left
    (``hooks.show_battle_mon`` runs them; ``search.playout.replacement_pass``, ruling G-32)."""
    fmt = format_of(state)
    entered = _switch_in(tables, state, side, slot, state.turn)
    effects = jnp.asarray(effects, bool)
    hazards = entry_hazards(tables, entered, side)  # hooks.on_switch_in's first part
    arrived = _where(effects, switch_in_effects(tables, hazards, side, trace_pick), hazards)
    out = _faint_check(tables, entered, arrived, entry_turn=state.turn)
    waits = ~effects & (out.hp[party(fmt, side), out.active[side]] > 0) & (out.over == 0)
    out = out._replace(
        entry_pending=out.entry_pending.at[side].set(waits.astype(out.entry_pending.dtype))
    )
    # TurnEnd's AnyExpPayout (bcp.c:1846) pays for a newcomer the hazards fainted. Learn work
    # it leaves inside the opponent's replacement loop would be read by the next replacement
    # pick before the learn resolve: refused as window (c) (design §3.3(c)).
    out, leveled = pay_exp_leveled(tables, entered, out)
    window_c = (jnp.asarray(side) == 1) & jnp.any(leveled) & (out.over == 0)
    out = out._replace(
        exp_refused=jnp.where(window_c, EXP_REFUSED_WINDOW_C, out.exp_refused).astype(
            out.exp_refused.dtype
        )
    )
    # The turn's turnFlags.roosting ends with the last replacement (SetupNextTurn, bcp.c:1856).
    pending = jnp.any(needs_replacement(out) != 0)
    out = out._replace(roosting=jnp.where(pending, out.roosting, 0).astype(out.roosting.dtype))
    # The hazards' assuranceDamageMask (effects.entry_hazards) is turnFlags too, which
    # SetupNextTurn clears before any action reads it (B13-17): a turn scratch, 0 between turns.
    out = out._replace(ts_assurance=jnp.zeros_like(out.ts_assurance))
    # The slot's speedRand stays; it is unread unless the newcomer (or, after an entry
    # faint, the next replacement) can be a Quick Claw holder (qc_kept).
    return _keep_quick_claw(tables, out, out.qc_roll)
