"""One battle turn: ``step(tables, state, actions, outcome) -> state``.

Pure, jit- and vmap-able, fixed shapes; every random event comes from
``outcome`` (``step`` draws nothing). The turn follows spec §4.2's six phases:

1. The opponent's choice is made outside ``step`` (``actions`` is an input).
2. Order: ``order.action_order``.
3. ``execute_actions``: the first actor, a faint check, the second actor, a faint
   check (``BattleControllerPlayer_CheckAnyFainted`` after each action). A switch
   facing Pursuit is preceded by the Pursuit hit and its own faint check.
4. ``end_of_turn``: the checks of ``END_OF_TURN_ORDER`` for each battler in
   ``end_of_turn_order``, each followed by a faint check (the decomp re-enters
   ``BattleControllerPlayer_CheckMonConditions``, battle_controller_player.c:1289,
   through ``AnyFainted`` / ``CheckBattleOver`` after every check that ran a script).
5. Faint replacement is a separate decision point outside ``step``:
   ``needs_replacement`` and ``replace``.
6. The battle ends at the first faint check that finds a player faint or no
   opponent left (``_faint_check``); later checks of the turn do nothing.

Implemented so far: the opponent's bag items (Potion, ``_use_trainer_item``),
switching (with Natural Cure) and the switch-in hooks
(``hooks.on_switch_in``: Stealth Rock, Intimidate, berries), damaging moves, the
accuracy check, PP, Struggle, the move effects of ``effects.py``, the berry check
after every hit (``hooks.after_damage_items``), Pursuit's pre-switch hit, the sleep,
freeze, flinch, Taunt, confusion and paralysis checks before a move, the held-item,
Leech Seed, poison, toxic, burn, Bind and Taunt end-of-turn checks, faints and the
end of the battle. A player faint is terminal (spec §3): it
sets ``over = 1, winner = 1``. A state that is already over is returned unchanged.

Not implemented (M6): the decomp's ``BattleControllerPlayer_CheckFieldConditions``
pass (battle_controller_player.c:932): hail and sandstorm chip damage and the
Reflect, Light Screen, Tailwind and Trick Room countdowns. A state with
``Weather.HAIL`` / ``Weather.SANDSTORM`` or a nonzero screen/Tailwind/Trick Room timer
is stepped with none of them, so M1b's coverage registry must mark those as unsupported
(docs/discrepancies.md, "Not implemented in either engine").
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.damage import confusion_damage
from nuzline.engine.effects import apply_move_effect, change_stage, defender_always
from nuzline.engine.hooks import after_damage_items, on_switch_in
from nuzline.engine.ids import POTION_HP_RESTORED, AbilityId, EffectId, Stat, Status, TrainerItemId
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    effective_speed,
    is_item,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import BAG_SLOTS, BattleState

_I32 = jnp.int32
WINNER_OPPONENT, WINNER_PLAYER = 1, 2
PURSUIT_POWER_MUL = 20  # BtlCmd_TryPursuit (battle_script.c:6884): powerMul = 20
# MON_CONDITION_TOXIC_COUNTER is 4 bits (include/constants/battle/condition.h:23).
TOXIC_COUNTER_MAX = 15

# Per-side fields of the active Pokemon that leave with it: the incoming battler is
# rebuilt from its party mon by ``BattleSystem_InitBattleMon`` (battle_lib.c:70), stat
# stages reset to neutral (:97, no Baton Pass in M1). Hazards and screens are per side
# and stay.
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


def _switch_in(state: BattleState, side, slot) -> BattleState:
    """``side`` sends out party ``slot``: stages and volatiles reset, and the Bind the
    outgoing Pokemon put on the other side ends (``BattleSystem_UpdateAfterSwitch``,
    battle_lib.c:2094)."""
    state = _reset_battler(state, side, True)
    state = state._replace(active=state.active.at[side].set(jnp.asarray(slot).astype(jnp.int8)))
    return _release_bind(state, 1 - side, True)


def _send_out(tables: Tables, state: BattleState, side, slot) -> BattleState:
    """A switch action: ``_switch_out``, ``_switch_in``, then ``hooks.on_switch_in``
    (entry hazards, Intimidate, berries)."""
    return on_switch_in(tables, _switch_in(_switch_out(state, side), side, slot), side)


def _use_move(
    tables: Tables, state: BattleState, side, action, choice, outcome: Outcome, target_moved
) -> BattleState:
    """``side``'s active Pokemon uses the move in slot ``action`` (or Struggle).
    ``choice``: its ``(move, struggling)`` from ``selected_move`` at the start of the
    turn, when the decomp fixes Struggle (battle_controller_player.c:433), so a Taunt
    landing earlier this turn makes a chosen power-0 move fail instead of Struggling.
    ``target_moved``: the other side has already acted this turn."""
    move, struggling = choice
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    target_side = 1 - side
    target = state.active[target_side]
    ready = (state.over == 0) & (state.hp[side, user] > 0)

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
    go = go & ~flinched
    # CHECK_STATUS_STATE_TAUNT (:2557): a taunted battler's power-0 move fails.
    go = go & ~((state.taunt[side] > 0) & (tables.move_power[move] == 0))
    # CHECK_STATUS_STATE_CONFUSION (:2614): the counter drops first; at 0 the battler snaps
    # out and moves, otherwise RandNext() & 1 (outcome.confusion_hit) makes it hit itself
    # (:2633, damage.confusion_damage) instead of moving.
    confused = go & (state.confusion[side] > 0)
    confusion = state.confusion[side] - confused.astype(state.confusion.dtype)
    self_hit = confused & (confusion > 0) & (outcome.confusion_hit[side] != 0)
    user_hp = state.hp[side, user].astype(_I32)
    hurt = jnp.minimum(confusion_damage(tables, state, side, outcome.roll[side, 0]), user_hp)
    user_hp = user_hp - jnp.where(self_hit, hurt, 0)
    state = state._replace(
        confusion=state.confusion.at[side].set(confusion),
        hp=state.hp.at[side, user].set(user_hp.astype(state.hp.dtype)),
    )
    go = go & ~self_hit
    # CHECK_STATUS_STATE_PARALYSIS (:2652): RandNext() % 4 == 0 (outcome.full_para) fully
    # paralyzes (Magic Guard is not implemented).
    full_para = (state.status[side, user] == Status.PARALYSIS) & (outcome.full_para[side] != 0)
    can_act = go & ~full_para
    # CHECK_STATUS_STATE_SELF_THAW (:2728): a still-frozen user of a thawing move thaws.
    state = _set_mon(state, "status", side, user, Status.NONE, can_act & still_frozen)

    # BattleControllerPlayer_DecrementPP (battle_controller_player.c:3182) runs before the
    # no-target check (:3190); Struggle has no PP to spend.
    spend = can_act & ~struggling
    pp = state.pp.at[side, user, slot].add(jnp.where(spend, -1, 0).astype(state.pp.dtype))
    state = state._replace(pp=pp)

    # Accuracy 0 never misses (CheckMoveHitAccuracy returns before rolling, :2901). A
    # multi-hit move checks accuracy on its first hit only (effects.py).
    # The no-target check (:3190) fails a move whose defender has fainted; a move whose
    # defender exists whatever the foe's HP (effects.defender_always) never fails there.
    never_misses = tables.move_accuracy[move] == 0
    has_target = (state.hp[target_side, target] > 0) | defender_always(tables, move)
    connected = can_act & has_target & ((outcome.hit[side, 0] != 0) | never_misses)
    for hit_index in range(MAX_HITS):
        state = apply_move_effect(
            tables,
            state,
            side,
            move,
            outcome,
            hit_index,
            connected=connected,
            target_moved=target_moved,
            struggling=struggling,
        )
        # BattleControllerPlayer_AfterMoveEffects (battle_controller_player.c:3638) runs
        # after every hit of a move that connected: AFTER_MOVE_EFFECT_ATTACKER_ITEM
        # (:3684), then _DEFENDER_ITEM (:3691). A disrupted move, a confusion self-hit,
        # a miss (MOVE_STATUS_DID_NOT_HIT, :3325) or a move with no target skips it. Its
        # AFTER_MOVE_EFFECT_HELD_ITEM_STATUS (:3730) re-checks the same two Pokemon in
        # singles and cannot trigger anything new. A later hit that did not happen
        # changes no HP, so its check finds what the previous one found.
        state = after_damage_items(tables, state, side, connected)
        state = after_damage_items(tables, state, 1 - side, connected)
    return state


def _bag_slot(action) -> jnp.ndarray:
    return jnp.clip(action - FIRST_ITEM_ACTION, 0, BAG_SLOTS - 1)


def _consume_trainer_items(state: BattleState, actions) -> BattleState:
    """Each side choosing an item action loses that bag slot when it chooses:
    ``TrainerAI_ShouldUseItem`` (trainer_ai.c:4193) zeroes ``trainerItems[i]`` at command
    selection, so the item is spent even if its action never runs."""
    for side in (0, 1):
        slot = _bag_slot(actions[side])
        kept = jnp.where(is_item(actions[side]), 0, state.trainer_items[side, slot])
        state = state._replace(
            trainer_items=state.trainer_items.at[side, slot].set(
                kept.astype(state.trainer_items.dtype)
            )
        )
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
) -> BattleState:
    action = actions[side]
    switched = _send_out(tables, state, side, jnp.clip(action - MOVE_ACTIONS, 0, 5))
    used = _use_trainer_item(tables, state, side, items[side])
    choice = (choices[0][side], choices[1][side])
    moved = _use_move(tables, state, side, action, choice, outcome, target_moved)
    live = state.over == 0
    acted = _where(is_item(action) & live, used, moved)
    return _where(is_switch(action) & live, switched, acted)


def _pursuit_fires(tables: Tables, state: BattleState, side, actions, choices) -> jnp.ndarray:
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


def _pursuit_hit(
    tables: Tables, state: BattleState, pursuer, actions, choices, outcome: Outcome
) -> BattleState:
    """The pursuer hits the outgoing Pokemon: ``BtlCmd_TryPursuit`` spends the PP and
    sets ``powerMul = 20``; ``subscript_pursuit.s`` then runs crit, damage and the type
    chart with no accuracy check. The pursuer's own action is over."""
    action = actions[pursuer]
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    mon = state.active[pursuer]
    state = state._replace(pp=state.pp.at[pursuer, mon, slot].add(jnp.int8(-1)))
    move = choices[0][pursuer]
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


def _faint_check(before: BattleState, after: BattleState) -> BattleState:
    """Count new player faints and end the battle: any player Pokemon fainting is a loss (the
    terminal rule, which wins over a simultaneous last opponent faint); no opponent
    Pokemon left standing is a win. A fainted active battler is cleaned up as
    ``BattleSystem_CleanupFaintedMon`` (battle_lib.c:2156, from ``PlayFaintAnimation``,
    battle_script.c:1803) does: its stat stages, ``statusVolatile``, ``moveEffectsMask``
    and ``moveEffectsData`` are cleared (``_reset_battler``), and the Bind it put on the
    other side ends (:2179). Every newly fainted Pokemon also loses its status condition
    and counter: the decomp writes ``status = 0`` for a Pokemon with no HP
    (battle_controller.c:1482-1486)."""
    fainted = (before.hp > 0) & (after.hp == 0)
    after = after._replace(
        status=jnp.where(fainted, 0, after.status).astype(after.status.dtype),
        status_counter=jnp.where(fainted, 0, after.status_counter).astype(
            after.status_counter.dtype
        ),
    )
    player_fainted = jnp.sum(fainted[0].astype(_I32))
    opp_out = jnp.all((after.species[1] == 0) | (after.hp[1] == 0))
    ended = (after.over == 0) & ((player_fainted > 0) | opp_out)
    winner = jnp.where(player_fainted > 0, WINNER_OPPONENT, WINNER_PLAYER)
    after = after._replace(
        player_faints=(after.player_faints + player_fainted).astype(after.player_faints.dtype),
        over=jnp.where(ended, 1, after.over).astype(after.over.dtype),
        winner=jnp.where(ended, winner, after.winner).astype(after.winner.dtype),
    )
    for side in (0, 1):
        active_fainted = fainted[side, after.active[side]]
        after = _reset_battler(after, side, active_fainted)
        after = _release_bind(after, 1 - side, active_fainted)
    return after


def execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 3: both actors in order, each followed by a faint check. A switch facing
    Pursuit is hit first (with its own faint check), and the pursuer does not act again.
    An item action's bag slot is spent first (``_consume_trainer_items``)."""
    first = order[0].astype(_I32)
    # Each side's move and bag item are fixed when chosen, from the turn's starting state.
    picks = [selected_move(tables, state, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    items = jnp.stack(
        [state.trainer_items[side, _bag_slot(actions[side])] for side in (0, 1)]
    ).astype(_I32)
    state = _consume_trainer_items(state, actions)
    pursued = _pursuit_fires(tables, state, first, actions, choices)
    state = _where(
        is_switch(actions[first]) & (state.over == 0), _natural_cure(state, first), state
    )
    before = state
    hit = _pursuit_hit(tables, state, 1 - first, actions, choices, outcome)
    state = _faint_check(before, _where(pursued, hit, state))
    for position in (0, 1):
        before = state
        side = order[position].astype(_I32)
        target_moved = position == 1  # the first actor's target has not moved yet
        acted = _act(tables, state, side, actions, choices, items, outcome, target_moved)
        if position == 1:
            acted = _where(pursued, state, acted)
        state = _faint_check(before, acted)
    return state


def _hp_change(state: BattleState, side, mon, delta) -> BattleState:
    hp = jnp.clip(state.hp[side, mon].astype(_I32) + delta, 0, state.max_hp[side, mon])
    return state._replace(hp=state.hp.at[side, mon].set(hp.astype(state.hp.dtype)))


def _leech_seed_check(tables: Tables, state: BattleState, side) -> BattleState:
    """``side``'s seeded active Pokemon with HP left loses ``BattleSystem_Divide(maxHP, 8)``
    (capped at its HP) and the other side's active Pokemon, if alive, gains what it lost
    (``subscript_leech_seed_effect.s``)."""
    other = 1 - side
    mon, recipient = state.active[side], state.active[other]
    hp = state.hp[side, mon].astype(_I32)
    on = (state.leech_seed[side] != 0) & (hp > 0) & (state.hp[other, recipient] > 0)
    drained = jnp.minimum(jnp.maximum(state.max_hp[side, mon].astype(_I32) // 8, 1), hp)
    drained = jnp.where(on, drained, 0)
    return _hp_change(_hp_change(state, side, mon, -drained), other, recipient, drained)


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


def _taunt_check(tables: Tables, state: BattleState, side) -> BattleState:
    """The Taunt counter drops by 1 (no HP condition in the decomp)."""
    taunt = state.taunt[side] - (state.taunt[side] > 0).astype(state.taunt.dtype)
    return state._replace(taunt=state.taunt.at[side].set(taunt))


# The implemented checks of ``enum MonCondCheckState`` (battle_controller_player.c:1256),
# in its order.
END_OF_TURN_ORDER = (
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
    # MON_COND_CHECK_STATE_TAUNT, battle_controller_player.c:1635
    ("taunt", _taunt_check),
    # MON_COND_CHECK_STATE_ITEM_CONDITION, battle_controller_player.c:1699: the same berry
    # check, after the residual damage above.
    ("item_condition", after_damage_items),
)


def end_of_turn_order(state: BattleState, outcome: Outcome) -> jnp.ndarray:
    """``int32[2]``: the sides in ``monSpeedOrder``, which
    ``BattleSystem_SortMonSpeedOrder`` (battle_lib.c:3364) sets after the last action
    (battle_controller_player.c:887). It compares with ``ignoreQuickClaw = TRUE``:
    speed only, no priority, Trick Room reversed; a tie swaps on ``RandNext() & 1``
    (``outcome.end_speed_tie``, 1 = opponent first)."""
    speed0, speed1 = effective_speed(state, 0), effective_speed(state, 1)
    slower = jnp.where(state.trick_room != 0, speed0 > speed1, speed0 < speed1)
    tie = (speed0 == speed1) & (outcome.end_speed_tie != 0)
    first = (slower | tie).astype(_I32)
    return jnp.stack([first, 1 - first])


def end_of_turn(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 4: ``BattleControllerPlayer_CheckMonConditions`` (battle_controller_player.c:
    1289) runs every check of ``END_OF_TURN_ORDER`` for one battler, then for the next,
    in ``end_of_turn_order``; a faint check follows each, and nothing runs once the
    battle is over. Flinch lasts only for the turn it is set in."""
    sides = end_of_turn_order(state, outcome)
    for position in (0, 1):
        side = sides[position]
        for _, check in END_OF_TURN_ORDER:
            before = state
            state = _faint_check(before, _where(state.over == 0, check(tables, state, side), state))
    return state._replace(flinched=jnp.zeros_like(state.flinched))


def step(tables: Tables, state: BattleState, actions, outcome: Outcome) -> BattleState:
    """Play one turn: ``actions`` is ``int32[2]`` (player, opponent; see ``order``)."""
    actions = jnp.asarray(actions, _I32)
    order = action_order(tables, state, actions, outcome)
    s = execute_actions(tables, state, actions, outcome, order)
    s = end_of_turn(tables, s, actions, outcome, order)
    s = s._replace(turn=(s.turn + 1).astype(s.turn.dtype))
    return _where(state.over != 0, state, s)


def needs_replacement(state: BattleState) -> jnp.ndarray:
    """``int8[2]``: 1 where a side's active Pokemon has fainted and a healthy one is
    left to send in, while the battle goes on. By the terminal rule only the
    opponent ever reaches this point."""
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
    that Stealth Rock faints triggers neither."""
    entered = _switch_in(state, side, slot)
    return _faint_check(entered, on_switch_in(tables, entered, side))
