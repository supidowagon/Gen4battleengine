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

Implemented so far: switching, Stealth Rock on a switch-in, damaging moves, the
accuracy check, PP, Struggle, the move effects of ``effects.py``, Pursuit's
pre-switch hit, the Taunt and confusion checks before a move, the Leech Seed, Bind
and Taunt end-of-turn checks, faints and the end of the battle. A player faint is
terminal (spec §3): it sets ``over = 1, winner = 1``. A state that is already over
is returned unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import confusion_damage
from nuzline.engine.effects import (
    apply_move_effect,
    change_stage,
    stealth_rock_damage,
    targets_user,
)
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    effective_speed,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import BattleState

_I32 = jnp.int32
WINNER_OPPONENT, WINNER_PLAYER = 1, 2
PURSUIT_POWER_MUL = 20  # BtlCmd_TryPursuit (battle_script.c:6884): powerMul = 20

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


def _switch_in(state: BattleState, side, slot) -> BattleState:
    """``side`` sends out party ``slot``: stages and volatiles reset, and the Bind the
    outgoing Pokemon put on the other side ends (``BattleSystem_UpdateAfterSwitch``,
    battle_lib.c:2094)."""
    fields = {name: getattr(state, name).at[side].set(0) for name in _VOLATILE}
    state = state._replace(
        active=state.active.at[side].set(jnp.asarray(slot).astype(jnp.int8)),
        stages=state.stages.at[side].set(0),
        **fields,
    )
    return _release_bind(state, 1 - side, True)


def _send_out(tables: Tables, state: BattleState, side, slot) -> BattleState:
    """A switch action: ``_switch_in``, then the entry hazards
    (``subscript_hazards_check.s``; Stealth Rock is the only one in M1)."""
    return stealth_rock_damage(tables, _switch_in(state, side, slot), side)


def _use_move(
    tables: Tables, state: BattleState, side, action, outcome: Outcome, target_moved
) -> BattleState:
    """``side``'s active Pokemon uses the move in slot ``action`` (or Struggle).
    ``target_moved``: the other side has already acted this turn."""
    move, struggling = selected_move(state, side, action)
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    target_side = 1 - side
    target = state.active[target_side]
    ready = (state.over == 0) & (state.hp[side, user] > 0)
    flinched = state.flinched[side] != 0

    # BattleControllerPlayer_CheckStatusDisruption (battle_controller_player.c:2414) runs
    # before the PP decrement; the first check that disrupts the move stops the rest.
    # CHECK_STATUS_STATE_FLINCH (:2529); subscript_flinched.s:10-14 gives Steadfast +1 Speed.
    steadfast = ready & flinched & (state.ability[side, user] == AbilityId.STEADFAST)
    state = change_stage(state, side, Stat.SPE, 1, steadfast)
    # CHECK_STATUS_STATE_TAUNT (:2557): a taunted battler's power-0 move fails.
    taunted = ready & ~flinched & (state.taunt[side] > 0) & (tables.move_power[move] == 0)
    # CHECK_STATUS_STATE_CONFUSION (:2614): the counter drops first; at 0 the battler snaps
    # out and moves, otherwise RandNext() & 1 (outcome.confusion_hit) makes it hit itself
    # (:2633, damage.confusion_damage) instead of moving.
    confused = ready & ~flinched & ~taunted & (state.confusion[side] > 0)
    confusion = state.confusion[side] - confused.astype(state.confusion.dtype)
    self_hit = confused & (confusion > 0) & (outcome.confusion_hit[side] != 0)
    user_hp = state.hp[side, user].astype(_I32)
    hurt = jnp.minimum(confusion_damage(tables, state, side, outcome.roll[side, 0]), user_hp)
    user_hp = user_hp - jnp.where(self_hit, hurt, 0)
    state = state._replace(
        confusion=state.confusion.at[side].set(confusion),
        hp=state.hp.at[side, user].set(user_hp.astype(state.hp.dtype)),
    )
    can_act = ready & ~flinched & ~taunted & ~self_hit

    # BattleControllerPlayer_DecrementPP (battle_controller_player.c:3182) runs before the
    # no-target check (:3190); Struggle has no PP to spend.
    spend = can_act & ~struggling
    pp = state.pp.at[side, user, slot].add(jnp.where(spend, -1, 0).astype(state.pp.dtype))
    state = state._replace(pp=pp)

    # Accuracy 0 never misses (CheckMoveHitAccuracy returns before rolling, :2901). A
    # multi-hit move checks accuracy on its first hit only (effects.py).
    # The no-target check (:3190) fails a move whose defender has fainted; a move whose
    # defender is its user (effects.targets_user) never fails there.
    never_misses = tables.move_accuracy[move] == 0
    has_target = (state.hp[target_side, target] > 0) | targets_user(tables, move)
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
        )

    # subscript_struggle.s: recoil BattleSystem_Divide(-maxHP, 4), at least 1, on a hit.
    recoil = jnp.where(
        connected & struggling, jnp.maximum(state.max_hp[side, user].astype(_I32) // 4, 1), 0
    )
    user_hp = jnp.maximum(state.hp[side, user].astype(_I32) - recoil, 0)
    return state._replace(hp=state.hp.at[side, user].set(user_hp.astype(state.hp.dtype)))


def _act(
    tables: Tables, state: BattleState, side, actions, outcome: Outcome, target_moved
) -> BattleState:
    action = actions[side]
    switched = _send_out(tables, state, side, jnp.clip(action - MOVE_ACTIONS, 0, 5))
    moved = _use_move(tables, state, side, action, outcome, target_moved)
    return _where(is_switch(action) & (state.over == 0), switched, moved)


def _pursuit_fires(tables: Tables, state: BattleState, side, actions) -> jnp.ndarray:
    """``side`` switches and the other side, which has not acted yet, chose Pursuit:
    ``BtlCmd_TryPursuit`` (battle_script.c:6884, called from
    ``subscript_switch_pokemon.s:11`` before the switch) needs the pursuer alive, not
    asleep or frozen, and PP left in the move (Truant is not implemented)."""
    pursuer = 1 - side
    move, struggling = selected_move(state, pursuer, actions[pursuer])
    mon = state.active[pursuer]
    status = state.status[pursuer, mon]
    return (
        (state.over == 0)
        & is_switch(actions[side])
        & ~is_switch(actions[pursuer])
        & ~struggling
        & (tables.move_effect[move] == EffectId.PURSUIT)
        & (state.hp[pursuer, mon] > 0)
        & (status != Status.SLEEP)
        & (status != Status.FREEZE)
    )


def _pursuit_hit(
    tables: Tables, state: BattleState, pursuer, actions, outcome: Outcome
) -> BattleState:
    """The pursuer hits the outgoing Pokemon: ``BtlCmd_TryPursuit`` spends the PP and
    sets ``powerMul = 20``; ``subscript_pursuit.s`` then runs crit, damage and the type
    chart with no accuracy check. The pursuer's own action is over."""
    action = actions[pursuer]
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    mon = state.active[pursuer]
    state = state._replace(pp=state.pp.at[pursuer, mon, slot].add(jnp.int8(-1)))
    move, _ = selected_move(state, pursuer, action)
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
    Pokemon left standing is a win. A fainted active Pokemon's Bind on the other side
    ends (``BattleSystem_CleanupFaintedMon``, battle_lib.c:2179)."""
    fainted = (before.hp > 0) & (after.hp == 0)
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
        after = _release_bind(after, 1 - side, fainted[side, after.active[side]])
    return after


def execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 3: both actors in order, each followed by a faint check. A switch facing
    Pursuit is hit first (with its own faint check), and the pursuer does not act again."""
    first = order[0].astype(_I32)
    pursued = _pursuit_fires(tables, state, first, actions)
    before = state
    hit = _pursuit_hit(tables, state, 1 - first, actions, outcome)
    state = _faint_check(before, _where(pursued, hit, state))
    for position in (0, 1):
        before = state
        side = order[position].astype(_I32)
        target_moved = position == 1  # the first actor's target has not moved yet
        acted = _act(tables, state, side, actions, outcome, target_moved)
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
# in its order; Task 8d inserts poison, toxic and burn between Leech Seed and Bind.
END_OF_TURN_ORDER = (
    # MON_COND_CHECK_STATE_LEECH_SEED, battle_controller_player.c:1378
    ("leech_seed", _leech_seed_check),
    # MON_COND_CHECK_STATE_BIND, battle_controller_player.c:1462
    ("bind", _bind_check),
    # MON_COND_CHECK_STATE_TAUNT, battle_controller_player.c:1635
    ("taunt", _taunt_check),
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


def replace(state: BattleState, side, slot) -> BattleState:
    """Send in party ``slot`` for ``side``'s fainted active Pokemon. Entry hazards on a
    replacement belong to Task 8e's ``hooks.on_switch_in`` (it needs ``tables``)."""
    return _switch_in(state, side, slot)
