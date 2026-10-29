"""One battle turn: ``step(tables, state, actions, outcome) -> state``.

Pure, jit- and vmap-able, fixed shapes; every random event comes from
``outcome`` (``step`` draws nothing). The turn follows spec §4.2's six phases:

1. The opponent's choice is made outside ``step`` (``actions`` is an input).
2. Order: ``order.action_order``.
3. ``execute_actions``: the first actor, a faint check, the second actor, a faint
   check (``BattleControllerPlayer_CheckAnyFainted`` after each action).
4. ``end_of_turn``: end-of-turn effects (none are implemented yet; Task 8c).
5. Faint replacement is a separate decision point outside ``step``:
   ``needs_replacement`` and ``replace``.
6. ``check_battle_over``.

Implemented so far: switching, damaging moves (every effect id behaves as a plain
hit until Tasks 8b-8e), the accuracy check, PP, Struggle, faints and the end of
the battle. A player faint is terminal (spec §3): it sets ``over = 1, winner = 1``.
A state that is already over is returned unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.damage import damage_for_roll
from nuzline.engine.order import MOVE_ACTIONS, action_order, is_switch, selected_move
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import BattleState

_I32 = jnp.int32
WINNER_OPPONENT, WINNER_PLAYER = 1, 2

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


def _switch_in(state: BattleState, side, slot) -> BattleState:
    """``side`` sends out party ``slot``: stages and volatiles reset."""
    fields = {name: getattr(state, name).at[side].set(0) for name in _VOLATILE}
    return state._replace(
        active=state.active.at[side].set(jnp.asarray(slot).astype(jnp.int8)),
        stages=state.stages.at[side].set(0),
        **fields,
    )


def _use_move(tables: Tables, state: BattleState, side, action, outcome: Outcome) -> BattleState:
    """``side``'s active Pokemon uses the move in slot ``action`` (or Struggle)."""
    move, struggling = selected_move(state, side, action)
    slot = jnp.clip(action, 0, MOVE_ACTIONS - 1)
    user = state.active[side]
    target_side = 1 - side
    target = state.active[target_side]
    can_act = (state.over == 0) & (state.hp[side, user] > 0) & (state.flinched[side] == 0)

    # BattleControllerPlayer_DecrementPP (battle_controller_player.c:3182) runs before the
    # no-target check (:3190); Struggle has no PP to spend.
    spend = can_act & ~struggling
    pp = state.pp.at[side, user, slot].add(jnp.where(spend, -1, 0).astype(state.pp.dtype))

    # Accuracy 0 never misses (CheckMoveHitAccuracy returns before rolling, :2901).
    never_misses = tables.move_accuracy[move] == 0
    lands = (
        can_act & (state.hp[target_side, target] > 0) & ((outcome.hit[side, 0] != 0) | never_misses)
    )
    damage = damage_for_roll(
        tables, state, side, move, outcome.crit[side, 0] != 0, outcome.roll[side, 0]
    )
    damage = jnp.where(lands, damage, 0)
    target_hp = jnp.maximum(state.hp[target_side, target].astype(_I32) - damage, 0)
    hp = state.hp.at[target_side, target].set(target_hp.astype(state.hp.dtype))

    # subscript_struggle.s: recoil BattleSystem_Divide(-maxHP, 4), at least 1, on a hit.
    recoil = jnp.where(
        lands & struggling, jnp.maximum(state.max_hp[side, user].astype(_I32) // 4, 1), 0
    )
    user_hp = jnp.maximum(hp[side, user].astype(_I32) - recoil, 0)
    hp = hp.at[side, user].set(user_hp.astype(state.hp.dtype))
    return state._replace(pp=pp, hp=hp)


def _act(tables: Tables, state: BattleState, side, actions, outcome: Outcome) -> BattleState:
    action = actions[side]
    switched = _switch_in(state, side, jnp.clip(action - MOVE_ACTIONS, 0, 5))
    moved = _use_move(tables, state, side, action, outcome)
    return _where(is_switch(action) & (state.over == 0), switched, moved)


def _faint_check(before: BattleState, after: BattleState) -> BattleState:
    """Count new player faints and end the battle: any player faint is a loss (the
    terminal rule, which wins over a simultaneous last opponent faint); no opponent
    Pokemon left standing is a win."""
    fainted = (before.hp > 0) & (after.hp == 0)
    player_fainted = jnp.sum(fainted[0].astype(_I32))
    opp_out = jnp.all((after.species[1] == 0) | (after.hp[1] == 0))
    ended = (after.over == 0) & ((player_fainted > 0) | opp_out)
    winner = jnp.where(player_fainted > 0, WINNER_OPPONENT, WINNER_PLAYER)
    return after._replace(
        player_faints=(after.player_faints + player_fainted).astype(after.player_faints.dtype),
        over=jnp.where(ended, 1, after.over).astype(after.over.dtype),
        winner=jnp.where(ended, winner, after.winner).astype(after.winner.dtype),
    )


def execute_actions(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 3: both actors in order, each followed by a faint check."""
    for position in (0, 1):
        before = state
        state = _act(tables, state, order[position].astype(_I32), actions, outcome)
        state = _faint_check(before, state)
    return state


def end_of_turn(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order
) -> BattleState:
    """Phase 4: end-of-turn effects. Flinch lasts only for the turn it is set in."""
    return state._replace(flinched=jnp.zeros_like(state.flinched))


def check_battle_over(
    tables: Tables, state: BattleState, actions, outcome: Outcome, order, *, before
) -> BattleState:
    """Phase 6: faints from end-of-turn effects (``before`` is the state entering
    phase 4) end the battle by the same rules as faints during the turn."""
    return _faint_check(before, state)


def step(tables: Tables, state: BattleState, actions, outcome: Outcome) -> BattleState:
    """Play one turn: ``actions`` is ``int32[2]`` (player, opponent; see ``order``)."""
    actions = jnp.asarray(actions, _I32)
    order = action_order(tables, state, actions, outcome)
    s = execute_actions(tables, state, actions, outcome, order)
    after_actions = s
    s = end_of_turn(tables, s, actions, outcome, order)
    s = check_battle_over(tables, s, actions, outcome, order, before=after_actions)
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
    """Send in party ``slot`` for ``side``'s fainted active Pokemon."""
    return _switch_in(state, side, slot)
