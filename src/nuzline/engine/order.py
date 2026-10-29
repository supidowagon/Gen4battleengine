"""Turn order and the move each action selects (pure ``jnp``, jit/vmap-able).

Actions are ``int32`` scalars: ``0-3`` use that move slot, ``4-9`` switch to party
slot ``action - 4``. ``action_order`` follows ``BattleControllerPlayer_CalcTurnOrder``
(``src/battle/battle_controller_player.c:707``): items and switches first, then
moves; two actions of the same kind are swapped when
``BattleSystem_CompareBattlerSpeed`` (``src/battle/battle_lib.c:1188``) says the
player's side does not go first (priority only between two moves, ``:1401``).

Speed modifiers of the implemented set only: stages (Simple doubled, ``:1222``),
paralysis ``/ 4`` (``:1260``), Tailwind ``x 2`` (``:1274``) and Trick Room's
reversal (``:1432``). Quick Claw, Custap, Lagging Tail, Stall, Quick Feet, Swift
Swim, Chlorophyll, Slow Start, Unburden, Choice Scarf and the speed-halving items
belong to unsupported abilities and items (``UNSUPPORTED`` ids), so they never apply.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS, usable_moves
from nuzline.engine.damage import apply_stage, simple_stage
from nuzline.engine.ids import MOVE_STRUGGLE, AbilityId, Stat, Status
from nuzline.engine.state import BattleState

_I32 = jnp.int32


def is_switch(action) -> jnp.ndarray:
    return jnp.asarray(action) >= MOVE_ACTIONS


def selected_move(
    tables: Tables, state: BattleState, side, action
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(move id, struggling)`` for a move action by ``side``'s active Pokemon in the
    state it was chosen in: the slot's move, or Struggle when no slot is usable
    (``actions.usable_moves``: no move, no PP, or power 0 while taunted), the decomp's
    ``STRUGGLING_ALL`` (battle_controller_player.c:433). The legality mask then only
    allows action 0. Meaningless for a switch action (callers mask it)."""
    slot = jnp.clip(jnp.asarray(action, _I32), 0, MOVE_ACTIONS - 1)
    mon = state.active[side]
    struggling = ~jnp.any(usable_moves(tables, state, side))
    move = jnp.where(struggling, MOVE_STRUGGLE, state.moves[side, mon, slot].astype(_I32))
    return move, struggling


def effective_speed(state: BattleState, side) -> jnp.ndarray:
    """``int32`` speed as ``BattleSystem_CompareBattlerSpeed`` computes it."""
    mon = state.active[side]
    stage = state.stages[side, Stat.SPE].astype(_I32)
    stage = jnp.where(state.ability[side, mon] == AbilityId.SIMPLE, simple_stage(stage), stage)
    speed = apply_stage(state.stats[side, mon, Stat.SPE - 1].astype(_I32), stage)
    speed = jnp.where(state.status[side, mon] == Status.PARALYSIS, speed // 4, speed)
    return jnp.where(state.tailwind[side] != 0, speed * 2, speed)


def action_order(tables: Tables, state: BattleState, actions, outcome) -> jnp.ndarray:
    """``int8[2]``: the two sides in execution order. ``outcome.speed_tie`` decides
    equal speeds (0 = player first): the decomp swaps on ``RandNext() & 1``."""
    actions = jnp.asarray(actions, _I32)
    switch = is_switch(actions)
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    both_moves = ~switch[0] & ~switch[1]
    priority = jnp.where(both_moves, tables.move_priority[moves].astype(_I32), 0)
    speed0, speed1 = effective_speed(state, 0), effective_speed(state, 1)
    tie = (speed0 == speed1) & (outcome.speed_tie != 0)
    slower = jnp.where(state.trick_room != 0, speed0 > speed1, speed0 < speed1)
    player_second = (priority[0] < priority[1]) | ((priority[0] == priority[1]) & (slower | tie))
    # A switch always precedes a move; otherwise compare the two actions.
    player_second = jnp.where(switch[0] == switch[1], player_second, switch[1])
    first = player_second.astype(jnp.int8)
    return jnp.stack([first, 1 - first]).astype(jnp.int8)
