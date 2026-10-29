"""Legal-action masks for the two decision points: choosing a turn's action and
choosing a replacement after a faint.

Actions are ``int32`` scalars ``0..NUM_ACTIONS-1``: ``0-3`` use the active
Pokemon's move slot, ``4-9`` switch to party slot ``action - 4`` (spec §4.2). The
active slot's own switch is always illegal, so at most 9 actions are legal.

Move legality follows ``BattleSystem_CheckInvalidMoves`` (battle_lib.c:2255) for the
checks M1 has state for: an empty slot (``CHECK_INVALID_NO_MOVE``), no PP
(``CHECK_INVALID_NO_PP``) and Taunt (``CHECK_INVALID_TAUNTED``: a taunted battler
cannot pick a move whose power is 0). Disable, Torment, Imprison, Gravity and Heal
Block come with their mechanics. When every slot is invalid the decomp sets
``struggling`` instead of letting the player pick a move
(battle_controller_player.c:433); here action 0 stays legal and means Struggle.

Switches are legal to a non-active, non-empty, non-fainted slot, and never while the
active Pokemon is trapped (``trapped > 0``, Bind and friends).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.state import SLOTS, BattleState

NUM_ACTIONS = MOVE_ACTIONS + SLOTS


def legal_replacements(state: BattleState, side) -> jnp.ndarray:
    """``bool[6]``: party slots ``side`` may send in, non-active, non-empty and not
    fainted."""
    slots = jnp.arange(SLOTS)
    return (slots != state.active[side]) & (state.species[side] > 0) & (state.hp[side] > 0)


def legal_actions(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[NUM_ACTIONS]``: the actions ``side`` may choose this turn."""
    mon = state.active[side]
    moves = state.moves[side, mon]
    usable = (moves > 0) & (state.pp[side, mon] > 0)
    taunted = state.taunt[side] > 0
    usable = usable & ~(taunted & (tables.move_power[moves] == 0))
    struggle = ~jnp.any(usable)
    usable = usable.at[0].set(usable[0] | struggle)
    switches = legal_replacements(state, side) & (state.trapped[side] == 0)
    return jnp.concatenate([usable, switches])
