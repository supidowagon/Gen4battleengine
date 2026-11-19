"""Builders for the D2 tests: four-battler states and per-battler move edits."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.spec import PokemonSpec
from nuzline.engine.format import DOUBLES, Format
from nuzline.engine.ids import Type
from nuzline.engine.state import BattleState, initial_arrays

DEFAULT_MOVES = ("Tackle", "Swords Dance", "Earthquake", "Curse")


def mon(names, moves=DEFAULT_MOVES, level: int = 50) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species("Bulbasaur"),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=0,
        gender=2,
        friendship=70,
    )


def four(tables, names, fmt: Format = DOUBLES, bench: int = 6) -> BattleState:
    """A turn-0 state of ``fmt`` with ``bench`` Pokemon (at most six) in each party, every battler
    leading from the first slots of its party, all with ``DEFAULT_MOVES``."""
    members = [mon(names) for _ in range(bench)]
    parties = [list(members) for _ in range(fmt.P)]
    a = initial_arrays(fmt, tables, parties, opp2_items=())
    return BattleState(**{k: jnp.asarray(v) for k, v in a.items()})


def party_of(state: BattleState, b: int) -> int:
    from nuzline.engine.format import format_of

    return format_of(state).party_of[b]


def with_moves(state: BattleState, names, b: int, moves, pp: int = 10) -> BattleState:
    """Battler ``b``'s active Pokemon knows ``moves`` (names, at most four) with ``pp`` each."""
    ids = [names.move(m) for m in moves]
    ids += [0] * (4 - len(ids))
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(
        moves=state.moves.at[p, slot].set(jnp.asarray(ids, state.moves.dtype)),
        pp=state.pp.at[p, slot].set(jnp.asarray([pp if m else 0 for m in ids], state.pp.dtype)),
    )


def with_hp(state: BattleState, b: int, hp: int) -> BattleState:
    """Battler ``b``'s active Pokemon has ``hp`` HP."""
    return state._replace(
        hp=state.hp.at[party_of(state, b), int(state.active[b])].set(hp),
    )


def as_ghost(state: BattleState, b: int) -> BattleState:
    """Battler ``b``'s active Pokemon is Ghost-typed."""
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(
        types=state.types.at[p, slot].set(jnp.asarray([Type.GHOST, Type.GHOST], state.types.dtype))
    )


def with_ability(state: BattleState, b: int, ability) -> BattleState:
    """Battler ``b``'s active Pokemon has ``ability``."""
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(ability=state.ability.at[p, slot].set(int(ability)))
