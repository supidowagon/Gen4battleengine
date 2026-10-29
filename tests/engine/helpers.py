"""Builders for engine tests."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables, load_names
from nuzline.engine.ids import AbilityId, EffectId
from nuzline.engine.state import BattleState, make_state

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category


@lru_cache(maxsize=1)
def _names() -> Names:
    return load_names()


def find_move(tables: Tables, power: int, type_id: int, category: int) -> int:
    """First supported move (lowest id) with this power, type and category."""
    power_ = np.asarray(tables.move_power)
    type_ = np.asarray(tables.move_type)
    cat = np.asarray(tables.move_category)
    effect = np.asarray(tables.move_effect)
    ids = np.flatnonzero(
        (power_ == power)
        & (type_ == type_id)
        & (cat == category)
        & (effect != EffectId.UNSUPPORTED)
    )
    ids = ids[ids > 0]
    if ids.size == 0:
        raise ValueError(
            f"no supported move with power {power}, type {type_id}, category {category}"
        )
    return int(ids[0])


def _spec(level: int, move: int) -> PokemonSpec:
    return PokemonSpec(
        species=1,
        level=level,
        nature=0,  # Hardy: neutral
        ivs=(0,) * 6,
        evs=(0,) * 6,
        moves=(move, 0, 0, 0),
        item=0,
        ability_slot=0,
        gender=2,
        friendship=70,
    )


def synthetic_state(
    tables: Tables,
    *,
    atk_level: int,
    atk_stats: Sequence[int],
    def_stats: Sequence[int],
    move_power: int,
    move_physical: bool,
    atk_types: tuple[str, str],
    def_types: tuple[str, str],
    move_type: str,
    def_level: int = 50,
    atk_ability: int = AbilityId.NONE,
    def_ability: int = AbilityId.NONE,
    atk_gender: int = 2,
    def_gender: int = 2,
) -> tuple[BattleState, int]:
    """One active Pokemon per side (player attacks, side 0). Stats are given as
    Atk, Def, SpA, SpD, Spe and written over the computed ones; types, abilities
    and genders (default genderless) are overridden too. The move is a real move
    from the tables with this power, type and category (``move_power == 0``
    picks a status move); the tables are not modified. Returns (state, move id).
    """
    names = _names()
    category = STATUS if move_power == 0 else (PHYSICAL if move_physical else SPECIAL)
    move = find_move(tables, move_power, names.type(move_type), category)
    s = make_state(tables, [_spec(atk_level, move)], [_spec(def_level, 0)])
    stats = np.asarray(s.stats).copy()
    stats[0, 0] = atk_stats
    stats[1, 0] = def_stats
    types = np.asarray(s.types).copy()
    types[0, 0] = [names.type(t) for t in atk_types]
    types[1, 0] = [names.type(t) for t in def_types]
    ability = np.asarray(s.ability).copy()
    ability[:, 0] = [atk_ability, def_ability]
    gender = np.asarray(s.gender).copy()
    gender[:, 0] = [atk_gender, def_gender]
    s = s._replace(
        stats=s.stats.at[...].set(stats),
        types=s.types.at[...].set(types),
        ability=s.ability.at[...].set(ability),
        gender=s.gender.at[...].set(gender),
    )
    return s, move
