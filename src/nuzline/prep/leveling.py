"""Level-to-cap: the user's Pokemon after Rare Candies, as a Pokemon Center visit leaves it."""

from __future__ import annotations

import dataclasses

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.saves.pkm import SaveMon, _as_np
from nuzline.saves.save import SaveTrainer


def candies_to_cap(level: int, cap: int) -> int:
    """Rare Candies to reach ``cap`` (one per level); 0 at or above the cap."""
    return max(0, cap - level)


def level_to_cap(
    mon: SaveMon, tables: Tables, cap: int, trainer: SaveTrainer | None = None
) -> PokemonSpec:
    """``mon`` at level ``max(level, cap)``: stats recomputed from the level (never evolved),
    full HP, no status, and every move at its maximum PP with its PP Ups
    (``MoveTable_CalcMaxPP``: ``move_pp * (5 + pp_ups) // 5``). Moves are unchanged.
    Raises what :meth:`SaveMon.to_spec` raises; ``trainer`` resolves ``exp_boost`` there.

    Exp: a Rare Candy adds ``Pokemon_GetExpToNextLevel`` (item_use_pokemon.c:291-293), the
    next level's base minus the exp (pokemon.c:2308-2319), so a raised Pokemon has exactly
    the base exp of the cap (``exp_table[rate(species)][cap]``, the rate by species as
    ``Pokemon_GetSpeciesBaseExpAt`` reads it, pokemon.c:2327-2330). One not raised keeps
    the save's exp."""
    spec = mon.to_spec(tables, trainer, healed=True)  # its status is dropped below
    move_pp = _as_np(tables.move_pp)
    pp = tuple(
        int(move_pp[m]) * (5 + ups) // 5 if m else 0
        for m, ups in zip(spec.moves, mon.pp_ups, strict=True)
    )
    if cap > spec.level:
        rate = int(_as_np(tables.species_exp_rate)[mon.species])
        exp = int(_as_np(tables.exp_table)[rate, cap])
    else:
        exp = spec.exp
    return dataclasses.replace(
        spec,
        level=max(spec.level, cap),
        exp=exp,
        stats=None,
        current_hp=None,
        status=0,
        status_counter=0,
        pp=pp,
    )
