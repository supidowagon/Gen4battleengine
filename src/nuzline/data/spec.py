"""A party member as the battle engine needs it, before stats are computed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PokemonSpec:
    """Ids are the decomp's own (species, nature, move, item).

    ``ivs``/``evs`` are in ``ids.Stat`` order; ``moves`` holds four move ids with
    0 for an empty slot; ``ability_slot`` (0 or 1) indexes the species' two
    abilities; ``gender`` is 0 male, 1 female, 2 none. ``current_hp=None`` means
    full HP; ``status`` is an ``ids.Status``.
    """

    species: int
    level: int
    nature: int
    ivs: tuple[int, int, int, int, int, int]
    evs: tuple[int, int, int, int, int, int]
    moves: tuple[int, int, int, int]
    item: int
    ability_slot: int
    gender: int
    friendship: int
    form: int = 0
    current_hp: int | None = None
    status: int = 0
