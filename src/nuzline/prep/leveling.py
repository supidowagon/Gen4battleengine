"""Level-to-cap: the user's Pokemon after Rare Candies, as a Pokemon Center visit leaves it."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from nuzline.data.gauntlet import Settings
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.prep.growth import (
    EVO_CLASS_BY_ITEM,
    EVO_CLASS_BY_LEVEL,
    GrowthPlan,
    evolution_target,
    evolve,
    rare_candy,
    use_stone,
)
from nuzline.saves.pkm import SaveMon, _as_np
from nuzline.saves.save import SaveTrainer


def candies_to_cap(level: int, cap: int) -> int:
    """Rare Candies to reach ``cap`` (one per level); 0 at or above the cap."""
    return max(0, cap - level)


@dataclasses.dataclass(frozen=True)
class PreFightGrowth:
    """The inputs of pre-fight growth (:func:`grow_to_cap`; GT-14, GT-15; ledger E3-B3-14).

    ``plan``: the member's growth plan; every learn prompt is answered by it, and the member
    evolves when ``plan.evolve_at == gap`` (pre-fight growth is gap 0, before the first fight),
    at the earliest point of the gap where an evolution is offered: with the bag holding a
    stone the member's species takes, before the first candy; else at the first level whose
    check passes. ``candy``: the Rare Candy item id (its friendship params come from the item
    table). ``bag``: item id -> count (stones, Poke Ball); ``party``: party 0's other members
    (``Party_HasSpecies``, the Shedinja room); ``map_method`` and ``time`` as
    :func:`~nuzline.prep.growth.evolution_target`; ``location`` the map the candies are used on
    (the egg-location friendship bonus; ``None`` none)."""

    learnsets: Learnsets
    candy: int
    plan: GrowthPlan
    settings: Settings
    personality: int
    beauty: int = 0
    map_method: int = 0
    time: str | None = None
    location: int | None = None
    party: Sequence[PokemonSpec] = ()
    bag: Mapping[int, int] = dataclasses.field(default_factory=dict)
    gap: int = 0


class Grown(NamedTuple):
    """The result of :func:`grow_to_cap`: the member, an added Shedinja (or ``None``), the bag
    afterwards and the Rare Candies used."""

    spec: PokemonSpec
    extra: PokemonSpec | None
    bag: dict[int, int]
    candies: int


def _centre(tables: Tables, spec: PokemonSpec) -> PokemonSpec:
    """``spec`` as a Pokemon Center visit leaves it: full HP, no status, every move at its
    maximum PP at its PP Ups (``MoveTable_CalcMaxPP``)."""
    move_pp = _as_np(tables.move_pp)
    pp = tuple(
        int(move_pp[m]) * (5 + ups) // 5 if m else 0
        for m, ups in zip(spec.moves, spec.pp_ups, strict=True)
    )
    return dataclasses.replace(spec, stats=None, current_hp=None, status=0, status_counter=0, pp=pp)


def _stone_in_bag(
    tables: Tables, mon: PokemonSpec, bag: Mapping[int, int], growth: PreFightGrowth
) -> int | None:
    """The first stone of ``mon``'s species rows that the bag holds and that evolves it."""
    learnsets = growth.learnsets
    for method, param, _ in learnsets.evolutions(mon.species):
        if not learnsets.evolution_methods[method].startswith("use_item") or bag.get(param, 0) < 1:
            continue
        found = evolution_target(
            tables,
            learnsets,
            mon,
            None,
            EVO_CLASS_BY_ITEM,
            param,
            None,
            personality=growth.personality,
            beauty=growth.beauty,
        )
        if found is not None:
            return param
    return None


def grow_to_cap(tables: Tables, spec: PokemonSpec, cap: int, growth: PreFightGrowth) -> Grown:
    """``spec`` grown to ``cap`` by Rare Candies with every prompt answered by the plan: the
    stone first where the plan evolves now and the bag holds one, then per candy
    (:func:`~nuzline.prep.growth.rare_candy`: level, friendship, learn prompts) the evolution
    check (``EVO_CLASS_BY_LEVEL``, ``src/applications/party_menu/callbacks.c:816-829``). A candy
    raises the experience to the next level on the species' table as it stands, so the exp rule
    of :func:`level_to_cap` holds for the species the Pokemon ends as. The result is healed
    (:func:`_centre`)."""
    plan, learnsets = growth.plan, growth.learnsets
    pending = plan.evolve_at == growth.gap
    bag = dict(growth.bag)
    others = list(growth.party)
    mon, extra, candies = spec, None, 0
    stone = _stone_in_bag(tables, mon, bag, growth) if pending else None
    if stone is not None:
        mon, extra, bag = use_stone(
            tables,
            learnsets,
            mon,
            stone,
            bag,
            [*others, mon],
            growth.settings,
            personality=growth.personality,
            learn_answers=plan.answers_at(mon.level),
        )
        pending = False
    while mon.level < cap:
        mon = rare_candy(
            tables,
            learnsets,
            mon,
            growth.candy,
            location=growth.location,
            learn_answers=plan.answers_at(mon.level + 1),
        )
        candies += 1
        found = None
        if pending:
            found = evolution_target(
                tables,
                learnsets,
                mon,
                [*others, mon],
                EVO_CLASS_BY_LEVEL,
                growth.map_method,
                growth.time,
                personality=growth.personality,
                beauty=growth.beauty,
            )
        if found is not None:
            mon, added, bag = evolve(
                tables,
                learnsets,
                mon,
                found[0],
                found[1],
                bag,
                [*others, mon],
                growth.settings,
                personality=growth.personality,
                learn_answers=plan.answers_at(mon.level),
            )
            extra = added if added is not None else extra
            pending = False
    return Grown(
        _centre(tables, mon), None if extra is None else _centre(tables, extra), bag, candies
    )


def level_to_cap(
    mon: SaveMon,
    tables: Tables,
    cap: int,
    trainer: SaveTrainer | None = None,
    *,
    growth: PreFightGrowth | None = None,
) -> PokemonSpec:
    """``mon`` at level ``max(level, cap)``: stats recomputed from the level (never evolved
    without ``growth``), full HP, no status, and every move at its maximum PP with its PP Ups
    (``MoveTable_CalcMaxPP``: ``move_pp * (5 + pp_ups) // 5``). Moves are unchanged without
    ``growth``. Raises what :meth:`SaveMon.to_spec` raises; ``trainer`` resolves ``exp_boost``
    there.

    Exp: a Rare Candy adds ``Pokemon_GetExpToNextLevel`` (item_use_pokemon.c:291-293), the
    next level's base minus the exp (pokemon.c:2308-2319), so a raised Pokemon has exactly
    the base exp of the cap (``exp_table[rate(species)][cap]``, the rate by species as
    ``Pokemon_GetSpeciesBaseExpAt`` reads it, pokemon.c:2327-2330). One not raised keeps
    the save's exp.

    ``growth`` (GT-14, GT-15): the Rare Candy levels run through :func:`grow_to_cap`, so the
    learn prompts, the candies' friendship and the evolution follow the plan; the exp rule holds
    for the species the Pokemon ends as. An added Shedinja cannot be returned here:
    ``ValueError``, use :func:`grow_to_cap`."""
    spec = mon.to_spec(tables, trainer, healed=True)  # its status is dropped below
    if growth is not None:
        grown = grow_to_cap(tables, spec, cap, growth)
        if grown.extra is not None:
            raise ValueError("the plan adds a Shedinja: use grow_to_cap")
        return grown.spec
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
