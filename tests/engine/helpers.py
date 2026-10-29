"""Builders for engine tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace as _replace_spec
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.party import load_party
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables, load_names
from nuzline.data.trainer import (
    IntermediateData,
    Trainer,
    build_trainer_party,
    default_moves,
    load_trainers,
)
from nuzline.engine.ids import AbilityId, EffectId, Stat
from nuzline.engine.state import BattleState, make_state

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = ROOT / "data" / "intermediate"
SAMPLE_PARTY = ROOT / "tests" / "fixtures" / "sample_party.toml"
_GENDERS = {"male": 0, "female": 1, "none": 2}  # BattleState.gender
# The fixture moveset every opponent gets (plan ruling): raw opponent action 0 is Tackle.
OPP_FIXTURE_MOVES = ("Tackle", "Rock Throw", "Stealth Rock", "Leer")


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


@lru_cache(maxsize=1)
def _roark() -> Trainer:
    if not (INTERMEDIATE / "trainers.json").exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")
    return next(t for t in load_trainers(INTERMEDIATE) if t.name.lower() == "roark")


@lru_cache(maxsize=1)
def _roark_party() -> tuple[PokemonSpec, ...]:
    return tuple(build_trainer_party(_roark(), IntermediateData(INTERMEDIATE)))


@lru_cache(maxsize=1)
def _inter() -> IntermediateData:
    return IntermediateData(INTERMEDIATE)


def _with_moves(mon: PokemonSpec, wanted: Sequence[int]) -> PokemonSpec:
    """``wanted`` first, then ``mon``'s own moves in order (duplicates skipped),
    up to four."""
    moves: list[int] = []
    for m in (*wanted, *mon.moves):
        if m and m not in moves:
            moves.append(m)
    moves = moves[:4]
    return _replace_spec(mon, moves=tuple(moves + [0] * (4 - len(moves))))


def _lead(party: list[PokemonSpec], species: int, inter: IntermediateData) -> list[PokemonSpec]:
    """Put ``species`` in slot 0: the party member of that species is swapped in
    (keeping its moves, item and ability); otherwise slot 0 becomes that species
    at the same level, with its default learnset moveset, ability slot 0 and no item."""
    party = list(party)
    for i, mon in enumerate(party):
        if mon.species == species:
            party[0], party[i] = mon, party[0]
            return party
    party[0] = _replace_spec(
        party[0],
        species=species,
        moves=default_moves(inter.learnsets[species], party[0].level),
        item=0,
        ability_slot=0,
    )
    return party


def roark_state(
    tables: Tables,
    names: Names,
    *,
    equal_speed: bool = False,
    opp_count: int = 3,
    player_species: str | None = None,
    player_moves: Sequence[str] | None = None,
    player_ability: str | None = None,
    player_gender: str | None = None,
    opp_species: str | None = None,
    opp_moves: Sequence[str] | None = None,
    opp_gender: str | None = None,
    opp_faster: bool = False,
) -> BattleState:
    """The sample party (``tests/fixtures/sample_party.toml``) against the first
    ``opp_count`` members of Roark's built party (Geodude L12, Onix L12, Cranidos
    L14), both leads in slot 0, Roark's bag (two Potions) in ``trainer_items[1]``.
    A test fixture, not Roark's real fight: every
    opponent knows Tackle, Rock Throw, Stealth Rock, Leer in that order (plan
    ruling).

    Overrides (plan rulings): ``player_species``/``opp_species`` put that species
    in slot 0 (``_lead``); ``player_moves``/``opp_moves`` go first in slot 0's
    moveset, the rest filled from its current moves (the fixture list for the
    opponent) skipping duplicates; ``player_ability`` must be one of the lead's
    species abilities; ``player_gender``/``opp_gender`` ("male", "female" or "none")
    set slot 0's gender. ``equal_speed`` sets the opponent lead's Spe stat to the
    player lead's, ``opp_faster`` to one more.
    """
    fixture_moves = tuple(names.move(m) for m in OPP_FIXTURE_MOVES)
    opp = [_replace_spec(mon, moves=fixture_moves) for mon in _roark_party()[:opp_count]]
    inter = _inter()
    player = load_party(SAMPLE_PARTY, names)
    if player_species is not None:
        player = _lead(player, names.species(player_species), inter)
    if opp_species is not None:
        opp = _lead(opp, names.species(opp_species), inter)
        opp[0] = _replace_spec(opp[0], moves=fixture_moves)
    if player_moves is not None:
        player[0] = _with_moves(player[0], [names.move(m) for m in player_moves])
    if opp_moves is not None:
        opp[0] = _with_moves(opp[0], [names.move(m) for m in opp_moves])
    if player_ability is not None:
        ability = names.ability(player_ability)
        slots = names.species_abilities(player[0].species)
        if ability == 0 or ability not in slots:
            raise ValueError(f"species {player[0].species} cannot have {player_ability!r}")
        player[0] = _replace_spec(player[0], ability_slot=slots.index(ability))
    if player_gender is not None:
        player[0] = _replace_spec(player[0], gender=_GENDERS[player_gender])
    if opp_gender is not None:
        opp[0] = _replace_spec(opp[0], gender=_GENDERS[opp_gender])
    s = make_state(tables, player, opp, opp_items=_roark().items)
    spe = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe
    if equal_speed or opp_faster:
        s = s._replace(stats=s.stats.at[1, 0, spe].set(s.stats[0, 0, spe] + int(opp_faster)))
    return s


def roark_real_state(
    tables: Tables, names: Names, *, player_species: str | None = None
) -> BattleState:
    """The sample party (``player_species`` swapped into slot 0, as in
    ``roark_state``) against Roark's real built party (Geodude, Onix, Cranidos with
    their own moves; docs/ai/roark.md section 4) and bag: the common setup of the
    roark.md section 13 traces."""
    player = load_party(SAMPLE_PARTY, names)
    if player_species is not None:
        player = _lead(player, names.species(player_species), _inter())
    return make_state(tables, player, list(_roark_party()), opp_items=_roark().items)


def A(player, opp, state: BattleState, names: Names) -> jnp.ndarray:
    """``int32[2]`` actions: a move name becomes that move's slot on the side's
    active Pokemon (``Names.move_slot``); an int is used as the raw action."""

    def one(side: int, a) -> int:
        return names.move_slot(state, side, a) if isinstance(a, str) else int(a)

    return jnp.array([one(0, player), one(1, opp)], jnp.int32)
