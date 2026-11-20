"""Builders for engine tests."""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import replace as _replace_spec
from functools import lru_cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.games import KAIZO
from nuzline.data.party import load_party
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables, load_names, to_numpy
from nuzline.data.trainer import (
    IntermediateData,
    Trainer,
    build_trainer_party,
    default_moves,
    load_trainers,
)
from nuzline.engine.ids import (
    FIELD_CONDITION_GRAVITY_SHIFT,
    FIELD_CONDITION_TRICK_ROOM_SHIFT,
    WEATHER_PERM_BIT,
    WEATHER_TEMP_BIT,
    AbilityId,
    EffectId,
    FieldCondition,
    Stat,
    Weather,
)
from nuzline.engine.outcome import Outcome, outcome_to_py, replacement_known, sample_outcome
from nuzline.engine.state import BattleState, make_state
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step

PHYSICAL, SPECIAL, STATUS = 0, 1, 2  # tables.move_category

ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE = ROOT / "data" / "intermediate"
SAMPLE_PARTY = ROOT / "tests" / "fixtures" / "sample_party.toml"
_GENDERS = {"male": 0, "female": 1, "none": 2}  # BattleState.gender
# The fixture moveset every opponent gets (plan ruling): raw opponent action 0 is Tackle.
OPP_FIXTURE_MOVES = ("Tackle", "Rock Throw", "Stealth Rock", "Leer")


# Reads of a clean ``field_mask`` (FCM 4.4: at most one weather group) for assertions; any
# shape (a state or a batch of states).
def weather_of(s) -> np.ndarray:
    """The ``Weather`` id of the one set weather group (TEMP or PERM; deep fog: bit 15), 0 with
    none."""
    mask = np.asarray(s.field_mask)
    out = np.zeros(mask.shape, np.int64)
    for weather in Weather:
        bits = int(WEATHER_TEMP_BIT[weather]) | int(WEATHER_PERM_BIT[weather])
        out = np.where(mask & bits, int(weather), out)
    return out


def weather_timer_of(s) -> np.ndarray:
    """``weather_turns`` while the weather is a TEMP-only group (a weather move's), else 0 (a
    permanent weather, deep fog or none): FCM 4.4's ``where(temp_only, weather_turns, 0)``."""
    mask = np.asarray(s.field_mask)
    temp = sum(int(WEATHER_TEMP_BIT[w]) for w in Weather if w != Weather.FOG)
    perm = sum(int(WEATHER_PERM_BIT[w]) for w in Weather if w != Weather.FOG)
    temp_only = ((mask & temp) != 0) & ((mask & perm) == 0)
    return np.where(temp_only, np.asarray(s.weather_turns), 0)


def gravity_of(s) -> np.ndarray:
    """The Gravity counter, bits 12-14 (condition.h:119-121)."""
    mask = np.asarray(s.field_mask)
    return (mask & int(FieldCondition.GRAVITY)) >> FIELD_CONDITION_GRAVITY_SHIFT


def trick_room_of(s) -> np.ndarray:
    """The Trick Room counter, bits 16-18 (condition.h:123-125)."""
    mask = np.asarray(s.field_mask)
    return (mask & int(FieldCondition.TRICK_ROOM)) >> FIELD_CONDITION_TRICK_ROOM_SHIFT


# Shared programs: every test module steps through these objects, so a program compiled in one
# module serves the next (the conftest keeps JAX's in-memory programs across modules until the
# process passes its memory threshold). ``jit_step`` is ``step`` compiled: an eager ``step``
# call runs the whole engine op by op and compiles each control-flow branch on its own (one
# eager module reached 16 GB). ``BATCH`` is the suite's one padded row count for vmapped
# programs: ``step_rows`` and ``sample_rows`` pad every call to it, so one program per kind
# serves every file whatever its row count.
BATCH = 1024
jit_step = jax.jit(step)
step_batch = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))


def _padded_calls(n: int):
    """``(rows, padded index)`` per ``BATCH``-row call over ``n`` rows (padding: row 0 of the
    call repeated); a row's result does not depend on the call it is in."""
    if n < 1:
        raise ValueError(f"padded calls need at least one row, got {n}")
    for start in range(0, n, BATCH):
        idx = np.arange(start, min(start + BATCH, n))
        yield idx, jnp.asarray(np.concatenate([idx, np.full(BATCH - len(idx), idx[0])]))


def _joined(parts, counts):
    """The calls' real rows, in order (the padding dropped)."""
    parts = [jax.tree.map(lambda f, n=n: f[:n], p) for p, n in zip(parts, counts, strict=True)]
    return parts[0] if len(parts) == 1 else jax.tree.map(lambda *fs: jnp.concatenate(fs), *parts)


def padded_rows(program, tables: Tables, state: BattleState, actions, outcomes: Outcome):
    """``program`` (a jitted vmap over the outcome rows, as ``step_batch``) of every outcome
    row, in ``BATCH``-row calls: one compile whatever the row count."""
    calls = list(_padded_calls(len(outcomes.hit)))
    parts = [
        program(tables, state, actions, jax.tree.map(lambda f, p=pad: jnp.asarray(f)[p], outcomes))
        for _, pad in calls
    ]
    return _joined(parts, [len(idx) for idx, _ in calls])


def step_rows(tables: Tables, state: BattleState, actions, outcomes: Outcome) -> BattleState:
    """``step`` of every outcome row, in ``BATCH``-row calls of one program."""
    return padded_rows(step_batch, tables, state, actions, outcomes)


def drop_programs() -> None:
    """Drop JAX's in-memory programs and traces (``jax.clear_caches``), between two whole-engine
    programs of one test: a compiled loop or turn program keeps gigabytes resident, and the
    next one's compile would add its own peak on top. The persistent compile cache keeps a
    later recompile of a dropped program cheap."""
    jax.clear_caches()
    gc.collect()


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


def unmodelled(tables: Tables, *moves: int) -> Tables:
    """``tables`` with each of ``moves`` given ``EffectId.UNSUPPORTED``: a refusal test's
    unsupported move, made so here rather than read from today's partial coverage (which
    breaks the test as soon as the move is modelled)."""
    effect = jnp.asarray(tables.move_effect)
    for move in moves:
        effect = effect.at[move].set(EffectId.UNSUPPORTED)
    return tables._replace(move_effect=effect)


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


def roark_trainer() -> Trainer:
    """Roark's ``trainers.json`` entry (the first trainer named Roark)."""
    return _roark()


def roark_party(names: Names) -> list[PokemonSpec]:
    """Roark's real built party (Geodude, Onix, Cranidos with their own moves)."""
    del names  # the party comes from the intermediate data; kept for a uniform signature
    return list(_roark_party())


def sample_party(
    names: Names,
    override_moves: dict[int, Sequence[str]] | None = None,
    override_item: dict[int, str] | None = None,
) -> list[PokemonSpec]:
    """``tests/fixtures/sample_party.toml``, with slot ``i``'s moves replaced by
    ``override_moves[i]`` (up to four names) and its held item by ``override_item[i]``."""
    party = load_party(SAMPLE_PARTY, names)
    for i, moves in (override_moves or {}).items():
        ids = [names.move(m) for m in moves][:4]
        party[i] = _replace_spec(party[i], moves=tuple(ids + [0] * (4 - len(ids))))
    for i, item in (override_item or {}).items():
        party[i] = _replace_spec(party[i], item=names.item(item))
    return party


def roark_state(
    tables: Tables,
    names: Names,
    *,
    equal_speed: bool = False,
    player_count: int = 6,
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
    """The first ``player_count`` members of the sample party
    (``tests/fixtures/sample_party.toml``) against the first ``opp_count`` members of
    Roark's built party (Geodude L12, Onix L12, Cranidos L14), both leads in slot 0,
    Roark's bag (two Potions) in ``trainer_items[1]``. A test fixture, not Roark's real
    fight: every
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
    player = player[:player_count]
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


def as_species(
    tables: Tables,
    state: BattleState,
    side: int,
    slot: int,
    species: int,
    ability_slot: int = 0,
    form: int = 0,
    ability_form: int | None = None,
) -> BattleState:
    """``state`` with ``side``'s party ``slot`` made ``species`` in ``form`` and ``ability_slot``
    (``ability_form`` defaults to ``form``), its engine ``ability`` that personal row's: a
    different one reads as a script's write (Trace, Worry Seed; ``damage.decomp_abilities``)."""
    ability_form = form if ability_form is None else ability_form
    row = int(np.asarray(tables.species_form_row)[species, ability_form])
    own = int(np.asarray(tables.species_abilities)[row, ability_slot])
    return state._replace(
        species=state.species.at[side, slot].set(species),
        form=state.form.at[side, slot].set(form),
        ability_form=state.ability_form.at[side, slot].set(ability_form),
        ability_slot=state.ability_slot.at[side, slot].set(ability_slot),
        ability=state.ability.at[side, slot].set(own),
    )


def A(player, opp, state: BattleState, names: Names) -> jnp.ndarray:
    """``int32[2]`` actions: a move name becomes that move's slot on the side's
    active Pokemon (``Names.move_slot``); an int is used as the raw action."""

    def one(side: int, a) -> int:
        return names.move_slot(state, side, a) if isinstance(a, str) else int(a)

    return jnp.array([one(0, player), one(1, opp)], jnp.int32)


def assert_matches_reference(
    tables: Tables, state: BattleState, actions, outcome: Outcome
) -> BattleState:
    """Step ``state`` with the JAX engine (``jit_step``) and the plain-Python
    reference engine (``ref_step`` through ``to_ref`` / ``from_ref``), assert every
    field equal (the field name is in the message), and return the JAX state."""
    acts = jnp.asarray(actions, jnp.int32)
    out = jit_step(tables, state, acts, outcome)
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(state),
            tuple(int(a) for a in np.asarray(acts)),
            outcome_to_py(outcome),
        )
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


EMPTY_COND = {
    "label": "",
    "formats": [],
    "areas": [],
    "moves": [],
    "abilities": [],
    "items": [],
    "trainers": [],
    "trainer_items": False,
}


def kaizo_names_with(manifest, conditions):
    """A Kaizo ``Names`` over the vanilla manifest with these ``rom_change_conditions``."""
    return Names(
        {
            **manifest,
            "game": KAIZO,
            "unclassified_rom_changes": 0,
            "rom_change_conditions": [{**EMPTY_COND, "label": "c", **c} for c in conditions],
        }
    )


@partial(jax.jit, static_argnames="replacement")
def _sample_rows(keys, tables: Tables, state: BattleState, actions, replacement: bool = True):
    return jax.vmap(lambda k: sample_outcome(k, tables, state, actions, replacement))(keys)


def sample_rows(keys, tables: Tables, state: BattleState, actions):
    """``sample_outcome`` of every key (one program per ``replacement`` value): its static
    ``replacement`` from ``outcome.replacement_known`` of the concrete ``state``. In
    ``BATCH``-key calls: one program whatever the key count."""
    replacement = replacement_known(tables, state)
    calls = list(_padded_calls(len(keys)))
    parts = [_sample_rows(keys[pad], tables, state, actions, replacement) for _, pad in calls]
    return _joined(parts, [len(idx) for idx, _ in calls])
