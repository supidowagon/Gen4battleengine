"""The prep problem: the usable candidate pool, the bag's resources, the user's selectors
(protect, exclude, reserve, prestatus) and the level-ups each Pokemon gets on the way to the
cap (spec §5.2, §6, §6.1; plan 4c rulings 4 and 9)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from nuzline.data.forms import personal_row
from nuzline.data.games import KAIZO
from nuzline.data.learnsets import Learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import Trainer
from nuzline.engine.coverage import (
    IMPLEMENTED_ABILITIES,
    IMPLEMENTED_EFFECTS,
    IMPLEMENTED_ITEMS,
    Support,
    rom_change_line,
    rom_change_refusals,
)
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Status
from nuzline.engine.state import MAX_DEATH_BUDGET
from nuzline.prep.pool import Candidate, pool_candidates
from nuzline.saves.save import SaveFile

PRESTATUS = {"poison": Status.POISON, "burn": Status.BURN, "paralysis": Status.PARALYSIS}
HELD_POCKETS = ("items", "berries", "medicine", "battle_items")
HEART_SCALE = "Heart Scale"
SYNTAX = "party:N or box:B:S"


@dataclass(frozen=True)
class Resources:
    """What the bag offers: Heart Scales (Move Relearner), TMs as ``(item, quantity)`` in item
    order, HMs (reusable), and implemented holdable items as ``(item, quantity)``."""

    heart_scales: int
    tms: tuple[tuple[int, int], ...]
    hms: tuple[int, ...]
    held: tuple[tuple[int, int], ...]


def bag_resources(
    bag: Mapping[str, list[tuple[int, int]]],
    tables: Tables,
    names: Names,
    learnsets: Learnsets,
    reserved: Collection[int],
) -> Resources:
    """The bag's resources with every ``reserved`` item id removed entirely (design.md §4.4:
    reserved resources are excluded from the search, never merely penalised)."""
    effect = np.asarray(tables.item_effect)
    scale = names.item(HEART_SCALE)
    scales = 0
    tms: dict[int, int] = {}
    hms: set[int] = set()
    held: dict[int, int] = {}
    for pocket, entries in bag.items():
        for item, qty in entries:
            if item in reserved:
                continue
            if item == scale:
                scales += qty
            elif pocket == "tms_hms":
                if learnsets.is_hm(item):
                    hms.add(item)
                else:
                    tms[item] = tms.get(item, 0) + qty
            elif pocket in HELD_POCKETS:
                kind = ItemId(int(effect[item]))
                if kind in IMPLEMENTED_ITEMS and kind != ItemId.NONE:
                    held[item] = held.get(item, 0) + qty
    return Resources(scales, tuple(sorted(tms.items())), tuple(sorted(hms)), tuple(held.items()))


def _move_refusal(tables: Tables, move: int) -> str | None:
    """Why a player's ``move`` cannot be used (engine effect or AI), or None."""
    from nuzline.ai.policy import player_move_refusal

    effect = EffectId(int(np.asarray(tables.move_effect)[move]))
    if IMPLEMENTED_EFFECTS.get(effect, Support.NONE) == Support.NONE:
        return "move effect not implemented"
    return player_move_refusal(move)


def _ability_refusal(tables: Tables, spec: PokemonSpec) -> str | None:
    """Why the engine or the AI refuses ``spec``'s ability, or None."""
    from nuzline.ai.policy import player_ability_refusal

    row = personal_row(tables, spec.species, spec.form)
    ability = AbilityId(int(np.asarray(tables.species_abilities)[row, spec.ability_slot]))
    if IMPLEMENTED_ABILITIES.get(ability, Support.NONE) == Support.NONE:
        return "ability not implemented"
    decomp = int(np.asarray(tables.species_abilities_decomp)[row][spec.ability_slot])
    return player_ability_refusal(decomp)


def _item_refusal(tables: Tables, item: int) -> str | None:
    effect = ItemId(int(np.asarray(tables.item_effect)[item]))
    if IMPLEMENTED_ITEMS.get(effect, Support.NONE) == Support.NONE:
        return "held item not implemented"
    return None


def _rom_change_refusal(
    tables: Tables,
    names: Names,
    trainer: Trainer | None,
    specs: Sequence[PokemonSpec] = (),
    moves: Sequence[int] = (),
) -> str | None:
    """Why a Kaizo ``not_reached`` ROM-change condition refuses these player-side ``specs`` or
    ``moves`` against ``trainer`` (``coverage.rom_change_refusals`` with ``spec_terms_only``:
    the fight-level terms are the fight's own report), or None; always None without a
    trainer or outside Kaizo (spec design.md §3)."""
    if trainer is None or names.game != KAIZO:
        return None
    labels = rom_change_refusals(tables, names, trainer, specs, moves, spec_terms_only=True)
    return "; ".join(rom_change_line(label) for label in labels) or None


def usable_pool(
    tables: Tables,
    names: Names,
    candidates: Sequence[Candidate],
    trainer: Trainer | None = None,
) -> tuple[list[Candidate], list[str]]:
    """The candidates the engine and the AI can play, and one line per exclusion or trim.

    A candidate whose ability or a current move is unsupported or refused is left out
    (``"<label>: left out: <what> (<reason>)"``). Plan 4c ruling (held items): one whose only
    unsupported part is its held item stays with the item taken off (``"<label>: <item> taken
    off (<reason>)"``), as spec §6.1 offers "none". An unsupported or refused relearner or TM
    option is removed (``"<label>: <move> not offered (<reason>)"``). With ``trainer`` on
    Kaizo, an ability, move or held item a ``not_reached`` ROM-change condition lists counts
    as refused the same way (``_rom_change_refusal``)."""

    def rom(specs: Sequence[PokemonSpec] = (), moves: Sequence[int] = ()) -> str | None:
        return _rom_change_refusal(tables, names, trainer, specs, moves)

    def move_refusal(move: int) -> str | None:
        return _move_refusal(tables, move) or rom(moves=[move])

    out: list[Candidate] = []
    lines: list[str] = []
    for cand in candidates:
        spec = cand.spec
        why: list[str] = []
        bare = dataclasses.replace(spec, moves=(), item=0)  # the ability alone
        if (r := _ability_refusal(tables, spec) or rom([bare])) is not None:
            ability = names.ability_names[
                names.species_abilities(spec.species, spec.form)[spec.ability_slot]
            ]
            why.append(f"{ability} ({r})")
        for move in spec.moves:
            if move and (r := move_refusal(move)) is not None:
                why.append(f"{names.move_names[move]} ({r})")
        if why:
            lines.extend(f"{cand.label}: left out: {w}" for w in why)
            continue
        if (r := _item_refusal(tables, spec.item) or rom([spec])) is not None:
            lines.append(f"{cand.label}: {names.item_names[spec.item]} taken off ({r})")
            cand = dataclasses.replace(cand, spec=dataclasses.replace(spec, item=0))
        relearner = []
        for move in cand.relearner:
            if (r := move_refusal(move)) is None:
                relearner.append(move)
            else:
                lines.append(f"{cand.label}: {names.move_names[move]} not offered ({r})")
        tms = []
        for item, move in cand.tms:
            if (r := move_refusal(move)) is None:
                tms.append((item, move))
            else:
                lines.append(f"{cand.label}: {names.move_names[move]} not offered ({r})")
        out.append(dataclasses.replace(cand, relearner=tuple(relearner), tms=tuple(tms)))
    return out, lines


def _where(cand: Candidate) -> str:
    kind, box, slot = cand.location
    return f"party:{slot}" if kind == "party" else f"box:{box + 1}:{slot}"


def pick(candidates: Sequence[Candidate], names: Names, value: str) -> int:
    """The index of the candidate ``value`` names: ``party:N`` (N as in the label),
    ``box:B:S`` (B 1-based, S as in the label), else a case-insensitive species name or
    nickname. ``ValueError`` for no match, and for several (each listed with its syntax)."""
    text = value.strip()
    if text.isdigit():
        mine = next((c for c in candidates if c.location == ("party", 0, int(text))), None)
        hint = f"; party:{text} is {mine.label}" if mine else ""
        raise ValueError(
            f"{value!r}: a bare number is ambiguous (party and boxes are both candidates); "
            f"use {SYNTAX}{hint}"
        )
    want = None
    if m := re.fullmatch(r"party:(\d+)", text, re.IGNORECASE):
        want = ("party", 0, int(m.group(1)))
    elif m := re.fullmatch(r"box:(\d+):(\d+)", text, re.IGNORECASE):
        want = ("box", int(m.group(1)) - 1, int(m.group(2)))
    if want is not None:
        for i, cand in enumerate(candidates):
            if cand.location == want:
                return i
        raise ValueError(f"no Pokemon at {value!r} ({SYNTAX}, as in the labels of `save show`)")
    key = text.casefold()
    hits = []
    for i, cand in enumerate(candidates):
        species = names.species_names[cand.spec.species].casefold()
        if key in {species, cand.nickname.casefold()}:
            hits.append(i)
    if not hits:
        raise ValueError(f"no Pokemon matches {value!r} (name it by species, nickname or {SYNTAX})")
    if len(hits) > 1:
        listing = "; ".join(f"{candidates[i].label} ({_where(candidates[i])})" for i in hits)
        raise ValueError(f"{value!r} names {len(hits)} Pokemon: {listing}; use {SYNTAX}")
    return hits[0]


def levelled_moves(
    tables: Tables,
    candidates: Sequence[Candidate],
    learnsets: Learnsets,
    cap: int,
    names: Names | None = None,
    trainer: Trainer | None = None,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Per candidate the ``(level, move)`` pairs it learns going from its save level
    (``spec.level - candies``) to ``cap``: levels above the save level and at most the cap,
    moves its spec does not know and the engine and AI support, ascending, first occurrence
    per move (plan 4c ruling 4). With ``names`` and ``trainer`` on Kaizo, a move a
    ``not_reached`` ROM-change condition lists is dropped too (``_rom_change_refusal``)."""
    out = []
    for cand in candidates:
        spec = cand.spec
        seen = set(spec.moves)
        learned: list[tuple[int, int]] = []
        level_up = learnsets.level_up(spec.species, spec.form)
        for level, move in sorted(level_up, key=lambda p: p[0]):
            if not spec.level - cand.candies < level <= cap or move in seen:
                continue
            seen.add(move)
            if _move_refusal(tables, move) is None and (
                names is None or _rom_change_refusal(tables, names, trainer, moves=[move]) is None
            ):
                learned.append((level, move))
        out.append(tuple(learned))
    return tuple(out)


# Abilities that rule out a declared prestatus (ruling 6), by name: each game's own
# ``Names`` gives their ids (``build_problem``).
STATUS_BLOCKING_ABILITIES = {
    "Limber": Status.PARALYSIS,
    "Immunity": Status.POISON,
    "Water Veil": Status.BURN,
}


def blocking_abilities(names: Names) -> tuple[tuple[int, int], ...]:
    """``STATUS_BLOCKING_ABILITIES`` as (ability id in ``names``' game, ``Status``)."""
    return tuple(
        sorted(
            (names.ability(name), int(status)) for name, status in STATUS_BLOCKING_ABILITIES.items()
        )
    )


@dataclass(frozen=True)
class PrepProblem:
    """Everything the prep search is given. ``protected``: candidate indices; ``prestatus``:
    ``ids.Status`` values the user declares; ``source``: ``"save:<sha256>"`` or a test name;
    ``levelled``: per candidate, see :func:`levelled_moves`; ``left_out_party``: ``(party slot,
    name)`` of every save party member that is not a candidate (eggs, excluded, unsupported),
    which the prep steps deposit."""

    candidates: tuple[Candidate, ...]
    trainer: Trainer
    opp: tuple[PokemonSpec, ...]
    start_condition: int
    resources: Resources
    deaths: int
    protected: frozenset[int]
    prestatus: tuple[int, ...]
    cap: int
    source: str
    levelled: tuple[tuple[tuple[int, int], ...], ...]
    left_out_party: tuple[tuple[int, str], ...] = ()
    # (decomp ability id, Status) of STATUS_BLOCKING_ABILITIES in the problem's game
    # (``build_problem`` looks the names up in its ``Names``); None: the shipped tables'.
    blocking_abilities: tuple[tuple[int, int], ...] | None = None


def _lookup(lookup: Callable[[str], int], name: str, what: str) -> int:
    try:
        return lookup(name)
    except KeyError:
        raise ValueError(f"unknown {what} {name!r}") from None


def build_problem(
    tables: Tables,
    names: Names,
    save: SaveFile,
    trainer: Trainer,
    opp: Sequence[PokemonSpec],
    start_condition: int,
    learnsets: Learnsets,
    *,
    cap: int,
    deaths: int = 0,
    protect: Sequence[str] = (),
    exclude: Sequence[str] = (),
    reserve: Sequence[str] = (),
    prestatus: Sequence[str] = (),
) -> tuple[PrepProblem, list[str]]:
    """The save's pool at ``cap`` as a :class:`PrepProblem`, and the warnings (every
    exclusion and trim, reserved items missing from the bag). ``ValueError`` for a ``deaths``
    outside ``0..MAX_DEATH_BUDGET``, an unknown item or prestatus name, a value both protected
    and excluded, a protected Pokemon left out as unsupported, and an empty pool."""
    if not 0 <= deaths <= MAX_DEATH_BUDGET:
        raise ValueError(f"deaths {deaths} outside [0, {MAX_DEATH_BUDGET}]")
    reserved = [_lookup(names.item, name, "item") for name in reserve]
    statuses = []
    for name in prestatus:
        if name.lower() not in PRESTATUS:
            raise ValueError(f"unknown prestatus {name!r} (one of {', '.join(PRESTATUS)})")
        statuses.append(int(PRESTATUS[name.lower()]))
    pool, warnings = pool_candidates(save, tables, names, learnsets, cap)
    protected_at = {pick(pool, names, v) for v in protect}
    excluded_at = {pick(pool, names, v) for v in exclude}
    if both := sorted(protected_at & excluded_at):
        raise ValueError(f"{pool[both[0]].label} is both protected and excluded")
    reasons = [f"{pool[i].label}: excluded" for i in sorted(excluded_at)] + warnings
    kept = [c for i, c in enumerate(pool) if i not in excluded_at]
    usable, lines = usable_pool(tables, names, kept, trainer)
    warnings += lines
    reasons += lines
    if not usable:
        raise ValueError("no usable Pokemon: " + "; ".join(reasons or ["the save has none"]))
    index = {c.location: i for i, c in enumerate(usable)}
    protected = set()
    for i in sorted(protected_at):
        if pool[i].location not in index:
            why = "; ".join(line for line in lines if line.startswith(pool[i].label + ":"))
            raise ValueError(f"cannot protect {pool[i].label}: {why or 'left out'}")
        protected.add(index[pool[i].location])
    resources = bag_resources(save.bag, tables, names, learnsets, set(reserved))
    if names.game == KAIZO:  # a held item a not_reached condition lists (design.md §3)
        held = []
        for item, qty in resources.held:
            labels = [c.label for c in names.rom_change_conditions if item in c.items]
            if labels:
                why = "; ".join(rom_change_line(label) for label in labels)
                warnings.append(f"{names.item_names[item]}: not offered ({why})")
            else:
                held.append((item, qty))
        resources = dataclasses.replace(resources, held=tuple(held))
    in_bag = {item for entries in save.bag.values() for item, _ in entries}
    warnings += [
        f"{name}: reserved but not in the bag"
        for name, item in zip(reserve, reserved, strict=True)
        if item not in in_bag
    ]
    problem = PrepProblem(
        candidates=tuple(usable),
        trainer=trainer,
        opp=tuple(opp),
        start_condition=start_condition,
        resources=resources,
        deaths=deaths,
        protected=frozenset(protected),
        prestatus=tuple(statuses),
        cap=cap,
        source=f"save:{save.sha256}",
        levelled=levelled_moves(tables, usable, learnsets, cap, names, trainer),
        left_out_party=tuple(
            (
                m.location[2],
                "Egg" if m.is_egg else (m.nickname or names.species_names[m.species]),
            )
            for m in save.party
            if m.location not in index
        ),
        blocking_abilities=blocking_abilities(names),
    )
    return problem, warnings
