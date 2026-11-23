"""Committed Kaizo overrides: loader, citation checks and the reviewed draft (plan Task 10).

``data/kaizo/overrides.toml`` holds the facts the extractors cannot settle alone (spec 6.4,
13; plan Rulings 8-9): Trick Room and Gravity areas, gauntlet groups, trainer-id settlements
for PKCalc labels and sheet headers, corrections to derived trainer fields and hand-read
move lists. Every entry carries ``source``: ``"<sheet>:<row>"``, ``"pkcalc:<file>:<line>"``,
``"rom:<what>"`` or ``"user, <YYYY-MM-DD>"``; anything else is an ``OverridesError``.

Tables (keys are ROM trainer ids unless said otherwise):

- ``[area.<id>]``: ``trick_room``, ``gravity`` (bool, default false), ``weather``
  (``rain|sun|sandstorm|hail|fog|none``; absent = the ROM-derived start condition stands),
  and ``map_header`` (int, optional): the placement a user answer chose. An entry whose
  source is ``"user, <date>"`` is the user's answer, which settles a ROM Trick Room trainer
  in place of an ``area_exception`` (``check_rom_trick_room``, ``check_two_placement``).
- ``[area_exception.<id>]``: ``reason``. A ROM Trick Room trainer (``rom_field_effects.json``)
  deliberately left without a ``trick_room`` override, e.g. one whose start condition the
  evidence does not settle (``data/kaizo/start_conditions.json`` ``unknown``).
- ``[[gauntlet]]``: ``name``, ``trainers`` (ids in the sheet's order) and the game facts of the
  group (design v5.3 section 2), all or none: ``orders`` (a list of orders, each a list of
  fights, each a list of one or two trainer ids; or ``{ permutations = [<fights>] }`` for every
  order of the fights), ``leave_gaps`` (the gaps where the player can leave the map on foot: one
  list for every order, one list per order, or ``"all"``), ``escape`` (bool), ``pc_gaps``,
  ``heal_gaps``, ``locked_gaps`` (a list of gaps or ``"all"``) and, optional, ``partner``
  (``trainer``, ``join_before``: the one fight before which the game makes the partner join,
  as a one-item list; ``source``) and ``fight_headers`` (the map header each fight starts on:
  one integer for every fight, or one list per order; it takes ``orders_source``) and
  ``live_next`` (the next-fight table by the live count, a list of rows ``{ fought, live,
  next, headers }`` plus ``gap`` on a plain row after a fight and ``to``, ``at`` on a change
  row: ``nuzline.data.gauntlet.LiveRow``; absent: the walk does not depend on the count and
  the table is the orders' prefixes; ``GauntletFacts.live_next``) and ``skips`` (orders that
  leave trainers out behind a pushed Strength boulder, each ``{ order, skipped, header,
  boulder, boulder_from, boulder_to, hold }``: ``nuzline.data.gauntlet.Skip``; with
  ``skips_source``). With ``live_next``,
  ``leave_gaps``, ``heal_gaps`` and ``pc_gaps`` are present exactly when the table's gap
  facts give one list for every order and live count (that list), and absent when they
  depend on the order or the count (``nuzline.data.gauntlet.flat_gaps``); either way
  round is an error. Optional ``field_moves`` (the relevant field moves, names of
  ``nuzline.data.gauntlet.FIELD_MOVES`` in that order; the facts above are the walk's with
  all of them) with ``field_moves_source`` (required with it), and one ``[[gauntlet.case]]``
  table per other subset of them, in order (``moves``, then ``orders``, ``leave_gaps``,
  ``pc_gaps``, ``heal_gaps``, ``locked_gaps``, ``fight_headers``, ``live_next`` as above, under
  the same rules; ``orders = []`` and no other key: the case has no full sequence;
  ``GauntletFacts.cases``, ``MoveCase``). Optional ``wild_levels`` (a list of tables
  ``header``, ``kind``, ``rate``, ``max_level``, ``tiles``, ``examples``, optional ``roaming``:
  ``WildLevel``; exactly when ``wild`` is a field move) with ``wild_levels_source``, and
  ``repel_steps`` (a list of tables ``order``, ``steps``: ``RepelSteps``; with ``wild``: every
  order, skip order and order of the other live count) with ``repel_steps_source``; a case
  with ``wild`` and orders holds its own ``repel_steps``; ``repel_refusal`` (a reason) in
  their place refuses ``wild = repel`` by name. Gap k follows
  the k-th fight of an order. Each fact has a ``<key>_source`` (``orders_source`` is required; the others default to it). The facts are
  read from the ROM by ``tools/kaizo/gauntlet_map.py``; user choices never appear here.
- ``[trainer_ids."<label or sheet cite>"]``: ``id``, read by ``trainer_map``.
- ``[sheet_unmapped."<sheet cite>"]``: ``kind`` and ``reason`` for a split-sheet header that
  maps to no trainer id on purpose (``SHEET_UNMAPPED_KINDS``): ``ally`` (the player's tag
  partner, not an opponent), ``duplicate`` (``of``: the id an earlier header already maps;
  the same battle listed twice) or ``user_call`` (the evidence does not settle it; the
  question is in ``reason``). Every unmapped header must have one and every entry must be
  an unmapped header (``check_sheet_unmapped``), so no battle drops out of the splits.
- ``[fields.<id>]``: corrections to derived fields: ``split``, ``optional``, ``left_side``,
  ``player_starter``, and ``format``: ``"unknown"`` for a trainer the ROM fights in two
  battle structures, e.g. a scripted pair on one map and placed alone on another, which
  one ``trainers.json`` format cannot hold (coverage refuses it, K2b Task 15); or, with a
  ``"user, <date>"`` source only, the user's answer: a concrete format (``FORMATS``), with
  ``opponent_2`` (the second trainer's id) for ``double_two_trainer`` (required) and
  ``multi`` (optional), and ``partner`` (the player's partner's trainer id, ``multi`` only,
  optional: absent, the ROM's partner stands).
- ``[move_lists.<sound|punch>]``: ``ids``, for ``read_move_lists(..., override=...)``.

:func:`check_citations` re-reads every sheet citation against ``docs.json``;
:func:`check_rom_trick_room` requires every ROM Trick Room trainer to be overridden or
excepted; :func:`draft_overrides` writes the reviewed draft.

Usage: python -m tools.kaizo.overrides draft|check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

from nuzline.data.gauntlet import (
    FLAT_GAP_FIELDS,
    GauntletFacts,
    LiveRow,
    MoveCase,
    Partner,
    RepelSteps,
    Skip,
    WildLevel,
)
from tools.kaizo.trainer_map import (
    Mapping,
    MappingError,
    check_override_keys,
    map_sheet_headers,
    map_trainers,
)
from tools.rom.inputs import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "data" / "kaizo" / "overrides.toml"
DRAFT_PATH = REPO_ROOT / ".cache" / "kaizo" / "overrides.draft.toml"
INTERMEDIATE = REPO_ROOT / "data" / "intermediate_kaizo"

FACT_KEYS = ("orders", "leave_gaps", "escape", "pc_gaps", "heal_gaps", "locked_gaps")
FACT_SOURCES = tuple(f"{k}_source" for k in FACT_KEYS)
WEATHERS = ("rain", "sun", "sandstorm", "hail", "fog", "none")
MOVE_LIST_KEYS = ("sound", "punch")
STARTERS = (387, 390, 393)
FIELD_TYPES = {
    "split": str,
    "optional": bool,
    "left_side": bool,
    "player_starter": int,
    "format": str,
    "opponent_2": int,
    "partner": int,
}
UNKNOWN_FORMAT = "unknown"  # the format a [fields.<id>] entry sets to refuse a trainer
# The concrete formats a user's answer may set (tools/battle_scripts.py battle structures).
FORMATS = ("single", "double", "double_two_trainer", "multi")
_PAIRED = ("double_two_trainer", "multi")
# Champion Cynthia's Gravity: PKCalc's UI ticks Gravity for this label only.
GRAVITY_LABEL = "Champion Cynthia"
GRAVITY_SOURCE = "pkcalc:src/js/shared_controls.js:611"
TRICK_ROOM_WEATHER = 18  # Kaizo map weather that starts Trick Room (tools/rom/readers/maps.py)

_SOURCE = re.compile(
    r"pkcalc:[^:\s][^:]*:\d+"
    r"|rom:\S.*"
    r"|user, \d{4}-\d{2}-\d{2}"
    r"|(?!pkcalc:|rom:)[^:]+:\d+"
)
_SHEET = re.compile(r"(?!pkcalc:|rom:|user, )[^:]+:\d+")
_CLAIM_NAMES = {"trick_room": "Trick Room", "gravity": "Gravity"}
SHEET_UNMAPPED_KINDS = ("ally", "duplicate", "user_call")


class OverridesError(ValueError):
    """``overrides.toml`` breaks the schema."""


@dataclass(frozen=True)
class AreaOverride:
    trick_room: bool
    gravity: bool
    weather: str | None
    source: str
    map_header: int | None = None  # the placement a user answer chose (optional)

    @property
    def user(self) -> bool:
        """Whether this entry is the user's answer (source ``"user, <date>"``)."""
        return self.source.startswith("user, ")


@dataclass(frozen=True)
class AreaException:
    reason: str
    source: str


@dataclass(frozen=True)
class SheetUnmapped:
    kind: str  # one of SHEET_UNMAPPED_KINDS
    reason: str
    source: str
    of: int | None = None  # duplicate: the trainer id the battle is already mapped to


@dataclass(frozen=True)
class Gauntlet:
    name: str
    trainers: tuple[int, ...]
    source: str
    facts: GauntletFacts | None = None  # None: the game facts are not recorded (a draft)


@dataclass(frozen=True)
class Overrides:
    """``sources`` holds the citation of every ``trainer_ids``, ``fields`` and
    ``move_lists`` entry, keyed ``"<table>.<key>"`` (area and gauntlet entries carry
    their own)."""

    area: dict[int, AreaOverride] = field(default_factory=dict)
    gauntlets: tuple[Gauntlet, ...] = ()
    trainer_ids: dict[str, int] = field(default_factory=dict)
    fields: dict[int, dict[str, object]] = field(default_factory=dict)
    move_lists: dict[str, list[int]] = field(default_factory=dict)
    area_exceptions: dict[int, AreaException] = field(default_factory=dict)
    sheet_unmapped: dict[str, SheetUnmapped] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)


# --- loader ---------------------------------------------------------------------------


def _reason(entry: str, value: object) -> str:
    """A plain-words reason (``repel_refusal``): a string, empty when absent."""
    if not isinstance(value, str):
        raise OverridesError(f"{entry}: repel_refusal must be a string")
    return value


def _source(entry: str, table: object) -> str:
    if not isinstance(table, dict) or "source" not in table:
        raise OverridesError(f"{entry}: entry has no source")
    src = table["source"]
    if not isinstance(src, str) or not _SOURCE.fullmatch(src):
        raise OverridesError(
            f"{entry}: source {src!r} is not '<sheet>:<row>', 'pkcalc:<file>:<line>', "
            "'rom:<what>' or 'user, <YYYY-MM-DD>'"
        )
    return src


def _keys(entry: str, table: dict, allowed: Iterable[str]) -> None:
    allowed = set(allowed) | {"source"}
    bad = sorted(k for k in table if k not in allowed)
    if bad:
        raise OverridesError(f"{entry}: unknown key {', '.join(map(repr, bad))}")


def _id(entry: str, key: str) -> int:
    if not key.isdigit():
        raise OverridesError(f"{entry}: key {key!r} is no trainer id")
    return int(key)


def _bool(entry: str, name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise OverridesError(f"{entry}: {name} must be true or false, not {value!r}")
    return value


def _ints(entry: str, name: str, value: object) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise OverridesError(f"{entry}: {name} must be a list of integers")
    return value


def _area(raw: dict) -> dict[int, AreaOverride]:
    out = {}
    for key, t in raw.items():
        entry = f"area.{key}"
        src = _source(entry, t)
        _keys(entry, t, ("trick_room", "gravity", "weather", "map_header"))
        tr = _bool(entry, "trick_room", t.get("trick_room", False))
        grav = _bool(entry, "gravity", t.get("gravity", False))
        weather = t.get("weather")
        if weather is not None and weather not in WEATHERS:
            raise OverridesError(f"{entry}: weather {weather!r} is not one of {WEATHERS}")
        if not (tr or grav or weather is not None):
            raise OverridesError(f"{entry}: claims nothing (trick_room, gravity, weather)")
        header = t.get("map_header")
        if header is not None and (not isinstance(header, int) or isinstance(header, bool)):
            raise OverridesError(f"{entry}: map_header must be an integer")
        if header is not None and not src.startswith("user, "):
            raise OverridesError(f"{entry}: map_header names a user's choice ('user, <date>')")
        out[_id(entry, key)] = AreaOverride(tr, grav, weather, src, header)
    return out


def _fight(entry: str, value: object) -> tuple[int, ...]:
    ids = _ints(entry, "a fight", value)
    if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids):
        raise OverridesError(f"{entry}: a fight is one or two distinct trainer ids, not {ids}")
    return tuple(ids)


def _orders(
    entry: str, raw: object, trainers: list[int], subset: bool = False
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """The orders; each covers ``trainers`` exactly once (``subset``: the same part of them,
    every trainer at most once)."""
    if isinstance(raw, dict):
        if set(raw) != {"permutations"} or not isinstance(raw["permutations"], list):
            raise OverridesError(f"{entry}: orders table is {{ permutations = [<fights>] }}")
        fights = tuple(_fight(entry, f) for f in raw["permutations"])
        orders = list(permutations(fights))
    elif isinstance(raw, list) and raw and all(isinstance(o, list) and o for o in raw):
        orders = [tuple(_fight(entry, f) for f in o) for o in raw]
    else:
        raise OverridesError(f"{entry}: orders must be a non-empty list of orders")
    for o in orders:
        ids = sorted(i for f in o for i in f)
        if subset and len(set(ids)) == len(ids) and set(ids) <= set(trainers):
            if ids != sorted(i for f in orders[0] for i in f):
                raise OverridesError(f"{entry}: the orders cover different trainers")
            continue
        if ids != sorted(trainers):
            raise OverridesError(
                f"{entry}: order {[list(f) for f in o]} does not cover the group's trainers "
                "exactly once"
            )
    if len(set(orders)) != len(orders):
        raise OverridesError(f"{entry}: an order is listed twice")
    return tuple(orders)


def _gaps(entry: str, name: str, raw: object, n: int) -> tuple[int, ...]:
    """A gap list within 1..n-1 (``"all"``: every gap)."""
    if raw == "all":
        return tuple(range(1, n))
    gaps = _ints(entry, name, raw)
    if any(not 1 <= k < n for k in gaps) or gaps != sorted(set(gaps)):
        raise OverridesError(f"{entry}: {name} {gaps} is not an ascending list within 1..{n - 1}")
    return tuple(gaps)


def _fact_source(entry: str, t: dict, key: str, default: str) -> str:
    return _source(f"{entry} {key}", {"source": t.get(key, default)})


def _partner(entry: str, p: object, n: int, default_source: str) -> Partner:
    if not isinstance(p, dict):
        raise OverridesError(f"{entry}: partner must be a table")
    _keys(f"{entry} partner", p, ("trainer", "join_before", "source"))
    if not isinstance(p.get("trainer"), int) or isinstance(p.get("trainer"), bool):
        raise OverridesError(f"{entry}: partner.trainer must be an integer")
    join = _ints(entry, "partner.join_before", p.get("join_before"))
    if len(join) != 1 or not 1 <= join[0] <= n:
        raise OverridesError(
            f"{entry}: partner.join_before {join} is not one fight within 1..{n} (the game "
            "fixes the join point)"
        )
    source = _source(f"{entry} partner", {"source": p.get("source", default_source)})
    return Partner(p["trainer"], tuple(join), source)


def _facts(entry: str, name: str, src: str, t: dict, trainers: list[int]) -> GauntletFacts | None:
    """The group's game facts, all or none (``FACT_KEYS``)."""
    if not any(k in t for k in (*FACT_KEYS, *FACT_SOURCES, "partner")):
        return None
    # A gap field the group's live_next makes order- or count-dependent is absent: required
    # without live_next, checked against it with one (GauntletFacts, flat_gaps).
    optional = set(FLAT_GAP_FIELDS) if "live_next" in t else set()
    missing = [k for k in (*FACT_KEYS, "orders_source") if k not in t and k not in optional]
    if missing:
        raise OverridesError(f"{entry}: facts are all or none; missing {', '.join(missing)}")
    orders = _orders(entry, t["orders"], trainers)
    sizes = [len(o) for o in orders]
    n = min(sizes)
    orders_source = _fact_source(entry, t, "orders_source", "")
    leave = _leave_gaps(entry, t.get("leave_gaps"), sizes)
    p = t.get("partner")
    heads = _fight_headers(entry, t.get("fight_headers"), sizes)

    def source(key: str) -> str:
        return _fact_source(entry, t, f"{key}_source", orders_source)

    live = _live_next(entry, t["live_next"]) if "live_next" in t else ()
    field_moves = t.get("field_moves", [])
    if not isinstance(field_moves, list) or not all(isinstance(m, str) for m in field_moves):
        raise OverridesError(f"{entry}: field_moves must be a list of move names")
    if "field_moves" in t and "field_moves_source" not in t:
        raise OverridesError(f"{entry}: field_moves needs field_moves_source")
    moves_source = (
        _fact_source(entry, t, "field_moves_source", "") if "field_moves_source" in t else ""
    )
    cases = _cases(entry, t.get("case"), trainers)
    wild = t.get("wild_levels", [])
    if not isinstance(wild, list) or not all(isinstance(w, dict) for w in wild):
        raise OverridesError(f"{entry}: wild_levels must be a list of tables")
    if "wild_levels" in t and "wild_levels_source" not in t:
        raise OverridesError(f"{entry}: wild_levels needs wild_levels_source")
    if "repel_steps" in t and "repel_steps_source" not in t:
        raise OverridesError(f"{entry}: repel_steps needs repel_steps_source")
    repel = _repel_steps(entry, t["repel_steps"]) if "repel_steps" in t else ()
    try:
        return GauntletFacts(
            name=name,
            source=src,
            orders=orders,
            orders_source=orders_source,
            leave_gaps=leave,
            leave_gaps_source=source("leave_gaps"),
            escape=_bool(entry, "escape", t["escape"]),
            escape_source=source("escape"),
            pc_gaps=_gaps(entry, "pc_gaps", t["pc_gaps"], n) if "pc_gaps" in t else None,
            pc_gaps_source=source("pc_gaps"),
            heal_gaps=_gaps(entry, "heal_gaps", t["heal_gaps"], n) if "heal_gaps" in t else None,
            heal_gaps_source=source("heal_gaps"),
            locked_gaps=_gaps(entry, "locked_gaps", t["locked_gaps"], n),
            locked_gaps_source=source("locked_gaps"),
            partner=None if p is None else _partner(entry, p, n, orders_source),
            fight_headers=heads,
            live_next=live,
            field_moves=tuple(field_moves),
            field_moves_source=moves_source,
            cases=cases,
            skips=_skips(entry, t["skips"]) if "skips" in t else (),
            skips_source=_fact_source(entry, t, "skips_source", "") if "skips" in t else "",
            requires_beaten=tuple(_ints(entry, "requires_beaten", t["requires_beaten"]))
            if "requires_beaten" in t
            else (),
            requires_beaten_source=_fact_source(entry, t, "requires_beaten_source", "")
            if "requires_beaten" in t
            else "",
            wild_levels=tuple(WildLevel.from_dict(w) for w in wild),
            wild_levels_source=_fact_source(entry, t, "wild_levels_source", "")
            if "wild_levels" in t
            else "",
            repel_steps=repel,
            repel_steps_source=_fact_source(entry, t, "repel_steps_source", "")
            if "repel_steps" in t
            else "",
            repel_refusal=_reason(entry, t.get("repel_refusal", "")),
        )
    except ValueError as e:
        if isinstance(e, OverridesError):
            raise
        raise OverridesError(f"{entry}: {e}") from None


def _leave_gaps(entry: str, raw: object, sizes: list[int]) -> tuple[tuple[int, ...], ...] | None:
    """``leave_gaps``: absent (None), ``"all"``, one list per order or one list for every
    order."""
    if raw is None:
        return None
    if raw == "all":
        return tuple(tuple(range(1, s)) for s in sizes)
    if isinstance(raw, list) and raw and all(isinstance(g, list) for g in raw):
        if len(raw) != len(sizes):
            raise OverridesError(f"{entry}: leave_gaps needs one list per order")
        return tuple(_gaps(entry, "leave_gaps", g, s) for g, s in zip(raw, sizes, strict=True))
    return (_gaps(entry, "leave_gaps", raw, min(sizes)),) * len(sizes)


CASE_KEYS = (
    "moves",
    "orders",
    "leave_gaps",
    "pc_gaps",
    "heal_gaps",
    "locked_gaps",
    "fight_headers",
    "live_next",
    "repel_steps",
)


def _repel_steps(where: str, raw: object) -> tuple[RepelSteps, ...]:
    """``repel_steps``: a list of tables ``{ order, steps }`` (``RepelSteps``)."""
    if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
        raise OverridesError(f"{where}: repel_steps must be a list of tables")
    out = []
    for r in raw:
        if set(r) != {"order", "steps"}:
            raise OverridesError(f"{where}: a repel_steps row holds order and steps only")
        if not isinstance(r["order"], list) or not all(
            isinstance(f, list) and f and all(isinstance(t, int) for t in f) for f in r["order"]
        ):
            raise OverridesError(f"{where}: repel_steps order must be a list of fights")
        steps = _ints(where, "repel_steps steps", r["steps"])
        try:
            out.append(RepelSteps(tuple(tuple(f) for f in r["order"]), tuple(steps)))
        except ValueError as e:
            raise OverridesError(f"{where}: {e}") from None
    return tuple(out)


def _cases(entry: str, raw: object, trainers: list[int]) -> tuple[MoveCase, ...]:
    """The ``[[gauntlet.case]]`` tables (``CASE_KEYS``): a field-move case's facts, read as the
    group's own; ``orders = []`` alone is a case with no full sequence. The set of cases is
    checked by ``GauntletFacts``."""
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(c, dict) for c in raw):
        raise OverridesError(f"{entry}: case must be [[gauntlet.case]] tables")
    out = []
    for i, c in enumerate(raw):
        where = f"{entry} case[{i}]"
        bad = sorted(k for k in c if k not in CASE_KEYS)
        if bad:
            raise OverridesError(f"{where}: unknown key {', '.join(map(repr, bad))}")
        moves = c.get("moves")
        if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
            raise OverridesError(f"{where}: moves must be a list of move names")
        where = f"{entry} case {moves}"
        if c.get("orders") == []:
            if set(c) != {"moves", "orders"}:
                raise OverridesError(f"{where}: a case with no orders holds no other fact")
            empty = ((), (), (), ())
            out.append(MoveCase(tuple(moves), (), *empty))
            continue
        optional = set(FLAT_GAP_FIELDS) if "live_next" in c else set()
        missing = [k for k in CASE_KEYS[:6] if k not in c and k not in optional]
        if missing:
            raise OverridesError(f"{where}: missing {', '.join(missing)}")
        orders = _orders(where, c["orders"], trainers)
        sizes = [len(o) for o in orders]
        n = min(sizes)
        try:
            out.append(
                MoveCase(
                    moves=tuple(moves),
                    orders=orders,
                    leave_gaps=_leave_gaps(where, c.get("leave_gaps"), sizes),
                    pc_gaps=_gaps(where, "pc_gaps", c["pc_gaps"], n) if "pc_gaps" in c else None,
                    heal_gaps=_gaps(where, "heal_gaps", c["heal_gaps"], n)
                    if "heal_gaps" in c
                    else None,
                    locked_gaps=_gaps(where, "locked_gaps", c["locked_gaps"], n),
                    fight_headers=_fight_headers(where, c.get("fight_headers"), sizes),
                    live_next=_live_next(where, c["live_next"]) if "live_next" in c else (),
                    repel_steps=_repel_steps(where, c["repel_steps"])
                    if "repel_steps" in c
                    else (),
                )
            )
        except ValueError as e:
            if isinstance(e, OverridesError):
                raise
            raise OverridesError(f"{where}: {e}") from None
    return tuple(out)


SKIP_KEYS = ("order", "skipped", "header", "boulder", "boulder_from", "boulder_to", "hold")


def _skips(entry: str, raw: object) -> tuple[Skip, ...]:
    """``skips``: a non-empty list of tables (``SKIP_KEYS``, all required), each an order that
    leaves trainers out behind a pushed boulder (``nuzline.data.gauntlet.Skip``); checked
    against the orders by ``GauntletFacts``."""
    entry = f"{entry} skips"
    if not isinstance(raw, list) or not raw:
        raise OverridesError(f"{entry}: must be a non-empty list of tables")
    out = []
    for i, r in enumerate(raw):
        where = f"{entry}[{i}]"
        if not isinstance(r, dict):
            raise OverridesError(f"{where}: must be a table")
        _keys(where, r, SKIP_KEYS)
        missing = [k for k in SKIP_KEYS if k not in r]
        if missing:
            raise OverridesError(f"{where}: missing {', '.join(missing)}")
        if not isinstance(r["order"], list) or not r["order"]:
            raise OverridesError(f"{where}: order must be a non-empty list of fights")
        if not isinstance(r["hold"], list) or not r["hold"]:
            raise OverridesError(f"{where}: hold must be a non-empty list of tiles")
        out.append(
            Skip(
                order=tuple(_fight(where, f) for f in r["order"]),
                skipped=tuple(_ints(where, "skipped", r["skipped"])),
                header=_ints(where, "header", [r["header"]])[0],
                boulder=_ints(where, "boulder", [r["boulder"]])[0],
                boulder_from=_skip_tile(where, r["boulder_from"]),
                boulder_to=_skip_tile(where, r["boulder_to"]),
                hold=tuple(_skip_tile(where, t) for t in r["hold"]),
            )
        )
    return tuple(out)


def _skip_tile(where: str, v: object) -> tuple[int, int]:
    t = _ints(where, "a tile", v)
    if len(t) != 2:
        raise OverridesError(f"{where}: a tile is [x, z], not {t}")
    return (t[0], t[1])


LIVE_ROW_KEYS = ("fought", "live", "next", "headers", "to", "at", "gap")


def _live_next(entry: str, raw: object) -> tuple[LiveRow, ...]:
    """``live_next``: a non-empty list of rows (``LIVE_ROW_KEYS``; ``fought``, ``live``,
    ``next`` and ``headers`` required), checked as a table by ``check_live_rows``."""
    entry = f"{entry} live_next"
    if not isinstance(raw, list) or not raw:
        raise OverridesError(f"{entry}: must be a non-empty list of rows")
    rows = []
    for i, r in enumerate(raw):
        where = f"{entry}[{i}]"
        if not isinstance(r, dict):
            raise OverridesError(f"{where}: must be a table")
        _keys(where, r, LIVE_ROW_KEYS)
        missing = [k for k in ("fought", "live", "next", "headers") if k not in r]
        if missing:
            raise OverridesError(f"{where}: missing {', '.join(missing)}")
        fights = []
        for key in ("fought", "next"):
            v = r[key]
            if not isinstance(v, list) or not all(isinstance(f, list) for f in v):
                raise OverridesError(f"{where}: {key} must be a list of fights")
            fights.append(tuple(tuple(_ints(where, key, f)) for f in v))
        for key in ("live", "to"):
            if key in r and (not isinstance(r[key], int) or isinstance(r[key], bool)):
                raise OverridesError(f"{where}: {key} must be an integer")
        if "at" in r and not isinstance(r["at"], str):
            raise OverridesError(f"{where}: at must be a string")
        gap = r.get("gap")
        if gap is not None and (
            not isinstance(gap, list) or not all(isinstance(g, str) for g in gap)
        ):
            raise OverridesError(f"{where}: gap must be a list of strings")
        rows.append(
            LiveRow(
                fought=fights[0],
                live=r["live"],
                next=fights[1],
                headers=tuple(_ints(where, "headers", r["headers"])),
                to=r.get("to", r["live"]),
                at=r.get("at", ""),
                gap=None if gap is None else tuple(gap),
            )
        )
    return tuple(rows)


def _fight_headers(entry: str, raw: object, sizes: list[int]) -> tuple[tuple[int, ...], ...]:
    """``fight_headers``: absent (not recorded), one header for every fight, or one list of
    headers per order (as long as the order)."""
    if raw is None:
        return ()
    if isinstance(raw, int) and not isinstance(raw, bool):
        return tuple((raw,) * s for s in sizes)
    if isinstance(raw, list) and len(raw) == len(sizes) and all(isinstance(h, list) for h in raw):
        out = tuple(_ints(entry, "fight_headers", h) for h in raw)
        if [len(h) for h in out] == sizes:
            return tuple(tuple(h) for h in out)
    raise OverridesError(f"{entry}: fight_headers is one header or one list per order")


def _gauntlets(raw: list) -> tuple[Gauntlet, ...]:
    out = []
    for i, t in enumerate(raw):
        entry = f"gauntlet[{i}]"
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            entry = f"gauntlet {t['name']!r}"
        src = _source(entry, t)
        _keys(
            entry,
            t,
            (
                "name",
                "trainers",
                "partner",
                "fight_headers",
                "live_next",
                "field_moves",
                "field_moves_source",
                "case",
                "skips",
                "skips_source",
                "requires_beaten",
                "requires_beaten_source",
                "wild_levels",
                "wild_levels_source",
                "repel_steps",
                "repel_steps_source",
                "repel_refusal",
                *FACT_KEYS,
                *FACT_SOURCES,
            ),
        )
        if not isinstance(t.get("name"), str) or not t["name"]:
            raise OverridesError(f"{entry}: name must be a non-empty string")
        trainers = _ints(entry, "trainers", t.get("trainers"))
        if len(trainers) < 2 or len(set(trainers)) != len(trainers):
            raise OverridesError(f"{entry}: trainers must be two or more distinct ids")
        facts = _facts(entry, t["name"], src, t, trainers)
        out.append(Gauntlet(t["name"], tuple(trainers), src, facts))
    owner: dict[int, str] = {}
    for g in out:
        for tid in g.trainers:
            if tid in owner:
                raise OverridesError(
                    f"gauntlet {g.name!r}: trainer {tid} is already in gauntlet {owner[tid]!r}"
                )
            owner[tid] = g.name
    return tuple(out)


def load_overrides(path: Path = DEFAULT_PATH) -> Overrides:
    """Read and validate ``overrides.toml``; an absent file is empty overrides."""
    if not path.exists():
        return Overrides()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise OverridesError(f"{path}: {e}") from e
    tables = (
        "area",
        "area_exception",
        "gauntlet",
        "trainer_ids",
        "sheet_unmapped",
        "fields",
        "move_lists",
    )
    bad = sorted(k for k in raw if k not in tables)
    if bad:
        raise OverridesError(f"{path.name}: unknown table {', '.join(map(repr, bad))}")
    sources: dict[str, str] = {}

    exceptions = {}
    for key, t in raw.get("area_exception", {}).items():
        entry = f"area_exception.{key}"
        src = _source(entry, t)
        _keys(entry, t, ("reason",))
        if not isinstance(t.get("reason"), str) or not t["reason"]:
            raise OverridesError(f"{entry}: reason must be a non-empty string")
        exceptions[_id(entry, key)] = AreaException(t["reason"], src)
    area = _area(raw.get("area", {}))
    both = sorted(set(area) & set(exceptions))
    if both:
        raise OverridesError(f"trainer ids {both} have both an area entry and an area_exception")

    trainer_ids = {}
    for label, t in raw.get("trainer_ids", {}).items():
        entry = f"trainer_ids.{label}"
        sources[entry] = _source(entry, t)
        _keys(entry, t, ("id",))
        tid = t.get("id")
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise OverridesError(f"{entry}: id must be an integer")
        trainer_ids[label] = tid

    sheet_unmapped = {}
    for cite, t in raw.get("sheet_unmapped", {}).items():
        entry = f"sheet_unmapped.{cite}"
        src = _source(entry, t)
        _keys(entry, t, ("kind", "reason", "of"))
        kind, reason, of = t.get("kind"), t.get("reason"), t.get("of")
        if kind not in SHEET_UNMAPPED_KINDS:
            raise OverridesError(f"{entry}: kind must be one of {SHEET_UNMAPPED_KINDS}")
        if not isinstance(reason, str) or not reason:
            raise OverridesError(f"{entry}: reason must be a non-empty string")
        if (kind == "duplicate") != (of is not None):
            raise OverridesError(f"{entry}: 'of' is required for a duplicate and only there")
        if of is not None and (not isinstance(of, int) or isinstance(of, bool)):
            raise OverridesError(f"{entry}: of must be an integer")
        sheet_unmapped[cite] = SheetUnmapped(kind, reason, src, of)

    fields_ = {}
    for key, t in raw.get("fields", {}).items():
        entry = f"fields.{key}"
        sources[entry] = _source(entry, t)
        _keys(entry, t, FIELD_TYPES)
        values = {k: v for k, v in t.items() if k != "source"}
        if not values:
            raise OverridesError(f"{entry}: corrects no field")
        for k, v in values.items():
            if type(v) is not FIELD_TYPES[k]:
                raise OverridesError(f"{entry}: {k} must be {FIELD_TYPES[k].__name__}")
        if "player_starter" in values and values["player_starter"] not in STARTERS:
            raise OverridesError(f"{entry}: player_starter must be one of {STARTERS}")
        fmt = values.get("format")
        user = sources[entry].startswith("user, ")
        if fmt is not None and fmt != UNKNOWN_FORMAT and not (user and fmt in FORMATS):
            raise OverridesError(
                f"{entry}: format must be {UNKNOWN_FORMAT!r}, or with a 'user, <date>' source "
                f"one of {FORMATS}"
            )
        if "opponent_2" in values and fmt not in _PAIRED:
            raise OverridesError(f"{entry}: opponent_2 needs format one of {_PAIRED}")
        if fmt == "double_two_trainer" and "opponent_2" not in values:
            raise OverridesError(f"{entry}: format double_two_trainer needs opponent_2")
        if "partner" in values and fmt != "multi":
            raise OverridesError(f"{entry}: partner needs format 'multi'")
        fields_[_id(entry, key)] = values

    move_lists = {}
    for key, t in raw.get("move_lists", {}).items():
        entry = f"move_lists.{key}"
        sources[entry] = _source(entry, t)
        _keys(entry, t, ("ids",))
        if key not in MOVE_LIST_KEYS:
            raise OverridesError(f"{entry}: no such move list (one of {MOVE_LIST_KEYS})")
        move_lists[key] = _ints(entry, "ids", t.get("ids"))

    return Overrides(
        area=area,
        gauntlets=_gauntlets(raw.get("gauntlet", [])),
        trainer_ids=trainer_ids,
        fields=fields_,
        move_lists=move_lists,
        area_exceptions=exceptions,
        sheet_unmapped=sheet_unmapped,
        sources=sources,
    )


def check_sheet_unmapped(overrides: Overrides, sheet_maps: Iterable[Mapping]) -> list[str]:
    """Problems: a split-sheet header that maps to no trainer id without a
    ``[sheet_unmapped]`` entry (a battle would drop out of its split), an entry for a header
    that maps, or a duplicate whose ``of`` no other header maps to."""
    sheet_maps = list(sheet_maps)
    unmapped = {m.label for m in sheet_maps if m.id is None}
    mapped_ids = {m.id for m in sheet_maps if m.id is not None}
    settled = overrides.sheet_unmapped
    out = [
        f"sheet header {c} maps to no trainer id: map it ([trainer_ids]) or settle it "
        "([sheet_unmapped])"
        for c in sorted(unmapped - set(settled))
    ]
    out += [
        f"sheet_unmapped.{c}: the header maps to a trainer id"
        for c in sorted(set(settled) - unmapped)
    ]
    out += [
        f"sheet_unmapped.{c}: duplicate of {e.of}, which no sheet header maps to"
        for c, e in sorted(settled.items())
        if e.kind == "duplicate" and e.of not in mapped_ids
    ]
    return out


# --- sheet rows -------------------------------------------------------------------------


def _sections(docs: dict) -> Iterator[dict]:
    for split in docs["splits"].values():
        yield from split["sections"]


def _rows(docs: dict) -> dict[str, tuple[dict, dict | None]]:
    """cite -> (section, header); a section header row has header None."""
    out: dict[str, tuple[dict, dict | None]] = {}
    for sec in _sections(docs):
        out[sec["cite"]] = (sec, None)
        for h in sec["trainers"]:
            out[h["cite"]] = (sec, h)
    return out


def _claims(tags: dict) -> set[str]:
    """Area conditions a tag dict claims: weathers, ``fog``, ``trick_room``, ``gravity``."""
    out = set()
    if tags.get("permanent"):
        out.add(tags["permanent"])
    if tags.get("weather"):
        out.add(tags["weather"])
    if tags.get("trick_room"):
        out.add("trick_room")
    if tags.get("fog"):
        out.add("fog")
    return out


def _row_claims(sec: dict, h: dict | None) -> set[str]:
    return _claims(sec["tags"]) | (_claims(h["tags"]) if h is not None else set())


def _says_gauntlet(sec: dict, h: dict | None) -> bool:
    sec_flag = any(k in sec["tags"] for k in ("gym_gauntlet", "gauntlet_with", "gauntlet"))
    return sec_flag or (h is not None and any(k in h["tags"] for k in ("gauntlet", "b2b_with")))


def check_citations(overrides: Overrides, docs: dict) -> list[str]:
    """Every sheet citation must still be a split row whose text supports its entry: the
    trainer header or its section carries the claimed tag (area: Trick Room, Gravity,
    the weather; gauntlet: a gauntlet tag). Other citations only need the row to exist."""
    rows = _rows(docs)
    out = []

    def row(entry: str, src: str) -> tuple[dict, dict | None] | None:
        if not _SHEET.fullmatch(src):
            return None
        if src not in rows:
            out.append(f"{entry}: {src} is no split sheet row")
            return None
        return rows[src]

    for tid, a in overrides.area.items():
        entry = f"area.{tid}"
        r = row(entry, a.source)
        if r is None:
            continue
        claims = _row_claims(*r)
        wanted = [k for k in ("trick_room", "gravity") if getattr(a, k)]
        if a.weather is not None:
            wanted.append(a.weather)
        for k in wanted:
            if k not in claims and not (k == "none" and not claims):
                out.append(f"{entry}: {a.source} does not say {_CLAIM_NAMES.get(k, k)}")
    for g in overrides.gauntlets:
        entry = f"gauntlet {g.name!r}"
        r = row(entry, g.source)
        if r is not None and not _says_gauntlet(*r):
            out.append(f"{entry}: {g.source} does not say gauntlet")
    for tid, e in overrides.area_exceptions.items():
        row(f"area_exception.{tid}", e.source)
    for entry, src in overrides.sources.items():
        row(entry, src)
    return out


_ROM_TR_CITE = re.compile(
    rf"rom:map headers? (\d+(?:, \d+)*) weather {TRICK_ROOM_WEATHER} \(Trick Room\)"
)
_ROM_ONLY_CITE = re.compile(
    r"rom:only (.+) on a Trick Room map \(rom_field_effects\.json, map headers? "
    r"(\d+(?:, \d+)*)\)"
)


def _headers(text: str) -> list[int]:
    return [int(x) for x in text.split(", ")]


def check_rom_trick_room(overrides: Overrides, field_effects: Iterable[dict]) -> list[str]:
    """Both directions of the ROM Trick Room join (Kaizo map weather 18, a ROM fact that
    supersedes Ruling 8's "no map header holds Trick Room"):

    - every ``rom_field_effects.json`` trick_room trainer needs an ``area`` entry with
      ``trick_room = true`` or an ``area_exception``;
    - every ``trick_room`` entry citing ``rom:`` must cite ``map header(s) N weather 18
      (Trick Room)`` with exactly that trainer's rom_field_effects headers;
    - every ``trainer_ids`` entry citing ``rom:only <name> on a Trick Room map (...)`` must
      name the only rom_field_effects trainer of that name, with those headers."""
    rows = [r for r in field_effects if r["effect"] == "trick_room"]
    by_id = {r["id"]: r for r in rows}
    out = []
    for tid, a in overrides.area.items():
        if not (a.trick_room and a.source.startswith("rom:")):
            continue
        m = _ROM_TR_CITE.fullmatch(a.source)
        if m is None:
            out.append(f"area.{tid}: {a.source!r} is no ROM Trick Room citation")
        elif tid not in by_id:
            out.append(f"area.{tid}: {a.source!r} but rom_field_effects.json has no Trick Room row")
        elif _headers(m[1]) != by_id[tid]["map_headers"]:
            out.append(
                f"area.{tid}: {a.source!r} but rom_field_effects.json has map headers "
                f"{by_id[tid]['map_headers']}"
            )
    for entry, src in overrides.sources.items():
        if not (entry.startswith("trainer_ids.") and src.startswith("rom:only ")):
            continue
        m = _ROM_ONLY_CITE.fullmatch(src)
        tid = overrides.trainer_ids[entry.removeprefix("trainer_ids.")]
        named = [r for r in rows if m is not None and r["name"] == m[1]]
        if m is None:
            out.append(f"{entry}: {src!r} is no ROM Trick Room citation")
        elif len(named) != 1 or named[0]["id"] != tid:
            ids = [r["id"] for r in named]
            out.append(
                f"{entry}: {src!r} but rom_field_effects.json has {m[1]} at {ids}, not {tid}"
            )
        elif _headers(m[2]) != named[0]["map_headers"]:
            out.append(
                f"{entry}: {src!r} but rom_field_effects.json has map headers "
                f"{named[0]['map_headers']}"
            )
    for r in rows:
        tid = r["id"]
        a = overrides.area.get(tid)
        if (a is not None and (a.trick_room or a.user)) or tid in overrides.area_exceptions:
            continue
        out.append(
            f"area.{tid}: ROM Trick Room ({r['name']}, map headers {r['map_headers']}) has no "
            "trick_room override and no area_exception"
        )
    return out


def two_placement_trick_room(
    field_effects: Iterable[dict], trainer_maps: MappingABC[int, list[int]]
) -> dict[int, tuple[list[int], list[int]]]:
    """Trainer id -> (its Trick Room headers, its other headers) for every ROM Trick Room
    trainer (map weather 18) the trainer -> map join also places on a header without Trick
    Room. Which fight is meant is a user call (the
    Rosalind/Puck/Kari precedent): ``check_two_placement``."""
    out = {}
    for r in field_effects:
        if r["effect"] != "trick_room":
            continue
        other = sorted(set(trainer_maps.get(r["id"], ())) - set(r["map_headers"]))
        if other:
            out[r["id"]] = (list(r["map_headers"]), other)
    return out


def check_two_placement(
    overrides: Overrides,
    field_effects: Iterable[dict],
    trainer_maps: MappingABC[int, list[int]],
) -> list[str]:
    """Every two-placement Trick Room trainer (``two_placement_trick_room``) needs either an
    ``area_exception`` (refused until the user answers) or the user's answer: an ``area``
    entry with source ``"user, <date>"``. A ``rom:`` or sheet Trick Room entry cannot settle
    it: the ROM holds both placements. An answer naming ``map_header`` must name one of the
    trainer's placements, with ``trick_room`` true exactly when that header has Trick Room."""
    out = []
    two = two_placement_trick_room(field_effects, trainer_maps)
    for tid, (tr, other) in sorted(two.items()):
        if tid in overrides.area_exceptions:
            continue
        a = overrides.area.get(tid)
        if a is None or not a.user:
            out.append(
                f"trainer {tid}: Trick Room headers {tr} and other headers {other}; needs an "
                "area_exception (the fight meant is a user call) or the user's answer as an "
                "area entry with source 'user, <date>', not a rom: or sheet trick_room entry"
            )
        elif a.map_header is None:
            continue
        elif a.map_header not in (*tr, *other):
            out.append(f"area.{tid}: map_header {a.map_header} is not one of {tr + other}")
        elif a.trick_room != (a.map_header in tr):
            has = "has" if a.map_header in tr else "has no"
            out.append(
                f"area.{tid}: map_header {a.map_header} {has} Trick Room, but trick_room is "
                f"{str(a.trick_room).lower()}"
            )
    return out


def check_trainer_ids(overrides: Overrides, rom_ids: set[int]) -> list[str]:
    """Every id an entry names must be a ROM trainer id."""
    out = []
    for table, ids in (
        ("area", overrides.area),
        ("area_exception", overrides.area_exceptions),
        ("fields", overrides.fields),
    ):
        out += [f"{table}.{i}: {i} is no ROM trainer id" for i in ids if i not in rom_ids]
    for g in overrides.gauntlets:
        out += [
            f"gauntlet {g.name!r}: {i} is no ROM trainer id" for i in g.trainers if i not in rom_ids
        ]
    for tid, values in overrides.fields.items():
        for k in ("opponent_2", "partner"):
            if k in values and values[k] not in rom_ids:
                out.append(f"fields.{tid}: {k} {values[k]} is no ROM trainer id")
    return out


def check_gauntlet_members(overrides: Overrides, docs: dict, sheet: Iterable[Mapping]) -> list[str]:
    """Every member of a sheet-cited gauntlet must map (``sheet``, Task 9 header mappings)
    from a row of the cited group: the drafted group whose source is that cite (a chain,
    a B2B pair, a gym or "Gauntlet With" section), else the cited row's section."""
    rows = _rows(docs)
    cites_of: dict[int, set[str]] = defaultdict(set)
    for m in sheet:
        if m.id is not None:
            cites_of[m.id].add(m.label)
    groups: dict[str, set[str]] = defaultdict(set)
    for _name, src, members, _problem in _gauntlet_groups(docs):
        groups[src] |= {h["cite"] for h in members}
    out = []
    for g in overrides.gauntlets:
        if not _SHEET.fullmatch(g.source) or g.source not in rows:
            continue  # check_citations reports a missing row
        allowed = groups.get(g.source) or {h["cite"] for h in rows[g.source][0]["trainers"]}
        for tid in g.trainers:
            if not cites_of[tid] & allowed:
                out.append(
                    f"gauntlet {g.name!r}: trainer {tid} maps to no row of {g.source}'s group "
                    f"(its rows: {sorted(cites_of[tid]) or 'none'})"
                )
    return out


# --- mappings -------------------------------------------------------------------------


@dataclass(frozen=True)
class Mappings:
    """Everything the draft joins: PKCalc and sheet-header mappings (Task 9), ROM trainer
    rows by id (``start_condition``) and ``rom_field_effects.json`` rows (the ROM trainer ->
    map join for Trick Room, Task 8)."""

    pkcalc: tuple[Mapping, ...]
    sheet: tuple[Mapping, ...]
    rom_trainers: dict[int, dict]
    field_effects: tuple[dict, ...]


def build_mappings(
    pkcalc_trainers: dict[str, dict],
    docs: dict,
    rom_trainers: list[dict],
    species: list[dict],
    field_effects: Iterable[dict],
    overrides: dict[str, int],
) -> Mappings:
    """Map PKCalc labels and sheet headers with the ``trainer_ids`` overrides, after
    ``check_override_keys`` (a key that is neither label nor cite raises)."""
    labels = [k for k, v in pkcalc_trainers.items() if not v.get("template")]
    cites = [h["cite"] for sec in _sections(docs) for h in sec["trainers"]]
    check_override_keys(overrides, labels, cites)
    return Mappings(
        pkcalc=tuple(map_trainers(pkcalc_trainers, rom_trainers, species, overrides)),
        sheet=tuple(map_sheet_headers(docs, rom_trainers, species, overrides)),
        rom_trainers={t["id"]: t for t in rom_trainers},
        field_effects=tuple(field_effects),
    )


# --- draft ----------------------------------------------------------------------------

HEADER = """\
# Platinum Kaizo overrides (plan K2b Task 10; spec 6.4, 12.4, 13). Hand-reviewed; every
# entry cites its source: "<sheet>:<row>" (Kaizo docs split sheet), "pkcalc:<file>:<line>",
# "rom:<what>" or "user, <date>". `python -m tools.kaizo.overrides check` re-validates every
# citation against docs.json and the ROM extraction; `... draft` regenerates the draft this
# file was reviewed from (.cache/kaizo/overrides.draft.toml).
#
# Ruling 8 (area conditions): weather and fog come from the ROM map headers through the
# trainer -> map join (permanent in battle); date-dependent maps are settled in
# data/kaizo/start_conditions.json, not here. Its "Trick Room ... no map header holds" is
# SUPERSEDED for Kaizo: Kaizo map weather 18 starts Trick Room (decoded from the patched
# battle weather switch, Task 8), so every trainer in rom_field_effects.json gets a
# trick_room entry citing the ROM, or an [area_exception] saying why not. Gravity (Champion
# Cynthia) and sheet-only Trick Room stay here. A trainer claimed by two areas is never
# settled by guess: its claim stays commented out with every citation. The start conditions
# left "unknown" by ruling (325 343 509 559 610 649 746 747; 318 and 676 left that list on
# 2026-11-14 when the shared-script join gave each one ROM condition) are not settled here.
#
# Ruling 9 (split, optional, left side): derived from the sheet at every build, with a
# per-field citation; [fields.<id>] only corrects them. Gauntlet groups are drafted from the
# sheet: a "Gauntlet Start" .. "Gauntlet End" chain, a "B2B With X" pair, a section tagged
# "GYM TRAINERS ARE A GAUNTLET" (its trainers in sheet order, leader excluded) and a
# "Gauntlet With X" section (its GAUNTLET-tagged trainers, the tag partner excluded).
# The game facts of each group (orders, gaps, escape, partner) are read from the ROM by
# tools/kaizo/gauntlet_map.py and written by hand below the drafted entry; the draft has none.
#
# Level caps are not here: data/kaizo/level_caps.toml (read at runtime by prep).
"""


def _q(s: str) -> str:
    return json.dumps(s)


def _ascii(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode("ascii")


def _area_lines(tid: int, claims: dict[str, bool], source: str) -> list[str]:
    lines = [f"[area.{tid}]"]
    lines += [f"{k} = true" for k in ("trick_room", "gravity") if claims.get(k)]
    lines.append(f"source = {_q(source)}")
    return lines


def _rom_tr_source(row: dict) -> str:
    hdrs = row["map_headers"]
    word = "header" if len(hdrs) == 1 else "headers"
    joined = ", ".join(map(str, hdrs))
    return f"rom:map {word} {joined} weather {TRICK_ROOM_WEATHER} (Trick Room)"


def _comment(lines: list[str]) -> list[str]:
    return [f"# {line}" for line in lines]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.casefold())


def _names_match(target: str, name: str) -> bool:
    """``B2B With Galactic Terra`` names ``GalacticfTerra`` (the sheet's mangled sign):
    equal, or same first and last word around a glued middle."""
    t, n = _norm(target), _norm(name)
    words = target.split()
    return t == n or (
        len(words) > 1 and n.startswith(_norm(words[0])) and n.endswith(_norm(words[-1]))
    )


def _is_leader(h: dict) -> bool:
    return h["name"].split()[0].casefold() == "leader" if h["name"].split() else False


def _chains(sec: dict) -> list[tuple[str, list[dict], str | None]]:
    """(source cite, headers, problem) per gauntlet chain in one section: "Gauntlet Start"
    opens a chain (a "Gauntlet With X" or section-level GAUNTLET section opens one at its
    top, cited by the section), GAUNTLET continues it, "Gauntlet End" closes it; several
    consecutive ends are one multi battle pair. A chain left open, an untagged header
    inside a chain, or a tag outside any chain is a problem (drafted commented out)."""
    chains: list[tuple[str, list[dict], str | None]] = []
    section_wide = "gauntlet_with" in sec["tags"] or "gauntlet" in sec["tags"]
    cur: list[dict] | None = [] if section_wide else None
    src, closed, problem = sec["cite"], False, None
    for h in [*sec["trainers"], None]:
        g = None if h is None else h["tags"].get("gauntlet")
        if cur and (h is None or g == "start" or (g is None and closed)):
            chains.append((src, cur, problem or (None if closed else "no Gauntlet End")))
            cur, closed, problem = None, False, None
        if h is None:
            break
        if g is None:
            if cur:
                problem = problem or f"untagged header {h['cite']} inside the chain"
            continue
        if g == "start":
            cur, src = [h], h["cite"]
        elif cur is None:
            chains.append((h["cite"], [h], f"'{g}' with no Gauntlet Start before it"))
        else:
            if g == "middle" and closed:
                problem = problem or "GAUNTLET after Gauntlet End"
            cur.append(h)
            closed = closed or g == "end"
    if section_wide and not chains:
        chains.append((sec["cite"], [], "no GAUNTLET-tagged trainer"))
    return chains


def _gauntlet_groups(docs: dict) -> list[tuple[str, str, list[dict], str | None]]:
    """(name, source cite, headers, problem) per sheet gauntlet group (Ruling 9)."""
    groups = []
    for sec in _sections(docs):
        hs = sec["trainers"]
        if sec["tags"].get("gym_gauntlet"):
            leaders = [h for h in hs if _is_leader(h)]
            problem = None if len(leaders) == 1 else f"{len(leaders)} leader headers"
            members = [h for h in hs if h not in leaders]
            groups.append((f"{sec['name']} gym trainers", sec["cite"], members, problem))
            continue
        chains = _chains(sec)
        many = len(chains) > 1
        for i, (src, members, prob) in enumerate(chains, 1):
            name = sec["name"] + (f" #{i}" if many else "")
            groups.append((name, src, members, prob))
        b2b = [h for h in hs if "b2b_with" in h["tags"]]
        paired: set[str] = set()
        for a in b2b:
            if a["cite"] in paired:
                continue
            mates = [
                b
                for b in b2b
                if b is not a
                and _names_match(a["tags"]["b2b_with"], b["name"])
                and _names_match(b["tags"]["b2b_with"], a["name"])
            ]
            if len(mates) == 1:
                paired |= {a["cite"], mates[0]["cite"]}
                pair = [a, mates[0]]
                groups.append((f"{sec['name']} B2B", a["cite"], pair, None))
            else:
                groups.append((f"{sec['name']} B2B", a["cite"], [a], f"{len(mates)} B2B partners"))
    return groups


def draft_overrides(docs: dict, mappings: Mappings) -> str:
    """TOML for every Trick Room trainer (ROM ``rom_field_effects.json`` -> ``rom:`` source;
    sheet-only -> the sheet header), Cynthia's Gravity and every gauntlet group, with
    no game facts. A trainer claimed by two areas (two sheet rows with different area tags,
    or a ROM start condition ``unknown``) is drafted commented out with every citation, and
    a ROM Trick Room one gets an ``[area_exception]``. Groups or claims whose trainers do
    not map are drafted commented out too."""
    sheet = {m.label: m for m in mappings.sheet}
    rom = mappings.rom_trainers
    rom_tr = {r["id"]: r for r in mappings.field_effects if r["effect"] == "trick_room"}
    rows = _rows(docs)

    by_id: dict[int, list[tuple[str, set[str]]]] = defaultdict(list)
    unmapped_claims = []
    for cite, (sec, h) in rows.items():
        if h is None:
            continue
        claims = _row_claims(sec, h)
        m = sheet.get(cite)
        if m is not None and m.id is not None:
            by_id[m.id].append((cite, claims))
        elif claims & {"trick_room", "gravity"}:
            cands = m.candidates if m is not None else ()
            unmapped_claims.append(
                f"# UNMAPPED {cite} {_ascii(h['text'])!r} claims {sorted(claims)}; "
                f"ROM candidates {list(cands)}"
            )

    def name_of(tid: int) -> str:
        return _ascii(rom.get(tid, {}).get("name") or "?")

    def sheet_note(tid: int) -> str:
        cs = [f"{c} {sorted(cl) or ['no area tag']}" for c, cl in by_id.get(tid, [])]
        return "sheet: " + ("; ".join(cs) if cs else "no mapped row")

    out = [HEADER, "# --- Trick Room -------------------------------------------------------\n"]
    sheet_tr = {tid for tid, rs in by_id.items() if any("trick_room" in cl for _c, cl in rs)}
    n_tr = n_conflict = 0
    for tid in sorted(set(rom_tr) | sheet_tr):
        rs = by_id.get(tid, [])
        if tid in rom_tr:
            source = _rom_tr_source(rom_tr[tid])
        else:
            source = next(c for c, cl in rs if "trick_room" in cl)
        others = [(c, cl) for c, cl in rs if "trick_room" not in cl]
        unknown = rom.get(tid, {}).get("start_condition") == "unknown"
        lines = _area_lines(tid, {"trick_room": True}, source)
        if not others and not unknown:
            n_tr += 1
            out.append(f"# {tid} {name_of(tid)}; {sheet_note(tid)}")
            if tid not in rom_tr:
                start = rom.get(tid, {}).get("start_condition")
                out.append(
                    f"# rom: no map of this trainer has weather {TRICK_ROOM_WEATHER} and "
                    f"start_condition is {start!r}: a sheet-only claim (Task 13 reports it)"
                )
            out += lines
            out.append("")
            continue
        n_conflict += 1
        out.append(f"# CONFLICT {tid} {name_of(tid)}: claimed by two areas; left unsettled.")
        if tid in rom_tr:
            out.append(f"#   {source}")
        if unknown:
            out.append(
                "#   rom:trainers.json start_condition unknown (data/kaizo/start_conditions.json:"
                " its maps start different conditions)"
            )
        out += [f"#   {c}: {sorted(cl) or ['no area tag']}" for c, cl in rs]
        out += _comment(lines)
        if tid in rom_tr:
            if unknown:
                ex_src = f"rom:trainers.json {tid} start_condition unknown"
                reason = "start condition unknown by ruling; ROM maps disagree"
            else:
                ex_src = others[0][0]
                reason = f"sheet also lists it under {sorted(others[0][1]) or 'no area tag'}"
            out += [
                f"[area_exception.{tid}]",
                f"reason = {_q(reason)}",
                f"source = {_q(ex_src)}",
            ]
        out.append("")
    out += [line for line in unmapped_claims if "trick_room" in line]
    if any("trick_room" in line for line in unmapped_claims):
        out.append("")

    out.append("# --- Gravity ----------------------------------------------------------\n")
    cynthia = next((m for m in mappings.pkcalc if m.label == GRAVITY_LABEL), None)
    grav: dict[int, str] = {}
    if cynthia is not None and cynthia.id is not None:
        grav[cynthia.id] = GRAVITY_SOURCE
    else:
        out.append(f"# UNMAPPED PKCalc {GRAVITY_LABEL!r}: Gravity entry not drafted")
    for tid, rs in sorted(by_id.items()):
        for c, cl in rs:
            if "gravity" in cl:
                grav.setdefault(tid, c)
    for tid, src in sorted(grav.items()):
        lines = _area_lines(tid, {"gravity": True}, src)
        if tid in set(rom_tr) | sheet_tr:
            out.append(f"# CONFLICT {tid}: Gravity on a Trick Room trainer; merge by hand")
            lines = _comment(lines)
        out.append(f"# {tid} {name_of(tid)}; {sheet_note(tid)}")
        out += lines
        out.append("")
    out += [line for line in unmapped_claims if "gravity" in line]

    out.append("# --- Gauntlets (game facts: tools/kaizo/gauntlet_map.py, added by hand) ---\n")
    seen: dict[int, str] = {}
    n_g = n_g_bad = 0
    for name, src, members, problem in _gauntlet_groups(docs):
        ids = [sheet.get(h["cite"]).id if h["cite"] in sheet else None for h in members]
        cites = ", ".join(f"{h['cite']}={i}" for h, i in zip(members, ids, strict=True))
        if problem is None and None in ids:
            problem = "unmapped member"
        if problem is None and len(set(ids)) < 2:
            problem = "fewer than two trainers"
        clash = [f"{i} (also in {seen[i]})" for i in ids if i in seen]
        if problem is None and clash:
            problem = "trainer in another group: " + ", ".join(clash)
        lines = [
            "[[gauntlet]]",
            f"name = {_q(_ascii(name))}",
            f"trainers = [{', '.join(str(i) for i in ids)}]",
            f"source = {_q(src)}",
        ]
        texts = "; ".join(_ascii(h["text"]) for h in members)
        if problem is None:
            n_g += 1
            seen.update({i: src for i in ids})
            out.append(f"# {texts}")
            out.append(f"# rows: {cites}")
            out += lines
        else:
            n_g_bad += 1
            out.append(f"# UNSETTLED gauntlet {_ascii(name)!r} ({problem}): {texts}")
            out.append(f"#   rows: {cites}")
            out += _comment(lines)
        out.append("")
    out.append(
        f"# draft totals: {n_tr} trick_room, {n_conflict} area conflicts, {len(grav)} gravity, "
        f"{n_g} gauntlets, {n_g_bad} unsettled gauntlets"
    )
    return "\n".join(out) + "\n"


# --- CLI ------------------------------------------------------------------------------


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs(inter: Path) -> tuple[dict, dict, list, list, list, dict]:
    paths = (
        inter / "docs" / "docs.json",
        inter / "pkcalc" / "pkcalc.json",
        inter / "rom" / "trainers.json",
        inter / "rom" / "species.json",
        inter / "rom" / "rom_field_effects.json",
        inter / "rom" / "rom_trainer_maps.json",
    )
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing Kaizo intermediates: " + ", ".join(missing))
    return tuple(_read(p) for p in paths)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.kaizo.overrides")
    ap.add_argument("command", choices=("draft", "check"))
    ap.add_argument("--overrides", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--intermediate", type=Path, default=INTERMEDIATE)
    ap.add_argument("--out", type=Path, default=DRAFT_PATH, help="draft output")
    args = ap.parse_args(argv)
    try:
        docs, pkcalc, rom_trainers, species, effects, where = _inputs(args.intermediate)
        o = load_overrides(args.overrides)
        m = build_mappings(pkcalc["trainers"], docs, rom_trainers, species, effects, o.trainer_ids)
    except (FileNotFoundError, OverridesError, MappingError) as e:
        print(f"error: {_ascii(str(e))}", file=sys.stderr)
        return 1
    if args.command == "draft":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(draft_overrides(docs, m), encoding="utf-8")
        print(args.out)
        return 0
    problems = (
        check_citations(o, docs)
        + check_rom_trick_room(o, effects)
        + check_two_placement(o, effects, {int(k): v for k, v in where.items()})
        + check_gauntlet_members(o, docs, m.sheet)
        + check_trainer_ids(o, {t["id"] for t in rom_trainers})
        + check_sheet_unmapped(o, m.sheet)
    )
    for p in problems:
        print(_ascii(p))
    if problems:
        return 1
    print(
        f"ok: {len(o.area)} area, {len(o.area_exceptions)} area_exception, "
        f"{len(o.gauntlets)} gauntlet, {len(o.trainer_ids)} trainer_ids, {len(o.fields)} fields, "
        f"{len(o.move_lists)} move_lists"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
