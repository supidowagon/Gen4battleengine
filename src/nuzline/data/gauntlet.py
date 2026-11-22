"""Gauntlet options, their precedence and the tracked game facts of a group (design v5.4
sections 1-3, 13).

``OPTIONS`` is the one registry of user options; the loader, CLI, coverage and report
iterate it and nothing else lists option names. Every option carries the name of its
transition in ``prep/gauntlet.py``; ``missing_transitions`` names the options without one
and the loader refuses them. Every default is the strictest value, the one whose score is
a lower bound under every user rule (design section 13): for ``order`` that is
``"worst"``. ``time`` has no default.

``GauntletFacts`` holds what the game fixes for one group (design section 2); user
choices never appear in it. Fights are numbered 1..n along an order; gap k sits after the
k-th fight played (1 <= k <= n-1).

The loader half (design section 3; ``load_rules_file``, ``load_local``, ``parse_flags``,
``resolve_settings``) turns the user's inputs into a ``GauntletSetup`` or raises a
``GauntletSettingsError`` naming the key, the group and the origin. Nothing here writes a
file or holds a user choice."""

from __future__ import annotations

import enum
import re
import tomllib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

LEVEL = "level"
COUNTED = "counted"
SET = "set"
CHOICE = "choice"
KINDS = (LEVEL, COUNTED, SET, CHOICE)
RUN = "run"
GAUNTLET = "gauntlet"
SCOPES = (RUN, GAUNTLET)

UNLIMITED = -1


class SettingsError(ValueError):
    """An input the settings refuse (unknown key or value, a key in the wrong scope, a gap
    outside the order)."""


class Swap(NamedTuple):
    """``swap``: ``bench`` box members chosen before fight 1, ``budget`` members swapped in
    over the chain; -1 = unlimited."""

    bench: int
    budget: int


class Origin(enum.Enum):
    """Where a resolved value came from; precedence locked > flag > rules file > local >
    default (design section 3)."""

    LOCKED = "locked"
    FLAG = "flag"
    RULES_FILE = "rules file"
    LOCAL = "local"
    DEFAULT = "default"


class Resolved(NamedTuple):
    """A value with its origin and the source text the report prints."""

    value: Any
    origin: Origin
    source: str


# --- validators ---------------------------------------------------------------------


def _one_of(values: tuple) -> Callable[[Any], Any]:
    def check(v):
        if v not in values:
            raise SettingsError(f"{v!r} is not one of {list(values)}")
        return v

    return check


def _count(v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < UNLIMITED:
        raise SettingsError(f"{v!r} is not an integer >= 0 or -1")
    return v


def _swap(v) -> Swap:
    if isinstance(v, Swap):
        pair = v
    elif isinstance(v, Mapping) and set(v) == {"bench", "budget"}:
        pair = Swap(v["bench"], v["budget"])
    else:
        raise SettingsError(f"{v!r} is not {{bench, budget}}")
    return Swap(_count(pair.bench), _count(pair.budget))


def _order(v):
    if v in ("worst", "sheet", "best"):
        return v
    if (
        isinstance(v, (list, tuple))
        and v
        and all(isinstance(i, int) and not isinstance(i, bool) for i in v)
    ):
        return tuple(v)
    raise SettingsError(f"{v!r} is not worst, sheet, best or a list of fight ids")


_MOVE_SOURCES = (("bag",), ("bag", "tutor", "relearner"))


def _move_sources(v) -> tuple[str, ...]:
    if isinstance(v, (list, tuple)) and tuple(v) in _MOVE_SOURCES:
        return tuple(v)
    raise SettingsError(f"{v!r} is not one of {[list(s) for s in _MOVE_SOURCES]}")


def _cap(v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise SettingsError(f"{v!r} is not an integer >= 0")
    return v


@dataclass(frozen=True)
class Option:
    """One registry entry. ``values`` lists the allowed values strictest first (None for a
    numeric option, which its validator checks); ``cap`` names the per-gap cap key of a
    counted option."""

    name: str
    kind: str
    scope: str
    per_gap: bool
    values: tuple | None
    default: Any
    validate: Callable[[Any], Any]
    transition: str
    cap: str | None = None
    has_default: bool = True


def _level(name: str, values: tuple[str, ...]) -> Option:
    return Option(name, LEVEL, GAUNTLET, True, values, values[0], _one_of(values), name)


def _run(name: str, values: tuple[str, ...]) -> Option:
    return Option(name, CHOICE, RUN, False, values, values[0], _one_of(values), name)


_OPTION_LIST = (
    _level("heal", ("none", "bag", "full")),
    _level("pc_heal", ("off", "on")),
    _level("team", ("none", "reorder", "moves")),
    _level("growth_items", ("forbidden", "allowed")),
    Option("swap", COUNTED, GAUNTLET, False, None, Swap(0, 0), _swap, "swap", cap="swap_cap"),
    Option("bag_uses", COUNTED, GAUNTLET, False, None, 0, _count, "bag_uses", cap="bag_cap"),
    Option("order", SET, GAUNTLET, False, ("worst", "sheet", "best"), "worst", _order, "order"),
    Option(
        "move_sources",
        SET,
        GAUNTLET,
        False,
        _MOVE_SOURCES,
        ("bag",),
        _move_sources,
        "move_sources",
    ),
    _run("deaths_scope", ("chain", "fight")),
    _run("cap_overflow", ("sit_out_after_fight", "fight")),
    _run("cap_sit_out_in_party", ("forbidden", "allowed")),
    _run("revived_dead", ("stay_dead", "revived")),
    _run("shedinja", ("forbidden", "allowed")),
    _run("field_poison", ("worst_case", "ignore")),
    Option(
        "time",
        CHOICE,
        GAUNTLET,
        False,
        ("day", "night"),
        None,
        _one_of(("day", "night")),
        "time",
        has_default=False,
    ),
)

OPTIONS: dict[str, Option] = {o.name: o for o in _OPTION_LIST}

# The per-gap keys: the Level options and the caps of the counted ones.
GAP_LEVEL_KEYS = tuple(o.name for o in _OPTION_LIST if o.kind == LEVEL)
GAP_CAP_KEYS = tuple(o.cap for o in _OPTION_LIST if o.cap is not None)


def missing_transitions(available: Collection[str]) -> tuple[str, ...]:
    """The options whose transition is not in ``available`` (the loader refuses any)."""
    return tuple(o.name for o in OPTIONS.values() if o.transition not in available)


def strictest(name: str):
    """The strictest value of a Level option."""
    option = OPTIONS[name]
    if option.kind != LEVEL:
        raise KeyError(name)
    return option.values[0]


# --- tracked facts ------------------------------------------------------------------

Fight = tuple[int, ...]
Order = tuple[Fight, ...]


@dataclass(frozen=True)
class Partner:
    """The Wayward Cave partner: ``trainer`` and ``join_before``, the one fight before which
    the game makes her join (a one-item tuple; no user choice)."""

    trainer: int
    join_before: tuple[int, ...]
    source: str

    def __post_init__(self):
        if len(self.join_before) != 1:
            raise ValueError(f"partner {self.trainer}: join_before {self.join_before} is not one")

    @property
    def joins(self) -> int:
        return self.join_before[0]


# The live count (``Party_HasTwoAliveMons``, ``src/unk_02054884.c:157-160``; the partner flag
# makes it two, ``src/overlay005/field_control.c:182-186``) decides which fight the walk starts:
# two singles trainers who see the player at once fight one VS2 battle only with two
# (``src/trainer_encounter.c:94-106``), and a double-battle trainer is fought only with two
# (:108-116; talked to, ``res/field/scripts/scripts_battles.s:19-22``). ``LiveRow`` is one row
# of the next-fight table ``GauntletFacts.live_next``.
LIVE_AT = ("", "bag", "pc", "heal")  # where in the gap the count changes ("": no change)
GAP_KINDS = ("leave", "heal", "pc", "locked")


@dataclass(frozen=True)
class LiveRow:
    """After the fights ``fought`` (in the order played), with the field count ``live`` (1 or
    2) at the end of the last one: the fights the walk can start next (``next``, each with the
    map header it starts on in ``headers``; empty ``headers``: not recorded; empty ``next``:
    the chain ends under that count).

    A plain row (``at == ""``, ``to == live``) is the walk from the last fight's tile under
    ``live``; it holds the gap's facts in ``gap`` (a subset of ``GAP_KINDS``; None before the
    first fight). A change row is a count change inside the gap, from ``live`` to ``to``:
    ``"bag"`` (a revive item, used on any tile the walk under ``live`` reaches: the start menu,
    ``field_control.c:331-333``), ``"pc"`` (a deposit or a withdraw, on a tile next to a PC the
    walk reaches) or ``"heal"`` (a healer's ``HealParty``, at the healer's talk tile). Its
    ``next`` is the plain row's fights plus those the walk under ``to`` starts from those
    tiles; its gap facts are the plain row's. A count change by the gap's forced steps, before
    any player step, acts on the fight tile: the plain row of the new count."""

    fought: Order
    live: int
    next: tuple[Fight, ...]
    headers: tuple[int, ...] = ()
    to: int = 0  # 0: the same as live (a plain row)
    at: str = ""
    gap: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.to == 0:
            object.__setattr__(self, "to", self.live)

    @property
    def key(self) -> tuple[Order, int, int, str]:
        return (self.fought, self.live, self.to, self.at)

    @classmethod
    def from_dict(cls, d: Mapping) -> LiveRow:
        gap = d.get("gap")
        return cls(
            fought=tuple(tuple(f) for f in d["fought"]),
            live=d["live"],
            next=tuple(tuple(f) for f in d["next"]),
            headers=tuple(d.get("headers", ())),
            to=d.get("to", d["live"]),
            at=d.get("at", ""),
            gap=None if gap is None else tuple(gap),
        )

    def to_dict(self) -> dict:
        out: dict = {
            "fought": [list(f) for f in self.fought],
            "live": self.live,
            "next": [list(f) for f in self.next],
            "headers": list(self.headers),
        }
        if self.at:
            out["to"] = self.to
            out["at"] = self.at
        if self.gap is not None:
            out["gap"] = list(self.gap)
        return out


def prefix_live_rows(
    orders: Sequence[Order],
    leave_gaps: Sequence[Sequence[int]],
    fight_headers: Sequence[Sequence[int]],
    pc_gaps: Collection[int],
    heal_gaps: Collection[int],
    locked_gaps: Collection[int],
) -> tuple[LiveRow, ...]:
    """The next-fight table of a group whose walk does not depend on the live count: after
    ``fought`` the next fights are those of the orders that start with ``fought``, the same
    under both counts; gap k's facts are ``leave`` when every such order leaves there and the
    group's ``heal_gaps``, ``pc_gaps`` and ``locked_gaps``. Change rows: a revive (``bag``) at
    every gap that is not locked, a PC change both ways at the PC gaps that are not locked, a
    healer at the heal gaps that are not locked; each gives the plain row's fights."""
    everyone = frozenset(t for f in orders[0] for t in f)
    nxt: dict[Order, dict[Fight, int | None]] = {}
    leave: dict[Order, bool] = {}
    for i, o in enumerate(orders):
        heads = fight_headers[i] if fight_headers else None
        for k in range(len(o)):
            fought = tuple(o[:k])
            nxt.setdefault(fought, {})[o[k]] = None if heads is None else heads[k]
            if k:
                leave[fought] = leave.get(fought, True) and k in leave_gaps[i]
    rows: list[LiveRow] = []
    for fought, fights in sorted(nxt.items(), key=lambda kv: (len(kv[0]), kv[0])):
        order = tuple(sorted(fights))
        heads = () if None in fights.values() else tuple(fights[f] for f in order)
        k = len(fought)
        gap = None
        if k and frozenset(t for f in fought for t in f) != everyone:
            kinds = {
                "leave": leave[fought],
                "heal": k in heal_gaps,
                "pc": k in pc_gaps,
                "locked": k in locked_gaps,
            }
            gap = tuple(g for g in GAP_KINDS if kinds[g])
        for live in (1, 2):
            rows.append(LiveRow(fought, live, order, heads, gap=gap))
        if gap is None or "locked" in gap:
            continue
        changes = [(1, 2, "bag")]
        if "pc" in gap:
            changes += [(1, 2, "pc"), (2, 1, "pc")]
        if "heal" in gap:
            changes.append((1, 2, "heal"))
        for live, to, at in changes:
            rows.append(LiveRow(fought, live, order, heads, to, at))
    return tuple(rows)


# The gap fields a ``live_next`` table can make order- or count-dependent: field -> the
# ``LiveRow.gap`` kind it lists.
FLAT_GAP_FIELDS = {"leave_gaps": "leave", "heal_gaps": "heal", "pc_gaps": "pc"}


def flat_gaps(rows: Sequence[LiveRow], kind: str) -> tuple[int, ...] | None:
    """The gaps where ``kind`` (one of ``GAP_KINDS``) holds, read from the plain rows of a
    ``live_next`` table, when that is the same for every path and live count: gap k holds it
    in every row after k fights, or in none. None when some gap k holds it in one row and not
    in another (it depends on the order or the live count): the group then has no flat list
    for it, and a reader takes ``live_next`` (``GauntletFacts.gap_holds``)."""
    by_k: dict[int, set[bool]] = {}
    for r in rows:
        if r.gap is not None:
            by_k.setdefault(len(r.fought), set()).add(kind in r.gap)
    if any(len(v) > 1 for v in by_k.values()):
        return None
    return tuple(sorted(k for k, v in by_k.items() if True in v))


def check_live_rows(name: str, rows: Sequence[LiveRow], orders: Sequence[Order]) -> None:
    """A stored ``live_next`` table: well-formed rows, one per key, both roots, a plain row
    for every state a row's fights lead to (both counts) and for every change row, and the
    paths under two live Pokemon are ``orders`` (the group's facts are the two-live ones).
    Raises ``ValueError``."""
    everyone = frozenset(t for f in orders[0] for t in f)
    keys: dict[tuple, LiveRow] = {}
    for r in rows:
        where = f"{name}: live_next row {r.to_dict()}"
        if r.live not in (1, 2) or r.to not in (1, 2) or r.at not in LIVE_AT:
            raise ValueError(f"{where}: live and to are 1 or 2, at one of {LIVE_AT}")
        if (r.at == "") != (r.to == r.live) or (r.at in ("bag", "heal") and r.to < r.live):
            raise ValueError(f"{where}: a change row changes the count; a revive or heal raises it")
        seen = [t for f in r.fought for t in f]
        done = frozenset(seen)
        if len(done) != len(seen) or not done <= everyone or done == everyone:
            raise ValueError(f"{where}: fought is distinct trainers of the group, not all of them")
        if any(len(f) not in (1, 2) or not set(f) <= everyone - done for f in r.next):
            raise ValueError(f"{where}: next holds unfought fights of the group")
        if len(set(r.next)) != len(r.next) or (r.headers and len(r.headers) != len(r.next)):
            raise ValueError(f"{where}: next is distinct, headers one per fight or none")
        has_gap = bool(r.fought) and r.at == ""
        if (r.gap is not None) != has_gap or any(g not in GAP_KINDS for g in r.gap or ()):
            raise ValueError(f"{where}: gap facts on a plain row after a fight, of {GAP_KINDS}")
        if r.key in keys:
            raise ValueError(f"{where}: two rows for one key")
        keys[r.key] = r
    plain = {(r.fought, r.live): r for r in rows if not r.at}
    for live in (1, 2):
        if ((), live) not in plain:
            raise ValueError(f"{name}: live_next has no row before the first fight at {live}")
    for r in rows:
        if r.at and (r.fought, r.live) not in plain:
            raise ValueError(f"{name}: live_next change row {r.to_dict()} has no plain row")
        for f in r.next:
            after = (*r.fought, f)
            if frozenset(t for g in after for t in g) == everyone:
                continue
            for live in (1, 2):
                if (after, live) not in plain:
                    raise ValueError(f"{name}: live_next has no row after {after} at {live}")
    paths: set[Order] = set()

    def follow(fought: Order) -> None:
        if frozenset(t for g in fought for t in g) == everyone:
            paths.add(fought)
            return
        for f in plain[(fought, 2)].next:
            follow((*fought, f))

    follow(())
    if paths != set(orders):
        raise ValueError(f"{name}: live_next paths at two live {sorted(paths)} are not the orders")


@dataclass(frozen=True)
class GauntletFacts:
    """The tracked game facts of one group (design section 2). ``leave_gaps`` holds one
    tuple of gaps per order of ``orders``. ``leave_gaps``, ``heal_gaps`` and ``pc_gaps`` are
    None (absent) exactly when the group's ``live_next`` makes them depend on the order or the
    live count (``flat_gaps``); a reader then answers from ``live_next`` (``gap_holds``), never
    from an empty list."""

    name: str
    source: str
    orders: tuple[Order, ...]
    orders_source: str
    leave_gaps: tuple[tuple[int, ...], ...] | None
    leave_gaps_source: str
    escape: bool
    escape_source: str
    pc_gaps: tuple[int, ...] | None
    pc_gaps_source: str
    heal_gaps: tuple[int, ...] | None
    heal_gaps_source: str
    locked_gaps: tuple[int, ...]
    locked_gaps_source: str
    partner: Partner | None = None
    # Per order, the map header each fight starts on (tools/kaizo/gauntlet_map.py); empty: not
    # recorded.
    fight_headers: tuple[tuple[int, ...], ...] = ()
    # The next-fight table by the live count (``LiveRow``; tools/kaizo/gauntlet_map.py
    # derive_groups), stored only where the walk depends on the count; empty: it does not, and
    # ``live_rows`` gives the orders' prefixes (``prefix_live_rows``).
    live_next: tuple[LiveRow, ...] = ()

    def __post_init__(self):
        if self.leave_gaps is not None and len(self.leave_gaps) != len(self.orders):
            raise ValueError(f"{self.name}: leave_gaps needs one entry per order")
        if not self.live_next:
            absent = [k for k in FLAT_GAP_FIELDS if getattr(self, k) is None]
            if absent:
                raise ValueError(
                    f"{self.name}: {', '.join(absent)} absent without a live_next table "
                    "(only a table that makes a field order- or count-dependent drops it)"
                )
            return
        check_live_rows(self.name, self.live_next, self.orders)
        for key, kind in FLAT_GAP_FIELDS.items():
            flat = flat_gaps(self.live_next, kind)
            value = getattr(self, key)
            if flat is None:
                if value is not None:
                    raise ValueError(
                        f"{self.name}: {key} must be absent: its live_next gap facts "
                        f"({kind!r}) depend on the order or the live count"
                    )
                continue
            if key == "leave_gaps":
                flat = tuple(tuple(k for k in flat if k < len(o)) for o in self.orders)
            if value != flat:
                raise ValueError(
                    f"{self.name}: {key} {value} is not the list its live_next gap facts "
                    f"({kind!r}) give for every order and live count: {flat}"
                )

    @classmethod
    def from_dict(cls, d: Mapping) -> GauntletFacts:
        """The ``gauntlet_facts`` row of ``trainers.json`` (lists read as tuples)."""
        p = d.get("partner")
        return cls(
            name=d["name"],
            source=d["source"],
            orders=tuple(tuple(tuple(f) for f in o) for o in d["orders"]),
            orders_source=d["orders_source"],
            leave_gaps=None
            if d.get("leave_gaps") is None
            else tuple(tuple(g) for g in d["leave_gaps"]),
            leave_gaps_source=d["leave_gaps_source"],
            escape=bool(d["escape"]),
            escape_source=d["escape_source"],
            pc_gaps=None if d.get("pc_gaps") is None else tuple(d["pc_gaps"]),
            pc_gaps_source=d["pc_gaps_source"],
            heal_gaps=None if d.get("heal_gaps") is None else tuple(d["heal_gaps"]),
            heal_gaps_source=d["heal_gaps_source"],
            locked_gaps=tuple(d["locked_gaps"]),
            locked_gaps_source=d["locked_gaps_source"],
            partner=None
            if p is None
            else Partner(p["trainer"], tuple(p["join_before"]), p["source"]),
            fight_headers=tuple(tuple(h) for h in d.get("fight_headers", ())),
            live_next=tuple(LiveRow.from_dict(r) for r in d.get("live_next") or ()),
        )

    def to_dict(self) -> dict:
        """The inverse of ``from_dict`` (JSON-ready). A field that is empty or absent is
        omitted, as in ``data/kaizo/overrides.toml``: ``fight_headers`` and ``live_next`` when
        empty, and a gap field the group's ``live_next`` makes order- or count-dependent."""
        p = self.partner
        out: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "orders": [[list(f) for f in o] for o in self.orders],
            "orders_source": self.orders_source,
        }
        if self.leave_gaps is not None:
            out["leave_gaps"] = [list(g) for g in self.leave_gaps]
        out["leave_gaps_source"] = self.leave_gaps_source
        out["escape"] = self.escape
        out["escape_source"] = self.escape_source
        if self.pc_gaps is not None:
            out["pc_gaps"] = list(self.pc_gaps)
        out["pc_gaps_source"] = self.pc_gaps_source
        if self.heal_gaps is not None:
            out["heal_gaps"] = list(self.heal_gaps)
        out["heal_gaps_source"] = self.heal_gaps_source
        out["locked_gaps"] = list(self.locked_gaps)
        out["locked_gaps_source"] = self.locked_gaps_source
        out["partner"] = (
            None
            if p is None
            else {"trainer": p.trainer, "join_before": list(p.join_before), "source": p.source}
        )
        if self.fight_headers:
            out["fight_headers"] = [list(h) for h in self.fight_headers]
        if self.live_next:
            out["live_next"] = [r.to_dict() for r in self.live_next]
        return out

    def live_rows(self) -> tuple[LiveRow, ...]:
        """The next-fight table: ``live_next``, or the orders' prefixes when it is empty."""
        if self.live_next:
            return self.live_next
        assert self.leave_gaps is not None and self.pc_gaps is not None
        assert self.heal_gaps is not None  # __post_init__: absent only with live_next
        return prefix_live_rows(
            self.orders,
            self.leave_gaps,
            self.fight_headers,
            self.pc_gaps,
            self.heal_gaps,
            self.locked_gaps,
        )

    def _index(self, order: Order) -> int:
        try:
            return self.orders.index(order)
        except ValueError:
            raise ValueError(
                f"{self.name}: order {order!r} is not one of the group's orders"
            ) from None

    @staticmethod
    def n_fights(order: Order) -> int:
        return len(order)

    @staticmethod
    def fight_ids(order: Order) -> tuple[int, ...]:
        """The first trainer id of each fight; the name every setting uses for a fight."""
        return tuple(fight[0] for fight in order)

    def gap_holds(self, order: Order, k: int, kind: str) -> bool | None:
        """Whether gap k of ``order`` holds ``kind`` (``"leave"``, ``"heal"`` or ``"pc"``):
        from the flat field where the group has one, else from ``live_next`` (its plain rows
        after ``order[:k]``, under both live counts). None when the two counts disagree: the
        answer depends on the live count."""
        key = next(f for f, g in FLAT_GAP_FIELDS.items() if g == kind)
        flat = getattr(self, key)
        i = self._index(order)
        if flat is not None:
            return k in (flat[i] if key == "leave_gaps" else flat)
        fought = tuple(order[:k])
        seen = {kind in r.gap for r in self.live_next if r.fought == fought and r.gap is not None}
        if not seen:
            raise ValueError(f"{self.name}: live_next has no gap row after {fought}")
        return seen.pop() if len(seen) == 1 else None

    def _escape_at(self, order: Order, k: int) -> bool:
        """``escape`` holds at gap k and no partner is set yet (the partner who joins before
        fight J is set at gaps J and later; gap k is after fight k)."""
        if not self.escape or not 1 <= k < self.n_fights(order):
            return False
        return self.partner is None or k < self.partner.joins

    def walkout_at(self, order: Order, k: int) -> bool | None:
        """Whether the player can leave the map at gap k of ``order`` (design section 3):
        ``escape`` with no partner set, or a leave on foot (``gap_holds``). None when that
        depends on the live count."""
        return True if self._escape_at(order, k) else self.gap_holds(order, k, "leave")

    def leave_gaps_of(self, order: Order) -> tuple[int, ...]:
        """The gaps of ``order`` where the player can leave on foot: ``leave_gaps``, or read
        from ``live_next`` where the group has no flat list. Raises ``ValueError`` when one
        depends on the live count."""
        if self.leave_gaps is not None:
            return self.leave_gaps[self._index(order)]
        out = []
        for k in range(1, self.n_fights(order)):
            holds = self.gap_holds(order, k, "leave")
            if holds is None:
                raise ValueError(
                    f"{self.name}: leaving at gap {k} of {list(self.fight_ids(order))} depends "
                    "on the live count (live_next)"
                )
            if holds:
                out.append(k)
        return tuple(out)

    def walkout_gaps(self, order: Order) -> tuple[int, ...]:
        """Gaps where the player can leave the map: ``leave_gaps_of`` the order plus every
        gap where ``escape`` holds and no partner is set (design section 3). Raises
        ``ValueError`` when a gap depends on the live count (``walkout_at`` answers per gap)."""
        gaps = set(self.leave_gaps_of(order))
        gaps |= {k for k in range(1, self.n_fights(order)) if self._escape_at(order, k)}
        return tuple(sorted(gaps))


# --- settings and precedence --------------------------------------------------------


@dataclass(frozen=True)
class Layer:
    """One input layer (flags, a rules file or ``data/local.toml``): ``run`` and
    ``gauntlet`` hold the option values of those scopes, ``gaps`` the per-gap values by
    gap number, ``source`` the text the report prints."""

    source: str
    run: Mapping[str, Any] = field(default_factory=dict)
    gauntlet: Mapping[str, Any] = field(default_factory=dict)
    gaps: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)


def _check_gap_value(where: str, k: int, key: str, value) -> Any:
    try:
        if key in GAP_LEVEL_KEYS:
            return OPTIONS[key].validate(value)
        if key in GAP_CAP_KEYS:
            return _cap(value)
    except SettingsError as e:
        raise SettingsError(f"{where}: gap {k} {key}: {e}") from None
    raise SettingsError(f"{where}: gap {k}: {key!r} is not a per-gap key")


def _check_layer(layer: Layer, facts: GauntletFacts, order: Order) -> None:
    where = layer.source
    for scope, table in ((RUN, layer.run), (GAUNTLET, layer.gauntlet)):
        for key, value in table.items():
            option = OPTIONS.get(key)
            if option is None:
                raise SettingsError(f"{where}: unknown option {key!r} in {scope}")
            if option.scope != scope:
                raise SettingsError(f"{where}: {key!r} belongs to {option.scope}, not {scope}")
            try:
                option.validate(value)
            except SettingsError as e:
                raise SettingsError(f"{where}: {key}: {e}") from None
    n = facts.n_fights(order)
    for k, table in layer.gaps.items():
        if not 1 <= k <= n - 1:
            raise SettingsError(f"{where}: gap {k} is outside 1..{n - 1}")
        for key, value in table.items():
            _check_gap_value(where, k, key, value)


@dataclass(frozen=True)
class Settings:
    """The resolved options of one group and order: ``values`` by option name, ``gaps`` the
    per-gap overrides by gap, ``locked`` the gaps where the game gives no field control."""

    values: Mapping[str, Resolved]
    gaps: Mapping[int, Mapping[str, Resolved]]
    locked: tuple[int, ...]
    locked_source: str

    def __getitem__(self, name: str) -> Resolved:
        return self.values[name]

    def at_gap(self, k: int) -> dict[str, Resolved]:
        """The per-gap view: each Level option and cap at gap k. The group value is
        overridden by a gap value; a cap with no value is None (no per-gap limit). At a
        locked gap every Level option is its strictest value and every cap 0, origin LOCKED
        with the ``locked_gaps`` source."""
        if k in self.locked:
            lock = self.locked_source
            out = {n: Resolved(strictest(n), Origin.LOCKED, lock) for n in GAP_LEVEL_KEYS}
            out.update({c: Resolved(0, Origin.LOCKED, lock) for c in GAP_CAP_KEYS})
            return out
        out = {n: self.values[n] for n in GAP_LEVEL_KEYS}
        out.update({c: Resolved(None, Origin.DEFAULT, "no per-gap cap") for c in GAP_CAP_KEYS})
        out.update(self.gaps.get(k, {}))
        return out


def resolve(
    facts: GauntletFacts,
    order: Order,
    flag: Layer | None = None,
    rules_file: Layer | None = None,
    local: Layer | None = None,
) -> Settings:
    """Merge the layers per value: flag > rules file > local > default (a locked gap
    overrides all of them in ``Settings.at_gap``). A gap value overrides the group value
    whichever layer holds it; among layers the same precedence applies. Raises
    ``SettingsError`` for an unknown key, an invalid value, a key in the wrong scope or a
    gap outside 1..n-1."""
    facts._index(order)
    layers = [
        (origin, layer)
        for origin, layer in (
            (Origin.FLAG, flag),
            (Origin.RULES_FILE, rules_file),
            (Origin.LOCAL, local),
        )
        if layer is not None
    ]
    for _, layer in layers:
        _check_layer(layer, facts, order)

    values: dict[str, Resolved] = {}
    for name, option in OPTIONS.items():
        for origin, layer in layers:
            table = layer.run if option.scope == RUN else layer.gauntlet
            if name in table:
                values[name] = Resolved(option.validate(table[name]), origin, layer.source)
                break
        else:
            values[name] = Resolved(
                option.default, Origin.DEFAULT, "default" if option.has_default else "unset"
            )

    gaps: dict[int, dict[str, Resolved]] = {}
    for origin, layer in reversed(layers):  # lowest precedence first; higher overwrites
        for k, table in layer.gaps.items():
            for key, value in table.items():
                checked = _check_gap_value(layer.source, k, key, value)
                gaps.setdefault(k, {})[key] = Resolved(checked, origin, layer.source)
    return Settings(values, gaps, facts.locked_gaps, facts.locked_gaps_source)


# --- the settings input: flags, a rules file, data/local.toml [run] ------------------

LOCAL_TOML = Path(__file__).resolve().parents[3] / "data" / "local.toml"
FLAG_SOURCE = "--rule/--gap-rule"
GAP_KEYS = GAP_LEVEL_KEYS + GAP_CAP_KEYS


class GauntletSettingsError(SettingsError):
    """An input the loader refuses; the message names the origin, the group and the key
    where they are known (design section 3, "Input errors")."""

    def __init__(
        self,
        detail: str,
        *,
        key: str | None = None,
        group: str | None = None,
        origin: str | None = None,
    ) -> None:
        self.detail, self.key, self.group, self.origin = detail, key, group, origin
        parts = [origin, None if group is None else f"group {group!r}", key, detail]
        super().__init__(": ".join(p for p in parts if p))


@dataclass(frozen=True)
class RawGroup:
    """The options and gap tables one rules file holds for one group."""

    gauntlet: Mapping[str, Any] = field(default_factory=dict)
    gaps: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class RawSettings:
    """One input as read, before validation: ``run`` the ``[run]`` table, ``gauntlet`` and
    ``gaps`` the values that apply to whichever group is selected (flags), ``groups`` the
    per-group tables of a rules file. ``layer`` is the ``Layer`` ``resolve`` merges."""

    source: str
    run: Mapping[str, Any] = field(default_factory=dict)
    gauntlet: Mapping[str, Any] = field(default_factory=dict)
    gaps: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    groups: Mapping[str, RawGroup] = field(default_factory=dict)

    def layer(self, group: str) -> Layer:
        mine = self.groups.get(group, RawGroup())
        gaps: dict[int, dict[str, Any]] = {k: dict(t) for k, t in self.gaps.items()}
        for k, table in mine.gaps.items():
            gaps.setdefault(k, {}).update(table)
        return Layer(
            self.source,
            run=dict(self.run),
            gauntlet={**self.gauntlet, **mine.gauntlet},
            gaps=gaps,
        )

    def unknown_groups(self, known: Collection[str]) -> tuple[str, ...]:
        return tuple(sorted(set(self.groups) - set(known)))


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except OSError as e:
        raise GauntletSettingsError(f"cannot read ({e.strerror or e})", origin=str(path)) from e
    except tomllib.TOMLDecodeError as e:
        raise GauntletSettingsError(str(e), origin=str(path)) from e


def _table(value: Any, what: str, origin: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise GauntletSettingsError(f"{what} is not a table", origin=origin)
    return value


def load_rules_file(path: Path | str) -> RawSettings:
    """A rules file the user keeps outside the tracked tree (design section 3): ``[run]``,
    ``[gauntlet."<group>"]`` and ``[gauntlet."<group>".gap.<k>]``. Values are checked by
    ``resolve_settings``, which knows the group."""
    path = Path(path)
    origin = str(path)
    data = _read_toml(path)
    unknown = sorted(set(data) - {RUN, GAUNTLET})
    if unknown:
        raise GauntletSettingsError(
            f"unknown top-level key(s) {', '.join(unknown)}; known: run, gauntlet", origin=origin
        )
    run = dict(_table(data.get(RUN, {}), "[run]", origin))
    groups: dict[str, RawGroup] = {}
    for name, body in _table(data.get(GAUNTLET, {}), "[gauntlet]", origin).items():
        body = _table(body, f"[gauntlet.{name!r}]", origin)
        gaps: dict[int, dict[str, Any]] = {}
        for k, table in _table(body.get("gap", {}), f"[gauntlet.{name!r}.gap]", origin).items():
            if not k.isdecimal():
                raise GauntletSettingsError(
                    f"gap table {k!r} is not a gap number",
                    key=f"gap.{k}",
                    group=name,
                    origin=origin,
                )
            gaps[int(k)] = dict(_table(table, f"[gauntlet.{name!r}.gap.{k}]", origin))
        groups[name] = RawGroup({k: v for k, v in body.items() if k != "gap"}, gaps)
    return RawSettings(origin, run=run, groups=groups)


def load_local(path: Path | str = LOCAL_TOML) -> RawSettings:
    """Only the ``[run]`` table of ``data/local.toml`` (no import from ``tools``). A missing
    file or table is no error; every other key is the ROM tools' (``tools/rom/inputs.py``)."""
    path = Path(path)
    source = f"{path} [run]"
    if not path.exists():
        return RawSettings(source)
    run = _read_toml(path).get(RUN, {})
    return RawSettings(source, run=dict(_table(run, "[run]", source)))


_INT = re.compile(r"-?\d+")


def _parse_rule_value(name: str, text: str) -> Any:
    if name == "swap":
        pairs: dict[str, int] = {}
        for part in text.split(","):
            key, _, num = part.partition(":")
            if key not in ("bench", "budget") or key in pairs or not _INT.fullmatch(num):
                raise SettingsError(f"{text!r} is not bench:N,budget:N")
            pairs[key] = int(num)
        return pairs
    if name == "order":
        if text in ("worst", "sheet", "best"):
            return text
        ids = text.split(",")
        if not all(_INT.fullmatch(i) for i in ids):
            raise SettingsError(f"{text!r} is not worst, sheet, best or comma-separated fight ids")
        return [int(i) for i in ids]
    if name == "move_sources":
        return text.split(",")
    if name == "bag_uses":
        return int(text) if _INT.fullmatch(text) else text
    return text


def parse_flags(rule: Sequence[str], gap_rule: Sequence[str] = ()) -> RawSettings:
    """``--rule heal=bag --rule swap=bench:1,budget:1 --gap-rule 2:heal=full`` as one
    input. A key goes to ``run`` or ``gauntlet`` by its option's scope; a key given twice,
    an unknown key or a malformed value is a ``GauntletSettingsError``; the value itself is
    checked by ``resolve_settings``."""
    run: dict[str, Any] = {}
    gauntlet: dict[str, Any] = {}
    for item in rule:
        name, eq, text = item.partition("=")
        if not eq or name not in OPTIONS:
            raise GauntletSettingsError(
                f"{item!r} is not KEY=VALUE with a key of {sorted(OPTIONS)}",
                key=name or item,
                origin="--rule",
            )
        table = run if OPTIONS[name].scope == RUN else gauntlet
        if name in table:
            raise GauntletSettingsError("given twice", key=name, origin="--rule")
        try:
            table[name] = _parse_rule_value(name, text)
        except SettingsError as e:
            raise GauntletSettingsError(str(e), key=name, origin="--rule") from None
    gaps: dict[int, dict[str, Any]] = {}
    for item in gap_rule:
        m = re.fullmatch(r"(\d+):([a-z_]+)=(.*)", item)
        if m is None or m.group(2) not in GAP_KEYS:
            raise GauntletSettingsError(
                f"{item!r} is not K:KEY=VALUE with a key of {list(GAP_KEYS)}",
                key=item,
                origin="--gap-rule",
            )
        k, key, text = int(m.group(1)), m.group(2), m.group(3)
        if key in GAP_CAP_KEYS and not _INT.fullmatch(text):
            raise GauntletSettingsError(
                f"{text!r} is not an integer", key=f"gap {k} {key}", origin="--gap-rule"
            )
        if key in gaps.setdefault(k, {}):
            raise GauntletSettingsError("given twice", key=f"gap {k} {key}", origin="--gap-rule")
        gaps[k][key] = int(text) if key in GAP_CAP_KEYS else text
    return RawSettings(FLAG_SOURCE, run=run, gauntlet=gauntlet, gaps=gaps)


# --- per-run inputs of a re-run ------------------------------------------------------


class Member(NamedTuple):
    """A save Pokemon as ``Candidate.location`` names it: ``("party", 0, slot)`` or
    ``("box", box, slot)`` with the box 0-based (the user's ``box:B:S`` is 1-based)."""

    kind: str
    box: int
    slot: int

    @property
    def label(self) -> str:
        return f"party:{self.slot}" if self.kind == "party" else f"box:{self.box + 1}:{self.slot}"


def parse_ids(values: Iterable[str], flag: str) -> tuple[int, ...]:
    """Trainer ids of ``--beaten`` (played order, repeats kept), commas or spaces."""
    out = []
    for value in values:
        for part in value.split(","):
            if not _INT.fullmatch(part.strip()):
                raise GauntletSettingsError(f"{part!r} is not a trainer id", key=flag, origin=flag)
            out.append(int(part))
    return tuple(out)


def parse_members(values: Iterable[str], flag: str) -> tuple[Member, ...]:
    """``party:N`` and ``box:B:S`` (the labels of ``save show``; B 1-based), commas or
    spaces; sorted, each once."""
    out: set[Member] = set()
    for value in values:
        for part in value.split(","):
            text = part.strip()
            if m := re.fullmatch(r"party:(\d+)", text, re.IGNORECASE):
                out.add(Member("party", 0, int(m.group(1))))
            elif (m := re.fullmatch(r"box:(\d+):(\d+)", text, re.IGNORECASE)) and int(m[1]) >= 1:
                out.add(Member("box", int(m[1]) - 1, int(m[2])))
            else:
                raise GauntletSettingsError(
                    f"{part!r} is not party:N or box:B:S (B from 1)", key=flag, origin=flag
                )
    return tuple(sorted(out))


@dataclass(frozen=True)
class RunInput:
    """The per-run inputs of a re-run mid-gauntlet (design sections 3 and 5): nothing is
    tracked or written, the save holds none of it."""

    beaten: tuple[int, ...] = ()
    dead: tuple[Member, ...] = ()
    bench: tuple[Member, ...] = ()
    swaps_left: int | None = None
    horizon: int | None = None
    growth_cap: int | None = None
    next_fight: int | None = None


def _check_members(run: RunInput, save: Any, err: Callable[..., GauntletSettingsError]) -> None:
    """``--dead`` and ``--bench`` against the save (``party`` of members with
    ``current_hp``, ``boxes`` of box rows with ``None`` for an empty slot)."""
    if (run.dead or run.bench) and save is None:
        raise err("needs --save: the party and the boxes come from it", key="--dead/--bench")

    def present(m: Member, flag: str):
        if m.kind == "party":
            if not 0 <= m.slot < len(save.party):
                raise err(f"{m.label}: the save's party has {len(save.party)} Pokemon", key=flag)
            return save.party[m.slot]
        if (
            not 0 <= m.box < len(save.boxes)
            or not 0 <= m.slot < len(save.boxes[m.box])
            or save.boxes[m.box][m.slot] is None
        ):
            raise err(f"{m.label}: no Pokemon there in the save's boxes", key=flag)
        return save.boxes[m.box][m.slot]

    for m in run.bench:
        if m.kind != "box":
            raise err(f"{m.label}: the bench holds box members only", key="--bench")
        present(m, "--bench")
    for m in run.dead:
        mon = present(m, "--dead")
        if m.kind == "party" and mon.current_hp == 0:
            raise err(f"{m.label}: already at 0 HP, so already dead", key="--dead")
    both = sorted(set(run.dead) & set(run.bench))
    if both:
        raise err(f"{both[0].label} is in both --dead and --bench", key="--bench")


# --- the open orders -----------------------------------------------------------------


def open_orders(
    facts: GauntletFacts,
    order_value: Any,
    beaten: Sequence[int] = (),
    next_fight: int | None = None,
) -> tuple[Order, ...]:
    """The orders still open: the one a list value names, else every order of the group
    (``"worst"``, ``"best"``, and ``"sheet"``, which falls back to them because the facts
    hold no sheet order); then those whose first fights match ``--beaten`` (any member id
    of a fight names it) and whose next fight holds ``next_fight``. Empty when none is
    left; ``resolve_settings`` raises then."""
    orders = facts.orders
    if isinstance(order_value, tuple):
        orders = tuple(o for o in orders if facts.fight_ids(o) == order_value)
    kept = []
    for o in orders:
        if len(beaten) > len(o) or any(b not in o[i] for i, b in enumerate(beaten)):
            continue
        if next_fight is not None and (len(beaten) >= len(o) or next_fight not in o[len(beaten)]):
            continue
        kept.append(o)
    return tuple(kept)


def _gap_allows(facts: GauntletFacts, order: Order, k: int, need: str) -> bool | None:
    """Whether the map gives ``need`` at gap k of ``order``: a walk-out, or the gap's own
    heal or PC. None when that depends on the live count (``live_next``)."""
    out = facts.walkout_at(order, k)
    own = facts.gap_holds(order, k, need)
    if out or own:
        return True
    return None if out is None or own is None else False


def _map_refusal(facts: GauntletFacts, orders: Sequence[Order], k: int, need: str) -> str | None:
    """Why the map gives no ``need`` ("heal" or "pc") at gap k, else None. Where an order
    is open, a value is refused only when refused under every open order in which gap k
    exists."""
    open_ = [o for o in orders if 1 <= k <= len(o) - 1]
    allows = [_gap_allows(facts, o, k, need) for o in open_]
    if not open_ or True in allows:
        return None
    what, listed, src = (
        ("a free full heal", facts.heal_gaps, facts.heal_gaps_source)
        if need == "heal"
        else ("a PC", facts.pc_gaps, facts.pc_gaps_source)
    )
    if None in allows:
        return (
            f"whether the map gives {what} at gap {k} depends on the live count there "
            f"(live_next: {facts.orders_source}), which the settings do not fix"
        )
    gaps = "by order and live count, live_next" if listed is None else f"gaps {list(listed)}"
    return (
        f"the map gives {what} at gap {k} under no open order "
        f"({gaps}: {src}; walk-out gaps: {facts.leave_gaps_source}, "
        f"{facts.escape_source})"
    )


def check_swap_gap(
    facts: GauntletFacts,
    order: Order,
    k: int,
    *,
    origin: str = "swap plan",
) -> None:
    """A swap at gap k needs a PC or a walk-out there (design section 3): a
    ``GauntletSettingsError`` otherwise."""
    why = _map_refusal(facts, (order,), k, "pc")
    if why is not None:
        raise GauntletSettingsError(
            f"a swap at gap {k}: {why}", key="swap", group=facts.name, origin=origin
        )


# --- the whole input ----------------------------------------------------------------


@dataclass(frozen=True)
class GauntletSetup:
    """One group's resolved input: ``settings`` merged over the longest open order,
    ``orders`` still open (consumers take the minimum over them under ``"worst"``), ``run``
    the re-run inputs, ``sheet_fallback`` true when ``order = "sheet"`` fell back to
    ``"worst"``."""

    facts: GauntletFacts
    settings: Settings
    orders: tuple[Order, ...]
    run: RunInput
    sheet_fallback: bool = False


def _first(layers, scope: str, key: str):
    for origin, layer in layers:
        table = layer.run if scope == RUN else layer.gauntlet
        if key in table:
            return table[key], origin, layer
    return None


def resolve_settings(
    facts: GauntletFacts,
    *,
    flags: RawSettings | None = None,
    rules_file: RawSettings | None = None,
    local: RawSettings | None = None,
    run: RunInput | None = None,
    save: Any = None,
    known_groups: Collection[str] | None = None,
) -> GauntletSetup:
    """The loader's one entry (design section 3): merge flag > rules file > local >
    default for ``facts``' group, then apply every input error. Locked gaps override all
    of it in ``Settings.at_gap``. Raises ``GauntletSettingsError`` naming key, group and
    origin; ``save`` (anything with ``party`` and ``boxes``) checks ``--dead`` and
    ``--bench``."""
    run = run or RunInput()
    group = facts.name

    def err(detail: str, *, key: str | None = None, origin: str | None = None):
        return GauntletSettingsError(detail, key=key, group=group, origin=origin)

    raws = [(Origin.FLAG, flags), (Origin.RULES_FILE, rules_file), (Origin.LOCAL, local)]
    if known_groups is not None:
        for _, raw in raws:
            if raw is not None and (bad := raw.unknown_groups(known_groups)):
                raise err(f"no such gauntlet group {bad[0]!r}", key="gauntlet", origin=raw.source)
    layers = [(origin, raw.layer(group)) for origin, raw in raws if raw is not None]

    for _, layer in layers:
        if "order" in layer.gauntlet:
            try:
                OPTIONS["order"].validate(layer.gauntlet["order"])
            except SettingsError as e:
                raise err(str(e), key="order", origin=layer.source) from None
    for name, value, check in (
        ("--swaps-left", run.swaps_left, _count),
        ("--horizon", run.horizon, _cap),
        ("--growth-cap", run.growth_cap, _count),
    ):
        if value is not None:
            try:
                check(value)
            except SettingsError as e:
                raise err(str(e), key=name, origin=name) from None

    # order: a list names one of the group's orders (checked in every layer)
    all_orders = [facts.fight_ids(o) for o in facts.orders]
    for _, layer in layers:
        ov = layer.gauntlet.get("order")
        if isinstance(ov, (list, tuple)) and tuple(ov) not in all_orders:
            raise err(
                f"{list(ov)} is not one of the group's orders {[list(o) for o in all_orders]} "
                f"({facts.orders_source})",
                key="order",
                origin=layer.source,
            )

    found = _first(layers, GAUNTLET, "order")
    order_value = OPTIONS["order"].validate(found[0]) if found else OPTIONS["order"].default

    orders = open_orders(facts, order_value, run.beaten, run.next_fight)
    if not orders:
        if run.next_fight is not None and open_orders(facts, order_value, run.beaten):
            raise err(
                f"{run.next_fight} is not the next fight of any open order",
                key="--next",
                origin="--next",
            )
        raise err(
            f"{list(run.beaten)} is not the start of any open order "
            f"{[list(facts.fight_ids(o)) for o in open_orders(facts, order_value)]}",
            key="--beaten",
            origin="--beaten",
        )
    longest = max(orders, key=len)

    for _, layer in layers:
        for k in layer.gaps:
            if not any(1 <= k <= len(o) - 1 for o in orders):
                top = len(longest) - 1
                raise err(f"gap {k} is outside 1..{top}", key=f"gap {k}", origin=layer.source)

    by_origin = dict(layers)
    try:
        settings = resolve(
            facts,
            longest,
            by_origin.get(Origin.FLAG),
            by_origin.get(Origin.RULES_FILE),
            by_origin.get(Origin.LOCAL),
        )
    except GauntletSettingsError:
        raise
    except SettingsError as e:
        raise err(str(e)) from None

    for _, layer in layers:
        for k, table in layer.gaps.items():
            for key, value in table.items():
                where = f"gap {k} {key}"
                if k in facts.locked_gaps:
                    strict = strictest(key) if key in GAP_LEVEL_KEYS else 0
                    if value != strict:
                        raise err(
                            f"{value!r} at locked gap {k}: only {strict!r} holds there "
                            f"({facts.locked_gaps_source})",
                            key=where,
                            origin=layer.source,
                        )
                    continue
                if (key, value) == ("heal", "full"):
                    need = "heal"
                elif (key, value) == ("pc_heal", "on") or (key == "swap_cap" and value > 0):
                    need = "pc"
                else:
                    continue
                if why := _map_refusal(facts, orders, k, need):
                    raise err(f"{value!r}: {why}", key=where, origin=layer.source)

    _check_members(run, save, err)
    return GauntletSetup(facts, settings, orders, run, sheet_fallback=order_value == "sheet")
