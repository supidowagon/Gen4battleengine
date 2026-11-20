"""Gauntlet options, their precedence and the tracked game facts of a group (design v5.3
sections 1-3, 13).

``OPTIONS`` is the one registry of user options; the loader, CLI, coverage and report
iterate it and nothing else lists option names. Every option carries the name of its
transition in ``prep/gauntlet.py``; ``missing_transitions`` names the options without one
and the loader refuses them. Every default is the strictest value, the one whose score is
a lower bound under every user rule (design section 13): for ``order`` and
``partner_from`` that is ``"worst"``. ``time`` has no default.

``GauntletFacts`` holds what the game fixes for one group (design section 2); user
choices never appear in it. Fights are numbered 1..n along an order; gap k sits after the
k-th fight played (1 <= k <= n-1)."""

from __future__ import annotations

import enum
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
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


def _partner_from(v):
    if v in ("worst", "best"):
        return v
    if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
        return v
    raise SettingsError(f"{v!r} is not worst, best or a fight number")


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
        "partner_from",
        CHOICE,
        GAUNTLET,
        False,
        ("worst", "best"),
        "worst",
        _partner_from,
        "partner_from",
    ),
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
    """The Wayward Cave partner: ``trainer`` and the fights before which she can join."""

    trainer: int
    join_before: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class GauntletFacts:
    """The tracked game facts of one group (design section 2). ``leave_gaps`` holds one
    tuple of gaps per order of ``orders``."""

    name: str
    source: str
    orders: tuple[Order, ...]
    orders_source: str
    leave_gaps: tuple[tuple[int, ...], ...]
    leave_gaps_source: str
    escape: bool
    escape_source: str
    pc_gaps: tuple[int, ...]
    pc_gaps_source: str
    heal_gaps: tuple[int, ...]
    heal_gaps_source: str
    locked_gaps: tuple[int, ...]
    locked_gaps_source: str
    partner: Partner | None = None

    def __post_init__(self):
        if len(self.leave_gaps) != len(self.orders):
            raise ValueError(f"{self.name}: leave_gaps needs one entry per order")

    @classmethod
    def from_dict(cls, d: Mapping) -> GauntletFacts:
        """The ``gauntlet_facts`` row of ``trainers.json`` (lists read as tuples)."""
        p = d.get("partner")
        return cls(
            name=d["name"],
            source=d["source"],
            orders=tuple(tuple(tuple(f) for f in o) for o in d["orders"]),
            orders_source=d["orders_source"],
            leave_gaps=tuple(tuple(g) for g in d["leave_gaps"]),
            leave_gaps_source=d["leave_gaps_source"],
            escape=bool(d["escape"]),
            escape_source=d["escape_source"],
            pc_gaps=tuple(d["pc_gaps"]),
            pc_gaps_source=d["pc_gaps_source"],
            heal_gaps=tuple(d["heal_gaps"]),
            heal_gaps_source=d["heal_gaps_source"],
            locked_gaps=tuple(d["locked_gaps"]),
            locked_gaps_source=d["locked_gaps_source"],
            partner=None
            if p is None
            else Partner(p["trainer"], tuple(p["join_before"]), p["source"]),
        )

    def to_dict(self) -> dict:
        """The inverse of ``from_dict`` (JSON-ready)."""
        p = self.partner
        return {
            "name": self.name,
            "source": self.source,
            "orders": [[list(f) for f in o] for o in self.orders],
            "orders_source": self.orders_source,
            "leave_gaps": [list(g) for g in self.leave_gaps],
            "leave_gaps_source": self.leave_gaps_source,
            "escape": self.escape,
            "escape_source": self.escape_source,
            "pc_gaps": list(self.pc_gaps),
            "pc_gaps_source": self.pc_gaps_source,
            "heal_gaps": list(self.heal_gaps),
            "heal_gaps_source": self.heal_gaps_source,
            "locked_gaps": list(self.locked_gaps),
            "locked_gaps_source": self.locked_gaps_source,
            "partner": None
            if p is None
            else {"trainer": p.trainer, "join_before": list(p.join_before), "source": p.source},
        }

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

    def leave_gaps_of(self, order: Order) -> tuple[int, ...]:
        return self.leave_gaps[self._index(order)]

    def walkout_gaps(self, order: Order, partner_from: int | None) -> tuple[int, ...]:
        """Gaps where the player can leave the map: ``leave_gaps`` of the order plus every
        gap where ``escape`` holds and no partner is set (design section 3). The partner
        who joins before fight J is set at gaps J and later (gap k is after fight k); with
        ``partner_from`` None she is never set. A group without a partner fact takes
        ``partner_from`` None."""
        if self.partner is None and partner_from is not None:
            raise ValueError(f"{self.name} has no partner")
        gaps = set(self.leave_gaps_of(order))
        if self.escape:
            for k in range(1, self.n_fights(order)):
                if partner_from is None or k < partner_from:
                    gaps.add(k)
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
    pf = layer.gauntlet.get("partner_from")
    if pf is not None:
        if facts.partner is None:
            raise SettingsError(f"{where}: partner_from needs a group with a partner")
        if pf not in ("worst", "best") and pf not in facts.partner.join_before:
            raise SettingsError(
                f"{where}: partner_from {pf} is not in join_before "
                f"{list(facts.partner.join_before)}"
            )
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
    ``SettingsError`` for an unknown key, an invalid value, a key in the wrong scope,
    ``partner_from`` outside ``partner.join_before`` or a gap outside 1..n-1."""
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
