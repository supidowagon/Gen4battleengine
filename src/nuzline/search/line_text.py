"""The line file (docs/superpowers/specs/2026-11-12-l1-line-player-design.md §6): a TOML
file with an optional ``[[team]]`` (the battle order, with ``--save``) and one ``[[rule]]``
per opposing Pokemon (spec §3).

``[[team]]``: ``pokemon`` is a species name or nickname (case-insensitive), ``party:N`` or
``box:B:S`` (the labels of ``save show``, B 1-based); an egg cannot be named and no Pokemon
twice; 1-6 entries. ``moves`` (1-4 names) replaces all four slots at full PP; ``item``
replaces the held item (``""`` = none). Neither is checked against the bag: an override is
the user's claim that the TM or item is available.

``[[rule]]``: ``against`` is an opposing species (case-insensitive) or ``slot:N`` (0-based),
which a species the opponent has twice needs; ``use``, ``via`` and ``from`` name a team
member by species, by its ``[[team]]`` ``pokemon`` string, or ``slot:N`` (0-based), the
latter needed for a name matching two members; ``move`` and ``first`` are moves ``use``
knows. ``from`` needs ``via``, ``wait`` excludes ``via``, ``via`` differs from ``use``,
``from`` differs from ``via``, and a foe has at most one rule.

Every failure is a ``ValueError`` naming the table (``rule 3 (against Gible): ...``).
This module does not import jax: ``Line`` is imported inside the two functions that build
or read one.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nuzline.data.spec import PokemonSpec

if TYPE_CHECKING:
    from nuzline.data.tables import Names
    from nuzline.search.line import Line

__all__ = [
    "LineFile",
    "Rule",
    "line_to_rules",
    "read_line_file",
    "render_rules",
    "resolve_rules",
    "rules_to_line",
    "team_from_save",
    "write_line_file",
]

SYNTAX = "party:N or box:B:S"  # as prep.problem.pick (not imported: it pulls in jax)
TEAM_KEYS = ("pokemon", "moves", "item")
RULE_KEYS = ("against", "use", "move", "first", "via", "from", "wait")
TEAM_MAX = 6


@dataclass(frozen=True)
class Rule:
    """One rule with its names resolved: opposing slot ``against``, party slots ``use``,
    ``via``, ``via_from`` and move slots ``move``, ``first`` (-1 = unset)."""

    against: int
    use: int
    move: int
    first: int = -1
    via: int = -1
    via_from: int = -1
    wait: bool = False


@dataclass(frozen=True)
class LineFile:
    """The ``[[team]]`` and ``[[rule]]`` tables as read (``()`` when absent)."""

    team: tuple[dict, ...]
    rules: tuple[dict, ...]


def read_line_file(path: Path) -> LineFile:
    """Parse ``path``; ``ValueError`` for bad TOML, an unknown top-level key, or a ``team``
    / ``rule`` that is not an array of tables. Names are checked by ``team_from_save`` and
    ``resolve_rules``."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None
    extra = sorted(set(data) - {"team", "rule"})
    if extra:
        raise ValueError(f"{path}: unknown key(s) {', '.join(extra)} (expected [[team]], [[rule]])")
    out = {}
    for key in ("team", "rule"):
        tables = data.get(key, [])
        if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
            raise ValueError(f"{path}: {key} must be [[{key}]] tables")
        out[key] = tuple(tables)
    return LineFile(team=out["team"], rules=out["rule"])


# --- [[team]] -----------------------------------------------------------------------------


def _check_keys(where: str, table: dict, allowed: Sequence[str]) -> None:
    extra = sorted(set(table) - set(allowed))
    if extra:
        raise ValueError(
            f"{where}: unknown key(s) {', '.join(extra)} (allowed: {', '.join(allowed)})"
        )


def _where(mon) -> str:
    kind, box, slot = mon.location
    return f"party:{slot}" if kind == "party" else f"box:{box + 1}:{slot}"


def _pick_mon(mons: Sequence, names: Names, value: str):
    """The save Pokemon ``value`` names, with ``prep.problem.pick``'s syntax and messages:
    ``party:N``, ``box:B:S`` (B 1-based), else a case-insensitive species name or nickname
    (eggs never match a name). ``ValueError`` for no match, several, or an egg."""
    text = value.strip()
    want = None
    if m := re.fullmatch(r"party:(\d+)", text, re.IGNORECASE):
        want = ("party", 0, int(m.group(1)))
    elif m := re.fullmatch(r"box:(\d+):(\d+)", text, re.IGNORECASE):
        want = ("box", int(m.group(1)) - 1, int(m.group(2)))
    if want is not None:
        mon = next((m for m in mons if m.location == want), None)
        if mon is None:
            raise ValueError(f"no Pokemon at {value!r} ({SYNTAX}, as in the labels of `save show`)")
        if mon.is_egg:
            raise ValueError(f"{value!r} is an egg (an egg cannot be on the team)")
        return mon
    key = text.casefold()
    hits = [
        m
        for m in mons
        if not m.is_egg
        and key in {names.species_names[m.species].casefold(), m.nickname.casefold()}
    ]
    if not hits:
        raise ValueError(f"no Pokemon matches {value!r} (name it by species, nickname or {SYNTAX})")
    if len(hits) > 1:
        listing = "; ".join(f"{names.species_names[m.species]} ({_where(m)})" for m in hits)
        raise ValueError(f"{value!r} names {len(hits)} Pokemon: {listing}; use {SYNTAX}")
    return hits[0]


def _lookup(fn, name, what: str) -> int:
    if not isinstance(name, str):
        raise ValueError(f"{what} must be a string, not {name!r}")  # noqa: TRY004
    try:
        return fn(name)
    except KeyError as e:
        raise ValueError(e.args[0]) from None


def _team_spec(entry: dict, mon, tables, names: Names) -> PokemonSpec:
    spec = mon.to_spec(tables)
    if "moves" in entry:
        moves = entry["moves"]
        if not isinstance(moves, list) or not 1 <= len(moves) <= 4:
            raise ValueError(f"moves must be a list of 1-4 move names, not {moves!r}")
        ids = [_lookup(names.move, m, "a move") for m in moves]
        if 0 in ids or len(set(ids)) != len(ids):
            raise ValueError(f"moves {moves!r}: empty or repeated move")
        spec = dataclasses.replace(spec, moves=tuple(ids + [0] * (4 - len(ids))), pp=None)
    if "item" in entry:
        spec = dataclasses.replace(spec, item=_lookup(names.item, entry["item"], "item"))
    return spec


def team_from_save(file: LineFile, save, tables, names: Names) -> list[PokemonSpec]:
    """The ``[[team]]`` tables as specs from ``save``'s party and boxes, in file order (the
    first leads), with each ``moves`` / ``item`` override applied."""
    if not 1 <= len(file.team) <= TEAM_MAX:
        raise ValueError(f"[[team]] needs 1-{TEAM_MAX} entries, the file has {len(file.team)}")
    mons = list(save.party) + [m for box in save.boxes for m in box if m is not None]
    team: list[PokemonSpec] = []
    seen: dict[tuple, int] = {}
    for i, entry in enumerate(file.team, 1):
        name = entry.get("pokemon")
        where = f"team {i} ({name})"
        try:
            _check_keys(where, entry, TEAM_KEYS)
            if not isinstance(name, str):
                raise ValueError(f"pokemon (species, nickname or {SYNTAX}) is required")  # noqa: TRY004
            mon = _pick_mon(mons, names, name)
            if mon.location in seen:
                raise ValueError(f"{_where(mon)} is listed twice (also team {seen[mon.location]})")
            seen[mon.location] = i
            team.append(_team_spec(entry, mon, tables, names))
        except ValueError as e:
            raise _named(where, e) from None
    return team


def _named(where: str, e: ValueError) -> ValueError:
    text = str(e)
    return ValueError(text if text.startswith(where) else f"{where}: {text}")


# --- [[rule]] -----------------------------------------------------------------------------

_SLOT = re.compile(r"slot:(\d+)", re.IGNORECASE)


def _slot_or_species(
    value,
    party: Sequence[PokemonSpec],
    names: Names,
    field: str,
    side: str,
    labels: Sequence[str] = (),
) -> int:
    """The index in ``party`` that ``value`` names: ``slot:N``, or a species name or a
    ``labels`` entry (slot ``i``'s listed name, case-insensitive) matching one slot only."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a species name or slot:N, not {value!r}")  # noqa: TRY004
    if m := _SLOT.fullmatch(value.strip()):
        slot = int(m.group(1))
        if slot >= len(party):
            raise ValueError(f"{field} {value!r}: {side} has slots 0-{len(party) - 1}")
        return slot
    species = names.species_names
    key = value.strip().casefold()
    hits = [
        i
        for i, p in enumerate(party)
        if key == species[p.species].casefold()
        or (i < len(labels) and key == labels[i].strip().casefold())
    ]
    if not hits:
        listed = ", ".join(species[p.species] for p in party)
        raise ValueError(f"{field} {value!r}: {side} has no such Pokemon ({listed})")
    if len(hits) > 1:
        slots = ", ".join(str(i) for i in hits)
        raise ValueError(
            f"{field} {value!r}: {side} has {len(hits)} of it (slots {slots}); use slot:N"
        )
    return hits[0]


def _move_slot(value, spec: PokemonSpec, names: Names, field: str) -> int:
    move = _lookup(names.move, value, field)
    if move == 0 or move not in spec.moves:
        raise ValueError(f"{field}: {names.species_names[spec.species]} does not know {value!r}")
    return spec.moves.index(move)


def _resolve_rule(
    table: dict,
    names: Names,
    player: Sequence[PokemonSpec],
    opp: Sequence[PokemonSpec],
    labels: Sequence[str],
) -> Rule:
    for key in ("against", "use", "move"):
        if key not in table:
            raise ValueError(f"{key} is required")
    against = _slot_or_species(table["against"], opp, names, "against", "the opponent")

    def member(field: str) -> int:
        return _slot_or_species(table[field], player, names, field, "the team", labels)

    use = member("use")
    move = _move_slot(table["move"], player[use], names, "move")
    first = _move_slot(table["first"], player[use], names, "first") if "first" in table else -1
    via = member("via") if "via" in table else -1
    if "from" in table and via < 0:
        raise ValueError("from needs via")
    via_from = member("from") if "from" in table else -1
    wait = table.get("wait", False)
    if not isinstance(wait, bool):
        raise ValueError(f"wait must be true or false, not {wait!r}")  # noqa: TRY004
    if wait and via >= 0:
        raise ValueError("wait cannot be set with via")
    if via == use:
        raise ValueError("via is the same Pokemon as use")
    if via_from >= 0 and via_from == via:
        raise ValueError("from is the same Pokemon as via")
    return Rule(against, use, move, first, via, via_from, wait)


def resolve_rules(
    file: LineFile, names: Names, player: Sequence[PokemonSpec], opp: Sequence[PokemonSpec]
) -> list[Rule]:
    """The ``[[rule]]`` tables resolved against the team ``player`` and the opponent's
    party ``opp`` (module docstring), in file order. A team member may also be named by its
    ``[[team]]`` entry's ``pokemon`` string (slot ``i`` is entry ``i``)."""
    labels = [e.get("pokemon") if isinstance(e.get("pokemon"), str) else "" for e in file.team]
    rules: list[Rule] = []
    owner: dict[int, int] = {}
    for i, table in enumerate(file.rules, 1):
        where = f"rule {i} (against {table.get('against')})"
        try:
            _check_keys(where, table, RULE_KEYS)
            rule = _resolve_rule(table, names, player, opp, labels)
            if rule.against in owner:
                raise ValueError(
                    f"opposing slot {rule.against} already has rule {owner[rule.against]}"
                )
        except ValueError as e:
            raise _named(where, e) from None
        owner[rule.against] = i
        rules.append(rule)
    return rules


# --- Line <-> rules -----------------------------------------------------------------------

_FIELDS = ("use", "move", "first", "via", "via_from", "wait")


def rules_to_line(rules: Sequence[Rule]) -> Line:
    """The ``Line`` (numpy arrays) holding ``rules``; slots without a rule stay unset."""
    from nuzline.search.line import Line, empty_line

    arrays = {f: np.array(a) for f, a in zip(_FIELDS, empty_line(), strict=True)}
    slots = len(arrays["use"])
    for r in rules:
        if not 0 <= r.against < slots:
            raise ValueError(f"opposing slot {r.against} outside 0-{slots - 1}")
        if arrays["use"][r.against] >= 0:
            raise ValueError(f"two rules for opposing slot {r.against}")
        for f in _FIELDS:
            arrays[f][r.against] = int(getattr(r, f))
    return Line(**arrays)


def line_to_rules(line: Line) -> list[Rule]:
    """One ``Rule`` per opposing slot with a ``use``, in slot order."""
    a = {f: np.asarray(getattr(line, f)).tolist() for f in _FIELDS}
    return [
        Rule(slot, *(a[f][slot] for f in _FIELDS[:-1]), wait=bool(a["wait"][slot]))
        for slot in range(len(a["use"]))
        if a["use"][slot] >= 0
    ]


# --- writing ------------------------------------------------------------------------------


def _toml_string(text: str) -> str:
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"cannot write {value!r} as a line-file value")


def _table(header: str, items: Sequence[tuple[str, object]]) -> str:
    return "\n".join([f"[[{header}]]", *(f"{k} = {_toml_value(v)}" for k, v in items)])


def _label(party: Sequence[PokemonSpec], slot: int, names: Names) -> str:
    """A species name, or ``slot:N`` where the name would be ambiguous."""
    species = party[slot].species
    if sum(p.species == species for p in party) > 1:
        return f"slot:{slot}"
    return names.species_names[species]


def render_rules(
    rules: Sequence[Rule], names: Names, player: Sequence[PokemonSpec], opp: Sequence[PokemonSpec]
) -> list[str]:
    """Each rule as a TOML ``[[rule]]`` table (names, ``slot:N`` where a name is ambiguous);
    ``resolve_rules`` reads them back to the same rules."""
    out = []
    for r in rules:
        moves = player[r.use].moves
        items: list[tuple[str, object]] = [
            ("against", _label(opp, r.against, names)),
            ("use", _label(player, r.use, names)),
            ("move", names.move_names[moves[r.move]]),
        ]
        if r.first >= 0:
            items.append(("first", names.move_names[moves[r.first]]))
        if r.via >= 0:
            items.append(("via", _label(player, r.via, names)))
        if r.via_from >= 0:
            items.append(("from", _label(player, r.via_from, names)))
        if r.wait:
            items.append(("wait", True))
        out.append(_table("rule", items))
    return out


def write_line_file(path: Path, file: LineFile, rules_toml: Sequence[str]) -> None:
    """Write ``file``'s ``[[team]]`` tables, then the ``rules_toml`` tables, to ``path``.
    ``ValueError`` (nothing written) for a team key the reader would refuse."""
    for i, entry in enumerate(file.team, 1):
        _check_keys(f"team {i} ({entry.get('pokemon')})", entry, TEAM_KEYS)
    team = [_table("team", list(entry.items())) for entry in file.team]
    text = "\n\n".join([*team, *rules_toml])
    Path(path).write_text(text + "\n" if text else "", encoding="utf-8", newline="\n")
