"""Resolve the Kaizo sources into the intermediates the runtime reads (plan K2b Task 15;
spec 6.5, 6.6; Rulings 2, 3, 7, 9, 12).

``resolve`` runs the three-way cross-check (Tasks 11-13) over ``<inter>/rom``,
``<inter>/pkcalc/pkcalc.json`` and ``<inter>/docs/docs.json`` with ``overrides.toml`` and
``resolutions.toml``, writes ``docs/kaizo/discrepancies.md`` either way, and stops on an
unresolved or stale row (``CrossCheckError``). Clean, it writes into ``inter``:

- the vanilla-schema JSON (``COPIED``) from the ROM, every value the cross-check's
  ``Outcome.used``. The ROM value is used everywhere except names (below) and rows with
  no ROM slot (a sheet-only id, an unmapped sheet header), which have nothing to write;
  any other non-ROM ``used`` raises (``ResolveError``), so nothing is silently dropped.
- names per Ruling 3: the display name is PKCalc's (joined by id; a slot PKCalc lacks
  keeps the ROM text). A slot is **renamed** when its display name differs from vanilla's
  beyond spelling (``spelling``: case, punctuation, the gender signs), and
  ``renamed.json`` maps the vanilla name (lower case) to the display name. An **alias**
  is a ROM text-bank name that differs from the display name (case-insensitively); the
  vanilla name of a renamed slot is never one, even when the ROM text still reads it
  (listed in the report), and an alias equal to another id's display name is an error.
- ``trainers.json`` in Task 14's schema (``trainer_record``): explicit nature, gender and
  ability slot from the patched routine's carry-over decode (``carry_over_sets``),
  friendship 255 (0 with Frustration, helper 0x02079570), ``pid_routine`` None, the
  PKCalc label, confirmed sight pairs (K2a Ruling 4), the ``format`` corrections of
  ``overrides.toml`` (``"unknown"``, or the user's answer with its ``opponent_2`` and
  ``partner``), the derived player starter, gauntlets, the split
  fields from the sheet headers (Ruling 9; None when no header maps to the trainer),
  ``area_condition`` with ``start_condition`` kept consistent, and ``sources`` per field.
- ``kaizo.json``: ``aliases`` and ``provenance`` (input hashes).

Usage: ``python -m tools.build_tables --game kaizo`` (resolve, then pack).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nuzline.data.games import GAMES, KAIZO, VANILLA
from nuzline.data.trainer import AreaCondition, default_moves
from tools.kaizo.crosscheck import (
    DEFAULT_REPORT,
    DEFAULT_RESOLUTIONS,
    CrossCheck,
    compare,
    load_resolutions,
    write_report,
)
from tools.kaizo.overrides import (
    DEFAULT_PATH as DEFAULT_OVERRIDES,
)
from tools.kaizo.overrides import (
    NOISY_MAP_JOIN,  # noqa: F401  (re-exported: callers and tests name it here)
    UNKNOWN_FORMAT,
    Mappings,
    Overrides,
    _row_claims,
    build_mappings,
    check_citations,
    check_gauntlet_members,
    check_rom_trick_room,
    check_trainer_ids,
    check_two_placement,
    load_overrides,
    two_placement_trick_room,
)
from tools.kaizo.rows_common import get
from tools.kaizo.rows_tables import load_rom, table_rows
from tools.kaizo.rows_trainers import (
    SIGHT_CITE,
    _joined,
    _partner_starters,
    _pk_starter,
    _sheet_by_id,
    _starter,
    area_section,
    carry_over_sets,
    mapping_section,
    pid_section,
    trainer_rows,
)
from tools.rom.inputs import KAIZO_PATCH_SHA256, REPO_ROOT, VANILLA_SHA1, RomInputs

ROM_DIFF = REPO_ROOT / "data" / "kaizo" / "rom_diff.toml"
UNKNOWN_START = "unknown"  # a start condition the evidence leaves to the user
START_CONDITIONS = REPO_ROOT / "data" / "kaizo" / "start_conditions.json"
BATTLE_FORMATS = REPO_ROOT / "data" / "kaizo" / "battle_formats.json"

# The vanilla-schema files the runtime and build_tables read, copied from <inter>/rom.
COPIED = (
    "species",
    "forms",
    "form_rows",
    "moves",
    "move_lists",
    "items",
    "abilities",
    "types",
    "natures",
    "exp_tables",
    "charmap",
    "learnsets",
    "tm_learnsets",
    "evolutions",
    "egg_moves",
    "evolution_methods",
    "tutor_moves",
    "tutor_learnsets",
)
# Ruling 3's kinds: (renamed.json / aliases key, intermediate file, PKCalc table).
KINDS = (
    ("species", "species", "species"),
    ("move", "moves", "moves"),
    ("item", "items", "items"),
    ("ability", "abilities", "abilities"),
)
FRUSTRATION = 218
FRIENDSHIP_MAX, FRIENDSHIP_FRUSTRATION = 255, 0  # patched builder's helper at 0x02079570
PATCHED_SETS = "rom: patched TrainerData_BuildParty carry-over decode (docs/kaizo/rom_diff.md)"
FRIENDSHIP_SOURCE = "rom: patched friendship helper 0x02079570 (docs/kaizo/rom_diff.md)"


class ResolveError(ValueError):
    """The resolved data cannot be written exactly (an unmodelled non-ROM value, a name
    collision, an inconsistent trainer field)."""


# --- inputs -----------------------------------------------------------------------------


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """SHA-256 of a committed text file with CRLF folded to LF, so the hash does not depend
    on a checkout's line endings (``.gitattributes`` also pins them to LF)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Sources:
    """The cross-check's inputs, read from one intermediate directory."""

    rom: dict[str, Any]  # rows_tables.load_rom bundle
    trainers: list[dict]
    field_effects: list[dict]
    trainer_maps: dict[int, list[int]]
    pkcalc: dict
    docs: dict


def load_sources(inter: Path) -> Sources:
    inter = Path(inter)
    rom_dir = inter / "rom"
    paths = [
        rom_dir / "trainers.json",
        rom_dir / "rom_field_effects.json",
        rom_dir / "rom_trainer_maps.json",
        inter / "pkcalc" / "pkcalc.json",
        inter / "docs" / "docs.json",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing Kaizo intermediates (run tools.extract_rom --game kaizo, "
            "tools.extract_pkcalc, tools.extract_kaizo_docs): " + ", ".join(missing)
        )
    trainers, effects, where, pkcalc, docs = (_read(p) for p in paths)
    return Sources(
        rom=load_rom(rom_dir),
        trainers=trainers,
        field_effects=effects,
        trainer_maps={int(k): list(v) for k, v in where.items()},
        pkcalc=pkcalc,
        docs=docs,
    )


def unplaced_ids(trainers: list[dict], trainer_maps: Mapping[int, list[int]]) -> list[int]:
    """Trainers with a party that the trainer -> map join places on no map header (none, or
    an empty list: a rematch of an unplaced trainer). Their start condition is unknown,
    never the extractor's default "none" (final review M4; Kaizo only)."""
    return [t["id"] for t in trainers if t["party"] and not trainer_maps.get(t["id"])]


def user_format(fields: Mapping[str, object], t: Mapping[str, Any]) -> dict:
    """``format``, ``opponent_2`` and ``partner`` from the user's ``[fields.<id>] format``
    answer (``load_overrides`` takes a concrete format from a ``"user, <date>"`` source
    only). Only a multi has a partner: ``partner`` when the answer names one, else the ROM
    trainer's (``None``: present, not evaluated)."""
    fmt = fields["format"]
    if fmt != "multi":
        partner: list | None = []
    elif "partner" in fields:
        partner = [{"trainer": fields["partner"], "starter": None, "player_gender": None}]
    else:
        partner = t.get("partner")
    return {"format": fmt, "opponent_2": fields.get("opponent_2"), "partner": partner}


def with_overrides(
    trainers: list[dict], overrides: Overrides, trainer_maps: Mapping[int, list[int]] | None = None
) -> list[dict]:
    """The ROM trainers with the format and refusing overrides applied, so the cross-check
    lists the refused ones with the unknowns instead of comparing them, and compares the
    user's format answers against PKCalc and the sheet: ``[fields.<id>] format =
    "unknown"`` (``opponent_2`` None: the structures disagree on it) or the user's format
    (``user_format``), ``[area_exception]`` (start condition ``"unknown"``) and, given
    ``trainer_maps``, each ``unplaced_ids`` trainer (start condition ``"unknown"``) unless an
    ``[area.<id>]`` weather settles it."""
    no_map = set(unplaced_ids(trainers, trainer_maps)) if trainer_maps is not None else set()
    out = []
    for t in trainers:
        tid = t["id"]
        fields = overrides.fields.get(tid, {})
        fmt = fields.get("format")
        if fmt == UNKNOWN_FORMAT:
            t = dict(t, format=fmt, opponent_2=None)
        elif fmt is not None:
            t = dict(t, **user_format(fields, t))
        settled = getattr(overrides.area.get(tid), "weather", None) is not None
        if tid in overrides.area_exceptions or (tid in no_map and not settled):
            t = dict(t, start_condition=UNKNOWN_START)
        out.append(t)
    return out


# --- cross-check ------------------------------------------------------------------------


def cross_check(
    src: Sources, overrides: Overrides, resolutions_path: Path
) -> tuple[CrossCheck, Mappings, list[dict], dict[str, list[str]]]:
    """Rows, outcomes and report sections. Raises ``CrossCheckError`` on a malformed
    resolutions file and ``ResolveError`` when the overrides fail their own checks."""
    problems = (
        check_citations(overrides, src.docs)
        + check_rom_trick_room(overrides, src.field_effects)
        + check_trainer_ids(overrides, {t["id"] for t in src.trainers})
    )
    problems += check_two_placement(overrides, src.field_effects, src.trainer_maps)
    trainers = with_overrides(src.trainers, overrides, src.trainer_maps)
    maps = build_mappings(
        src.pkcalc.get("trainers", {}),
        src.docs,
        trainers,
        src.rom["species"],
        src.field_effects,
        overrides.trainer_ids,
    )
    problems += check_gauntlet_members(overrides, src.docs, maps.sheet)
    if problems:
        raise ResolveError("overrides.toml fails its checks:\n" + "\n".join(problems))
    rows = table_rows(src.rom, src.pkcalc, src.docs) + trainer_rows(
        trainers, maps, src.pkcalc, src.docs, src.rom, overrides, src.trainer_maps
    )
    check = compare(rows, load_resolutions(resolutions_path))
    sections = {
        "trainer mappings": mapping_section(maps),
        "PID cross-check (Ruling 19; recorded, not asserted)": pid_section(
            trainers, maps, src.pkcalc, src.rom
        ),
        "battle setup": area_section(
            trainers, maps, src.pkcalc, src.docs, overrides, src.trainer_maps
        ),
    }
    return check, maps, trainers, sections


def check_used(check: CrossCheck) -> list[str]:
    """Rows whose used value is not the ROM's and that ``resolve`` writes nowhere: names
    (Ruling 3 writes the display name), rows with no ROM slot, the ``mapped`` rows.
    Raises ``ResolveError`` for any other non-ROM used value."""
    skipped, bad = [], []
    for o in check.outcomes:
        r = o.row
        if o.used == r.rom and type(o.used) is type(r.rom):
            continue
        if r.field in ("name", "mapped") or (o.kind == "sheet_only" and r.rom is None):
            skipped.append(f"{r.table} {r.key} {r.field}: {o.kind}, used {o.used!r}")
            continue
        bad.append(f"{r.table} {r.key!r} {r.field}: {o.kind} uses {o.used!r}, ROM {r.rom!r}")
    if bad:
        raise ResolveError(
            "resolve writes only ROM values for these fields; a non-ROM value needs a "
            "writer in tools/kaizo/resolve.py:\n" + "\n".join(bad)
        )
    return skipped


# --- names (Ruling 3) -------------------------------------------------------------------

_SIGNS = {"♀": "f", "♂": "m"}


def spelling(name: str) -> str:
    """A name with case, punctuation and the gender signs folded: two names with the same
    spelling are one name written two ways ("Nidoran-F" / "NIDORAN♀", "King's Rock" /
    "King’s Rock")."""
    text = "".join(_SIGNS.get(c, c) for c in name.casefold())
    return re.sub(r"[^0-9a-z]", "", text)


def _pk_names(pkcalc: dict, table: str) -> dict[int, str]:
    if table == "species":
        return {
            int(v["num"]): v["name"] for v in pkcalc.get("species", {}).values() if v["form"] == 0
        }
    return {int(k): v["name"] for k, v in pkcalc.get(table, {}).items()}


@dataclass(frozen=True)
class NameSet:
    display: dict[str, list[str]]  # kind -> display name per id
    renamed: dict[str, dict[str, str]]  # kind -> {vanilla lower: display}
    aliases: dict[str, dict[str, int]]  # kind -> {alias: id}
    notes: list[str]  # report lines


def names(
    rom: Mapping[str, list[dict]], vanilla: Mapping[str, list[dict]], pkcalc: dict
) -> NameSet:
    display: dict[str, list[str]] = {}
    renamed: dict[str, dict[str, str]] = {}
    aliases: dict[str, dict[str, int]] = {}
    notes: list[str] = []
    for kind, table, pk_table in KINDS:
        rom_rows, van_rows = rom[table], vanilla[table]
        if len(rom_rows) != len(van_rows):
            raise ResolveError(f"{table}: {len(rom_rows)} ROM rows, {len(van_rows)} vanilla rows")
        pk = _pk_names(pkcalc, pk_table)
        shown = [pk.get(i, r["name"]) for i, r in enumerate(rom_rows)]
        index: dict[str, int] = {}
        for i, n in enumerate(shown):
            if index.setdefault(n.casefold(), i) != i:
                raise ResolveError(
                    f"{kind} display name {n!r} is ids {index[n.casefold()]} and {i}"
                )
        ren: dict[str, str] = {}
        for v, n in zip(van_rows, shown, strict=True):
            if spelling(v["name"]) != spelling(n) and ren.setdefault(v["name"].lower(), n) != n:
                raise ResolveError(f"{kind}: vanilla name {v['name']!r} renamed twice")
        al: dict[str, int] = {}
        for i, (r, n) in enumerate(zip(rom_rows, shown, strict=True)):
            text = r["name"]
            if text.casefold() == n.casefold():
                continue
            if text.lower() in ren:
                notes.append(
                    f"{kind} {i} {n!r}: the ROM text reads {text!r}, the vanilla name of a "
                    "renamed slot; not an alias"
                )
                continue
            other = index.get(text.casefold())
            if other is not None:
                raise ResolveError(
                    f"{kind} alias {text!r} (id {i}) is the display name of id {other}"
                )
            if al.setdefault(text, i) != i:
                raise ResolveError(f"{kind} alias {text!r} is ids {al[text]} and {i}")
        display[kind], renamed[kind], aliases[kind] = shown, ren, al
        notes.append(f"{kind}: {len(ren)} renamed, {len(al)} aliases")
        notes += [f"{kind} renamed: {old!r} -> {new!r}" for old, new in sorted(ren.items())]
        notes += [f"{kind} alias: {a!r} -> {i}" for a, i in sorted(al.items(), key=lambda x: x[1])]
    return NameSet(display, renamed, aliases, notes)


# --- trainers ---------------------------------------------------------------------------


# Ruling 22: a bare sheet weather tag such as "(SUN)" (extract_kaizo_docs marks it
# ``assumed_permanent``) is read as permanent: user-confirmed 2026-11-05. Flip this one
# constant (and rebuild) if play shows such weather ending.
BARE_WEATHER_PERMANENT = True


def bare_tags(rows: list[tuple[dict, dict]]) -> frozenset[str]:
    """The conditions a trainer's sheet section or header tags only barely (``weather``
    or ``trick_room`` with ``assumed_permanent``), e.g. ``{"sun"}`` for ``(SUN)``."""
    out = set()
    for sec, h in rows:
        for tags in (sec.get("tags", {}), h.get("tags", {})):
            if tags.get("assumed_permanent"):
                if tags.get("weather"):
                    out.add(tags["weather"])
                if tags.get("trick_room"):
                    out.add("trick_room")
    return frozenset(out)


def _area(start_condition: str, area: Any, bare: frozenset[str] = frozenset()) -> tuple[str, dict]:
    """(start_condition, area_condition) from the ROM start condition and an overrides
    ``[area.<id>]`` entry (weather replaces the start condition; Trick Room, Gravity add).
    ``permanent``: map weather is permanent in battle (``FIELD_CONDITION_*_PERM``, Ruling 8),
    so it is True, except where the trainer's sheet tags its weather or Trick Room only
    barely (``bare``, ``bare_tags``): there it is ``BARE_WEATHER_PERMANENT`` (Ruling 22)."""
    weather = getattr(area, "weather", None)
    start = start_condition if weather is None else weather
    cond = AreaCondition.from_start_condition(start)
    out = {
        "weather": cond.weather,
        "fog": cond.fog,
        "trick_room": bool(getattr(area, "trick_room", False)),
        "gravity": bool(getattr(area, "gravity", False)),
        "permanent": True,
    }
    barely = (out["weather"] in bare) or (out["trick_room"] and "trick_room" in bare)
    if barely:
        out["permanent"] = BARE_WEATHER_PERMANENT
    # Round trip: the start condition coverage reads (start_weather) is rebuilt from the
    # area condition's weather and fog.
    back = "fog" if out["fog"] else (out["weather"] or "none")
    if (out["fog"] and out["weather"] is not None) or back != start:
        raise ResolveError(f"start condition {start!r} does not round-trip through {out}")
    return start, out


def _split_rows(docs: dict) -> dict[str, tuple[int, str, dict, dict]]:
    """cite -> (split index, sheet name, section, header), in workbook order."""
    out = {}
    for i, split in enumerate(docs["splits"].values()):
        for sec in split["sections"]:
            for h in sec["trainers"]:
                out[h["cite"]] = (i, split["sheet"], sec, h)
    return out


def _line(cite: str) -> int:
    return int(cite.rsplit(":", 1)[1])


@dataclass(frozen=True)
class TrainerContext:
    labels: dict[int, tuple[str, str]]  # id -> (PKCalc label, mapping how)
    headers: dict[int, list[str]]  # id -> sheet cites, workbook order
    sight: dict[int, tuple[int, int]]  # id -> (partner id, shared map header)
    gauntlets: dict[int, tuple[str, int, dict, str]]  # id -> (name, position, rules, source)


def trainer_context(check: CrossCheck, maps: Mappings, overrides: Overrides,
                    trainer_maps: dict[int, list[int]], docs: dict,
                    formats: Mapping[int, str] | None = None) -> TrainerContext:  # fmt: skip
    """The per-trainer facts ``trainer_record`` joins. A confirmed sight pair must be
    confirmed from both sides: a pair claimed by one trainer only would make it a
    two-trainer double while its partner stays a runnable single, so it raises, unless the
    partner's format (``formats``, the trainers after ``with_overrides``) is ``"unknown"``
    and so refused anyway."""
    labels = {m.id: (m.label, m.how) for m in maps.pkcalc if m.id is not None}
    order = {c: n for n, c in enumerate(_split_rows(docs))}
    headers: dict[int, list[str]] = defaultdict(list)
    for m in sorted((m for m in maps.sheet if m.id is not None), key=lambda m: order[m.label]):
        headers[m.id].append(m.label)
    sight: dict[int, tuple[int, int]] = {}
    for o in check.outcomes:
        if o.row.field == "sight_pair" and isinstance(o.used, int):
            tid = int(o.row.key.split(" ", 1)[0])
            shared = set(trainer_maps.get(tid, ())) & set(trainer_maps.get(o.used, ()))
            if not shared:
                raise ResolveError(f"sight pair {tid}-{o.used}: no shared ROM map header")
            sight[tid] = (o.used, min(shared))
    for tid, (pair, _) in sight.items():
        if pair in sight and sight[pair][0] != tid:
            raise ResolveError(f"sight pairs disagree: {tid} -> {pair} -> {sight[pair][0]}")
        if pair not in sight and (formats or {}).get(pair) != UNKNOWN_FORMAT:
            raise ResolveError(
                f"sight pair {tid} -> {pair} is confirmed on {tid}'s side only: {pair} would "
                "stay a runnable single (needs a trainer_ids or fields correction)"
            )
    gauntlets: dict[int, tuple[str, int, dict, str]] = {}
    for g in overrides.gauntlets:
        for pos, tid in enumerate(g.trainers, 1):
            if tid in gauntlets:
                raise ResolveError(
                    f"trainer {tid} is in gauntlets {gauntlets[tid][0]!r} and {g.name!r}"
                )
            gauntlets[tid] = (g.name, pos, dict(g.rules), g.source)
    return TrainerContext(labels, dict(headers), sight, gauntlets)


def trainer_record(
    t: dict,
    ctx: TrainerContext,
    overrides: Overrides,
    src: Sources,
    sheet: dict[int, list[tuple[dict, dict]]],
    partner_starts: dict[int, set[int]],
    split_rows: dict[str, tuple[int, str, dict, dict]],
) -> dict:
    """One ``trainers.json`` entry in Task 14's schema, with ``sources`` per field."""
    tid = t["id"]
    fields = overrides.fields.get(tid, {})
    fsrc = f"overrides fields.{tid} ({overrides.sources.get(f'fields.{tid}', '?')})"
    sources: dict[str, str] = {"party": "rom: trainer party (trainers.json)"}
    out = {k: v for k, v in t.items() if k != "party"}

    party = []
    learnsets = src.rom["learnsets"]
    for mon, (nature, gender, slot) in zip(t["party"], carry_over_sets(t, src.rom), strict=True):
        moves = mon["moves"]
        if moves is None:
            moves = default_moves(get(learnsets, mon["species"]) or [], mon["level"])
        friendship = FRIENDSHIP_FRUSTRATION if FRUSTRATION in moves else FRIENDSHIP_MAX
        party.append(
            dict(mon, nature=nature, gender=gender, ability_slot=slot, friendship=friendship)
        )
    out["party"] = party
    sources.update(nature=PATCHED_SETS, gender=PATCHED_SETS, ability_slot=PATCHED_SETS)
    sources["friendship"] = FRIENDSHIP_SOURCE
    out["pid_routine"] = None
    sources["pid_routine"] = "rom: patched routine not transcribed (Ruling 7)"

    label = ctx.labels.get(tid)
    out["label"] = None if label is None else label[0]
    if label is not None:
        sources["label"] = f"pkcalc: trainer label (mapping how={label[1]})"

    fmt_source = "rom: trainers.json (K2a battle_structures over the patched field scripts)"
    if fields.get("format") is not None:
        out["format"] = fields["format"]
        if fields["format"] != UNKNOWN_FORMAT:  # the user's answer (overrides FORMATS)
            out.update(user_format(fields, t))
            sources["partner"] = fsrc
        fmt_source = fsrc
    elif tid in ctx.sight and t["format"] != UNKNOWN_FORMAT:
        pair, header = ctx.sight[tid]
        if t["format"] == "single":
            out["format"] = "double_two_trainer"
        elif not (t["format"] == "multi" and t["opponent_2"] is None):
            raise ResolveError(f"trainer {tid}: sight pair {pair} beside ROM format {t['format']}")
        out["opponent_2"] = pair  # a walking-partner multi keeps multi (K2a Ruling 3)
        fmt_source = SIGHT_CITE.format(header=header)
    sources["format"] = sources["opponent_2"] = fmt_source

    pk_start = _pk_starter(label[0]) if label is not None else None
    sheet_start = _joined(h["tags"].get("player_starter") for _, h in sheet.get(tid, []))
    start, how = _starter(tid, overrides, partner_starts, sheet_start, pk_start)
    out["player_starter"] = start
    if start is not None:
        sources["player_starter"] = how

    g = ctx.gauntlets.get(tid)
    out["gauntlet"] = None if g is None else [g[0], g[1]]
    out["gauntlet_rules"] = {} if g is None else g[2]
    if g is not None:
        sources["gauntlet"] = sources["gauntlet_rules"] = g[3]

    cites = ctx.headers.get(tid, [])
    if cites:
        idx, sheet_name, _, _ = split_rows[cites[0]]
        out["split"] = sheet_name
        out["split_order"] = [idx, _line(cites[0])]
        cite = ", ".join(cites)
        for k in ("split", "split_order"):
            sources[k] = f"sheet {cite}"
        for k, values in _header_fields(cites, split_rows).items():
            # Several headers that disagree settle nothing (Task 13 M1 ruling): unknown.
            out[k] = values[0][1] if len({v for _, v in values}) == 1 else None
            sources[k] = f"sheet {cite}"
            if out[k] is None and len(values) > 1:
                said = ", ".join(f"{c} {_cell_text(v)}" for c, v in values)
                sources[k] += f" (headers disagree: {said}; unknown)"
    else:
        out.update(split=None, split_order=None, optional=None, left_side=None)
    for k in ("split", "optional", "left_side"):
        if k in fields:
            out[k] = fields[k]
            sources[k] = fsrc
            if k == "split":
                # The sheet position belongs to the old split: none is known for the new one.
                out["split_order"] = None
                sources["split_order"] = fsrc + "; no sheet position for the corrected split"

    start, area = _area(
        t["start_condition"], overrides.area.get(tid), bare_tags(sheet.get(tid, []))
    )
    out["start_condition"], out["area_condition"] = start, area
    where = ", ".join(map(str, sorted(src.trainer_maps.get(tid, [])))) or "none"
    sources["start_condition"] = sources["area_condition"] = (
        f"rom: map header weather through the trainer -> map join (headers {where}; "
        "data/kaizo/start_conditions.json)"
        if where != "none" or not t["party"]
        else "rom: no map placement found (trainer -> map join): unknown (user call)"
    )
    a = overrides.area.get(tid)
    if a is not None:
        sources["area_condition"] += f"; overrides area.{tid} ({a.source})"
        if a.weather is not None:  # the override replaces the ROM start condition
            sources["start_condition"] += f"; overrides area.{tid} weather ({a.source})"
    if tid in overrides.area_exceptions:
        e = overrides.area_exceptions[tid]
        sources["area_condition"] += f"; area_exception ({e.reason}; {e.source})"
    out["sources"] = sources
    return out


def _header_fields(
    cites: list[str], split_rows: dict[str, tuple[int, str, dict, dict]]
) -> dict[str, list[tuple[str, object]]]:
    """``optional`` and ``left_side`` as each of a trainer's sheet headers says them:
    field -> [(cite, value)] in workbook order."""
    out: dict[str, list[tuple[str, object]]] = {"optional": [], "left_side": []}
    for c in cites:
        _, _, sec, h = split_rows[c]
        out["optional"].append((c, bool(h["tags"].get("optional") or sec["tags"].get("optional"))))
        out["left_side"].append((c, h.get("left_side") if h.get("ai_known") else None))
    return out


def _cell_text(v: object) -> str:
    return "-" if v is None else str(v).lower()


def resolve_trainers(
    src: Sources, trainers: list[dict], check: CrossCheck, maps: Mappings, overrides: Overrides
) -> tuple[list[dict], list[str]]:
    formats = {t["id"]: t["format"] for t in trainers}
    no_map = set(unplaced_ids(src.trainers, src.trainer_maps))
    ctx = trainer_context(check, maps, overrides, src.trainer_maps, src.docs, formats)
    sheet = _sheet_by_id(src.docs, maps)
    partner_starts = _partner_starters(trainers)
    split_rows = _split_rows(src.docs)
    out = [
        trainer_record(t, ctx, overrides, src, sheet, partner_starts, split_rows) for t in trainers
    ]
    sight = [t for t in out if t["sources"]["format"].startswith("rom: map header")]
    n_double = sum(1 for t in sight if t["format"] == "double_two_trainer")
    lines = [
        (
            f"sight pairs applied (K2a Ruling 4): {len(sight)}; double_two_trainer {n_double}, "
            f"multi beside a walking partner (opponent_2 set) {len(sight) - n_double}"
        ),
        "format unknown (user call): "
        + ", ".join(f"{t['id']} {t['name']}" for t in out if t["format"] == UNKNOWN_FORMAT),
        "start condition unknown (user call): "
        + ", ".join(
            f"{t['id']} {t['name']}"
            for t in out
            if t["start_condition"] == "unknown" and t["id"] not in no_map
        ),
        f"start condition unknown, no map placement found (one user call): {len(no_map)}",
    ]
    lines += [
        f"sheet headers disagree on {k}: {t['id']} {t['name']}: {t['sources'][k]}"
        for t in out
        for k in ("optional", "left_side")
        if "headers disagree" in t["sources"].get(k, "")
    ]
    unplaced = [
        f"{t['id']} {t['label']}" for t in out if t["label"] is not None and t["split"] is None
    ]
    lines.append(
        f"PKCalc trainers with no mapped sheet header (split, optional, left_side unknown): "
        f"{len(unplaced)}" + (": " + "; ".join(unplaced) if unplaced else "")
    )
    return out, lines


# --- user calls -------------------------------------------------------------------------


def user_calls(
    trainers: list[dict],
    check: CrossCheck,
    overrides: Overrides,
    src: Sources,
    unclassified: int,
    maps: Mappings | None = None,
) -> list[str]:
    """Every item only the user can settle, one line each with its evidence. An area
    weather the user's ``[area.<id>]`` answer sets is settled; Gravity is listed only when
    its entry cites PKCalc and no sheet row mapped to the trainer (``maps``) claims it."""
    by_id = {t["id"]: t for t in trainers}
    sheet = _sheet_by_id(src.docs, maps) if maps is not None else {}
    two = two_placement_trick_room(src.field_effects, src.trainer_maps)
    no_map = [
        tid for tid in unplaced_ids(src.trainers, src.trainer_maps)
        if by_id[tid]["start_condition"] == UNKNOWN_START
    ]  # fmt: skip
    lines = []

    def name(tid: int) -> str:
        return f"{tid} {by_id[tid]['name']}"

    for t in trainers:
        if t["start_condition"] == UNKNOWN_START and t["id"] not in two and t["id"] not in no_map:
            lines.append(
                f"- start condition unknown: {name(t['id'])} (data/kaizo/start_conditions.json: "
                "its maps start different conditions)"
            )
    if no_map:
        lines.append(
            f"- no map placement found: {len(no_map)} trainers with a party that the trainer "
            "-> map join places on no map header (rom_trainer_maps.json), so their start "
            "condition is refused as unknown, never read as no weather; settle one with an "
            "[area.<id>] weather entry: " + ", ".join(map(str, no_map))
        )
    for tid, (tr, other) in sorted(two.items()):
        exc = overrides.area_exceptions.get(tid)
        if exc is None:
            continue  # settled by the user's answer (check_two_placement)
        lines.append(
            f"- Trick Room on one placement only: {name(tid)}: Trick Room headers {tr}, other "
            f"headers {other}; start condition refused ({exc.source})"
        )
    for t in trainers:
        if t["format"] == UNKNOWN_FORMAT:
            tid = t["id"]
            if "format" in overrides.fields.get(tid, {}):
                why = overrides.sources.get(f"fields.{tid}", "?")
            else:
                why = "data/kaizo/battle_formats.json (script claims conflict)"
            lines.append(f"- format unknown: {name(tid)} ({why})")
    for o in check.outcomes:
        r = o.row
        head = str(r.key).split(" ", 1)[0]
        a = overrides.area.get(int(head)) if head.isdigit() else None
        if (
            o.kind == "resolved"
            and r.field == "area_weather"
            and r.sheet is not None
            and r.sheet == r.pkcalc
            and not (a is not None and a.user and a.weather is not None)
        ):
            lines.append(
                f"- area weather {r.key}: ROM {r.rom} against PKCalc and the sheet {r.pkcalc}; "
                f"ROM used ({o.resolution.evidence})"
            )
    for tid, a in sorted(overrides.area.items()):
        on_sheet = any("gravity" in _row_claims(sec, h) for sec, h in sheet.get(tid, []))
        if a.gravity and a.source.startswith("pkcalc:") and not on_sheet:
            lines.append(
                f"- Gravity: {name(tid)} rests on PKCalc only ({a.source}); no ROM field effect "
                f"on its headers {sorted(src.trainer_maps.get(tid, []))}"
            )
    lines.append(
        f"- {unclassified} ROM changes unclassified in data/kaizo/rom_diff.toml: every Kaizo "
        "fight is refused until they are classified (Ruling 13)"
    )
    return lines


# --- resolve ----------------------------------------------------------------------------


def _vanilla_tables(vanilla: Path) -> dict[str, list[dict]]:
    tables = {}
    for _, table, _ in KINDS:
        path = Path(vanilla) / f"{table}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no {path}: the vanilla names come from it (python -m tools.extract_decomp)"
            )
        tables[table] = _read(path)
    return tables


def _report_sources(src: Sources, overrides_path: Path, resolutions_path: Path) -> dict:
    out = {
        "PKCalc zip (pkcalc.json sha256)": src.pkcalc.get("sha256", "-"),
        "Kaizo docs workbook (docs.json sha256)": src.docs.get("sha256", "-"),
    }
    for name, path in (
        ("data/kaizo/overrides.toml", overrides_path),
        ("data/kaizo/resolutions.toml", resolutions_path),
    ):
        out[name] = _sha256(path) if Path(path).exists() else "absent"
    return out


def check_extract_inputs(inter: Path, overrides: Overrides) -> None:
    """Raise unless ``<inter>/rom/rom_sources.json`` records the committed inputs that only
    ``extract_rom`` reads (``start_conditions.json``, ``battle_formats.json``, the
    ``[move_lists]`` of ``overrides``) with the hashes they have now: an edit to one of them
    after the extraction leaves the ROM intermediates stale, and a rebuild alone would keep
    the old values."""
    from tools.extract_rom import INPUTS_KEY, kaizo_input_hashes

    path = Path(inter) / "rom" / "rom_sources.json"
    recorded = _read(path).get(INPUTS_KEY) if path.exists() else None
    current = kaizo_input_hashes(overrides.move_lists)
    if recorded is None:
        stale = ["no input hashes recorded"]
    else:
        stale = [k for k in sorted(current) if recorded.get(k) != current[k]]
    if stale:
        raise ResolveError(
            f"{path}: the ROM intermediates are stale ({', '.join(stale)}); re-run "
            "python -m tools.extract_rom --game kaizo"
        )


def check_input_hashes(inputs: RomInputs, src: Sources) -> list[str]:
    """Re-hash the configured PKCalc zip and docs workbook against the SHA-256 their
    extracted JSON records; a mismatch raises, an input not configured or absent is a
    report line (its JSON is used as extracted). The committed report names each input by
    its file name only, never its local path."""
    lines = []
    for what, configured, recorded in (
        ("PKCalc zip", inputs.pkcalc_zip, src.pkcalc.get("sha256")),
        ("Kaizo docs workbook", inputs.kaizo_docs, src.docs.get("sha256")),
    ):
        if configured is None or not configured.exists():
            shown = "not configured" if configured is None else configured.name
            lines.append(
                f"{what}: not configured or absent ({shown}); re-hash skipped, the "
                f"extracted JSON's sha256 {recorded} is used as recorded"
            )
        elif _file_sha256(configured) != recorded:
            raise ResolveError(
                f"{what} {configured} does not match the extracted intermediates "
                f"(sha256 {recorded}); re-run its extractor"
            )
        else:
            lines.append(f"{what}: {configured.name} re-hashed, matches {recorded}")
    return lines


def provenance(
    inputs: RomInputs, src: Sources, overrides_path: Path, resolutions_path: Path
) -> dict[str, str]:
    """Input hashes for the manifest (the patched ROM is built and checked by K1's
    ``kaizo_rom``, which verifies the vanilla SHA-1 and the patch SHA-256 first). The
    committed text files hash LF-normalised (``_sha256``)."""
    from tools.rom.patch import kaizo_rom

    _, patched_sha1 = kaizo_rom(inputs)
    return {
        "vanilla_rom_sha1": VANILLA_SHA1,
        "patch_sha256": KAIZO_PATCH_SHA256,
        "patched_rom_sha1": patched_sha1,
        "pkcalc_sha256": src.pkcalc["sha256"],
        "kaizo_docs_sha256": src.docs["sha256"],
        "overrides_sha256": _sha256(overrides_path),
        "resolutions_sha256": _sha256(resolutions_path),
        "rom_diff_sha256": _sha256(ROM_DIFF),
        "start_conditions_sha256": _sha256(START_CONDITIONS),
        "battle_formats_sha256": _sha256(BATTLE_FORMATS),
    }


def resolve(
    inputs: RomInputs,
    inter: Path = GAMES[KAIZO].intermediate_dir,
    report: Path = DEFAULT_REPORT,
    *,
    resolutions: Path = DEFAULT_RESOLUTIONS,
    overrides: Path = DEFAULT_OVERRIDES,
    vanilla: Path = GAMES[VANILLA].intermediate_dir,
) -> Path:
    """Cross-check, write ``report`` (always), stop on unresolved or stale rows, then write
    the resolved intermediates into ``inter``; returns ``inter``."""
    inter = Path(inter)
    src = load_sources(inter)
    ov = load_overrides(overrides)
    check_extract_inputs(inter, ov)
    check, maps, trainers, sections = cross_check(src, ov, resolutions)
    sources = _report_sources(src, overrides, resolutions)
    if not (check.unresolved or check.stale):
        try:
            skipped = check_used(check)
            name_set = names(src.rom, _vanilla_tables(vanilla), src.pkcalc)
            resolved_trainers, trainer_lines = resolve_trainers(src, trainers, check, maps, ov)
            hash_lines = check_input_hashes(inputs, src)
            from tools.build_tables import _kaizo_rom_changes

            _, unclassified = _kaizo_rom_changes(inter)
            calls = user_calls(resolved_trainers, check, ov, src, unclassified, maps)
        except Exception:
            write_report(check, sections, sources, report)
            raise
        sections = {"User calls": calls, **sections}
        sections["input re-hash"] = hash_lines
        sections["names (Ruling 3)"] = name_set.notes
        sections["resolved trainers"] = trainer_lines
        sections["rows with nothing to write (names, no ROM slot, mapped)"] = skipped
    write_report(check, sections, sources, report)
    check.require_clean()

    prov = provenance(inputs, src, overrides, resolutions)
    rom_dir = inter / "rom"
    for name in COPIED:
        data = _read(rom_dir / f"{name}.json")
        kind = next((k for k, table, _ in KINDS if table == name), None)
        if kind is not None:
            data = [dict(row, name=n) for row, n in zip(data, name_set.display[kind], strict=True)]
        _write(inter / f"{name}.json", data)
    _write(inter / "trainers.json", resolved_trainers)
    _write(inter / "renamed.json", name_set.renamed)
    _write(inter / "kaizo.json", {"aliases": name_set.aliases, "provenance": prov})
    return inter
