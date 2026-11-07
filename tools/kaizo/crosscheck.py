"""Generic three-way cross-check: ROM vs PKCalc vs spreadsheet (plan Task 11, spec 6.5).

Policy, per row (``None`` = that source has no value and never counts as a disagreement):

- PKCalc ``None``: ``rom_vs_sheet`` when the sheet has a value that differs from the ROM,
  else ``agree``; the ROM value is used.
- ROM = PKCalc: ``agree``, or ``sheet_only`` when the sheet has a different value.
- ROM != PKCalc (ROM ``None`` with PKCalc set counts), not executed: ``unexecuted``, PKCalc used.
- ROM != PKCalc, executed: ``resolved`` through the matching entry of ``resolutions.toml``
  (``use`` = ``rom`` or ``pkcalc``), else ``unresolved`` (ROM used) and the build fails.

A resolution matches on table, field, key and the PKCalc value it was written against; one
that matches no executed disagreement is stale and fails the build too. A resolution against
the ROM (``use = "pkcalc"``) needs ``evidence`` starting ``trace:``.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tools.rom.inputs import REPO_ROOT

DEFAULT_RESOLUTIONS = REPO_ROOT / "data" / "kaizo" / "resolutions.toml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "kaizo" / "discrepancies.md"

KINDS = ("agree", "sheet_only", "rom_vs_sheet", "unexecuted", "resolved", "unresolved")
USES = ("rom", "pkcalc")
_RES_KEYS = ("table", "keys", "field", "pkcalc", "use", "reason", "evidence")


class CrossCheckError(ValueError):
    """A resolutions file is malformed, or the cross-check has unresolved or stale entries."""


@dataclass(frozen=True)
class Row:
    table: str
    key: str
    field: str
    rom: object
    pkcalc: object
    sheet: object
    executed: bool


@dataclass(frozen=True)
class Resolution:
    table: str
    keys: tuple[str, ...]
    field: str
    pkcalc: object
    use: str
    reason: str
    evidence: str


@dataclass(frozen=True)
class Outcome:
    row: Row
    kind: str
    used: object
    resolution: Resolution | None


@dataclass
class CrossCheck:
    outcomes: list[Outcome] = field(default_factory=list)
    unresolved: list[Row] = field(default_factory=list)
    stale: list[Resolution] = field(default_factory=list)

    def require_clean(self) -> None:
        lines = [
            f"unresolved: {r.table} {r.key!r} {r.field}: ROM {r.rom!r} vs PKCalc {r.pkcalc!r}"
            for r in self.unresolved
        ]
        lines += [
            f"stale resolution: {s.table} {', '.join(map(repr, s.keys))} {s.field} "
            f"(pkcalc {s.pkcalc!r}) matches no current ROM-vs-PKCalc conflict"
            for s in self.stale
        ]
        if lines:
            raise CrossCheckError("cross-check is not clean:\n" + "\n".join(lines))


# --- resolutions ----------------------------------------------------------------------


def _text(entry: str, name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CrossCheckError(f"{entry}: {name} must be a non-empty string")
    return value


def load_resolutions(path: Path = DEFAULT_RESOLUTIONS) -> list[Resolution]:
    """Read and validate ``resolutions.toml``; an absent file has no entries."""
    if not path.exists():
        return []
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise CrossCheckError(f"{path}: {e}") from e
    bad = sorted(k for k in raw if k != "resolution")
    if bad:
        raise CrossCheckError(f"{path.name}: unknown table {', '.join(map(repr, bad))}")
    out: list[Resolution] = []
    seen: set[tuple[str, str, str]] = set()
    for i, t in enumerate(raw.get("resolution", [])):
        entry = f"resolution[{i}]"
        unknown = sorted(k for k in t if k not in _RES_KEYS)
        if unknown:
            raise CrossCheckError(f"{entry}: unknown key {', '.join(map(repr, unknown))}")
        table = _text(entry, "table", t.get("table"))
        fld = _text(entry, "field", t.get("field"))
        keys = t.get("keys")
        if (
            not isinstance(keys, list)
            or not keys
            or not all(isinstance(k, str) and k for k in keys)
        ):
            raise CrossCheckError(f"{entry}: keys must be a non-empty list of strings")
        if "pkcalc" not in t:
            raise CrossCheckError(f"{entry}: pkcalc (the PKCalc value resolved) is required")
        use = t.get("use")
        if use not in USES:
            raise CrossCheckError(f"{entry}: use must be one of {USES}, not {use!r}")
        reason = _text(entry, "reason", t.get("reason"))
        evidence = _text(entry, "evidence", t.get("evidence"))
        if use == "pkcalc" and not evidence.startswith("trace:"):
            raise CrossCheckError(
                f"{entry}: use = 'pkcalc' goes against the ROM and needs evidence starting "
                f"'trace:', not {evidence!r}"
            )
        for k in keys:
            if (table, fld, k) in seen:
                raise CrossCheckError(f"{entry}: {table} {k!r} {fld} is resolved twice")
            seen.add((table, fld, k))
        out.append(Resolution(table, tuple(keys), fld, t["pkcalc"], use, reason, evidence))
    return out


# --- compare --------------------------------------------------------------------------


def compare(rows: Iterable[Row], resolutions: Sequence[Resolution]) -> CrossCheck:
    index: dict[tuple[str, str, str], Resolution] = {}
    for res in resolutions:
        for k in res.keys:
            index[(res.table, res.field, k)] = res
    used_res: set[int] = set()
    check = CrossCheck()
    for row in rows:
        resolution = None
        if row.pkcalc is None:
            differs = row.sheet is not None and row.sheet != row.rom
            kind, used = ("rom_vs_sheet" if differs else "agree"), row.rom
        elif row.rom == row.pkcalc:
            differs = row.sheet is not None and row.sheet != row.rom
            kind, used = ("sheet_only" if differs else "agree"), row.rom
        elif not row.executed:
            kind, used = "unexecuted", row.pkcalc
        else:
            cand = index.get((row.table, row.field, row.key))
            if cand is not None and cand.pkcalc == row.pkcalc:
                resolution = cand
                used_res.add(id(cand))
                kind = "resolved"
                used = row.pkcalc if cand.use == "pkcalc" else row.rom
            else:
                kind, used = "unresolved", row.rom
                check.unresolved.append(row)
        check.outcomes.append(Outcome(row, kind, used, resolution))
    check.stale = [r for r in resolutions if id(r) not in used_res]
    return check


# --- report ---------------------------------------------------------------------------


def _ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _cell(value: object) -> str:
    """Deterministic, ASCII-only, table-safe rendering of a row value."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_cell(v) for v in value) + "]"
    text = str(value).replace("\\", "\\\\").replace("|", "\\|")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return _ascii(text)


def _why(o: Outcome) -> str:
    if o.resolution is not None:
        return f"{o.resolution.reason} ({o.resolution.evidence})"
    return {
        "sheet_only": "sheet differs; ROM and PKCalc agree",
        "rom_vs_sheet": "PKCalc has no value; ROM differs from the sheet",
        "unexecuted": "field not executed; PKCalc value used",
        "unresolved": "needs an entry in resolutions.toml",
    }.get(o.kind, "")


def _sort_key(o: Outcome) -> tuple[str, str, str]:
    return (o.row.table, o.row.key, o.row.field)


def write_report(
    check: CrossCheck, sections: dict[str, list[str]], sources: dict[str, str], path: Path
) -> None:
    lines = [
        "# Kaizo discrepancies",
        "",
        "Generated by tools/crosscheck_kaizo.py; do not edit.",
        "",
    ]
    lines += ["## Sources", "", "| source | SHA-256 |", "| --- | --- |"]
    lines += [f"| {_cell(k)} | {_cell(sources[k])} |" for k in sorted(sources)]
    counts = Counter((o.row.table, o.kind) for o in check.outcomes)
    lines += ["", "## Counts", "", "| table | " + " | ".join(KINDS) + " |"]
    lines.append("| --- |" + " --- |" * len(KINDS))
    for table in sorted({t for t, _ in counts}):
        n = " | ".join(str(counts[table, k]) for k in KINDS)
        lines.append(f"| {_cell(table)} | {n} |")
    for kind in KINDS:
        if kind == "agree":
            continue
        group = sorted((o for o in check.outcomes if o.kind == kind), key=_sort_key)
        lines += ["", f"## {kind} ({len(group)})", ""]
        if not group:
            lines.append("None.")
            continue
        lines += [
            "| table | key | field | ROM | PKCalc | sheet | used | why |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for o in group:
            r = o.row
            cells = [r.table, r.key, r.field, r.rom, r.pkcalc, r.sheet, o.used, _why(o)]
            lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    lines += ["", f"## stale resolutions ({len(check.stale)})", ""]
    if check.stale:
        lines += ["| table | keys | field | pkcalc | use |", "| --- | --- | --- | --- | --- |"]
        for s in sorted(check.stale, key=lambda s: (s.table, s.field, s.keys)):
            cells = [s.table, ", ".join(s.keys), s.field, s.pkcalc, s.use]
            lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    else:
        lines.append("None.")
    for name, body in sections.items():
        lines += ["", f"## {_ascii(name)}", ""]
        lines += [_ascii(str(ln)) for ln in body]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii"))
