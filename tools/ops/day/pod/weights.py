"""Expected peak (GB) per unit for runner.sh's memory gate. Imported by setup.sh from bin/.

weights.txt lines: "<entry> <GB> <cite...>"; "#" starts a comment. An entry matches a test id
equal to it or starting with it plus "::", "[" or "/". An id's weight is the GB of its longest
matching entry, else DEFAULT_GB; a unit's weight is the largest weight of its ids. A list
line's weight (its "<GB>G" field) replaces the table for every unit cut from that line.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_GB = 12  # docs/process.md section 5, Chunking: a chunk grows to 8-12 GB


def load(path: Path) -> list[tuple[str, int]]:
    table = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].split()
        if not line:
            continue
        if len(line) < 2 or not line[1].isdigit() or int(line[1]) < 1:
            raise ValueError(f"{path}:{n}: expected '<entry> <GB> <cite>', got {raw!r}")
        table.append((line[0], int(line[1])))
    return table


def _matches(entry: str, test_id: str) -> bool:
    return test_id == entry or test_id.startswith((entry + "::", entry + "[", entry + "/"))


def id_weight(test_id: str, table: list[tuple[str, int]]) -> int:
    best = None
    for entry, gb in table:
        if _matches(entry, test_id) and (best is None or len(entry) > len(best[0])):
            best = (entry, gb)
    return DEFAULT_GB if best is None else best[1]


def unit_weight(ids: list[str], table: list[tuple[str, int]], line_gb: int | None = None) -> int:
    if line_gb is not None:
        return line_gb
    return max((id_weight(i, table) for i in ids), default=DEFAULT_GB)


def parse_line(line: str) -> tuple[str, int, str, int | None]:
    """files.txt line "<entry> <ids per chunk> [A] [<GB>G]" -> (entry, chunk, "A"|"N", GB|None)."""
    p = line.split()
    entry, chunk, rest = p[0], int(p[1]), p[2:]
    mk = "N"
    if rest and rest[0] == "A":
        mk, rest = "A", rest[1:]
    gb = None
    if rest:
        if len(rest) > 1 or not rest[0].endswith("G") or not rest[0][:-1].isdigit() or int(rest[0][:-1]) < 1:
            raise ValueError(f"files.txt line {line!r}: expected '<entry> <ids per chunk> [A] [<GB>G]'")
        gb = int(rest[0][:-1])
    return entry, chunk, mk, gb
