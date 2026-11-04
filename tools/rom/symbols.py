"""Symbols of the vanilla decomp build, from the Metrowerks linker map.

The decomp links ``main.nef`` with ``-map closure,unused`` (``meson.build:99``), which
writes ``build/main.nef.xMAP``. ``docs/rom_pipeline.md`` ("Decomp build") copies it to
``DEFAULT_MAP``. The map's layout, as measured on the build at ``DECOMP_COMMIT``:

- A preamble (``# Unreferenced symbols:``, ``# Link map of _start`` closure tree) that
  is not parsed.
- One ``# .<name>`` header per output section, each followed by ``# .<name>.bss``:
  ``main``, ``ITCM`` and ``DTCM`` are the ARM9 static module and its autoloads; every
  ``Overlay <name>`` of ``platinum.us/main.lsf`` gets its own pair. Linker-generated
  symbols are ``#>ADDRESS          NAME (linker command file)``; among them
  ``SDK_OVERLAY_<name>_ID`` gives the overlay's id (what ``FS_OVERLAY_ID(battle)``
  resolves to, ``src/unk_0203D1B8.c:238``), so section ``battle`` is ``overlay16``.
- Symbol lines ``  ADDRESS SIZE SECTION NAME\\t(OBJECT)``, hex fields of 8 digits; the
  one ``Exception Table Index`` line has no section. Names may contain spaces
  (``virtual table for ...``). Zero-size lines are labels (``$t``, ``$d``, ``.text``).
- ``# Memory map:`` and the link times close the file; nothing after it is parsed.

Any line in a section that is none of these raises ``SymbolMapError`` with its line
number, and so does a symbol in a section that is neither ARM9 nor an overlay with an id.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.rom.inputs import REPO_ROOT

DECOMP_COMMIT = "579d2d5c65e736a1a4268fb7acbfd70712177c6c"
DEFAULT_MAP = REPO_ROOT / ".cache" / "decomp" / DECOMP_COMMIT / "main.nef.xMAP"

_ARM9_SECTIONS = {"main", "ITCM", "DTCM"}
_HEADER = re.compile(r"# \.(\S+)")
_LINKER = re.compile(r"#>([0-9A-F]{8}) +(\S+) \(linker command file\)")
_OVERLAY_ID = re.compile(r"SDK_OVERLAY_(\w+)_ID")
_SYMBOL = re.compile(r"  ([0-9A-F]{8}) ([0-9A-F]{8}) (?:(\.\S+) +| +)(\S.*?)\t\((.*)\)")


class SymbolMapError(ValueError):
    """The linker map has a line or section this parser does not account for."""


@dataclass(frozen=True)
class Symbol:
    module: str  # "arm9" or "overlay<N>"
    address: int  # RAM address
    size: int
    name: str
    obj: str  # object file, as the map names it ("libos.a src_os_irqhandl")
    section: str  # ".text", ".rodata", ".data", ".bss", ".itcm", ...; "" for the ETABLE index


class SymbolMap:
    """Symbols by module; ``lookup`` names the smallest symbol containing an address."""

    def __init__(self, symbols: Iterable[Symbol]):
        self._all = list(symbols)
        by_name: dict[str, list[Symbol]] = {}
        by_module: dict[str, list[tuple[int, Symbol]]] = {}
        for order, s in enumerate(self._all):
            by_name.setdefault(s.name, []).append(s)
            by_module.setdefault(s.module, []).append((order, s))
        self._by_name = by_name
        self._sorted = {
            m: [s for _, s in sorted(entries, key=lambda e: (e[1].address, e[0]))]
            for m, entries in by_module.items()
        }
        self._segments = {m: _segments(entries) for m, entries in by_module.items()}

    @classmethod
    def from_map(cls, path: Path) -> SymbolMap:
        return cls(_parse(path))

    def lookup(self, module: str, address: int) -> str | None:
        """Name of the smallest sized symbol whose ``[address, address + size)`` holds it."""
        starts, winners = self._segments.get(module, ([], []))
        i = bisect.bisect_right(starts, address) - 1
        if i < 0:
            return None
        w = winners[i]
        return w.name if w is not None and address < w.address + w.size else None

    def find(self, name: str) -> Symbol:
        found = self._by_name.get(name, [])
        if len(found) != 1:
            raise KeyError(
                f"symbol {name!r} is {'ambiguous' if found else 'not in the map'}"
                + (f" ({len(found)} definitions)" if found else "")
            )
        return found[0]

    def symbols(self, module: str) -> list[Symbol]:
        """The module's symbols sorted by address (map order among equal addresses)."""
        return list(self._sorted.get(module, []))

    def modules(self) -> list[str]:
        """Every module with at least one symbol."""
        return sorted(self._sorted)


def _segments(entries: list[tuple[int, Symbol]]) -> tuple[list[int], list[Symbol | None]]:
    """Split a module at every symbol boundary; each piece keeps its smallest container.

    Returns piece start addresses and, per piece, the winning symbol (smallest size, then
    first in map order) or ``None``. A piece ends where the next one starts.
    """
    sized = sorted(((s.address, o, s) for o, s in entries if s.size), key=lambda e: e[:2])
    bounds = sorted({b for _, _, s in sized for b in (s.address, s.address + s.size)})
    starts: list[int] = []
    winners: list[Symbol | None] = []
    active: list[tuple[int, Symbol]] = []
    j = 0
    for lo in bounds:
        active = [(o, s) for o, s in active if s.address + s.size > lo]
        while j < len(sized) and sized[j][0] == lo:
            active.append((sized[j][1], sized[j][2]))
            j += 1
        best = min(active, key=lambda e: (e[1].size, e[0]))[1] if active else None
        starts.append(lo)
        winners.append(best)
    return starts, winners


def _parse(path: Path) -> list[Symbol]:
    text = path.read_text(encoding="utf-8")
    ids: dict[str, int] = {}
    section: str | None = None  # current header's name without ".bss"
    out: list[Symbol] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if line == "# Memory map:":
            break
        header = _HEADER.fullmatch(line)
        if header:
            section = header[1].removesuffix(".bss")
            continue
        if section is None or not line.strip() or line == "# Exception index":
            continue  # preamble, blank line, or the ETABLE index title inside ".main"
        linker = _LINKER.fullmatch(line)
        if linker:
            ov = _OVERLAY_ID.fullmatch(linker[2])
            if ov:
                value = int(linker[1], 16)
                if ids.setdefault(ov[1], value) != value:
                    raise SymbolMapError(f"{path}:{lineno}: {linker[2]} redefined as {value}")
            continue
        sym = _SYMBOL.fullmatch(line)
        if not sym:
            raise SymbolMapError(f"{path}:{lineno}: unrecognised line {line!r}")
        if section in _ARM9_SECTIONS:
            module = "arm9"
        elif section in ids:
            module = f"overlay{ids[section]}"
        else:
            raise SymbolMapError(
                f"{path}:{lineno}: symbol in section .{section}, which is neither ARM9 "
                f"nor an overlay with an SDK_OVERLAY_{section}_ID line before it"
            )
        out.append(
            Symbol(module, int(sym[1], 16), int(sym[2], 16), sym[4], sym[5].strip(), sym[3] or "")
        )
    else:
        raise SymbolMapError(f"{path}: no '# Memory map:' line; the map looks truncated")
    return out
