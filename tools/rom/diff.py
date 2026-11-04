"""What one NDS ROM changes against another: header fields, files, NARC members, code.

``diff_roms(vanilla, patched, symbols)`` compares two ``RomFiles`` (any object with its
methods). Task 7 turns the result into ``docs/kaizo/rom_diff.md``.

**Files.** Every FNT path is compared; a path only one side has is ``added`` or
``removed``. A changed file that is a NARC on both sides (by its ``NARC`` magic, not its
name: ``wazaeffect/we.arc`` is one, ``poketool/personal/pms.narc`` is not) is unpacked and
compared member by member. A NARC can change with every member equal (its container
layout alone differs): it is listed with no ``members_changed``.

**Code** (plan Ruling 7). The modules are ARM9, one per ndspy ``MainCodeFile`` section
(static first, then the ITCM and DTCM autoloads, each at its own RAM address) and each ARM9
overlay at its y9 RAM address. Modules of equal length are compared byte by byte at equal
offsets. Others are aligned with ``difflib.SequenceMatcher(None, a, b, autojunk=False)``
over 4-byte words; an unequal block of equal length is then compared byte by byte, one of
unequal length is narrowed to the bytes between its common prefix and suffix.

A changed byte is a **relocation**, counted per symbol and not listed, when it lies in a
unit that differs only because what it points at moved:

- an aligned 32-bit word whose vanilla value, as an address in the module (bit 0, the
  Thumb bit, kept apart), maps through the alignment to the patched value;
- a Thumb ``BL``/``BLX`` pair (``11110 hi11`` then ``11111 lo11`` / ``11101 lo11``, target
  ``pc + 4 + (sext(hi11) << 12) + (lo11 << 1)``, ``& ~3`` for ``BLX``) or an ARM ``BL``
  (cond ≠ ``1111``, bits 27-24 ``1011``, target ``pc + 8 + (sext(imm24) << 2)``) whose
  vanilla target, decoded at the vanilla ``pc``, maps to the target decoded at the patched
  ``pc``.

"Maps through the alignment" means: a vanilla address in an equal block of the same
module moves by that block's offset; the module's end and an overlay's BSS after it move
with the end. A target inside changed code maps to nothing, so such a unit stays a change.
An address outside the module (ARM9 called from an overlay, say) maps to itself: this
module's alignment does not move it, so a call re-encoded only because the caller moved
is a relocation. The identity never turns a changed word into a relocation by itself,
since a word is only looked at when it differs and the identity would leave it equal.
Nothing else is a relocation: a PC-relative ``LDR``/``ADR`` or a ``B`` whose offset changed
because code was inserted between it and its target stays a listed change (on Kaizo these
only occur inside functions that changed anyway), and no unit inside an unequal-length
block is examined.

The remaining changed bytes form ranges; ranges with fewer than 4 equal bytes between them
merge. With a symbol map, a range is split at every symbol boundary inside it and each
piece is named by ``SymbolMap.lookup`` at its start (adjacent pieces with the same name
join); an insertion (``start == end``) is named by the symbol holding its address, else
the one holding the byte before. A piece of a range whose two sides differ in length gets
the patched bytes in step with its vanilla bytes, and the last piece keeps the rest.

Anything this cannot pair (an overlay or ARM9 section only one ROM has, or ARM9 sections
at different addresses) raises ``ValueError`` naming it.
"""

from __future__ import annotations

import bisect
import difflib
import functools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# GBATEK "DS Cartridge Header": (offset, width, kind).
HEADER_FIELDS: dict[str, tuple[int, int, str]] = {
    "title": (0x000, 12, "ascii"),
    "game_code": (0x00C, 4, "ascii"),
    "maker_code": (0x010, 2, "ascii"),
    "unit_code": (0x012, 1, "int"),
    "device_capacity": (0x014, 1, "int"),
    "rom_version": (0x01E, 1, "int"),
    "autostart": (0x01F, 1, "int"),
    "arm9_rom_offset": (0x020, 4, "int"),
    "arm9_entry": (0x024, 4, "int"),
    "arm9_ram_address": (0x028, 4, "int"),
    "arm9_size": (0x02C, 4, "int"),
    "arm7_rom_offset": (0x030, 4, "int"),
    "arm7_entry": (0x034, 4, "int"),
    "arm7_ram_address": (0x038, 4, "int"),
    "arm7_size": (0x03C, 4, "int"),
    "fnt_offset": (0x040, 4, "int"),
    "fnt_size": (0x044, 4, "int"),
    "fat_offset": (0x048, 4, "int"),
    "fat_size": (0x04C, 4, "int"),
    "arm9_overlay_offset": (0x050, 4, "int"),
    "arm9_overlay_size": (0x054, 4, "int"),
    "arm7_overlay_offset": (0x058, 4, "int"),
    "arm7_overlay_size": (0x05C, 4, "int"),
    "icon_title_offset": (0x068, 4, "int"),
    "secure_area_crc": (0x06C, 2, "int"),
    "total_rom_size": (0x080, 4, "int"),
    "header_size": (0x084, 4, "int"),
    "logo_crc": (0x15C, 2, "int"),
    "header_crc": (0x15E, 2, "int"),
}

MERGE_GAP = 4  # change ranges with fewer equal bytes than this between them merge


@dataclass(frozen=True)
class HeaderChange:
    field: str  # a HEADER_FIELDS key
    vanilla: str  # ASCII text, or 0x-hex of the integer
    patched: str


@dataclass(frozen=True)
class FileChange:
    path: str
    status: str  # "changed" | "added" | "removed"
    vanilla_size: int | None
    patched_size: int | None
    members_changed: tuple[int, ...] = ()
    member_counts: tuple[int, int] | None = None


@dataclass(frozen=True)
class CodeChange:
    module: str  # "arm9" | "overlay<N>"
    start: int  # vanilla RAM addresses, half-open; an insertion has start == end
    end: int
    patched_start: int
    patched_end: int
    symbol: str | None


@dataclass(frozen=True)
class RomDiff:
    header: tuple[HeaderChange, ...]
    files: tuple[FileChange, ...]
    code: tuple[CodeChange, ...]
    relocations: dict[tuple[str, str | None], int]  # (module, symbol) -> relocated units
    modules: dict[str, tuple[int, int]]  # vanilla and patched size of each differing module


def diff_roms(vanilla, patched, symbols) -> RomDiff:
    """Compare two ROMs (``RomFiles``-like); ``symbols`` is a ``SymbolMap`` or ``None``."""
    code: list[CodeChange] = []
    relocations: Counter = Counter()
    modules: dict[str, tuple[int, int]] = {}
    for name, regions in _modules(vanilla, patched):
        if all(r.a == r.b for r in regions):
            continue
        modules[name] = (sum(len(r.a) for r in regions), sum(len(r.b) for r in regions))
        _Module(name, regions, symbols, code, relocations).diff()
    return RomDiff(_header(vanilla.header(), patched.header()), _files(vanilla, patched),
                   tuple(code), dict(relocations), modules)


# --- header -----------------------------------------------------------------------------


def _field(header: bytes, offset: int, width: int, kind: str) -> str:
    raw = header[offset:offset + width]
    if kind == "int":
        return f"0x{int.from_bytes(raw, 'little'):X}"
    text = raw.rstrip(b"\x00")
    if all(0x20 <= c < 0x7F for c in text):
        return text.decode("ascii")
    return "0x" + raw.hex().upper()  # not printable ASCII: the raw bytes, still < 32 chars


def _header(a: bytes, b: bytes) -> tuple[HeaderChange, ...]:
    out = []
    for name, (offset, width, kind) in HEADER_FIELDS.items():
        va, vb = _field(a, offset, width, kind), _field(b, offset, width, kind)
        if va != vb:
            out.append(HeaderChange(name, va, vb))
    return tuple(out)


# --- files ------------------------------------------------------------------------------


def _files(vanilla, patched) -> tuple[FileChange, ...]:
    va, pa = set(vanilla.paths()), set(patched.paths())
    out = []
    for path in sorted(va | pa):
        if path not in pa:
            out.append(FileChange(path, "removed", len(vanilla.file(path)), None))
        elif path not in va:
            out.append(FileChange(path, "added", None, len(patched.file(path))))
        else:
            a, b = vanilla.file(path), patched.file(path)
            if a == b:
                continue
            if a[:4] == b"NARC" and b[:4] == b"NARC":  # we.arc is one, pms.narc is not
                ma, mb = vanilla.narc(path), patched.narc(path)
                changed = tuple(i for i in range(max(len(ma), len(mb)))
                                if i >= len(ma) or i >= len(mb) or ma[i] != mb[i])
                out.append(FileChange(path, "changed", len(a), len(b), changed,
                                      (len(ma), len(mb))))
            else:
                out.append(FileChange(path, "changed", len(a), len(b)))
    return tuple(out)


# --- code -------------------------------------------------------------------------------


def _modules(vanilla, patched):
    """(module name, regions); ARM9's sections are regions of one module ``arm9``."""
    sa, sb = vanilla.arm9_sections(), patched.arm9_sections()
    if [r for r, _ in sa] != [r for r, _ in sb]:
        raise ValueError("ARM9 sections differ in number or RAM address: "
                         f"{[hex(r) for r, _ in sa]} vs {[hex(r) for r, _ in sb]}")
    yield "arm9", [_Region(ra, a, rb, b, 0) for (ra, a), (rb, b) in zip(sa, sb, strict=True)]
    ia, ib = vanilla.overlay_ids(), patched.overlay_ids()
    if ia != ib:
        raise ValueError(f"overlay ids differ: only vanilla {sorted(set(ia) - set(ib))}, "
                         f"only patched {sorted(set(ib) - set(ia))}")
    for i in ia:
        ra, _, bss = vanilla.overlay_info(i)
        rb = patched.overlay_info(i)[0]
        yield f"overlay{i}", [_Region(ra, vanilla.overlay(i), rb, patched.overlay(i), bss)]


@dataclass(frozen=True)
class _Region:
    """One separately aligned piece of a module: an ARM9 section or an overlay."""

    ra: int  # vanilla RAM address
    a: bytes  # vanilla bytes
    rb: int  # patched RAM address
    b: bytes
    bss: int  # vanilla BSS size right after it (0 when unknown, as for ARM9 sections)


def _sext(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _thumb_bl(data: bytes, off: int, pc: int) -> int | None:
    """Target of the Thumb BL/BLX pair at ``off`` (RAM ``pc``), or None."""
    if off < 0 or off + 4 > len(data):
        return None
    first = int.from_bytes(data[off:off + 2], "little")
    second = int.from_bytes(data[off + 2:off + 4], "little")
    if first >> 11 != 0b11110 or second >> 11 not in (0b11111, 0b11101):
        return None
    target = pc + 4 + (_sext(first & 0x7FF, 11) << 12) + ((second & 0x7FF) << 1)
    return target & ~3 if second >> 11 == 0b11101 else target


def _arm_bl(word: int, pc: int) -> int | None:
    if word >> 28 == 0xF or (word >> 24) & 0xF != 0b1011:
        return None
    return pc + 8 + (_sext(word & 0xFFFFFF, 24) << 2)


class _Module:
    """Diff of one module's regions, appended to the shared ``code``/``relocations``."""

    def __init__(self, name, regions, symbols, code, relocations):
        self.name = name
        self.regions = regions
        self.symbols = symbols
        self.code = code
        self.relocations = relocations
        self.maps = [self._align(r) for r in regions]
        self._starts = [[op[1] for op in ops] for ops in self.maps]
        self._boundaries: list[int] | None = None

    def _bounds(self) -> list[int]:
        """Every start and end of a sized symbol of this module, sorted."""
        if self._boundaries is None:
            self._boundaries = sorted({x for s in self.symbols.symbols(self.name) if s.size
                                       for x in (s.address, s.address + s.size)})
        return self._boundaries

    @staticmethod
    def _align(region: _Region):
        a, b = region.a, region.b
        if len(a) == len(b):
            return [("equal" if a == b else "replace", 0, len(a), 0, len(b))]
        wa = [a[i:i + 4] for i in range(0, len(a), 4)]
        wb = [b[i:i + 4] for i in range(0, len(b), 4)]
        ops = difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
        out = []
        for tag, i1, i2, j1, j2 in ops:  # word indices to byte offsets
            out.append((tag, 4 * i1, min(4 * i2, len(a)), 4 * j1, min(4 * j2, len(b))))
        return out

    def map_address(self, addr: int) -> int | None:
        """A vanilla RAM address moved through this module's alignment, or None.

        An address outside every region of the module (another module's code, which this
        module's alignment does not move) maps to itself.
        """
        for region, ops, starts in zip(self.regions, self.maps, self._starts, strict=True):
            ra, a, rb, b = region.ra, region.a, region.rb, region.b
            off = addr - ra
            if 0 <= off < len(a):
                tag, i1, _, j1, _ = ops[bisect.bisect_right(starts, off) - 1]
                return rb + j1 + off - i1 if tag == "equal" else None
            if len(a) <= off < len(a) + region.bss or off == len(a):
                return rb + off - len(a) + len(b)
        return addr

    def _map_pointer(self, value: int) -> int | None:
        thumb = value & 1
        mapped = self.map_address(value & ~1)
        return None if mapped is None else mapped | thumb

    def _relocated(self, a: bytes, b: bytes, ra: int, rb: int, off: int, delta: int):
        """The (start, end) vanilla offsets of a relocated unit holding byte ``off``."""
        word = off & ~3
        if word + 4 <= len(a) and word + delta + 4 <= len(b):
            wa = int.from_bytes(a[word:word + 4], "little")
            wb = int.from_bytes(b[word + delta:word + delta + 4], "little")
            if wa != wb and self._map_pointer(wa) == wb:
                return word, word + 4
            ta, tb = _arm_bl(wa, ra + word), _arm_bl(wb, rb + word + delta)
            if (ta is not None and tb is not None and wa >> 24 == wb >> 24
                    and self.map_address(ta) == tb):
                return word, word + 4
        half = off & ~1
        for start in (half - 2, half):
            ta = _thumb_bl(a, start, ra + start)
            tb = _thumb_bl(b, start + delta, rb + start + delta)
            if ta is None or tb is None or a[start:start + 4] == b[start + delta:start + delta + 4]:
                continue
            blx = int.from_bytes(a[start + 2:start + 4], "little") >> 11 == 0b11101
            if blx != (int.from_bytes(b[start + delta + 2:start + delta + 4], "little") >> 11
                       == 0b11101):
                continue
            mapped = self.map_address(ta)
            if mapped is not None and (mapped & ~3 if blx else mapped) == tb:
                return start, start + 4
        return None

    def diff(self) -> None:
        for region, ops in zip(self.regions, self.maps, strict=True):
            ra, a, rb, b = region.ra, region.a, region.rb, region.b
            ranges: list[list[int]] = []  # [vanilla start, end, patched start, end] offsets
            for tag, i1, i2, j1, j2 in ops:
                if tag == "equal":
                    continue
                if i2 - i1 == j2 - j1:
                    ranges.extend(self._positional(a, b, ra, rb, i1, i2, j1 - i1))
                else:
                    while i1 < i2 and j1 < j2 and a[i1] == b[j1]:
                        i1, j1 = i1 + 1, j1 + 1
                    while i2 > i1 and j2 > j1 and a[i2 - 1] == b[j2 - 1]:
                        i2, j2 = i2 - 1, j2 - 1
                    ranges.append([i1, i2, j1, j2])
            for r in _merge(ranges):
                self._emit(ra, rb, *r)

    def _positional(self, a, b, ra, rb, i1, i2, delta):
        """Changed byte ranges of an equal-length block, relocated units counted apart."""
        out: list[list[int]] = []
        covered = -1  # vanilla offsets below this are in a relocated unit already counted
        for off in range(i1, i2):
            if a[off] == b[off + delta] or off < covered:
                continue
            unit = self._relocated(a, b, ra, rb, off, delta)
            if unit is not None:
                covered = unit[1]
                sym = self.symbols.lookup(self.name, ra + unit[0]) if self.symbols else None
                self.relocations[(self.name, sym)] += 1
                continue
            if out and out[-1][1] == off:
                out[-1][1] += 1
                out[-1][3] += 1
            else:
                out.append([off, off + 1, off + delta, off + delta + 1])
        return out

    def _emit(self, ra: int, rb: int, i1: int, i2: int, j1: int, j2: int) -> None:
        start, end, pstart, pend = ra + i1, ra + i2, rb + j1, rb + j2
        if self.symbols is None:
            self.code.append(CodeChange(self.name, start, end, pstart, pend, None))
            return
        if start == end:
            sym = self.symbols.lookup(self.name, start)
            if sym is None:
                sym = self.symbols.lookup(self.name, start - 1)
            self.code.append(CodeChange(self.name, start, end, pstart, pend, sym))
            return
        bounds = self._bounds()
        cuts = bounds[bisect.bisect_right(bounds, start):bisect.bisect_left(bounds, end)]
        pieces: list[list] = []
        for lo, hi in zip([start, *cuts], [*cuts, end], strict=True):
            sym = self.symbols.lookup(self.name, lo)
            if pieces and pieces[-1][2] == sym:
                pieces[-1][1] = hi
            else:
                pieces.append([lo, hi, sym])
        for k, (lo, hi, sym) in enumerate(pieces):
            plo = min(pstart + lo - start, pend)
            phi = pend if k == len(pieces) - 1 else min(pstart + hi - start, pend)
            self.code.append(CodeChange(self.name, lo, hi, plo, phi, sym))


def _merge(ranges: list[list[int]]) -> list[list[int]]:
    """Join ranges with fewer than ``MERGE_GAP`` equal bytes between them."""
    out: list[list[int]] = []
    for r in sorted(ranges):
        if out and r[0] - out[-1][1] < MERGE_GAP:
            out[-1][1], out[-1][3] = r[1], r[3]
        else:
            out.append(list(r))
    return out


# --- NARC member names ------------------------------------------------------------------

_MEMBER_CONSTANTS = {
    "pl_msg.narc": ("text_banks", "TEXT_BANK_"),
    "pl_personal.narc": ("species", "SPECIES_"),
    "wotbl.narc": ("species", "SPECIES_"),
    "evo.narc": ("species", "SPECIES_"),
    "pl_waza_tbl.narc": ("moves", "MOVE_"),
    "trdata.narc": ("trainers", "TRAINER_"),
    "trpoke.narc": ("trainers", "TRAINER_"),
}


@functools.cache
def _member_names(decomp: Path, name: str, prefix: str) -> tuple[str, ...]:
    from tools.extract_decomp import _constants

    return tuple(_constants(decomp, name, prefix))


def member_label(path: str, index: int, decomp: Path) -> str | None:
    """The decomp constant naming NARC member ``index`` of ``path``, or None.

    ``pl_msg.narc`` members are text banks in ``generated/text_banks.txt`` order
    (``res/text/meson.build:81``); personal, learnset and evolution members are species,
    ``pl_waza_tbl.narc`` members moves, ``trdata``/``trpoke`` members trainers.
    """
    kind = _MEMBER_CONSTANTS.get(path.rsplit("/", 1)[-1])
    if kind is None:
        return None
    names = _member_names(Path(decomp), *kind)
    return names[index] if 0 <= index < len(names) else None
