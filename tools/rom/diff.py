"""What one NDS ROM changes against another: header fields, files, NARC members, code.

``diff_roms(vanilla, patched, symbols)`` compares two ``RomFiles`` (any object with its
methods). ``write_report`` turns the result into ``docs/kaizo/rom_diff.md``, with the
hand-edited classification in ``data/kaizo/rom_diff.toml`` (plan Ruling 16); ``main`` is the
``python -m tools.rom.diff`` command that does both on the real ROMs.

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
module, or in a non-equal block whose two sides have equal length (a table whose own
entries were relocated, say), moves by that block's constant offset; the module's end and an overlay's BSS after it move
with the end. A target inside changed code maps to nothing, so such a unit stays a change.
An address outside the module (ARM9 called from an overlay, say) maps to itself: this
module's alignment does not move it, so a call re-encoded only because the caller moved
is a relocation. The identity never turns a changed word into a relocation by itself,
since a word is only looked at when it differs and the identity would leave it equal.
Nothing else is a relocation: a PC-relative ``LDR``/``ADR`` or a ``B`` whose offset changed
because code was inserted between it and its target stays a listed change (on Kaizo these
only occur inside functions that changed anyway), and no unit inside an unequal-length
block is examined. Re-encoded BLs straddling words can fragment the word alignment into an
insert, a coincidentally equal word and a delete; a run of non-equal blocks split only by
such short equal runs is first joined into one block when its two sides have equal length
(``_coalesce``), so its units are examined at that block's constant offset.

The remaining changed bytes form ranges; ranges with fewer than 4 equal bytes between them
merge. With a symbol map, a range is split at every symbol boundary inside it and each
piece is named by ``SymbolMap.lookup`` at its start (adjacent pieces with the same name
join); an insertion (``start == end``) is named by the symbol holding its address, else
the one holding the byte before. A piece of a range whose two sides differ in length gets
the patched bytes in step with its vanilla bytes, and the last piece keeps the rest.

Anything this cannot pair (an overlay or ARM9 section only one ROM has, or ARM9 sections
at different addresses) raises ``ValueError`` naming it.

**Outside the file system** (``parts``). The ARM9 overlay table is compared field by field
per 32-byte entry (GBATEK "DS Cartridge ARM9/ARM7 Overlay Tables": id, RAM address, RAM
size, BSS size, static-initialiser start and end, file id, and a word holding the
compressed size in bits 0-23 and flags in bits 24-31); so is the ARM7 overlay table. The
12-byte ARM9 footer is three words (``0xDEC00621``, the ModuleParams offset, 0). The ARM7
binary and the banner are compared by size and SHA-1. Only differing fields are listed.

**Injected code** (``injected``, ``far_calls``). Every Thumb ``BL``/``BLX`` inside a listed
code change is decoded at its patched address. A target outside every patched module
image (ARM9 sections, overlays) is a call into code loaded from elsewhere: it is assigned to
the declared injected region (``injected_code``: a NARC member and the address it is loaded
at, ``KAIZO_INJECTED_CODE``) whose ``[address, address + member size)`` holds it, else it is
a ``far_calls`` entry, which the report lists as unexplained. ARM ``BL`` is not decoded.
"""

from __future__ import annotations

import bisect
import difflib
import functools
import hashlib
import json
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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

# Code the Kaizo patch adds outside every module (K1 final review): data/weather_sys.narc
# member 9 grows from 552 to 90112 bytes and is loaded at 0x023C8000 by the changed
# NitroMain and OS_GetInitArenaLo; HandleInput_Main, PartyMenuCB_LevelUp and
# SetMonDataFromMon call into it. (path, member, load address).
KAIZO_INJECTED_CODE: tuple[tuple[str, int, int], ...] = (("data/weather_sys.narc", 9, 0x023C8000),)

OVERLAY_ENTRY_SIZE = 32
OVERLAY_ENTRY_FIELDS = (
    "id",
    "ram_address",
    "ram_size",
    "bss_size",
    "sinit_start",
    "sinit_end",
    "file_id",
)
FOOTER_FIELDS = ("nitrocode", "module_params_offset", "reserved")

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
class PartChange:
    part: str  # "arm9_overlay_table" | "arm9_footer" | "arm7" | "arm7_overlay_table" | "banner"
    field: str
    vanilla: str  # 0x-hex, a count, or a SHA-1
    patched: str


@dataclass(frozen=True)
class FarCall:
    module: str  # the module of the calling code change
    symbol: str | None  # its symbol
    address: int  # patched RAM address of the BL
    target: int


@dataclass(frozen=True)
class InjectedCode:
    path: str
    member: int
    load_address: int
    vanilla_size: int | None  # None when the member is new
    patched_size: int
    calls: tuple[FarCall, ...]  # listed code changes calling into it


@dataclass(frozen=True)
class RomDiff:
    header: tuple[HeaderChange, ...]
    files: tuple[FileChange, ...]
    code: tuple[CodeChange, ...]
    relocations: dict[tuple[str, str | None], int]  # (module, symbol) -> relocated units
    modules: dict[str, tuple[int, int]]  # vanilla and patched size of each differing module
    parts: tuple[PartChange, ...] = ()
    injected: tuple[InjectedCode, ...] = ()
    far_calls: tuple[FarCall, ...] = ()  # calls outside every module and injected region


def diff_roms(vanilla, patched, symbols, injected_code=()) -> RomDiff:
    """Compare two ROMs (``RomFiles``-like); ``symbols`` is a ``SymbolMap`` or ``None``;
    ``injected_code`` declares ``(path, member, load address)`` regions of code the patched
    ROM loads from a NARC member (``KAIZO_INJECTED_CODE``). A declared member the patched
    ROM lacks raises ``ValueError``."""
    code: list[CodeChange] = []
    relocations: Counter = Counter()
    modules: dict[str, tuple[int, int]] = {}
    images: dict[str, list[tuple[int, bytes]]] = {}
    for name, regions in _modules(vanilla, patched):
        images[name] = [(r.rb, r.b) for r in regions]
        if all(r.a == r.b for r in regions):
            continue
        modules[name] = (sum(len(r.a) for r in regions), sum(len(r.b) for r in regions))
        _Module(name, regions, symbols, code, relocations).diff()
    injected, far = _far_calls(vanilla, patched, code, images, injected_code)
    return RomDiff(
        _header(vanilla.header(), patched.header()),
        _files(vanilla, patched),
        tuple(code),
        dict(relocations),
        modules,
        _parts(vanilla, patched),
        injected,
        far,
    )


# --- parts outside the file system ------------------------------------------------------


def _hex(n: int) -> str:
    return f"0x{n:X}"


def _words(data: bytes) -> list[int]:
    return [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data) - 3, 4)]


def _overlay_table(part: str, a: bytes, b: bytes) -> list[PartChange]:
    for side, data in (("vanilla", a), ("patched", b)):
        if len(data) % OVERLAY_ENTRY_SIZE:
            raise ValueError(f"{part}: {side} table is {len(data)} bytes, not whole entries")
    out = []
    na, nb = len(a) // OVERLAY_ENTRY_SIZE, len(b) // OVERLAY_ENTRY_SIZE
    if na != nb:
        out.append(PartChange(part, "entries", str(na), str(nb)))
    names = (*OVERLAY_ENTRY_FIELDS, "compressed_size", "compressed_flags")
    for i in range(min(na, nb)):
        ea = _words(a[i * OVERLAY_ENTRY_SIZE : (i + 1) * OVERLAY_ENTRY_SIZE])
        eb = _words(b[i * OVERLAY_ENTRY_SIZE : (i + 1) * OVERLAY_ENTRY_SIZE])
        fa = [*ea[:7], ea[7] & 0xFFFFFF, ea[7] >> 24]
        fb = [*eb[:7], eb[7] & 0xFFFFFF, eb[7] >> 24]
        for field, va, vb in zip(names, fa, fb, strict=True):
            if va != vb:
                out.append(PartChange(part, f"entry {i} {field}", _hex(va), _hex(vb)))
    return out


def _blob(part: str, a: bytes, b: bytes) -> list[PartChange]:
    if a == b:
        return []
    out = []
    if len(a) != len(b):
        out.append(PartChange(part, "size", _hex(len(a)), _hex(len(b))))
    out.append(PartChange(part, "sha1", hashlib.sha1(a).hexdigest(), hashlib.sha1(b).hexdigest()))
    return out


def _parts(vanilla, patched) -> tuple[PartChange, ...]:
    out = _overlay_table(
        "arm9_overlay_table", vanilla.arm9_overlay_table(), patched.arm9_overlay_table()
    )
    fa, fb = vanilla.arm9_footer(), patched.arm9_footer()
    if fa != fb:
        if len(fa) != 12 or len(fb) != 12:
            out += _blob("arm9_footer", fa, fb)
        else:
            for field, va, vb in zip(FOOTER_FIELDS, _words(fa), _words(fb), strict=True):
                if va != vb:
                    out.append(PartChange("arm9_footer", field, _hex(va), _hex(vb)))
    out += _blob("arm7", vanilla.arm7(), patched.arm7())
    out += _overlay_table(
        "arm7_overlay_table", vanilla.arm7_overlay_table(), patched.arm7_overlay_table()
    )
    out += _blob("banner", vanilla.banner(), patched.banner())
    return tuple(out)


# --- calls into injected code -----------------------------------------------------------


def _far_calls(vanilla, patched, code, images, injected_code):
    """(declared injected regions with their callers, calls into no module and no region)."""
    regions = []
    for path, member, address in injected_code:
        members = patched.narc(path)
        if member >= len(members):
            raise ValueError(
                f"injected code {path} member {member}: the patched ROM has {len(members)} members"
            )
        old = vanilla.narc(path) if path in vanilla.paths() else []
        vsize = len(old[member]) if member < len(old) else None
        regions.append((path, member, address, vsize, len(members[member])))
    known = [(ram, ram + len(data)) for imgs in images.values() for ram, data in imgs]
    calls: list[list[FarCall]] = [[] for _ in regions]
    far: list[FarCall] = []
    for c in code:
        for ram, data in images[c.module]:
            if not ram <= c.patched_start < ram + len(data):
                continue
            for pc in range(c.patched_start & ~1, c.patched_end, 2):
                target = _thumb_bl(data, pc - ram, pc)
                if target is None or any(lo <= target < hi for lo, hi in known):
                    continue
                call = FarCall(c.module, c.symbol, pc, target)
                for k, (_, _, address, _, size) in enumerate(regions):
                    if address <= target < address + size:
                        calls[k].append(call)
                        break
                else:
                    far.append(call)
            break
    injected = tuple(
        InjectedCode(path, member, address, vsize, psize, tuple(calls[k]))
        for k, (path, member, address, vsize, psize) in enumerate(regions)
    )
    return injected, tuple(far)


# --- header -----------------------------------------------------------------------------


def _field(header: bytes, offset: int, width: int, kind: str) -> str:
    raw = header[offset : offset + width]
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
                changed = tuple(
                    i
                    for i in range(max(len(ma), len(mb)))
                    if i >= len(ma) or i >= len(mb) or ma[i] != mb[i]
                )
                out.append(FileChange(path, "changed", len(a), len(b), changed, (len(ma), len(mb))))
            else:
                out.append(FileChange(path, "changed", len(a), len(b)))
    return tuple(out)


# --- code -------------------------------------------------------------------------------


def _modules(vanilla, patched):
    """(module name, regions); ARM9's sections are regions of one module ``arm9``."""
    sa, sb = vanilla.arm9_sections(), patched.arm9_sections()
    if [r for r, _ in sa] != [r for r, _ in sb]:
        raise ValueError(
            "ARM9 sections differ in number or RAM address: "
            f"{[hex(r) for r, _ in sa]} vs {[hex(r) for r, _ in sb]}"
        )
    yield "arm9", [_Region(ra, a, rb, b, 0) for (ra, a), (rb, b) in zip(sa, sb, strict=True)]
    ia, ib = vanilla.overlay_ids(), patched.overlay_ids()
    if ia != ib:
        raise ValueError(
            f"overlay ids differ: only vanilla {sorted(set(ia) - set(ib))}, "
            f"only patched {sorted(set(ib) - set(ia))}"
        )
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
    first = int.from_bytes(data[off : off + 2], "little")
    second = int.from_bytes(data[off + 2 : off + 4], "little")
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
            self._boundaries = sorted(
                {
                    x
                    for s in self.symbols.symbols(self.name)
                    if s.size
                    for x in (s.address, s.address + s.size)
                }
            )
        return self._boundaries

    @staticmethod
    def _align(region: _Region):
        a, b = region.a, region.b
        if len(a) == len(b):
            return [("equal" if a == b else "replace", 0, len(a), 0, len(b))]
        wa = [a[i : i + 4] for i in range(0, len(a), 4)]
        wb = [b[i : i + 4] for i in range(0, len(b), 4)]
        ops = difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
        out = []
        for tag, i1, i2, j1, j2 in _coalesce(ops):  # word indices to byte offsets
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
                _, i1, i2, j1, j2 = ops[bisect.bisect_right(starts, off) - 1]
                # an equal block, or a replace block of equal length: a constant offset
                return rb + j1 + off - i1 if i2 - i1 == j2 - j1 else None
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
            wa = int.from_bytes(a[word : word + 4], "little")
            wb = int.from_bytes(b[word + delta : word + delta + 4], "little")
            if wa != wb and self._map_pointer(wa) == wb:
                return word, word + 4
            ta, tb = _arm_bl(wa, ra + word), _arm_bl(wb, rb + word + delta)
            if (
                ta is not None
                and tb is not None
                and wa >> 24 == wb >> 24
                and self.map_address(ta) == tb
            ):
                return word, word + 4
        half = off & ~1
        for start in (half - 2, half):
            ta = _thumb_bl(a, start, ra + start)
            tb = _thumb_bl(b, start + delta, rb + start + delta)
            if (
                ta is None
                or tb is None
                or a[start : start + 4] == b[start + delta : start + delta + 4]
            ):
                continue
            blx = int.from_bytes(a[start + 2 : start + 4], "little") >> 11 == 0b11101
            if blx != (
                int.from_bytes(b[start + delta + 2 : start + delta + 4], "little") >> 11 == 0b11101
            ):
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
        cuts = bounds[bisect.bisect_right(bounds, start) : bisect.bisect_left(bounds, end)]
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


SHORT_EQUAL = 8  # words; an equal run this short between non-equal opcodes may be chance


def _coalesce(ops: list[tuple]) -> list[tuple]:
    """Join non-equal opcodes split by short equal runs when the join has equal lengths.

    Re-encoded Thumb BLs straddling words can make the word alignment pair a patched word
    with a coincidentally equal vanilla word elsewhere, splitting a region of constant
    offset into an insert, a short equal run and a delete. A maximal run of non-equal
    opcodes separated by equal runs of at most ``SHORT_EQUAL`` words becomes one
    ``replace`` when its vanilla and patched lengths agree, so it is compared at that
    constant offset and its relocations are recognised; otherwise it is kept as found.
    """
    out: list[tuple] = []
    k = 0
    while k < len(ops):
        if ops[k][0] == "equal":
            out.append(ops[k])
            k += 1
            continue
        end = k  # the run is ops[k:end + 1], from non-equal to non-equal
        n = k + 1
        while (
            n + 1 < len(ops)
            and ops[n][0] == "equal"
            and ops[n][2] - ops[n][1] <= SHORT_EQUAL
            and ops[n + 1][0] != "equal"
        ):
            end = n + 1
            n += 2
        run = ops[k : end + 1]
        i1, j1, i2, j2 = run[0][1], run[0][3], run[-1][2], run[-1][4]
        if len(run) > 1 and i2 - i1 == j2 - j1:
            out.append(("replace", i1, i2, j1, j2))
        else:
            out.extend(run)
        k = end + 1
    return out


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


# --- classification ---------------------------------------------------------------------

STATUSES = ("modelled", "irrelevant", "unknown")


def _key(c: CodeChange) -> tuple[str, str]:
    """(module, symbol), or (module, "0xSTART-0xEND") for a range outside every symbol."""
    return c.module, c.symbol if c.symbol is not None else f"0x{c.start:08X}-0x{c.end:08X}"


def _module_order(module: str) -> tuple[int, int]:
    return (0, 0) if module == "arm9" else (1, int(module.removeprefix("overlay")))


def load_classification(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """``(module, symbol-or-range) -> (status, note)`` from the ``[[code]]`` tables."""
    with Path(path).open("rb") as f:
        data = tomllib.load(f)
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for entry in data.get("code", []):
        try:
            key = (entry["module"], entry["symbol"])
            status, note = entry["status"], entry.get("note", "")
        except KeyError as e:
            raise ValueError(f"{path}: [[code]] entry {entry} lacks {e}") from e
        if status not in STATUSES:
            raise ValueError(
                f"{path}: {key[0]} {key[1]}: unknown status {status!r}; expected one of {STATUSES}"
            )
        if key in out:
            raise ValueError(f"{path}: duplicate entry {key[0]} {key[1]}")
        out[key] = (status, note)
    return out


def _changed_keys(diff: RomDiff) -> list[tuple[str, str]]:
    """Each changed (module, symbol-or-range) once, by module then first address."""
    first: dict[tuple[str, str], int] = {}
    for c in diff.code:
        first.setdefault(_key(c), c.start)
    return sorted(first, key=lambda k: (_module_order(k[0]), first[k], k[1]))


def classification_skeleton(diff: RomDiff) -> str:
    """TOML with an ``unknown`` entry for every changed (module, symbol), by address."""
    lines = [
        "# Classification of every code change the Kaizo patch makes (plan K1 Ruling 16).",
        "# status: modelled | irrelevant | unknown.  symbol: a name, or 0xSTART-0xEND for a",
        "# range outside every symbol.  An entry matching no current change is an error.",
        "",
    ]
    for module, symbol in _changed_keys(diff):
        lines += [
            "[[code]]",
            f"module = {json.dumps(module)}",
            f"symbol = {json.dumps(symbol)}",
            'status = "unknown"',
            'note = ""',
            "",
        ]
    return "\n".join(lines)


def stale_entries(diff: RomDiff, classification: dict) -> list[tuple[str, str]]:
    """Classification keys that match no change in ``diff``, sorted."""
    current = set(_changed_keys(diff))
    return sorted(k for k in classification if k not in current)


# --- report -----------------------------------------------------------------------------


def default_provenance() -> dict[str, str]:
    """The pinned inputs' hashes and the decomp commit; reads no input file."""
    from tools.rom.inputs import KAIZO_PATCH_SHA256, VANILLA_SHA1
    from tools.rom.patch import expected_patched
    from tools.rom.symbols import DECOMP_COMMIT

    exp = expected_patched()
    return {
        "vanilla_sha1": VANILLA_SHA1,
        "patch_sha256": KAIZO_PATCH_SHA256,
        "patched_sha1": exp["sha1"] if exp else "unknown",
        "decomp_commit": DECOMP_COMMIT,
    }


def _cell(text: object) -> str:
    return (
        str(text)
        .encode("ascii", "backslashreplace")
        .decode("ascii")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return [*out, ""]


def _index_ranges(indices: tuple[int, ...]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for i in sorted(indices):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def _members(f: FileChange, decomp: Path | None) -> str:
    if f.member_counts is None:
        return ""
    parts = []
    if f.member_counts[0] != f.member_counts[1]:
        parts.append(f"{f.member_counts[0]} -> {f.member_counts[1]} members")
    if not f.members_changed:
        if f.member_counts[0] == f.member_counts[1]:
            parts.append("container changed, no member changed")
        return "; ".join(parts)
    texts = []
    for lo, hi in _index_ranges(f.members_changed):
        text = str(lo) if lo == hi else f"{lo}-{hi}"
        if decomp is not None:
            a, b = member_label(f.path, lo, decomp), member_label(f.path, hi, decomp)
            if a and b:
                text += f" ({a})" if lo == hi else f" ({a} .. {b})"
        texts.append(text)
    parts.append(f"{len(f.members_changed)} changed: " + ", ".join(texts))
    return "; ".join(parts)


def _size(n: int | None) -> str:
    return "-" if n is None else str(n)


def _files_lines(diff: RomDiff, decomp: Path | None) -> list[str]:
    changed = [f for f in diff.files if f.status == "changed"]
    added = [f for f in diff.files if f.status == "added"]
    removed = [f for f in diff.files if f.status == "removed"]
    out = [f"## Changed files ({len(changed)})", ""]
    out += _table(
        ["path", "vanilla bytes", "patched bytes", "members"],
        [
            [f.path, _size(f.vanilla_size), _size(f.patched_size), _members(f, decomp)]
            for f in changed
        ],
    )
    out += [f"## Added files ({len(added)})", "", "Collapsed per folder.", ""]
    folders: dict[str, list[int]] = {}
    for f in added:
        entry = folders.setdefault(f.path.rpartition("/")[0] or "(root)", [0, 0])
        entry[0] += 1
        entry[1] += f.patched_size or 0
    out += _table(
        ["folder", "files", "bytes"], [[k, v[0], v[1]] for k, v in sorted(folders.items())]
    )
    out += [f"## Removed files ({len(removed)})", ""]
    out += (
        _table(["path", "vanilla bytes"], [[f.path, _size(f.vanilla_size)] for f in removed])
        if removed
        else ["None.", ""]
    )
    return out


def _code_lines(diff: RomDiff, classification: dict) -> list[str]:
    total = sum(diff.relocations.values())
    summary = (
        f"{len(diff.code)} changed ranges; {total} relocated units are counted apart "
        "(re-encoded only because what they point at moved). Rows with status `-` and 0 bytes "
        "changed are symbols with no changed range, only such re-encoded references to "
        "moved code or data: they are counted, not classified, and not stale-checked."
    )
    out = ["## Code", "", summary, ""]
    by_key: dict[tuple[str, str], list[CodeChange]] = {}
    for c in diff.code:
        by_key.setdefault(_key(c), []).append(c)
    keys = _changed_keys(diff)
    modules = sorted({m for m, _ in keys} | {m for m, _ in diff.relocations}, key=_module_order)
    for module in modules:
        out += [f"### {module}", ""]
        rows: list[list[object]] = []
        for m, name in (k for k in keys if k[0] == module):
            cs = by_key[(m, name)]
            nbytes = sum(max(c.end - c.start, c.patched_end - c.patched_start) for c in cs)
            status, note = classification.get((m, name), ("unknown", ""))
            reloc = diff.relocations.get((m, cs[0].symbol), 0) if cs[0].symbol else 0
            rows.append([name, len(cs), nbytes, reloc, status, note])
        listed = {r[0] for r in rows}
        for (m, sym), n in sorted(
            diff.relocations.items(), key=lambda kv: (kv[0][1] is None, kv[0][1] or "")
        ):
            if m == module and sym not in listed:
                rows.append([sym or "(outside symbols)", 0, 0, n, "-", "relocations only"])
        out += _table(["symbol", "ranges", "bytes changed", "relocations", "status", "note"], rows)
    return out


def _report_lines(diff, classification, provenance, decomp) -> list[str]:
    out = [
        "# Kaizo ROM diff",
        "",
        "Generated by `python -m tools.rom.diff`; names, addresses, sizes and counts only.",
        "Statuses come from `data/kaizo/rom_diff.toml`.",
        "",
        "## Provenance",
        "",
    ]
    out += _table(
        ["input", "value"],
        [
            ["vanilla SHA-1", provenance["vanilla_sha1"]],
            ["patch SHA-256", provenance["patch_sha256"]],
            ["patched SHA-1", provenance["patched_sha1"]],
            ["decomp commit", provenance["decomp_commit"]],
        ],
    )
    out += ["## Header changes", ""]
    out += (
        _table(
            ["field", "vanilla", "patched"], [[h.field, h.vanilla, h.patched] for h in diff.header]
        )
        if diff.header
        else ["None.", ""]
    )
    out += [
        "## Module sizes",
        "",
        "Modules whose bytes differ (ARM9 is the sum of its sections).",
        "",
    ]
    out += _table(
        ["module", "vanilla bytes", "patched bytes", "delta"],
        [
            [m, a, b, f"{b - a:+d}"]
            for m, (a, b) in sorted(diff.modules.items(), key=lambda kv: _module_order(kv[0]))
        ],
    )
    return (
        out
        + _files_lines(diff, decomp)
        + _parts_lines(diff)
        + _injected_lines(diff)
        + _code_lines(diff, classification)
    )


def _parts_lines(diff: RomDiff) -> list[str]:
    out = [
        "## Outside the file system",
        "",
        (
            "ARM9 overlay table (per entry), ARM9 footer, ARM7 binary, ARM7 overlay table and "
            "banner; differing fields only (binaries by size and SHA-1)."
        ),
        "",
    ]
    if not diff.parts:
        return [*out, "None.", ""]
    return out + _table(
        ["part", "field", "vanilla", "patched"],
        [[p.part, p.field, p.vanilla, p.patched] for p in diff.parts],
    )


def _call_rows(calls) -> list[list[object]]:
    return [
        [c.module, c.symbol or "(outside symbols)", f"0x{c.address:08X}", f"0x{c.target:08X}"]
        for c in calls
    ]


def _injected_lines(diff: RomDiff) -> list[str]:
    out = [
        "## Injected code",
        "",
        (
            "Code loaded from a file-system member to an address outside every module. Its callers "
            "are the Thumb BL/BLX inside listed code changes whose target lies in the region."
        ),
        "",
    ]
    if not diff.injected:
        out += ["None declared.", ""]
    for r in diff.injected:
        end = r.load_address + r.patched_size
        out += [
            f"### {r.path} member {r.member}",
            "",
            (
                f"{_size(r.vanilla_size)} -> {r.patched_size} bytes, loaded at "
                f"0x{r.load_address:08X}-0x{end:08X}; {len(r.calls)} calls from listed changes."
            ),
            "",
        ]
        out += _table(["module", "symbol", "BL at", "target"], _call_rows(r.calls))
    out += [f"### Unexplained far calls ({len(diff.far_calls)})", ""]
    out += (
        _table(["module", "symbol", "BL at", "target"], _call_rows(diff.far_calls))
        if diff.far_calls
        else ["None: every call outside the modules lands in a declared region.", ""]
    )
    return out


def write_report(
    diff: RomDiff,
    classification: dict,
    path: Path,
    provenance: dict[str, str] | None = None,
    decomp: Path | None = None,
) -> None:
    """Write the deterministic ASCII Markdown report; stale classification entries raise."""
    stale = stale_entries(diff, classification)
    if stale:
        raise ValueError(
            "classification entries matching no current change: "
            + ", ".join(f"{m} {s}" for m, s in stale)
        )
    lines = _report_lines(diff, classification, provenance or default_provenance(), decomp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines).rstrip("\n") + "\n").encode("ascii"))


# --- command line -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from tools.rom.symbols import DEFAULT_MAP

    ap = argparse.ArgumentParser(
        prog="python -m tools.rom.diff", description="Diff vanilla Platinum against the Kaizo ROM."
    )
    ap.add_argument("--symbols", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--decomp", type=Path, default=REPO_ROOT / "external" / "pokeplatinum")
    ap.add_argument(
        "--classification", type=Path, default=REPO_ROOT / "data" / "kaizo" / "rom_diff.toml"
    )
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "kaizo" / "rom_diff.md")
    ap.add_argument(
        "--init-classification",
        action="store_true",
        help="write an all-unknown classification; refuses to overwrite one",
    )
    args = ap.parse_args(argv)

    if args.init_classification and args.classification.exists():
        print(f"{args.classification} exists; not overwriting it", file=sys.stderr)
        return 1
    if not args.decomp.exists():
        print(
            f"decomp not found at {args.decomp}: the report's member labels come from it "
            "(clone it with tools/clone_decomp.sh, or pass --decomp)",
            file=sys.stderr,
        )
        return 1

    from tools.rom.inputs import VANILLA_SHA1, load_inputs, read_checked
    from tools.rom.nds import RomFiles
    from tools.rom.patch import kaizo_rom
    from tools.rom.symbols import SymbolMap

    inputs = load_inputs()
    vanilla = RomFiles.open(read_checked(inputs.vanilla_rom, VANILLA_SHA1, "sha1"))
    patched_path, _ = kaizo_rom(inputs)
    patched = RomFiles.open(patched_path.read_bytes())
    diff = diff_roms(vanilla, patched, SymbolMap.from_map(args.symbols), KAIZO_INJECTED_CODE)

    if args.init_classification:
        args.classification.parent.mkdir(parents=True, exist_ok=True)
        args.classification.write_text(
            classification_skeleton(diff), encoding="ascii", newline="\n"
        )
        print(f"wrote {args.classification}: {len(_changed_keys(diff))} entries")
        return 0
    if args.classification.exists():
        classification = load_classification(args.classification)
    else:
        print(
            f"{args.classification} not found; every change is reported as unknown "
            "(--init-classification writes one)",
            file=sys.stderr,
        )
        classification = {}
    write_report(diff, classification, args.out, decomp=args.decomp)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
