"""Synthetic ROMs for the ROM diff tests (``test_diff.py``, and Task 7's report tests).

``FakeRom`` has the ``RomFiles`` methods ``diff_roms`` uses, built from plain bytes, so
the diff is tested without ndspy or a real ROM.
"""

from __future__ import annotations

import struct

from tools.rom.symbols import Symbol, SymbolMap

BASE = 0x02200000  # overlay 3's RAM address
CODE = bytes(range(256))  # overlay 3's default code

SYMS = SymbolMap([
    Symbol("overlay3", BASE + 0x40, 0x20, "Foo", "a.o", ".text"),
    Symbol("overlay3", BASE + 0x60, 0x20, "Bar", "a.o", ".text"),
])


class FakeRom:
    """A ROM with the ``RomFiles`` interface, from in-memory files and code."""

    def __init__(self, files: dict, arm9: list[tuple[int, bytes]], ov3: bytes, header: bytes):
        self._files = files
        self._arm9 = arm9
        self._ov3 = ov3
        self._header = header

    @property
    def game_code(self) -> str:
        return self._header[0x0C:0x10].decode("ascii")

    def header(self) -> bytes:
        return self._header

    def paths(self) -> list[str]:
        return sorted(self._files)

    def file(self, path: str) -> bytes:
        f = self._files[path]
        if isinstance(f, list):  # a NARC: its magic, then any encoding of the members
            return b"NARC" + b"".join(struct.pack("<I", len(m)) + m for m in f)
        return f

    def narc(self, path: str) -> list[bytes]:
        f = self._files[path]
        if isinstance(f, list):
            return list(f)
        raise ValueError(f"{path}: not a readable NARC")

    def arm9(self) -> bytes:
        return self._arm9[0][1]

    def arm9_sections(self) -> list[tuple[int, bytes]]:
        return list(self._arm9)

    def overlay_ids(self) -> list[int]:
        return [3]

    def overlay(self, i: int) -> bytes:
        if i != 3:
            raise KeyError(f"no ARM9 overlay {i} in the ROM")
        return self._ov3

    def overlay_info(self, i: int) -> tuple[int, int, int]:
        return BASE, len(self.overlay(i)), 0


def _default_header() -> bytes:
    h = bytearray(0x200)
    h[0x0C:0x10] = b"CPUE"
    return bytes(h)


def fake(files=None, arm9=None, ov3=None, header=None) -> FakeRom:
    """A ``FakeRom``; ``files`` maps a path to bytes, or to a member list for a NARC."""
    return FakeRom(
        files or {},
        arm9 if arm9 is not None else [(0x02000000, b"")],
        CODE if ov3 is None else ov3,
        header if header is not None else _default_header(),
    )


def poke(data: bytes, off: int, new: bytes) -> bytes:
    """``data`` with ``new`` written at ``off``."""
    return data[:off] + new + data[off + len(new):]


# Relocation fixture: 0x100 bytes of overlay 3 made of distinct filler words that are
# neither module addresses nor branch encodings. Grown [0x40, 0x80) gets 8 bytes inserted
# at 0x60; Caller [0x00, 0x20) has a Thumb BL at 0x10 to Callee [0xC0, 0xE0), after the
# insertion, so it is re-encoded 8 bytes further; Holder [0x80, 0xA0) has a literal word
# at 0x90 holding Callee's Thumb address (bit 0 set), which moves by 8 too.


def thumb_bl(pc: int, target: int) -> bytes:
    off = target - (pc + 4)
    hi, lo = (off >> 12) & 0x7FF, (off >> 1) & 0x7FF
    return struct.pack("<HH", 0xF000 | hi, 0xF800 | lo)


def _filler() -> bytes:
    return b"".join(struct.pack("<I", 0x11000000 + i) for i in range(0x40))


RELOC_SYMS = SymbolMap([
    Symbol("overlay3", BASE + 0x00, 0x20, "Caller", "b.o", ".text"),
    Symbol("overlay3", BASE + 0x40, 0x40, "Grown", "b.o", ".text"),
    Symbol("overlay3", BASE + 0x80, 0x20, "Holder", "b.o", ".rodata"),
    Symbol("overlay3", BASE + 0xC0, 0x20, "Callee", "b.o", ".text"),
])


def reloc_van() -> bytes:
    code = poke(_filler(), 0x10, thumb_bl(BASE + 0x10, BASE + 0xC0))
    return poke(code, 0x90, struct.pack("<I", BASE + 0xC0 + 1))


def reloc_pat() -> bytes:
    van = reloc_van()
    code = van[:0x60] + b"\xee" * 8 + van[0x60:]
    code = poke(code, 0x10, thumb_bl(BASE + 0x10, BASE + 0xC8))
    return poke(code, 0x98, struct.pack("<I", BASE + 0xC8 + 1))


RELOC_VAN = fake(ov3=reloc_van())
RELOC_PAT = fake(ov3=reloc_pat())
