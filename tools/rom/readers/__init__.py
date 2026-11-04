"""Shared helpers for the ROM table readers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap


class ReaderError(RuntimeError):
    """The ROM holds bytes a reader cannot account for."""


def symbol_bytes(rom: RomFiles, symbols: SymbolMap, name: str) -> bytes:
    """The bytes of linker-map symbol ``name`` from its module's image.

    ARM9: the section whose RAM range contains the symbol; overlay ``N``: overlay ``N``
    at ``address - overlay_info(N)[0]``. A symbol outside the stored bytes (BSS) raises.
    """
    sym = symbols.find(name)
    lo, hi = sym.address, sym.address + sym.size
    if sym.module == "arm9":
        for ram, data in rom.arm9_sections():
            if ram <= lo and hi <= ram + len(data):
                return data[lo - ram : hi - ram]
        raise ReaderError(
            f"symbol {name} ({sym.address:#x}+{sym.size}) is outside the ARM9 sections"
        )
    if not sym.module.startswith("overlay"):
        raise ReaderError(f"symbol {name}: unknown module {sym.module!r}")
    n = int(sym.module[len("overlay") :])
    ram = rom.overlay_info(n)[0]
    data = rom.overlay(n)
    if lo < ram or hi > ram + len(data):
        raise ReaderError(
            f"symbol {name} ({sym.address:#x}+{sym.size}) is outside overlay {n} "
            f"({ram:#x}+{len(data)}); a BSS symbol has no stored bytes"
        )
    return data[lo - ram : hi - ram]


def fixed_members(members: list[bytes], size: int, path: str) -> list[bytes]:
    """``members`` unchanged once every one is exactly ``size`` bytes, else raise."""
    for i, m in enumerate(members):
        if len(m) != size:
            raise ReaderError(f"{path}: member {i}: expected {size} bytes, got {len(m)}")
    return members
