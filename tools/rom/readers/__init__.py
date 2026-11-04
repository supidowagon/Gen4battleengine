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

    The map's addresses hold only while the ROM's image is laid out as the vanilla build's:
    the image must start where the map's extent for it starts (``SymbolMap.extent``:
    ``SDK_STATIC``/``SDK_AUTOLOAD.<name>``/``SDK_OVERLAY.<name>`` ``START`` and ``SIZE``)
    and have exactly that many stored bytes, else this raises. An image of another size
    has had code or data inserted or removed, so a symbol may sit elsewhere: on Kaizo,
    overlay 16 is 32 bytes longer and ``sTypeMatchupMultipliers`` is 28 bytes further on
    (K1 final review). A caller that reads a moved symbol must map its address itself.
    """
    sym = symbols.find(name)
    lo, hi = sym.address, sym.address + sym.size
    if sym.module == "arm9":
        for ram, data in rom.arm9_sections():
            if ram <= lo and hi <= ram + len(data):
                _check_extent(symbols, name, "ARM9 section", sym.module, ram, len(data))
                return data[lo - ram : hi - ram]
        raise ReaderError(
            f"symbol {name} ({sym.address:#x}+{sym.size}) is outside the ARM9 sections"
        )
    if not sym.module.startswith("overlay"):
        raise ReaderError(f"symbol {name}: unknown module {sym.module!r}")
    n = int(sym.module[len("overlay") :])
    ram = rom.overlay_info(n)[0]
    data = rom.overlay(n)
    _check_extent(symbols, name, f"overlay {n}", sym.module, ram, len(data))
    if lo < ram or hi > ram + len(data):
        raise ReaderError(
            f"symbol {name} ({sym.address:#x}+{sym.size}) is outside overlay {n} "
            f"({ram:#x}+{len(data)}); a BSS symbol has no stored bytes"
        )
    return data[lo - ram : hi - ram]


def _check_extent(
    symbols: SymbolMap, name: str, what: str, module: str, ram: int, size: int
) -> None:
    want = symbols.extent(module, ram)
    if want is None:
        raise ReaderError(
            f"symbol {name}: the linker map gives no extent for {what} at {ram:#x}; "
            "its layout cannot be checked against the vanilla build"
        )
    if want != size:
        raise ReaderError(
            f"symbol {name}: {what} at {ram:#x} is {size:#x} bytes in this ROM but "
            f"{want:#x} in the linker map's build; its symbols may have moved, so their "
            "vanilla addresses are not read (pass an explicit address mapping)"
        )


def fixed_members(members: list[bytes], size: int, path: str) -> list[bytes]:
    """``members`` unchanged once every one is exactly ``size`` bytes, else raise."""
    for i, m in enumerate(members):
        if len(m) != size:
            raise ReaderError(f"{path}: member {i}: expected {size} bytes, got {len(m)}")
    return members
