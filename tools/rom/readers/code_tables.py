"""Tables that live in the ARM9/overlay code rather than in a NARC (Ruling 13)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError, symbol_bytes
from tools.rom.readers.personal import SPECIES_COUNT

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

TM_SYMBOL = "sTMHMMoves"
TM_COUNT = 100  # NUM_TMS (92) + NUM_HMS (8)
EGG_SYMBOL = "sEggMoves"
EGG_OFFSET = 20000  # EGG_MOVES_SPECIES_OFFSET
EGG_TERMINATOR = 0xFFFF  # EGG_MOVES_TERMINATOR
MAX_EGG_MOVES = 16  # include/constants/daycare.h:19


def tm_moves(rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None) -> list[int]:
    """Move id of ITEM_TM01..ITEM_HM08, from ``sTMHMMoves``: 100 u16 in
    ``res/items/item_tm_move_map.h`` (template ``tools/dataproc/data/item_tm_move_map.h.template:6``,
    included at ``src/item.c:17``, read by ``Item_MoveForTMHM``, ``src/item.c:296-304``).
    Each must be a real move: ``0 < m < len(moves)``. ``decomp`` defaults to
    ``external/pokeplatinum``."""
    if decomp is None:
        from tools.rom.inputs import REPO_ROOT

        decomp = REPO_ROOT / "external" / "pokeplatinum"
    n_moves = len(_constants(Path(decomp), "moves", "MOVE_"))
    data = symbol_bytes(rom, symbols, TM_SYMBOL)
    if len(data) != 2 * TM_COUNT:
        raise ReaderError(f"{TM_SYMBOL}: {len(data)} bytes, expected {2 * TM_COUNT}")
    out = list(struct.unpack(f"<{TM_COUNT}H", data))
    for i, m in enumerate(out):
        if not 0 < m < n_moves:
            raise ReaderError(f"{TM_SYMBOL}[{i}]: move {m} is out of range (1..{n_moves - 1})")
    return out


def egg_moves(
    rom: RomFiles, symbols: SymbolMap, decomp: Path | None = None
) -> dict[int, list[int]]:
    """Species id -> egg moves from ``sEggMoves``: u16s, each species' marker
    ``20000 + species`` followed by its moves, the whole table ended by ``0xFFFF``
    (``tools/dataproc/data/species_egg_moves.h.template:7-8``; ``EmitEggMoves``,
    ``speciesproc.c:681-692``). ``LoadSpeciesEggMoves`` (``src/overlay005/daycare.c:445``)
    scans for the marker and reads up to 16 values until one exceeds 20000.

    Shape checks: a whole number of u16s ending in the terminator and containing it nowhere
    else, a marker for a real species (1..495) at most once, every move a real move
    (``0 < m < len(moves)``), and at most 16 moves per species. A species with no moves
    after its marker is not recorded, as the decomp side does."""
    if decomp is None:
        from tools.rom.inputs import REPO_ROOT

        decomp = REPO_ROOT / "external" / "pokeplatinum"
    n_moves = len(_constants(Path(decomp), "moves", "MOVE_"))
    data = symbol_bytes(rom, symbols, EGG_SYMBOL)
    if len(data) % 2 or len(data) < 2:
        raise ReaderError(f"{EGG_SYMBOL}: {len(data)} bytes is not a whole number of u16")
    vals = struct.unpack(f"<{len(data) // 2}H", data)
    if vals[-1] != EGG_TERMINATOR or EGG_TERMINATOR in vals[:-1]:
        raise ReaderError(f"{EGG_SYMBOL}: not terminated by a single 0xffff at the end")
    out: dict[int, list[int]] = {}
    cur: list[int] | None = None
    for i, v in enumerate(vals[:-1]):
        if v > EGG_OFFSET:
            species = v - EGG_OFFSET
            if not 0 < species < SPECIES_COUNT:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: marker for species {species}")
            if species in out:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: a second marker for species {species}")
            cur = out[species] = []
        else:
            if cur is None:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: move {v} before any species marker")
            if not 0 < v < n_moves:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: move {v} is out of range (1..{n_moves - 1})")
            cur.append(v)
            if len(cur) > MAX_EGG_MOVES:
                raise ReaderError(f"{EGG_SYMBOL}[{i}]: more than {MAX_EGG_MOVES} egg moves")
    return {k: v for k, v in out.items() if v}
