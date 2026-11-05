"""Tables that live in the ARM9/overlay code rather than in a NARC (Ruling 13)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError, symbol_bytes

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

TM_SYMBOL = "sTMHMMoves"
TM_COUNT = 100  # NUM_TMS (92) + NUM_HMS (8)


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
