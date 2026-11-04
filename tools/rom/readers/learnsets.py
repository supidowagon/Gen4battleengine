"""Level-up learnsets from ``poketool/personal/wotbl.narc``."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError
from tools.rom.readers.personal import SPECIES_COUNT

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles

WOTBL_NARC = "poketool/personal/wotbl.narc"
TERMINATOR = 0xFFFF  # move 0x1FF, level 0x7F (tools/dataproc/src/speciesproc.c:490)


def read_learnsets(rom: RomFiles, decomp: Path) -> dict[str, list[list[int]]]:
    """``{species id: [[level, move], ...]}`` for ids ``0..495``.

    Each member is u16 entries ``move | level << 9`` up to ``0xFFFF``, padded to a multiple
    of 4 bytes with zeros (``speciesproc.c:472-494``).
    """
    members = rom.narc(WOTBL_NARC)
    if len(members) < SPECIES_COUNT:
        raise ReaderError(f"{WOTBL_NARC}: {len(members)} members, need {SPECIES_COUNT}")
    n_moves = len(_constants(decomp, "moves", "MOVE_"))
    out = {}
    for sid in range(SPECIES_COUNT):
        m = members[sid]
        where = f"{WOTBL_NARC} member {sid}"
        rows = []
        off = 0
        while True:
            if off + 2 > len(m):
                raise ReaderError(f"{where} offset {off}: no 0xffff terminator")
            (v,) = struct.unpack_from("<H", m, off)
            if v == TERMINATOR:
                break
            move, level = v & 0x1FF, v >> 9
            if move >= n_moves:
                raise ReaderError(f"{where} offset {off}: move {move} is out of range")
            rows.append([level, move])
            off += 2
        tail = m[off + 2 :]
        if len(tail) > 2 or any(tail):
            raise ReaderError(
                f"{where} offset {off + 2}: {len(tail)} unaccounted bytes after the terminator"
            )
        out[str(sid)] = rows
    return out
