"""Level-up learnsets from ``poketool/personal/wotbl.narc``."""

from __future__ import annotations

import re
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
SPECIES_H = "include/struct_defs/species.h"


def max_learnset_entries(decomp: Path) -> int:
    """``MAX_LEARNSET_ENTRIES`` (``include/struct_defs/species.h:10``): the packer drops
    entries past it (``speciesproc.c:478-483``), and the game loads a member into a
    ``SpeciesLearnset``, ``ALIGN_4 entries[MAX_LEARNSET_ENTRIES + 1]`` (``species.h:73-75``,
    ``Heap_Alloc(..., sizeof(SpeciesLearnset))`` at ``src/pokemon.c:3822,3938,4131``)."""
    path = Path(decomp) / SPECIES_H
    m = re.search(
        r"^#define MAX_LEARNSET_ENTRIES\s+(\d+)\s*$", path.read_text("utf-8"), re.MULTILINE
    )
    if not m:
        raise ReaderError(f"{path}: no MAX_LEARNSET_ENTRIES")
    return int(m.group(1))


def read_learnsets(rom: RomFiles, decomp: Path) -> dict[str, list[list[int]]]:
    """``{species id: [[level, move], ...]}`` for ids ``0..495``.

    Each member is u16 entries ``move | level << 9`` up to ``0xFFFF``, then at most one
    zero u16 of padding (``speciesproc.c:472-494`` pads to 4 bytes). An odd length, any
    other tail, or more entries than the game's buffer holds with its terminator raises.

    The buffer bound is ``MAX_LEARNSET_ENTRIES + 1`` entries: ``sizeof(SpeciesLearnset)`` is
    44 bytes (21 u16 slots aligned to 4), so 21 entries and the terminator fit. Vanilla
    never exceeds ``MAX_LEARNSET_ENTRIES`` (20; pinned by the tests). Kaizo (K1 final fix
    wave, measured) has one 21-entry species (member 359, 46 bytes with its padding) and 48
    members whose length is 2 mod 4, so neither "at most 20 entries" nor "a multiple of 4
    bytes" holds there; K2b must confirm how the patched game loads member 359.
    """
    members = rom.narc(WOTBL_NARC)
    if len(members) < SPECIES_COUNT:
        raise ReaderError(f"{WOTBL_NARC}: {len(members)} members, need {SPECIES_COUNT}")
    n_moves = len(_constants(decomp, "moves", "MOVE_"))
    max_entries = max_learnset_entries(decomp) + 1  # the buffer's slots, terminator aside
    out = {}
    for sid in range(SPECIES_COUNT):
        m = members[sid]
        where = f"{WOTBL_NARC} member {sid}"
        if len(m) % 2:
            raise ReaderError(f"{where}: {len(m)} bytes, an odd length")
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
        if len(rows) > max_entries:
            raise ReaderError(
                f"{where}: {len(rows)} entries, more than the {max_entries} the game's "
                "SpeciesLearnset buffer holds with its terminator"
            )
        tail = m[off + 2 :]
        if len(tail) > 2 or any(tail):
            raise ReaderError(
                f"{where} offset {off + 2}: {len(tail)} unaccounted bytes after the terminator"
            )
        out[str(sid)] = rows
    return out
