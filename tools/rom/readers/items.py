"""Item hold effects from ``itemtool/itemdata/pl_item_data.narc`` and the TM table."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError, symbol_bytes
from tools.rom.readers.code_tables import tm_moves
from tools.rom.readers.text import message_bank

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

ITEM_DATA_NARC = "itemtool/itemdata/pl_item_data.narc"
# ItemData (include/item.h:56-77) is ALIGN_4, so sizeof is 36; the game reads whole members
# (src/item.c:118), so a member may be the 34 bytes of fields (every vanilla member) or the
# 36 with two bytes of alignment padding, which must be zero (some Kaizo members).
ITEM_DATA_SIZES = (34, 36)
ITEM_FIELDS_SIZE = 34
ARCHIVE_SYMBOL = "sItemArchiveIDs"  # src/item.c:48-57; 4 x u16: data, icon, palette, gen3
ARCHIVE_ENTRY = 8
HOLD_EFFECT_OFFSET = 2  # include/item.h:57-59
HOLD_PARAM_OFFSET = 3
# ItemData.partyUseParam starts at byte 14, after padding_0D (include/item.h:72-75); its u8
# hpRestored follows seven bytes of bit fields and six s8 EVs (include/item.h:9-48): byte 27.
HP_RESTORED_OFFSET = 27


def read_items(
    rom: RomFiles, decomp: Path, symbols: SymbolMap, charmap: dict[int, str]
) -> list[dict]:
    """``items.json`` rows (Ruling 10). Unused ids keep the extractor's constant name,
    ``HOLD_EFFECT_NONE`` and 0; every other id reads the data member named by
    ``sItemArchiveIDs[id].dataID`` (item 0 uses member 0, ``src/item.c:48-57``)."""
    ids = _constants(decomp, "items", "ITEM_")
    tm01, hm08 = ids.index("ITEM_TM01"), ids.index("ITEM_HM08")
    teaches = tm_moves(rom, symbols, decomp)
    if hm08 - tm01 + 1 != len(teaches):
        raise ReaderError(f"ITEM_TM01..ITEM_HM08 span {hm08 - tm01 + 1} ids, not {len(teaches)}")
    holds = _constants(decomp, "item_hold_effects", "HOLD_EFFECT_")
    none = holds.index("HOLD_EFFECT_NONE")
    members = rom.narc(ITEM_DATA_NARC)
    for i, m in enumerate(members):
        if len(m) not in ITEM_DATA_SIZES:
            raise ReaderError(
                f"{ITEM_DATA_NARC}: member {i}: expected {ITEM_DATA_SIZES} bytes, got {len(m)}"
            )
        if any(m[ITEM_FIELDS_SIZE:]):
            raise ReaderError(
                f"{ITEM_DATA_NARC} member {i} offset {ITEM_FIELDS_SIZE}: nonzero padding"
            )
    table = symbol_bytes(rom, symbols, ARCHIVE_SYMBOL)
    if len(table) != ARCHIVE_ENTRY * len(ids):
        raise ReaderError(
            f"{ARCHIVE_SYMBOL}: {len(table)} bytes, expected {ARCHIVE_ENTRY * len(ids)}"
        )
    names = message_bank(rom, decomp, "TEXT_BANK_ITEM_NAMES", charmap)
    if len(names) < len(ids):
        raise ReaderError(f"TEXT_BANK_ITEM_NAMES: {len(names)} entries, need {len(ids)}")
    out = []
    for item, const in enumerate(ids):
        if const.startswith("ITEM_UNUSED_"):
            out.append(
                {
                    "name": const,
                    "hold_effect": none,
                    "hold_param": 0,
                    "teaches_move": None,
                    "hp_restored": 0,
                }
            )
            continue
        data_id = 0 if item == 0 else struct.unpack_from("<H", table, item * ARCHIVE_ENTRY)[0]
        if data_id >= len(members):
            raise ReaderError(
                f"{ARCHIVE_SYMBOL}[{item}] ({const}): data member {data_id} but "
                f"{ITEM_DATA_NARC} has {len(members)}"
            )
        effect = members[data_id][HOLD_EFFECT_OFFSET]
        if effect >= len(holds):
            raise ReaderError(
                f"{ITEM_DATA_NARC} member {data_id} offset {HOLD_EFFECT_OFFSET}: "
                f"hold effect {effect} is out of range"
            )
        out.append(
            {
                "name": names[item],
                "hold_effect": effect,
                "hold_param": members[data_id][HOLD_PARAM_OFFSET],
                # sTMHMMoves[item - ITEM_TM01] (src/item.c:296-304); null for every other item.
                "teaches_move": teaches[item - tm01] if tm01 <= item <= hm08 else None,
                "hp_restored": members[data_id][HP_RESTORED_OFFSET],
            }
        )
    return out
