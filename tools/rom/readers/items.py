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
# u16 price, holdEffect, effectParam, pluckEffect, flingEffect, flingPower (include/item.h:56-62)
FLING_POWER_OFFSET = 6
# ItemData.partyUseParam starts at byte 14, after padding_0D (include/item.h:72-75); its u8
# hpRestored follows seven bytes of bit fields and six s8 EVs (include/item.h:9-48): byte 27.
HP_RESTORED_OFFSET = 27
PARTY_USE_OFFSET = 12  # ItemData.partyUse (include/item.h:71)
PARTY_PARAM_OFFSET = 14  # the union of dummy and partyUseParam (include/item.h:73-76)
# The ItemPartyParam fields ``Item_Get`` returns that BattleSystem_UseBagItem
# (src/battle/battle_system.c:435-758) and TrainerAI_ShouldUseItem (trainer_ai.c:4056-4197)
# read, as (name, byte, low bit, bits, signed) within partyUseParam: seven bytes of u8 bit
# fields, LSB first, a field that does not fit starting the next byte (include/item.h:9-41),
# then six s8 EVs (21-26 of ItemData), u8 hpRestored and ppRestored, s8 friendship amounts.
# ``nuzline.engine.ids.ItemParam`` names the same columns in this order.
PARTY_PARAM_FIELDS = (
    ("heal_sleep", 0, 0, 1, False),
    ("heal_poison", 0, 1, 1, False),
    ("heal_burn", 0, 2, 1, False),
    ("heal_freeze", 0, 3, 1, False),
    ("heal_paralysis", 0, 4, 1, False),
    ("heal_confusion", 0, 5, 1, False),
    ("heal_attract", 0, 6, 1, False),
    ("guard_spec", 0, 7, 1, False),
    ("revive", 1, 0, 1, False),
    ("atk_stages", 1, 4, 4, False),
    ("def_stages", 2, 0, 4, False),
    ("spatk_stages", 2, 4, 4, False),
    ("spdef_stages", 3, 0, 4, False),
    ("speed_stages", 3, 4, 4, False),
    ("acc_stages", 4, 0, 4, False),
    ("crit_stages", 4, 4, 2, False),
    ("pp_restore", 5, 0, 1, False),
    ("pp_restore_all", 5, 1, 1, False),
    ("hp_restore", 5, 2, 1, False),
    ("give_friendship_low", 6, 1, 1, False),
    ("give_friendship_med", 6, 2, 1, False),
    ("give_friendship_high", 6, 3, 1, False),
    ("hp_restored", 13, 0, 8, False),
    ("pp_restored", 14, 0, 8, False),
    ("friendship_low", 15, 0, 8, True),
    ("friendship_med", 16, 0, 8, True),
    ("friendship_high", 17, 0, 8, True),
)
PARTY_PARAM_BYTES = 20  # sizeof(ItemPartyParam), include/item.h:8-54


def party_params(party_use: int, union: bytes) -> list[int]:
    """``Item_Get`` (src/item.c:153-198) of each ``PARTY_PARAM_FIELDS`` param: with
    ``partyUse`` TRUE the ``ItemPartyParam`` field (``ItemPartyParam_Get``, :200-294); with
    FALSE ``itemData->dummy``, the union's first byte, for every such param (:186-191, e.g.
    a Poke Ball's ball index); any other ``partyUse`` byte falls through to 0 (:192-197).
    ``union`` is ItemData's 20 bytes from offset 14."""
    fields = {}
    for name, byte, low, bits, signed in PARTY_PARAM_FIELDS:
        v = (union[byte] >> low) & ((1 << bits) - 1)
        fields[name] = v - 256 if signed and v >= 128 else v
    return project_party_params(party_use, union[0], fields)


def project_party_params(party_use: int, dummy: int, fields: dict[str, int]) -> list[int]:
    """``Item_Get``'s value of each ``PARTY_PARAM_FIELDS`` param from the item's ``partyUse``
    byte, ``dummy`` byte and ``ItemPartyParam`` field values (``party_params``)."""
    if party_use == 0:
        return [dummy] * len(PARTY_PARAM_FIELDS)
    if party_use != 1:
        return [0] * len(PARTY_PARAM_FIELDS)
    return [fields[name] for name, *_ in PARTY_PARAM_FIELDS]


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
                    "fling_power": 0,
                    "party_params": [0] * len(PARTY_PARAM_FIELDS),
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
                "fling_power": members[data_id][FLING_POWER_OFFSET],
                "party_params": party_params(
                    members[data_id][PARTY_USE_OFFSET],
                    members[data_id][PARTY_PARAM_OFFSET : PARTY_PARAM_OFFSET + PARTY_PARAM_BYTES],
                ),
            }
        )
    return out
