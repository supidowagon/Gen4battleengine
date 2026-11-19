"""Platinum (US) save layout: file size, the two copies, block sizes and the offsets of
everything ``load_save`` reads (spec §1-3). Derived from the decomp, not from PKHeX.
"""

from __future__ import annotations

__all__ = [
    "BADGES",
    "BAG",
    "BLOCK_IDS",
    "BOXES_OFFSET",
    "BOXES_SIZE",
    "BOX_COUNT",
    "BOX_MONS",
    "BOX_MON_SIZE",
    "BOX_SLOTS",
    "CHATOT_CRY",
    "CHATOT_CRY_SIZE",
    "CHATOT_PARAM_SAMPLE",
    "COPY_OFFSETS",
    "FOOTER_SIZE",
    "MAP_HEADER",
    "NORMAL_PAGES",
    "NORMAL_SIZE",
    "NUM_VARS",
    "OVERWORLD_WEATHER",
    "PARTY_COUNT",
    "PARTY_MAX",
    "PARTY_MONS",
    "PARTY_MON_SIZE",
    "POCKETS",
    "SAVE_SIZE",
    "SECTOR_SIZE",
    "SIGNATURE",
    "TRAINER_GENDER",
    "TRAINER_ID",
    "TRAINER_NAME",
    "TRAINER_NAME_LEN",
    "VARS",
    "VARS_START",
    "VAR_PLAYER_STARTER_INDEX",
    "page_body_size",
    "page_offset",
]

# --- container -------------------------------------------------------------------------
SAVE_SIZE = 0x80000  # 512 KiB flash image
SECTOR_SIZE = 0x1000  # SAVE_SECTOR_SIZE, include/constants/savedata/save_table.h
# PRIMARY_SECTOR_START = 0, BACKUP_SECTOR_START = 64 (include/constants/savedata/savedata.h);
# SaveData_SaveOffset, src/savedata.c:299.
COPY_OFFSETS = (0 * SECTOR_SIZE, 64 * SECTOR_SIZE)

# --- blocks (SaveBlockInfo_Init, src/savedata.c:852: offsets accumulate block sizes) ------
# Block size = sum of page sizes, each rounded as ``s += 4 - s % 4; s += 4``
# (SaveTableEntry_BodySize, src/savedata.c:813), plus the 20-byte footer.
FOOTER_SIZE = 20  # SaveBlockFooter, include/savedata.h:8 (u32 x4, u8 id, u16 checksum @18)
SIGNATURE = 0x20060623  # SECTOR_SIGNATURE
NORMAL_SIZE = 0xCF2C  # SAVE_BLOCK_ID_NORMAL: pages SYSTEM .. WIFI_PLAZA_QUESTIONS + footer
BOXES_OFFSET = NORMAL_SIZE  # boxes block follows the normal block, not sector-aligned
# PCBoxes (include/pc_boxes.h:17) fields = 4 + 18*30*0x88 + 18*20*2 + 18 + 1 = 0x121C7
# bytes; sizeof pads it to the u32 alignment: 0x121C8 (PCBoxes_SaveSize returns sizeof).
# Body size: 0x121C8 + (4 - 0x121C8 % 4) + 4 = 0x121D0; + footer 20 = 0x121E4.
BOXES_SIZE = 0x121E4
BLOCK_IDS = {"normal": 0, "boxes": 1}  # enum SaveBlockID

# --- normal block (SavePageInfo_Init, src/savedata.c:826) --------------------------------
# SystemData page 0x00 (0x5C rounded to 0x64), PlayerSave page 0x64
# (include/save_player.h:9: u16 options + 2 padding, then TrainerInfo).
TRAINER_NAME = 0x64 + 0x04  # TrainerInfo.name, include/trainer_info.h:9
TRAINER_NAME_LEN = 8  # charcode_t name[TRAINER_NAME_LEN + 1], TRAINER_NAME_LEN = 7
TRAINER_ID = TRAINER_NAME + 2 * TRAINER_NAME_LEN  # 0x78, u32: low = TID, high = SID
# TrainerInfo (include/trainer_info.h:9-14): name, u32 id, u32 money, u8 gender, u8 language.
TRAINER_GENDER = TRAINER_ID + 4 + 4  # 0x80: after id and money
BADGES = TRAINER_ID + 4 + 4 + 1 + 1  # 0x82: after id, money, gender, language
# Party page 0x98 (PlayerSave 0x2C rounded to 0x34): int capacity, int currentCount, mons.
PARTY_COUNT = 0x98 + 4  # 0x9C
PARTY_MONS = 0x98 + 8  # 0xA0
PARTY_MON_SIZE = 0xEC  # struct Pokemon (BoxPokemon 0x88 + party extension 0x64)
PARTY_MAX = 6  # MAX_PARTY_SIZE
# Bag page: 0x98 + (8 + 6*0xEC = 0x590 rounded to 0x598) = 0x630.
BAG = 0x630
# Pockets of BagItem {u16 item, u16 quantity} in struct Bag order (include/bag.h:9-16).
POCKETS: tuple[tuple[str, int], ...] = (
    ("items", 165),
    ("key_items", 50),
    ("tms_hms", 100),
    ("mail", 12),
    ("medicine", 40),
    ("berries", 64),
    ("poke_balls", 15),
    ("battle_items", 30),
)
# VarsFlags page (SAVE_TABLE_ENTRY_VARS_FLAGS, src/savedata/save_table.c:53) follows the
# bag: struct Bag (include/bag.h:35) is the pockets above (476 BagItems, 0x770) plus
# u32 registeredItem = 0x774, rounded to 0x77C. VarsFlags.vars (include/vars_flags.h:12,
# u16 vars[NUM_VARS]) is the page's first member: 0x630 + 0x77C = 0xDAC.
VARS = 0xDAC
# VAR_PLAYER_STARTER = 0x4030 - VARS_START 0x4000 (VarsFlags_GetVarAddress,
# src/vars_flags.c:65). generated/vars_flags.txt:4107-4159 counted as C enum values (alias
# lines such as MAP_LOCAL_VARS_END = VAR_MAP_LOCAL_0x1F do not advance); the next line,
# VAR_UNUSED_0x4031, confirms it. Holds the starter's species id, 0 before the choice.
VAR_PLAYER_STARTER_INDEX = 48
# NUM_VARS = VARS_END - VARS_START (include/constants/savedata/vars_flags.h:6): VARS_END is
# 0x4120 (generated/vars_flags.txt:4399, counted as above), so the page holds 288 vars.
NUM_VARS = 288
VARS_START = 0x4000  # the first saved variable's id (generated/vars_flags.txt: VARS_START)

# --- the normal block's pages (gSaveTable, src/savedata/save_table.c:49-86) -------------
# Each page's ``sizeFunc`` result in table order (SavePageInfo_Init, src/savedata.c:826-850:
# ``location`` accumulates ``SaveTableEntry_BodySize``, :813-823). Every size function returns
# ``sizeof`` of its struct (PalPad, RecordMixedRNG and Journal times their entry count;
# Ribbons RIBBONS_SAVE_SIZE, include/ribbon_save_data.h:6; Email sizeof(UnkStruct_02030CEC)),
# taken under the game's ABI: 4-byte alignment of s64 (SystemData's own offsets,
# padding_49 / unused_50, include/system_data.h), enums as int (meson.build:24 ``-enum int``).
# WiFiList's DWC types are not in the decomp; its own field name ``unk_1C0`` puts its third
# member at 0x1C0, after which 32 x 0x38 (UnkStruct_0202B370_sub1, the ``unk_36`` field + 2):
# 0x8C0. The pages sum to NORMAL_SIZE (with the footer) exactly, and the per-page CRC16 the game
# keeps in the last 4 bytes of some pages (SaveData_SetChecksum, src/savedata.c:1349-1362: TV
# broadcast, rankings, Wi-Fi history, mystery gift, contest records, easy-chat words, email)
# matches at these offsets in real saves (tests/saves/test_save.py).
NORMAL_PAGES: tuple[tuple[str, int], ...] = (
    ("SYSTEM", 0x5C),
    ("PLAYER", 0x2C),
    ("PARTY", 0x590),
    ("BAG", 0x774),
    ("VARS_FLAGS", 0x3AC),
    ("POKETCH", 0x118),
    ("FIELD_PLAYER_STATE", 0xA0),
    ("POKEDEX", 0x324),
    ("DAYCARE", 0x1E0),
    ("PAL_PAD", 0x880),
    ("MISC", 0x78C),
    ("FIELD_OVERWORLD_STATE", 0x1400),
    ("UNDERGROUND", 0x9B4),
    ("REGULATION_BATTLES", 0x20),
    ("IMAGE_CLIPS", 0x834),
    ("MAILBOX", 0x460),
    ("POFFINS", 0x320),
    ("RECORD_MIXED_RNG", 0x108),
    ("JOURNAL", 0x460),
    ("TRAINER_CASE", 0x620),
    ("GAME_RECORDS", 0x1BC),
    ("SEAL_CASE", 0x170),
    ("CHATOT", 0x3EC),
    ("FRONTIER", 0x1628),
    ("RIBBONS", 0xE),
    ("ENCOUNTERS", 0x154),
    ("GLOBAL_TRADE", 0xF8),
    ("TV_BROADCAST", 0xBC8),
    ("RANKINGS", 0xEA0),
    ("WIFI_LIST", 0x8C0),
    ("WIFI_HISTORY", 0xFF8),
    ("MYSTERY_GIFT", 0x1328),
    ("PAL_PARK_TRANSFER", 0x688),
    ("LINK_CONTEST_RECORDS", 0x28),
    ("UNLOCKED_EASY_CHAT_WORDS", 0x8),
    ("EMAIL", 0x40),
    ("WIFI_PLAZA_QUESTIONS", 0x8),
)


def page_body_size(size: int) -> int:
    """``SaveTableEntry_BodySize`` (src/savedata.c:813-823): ``s += 4 - s % 4; s += 4``."""
    size += 4 - size % 4
    return size + 4


def page_offset(name: str) -> int:
    """The offset of page ``name`` (``NORMAL_PAGES``) in the normal block."""
    offset = 0
    for page, size in NORMAL_PAGES:
        if page == name:
            return offset
        offset += page_body_size(size)
    raise KeyError(name)


# FieldOverworldState (include/field_overworld_state.h:13-29) is the FIELD_PLAYER_STATE page
# (save_table.c:55: FieldOverworldState_Size = sizeof, src/field_overworld_state.c:17-20).
# Its first member is the player's Location (FieldOverworldState_GetPlayerLocation, :42-45;
# the FieldSystem's ``location``, src/field_system.c:163), whose first member is the enum
# (int) mapHeaderID (include/location.h:18-24). After the five Locations (5 ints each) and
# u16 bgm comes u16 weather, the overworld weather a trainer battle starts with
# (field_battle_data_transfer.c:267); Continue keeps it (field_map_change.c:521-525).
MAP_HEADER = page_offset("FIELD_PLAYER_STATE")
OVERWORLD_WEATHER = MAP_HEADER + 5 * 5 * 4 + 2

# ChatotCry (include/struct_defs/chatot_cry.h:6-9): BOOL valid, then s8 audioData[1000]
# (CHATOT_CRY_SIZE). SaveData_GetChatotCry (src/chatot_cry.c:27-30) is the page itself.
CHATOT_CRY = page_offset("CHATOT")  # 0x64EC
CHATOT_CRY_SIZE = 1000
# Sound_GetChatterActivationParameter (src/sound_chatot.c:193-213) reads audioData[15].
CHATOT_PARAM_SAMPLE = 15

# --- boxes block (PCBoxes, include/pc_boxes.h:17) ----------------------------------------
BOX_MONS = 4  # after u32 currentBoxID
BOX_MON_SIZE = 0x88  # BoxPokemon
BOX_COUNT = 18  # MAX_PC_BOXES
BOX_SLOTS = 30  # MAX_MONS_PER_BOX
