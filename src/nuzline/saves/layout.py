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
    "COPY_OFFSETS",
    "FOOTER_SIZE",
    "NORMAL_SIZE",
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
    "VAR_PLAYER_STARTER_INDEX",
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

# --- boxes block (PCBoxes, include/pc_boxes.h:17) ----------------------------------------
BOX_MONS = 4  # after u32 currentBoxID
BOX_MON_SIZE = 0x88  # BoxPokemon
BOX_COUNT = 18  # MAX_PC_BOXES
BOX_SLOTS = 30  # MAX_MONS_PER_BOX
