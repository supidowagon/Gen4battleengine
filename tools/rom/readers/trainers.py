"""Trainer headers and parties from ``poketool/trainer/trdata.narc`` and ``trpoke.narc``."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError, symbol_bytes
from tools.rom.readers.text import message_bank

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

TRDATA_NARC = "poketool/trainer/trdata.narc"
TRPOKE_NARC = "poketool/trainer/trpoke.narc"
# TrainerHeader (include/struct_defs/trainer_data.h:24-32): monDataType u8, trainerType u8,
# sprite u8, partySize u8, items u16 x MAX_TRAINER_ITEMS, aiMask u32, battleType u32.
HEADER_SIZE = 20
MAX_ITEMS = 4  # MAX_TRAINER_ITEMS, trainer_data.h:14
PARTY_MAX = 6
MOVES_MAX = 4  # LEARNED_MOVES_MAX
# sizeof of TrainerMonBase / WithMoves / WithItem / WithMovesAndItem (trainer_data.h:34-64),
# indexed by TRDATATYPE_* (trainer_data.h:17-22).
MON_SIZES = (8, 16, 10, 18)
HAS_MOVES = (False, True, False, True)
HAS_ITEM = (False, False, True, True)
FORM_SHIFT = 10  # TRAINER_MON_FORM_SHIFT, trainer_data.h:11
SPECIES_MASK = (1 << FORM_SHIFT) - 1
GENDER_SYMBOL = "sTrainerClassGender"  # u8 per class, src/trainer_data.c:9,166


def party_bytes(member: bytes, trainer_id: int, data_type: int, party_size: int) -> bytes:
    """The mon records of one ``trpoke.narc`` member, without its padding.

    The member is ``round_up_4(party_size * MON_SIZES[data_type])`` bytes (measured on
    vanilla: the packer pads to 4), every padding byte zero; a party of 0 is an 8-byte
    all-zero placeholder (trainer 0). Any other size or a non-zero padding byte raises.
    """
    if not 0 <= data_type < len(MON_SIZES):
        raise ReaderError(f"trainer {trainer_id}: unknown data type {data_type}")
    used = party_size * MON_SIZES[data_type]
    want = (used + 3) & ~3 if party_size else 8
    if len(member) != want:
        raise ReaderError(
            f"{TRPOKE_NARC} trainer {trainer_id}: expected {want} bytes, got {len(member)} "
            f"(party {party_size}, data type {data_type})"
        )
    for off in range(used, len(member)):
        if member[off]:
            raise ReaderError(
                f"{TRPOKE_NARC} trainer {trainer_id} offset {off}: non-zero padding byte "
                f"{member[off]:#x} (party {party_size}, data type {data_type})"
            )
    return member[:used]


def read_trainers(
    rom: RomFiles,
    decomp: Path,
    symbols: SymbolMap,
    charmap: dict[int, str],
    start_conditions: list[str],
) -> list[dict]:
    """``trainers.json`` rows. ``start_conditions[id]`` is supplied by the caller."""
    ids = _constants(decomp, "trainers", "TRAINER_")
    classes = _constants(decomp, "trainer_classes", "TRAINER_CLASS_")
    species_n = len(_constants(decomp, "species", "SPECIES_"))
    move_n = len(_constants(decomp, "moves", "MOVE_"))
    item_n = len(_constants(decomp, "items", "ITEM_"))
    if len(start_conditions) != len(ids):
        raise ReaderError(f"{len(start_conditions)} start conditions for {len(ids)} trainers")
    headers = rom.narc(TRDATA_NARC)
    parties = rom.narc(TRPOKE_NARC)
    if len(headers) != len(ids) or len(parties) != len(ids):
        raise ReaderError(
            f"{TRDATA_NARC} has {len(headers)} and {TRPOKE_NARC} {len(parties)} members, "
            f"decomp has {len(ids)} trainers"
        )
    genders = symbol_bytes(rom, symbols, GENDER_SYMBOL)
    if len(genders) != len(classes):
        raise ReaderError(f"{GENDER_SYMBOL}: {len(genders)} bytes for {len(classes)} classes")
    names = message_bank(rom, decomp, "TEXT_BANK_NPC_TRAINER_NAMES", charmap, True)
    class_names = message_bank(rom, decomp, "TEXT_BANK_TRAINER_CLASS_NAMES", charmap)
    if len(names) < len(ids):
        raise ReaderError(f"TEXT_BANK_NPC_TRAINER_NAMES: {len(names)} names for {len(ids)}")
    if len(class_names) < len(classes):
        raise ReaderError(f"TEXT_BANK_TRAINER_CLASS_NAMES: {len(class_names)} < {len(classes)}")

    out = []
    for tid, (head, member) in enumerate(zip(headers, parties, strict=True)):
        if len(head) != HEADER_SIZE:
            raise ReaderError(f"{TRDATA_NARC} member {tid}: {len(head)} bytes, need {HEADER_SIZE}")
        data_type, cls, size = head[0], head[1], head[3]
        bag = struct.unpack_from(f"<{MAX_ITEMS}H", head, 4)
        (ai_mask,) = struct.unpack_from("<I", head, 12)
        if cls >= len(classes):
            raise ReaderError(f"{TRDATA_NARC} member {tid} offset 1: class {cls} out of range")
        if size > PARTY_MAX:
            raise ReaderError(f"{TRDATA_NARC} member {tid} offset 3: party size {size}")
        for i, it in enumerate(bag):
            if it >= item_n:
                raise ReaderError(f"{TRDATA_NARC} member {tid} offset {4 + 2 * i}: item {it}")
        body = party_bytes(member, tid, data_type, size)
        rec = MON_SIZES[data_type]
        party = []
        for k in range(size):
            at = k * rec
            iv, level, sp = struct.unpack_from("<3H", body, at)
            pos = at + 6
            item = 0
            if HAS_ITEM[data_type]:
                (item,) = struct.unpack_from("<H", body, pos)
                pos += 2
            moves = None
            if HAS_MOVES[data_type]:
                moves = list(struct.unpack_from(f"<{MOVES_MAX}H", body, pos))
                pos += 2 * MOVES_MAX
            # The trailing u16 is cbSeal, the ball capsule seal (trainer_data.c:217); it is
            # cosmetic, trainers.json has no field for it, and vanilla uses non-zero values.
            where = f"{TRPOKE_NARC} trainer {tid} offset {at}"
            species = sp & SPECIES_MASK
            if species >= species_n:
                raise ReaderError(f"{where}: species {species} out of range")
            if item >= item_n or (moves and max(moves) >= move_n):
                raise ReaderError(f"{where}: item or move out of range")
            party.append(
                {
                    "species": species,
                    "level": level,
                    "difficulty": iv,
                    "moves": moves,
                    "item": item,
                    "form": sp >> FORM_SHIFT,
                }
            )
        out.append(
            {
                "id": tid,
                "name": names[tid],
                "class": class_names[cls],
                "class_id": cls,
                "class_gender": genders[cls],
                "ai_flags": ai_mask,
                "items": [i for i in bag if i],
                "party": party,
                "start_condition": start_conditions[tid],
            }
        )
    return out
