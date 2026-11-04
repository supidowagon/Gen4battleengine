"""Move data from ``poketool/waza/pl_waza_tbl.narc``."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import _constants
from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.text import message_bank

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles

WAZA_NARC = "poketool/waza/pl_waza_tbl.narc"
MOVE_SIZE = 16  # sizeof(MoveTable), include/move_table.h:12-24


def _range_names(decomp: Path) -> dict[int, str]:
    """Range value -> ``RANGE_*`` name. ``move_ranges`` is a ``--no-auto`` mask
    (``generated/meson.build:72``): line 1 is 0 and line n + 2 is bit n."""
    names = _constants(decomp, "move_ranges", "RANGE_")
    return {0 if i == 0 else 1 << (i - 1): n for i, n in enumerate(names)}


def decode_move(decomp: Path, move_id: int, m: bytes, name: str = "") -> dict:
    """One ``moves.json`` row from a 16-byte ``MoveTable``: effect u16 at 0, class 2,
    power 3, type 4, accuracy 5, pp 6, effect chance 7, range u16 at 8, priority s8 at 10,
    flags 11 (``include/move_table.h:12-24``). Values the decomp has no name for raise."""
    effect, cls, power, typ, acc, pp, chance, rng, prio, flags = struct.unpack(
        "<HBBBBBBHbB", m[:12]
    )
    where = f"{WAZA_NARC} member {move_id}"
    ranges = _range_names(decomp)
    if rng not in ranges:
        raise ReaderError(f"{where} offset 8: move {move_id} has unknown range {rng:#x}")
    limits = (
        ("effect", effect, "move_battle_effects", "BATTLE_EFFECT_", 0),
        ("class", cls, "move_classes", "CLASS_", 2),
        ("type", typ, "pokemon_types", "TYPE_", 4),
    )
    for what, value, table, prefix, off in limits:
        if value >= len(_constants(decomp, table, prefix)):
            raise ReaderError(f"{where} offset {off}: move {move_id} {what} {value} out of range")
    flag_names = _constants(decomp, "move_flags", "MOVE_FLAG_")
    if flags >> len(flag_names):
        raise ReaderError(f"{where} offset 11: move {move_id} flags {flags:#x} set unnamed bits")
    return {
        "name": name,
        "power": power,
        "type": typ,
        "category": cls,
        "accuracy": acc,
        "pp": pp,
        "priority": prio,
        "target": ranges[rng],
        "effect": effect,
        "effect_chance": chance,
        "flags": [n for bit, n in enumerate(flag_names) if flags >> bit & 1],
    }


def read_moves(rom: RomFiles, decomp: Path, charmap: dict[int, str]) -> list[dict]:
    """``moves.json`` rows for every ``MOVE_*`` id; names from ``TEXT_BANK_MOVE_NAMES``."""
    members = fixed_members(rom.narc(WAZA_NARC), MOVE_SIZE, WAZA_NARC)
    count = len(_constants(decomp, "moves", "MOVE_"))
    names = message_bank(rom, decomp, "TEXT_BANK_MOVE_NAMES", charmap)
    if len(members) < count:
        raise ReaderError(f"{WAZA_NARC}: {len(members)} members, need {count}")
    if len(names) < count:
        raise ReaderError(f"TEXT_BANK_MOVE_NAMES: {len(names)} entries, need {count}")
    return [decode_move(decomp, i, members[i], names[i]) for i in range(count)]
