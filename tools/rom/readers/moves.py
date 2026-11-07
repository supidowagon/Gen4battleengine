"""Move data from ``poketool/waza/pl_waza_tbl.narc``."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import MOVE_LIST_ARRAYS, _constants
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
        "ai_effect": effect,  # both come from MoveTable.effect in the game's own data
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


VANILLA_GAME_CODE = "CPUE"


def _hits(rom: RomFiles, needle: bytes) -> list[tuple[str, int, bytes]]:
    """Every (image, offset, image bytes) where ``needle`` occurs in the decompressed ARM9
    and overlays of ``rom``, overlapping hits included."""
    images = [("ARM9", rom.arm9())] + [(f"overlay {i}", rom.overlay(i)) for i in rom.overlay_ids()]
    out = []
    for where, data in images:
        at = data.find(needle)
        while at != -1:
            out.append((where, at, data))
            at = data.find(needle, at + 1)
    return out


def _context(hit: tuple[str, int, bytes], size: int) -> tuple[str, bytes, bytes]:
    """The image, and the u16 just before and just after a hit of ``size`` bytes."""
    where, at, data = hit
    return where, data[max(at - 2, 0) : at], data[at + size : at + size + 2]


def read_move_lists(
    rom: RomFiles,
    vanilla_lists: dict[str, list[int]],
    override: dict[str, list[int]] | None = None,
    vanilla_rom: RomFiles | None = None,
) -> dict[str, list[int]]:
    """``move_lists.json`` for the ROM: ``{"sound": [ids], "punch": [ids]}``.

    The game keeps no table for these (``sSoundMoves`` and ``sPunchingMoves`` are C arrays
    in ``battle_lib.c``), so each vanilla list is searched for as consecutive little-endian
    u16 in the decompressed ARM9 and every decompressed overlay. A list is unchanged when
    it is found exactly once and, unless ``rom`` is the vanilla ROM, the u16 before and
    after it equal those around the single hit in ``vanilla_rom`` (required for any other
    ROM): a move appended or prepended to the array leaves the vanilla bytes in place, so
    the bytes alone cannot show it. A list not found unchanged takes ``override[key]``
    (transcribed by hand) if given, else this raises. More than one hit is ambiguous and
    always raises."""
    patched = rom.game_code != VANILLA_GAME_CODE
    if patched and vanilla_rom is None:
        raise ReaderError(
            f"ROM {rom.game_code!r} is not the vanilla ROM: pass vanilla_rom to compare "
            "the move lists' neighbouring bytes"
        )
    out = {}
    for key, array in MOVE_LIST_ARRAYS.items():
        needle = b"".join(struct.pack("<H", m) for m in vanilla_lists[key])
        hits = _hits(rom, needle)
        if len(hits) > 1:
            raise ReaderError(
                f"{array} found {len(hits)} times ({[(w, a) for w, a, _ in hits]}); "
                "it must be unique"
            )
        unchanged = len(hits) == 1
        if unchanged and patched:
            base = _hits(vanilla_rom, needle)
            if len(base) != 1:
                raise ReaderError(f"{array} found {len(base)} times in the vanilla ROM, need 1")
            unchanged = _context(hits[0], len(needle)) == _context(base[0], len(needle))
        if unchanged:
            out[key] = list(vanilla_lists[key])
        elif override is not None and key in override:
            out[key] = [int(m) for m in override[key]]
        else:
            raise ReaderError(
                f"{array} not found unchanged: transcribe it into "
                "data/kaizo/overrides.toml [move_lists]"
            )
    return out
