"""Soundproof and punch lists searched for in the ROM's code, and the Kaizo TM and tutor
tables."""

import json
import struct
from pathlib import Path

import pytest

from tools.extract_decomp import _constants, extract_move_lists
from tools.rom.readers import ReaderError
from tools.rom.readers.moves import read_move_lists


class FakeRom:
    """Just the accessors ``read_move_lists`` uses."""

    def __init__(
        self, arm9: bytes, overlays: dict[int, bytes] | None = None, game_code: str = "CPUE"
    ):
        self._arm9, self._overlays, self.game_code = arm9, overlays or {}, game_code

    def arm9(self) -> bytes:
        return self._arm9

    def overlay_ids(self) -> list[int]:
        return sorted(self._overlays)

    def overlay(self, i: int) -> bytes:
        return self._overlays[i]


def _u16(ids: list[int]) -> bytes:
    return b"".join(m.to_bytes(2, "little") for m in ids)


LISTS = {"sound": [45, 46, 47], "punch": [4, 5]}
BOTH = _u16(LISTS["sound"]) + b"\xff\xff" + _u16(LISTS["punch"])


def test_vanilla_rom_lists_match_decomp(vanilla_files, decomp):
    lists = extract_move_lists(decomp)
    assert read_move_lists(vanilla_files, lists) == lists


def test_found_once_in_arm9_or_an_overlay_is_unchanged():
    assert read_move_lists(FakeRom(b"\x00" * 8 + BOTH), LISTS) == LISTS
    rom = FakeRom(_u16(LISTS["sound"]), {3: b"\x00\x00" + _u16(LISTS["punch"])})
    assert read_move_lists(rom, LISTS) == LISTS


def test_changed_list_without_override_raises():
    rom = FakeRom(_u16(LISTS["sound"]) + _u16([4, 6]))
    with pytest.raises(ReaderError, match=r"sPunchingMoves not found unchanged.*\[move_lists\]"):
        read_move_lists(rom, LISTS)
    with pytest.raises(ReaderError, match="sPunchingMoves not found unchanged"):
        read_move_lists(rom, LISTS, {"sound": [1]})  # an override for the other list


def test_override_used_only_for_a_changed_list():
    rom = FakeRom(_u16(LISTS["sound"]) + _u16([4, 6]))
    got = read_move_lists(rom, LISTS, {"punch": [4, 6], "sound": [99]})
    assert got == {"sound": LISTS["sound"], "punch": [4, 6]}  # the unchanged one ignores it


def test_ambiguous_list_raises_even_with_override():
    rom = FakeRom(BOTH + BOTH)
    with pytest.raises(ReaderError, match="sSoundMoves found 2 times"):
        read_move_lists(rom, LISTS, {"sound": [1], "punch": [2]})


PAD = _u16([0x1111])
TAIL = _u16([0x2222])
VANILLA = FakeRom(PAD + BOTH + TAIL)


def _patched(image: bytes) -> FakeRom:
    return FakeRom(image, game_code="CPUK")


def test_patched_rom_with_unchanged_neighbours_is_accepted():
    rom = _patched(_u16([0, 0]) + PAD + BOTH + TAIL)  # moved, same context
    assert read_move_lists(rom, LISTS, vanilla_rom=VANILLA) == LISTS


def test_appended_move_is_refused():
    rom = _patched(PAD + BOTH + _u16([7]) + TAIL)  # punch list gains a move
    with pytest.raises(ReaderError, match="sPunchingMoves not found unchanged"):
        read_move_lists(rom, LISTS, vanilla_rom=VANILLA)
    got = read_move_lists(rom, LISTS, {"punch": [4, 5, 7]}, vanilla_rom=VANILLA)
    assert got == {"sound": LISTS["sound"], "punch": [4, 5, 7]}


def test_prepended_move_is_refused():
    rom = _patched(_u16([9]) + BOTH + TAIL)  # sound list gains a move at the front
    with pytest.raises(ReaderError, match="sSoundMoves not found unchanged"):
        read_move_lists(rom, LISTS, vanilla_rom=VANILLA)


def test_non_vanilla_rom_without_vanilla_rom_raises():
    with pytest.raises(ReaderError, match="pass vanilla_rom"):
        read_move_lists(_patched(PAD + BOTH + TAIL), LISTS)


def test_vanilla_rom_skips_the_neighbour_comparison():
    assert read_move_lists(FakeRom(_u16([0x3333]) + BOTH + _u16([0x4444])), LISTS) == LISTS


# --- K3a Task 5: the Kaizo TM and tutor tables the K2b readers feed the intermediates -----
# sTMHMMoves (arm9) and sTeachableMoves / sSpeciesLearnsetsByTutor (overlay5) differ in the
# Kaizo ROM. Each table is decoded here straight from the patched module bytes at its
# linker-map address and compared with what data/intermediate_kaizo holds (written by
# tools.extract_rom through tools/rom/readers/code_tables.py); the rom_diff.toml entries are
# ``modelled`` on the strength of these two tests.

KAIZO_INTER = Path(__file__).resolve().parents[3] / "data" / "intermediate_kaizo"
ITEM_TM01 = 328  # ITEM_TM01..ITEM_HM08 are items 328..427 (generated/items.txt)


def _kaizo_intermediate(name: str):
    path = KAIZO_INTER / name
    if not path.exists():
        pytest.skip(f"no {path}: run tools.extract_rom --game kaizo and tools.build_tables")
    return json.loads(path.read_text("utf-8"))


def _module_bytes(rom, symbols, name: str) -> bytes:
    """The bytes of ``name`` at its vanilla linker-map address in ``rom``'s module image."""
    sym = symbols.find(name)
    if sym.module == "arm9":
        regions = list(rom.arm9_sections())
    else:
        n = int(sym.module.removeprefix("overlay"))
        regions = [(rom.overlay_info(n)[0], rom.overlay(n))]
    for ram, data in regions:
        if ram <= sym.address and sym.address + sym.size <= ram + len(data):
            assert len(data) == symbols.extent(sym.module, ram), f"{sym.module} moved"
            return data[sym.address - ram : sym.address - ram + sym.size]
    raise AssertionError(f"{name} is outside {sym.module}")


def test_tm_table_matches_rom(kaizo_files, symbols):
    raw = _module_bytes(kaizo_files, symbols, "sTMHMMoves")
    decoded = list(struct.unpack("<100H", raw))
    items = _kaizo_intermediate("items.json")
    assert [i["teaches_move"] for i in items[ITEM_TM01 : ITEM_TM01 + 100]] == decoded
    assert decoded[6] == 62  # TM07: vanilla Hail (258), Kaizo Aurora Beam


def test_tutor_table_matches_rom(kaizo_files, symbols, decomp):
    raw = _module_bytes(kaizo_files, symbols, "sTeachableMoves")
    locations = _constants(decomp, "tutor_locations", "TUTOR_LOCATION_")
    decoded = [
        {
            "move": move,
            "costs": dict(zip(("red", "blue", "yellow", "green"), costs, strict=True)),
            "location": locations[loc].removeprefix("TUTOR_LOCATION_").lower(),
        }
        for move, *costs, _pad, loc in struct.iter_unpack("<H4BHI", raw)
    ]
    assert _kaizo_intermediate("tutor_moves.json") == decoded
    assert decoded[0]["move"] == 392  # vanilla Dive (291), Kaizo Aqua Ring

    masks = _module_bytes(kaizo_files, symbols, "sSpeciesLearnsetsByTutor")
    width = (len(decoded) + 7) // 8
    learnsets = _kaizo_intermediate("tutor_learnsets.json")
    # Moveset m (1-based) is mask m - 1; personal row r < 496 uses moveset r, a form row
    # r >= 496 uses 493 + (r - 495); rows 0, 494, 495 have none (include/tutor_movesets.h:20-26,
    # src/overlay005/scrcmd_move_tutor.c:211-259).
    for row, tutors in learnsets.items():
        r = int(row)
        if r in (0, 494, 495):
            assert tutors == [], row
            continue
        moveset = r if r < 496 else 493 + (r - 495)
        mask = int.from_bytes(masks[(moveset - 1) * width : moveset * width], "little")
        assert tutors == [j for j in range(len(decoded)) if mask >> j & 1], row
    assert len(masks) == width * (493 + len(learnsets) - 496)
