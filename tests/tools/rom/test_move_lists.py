"""Soundproof and punch lists searched for in the ROM's code."""

import pytest

from tools.extract_decomp import extract_move_lists
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
