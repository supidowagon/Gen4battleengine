"""Moves and items read from the ROM."""

import struct

import pytest

from tools.extract_decomp import extract_items, extract_moves
from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.items import read_items
from tools.rom.readers.moves import MOVE_SIZE, decode_move, read_moves


def test_moves_match_decomp(vanilla_files, decomp, charmap):
    assert read_moves(vanilla_files, decomp, charmap) == extract_moves(decomp)


def test_items_match_decomp(vanilla_files, decomp, symbols, charmap):
    assert read_items(vanilla_files, decomp, symbols, charmap) == extract_items(decomp)


def _member(range_value: int) -> bytes:
    return struct.pack("<HBBBBBBHbB4x", 0, 0, 40, 0, 100, 10, 0, range_value, 0, 0)


def test_unknown_range_value_raises(decomp):
    assert len(_member(0)) == MOVE_SIZE
    with pytest.raises(ReaderError, match=r"move 7.*range 0x3"):
        decode_move(decomp, 7, _member(0x3))
    assert decode_move(decomp, 7, _member(0x1))["target"] == "RANGE_SINGLE_TARGET_SPECIAL"


def test_move_member_size_mismatch_raises():
    with pytest.raises(ReaderError, match=r"pl_waza_tbl\.narc.*member 1.*16.*15"):
        fixed_members([bytes(16), bytes(15)], 16, "poketool/waza/pl_waza_tbl.narc")


def _by_name(rows, name):
    hits = [i for i, r in enumerate(rows) if r["name"] == name]
    if not hits:
        pytest.fail(f"no {name!r} in this ROM's names")
    return rows[hits[0]]


def test_kaizo_changes(vanilla_files, kaizo_files, decomp, symbols, charmap):
    v_moves = read_moves(vanilla_files, decomp, charmap)
    k_moves = read_moves(kaizo_files, decomp, charmap)
    assert _by_name(v_moves, "Trick Room")["priority"] == -7
    assert _by_name(k_moves, "Trick Room")["priority"] == 7
    v_items = read_items(vanilla_files, decomp, symbols, charmap)
    k_items = read_items(kaizo_files, decomp, symbols, charmap)
    assert _by_name(v_items, "Flame Plate")["hold_param"] == 20
    assert _by_name(k_items, "Flame Plate")["hold_param"] == 50
    assert _by_name(v_items, "Town Map")["fling_power"] == 0  # spec 5.5: 0 -> 120
    assert _by_name(k_items, "Town Map")["fling_power"] == 120
