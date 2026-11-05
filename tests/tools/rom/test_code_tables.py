"""Code-resident tables read from the ROM (Ruling 13)."""

import json

import pytest

from tools.rom.readers import ReaderError
from tools.rom.readers.code_tables import egg_moves, tm_moves, tutor_learnsets, tutor_moves
from tools.rom.readers.evolutions import read_evolutions


def test_tm_table_symbol(symbols):
    s = symbols.find("sTMHMMoves")
    assert s.module == "arm9" and s.size == 200


def test_tm_moves_match_decomp(vanilla_files, symbols, decomp_inter):
    tms = tm_moves(vanilla_files, symbols)
    assert len(tms) == 100 and tms[0] == 264  # TM01 Focus Punch
    items = json.loads((decomp_inter / "items.json").read_text("utf-8"))
    assert tms == [i["teaches_move"] for i in items[328:428]]  # ITEM_TM01..ITEM_HM08


def test_tm_moves_out_of_range_raises(vanilla_files, symbols, monkeypatch):
    import struct

    import tools.rom.readers.code_tables as ct

    monkeypatch.setattr(ct, "symbol_bytes", lambda *a: struct.pack("<100H", *([0] + [1] * 99)))
    with pytest.raises(ReaderError, match="sTMHMMoves"):
        tm_moves(vanilla_files, symbols)


def test_tm_learnsets_match_decomp(vanilla_files, decomp, decomp_inter):
    from tools.rom.readers.personal import read_tm_learnsets

    want = json.loads((decomp_inter / "tm_learnsets.json").read_text("utf-8"))
    assert read_tm_learnsets(vanilla_files) == want


def test_rom_egg_moves_match(vanilla_files, symbols, decomp_inter):
    want = json.loads((decomp_inter / "egg_moves.json").read_text("utf-8"))
    assert egg_moves(vanilla_files, symbols) == {int(k): v for k, v in want.items()}


def test_rom_evolutions_match(vanilla_files, decomp_inter):
    want = json.loads((decomp_inter / "evolutions.json").read_text("utf-8"))
    assert read_evolutions(vanilla_files) == want


def test_rom_tutor_tables_match(vanilla_files, symbols, decomp_inter):
    want_m = json.loads((decomp_inter / "tutor_moves.json").read_text("utf-8"))
    want_l = json.loads((decomp_inter / "tutor_learnsets.json").read_text("utf-8"))
    assert tutor_moves(vanilla_files, symbols) == want_m
    assert tutor_learnsets(vanilla_files, symbols, len(want_m)) == want_l


def test_tutor_symbols(symbols):
    assert symbols.find("sTeachableMoves").size == 12 * 38
    assert symbols.find("sSpeciesLearnsetsByTutor").size == 5 * 505


def _patch(monkeypatch, data):
    import tools.rom.readers.code_tables as ct

    monkeypatch.setattr(ct, "symbol_bytes", lambda rom, syms, sym: data[sym])
    return ct


def _move(m=291, costs=(2, 4, 2, 0), pad=0, loc=0):
    import struct

    return struct.pack("<H4BHI", m, *costs, pad, loc)


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        (_move(m=0), "move"),
        (_move(m=60000), "move"),
        (_move(costs=(100, 0, 0, 0)), "cost"),
        (_move(pad=1), "pad"),
        (_move(loc=3), "location"),
        (_move() + b"\0", "bytes"),
    ],
)
def test_tutor_moves_shape_errors(vanilla_files, symbols, monkeypatch, raw, why):
    _patch(monkeypatch, {"sTeachableMoves": raw})
    with pytest.raises(ReaderError, match="sTeachableMoves"):
        tutor_moves(vanilla_files, symbols)


def test_tutor_learnsets_shape_errors(vanilla_files, symbols, monkeypatch):
    good = bytes(5 * 505)
    _patch(monkeypatch, {"sSpeciesLearnsetsByTutor": good + b"\0"})
    with pytest.raises(ReaderError, match="sSpeciesLearnsetsByTutor"):
        tutor_learnsets(vanilla_files, symbols, 38)
    bad = bytearray(good)
    bad[4] = 0x40  # bit 6 of the last byte = tutor 38, past the 38 tutors
    _patch(monkeypatch, {"sSpeciesLearnsetsByTutor": bytes(bad)})
    with pytest.raises(ReaderError, match="sSpeciesLearnsetsByTutor"):
        tutor_learnsets(vanilla_files, symbols, 38)
