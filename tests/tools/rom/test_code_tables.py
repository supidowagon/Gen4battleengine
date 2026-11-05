"""Code-resident tables read from the ROM (Ruling 13)."""

import json

import pytest

from tools.rom.readers import ReaderError
from tools.rom.readers.code_tables import tm_moves


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
