"""Trainer headers and parties read from the ROM."""

import json

import pytest

from tools.extract_decomp import extract_trainers
from tools.rom.readers import ReaderError
from tools.rom.readers.trainers import TRPOKE_NARC, party_bytes, read_trainers


def test_trainers_match_decomp_except_start(vanilla_files, decomp, symbols, charmap, tmp_path):
    extract_trainers(decomp, tmp_path)
    want = json.loads((tmp_path / "trainers.json").read_text("utf-8"))
    starts = [t["start_condition"] for t in want]
    got = read_trainers(vanilla_files, decomp, symbols, charmap, starts)
    assert got == want


def test_rom_trainer_format_from_battle_type(vanilla_files, decomp, symbols, charmap, decomp_inter):
    want = json.loads((decomp_inter / "trainers.json").read_text("utf-8"))
    rows = read_trainers(
        vanilla_files, decomp, symbols, charmap, [t["start_condition"] for t in want]
    )
    assert sum(t["format"] == "double" for t in rows) == 28
    assert [t["format"] for t in rows] == [t["format"] for t in want]


def test_roark(vanilla_files, decomp, symbols, charmap):
    t = read_trainers(vanilla_files, decomp, symbols, charmap, ["none"] * 928)[246]
    assert (t["name"], t["class_id"], t["ai_flags"], t["items"]) == ("Roark", 62, 7, [17, 17])
    assert [m["species"] for m in t["party"]] == [74, 95, 408]


def test_party_size_rule():
    assert len(party_bytes(bytes(92), 1, 3, 5)) == 90
    assert len(party_bytes(bytes(56), 2, 3, 3)) == 54
    assert len(party_bytes(bytes(12), 3, 2, 1)) == 10
    assert party_bytes(bytes(8), 0, 0, 0) == b""
    with pytest.raises(ReaderError, match="trainer 4.*56.*48"):
        party_bytes(bytes(48), 4, 3, 3)
    with pytest.raises(ReaderError, match="trainer 5.*padding"):
        party_bytes(bytes(90) + b"\x01\x00", 5, 3, 5)
    with pytest.raises(ReaderError, match="trainer 6"):
        party_bytes(bytes(96), 6, 3, 5)


class _Patched:
    """``rom`` with one ``trpoke.narc`` member replaced."""

    def __init__(self, rom, trainer, member):
        self._rom, self._trainer, self._member = rom, trainer, member

    def __getattr__(self, name):
        return getattr(self._rom, name)

    def narc(self, path):
        members = self._rom.narc(path)
        if path == TRPOKE_NARC:
            members = [*members]
            members[self._trainer] = self._member
        return members


def test_level_above_100_raises(vanilla_files, decomp, symbols, charmap):
    # Roark (246): the first mon's level is the u16 at offset 2.
    member = vanilla_files.narc(TRPOKE_NARC)[246]
    bad = member[:2] + (101).to_bytes(2, "little") + member[4:]
    with pytest.raises(ReaderError, match="trainer 246 offset 0: level 101 exceeds 100"):
        read_trainers(_Patched(vanilla_files, 246, bad), decomp, symbols, charmap, ["none"] * 928)
