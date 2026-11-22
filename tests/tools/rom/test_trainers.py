"""Trainer headers and parties read from the ROM."""

import json

import pytest

from tools.extract_decomp import extract_trainers
from tools.rom.readers import ReaderError
from tools.rom.readers.maps import trainer_fog_headers, trainer_start_conditions
from tools.rom.readers.trainers import TRPOKE_NARC, party_bytes, read_trainers


def _no_fog(n):
    """Fog headers for tests that read no start in fog: a fresh empty list per trainer."""
    return [[] for _ in range(n)]


def _n_trainers(decomp):
    """The decomp's trainer count (generated/trainers.txt), the length read_trainers checks."""
    from tools.extract_decomp import _constants

    return len(_constants(decomp, "trainers", "TRAINER_"))


def test_trainers_match_decomp_except_start(vanilla_files, decomp, symbols, charmap, tmp_path):
    extract_trainers(decomp, tmp_path)
    want = json.loads((tmp_path / "trainers.json").read_text("utf-8"))
    starts = [t["start_condition"] for t in want]
    where: dict[int, list[int]] = {}
    trainer_start_conditions(vanilla_files, decomp, symbols, maps=where)
    fog = trainer_fog_headers(vanilla_files, decomp, symbols, where, len(want))
    got = read_trainers(vanilla_files, decomp, symbols, charmap, starts, fog, layout="vanilla")
    # The terrain fields are joined in by extract_rom (maps.trainer_terrain_rows); the
    # round trip (test_extract_rom.py) compares them. fog_headers is compared here.
    assert got == [{k: v for k, v in t.items() if k not in ("terrain", "terrains")} for t in want]
    # G-15: the ROM's header weather bytes give the decomp's fog headers (Luna, ROUTE_210_NORTH).
    assert got[66]["start_condition"] == "fog" and got[66]["fog_headers"] == [363]
    # The headers of weather OVERWORLD_WEATHER_FOG in include/data/map_headers.h that place a
    # fog fight: 247 VICTORY_ROAD_1F_ROOM_2, 359 ROUTE_209_LOST_TOWER_3F,
    # 360 ROUTE_209_LOST_TOWER_4F, 363 ROUTE_210_NORTH (ids by tools.terrain.header_names).
    assert {h for t in got for h in t["fog_headers"]} == {247, 359, 360, 363}


def test_fog_headers_only_for_fog_starts(vanilla_files, decomp, symbols, charmap):
    n = _n_trainers(decomp)
    fog = [[363] for _ in range(n)]
    rows = read_trainers(vanilla_files, decomp, symbols, charmap, ["none"] * n, fog)
    assert all(t["fog_headers"] == [] for t in rows)
    with pytest.raises(ReaderError, match=f"{n - 1} fog header lists for {n} trainers"):
        read_trainers(vanilla_files, decomp, symbols, charmap, ["none"] * n, fog[1:])


def test_rom_trainer_format_from_battle_type(vanilla_files, decomp, symbols, charmap, decomp_inter):
    want = json.loads((decomp_inter / "trainers.json").read_text("utf-8"))
    rows = read_trainers(
        vanilla_files,
        decomp,
        symbols,
        charmap,
        [t["start_condition"] for t in want],
        _no_fog(len(want)),
    )
    assert sum(t["format"] == "double" for t in rows) == 28
    assert [t["format"] for t in rows] == [t["format"] for t in want]


def test_roark(vanilla_files, decomp, symbols, charmap):
    n = _n_trainers(decomp)
    t = read_trainers(vanilla_files, decomp, symbols, charmap, ["none"] * n, _no_fog(n))[246]
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
    # Kaizo's packer leaves most members unpadded (K2b Task 8)
    with pytest.raises(ReaderError, match="trainer 7.*56.*54"):
        party_bytes(bytes(54), 7, 3, 3)
    assert len(party_bytes(bytes(54), 7, 3, 3, unpadded=True)) == 54
    assert len(party_bytes(bytes(56), 7, 3, 3, unpadded=True)) == 54
    with pytest.raises(ReaderError, match="trainer 7"):
        party_bytes(bytes(55), 7, 3, 3, unpadded=True)


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
    n = _n_trainers(decomp)
    with pytest.raises(ReaderError, match="trainer 246 offset 0: level 101 exceeds 100"):
        read_trainers(
            _Patched(vanilla_files, 246, bad), decomp, symbols, charmap, ["none"] * n, _no_fog(n)
        )
