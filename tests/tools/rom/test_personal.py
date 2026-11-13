"""Species, experience tables and learnsets read from the ROM."""

import json
import struct

import numpy as np
import pytest

from tools.extract_decomp import (
    _constants,
    extract_exp_tables,
    extract_learnsets,
    extract_species,
)
from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.learnsets import read_learnsets
from tools.rom.readers.personal import read_exp_tables, read_forms, read_species


def test_species_match_decomp(vanilla_files, decomp, charmap):
    assert read_species(vanilla_files, decomp, charmap) == extract_species(decomp)


def test_exp_tables_match_decomp(vanilla_files, decomp):
    assert read_exp_tables(vanilla_files) == extract_exp_tables(decomp)


def test_learnsets_match_decomp(vanilla_files, decomp):
    assert read_learnsets(vanilla_files, decomp) == extract_learnsets(decomp)


def test_member_size_mismatch_raises():
    with pytest.raises(ReaderError, match=r"pl_personal\.narc.*member 1.*44.*46"):
        fixed_members([b"\0" * 44, b"\0" * 46], 44, "pl_personal.narc")


def test_kaizo_species_read(kaizo_files, decomp, charmap):
    species = read_species(kaizo_files, decomp, charmap)
    assert len(species) == 496
    solid_rock = _constants(decomp, "abilities", "ABILITY_").index("ABILITY_SOLID_ROCK")
    assert solid_rock in species[345]["abilities"]
    # spec 5.3: catch rate changed on 452 species; Bulbasaur 45 -> 175 (PKCalc agrees)
    assert (species[1]["name"], species[1]["catch_rate"]) == ("BULBASAUR", 175)


class _Wotbl:
    """A ROM whose wotbl.narc is ``members`` padded with empty learnsets to 496."""

    def __init__(self, first: bytes):
        self.members = [first] + [b"\xff\xff\x00\x00"] * 507

    def narc(self, path):
        return self.members


def _learnset(n, tail=b"\x00\x00"):
    import struct

    return b"".join(struct.pack("<H", 33 | 5 << 9) for _ in range(n)) + b"\xff\xff" + tail


def test_learnset_layout_checks(decomp):
    from tools.rom.readers.learnsets import max_learnset_entries

    assert max_learnset_entries(decomp) == 20
    assert read_learnsets(_Wotbl(_learnset(21, b"")), decomp)["0"] == [[5, 33]] * 21
    with pytest.raises(ReaderError, match="member 0: 22 entries, more than the 21"):
        read_learnsets(_Wotbl(_learnset(22)), decomp)
    with pytest.raises(ReaderError, match="member 0: 7 bytes, an odd length"):
        read_learnsets(_Wotbl(_learnset(2, b"\x00")), decomp)
    with pytest.raises(ReaderError, match="unaccounted bytes"):
        read_learnsets(_Wotbl(_learnset(1, b"\x00\x01")), decomp)
    with pytest.raises(ReaderError, match="unaccounted bytes"):
        read_learnsets(_Wotbl(_learnset(1, b"\x00" * 4)), decomp)
    short = _Wotbl(_learnset(1))
    short.members = short.members[:496]
    with pytest.raises(ReaderError, match=r"wotbl\.narc: 496 members, expected 508"):
        read_learnsets(short, decomp)


def test_vanilla_learnsets_within_max_entries(vanilla_files, decomp):
    got = read_learnsets(vanilla_files, decomp)
    assert max(len(v) for v in got.values()) == 20
    assert all(len(m) % 4 == 0 for m in vanilla_files.narc("poketool/personal/wotbl.narc"))


def test_personal_narcs_have_508_members(vanilla_files):
    for path in ("poketool/personal/pl_personal.narc", "poketool/personal/wotbl.narc"):
        assert len(vanilla_files.narc(path)) == 508


def test_rom_forms_match_decomp(vanilla_files, decomp, charmap, decomp_inter):
    want = json.loads((decomp_inter / "forms.json").read_text("utf-8"))
    assert read_forms(vanilla_files, decomp, charmap) == want  # row fields read from the NARC


# --- K3d-exp T2: EV yields, learnsets and growth tables of the Kaizo build ------------------


def test_ev_yield_bit_layout():
    # SpeciesEVYields (include/struct_defs/species.h:25-32): 2-bit hp, attack, defense, speed,
    # spAttack, spDefense from bit 0 of the u16 at byte 10 (species.h:43-44).
    from tools.rom.readers.personal import _ev_yield

    word = 1 | 2 << 2 | 3 << 4 | 3 << 6 | 1 << 8 | 2 << 10
    member = bytes(10) + struct.pack("<H", word) + bytes(32)
    assert _ev_yield(member, 0) == [1, 2, 3, 1, 2, 3]  # ids.Stat: HP, Atk, Def, SpA, SpD, Spe
    with pytest.raises(ReaderError, match="member 7 offset 10: EV yield bits above 11"):
        _ev_yield(bytes(10) + struct.pack("<H", 1 << 12) + bytes(32), 7)


def _raw_ev_yield(member: bytes) -> list[int]:
    """An independent decode of species.h:25-32 into ids.Stat order."""
    (w,) = struct.unpack_from("<H", member, 10)
    hp, atk, df, spe, spa, spd = (w >> s & 3 for s in range(0, 12, 2))
    return [hp, atk, df, spa, spd, spe]


def test_kaizo_ev_yield_equals_the_rom_decode(kaizo_files, kaizo_tables, kaizo_names):
    members = kaizo_files.narc("poketool/personal/pl_personal.narc")
    want = [_raw_ev_yield(m) for m in members]
    assert np.asarray(kaizo_tables.species_ev_yield).tolist() == want
    wormadam = kaizo_names.species("Wormadam")
    for row in (
        kaizo_names.species("Bonsly"),
        kaizo_names.species("Geodude"),
        *(kaizo_names.form_row(wormadam, f) for f in range(3)),  # Plant, Sandy, Trash
    ):
        assert want[row] == [0] * 6, row  # Kaizo zeroes every EV yield (measured, K3d-exp T2)


def test_kaizo_learnsets_equal_the_rom(kaizo_files, decomp, kaizo_tables):
    rom = read_learnsets(kaizo_files, decomp)
    for row in range(508):
        lv = np.asarray(kaizo_tables.learnset_level[row]).tolist()
        mv = np.asarray(kaizo_tables.learnset_move[row]).tolist()
        n = len(rom[str(row)])
        assert [list(p) for p in zip(lv[:n], mv[:n], strict=True)] == rom[str(row)], row
        assert lv[n:] == mv[n:] == [0] * (21 - n), row


def test_kaizo_exp_tables_equal_the_rom(kaizo_files, kaizo_tables):
    assert read_exp_tables(kaizo_files) == np.asarray(kaizo_tables.exp_table).tolist()
