"""Species, experience tables and learnsets read from the ROM."""

import json

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
