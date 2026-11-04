"""Species, experience tables and learnsets read from the ROM."""

import pytest

from tools.extract_decomp import (
    _constants,
    extract_exp_tables,
    extract_learnsets,
    extract_species,
)
from tools.rom.readers import ReaderError, fixed_members
from tools.rom.readers.learnsets import read_learnsets
from tools.rom.readers.personal import read_exp_tables, read_species


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
