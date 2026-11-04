"""``tools.extract_rom``: the vanilla round trip against the decomp extractor (spec §6.3, §9.1)."""

import json

import pytest

pytest.importorskip("ndspy")

from tools.extract_decomp import extract_static, extract_trainers
from tools.extract_rom import extract, parse_args
from tools.rom.inputs import RomInputError
from tools.rom.symbols import DEFAULT_MAP


def _json(path):
    return json.loads(path.read_text("utf-8"))


def test_vanilla_round_trip(rom_inputs, decomp, symbols, tmp_path):
    extract_static(decomp, tmp_path / "d")
    extract_trainers(decomp, tmp_path / "d")
    sources = extract("vanilla", tmp_path / "r", decomp, DEFAULT_MAP, rom_inputs)
    names = sorted(p.name for p in (tmp_path / "d").glob("*.json"))
    got = sorted(p.name for p in (tmp_path / "r").glob("*.json") if p.name != "rom_sources.json")
    assert got == names
    for n in names:
        assert _json(tmp_path / "r" / n) == _json(tmp_path / "d" / n), n
    assert sources["types.json"].startswith("decomp") and sources["species.json"] == "rom"
    assert _json(tmp_path / "r" / "rom_sources.json") == sources
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d", "r"]  # no staging dir left


def test_kaizo_refused_and_nothing_written(kaizo_inputs, decomp, symbols, tmp_path):
    with pytest.raises(RomInputError, match="K2b"):
        extract("kaizo", tmp_path / "k", decomp, DEFAULT_MAP, kaizo_inputs)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_existing_out_files_kept(rom_inputs, decomp, symbols, tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "other.json").write_text("{}")
    extract("vanilla", out, decomp, DEFAULT_MAP, rom_inputs)
    assert (out / "other.json").read_text() == "{}" and (out / "species.json").exists()


def test_failure_leaves_out_untouched(rom_inputs, decomp, symbols, tmp_path):
    with pytest.raises(OSError):
        extract("vanilla", tmp_path / "r", tmp_path / "no-decomp", DEFAULT_MAP, rom_inputs)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_cli_default_out():
    assert parse_args(["--game", "vanilla"]).out.as_posix().endswith("data/intermediate")
    assert parse_args(["--game", "kaizo"]).out.as_posix().endswith("data/intermediate_kaizo")
    assert parse_args(["--game", "vanilla", "--out", "x"]).out.as_posix() == "x"
    with pytest.raises(SystemExit) as e:
        parse_args(["--game", "emerald"])
    assert e.value.code == 2
