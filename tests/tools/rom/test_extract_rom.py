"""``tools.extract_rom``: the vanilla round trip against the decomp extractor (spec §6.3, §9.1)."""

import json

import pytest

pytest.importorskip("ndspy")

from tools.extract_decomp import extract_static, extract_trainers
from tools.extract_rom import (
    CHANGED_SCRIPTS_FILE,
    REUSE_BANKS,
    ROM_FILES,
    UNRESOLVED_FILE,
    check_reuse_names,
    extract,
    parse_args,
    reuse_violations,
)
from tools.rom.diff import CodeChange, FileChange, RomDiff
from tools.rom.inputs import RomInputError
from tools.rom.symbols import DEFAULT_MAP


def _json(path):
    return json.loads(path.read_text("utf-8"))


def test_vanilla_round_trip(rom_inputs, decomp, symbols, tmp_path, capsys):
    extract_static(decomp, tmp_path / "d")
    extract_trainers(decomp, tmp_path / "d")
    sources = extract("vanilla", tmp_path / "r", decomp, DEFAULT_MAP, rom_inputs)
    names = sorted(p.name for p in (tmp_path / "d").glob("*.json"))
    extra = {"rom_sources.json", UNRESOLVED_FILE}
    got = sorted(p.name for p in (tmp_path / "r").glob("*.json") if p.name not in extra)
    assert got == names
    for n in names:
        assert _json(tmp_path / "r" / n) == _json(tmp_path / "d" / n), n
    assert sources["types.json"].startswith("decomp") and sources["species.json"] == "rom"
    assert _json(tmp_path / "r" / "rom_sources.json") == sources
    unresolved = _json(tmp_path / "r" / UNRESOLVED_FILE)
    assert [(u["member"], u["offset"], u["variable_name"]) for u in unresolved] == [
        (454, 0x321, "VAR_0x8007"),
        (1101, 0x341, "VAR_MAP_LOCAL_0x05"),
    ]
    err = capsys.readouterr().err
    assert "member 454 offset 0x321" in err and "member 1101 offset 0x341" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d", "r"]  # no staging dir left


def test_kaizo_extract_writes_patched_data(kaizo_rom_dir):
    # The K1 refusal is gone (K2b Task 8): the Kaizo ROM's own tables are written.
    names = {p.name for p in kaizo_rom_dir.glob("*.json")}
    assert {f"{n}.json" for n in ROM_FILES} | {CHANGED_SCRIPTS_FILE, UNRESOLVED_FILE} <= names
    sources = _json(kaizo_rom_dir / "rom_sources.json")
    assert sources["trainers.json"] == "rom" and sources[CHANGED_SCRIPTS_FILE] == "rom diff"
    roark = _json(kaizo_rom_dir / "trainers.json")[246]
    assert [m["species"] for m in roark["party"]] == [438, 345, 443, 222, 213, 408]
    unresolved = _json(kaizo_rom_dir / UNRESOLVED_FILE)
    assert (136, 909, "VAR_0x8004") in [
        (u["member"], u["offset"], u["variable_name"]) for u in unresolved
    ]


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
    assert parse_args(["--game", "kaizo"]).out.as_posix().endswith("data/intermediate_kaizo/rom")
    assert parse_args(["--game", "vanilla", "--out", "x"]).out.as_posix() == "x"
    with pytest.raises(SystemExit) as e:
        parse_args(["--game", "emerald"])
    assert e.value.code == 2


def test_reuse_violations_names_changed_sources(decomp):
    from tools.extract_decomp import _constants

    banks = _constants(decomp, "text_banks", "TEXT_BANK_")
    nature = banks.index("TEXT_BANK_NATURE_NAMES")
    other = banks.index("TEXT_BANK_SPECIES_NAME")
    diff = RomDiff(
        (),
        (
            FileChange("msgdata/pl_msg.narc", "changed", 1, 2, (other, nature), (900, 900)),
            FileChange("poketool/personal/pl_personal.narc", "changed", 1, 2, (0,), (9, 9)),
        ),
        (
            CodeChange("overlay16", 1, 2, 1, 2, "sTypeMatchupMultipliers"),
            CodeChange("arm9", 1, 2, 1, 2, "SomethingElse"),
        ),
        {},
        {},
    )
    assert reuse_violations(diff, decomp) == [
        "symbol sTypeMatchupMultipliers (overlay16)",
        f"msgdata/pl_msg.narc member {nature} (TEXT_BANK_NATURE_NAMES)",
    ]
    assert reuse_violations(RomDiff((), (), (), {}, {}), decomp) == []


def test_reuse_names_checked(symbols, decomp, monkeypatch):
    import tools.extract_rom as er

    check_reuse_names(symbols, decomp)
    monkeypatch.setattr(er, "REUSE_SYMBOLS", ("sTypeMatchupMultiplier",))
    with pytest.raises(RomInputError, match="sTypeMatchupMultiplier .*not in the map"):
        check_reuse_names(symbols, decomp)
    monkeypatch.setattr(er, "REUSE_SYMBOLS", ())
    monkeypatch.setattr(er, "REUSE_BANKS", (*REUSE_BANKS, "TEXT_BANK_NATURES"))
    with pytest.raises(RomInputError, match="TEXT_BANK_NATURES"):
        check_reuse_names(symbols, decomp)
