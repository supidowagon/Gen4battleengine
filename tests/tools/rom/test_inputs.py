import hashlib
from pathlib import Path

import pytest

from tools.rom.inputs import REPO_ROOT, RomInputError, load_inputs, read_checked


def test_toml_and_env_precedence(tmp_path):
    cfg = tmp_path / "local.toml"
    patch = tmp_path / "k.xdelta"  # absolute on every OS: an absolute TOML path is kept as given
    cfg.write_text(f"vanilla_rom = 'roms/v.nds'\nkaizo_patch = '{patch}'\n")
    got = load_inputs(cfg, environ={"NUZLINE_KAIZO_SAVE": "s.sav"})
    assert got.vanilla_rom == REPO_ROOT / "roms" / "v.nds"
    assert got.kaizo_patch == patch and got.kaizo_save == Path("s.sav")
    assert got.pkcalc_zip is None and got.kaizo_docs is None
    assert got.kaizo_save_pre_roark is None and got.kaizo_save_probe is None
    probe = {"NUZLINE_KAIZO_SAVE_PROBE": "probe.dsv"}
    assert load_inputs(cfg, environ=probe).kaizo_save_probe == Path("probe.dsv")
    pre = {"NUZLINE_KAIZO_SAVE_PRE_ROARK": "base.dsv"}
    assert load_inputs(cfg, environ=pre).kaizo_save_pre_roark == Path("base.dsv")
    cfg.write_text("vanilla_rom = 'v.nds'\nkaizo_save_pre_roark = 'saves/base.dsv'\n")
    got = load_inputs(cfg, environ={})
    assert got.kaizo_save_pre_roark == REPO_ROOT / "saves" / "base.dsv"
    assert load_inputs(cfg, environ={"NUZLINE_VANILLA_ROM": "x.nds"}).vanilla_rom == Path("x.nds")


def test_missing_vanilla_names_key_and_env(tmp_path):
    with pytest.raises(RomInputError, match="vanilla_rom.*NUZLINE_VANILLA_ROM"):
        load_inputs(tmp_path / "absent.toml", environ={})


def test_backslash_basic_string_explained(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text('vanilla_rom = "G:\\Farry\\v.nds"\n')
    with pytest.raises(RomInputError, match="single-quoted"):
        load_inputs(cfg, environ={})


def test_unknown_key_raises(tmp_path):
    cfg = tmp_path / "local.toml"
    cfg.write_text("vanila_rom = 'v.nds'\n")
    with pytest.raises(RomInputError, match="vanila_rom"):
        load_inputs(cfg, environ={})


def test_read_checked(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert read_checked(p, hashlib.sha1(b"abc").hexdigest(), "sha1") == b"abc"
    with pytest.raises(RomInputError, match="ce81046e.*a9993e36"):  # expected, then actual
        read_checked(p, "ce81046eda7d232513069519cb2085349896dec7", "sha1")
    with pytest.raises(RomInputError, match="absent.bin"):
        read_checked(tmp_path / "absent.bin", "0" * 40, "sha1")


def test_example_config_parses():
    got = load_inputs(REPO_ROOT / "data" / "local.example.toml", environ={})
    assert got.vanilla_rom.suffix == ".nds" and got.kaizo_patch.suffix == ".xdelta"
    assert got.kaizo_save.suffix == ".sav"
    assert got.kaizo_save_pre_roark.suffix == ".dsv"


def test_vanilla_rom_hash(vanilla_bytes):  # skips without inputs
    assert len(vanilla_bytes) == 134217728 and vanilla_bytes[12:16] == b"CPUE"


def test_rom_fixture_skips_only_when_not_configured(tmp_path):
    from tests.tools.rom.conftest import inputs_or_skip

    with pytest.raises(pytest.skip.Exception, match="vanilla_rom is not set"):
        inputs_or_skip(tmp_path / "absent.toml", {})
    with pytest.raises(pytest.skip.Exception, match="absent"):
        inputs_or_skip(tmp_path / "absent.toml", {"NUZLINE_VANILLA_ROM": str(tmp_path / "v.nds")})
    bad = tmp_path / "local.toml"
    bad.write_text('vanilla_rom = "G:\\x"\n')  # a basic string with a bad escape
    with pytest.raises(RomInputError, match="literal strings"):
        inputs_or_skip(bad, {})
    bad.write_text("vanila_rom = 'x.nds'\n")
    with pytest.raises(RomInputError, match="unknown key"):
        inputs_or_skip(bad, {})
