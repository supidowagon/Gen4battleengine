"""Patch application, the verified cache and the patched header (plan K1 Task 3)."""

import hashlib

import pytest

import tools.rom.patch as patch_mod
from tools.rom.inputs import KAIZO_PATCH_SHA256, VANILLA_SHA1, RomInputError, RomInputs
from tools.rom.patch import expected_patched, kaizo_rom, rom_header


def test_rom_toml_matches_inputs_constants():
    exp = expected_patched()
    assert (exp["vanilla_sha1"], exp["patch_sha256"]) == (VANILLA_SHA1, KAIZO_PATCH_SHA256)


def test_wrong_vanilla_refused_before_patching(tmp_path):
    v = tmp_path / "v.nds"
    v.write_bytes(b"x")
    p = tmp_path / "k.xdelta"
    p.write_bytes(b"y")
    with pytest.raises(RomInputError, match="v.nds.*0862ec35"):
        kaizo_rom(RomInputs(v, p, None, None, None), tmp_path / "c")
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())


def test_missing_patch_refused(tmp_path):
    with pytest.raises(RomInputError, match="kaizo_patch"):
        kaizo_rom(RomInputs(tmp_path / "v.nds", None, None, None, None), tmp_path)


def test_kaizo_rom_and_header(kaizo_inputs, tmp_path):
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    exp = expected_patched()
    h = rom_header(path.read_bytes())
    assert sha == exp["sha1"] and path.stat().st_size == exp["size"]
    assert (h.title, h.maker, h.revision) == ("POKEMON PL", "01", 1)
    assert h.game_code == exp["game_code"]


def _count_builds(monkeypatch):
    calls = []
    real = patch_mod.apply_vcdiff
    monkeypatch.setattr(patch_mod, "apply_vcdiff", lambda s, p: calls.append(1) or real(s, p))
    return calls


def test_cache_hit_and_corrupt_cache_rebuilt(kaizo_inputs, tmp_path, monkeypatch):
    calls = _count_builds(monkeypatch)
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 1
    path.write_bytes(path.read_bytes()[:-1])
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 2
    assert hashlib.sha1(path.read_bytes()).hexdigest() == sha


def test_foreign_sidecar_rebuilt(kaizo_inputs, tmp_path, monkeypatch):
    calls = _count_builds(monkeypatch)
    path, sha = kaizo_rom(kaizo_inputs, tmp_path)
    sidecar = path.with_name(path.name + ".sha1")
    sidecar.write_text("0" * 40)
    assert kaizo_rom(kaizo_inputs, tmp_path) == (path, sha) and len(calls) == 2
    assert sidecar.read_text().strip() == sha
