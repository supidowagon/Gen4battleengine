"""Tests for tools.rom.nds against the real vanilla and Kaizo ROMs."""

import subprocess
import sys
from pathlib import Path

import pytest

import tools.rom


def test_vanilla_layout(vanilla_files):
    assert vanilla_files.game_code == "CPUE"
    assert len(vanilla_files.paths()) == 340
    assert len(vanilla_files.overlay_ids()) == 122
    assert len(vanilla_files.arm9()) == 0x1023F8
    assert vanilla_files.overlay_info(16)[:2] == (0x223B140, 0x35A80)
    # sizeof(SpeciesData)
    assert len(vanilla_files.narc("poketool/personal/pl_personal.narc")[1]) == 44
    assert len(vanilla_files.header()) == 0x200
    assert vanilla_files.paths() == sorted(vanilla_files.paths())
    sections = vanilla_files.arm9_sections()
    assert sections and sections[0][1]  # static section first
    assert sum(len(d) for _, d in sections) >= len(vanilla_files.arm9()) - 0x100


def test_kaizo_layout(kaizo_files):
    assert kaizo_files.game_code == "JAK7"
    assert len(kaizo_files.paths()) == 2091
    assert len(kaizo_files.overlay(16)) == 0x35AA0


def test_missing_path_names_it(vanilla_files):
    with pytest.raises(KeyError, match="nope/x.narc"):
        vanilla_files.file("nope/x.narc")


def test_rom_package_imports_without_ndspy():
    package = Path(tools.rom.__file__).parent
    names = sorted(p.stem for p in package.glob("*.py") if p.stem != "__init__")
    modules = ["tools.rom"] + [f"tools.rom.{n}" for n in names]
    code = (
        "import sys, importlib\n"
        "sys.modules['ndspy'] = None\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package.parent.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
