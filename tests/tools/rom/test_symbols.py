"""Tests for tools.rom.symbols: the decomp build's Metrowerks linker map."""

import json
import re
from pathlib import Path

import pytest

from tools.rom.inputs import REPO_ROOT
from tools.rom.symbols import DECOMP_COMMIT, Symbol, SymbolMap, SymbolMapError

FIXTURES = Path(__file__).parent / "fixtures"


def _code_at(rom, module: str, address: int, n: int) -> bytes:
    """``n`` bytes at RAM ``address`` of ``module`` ("arm9" or "overlay<N>")."""
    if module == "arm9":
        for ram, data in rom.arm9_sections():
            if ram <= address and address + n <= ram + len(data):
                return data[address - ram : address - ram + n]
        raise AssertionError(f"0x{address:08X} is in no ARM9 section")
    i = int(module.removeprefix("overlay"))
    off = address - rom.overlay_info(i)[0]
    return rom.overlay(i)[off : off + n]


def test_excerpt_parses():
    m = SymbolMap.from_map(FIXTURES / "xmap_excerpt.txt")
    assert m.lookup("arm9", 0x020793B8 + 0x40F) == "TrainerData_BuildParty"
    assert m.lookup("arm9", 0x020793A8) == "Trainer_LoadParty"  # the $d label is skipped
    assert m.lookup("arm9", 0x020793B0) is None  # between symbols (cut from the excerpt)
    assert m.lookup("arm9", 0x020793B8 + 0x410) is None  # one past the end
    assert m.lookup("arm9", 0x01FF8010) == "OS_IrqHandler"
    assert m.find("OS_IrqHandler").section == ".itcm"
    assert m.lookup("overlay14", 0x02222E84) == "TrainerAI_CalcDamage"
    assert m.lookup("overlay16", 0x0226ECD4 + 0x14F) == "sTypeMatchupMultipliers"
    assert m.lookup("overlay16", 0x02222E84) is None  # wrong module
    s = m.find("TrainerData_BuildParty")
    assert s == Symbol("arm9", 0x020793B8, 0x410, "TrainerData_BuildParty",
                       "src_trainer_data.c.o", ".text")
    assert [x.name for x in m.symbols("overlay14")] == ["TrainerAI_Init", "TrainerAI_CalcDamage"]
    with pytest.raises(KeyError):
        m.find("$t")  # ambiguous
    with pytest.raises(KeyError):
        m.find("NoSuchSymbol")
    assert "v_addr" not in {x.name for x in m.symbols("arm9")}  # memory map not parsed


def test_symbol_before_overlay_id_raises(tmp_path):
    bad = tmp_path / "bad.xMAP"
    bad.write_text("# .battle\n  0223B140 00000244 .text   Battle_Main\t(a.o)\n")
    with pytest.raises(SymbolMapError, match="battle"):
        SymbolMap.from_map(bad)


def test_unparsed_line_raises(tmp_path):
    bad = tmp_path / "bad.xMAP"
    bad.write_text("# .main\n  garbage line\n")
    with pytest.raises(SymbolMapError, match="2"):
        SymbolMap.from_map(bad)


def test_decomp_commit_pinned():
    manifest = json.loads((REPO_ROOT / "data/gen4_platinum/manifest.json").read_text(encoding="utf-8"))
    assert DECOMP_COMMIT == manifest["decomp_commit"]
    script = (REPO_ROOT / "tools/clone_decomp.sh").read_text(encoding="utf-8")
    pinned = re.search(r'^PINNED_COMMIT="([0-9a-f]{40})"\r?$', script, re.MULTILINE)
    assert pinned and pinned[1] == DECOMP_COMMIT


def test_zero_size_and_alias_lookup():
    m = SymbolMap([Symbol("arm9", 0x100, 0, "Label", "a.o", ".text"),
                   Symbol("arm9", 0x100, 0x20, "Outer", "a.o", ".text"),
                   Symbol("arm9", 0x100, 0x10, "Inner", "a.o", ".text")])
    assert m.lookup("arm9", 0x104) == "Inner" and m.lookup("arm9", 0x118) == "Outer"
    assert m.lookup("arm9", 0x120) is None and m.find("Label").size == 0
    assert m.lookup("arm9", 0x100) == "Inner"


def test_alias_tie_goes_to_first():
    m = SymbolMap([Symbol("arm9", 0x100, 0x10, "First", "a.o", ".text"),
                   Symbol("arm9", 0x100, 0x10, "Second", "a.o", ".text")])
    assert m.lookup("arm9", 0x108) == "First"


def test_known_symbols(symbols):
    assert symbols.find("TrainerData_BuildParty").module == "arm9"
    assert symbols.find("TrainerAI_CalcDamage").module == "overlay14"  # trainer_ai, main.lsf order
    assert symbols.find("sTypeMatchupMultipliers").module == "overlay16"  # battle
    for name in ("sMapHeaders", "sYearlyWeather", "sTrainerClassGender", "sItemArchiveIDs",
                 "sNatureStatAffinities"):
        assert symbols.find(name).module == "arm9"
    assert symbols.find("gVsSeekerRematchData").module == "overlay5"


def test_overlay_symbols_inside_rom_overlays(symbols, vanilla_files):
    for i in vanilla_files.overlay_ids():
        ram, size, bss = vanilla_files.overlay_info(i)
        assert all(ram <= s.address and s.address + s.size <= ram + size + bss
                   for s in symbols.symbols(f"overlay{i}"))
    assert {f"overlay{i}" for i in vanilla_files.overlay_ids()} >= set(symbols.modules()) - {"arm9"}


def test_arm9_symbols_inside_rom_sections(symbols, vanilla_files):
    # The map against the ROM, not against itself. BSS (".bss", ".dtcm.bss") has no ROM bytes.
    ranges = [(ram, ram + len(data)) for ram, data in vanilla_files.arm9_sections()]
    for s in symbols.symbols("arm9"):
        if s.size and not s.section.endswith("bss"):
            assert any(lo <= s.address and s.address + s.size <= hi for lo, hi in ranges), s


def test_known_functions_start_with_thumb_push(symbols, vanilla_files):
    for name in ("TrainerData_BuildParty", "TrainerAI_CalcDamage"):
        s = symbols.find(name)
        first = _code_at(vanilla_files, s.module, s.address, 2)
        assert first[1] == 0xB5, (name, first.hex())  # Thumb push {..., lr}, little-endian
