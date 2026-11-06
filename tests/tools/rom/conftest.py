"""Shared fixtures for the ROM pipeline tests.

Every fixture that needs a local input skips when it is absent, like the decomp
tests in ``tests/tools/test_extract_decomp.py``.
"""

import os

import pytest

from tools.extract_decomp import extract_charmap
from tools.rom.inputs import REPO_ROOT, VANILLA_SHA1, RomInputNotSet, load_inputs, read_checked


@pytest.fixture(scope="session")
def rom_inputs():
    """The configured inputs; skips when no vanilla ROM is configured or present.

    Only "not configured" skips (no ``data/local.toml`` and no ``NUZLINE_VANILLA_ROM``, or a
    configured file that is absent): any other ``RomInputError``, such as a malformed
    ``local.toml``, fails, so it cannot silently skip every ROM test.
    """
    return inputs_or_skip(REPO_ROOT / "data" / "local.toml", os.environ)


def inputs_or_skip(config, environ):
    """``load_inputs(config, environ)``; skips only when ``vanilla_rom`` is not set
    (``RomInputNotSet``) or the configured file is absent."""
    try:
        inputs = load_inputs(config, environ)
    except RomInputNotSet as e:
        pytest.skip(str(e))
    if not inputs.vanilla_rom.exists():
        pytest.skip(f"configured vanilla ROM {inputs.vanilla_rom} is absent")
    return inputs


@pytest.fixture(scope="session")
def vanilla_bytes(rom_inputs):
    """The vanilla ROM bytes, SHA-1 checked."""
    return read_checked(rom_inputs.vanilla_rom, VANILLA_SHA1, "sha1")


@pytest.fixture(scope="session")
def decomp():
    """The pinned decomp checkout; skips when it is not cloned."""
    path = REPO_ROOT / "external" / "pokeplatinum"
    if not path.exists():
        pytest.skip("decomp not cloned")
    return path


@pytest.fixture(scope="session")
def charmap(decomp):
    """The decomp's font charmap, for decoding message banks."""
    return extract_charmap(decomp)


@pytest.fixture(scope="session")
def kaizo_inputs(rom_inputs):
    """The configured inputs with a Kaizo patch; skips when the patch is absent."""
    if rom_inputs.kaizo_patch is None or not rom_inputs.kaizo_patch.exists():
        pytest.skip("no kaizo_patch configured")
    return rom_inputs


@pytest.fixture(scope="session")
def kaizo_bytes(kaizo_inputs):
    """The patched Kaizo ROM bytes, built once into the default cache."""
    from tools.rom.patch import kaizo_rom

    path, _ = kaizo_rom(kaizo_inputs)
    return path.read_bytes()


@pytest.fixture(scope="session")
def vanilla_files(vanilla_bytes):
    """The vanilla ROM's unpacked file system."""
    pytest.importorskip("ndspy")
    from tools.rom.nds import RomFiles

    return RomFiles.open(vanilla_bytes)


@pytest.fixture(scope="session")
def kaizo_files(kaizo_bytes):
    """The patched Kaizo ROM's unpacked file system."""
    pytest.importorskip("ndspy")
    from tools.rom.nds import RomFiles

    return RomFiles.open(kaizo_bytes)


@pytest.fixture(scope="session")
def symbols():
    """The vanilla decomp build's linker-map symbols; skips when the map is not copied in."""
    from tools.rom.symbols import DEFAULT_MAP, SymbolMap

    if not DEFAULT_MAP.exists():
        pytest.skip(f"no linker map at {DEFAULT_MAP} (docs/rom_pipeline.md, Decomp build)")
    return SymbolMap.from_map(DEFAULT_MAP)


@pytest.fixture(scope="session")
def decomp_inter(decomp, tmp_path_factory):
    """The decomp extractor's static and trainer intermediate files, written once."""
    from tools.extract_decomp import extract_static, extract_trainers

    d = tmp_path_factory.mktemp("decomp_inter")
    extract_static(decomp, d)
    extract_trainers(decomp, d)
    return d
