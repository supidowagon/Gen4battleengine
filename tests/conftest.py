"""Shared fixtures. Tables are loaded once per session, never at import time, and
every test that needs them skips when ``data/gen4_platinum/tables.npz`` is absent."""

import gc
import json
import sys
from pathlib import Path

import pytest

from nuzline.data.games import GAMES, KAIZO, VANILLA
from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import cache_root, code_version, prune_other_versions, set_compile_cache_dir

TABLES_DIR = Path(__file__).resolve().parents[1] / "data" / "gen4_platinum"

# Reruns of the suite skip XLA compilation (only the compilation cache: the
# serialized-trace cache is left as ``rank_actions`` uses it). Set at import, before
# any test compiles. Earlier versions' directories are deleted.
_COMPILE_CACHE = cache_root() / "jax-tests" / code_version()
prune_other_versions(_COMPILE_CACHE)
set_compile_cache_dir(_COMPILE_CACHE)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Drop JAX's in-memory compiled programs after each test module. One process
    running the whole suite otherwise keeps every module's executables and reached
    38 GB of commit (29 GB for tests/engine alone; 6 GB peak with this)."""
    yield
    if (nextitem is None or nextitem.module is not item.module) and "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
        gc.collect()


def _require_tables() -> None:
    if not (TABLES_DIR / "tables.npz").exists():
        pytest.skip("tables not built (run tools/build_tables.py)")


@pytest.fixture(scope="session")
def tables():
    _require_tables()
    return load_tables(TABLES_DIR)


@pytest.fixture(scope="session")
def names():
    _require_tables()
    return load_names(TABLES_DIR)


@pytest.fixture(scope="session")
def vanilla_manifest():
    _require_tables()
    return json.loads((TABLES_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def vanilla_items():
    """``data/intermediate/items.json`` rows (skips when the vanilla data is absent)."""
    path = GAMES[VANILLA].intermediate_dir / "items.json"
    if not path.exists():
        pytest.skip("vanilla intermediates not extracted (run python -m tools.extract_decomp)")
    return json.loads(path.read_text(encoding="utf-8"))


# --- Kaizo source extractions (each skips when its local input is absent) -----------


@pytest.fixture(scope="session")
def docs(tmp_path_factory):
    """``docs.json`` extracted once per session from the configured Kaizo docs workbook.

    Skips when ``load_inputs()`` raises ``RomInputError``, ``kaizo_docs`` is not set or
    absent, openpyxl is not importable, or the vanilla intermediates are missing.
    """
    from tools.rom.inputs import RomInputError, load_inputs

    try:
        path = load_inputs().kaizo_docs
    except RomInputError as e:
        pytest.skip(str(e))
    if path is None:
        pytest.skip("kaizo_docs is not configured (data/local.toml or NUZLINE_KAIZO_DOCS)")
    if not path.exists():
        pytest.skip(f"configured Kaizo docs workbook {path} is absent")
    pytest.importorskip("openpyxl")
    inter = GAMES[VANILLA].intermediate_dir
    if not (inter / "moves.json").exists() or not (inter / "abilities.json").exists():
        pytest.skip("vanilla intermediates not extracted (run python -m tools.extract_decomp)")
    from tools.extract_kaizo_docs import extract_kaizo_docs, load_vanilla

    moves, abilities = load_vanilla(inter)
    return extract_kaizo_docs(path, moves, abilities, tmp_path_factory.mktemp("docs"))


@pytest.fixture(scope="session")
def kaizo_rom_dir(tmp_path_factory):
    """``extract_rom.extract("kaizo", ...)`` output, written once per session.

    Skips when ``load_inputs()`` raises ``RomInputError``, the vanilla ROM or Kaizo patch
    is not set or absent, ndspy is not importable, or the decomp or its linker map is
    missing. ``data/kaizo/rom_diff.toml`` is only read (``extract`` never appends to it).
    """
    from tools.rom.inputs import REPO_ROOT, RomInputError, load_inputs

    try:
        inputs = load_inputs()
    except RomInputError as e:
        pytest.skip(str(e))
    if not inputs.vanilla_rom.exists():
        pytest.skip(f"configured vanilla ROM {inputs.vanilla_rom} is absent")
    if inputs.kaizo_patch is None or not inputs.kaizo_patch.exists():
        pytest.skip("no kaizo_patch configured")
    pytest.importorskip("ndspy")
    from tools.rom.symbols import DEFAULT_MAP

    decomp = REPO_ROOT / "external" / "pokeplatinum"
    if not decomp.exists():
        pytest.skip("decomp not cloned")
    if not DEFAULT_MAP.exists():
        pytest.skip(f"no linker map at {DEFAULT_MAP}")
    from tools.extract_rom import extract

    out = tmp_path_factory.mktemp("kaizo_rom") / "rom"
    extract("kaizo", out, decomp, DEFAULT_MAP, inputs)
    return out


@pytest.fixture(scope="session")
def kaizo_rom_trainers(kaizo_rom_dir):
    """The Kaizo ROM's ``trainers.json`` rows (``kaizo_rom_dir``)."""
    return json.loads((kaizo_rom_dir / "trainers.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def species(kaizo_rom_dir):
    """The Kaizo ROM's ``species.json`` rows (``kaizo_rom_dir``)."""
    return json.loads((kaizo_rom_dir / "species.json").read_text(encoding="utf-8"))


# --- Kaizo: every fixture skips until the Kaizo tables are built (Task 15) ----------

KAIZO_TABLES_DIR = GAMES[KAIZO].tables_dir
KAIZO_INTERMEDIATE = GAMES[KAIZO].intermediate_dir


def _require_kaizo_tables() -> None:
    if not (KAIZO_TABLES_DIR / "tables.npz").exists():
        pytest.skip("Kaizo tables not built (run python -m tools.build_tables --game kaizo)")


@pytest.fixture(scope="session")
def kaizo_tables():
    _require_kaizo_tables()
    return load_tables(KAIZO_TABLES_DIR)


@pytest.fixture(scope="session")
def kaizo_names():
    _require_kaizo_tables()
    return load_names(KAIZO_TABLES_DIR)


@pytest.fixture(scope="session")
def kaizo_manifest():
    _require_kaizo_tables()
    return json.loads((KAIZO_TABLES_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def kaizo_learnsets():
    from nuzline.data.learnsets import load_learnsets

    _require_kaizo_tables()
    return load_learnsets(KAIZO_TABLES_DIR)


@pytest.fixture(scope="session")
def kaizo_trainers():
    from nuzline.data.trainer import load_trainers

    _require_kaizo_tables()
    return load_trainers(KAIZO_INTERMEDIATE)


@pytest.fixture(scope="session")
def kaizo_roark(kaizo_trainers):
    """Kaizo's Roark: the first trainer named Roark, as in vanilla."""
    roark = next((t for t in kaizo_trainers if t.name == "Roark"), None)
    if roark is None:
        pytest.fail("no trainer named Roark in the Kaizo trainers")
    return roark


@pytest.fixture(scope="session")
def kaizo_roark_party(kaizo_roark):
    from nuzline.data.trainer import IntermediateData, build_trainer_party

    return build_trainer_party(kaizo_roark, IntermediateData(KAIZO_INTERMEDIATE))
