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
    return next(t for t in kaizo_trainers if t.name == "Roark")


@pytest.fixture(scope="session")
def kaizo_roark_party(kaizo_roark):
    from nuzline.data.trainer import IntermediateData, build_trainer_party

    return build_trainer_party(kaizo_roark, IntermediateData(KAIZO_INTERMEDIATE))
