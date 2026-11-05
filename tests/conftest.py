"""Shared fixtures. Tables are loaded once per session, never at import time, and
every test that needs them skips when ``data/gen4_platinum/tables.npz`` is absent."""

import gc
import sys
from pathlib import Path

import pytest

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
