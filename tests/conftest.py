"""Shared fixtures. Tables are loaded once per session, never at import time, and
every test that needs them skips when ``data/gen4_platinum/tables.npz`` is absent."""

from pathlib import Path

import pytest

from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import cache_root, code_version, set_compile_cache_dir

TABLES_DIR = Path(__file__).resolve().parents[1] / "data" / "gen4_platinum"

# Reruns of the suite skip XLA compilation (only the compilation cache: the
# serialized-trace cache is left as ``rank_actions`` uses it). Set at import, before
# any test compiles.
set_compile_cache_dir(cache_root() / "jax-tests" / code_version())


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
