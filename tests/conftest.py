"""Shared fixtures. Tables are loaded once per session, never at import time, and
every test that needs them skips when ``data/gen4_platinum/tables.npz`` is absent."""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nuzline.data.games import GAMES, KAIZO, VANILLA
from nuzline.data.tables import load_names, load_tables
from nuzline.jaxcache import (
    cache_root,
    cap_directory,
    jax_version,
    prune_other_versions,
    set_compile_cache_dir,
)

TABLES_DIR = Path(__file__).resolve().parents[1] / "data" / "gen4_platinum"

# Reruns of the suite skip XLA compilation (only the compilation cache: the
# serialized-trace cache is left as ``rank_actions`` uses it). Set at import, before
# any test compiles. Keyed by the jax and jaxlib versions only (``jax_version``): JAX keys
# each entry by the compiled program, so a commit keeps every program it leaves unchanged.
# Other versions' directories are deleted, and the oldest entries past 20 GB.
_COMPILE_CACHE = cache_root() / "jax-tests" / jax_version()
prune_other_versions(_COMPILE_CACHE)
cap_directory(_COMPILE_CACHE, 20 * 2**30)
set_compile_cache_dir(_COMPILE_CACHE)


def _resident_bytes() -> int:
    """This process's resident memory (working set on Windows), 0 when unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            c = Counters()
            c.cb = ctypes.sizeof(c)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            info = kernel32.K32GetProcessMemoryInfo
            info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            info.restype = wintypes.BOOL
            if info(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return int(c.WorkingSetSize)
            return 0
        with open("/proc/self/statm") as f:
            import os

            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, AttributeError, ValueError):
        return 0


# Above this resident size, JAX's in-memory programs are dropped after the current test.
_CLEAR_ABOVE = 6 * 2**30


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Drop JAX's in-memory compiled programs after each test module, and after any test that
    leaves the process above 6 GB resident. One process running the whole suite otherwise
    keeps every module's executables and reached 38 GB of commit (29 GB for tests/engine
    alone; 6 GB peak with this). The trainer-AI modules compile a program per layout (flag
    set, read bound and a fight's effects, E1 Rulings R2, R6, R7) and one of them alone passed
    20 GB (E1 batch 8); clearing after every test there made them re-trace every program, so
    the clear follows memory instead. The persistent compile cache keeps recompiles cheap."""
    yield
    next_module = nextitem is None or nextitem.module is not item.module
    if "jax" in sys.modules and (next_module or _resident_bytes() > _CLEAR_ABOVE):
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
def learnsets():
    from nuzline.data.learnsets import load_learnsets

    _require_tables()
    return load_learnsets(TABLES_DIR)


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


def _require_vanilla_intermediate() -> None:
    if not (GAMES[VANILLA].intermediate_dir / "trainers.json").exists():
        pytest.skip("vanilla intermediates not extracted (run python -m tools.extract_decomp)")


@pytest.fixture(scope="session")
def inter():
    """``IntermediateData`` of ``data/intermediate`` (skips when it is absent)."""
    from nuzline.data.trainer import IntermediateData

    _require_vanilla_intermediate()
    return IntermediateData(GAMES[VANILLA].intermediate_dir)


@pytest.fixture(scope="session")
def trainers():
    """Vanilla ``load_trainers(data/intermediate)`` (skips when it is absent)."""
    from nuzline.data.trainer import load_trainers

    _require_vanilla_intermediate()
    return load_trainers(GAMES[VANILLA].intermediate_dir)


# --- Kaizo saves (each skips when its save is not configured or absent) --------------


def _configured_save(key: str, env: str) -> Path:
    """The save ``RomInputs.<key>`` names; skips on ``RomInputError``, an unset key or a
    missing file. Not ``rom_inputs``: that gates on the vanilla ROM existing."""
    import os

    from tools.rom.inputs import REPO_ROOT, RomInputError, load_inputs

    try:
        path = getattr(load_inputs(REPO_ROOT / "data" / "local.toml", os.environ), key)
    except RomInputError as e:
        pytest.skip(str(e))
    if path is None:
        pytest.skip(f"{key} is not configured (data/local.toml or {env})")
    if not path.exists():
        pytest.skip(f"configured {key} {path} is absent")
    return path


@pytest.fixture(scope="session")
def kaizo_save_path():
    """The Maylene Kaizo save (``kaizo_save``)."""
    return _configured_save("kaizo_save", "NUZLINE_KAIZO_SAVE")


@pytest.fixture(scope="session")
def kaizo_save_pre_roark_path():
    """The pre-Roark Kaizo save (``kaizo_save_pre_roark``)."""
    return _configured_save("kaizo_save_pre_roark", "NUZLINE_KAIZO_SAVE_PRE_ROARK")


@pytest.fixture(scope="session")
def kaizo_save_probe_path():
    """The trace probe's Kaizo save (``kaizo_save_probe``): the party taken to Roark."""
    return _configured_save("kaizo_save_probe", "NUZLINE_KAIZO_SAVE_PROBE")


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


# The battle effects of the K3c Kaizo transcription: Kaizo Roark's party's, and the moves the
# K3c tests put in it (Leer, Screech, Pursuit, Tackle). The Kaizo tests run on this fight.
K3C_EFFECTS = frozenset(
    {
        0,
        2,
        3,
        4,
        6,
        7,
        19,
        31,
        32,
        33,
        35,
        52,
        59,
        69,
        72,
        76,
        109,
        128,
        140,
        186,
        261,
        266,
        269,
        275,
    }
)


@pytest.fixture(scope="session")
def kaizo_tables():
    """The Kaizo tables of a fight under Kaizo Roark's trainer AI flags 23 (``Tables.ai_flags``,
    E1 Ruling R2) against opponents with one Kaizo move of each ``K3C_EFFECTS`` effect
    (``Tables.ai_moves``, E1 Rulings R6, R7; ``ai.layout.layout_of`` refuses either unset), so
    the layout transcribes the effects the K3c tests use and no more: a program no wider than
    a real fight's (E1 batch 8)."""
    _require_kaizo_tables()
    tables = load_tables(KAIZO_TABLES_DIR)
    effect = np.asarray(tables.move_ai_effect)
    moves = [1 + int(np.flatnonzero(effect[1:] == e)[0]) for e in sorted(K3C_EFFECTS)]
    return tables._replace(ai_flags=np.int32(0b10111), ai_moves=np.asarray(moves, np.int16))


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
