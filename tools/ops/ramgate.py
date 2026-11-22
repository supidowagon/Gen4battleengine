"""Local test gate (docs/process.md section 5). Pure functions, the pytest gate and a small CLI.

Rule (owner, 22 Nov 2026): no engine tests on the PC. A local pytest run is allowed only when every
selected test is ROM-only (needs a local input the pod does not have) or the run is engine-free
(after collection `nuzline.engine` is not imported). Everything else goes to the pod through the
dispatcher's quick lane. An allowed run waits for free RAM > 6 GB + its weight (weights.txt, the
longest key matching an id; no key = 20 GB). The root conftest.py applies it at collection.

CLI, for `tools.kaizo.*` runs and table builds:
    python tools/ops/ramgate.py <ids> [--weight GB] [--rom-gate | --build]
    python tools/ops/ramgate.py --rom-ids [file ...]   (collect only; one ready command per file)
prints `GO <w>` (exit 0) after any wait, `POD <w>` (exit 3, weight >= 8), or `TIMEOUT <w>`
(exit 4, after NUZLINE_RAMGATE_TIMEOUT minutes, default 30). --build (table
builds, weight 20 unless --weight) never prints POD: it only waits for RAM.
"""

import ast
import importlib.abc
import inspect
import os
import shlex
import sys
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "tools" / "ops" / "day" / "pod" / "weights.txt"

DEFAULT_WEIGHT = 20.0
POD_WEIGHT = 8.0
RESERVE_GB = 6.0
POLL_S = 60.0
DEFAULT_TIMEOUT_MIN = 30.0

# Fixtures that need a local input the pod never gets (ROM, patch, linker map, saves, docs sheet,
# PKCalc archive). A test whose fixture closure (`item.fixturenames`) holds one is ROM-only. List
# from the e3 ROM-only test inventory of 21 Nov 2026 (collected with a plugin flagging exactly this).
ROM_FIXTURES = frozenset(
    {
        *("rom_inputs", "vanilla_bytes", "vanilla_files", "kaizo_inputs", "kaizo_bytes"),
        *("kaizo_files", "symbols", "kaizo_rom_dir", "kaizo_rom_trainers", "species"),
        *("kaizo_save_path", "kaizo_save_pre_roark_path", "kaizo_save_probe_path"),
        *("docs", "pkcalc_zip", "pkcalc"),
    }
)
# ROM-only tests the fixture closure cannot show (every param of a parametrized id counts):
# two skip on a missing docs extraction by path, and the test_cli save export gets its save
# through `request.getfixturevalue(which)`, so no fixture name in `item.fixturenames` shows it.
# Without it a local ROM-only run holding these ids is no longer all ROM-only, and the engine
# blocker then breaks the two test_cli tests next to it that import the engine in their body.
# (The pod skips 4 test_cli tests: these 2 params plus test_kaizo_prep_refusal_exits_2_before_the_cap
# and test_kaizo_pool_cap_is_never_defaulted, which hold kaizo_save_pre_roark_path.)
ROM_ONLY_IDS = frozenset(
    {
        "tests/test_cli.py::test_kaizo_save_export_checks_kaizo_tables",
        "tests/test_cli.py::test_kaizo_roark_coverage_on_the_saves_is_exact",
        "tests/kaizo/test_roark_ai_audit.py::test_roark_ai_against_the_pool",
        "tests/kaizo/test_roark_widths.py::test_roark_pairs_fit_the_width_cap",
        "tests/kaizo/test_overrides.py::test_overrides_check_runs_the_two_placement_rule",
        "tests/kaizo/test_sheet_unmapped.py::test_ally_reasons_name_the_partner_rows_starter",
    }
)
# Engine-free means none of these is imported once collection ends. `jax` is not listed:
# tests/conftest.py imports it for every run under tests/.
# A ROM-only item's pod weight was measured while it skipped, so a ROM-only item weighs
# ROM_UNMEASURED_GB until rom_weights.txt (same format as weights.txt, measured locally) has it.
ROM_UNMEASURED_GB = 20.0
ROM_WEIGHTS = ROOT / "tools" / "ops" / "day" / "rom_weights.txt"
ENGINE_PREFIX = "nuzline.engine"
MAX_LISTED = 5

ENV_ON_POD = "NUZLINE_ON_POD"
ENV_TIMEOUT = "NUZLINE_RAMGATE_TIMEOUT"
ENV_ROM_GATE = "NUZLINE_RAMGATE_ROM_GATE"
ENV_ENGINE_BLOCKED = "NUZLINE_ENGINE_BLOCKED"  # src/nuzline/engine/__init__.py refuses on it
ENV_DONE = "NUZLINE_RAMGATE_DONE"
ENV_NESTED = "PYTEST_CURRENT_TEST"  # set while a test runs: a pytest inside it is not gated

# Pytest options (and common plugin options) that take a separate value argument; the value is
# not a test id. A value of an option missing here reads as an id and weighs 20: it errs to POD.
_VALUE_OPTIONS = frozenset(
    {
        *("-m", "-k", "-p", "-o", "-c", "-n", "-W", "-r"),
        *("--durations", "--durations-min", "--maxfail", "--deselect", "--ignore"),
        *("--ignore-glob", "--rootdir", "--confcutdir", "--basetemp", "--junitxml"),
        *("--junit-xml", "--tb", "--color", "--import-mode", "--override-ini"),
        *("--log-level", "--log-file", "--log-format", "--log-cli-level"),
        *("--log-date-format", "--capture", "--assert", "--pythonwarnings"),
        *("--numprocesses", "--dist", "--verbosity", "--report-log", "--pastebin"),
        *("--doctest-glob", "--doctest-report", "--hypothesis-seed"),
        *("--hypothesis-profile", "--timeout", "--show-capture", "--junitprefix"),
        *("--junit-prefix", "--max-worker-restart", "--maxprocesses", "--tx", "--rsyncdir"),
    }
)


def load_weights(path: str | Path = WEIGHTS) -> dict[str, float]:
    """`<id> <GB> ...` lines of weights.txt; comments and unreadable lines are skipped."""
    out: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def lookup(ident: str, table: Mapping[str, float]) -> float | None:
    """Longest key that equals the id or that the id starts with plus "::", "[" or "/"."""
    ident = ident.replace("\\", "/")
    best: tuple[str, float] | None = None
    for key, gb in table.items():
        hit = ident == key or any(ident.startswith(key + s) for s in ("::", "[", "/"))
        if hit and (best is None or len(key) > len(best[0])):
            best = (key, gb)
    return best[1] if best else None


def weight(ident: str, table: Mapping[str, float]) -> float:
    w = lookup(ident, table)
    return DEFAULT_WEIGHT if w is None else w


def load_rom_weights(path: str | Path = ROM_WEIGHTS) -> dict[str, float]:
    return load_weights(path) if Path(path).exists() else {}


def _relative(arg: str, rootdir: Path, invocation_dir: Path | None) -> str:
    path, sep, node = arg.partition("::")
    p = Path(path)
    if not p.is_absolute():
        p = (invocation_dir or rootdir) / p
    try:
        rel = p.resolve().relative_to(rootdir.resolve()).as_posix()
    except ValueError:
        rel = path.replace("\\", "/")
    rel = rel.removeprefix("./").rstrip("/") or "."
    return rel + sep + node


def selected_ids(
    args: Iterable[str], rootdir: str | Path, invocation_dir: str | Path | None = None
) -> list[str]:
    """Path and node-id args of a pytest command line, repo-relative with forward slashes.

    Options and their values are dropped. An empty result means the whole tests/ tree."""
    root = Path(rootdir)
    inv = Path(invocation_dir) if invocation_dir else None
    ids: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
        elif a.startswith("-") and a != "-":
            skip = a in _VALUE_OPTIONS
        else:
            ids.append(_relative(a, root, inv))
    return ids


def run_weight(ids: Iterable[str], table: Mapping[str, float]) -> float:
    """Heaviest id; no ids = the whole tree = the heaviest entry of the table."""
    ids = list(ids)
    if not ids or ids == ["."]:
        return max(table.values(), default=DEFAULT_WEIGHT)
    return max(weight(i, table) for i in ids)


def decide(weight_gb: float, free: float, rom_gate: bool = False) -> str:
    if weight_gb >= POD_WEIGHT and not rom_gate:
        return "POD"
    if free > RESERVE_GB + weight_gb:
        return "GO"
    return "WAIT"


def free_gb() -> float:
    """Available physical RAM in GB (10^9 bytes)."""
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            raise OSError("GlobalMemoryStatusEx failed")
        return st.ullAvailPhys / 1e9
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    raise OSError("MemAvailable not found in /proc/meminfo")


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "") not in ("", "0")


def wait_for_ram(
    weight_gb: float,
    *,
    timeout_min: float,
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> bool:
    """Poll every 60 s until free RAM > 6 + weight; False after timeout_min minutes."""
    need = RESERVE_GB + weight_gb
    deadline = clock() + timeout_min * 60
    while True:
        f = free()
        if f > need:
            return True
        if clock() >= deadline:
            return False
        say(f"ramgate: {f:.1f} GB free, need > {need:g} GB (weight {weight_gb:g}); waiting")
        sleep(POLL_S)


def _direct_params(item) -> set[str]:
    """Argnames the item gets from a direct (non-indirect) parametrize: pytest gives them a
    pseudo fixture, so they are in `fixturenames` but are not the fixture of the same name."""
    spec = getattr(item, "callspec", None)
    info = getattr(item, "_fixtureinfo", None)
    if spec is None or info is None:
        return set()
    defs = info.name2fixturedefs
    return {
        n
        for n in spec.params
        if all(
            getattr(d.func, "__name__", "") == "get_direct_param_fixture_func"
            for d in defs.get(n, ())
        )
    }


def _fetches_rom_fixture(item) -> bool:
    """The test body calls request.getfixturevalue(...) with a ROM fixture name among the string
    literals of the call, so `getfixturevalue("a" if g else "b")` counts too. A name held in a
    variable is invisible here: those tests are listed in ROM_ONLY_IDS."""
    func = getattr(item, "function", None)
    if func is None:
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        return False
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "getfixturevalue":
            for node in ast.walk(call):
                if isinstance(node, ast.Constant) and node.value in ROM_FIXTURES:
                    return True
    return False


def is_rom_only(item) -> bool:
    """A test whose fixture closure holds a local-input fixture, or one of the path/id tests."""
    node = item.nodeid.replace("\\", "/")
    fixtures = set(item.fixturenames) - _direct_params(item)
    return (
        bool(ROM_FIXTURES & fixtures)
        or node.split("[")[0] in ROM_ONLY_IDS
        or _fetches_rom_fixture(item)
    )


def item_weight(item, table: Mapping[str, float], rom_table: Mapping[str, float]) -> float:
    """Pod weights come from runs where ROM-only tests skip: a ROM-only item weighs the figure of
    rom_weights.txt (measured locally), else 20 GB or its pod weight if higher."""
    w = weight(item.nodeid, table)
    if is_rom_only(item):
        measured = lookup(item.nodeid, rom_table)
        return measured if measured is not None else max(ROM_UNMEASURED_GB, w)
    return w


def engine_loaded(modules: Iterable[str]) -> bool:
    return any(m == ENGINE_PREFIX or m.startswith(ENGINE_PREFIX + ".") for m in modules)


def disallowed(items: Iterable, modules: Iterable[str]) -> list[str]:
    """Node ids the rule refuses: none when the run is engine-free, else every non-ROM-only item."""
    items = list(items)
    if not engine_loaded(modules):
        return []
    return [i.nodeid for i in items if not is_rom_only(i)]


QUICK_LANE_HINT = (
    "send at most 5 ids to the dispatcher's quick lane, run name q-<stream>-<sha7> "
    "(docs/process.md section 5)"
)


class EngineBlocker(importlib.abc.MetaPathFinder):
    """Refuses every import of `nuzline.engine` or a submodule, so a test that imports the engine
    at run time (in its body, a fixture or a lazy CLI path) fails loudly and loads nothing."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == ENGINE_PREFIX or fullname.startswith(ENGINE_PREFIX + "."):
            raise ImportError(f"ramgate: no engine tests on the PC: {fullname}: {QUICK_LANE_HINT}")


def blocker_wanted(items: Iterable, env: Mapping[str, str]) -> bool:
    """An engine-free run (not all ROM-only, no override) gets the finder."""
    return not _flag(env, ENV_ROM_GATE) and not all(is_rom_only(i) for i in items)


def install_blocker(meta_path: list | None = None) -> EngineBlocker:
    meta_path = sys.meta_path if meta_path is None else meta_path
    blocker = EngineBlocker()
    meta_path.insert(0, blocker)
    os.environ[ENV_ENGINE_BLOCKED] = "1"  # children: src/nuzline/engine/__init__.py refuses
    return blocker


def remove_blocker(blocker, meta_path: list | None = None) -> None:
    meta_path = sys.meta_path if meta_path is None else meta_path
    if blocker in meta_path:
        meta_path.remove(blocker)
    os.environ.pop(ENV_ENGINE_BLOCKED, None)


def pytest_gate(
    items: Iterable,
    *,
    collect_only: bool = False,
    modules: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
    table: Mapping[str, float] | None = None,
    rom_table: Mapping[str, float] | None = None,
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> tuple[str, str]:
    """The root conftest's decision after collection: (SKIP | GO | REFUSE | TIMEOUT, message)."""
    env = os.environ if env is None else env
    if _flag(env, ENV_ON_POD):
        return "SKIP", "on the pod"
    if _flag(env, ENV_DONE) or ENV_NESTED in env:
        return "SKIP", "inside a gated run"
    if collect_only:  # --collect-only only imports test modules: allowed, no wait
        return "GO", "ramgate: collect-only"
    bad = (
        []
        if _flag(env, ENV_ROM_GATE)
        else disallowed(items, sys.modules if modules is None else modules)
    )
    if bad:
        more = f" and {len(bad) - MAX_LISTED} more" if len(bad) > MAX_LISTED else ""
        return "REFUSE", (
            f"ramgate: no engine tests on the PC: {len(bad)} selected test(s) are neither ROM-only "
            f"nor engine-free, first: {', '.join(bad[:MAX_LISTED])}{more}. "
            f"{QUICK_LANE_HINT[0].upper() + QUICK_LANE_HINT[1:]}. "
            "Pass ROM-only node ids, never a file path (python tools/ops/ramgate.py --rom-ids <file>). "
            f"{ENV_ROM_GATE}=1 is a last resort: it also lets engine tests run on the PC."
        )
    table = load_weights() if table is None else table
    rom_table = load_rom_weights() if rom_table is None else rom_table
    w = max((item_weight(i, table, rom_table) for i in items), default=0.0)
    try:
        timeout = float(env.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_MIN))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_MIN
    if not wait_for_ram(w, timeout_min=timeout, free=free, sleep=sleep, clock=clock, say=say):
        return "TIMEOUT", (
            f"ramgate: free RAM stayed <= {RESERVE_GB + w:g} GB for {timeout:g} min "
            f"(weight {w:g} GB; {ENV_TIMEOUT} sets the minutes, weights in "
            "tools/ops/day/pod/weights.txt and tools/ops/day/rom_weights.txt). Run again later."
        )
    return "GO", f"ramgate: GO weight {w:g} GB"


def rom_ids_by_file(items: Iterable) -> dict[str, list[str]]:
    """Node ids of the ROM-only items, grouped by test file, in collection order."""
    out: dict[str, list[str]] = {}
    for it in items:
        if is_rom_only(it):
            out.setdefault(it.nodeid.split("::")[0], []).append(it.nodeid)
    return out


def rom_commands(by_file: Mapping[str, list[str]]) -> list[str]:
    """One ready command line per file: exact ids, never the file path (a path makes the run
    mixed, so the engine blocker goes on and the file's engine tests would be refused)."""
    return [
        "python -m pytest -p no:cacheprovider -q " + " ".join(shlex.quote(i) for i in ids)
        for ids in by_file.values()
    ]


def rom_ids(files: list[str], say: Callable[[str], None] = print) -> int:
    """`--rom-ids [file ...]`: collect only (no wait), print the commands for the ROM-only tests."""
    import pytest

    class Collect:
        def __init__(self):
            self.items: list = []

        def pytest_collection_modifyitems(self, items):
            self.items = list(items)

    plugin = Collect()
    code = pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider", *(files or ["tests"])], plugins=[plugin]
    )
    if code not in (0, 5):
        say(f"ramgate: collection failed (pytest exit {int(code)})")
        return 1
    for line in rom_commands(rom_ids_by_file(plugin.items)):
        say(line)
    return 0


def main(
    argv: list[str],
    *,
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = lambda m: print(m, flush=True),
) -> int:
    """CLI. `--build`: a table build cannot go to the pod, so it only waits for RAM."""
    if "--rom-ids" in argv:
        return rom_ids([a for a in argv if a != "--rom-ids"], say)
    rom_gate = "--rom-gate" in argv or _flag(os.environ, ENV_ROM_GATE)
    build = "--build" in argv
    argv = [a for a in argv if a not in ("--rom-gate", "--build")]
    w = None
    if "--weight" in argv:
        i = argv.index("--weight")
        try:
            w = float(argv[i + 1])
        except (IndexError, ValueError):
            say(
                "usage: ramgate.py <ids> [--weight GB] [--rom-gate | --build]; --weight needs a number"
            )
            return 2
        argv = argv[:i] + argv[i + 2 :]
    if w is None:
        table = load_weights()
        w = (
            DEFAULT_WEIGHT
            if build
            else max((weight(a, table) for a in argv), default=DEFAULT_WEIGHT)
        )
    if not build and decide(w, 0.0, rom_gate) == "POD":
        say(f"POD {w:g}")
        return 3
    try:
        timeout = float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_MIN))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_MIN
    if not wait_for_ram(w, timeout_min=timeout, free=free, sleep=sleep, clock=clock, say=say):
        say(f"TIMEOUT {w:g}")
        return 4
    say(f"GO {w:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
