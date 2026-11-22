"""Local test gate (docs/process.md section 5). Pure functions, the pytest gate and a small CLI.

Rule (owner, 22 Nov 2026): no engine tests on the PC. A local pytest run is allowed only when every
selected test is ROM-only (needs a local input the pod does not have) or the run is engine-free
(after collection `nuzline.engine` is not imported). Everything else goes to the pod through the
dispatcher's quick lane. An allowed run waits for free RAM > 6 GB + its weight (weights.txt, the
longest key matching an id; no key = 20 GB). The root conftest.py applies it at collection.

CLI, for `tools.kaizo.*` runs and table builds:
    python tools/ops/ramgate.py <ids> [--weight GB] [--rom-gate | --build]
prints `GO <w>` (exit 0) after any wait, or `POD <w>` (exit 3, weight >= 8). --build (table
builds, weight 20 unless --weight) never prints POD: it only waits for RAM.
"""

import os
import sys
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
# The two ROM-only tests that skip on a missing docs extraction by path, not through a fixture.
ROM_ONLY_IDS = frozenset(
    {
        "tests/kaizo/test_overrides.py::test_overrides_check_runs_the_two_placement_rule",
        "tests/kaizo/test_sheet_unmapped.py::test_ally_reasons_name_the_partner_rows_starter",
    }
)
# Engine-free means none of these is imported once collection ends. `jax` is not listed:
# tests/conftest.py imports it for every run under tests/.
ENGINE_PREFIX = "nuzline.engine"
MAX_LISTED = 5

ENV_ON_POD = "NUZLINE_ON_POD"
ENV_TIMEOUT = "NUZLINE_RAMGATE_TIMEOUT"
ENV_ROM_GATE = "NUZLINE_RAMGATE_ROM_GATE"
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


def weight(ident: str, table: Mapping[str, float]) -> float:
    ident = ident.replace("\\", "/")
    best: tuple[str, float] | None = None
    for key, gb in table.items():
        hit = ident == key or any(ident.startswith(key + s) for s in ("::", "[", "/"))
        if hit and (best is None or len(key) > len(best[0])):
            best = (key, gb)
    return best[1] if best else DEFAULT_WEIGHT


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


def is_rom_only(item) -> bool:
    """A test whose fixture closure holds a local-input fixture, or one of the two path tests."""
    node = item.nodeid.replace("\\", "/")
    return bool(ROM_FIXTURES.intersection(item.fixturenames)) or node.split("[")[0] in ROM_ONLY_IDS


def engine_loaded(modules: Iterable[str]) -> bool:
    return any(m == ENGINE_PREFIX or m.startswith(ENGINE_PREFIX + ".") for m in modules)


def disallowed(items: Iterable, modules: Iterable[str]) -> list[str]:
    """Node ids the rule refuses: none when the run is engine-free, else every non-ROM-only item."""
    items = list(items)
    if not engine_loaded(modules):
        return []
    return [i.nodeid for i in items if not is_rom_only(i)]


def pytest_gate(
    items: Iterable,
    args: Iterable[str],
    rootdir: str | Path,
    *,
    invocation_dir: str | Path | None = None,
    collect_only: bool = False,
    modules: Iterable[str] | None = None,
    env: Mapping[str, str] | None = None,
    table: Mapping[str, float] | None = None,
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
            f"nor engine-free, first: {', '.join(bad[:MAX_LISTED])}{more}. Send at most 5 ids to the "
            "dispatcher's quick lane, run name q-<stream>-<sha7> (docs/process.md section 5). "
            f"{ENV_ROM_GATE}=1 is for the merge gate's ROM-only run only."
        )
    ids = selected_ids(args, rootdir, invocation_dir)
    w = run_weight(ids, load_weights() if table is None else table)
    try:
        timeout = float(env.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_MIN))
    except ValueError:
        timeout = DEFAULT_TIMEOUT_MIN
    if not wait_for_ram(w, timeout_min=timeout, free=free, sleep=sleep, clock=clock, say=say):
        return "TIMEOUT", (
            f"ramgate: free RAM stayed <= {RESERVE_GB + w:g} GB for {timeout:g} min "
            f"(weight {w:g} GB). Run again later."
        )
    return "GO", f"ramgate: GO weight {w:g} GB"


def main(
    argv: list[str],
    *,
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = lambda m: print(m, flush=True),
) -> int:
    """CLI. `--build`: a table build cannot go to the pod, so it only waits for RAM."""
    rom_gate = "--rom-gate" in argv or _flag(os.environ, ENV_ROM_GATE)
    build = "--build" in argv
    argv = [a for a in argv if a not in ("--rom-gate", "--build")]
    w = None
    if "--weight" in argv:
        i = argv.index("--weight")
        w = float(argv[i + 1])
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
    wait_for_ram(w, timeout_min=float("inf"), free=free, sleep=sleep, clock=clock, say=say)
    say(f"GO {w:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
