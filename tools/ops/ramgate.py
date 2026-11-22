"""Local test gate (docs/process.md section 5). Pure functions, the pytest gate and a small CLI.

Rule (owner, 22 Nov 2026): no engine tests on the PC. A local pytest run is allowed only when every
selected test is ROM-only (needs a local input the pod does not have) or the run is engine-free
(after collection `nuzline.engine` is not imported). Everything else goes to the pod through the
dispatcher's quick lane. An allowed run waits for free RAM > 6 GB + its weight (weights.txt, the
longest key matching an id; no key = 6 GB in an engine-free run, else 20 GB). The root conftest.py
applies it at collection.

One local run at a time on the PC: after the RAM check a gated run takes the machine lock
(`<main checkout>/_scratch/ramgate.lock`, MachineLock) and holds it until it ends; waiting on it
counts toward the same timeout.

CLI:
    python tools/ops/ramgate.py --build [--weight GB] -- <command ...>
        a local job that is not pytest (table builds, `tools.kaizo.*` runs): waits for free RAM >
        6 GB + weight (--weight, else the weights.txt line `tools/build_tables.py`, else 14), takes
        the machine lock, runs the command as its child under the lock and exits with its code.
    python tools/ops/ramgate.py <ids> [--weight GB] [--rom-gate]
        a check only (no lock): `GO <w>` (exit 0) after any wait, `POD <w>` (weight >= 8).
    python tools/ops/ramgate.py --rom-ids [file ...]   (collect only; one ready command per file)
`POD` exits 13, `TIMEOUT <w>` (after NUZLINE_RAMGATE_TIMEOUT minutes, default 30) exits 14, an
unreadable free-RAM figure exits 16. The pytest gate exits 13 (refused), 14 (timeout) and 16 too:
codes apart from pytest's own 0-5.
"""

import ast
import importlib.abc
import inspect
import math
import os
import shlex
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "tools" / "ops" / "day" / "pod" / "weights.txt"

DEFAULT_WEIGHT = 20.0  # an item with no weights.txt key in a run that loads the engine
ENGINE_FREE_WEIGHT = 6.0  # the same in an engine-free run (the blocker is on): what tools tests measure
POD_WEIGHT = 8.0
RESERVE_GB = 6.0
BUILD_KEY = "tools/build_tables.py"  # the weights.txt line a table build waits for
BUILD_DEFAULT_WEIGHT = 14.0  # without that line: a build waits for > 6 + 14 = 20 GB free
POLL_S = 60.0
EXIT_REFUSE = 13  # refused (pytest) or POD (CLI)
EXIT_TIMEOUT = 14
EXIT_NO_RAM_READ = 16  # free_gb raised OSError
LOCK_POLL_S = 10.0  # free RAM is enough but another local run holds the machine lock
LOCK_NAME = "ramgate.lock"  # under the main checkout's _scratch/: one local run on the PC at a time
LOCK_EMPTY_STALE_S = 10.0  # a lock file without a pid (its writer died between create and write)
DEFAULT_TIMEOUT_MIN = 30.0

# Fixtures that need a local input the pod never gets (ROM, patch, linker map, saves, docs sheet,
# PKCalc archive). A test whose fixture closure (`item.fixturenames`) holds one, defined in one of
# ROM_FIXTURE_MODULES, is ROM-only. List from the e3 ROM-only test inventory of 21 Nov 2026
# (collected with a plugin flagging exactly this).
ROM_FIXTURES = frozenset(
    {
        *("rom_inputs", "vanilla_bytes", "vanilla_files", "kaizo_inputs", "kaizo_bytes"),
        *("kaizo_files", "symbols", "kaizo_rom_dir", "kaizo_rom_trainers", "species"),
        *("kaizo_save_path", "kaizo_save_pre_roark_path", "kaizo_save_probe_path"),
        *("docs", "pkcalc_zip", "pkcalc"),
    }
)
ROM_FIXTURE_MODULES = frozenset({"tests.conftest", "tests.tools.rom.conftest", "tests.kaizo.conftest"})
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
ROM_UNMEASURED_GB = 14.0  # an unmeasured ROM-only item waits for > 6 + 14 = 20 GB free
ROM_WEIGHTS = ROOT / "tools" / "ops" / "day" / "rom_weights.txt"
POD_ONLY = ROOT / "tools" / "ops" / "day" / "pod_only.txt"  # refused on the PC whatever they import
ENGINE_PREFIX = "nuzline.engine"
MAX_LISTED = 5

ENV_ON_POD = "NUZLINE_ON_POD"
ENV_TIMEOUT = "NUZLINE_RAMGATE_TIMEOUT"
ENV_ROM_GATE = "NUZLINE_RAMGATE_ROM_GATE"
ENV_ENGINE_BLOCKED = "NUZLINE_ENGINE_BLOCKED"  # src/nuzline/engine/__init__.py refuses on it
# Honoured only while a live ancestor holds the machine lock as kind "measure" (the measuring
# harness, tools/ops/ramgate_measure.py, which checks RAM and runs a watchdog itself).
ENV_NO_WAIT = "NUZLINE_RAMGATE_NO_WAIT"
ENV_DONE = "NUZLINE_RAMGATE_DONE"  # the gated run's pid; honoured only when it is a live ancestor

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


def weight(ident: str, table: Mapping[str, float], default: float = DEFAULT_WEIGHT) -> float:
    w = lookup(ident, table)
    return default if w is None else w


def load_rom_weights(path: str | Path = ROM_WEIGHTS) -> dict[str, float]:
    return load_weights(path) if Path(path).exists() else {}


def load_pod_only(path: str | Path = POD_ONLY) -> frozenset[str]:
    """`<id> <reason>` lines of pod_only.txt (comments and blank lines skipped)."""
    return frozenset(
        line.split()[0]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


def pod_only_ids(items: Iterable, keys: Iterable[str]) -> list[str]:
    """Node ids under a pod_only.txt key (same matching as a weights.txt key)."""
    table = dict.fromkeys(keys, 0.0)
    return [i.nodeid for i in items if lookup(i.nodeid, table) is not None]


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


def timeout_minutes(env: Mapping[str, str]) -> float:
    """NUZLINE_RAMGATE_TIMEOUT minutes; unreadable, nan, infinite or negative = the default 30."""
    try:
        minutes = float(env.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_MIN))
    except ValueError:
        return DEFAULT_TIMEOUT_MIN
    return minutes if math.isfinite(minutes) and minutes >= 0 else DEFAULT_TIMEOUT_MIN


def on_pod(env: Mapping[str, str], platform: str = sys.platform) -> bool:
    """NUZLINE_ON_POD counts on Linux only: the pods are Linux, the PC is Windows."""
    return platform != "win32" and _flag(env, ENV_ON_POD)


def process_table() -> dict[int, int]:
    """pid -> parent pid of every live process (Windows: a Toolhelp32 snapshot; else /proc)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Entry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Entry)]
        k32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Entry)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if snap in (None, wintypes.HANDLE(-1).value):
            raise OSError("CreateToolhelp32Snapshot failed")
        out: dict[int, int] = {}
        try:
            entry = _Entry()
            entry.dwSize = ctypes.sizeof(_Entry)
            ok = k32.Process32First(snap, ctypes.byref(entry))
            while ok:
                out[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                ok = k32.Process32Next(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(snap)
        return out
    out = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                stat = Path("/proc", name, "stat").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            out[int(name)] = int(stat.rsplit(")", 1)[1].split()[1])
    return out


def lineage(pid: int, procs: Mapping[int, int]) -> list[int]:
    """The pid and its live ancestors, nearest first (a pid cycle or a dead parent ends it)."""
    out: list[int] = []
    while pid in procs and pid not in out and pid > 0:
        out.append(pid)
        pid = procs[pid]
    return out


def main_checkout(root: Path = ROOT) -> Path:
    """The main checkout of a git worktree (its .git file -> gitdir -> commondir), else root."""
    try:
        dot = root / ".git"
        if dot.is_file():
            text = dot.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text[len("gitdir:") :].strip())
                gitdir = gitdir if gitdir.is_absolute() else root / gitdir
                common = (gitdir / "commondir").read_text(encoding="utf-8").strip()
                common_dir = Path(common) if Path(common).is_absolute() else gitdir / common
                return common_dir.resolve().parent
    except OSError:
        pass
    return root


def lock_path(root: Path = ROOT) -> Path:
    return main_checkout(root) / "_scratch" / LOCK_NAME


class MachineLock:
    """One local gated run on the PC at a time: `<main checkout>/_scratch/ramgate.lock`, created
    with O_EXCL, holding `<pid> <kind>`. Stale (removed, then retried) when that pid is gone.
    A process whose ancestor (or itself) holds it runs under that hold: `inherited`, no file."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        procs: Callable[[], Mapping[int, int]] = process_table,
        me: int | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.path = Path(path) if path is not None else lock_path()
        self._procs = procs
        self.me = os.getpid() if me is None else me
        self._now = now
        self.owned = False
        self.inherited = False

    def in_lineage(self, pid: int) -> bool:
        """pid is this process or one of its live ancestors."""
        return pid in lineage(self.me, self._procs())

    @property
    def held(self) -> bool:
        return self.owned or self.inherited

    def holder(self) -> tuple[int | None, str]:
        """(pid, kind) written in the file; (None, "") when absent, empty or unreadable."""
        try:
            parts = self.path.read_text(encoding="utf-8").split()
            return int(parts[0]), (parts[1] if len(parts) > 1 else "")
        except (OSError, ValueError, IndexError):
            return None, ""

    def held_by_lineage(self, kind: str | None = None) -> bool:
        """This process or a live ancestor holds the file (of that kind, when given)."""
        pid, k = self.holder()
        if pid is None or (kind is not None and k != kind):
            return False
        return self.in_lineage(pid)

    def acquire(self, kind: str = "pytest") -> bool:
        """Take the lock once (no waiting). True when it is now held by this process or its line."""
        if self.held:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{self.me} {kind}\n")
                self.owned = True
                return True
            pid, _ = self.holder()
            procs = self._procs()
            if pid is not None and pid in lineage(self.me, procs):
                self.inherited = True
                return True
            if pid is None:
                try:
                    age = self._now() - self.path.stat().st_mtime
                except OSError:
                    continue  # removed meanwhile: try again
                if age < LOCK_EMPTY_STALE_S:
                    return False  # being written right now
            elif pid in procs:
                return False
            self._remove_if_holder(pid)
        return False

    def _remove_if_holder(self, pid: int | None) -> None:
        if self.holder()[0] == pid:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def release(self) -> None:
        if self.owned:
            self._remove_if_holder(self.me)
        self.owned = self.inherited = False


def wait_for_ram(
    weight_gb: float,
    *,
    timeout_min: float,
    lock: MachineLock | None = None,
    kind: str = "pytest",
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> bool:
    """Poll every 60 s until free RAM > 6 + weight, then take the machine lock (polled every 10 s
    while another run holds it, RAM re-read each time). The lock wait counts toward the same
    timeout: False after timeout_min minutes."""
    need = RESERVE_GB + weight_gb
    deadline = clock() + timeout_min * 60
    told = None
    while True:
        f = free()
        if f > need:
            if lock is None or lock.acquire(kind):
                return True
            if clock() >= deadline:
                return False
            pid, _ = lock.holder()
            if pid != told:
                say(f"ramgate: another local run (pid {pid}) holds {lock.path}; waiting")
                told = pid
            sleep(LOCK_POLL_S)
            continue
        if clock() >= deadline:
            return False
        say(f"ramgate: {f:.1f} GB free, need > {need:g} GB (weight {weight_gb:g}); waiting")
        sleep(POLL_S)


def _rom_fixtures(item) -> set[str]:
    """ROM fixture names in the item's closure whose active definition (the last fixturedef of the
    name) comes from one of ROM_FIXTURE_MODULES. A direct parametrize argname (pytest's own pseudo
    fixture) or a test module's local fixture of the same name is not one."""
    info = getattr(item, "_fixtureinfo", None)
    defs = getattr(info, "name2fixturedefs", None) or {}
    out = set()
    for name in ROM_FIXTURES & set(item.fixturenames):
        chain = defs.get(name) or ()
        func = getattr(chain[-1], "func", None) if chain else None
        if getattr(func, "__module__", None) in ROM_FIXTURE_MODULES:
            out.add(name)
    return out


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
    return (
        bool(_rom_fixtures(item))
        or node.split("[")[0] in ROM_ONLY_IDS
        or _fetches_rom_fixture(item)
    )


def item_weight(
    item, table: Mapping[str, float], rom_table: Mapping[str, float], engine_free: bool = False
) -> float:
    """Pod weights come from runs where ROM-only tests skip: a ROM-only item weighs the figure of
    rom_weights.txt (measured locally), else 14 GB or its pod weight (if it has a key) when higher.
    Any other item weighs its weights.txt key; without one 6 GB in an engine-free run, else 20."""
    if is_rom_only(item):
        measured = lookup(item.nodeid, rom_table)
        if measured is not None:
            return measured
        pod = lookup(item.nodeid, table)
        return ROM_UNMEASURED_GB if pod is None else max(ROM_UNMEASURED_GB, pod)
    return weight(item.nodeid, table, ENGINE_FREE_WEIGHT if engine_free else DEFAULT_WEIGHT)


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
    pod_only: Iterable[str] | None = None,
    lock: "MachineLock | None" = None,
    platform: str = sys.platform,
    free: Callable[[], float] = free_gb,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    say: Callable[[str], None] = print,
) -> tuple[str, str]:
    """The root conftest's decision after collection: (SKIP | GO | REFUSE | TIMEOUT | ERROR,
    message); ERROR = free RAM could not be read.
    GO after the wait holds `lock` (default: the machine lock); the caller releases it."""
    env = os.environ if env is None else env
    if on_pod(env, platform):
        return "SKIP", "on the pod"
    lock = MachineLock() if lock is None else lock
    if inside_gated_run(env, lock):
        return "SKIP", "inside a gated run"
    if collect_only:  # --collect-only only imports test modules: allowed, no wait
        return "GO", "ramgate: collect-only"
    items = list(items)
    pod = pod_only_ids(items, load_pod_only() if pod_only is None else pod_only)
    if pod:  # even with NUZLINE_RAMGATE_ROM_GATE=1
        more = f" and {len(pod) - MAX_LISTED} more" if len(pod) > MAX_LISTED else ""
        return "REFUSE", (
            f"ramgate: pod-only tests (tools/ops/day/pod_only.txt) are never run on the PC: "
            f"{', '.join(pod[:MAX_LISTED])}{more}. "
            f"{QUICK_LANE_HINT[0].upper() + QUICK_LANE_HINT[1:]}."
        )
    modules = sys.modules if modules is None else modules
    engine_free = not engine_loaded(modules)
    bad = [] if _flag(env, ENV_ROM_GATE) else disallowed(items, modules)
    if bad:
        more = f" and {len(bad) - MAX_LISTED} more" if len(bad) > MAX_LISTED else ""
        return "REFUSE", (
            f"ramgate: no engine tests on the PC: {len(bad)} selected test(s) are neither ROM-only "
            f"nor engine-free, first: {', '.join(bad[:MAX_LISTED])}{more}. "
            f"{QUICK_LANE_HINT[0].upper() + QUICK_LANE_HINT[1:]}. "
            "Pass ROM-only node ids, never a file path (python tools/ops/ramgate.py --rom-ids <file>). "
            f"{ENV_ROM_GATE}=1 is a last resort: it also lets engine tests run on the PC."
        )
    if _flag(env, ENV_NO_WAIT):
        if lock.held_by_lineage("measure"):
            return "GO", "ramgate: GO (no wait: the measuring harness holds the lock)"
        say(f"ramgate: {ENV_NO_WAIT} ignored: no measuring harness holds the machine lock")
    table = load_weights() if table is None else table
    rom_table = load_rom_weights() if rom_table is None else rom_table
    w = max((item_weight(i, table, rom_table, engine_free) for i in items), default=0.0)
    timeout = timeout_minutes(env)
    try:
        ok = wait_for_ram(
            w, timeout_min=timeout, lock=lock, free=free, sleep=sleep, clock=clock, say=say
        )
    except OSError as e:
        return "ERROR", f"ramgate: cannot read free RAM ({e}); not starting. Run again later."
    if not ok:
        return "TIMEOUT", (
            f"ramgate: for {timeout:g} min free RAM stayed <= {RESERVE_GB + w:g} GB or another "
            f"local run held {lock.path} (weight {w:g} GB; {ENV_TIMEOUT} sets the minutes, weights "
            "in tools/ops/day/pod/weights.txt and tools/ops/day/rom_weights.txt). Run again later."
        )
    return "GO", f"ramgate: GO weight {w:g} GB"


def xdist_refusal(option, env: Mapping[str, str], platform: str = sys.platform) -> str | None:
    """pytest-xdist off the pod: its controller never collects, so the gate would never run."""
    if on_pod(env, platform):
        return None
    n = getattr(option, "numprocesses", None)
    dist = getattr(option, "dist", None)
    if n or (dist not in (None, "no")):
        return (
            "ramgate: no -n / --dist (pytest-xdist) on the PC: its controller does not collect, so "
            "the local gate cannot check the run. One pytest process, one file or chunk; "
            "parallel runs go to the pod (docs/process.md section 5)."
        )
    return None


def inside_gated_run(env: Mapping[str, str], lock: "MachineLock") -> bool:
    """NUZLINE_RAMGATE_DONE holds the pid of a gated run that is this process or a live ancestor:
    a pytest started by a test of that run (or inside it) is part of it."""
    try:
        pid = int(env.get(ENV_DONE, ""))
    except ValueError:
        return False
    return pid > 0 and lock.in_lineage(pid)


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
    lock: MachineLock | None = None,
) -> int:
    """CLI. `--build ... -- <command>`: a local job that cannot go to the pod: it waits for RAM
    and the machine lock, then runs the command as its child under the lock."""
    if "--rom-ids" in argv:
        return rom_ids([a for a in argv if a != "--rom-ids"], say)
    command: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1 :]
    rom_gate = "--rom-gate" in argv or _flag(os.environ, ENV_ROM_GATE)
    build = "--build" in argv
    argv = [a for a in argv if a not in ("--rom-gate", "--build")]
    usage = "usage: ramgate.py --build [--weight GB] -- <command> | <ids> [--weight GB] [--rom-gate]"
    if build != bool(command):
        say(usage + "; --build needs a command after --, and only --build takes one")
        return 2
    w = None
    if "--weight" in argv:
        i = argv.index("--weight")
        try:
            w = float(argv[i + 1])
        except (IndexError, ValueError):
            say(usage + "; --weight needs a number")
            return 2
        argv = argv[:i] + argv[i + 2 :]
    if w is None:
        table = load_weights()
        if build:
            measured = lookup(BUILD_KEY, table)
            w = BUILD_DEFAULT_WEIGHT if measured is None else measured
        else:
            w = max((weight(a, table) for a in argv), default=DEFAULT_WEIGHT)
    if not build and decide(w, 0.0, rom_gate) == "POD":
        say(f"POD {w:g}")
        return EXIT_REFUSE
    timeout = timeout_minutes(os.environ)
    if build:
        lock = MachineLock() if lock is None else lock
    try:
        ok = wait_for_ram(
            w, timeout_min=timeout, lock=lock if build else None, kind="build",
            free=free, sleep=sleep, clock=clock, say=say,
        )  # fmt: skip
    except OSError as e:
        say(f"ERROR cannot read free RAM ({e})")
        return EXIT_NO_RAM_READ
    if not ok:
        say(f"TIMEOUT {w:g}")
        return EXIT_TIMEOUT
    say(f"GO {w:g}")
    if not build:
        return 0
    try:
        return subprocess.call(command)
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
