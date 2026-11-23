"""Local test gate (docs/process.md section 5). Pure functions, the pytest gate and a small CLI.

Rule (owner, 22 Nov 2026): no engine tests on the PC. A local pytest run is allowed only when every
selected test is ROM-only (needs a local input the pod does not have) or the run is engine-free
(after collection `nuzline.engine` is not imported). Everything else goes to the pod through the
dispatcher's quick lane. An allowed run waits for free RAM > 6 GB + its weight (weights.txt, the
longest key matching an id; no key = 6 GB in an engine-free run, else 20 GB). The root conftest.py
applies it at collection.

One local run at a time on the PC: after the RAM check a gated run takes the machine lock, an OS
lock on `<main checkout>/_scratch/ramgate.lock` (MachineLock), and holds it until it ends; the
kernel frees it when the holder dies. Waiting on it counts toward the same timeout. While it runs a watchdog thread kills its child processes and
stops it when free RAM drops below 4 GB (exit 15).

CLI:
    python tools/ops/ramgate.py --build [--weight GB] -- <command ...>
        a local job that is not pytest (table builds, `tools.kaizo.*` runs): waits for free RAM >
        6 GB + weight (--weight, else the weights.txt line `tools/build_tables.py`, else 6), takes
        the machine lock, runs the command as its child under the lock and exits with its code.
        On the pod or off Windows it runs the command directly (no wait, lock or watchdog).
    python tools/ops/ramgate.py <ids> [--weight GB] [--rom-gate]
        a check only (no lock): `GO <w>` (exit 0) after any wait, `POD <w>` (weight >= 8).
    python tools/ops/ramgate.py --rom-ids [file ...]   (collect only; one ready command per file)
`POD` exits 13, `TIMEOUT <w>` (after NUZLINE_RAMGATE_TIMEOUT minutes, default 30) exits 14, an
unreadable free-RAM figure exits 16. The pytest gate exits 13 (refused), 14 (timeout) and 16 too:
codes apart from pytest's own 0-5.
"""

import _thread
import ast
import importlib.abc
import inspect
import math
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
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
BUILD_DEFAULT_WEIGHT = 6.0  # without that line: > 6 + 6 = 12 GB free (measured build peak 0.64 GB)
POLL_S = 60.0
EXIT_REFUSE = 13  # refused (pytest) or POD (CLI)
EXIT_TIMEOUT = 14
EXIT_WATCHDOG = 15  # free RAM fell below WATCHDOG_GB during the run
EXIT_NO_RAM_READ = 16  # free_gb raised OSError
WATCHDOG_GB = 4.0
WATCHDOG_POLL_S = 0.5
LOCK_POLL_S = 10.0  # free RAM is enough but another local run holds the machine lock
LOCK_NAME = "ramgate.lock"  # under the main checkout's _scratch/: one local run on the PC at a time
LOCK_OFFSET = 1 << 20  # Windows: the byte msvcrt.locking holds, far past the holder line
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
# The 6 holds only while the engine stays out (rom_measure_w3.md measured engine-free tests):
# an unmeasured ROM-only item in a run that loaded the engine at collection weighs
# ROM_ENGINE_UNMEASURED_GB, and a run with an unmeasured ROM-only item gets the engine blocker
# (blocker_wanted), so an engine import in a test body fails instead of loading a step.
ROM_UNMEASURED_GB = 6.0  # an unmeasured engine-free ROM-only item waits for > 6 + 6 = 12 GB free (largest measured figure: 3)
ROM_ENGINE_UNMEASURED_GB = 14.0  # the same when the engine is loaded: > 20 GB free (one loaded step is about 6.3 GB)
ROM_WEIGHTS = ROOT / "tools" / "ops" / "day" / "rom_weights.txt"
POD_ONLY = ROOT / "tools" / "ops" / "day" / "pod_only.txt"  # refused on the PC whatever they import
ENGINE_PREFIX = "nuzline.engine"
# Engine modules that load nothing heavy: the package (its __init__ imports os and sys only) and
# the id enums (ids.py imports enum only; tools.kaizo.rows_tables needs them). Importing them at
# collection does not make a run engine-loaded; the blocker still refuses any engine import after.
LIGHT_ENGINE_MODULES = frozenset({ENGINE_PREFIX, ENGINE_PREFIX + ".ids"})
MAX_LISTED = 5

ENV_ON_POD = "NUZLINE_ON_POD"
ENV_TIMEOUT = "NUZLINE_RAMGATE_TIMEOUT"
ENV_ROM_GATE = "NUZLINE_RAMGATE_ROM_GATE"  # CLI only: lifts the POD cut-off for ROM-only ids
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


def lookup_entry(ident: str, table: Mapping[str, float]) -> tuple[str, float] | None:
    """(key, GB) of the longest key that equals the id or that the id starts with plus "::", "[" or "/"."""
    ident = ident.replace("\\", "/")
    best: tuple[str, float] | None = None
    for key, gb in table.items():
        hit = ident == key or any(ident.startswith(key + s) for s in ("::", "[", "/"))
        if hit and (best is None or len(key) > len(best[0])):
            best = (key, gb)
    return best


def lookup(ident: str, table: Mapping[str, float]) -> float | None:
    """GB of the longest matching key (lookup_entry), else None."""
    entry = lookup_entry(ident, table)
    return entry[1] if entry else None


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


def load_pod_only_reasons(path: str | Path = POD_ONLY) -> dict[str, str]:
    """pod_only.txt as {id: reason}."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, reason = line.strip().partition(" ")
            out[key] = reason.strip()
    return out


ROM_AND_ENGINE_REASON = "rom-and-engine"  # pod_only.txt reasons starting so: ROM + engine tests


def pod_only_ids(items: Iterable, keys: Iterable[str]) -> list[str]:
    """Node ids under a pod_only.txt key (same matching as a weights.txt key)."""
    table = dict.fromkeys(keys, 0.0)
    return [i.nodeid for i in items if lookup(i.nodeid, table) is not None]


def pod_only_refusal(ids: list[str], reasons: Mapping[str, str]) -> str:
    """The refusal text. A ROM + engine test (reason `rom-and-engine...`) needs the ROM, which never
    leaves the PC, so the quick lane cannot run it either: no quick-lane hint for those."""
    both = [i for i in ids if (lookup(i, reasons) or "").startswith(ROM_AND_ENGINE_REASON)]
    rest = [i for i in ids if i not in both]

    def listed(xs):
        more = f" and {len(xs) - MAX_LISTED} more" if len(xs) > MAX_LISTED else ""
        return f"{', '.join(xs[:MAX_LISTED])}{more}"

    parts = []
    if rest:
        parts.append(
            f"ramgate: pod-only tests (tools/ops/day/pod_only.txt) are never run on the PC: "
            f"{listed(rest)}. {QUICK_LANE_HINT[0].upper() + QUICK_LANE_HINT[1:]}."
        )
    if both:
        parts.append(
            f"ramgate: ROM + engine tests (tools/ops/day/pod_only.txt) run neither on the PC nor on "
            f"a pod (the ROM never leaves the PC; not the quick lane either): {listed(both)}. Drop "
            "them from this run; they are never in a merge-gate ROM-only run. Running them at a "
            "split close is an owner question that the managing session asks (docs/process.md "
            "section 3)."
        )
    return " ".join(parts)


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
    return meminfo_available_gb(Path("/proc/meminfo").read_text(encoding="utf-8"))


def meminfo_available_gb(text: str) -> float:
    """MemAvailable of a /proc/meminfo text, in GB (10^9 bytes)."""
    for line in text.splitlines():
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
    """pid -> parent pid of every live process (Windows: a Toolhelp32 snapshot; else /proc, where
    a zombie counts as gone: proc_table)."""
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
    return proc_table()


def proc_table(root: str | Path = "/proc") -> dict[int, int]:
    """pid -> parent pid from a /proc tree. A zombie (state Z) or dead (X) process has ended: it
    only waits for its parent to reap it (a pod's PID 1 may never do), so it is left out, for
    kill_tree, lineage and the machine lock alike."""
    out: dict[int, int] = {}
    for name in os.listdir(root):
        if name.isdigit():
            try:
                stat = Path(root, name, "stat").read_text(encoding="utf-8", errors="replace")
                state, ppid = stat.rsplit(")", 1)[1].split()[:2]
            except (OSError, ValueError, IndexError):  # IndexError: a stat text with no ")"
                continue
            if state not in ("Z", "X", "x"):
                out[int(name)] = int(ppid)
    return out


def start_time(pid: int) -> float | None:
    """When the process started (seconds, platform clock), None when it cannot be read."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return (times[0].dwHighDateTime << 32 | times[0].dwLowDateTime) / 1e7
        finally:
            k32.CloseHandle(handle)
    try:
        stat = Path("/proc", str(pid), "stat").read_text(encoding="utf-8", errors="replace")
        return float(stat.rsplit(")", 1)[1].split()[19])  # field 22, starttime in clock ticks
    except (OSError, IndexError, ValueError):
        return None


def lineage(
    pid: int,
    procs: Mapping[int, int],
    started: Callable[[int], float | None] = start_time,
) -> list[int]:
    """The pid and its live ancestors, nearest first. Windows keeps a dead parent's pid as the
    recorded parent and hands pids out again, so a link counts only when the parent started no
    later than the child (both start times readable); a pid cycle or a dead parent ends it too."""
    out: list[int] = []
    born = started(pid) if pid in procs else None
    while pid in procs and pid not in out and pid > 0:
        out.append(pid)
        parent = procs[pid]
        if parent not in procs or parent in out or parent <= 0:
            break
        parent_born = started(parent)
        if born is None or parent_born is None or parent_born > born:
            break
        pid, born = parent, parent_born
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


def _os_lock(fd: int) -> bool:
    """Exclusive, non-blocking OS lock on the open file; False when another open file holds it.
    Windows: msvcrt.locking of one byte at LOCK_OFFSET (far past the holder line, which others
    must still read); elsewhere fcntl.flock. The kernel drops it when the process ends."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _os_unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class MachineLock:
    """One local gated run on the PC at a time: an OS lock (msvcrt.locking on Windows, fcntl.flock
    elsewhere) on `<main checkout>/_scratch/ramgate.lock`, held for the run. The kernel releases it
    when the holder ends, however it ends, so there is no stale lock. The file's line
    `<pid> <kind> <start time>` only names the holder: a process whose live ancestor (pid and
    start time both matching, so a reused pid never counts) holds the lock runs under that hold
    (`inherited`). The file itself is never deleted."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        procs: Callable[[], Mapping[int, int]] = process_table,
        started: Callable[[int], float | None] = start_time,
        me: int | None = None,
    ):
        self.path = Path(path) if path is not None else lock_path()
        self._procs = procs
        self._started = started
        self.me = os.getpid() if me is None else me
        self._fd: int | None = None
        self.inherited = False

    @property
    def owned(self) -> bool:
        return self._fd is not None

    @property
    def held(self) -> bool:
        return self.owned or self.inherited

    def in_lineage(self, pid: int, start: float | None = None) -> bool:
        """pid is this process or a live ancestor (start-time checked links); with `start`, that
        process must also have started at exactly that time."""
        if pid not in lineage(self.me, self._procs(), self._started):
            return False
        return start is None or self._started(pid) == start

    def holder(self) -> tuple[int | None, str, float | None]:
        """(pid, kind, start time) of the line in the file; (None, "", None) when there is none."""
        try:
            parts = self.path.read_text(encoding="utf-8").split()
            return int(parts[0]), parts[1], float(parts[2])
        except (OSError, ValueError, IndexError):
            return None, "", None

    def _open(self) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))

    def _busy(self) -> bool:
        """Some open file holds the OS lock right now (probed with a second open, then let go)."""
        fd = self._open()
        try:
            if _os_lock(fd):
                _os_unlock(fd)
                return False
            return True
        finally:
            os.close(fd)

    def held_by_lineage(self, kind: str | None = None) -> bool:
        """The lock is held now, and its line names this process or a live ancestor (pid and start
        time), of that kind when given."""
        pid, k, start = self.holder()
        if pid is None or start is None or (kind is not None and k != kind):
            return False
        return self.in_lineage(pid, start) and (self.owned or self._busy())

    def acquire(self, kind: str = "pytest") -> bool:
        """Take the lock once (no waiting). True when it is now held by this process or its line."""
        if self.held:
            return True
        fd = self._open()
        if _os_lock(fd):
            start = self._started(self.me)
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{self.me} {kind} {start!r}\n".encode())
            self._fd = fd
            return True
        os.close(fd)
        pid, _, start = self.holder()
        if pid is not None and start is not None and self.in_lineage(pid, start):
            self.inherited = True
            return True
        return False

    def release(self) -> None:
        fd, self._fd = self._fd, None
        self.inherited = False
        if fd is None:
            return
        try:
            os.ftruncate(fd, 0)  # the line names no one once the lock is free
            _os_unlock(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def busy_message(self) -> str:
        pid, kind, _ = self.holder()
        who = f"pid {pid} ({kind})" if pid is not None else "an unnamed process"
        return (
            f"ramgate: another local run, {who}, holds the lock on {self.path}; waiting. The OS "
            f"frees it the moment that process ends; to clear it, end {who} if it is no pytest "
            "or build of yours (deleting the file does not free it)"
        )


def kill_tree(
    pid: int,
    procs: Mapping[int, int] | None = None,
    *,
    started: Callable[[int], float | None] = start_time,
    kill: Callable[[int, int], None] = os.kill,
) -> list[int]:
    """Kill every process below pid (never pid itself), found by its recorded parent pid. A child
    that started before its parent holds a reused pid as parent (Windows keeps dead parents'
    pids) and is not ours: it is skipped with its subtree. Returns the pids killed, leaves first."""
    procs = process_table() if procs is None else procs
    seen: dict[int, float | None] = {}

    def born(p: int) -> float | None:
        if p not in seen:
            seen[p] = started(p)
        return seen[p]

    victims: list[int] = []
    todo = [pid]
    while todo:
        parent = todo.pop(0)
        for child, pp in procs.items():
            if pp != parent or child in (pid, parent) or child in victims:
                continue
            tc, tp = born(child), born(parent)
            if tc is None or tp is None or tc < tp:
                continue
            victims.append(child)
            todo.append(child)
    sig = getattr(signal, "SIGKILL", signal.SIGTERM)  # Windows: TerminateProcess
    for victim in reversed(victims):
        try:
            kill(victim, sig)
        except OSError:
            pass
    return victims[::-1]


class Watchdog(threading.Thread):
    """While a gated local run goes: when free RAM drops below 4 GB, kill every child process of
    `pid` (default: this process) and call on_fire(self) (the pytest gate: interrupt the run)."""

    def __init__(
        self,
        *,
        pid: int | None = None,
        floor_gb: float = WATCHDOG_GB,
        poll_s: float = WATCHDOG_POLL_S,
        free: Callable[[], float] = free_gb,
        kill: Callable[[int], list[int]] = kill_tree,
        on_fire: "Callable[[Watchdog], None] | None" = None,
        say: Callable[[str], None] = print,
    ):
        super().__init__(name="ramgate-watchdog", daemon=True)
        self.pid = os.getpid() if pid is None else pid
        self.floor_gb, self.poll_s = floor_gb, poll_s
        self._free, self._kill, self._on_fire, self._say = free, kill, on_fire, say
        self.halted = threading.Event()
        self.fired = False

    def halt(self) -> None:
        self.halted.set()

    def check(self) -> bool:
        """One poll. True once it has fired."""
        if self.fired:
            return True
        try:
            f = self._free()
        except OSError:
            return False
        if f >= self.floor_gb:
            return False
        self.fired = True
        killed = self._kill(self.pid)
        self._say(
            f"ramgate watchdog: {f:.1f} GB free < {self.floor_gb:g} GB: killed {len(killed)} child "
            f"process(es) of pid {self.pid}, stopping the run (exit {EXIT_WATCHDOG}). Run the ids on "
            "the pod, or again once RAM is free."
        )
        if self._on_fire is not None:
            self._on_fire(self)
        return True

    def run(self) -> None:
        while not self.halted.wait(self.poll_s):
            if self.check():
                return


def interrupt_pytest(watchdog: Watchdog, grace_s: float = 30.0) -> None:
    """on_fire of the pytest gate: KeyboardInterrupt in the main thread, so pytest ends the
    session (unconfigure releases the lock); if it is still running after grace_s (stuck in C
    code), the process exits at once with EXIT_WATCHDOG (the kernel then frees its OS lock)."""
    _thread.interrupt_main()
    if not watchdog.halted.wait(grace_s):
        os._exit(EXIT_WATCHDOG)


_NOT_TOLD = object()


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
    told: object = _NOT_TOLD
    while True:
        f = free()
        if f > need:
            if lock is None or lock.acquire(kind):
                return True
            if clock() >= deadline:
                return False
            pid = lock.holder()[0]
            if pid != told:  # once per holder; "no holder named" counts as one too
                say(lock.busy_message())
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
    rom_weights.txt (measured locally), else 6 GB in an engine-free run (14 when the engine is
    loaded), or its pod weight (if it has a key) when higher.
    Any other item weighs its weights.txt key; without one 6 GB in an engine-free run, else 20.
    Exception: in an engine-free run a non-ROM-only item takes a rom_weights.txt match when no
    weights.txt key is more specific (longest key wins across both tables; equal length: rom
    line). The figure is the measured peak of the file's ROM-only ids; it serves the file's other
    engine-free ids because the engine blocker keeps the engine out. The weights.txt nightly line
    of a ROM-only file counts skipped tests, so its 6 would otherwise outrank it."""
    if is_rom_only(item):
        measured = lookup(item.nodeid, rom_table)
        if measured is not None:
            return measured
        base = ROM_UNMEASURED_GB if engine_free else ROM_ENGINE_UNMEASURED_GB
        pod = lookup(item.nodeid, table)
        return base if pod is None else max(base, pod)
    if engine_free:
        rom = lookup_entry(item.nodeid, rom_table)
        pod = lookup_entry(item.nodeid, table)
        if rom is not None and (pod is None or len(rom[0]) >= len(pod[0])):
            return rom[1]
    return weight(item.nodeid, table, ENGINE_FREE_WEIGHT if engine_free else DEFAULT_WEIGHT)


def engine_loaded(modules: Iterable[str]) -> bool:
    """A module of the engine is imported, other than the light ones (LIGHT_ENGINE_MODULES)."""
    return any(
        (m == ENGINE_PREFIX or m.startswith(ENGINE_PREFIX + ".")) and m not in LIGHT_ENGINE_MODULES
        for m in modules
    )


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


def blocker_wanted(
    items: Iterable,
    env: Mapping[str, str],
    rom_table: Mapping[str, float] | None = None,
    measuring: Callable[[], bool] | None = None,
) -> bool:
    """The finder goes on in an engine-free run (not all ROM-only), and in an all-ROM-only run
    with an item that rom_weights.txt does not list: its 6 GB default assumes no engine. Not while
    the measuring harness (machine lock kind "measure") runs the unmeasured items to measure them."""
    items = list(items)
    if not all(is_rom_only(i) for i in items):
        return True
    rom_table = load_rom_weights() if rom_table is None else rom_table
    if all(lookup(i.nodeid, rom_table) is not None for i in items):
        return False
    measuring = (lambda: MachineLock().held_by_lineage("measure")) if measuring is None else measuring
    return not measuring()


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
    if pod_only is None:
        pod_only = load_pod_only_reasons()
    reasons = pod_only if isinstance(pod_only, Mapping) else dict.fromkeys(pod_only, "")
    pod = pod_only_ids(items, reasons)
    if pod:
        return "REFUSE", pod_only_refusal(pod, reasons)
    modules = sys.modules if modules is None else modules
    engine_free = not engine_loaded(modules)
    bad = disallowed(items, modules)  # NUZLINE_RAMGATE_ROM_GATE lifts nothing here
    if bad:
        more = f" and {len(bad) - MAX_LISTED} more" if len(bad) > MAX_LISTED else ""
        return "REFUSE", (
            f"ramgate: no engine tests on the PC: {len(bad)} selected test(s) are neither ROM-only "
            f"nor engine-free, first: {', '.join(bad[:MAX_LISTED])}{more}. "
            f"{QUICK_LANE_HINT[0].upper() + QUICK_LANE_HINT[1:]}. "
            "Pass ROM-only node ids, never a file path (python tools/ops/ramgate.py --rom-ids <file>)."
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
    """NUZLINE_RAMGATE_DONE holds the pid of a gated run that is this process or a live ancestor
    (lineage: every link start-time checked, so a reused pid is no ancestor): a pytest started by
    a test of that run (or inside it) is part of it."""
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
    platform: str = sys.platform,
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
    if build and (on_pod(os.environ, platform) or platform != "win32"):
        return subprocess.call(command)  # the gate guards the PC only: no wait, lock or watchdog
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
    watchdog = Watchdog(free=free, say=say)  # kills the command's tree below 4 GB free
    watchdog.start()
    try:
        code = subprocess.call(command)
    finally:
        watchdog.halt()
        lock.release()
    return EXIT_WATCHDOG if watchdog.fired else code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
