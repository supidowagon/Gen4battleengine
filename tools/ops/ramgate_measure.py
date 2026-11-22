"""Local measuring pass for tools/ops/day/rom_weights.txt (docs/process.md section 5).

    python tools/ops/ramgate_measure.py IDS_FILE OUT_JSON [--skip SKIP_FILE]

IDS_FILE: ROM-only node ids, one per line (`python tools/ops/ramgate.py --rom-ids` lists them);
SKIP_FILE: ids left unmeasured (the slow ones). One file's ids per pytest process, strictly
serial, from the checkout this script is in, with its python. The harness takes the machine lock
as kind "measure" (tools/ops/ramgate.py MachineLock) for the whole pass, so the gate of each child
pytest honours NUZLINE_RAMGATE_NO_WAIT=1; before each file it waits for free RAM > 20 GB. Every
0.5 s it sums the resident size of the pytest process and all its descendants; when free RAM
drops below 4 GB it kills that tree (by the pid it started, never by name) and records the file
as `pod-measure needed`, weight 20. Figure = ceil(max(1.25 m, m + 2)) GB of the peak m.
OUT_JSON keeps the results (a rerun skips the files in it); the rom_weights.txt lines are printed
at the end. A figure comes only from a run where the tests ran: exit 0 and no `skipped` in the
summary gives tag `local-measure-<sha7>-<yyyymmdd>`; any other run (failed, error, skipped ids,
no tests) gives weight 14 (one loaded step), tag `unmeasured-skipped-<sha7>-<yyyymmdd>` or
`unmeasured-failed-rc<rc>-<sha7>-<yyyymmdd>`. Exit 0, or 14 when the RAM or lock wait timed out.
"""

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("ramgate_for_measure", HERE / "ramgate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["ramgate_for_measure"] = rg
_spec.loader.exec_module(rg)

START_FREE_GB = 20.0  # free RAM before each file: > 6 GB reserve + 14 GB
POLL_S = 0.5
PEAK_KILLED_WEIGHT = 20
UNMEASURED_WEIGHT = 14  # a run that did not run every id: one loaded step, as rom_weights.txt


def rss_bytes(pid: int) -> int:
    """Resident size of one process; 0 when it cannot be read (gone, or not ours)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
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

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD
        ]  # fmt: skip
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
        if not handle:
            return 0
        try:
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if not k32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return 0
            return int(counters.WorkingSetSize)
        finally:
            k32.CloseHandle(handle)
    try:
        for line in Path("/proc", str(pid), "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def tree_pids(pid: int) -> list[int]:
    procs = rg.process_table()
    out, todo = [pid], [pid]
    while todo:
        parent = todo.pop(0)
        kids = [c for c, pp in procs.items() if pp == parent and c not in out]
        out.extend(kids)
        todo.extend(kids)
    return out


def figure(peak_gb: float) -> int:
    return math.ceil(max(1.25 * peak_gb, peak_gb + 2))


def weight_line(f: str, r: dict, stamp: str) -> str:
    """The rom_weights.txt line of one file's result; stamp = `<sha7>-<yyyymmdd>`. Only a run
    with exit 0 and no skipped id gives a figure; a skip or a failure measured too little."""
    if r["rc"] == "killed":
        return f"{f} {r['figure']} pod-measure-needed"
    summary = " ".join(r["summary"]) if isinstance(r["summary"], list) else str(r["summary"])
    if r["rc"] != 0:
        return f"{f} {UNMEASURED_WEIGHT} unmeasured-failed-rc{r['rc']}-{stamp}"
    if re.search(r"\bskipped\b", summary) or not re.search(r"\bpassed\b", summary):
        return f"{f} {UNMEASURED_WEIGHT} unmeasured-skipped-{stamp}"
    return f"{f} {r['figure']} local-measure-{stamp}"


def measure_file(ids: list[str], log: Path) -> dict:
    """Run one file's ids; poll the tree's summed RSS and the 4 GB watchdog."""
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *ids]
    env = {**os.environ, rg.ENV_NO_WAIT: "1"}
    with log.open("w+", encoding="utf-8") as out:  # a file, not a pipe: a full pipe blocks the child
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, text=True)
        peak, killed = 0, False
        while p.poll() is None:
            try:
                low = rg.free_gb() < rg.WATCHDOG_GB
            except OSError:
                low = False
            if low:
                rg.kill_tree(p.pid)
                p.kill()
                killed = True
                break
            peak = max(peak, sum(rss_bytes(q) for q in tree_pids(p.pid)))
            time.sleep(POLL_S)
        p.wait()
        out.seek(0)
        text = out.read()
    m = peak / 1e9
    if killed:
        return {"n": len(ids), "m": round(m, 3), "figure": PEAK_KILLED_WEIGHT, "rc": "killed",
                "summary": "pod-measure needed (free RAM < 4 GB)"}  # fmt: skip
    last = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed|skipped|error)", ln)]
    return {"n": len(ids), "m": round(m, 3), "figure": figure(m), "rc": p.returncode,
            "summary": last[-1] if last else text.strip().splitlines()[-1:]}  # fmt: skip


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ids_file", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--skip", type=Path, help="ids left unmeasured (slow)")
    a = ap.parse_args(argv)
    ids = [ln.strip() for ln in a.ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    skip = set()
    if a.skip:
        skip = {ln.strip() for ln in a.skip.read_text(encoding="utf-8").splitlines() if ln.strip()}
    by_file: dict[str, list[str]] = {}
    for i in ids:
        if i in skip:
            print("SKIP unmeasured", i, flush=True)
        else:
            by_file.setdefault(i.split("::")[0], []).append(i)
    res = json.loads(a.out_json.read_text(encoding="utf-8")) if a.out_json.exists() else {}
    timeout = rg.timeout_minutes(os.environ)
    weight = START_FREE_GB - rg.RESERVE_GB
    lock = rg.MachineLock()
    if not rg.wait_for_ram(weight, timeout_min=timeout, lock=lock, kind="measure"):
        print(f"STOP: no {START_FREE_GB:g} GB free or the machine lock busy for {timeout:g} min")
        return rg.EXIT_TIMEOUT
    code = 0
    try:
        for f, fids in by_file.items():
            if f in res:
                continue
            if not rg.wait_for_ram(weight, timeout_min=timeout):  # the lock stays held
                print(f"STOP: free RAM <= {START_FREE_GB:g} GB for {timeout:g} min before {f}")
                code = rg.EXIT_TIMEOUT
                break
            res[f] = measure_file(fids, a.out_json.with_suffix(".pytest.log"))
            a.out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(f, res[f]["m"], res[f]["figure"], res[f]["rc"], res[f]["summary"], flush=True)
    finally:
        lock.release()
    sha = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    stamp = f"{sha}-{datetime.date.today():%Y%m%d}"
    for f, r in sorted(res.items()):
        print(weight_line(f, r, stamp))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
