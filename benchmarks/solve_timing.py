"""Wall-time benchmark of ``nuzline solve``: cold then warm at depths 1 and 2.

    uv run python -m benchmarks.solve_timing [--depths 1 2] [--playouts D=B ...] [--calibrate]
                                             [--devices N]

Every run is a subprocess sharing a fresh temporary ``NUZLINE_CACHE_DIR`` (cold: the cache is
empty; warm: the cold run's cache is reused). The subprocess times the two phases of a solve,
the leaf playouts (``evaluate_leaves`` and ``evaluate_forced``) and everything else
(expansion: the exact chance and AI enumeration, merging, backup), and reports its peak
resident set. With ``--calibrate`` the leaf budget ``B_d`` starts at ``2 * K_d * 4300`` (K_1 = 9,
K_2 = 81 player-action combinations; 4300 effective playouts is a 1.5 pp half-width,
notes/critic.md C7; the factor 2 is the ruling-5 half split) and doubles, on a warm cache,
until the leader's half-width is at most 1.5 percentage points; the cold and warm runs then
use that budget. Results go to ``benchmarks/results/solve_<timestamp>.json`` (gitignored).
The command exits non-zero if a run prints no ranking header.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

from benchmarks.rank_timing import PARTY, RESULTS, ROOT, machine

K = {1: 9, 2: 81}
TARGET_HALF_WIDTH = 1.5  # percentage points
TARGET_SECONDS = {1: 120.0, 2: 600.0}

# Runs ``nuzline solve`` in-process with the leaf evaluators timed; the last stderr line is
# a JSON object {leaf_s, total_s, peak_rss_mb}.
DRIVER = """
import json, sys, time
import nuzline.search.stage_b as sb
leaf = [0.0]
def timed(fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            leaf[0] += time.perf_counter() - t
    return wrapper
sb.evaluate_leaves = timed(sb.evaluate_leaves)
sb.evaluate_forced = timed(sb.evaluate_forced)
from nuzline.cli import main
t0 = time.perf_counter()
code = main(sys.argv[1:])
total = time.perf_counter() - t0
sys.stdout.flush()
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    peak = pmc.PeakWorkingSetSize / 2**20
else:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        2**20 if sys.platform == "darwin" else 2**10)
print(json.dumps({"leaf_s": leaf[0], "total_s": total, "peak_rss_mb": peak}), file=sys.stderr)
sys.exit(code)
"""

HEADER = re.compile(r"depth reached (\d+) of (\d+); (\d+) playouts, ([\d.]+) s")
LEADER = re.compile(r"^\s*\*?\s+1\s+.*\[\s*([\d.]+),\s*([\d.]+)\]", re.MULTILINE)


def run_solve(depth: int, playouts: int, devices: int | None, cache_dir: str) -> dict:
    """One ``nuzline solve`` subprocess: wall seconds, phases, peak RSS and the output facts."""
    cmd = [
        sys.executable,
        "-c",
        DRIVER,
        "solve",
        "--party",
        str(PARTY),
        "--trainer",
        "roark",
        "--max-depth",
        str(depth),
        "--playouts",
        str(playouts),
    ]
    if devices is not None:
        cmd += ["--devices", str(devices)]
    env = {**os.environ, "NUZLINE_CACHE_DIR": cache_dir}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    start = time.perf_counter()
    done = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - start
    header, leader = HEADER.search(done.stdout), LEADER.search(done.stdout)
    if header is None or leader is None:
        raise RuntimeError(f"nuzline solve printed no ranking:\n{done.stdout}\n{done.stderr}")
    phases = json.loads(done.stderr.strip().splitlines()[-1])
    low, high = float(leader.group(1)), float(leader.group(2))
    return {
        "wall_s": round(wall, 2),
        "solve_s": round(phases["total_s"], 2),
        "leaf_playouts_s": round(phases["leaf_s"], 2),
        "expansion_s": round(phases["total_s"] - phases["leaf_s"], 2),
        "peak_rss_mb": round(phases["peak_rss_mb"]),
        "completed_depth": int(header.group(1)),
        "playouts": int(header.group(3)),
        "leader_ci": [low, high],
        "leader_half_width_pp": round((high - low) / 2, 2),
    }


def calibrate(depth: int, devices: int | None, cache_dir: str) -> tuple[int, list[dict]]:
    """Double ``B_d`` from ``2 * K_d * 4300`` until the leader's half-width is at most 1.5 pp."""
    budget, trail = 2 * K[depth] * 4300, []
    while True:
        run = run_solve(depth, budget, devices, cache_dir)
        trail.append({"budget": budget, **run})
        print(
            f"  depth {depth} B={budget}: half-width {run['leader_half_width_pp']} pp, "
            f"{run['solve_s']} s",
            flush=True,
        )
        if run["leader_half_width_pp"] <= TARGET_HALF_WIDTH:
            return budget, trail
        budget *= 2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2], choices=sorted(K))
    parser.add_argument(
        "--playouts", nargs="*", default=[], metavar="D=B", help="leaf budget B_d, e.g. 1=80000"
    )
    parser.add_argument("--calibrate", action="store_true", help="find B_d by doubling")
    parser.add_argument("--devices", type=int, default=None)
    args = parser.parse_args(argv)
    fixed = {int(d): int(b) for d, b in (item.split("=") for item in args.playouts)}

    records = []
    for depth in args.depths:
        trail: list[dict] = []
        if args.calibrate:
            with tempfile.TemporaryDirectory(prefix="nuzline-bench-") as cache_dir:
                budget, trail = calibrate(depth, args.devices, cache_dir)
        else:
            budget = fixed.get(depth, 2 * K[depth] * 4300)
        with tempfile.TemporaryDirectory(prefix="nuzline-bench-") as cache_dir:
            cold = run_solve(depth, budget, args.devices, cache_dir)
            print(f"depth {depth} B={budget} cold: {cold}", flush=True)
            warm = run_solve(depth, budget, args.devices, cache_dir)
            print(f"depth {depth} B={budget} warm: {warm}", flush=True)
        records.append(
            {
                "depth": depth,
                "budget": budget,
                "target_s": TARGET_SECONDS[depth],
                "cold": cold,
                "warm": warm,
                "calibration": trail,
            }
        )

    RESULTS.mkdir(exist_ok=True)
    now = datetime.now()
    out = RESULTS / f"solve_{now:%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(timespec="seconds"),
                "devices": args.devices,
                "runs": records,
                "machine": machine(),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
