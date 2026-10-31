"""Wall-time benchmark of ``nuzline solve``: cold then warm at depths 1 and 2.

    uv run python -m benchmarks.solve_timing [--depths 1 2] [--precision PP] [--playouts D=B ...]
                                             [--devices N]

Every run is a subprocess sharing a fresh temporary ``NUZLINE_CACHE_DIR`` (cold: the cache is
empty; warm: the cold run's cache is reused). The subprocess times the two phases of a solve,
the leaf playouts (``evaluate_leaves`` and ``evaluate_forced``, every round) and everything
else (expansion: the exact chance and AI enumeration, merging, backup), and reports its peak
resident set. Runs use the solver's own precision target (``nuzline solve --precision PP``,
default 1.5 percentage points: each pass tops up its leaf playouts until the leader's half-A
half-width is at most that, rulings 8 and 9 as amended 2026-10-31); ``--playouts D=B`` runs
depth ``D`` with the fixed leaf budget ``B`` instead (no top-ups). Each run reports the
playouts run, the top-up rounds of the final pass, the leader's half-A half-width and whether
the target was met, and the leaf and total seconds. Results go to
``benchmarks/results/solve_<timestamp>.json`` (gitignored). The command exits non-zero if a
run prints no ranking header.
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

DEPTHS = (1, 2)
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
PRECISION = re.compile(
    r"^precision target [\d.]+ pp: (met|not met) \(leader's half-A half-width ([\d.]+) pp\) "
    r"after (\d+) top-up rounds?",
    re.MULTILINE,
)


def solve_command(
    depth: int, playouts: int | None, precision: float, devices: int | None
) -> list[str]:
    """The ``nuzline solve`` arguments of one run: the fixed budget ``playouts`` when given,
    else the precision target ``precision`` (percentage points)."""
    cmd = ["solve", "--party", str(PARTY), "--trainer", "roark", "--max-depth", str(depth)]
    if playouts is not None:
        cmd += ["--playouts", str(playouts)]
    else:
        cmd += ["--precision", str(precision)]
    if devices is not None:
        cmd += ["--devices", str(devices)]
    return cmd


def parse_output(stdout: str) -> dict:
    """The facts a run's output states: completed depth, playouts, the leader's printed
    (half-B) interval and half-width, and for a precision run the final pass's top-up
    rounds, its half-A half-width and whether the target was met (``0``, ``None``,
    ``None`` for a fixed budget). ``RuntimeError`` without a ranking."""
    header, leader = HEADER.search(stdout), LEADER.search(stdout)
    if header is None or leader is None:
        raise RuntimeError(f"nuzline solve printed no ranking:\n{stdout}")
    low, high = float(leader.group(1)), float(leader.group(2))
    precision = PRECISION.search(stdout)
    return {
        "completed_depth": int(header.group(1)),
        "playouts": int(header.group(3)),
        "leader_ci": [low, high],
        "leader_half_width_pp": round((high - low) / 2, 2),
        "rounds": int(precision.group(3)) if precision else 0,
        "half_width_a_pp": float(precision.group(2)) if precision else None,
        "precision_met": precision.group(1) == "met" if precision else None,
    }


def run_solve(
    depth: int, playouts: int | None, precision: float, devices: int | None, cache_dir: str
) -> dict:
    """One ``nuzline solve`` subprocess: wall seconds, phases, peak RSS and the output facts."""
    cmd = [sys.executable, "-c", DRIVER, *solve_command(depth, playouts, precision, devices)]
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
    try:
        facts = parse_output(done.stdout)
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n{done.stderr}") from None
    phases = json.loads(done.stderr.strip().splitlines()[-1])
    return {
        "wall_s": round(wall, 2),
        "solve_s": round(phases["total_s"], 2),
        "leaf_playouts_s": round(phases["leaf_s"], 2),
        "expansion_s": round(phases["total_s"] - phases["leaf_s"], 2),
        "peak_rss_mb": round(phases["peak_rss_mb"]),
        **facts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS), choices=DEPTHS)
    parser.add_argument(
        "--precision",
        type=float,
        default=TARGET_HALF_WIDTH,
        help="precision target, percentage points (default 1.5)",
    )
    parser.add_argument(
        "--playouts",
        nargs="*",
        default=[],
        metavar="D=B",
        help="fixed leaf budget B_d for depth D instead of the precision target, e.g. 1=80000",
    )
    parser.add_argument("--devices", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.precision > 0:
        parser.error("--precision must be positive")
    fixed = {int(d): int(b) for d, b in (item.split("=") for item in args.playouts)}

    records = []
    for depth in args.depths:
        budget = fixed.get(depth)
        mode = f"B={budget}" if budget is not None else f"precision {args.precision} pp"
        with tempfile.TemporaryDirectory(prefix="nuzline-bench-") as cache_dir:
            cold = run_solve(depth, budget, args.precision, args.devices, cache_dir)
            print(f"depth {depth} {mode} cold: {cold}", flush=True)
            warm = run_solve(depth, budget, args.precision, args.devices, cache_dir)
            print(f"depth {depth} {mode} warm: {warm}", flush=True)
        records.append(
            {
                "depth": depth,
                "budget": budget,
                "precision_pp": None if budget is not None else args.precision,
                "target_s": TARGET_SECONDS[depth],
                "cold": cold,
                "warm": warm,
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
