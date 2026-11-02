"""Wall-time benchmark of ``nuzline prep`` on a real save: one cold run, then one warm run.

    uv run python -m benchmarks.prep_timing --save "$NUZLINE_USER_SAVE" [--devices N]

Both runs are subprocesses of ``nuzline prep --save S --trainer roark`` (defaults otherwise)
sharing a fresh temporary ``NUZLINE_CACHE_DIR``: the first finds the cache empty (cold), the
second reuses what the first wrote (warm). Each run's record (seconds, phase seconds, setups
screened, playouts, chosen setup, rung values) is read back, and both runs and the machine are
written to ``benchmarks/results/prep_<timestamp>.json`` (gitignored). Exits 1 when the cold run
printed no ``Setup:`` block. The target is a cold run of ``TARGET_SECONDS`` (plan 4c Task 13).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from benchmarks.rank_timing import RESULTS, ROOT, machine

TARGET_SECONDS = 900.0


def prep_command(save: Path, devices: int | None, finalists: int | None = None) -> list[str]:
    """The ``nuzline prep`` command line: Roark, every other option at its default."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; from nuzline.cli import main; sys.exit(main())",
        "prep",
        "--save",
        str(save),
        "--trainer",
        "roark",
    ]
    if devices is not None:
        cmd += ["--devices", str(devices)]
    if finalists is not None:
        cmd += ["--finalists", str(finalists)]
    return cmd


def parse_record(stdout: str) -> dict:
    """The numbers of the run record named by stdout's ``run record:`` line."""
    path = None
    for line in stdout.splitlines():
        if line.startswith("run record: "):
            path = Path(line[len("run record: ") :].strip())
    if path is None:
        raise ValueError("no 'run record:' line in the output of nuzline prep")
    rec = json.loads(path.read_text(encoding="utf-8"))
    stats = rec["stats"]
    return {
        "record": str(path),
        "seconds": stats["seconds"],
        "phase_seconds": {name: secs for name, secs in stats["phase_seconds"]},
        "setups_screened": stats["setups_screened"],
        "playouts": stats["playouts"],
        "stopped": stats["stopped"],
        "setup": rec["setup"],
        "rungs": {str(r["k"]): r["p"] for r in rec["ladder"]},
    }


def run_prep_timed(
    save: Path, devices: int | None, cache_dir: str, finalists: int | None = None
) -> dict:
    """One ``nuzline prep`` subprocess: its record's numbers plus ``wall_s`` and whether the
    output held a ``Setup:`` block."""
    env = {**os.environ, "NUZLINE_CACHE_DIR": cache_dir}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    start = time.perf_counter()
    done = subprocess.run(
        prep_command(save, devices, finalists),
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    wall = time.perf_counter() - start
    out = parse_record(done.stdout)
    out["wall_s"] = round(wall, 2)
    out["has_setup"] = "Setup:" in done.stdout
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument(
        "--finalists", type=int, default=None, help="pass --finalists (the CLI default is 3)"
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="nuzline-bench-") as cache_dir:
        cold = run_prep_timed(args.save, args.devices, cache_dir, args.finalists)
        print(f"cold: {cold['wall_s']:.1f} s (target {TARGET_SECONDS:.0f} s)", flush=True)
        warm = run_prep_timed(args.save, args.devices, cache_dir, args.finalists)
        print(f"warm: {warm['wall_s']:.1f} s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    now = datetime.now()
    out = RESULTS / f"prep_{now:%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(timespec="seconds"),
                "devices": args.devices,
                "target_s": TARGET_SECONDS,
                "cold": cold,
                "warm": warm,
                "machine": machine(),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")
    if not cold["has_setup"]:
        print("the cold run printed no Setup: block", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
