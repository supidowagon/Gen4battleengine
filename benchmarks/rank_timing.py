"""Wall-time benchmark of ``nuzline rank``: one cold run, then one warm run.

    uv run python -m benchmarks.rank_timing [--n 2000] [--devices N]

Both runs are subprocesses sharing a fresh temporary ``NUZLINE_CACHE_DIR``: the first
finds the cache empty (cold), the second reuses what the first wrote (warm). The times
and the machine are written to ``benchmarks/results/rank_<timestamp>.json`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTY = ROOT / "tests" / "fixtures" / "sample_party.toml"
RESULTS = Path(__file__).resolve().parent / "results"


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def machine() -> dict:
    import jax

    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "jax": jax.__version__,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def run_rank(n: int, devices: int | None, cache_dir: str) -> float:
    """Wall seconds of one ``nuzline rank`` subprocess."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; from nuzline.cli import main; sys.exit(main())",
        "rank",
        "--party",
        str(PARTY),
        "--trainer",
        "roark",
        "--n",
        str(n),
    ]
    if devices is not None:
        cmd += ["--devices", str(devices)]
    env = {**os.environ, "NUZLINE_CACHE_DIR": cache_dir}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    start = time.perf_counter()
    done = subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - start
    if "playouts per action" not in done.stdout:
        raise RuntimeError(f"nuzline rank printed no ranking:\n{done.stdout}")
    return elapsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--devices", type=int, default=None)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="nuzline-bench-") as cache_dir:
        cold = run_rank(args.n, args.devices, cache_dir)
        print(f"cold: {cold:.1f} s", flush=True)
        warm = run_rank(args.n, args.devices, cache_dir)
        print(f"warm: {warm:.1f} s", flush=True)

    RESULTS.mkdir(exist_ok=True)
    now = datetime.now()
    out = RESULTS / f"rank_{now:%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": now.isoformat(timespec="seconds"),
                "n": args.n,
                "devices": args.devices,
                "cold_s": round(cold, 2),
                "warm_s": round(warm, 2),
                "machine": machine(),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
