"""Seeds of the nightly soundness sweep. Plain Python: no nuzline or jax import.

    python -m tools.ops.sweep.seeds [<yyyymmdd>] [--n 15]

Prints one job line per process, `<node id> <short name> <seed>` (the nodes come from
tools/ops/sweep/nodes.txt), the format tools/ops/day/pod/sweep.sh reads. The date defaults to
today's local date. The seeds depend on night and node only, so the printed lines can be kept with
the results and a failing seed rerun from its line.
"""

import argparse
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

NODES = Path(__file__).resolve().parent / "nodes.txt"
NIGHT = re.compile(r"\d{8}")


def seeds(night: str, node: str, n: int = 15) -> list[int]:
    """n seeds for one node on one night: sha256 of "<night>:<node>:<i>" (i = 1..n), first 32 bits."""
    return [
        int.from_bytes(hashlib.sha256(f"{night}:{node}:{i}".encode()).digest()[:4], "big")
        for i in range(1, n + 1)
    ]


def nodes(path: Path = NODES) -> list[tuple[str, str]]:
    """(node id, short name) per line of the committed list; '#' lines and blanks are skipped."""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}: expected '<node id> <short name>', got {raw!r}")
        out.append((parts[0], parts[1]))
    return out


def job_lines(night: str, n: int = 15, path: Path = NODES) -> list[str]:
    return [
        f"{node} {short} {s}" for node, short in nodes(path) for s in seeds(night, node, n)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("night", nargs="?", help="yyyymmdd (default: today, local date)")
    ap.add_argument("--n", type=int, default=15, help="seeds per node (default 15)")
    a = ap.parse_args(argv)
    night = a.night or date.today().strftime("%Y%m%d")
    if not NIGHT.fullmatch(night):
        print(f"night must be yyyymmdd, got {night!r}", file=sys.stderr)
        return 2
    if a.n < 1:
        print("--n must be at least 1", file=sys.stderr)
        return 2
    print("\n".join(job_lines(night, a.n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
