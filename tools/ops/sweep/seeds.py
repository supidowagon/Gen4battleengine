"""Seeds of the nightly soundness sweep. Plain Python: no nuzline or jax import.

    python -m tools.ops.sweep.seeds [<yyyymmdd>] [--n 15]
    python -m tools.ops.sweep.seeds --seeds 1..15

The second form prints the same jobs for fixed seeds (`1..15`, `3`, `1,4,9` or `1..5,9`: whole
numbers up to 2**32 - 1) on every node, whatever the night: a rerun with the same seeds on another
commit (a gate or a road test) needs no hand-written jobs file.

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
    return [f"{node} {short} {s}" for node, short in nodes(path) for s in seeds(night, node, n)]


def parse_seeds(spec: str) -> list[int]:
    """`1..15`, `3`, `1,4,9`, `1..5,9` -> the seeds in the order written (a repeat is an error)."""
    out: list[int] = []
    for part in spec.split(","):
        m = re.fullmatch(r"(\d+)(?:\.\.(\d+))?", part.strip())
        if not m:
            raise ValueError(f"bad seed list {spec!r}: expected N, A..B or a comma list of them")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ValueError(f"bad seed range {part!r}: {hi} is under {lo}")
        if hi >= 2**32:  # before the range is built: a huge B must not fill memory
            raise ValueError(f"bad seed list {spec!r}: seeds must be under 2**32")
        out.extend(range(lo, hi + 1))
    if not out or max(out) >= 2**32:
        raise ValueError(f"bad seed list {spec!r}: seeds must be under 2**32")
    if len(set(out)) != len(out):
        raise ValueError(f"bad seed list {spec!r}: a seed is listed twice")
    return out


def fixed_job_lines(seed_list: list[int], path: Path = NODES) -> list[str]:
    return [f"{node} {short} {s}" for node, short in nodes(path) for s in seed_list]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("night", nargs="?", help="yyyymmdd (default: today, local date)")
    ap.add_argument("--n", type=int, default=None, help="seeds per node (default 15)")
    ap.add_argument("--seeds", help="fixed seeds on every node, e.g. 1..15 (no night, no --n)")
    a = ap.parse_args(argv)
    if a.seeds is not None:
        if a.night or a.n is not None:
            print("--seeds takes no night and no --n", file=sys.stderr)
            return 2
        try:
            print("\n".join(fixed_job_lines(parse_seeds(a.seeds))))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        return 0
    a.n = 15 if a.n is None else a.n
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
