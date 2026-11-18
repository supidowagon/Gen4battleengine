"""Split the fast-tier test files into N parts balanced by estimated time.

Inputs: a `pytest --collect-only -q -m "not slow"` listing (node ids) and the local run log
with its "slowest 40 durations" block. A file's estimate = its known durations (from the
block) + (its other tests) x (average time of the tests not in the block).
Files named in ALONE each get a part of their own. Greedy longest-first into the rest.
Usage: python split_parts.py collect.txt local.log N outdir
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ALONE = ["tests/search/test_player_learn.py", "tests/test_cli.py"]

collect, log, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
counts = defaultdict(int)
for line in Path(collect).read_text(encoding="utf-8").splitlines():
    if "::" in line and line.startswith("tests/"):
        counts[line.split("::")[0]] += 1

text = Path(log).read_text(encoding="utf-8", errors="replace")
block = text.split("slowest 40 durations")[-1]
known = defaultdict(float)
known_ids = set()
for m in re.finditer(r"^([\d.]+)s (call|setup|teardown)\s+(\S+)", block, re.M):
    t, _, nid = m.groups()
    known[nid.split("::")[0]] += float(t)
    known_ids.add(nid)
total = float(re.search(r"in ([\d.]+)s \(", text).group(1))
ntests = sum(counts.values())
avg = (total - sum(known.values())) / max(1, ntests - len(known_ids))

# Units: whole files; a file whose estimate exceeds the per-part target is split by test
# function (file::func, all its parameter cases), so one file cannot be the long pole.
func_counts = defaultdict(int)
for line in Path(collect).read_text(encoding="utf-8").splitlines():
    if "::" in line and line.startswith("tests/"):
        func_counts[line.split("[")[0]] += 1
known_func = defaultdict(float)
for m in re.finditer(r"^([\d.]+)s (call|setup|teardown)\s+(\S+)", block, re.M):
    known_func[m.group(3).split("[")[0]] += float(m.group(1))
known_func_n = defaultdict(int)
for i in known_ids:
    known_func_n[i.split("[")[0]] += 1

def unit_est(u, c):
    return known_func[u] + (c - known_func_n[u]) * avg

file_est = {f: sum(unit_est(u, c) for u, c in func_counts.items() if u.split("::")[0] == f)
            for f in counts}
target = sum(file_est.values()) / n
units = {}
for f, e in file_est.items():
    if e > target * 0.6 and f not in ALONE:
        for u, c in func_counts.items():
            if u.split("::")[0] == f:
                units[u] = unit_est(u, c)
    else:
        units[f] = e
parts = [[f] for f in ALONE if f in units]
loads = [units[f] for f in ALONE if f in units]
parts += [[] for _ in range(n - len(parts))]
loads += [0.0] * (n - len(loads))
for u in sorted((u for u in units if u not in ALONE), key=lambda u: -units[u]):
    i = min(range(n), key=lambda j: loads[j])
    parts[i].append(u)
    loads[i] += units[u]
est = units
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("part_*.txt"):
    old.unlink()
for i, (p, load) in enumerate(zip(parts, loads)):
    (out / f"part_{i:02d}.txt").write_text("\n".join(sorted(p)) + "\n", encoding="utf-8")
    print(f"part_{i:02d}: {len(p):3d} units, est {load/60:6.1f} min")
print(f"tests {ntests}, avg residual {avg:.2f}s, est total {sum(loads)/3600:.2f} h")
