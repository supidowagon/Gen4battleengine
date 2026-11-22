"""Collect a nightly soundness sweep's results. Plain Python: no nuzline or jax import.

    python tools/ops/sweep/collect.py <out dir> [--night <yyyymmdd>] [--tip <sha8>]

<out dir> is what tools/ops/day/pod/sweep.sh wrote (jobs.txt, summary.txt, <short>_<seed>.log,
replay.txt, SWEEP_DONE or SWEEP_STOPPED). It writes <out dir>/sweep_failures.txt and prints its first
line and its class list. Always exit 0: the result is the file's first line,
  SWEEP <night> <tip sha8>: 0 FAIL: none in <n> processes          (clean)
  SWEEP <night> <tip sha8>: <n> processes, <p> PASS, <f> FAIL, <k> classes, <u> unclassified
  SWEEP <night> <tip sha8>: INCOMPLETE (<why>; ...): <n> processes, <p> PASS, ...
The clean form needs every check to hold: SWEEP_DONE present, no SWEEP_STOPPED, a summary.txt, a
jobs.txt with at least one job line, and as many summary rows as jobs.txt has job lines (never 0).
Each check that fails is named in the INCOMPLETE line: "SWEEP_STOPPED present (the sweep was
stopped)", "no SWEEP_DONE (the driver did not finish)", "no summary.txt", "no jobs.txt (the expected
count is unknown)", "jobs.txt has no job line", "0 processes in summary.txt", "<r> of <e> processes in
summary.txt". <k> counts the classes the replay gave (a CLASS KEY with its diverging fields); <u>
counts the failed processes and falsifiers in the fallback groups below, which the replay did not
classify (one such group can hold several different bugs).
The night and tip default to the run's meta.env (pack.sh writes it: NIGHT=<yyyymmdd>, the packing
date or SWEEP_NIGHT, and COMMIT=<full sha>, whose first 8 characters are the tip), read from
<out dir>/meta.env, else <out dir>/../meta.env (`<run>/sweep_out`, the fetched layout); a value
meta.env lacks (a run packed before NIGHT) comes from the out folder's name
(`...-<yyyymmdd>-<sha>`), else its parent's, else "?".
A log with more falsifying examples than @reproduce_failure lines gives the extra examples
"(repro line not matched)".

Then the class list (grouped by CLASS KEY, with a count and the seeds), then one block per
falsifier: node, seed, the @reproduce_failure line, the diverging Outcome field(s) and CLASS KEY
from replay.txt (matched per example), the drawn bag item and choice. A falsifier whose replay is
missing or failed has the key (UNREPLAYED, <node>, <bag item or ->) and the reason is said; one
whose replay ran and passed (the failure did not reproduce from the extracted draws: replay.py's
field part PASS or DRIFT-PASS) has
(NOT-REPRODUCED, <node>:<seed>, <bag item or ->). A process that failed without a falsifier (crash,
kill, a job over SWEEP_JOB_S: rc 124 or 137, no result) is grouped as (NO-FALSIFIER rc=<rc>, <node>,
-) or (LOST, <node>, -). These four are the fallback groups. Only the logs of summary.txt's rows
are read (all logs when it is missing).
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tools.ops.sweep import extract
except ImportError:  # run as a script: this folder is on sys.path
    import extract  # type: ignore[no-redef]

FALLBACK = ("(UNREPLAYED, ", "(NOT-REPRODUCED, ", "(NO-FALSIFIER ", "(LOST, ")
LOG = re.compile(r"(test_\w+?)_(\d+)")
REPRO = re.compile(r"(@reproduce_failure\(.*?\))(?: as a decorator|\s*$)")


def read_meta(out: Path) -> dict[str, str]:
    """KEY=VALUE lines of the run's meta.env (in the out dir, else in its parent), {} without one.
    The beacon codes (WATCH_*) are never read."""
    here = out.resolve()
    for f in (here / "meta.env", here.parent / "meta.env"):
        if f.is_file():
            meta = {}
            for line in f.read_text(errors="replace").splitlines():
                k, sep, v = line.strip().partition("=")
                if sep and re.fullmatch(r"[A-Z_]+", k) and not k.startswith("WATCH_"):
                    meta[k] = v.strip().strip("\"'")
            return meta
    return {}


def read_summary(out: Path) -> list[tuple[str, str, str, str]] | None:
    """(short, seed, result, rc) per summary.txt row, or None without the file."""
    f = out / "summary.txt"
    if not f.is_file():
        return None
    rows = []
    for line in f.read_text(errors="replace").splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 5:
            rows.append((c[0], c[1], c[2], c[4]))
    return rows


def replay_sections(out: Path) -> tuple[dict[str, list[str]] | None, str]:
    """{example name: its lines} from replay.txt, or (None, reason)."""
    f = out / "replay.txt"
    if not f.is_file():
        return None, "no replay.txt"
    secs: dict[str, list[str]] = {}
    cur = None
    for line in f.read_text(errors="replace").splitlines():
        if line.startswith("== CLASS KEYS"):
            break
        if line.startswith("== "):
            cur = line[3:].strip()
            secs[cur] = []
        elif cur is not None:
            secs[cur].append(line)
    if not secs:
        return None, "replay.txt has no example section (the replay did not start or crashed)"
    return secs, ""


def replayed(lines: list[str] | None) -> tuple[str | None, str, str]:
    """(class key, fields, reason) of one replay section; key None when it has none."""
    if lines is None:
        return None, "", "replay.txt has no section for this example"
    for line in lines:
        m = re.match(r"\s*CLASS KEY (\(.*\))\s*$", line)
        if m:
            key = m.group(1)
            return key, key[1:].split(", ")[0], ""
    err = [ln.strip() for ln in lines if ln.strip().startswith("ERROR")]
    return None, "", "replay failed: " + (err[0] if err else "no CLASS KEY line")


def expected_jobs(out: Path) -> int | None:
    """The job lines of jobs.txt (what sweep.sh was given), or None without the file."""
    f = out / "jobs.txt"
    if not f.is_file():
        return None
    lines = f.read_text(errors="replace").splitlines()
    return sum(1 for ln in lines if ln.strip() and not ln.lstrip().startswith("#"))


def falsifiers(out: Path, secs: dict[str, list[str]] | None, why: str, rows=None) -> list[dict]:
    """One dict per falsifying example of every log of summary.txt's rows (all logs without it)."""
    found = []
    if rows is None:
        logs = sorted(out.glob("*.log"))
    else:
        logs = [out / f"{short}_{seed}.log" for short, seed, _, _ in rows]
    for log in logs:
        m = LOG.fullmatch(log.stem)
        if not m or m.group(1) not in extract.LAYOUT or not log.is_file():
            continue
        node, seed = m.groups()
        text = log.read_text(errors="replace")
        if not extract.has_falsifier(text):
            continue
        lines = text.splitlines()
        repro = [r.group(1) for ln in lines if (r := REPRO.search(ln))]
        layout = extract.LAYOUT[node]
        for j, ex in enumerate(extract.examples(lines) or [{}]):
            name = log.stem + ("" if j == 0 else "_" + "abcdefgh"[j])
            item = choice = None
            if "item" in layout and layout.index("item") + 1 in ex:
                item = ex[layout.index("item") + 1].strip("'\"")
            if "choice" in layout and layout.index("choice") + 1 in ex:
                choice = ex[layout.index("choice") + 1]
            if secs is None:
                key, fields, reason = None, "", why
            else:
                key, fields, reason = replayed(secs.get(name))
            if fields in ("PASS", "DRIFT-PASS"):  # the test body passed on the drawn inputs
                key = f"(NOT-REPRODUCED, {node}:{seed}, {item or '-'})"
                reason = (
                    "the replay did not reproduce the failure (the test body passed on the draws)"
                )
            found.append(
                {
                    "node": node,
                    "seed": seed,
                    "name": name,
                    "repro": repro[j]
                    if j < len(repro)
                    else ("(repro line not matched)" if repro else "(no @reproduce_failure)"),
                    "item": item,
                    "choice": choice,
                    "fields": fields or "-",
                    "key": key or f"(UNREPLAYED, {node}, {item or '-'})",
                    "reason": reason,
                }
            )
    return found


def other_failures(out: Path, rows, with_falsifier: set[tuple[str, str]]) -> list[dict]:
    """Processes that failed, or left no result, without a falsifier in their log."""
    found = []
    for short, seed, result, rc in rows or []:
        if result == "PASS" or (short, seed) in with_falsifier:
            continue
        log = out / f"{short}_{seed}.log"
        tail = []
        if log.is_file():
            tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()][-3:]
        key = f"(LOST, {short}, -)" if result == "LOST" else f"(NO-FALSIFIER rc={rc}, {short}, -)"
        found.append(
            {
                "node": short,
                "seed": seed,
                "name": f"{short}_{seed}",
                "key": key,
                "tail": tail,
                "result": result,
                "rc": rc,
                "reason": "",
            }
        )
    return found


def main(argv: list[str]) -> int:
    args = list(argv)
    opts = {}
    for flag in ("--night", "--tip"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1] if i + 1 < len(args) else "?"
            del args[i : i + 2]
    if len(args) != 1:
        print("usage: collect.py <out dir> [--night <yyyymmdd>] [--tip <sha8>]", file=sys.stderr)
        return 0
    out = Path(args[0])
    here = out.resolve()
    m = None
    for name in (here.name, here.parent.name):  # <run>/sweep_out: the run folder has the name
        if m := re.search(r"(\d{8})-([0-9a-f]{7,40})", name):
            break
    meta = read_meta(out)
    mnight = meta.get("NIGHT", "") if re.fullmatch(r"\d{8}", meta.get("NIGHT", "")) else ""
    mtip = meta.get("COMMIT", "") if re.fullmatch(r"[0-9a-f]{7,40}", meta.get("COMMIT", "")) else ""
    night = opts.get("--night") or mnight or (m.group(1) if m else "?")
    tip = (opts.get("--tip") or mtip or (m.group(2) if m else "?"))[:8]
    rows = read_summary(out)
    secs, why = replay_sections(out)
    fals = falsifiers(out, secs, why, rows)
    others = other_failures(out, rows, {(f["node"], f["seed"]) for f in fals})
    classes: dict[str, list[str]] = defaultdict(list)
    reasons: dict[str, str] = {}
    for f in fals + others:
        classes[f["key"]].append(f"{f['node']}:{f['seed']}")
        if f["reason"]:
            reasons[f["key"]] = f["reason"]
    problems = []
    if (out / "SWEEP_STOPPED").exists():
        problems.append("SWEEP_STOPPED present (the sweep was stopped)")
    if not (out / "SWEEP_DONE").exists():
        problems.append("no SWEEP_DONE (the driver did not finish)")
    expected = expected_jobs(out)
    if rows is None:
        problems.append("no summary.txt")
    if expected is None:
        problems.append("no jobs.txt (the expected count is unknown)")
    elif expected == 0:
        problems.append("jobs.txt has no job line")
    if rows is not None:
        if not rows:
            problems.append("0 processes in summary.txt")
        elif expected and len(rows) != expected:
            problems.append(f"{len(rows)} of {expected} processes in summary.txt")
    n = len(rows or [])
    p = sum(1 for r in rows or [] if r[2] == "PASS")
    real = [k for k in classes if not k.startswith(FALLBACK)]
    unclassified = sum(len(v) for k, v in classes.items() if k.startswith(FALLBACK))
    counts = (
        f"{n} processes, {p} PASS, {n - p} FAIL, {len(real)} classes, {unclassified} unclassified"
    )
    if problems:
        head = f"SWEEP {night} {tip}: INCOMPLETE ({'; '.join(problems)}): {counts}"
        if rows is None:
            head += f" ({len(fals)} falsifiers in the logs)"
    elif not fals and not others:
        head = f"SWEEP {night} {tip}: 0 FAIL: none in {n} processes"
    else:
        head = f"SWEEP {night} {tip}: {counts}"
    top = [head]
    if classes:
        top += ["", "CLASSES (CLASS KEY: count, seeds)"]
        for key, seeds in sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            top.append(f"  {len(seeds):3d}  {key}  seeds {', '.join(seeds)}")
            if key in reasons:
                top.append(f"       reason: {reasons[key]}")
    body = []
    for f in fals:
        body += [
            f"== {f['name']}  node {f['node']}  seed {f['seed']}",
            f"  {f['repro']}",
            f"  diverging field(s): {f['fields']}",
            f"  CLASS KEY: {f['key']}",
            f"  bag item: {f['item'] or '-'}  choice: {f['choice'] or '-'}",
        ]
        if f["reason"]:
            body.append(f"  replay: {f['reason']}")
        body.append("")
    for f in others:
        body += [
            f"== {f['name']}  node {f['node']}  seed {f['seed']}  {f['result']} rc={f['rc']}, no falsifier in the log",
            f"  CLASS KEY: {f['key']}",
            *[f"  log tail: {t[:200]}" for t in f["tail"]],
            "",
        ]
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(top + ([""] + body if body else [])).rstrip() + "\n"
    (out / "sweep_failures.txt").write_text(text, encoding="utf-8")
    print("\n".join(top))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
