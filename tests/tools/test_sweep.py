"""The nightly soundness sweep (tools/ops/sweep, tools/ops/day/pod/sweep.sh, sweep_body.sh).
No engine import (no nuzline.engine, no jax): seeds, extract and collect are plain Python, and
sweep.sh runs under bash with a fake python and a fake cgroup folder."""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.tools.test_pod_gate import BASH, _clean_env, _linux_tools, _posix, _tool_body
from tools.ops.sweep import collect, extract, seeds

ROOT = Path(__file__).resolve().parents[2]
POD = ROOT / "tools" / "ops" / "day" / "pod"
linux = pytest.mark.skipif(
    not _linux_tools(), reason="needs setsid, flock and a procps ps (the pod)"
)
NODE_A = "tests/engine/test_enumerate.py::test_mask_sound"
NODE_B = "tests/engine/test_enumerate.py::test_mask_sound_with_a_trainer_item"
SHORT_A, SHORT_B = "test_mask_sound", "test_mask_sound_with_a_trainer_item"


# --- seeds -------------------------------------------------------------------------------------


def test_seeds_are_deterministic_and_distinct():
    a = seeds.seeds("20261122", NODE_A)
    assert a == seeds.seeds("20261122", NODE_A)
    assert len(a) == 15 == len(set(a))
    assert all(0 <= s < 2**32 for s in a)
    assert len(seeds.seeds("20261122", NODE_A, 4)) == 4
    assert seeds.seeds("20261122", NODE_A, 4) == a[:4]


def test_seeds_differ_across_nights_and_nodes():
    a = seeds.seeds("20261122", NODE_A)
    assert a != seeds.seeds("20261123", NODE_A)
    assert a != seeds.seeds("20261122", NODE_B)
    assert not set(a) & set(seeds.seeds("20261123", NODE_A))


def test_nodes_list_is_the_committed_one():
    got = seeds.nodes()
    assert (NODE_A, SHORT_A) in got and (NODE_B, SHORT_B) in got


def test_seeds_cli_prints_one_job_line_per_process(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "tools.ops.sweep.seeds", "20261122"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )  # fmt: skip
    lines = r.stdout.splitlines()
    assert len(lines) == 30
    for i, node in enumerate([(NODE_A, SHORT_A)] * 15 + [(NODE_B, SHORT_B)] * 15):
        n, short, seed = lines[i].split(" ")
        assert (n, short) == node
        assert seed == str(seeds.seeds("20261122", node[0])[i % 15])
    r2 = subprocess.run(
        [sys.executable, "-m", "tools.ops.sweep.seeds", "20261122", "--n", "2"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )  # fmt: skip
    assert len(r2.stdout.splitlines()) == 4
    bad = subprocess.run(
        [sys.executable, "-m", "tools.ops.sweep.seeds", "yesterday"],
        check=False, capture_output=True, text=True, cwd=ROOT,
    )  # fmt: skip
    assert bad.returncode == 2


def test_seeds_without_a_date_use_today():
    r = subprocess.run(
        [sys.executable, "-m", "tools.ops.sweep.seeds"],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )  # fmt: skip
    today = time.strftime("%Y%m%d")
    assert r.stdout.splitlines()[0].endswith(f" {seeds.seeds(today, NODE_A)[0]}")


# --- fixture logs ------------------------------------------------------------------------------


def _fail_log(node: str, seed: int, item: str | None, choice: int | None, repro: str) -> str:
    """A short log with the structure of a real Hypothesis failure (draws, reproduce line)."""
    draws = [
        "E       Draw 1: BattleState(species=Array([[318, 453],",
        "E              [403, 408]], dtype=int16))",
    ]
    if item is not None:
        draws += [
            f"E       Draw 2: '{item}'",
            f"E       Draw 3: {choice}",
            "E       Draw 4: Outcome(speed_tie=Array(0, dtype=int8))",
            "E       Draw 5: 170886",
        ]
    else:
        draws += [
            "E       Draw 2: Array([3, 4], dtype=int32)",
            "E       Draw 3: Outcome(speed_tie=Array(0, dtype=int8))",
            "E       Draw 4: 7",
        ]
    return (
        "\n".join(
            [
                "=================== FAILURES ===================",
                "E       AssertionError: assert [b'a'] == [b'b']",
                f"E       Failing test case: {node.rsplit('::', 1)[-1]}(",
                "E           tables=Tables(species_base_stats=Array([[  0,   0],",
                "E              ...,",
                "E              [ 50,  65]], dtype=int16)),",
                *draws,
                "E       Explanation:",
                "E           These lines were always and only run by failing test cases:",
                "E               /work/x/repo/tests/strategies.py:806",
                "E       ",
                f"E       You can reproduce this test case by temporarily adding {repro} as a decorator on your test function",
                "1 failed, 1 warning in 1104.23s (0:18:24)",
            ]
        )
        + "\n"
    )


REPRO_1 = "@reproduce_failure('6.170.13', b'AXicbVVb')"
REPRO_2 = "@reproduce_failure('6.170.13', b'AXicQQQQ')"
REPRO_3 = "@reproduce_failure('6.170.13', b'AXicRRRR')"


def _done(out: Path, jobs: list[tuple[str, int]]) -> None:
    """jobs.txt (the expected count) and SWEEP_DONE, as sweep.sh leaves a finished sweep."""
    full = {SHORT_A: NODE_A, SHORT_B: NODE_B}
    (out / "jobs.txt").write_text(
        "".join(f"{full[short]} {short} {seed}\n" for short, seed in jobs), encoding="utf-8"
    )
    (out / "SWEEP_DONE").write_text("1\n", encoding="utf-8")


def _sweep_out(tmp_path: Path, name: str = "sweep-20261122-abcdef0123456789") -> Path:
    out = tmp_path / name
    out.mkdir()
    rows = [(SHORT_A, 1, "PASS", 1100, 0, "11.5"), (SHORT_A, 2, "PASS", 1100, 0, "11.5"),
            (SHORT_B, 6, "FAIL", 1100, 1, "12.7"), (SHORT_B, 7, "FAIL", 1100, 1, "11.9"),
            (SHORT_B, 8, "FAIL", 1100, 1, "11.9"), (SHORT_B, 9, "PASS", 1100, 0, "11.5")]  # fmt: skip
    (out / "summary.txt").write_text(
        "node\tseed\tresult\twall_s\trc\tpeak_GB\n"
        + "".join("\t".join(map(str, r)) + "\n" for r in rows),
        encoding="utf-8",
    )
    _done(out, [(r[0], r[1]) for r in rows])
    for r in rows:
        (out / f"{r[0]}_{r[1]}.log").write_text("1 passed in 1s\n", encoding="utf-8")
    (out / f"{SHORT_B}_6.log").write_text(
        _fail_log(NODE_B, 6, "X Attack", 3, REPRO_1), encoding="utf-8"
    )
    (out / f"{SHORT_B}_7.log").write_text(
        _fail_log(NODE_B, 7, "X Attack", 5, REPRO_2), encoding="utf-8"
    )
    (out / f"{SHORT_B}_8.log").write_text(
        _fail_log(NODE_B, 8, "Potion", 1, REPRO_3), encoding="utf-8"
    )
    return out


FAKE_REPLAY = f"""\
== {SHORT_B}_6
  node {SHORT_B} seed 170886
  RESULT FAIL bad rows 3 of 40
  FIELD hit bad rows 3 of 40 first 0
  CLASS KEY (hit, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)
== {SHORT_B}_7
  RESULT FAIL bad rows 1 of 40
  CLASS KEY (hit, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)
== CLASS KEYS
    2  (hit, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)
done
"""


def test_extract_cuts_the_draws_out_of_a_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"{SHORT_B}_6.log").write_text(
        _fail_log(NODE_B, 6, "X Attack", 3, REPRO_1), encoding="utf-8"
    )
    (logs / f"{SHORT_A}_2.log").write_text(
        _fail_log(NODE_A, 2, None, None, REPRO_2), encoding="utf-8"
    )
    (logs / f"{SHORT_A}_3.log").write_text("1 passed in 1s\n", encoding="utf-8")
    (logs / "driver.log").write_text("START\n", encoding="utf-8")
    out = tmp_path / "ex"
    r = subprocess.run([sys.executable, str(ROOT / "tools/ops/sweep/extract.py"), str(logs), str(out)],
                       check=True, capture_output=True, text=True)  # fmt: skip
    assert "PASS/no falsifier" in r.stdout
    d = out / f"{SHORT_B}_6"
    assert (d / "node.txt").read_text().strip() == SHORT_B
    assert (d / "item.txt").read_text().strip() == "'X Attack'"
    assert (d / "choice.txt").read_text().strip() == "3"
    assert (d / "seed.txt").read_text().strip() == "170886"
    assert (d / "outcome.txt").read_text().startswith("Outcome(")
    assert (d / "state.txt").read_text().splitlines()[0].startswith("BattleState(")
    assert "[403, 408]" in (d / "state.txt").read_text()
    assert (
        (d / "reproduce.txt")
        .read_text()
        .strip()
        .endswith(REPRO_1 + " as a decorator on your test function")
    )
    a = out / f"{SHORT_A}_2"
    assert (a / "acts.txt").read_text().strip() == "Array([3, 4], dtype=int32)"
    assert (a / "seed.txt").read_text().strip() == "7"
    assert not (out / f"{SHORT_A}_3").exists()
    assert extract.examples(_fail_log(NODE_B, 6, "X", 1, REPRO_1).splitlines())[0][2] == "'X'"


def _group_log(log: str) -> str:
    """The falsifier as pytest prints it inside an exception group: `E       | ` before each
    line, `+-` separator lines around it."""
    out = ["E     +-+---------------- 1 ----------------"]
    for ln in log.splitlines():
        out.append("E       | " + ln[1:].lstrip() if ln.startswith("E") else ln)
    out.insert(-1, "E       +------------------------------------")
    return "\n".join(out) + "\n"


def test_extract_reads_draws_behind_an_exception_group_pipe(tmp_path):
    lines = _group_log(_fail_log(NODE_B, 6, "X Attack", 3, REPRO_1)).splitlines()
    assert any(ln.startswith("E       | Draw 1: ") for ln in lines)
    ex = extract.examples(lines)
    assert len(ex) == 1 and set(ex[0]) == {1, 2, 3, 4, 5}
    assert ex[0][1].startswith("BattleState(") and "|" not in ex[0][1] and "[403, 408]" in ex[0][1]
    assert ex[0][2] == "'X Attack'" and ex[0][3] == "3" and ex[0][5] == "170886"
    logs, out = tmp_path / "logs", tmp_path / "ex"
    logs.mkdir()
    (logs / f"{SHORT_B}_6.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "tools/ops/sweep/extract.py"), str(logs), str(out)],
                       check=True, capture_output=True, text=True)  # fmt: skip
    assert f"{SHORT_B}_6 {SHORT_B} item='X Attack' choice=3 seed=170886 ok" in r.stdout
    assert (out / f"{SHORT_B}_6" / "item.txt").read_text().strip() == "'X Attack'"


def test_extract_goes_on_past_a_draw_that_is_not_a_literal(tmp_path):
    logs, out = tmp_path / "logs", tmp_path / "ex"
    logs.mkdir()
    bad = _fail_log(NODE_B, 6, "X Attack", 3, REPRO_1).replace(
        "E       Draw 3: 3", "E       Draw 3: <Act 3>"
    )
    (logs / f"{SHORT_B}_6.log").write_text(bad, encoding="utf-8")
    (logs / f"{SHORT_B}_7.log").write_text(
        _fail_log(NODE_B, 7, "Potion", 1, REPRO_2), encoding="utf-8"
    )
    r = subprocess.run([sys.executable, str(ROOT / "tools/ops/sweep/extract.py"), str(logs), str(out)],
                       check=True, capture_output=True, text=True)  # fmt: skip
    assert "choice=<Act 3>" in r.stdout
    assert "PROBLEMS: choice (Draw 3) not a literal: SyntaxError" in r.stdout
    assert f"{SHORT_B}_7 {SHORT_B} item='Potion' choice=1 seed=170886 ok" in r.stdout


# --- collect -----------------------------------------------------------------------------------


def test_collect_writes_blocks_and_classes_from_the_replay(tmp_path):
    out = _sweep_out(tmp_path)
    (out / "replay.txt").write_text(FAKE_REPLAY, encoding="utf-8")
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert (
        first == "SWEEP 20261122 abcdef01: 6 processes, 3 PASS, 3 FAIL, 1 classes, 1 unclassified"
    )
    for seed, repro, item, choice in [
        (6, REPRO_1, "X Attack", 3),
        (7, REPRO_2, "X Attack", 5),
        (8, REPRO_3, "Potion", 1),
    ]:
        block = text.split(f"== {SHORT_B}_{seed}  node {SHORT_B}  seed {seed}\n", 1)[1].split(
            "\n\n", 1
        )[0]
        assert repro in block and f"bag item: {item}  choice: {choice}" in block
    b6 = text.split(f"== {SHORT_B}_6  ", 1)[1].split("\n\n", 1)[0]
    assert "diverging field(s): hit" in b6
    assert "CLASS KEY: (hit, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)" in b6
    classes = text.split("\n\n")[1]
    # dedup by CLASS KEY: seeds 6 and 7 share a class, seed 8 has no replay section
    assert (
        f"  2  (hit, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)  seeds {SHORT_B}:6, {SHORT_B}:7"
        in classes
    )
    assert f"  1  (UNREPLAYED, {SHORT_B}, Potion)  seeds {SHORT_B}:8" in classes
    assert "reason: replay.txt has no section for this example" in classes


@pytest.mark.parametrize(
    ("replay", "reason"),
    [
        (None, "no replay.txt"),
        ("", "no example section"),
        (
            f"== {SHORT_B}_6\n  ERROR AssertionError boom\n== CLASS KEYS\n",
            "replay failed: ERROR AssertionError boom",
        ),
    ],
)
def test_collect_marks_a_missing_or_failed_replay_unreplayed(tmp_path, replay, reason):
    out = _sweep_out(tmp_path)
    if replay is not None:
        (out / "replay.txt").write_text(replay, encoding="utf-8")
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text(encoding="utf-8")
    assert (
        f"(UNREPLAYED, {SHORT_B}, X Attack)" in text and f"(UNREPLAYED, {SHORT_B}, Potion)" in text
    )
    assert reason in text


def test_collect_reads_night_and_tip_from_the_run_folder_above_sweep_out(tmp_path):
    run = tmp_path / "sweep-20261122-abcdef01"
    run.mkdir()
    out = _sweep_out(run, "sweep_out")
    assert collect.main([str(out)]) == 0
    first = (out / "sweep_failures.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("SWEEP 20261122 abcdef01: 6 processes")


def test_collect_marks_examples_past_the_last_repro_line(tmp_path):
    out = _sweep_out(tmp_path)
    log = _fail_log(NODE_B, 6, "X Attack", 3, REPRO_1)
    draws = [ln for ln in log.splitlines() if "Draw" in ln or ln.startswith("E              [403")]
    two = log.replace("E       Explanation:", "\n".join(draws) + "\nE       Explanation:", 1)
    (out / f"{SHORT_B}_6.log").write_text(two, encoding="utf-8")
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text(encoding="utf-8")
    a = text.split(f"== {SHORT_B}_6  ", 1)[1].split("\n\n", 1)[0]
    b = text.split(f"== {SHORT_B}_6_b  ", 1)[1].split("\n\n", 1)[0]
    assert REPRO_1 in a and "(repro line not matched)" in b and REPRO_1 not in b


def test_collect_clean_sweep_first_line_and_exit_zero(tmp_path):
    out = tmp_path / "x-20261201-0123456789abcdef"
    out.mkdir()
    (out / "summary.txt").write_text(
        "node\tseed\tresult\twall_s\trc\tpeak_GB\n"
        + "".join(f"{SHORT_A}\t{i}\tPASS\t10\t0\t1.0\n" for i in (1, 2, 3)),
        encoding="utf-8",
    )
    _done(out, [(SHORT_A, 1), (SHORT_A, 2), (SHORT_A, 3)])
    r = subprocess.run([sys.executable, str(ROOT / "tools/ops/sweep/collect.py"), str(out)],
                       check=False, capture_output=True, text=True)  # fmt: skip
    assert r.returncode == 0
    assert r.stdout.strip() == "SWEEP 20261201 01234567: 0 FAIL: none in 3 processes"
    assert (
        out / "sweep_failures.txt"
    ).read_text() == "SWEEP 20261201 01234567: 0 FAIL: none in 3 processes\n"


def test_collect_a_failure_without_a_falsifier_is_its_own_class(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "summary.txt").write_text(
        "node\tseed\tresult\twall_s\trc\tpeak_GB\n"
        f"{SHORT_A}\t1\tFAIL\t10\t137\t1.0\n"
        f"{SHORT_A}\t2\tLOST\t-\t-\t-\n",
        encoding="utf-8",
    )
    (out / f"{SHORT_A}_1.log").write_text("Killed\n", encoding="utf-8")
    _done(out, [(SHORT_A, 1), (SHORT_A, 2)])
    assert collect.main([str(out), "--night", "20261122", "--tip", "deadbeefcafe"]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert (
        text.splitlines()[0]
        == "SWEEP 20261122 deadbeef: 2 processes, 0 PASS, 2 FAIL, 0 classes, 2 unclassified"
    )
    assert f"(NO-FALSIFIER rc=137, {SHORT_A}, -)" in text and f"(LOST, {SHORT_A}, -)" in text


# --- sweep_body.sh -----------------------------------------------------------------------------


def test_sweep_body_ships_the_jobs_in_a_tar_and_stays_short(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("\n".join(seeds.job_lines("20261122")) + "\n", encoding="utf-8", newline="\n")
    r, body, send = _tool_body(
        tmp_path,
        (ROOT / "tools/ops/day/sweep_body.sh").as_posix(),
        "sweep-20261122-abc",
        _posix(jobs),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    script = body["command"][2]
    assert len(r.stdout.strip().splitlines()[-1]) <= 950 and "base64" not in script
    assert "bash start.sh" in script and "sweep.tar" in script
    names = subprocess.run(
        ["tar", "-tf", _posix(send / "sweep.tar")], check=True, capture_output=True, text=True
    ).stdout.split()
    assert sorted(names) == ["start.sh", "sweep_jobs.txt"]
    start = subprocess.run(
        ["tar", "-xOf", _posix(send / "sweep.tar"), "start.sh"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "R=/work/sweep-20261122-abc" in start
    # absolute jobs and out paths: sweep.sh works from <run>/repo
    assert 'setsid nohup bash bin/sweep.sh "$R/$J" "$R/$o"' in start
    assert "o=sweep_out_$e" in start  # a second sweep of the run gets a new out dir
    assert "*[!0-9]*|0|1)" in start  # a pid of 0 or 1 in sweep.pid is never signalled
    assert "export SWEEP_CAP" not in start  # unset: sweep.sh takes the cap from the pod's limit
    bad = tmp_path / "bad.txt"
    bad.write_text("tests/x.py::t only_two\n", encoding="utf-8")
    r2, _, _ = _tool_body(
        tmp_path,
        (ROOT / "tools/ops/day/sweep_body.sh").as_posix(),
        "sweep-20261122-abc",
        _posix(bad),
    )
    assert r2.returncode == 2


# --- sweep.sh ----------------------------------------------------------------------------------

FAKE_PYTHON = r"""#!/bin/bash
# fake <repo>/.venv/bin/python: the pytest job, extract.py and replay.py
case "$1" in
  *extract.py) mkdir -p "$3"; echo extracted "$2" > "$3/extract.ok"; exit 0;;
  *replay.py) echo "== replayed $3"; echo "replay $$ $(date +%s.%N)" >> "$FAKE_TRACE"; sleep "${FAKE_REPLAY_S:-0}"; exit 0;;
esac
for a in "$@"; do case "$a" in --hypothesis-seed=*) seed=${a#*=};; esac; done
T="$FAKE_TRACE"
echo "start $seed $(date +%s.%N) $NUZLINE_HYPOTHESIS_PROFILE $*" >> "$T"
if [ "$seed" = "${FAKE_HANG_SEED:-}" ]; then sleep "${FAKE_HANG_S:-600}"; fi
sleep "${FAKE_JOB_S:-1}"
echo "end $seed" >> "$T"
if [ "$seed" = "${FAKE_FAIL_SEED:-}" ]; then echo "E       Failing test case: t("; exit 1; fi
echo "1 passed"; exit 0
"""


def _run_folder(tmp_path: Path) -> Path:
    d = tmp_path / "work" / "sweep-x"
    (d / "bin").mkdir(parents=True)
    shutil.copy(POD / "sweep.sh", d / "bin" / "sweep.sh")
    shutil.copy(POD / "gate.sh", d / "bin" / "gate.sh")
    py = d / "repo" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
    os.chmod(py, 0o755)
    return d


def _env(
    tmp_path: Path, mem_bytes: int, limit: str | None = str(128 * 2**30), **kw: str
) -> dict[str, str]:
    """A fake cgroup (memory.current; memory.max = <limit>, no file when None), a gate lock, a
    claims folder and a /work of the test's own."""
    cg = tmp_path / "cg"
    cg.mkdir(exist_ok=True)
    (cg / "memory.current").write_text(f"{mem_bytes}\n", encoding="utf-8", newline="\n")
    if limit is not None:
        (cg / "memory.max").write_text(f"{limit}\n", encoding="utf-8", newline="\n")
    return {**_clean_env(), "CGROOT": _posix(cg), "FAKE_TRACE": _posix(tmp_path / "trace.txt"), "SWEEP_SPACING": "0",
            "SWEEP_POLL": "0.2", "SWEEP_TOOLS": _posix(tmp_path / "tools"), "RUNNER_LOCK": _posix(tmp_path / "lock"),
            "CLAIMS": _posix(tmp_path / "claims"), "WORK": _posix(tmp_path / "work"), **kw}  # fmt: skip


def _jobs(tmp_path: Path, seeds_: list[int]) -> Path:
    j = tmp_path / "jobs.txt"
    j.write_text(
        "# comment\n\n" + "".join(f"{NODE_A} {SHORT_A} {s}\n" for s in seeds_),
        encoding="utf-8",
        newline="\n",
    )
    return j


def _live(pgid: str, args: str) -> list[str]:
    """The processes that live (state not Z) in the group <pgid> or with exactly the command line
    <args>. Never a text search over the pod: other runs' processes are on it too."""
    ps = subprocess.run(["ps", "-ww", "-eo", "pgid=,stat=,args="], check=True, capture_output=True, text=True)  # fmt: skip
    rows = [ln.split(None, 2) for ln in ps.stdout.splitlines()]
    return [" ".join(r) for r in rows
            if not r[1].startswith("Z") and (r[0] == pgid or r[2:] == [args])]  # fmt: skip


def _max_concurrent(trace: Path) -> int:
    cur = best = 0
    for line in trace.read_text().splitlines():
        cur += {"start": 1, "end": -1}.get(line.split()[0], 0)
        best = max(best, cur)
    return best


@linux
def test_sweep_runs_the_jobs_under_the_p_cap_and_writes_the_results(tmp_path):
    d = _run_folder(tmp_path)
    jobs = _jobs(tmp_path, [11, 12, 13, 14, 15, 16])
    env = _env(tmp_path, 10 * 10**9, SWEEP_P="2", FAKE_FAIL_SEED="13")
    r = subprocess.run(
        [BASH, _posix(d / "bin/sweep.sh"), _posix(jobs)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr  # a failing job is a result, not an error
    out = d / "sweep_out"
    assert (out / "SWEEP_DONE").is_file()
    trace = tmp_path / "trace.txt"
    assert _max_concurrent(trace) == 2
    starts = [ln for ln in trace.read_text().splitlines() if ln.startswith("start")]
    assert len(starts) == 6
    assert all(" explore " in ln for ln in starts)
    assert all(
        f"-p no:cacheprovider -q -x {NODE_A} --hypothesis-seed=" in ln
        and ln.rstrip().endswith("-m")  # `echo "$*"` ends `-m ` (the empty -m "")
        for ln in starts
    )
    rows = [ln.split("\t") for ln in (out / "summary.txt").read_text().splitlines()]
    assert rows[0] == ["node", "seed", "result", "wall_s", "rc", "peak_GB"]
    assert [(x[0], x[1], x[2], x[4]) for x in rows[1:]] == [
        (SHORT_A, str(s), "FAIL" if s == 13 else "PASS", "1" if s == 13 else "0")
        for s in (11, 12, 13, 14, 15, 16)
    ]
    assert (out / f"{SHORT_A}_13.log").read_text().startswith("E       Failing test case")
    assert (out / f"{SHORT_A}_11.time").read_text().startswith("wall=")
    assert "START test_mask_sound_11" in (out / "driver.log").read_text()
    # a falsifier in a log starts the replay step (the fake python stands in for extract and replay)
    assert (out / "replay_in" / "extract.ok").is_file() and "== replayed" in (
        out / "replay.txt"
    ).read_text()
    # the replay is a gate unit like a job: started through the gate, its run file gone after it
    assert re.search(r"REPLAY_RUN pgid=[0-9]+ ", (out / "driver.log").read_text())
    assert "REPLAY_END rc=0" in (out / "driver.log").read_text()
    assert (d / "sweep" / "out" / "replay.w").read_text().strip() == "13"
    assert not list((d / "sweep" / "out" / "run").iterdir())


def _driver_output(d: Path, p: subprocess.Popen, out: str = "sweep_out") -> str:
    """sweep.sh's own stdout/stderr and driver.log, for an assertion message."""
    try:
        own = p.communicate(timeout=5)[0] or ""
    except (subprocess.TimeoutExpired, ValueError):
        own = "(no output read)"
    log = d / out / "driver.log"
    text = log.read_text() if log.is_file() else "(no driver.log)"
    return f"sweep.sh rc {p.returncode}\n--- its output:\n{own}\n--- driver.log:\n{text}"


@linux
def test_sweep_waits_while_container_memory_plus_w_is_over_the_cap(tmp_path):
    d = _run_folder(tmp_path)
    jobs = _jobs(tmp_path, [1, 2])
    env = _env(tmp_path, 185 * 10**9, SWEEP_P="5", SWEEP_W="13", SWEEP_CAP="190", FAKE_JOB_S="0")
    p = subprocess.Popen(
        [BASH, _posix(d / "bin/sweep.sh"), _posix(jobs)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(2)
        assert (
            p.poll() is None and not (tmp_path / "trace.txt").exists()
        )  # 185 + 13 > 190: no job yet
        # 177 + 13 = 190. Replaced, not rewritten in place: write_text truncates first, and a
        # reading between the truncation and the write sees an empty file, which the driver
        # takes for an unreadable memory file (it stops, exit 1), as it must on a pod.
        tmp = tmp_path / "cg" / "memory.current.tmp"
        tmp.write_text(f"{177 * 10**9}\n", encoding="utf-8", newline="\n")
        os.replace(tmp, tmp_path / "cg" / "memory.current")
        rc = p.wait(timeout=60)
        assert rc == 0, _driver_output(d, p)
    finally:
        p.kill()
    trace = (tmp_path / "trace.txt").read_text()
    assert trace.count("start") == 2
    assert (d / "sweep_out" / "SWEEP_DONE").is_file()
    assert not (d / "sweep_out" / "replay.txt").exists()  # no falsifier, no replay


@linux
def test_sweep_stops_when_no_memory_reader_exists(tmp_path):
    d = _run_folder(tmp_path)
    jobs = _jobs(tmp_path, [1])
    env = {**_env(tmp_path, 0), "CGROOT": _posix(tmp_path / "nocg")}
    r = subprocess.run(
        [BASH, _posix(d / "bin/sweep.sh"), _posix(jobs)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert r.returncode == 1 and "no cgroup memory file" in r.stderr
    assert not (d / "sweep_out" / "SWEEP_DONE").exists()


@linux
@pytest.mark.parametrize(
    ("limit", "cap"),
    [
        ("max", 88),
        (str(2**63 - 4096), 88),
        (str(128 * 2**30), 88),
        (str(256 * 2**30), 200),
        (str(100 * 10**9), 60),
    ],
)
def test_sweep_cap_comes_from_the_container_limit(tmp_path, limit, cap):
    d = _run_folder(tmp_path)
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, []))], check=False,
                       capture_output=True, text=True, env=_env(tmp_path, 10**9, limit, SWEEP_P="1"),
                       timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert f" CAP={cap} (limit " in (d / "sweep_out" / "driver.log").read_text()


@linux
@pytest.mark.parametrize(
    ("limit", "why"),
    [(None, "container memory limit unreadable"), ("lots", "container memory limit unreadable"),
     (str(50 * 10**9), "cap 10GB (limit 50GB) is under one job's weight SWEEP_W 13GB")],
)  # fmt: skip
def test_sweep_refuses_without_a_limit_or_with_too_small_a_cap(tmp_path, limit, why):
    d = _run_folder(tmp_path)
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], check=False,
                       capture_output=True, text=True, env=_env(tmp_path, 10**9, limit), timeout=60)  # fmt: skip
    assert r.returncode == 1 and why in r.stderr, r.stderr
    assert not (tmp_path / "trace.txt").exists()
    assert not (d / "sweep_out" / "SWEEP_DONE").exists()


@linux
def test_sweep_holds_its_starts_while_a_gate_claim_exists(tmp_path):
    d = _run_folder(tmp_path)
    claims = tmp_path / "claims"
    claims.mkdir()
    holder = subprocess.Popen([BASH, "-c", "sleep 60"])  # a live pid for the claim
    claim = claims / "gate-run.u001"
    claim.write_text(f"{int(time.time())} 40 {holder.pid}\n", encoding="utf-8", newline="\n")
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0")
    p = subprocess.Popen([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    try:
        time.sleep(2)
        assert p.poll() is None and not (tmp_path / "trace.txt").exists()
        assert "HOLD claim gate-run.u001" in (d / "sweep_out" / "driver.log").read_text()
        claim.unlink()
        rc = p.wait(timeout=60)
        assert rc == 0, _driver_output(d, p)
    finally:
        p.kill()
        holder.kill()
    assert (tmp_path / "trace.txt").read_text().count("start") == 1


@linux
def test_sweep_term_stops_the_jobs_without_sweep_done(tmp_path):
    d = _run_folder(tmp_path)
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="30", SWEEP_P="2", SWEEP_POLL="1")
    p = subprocess.Popen([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1, 2, 3]))], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    trace = tmp_path / "trace.txt"
    try:
        for _ in range(100):
            if trace.exists() and trace.read_text().count("start") == 2:
                break
            time.sleep(0.1)
        p.terminate()
        assert p.wait(timeout=30) == 143
    finally:
        p.kill()
    time.sleep(1)
    out = d / "sweep_out"
    assert (out / "SWEEP_STOPPED").is_file() and not (out / "SWEEP_DONE").exists()
    assert "STOPPED by TERM" in (out / "driver.log").read_text()
    left = subprocess.run(["ps", "-ww", "-eo", "args="], check=True, capture_output=True, text=True).stdout
    assert _posix(d / "repo") not in left  # the fake pytest processes are gone
    assert "end" not in trace.read_text().split()


def test_sweep_scripts_parse_and_ship_in_the_pod_bin():
    for f in (POD / "sweep.sh", ROOT / "tools/ops/day/sweep_body.sh"):
        assert subprocess.run([BASH, "-n", _posix(f)], check=False).returncode == 0
    assert (POD / "sweep.sh") in set(
        POD.glob("*.sh")
    )  # pod_bin (common.sh) packs every *.sh of pod/
    assert re.search(r"\*\.sh", (ROOT / "tools/ops/common.sh").read_text(encoding="utf-8"))


# --- collect: the first line is clean only for a whole sweep -----------------------------------

HEADER = "node\tseed\tresult\twall_s\trc\tpeak_GB\n"


def _clean_out(tmp_path: Path, n: int = 3) -> Path:
    """A finished clean sweep of n processes: jobs.txt, summary.txt, logs, SWEEP_DONE."""
    out = tmp_path / "sweep-20261122-abcdef01"
    out.mkdir()
    (out / "summary.txt").write_text(
        HEADER + "".join(f"{SHORT_A}\t{i}\tPASS\t10\t0\t1.0\n" for i in range(1, n + 1)),
        encoding="utf-8",
    )
    for i in range(1, n + 1):
        (out / f"{SHORT_A}_{i}.log").write_text("1 passed in 1s\n", encoding="utf-8")
    _done(out, [(SHORT_A, i) for i in range(1, n + 1)])
    return out


def _zero_rows(out: Path) -> None:
    (out / "summary.txt").write_text(HEADER, encoding="utf-8")


def _short(out: Path) -> None:
    rows = (out / "summary.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    (out / "summary.txt").write_text("".join(rows[:-1]), encoding="utf-8")


def _stopped(out: Path) -> None:
    (out / "SWEEP_DONE").unlink()
    (out / "SWEEP_STOPPED").write_text("1\n", encoding="utf-8")


def _no_done(out: Path) -> None:
    (out / "SWEEP_DONE").unlink()


def _no_jobs(out: Path) -> None:
    (out / "jobs.txt").unlink()


def _empty_jobs(out: Path) -> None:
    (out / "jobs.txt").write_text("", encoding="utf-8")
    _zero_rows(out)


def _no_summary(out: Path) -> None:
    (out / "summary.txt").unlink()


@pytest.mark.parametrize(
    ("spoil", "first"),
    [
        (_zero_rows, "INCOMPLETE (0 processes in summary.txt): 0 processes, 0 PASS, 0 FAIL, 0 classes, 0 unclassified"),
        (_short, "INCOMPLETE (2 of 3 processes in summary.txt): 2 processes, 2 PASS, 0 FAIL, 0 classes, 0 unclassified"),
        (_stopped, ("INCOMPLETE (SWEEP_STOPPED present (the sweep was stopped); no SWEEP_DONE (the driver did not "
                   "finish)): 3 processes, 3 PASS, 0 FAIL, 0 classes, 0 unclassified")),
        (_no_done, ("INCOMPLETE (no SWEEP_DONE (the driver did not finish)): 3 processes, 3 PASS, 0 FAIL, 0 classes, "
                   "0 unclassified")),
        (_no_jobs, ("INCOMPLETE (no jobs.txt (the expected count is unknown)): 3 processes, 3 PASS, 0 FAIL, "
                   "0 classes, 0 unclassified")),
        (_empty_jobs, ("INCOMPLETE (jobs.txt has no job line; 0 processes in summary.txt): 0 processes, 0 PASS, "
                      "0 FAIL, 0 classes, 0 unclassified")),
        (_no_summary, ("INCOMPLETE (no summary.txt): 0 processes, 0 PASS, 0 FAIL, 0 classes, 0 unclassified "
                      "(0 falsifiers in the logs)")),
    ],
)  # fmt: skip
def test_collect_names_every_reason_a_sweep_is_not_whole(tmp_path, spoil, first):
    out = _clean_out(tmp_path)
    assert collect.main([str(out)]) == 0
    assert (out / "sweep_failures.txt").read_text().splitlines()[0] == (
        "SWEEP 20261122 abcdef01: 0 FAIL: none in 3 processes"
    )
    spoil(out)
    assert collect.main([str(out)]) == 0
    assert (out / "sweep_failures.txt").read_text().splitlines()[0] == (
        f"SWEEP 20261122 abcdef01: {first}"
    )


def test_collect_a_falsifier_the_replay_does_not_reproduce_is_named_per_seed(tmp_path):
    out = _sweep_out(tmp_path)
    (out / "replay.txt").write_text(
        f"== {SHORT_B}_6\n  RESULT PASS bad rows 0 of 40\n"
        f"  CLASS KEY (PASS, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)\n"
        f"== {SHORT_B}_7\n  CLASS KEY (PASS, X Attack(57), Pressure(46)/Static(9), BAG/EFFECT_HIT)\n"
        "== CLASS KEYS\n",
        encoding="utf-8",
    )
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert text.splitlines()[0].endswith("3 FAIL, 0 classes, 3 unclassified")
    assert f"  1  (NOT-REPRODUCED, {SHORT_B}:6, X Attack)  seeds {SHORT_B}:6" in text
    assert f"  1  (NOT-REPRODUCED, {SHORT_B}:7, X Attack)  seeds {SHORT_B}:7" in text
    assert "reason: the replay did not reproduce the failure" in text
    assert "(PASS, X Attack(57)" not in text.split("\n\n")[1]  # not listed as a class


def test_collect_reads_only_the_logs_of_its_summary(tmp_path):
    out = _sweep_out(tmp_path)
    (out / f"{SHORT_B}_99.log").write_text(  # an older sweep's log in the same folder
        _fail_log(NODE_B, 99, "Ether", 2, REPRO_1), encoding="utf-8"
    )
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert f"{SHORT_B}:99" not in text and "Ether" not in text


def test_collect_and_sweep_sh_use_extract_markers(tmp_path):
    marks = re.search(r"^MARKS='([^']*)'$", (POD / "sweep.sh").read_text(), re.MULTILINE)
    assert marks and marks.group(1) == "|".join(extract.MARKERS)
    out = _sweep_out(tmp_path)
    old = (out / f"{SHORT_B}_6.log").read_text().replace("Failing test case", "Falsifying example")
    (out / f"{SHORT_B}_6.log").write_text(old, encoding="utf-8")
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert f"== {SHORT_B}_6  node {SHORT_B}  seed 6" in text  # still a falsifier, not NO-FALSIFIER
    assert f"NO-FALSIFIER rc=1, {SHORT_B}" not in text


# --- a real failing log ------------------------------------------------------------------------

REAL_LOG = ROOT / "tests" / "fixtures" / "sweep" / f"{SHORT_B}_7.log.gz"
# A real falsifier log (Hypothesis 6.170.13 through `pytest -q`, the explore profile): seed 7 of
# the hand-run sweep at b83a1247, one bug, no exception group.


def test_extract_and_collect_read_a_real_hypothesis_log(tmp_path):
    import gzip
    import hashlib

    out = tmp_path / "sweep-20261122-b83a1247"
    out.mkdir()
    (out / f"{SHORT_B}_7.log").write_bytes(gzip.decompress(REAL_LOG.read_bytes()))
    ex = tmp_path / "ex"
    r = subprocess.run([sys.executable, str(ROOT / "tools/ops/sweep/extract.py"), str(out), str(ex)],
                       check=True, capture_output=True, text=True)  # fmt: skip
    assert f"{SHORT_B}_7 {SHORT_B} item='Persim Berry' choice=0 seed=1438 ok" in r.stdout
    d = ex / f"{SHORT_B}_7"
    assert sorted(f.name for f in d.iterdir()) == [
        "choice.txt", "item.txt", "node.txt", "outcome.txt", "reproduce.txt", "seed.txt", "state.txt",
    ]  # fmt: skip
    digest = {
        f: hashlib.sha256((d / f"{f}.txt").read_text().encode()).hexdigest()
        for f in ("state", "outcome")
    }
    assert digest == {  # the draws E2's extract.py cut from the same log
        "state": "dd104365c497bff24295e61e6c6e99e1d06cee7a3e7b4aa0b431a345f06241a3",
        "outcome": "b5df612e4d410f8fb43ea1d325cc42b728ed1de54c3375e0ad210c2ae99d5a21",
    }
    (out / "summary.txt").write_text(
        HEADER + f"{SHORT_B}\t7\tFAIL\t1215\t1\t12.1\n", encoding="utf-8"
    )
    _done(out, [(SHORT_B, 7)])
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert text.splitlines()[0] == (
        "SWEEP 20261122 b83a1247: 1 processes, 0 PASS, 1 FAIL, 0 classes, 1 unclassified"
    )
    block = text.split(f"== {SHORT_B}_7  node {SHORT_B}  seed 7\n", 1)[1].split("\n\n", 1)[0]
    assert "  @reproduce_failure('6.170.13', b'AXicXVRtaJVlGH7u9z0f2znb" in block
    assert "bag item: Persim Berry  choice: 0" in block
    assert f"(UNREPLAYED, {SHORT_B}, Persim Berry)" in block


# --- sweep.sh: paths, gate units, refusals, job limit ------------------------------------------


@linux
def test_sweep_takes_relative_jobs_and_out_paths_from_the_callers_folder(tmp_path):
    d = _run_folder(tmp_path)
    _jobs(tmp_path, [1, 2])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy(tmp_path / "jobs.txt", elsewhere / "rel_jobs.txt")
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0", SWEEP_P="2")
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), "rel_jobs.txt", "rel_out"], cwd=elsewhere,
                       check=False, capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    out = elsewhere / "rel_out"
    assert (out / "SWEEP_DONE").is_file()
    assert len((out / "summary.txt").read_text().splitlines()) == 3  # header + 2 jobs
    assert (tmp_path / "trace.txt").read_text().count("start") == 2
    assert collect.main([str(out), "--night", "20261122", "--tip", "abcdef01"]) == 0
    first = (out / "sweep_failures.txt").read_text().splitlines()[0]
    assert first == "SWEEP 20261122 abcdef01: 0 FAIL: none in 2 processes"


@linux
def test_sweep_jobs_are_gate_units_while_they_run(tmp_path):
    d = _run_folder(tmp_path)
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="4", SWEEP_P="1")
    p = subprocess.Popen([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    unit = d / "sweep" / "out" / "run" / f"{SHORT_A}_1"
    try:
        for _ in range(100):
            if unit.is_file() and (tmp_path / "trace.txt").exists():
                break
            time.sleep(0.05)
        g = unit.read_text().strip()
        assert (d / "sweep" / "out" / f"{SHORT_A}_1.w").read_text().strip() == "13"
        ps = subprocess.run(
            ["ps", "-ww", "-eo", "pgid=,args="], check=True, capture_output=True, text=True
        )  # -ww: a piped ps cuts args at 80 columns, before the seed of a long tmp path
        assert any(
            ln.split(None, 1)[0] == g and "--hypothesis-seed=1" in ln
            for ln in ps.stdout.splitlines()
        ), (g, ps.stdout)  # the pytest process is in the recorded group
        work = _posix(tmp_path / "work")
        gate = subprocess.run(
            [BASH, "-c", (f'. "{_posix(d / "bin/gate.sh")}"; NO_W_FILE_GB=12 podgate '
                         f'"{work}"/*/out/run/* "{work}"/*/*/out/run/*')],
            check=True, capture_output=True, text=True, env=env,
        )  # fmt: skip
        live, reserve = map(int, gate.stdout.split())
        assert live == 1 and reserve >= 12  # what every runner's gate now holds for it
        rc = p.wait(timeout=60)
        assert rc == 0, _driver_output(d, p)
    finally:
        p.kill()
    assert not unit.exists()  # the run file goes with the job


@linux
@pytest.mark.parametrize(
    ("knobs", "why"),
    [({"SWEEP_P": "7"}, "SWEEP_P 7 x SWEEP_W 13GB = 91GB is over the cap 88GB (limit 137GB)"),
     ({"SWEEP_P": "0"}, "SWEEP_P '0' is under 1"),
     ({"SWEEP_JOB_S": "1h"}, "SWEEP_JOB_S '1h' is not a whole number")],
)  # fmt: skip
def test_sweep_refuses_knobs_that_cannot_work(tmp_path, knobs, why):
    d = _run_folder(tmp_path)
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], check=False,
                       capture_output=True, text=True, env=_env(tmp_path, 10**9, **knobs), timeout=60)  # fmt: skip
    assert r.returncode == 1 and why in r.stderr, r.stderr
    assert not (tmp_path / "trace.txt").exists()


@linux
def test_sweep_refuses_an_out_dir_that_holds_a_sweep(tmp_path):
    d = _run_folder(tmp_path)
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0")
    args = [BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))]
    assert (
        subprocess.run(args, check=False, capture_output=True, env=env, timeout=60).returncode == 0
    )
    r = subprocess.run(args, check=False, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and "already holds a sweep (jobs.done)" in r.stderr
    assert (tmp_path / "trace.txt").read_text().count("start") == 1


@linux
def test_sweep_a_job_over_its_time_limit_fails_and_the_sweep_goes_on(tmp_path):
    d = _run_folder(tmp_path)
    hang = f"600.{os.getpid()}"  # this process's own duration: no other `sleep` of the pod has it
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0", FAKE_HANG_SEED="2", FAKE_HANG_S=hang,
               SWEEP_JOB_S="2", SWEEP_P="3")  # fmt: skip
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1, 2, 3]))], check=False,
                       capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    out = d / "sweep_out"
    rows = [ln.split("\t") for ln in (out / "summary.txt").read_text().splitlines()[1:]]
    assert [(x[1], x[2], x[4]) for x in rows] == [
        ("1", "PASS", "0"),
        ("2", "FAIL", "124"),
        ("3", "PASS", "0"),
    ]
    assert (out / "SWEEP_DONE").is_file()
    assert not list((d / "sweep" / "out" / "run").iterdir())
    m = re.search(rf"START {SHORT_A}_2 pgid=(\d+) ", (out / "driver.log").read_text())
    assert m, (out / "driver.log").read_text()
    # the hung job's leftovers were killed (under the pod's PID 1 they stay as zombies: dead)
    assert _live(m[1], f"sleep {hang}") == []
    assert collect.main([str(out), "--night", "20261122", "--tip", "abcdef01"]) == 0
    assert f"(NO-FALSIFIER rc=124, {SHORT_A}, -)" in (out / "sweep_failures.txt").read_text()


@linux
def test_a_zombie_of_a_hung_job_counts_as_dead():
    hang = f"601.{os.getpid()}"
    proc = subprocess.Popen(["sleep", hang], start_new_session=True)  # its own group, as a job
    g, args = str(proc.pid), f"sleep {hang}"
    grouplive = [BASH, "-c", f'. "{_posix(POD / "gate.sh")}"; grouplive {g}']  # reap's check
    try:
        for _ in range(100):  # the child's exec
            if _live("", args):
                break
            time.sleep(0.05)
        assert len(_live("", args)) == len(_live(g, "")) == 1  # both ways find the live one
        assert subprocess.run(grouplive, check=False).returncode == 0
        proc.kill()  # not waited for: a zombie, as under the pod's PID 1, which never reaps
        for _ in range(100):
            st = subprocess.run(
                ["ps", "-o", "stat=", "-p", g], check=False, capture_output=True, text=True
            ).stdout
            if st.startswith("Z"):
                break
            time.sleep(0.05)
        assert st.startswith("Z"), st
        assert _live(g, args) == []
        assert subprocess.run(grouplive, check=False).returncode == 1
    finally:
        proc.kill()
        proc.wait()


# --- the driver-alive checks: a zombie or dead driver is not a running sweep --------------------


def _start_sh(run_dir: Path) -> str:
    """sweep_body.sh's start.sh for the run folder <run_dir> (its R line pointed there)."""
    src = (ROOT / "tools/ops/day/sweep_body.sh").read_text(encoding="utf-8")
    body = src.split("cat <<'START'\n", 1)[1].split("\nSTART\n", 1)[0]
    return f"R='{_posix(run_dir)}'; export SWEEP_P=1 SWEEP_W=13\n{body}\n"


def _daybox_check(run_dir: Path) -> str:
    """The check exec of DAYBOX.md "Soundness sweep" step 4, for the run folder <run_dir>."""
    import json

    text = (ROOT / "tools/ops/day/DAYBOX.md").read_text(encoding="utf-8")
    m = re.search(r'Check: exec `(\{"command":\["bash","-c","cd /work/<run>; o=.*?\}`)', text)
    assert m, "the check exec is not in DAYBOX.md"
    cmd = json.loads(m.group(1)[:-1])["command"]
    return cmd[2].replace("/work/<run>", f"'{_posix(run_dir)}'")


@linux
@pytest.mark.parametrize("driver", ["dead", "zombie", "live"])
def test_start_and_the_check_see_a_dead_or_zombie_driver_as_gone(tmp_path, driver):
    d = _run_folder(tmp_path)
    if driver == "live":
        proc = subprocess.Popen(["sleep", "60"])
    else:
        proc = subprocess.Popen(["true"])
        if driver == "dead":
            proc.wait()
        else:  # exited and not waited for: a zombie, as under the pod's PID 1, which never reaps
            for _ in range(100):
                st = subprocess.run(["ps", "-o", "stat=", "-p", str(proc.pid)],
                                    check=False, capture_output=True, text=True).stdout.strip()  # fmt: skip
                if st.startswith("Z"):
                    break
                time.sleep(0.05)
            assert st.startswith("Z")
    try:
        (d / "sweep.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        (d / "sweep.outdir").write_text("sweep_out\n", encoding="utf-8")
        (d / "sweep_out").mkdir()  # an earlier sweep that ended without SWEEP_DONE
        check = subprocess.run(
            [BASH, "-c", _daybox_check(d)], check=False, capture_output=True, text=True, timeout=30
        )
        assert ("DRIVER_ALIVE" in check.stdout) == (driver == "live"), check.stdout
        send = tmp_path / "send"
        send.mkdir()
        (send / "start.sh").write_text(_start_sh(d), encoding="utf-8", newline="\n")
        (send / "sweep_jobs.txt").write_text(
            f"{NODE_A} {SHORT_A} 1\n", encoding="utf-8", newline="\n"
        )
        r = subprocess.run([BASH, "start.sh"], cwd=send, check=False, capture_output=True, text=True, timeout=30,
                           env=_env(tmp_path, 10 * 10**9, FAKE_JOB_S="0"))  # fmt: skip
        last = r.stdout.strip().splitlines()[-1]
        if driver == "live":
            assert last == f"SWEEP_REFUSED a sweep of this run is running (pid {proc.pid})", (
                r.stdout
            )
            return
        assert re.fullmatch(r"SWEEP_STARTED sweep-x out=sweep_out_[0-9]+ jobs=1", last), (
            r.stdout + r.stderr
        )
        o = d / (d / "sweep.outdir").read_text().strip()
        for _ in range(200):
            if (o / "SWEEP_DONE").exists():
                break
            time.sleep(0.1)
        assert (o / "SWEEP_DONE").is_file(), (d / "sweep.log").read_text()
    finally:
        proc.kill()
        proc.wait()


@linux
def test_sweep_term_during_the_replay_stops_it_at_once(tmp_path):
    d = _run_folder(tmp_path)
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0", FAKE_FAIL_SEED="1", FAKE_REPLAY_S="60")
    p = subprocess.Popen([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # fmt: skip
    trace = tmp_path / "trace.txt"
    try:
        for _ in range(200):
            if trace.exists() and "replay " in trace.read_text():
                break
            time.sleep(0.05)
        assert (d / "sweep" / "out" / "run" / "replay").is_file()  # a unit while it runs
        t0 = time.time()
        p.terminate()
        assert p.wait(timeout=30) == 143, _driver_output(d, p)
        assert time.time() - t0 < 15  # not held until the replay ends (60 s)
    finally:
        p.kill()
    rpid = trace.read_text().split("replay ", 1)[1].split()[0]
    st = subprocess.run(
        ["ps", "-o", "stat=", "-p", rpid], check=False, capture_output=True, text=True
    ).stdout.strip()
    assert st == "" or st.startswith("Z")  # the replay's group was stopped too
    assert (d / "sweep_out" / "SWEEP_STOPPED").is_file()
    assert not (d / "sweep_out" / "SWEEP_DONE").exists()


@linux
def test_podcheck_lists_a_live_sweep_run_with_its_units(tmp_path):
    root = tmp_path / "work"
    d = root / "sweep-x"
    (d / "sweep" / "out" / "run").mkdir(parents=True)
    driver = subprocess.Popen(["sleep", "60"])
    job = subprocess.Popen(["setsid", "sleep", "60"])
    try:
        (d / "sweep.pid").write_text(f"{driver.pid}\n", encoding="utf-8")
        (d / "sweep" / "out" / "run" / "j1").write_text(f"{job.pid}\n", encoding="utf-8")
        (d / "sweep" / "out" / "j1.w").write_text("13\n", encoding="utf-8")
        r = subprocess.run([BASH, _posix(POD / "podcheck.sh"), _posix(root)], check=False, capture_output=True, text=True,
                           timeout=60, env={**_clean_env(), "CGROOT": _posix(tmp_path / "nocg")})  # fmt: skip
        assert "run sweep-x gate=old scripts=old runner=sweep live=1 pending=0" in r.stdout, (
            r.stdout + r.stderr
        )
    finally:
        driver.kill()
        job.kill()
        driver.wait()
        job.wait()


# --- quick-lane-3: fixed seeds, meta.env, the replay tools, run.env knobs, a dead driver ---------


def test_fixed_seeds_give_every_node_the_same_seeds_whatever_the_night(capsys):
    assert seeds.parse_seeds("1..3") == [1, 2, 3]
    assert seeds.parse_seeds("7") == [7]
    assert seeds.parse_seeds("1,4,9") == [1, 4, 9]
    assert seeds.parse_seeds("1..2,9") == [1, 2, 9]
    assert len(seeds.parse_seeds(f"1..{seeds.MAX_SEEDS}")) == seeds.MAX_SEEDS
    for bad in ("", "a", "3..1", "1,1", "1..2,2", "-1", "4294967296", "1..", "1,,2",
                "1..99999999999999999999", "1..4294967295", "1..1001", "1..600,601..1001"):
        # refused before the range is built: a seed of 2**32 or more, more than MAX_SEEDS seeds
        with pytest.raises(ValueError):
            seeds.parse_seeds(bad)
    assert seeds.main(["--seeds", "1..15"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{n} {s} {i}" for n, s in seeds.nodes() for i in range(1, 16)]
    # the printed lines are what sweep_body.sh accepts: '<node id> <short name> <seed>'
    assert all(re.fullmatch(r"[^ ]+ [^ ]+ [0-9]+", ln) for ln in lines)
    # no night and no --n beside --seeds; a bad list is refused (rc 2, nothing on stdout)
    for argv in (["20261122", "--seeds", "1"], ["--seeds", "1", "--n", "3"], ["--seeds", "x"]):
        assert seeds.main(argv) == 2
    assert capsys.readouterr().out == ""


def _meta_run(tmp_path: Path, meta: str, run_name: str = "run-without-a-date") -> Path:
    run = tmp_path / run_name
    run.mkdir()
    (run / "meta.env").write_text(meta, encoding="utf-8", newline="\n")
    return _sweep_out(run, "sweep_out")


def _first(out: Path) -> str:
    assert collect.main([str(out)]) == 0
    return (out / "sweep_failures.txt").read_text(encoding="utf-8").splitlines()[0]


def test_collect_reads_night_and_tip_from_the_runs_meta_env(tmp_path):
    full = "0123456789abcdef0123456789abcdef01234567"
    out = _meta_run(tmp_path, f'COMMIT={full}\nMARK="not slow"\nNIGHT=20261123\nWATCH_F=secret\n')
    assert _first(out).startswith("SWEEP 20261123 01234567: 6 processes")
    assert "secret" not in (out / "sweep_failures.txt").read_text(encoding="utf-8")
    assert collect.read_meta(out) == {"COMMIT": full, "MARK": "not slow", "NIGHT": "20261123"}


def test_collect_night_from_the_run_name_tip_from_meta_and_the_options_beat_both(tmp_path):
    # the name's date is the jobs file's night: it wins over NIGHT (a pack that slipped past midnight)
    out = _meta_run(tmp_path, "COMMIT=aaaaaaaa11111111\nNIGHT=20261123\n", "sweep-20261101-bbbbbbbb")
    assert _first(out).startswith("SWEEP 20261101 aaaaaaaa: ")
    assert collect.main([str(out), "--night", "20261124", "--tip", "cccccccc99"]) == 0
    head = (out / "sweep_failures.txt").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("SWEEP 20261124 cccccccc: ")


def test_collect_takes_a_value_meta_lacks_from_the_name_and_ignores_a_bad_one(tmp_path):
    # a run packed before NIGHT: the night comes from the name, the tip from COMMIT
    out = _meta_run(tmp_path, "COMMIT=aaaaaaaa11111111\n", "sweep-20261101-bbbbbbbb")
    assert _first(out).startswith("SWEEP 20261101 aaaaaaaa: ")
    # a malformed NIGHT and COMMIT are not used
    other = tmp_path / "other"
    other.mkdir()
    out = _meta_run(other, "COMMIT=zz\nNIGHT=tomorrow\n")
    assert _first(out).startswith("SWEEP ? ?: ")
    # meta.env inside the out dir counts too (its COMMIT; the out dir's name sweep-20261122-... has the night)
    third = tmp_path / "third"
    third.mkdir()
    out3 = _clean_out(third)
    (out3 / "meta.env").write_text("COMMIT=feedface0000\nNIGHT=20260101\n", encoding="utf-8")
    assert _first(out3).startswith("SWEEP 20261122 feedface: ")


def test_logfields_zero_fills_a_state_field_an_old_log_lacks():
    import collections

    from tools.ops.sweep.logfields import keep

    State = collections.namedtuple("State", ["hp", "sketch_move", "gone"])
    said: list[tuple] = []
    zero = {"sketch_move": 0, "hp": -1}  # 'gone' has no layout entry

    def zero_of(name):
        return zero[name]  # KeyError for a field the layout does not know

    make = keep(State, zero_of, lambda *a: said.append(a))
    # an old log without sketch_move, and with a field this repo dropped
    s = make(hp=5, gone=7, retired=9)
    assert s == State(hp=5, sketch_move=0, gone=7)
    assert any("dropped" in a[0] and a[1] == ["retired"] for a in said)
    assert any("zero-filled" in a[0] and a[1] == ["sketch_move"] for a in said)
    # a missing field the layout cannot fill is still an error, and named
    with pytest.raises(ValueError, match=r"State fields missing from the log: \['gone'\]"):
        make(hp=5)
    # without a zero maker (Outcome) a missing field is an error as before
    with pytest.raises(ValueError, match=r"fields missing from the log: \['sketch_move'\]"):
        keep(State)(hp=1, gone=2)
    # replay.py wires it: BattleState is zero-filled from LAYOUT, Outcome is not
    text = (ROOT / "tools/ops/sweep/replay.py").read_text(encoding="utf-8")
    assert '"BattleState": keep(BattleState, zero_state_field)' in text
    assert '"Outcome": keep(Outcome),' in text and "LAYOUT[name]" in text
    assert text.index("from logfields import keep") < text.index("sys.path[:0]")


def test_sweep_tools_ship_with_the_pod_scripts(tmp_path):
    rent = tmp_path / "rent"
    rent.mkdir()
    dest = tmp_path / "bin"
    script = (
        f"set -euo pipefail; HERE='{_posix(ROOT / 'tools/ops/day')}'; . \"$HERE/pack_lib.sh\"; "
        f"pack_sweep_tools '{_posix(ROOT)}' HEAD '{_posix(dest)}'"
    )
    r = subprocess.run([BASH, "-c", script], check=False, capture_output=True, text=True,
                       env={**_clean_env(), "RENT": _posix(rent)})  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    got = {p.name for p in (dest / "sweep").iterdir()}
    assert {"extract.py", "replay.py", "collect.py", "seeds.py", "nodes.txt"} <= got
    assert not list(rent.iterdir())  # its temp folder is gone
    assert b"\r" not in (dest / "sweep" / "replay.py").read_bytes()  # LF, as on the pod
    # pack.sh calls it after the pod scripts, and sweep.sh looks in bin/sweep
    pack = (ROOT / "tools/ops/day/pack.sh").read_text(encoding="utf-8")
    assert pack.index("pod_bin ") < pack.index('pack_sweep_tools "$TOP" "$OPS" "$P/bin"')
    assert '"$D/bin/sweep"' in (POD / "sweep.sh").read_text(encoding="utf-8")


@linux
def test_sweep_of_a_commit_without_sweep_tools_uses_the_copy_in_bin(tmp_path):
    d = _run_folder(tmp_path)
    (d / "bin" / "sweep").mkdir()
    for n in ("extract.py", "replay.py"):  # the fake python answers by the script's name
        (d / "bin" / "sweep" / n).write_text("", encoding="utf-8")
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0", FAKE_FAIL_SEED="2")
    env.pop("SWEEP_TOOLS")  # the default: <repo>/tools/ops/sweep, which this repo lacks
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1, 2]))], check=False,
                       capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    out = d / "sweep_out"
    assert (out / "replay_in" / "extract.ok").is_file()
    assert "== replayed" in (out / "replay.txt").read_text(encoding="utf-8")


@linux
def test_sweep_uses_the_repos_own_tools_when_it_has_both(tmp_path):
    d = _run_folder(tmp_path)
    for n in ("extract.py", "replay.py"):
        (d / "repo" / "tools" / "ops" / "sweep").mkdir(parents=True, exist_ok=True)
        (d / "repo" / "tools" / "ops" / "sweep" / n).write_text("", encoding="utf-8")
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0")
    env.pop("SWEEP_TOOLS")  # no bin/sweep exists: only the repo's copy can satisfy the check
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], check=False,
                       capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert (d / "sweep_out" / "SWEEP_DONE").is_file()


@linux
def test_sweep_refuses_at_the_start_when_neither_tool_copy_exists(tmp_path):
    d = _run_folder(tmp_path)
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0")
    env.pop("SWEEP_TOOLS")
    r = subprocess.run([BASH, _posix(d / "bin/sweep.sh"), _posix(_jobs(tmp_path, [1]))], check=False,
                       capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    assert r.returncode == 1 and "no extract.py and replay.py in the repo's tools/ops/sweep" in r.stderr
    assert not (d / "sweep_out" / "SWEEP_DONE").exists()


@linux
def test_start_reads_the_sweep_knobs_the_pc_did_not_set_from_run_env(tmp_path):
    d = _run_folder(tmp_path)
    (d / "run.env").write_text("export SWEEP_NICE='5'\nexport SWEEP_JOB_S='1234'\n", encoding="utf-8", newline="\n")
    send = tmp_path / "send"
    send.mkdir()
    (send / "start.sh").write_text(_start_sh(d), encoding="utf-8", newline="\n")
    (send / "sweep_jobs.txt").write_text(f"{NODE_A} {SHORT_A} 1\n", encoding="utf-8", newline="\n")
    # SWEEP_JOB_S set where start.sh runs (the PC's value, baked in) beats run.env
    env = _env(tmp_path, 10 * 10**9, FAKE_JOB_S="0", SWEEP_JOB_S="99")
    r = subprocess.run([BASH, "start.sh"], cwd=send, check=False, capture_output=True, text=True,
                       timeout=30, env=env)  # fmt: skip
    assert r.stdout.strip().splitlines()[-1].startswith("SWEEP_STARTED"), r.stdout + r.stderr
    out = d / (d / "sweep.outdir").read_text().strip()
    for _ in range(200):
        if (out / "SWEEP_DONE").exists():
            break
        time.sleep(0.1)
    assert (out / "SWEEP_DONE").is_file(), (d / "sweep.log").read_text()
    start = next(ln for ln in (out / "driver.log").read_text().splitlines() if "DRIVER_START" in ln)
    assert "nice=5 job_s=99" in start, start  # NICE from run.env, JOB_S from the environment


def test_sweep_body_bakes_the_pcs_knobs_and_refuses_a_bad_one(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text(f"{NODE_A} {SHORT_A} 1\n", encoding="utf-8", newline="\n")
    body = (ROOT / "tools/ops/day/sweep_body.sh").as_posix()

    def start(n: int, **knobs: str) -> tuple[subprocess.CompletedProcess, str]:
        sub = tmp_path / f"s{n}"
        sub.mkdir()
        r, _, send = _tool_body(sub, body, "sweep-20261122-abc", _posix(jobs), env=knobs)
        if r.returncode != 0:
            return r, ""
        tar = ["tar", "-xOf", _posix(send / "sweep.tar"), "start.sh"]
        return r, subprocess.run(tar, check=True, capture_output=True, text=True).stdout

    r, text = start(1, SWEEP_NICE="3", SWEEP_POLL="0.5", SWEEP_REPLAY_S="60")
    assert r.returncode == 0, r.stdout + r.stderr
    head = text.split("for n in", 1)[0]  # what the PC baked in; the rest reads run.env
    assert "export SWEEP_NICE=3\n" in head and "export SWEEP_POLL=0.5\n" in head
    assert "export SWEEP_REPLAY_S=60\n" in head
    assert "export SWEEP_P=" not in head and "export SWEEP_CAP" not in head
    assert "export SWEEP_P=${SWEEP_P:-5} SWEEP_W=${SWEEP_W:-13}" in text and "run.env" in text
    r, _ = start(2, SWEEP_NICE="x")
    assert r.returncode == 3 and "SWEEP_NICE 'x' is not a whole number" in r.stdout
    r, _ = start(3, SWEEP_SPACING="1.2.3")
    assert r.returncode == 3 and "SWEEP_SPACING '1.2.3' is not a number" in r.stdout


@linux
def test_podcheck_lists_a_sweep_whose_driver_died_while_its_job_group_lives(tmp_path):
    root = tmp_path / "work"
    d = root / "sweep-x"
    (d / "sweep" / "out" / "run").mkdir(parents=True)
    driver = subprocess.Popen(["true"])
    driver.wait()  # the driver is gone; sweep.pid names a dead process
    # a job as sweep.sh starts it: its own group, the run's repo among its arguments
    job = subprocess.Popen(["setsid", "bash", "-c", "sleep 60; true", "j1", _posix(d / "repo")])
    want = "run sweep-x gate=old scripts=old runner=sweep live=1 pending=0"

    def check() -> str:
        return subprocess.run([BASH, _posix(POD / "podcheck.sh"), _posix(root)], check=False, capture_output=True,
                              text=True, timeout=60, env={**_clean_env(), "CGROOT": _posix(tmp_path / "nocg")}).stdout  # fmt: skip

    try:
        (d / "sweep.pid").write_text(f"{driver.pid}\n", encoding="utf-8")
        # the unit's run file holds the group id; setsid made the job its own group leader
        (d / "sweep" / "out" / "run" / "j1").write_text(f"{job.pid}\n", encoding="utf-8")
        (d / "sweep" / "out" / "j1.w").write_text("13\n", encoding="utf-8")
        assert want in check()
        # no sweep.pid at all: the live group alone still lists the run
        (d / "sweep.pid").unlink()
        assert want in check()
        # once the group is gone the run is not listed
        job.kill()
        job.wait()
        for _ in range(50):
            if "runner=sweep" not in check():
                break
            time.sleep(0.1)
        assert "runner=sweep" not in check()
    finally:
        job.kill()
        job.wait()


@linux
def test_sweeplive_in_gate_sh_counts_a_live_group_and_not_a_dead_or_malformed_one(tmp_path):
    run = tmp_path / "r"
    (run / "sweep" / "out" / "run").mkdir(parents=True)
    f = run / "sweep" / "out" / "run" / "j1"
    job = subprocess.Popen(["setsid", "bash", "-c", "sleep 60; true", "j1", _posix(run / "repo")])
    # a live group that is not this run's (its id reused after the job ended): no run path in its args
    other = subprocess.Popen(["setsid", "bash", "-c", "sleep 60; true", "x", _posix(tmp_path / "r2" / "repo")])
    gate = _posix(POD / "gate.sh")

    def live(content: str) -> bool:
        f.write_text(content, encoding="utf-8")
        cmd = f". '{gate}'; sweeplive '{_posix(run)}'"
        return subprocess.run([BASH, "-c", cmd], check=False).returncode == 0

    try:
        assert live(f"{job.pid}\n")
        for bad in ("", "abc\n", "0\n", "1\n"):
            assert not live(bad)
        assert not live(f"{other.pid}\n")  # r2/ is not r/: another run's group does not count
        job.kill()
        job.wait()
        assert not live(f"{job.pid}\n")
    finally:
        for p in (job, other):
            p.kill()
            p.wait()
    # status.sh and podcheck.sh both call it
    assert "sweeplive" in (POD / "status.sh").read_text(encoding="utf-8")
    assert "sweeplive" in (POD / "podcheck.sh").read_text(encoding="utf-8")


def test_daybox_names_every_runner_value_podcheck_prints():
    text = (ROOT / "tools/ops/day/DAYBOX.md").read_text(encoding="utf-8")
    assert "runner=alive|setup|quick|sweep live=" in text
    pod = (POD / "podcheck.sh").read_text(encoding="utf-8")
    assert "runner=<alive|setup|quick|sweep>" in pod
