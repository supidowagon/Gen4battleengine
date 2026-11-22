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
    assert "SWEEP_CAP" not in start  # unset: sweep.sh takes the cap from the pod's limit
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
  *replay.py) echo "== replayed $3"; exit 0;;
esac
for a in "$@"; do case "$a" in --hypothesis-seed=*) seed=${a#*=};; esac; done
T="$FAKE_TRACE"
echo "start $seed $(date +%s.%N) $NUZLINE_HYPOTHESIS_PROFILE $*" >> "$T"
if [ "$seed" = "${FAKE_HANG_SEED:-}" ]; then sleep 600; fi
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
    left = subprocess.run(["ps", "-eo", "args="], check=True, capture_output=True, text=True).stdout
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
            ["ps", "-eo", "pgid=,args="], check=True, capture_output=True, text=True
        )
        assert any(
            ln.split(None, 1)[0] == g and "--hypothesis-seed=1" in ln
            for ln in ps.stdout.splitlines()
        )  # the pytest process is in the recorded group
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
    env = _env(
        tmp_path, 10 * 10**9, FAKE_JOB_S="0", FAKE_HANG_SEED="2", SWEEP_JOB_S="2", SWEEP_P="3"
    )
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
    left = subprocess.run(["ps", "-eo", "args="], check=True, capture_output=True, text=True).stdout
    assert "sleep 600" not in left  # the hung job's leftovers were killed
    assert collect.main([str(out), "--night", "20261122", "--tip", "abcdef01"]) == 0
    assert f"(NO-FALSIFIER rc=124, {SHORT_A}, -)" in (out / "sweep_failures.txt").read_text()
