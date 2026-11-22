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


# --- collect -----------------------------------------------------------------------------------


def test_collect_writes_blocks_and_classes_from_the_replay(tmp_path):
    out = _sweep_out(tmp_path)
    (out / "replay.txt").write_text(FAKE_REPLAY, encoding="utf-8")
    assert collect.main([str(out)]) == 0
    text = (out / "sweep_failures.txt").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first == "SWEEP 20261122 abcdef01: 6 processes, 3 PASS, 3 FAIL, 2 classes"
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


def test_collect_clean_sweep_first_line_and_exit_zero(tmp_path):
    out = tmp_path / "x-20261201-0123456789abcdef"
    out.mkdir()
    (out / "summary.txt").write_text(
        "node\tseed\tresult\twall_s\trc\tpeak_GB\n" + f"{SHORT_A}\t1\tPASS\t10\t0\t1.0\n" * 3,
        encoding="utf-8",
    )
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
    assert collect.main([str(out), "--night", "20261122", "--tip", "deadbeefcafe"]) == 0
    text = (out / "sweep_failures.txt").read_text()
    assert text.splitlines()[0] == "SWEEP 20261122 deadbeef: 2 processes, 0 PASS, 2 FAIL, 2 classes"
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
    assert "R=/work/sweep-20261122-abc" in start and "setsid nohup bash bin/sweep.sh" in start
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


def _env(tmp_path: Path, mem_bytes: int, **kw: str) -> dict[str, str]:
    cg = tmp_path / "cg"
    cg.mkdir(exist_ok=True)
    (cg / "memory.current").write_text(f"{mem_bytes}\n", encoding="utf-8", newline="\n")
    return {**_clean_env(), "CGROOT": _posix(cg), "FAKE_TRACE": _posix(tmp_path / "trace.txt"), "SWEEP_SPACING": "0",
            "SWEEP_POLL": "0.2", "SWEEP_TOOLS": _posix(tmp_path / "tools"), **kw}  # fmt: skip


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
        f"-p no:cacheprovider -q -x {NODE_A} --hypothesis-seed=" in ln and ln.endswith("-m")
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
        (tmp_path / "cg" / "memory.current").write_text(
            f"{177 * 10**9}\n", encoding="utf-8", newline="\n"
        )  # 177 + 13 = 190
        assert p.wait(timeout=60) == 0
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


def test_sweep_scripts_parse_and_ship_in_the_pod_bin():
    for f in (POD / "sweep.sh", ROOT / "tools/ops/day/sweep_body.sh"):
        assert subprocess.run([BASH, "-n", _posix(f)], check=False).returncode == 0
    assert (POD / "sweep.sh") in set(
        POD.glob("*.sh")
    )  # pod_bin (common.sh) packs every *.sh of pod/
    assert re.search(r"\*\.sh", (ROOT / "tools/ops/common.sh").read_text(encoding="utf-8"))
