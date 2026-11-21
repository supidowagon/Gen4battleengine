"""tools/ops/day/pod/gate.sh: runner.sh's start-gate arithmetic, run in bash with a fake ps."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tools" / "ops" / "day" / "pod" / "gate.sh"


def _bash() -> str | None:
    """A POSIX bash with awk: on Windows Git's bash (System32's bash.exe is WSL, which cannot see
    these paths), elsewhere the bash on PATH."""
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    git = shutil.which("git")
    if git:
        exec_path = subprocess.run([git, "--exec-path"], check=False, capture_output=True, text=True).stdout.strip()
        for b in (Path(exec_path).parents[2] / "bin" / "bash.exe", Path(git).parents[1] / "bin" / "bash.exe"):
            if b.is_file():
                return str(b)
    return None


BASH = _bash()


def _runner_knobs() -> set[str]:
    """setup.sh's RUN_ENV_NAMES (the knobs a pod run's exec-pod body may set, which every process of
    the run, this test included, inherits), plus the other variables the pod scripts read."""
    text = (GATE.parent / "setup.sh").read_text(encoding="utf-8")
    line = next(x for x in text.splitlines() if x.startswith("RUN_ENV_NAMES="))
    return set(line.split("=", 1)[1].strip('"').split()) | {
        "MARK", "NO_W_FILE_GB", "NOW", "CLAIM_PID", "WATCH_F", "WATCH_D", "PACK_OFF_MAIN_POD", "NUZLINE_REPO", "NUZLINE_SCRATCH",
        "GATE_VERSION", "POD_SCRIPTS", "POD_WEIGHTS", "RUNPODCTL"}


def _clean_env() -> dict[str, str]:
    """os.environ without any runner knob: on the pod the tests run inside a run whose knobs
    (KILL_GB, GATE_GB, MAXP, SPACING, ...) would otherwise reach the scripts under test."""
    knobs = _runner_knobs()
    return {k: v for k, v in os.environ.items() if k not in knobs}


pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _run(script: str, env: str = "NO_W_FILE_GB=12 MAXP=10 GATE_GB=88 KILL_GB=118 IDLE_WAIT_S=300") -> str:
    full = f"set -u; {env}; . '{GATE.as_posix()}'\n{script}"
    r = subprocess.run([BASH, "-c", full], capture_output=True, text=True, check=True, env=_clean_env())
    return r.stdout.strip()


def _unit(run_dir: Path, u: str, pgid: str, weight: str | None) -> str:
    (run_dir / "run").mkdir(parents=True, exist_ok=True)
    (run_dir / "run" / u).write_text(pgid + "\n", encoding="utf-8", newline="\n")
    if weight is not None:
        (run_dir / f"{u}.w").write_text(weight + "\n", encoding="utf-8", newline="\n")
    return (run_dir / "run" / u).as_posix()


# ps -eo pid=,pgid=,stat=,rss=  (rss in KiB); 29296875 KiB = 30.0 GB
PS = ["101 101 S 20000000", "1101 101 S 9296875", "102 102 R 19531250", "103 103 S 9765625",
      "201 201 Z 0", "203 203 S 0", "204 204 S 0", "1 1 S 100", "7 0 S 100"]


def test_podgate_counts_live_groups_and_reserves_unreached_peaks(tmp_path):
    a, b = tmp_path / "r1" / "out", tmp_path / "r2" / "sub" / "out"
    files = [
        _unit(a, "u_001", "101", "90"),    # 30 GB used of 90: reserve 60
        _unit(a, "u_002", "102", None),    # no .w: NO_W_FILE_GB 12, 20 GB used: reserve 0
        _unit(a, "u_003", "103", "x7"),    # malformed .w: NO_W_FILE_GB 12, 10 GB used: reserve 2
        _unit(b, "u_001", "201", "40"),    # only a zombie: not live
        _unit(b, "u_002", "1", "50"),      # pgid 1 (PID 1): never counted
        _unit(b, "u_003", "0", "50"),      # pgid 0: never counted
        _unit(b, "u_004", "abc", "50"),    # malformed pgid
        _unit(b, "u_005", "203", "25"),    # just started: reserve 25
        _unit(b, "u_006", "999", "25"),    # group gone
    ]
    fake = "ps() { printf '%s\\n' " + " ".join(f"'{p}'" for p in PS) + "; }"
    args = " ".join(f"'{f}'" for f in files + [(tmp_path / "none" / "run" / "u_9").as_posix()])
    assert _run(f"{fake}\npodgate {args}") == "4 87"


def test_podgate_rounds_a_fractional_reserve_up_and_handles_no_units(tmp_path):
    f = _unit(tmp_path / "out", "u_001", "204", "10")
    fake = "ps() { echo '204 204 S 9277343'; }"  # 9.4999 GB used of 10: 0.5 GB rounds up to 1
    assert _run(f"{fake}\npodgate '{f}'") == "1 1"
    assert _run(f"{fake}\npodgate") == "0 0"


def test_wgb_falls_back_to_the_default(tmp_path):
    (tmp_path / "a.w").write_text("33\n", encoding="utf-8", newline="\n")
    (tmp_path / "b.w").write_text("3.5\n", encoding="utf-8", newline="\n")
    got = _run(" ".join(f"wgb '{(tmp_path / n).as_posix()}';" for n in ("a.w", "b.w", "c.w")))
    assert got.split() == ["33", "12", "12"]


@pytest.mark.parametrize(("args", "want"), [
    ("10 0 0 5 0", "hold-maxp"),
    ("3 40 30 17 0", "start"),          # 30 + 40 + 17 = 87 < 88
    ("3 40 30 18 0", "hold"),           # 88: not under GATE_GB
    ("0 0 10 100 0", "start"),          # idle, 110 < KILL_GB 118: heavier than the gate, alone
    ("0 0 20 100 0", "wait-idle"),      # idle, 120 >= KILL_GB, could fit once memory falls
    ("0 0 20 100 299", "wait-idle"),
    ("0 0 20 100 300", "start-idle-wait"),
    ("0 0 5 118 0", "start-over-kill"),  # can never fit: alone, at once
])
def test_admit(args, want):
    assert _run(f"admit {args}") == want


# Skip-ahead and pod-wide claims (choose, claim_state, claim_write). A fake ps answers both
# formats the gate uses: process groups (pgid,stat,rss) and runner pids (pid,stat).
def _fake_ps(pids: dict[int, str]) -> str:
    rows = " ".join(f"'{p} {s}'" for p, s in pids.items())
    return ("ps() { case \"$*\" in *pgid=*) :;; *pid=,stat=*) printf '%s\n' " + rows
            + ";; esac; }")


def _run_dir(root: Path, name: str, units: list[tuple[str, int]]) -> Path:
    d = root / name / "out"
    d.mkdir(parents=True)
    (d / "pending.txt").write_text("".join(f"{u} tests/{u}.py\n" for u, _ in units), encoding="utf-8", newline="\n")
    for u, gb in units:
        (d / f"{u}.w").write_text(f"{gb}\n", encoding="utf-8", newline="\n")
    return d


def _in(d: Path, claims: Path, script: str, pids: dict[int, str] | None = None) -> str:
    pids = pids or {}
    env = (f"NO_W_FILE_GB=12 MAXP=20 GATE_GB=176 KILL_GB=236 IDLE_WAIT_S=300 CLAIM_S=120 NOW=2000 "
           f"CLAIMS='{claims.as_posix()}'")
    return _run(f"{_fake_ps(pids)}\ncd '{d.as_posix()}'\n{script}", env)


def test_a_heavy_head_lets_a_later_unit_that_fits_start(tmp_path):
    d = _run_dir(tmp_path, "r1", [("u_001", 30), ("u_002", 30), ("u_003", 9)])
    # 256 GB pod: 137 + 10 + 30 >= 176 holds the head; 137 + 10 + 9 < 176 lets u_003 start
    assert _in(d, tmp_path / "c", "choose 18 10 137 0 0 none") == "start u_003 skip-ahead -"
    assert _in(d, tmp_path / "c", "choose 18 30 137 0 0 none") == "none - hold -"   # nothing fits
    assert _in(d, tmp_path / "c", "choose 18 0 100 0 0 none") == "start u_001 start -"  # the head fits


def test_the_head_is_claimed_after_claim_s_and_skip_ahead_stops(tmp_path):
    d = _run_dir(tmp_path, "r1", [("u_001", 30), ("u_002", 9)])
    assert _in(d, tmp_path / "c", "choose 18 10 137 0 119 none") == "start u_002 skip-ahead -"
    assert _in(d, tmp_path / "c", "choose 18 10 137 0 120 none") == "none - hold claim"
    assert _in(d, tmp_path / "c", "choose 18 10 137 0 500 mine") == "none - hold -"
    assert _in(d, tmp_path / "c", "choose 18 0 100 0 500 mine") == "start u_001 start -"
    assert _in(d, tmp_path / "c", "choose 1 0 0 0 0 other 176") == "none - claim-other -"


def test_claim_hand_over_between_two_runners(tmp_path):
    claims = tmp_path / "claims"
    a = _run_dir(tmp_path, "run-a", [("u_001", 30)])
    b = _run_dir(tmp_path, "run-b", [("u_001", 9), ("u_002", 9)])
    live = {4001: "S", 4002: "S"}
    _in(a, claims, "CLAIM_PID=4001 claim_write run-a u_001 30 1000", live)
    assert (claims / "run-a.u_001").read_text(encoding="utf-8").split() == ["1000", "30", "4001"]
    assert _in(a, claims, "claim_state run-a", live) == "mine u_001 30"
    assert _in(b, claims, "claim_state run-b", live) == "other run-a.u_001 30"
    # b's own later claim does not outrank the older one
    _in(b, claims, "CLAIM_PID=4002 claim_write run-b u_001 9 1005", live)
    assert _in(b, claims, "claim_state run-b", live) == "other run-a.u_001 30"
    # a starts its unit and drops the claim: b's claim is now the oldest
    _in(a, claims, "claim_drop run-a u_001", live)
    assert _in(b, claims, "claim_state run-b", live) == "mine u_001 9"
    assert _in(a, claims, "claim_state run-a", live) == "other run-b.u_001 9"


def test_a_dead_or_zombie_claimant_is_ignored_and_deleted(tmp_path):
    claims = tmp_path / "claims"
    b = _run_dir(tmp_path, "run-b", [("u_001", 9)])
    _in(b, claims, "CLAIM_PID=5001 claim_write run-a u_001 30 1000; CLAIM_PID=5002 claim_write run-c u_004 30 1001")
    (claims / "run-d.u_002").write_text("garbage\n", encoding="utf-8", newline="\n")
    got = _in(b, claims, "claim_state run-b 2> err.txt; cat err.txt", {5002: "Z"}).splitlines()
    assert got == ["none", "2000 claim-dead run-a.u_001 pid 5001 age 1000 s", "2000 claim-dead run-c.u_004 pid 5002 age 999 s"]
    assert sorted(p.name for p in claims.iterdir()) == []


def test_podgate_counts_a_unit_that_has_not_called_setsid_yet(tmp_path):
    # pid 301 is alive but still in the runner's group 50: live, nothing of group 301 yet
    f = _unit(tmp_path / "out", "u_001", "301", "30")
    fake = "ps() { printf '%s\\n' '50 50 S 1000' '301 50 S 2000'; }"
    assert _run(f"{fake}\npodgate '{f}'") == "1 30"
    fake_z = "ps() { printf '%s\\n' '50 50 S 1000' '301 50 Z 0'; }"
    assert _run(f"{fake_z}\npodgate '{f}'") == "0 0"


GB_KIB = 1e9 / 1024   # ps rss is KiB; the gate's GB is 10^9 bytes


def _units(root: Path, specs: list[tuple[int, float]]) -> tuple[str, str]:
    """Live units of weight GB at RSS GB, pgids 1001...: (podgate's file args, a fake ps)."""
    files, rows = [], []
    for i, (w, rss) in enumerate(specs):
        files.append(_unit(root / "out", f"u_{i:03d}", str(1001 + i), str(w)))
        rows.append(f"'{1001 + i} {1001 + i} S {int(rss * GB_KIB)}'")
    return " ".join(f"'{f}'" for f in files), "ps() { printf '%s\n' " + " ".join(rows) + "; }"


def test_podgate_reserves_every_heavy_gap_and_only_the_largest_light_gaps(tmp_path):
    # heavy (w >= 40): gaps 72, 50, 0 (over its weight); light: gaps 24, 20, 14, 5, 0
    args, fake = _units(tmp_path, [(82, 10), (60, 10), (45, 50), (30, 6), (26, 6), (20, 6), (11, 6), (5, 6)])
    assert _run(f"{fake}\npodgate {args}") == "8 166"                          # 72 + 50 + 24 + 20
    assert _run(f"{fake}\nRESERVE_TOP=3; podgate {args}") == "8 180"           # + 14
    assert _run(f"{fake}\nRESERVE_TOP=0; podgate {args}") == "8 122"           # heavy gaps only
    assert _run(f"{fake}\nRESERVE_FULL_GB=0; podgate {args}") == "8 185"       # every gap: the old sum
    assert _run(f"{fake}\nRESERVE_FULL_GB=999; podgate {args}") == "8 122"     # top 2 of all: 72 + 50


def test_17_live_units_of_379_gb_weight_admit_the_next_unit(tmp_path):
    # Pod 3, 21 Nov: 17 live units, weight sum 379 GB, ran without a kill. Early in their lives
    # (6 GB each, 10 GB of other memory) the old sum of every gap held the pod; now two light
    # gaps are reserved: 112 + 48 + 12 < 176 (and < 200).
    weights = [30, 30] + [21] * 11 + [22] * 4
    assert sum(weights) == 379
    args, fake = _units(tmp_path, [(w, 6) for w in weights])
    live, res = _run(f"{fake}\npodgate {args}").split()
    assert (live, res) == ("17", "48")
    for gate in (176, 200):
        env = f"NO_W_FILE_GB=12 MAXP=18 GATE_GB={gate} KILL_GB=236 IDLE_WAIT_S=300"
        assert _run(f"admit 17 {res} 112 12 0", env) == "start"
    _, old = _run(f"{fake}\nRESERVE_FULL_GB=0; podgate {args}").split()   # the old formula
    assert old == "277"
    assert _run(f"admit 17 {old} 112 12 0", "NO_W_FILE_GB=12 MAXP=18 GATE_GB=200 KILL_GB=236 IDLE_WAIT_S=300") == "hold"


def test_a_heavy_unit_keeps_its_whole_reserve(tmp_path):
    env = "NO_W_FILE_GB=12 MAXP=18 GATE_GB=200 KILL_GB=236 IDLE_WAIT_S=300"
    # an 80 GB head at 150 GB of real memory holds, even with nothing reserved
    assert _run("admit 10 0 150 80 0", env) == "hold"
    assert _run("admit 10 0 100 80 0", env) == "start"    # 180 < 200
    # once started (4 GB used) it reserves its whole gap, with no light gap reserved at all
    args, fake = _units(tmp_path, [(80, 4), (20, 6), (20, 6)])
    assert _run(f"{fake}\nRESERVE_TOP=0; podgate {args}") == "3 76"
    _, res = _run(f"{fake}\npodgate {args}").split()
    assert res == "104"
    assert _run(f"admit 3 {res} 60 35 0", env) == "start"    # 199
    assert _run(f"admit 3 {res} 60 36 0", env) == "hold"     # 200


def test_another_runs_claim_lets_a_unit_start_only_beside_it(tmp_path):
    d = _run_dir(tmp_path, "r1", [("u_001", 30), ("u_002", 20), ("u_003", 9)])
    c = tmp_path / "c"
    # another run's claim of 30 GB: a unit starts only when it fits beside it
    assert _in(d, c, "choose 10 10 100 0 0 other 30") == "start u_001 claim-fit -"   # 100 + 10 + 30 + 30 = 170
    assert _in(d, c, "choose 10 10 120 0 0 other 30") == "start u_003 claim-fit -"   # 120 + 40 + 9 = 169
    assert _in(d, c, "choose 10 10 130 0 0 other 30") == "none - claim-other -"      # 9 fits alone, not beside
    assert _in(d, c, "choose 0 0 0 0 0 other 30") == "none - claim-other -"          # idle: the claimant goes first
    assert _in(d, c, "choose 20 0 0 0 0 other 30") == "none - claim-other -"         # MAXP
    # this run's own head: skip-ahead while its hold is young; claimed, only the head
    assert _in(d, c, "choose 10 10 140 0 119 none") == "start u_002 skip-ahead -"    # 140 + 10 + 20 = 170
    assert _in(d, c, "choose 10 10 140 0 120 none") == "none - hold claim"
    assert _in(d, c, "choose 10 10 140 0 500 mine") == "none - hold -"
    assert _in(d, c, "choose 10 10 120 0 500 mine") == "start u_001 start -"


# runnerpid stand-in: pids 6001 and 6002 run runner.sh; 6003 is alive but a reused pid.
RUNNERS = "runnerpid() { case \"$1\" in 6001|6002) return 0;; esac; return 1; }\n"


def test_only_a_reused_pids_claim_expires_by_time_a_live_runners_never(tmp_path):
    claims = tmp_path / "claims"
    b = _run_dir(tmp_path, "run-b", [("u_001", 9)])
    live = {6001: "S", 6002: "S", 6003: "S"}
    _in(b, claims, "CLAIM_PID=6001 claim_write run-a u_001 30 1000; CLAIM_PID=6002 claim_write run-b u_001 9 4000; "
        "CLAIM_PID=6003 claim_write run-c u_002 30 900", live)
    # at 4500 the reused pid's claim is 3600 s old, not older than CLAIM_MAX_S: it still wins
    assert _in(b, claims, RUNNERS + "NOW=4500 CLAIM_MAX_S=3600 claim_state run-b", live) == "other run-c.u_002 30"
    script = RUNNERS + "NOW=4501 CLAIM_MAX_S=3600 claim_state run-b 2> err.txt; cat err.txt"
    assert _in(b, claims, script, live).splitlines() == ["other run-a.u_001 30", "4501 claim-expired run-c.u_002 age 3601 s"]
    # run-a's runner is alive: its 3601 s old claim stays and stays the oldest
    assert sorted(p.name for p in claims.iterdir()) == ["run-a.u_001", "run-b.u_001"]
    assert _in(b, claims, RUNNERS + "NOW=99999 CLAIM_MAX_S=3600 claim_state run-a", live) == "mine u_001 30"


PODCHECK = GATE.parent / "podcheck.sh"


def _cg(root: Path, files: dict[str, str]) -> Path:
    cg = root / "cg"
    for name, v in files.items():
        (cg / name).parent.mkdir(parents=True, exist_ok=True)
        (cg / name).write_text(v + "\n", encoding="utf-8", newline="\n")
    cg.mkdir(parents=True, exist_ok=True)
    return cg


@pytest.mark.parametrize("files, want", [
    ({"memory.current": "5", "memory.max": "max", "memory.peak": "7"}, "5 max 7"),
    ({"memory/memory.usage_in_bytes": "6", "memory/memory.limit_in_bytes": "9223372036854771712",
      "memory/memory.max_usage_in_bytes": "8"}, "6 9223372036854771712 8"),
    ({"memory.current": "5", "memory/memory.usage_in_bytes": "6"}, "5 none none"),   # v2 wins
    ({}, "none none none"),
])
def test_cgmem_reads_cgroup_v2_else_v1_else_fails(tmp_path, files, want):
    cg = _cg(tmp_path, files)
    got = _run(f"export CGROOT='{cg.as_posix()}'; for k in current max peak; do cgmem $k || echo none; done")
    assert got.split() == want.split()


def test_unit_overdue_counts_from_the_start_file(tmp_path):
    (tmp_path / "u_001.start").write_text("1000\n", encoding="utf-8", newline="\n")
    (tmp_path / "u_003.start").write_text("x\n", encoding="utf-8", newline="\n")
    script = (f"cd '{tmp_path.as_posix()}'; UNIT_MAX_S=14400; for a in 'u_001 15400' 'u_001 15401' 'u_002 99999' 'u_003 99999'; do "
              "unit_overdue $a && echo y || echo n; done")
    assert _run(script).split() == ["n", "y", "n", "n"]


def _podcheck(root: Path, pids: dict[int, str], cg: Path, want: str = "") -> tuple[int, list[str]]:
    """want: podcheck.sh's arguments after <root> (POD_SCRIPTS, POD_WEIGHTS)."""
    script = (f"{_fake_ps(pids)}\nexport -f ps\nCGROOT='{cg.as_posix()}' bash '{PODCHECK.as_posix()}' "
              f"'{root.as_posix()}' {want}")
    r = subprocess.run([BASH, "-c", script], check=False, capture_output=True, text=True, env=_clean_env())
    return r.returncode, r.stdout.strip().splitlines()


def _pod_run(root: Path, name: str, runner_pid: int | None, scripts: str | None, pending: int,
             weights: str = "c" * 64) -> Path:
    d = root / name
    (d / "out" / "run").mkdir(parents=True)
    if runner_pid is not None:
        (d / "runner.pid").write_text(f"{runner_pid}\n", encoding="utf-8", newline="\n")
    if scripts is not None:
        (d / "gate.env").write_text(f"GATE_VERSION=2\nPOD_SCRIPTS={scripts}\nPOD_WEIGHTS={weights}\n", encoding="utf-8", newline="\n")
    (d / "out" / "pending.txt").write_text("".join(f"u_{i:03d} tests/x.py\n" for i in range(pending)), encoding="utf-8", newline="\n")
    (d / "meta.env").write_text("WATCH_F=secretcode\n", encoding="utf-8", newline="\n")
    return d


A = "a" * 64
B = "b" * 64


def test_podcheck_lists_runs_gate_versions_memory_and_claims(tmp_path):
    root = tmp_path / "work"
    _pod_run(root, "new-a", 9000001, A, 3)
    _pod_run(root, "done-b", 9000002, B, 0)          # runner gone: not active
    _pod_run(root, "_podcheck", 9000001, None, 0)    # tool folders are skipped
    (root / ".claims").mkdir(parents=True)
    (root / ".claims" / "new-a.u_004").write_text("1 30 9000001\n", encoding="utf-8", newline="\n")
    cg = _cg(tmp_path, {"memory.current": "94000000000", "memory.max": "137438953472"})
    rc, out = _podcheck(root, {9000001: "S"}, cg, A)
    assert rc == 0
    assert out[0] == "run new-a gate=2 scripts=aaaaaaaaaaaa runner=alive live=0 pending=3"
    assert out[1] == "memory 94 GB limit 137 GB"
    assert out[2].startswith("claim new-a.u_004 age ") and out[2].endswith(" s weight 30 GB runner not-runner")
    assert out[-1] == "PODCHECK OK"
    assert not any("secretcode" in x for x in out)


def test_podcheck_reads_cgroup_v1_and_an_unlimited_limit(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    cg = _cg(tmp_path, {"memory/memory.usage_in_bytes": "3000000000", "memory/memory.limit_in_bytes": "9223372036854771712"})
    assert _podcheck(root, {}, cg) == (0, ["memory 3 GB limit max", "PODCHECK OK"])
    assert _podcheck(root, {}, tmp_path / "nocg") == (0, ["memory none limit none", "PODCHECK OK"])


def test_podcheck_fails_on_an_active_old_script_or_other_script_run(tmp_path):
    root = tmp_path / "work"
    _pod_run(root, "new-a", 9000001, A, 1)
    _pod_run(root, "old-c", 9000003, None, 5)
    _pod_run(root, "other-d", 9000004, B, 2)
    cg = _cg(tmp_path, {"memory.current": "0"})
    rc, out = _podcheck(root, {9000001: "S", 9000003: "S", 9000004: "S"}, cg, A)
    assert rc == 1
    assert "run old-c gate=old scripts=old runner=alive live=0 pending=5" in out
    assert out[-1] == "PODCHECK MIXED old=old-c other=other-d"
    rc, out = _podcheck(root, {9000001: "S", 9000004: "S"}, cg)   # no POD_SCRIPTS given: only old ones count
    assert (rc, out[-1]) == (0, "PODCHECK OK")


def test_podcheck_reports_other_weights_on_their_own_line_not_as_mixed(tmp_path):
    root = tmp_path / "work"
    _pod_run(root, "new-a", 9000001, A, 1)
    _pod_run(root, "new-e", 9000005, A, 1, weights="d" * 64)
    cg = _cg(tmp_path, {"memory.current": "0"})
    rc, out = _podcheck(root, {9000001: "S", 9000005: "S"}, cg, f"{A} {'c' * 64}")
    assert rc == 0
    assert "weights other=new-e" in out
    assert out[-1] == "PODCHECK OK"


BODY = ROOT / "tools" / "ops" / "day" / "podcheck_body.sh"


def _exec_body_files(body: dict) -> tuple[str, dict[str, bytes]]:
    """common.sh exec_body: the argv is ["sh","-c",<script>,"_","=<name>",<chunk>,...]; every item
    at most 4000 characters (exec-pod refuses one over 4096), at most 256 items. Returns the script
    and the decoded files."""
    import base64
    cmd = body["command"]
    assert cmd[:2] == ["sh", "-c"] and cmd[3] == "_"
    assert len(cmd) <= 256
    assert max(len(x) for x in cmd) <= 4000
    files: dict[str, str] = {}
    name = None
    for a in cmd[4:]:
        if a.startswith("="):
            name = a[1:]
            files[name] = ""
        else:
            assert name is not None
            files[name] += a
    return cmd[2], {n: base64.b64decode(b64) for n, b64 in files.items()}


def _head(name: str) -> bytes:
    out = subprocess.run(["git", "show", f"HEAD:tools/ops/day/pod/{name}"], check=True, capture_output=True, cwd=ROOT).stdout
    return out.replace(b"\r", b"")


def test_podcheck_body_carries_heads_gate_and_podcheck_and_their_sum():
    import json
    import re
    env = {**_clean_env(), "PACK_OFF_MAIN_POD": "1"}   # this checkout may be an ops branch
    r = subprocess.run([BASH, BODY.as_posix()], check=False, capture_output=True, text=True, cwd=ROOT, env=env)
    if "REFUSED: uncommitted changes" in r.stdout:
        pytest.skip("tools/ops/day/pod has uncommitted changes")
    assert r.returncode == 0, r.stdout + r.stderr
    script, files = _exec_body_files(json.loads(r.stdout.strip().splitlines()[-1]))
    assert list(files) == ["gate.sh", "podcheck.sh"]
    for name, data in files.items():
        assert data == _head(name)
    assert re.search(r"; bash \$W/podcheck.sh /work [0-9a-f]{64} [0-9a-f]{64}$", script)
    assert script.startswith("W=/work/_podcheck;")


def test_exec_body_script_writes_the_files_byte_for_byte(tmp_path):
    import json
    env = {**_clean_env(), "PACK_OFF_MAIN_POD": "1"}
    r = subprocess.run([BASH, BODY.as_posix()], check=False, capture_output=True, text=True, cwd=ROOT, env=env)
    if "REFUSED: uncommitted changes" in r.stdout:
        pytest.skip("tools/ops/day/pod has uncommitted changes")
    cmd = json.loads(r.stdout.strip().splitlines()[-1])["command"]
    for i, sh in enumerate(_posix_shells()):
        w = tmp_path / str(i)
        script = cmd[2].replace("W=/work/_podcheck;", f"W='{w.as_posix()}';").replace("bash $W/podcheck.sh", "echo RAN")
        run = subprocess.run([*sh, script, *cmd[3:]], check=False, capture_output=True, text=True, env=_clean_env())
        assert run.returncode == 0, (sh, run.stderr)
        assert run.stdout.startswith("RAN /work "), sh
        assert sorted(p.name for p in w.iterdir()) == ["gate.sh", "podcheck.sh"], sh
        for n in ("gate.sh", "podcheck.sh"):
            assert (w / n).read_bytes() == _head(n), sh


def _posix_shells() -> list[list[str]]:
    """bash in POSIX mode, and dash when it exists: the pod runs an exec body's script under sh."""
    dash = shutil.which("dash")
    return [[BASH, "--posix", "-c"]] + ([[dash, "-c"]] if dash else [])


def test_exec_body_writes_a_multi_chunk_file_and_an_empty_file_byte_for_byte(tmp_path):
    import json
    import random
    big = random.Random(0).randbytes(7001)   # base64 9336 characters: chunks 4000, 4000 and a partial 1336
    (tmp_path / "big.bin").write_bytes(big)
    (tmp_path / "empty.bin").write_bytes(b"")
    common = (ROOT / "tools" / "ops" / "common.sh").as_posix()
    r = subprocess.run([BASH, "-c", f". '{common}'; cd '{tmp_path.as_posix()}' && exec_body 'echo RAN' big=big.bin empty=empty.bin"],
                       check=False, capture_output=True, text=True, env=_clean_env())
    assert r.returncode == 0, r.stdout + r.stderr
    script, files = _exec_body_files(json.loads(r.stdout.strip().splitlines()[-1]))
    assert files == {"big": big, "empty": b""}
    cmd = json.loads(r.stdout.strip().splitlines()[-1])["command"]
    assert [len(x) for x in cmd[4:]] == [4, 4000, 4000, 1336, 6]
    for i, sh in enumerate(_posix_shells()):
        w = tmp_path / f"w{i}"
        run = subprocess.run([*sh, script.replace("W=/work/_podcheck;", f"W='{w.as_posix()}';"), *cmd[3:]],
                             check=False, capture_output=True, text=True, env=_clean_env())
        assert run.returncode == 0, (sh, run.stderr)
        assert run.stdout.strip() == "RAN", sh
        assert sorted(p.name for p in w.iterdir()) == ["big", "empty"], sh
        assert (w / "big").read_bytes() == big, sh
        assert (w / "empty").read_bytes() == b"", sh


# pack.sh's main check, in a throwaway repo holding only the scripts it needs before that check.
def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@localhost", "-c", "core.autocrlf=false",
                    *args], check=True, capture_output=True)


def test_pack_refuses_pod_scripts_that_differ_from_main(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools/ops/day/pod").mkdir(parents=True)
    for rel in ("tools/ops/common.sh", "tools/ops/day/pack.sh"):
        (repo / rel).write_bytes((ROOT / rel).read_bytes().replace(b"\r", b""))
    (repo / "tools/ops/day/pod/a.sh").write_text("echo 1\n", encoding="utf-8", newline="\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "main")
    _git(repo, "checkout", "-qb", "ops")
    (repo / "tools/ops/day/pod/a.sh").write_text("echo 2\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "ops")
    env = _clean_env()   # no NUZLINE_REPO, NUZLINE_SCRATCH or PACK_OFF_MAIN_POD either

    def pack(**extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, "tools/ops/day/pack.sh", "HEAD", "t.tar.xz", "r1", "nolist.txt"], check=False,
                              capture_output=True, text=True, cwd=repo, env={**env, **extra})

    r = pack()
    assert r.returncode == 3
    assert "REFUSED: tools/ops/day/pod at this checkout's HEAD differs from main's" in r.stdout
    assert "run it from a checkout at main (the main checkout)" in r.stdout
    assert "merge" not in r.stdout
    # the override, and a checkout at main, pass the check and stop at the next one
    off = pack(PACK_OFF_MAIN_POD="1").stdout
    assert "REFUSED: no such test list" in off
    assert sum(x.startswith("WARNING: PACK_OFF_MAIN_POD=1") for x in off.splitlines()) == 1
    _git(repo, "checkout", "-q", "main")
    r = pack()
    assert (r.returncode, "REFUSED: no such test list" in r.stdout) == (3, True)
    assert "WARNING" not in r.stdout


def test_setup_writes_gate_env_without_the_beacon_codes(tmp_path):
    d = tmp_path / "run1"
    (d / "bin").mkdir(parents=True)
    for n in ("setup.sh", "gate.sh"):
        (d / "bin" / n).write_bytes((GATE.parent / n).read_bytes().replace(b"\r", b""))
    (d / "meta.env").write_text("COMMIT=x\nGATE_VERSION=2\nPOD_SCRIPTS=abc\nPOD_WEIGHTS=def\nWATCH_F=secretf\nWATCH_D=secretd\n",
                                encoding="utf-8", newline="\n")
    # no cgroup memory file: setup stops right after writing gate.env
    knobs = "GATE_GB=176 MAXP=18 SPACING=4 RUNMAXP=6 WARMUP=\"it's x\" NUZLINE_HYPOTHESIS_PROFILE=explore DUR='40 x'"
    r = subprocess.run([BASH, "-c", f"{knobs} CGROOT='{(tmp_path / 'nocg').as_posix()}' bash '{(d / 'bin/setup.sh').as_posix()}'"],
                       check=False, capture_output=True, text=True, env=_clean_env())
    assert r.returncode != 0
    assert (d / "gate.env").read_text(encoding="utf-8") == "GATE_VERSION=2\nPOD_SCRIPTS=abc\nPOD_WEIGHTS=def\n"
    env = (d / "run.env").read_text(encoding="utf-8")
    assert "secret" not in env and "WATCH" not in env
    show = (f". '{(d / 'run.env').as_posix()}'; echo \"$GATE_GB|$MAXP|$SPACING|$RUNMAXP|$WARMUP|"
            f"$NUZLINE_HYPOTHESIS_PROFILE|$DUR|${{KILL_GB:-unset}}\"")
    # POSIX quoting: the restart body sources run.env under sh (dash on the pod)
    shells = [[BASH, "-c"], [BASH, "--posix", "-c"]] + ([[shutil.which("dash"), "-c"]] if shutil.which("dash") else [])
    for sh in shells:
        got = subprocess.run([*sh, show], check=True, capture_output=True, text=True, env=_clean_env())
        assert got.stdout.strip() == "176|18|4|6|it's x|explore|40 x|unset", sh
    assert "ERROR: no cgroup memory file" in r.stdout
    assert "SETUP_FAILED" in r.stdout
    assert "secret" not in r.stdout + r.stderr


def _linux_tools() -> bool:
    if BASH is None:
        return False
    r = subprocess.run([BASH, "-c", "command -v setsid flock >/dev/null && ps -eo pid=,pgid=,stat= >/dev/null"],
                       check=False, capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _linux_tools(), reason="needs setsid, flock and a procps ps (the pod)")
def test_runner_kills_a_unit_over_unit_max_s_with_rc_96(tmp_path):
    work = tmp_path / "work"
    d = work / "run-t"
    (d / "bin").mkdir(parents=True)
    (d / "out").mkdir()
    for n in ("runner.sh", "gate.sh"):
        (d / "bin" / n).write_bytes((GATE.parent / n).read_bytes().replace(b"\r", b""))
    # a unit that never ends: it writes its start time like unit.sh and sleeps
    (d / "bin" / "unit.sh").write_text('u="$1"; O="$(cd "$(dirname "$0")/.." && pwd)/out"\n'
                                       'date +%s > "$O/$u.start"; exec sleep 300\n', encoding="utf-8", newline="\n")
    (d / "out" / "queue.txt").write_text("u_001 tests/x.py\n", encoding="utf-8", newline="\n")
    (d / "out" / "u_001.w").write_text("1\n", encoding="utf-8", newline="\n")
    cg = _cg(tmp_path, {"memory.current": "1000000000"})
    env = (f"CGROOT='{cg}' WORK='{work}' RUNNER_LOCK='{tmp_path / 'lock'}' CLAIMS='{tmp_path / 'claims'}' "
           "WARMUP=none SPACING=0 UNIT_MAX_S=2")
    log = d / "runner.log"
    subprocess.run([BASH, "-c", f"{env} bash '{d / 'bin/runner.sh'}' > '{log}' 2>&1"], check=True, timeout=120, env=_clean_env())
    o = d / "out"
    assert (o / "u_001.rc").read_text(encoding="utf-8").strip() == "96"
    assert (o / "timedout.txt").read_text(encoding="utf-8").split() == ["u_001"]
    assert (o / "DONE").is_file()
    assert "TIMEOUT u_001: over UNIT_MAX_S 2 s, killed: rc 96" in log.read_text(encoding="utf-8")



STOP = GATE.parent / "stop.sh"


def _alive(pid: int) -> bool:
    r = subprocess.run([BASH, "-c", f"ps -o stat= -p {pid}"], check=False, capture_output=True, text=True)
    return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


def _launch(script: str) -> None:
    subprocess.run([BASH, "-c", script], check=True, timeout=30, env=_clean_env())


def _children(pid: int) -> list[int]:
    out = subprocess.run([BASH, "-c", f"sleep 1; ps -eo pid=,ppid= | awk '$2 == {pid} {{print $1}}'"],
                         check=True, capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def _stop(run: str, root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(STOP), run, str(root)], check=False, capture_output=True, text=True, timeout=60,
                          env=_clean_env())


Q = "> /dev/null 2>&1 < /dev/null"


@pytest.mark.skipif(not _linux_tools(), reason="needs setsid, /proc and a procps ps (the pod)")
def test_stop_kills_setup_notify_runner_units_and_the_orphaned_memory_loop_but_nothing_else(tmp_path):
    root = tmp_path / "work"
    d = root / "run-s"
    (d / "out" / "run").mkdir(parents=True)
    (d / "repo").mkdir()
    (d / "bin").mkdir()
    (d / "bin" / "runner.sh").write_text("sleep 300 & wait\n", encoding="utf-8", newline="\n")
    (root / ".claims").mkdir()
    (root / ".claims" / "run-s.u_002").write_text("1 9 1\n", encoding="utf-8", newline="\n")
    (root / ".claims" / "run-sx.u_001").write_text("1 9 1\n", encoding="utf-8", newline="\n")
    t = tmp_path
    _launch(f"cd '{t}'\n"
            # setup.sh: cwd the run; its child works elsewhere (found only as a descendant)
            f"(cd '{d}' && exec bash -c \"(cd '{t}' && exec sleep 300) & wait\") {Q} & echo $! > '{d}/setup.pid'\n"
            # the runner: cwd elsewhere, cmdline names the run's bin/; its child is a descendant
            f"bash '{d}/bin/runner.sh' {Q} & echo $! > '{d}/runner.pid'\n"
            f"(cd '{d}' && exec sleep 300) {Q} & echo $! > '{d}/notify.pid'\n"
            f"(cd '{d}/repo' && exec setsid sleep 300) {Q} & echo $! > '{d}/out/run/u_001'\n"
            f"(cd '{d}/out' && exec sleep 300) {Q} & echo $! > '{t}/memloop.pid'\n"   # the runner's loop, orphaned
            f"sleep 300 {Q} & echo $! > '{t}/bystander.pid'\n")
    pid = {n: int((d / f"{n}.pid").read_text()) for n in ("setup", "runner", "notify")}
    pid["unit"] = int((d / "out/run/u_001").read_text())
    pid["memloop"] = int((t / "memloop.pid").read_text())
    for n in ("setup", "runner"):
        kids = _children(pid[n])
        assert kids, f"{n} stand-in has no child"
        pid[f"{n}_child"] = kids[0]
    bystander = int((t / "bystander.pid").read_text())
    try:
        assert all(_alive(x) for x in pid.values())
        r = _stop("run-s", root)
        assert r.stdout.strip().splitlines()[-1] == "STOP_OK run-s", r.stdout + r.stderr
        assert {n: _alive(x) for n, x in pid.items()} == {n: False for n in pid}
        assert _alive(bystander)
        assert (d / "STOPPED").is_file()
        assert sorted(x.name for x in (root / ".claims").iterdir()) == ["run-sx.u_001"]
    finally:
        subprocess.run([BASH, "-c", f"kill -9 {bystander} {' '.join(map(str, pid.values()))} 2>/dev/null"], check=False)


@pytest.mark.skipif(not _linux_tools(), reason="needs setsid, /proc and a procps ps (the pod)")
def test_stop_skips_stale_pid_files_and_unit_groups_that_name_other_processes(tmp_path):
    root = tmp_path / "work"
    d = root / "run-s"
    (d / "out" / "run").mkdir(parents=True)
    other = root / "run-s2"                       # a prefix-named run: its processes are not run-s's
    (other / "bin").mkdir(parents=True)
    (other / "bin" / "runner.sh").write_text("sleep 300 & wait\n", encoding="utf-8", newline="\n")
    t = tmp_path
    _launch(f"cd '{t}'\n"
            f"sleep 300 {Q} & echo $! > '{d}/setup.pid'\n"                 # setup.sh's pid, reused by a bystander
            f"setsid sleep 300 {Q} & echo $! > '{d}/out/run/u_001'\n"      # a unit group id naming a bystander's group
            f"(cd '{other}' && exec bash '{other}/bin/runner.sh') {Q} & echo $! > '{d}/runner.pid'\n")
    keep = {"setup": int((d / "setup.pid").read_text()), "unit": int((d / "out/run/u_001").read_text()),
            "other_runner": int((d / "runner.pid").read_text())}
    keep["other_child"] = _children(keep["other_runner"])[0]
    try:
        assert all(_alive(x) for x in keep.values())
        r = _stop("run-s", root)
        assert r.stdout.strip().splitlines()[-1] == "STOP_OK run-s", r.stdout + r.stderr
        assert {n: _alive(x) for n, x in keep.items()} == {n: True for n in keep}
        for n, src in (("setup", "setup.pid"), ("unit", "out/run/u_001"), ("other_runner", "runner.pid")):
            assert f"skip {keep[n]} {src}: not this run's" in r.stderr
    finally:
        subprocess.run([BASH, "-c", f"kill -9 {' '.join(map(str, keep.values()))} 2>/dev/null"], check=False)


def test_stop_refuses_bad_names_and_missing_runs(tmp_path):
    for name in ("", "_bin", "a/b", "A"):
        r = subprocess.run([BASH, STOP.as_posix(), name, tmp_path.as_posix()], check=False, capture_output=True, text=True,
                           env=_clean_env())
        assert (r.returncode, r.stdout.startswith("STOP_REFUSED")) == (2, True), name
    r = subprocess.run([BASH, STOP.as_posix(), "nope", tmp_path.as_posix()], check=False, capture_output=True, text=True,
                       env=_clean_env())
    assert (r.returncode, r.stdout.strip()) == (2, f"STOP_REFUSED: no folder {tmp_path.as_posix()}/nope")



def test_stop_body_carries_heads_stop_sh_and_the_run_name():
    import json
    env = {**_clean_env(), "PACK_OFF_MAIN_POD": "1"}   # this checkout may be an ops branch
    body_sh = (ROOT / "tools" / "ops" / "day" / "stop_body.sh").as_posix()
    r = subprocess.run([BASH, body_sh, "run-x"], check=False, capture_output=True, text=True, cwd=ROOT, env=env)
    if "REFUSED: uncommitted changes" in r.stdout:
        pytest.skip("tools/ops/day/pod has uncommitted changes")
    assert r.returncode == 0, r.stdout + r.stderr
    script, files = _exec_body_files(json.loads(r.stdout.strip().splitlines()[-1]))
    assert files == {"stop.sh": _head("stop.sh")}
    assert script.startswith("W=/work/_podcheck;") and script.endswith("; bash $W/stop.sh run-x")
    bad = subprocess.run([BASH, body_sh, "Bad/name"], check=False, capture_output=True, text=True, cwd=ROOT, env=env)
    assert bad.returncode == 2


def _agg_peak_line(tmp_path: Path, mem: str | None) -> str:
    d = tmp_path / "run1"
    (d / "bin").mkdir(parents=True)
    (d / "out").mkdir()
    for n in ("agg.sh", "gate.sh"):
        (d / "bin" / n).write_bytes((GATE.parent / n).read_bytes().replace(b"\r", b""))
    (d / "out" / "queue.txt").write_text("", encoding="utf-8", newline="\n")
    if mem is not None:
        (d / "out" / "mem_mb.txt").write_text(mem, encoding="utf-8", newline="\n")
    r = subprocess.run([BASH, "-c", f"CGROOT='{(tmp_path / 'nocg').as_posix()}' bash '{(d / 'bin/agg.sh').as_posix()}'"],
                       check=False, capture_output=True, text=True, env=_clean_env())
    return next(x for x in r.stdout.splitlines() if x.startswith("--- wall "))


@pytest.mark.parametrize(("mem", "want"), [
    ("1700000000 51234\n1700000015 60000\nbad\n1700000030 \n1700000045 x\n1700000060 9999\n1700000075 99999x\n", "60000 MB"),
    ("", "n/a"),
    (None, "n/a"),
])
def test_agg_prints_the_runs_peak_from_mem_mb_txt(tmp_path, mem, want):
    """agg.sh's peak line: the largest numeric second field of mem_mb.txt; malformed lines skipped."""
    line = _agg_peak_line(tmp_path, mem)
    assert f"container peak while this run ran (max of its mem_mb.txt, 15 s samples, incl. other runs) {want};" in line
    assert line.endswith("container lifetime peak (since the pod started, incl. earlier and other runs) n/a")
