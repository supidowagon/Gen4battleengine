"""tools/ops/day/pod/gate.sh: runner.sh's start-gate arithmetic, run in bash with a fake ps."""

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
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _run(script: str, env: str = "NO_W_FILE_GB=12 MAXP=10 GATE_GB=88 KILL_GB=118 IDLE_WAIT_S=300") -> str:
    full = f"set -u; {env}; . '{GATE.as_posix()}'\n{script}"
    r = subprocess.run([BASH, "-c", full], capture_output=True, text=True, check=True)
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
    assert _in(d, tmp_path / "c", "choose 1 0 0 0 0 other") == "none - claim-other -"


def test_claim_hand_over_between_two_runners(tmp_path):
    claims = tmp_path / "claims"
    a = _run_dir(tmp_path, "run-a", [("u_001", 30)])
    b = _run_dir(tmp_path, "run-b", [("u_001", 9), ("u_002", 9)])
    live = {4001: "S", 4002: "S"}
    _in(a, claims, "CLAIM_PID=4001 claim_write run-a u_001 30 1000", live)
    assert (claims / "run-a.u_001").read_text(encoding="utf-8").split() == ["1000", "30", "4001"]
    assert _in(a, claims, "claim_state run-a", live) == "mine u_001"
    assert _in(b, claims, "claim_state run-b", live) == "other run-a.u_001"
    # b's own later claim does not outrank the older one
    _in(b, claims, "CLAIM_PID=4002 claim_write run-b u_001 9 1005", live)
    assert _in(b, claims, "claim_state run-b", live) == "other run-a.u_001"
    # a starts its unit and drops the claim: b's claim is now the oldest
    _in(a, claims, "claim_drop run-a u_001", live)
    assert _in(b, claims, "claim_state run-b", live) == "mine u_001"
    assert _in(a, claims, "claim_state run-a", live) == "other run-b.u_001"


def test_a_dead_or_zombie_claimant_is_ignored_and_deleted(tmp_path):
    claims = tmp_path / "claims"
    b = _run_dir(tmp_path, "run-b", [("u_001", 9)])
    _in(b, claims, "CLAIM_PID=5001 claim_write run-a u_001 30 1000; CLAIM_PID=5002 claim_write run-c u_004 30 1001")
    (claims / "run-d.u_002").write_text("garbage\n", encoding="utf-8", newline="\n")
    assert _in(b, claims, "claim_state run-b", {5002: "Z"}) == "none"
    assert sorted(p.name for p in claims.iterdir()) == []


def test_podgate_counts_a_unit_that_has_not_called_setsid_yet(tmp_path):
    # pid 301 is alive but still in the runner's group 50: live, nothing of group 301 yet
    f = _unit(tmp_path / "out", "u_001", "301", "30")
    fake = "ps() { printf '%s\\n' '50 50 S 1000' '301 50 S 2000'; }"
    assert _run(f"{fake}\npodgate '{f}'") == "1 30"
    fake_z = "ps() { printf '%s\\n' '50 50 S 1000' '301 50 Z 0'; }"
    assert _run(f"{fake_z}\npodgate '{f}'") == "0 0"


def test_an_expired_claim_is_deleted_and_logged_and_the_next_one_wins(tmp_path):
    claims = tmp_path / "claims"
    b = _run_dir(tmp_path, "run-b", [("u_001", 9)])
    live = {6001: "S", 6002: "S"}
    _in(b, claims, "CLAIM_PID=6001 claim_write run-a u_001 30 1000; CLAIM_PID=6002 claim_write run-b u_001 9 4000", live)
    script = "NOW=4601 CLAIM_MAX_S=3600 claim_state run-b 2> err.txt; cat err.txt"
    assert _in(b, claims, script, live).splitlines() == ["mine u_001", "4601 claim-expired run-a.u_001 age 3601 s"]
    assert sorted(p.name for p in claims.iterdir()) == ["run-b.u_001"]
    assert _in(b, claims, "NOW=4600 CLAIM_MAX_S=3600 claim_state run-a", live) == "other run-b.u_001"


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
    script = (f"{_fake_ps(pids)}\nexport -f ps\nCGROOT='{cg.as_posix()}' bash '{PODCHECK.as_posix()}' "
              f"'{root.as_posix()}' {want}")
    r = subprocess.run([BASH, "-c", script], check=False, capture_output=True, text=True)
    return r.returncode, r.stdout.strip().splitlines()


def _pod_run(root: Path, name: str, runner_pid: int | None, scripts: str | None, pending: int) -> Path:
    d = root / name
    (d / "out" / "run").mkdir(parents=True)
    if runner_pid is not None:
        (d / "runner.pid").write_text(f"{runner_pid}\n", encoding="utf-8", newline="\n")
    if scripts is not None:
        (d / "gate.env").write_text(f"GATE_VERSION=2\nPOD_SCRIPTS={scripts}\n", encoding="utf-8", newline="\n")
    (d / "out" / "pending.txt").write_text("".join(f"u_{i:03d} tests/x.py\n" for i in range(pending)), encoding="utf-8", newline="\n")
    (d / "meta.env").write_text("WATCH_F=secretcode\n", encoding="utf-8", newline="\n")
    return d


A = "a" * 64
B = "b" * 64


def test_podcheck_lists_runs_gate_versions_memory_and_claims(tmp_path):
    root = tmp_path / "work"
    _pod_run(root, "new-a", 7001, A, 3)
    _pod_run(root, "done-b", 7002, B, 0)          # runner gone: not active
    _pod_run(root, "_podcheck", 7001, None, 0)    # tool folders are skipped
    (root / ".claims").mkdir(parents=True)
    (root / ".claims" / "new-a.u_004").write_text("1 30 7001\n", encoding="utf-8", newline="\n")
    cg = _cg(tmp_path, {"memory.current": "94000000000", "memory.max": "137438953472"})
    rc, out = _podcheck(root, {7001: "S"}, cg, A)
    assert rc == 0
    assert out[0] == "run new-a gate=2 scripts=aaaaaaaaaaaa runner=alive live=0 pending=3"
    assert out[1] == "memory 94 GB limit 137 GB"
    assert out[2].startswith("claim new-a.u_004 age ") and out[2].endswith(" s weight 30 GB runner alive")
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
    _pod_run(root, "new-a", 7001, A, 1)
    _pod_run(root, "old-c", 7003, None, 5)
    _pod_run(root, "other-d", 7004, B, 2)
    cg = _cg(tmp_path, {"memory.current": "0"})
    rc, out = _podcheck(root, {7001: "S", 7003: "S", 7004: "S"}, cg, A)
    assert rc == 1
    assert "run old-c gate=old scripts=old runner=alive live=0 pending=5" in out
    assert out[-1] == "PODCHECK MIXED old=old-c other=other-d"
    rc, out = _podcheck(root, {7001: "S", 7004: "S"}, cg)   # no POD_SCRIPTS given: only old ones count
    assert (rc, out[-1]) == (0, "PODCHECK OK")


BODY = ROOT / "tools" / "ops" / "day" / "podcheck_body.sh"


def test_podcheck_body_carries_heads_gate_and_podcheck_and_their_sum():
    import base64
    import json
    import re
    r = subprocess.run([BASH, BODY.as_posix()], check=False, capture_output=True, text=True, cwd=ROOT)
    if "REFUSED: uncommitted changes" in r.stdout:
        pytest.skip("tools/ops/day/pod has uncommitted changes")
    assert r.returncode == 0, r.stdout + r.stderr
    body = json.loads(r.stdout.strip().splitlines()[-1])
    cmd = body["command"][2]
    parts = re.findall(r"echo ([A-Za-z0-9+/=]+) \| base64 -d > /work/_podcheck/(\w+\.sh)", cmd)
    assert [n for _, n in parts] == ["gate.sh", "podcheck.sh"]
    for b64, name in parts:
        head = subprocess.run(["git", "show", f"HEAD:tools/ops/day/pod/{name}"], check=True, capture_output=True, cwd=ROOT).stdout
        assert base64.b64decode(b64) == head.replace(b"\r", b"")
    assert re.search(r"bash /work/_podcheck/podcheck.sh /work [0-9a-f]{64}$", cmd)
