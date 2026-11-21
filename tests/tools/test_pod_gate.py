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
        exec_path = subprocess.run([git, "--exec-path"], capture_output=True, text=True).stdout.strip()
        for b in (Path(exec_path).parents[2] / "bin" / "bash.exe", Path(git).parents[1] / "bin" / "bash.exe"):
            if b.is_file():
                return str(b)
    return None


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _run(script: str, env: str = "DEFW=20 MAXP=10 GATE_GB=88 KILL_GB=118 IDLE_WAIT_S=300") -> str:
    full = f"set -u; {env}; . '{GATE.as_posix()}'\n{script}"
    r = subprocess.run([BASH, "-c", full], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _unit(run_dir: Path, u: str, pgid: str, weight: str | None) -> str:
    (run_dir / "run").mkdir(parents=True, exist_ok=True)
    (run_dir / "run" / u).write_text(pgid + "\n", encoding="utf-8")
    if weight is not None:
        (run_dir / f"{u}.w").write_text(weight + "\n", encoding="utf-8")
    return (run_dir / "run" / u).as_posix()


# ps -eo pgid=,stat=,rss=  (rss in KiB); 29296875 KiB = 30.0 GB
PS = ["101 S 20000000", "101 S 9296875", "102 R 19531250", "103 S 9765625",
      "201 Z 0", "203 S 0", "204 S 0", "1 S 100", "0 S 100"]


def test_podgate_counts_live_groups_and_reserves_unreached_peaks(tmp_path):
    a, b = tmp_path / "r1" / "out", tmp_path / "r2" / "sub" / "out"
    files = [
        _unit(a, "u_001", "101", "90"),    # 30 GB used of 90: reserve 60
        _unit(a, "u_002", "102", None),    # no .w: DEFW 20, 20 GB used: reserve 0
        _unit(a, "u_003", "103", "x7"),    # malformed .w: DEFW 20, 10 GB used: reserve 10
        _unit(b, "u_001", "201", "40"),    # only a zombie: not live
        _unit(b, "u_002", "1", "50"),      # pgid 1 (PID 1): never counted
        _unit(b, "u_003", "0", "50"),      # pgid 0: never counted
        _unit(b, "u_004", "abc", "50"),    # malformed pgid
        _unit(b, "u_005", "203", "25"),    # just started: reserve 25
        _unit(b, "u_006", "999", "25"),    # group gone
    ]
    fake = "ps() { printf '%s\\n' " + " ".join(f"'{p}'" for p in PS) + "; }"
    args = " ".join(f"'{f}'" for f in files + [(tmp_path / "none" / "run" / "u_9").as_posix()])
    assert _run(f"{fake}\npodgate {args}") == "4 95"


def test_podgate_rounds_a_fractional_reserve_up_and_handles_no_units(tmp_path):
    f = _unit(tmp_path / "out", "u_001", "204", "10")
    fake = "ps() { echo '204 S 9277343'; }"  # 9.4999 GB used of 10: 0.5 GB rounds up to 1
    assert _run(f"{fake}\npodgate '{f}'") == "1 1"
    assert _run(f"{fake}\npodgate") == "0 0"


def test_wgb_falls_back_to_the_default(tmp_path):
    (tmp_path / "a.w").write_text("33\n", encoding="utf-8")
    (tmp_path / "b.w").write_text("3.5\n", encoding="utf-8")
    got = _run(" ".join(f"wgb '{(tmp_path / n).as_posix()}';" for n in ("a.w", "b.w", "c.w")))
    assert got.split() == ["33", "20", "20"]


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
    return ("ps() { case \"$*\" in *pid=,stat=,rss=*) :;; *pid=,stat=*) printf '%s\n' " + rows
            + ";; esac; }")


def _run_dir(root: Path, name: str, units: list[tuple[str, int]]) -> Path:
    d = root / name / "out"
    d.mkdir(parents=True)
    (d / "pending.txt").write_text("".join(f"{u} tests/{u}.py\n" for u, _ in units), encoding="utf-8")
    for u, gb in units:
        (d / f"{u}.w").write_text(f"{gb}\n", encoding="utf-8")
    return d


def _in(d: Path, claims: Path, script: str, pids: dict[int, str] = {}) -> str:
    env = (f"DEFW=20 MAXP=20 GATE_GB=176 KILL_GB=236 IDLE_WAIT_S=300 CLAIM_S=120 "
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
    (claims / "run-d.u_002").write_text("garbage\n", encoding="utf-8")
    assert _in(b, claims, "claim_state run-b", {5002: "Z"}) == "none"
    assert sorted(p.name for p in claims.iterdir()) == []
