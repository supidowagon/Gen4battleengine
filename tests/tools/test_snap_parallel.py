"""tools/snap_compare.py's parallel driver (--workers > 1): print order, failure propagation,
the capture's step-first order and the memory gate. Children are fake processes on a fake
clock, so no engine is imported and no test waits."""

import itertools
import json
import re
import subprocess
import sys

import pytest

from tools import snap_compare as sc

BASE = "a" * 40


def result():
    return {"steps": 2, "diverged": None, "only_base": [], "only_new": [], "calls": 4,
            "raised": {}, "peak_bytes": 1}  # fmt: skip


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    def __init__(self, clock, ends, code, pid):
        self.clock, self.ends, self.code, self.pid = clock, ends, code, pid
        self.killed = False

    def poll(self):
        return self.code if self.killed or self.clock() >= self.ends else None

    def kill(self):
        self.killed, self.code = True, -9

    def wait(self):
        return self.code


class Fake:
    """``launch`` for PartPool: part ``p`` runs ``took(p)`` s and exits ``codes.get(p, 0)``,
    printing a RESULT line unless its code is not 0. Records every start and end time."""

    def __init__(self, clock, took=lambda p: 10.0, codes=None):
        self.clock, self.took, self.codes = clock, took, codes or {}
        self.starts, self.procs = {}, {}

    def __call__(self, cmd, out, err):
        part = cmd[5]
        code = self.codes.get(part, 0)
        if code == 0:
            out.write("RESULT " + json.dumps(result()) + "\n")
        else:
            err.write("Traceback (most recent call last):\nKilled\n")
        out.flush()
        err.flush()
        self.starts[part] = self.clock()
        proc = FakeProc(self.clock, self.clock() + self.took(part), code, 1000 + len(self.procs))
        self.procs[part] = proc
        return proc

    def end(self, part):
        return self.starts[part] + self.took(part)


@pytest.fixture
def pool(tmp_path):
    """``pool(parts, ...) -> (PartPool, Fake, Clock)`` on a fake clock, its temp files under
    ``tmp_path``; every pool made is closed at teardown."""
    made = []

    def make(parts, workers=4, gate_gb=1000, mode="compare", **kw):
        clock = kw.pop("clock", None) or Clock()
        fake = kw.pop("fake", None) or Fake(clock)
        p = sc.PartPool(mode, "snap", parts, workers, gate_gb, launch=fake, clock=clock,
                        sleep=clock.sleep, memory=kw.pop("memory", lambda: None),
                        rss=kw.pop("rss", lambda pid: 0), tmp=tmp_path, **kw)  # fmt: skip
        made.append(p)
        return p, fake, clock

    yield make
    for p in made:
        p.close()


@pytest.fixture
def run(tmp_path, monkeypatch):
    """``run(argv, took=..., codes=...) -> (exit code, printed lines)`` with fake parts at a
    git-less checkout; ``--workers`` > 1 goes through PartPool on a fake clock."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(sc, "ROOT", repo)
    monkeypatch.setattr(sc, "missing", lambda part: [])
    monkeypatch.setattr(sc, "git_info", lambda r: None)
    monkeypatch.setattr(sc, "content_hash", lambda r: "code-a")
    real = sc.PartPool

    def go(argv, capsys, took=lambda p: 10.0, codes=None):
        codes = codes or {}

        def serial(mode, d, part):
            if codes.get(part, 0):
                return codes[part], "", "Traceback (most recent call last):\nKilled\n"
            return 0, "RESULT " + json.dumps(result()) + "\n", ""

        monkeypatch.setattr(sc, "run_part", serial)

        def fake_pool(mode, directory, parts, workers, gate_gb):
            clock = Clock()
            return real(mode, directory, parts, workers, gate_gb, launch=Fake(clock, took, codes),
                        clock=clock, sleep=clock.sleep, memory=lambda: None, rss=lambda pid: 0,
                        tmp=tmp_path)  # fmt: skip

        monkeypatch.setattr(sc, "PartPool", fake_pool)
        capsys.readouterr()
        rc = sc.parent(sc.build_parser().parse_args(argv))
        return rc, capsys.readouterr().out.splitlines()

    go.snap = tmp_path / "snap"
    return go


def _parts_only(lines):
    """The printed lines without timings and without the driver's own two lines."""
    keep = [ln for ln in lines
            if "parts at once" not in ln and "parts ran at once" not in ln and "gate wait" not in ln]  # fmt: skip
    return [re.sub(r"\(?\d+(\.\d+)? s\)?", "<t>", ln) for ln in keep]


def test_parallel_prints_what_serial_prints_in_the_same_order(run, capsys):
    argv = ["capture", str(run.snap), "0-29", "--commit", BASE]
    serial_rc, serial = run(argv, capsys)
    serial_rec = json.loads((run.snap / "capture.json").read_text())
    (run.snap / "capture.json").unlink()
    # Later parts end first: the print order must still be the part list's.
    order = sc.plan_parts("0-29", 10, None, False)[2]
    rank = {p: i for i, p in enumerate(order)}
    # Under 60 s each, so no part is OVER LIMIT on either side.
    rc, par = run([*argv, "--workers", "5"], capsys, took=lambda p: 50.0 - rank[p])
    assert rc == serial_rc == 0
    assert _parts_only(par) == _parts_only(serial)
    printed = [ln.split(":")[0] for ln in par if ln.split(":")[0] in rank]
    assert printed == order
    assert json.loads((run.snap / "capture.json").read_text())["parts"] == serial_rec["parts"]
    assert any("at most 5 parts ran at once" in ln for ln in par)


def test_a_killed_part_fails_and_is_never_ok(run, capsys):
    argv = ["capture", str(run.snap), "0-9", "--commit", BASE, "--workers", "3"]
    rc, lines = run(argv, capsys, codes={"0-9.sample": -9})
    assert rc == 1
    assert any(ln.startswith("0-9.sample: ERROR (exit -9") for ln in lines)
    assert "FAILED (1 failure(s)" in lines[-1]
    parts = json.loads((run.snap / "capture.json").read_text())["parts"]
    assert parts["0-9.sample"] == "failed"
    assert {v for k, v in parts.items() if k != "0-9.sample"} == {"ok"}


def test_compare_exit_codes_match_serial(run, capsys, monkeypatch):
    snap = str(run.snap)
    assert run(["capture", snap, "0-9", "--commit", BASE], capsys)[0] == 0
    monkeypatch.setattr(sc, "content_hash", lambda r: "code-b")
    new = "b" * 40
    for codes in ({}, {"roark.rank": 2}, {"0-9.gen": -15}):
        s = run(["compare", snap, "0-9", "--commit", new], capsys, codes=codes)
        p = run(["compare", snap, "0-9", "--commit", new, "--workers", "4"], capsys, codes=codes)
        assert s[0] == p[0] == (1 if codes else 0)
        assert _parts_only(p[1]) == _parts_only(s[1])


def test_capture_starts_a_chunk_s_groups_after_its_step_part(pool):
    parts = sc.plan_parts("0-29", 10, None, False)[2]
    p, fake, _ = pool(parts, workers=8, mode="capture")
    for part in parts:
        p.result(part)
    for part in parts:
        head, _, group = part.partition(".")
        if head != "roark" and group != "step":
            assert fake.starts[part] >= fake.end(f"{head}.step"), part
    # compare reads saved inputs only: no order between a chunk's groups.
    assert sc.part_deps("compare", "0-9.rest", parts) == set()
    assert sc.part_deps("capture", "0-9.rest", parts) == {"0-9.step"}
    assert sc.part_deps("capture", "0-9.rest", ["0-9.rest"]) == set()  # step not planned
    assert sc.part_deps("capture", "roark.rank", parts) == set()


def test_longest_part_starts_first_and_the_worker_cap_holds(pool):
    parts = sc.plan_parts("0-29", 10, None, False)[2]
    p, fake, _ = pool(parts, workers=3)
    live_max = 0
    for part in parts:
        p.result(part)
        live_max = max(live_max, p.peak_live)
    assert min(fake.starts, key=fake.starts.get) == "roark.rank"
    assert live_max == 3


def test_admit_counts_memory_reserve_and_own_peak():
    gb = 10**9
    assert sc.admit(13, {}, None, 88)  # nothing live, no reading
    # Nothing live: starts whatever memory reads (page cache counts in memory.current).
    assert sc.admit(13, {}, 87 * gb, 88) and sc.admit(13, {}, 88 * gb, 88)
    assert sc.admit(13, {}, 500 * gb, 88)
    assert sc.admit(200, {}, 0, 88)  # too heavy for the gate: starts alone
    live = {"a": (13, 3 * gb), "b": (13, 13 * gb)}
    # memory 40 + reserve (10 + 0) + 13 = 63
    assert sc.admit(13, live, 40 * gb, 63) and not sc.admit(13, live, 40 * gb, 62)
    # no memory reading: the live RSS sum (16) + reserve (10) + 13 = 39
    assert sc.admit(13, live, None, 39) and not sc.admit(13, live, None, 38)


def test_gate_keeps_live_parts_under_it(pool):
    parts = sc.plan_parts("0-29", 10, "step,jitstep,sample", True)[2]  # 13 GB each
    holder = {}

    def mem():  # every live part already at its peak
        return sum(sc.part_gb(q) for q in holder["p"].live) * 10**9

    p, fake, _ = pool(parts, workers=16, gate_gb=40, memory=mem)
    holder["p"] = p
    for part in parts:
        p.result(part)
    # A second part: memory 13 + reserve 13 (RSS reads 0) + 13 = 39 <= 40; a third: 65 > 40.
    assert p.peak_live == 2
    intervals = sorted((fake.starts[q], fake.end(q)) for q in parts)
    for t, _ in intervals:
        assert sum(1 for s, e in intervals if s <= t < e) <= 2


def test_heavy_part_waits_then_runs_alone(pool):
    parts = ["0-9.ai", "roark.rank", "0-9.gen"]  # roark.rank 27 GB against a 20 GB gate
    p, fake, _ = pool(parts, workers=4, gate_gb=20)
    for part in parts:
        p.result(part)
    rank = (fake.starts["roark.rank"], fake.end("roark.rank"))
    for q in ("0-9.ai", "0-9.gen"):
        assert fake.end(q) <= rank[0] or fake.starts[q] >= rank[1]


def test_close_kills_live_children(pool):
    p, fake, _ = pool(["roark.rank", "roark.trace"], workers=2, spacing=0)
    p._start()
    p._start()
    assert len(p.live) == 2
    p.close()
    assert all(proc.killed for proc in fake.procs.values())
    assert not p.tmp.exists()


def test_a_real_child_s_output_is_read_back(tmp_path):
    def launch(cmd, out, err):
        code = "import sys; print('RESULT {}'); print('warn', file=sys.stderr); sys.exit(3)"
        return subprocess.Popen([sys.executable, "-c", code], stdout=out, stderr=err)

    p = sc.PartPool("compare", "snap", ["0-9.ai"], 2, 88, launch=launch, memory=lambda: None,
                    poll=0.05, tmp=tmp_path)  # fmt: skip
    try:
        code, out, err, took = p.result("0-9.ai")
    finally:
        p.close()
    assert (code, out.strip(), err.strip()) == (3, "RESULT {}", "warn")
    assert took >= 0


@pytest.mark.parametrize("extra", [["--workers", "0"], ["--gate-gb", "0"]])
def test_bad_worker_knobs_are_usage_errors(run, capsys, extra):
    assert run(["capture", str(run.snap), "0-9", "--commit", BASE, *extra], capsys)[0] == 2
    assert not run.snap.exists()


def test_env_knobs(run, capsys, monkeypatch):
    monkeypatch.setenv("SNAP_WORKERS", "x")
    assert run(["capture", str(run.snap), "0-9", "--commit", BASE], capsys)[0] == 2
    monkeypatch.setenv("SNAP_WORKERS", "3")
    monkeypatch.setenv("SNAP_GATE_GB", "176")
    rc, lines = run(["capture", str(run.snap), "0-9", "--commit", BASE], capsys)
    assert rc == 0 and "capture: up to 3 parts at once, memory gate 176 GB" in lines
    monkeypatch.delenv("SNAP_WORKERS")
    rc, lines = run(["capture", str(run.snap), "0-9", "--commit", BASE], capsys)
    assert rc == 0 and not any("parts at once" in ln for ln in lines)  # serial by default


def test_env_knob_errors_name_the_variable(run, capsys, monkeypatch):
    monkeypatch.setenv("SNAP_GATE_GB", "lots")
    rc, lines = run(["capture", str(run.snap), "0-9", "--commit", BASE], capsys)
    assert rc == 2 and "capture: usage error: env SNAP_GATE_GB='lots': not a whole number" in lines
    monkeypatch.setenv("SNAP_GATE_GB", "88")
    monkeypatch.setenv("SNAP_WORKERS", "0")
    rc, lines = run(["capture", str(run.snap), "0-9", "--commit", BASE], capsys)
    assert rc == 2 and "capture: usage error: SNAP_WORKERS 0: must be at least 1" in lines


def test_memory_over_the_gate_with_nothing_live_still_runs_every_part(pool):
    """Container memory counts page cache: at or over the gate with no part of the job live,
    the next part starts anyway; a ready part held by the gate prints "gate wait" about once a
    minute."""
    parts = ["0-9.step", "10-19.step", "20-29.step"]
    said, clock = [], Clock()
    p, fake, _ = pool(parts, workers=4, gate_gb=88, memory=lambda: 100 * 10**9, clock=clock,
                      fake=Fake(clock, took=lambda q: 150.0),
                      say=lambda s: said.append((clock(), s)))  # fmt: skip
    for part in parts:
        assert p.result(part)[0] == 0
    assert p.peak_live == 1  # one at a time: beside a live part 100 + 13 + 13 > 88
    starts = sorted(fake.starts.values())
    assert all(b >= a + 150.0 for a, b in itertools.pairwise(starts))
    assert 4 <= len(said) <= 7, said
    times = [t for t, _ in said]
    assert all(b - a >= 60.0 for a, b in itertools.pairwise(times))
    assert all(re.fullmatch(r"gate wait \S+: \d+ ready, 1 live, memory 100 GB, gate 88 GB", s)
               for _, s in said)  # fmt: skip


def test_term_between_a_launch_and_its_record_still_kills_that_child(pool):
    """The TERM handler only flags the pool: a TERM that lands while a child is being launched
    leaves it recorded, the next result() raises SystemExit(143) and close() kills every
    child."""
    import signal

    clock = Clock()
    inner = Fake(clock, took=lambda q: 1000.0)

    def launch(cmd, out, err):
        proc = inner(cmd, out, err)
        if len(inner.procs) == 2:
            signal.raise_signal(signal.SIGTERM)  # handled before the pool records the child
        return proc

    p, _, _ = pool(["roark.rank", "roark.trace", "0-9.step"], workers=3, spacing=0, clock=clock,
                   fake=launch)  # fmt: skip
    undo = sc._stop_on_term(p)
    try:
        with pytest.raises(SystemExit) as e:
            try:
                p.result("0-9.step")
            finally:
                p.close()
    finally:
        undo()
    assert e.value.code == 143
    assert len(inner.procs) == 2 and all(q.killed for q in inner.procs.values())
    assert not p.live and not p.tmp.exists()


def test_parent_exits_143_on_term_and_kills_its_children(run, capsys, monkeypatch):
    import signal

    procs = []
    real = sc.PartPool

    def fake_pool(mode, directory, parts, workers, gate_gb):
        clock = Clock()
        inner = Fake(clock, took=lambda q: 1000.0)

        def launch(cmd, out, err):
            proc = inner(cmd, out, err)
            procs.append(proc)
            if len(procs) == 3:
                signal.raise_signal(signal.SIGTERM)
            return proc

        return real(mode, directory, parts, workers, gate_gb, launch=launch, clock=clock,
                    sleep=clock.sleep, memory=lambda: None, rss=lambda pid: 0,
                    tmp=run.snap.parent)  # fmt: skip

    old = signal.getsignal(signal.SIGTERM)
    args = sc.build_parser().parse_args(["capture", str(run.snap), "0-9", "--commit", BASE,
                                         "--workers", "4"])  # fmt: skip
    monkeypatch.setattr(sc, "PartPool", fake_pool)  # the run fixture patched ROOT, git and hash
    with pytest.raises(SystemExit) as e:
        sc.parent(args)
    assert e.value.code == 143
    assert len(procs) == 3 and all(q.killed for q in procs)
    assert signal.getsignal(signal.SIGTERM) is old  # the handler is undone
    assert not list(run.snap.parent.glob("snap_parts_*"))


# A driver process: a PartPool of two real children that sleep 120 s, each pid appended to
# argv[2]; argv: <checkout> <pid file> <temp folder>.
TERM_DRIVER = """
import subprocess, sys
root, pids, tmp = sys.argv[1:4]
sys.path[:0] = [root, root + "/src"]
from tools import snap_compare as sc
def launch(cmd, out, err):
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdout=out,
                         stderr=err)
    with open(pids, "a") as f:
        f.write(str(p.pid) + " ")
    return p
pool = sc.PartPool("compare", "snap", ["roark.rank", "roark.trace"], 2, 88, launch=launch,
                   memory=lambda: None, spacing=0, poll=0.05, tmp=tmp)
sc._stop_on_term(pool)
try:
    pool.result("roark.trace")
finally:
    pool.close()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="a real SIGTERM needs POSIX")
def test_a_real_term_kills_the_real_children(tmp_path):
    """A driver process with two real sleeping children gets SIGTERM: it exits 143 and both
    children are gone."""
    import os
    import signal
    import time

    from tests.tools.test_pod_gate import ROOT

    pids = tmp_path / "pids"
    proc = subprocess.Popen(
        [sys.executable, "-c", TERM_DRIVER, str(ROOT), str(pids), str(tmp_path)]
    )
    deadline = time.monotonic() + 30
    while (
        time.monotonic() < deadline and len(pids.read_text().split() if pids.exists() else []) < 2
    ):
        time.sleep(0.1)
    kids = [int(x) for x in pids.read_text().split()]
    assert len(kids) == 2
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 143
    for k in kids:
        with pytest.raises(ProcessLookupError):
            os.kill(k, 0)
    assert not list(tmp_path.glob("snap_parts_*"))


# --- tools/ops/day/snap.sh: the generated run.sh ------------------------------------------


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@localhost",
                    "-c", "core.autocrlf=false", *args], check=True, capture_output=True)  # fmt: skip


OLD_SC = 'p.add_argument("--dirty-ok", action="store_true")\n'
# The flag on a line of its own, as a formatter may split the call: still detected.
NEW_SC = OLD_SC + 'p.add_argument(\n    "--workers", type=int, help="parts at once"\n)\n'


@pytest.fixture
def snap_dry(tmp_path):
    """``snap_dry(mode, commit, *args, **env) -> (CompletedProcess, run.sh text)``: ``snap.sh
    <mode> <commit> <tables> <args> --dry-run`` in a throwaway checkout whose HEAD's
    snap_compare.py has ``--workers`` and HEAD~1's has not (``snap_dry.new`` / ``.old``)."""
    import tarfile

    from tests.tools.test_pod_gate import BASH, ROOT, _clean_env

    if BASH is None:
        pytest.skip("no POSIX bash")
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    for rel in ("tools/ops/common.sh", "tools/ops/day/snap.sh", "tools/ops/day/pod/gate.sh"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((ROOT / rel).read_bytes().replace(b"\r", b""))
    (repo / ".gitignore").write_text("external/\n", encoding="utf-8", newline="\n")
    sc_file = repo / "tools" / "snap_compare.py"
    sc_file.write_text(OLD_SC, encoding="utf-8", newline="\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "old")
    sc_file.write_text(NEW_SC, encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "new")
    dec = repo / "external" / "pokeplatinum"
    dec.mkdir(parents=True)
    (dec / "a").write_text("a\n")
    _git(dec, "init", "-q")
    _git(dec, "add", "-A")
    _git(dec, "commit", "-qm", "decomp")
    data = tmp_path / "data" / "gen4_platinum"
    data.mkdir(parents=True)
    (data / "x.json").write_text("{}\n")
    tables = tmp_path / "tables.tar.xz"
    with tarfile.open(tables, "w:xz") as t:
        t.add(data / "x.json", arcname="data/gen4_platinum/x.json")
    scratch.mkdir()
    env = {k: v for k, v in _clean_env().items() if not k.startswith("SNAP_")}
    env.update(NUZLINE_REPO=str(repo), NUZLINE_SCRATCH=str(scratch))

    def sha(ref):
        return subprocess.run(["git", "-C", str(repo), "rev-parse", ref], check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    def go(mode, commit, *rest, **knobs):
        name = rest[1] if mode == "compare" else (rest[0] if rest else f"snapmain-{commit[:7]}")
        argv = [BASH, "tools/ops/day/snap.sh", mode, commit, tables.as_posix(), *rest, "--dry-run"]
        r = subprocess.run(argv, cwd=repo, env={**env, **knobs}, capture_output=True, text=True,
                           check=False)  # fmt: skip
        out = scratch / "rent" / "out" / name / "run.sh"
        return r, out.read_text(encoding="utf-8") if out.exists() else ""

    go.old, go.new, go.bash = sha("HEAD~1"), sha("HEAD"), BASH
    return go


def test_snap_sh_capture_run_sh_passes_workers_gate_and_takes_the_lock(snap_dry):
    r, run_sh = snap_dry("capture", snap_dry.new)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NOTE:" not in r.stdout
    cmd = (f'uv run --frozen python "tools/snap_compare.py" capture "$W/snap" 0-199 --repo '
           f'"$W/repo" --commit {snap_dry.new} --workers 16 --gate-gb 88 -v > "$W/capture.log" '
           f"2>&1 & wait $!; ")  # fmt: skip
    assert cmd in run_sh
    lines = run_sh.splitlines()
    assert any(ln.startswith(f'W="/work/snapmain-{snap_dry.new[:7]}";') for ln in lines)
    assert any(ln.startswith("m=$(tr -dc 0-9 < /proc/sys/vm/max_map_count") and
               'echo "MAP_COUNT_LOW ${m:-none} $(date +%s)" >> "$W/times.txt"; exit 1;' in ln
               for ln in lines)  # fmt: skip
    assert "L=/work/.snaplock" in lines
    assert "trap 'snapunlock \"$L\"' EXIT" in lines
    assert any(ln.startswith("trap 'trap \"\" TERM; kill -TERM -- -$$")
               and ln.endswith("exit 143' TERM") for ln in lines)  # fmt: skip
    for fn in ("cgmem", "snaplive", "snaplock", "snapunlock"):
        assert f"{fn} () " in run_sh
    order = ["MAP_COUNT_LOW", "SYNC_RC=", 'snaplock "$L" "$W/times.txt"; echo "SNAP_LOCK',
             'gate; echo "CAPTURE_START', "CAPTURE_RC=", "echo ALL_DONE"]  # fmt: skip
    at = [run_sh.index(k) for k in order]
    assert at == sorted(at)


def test_snap_sh_compare_run_sh_quotes_and_knobs(snap_dry):
    cap = "/work/snapmain-abc1234/snap"
    r, run_sh = snap_dry("compare", snap_dry.new, cap, "snapcmp-e1-x", "--allow-new", "a,b.c",
                         SNAP_WORKERS="4", SNAP_GATE_GB="176")  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert (f'compare "{cap}" 0-199 --repo "$W/repo" --commit {snap_dry.new} --allow-new a,b.c '
            f'--workers 4 --gate-gb 176 -v > "$W/compare.log" 2>&1 & wait $!;') in run_sh  # fmt: skip
    assert f'printf \'%s\\n\' "{cap}" > "$W/reads"' in run_sh
    assert "-ge 176 ]" in run_sh  # run.sh's own gate


def test_snap_sh_old_snap_compare_runs_serially_with_a_note(snap_dry):
    r, run_sh = snap_dry("capture", snap_dry.old)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"NOTE: tools/snap_compare.py at {snap_dry.old[:8]} has no --workers" in r.stdout
    assert "--workers" not in run_sh and "--gate-gb" not in run_sh
    assert "L=/work/.snaplock" in run_sh  # the lock does not depend on the tool


@pytest.mark.parametrize("knob", ["SNAP_WORKERS", "SNAP_GATE_GB"])
def test_snap_sh_refuses_a_bad_knob(snap_dry, knob):
    r, run_sh = snap_dry("capture", snap_dry.new, **{knob: "0"})
    assert r.returncode == 3 and f"REFUSED: {knob} must be a whole number" in r.stdout
    assert run_sh == ""


# --- the generated run.sh, run with a fake uv ---------------------------------------------


@pytest.fixture
def pod(tmp_path, snap_dry):
    """``pod.job(name, uv_s) -> Popen``: a capture run.sh from snap.sh, rewritten so its /work,
    max_map_count, cgroup and uv are under ``tmp_path`` (fake uv: ``sync`` passes, ``run``
    writes its pid to ``<job>/uv.pid`` and sleeps ``uv_s`` s). ``pod.lock``: the pod lock
    folder. Every job left is killed at teardown."""
    import os

    bash = snap_dry.bash
    work, fake = tmp_path / "work", tmp_path / "fakebin"
    fake.mkdir()
    uv = '#!/bin/bash\n[ "$1" = sync ] && exit 0\necho $$ > "$W/uv.pid"\nexec sleep "$FAKE_UV_S"\n'
    (fake / "uv").write_text(uv, encoding="utf-8", newline="\n")
    os.chmod(fake / "uv", 0o755)
    (tmp_path / "mc").write_text("1048576\n")
    (tmp_path / "cg").mkdir()
    (tmp_path / "cg" / "memory.current").write_text("0\n")
    procs = []

    def job(name, uv_s):
        r, run_sh = snap_dry("capture", snap_dry.new, name)
        assert r.returncode == 0, r.stdout + r.stderr
        # The paths go in as variables: the project path has spaces, and run.sh writes the
        # pod's space-free paths unquoted.
        run_sh = (run_sh.replace("/proc/sys/vm/max_map_count", '"${MC}"')
                  .replace("/root/.local/bin", "${FAKEBIN}").replace("/work/", "${POD}/"))  # fmt: skip
        d = work / name
        (d / "repo").mkdir(parents=True)
        (d / "run.sh").write_text(run_sh, encoding="utf-8", newline="\n")
        env = {**os.environ, "FAKE_UV_S": str(uv_s), "CGROOT": posix(tmp_path / "cg"),
               "W": posix(d), "POD": posix(work), "FAKEBIN": posix(fake),
               "MC": posix(tmp_path / "mc")}  # fmt: skip
        out = open(d / "run.out", "w")  # noqa: SIM115
        p = subprocess.Popen([bash, (d / "run.sh").as_posix()], stdout=out,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)  # fmt: skip
        p.dir = d
        procs.append((p, out))
        return p

    def sh(cmd):
        return subprocess.run([bash, "-c", cmd], check=False, capture_output=True, text=True)

    def posix(path):  # a path bash's PATH and redirections take (Git Bash: /g/...)
        if sys.platform != "win32":
            return str(path)
        return subprocess.run([bash, "-c", 'cygpath -u "$0"', str(path)], check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    job.job = job
    job.lock = work / ".snaplock"
    job.term = lambda pid: sh(f"kill -TERM {pid}")
    job.alive = lambda pid: sh(f"kill -0 {pid}").returncode == 0
    job.dead_pid = lambda: sh("echo $$").stdout.strip()
    yield job
    for p, out in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
        out.close()
    for p, _ in procs:  # a fake uv a failed test left behind
        pid = _text(p.dir / "uv.pid").strip()
        if pid:
            sh(f"kill -9 {pid}")


def _until(cond, s=30.0):
    import time

    end = time.monotonic() + s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def _text(path):
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def test_run_sh_one_job_per_pod_and_term_releases_the_lock_and_kills_the_job(pod):
    a = pod.job("snapt-a", 120)
    assert _until(lambda: _text(a.dir / "uv.pid").strip()), _text(a.dir / "run.out")
    a_uv = _text(a.dir / "uv.pid").strip()
    holder = _text(pod.lock / "pid").strip()
    assert _text(pod.lock / "name").strip().endswith("/snapt-a")
    b = pod.job("snapt-b", 120)
    assert _until(lambda: "lock wait" in _text(b.dir / "run.out")), _text(b.dir / "run.out")
    assert "SYNC_RC=0" in _text(b.dir / "times.txt")
    assert "SNAP_LOCK" not in _text(b.dir / "times.txt")  # waits while a's run.sh lives
    lines = _text(b.dir / "run.out").splitlines()
    assert len(lines) == 1  # one line a minute
    assert re.fullmatch(r"lock wait \S+: .+/snapt-a holds .+/\.snaplock", lines[0])
    pod.term(holder)
    assert a.wait(timeout=30) == 143
    assert _until(lambda: not pod.alive(a_uv), 10)  # the job a's run.sh started is gone
    times_a = _text(a.dir / "times.txt")
    assert "CAPTURE_START" in times_a
    assert "ALL_DONE" not in times_a and "CAPTURE_RC" not in times_a
    # b takes the lock once a's TERM released it, never judging it stale
    assert _until(lambda: _text(b.dir / "uv.pid").strip()), _text(b.dir / "run.out")
    assert _text(pod.lock / "name").strip().endswith("/snapt-b")
    assert "SNAP_LOCK_STALE" not in _text(b.dir / "times.txt")
    b_uv = _text(b.dir / "uv.pid").strip()
    pod.term(_text(pod.lock / "pid").strip())
    assert b.wait(timeout=30) == 143
    assert _until(lambda: not pod.alive(b_uv), 10)
    assert not pod.lock.exists()


def test_run_sh_removes_a_stale_lock_of_a_dead_pid(pod):
    dead = pod.dead_pid()
    pod.lock.mkdir(parents=True)
    (pod.lock / "pid").write_text(dead + "\n")
    (pod.lock / "name").write_text("/work/gone\n")
    a = pod.job("snapt-s", 0)
    assert a.wait(timeout=60) == 0, _text(a.dir / "run.out")
    times = _text(a.dir / "times.txt")
    assert f"SNAP_LOCK_STALE {dead} " in times and "CAPTURE_RC=0" in times
    assert times.rstrip().endswith("ALL_DONE")
    assert not pod.lock.exists()
