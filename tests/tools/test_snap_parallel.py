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
    monkeypatch.setattr(sc, "container_bytes", lambda: 0)  # a cgroup reading, as on the pod
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
        assert _gone(k), k
    assert not list(tmp_path.glob("snap_parts_*"))


def _gone(pid) -> bool:
    """run.sh's rule (procscan): a process is gone when /proc/<pid>/stat is missing or its state
    is Z. On the test pod PID 1 (sleep infinity) never reaps an orphan, so a killed orphan stays
    a zombie, which kill -0 still finds."""
    import os

    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True
    except OSError:  # no /proc: kill -0
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        return False


# Bash: exit 0 while process {} lives (not a zombie), the rule of run.sh's procscan. Git Bash's
# /proc/<pid>/stat has the same layout, by bash's pid numbers.
LIVE_SH = 'read -r s < /proc/{}/stat 2>/dev/null || exit 1; read -r st _ <<< "${{s##*) }}"; [ "$st" != Z ]'


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
    for rel in ("tools/ops/common.sh", "tools/ops/day/snap.sh", "tools/ops/day/snap_body.sh",
                "tools/ops/day/pod/gate.sh", "tools/ops/day/pod/cache_sync.sh"):  # fmt: skip
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
    (tmp_path / "tables.key").write_text("c" * 64 + "\n")  # build_tables.sh's key: a store key
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
           f'2>&1 &\nsnapwatch $! "$L"; ')  # fmt: skip
    assert cmd in run_sh
    lines = run_sh.splitlines()
    assert any(ln.startswith(f'W="/work/snapmain-{snap_dry.new[:7]}";') for ln in lines)
    assert any(ln.startswith("m=$(tr -dc 0-9 < /proc/sys/vm/max_map_count") and
               'echo "MAP_COUNT_LOW ${m:-none} $(date +%s)" >> "$W/times.txt"; exit 1;' in ln
               for ln in lines)  # fmt: skip
    assert "L=/work/.snaplock; SG=$$" in lines
    assert "trap 'snapunlock \"$L\"' EXIT" in lines
    assert ("trap 'trap \"\" TERM; echo \"TERM $(date +%s)\" >> \"$W/times.txt\"; snapstop "
            "\"$W/times.txt\" || trap - EXIT; exit 143' TERM") in lines  # fmt: skip
    for fn in ("cgused", "procscan", "snapmembers", "snaplive", "snaplock", "snapunlock",
               "sgprune", "snapwatch", "snapstop", "snapgate", "reserve_knobs", "podgate"):  # fmt: skip
        assert f"{fn} () " in run_sh
    order = ["MAP_COUNT_LOW", "SYNC_RC=", 'snaplock "$L" "$W/times.txt"; echo "SNAP_LOCK',
             "snap-fetch", 'snapgate 88 "$W/times.txt"; echo "CAPTURE_START', "CAPTURE_RC=$rc",
             'snapstop "$W/times.txt" || { trap - EXIT;', "snap-publish", "\necho ALL_DONE"]  # fmt: skip
    at = [run_sh.index(k) for k in order]
    assert at == sorted(at)


def test_snap_sh_compare_run_sh_quotes_and_knobs(snap_dry):
    cap = "/work/snapmain-abc1234/snap"
    r, run_sh = snap_dry("compare", snap_dry.new, cap, "snapcmp-e1-x", "--allow-new", "a,b.c",
                         SNAP_WORKERS="4", SNAP_GATE_GB="176")  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert (f'compare "{cap}" 0-199 --repo "$W/repo" --commit {snap_dry.new} --allow-new a,b.c '
            f'--workers 4 --gate-gb 176 -v > "$W/compare.log" 2>&1 &\n') in run_sh  # fmt: skip
    assert f'printf \'%s\\n\' "{cap}" > "$W/reads"' in run_sh
    assert 'snapgate 176 "$W/times.txt"' in run_sh  # run.sh's own gate


def test_snap_sh_old_snap_compare_runs_serially_with_a_note(snap_dry):
    r, run_sh = snap_dry("capture", snap_dry.old)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"NOTE: tools/snap_compare.py at {snap_dry.old[:8]} has no --workers" in r.stdout
    assert "--workers" not in run_sh and "--gate-gb" not in run_sh
    assert "L=/work/.snaplock" in run_sh  # the lock does not depend on the tool


def _snap_body(snap_dry, mode):
    """The dry-run exec-pod body (argv) and run.sh of a capture or a compare."""
    import json

    if mode == "capture":
        r, run_sh = snap_dry("capture", snap_dry.new, f"snapt-body-{mode}")
        name = f"snapt-body-{mode}"
    else:
        name = f"snapcmp-body-{mode}"
        r, run_sh = snap_dry("compare", snap_dry.new, "/work/snapmain-abc1234/snap", name)
    assert r.returncode == 0, r.stdout + r.stderr
    body = r.stdout[r.stdout.index("== exec-pod body") :].splitlines()[1]
    return json.loads(body)["command"], run_sh, name


@pytest.mark.parametrize("mode", ["capture", "compare"])
def test_snap_sh_body_stays_within_exec_pod_limits(snap_dry, mode):
    """exec-pod refuses an argv item over 4096 characters and an argv over 256 items (road test:
    a 15,124-character run.sh item was refused with 422)."""
    import base64

    cmd, run_sh, _ = _snap_body(snap_dry, mode)
    assert len(cmd) <= 256
    assert max(len(x) for x in cmd) <= 4096
    assert cmd[:2] == ["sh", "-c"] and cmd[3] == "_"
    assert len(run_sh) > 4096  # the body really is split: run.sh alone is over one item
    assert base64.b64decode("".join(cmd[4:])).decode() == run_sh


@pytest.mark.parametrize("mode", ["capture", "compare"])
def test_snap_sh_body_writes_run_sh_and_refuses_a_changed_one(snap_dry, tmp_path, mode):
    """The body's script joins the chunks, decodes run.sh and goes on only when its sha256 is the
    PC's; a changed chunk ends RUNSH_SHA_MISMATCH, exit 1, run.sh removed, nothing started."""
    import hashlib

    from tests.tools.test_pod_gate import _clean_env, _posix_shells

    cmd, run_sh, name = _snap_body(snap_dry, mode)
    want = hashlib.sha256(run_sh.encode()).hexdigest()
    assert f"= {want} ]" in cmd[2]
    # Only the part before the start: the rest receives the payload and starts run.sh.
    head = cmd[2][: cmd[2].index("chmod +x run.sh")] + "echo RUNSH_OK"
    for i, sh in enumerate(_posix_shells()):
        for bad in (False, True):
            w = tmp_path / f"{mode}{i}{int(bad)}"
            script = head.replace(f"/work/{name}", f"'{w.as_posix()}'")
            chunks = list(cmd[4:])
            if bad:  # a valid base64 chunk with other bytes
                chunks[0] = "QUJD" + chunks[0][4:]
            r = subprocess.run([*sh, script, "_", *chunks], capture_output=True, text=True, env=_clean_env(),
                               check=False)  # fmt: skip
            if bad:
                assert r.returncode == 1 and "RUNSH_SHA_MISMATCH" in r.stdout, (sh, r.stdout)
                assert not (w / "run.sh").exists()
            else:
                assert r.returncode == 0 and r.stdout.strip() == "RUNSH_OK", (sh, r.stderr)
                assert (w / "run.sh").read_bytes() == run_sh.encode()


def _body_fn(tmp_path, call):
    """<call> in bash after sourcing tools/ops/day/snap_body.sh, in <tmp_path>."""
    from tests.tools.test_pod_gate import BASH, ROOT, _clean_env

    if BASH is None:
        pytest.skip("no POSIX bash")
    body = (ROOT / "tools" / "ops" / "day" / "snap_body.sh").as_posix()
    return subprocess.run([BASH, "-c", f". '{body}'; cd '{tmp_path.as_posix()}' && {call}"],
                          capture_output=True, text=True, env=_clean_env(), check=False)  # fmt: skip


def test_snap_body_refuses_a_run_sh_over_256_argv_items(tmp_path):
    """252 chunks of 4000 base64 characters (756,000 bytes) is the most run.sh may be."""
    (tmp_path / "ok.sh").write_bytes(b"x" * 756000)
    (tmp_path / "big.sh").write_bytes(b"x" * 756001)
    r = _body_fn(tmp_path, 'body_chunks ok.sh && echo "N ${#CH[@]}"')
    assert r.returncode == 0 and r.stdout.strip() == "N 252", r.stdout + r.stderr
    r = _body_fn(tmp_path, "body_chunks big.sh; echo rc $?")
    assert r.stdout.splitlines() == ["REFUSED: exec body would have 257 argv items, over 256 (run.sh too large)",
                                     "rc 3"], r.stdout + r.stderr  # fmt: skip


def test_snap_body_refuses_a_script_over_4000_characters(tmp_path):
    import json

    (tmp_path / "r.sh").write_bytes(b"echo hi\n")
    ok, big = "a" * 3998 + '"\\', "a" * 4001
    r = _body_fn(tmp_path, f"body_chunks r.sh && body_json '{ok}'")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["command"] == ["sh", "-c", ok, "_", "ZWNobyBoaQo="]
    r = _body_fn(tmp_path, f"body_chunks r.sh && body_json '{big}'; echo rc $?")
    assert r.stdout.splitlines() == ["REFUSED: exec body script is 4001 characters, over 4000", "rc 3"], r.stdout


@pytest.mark.parametrize("knob", ["SNAP_WORKERS", "SNAP_GATE_GB"])
def test_snap_sh_refuses_a_bad_knob(snap_dry, knob):
    r, run_sh = snap_dry("capture", snap_dry.new, **{knob: "0"})
    assert r.returncode == 3 and f"REFUSED: {knob} must be a whole number" in r.stdout
    assert run_sh == ""


def test_workers_need_a_cgroup_memory_reading(run, capsys, monkeypatch):
    """Off Linux (no cgroup file) --workers > 1 would admit parts blind: a usage error."""
    monkeypatch.setattr(sc, "container_bytes", lambda: None)
    for argv in (["--workers", "3"], []):
        if not argv:
            monkeypatch.setenv("SNAP_WORKERS", "4")
        rc, lines = run(["capture", str(run.snap), "0-9", "--commit", BASE, *argv], capsys)
        src = "--workers 3" if argv else "SNAP_WORKERS 4"
        assert rc == 2 and any(ln.startswith(f"capture: usage error: {src}: parts at once need "
                                             "the container memory reading") for ln in lines)  # fmt: skip
        assert not run.snap.exists()
    rc, _ = run(["capture", str(run.snap), "0-9", "--commit", BASE, "--workers", "1"], capsys)
    assert rc == 0  # one at a time needs no reading


# --- tools/ops/day/snap.sh: the generated run.sh's functions -------------------------------

GB = 10**9
# cgroup trees: {file: text}, and the bytes both readers give (None: no reading).
CG_TREES = {
    "v2": ({"memory.current": f"{200 * GB}\n",
            "memory.stat": f"anon 1\nactive_file 5\ninactive_file {150 * GB}\nfile 9\n"}, 50 * GB),
    "v2-no-stat": ({"memory.current": f"{200 * GB}\n"}, 200 * GB),
    "v2-stat-over": ({"memory.current": "100\n", "memory.stat": "inactive_file 500\n"}, 0),
    "v1": ({"memory/memory.usage_in_bytes": f"{90 * GB}\n",
            "memory/memory.stat": f"inactive_file 7\ntotal_inactive_file {30 * GB}\n"}, 60 * GB),
    "v1-no-stat": ({"memory/memory.usage_in_bytes": "1234\n"}, 1234),
    "v2-garbage": ({"memory.current": "max\n", "memory/memory.usage_in_bytes": "77\n"}, 77),
    "missing": ({}, None),
}  # fmt: skip


@pytest.fixture
def fns(tmp_path, snap_dry):
    """The function definitions of a dry-run capture run.sh, in a file to source."""
    r, run_sh = snap_dry("capture", snap_dry.new, "snapt-fns")
    assert r.returncode == 0, r.stdout + r.stderr
    body = run_sh[run_sh.index("cgused () ") : run_sh.index("L=/work/.snaplock")]
    f = tmp_path / "fns.sh"
    f.write_text(body, encoding="utf-8", newline="\n")
    return f


def _bash(cmd, timeout=60, **env):
    from tests.tools.test_pod_gate import BASH, _clean_env

    return subprocess.run([BASH, "-c", cmd], check=False, capture_output=True, text=True,
                          timeout=timeout, env={**_clean_env(), **env})  # fmt: skip


@pytest.mark.parametrize("tree", sorted(CG_TREES))
def test_cgused_and_container_bytes_subtract_inactive_file_alike(tmp_path, monkeypatch, fns, tree):
    files, want = CG_TREES[tree]
    root = tmp_path / "cg"
    root.mkdir()
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8", newline="\n")
    monkeypatch.setenv("CGROOT", str(root))
    assert sc.container_bytes() == want
    r = _bash(f". '{fns.as_posix()}'; cgused", CGROOT=root.as_posix())
    if want is None:
        assert (r.returncode, r.stdout) == (1, "")
    else:
        assert (r.returncode, r.stdout.strip()) == (0, str(want))


def test_snaplock_treats_its_own_pid_in_the_lock_as_stale(tmp_path, fns):
    """A lock whose pid is the waiting shell's own is another holder's pid, reused: stale. The
    shell runs a script named run.sh, so its own pid would pass snaplive's run.sh test."""
    lock, times, script = tmp_path / "lock", tmp_path / "times.txt", tmp_path / "j" / "run.sh"
    script.parent.mkdir()
    script.write_text(f". '{fns.as_posix()}'; W=/work/me; d='{lock.as_posix()}'; mkdir -p \"$d\"; "
                      f"echo $$ > \"$d/pid\"; echo /work/old > \"$d/name\"; snaplock \"$d\" "
                      f"'{times.as_posix()}'; echo \"$$ $(cat \"$d/pid\") $(cat \"$d/name\")\"\n",
                      encoding="utf-8", newline="\n")  # fmt: skip
    r = _bash(f"bash '{script.as_posix()}'")
    assert r.returncode == 0, r.stderr
    me, held, name = r.stdout.split()
    assert held == me and name == "/work/me"
    assert re.fullmatch(rf"SNAP_LOCK_STALE {me} \d+\n", _text(times))


def test_snaplock_removes_a_lock_without_a_pid_file_after_10_s(tmp_path, fns):
    import time

    lock, times = tmp_path / "lock", tmp_path / "times.txt"
    lock.mkdir()
    (lock / "name").write_text("/work/died\n")
    t0 = time.monotonic()
    r = _bash(f". '{fns.as_posix()}'; W=/work/me; snaplock '{lock.as_posix()}' "
              f"'{times.as_posix()}' > /dev/null; cat '{lock.as_posix()}/name'")  # fmt: skip
    took = time.monotonic() - t0
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "/work/me"
    assert re.fullmatch(r"SNAP_LOCK_STALE none \d+\n", _text(times))
    assert 9 <= took < 40, took


def test_snaplive_counts_a_dead_leader_s_live_group_and_the_groups_file(tmp_path, fns):
    """A holder pid that is gone still holds while a process of its group, or of a group in
    <lock>/groups, is live; a dead pid with nothing of its groups live does not."""
    from tests.tools.test_pod_gate import BASH, _clean_env

    p = subprocess.Popen([BASH, "-c", "sleep 60 > /dev/null 2>&1 < /dev/null & echo $$ $!"],
                         stdout=subprocess.PIPE, text=True, env=_clean_env(),
                         start_new_session=True)  # fmt: skip
    leader, kid = p.stdout.read().split()
    p.wait()
    lock = tmp_path / "lock"
    lock.mkdir()
    live = f". '{fns.as_posix()}'; snaplive {{}} '{lock.as_posix()}' && echo LIVE || echo GONE"
    dead = _bash("echo $$").stdout.strip()
    try:
        assert _until(
            lambda: _bash(LIVE_SH.format(leader)).returncode != 0, 10
        )
        assert _bash(live.format(leader)).stdout.strip() == "LIVE"
        assert _bash(live.format(dead)).stdout.strip() == "GONE"
        (lock / "groups").write_text(f"{dead} {leader}\n")  # the job's groups, snapwatch's file
        assert _bash(live.format(dead)).stdout.strip() == "LIVE"
    finally:
        _bash(f"kill -9 {kid}")
    assert _until(lambda: _bash(live.format(leader)).stdout.strip() == "GONE", 10)
    assert _bash(live.format(dead)).stdout.strip() == "GONE"


def _own_group_bash(script):
    """Run ``script`` in a bash that leads its own process group (as run.sh under setsid)."""
    from tests.tools.test_pod_gate import BASH, _clean_env

    return subprocess.run([BASH, "-c", 'read -r s < /proc/$$/stat; read -r _ _ pg _ <<< "${s##*) }"; '
                           '[ "$pg" = "$$" ] || { echo NOT_LEADER; exit 0; }; ' + script],
                          check=False, capture_output=True, text=True, timeout=60, env=_clean_env(),
                          start_new_session=True)  # fmt: skip


def test_snaplive_never_counts_the_waiting_shell_s_own_group(tmp_path, fns):
    """A dead holder whose groups file names the waiter's own group id (a reused id): the waiter
    is a member of that group, but it does not wait on itself."""
    lock = tmp_path / "lock"
    lock.mkdir()
    dead = _bash("echo $$").stdout.strip()
    (lock / "pid").write_text(dead + "\n")
    r = _own_group_bash(f". '{fns.as_posix()}'; echo $$ > '{lock.as_posix()}/groups'; "
                        f"snaplive {dead} '{lock.as_posix()}' && echo LIVE || echo GONE")  # fmt: skip
    assert r.stdout.strip() == "GONE", r.stdout + r.stderr


def _group_leader(s=60):
    """A live sleep leading a process group of its own: (Popen, its pid as bash numbers it)."""
    from tests.tools.test_pod_gate import BASH, _clean_env

    p = subprocess.Popen([BASH, "-c", "echo $$; exec sleep %d" % s], stdout=subprocess.PIPE,
                         text=True, env=_clean_env(), start_new_session=True)  # fmt: skip
    return p, p.stdout.readline().strip()


def test_sgprune_drops_a_group_once_it_has_no_member(tmp_path, fns):
    p, g = _group_leader()
    try:
        keep = _own_group_bash(f". '{fns.as_posix()}'; SG=\"$$ {g}\"; procscan; sgprune; echo \"$SG\"")
        assert keep.stdout.split()[1:] == [g], keep.stdout + keep.stderr
    finally:
        _bash(f"kill -9 {g}")  # by bash's pid (Git Bash: Popen.kill may miss the exec'd sleep)
        p.kill()
        p.wait()
    assert _until(lambda: _bash(LIVE_SH.format(g)).returncode != 0, 10)
    gone = _own_group_bash(f". '{fns.as_posix()}'; SG=\"$$ {g}\"; procscan; sgprune; echo \"$$ $SG\"")
    me, *sg = gone.stdout.split()
    assert sg == [me], gone.stdout + gone.stderr


def test_snapwatch_drops_a_dead_group_from_sg_and_the_groups_file(tmp_path, fns):
    """snapwatch re-checks SG on every scan: a job group with no member left leaves SG and the
    lock's groups file, so snapstop cannot later TERM a process that reused its id."""
    p, g = _group_leader()
    _bash(f"kill -9 {g}")
    p.kill()
    p.wait()
    assert _until(lambda: _bash(LIVE_SH.format(g)).returncode != 0, 10)
    lock = tmp_path / "lock"
    lock.mkdir()
    # SG starts as the shell's pid, as in run.sh, which leads its own group under setsid.
    r = _own_group_bash(f". '{fns.as_posix()}'; L='{lock.as_posix()}'; echo $$ > \"$L/pid\"; "
                        f"SG=\"$$ {g}\"; sleep 3 & snapwatch $! \"$L\"; "
                        f"echo \"rc=$? $$|$SG|$(cat \"$L/groups\")\"")  # fmt: skip
    head, sg, groups = r.stdout.strip().split("|")
    rc, me = head.split()
    assert (rc, sg.split(), groups.split()) == ("rc=0", [me], [me]), r.stdout + r.stderr


# --- the generated run.sh, run with a fake uv ---------------------------------------------

# Git Bash's ps has no -o: the columns podgate reads (pid pgid stat rss) from ps -l, whose first
# column may be a status letter.
PS_SHIM = """#!/bin/bash
/usr/bin/ps -l | awk 'NR > 1 { if ($1 !~ /^[0-9]+$/) { $1 = ""; $0 = $0 }; print $1, $3, "S", 0 }'
"""
# Fake uv: sync passes; run writes its pid to <job>/uv.pid, and then: with FAKE_UV_LEAVE starts a
# sleep it leaves behind (pid in <job>/left.pid; FAKE_UV_LEAVE=stuck: one that ignores TERM) and
# exits 9; with FAKE_UV_SETSID becomes a
# sleep in a session of its own; else a sleep of FAKE_UV_S s. FAKE_UV_CAPTURE first writes a
# complete capture (snap/capture.json, every part ok, each with its two files).
FAKE_UV = """#!/bin/bash
[ "$1" = sync ] && exit 0
echo $$ > "$W/uv.pid"
if [ -n "$FAKE_UV_CAPTURE" ]; then mkdir -p "$W/snap"; echo '{"parts": {"p1": "ok"}}' > "$W/snap/capture.json"
  echo states > "$W/snap/p1.pkl.gz"; echo "0  p1" > "$W/snap/p1.sha256"; fi
[ "$FAKE_UV_LEAVE" = stuck ] && trap '' TERM
if [ -n "$FAKE_UV_LEAVE" ]; then sleep 120 > /dev/null 2>&1 < /dev/null & echo $! > "$W/left.pid"; exit 9; fi
[ -n "$FAKE_UV_SETSID" ] && exec setsid sleep "$FAKE_UV_S"
exec sleep "$FAKE_UV_S"
"""


@pytest.fixture
def pod(tmp_path, snap_dry):
    """``pod.job(name, uv_s, **env) -> Popen``: a capture run.sh from snap.sh, rewritten so its
    /work, max_map_count, cgroup and uv are under ``tmp_path`` (FAKE_UV), with IDLE_WAIT_S 300
    unless ``env`` sets it. ``pod.mem(current, inactive)``: the fake cgroup v2 reading.
    ``pod.lock``: the pod lock folder; ``pod.work``: the fake /work. ``job(..., vol=<dir>)``: a
    pod whose network volume is <dir> (cache_sync.sh in the job folder, as the payload puts it;
    run.sh's mount test always true). ``job(..., nokill=True)``: snapstop's KILL does nothing (a
    process the kernel does not let die). Every job and pid left is killed at teardown."""
    import os

    from tests.tools.test_pod_gate import ROOT

    bash = snap_dry.bash
    work, fake = tmp_path / "work", tmp_path / "fakebin"
    fake.mkdir()
    (fake / "uv").write_text(FAKE_UV, encoding="utf-8", newline="\n")
    os.chmod(fake / "uv", 0o755)
    if subprocess.run([bash, "-c", "ps -eo pid=,pgid=,stat=,rss= > /dev/null 2>&1"],
                      check=False).returncode != 0:  # fmt: skip
        (fake / "ps").write_text(PS_SHIM, encoding="utf-8", newline="\n")
        os.chmod(fake / "ps", 0o755)
    (tmp_path / "mc").write_text("1048576\n")
    cg = tmp_path / "cg"
    cg.mkdir()

    def mem(current, inactive=None):
        (cg / "memory.current").write_text(f"{current}\n")
        if inactive is not None:
            (cg / "memory.stat").write_text(f"active_file 1\ninactive_file {inactive}\n")

    mem(0)
    procs, pids = [], []

    def job(name, uv_s, vol=None, cap=None, sync=None, nokill=False, **extra):
        if cap is None:
            r, run_sh = snap_dry("capture", snap_dry.new, name)
        else:  # a compare of <cap>, a /work/snapmain-<short>/snap
            r, run_sh = snap_dry("compare", snap_dry.new, cap, name)
        assert r.returncode == 0, r.stdout + r.stderr
        if sync is not None:  # a stand-in cache_sync.sh; every volume call times out after 2 s
            run_sh = run_sh.replace("timeout -k 30 900 ", "timeout -k 2 2 ")
        if nokill:
            assert run_sh.count("kill -KILL $SM") == 1
            run_sh = run_sh.replace("kill -KILL $SM", ": KILL $SM")
        # Never the real /vol of the pod the tests run on.
        run_sh = run_sh.replace("mountpoint -q /vol 2>/dev/null", "false" if vol is None else "true")
        if vol is not None:
            run_sh = (run_sh
                      .replace('export CACHE_PY="$W/repo/.venv/bin/python"', 'export CACHE_PY="$TEST_PY"'))  # fmt: skip
            extra = {"CACHE_VOL": posix(vol), "CACHE_VOL_MOUNTCHECK": "0", "TEST_PY": sys.executable,
                     "CACHE_POD": "t", **extra}  # fmt: skip
        # The paths go in as variables: the project path has spaces, and run.sh writes the
        # pod's space-free paths unquoted.
        run_sh = (run_sh.replace("/proc/sys/vm/max_map_count", '"${MC}"')
                  .replace("/root/.local/bin", "${FAKEBIN}").replace("/work/", "${POD}/"))  # fmt: skip
        d = work / name
        (d / "repo").mkdir(parents=True)
        (d / "run.sh").write_text(run_sh, encoding="utf-8", newline="\n")
        if vol is not None:
            src = ROOT / "tools/ops/day/pod/cache_sync.sh"
            (d / "cache_sync.sh").write_bytes(src.read_bytes().replace(b"\r", b"") if sync is None
                                              else sync.encode())  # fmt: skip
        env = {**os.environ, "FAKE_UV_S": str(uv_s), "CGROOT": posix(cg), "W": posix(d),
               "POD": posix(work), "FAKEBIN": posix(fake), "MC": posix(tmp_path / "mc"),
               "IDLE_WAIT_S": "300", **extra}  # fmt: skip
        out = open(d / "run.out", "w")  # noqa: SIM115
        p = subprocess.Popen([bash, (d / "run.sh").as_posix()], stdout=out,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)  # fmt: skip
        p.dir = d
        procs.append((p, out))
        return p

    def sh(cmd):
        return subprocess.run([bash, "-c", cmd], check=False, capture_output=True, text=True,
                              timeout=60)  # fmt: skip

    def posix(path):  # a path bash's PATH and redirections take (Git Bash: /g/...)
        if sys.platform != "win32":
            return str(path)
        return subprocess.run([bash, "-c", 'cygpath -u "$0"', str(path)], check=True,
                              capture_output=True, text=True).stdout.strip()  # fmt: skip

    def sleeper(s=120):  # a live process's pid, as bash numbers it
        pid = sh(f"sleep {s} > /dev/null 2>&1 < /dev/null & echo $!").stdout.strip()
        pids.append(pid)
        return pid

    job.job, job.mem, job.sleeper, job.sh = job, mem, sleeper, sh
    job.key = f"{snap_dry.new}-{'c' * 16}"  # snap_dry's store key (its tables.key)
    job.work, job.lock = work, work / ".snaplock"
    job.term = lambda pid: sh(f"kill -TERM {pid}")
    job.kill = lambda pid: sh(f"kill -KILL {pid}")
    job.alive = lambda pid: sh(LIVE_SH.format(pid)).returncode == 0  # a zombie is gone
    job.dead_pid = lambda: sh("echo $$").stdout.strip()
    yield job
    for p, out in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
        out.close()
    for p, _ in procs:  # a fake uv or left sleep a failed test left behind
        for f in ("uv.pid", "left.pid"):
            pid = _text(p.dir / f).strip()
            if pid:
                sh(f"kill -9 {pid}")
    for pid in pids:
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


def _setsid() -> bool:
    import shutil

    from tests.tools.test_pod_gate import BASH

    return BASH is not None and shutil.which("setsid") is not None and sys.platform != "win32"


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
    assert a.wait(timeout=60) == 143
    assert _until(lambda: not pod.alive(a_uv), 10)  # the job a's run.sh started is gone
    times_a = _text(a.dir / "times.txt").splitlines()
    assert any(ln.startswith("CAPTURE_START") for ln in times_a)
    term = [i for i, ln in enumerate(times_a) if re.fullmatch(r"TERM \d+", ln)]
    assert term and times_a[term[0] + 1].startswith("JOB_LEFT ")
    assert a_uv in times_a[term[0] + 1].split()
    assert not any(ln.startswith(("ALL_DONE", "CAPTURE_RC")) for ln in times_a)
    # b takes the lock once a's TERM released it, never judging it stale
    assert _until(lambda: _text(b.dir / "uv.pid").strip()), _text(b.dir / "run.out")
    assert _text(pod.lock / "name").strip().endswith("/snapt-b")
    assert "SNAP_LOCK_STALE" not in _text(b.dir / "times.txt")
    b_uv = _text(b.dir / "uv.pid").strip()
    pod.term(_text(pod.lock / "pid").strip())
    assert b.wait(timeout=60) == 143
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
    assert "JOB_LEFT" not in times and times.rstrip().endswith("ALL_DONE")
    assert not pod.lock.exists()


def test_run_sh_a_killed_holder_with_a_live_job_keeps_the_lock(pod):
    """run.sh KILLed (no trap runs) while its job runs: the job's group still holds the lock;
    once the job is gone the next job judges it stale and runs."""
    a = pod.job("snapt-k", 120)
    assert _until(lambda: _text(a.dir / "uv.pid").strip()), _text(a.dir / "run.out")
    a_uv, holder = _text(a.dir / "uv.pid").strip(), _text(pod.lock / "pid").strip()
    pod.kill(holder)
    a.wait(timeout=30)
    assert pod.alive(a_uv) and pod.lock.exists()
    b = pod.job("snapt-l", 0)
    assert _until(lambda: "lock wait" in _text(b.dir / "run.out")), _text(b.dir / "run.out")
    assert not _until(lambda: "SNAP_LOCK" in _text(b.dir / "times.txt"), 4)
    pod.kill(a_uv)
    assert b.wait(timeout=60) == 0, _text(b.dir / "run.out")
    times = _text(b.dir / "times.txt")
    assert f"SNAP_LOCK_STALE {holder} " in times and times.rstrip().endswith("ALL_DONE")


def test_run_sh_stops_what_the_driver_left_before_all_done(pod):
    a = pod.job("snapt-x", 0, FAKE_UV_LEAVE="1")
    assert a.wait(timeout=90) == 0, _text(a.dir / "run.out")
    left = _text(a.dir / "left.pid").strip()
    lines = _text(a.dir / "times.txt").splitlines()
    assert re.fullmatch(r"CAPTURE_RC=9 \d+", lines[-3])
    assert re.fullmatch(rf"JOB_LEFT {left} \d+", lines[-2]) and lines[-1] == "ALL_DONE"
    assert not pod.alive(left), pod.sh(f"cat /proc/{left}/stat").stdout  # gone or a zombie
    assert not pod.lock.exists()


def test_run_sh_keeps_the_lock_when_a_job_process_outlives_the_kill(pod):
    """A leftover that TERM and KILL do not stop (a process stuck in the kernel): run.sh writes
    JOB_STUCK and exits 1 without ALL_DONE, keeping the pod lock; the next job waits until that
    process is gone, then judges the lock stale."""
    a = pod.job("snapt-y", 0, nokill=True, FAKE_UV_LEAVE="stuck", SNAPSTOP_KILL_S="1",
                SNAPSTOP_MAX_S="3")  # fmt: skip
    assert a.wait(timeout=90) == 1, _text(a.dir / "run.out")
    left = _text(a.dir / "left.pid").strip()
    lines = _text(a.dir / "times.txt").splitlines()
    assert re.fullmatch(r"CAPTURE_RC=9 \d+", lines[-3]), lines
    assert re.fullmatch(rf"JOB_LEFT {left} \d+", lines[-2])
    assert re.fullmatch(rf"JOB_STUCK {left} \d+", lines[-1])
    assert "JOB_STUCK in times.txt" in _text(a.dir / "run.out")
    assert pod.alive(left) and pod.lock.exists()
    b = pod.job("snapt-z", 0)
    assert _until(lambda: "lock wait" in _text(b.dir / "run.out")), _text(b.dir / "run.out")
    assert not _until(lambda: "SNAP_LOCK" in _text(b.dir / "times.txt"), 4)
    pod.kill(left)
    assert b.wait(timeout=60) == 0, _text(b.dir / "run.out")
    times = _text(b.dir / "times.txt")
    assert "SNAP_LOCK_STALE " in times and times.rstrip().endswith("ALL_DONE")
    assert not pod.lock.exists()


def test_snapstop_returns_1_with_job_stuck_when_a_member_survives(tmp_path, fns):
    """snapstop's bound: a member it cannot stop (kill shadowed) is logged JOB_STUCK and snapstop
    returns 1 after SNAPSTOP_MAX_S, never 0."""
    p, g = _group_leader()
    times = tmp_path / "times.txt"
    try:
        r = _own_group_bash(f". '{fns.as_posix()}'; kill() {{ :; }}; SG=\"$$ {g}\"; "
                            f"SNAPSTOP_KILL_S=1 SNAPSTOP_MAX_S=2 snapstop '{times.as_posix()}'; "
                            f"echo rc=$?")  # fmt: skip
    finally:
        _bash(f"kill -9 {g}")
        p.kill()
        p.wait()
    assert r.stdout.strip() == "rc=1", r.stdout + r.stderr
    t = _text(times).splitlines()
    assert re.fullmatch(rf"JOB_LEFT {g} \d+", t[0]) and re.fullmatch(rf"JOB_STUCK {g} \d+", t[1]), t


@pytest.mark.skipif(not _setsid(), reason="needs setsid (Linux, the pod)")
def test_run_sh_tracks_a_job_in_a_group_of_its_own(pod):
    """A job process in its own session (as uv might start the driver): snapwatch records its
    group in the lock, so a KILLed run.sh's lock stays held, and a TERM stops it."""
    a = pod.job("snapt-g1", 120, FAKE_UV_SETSID="1")
    assert _until(lambda: _text(a.dir / "uv.pid").strip()), _text(a.dir / "run.out")
    a_uv = _text(a.dir / "uv.pid").strip()
    assert _until(lambda: a_uv in _text(pod.lock / "groups").split(), 10)
    pod.term(_text(pod.lock / "pid").strip())
    assert a.wait(timeout=60) == 143
    assert _until(lambda: not pod.alive(a_uv), 10)
    b = pod.job("snapt-g2", 120, FAKE_UV_SETSID="1")
    assert _until(lambda: _text(b.dir / "uv.pid").strip()), _text(b.dir / "run.out")
    b_uv, holder = _text(b.dir / "uv.pid").strip(), _text(pod.lock / "pid").strip()
    assert _until(lambda: b_uv in _text(pod.lock / "groups").split(), 10)
    pod.kill(holder)
    b.wait(timeout=30)
    c = pod.job("snapt-g3", 0)
    assert _until(lambda: "lock wait" in _text(c.dir / "run.out")), _text(c.dir / "run.out")
    assert not _until(lambda: "SNAP_LOCK" in _text(c.dir / "times.txt"), 4)
    pod.kill(b_uv)
    assert c.wait(timeout=60) == 0, _text(c.dir / "run.out")


def test_run_sh_gate_starts_alone_after_idle_wait_when_memory_stays_high(pod):
    """Memory stuck over the gate (page cache no unit holds) with no pack unit live: after
    IDLE_WAIT_S the job starts anyway, GATE_IDLE_START in times.txt."""
    pod.mem(200 * GB)
    a = pod.job("snapt-i", 0, IDLE_WAIT_S="2")
    assert a.wait(timeout=60) == 0, _text(a.dir / "run.out")
    lines = [ln.split()[0] for ln in _text(a.dir / "times.txt").splitlines()]
    assert lines[lines.index("SNAP_LOCK") + 1:] == ["GATE_IDLE_START", "CAPTURE_START",
                                                    "CAPTURE_RC=0", "ALL_DONE"]  # fmt: skip
    out = _text(a.dir / "run.out").splitlines()
    assert out and all(re.fullmatch(r"gate wait \S+: memory 200 GB without page cache, gate 88 "
                                    r"GB, 0 pack units live", ln) for ln in out)  # fmt: skip


@pytest.mark.skipif(sys.platform == "win32", reason="podgate finds the live unit by its process "
                    "group with a procps ps; Git Bash's ps does not give it: Linux only (WSL runner, the pod)")
def test_run_sh_gate_waits_while_a_pack_unit_is_live(pod):
    pod.mem(200 * GB)
    unit = pod.sleeper()
    run_dir = pod.work / "r1" / "out" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "u_001").write_text(unit + "\n")
    a = pod.job("snapt-u", 0, IDLE_WAIT_S="2")
    assert _until(lambda: "1 pack units live" in _text(a.dir / "run.out")), _text(a.dir / "run.out")
    assert not _until(lambda: "CAPTURE_START" in _text(a.dir / "times.txt"), 5)
    pod.kill(unit)
    assert a.wait(timeout=60) == 0, _text(a.dir / "run.out")
    assert "GATE_IDLE_START" in _text(a.dir / "times.txt")


def test_run_sh_gate_does_not_count_inactive_file_cache(pod):
    pod.mem(200 * GB, inactive=150 * GB)  # 50 GB without the cache: under the 88 GB gate
    a = pod.job("snapt-c", 0)
    assert a.wait(timeout=60) == 0, _text(a.dir / "run.out")
    assert "gate wait" not in _text(a.dir / "run.out")
    assert "GATE_IDLE_START" not in _text(a.dir / "times.txt")


def _times(d):
    return [ln.split()[0] for ln in _text(d / "times.txt").splitlines()]


def test_run_sh_stores_a_passed_capture_after_the_job_under_the_lock(pod, tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    a = pod.job("snapt-st", 0, vol=vol, FAKE_UV_CAPTURE="1")
    assert a.wait(timeout=90) == 0, _text(a.dir / "run.out")
    t = _times(a.dir)
    assert t[t.index("SNAP_LOCK"):] == ["SNAP_LOCK", "CAPTURE_START", "CAPTURE_RC=0", "STORE_RC=0",
                                        "ALL_DONE"], (t, _text(a.dir / "store.log"))  # fmt: skip
    assert (vol / "snap" / pod.key / "snap" / "capture.json").is_file()
    assert "CAPTURE_RC=0" in _text(vol / "snap" / pod.key / "times.txt")
    assert not pod.lock.exists()


def test_run_sh_fetches_under_the_lock_before_the_gate(pod, tmp_path):
    """A capture whose key is on the volume: it waits for the lock like any job, then fetches
    and ends without the memory gate (memory stays over it) and without capturing."""
    vol = tmp_path / "vol"
    stored = vol / "snap" / pod.key
    (stored / "snap").mkdir(parents=True)
    (stored / "snap" / "capture.json").write_text('{"parts": {"p1": "ok"}}')
    (stored / "snap" / "p1.pkl.gz").write_text("states")
    (stored / "snap" / "p1.sha256").write_text("0  p1\n")
    (stored / "KEY").write_text(pod.key + "\n")
    (stored / "times.txt").write_text("CAPTURE_RC=0 1\nALL_DONE\n")
    a = pod.job("snapt-hold", 120)  # no volume: it captures and holds the lock
    assert _until(lambda: _text(a.dir / "uv.pid").strip()), _text(a.dir / "run.out")
    pod.mem(200 * GB)
    b = pod.job("snapt-fv", 0, vol=vol, IDLE_WAIT_S="600")
    assert _until(lambda: "lock wait" in _text(b.dir / "run.out")), _text(b.dir / "run.out")
    assert not (b.dir / "snap").exists() and not (b.dir / "store.log").exists()  # nothing before the lock
    pod.term(_text(pod.lock / "pid").strip())
    assert a.wait(timeout=60) == 143
    assert b.wait(timeout=60) == 0, _text(b.dir / "run.out")
    t = _times(b.dir)
    assert t[t.index("SYNC_RC=0"):] == ["SYNC_RC=0", "SNAP_LOCK", "CAPTURE_FROM_VOLUME", "CAPTURE_RC=0",
                                        "ALL_DONE"], (t, _text(b.dir / "store.log"))  # fmt: skip
    assert "gate wait" not in _text(b.dir / "run.out") and not (b.dir / "uv.pid").exists()
    assert (b.dir / "snap" / "capture.json").is_file()
    assert not pod.lock.exists()


# A stand-in cache_sync.sh whose calls hang: a fetch leaves a partial snap/ first, as a copy cut
# off mid-way; a store only hangs.
HANG_SYNC = """#!/bin/bash
if [ "$1" = snap-fetch ]; then mkdir -p "$3/snap" "$3/.snapfetch.1"; echo half > "$3/snap/partial"; fi
exec sleep 60
"""


def test_run_sh_captures_after_a_timed_out_fetch_and_records_a_timed_out_store(pod, tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    a = pod.job("snapt-to", 0, vol=vol, sync=HANG_SYNC, FAKE_UV_CAPTURE="1")
    assert a.wait(timeout=90) == 0, _text(a.dir / "run.out")
    t = _times(a.dir)
    assert t[t.index("SNAP_LOCK"):] == ["SNAP_LOCK", "VOLUME_TIMEOUT", "CAPTURE_START", "CAPTURE_RC=0",
                                        "VOLUME_TIMEOUT", "STORE_RC=124", "ALL_DONE"], t  # fmt: skip
    vt = [ln.split()[1] for ln in _text(a.dir / "times.txt").splitlines() if ln.startswith("VOLUME_TIMEOUT")]
    assert vt == ["fetch", "store"]
    assert not (a.dir / "snap" / "partial").exists() and not list(a.dir.glob(".snapfetch.*"))
    out = _text(a.dir / "run.out")
    assert "timed out after 900 s (rc 124): captured instead" in out and "not stored" in out
    assert not pod.lock.exists()


def test_run_sh_compare_stops_with_no_capture_after_a_timed_out_or_failed_fetch(pod, tmp_path):
    vol = tmp_path / "vol"
    vol.mkdir()
    scratch = tmp_path / "scratch" / "rent" / "tables"
    scratch.mkdir(parents=True)
    (scratch / "index.txt").write_text(f"{pod.key.split('-')[0]} {'c' * 64}\n")
    cap = f"/work/snapmain-{pod.key[:7]}/snap"
    a = pod.job("snapcmp-to-1", 0, vol=vol, cap=cap, sync=HANG_SYNC)
    assert a.wait(timeout=90) == 1, _text(a.dir / "run.out")
    t = _times(a.dir)
    assert t[t.index("SNAP_LOCK"):] == ["SNAP_LOCK", "FETCH_RC=124", "VOLUME_TIMEOUT", "NO_CAPTURE"], t
    main = pod.work / f"snapmain-{pod.key[:7]}"
    assert not list(main.glob(".snapfetch.*"))
    assert not (main / "snap" / "capture.json").exists() and "COMPARE_START" not in t
    b = pod.job("snapcmp-to-2", 0, vol=vol, cap=cap)  # the real cache_sync.sh: the key is not stored
    assert b.wait(timeout=90) == 1, _text(b.dir / "run.out")
    t = _times(b.dir)
    assert t[t.index("SNAP_LOCK"):] == ["SNAP_LOCK", "FETCH_RC=1", "NO_CAPTURE"], (t, _text(b.dir / "store.log"))
    assert not pod.lock.exists()
