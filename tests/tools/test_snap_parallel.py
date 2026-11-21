"""tools/snap_compare.py's parallel driver (--workers > 1): print order, failure propagation,
the capture's step-first order and the memory gate. Children are fake processes on a fake
clock, so no engine is imported and no test waits."""

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


def pool(parts, workers=4, gate_gb=1000, mode="compare", **kw):
    clock = kw.pop("clock", None) or Clock()
    fake = kw.pop("fake", None) or Fake(clock)
    p = sc.PartPool(mode, "snap", parts, workers, gate_gb, launch=fake, clock=clock,
                    sleep=clock.sleep, memory=kw.pop("memory", lambda: None),
                    rss=kw.pop("rss", lambda pid: 0), **kw)  # fmt: skip
    return p, fake, clock


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
                        clock=clock, sleep=clock.sleep, memory=lambda: None, rss=lambda pid: 0)  # fmt: skip

        monkeypatch.setattr(sc, "PartPool", fake_pool)
        capsys.readouterr()
        rc = sc.parent(sc.build_parser().parse_args(argv))
        return rc, capsys.readouterr().out.splitlines()

    go.snap = tmp_path / "snap"
    return go


def _parts_only(lines):
    """The printed lines without timings and without the driver's own two lines."""
    keep = [ln for ln in lines if "parts at once" not in ln and "parts ran at once" not in ln]
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


def test_capture_starts_a_chunk_s_groups_after_its_step_part():
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


def test_longest_part_starts_first_and_the_worker_cap_holds():
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
    assert sc.admit(13, {}, 87 * gb, 88) and not sc.admit(13, {}, 88 * gb, 88)
    assert sc.admit(200, {}, 0, 88)  # too heavy for the gate: starts alone
    live = {"a": (13, 3 * gb), "b": (13, 13 * gb)}
    # memory 40 + reserve (10 + 0) + 13 = 63
    assert sc.admit(13, live, 40 * gb, 63) and not sc.admit(13, live, 40 * gb, 62)
    # no memory reading: the live RSS sum (16) + reserve (10) + 13 = 39
    assert sc.admit(13, live, None, 39) and not sc.admit(13, live, None, 38)


def test_gate_keeps_live_parts_under_it():
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


def test_heavy_part_waits_then_runs_alone():
    parts = ["0-9.ai", "roark.rank", "0-9.gen"]  # roark.rank 27 GB against a 20 GB gate
    p, fake, _ = pool(parts, workers=4, gate_gb=20)
    for part in parts:
        p.result(part)
    rank = (fake.starts["roark.rank"], fake.end("roark.rank"))
    for q in ("0-9.ai", "0-9.gen"):
        assert fake.end(q) <= rank[0] or fake.starts[q] >= rank[1]


def test_close_kills_live_children(tmp_path):
    p, fake, _ = pool(["roark.rank", "roark.trace"], workers=2, spacing=0)
    p._start()
    p._start()
    assert len(p.live) == 2
    p.close()
    assert all(proc.killed for proc in fake.procs.values())
    assert not p.tmp.exists()


def test_a_real_child_s_output_is_read_back():
    def launch(cmd, out, err):
        code = "import sys; print('RESULT {}'); print('warn', file=sys.stderr); sys.exit(3)"
        return subprocess.Popen([sys.executable, "-c", code], stdout=out, stderr=err)

    p = sc.PartPool("compare", "snap", ["0-9.ai"], 2, 88, launch=launch, memory=lambda: None,
                    poll=0.05)  # fmt: skip
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
