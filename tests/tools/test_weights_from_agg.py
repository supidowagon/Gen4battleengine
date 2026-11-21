"""tools/ops/day/weights_from_agg.py: weights.txt refreshed from pod-run agg.txt files."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("weights_from_agg", ROOT / "tools/ops/day/weights_from_agg.py")
wfa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfa)

HEADER = "file collected passed failed skipped errors xfailed xpassed not_run max_rss_MB wall_sum_s complete"
UNITS = "--- per unit: unit ids max_rss_MB wall_s rc first_id"
OK = "1 passed in 3.10s (0:00:03)"


def _agg(root: Path, run: str, units: list[str], ids: dict[str, list[str]] | None = None,
         logs: dict[str, str] | None = None, mark: str = "not slow and not fullset", commit: str = "c0",
         a_units: tuple[str, ...] = ()) -> Path:
    """logs: unit -> the last line of its pytest log (out/<unit>.log); a_units: units of an A line."""
    d = root / run / "results"
    (d / "out").mkdir(parents=True)
    for u, xs in (ids or {}).items():
        (d / "out" / f"{u}.ids").write_text("\n".join(xs) + "\n", encoding="utf-8")
    for u, last in (logs or {}).items():
        (d / "out" / f"{u}.log").write_text(f".  [100%]\nslowest 40 durations\n{last}\n", encoding="utf-8")
    for u in a_units:
        (d / "out" / f"{u}.m").write_text("A\n", encoding="utf-8")
    (d / "meta.env").write_text(f'COMMIT={commit}\nMARK="{mark}"\n', encoding="utf-8")
    (d / "agg.txt").write_text("\n".join([HEADER, "tests/x.py 1 1 0 0 0 0 0 0 1 1 yes", "--- wall 1 s", UNITS, *units,
                                          "--- slowest 15", "RESULT: PASS"]) + "\n", encoding="utf-8")
    return d / "agg.txt"


def _table(tmp_path: Path, head: str, gen: str) -> Path:
    w = tmp_path / "weights.txt"
    w.write_text(head + wfa.MARK + "\n" + gen, encoding="utf-8")
    return w


@pytest.mark.parametrize(("m", "floor"), [(4, 6), (10, 13), (30, 38), (39, 49), (40, 44), (80, 88)])
def test_floor_is_1_25_x_or_plus_2_gb_and_1_1_x_from_40_gb(m, floor):
    assert wfa.floor_gb(m) == floor


@pytest.mark.parametrize(("old", "m", "new"), [
    (30, 10, 13),    # down to the floor
    (12, 10, 12),    # the floor (13) is above the line: it stays
    (13, 10, 13),
    (10, 20, 20),    # a raise takes the measured peak
    (82, 50, 55),    # 40 GB and more: x 1.1
    (82, None, 82),  # no fresh measurement: stays
])
def test_a_line_goes_down_only_to_the_floor(old, m, new):
    assert wfa.new_value(old, m) == new


def test_file_lines_take_the_max_with_floor_and_skip_hand_kept_units(tmp_path):
    w = _table(tmp_path, "# head\ntests/h.py::test_big 90 r0\ntests/f.py 7 r0\n", "tests/old.py 9 x\n")
    a1 = _agg(tmp_path / "out", "run-a", [
        "u_001 2 20000 10 0 tests/a.py::test_1",        # 20000 MiB -> 21 GB
        "u_002 2 100 1 0 tests/b.py::test_1",           # 1 GB -> floor 4
        "u_003 1 80000 9 137 tests/h.py::test_big",     # only a hand-kept id: left out
        "u_004 2 30000 9 0 tests/h.py::test_big",       # ids file adds an unlisted id: counts
        "u_005 2 50000 1 0 tests/f.py::test_1",         # file with a hand file-level line: not printed
        "u_006 1 0 1 0 tests/c.py::test_1",             # no measurement
    ], ids={"u_004": ["tests/h.py::test_big", "tests/h.py::test_small"]})
    a2 = _agg(tmp_path / "out", "run-b", ["u_001 2 30000 10 0 tests/a.py::test_2"])
    out = wfa.render(w, [a1, a2], [], report=[]).splitlines()
    # tests/f.py's units have only hand-kept ids: its hand line has no fresh measurement
    assert out[:4] == ["# head", "tests/h.py::test_big 90 r0", "tests/f.py 7 r0", wfa.MARK]
    assert out[4:] == ["tests/a.py 32 run-b", "tests/h.py 32 run-a", "tests/old.py 9 x", "tests/b.py 4 run-a"]


def test_an_agg_without_a_unit_table_uses_its_file_rows(tmp_path):
    w = _table(tmp_path, "", "")
    d = tmp_path / "out" / "old" / "results"
    d.mkdir(parents=True)
    (d / "agg.txt").write_text(HEADER + "\ntests/q.py 3 3 0 0 0 0 0 0 9000 5 yes\n--- wall 5 s\n", encoding="utf-8")
    assert wfa.render(w, [d / "agg.txt"], [], report=[]).splitlines()[1:] == ["tests/q.py 10 old"]


def test_tracked_table_has_the_marker_and_loads():
    lines = (ROOT / "tools/ops/day/pod/weights.txt").read_text(encoding="utf-8").splitlines()
    assert wfa.MARK in lines
    wfa.weights.load_lines(lines)


def test_warnings_for_kernel_kills_triple_kills_and_hand_kept_overruns(tmp_path):
    w = _table(tmp_path, "tests/h.py::test_big 10 r0\n", "tests/a.py 70 old\n")
    agg = _agg(tmp_path / "out", "run-a", [
        "u_001 2 20000 10 0 tests/a.py::test_1",       # 21 GB measured
        "u_002 2 60000 10 137 tests/a.py::test_2",     # kernel kill: at least 63 GB, the line takes it
        "u_003 1 0 10 98 tests/b.py::test_1",          # killed three times: no peak
        "u_004 2 30000 10 0 tests/h.py::test_big",     # hand-kept 10 GB, peaked at 32
    ], ids={"u_004": ["tests/h.py::test_big", "tests/h.py::test_big[x]"]})
    warn: list[str] = []
    out = wfa.render(w, [agg], warn, report=[]).splitlines()
    assert out[-1] == "tests/a.py 70 old"   # 63 measured, the cited run not given: the line stays
    assert sorted(warn) == sorted([
        "killed three times, no peak: run-a u_003 tests/b.py",
        "hand-kept ids peaked at 32 GB, above their largest hand-kept weight 10 GB: run-a u_004 tests/h.py::test_big",
        ("rc 137 lower bound 63 GB above the file's 21 GB; the line takes the bound, the real peak may be higher "
         "(a hand-kept line or a rerun): tests/a.py (run-a u_002)"),
    ])


def test_per_test_lines_come_from_1_id_units_that_passed_their_test(tmp_path):
    # test_h and test_e cite run-a, whose units holding them are known id by id: they may go down
    w = _table(tmp_path, "tests/p.py::test_h 80 run-a\ntests/p.py::test_k 80 r0 keep\ntests/p.py::test_z 80 r0\n",
               "tests/p.py 80 old\ntests/p.py::test_e 30 run-a\n")
    a1 = _agg(tmp_path / "out", "run-a", [
        "u_001 5 76000 50 0 tests/p.py::test_a",        # 80 GB: the file line
        "u_002 1 8000 10 0 tests/p.py::test_a",         # passed alone at 9
        "u_003 1 3000 10 0 tests/p.py::test_b",         # skipped: not its cost
        "u_004 1 4000 10 1 tests/p.py::test_c",         # failed
        "u_005 1 58000 10 0 tests/p.py::test_d",        # 61: floor 68 = file line - 12
        "u_006 1 5000 10 0 tests/p.py::test_h",         # hand-kept 80, alone at 6: down to 8
        "u_007 1 5000 10 0 tests/p.py::test_k",         # hand-kept 80 with "keep": stays
        "u_008 2 9000 10 0 tests/p.py::test_e",         # 2 ids: no measurement of e
        "u_009 1 69000 10 0 tests/p.py::test_g",        # 73: floor 81 above the file line: its own line
        "u_010 1 4000 10 0 tests/p.py::test_x[a b]",    # an id with a space: no line
        "u_011 1 4000 10 0 tests/p.py::test_y",         # no log: no line
        "u_012 1 5000 10 0 tests/p.py::test_z",         # hand-kept 80, alone at 6, but also killed:
        "u_013 2 9000 10 137 tests/p.py::test_z",       # it stays
        "u_014 1 55000 10 0 tests/p.py::test_q",        # 58: floor 64 = file line 80 - 16
    ], {"u_001": [f"tests/p.py::test_{c}" for c in "abcdh"], "u_008": ["tests/p.py::test_e", "tests/p.py::test_f"],
        "u_010": ["tests/p.py::test_x[a b]"], "u_013": ["tests/p.py::test_z", "tests/p.py::test_w"]},
        {u: OK for u in ("u_002", "u_005", "u_006", "u_007", "u_009", "u_010", "u_012", "u_014")}
        | {"u_003": "1 skipped in 0.10s", "u_004": "1 failed in 1.00s", "u_008": "2 passed in 9.00s"})
    a2 = _agg(tmp_path / "out", "run-b", ["u_001 1 20000 10 0 tests/p.py::test_e", "u_002 1 9000 10 0 tests/p.py::test_a"],
              logs={"u_001": "1 passed, 2 warnings in 5.00s", "u_002": OK})
    report: list[str] = []
    out = wfa.render(w, [a1, a2], [], report=report).splitlines()
    assert out[:4] == [f"{'tests/p.py::test_h':<74} 8 run-a", "tests/p.py::test_k 80 r0 keep", "tests/p.py::test_z 80 r0",
                       wfa.MARK]
    assert out[4:] == ["tests/p.py 80 old",          # the cited run is not given: not lowered, kept
                       "tests/p.py::test_a 13 run-b",       # the larger of two passed 1-id units (10), floor 13
                       "tests/p.py::test_d 68 run-a",
                       "tests/p.py::test_e 27 run-b",       # old line 30, measured 21 (a warnings count is a pass)
                       "tests/p.py::test_g 81 run-a",
                       "tests/p.py::test_q 64 run-a"]
    assert report == ["REDUCED tests/p.py::test_h 80 -> 8 GB", "REDUCED tests/p.py::test_a 80 -> 13 GB",
                      "REDUCED tests/p.py::test_q 80 -> 64 GB", "REDUCED tests/p.py::test_d 80 -> 68 GB",
                      "REDUCED tests/p.py::test_e 30 -> 27 GB", "changed: 5 down, 1 up"]
    # the new table loads; a multi-id chunk weighs the max over its ids; a rerun changes nothing
    w2 = tmp_path / "w2.txt"
    w2.write_text("\n".join(out) + "\n", encoding="utf-8")
    t = wfa.weights.load(w2)
    assert wfa.weights.unit_weight(["tests/p.py::test_a", "tests/p.py::test_e"], t) == 27
    assert wfa.weights.unit_weight(["tests/p.py::test_a", "tests/p.py::test_b"], t) == 80
    assert wfa.render(w2, [a1, a2], [], report=[]) == w2.read_text(encoding="utf-8")


def _slow_peak(out: Path) -> Path:
    """Run old-1: an A unit of tests/s.py (a fast and a slow id) peaked at 28 GB; the file line cites it."""
    return _agg(out, "old-1", ["u_001 2 26000 300 0 tests/s.py::test_fast"],
                ids={"u_001": ["tests/s.py::test_fast", "tests/s.py::test_slow"]}, a_units=("u_001",))


def test_a_default_mark_run_does_not_lower_a_line_whose_peak_came_from_an_a_unit(tmp_path):
    out = tmp_path / "out"
    old = _slow_peak(out)
    w = _table(tmp_path, "", "tests/s.py 28 old-1\ntests/u.py 28 gone-1\n")
    # the default-mark run collected only the fast id (and tests/u.py's cited run is not given)
    a = _agg(out, "fresh-1", ["u_001 1 9000 10 0 tests/s.py::test_fast", "u_002 1 9000 10 0 tests/u.py::test_1"],
             ids={"u_001": ["tests/s.py::test_fast"], "u_002": ["tests/u.py::test_1"]})
    report: list[str] = []
    got = wfa.render(w, [old, a], [], since=["fresh-*"], report=report)
    assert got.splitlines()[1:] == ["tests/s.py 28 old-1", "tests/u.py 28 gone-1"]
    assert report == ["changed: 0 down, 0 up"]


def test_a_fresh_a_unit_that_ran_the_peak_ids_lowers_the_line(tmp_path):
    out = tmp_path / "out"
    old = _slow_peak(out)
    w = _table(tmp_path, "", "tests/s.py 28 old-1\n")
    a = _agg(out, "fresh-2", ["u_001 2 9000 300 0 tests/s.py::test_fast"],
             ids={"u_001": ["tests/s.py::test_fast", "tests/s.py::test_slow"]}, a_units=("u_001",))
    assert wfa.render(w, [old, a], [], since=["fresh-*"], report=[]).splitlines()[1:] == ["tests/s.py 13 fresh-2"]
    # an old unit without its ids file (2 ids, only the first known) cannot be checked: the line stays
    (out / "old-1" / "results" / "out" / "u_001.ids").unlink()
    assert wfa.render(w, [old, a], [], since=["fresh-*"], report=[]).splitlines()[1:] == ["tests/s.py 28 old-1"]


def test_a_per_test_line_goes_down_only_when_the_ids_behind_its_peak_passed_alone(tmp_path):
    out = tmp_path / "out"
    old = _agg(out, "old-1", ["u_001 2 30000 300 0 tests/v.py::test_p[1]"],
               ids={"u_001": ["tests/v.py::test_p[1]", "tests/v.py::test_p[2]"]}, a_units=("u_001",))
    w = _table(tmp_path, "tests/v.py::test_p 33 old-1\n", "tests/v.py 40 x\n")
    a = _agg(out, "fresh-1", ["u_001 1 5000 10 0 tests/v.py::test_p[1]"], logs={"u_001": OK})
    assert wfa.render(w, [old, a], [], since=["fresh-*"], report=[]).splitlines()[0] == "tests/v.py::test_p 33 old-1"
    b = _agg(out, "fresh-2", ["u_001 1 5000 10 0 tests/v.py::test_p[2]"], logs={"u_001": OK})
    got = wfa.render(w, [old, a, b], [], since=["fresh-*"], report=[]).splitlines()[0]
    assert got == f"{'tests/v.py::test_p':<74} 8 fresh-2"


def test_since_counts_only_the_named_commits_or_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(wfa, "_descends", lambda base, commit, cache: (base, commit) in {("b1", "c2"), ("b2", "c2")})
    w = _table(tmp_path, "", "")
    out = tmp_path / "out"
    old = _agg(out, "old-run", ["u_001 2 30000 1 0 tests/a.py::t", "u_002 2 30000 1 0 tests/b.py::t"], commit="c1")
    new = _agg(out, "new-run", ["u_001 2 10000 1 0 tests/a.py::t", "u_002 2 10000 1 0 tests/b.py::t"], commit="c2")
    both = [old, new]
    assert wfa.render(w, both, [], report=[]).splitlines()[1:] == ["tests/a.py 32 old-run", "tests/b.py 32 old-run"]
    assert wfa.render(w, both, [], since=["b1"], report=[]).splitlines()[1:] == ["tests/a.py 11 new-run",
                                                                                 "tests/b.py 11 new-run"]
    assert wfa.render(w, both, [], since=["new-*"], report=[]).splitlines()[1:] == ["tests/a.py 11 new-run",
                                                                                   "tests/b.py 11 new-run"]
    # a per-file spec narrows that file only
    got = wfa.render(w, both, [], since=["*-run", "tests/b.py=b2"], report=[]).splitlines()[1:]
    assert got == ["tests/a.py 32 old-run", "tests/b.py 11 new-run"]


def test_main_takes_since_and_prints_the_report(tmp_path, capsys):
    w = _table(tmp_path, "", "tests/a.py 30 run-a\n")
    agg = _agg(tmp_path / "out", "run-a", ["u_001 2 9000 10 0 tests/a.py::test_1"])
    assert wfa.main(["--since", "run-*", str(w), str(agg)]) == 0
    io = capsys.readouterr()
    assert io.out.splitlines()[-1] == "tests/a.py 13 run-a"
    assert io.err.splitlines() == ["REDUCED tests/a.py 30 -> 13 GB", "changed: 1 down, 0 up"]
    assert wfa.main(["--bad", str(w), str(agg)]) == 2
