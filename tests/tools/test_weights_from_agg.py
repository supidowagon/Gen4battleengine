"""tools/ops/day/weights_from_agg.py: weights.txt's generated part from pod-run agg.txt files."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("weights_from_agg", ROOT / "tools/ops/day/weights_from_agg.py")
wfa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfa)

HEADER = "file collected passed failed skipped errors xfailed xpassed not_run max_rss_MB wall_sum_s complete"
UNITS = "--- per unit: unit ids max_rss_MB wall_s rc first_id"


def _agg(root: Path, run: str, units: list[str], ids: dict[str, list[str]] = {}) -> Path:
    d = root / run / "results"
    (d / "out").mkdir(parents=True)
    for u, xs in ids.items():
        (d / "out" / f"{u}.ids").write_text("\n".join(xs) + "\n", encoding="utf-8")
    (d / "agg.txt").write_text("\n".join([HEADER, "tests/x.py 1 1 0 0 0 0 0 0 1 1 yes", "--- wall 1 s", UNITS, *units,
                                          "--- slowest 15", "RESULT: PASS"]) + "\n", encoding="utf-8")
    return d / "agg.txt"


def test_generated_lines_take_the_max_per_file_with_floor_and_skip_hand_kept_units(tmp_path):
    w = tmp_path / "weights.txt"
    w.write_text("# head\ntests/h.py::test_big 90 r0\ntests/f.py 7 r0\n" + wfa.MARK + "\nold line 1 x\n",
                 encoding="utf-8")
    a1 = _agg(tmp_path, "run-a", [
        "u_001 2 20000 10 0 tests/a.py::test_1",        # 20000 MiB -> 21 GB
        "u_002 1 100 1 0 tests/b.py::test_1",           # 1 GB -> floor 4
        "u_003 1 80000 9 137 tests/h.py::test_big",     # only a hand-kept id: left out
        "u_004 2 30000 9 0 tests/h.py::test_big",       # ids file adds an unlisted id: counts
        "u_005 1 50000 1 0 tests/f.py::test_1",         # file with a hand file-level line: not printed
        "u_006 1 0 1 0 tests/c.py::test_1",             # no measurement
    ], ids={"u_004": ["tests/h.py::test_big", "tests/h.py::test_small"]})
    a2 = _agg(tmp_path, "run-b", ["u_001 1 30000 10 0 tests/a.py::test_2", "u_002 1 10 1 0 tests/b.py::test_2"])
    out = wfa.render(w, [a1, a2]).splitlines()
    assert out[:4] == ["# head", "tests/h.py::test_big 90 r0", "tests/f.py 7 r0", wfa.MARK]
    assert out[4:] == ["tests/a.py 32 run-b", "tests/h.py 32 run-a", "tests/b.py 4 run-a"]


def test_an_agg_without_a_unit_table_uses_its_file_rows(tmp_path):
    w = tmp_path / "weights.txt"
    w.write_text(wfa.MARK + "\n", encoding="utf-8")
    d = tmp_path / "old" / "results"
    d.mkdir(parents=True)
    (d / "agg.txt").write_text(HEADER + "\ntests/q.py 3 3 0 0 0 0 0 0 9000 5 yes\n--- wall 5 s\n", encoding="utf-8")
    assert wfa.render(w, [d / "agg.txt"]).splitlines()[1:] == ["tests/q.py 10 old"]


def test_tracked_table_has_the_marker():
    assert wfa.MARK in (ROOT / "tools/ops/day/pod/weights.txt").read_text(encoding="utf-8").splitlines()
