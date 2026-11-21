"""tools/ops/day/pod/weights.py: expected unit peaks for runner.sh's memory gate."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POD = ROOT / "tools" / "ops" / "day" / "pod"
_spec = importlib.util.spec_from_file_location("pod_weights", POD / "weights.py")
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

TABLE = [
    ("tests/search", 20),
    ("tests/search/test_x.py", 30),
    ("tests/search/test_x.py::test_big", 90),
]


@pytest.mark.parametrize(("test_id", "gb"), [
    ("tests/search/test_x.py::test_big", 90),
    ("tests/search/test_x.py::test_big[3]", 90),
    ("tests/search/test_x.py::test_bigger", 30),  # a longer name is another test
    ("tests/search/test_x.py::test_small", 30),
    ("tests/search/test_y.py::test_a", 20),
    ("tests/searcher/test_z.py::test_a", w.DEFAULT_GB),
    ("tests/engine/test_q.py::test_a", w.DEFAULT_GB),
])
def test_longest_matching_entry_gives_the_weight(test_id, gb):
    assert w.id_weight(test_id, TABLE) == gb


def test_unit_weight_is_its_heaviest_id_and_a_line_weight_only_raises_it():
    ids = ["tests/search/test_x.py::test_small", "tests/search/test_x.py::test_big[0]"]
    assert w.unit_weight(ids, TABLE) == 90
    assert w.unit_weight(ids, TABLE, 40) == 90
    assert w.unit_weight(ids, TABLE, 120) == 120
    assert w.unit_weight(["tests/engine/test_q.py::test_a"], TABLE, 40) == 40
    assert w.unit_weight([], TABLE) == w.DEFAULT_GB


LINES = {
    "tests/search/test_x.py 8": ("tests/search/test_x.py", 8, "N", None),
    "tests 40 A": ("tests", 40, "A", None),
    "tests/search/test_x.py 1 A 85G": ("tests/search/test_x.py", 1, "A", 85),
    "tests/search/test_x.py::test_big 1 90G": ("tests/search/test_x.py::test_big", 1, "N", 90),
}
BAD = ["tests/x.py 1 G", "tests/x.py 1 85", "tests/x.py 1 0G", "tests/x.py 1 85G A", "tests/x.py 1 A 8G 9G"]


def test_list_lines_parse_and_pack_accepts_exactly_the_same_lines():
    m = re.search(r"^LINE='(.*)'$", (ROOT / "tools/ops/day/pack.sh").read_text(encoding="utf-8"), re.M)
    line_re = re.compile(m.group(1).replace("[:space:]", r"\s"))
    for line, want in LINES.items():
        assert w.parse_line(line) == want
        assert line_re.match(line), line
    for line in BAD:
        assert not line_re.match(line), line
        with pytest.raises(ValueError):
            w.parse_line(line)


def test_tracked_defaults_load_with_one_entry_per_id():
    table = w.load(POD / "weights.txt")
    entries = [e for e, _ in table]
    assert len(entries) == len(set(entries))
    assert all(e.startswith("tests/") and gb >= 1 for e, gb in table)
    assert w.id_weight("tests/search/test_stage_b_budget.py::test_protected_line_matches_exhaustive", table) > 80
