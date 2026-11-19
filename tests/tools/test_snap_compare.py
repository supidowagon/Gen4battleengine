"""tools/snap_compare.py's failure modes on fake captures: the child processes are replaced by
canned RESULT lines, so no engine is imported or compiled."""

import json
import subprocess

import numpy as np
import pytest

from tools import snap_compare as sc

BASE, NEW = "a" * 40, "b" * 40


def result(raised=None, calls=4, diverged=None, only_base=(), only_new=()):
    return {
        "steps": 2,
        "diverged": diverged,
        "only_base": list(only_base),
        "only_new": list(only_new),
        "calls": calls,
        "raised": raised or {},
        "peak_bytes": 1,
    }


@pytest.fixture
def run(tmp_path, monkeypatch):
    """``run(argv, res=...) -> (exit, printed)`` with fake parts at a git-less checkout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(sc, "ROOT", repo)
    monkeypatch.setattr(sc, "missing", lambda part: [])
    monkeypatch.setattr(sc, "git_info", lambda r: None)

    def go(argv, res=None):
        line = "RESULT " + json.dumps(res or result())
        monkeypatch.setattr(sc, "run_part", lambda mode, d, part: (0, line + "\n", ""))
        return sc.parent(sc.build_parser().parse_args(argv))

    go.snap = tmp_path / "snap"
    return go


def args(*extra):
    return sc.build_parser().parse_args(["compare", "d", "0-1", *extra])


# --- part list (MUST 1) ------------------------------------------------------------------


@pytest.mark.parametrize(
    "seeds, chunk, parts, no_roark",
    [
        ("0-9", 10, "stepp", False),  # unknown name
        ("0-9", 10, "step,roark", False),  # unknown name beside a known one
        ("5-2", 10, None, False),  # B < A
        ("0-9", 0, None, False),  # chunk < 1
        ("0-9", 10, "roark.trace", True),  # nothing left
        ("0-9", 10, "", False),  # empty list
        ("x", 10, None, False),
    ],
)
def test_bad_part_lists_are_usage_errors(seeds, chunk, parts, no_roark):
    with pytest.raises(sc.UsageError):
        sc.plan_parts(seeds, chunk, parts, no_roark)


def test_part_list():
    assert sc.plan_parts("0-14", 10, "step,roark.rank", False) == (
        0,
        14,
        ["0-9.step", "10-14.step", "roark.rank"],
    )
    assert len(sc.plan_parts("3", 10, None, False)[2]) == 8
    assert sc.plan_parts("0-14", 10, "ai", True) == (0, 14, ["0-9.ai", "10-14.ai"])


def test_usage_error_exits_2(run):
    assert run(["capture", str(run.snap), "0-1", "--commit", BASE, "--parts", "nope"]) == 2
    assert run(["capture", str(run.snap), "0-1", "--commit", BASE, "--chunk", "0"]) == 2
    assert not run.snap.exists()


# --- all-raise per function (MUST 2) -------------------------------------------------------


def test_one_function_raising_on_every_call_fails(capsys):
    res = result(raised={"enumerate_turn": [2, 2, "E: x"], "root_belief": [1, 2, "E: y"]})
    assert sc.report(args(), "0-0.rest", res, 1.0, set(), set(), "") == 1
    assert "FAIL raised on every call: enumerate_turn" in capsys.readouterr().out
    assert sc.report(args(), "0-0.rest", res, 1.0, set(), {"enumerate_turn"}, "") == 0
    some = result(raised={"root_belief": [1, 2, "E: y"]})
    assert sc.report(args(), "0-0.rest", some, 1.0, set(), set(), "") == 0


def test_one_sided_and_diverged_fail():
    assert sc.report(args(), "p", result(diverged="k: 1 -> 2"), 1.0, set(), set(), "") == 1
    one = result(only_new=[".jit_step.new_field"])
    assert sc.report(args(), "p", one, 1.0, set(), set(), "") == 1
    assert sc.report(args(), "p", one, 1.0, {"new_field"}, set(), "") == 0


# --- capture.json, subset, same commit, folder reuse (3, 5, nit) ----------------------------


def test_capture_then_compare(run):
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 0
    rec = json.loads((run.snap / "capture.json").read_text())
    assert rec["commit"] == BASE and rec["seeds"] == [[0, 1, 10]]
    assert set(rec["parts"]) == {f"0-1.{g}" for g in sc.SEED_GROUPS} | set(sc.ROARK_PARTS)
    assert set(rec["parts"].values()) == {"ok"}
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 0
    assert run(["compare", snap, "0-1", "--commit", BASE]) == 1  # same commit
    assert run(["compare", snap, "0-1", "--commit", BASE, "--same-ok"]) == 0
    assert run(["compare", snap, "0-1", "--commit", NEW, "--no-roark"]) == 1  # strict subset
    assert run(["compare", snap, "0-1", "--commit", NEW, "--parts", "step"]) == 1
    assert run(["compare", snap, "0-1", "--commit", NEW, "--parts", "step", "--subset-ok"]) == 0
    assert run(["compare", snap, "0-3", "--commit", NEW]) == 1  # not captured
    assert run(["compare", snap, "0-1", "--chunk", "1", "--commit", NEW]) == 1


def test_compare_needs_a_commit_on_both_sides(run):
    snap = str(run.snap)
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1  # no capture.json
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 0
    assert run(["compare", snap, "0-1"]) == 1  # no git, no --commit
    assert run(["compare", snap, "0-1", "--commit", "b" * 7]) == 1  # not a full commit
    rec = json.loads((run.snap / "capture.json").read_text())
    del rec["parts"]
    (run.snap / "capture.json").write_text(json.dumps(rec))
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1  # no part list


def test_failed_capture_part_fails_compare(run):
    snap = str(run.snap)
    bad = result(raised={"jit_step": [4, 4, "E: x"]})
    assert run(["capture", snap, "0-1", "--commit", BASE], res=bad) == 1
    assert set(json.loads((run.snap / "capture.json").read_text())["parts"].values()) == {"failed"}
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1


def test_recapture_never_mixes_commits(run):
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "step"]) == 0
    assert run(["capture", snap, "0-1", "--commit", NEW, "--parts", "rest"]) == 1
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "rest"]) == 0
    rec = json.loads((run.snap / "capture.json").read_text())
    assert rec["commit"] == BASE and set(rec["parts"]) == {"0-1.step", "0-1.rest"}
    assert rec["seeds"] == [[0, 1, 10], [0, 1, 10]]
    other = run.snap.parent / "other"
    other.mkdir()
    (other / "stray.txt").write_text("x")
    assert run(["capture", str(other), "0-1", "--commit", BASE]) == 1  # no capture.json


def test_capture_without_commit_fails(run):
    assert run(["capture", str(run.snap), "0-1"]) == 1
    assert not run.snap.exists()


# --- git: dirty checkouts and folders inside another repository (4, 5) ----------------------


def _git(repo, *a):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
        check=True,
        capture_output=True,
    )


def test_git_commit_and_dirty(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    info = sc.git_info(repo)
    assert info["dirty"] is False and sc.SHA.fullmatch(info["commit"])
    here, problems = sc.identify(repo, None)
    assert here["commit"] == info["commit"] and not problems
    assert sc.identify(repo, BASE)[1]  # --commit contradicts git
    (repo / "notes.txt").write_text("untracked outside src/ and tools/\n")
    assert sc.git_info(repo)["dirty"] is False
    (repo / "src" / "b.py").write_text("y = 2\n")  # untracked under src/
    assert sc.git_info(repo)["dirty"] is True
    (repo / "src" / "b.py").unlink()
    (repo / "src" / "a.py").write_text("x = 2\n")  # tracked change
    here, problems = sc.identify(repo, None)
    assert here["dirty"] and any(p.startswith("dirty:") for p in problems)
    assert sc.git_info(repo / "src") is None  # a folder inside a repository is not one
    assert sc.identify(repo / "src", None)[1]  # ... so it needs --commit


def test_dirty_fails_both_sides_unless_dirty_ok(run, monkeypatch):
    snap = str(run.snap)
    dirty = {"commit": BASE, "dirty": True, "changes": [" M src/a.py"]}
    monkeypatch.setattr(sc, "git_info", lambda r: dirty)
    assert run(["capture", snap, "0-1"]) == 1
    assert run(["capture", snap, "0-1", "--dirty-ok"]) == 0
    monkeypatch.setattr(sc, "git_info", lambda r: {**dirty, "commit": NEW, "dirty": False})
    assert run(["compare", snap, "0-1"]) == 1  # the capture side was dirty
    assert run(["compare", snap, "0-1", "--dirty-ok"]) == 0


# --- helpers (6, nits) ----------------------------------------------------------------------


def test_first_diff_on_negative_zero():
    want = sc.canon({"x": np.array([1.0, 0.0])})
    got = sc.canon({"x": np.array([1.0, -0.0])})
    diff = sc.first_diff(want, got)
    assert diff.startswith(".x[1]: ") and "-0.0" in diff
    grid = np.zeros((2, 3), np.float32)
    flipped = grid.copy()
    flipped[1, 2] = -0.0
    assert sc.first_diff(sc.canon(grid), sc.canon(flipped)).startswith("[1, 2]: ")


def test_check_layout():
    layout = {"hp": ((2, 6), np.int16), "turn": ((), np.int8)}
    sc.check_layout({"hp": np.zeros((2, 6), np.int16), "turn": np.int8(0), "x": 1}, layout)
    with pytest.raises(ValueError, match="hp"):
        sc.check_layout({"hp": np.zeros((2, 6), np.int32)}, layout)
    with pytest.raises(ValueError, match="turn"):
        sc.check_layout({"turn": np.zeros((1,), np.int8)}, layout)


def test_check_imported_from(tmp_path):
    sc.check_imported_from(str(tmp_path / "src" / "nuzline" / "__init__.py"), tmp_path)
    with pytest.raises(RuntimeError):
        sc.check_imported_from(str(tmp_path / "other" / "nuzline" / "__init__.py"), tmp_path)
