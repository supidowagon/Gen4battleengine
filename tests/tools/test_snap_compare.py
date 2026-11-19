"""tools/snap_compare.py's failure modes on fake captures: the child processes are replaced by
canned RESULT lines, so no engine is imported or compiled."""

import json
import struct
import subprocess
import sys
from pathlib import Path

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
    monkeypatch.setattr(sc, "content_hash", lambda r: go.code)  # go.code: the code under test

    def go(argv, res=None):
        line = "RESULT " + json.dumps(res or result())
        monkeypatch.setattr(sc, "run_part", lambda mode, d, part: (0, line + "\n", ""))
        return sc.parent(sc.build_parser().parse_args(argv))

    go.snap = tmp_path / "snap"
    go.code = "code-a"
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
    run.code = "code-b"  # compare runs other code
    rec = json.loads((run.snap / "capture.json").read_text())
    assert rec["commit"] == BASE and rec["seeds"] == [[0, 1, 10]]
    assert rec["content_hash"] == "code-a" and rec["dirty"] is None  # no git
    assert set(rec["parts"]) == {f"0-1.{g}" for g in sc.SEED_GROUPS} | set(sc.ROARK_PARTS)
    assert set(rec["parts"].values()) == {"ok"}
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 0
    assert run(["compare", snap, "0-1", "--commit", BASE]) == 1  # same commit
    assert run(["compare", snap, "0-1", "--commit", BASE, "--same-ok"]) == 0
    run.code = "code-a"  # the capture's code at another commit: still the same code
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1
    assert run(["compare", snap, "0-1", "--commit", NEW, "--same-ok"]) == 0
    run.code = "code-b"
    assert run(["compare", snap, "0-1", "--commit", NEW, "--no-roark"]) == 1  # strict subset
    assert run(["compare", snap, "0-1", "--commit", NEW, "--parts", "step"]) == 1
    assert run(["compare", snap, "0-1", "--commit", NEW, "--parts", "step", "--subset-ok"]) == 0
    assert run(["compare", snap, "0-3", "--commit", NEW]) == 1  # not captured
    assert run(["compare", snap, "0-1", "--chunk", "1", "--commit", NEW]) == 1


def test_compare_needs_a_commit_on_both_sides(run):
    snap = str(run.snap)
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1  # no capture.json
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 0
    run.code = "code-b"
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
    run.code = "code-b"
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
    run.code = "code-b"
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


# --- M1: the content hash ties the code to the capture ----------------------------------------


def _tree(root):
    files = {
        "src/nuzline/a.py": "x = 1\n",
        "src/nuzline/sub/b.py": "y = 1\n",
        "tools/t.py": "t = 1\n",
        "tests/__init__.py": "\n",
        "tests/strategies.py": "s = 1\n",
        "tests/ai_legality.py": "g = 1\n",
        "tests/engine/__init__.py": "\n",
        "tests/engine/helpers.py": "h = 1\n",
        "tests/ai/test_distribution.py": "d = 1\n",
        "tests/fixtures/sample_party.toml": "[party]\n",
        "tests/golden/kaizo_roark_02.json": "{}\n",
        "data/gen4_platinum/manifest.json": '{"version": 1}\n',
        **{f"data/intermediate/{n}.json": "[1, 2]\n" for n in sc.INTERMEDIATE},
        **{f"data/intermediate_kaizo/{n}.json": "[3]\n" for n in sc.INTERMEDIATE},
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for d in ("gen4_platinum", "gen4_platinum_kaizo"):
        (root / "data" / d).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "data" / d / "tables.npz", a=np.arange(3), b=np.ones(2))
    return files


def test_content_hash_covers_what_the_child_runs(tmp_path):
    files = _tree(tmp_path)
    base = sc.content_hash(tmp_path)
    assert base == sc.content_hash(tmp_path)
    (tmp_path / "README.md").write_text("not part of the hash")
    (tmp_path / "tests" / "other.py").write_text("o = 1\n")
    (tmp_path / "tools" / "snap_compare.py").write_text("# not the script that runs\n")
    assert sc.content_hash(tmp_path) == base
    for rel in files:
        path = tmp_path / rel
        path.write_text("[9]\n" if rel.endswith(".json") else files[rel] + "# edit\n")
        assert sc.content_hash(tmp_path) != base, rel
        path.write_text(files[rel])
        assert sc.content_hash(tmp_path) == base, rel
    (tmp_path / "src" / "nuzline" / "new.py").write_text("n = 1\n")
    assert sc.content_hash(tmp_path) != base
    (tmp_path / "src" / "nuzline" / "new.py").unlink()
    (tmp_path / "tools" / "a.py").write_bytes(b"a = 1\r\n")
    crlf = sc.content_hash(tmp_path)
    (tmp_path / "tools" / "a.py").write_bytes(b"a = 1\n")
    assert sc.content_hash(tmp_path) == crlf  # a git archive with other line endings


def test_content_hash_reads_tables_by_content(tmp_path):
    """A rebuild of the same tables (other zip timestamps, order, compression, JSON layout)
    hashes equal; another array, dtype, shape or value does not."""
    _tree(tmp_path)
    base = sc.content_hash(tmp_path)
    npz = tmp_path / "data" / "gen4_platinum" / "tables.npz"
    raw = npz.read_bytes()
    np.savez(npz, b=np.ones(2), a=np.arange(3))  # uncompressed, other member order
    assert npz.read_bytes() != raw and sc.content_hash(tmp_path) == base
    for arrays in (
        {"a": np.arange(3), "b": np.ones(2), "c": np.zeros(1)},
        {"a": np.arange(3, dtype=np.int16), "b": np.ones(2)},
        {"a": np.arange(3).reshape(1, 3), "b": np.ones(2)},
        {"a": np.arange(3), "b": np.ones(2) * 2},
    ):
        np.savez_compressed(npz, **arrays)
        assert sc.content_hash(tmp_path) != base, arrays
    np.savez_compressed(npz, a=np.arange(3), b=np.ones(2))
    manifest = tmp_path / "data" / "gen4_platinum" / "manifest.json"
    manifest.write_bytes(b'{\r\n  "version":   1\r\n}')
    assert sc.content_hash(tmp_path) == base


def test_content_hash_covers_the_running_script(tmp_path, monkeypatch):
    """The pod captures BASE with NEW's script: the script that runs is hashed, not BASE's."""
    _tree(tmp_path)
    base = sc.content_hash(tmp_path)
    running = tmp_path / "elsewhere" / "snap_compare.py"
    running.parent.mkdir()
    running.write_bytes(Path(sc.__file__).read_bytes())
    monkeypatch.setattr(sc, "__file__", str(running))
    assert sc.content_hash(tmp_path) == base
    running.write_text("# another script\n")
    assert sc.content_hash(tmp_path) != base


def test_capture_without_git_records_not_checkable(run, capsys):
    assert run(["capture", str(run.snap), "0-1", "--commit", BASE]) == 0
    out = capsys.readouterr().out
    assert "not checkable" in out and "content hash code-a" in out
    assert json.loads((run.snap / "capture.json").read_text())["dirty"] is None
    run.code = "code-b"
    assert run(["compare", str(run.snap), "0-1", "--commit", NEW]) == 0
    assert "not checkable" in capsys.readouterr().out


def test_equal_hashes_fail_and_print_both(run, capsys):
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 0
    capsys.readouterr()
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1
    out = capsys.readouterr().out
    assert "same content hash" in out and out.count("code-a") >= 3  # both hashes, the reason


def test_capture_with_other_code_into_a_used_folder_fails(run):
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "step"]) == 0
    run.code = "code-b"
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "rest"]) == 1


@pytest.mark.parametrize(
    "edit", [{"drop": "content_hash"}, {"drop": "dirty"}, {"drop": "format"}, {"format": 1}]
)
def test_an_older_capture_is_refused_with_re_capture(run, capsys, edit):
    """A capture.json without ``content_hash`` or ``dirty`` (or another format) was written by
    an older snap_compare: compare and a re-capture into its folder refuse it and say to
    re-capture; a missing ``dirty`` key is never read as "not checkable"."""
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 0
    rec = json.loads((run.snap / "capture.json").read_text())
    if "drop" in edit:
        del rec[edit["drop"]]
    else:
        rec.update(edit)
    (run.snap / "capture.json").write_text(json.dumps(rec))
    capsys.readouterr()
    run.code = "code-b"
    assert run(["compare", snap, "0-1", "--commit", NEW]) == 1
    out = capsys.readouterr().out
    assert "older snap_compare" in out and "re-capture" in out
    assert "dirty state not checkable" not in out
    if edit == {"drop": "dirty"}:  # the capture side is shown without a dirty mark
        assert f"capture commit {BASE};" in out
    run.code = "code-a"
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "rest"]) == 1
    assert "re-capture" in capsys.readouterr().out


def test_dirty_also_covers_strategies_and_helpers(tmp_path):
    repo = tmp_path / "r"
    for rel in ("src/a.py", "tests/strategies.py", "tests/engine/helpers.py", "tests/ai/x.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    assert sc.git_info(repo)["dirty"] is False
    for rel in ("tests/strategies_extra.py", "tests/engine/helpers.py", "tests/ai/y.py"):
        path = repo / rel
        existed = path.exists()
        path.write_text("x = 2\n")
        assert sc.git_info(repo)["dirty"] is True, rel
        if existed:
            _git(repo, "checkout", "--", rel)
        else:
            path.unlink()
        assert sc.git_info(repo)["dirty"] is False, rel
    (repo / "tests" / "unrelated.py").write_text("u = 1\n")  # untracked, outside the list
    assert sc.git_info(repo)["dirty"] is False


# --- N1: the final line ---------------------------------------------------------------------


def test_final_line_names_groups_and_seed_count(run, capsys):
    snap = str(run.snap)
    assert run(["capture", snap, "0-24", "--commit", BASE, "--parts", "step,rest"]) == 0
    capsys.readouterr()
    run.code = "code-b"
    argv = ["compare", snap, "0-24", "--commit", NEW, "--parts", "step,rest"]
    assert run(argv) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("compare: ok") and "groups: step, rest" in last
    assert "seeds 0-24 (25)" in last


def test_final_line_without_a_seed_part_names_no_seeds(run, capsys):
    roark = ["--parts", "roark.trace,roark.rank"]
    assert run(["capture", str(run.snap), "0-199", "--commit", BASE, *roark]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert "groups: roark.trace, roark.rank;" in last and "seeds: none" in last
    assert "0-199" not in last and "(200)" not in last


@pytest.mark.parametrize(
    "warn, fail", [("nan", "0.9"), ("0.5", "nan"), ("0.5", "1"), ("0.5", "1.5"), ("0.9", "0.5"),
                   ("-0.1", "0.5")]
)  # fmt: skip
def test_bad_raise_shares_are_usage_errors(run, warn, fail):
    argv = ["capture", str(run.snap), "0-1", "--commit", BASE]
    assert run([*argv, "--raise-warn", warn, "--raise-fail", fail]) == 2
    assert not run.snap.exists()
    assert run([*argv, "--raise-warn", "0", "--raise-fail", "0"]) == 0


# --- N2, R5, N3: a part that ran nothing, a high raise share ----------------------------------


def test_zero_steps_or_calls_fail(capsys):
    zero = result(calls=0)
    assert sc.report(args(), "0-0.rest", zero, 1.0, set(), set(), "") == 1
    assert "zero calls" in capsys.readouterr().out
    assert sc.report(args(), "0-0.gen", zero, 1.0, set(), set(), "") == 0  # the check has none
    assert sc.report(args(), "0-0.rest", {**result(), "steps": 0}, 1.0, set(), set(), "") == 1
    assert "zero steps" in capsys.readouterr().out
    assert sc.report(args(), "0-0.gen", {**result(), "steps": 0}, 1.0, set(), set(), "") == 1


def test_raise_share_warns_then_fails(capsys):
    def res(n):  # n of 40 calls raised, spread over two functions
        return result(calls=40, raised={"a": [n // 2, 20, "E: x"], "b": [n - n // 2, 20, "E: y"]})

    assert sc.report(args(), "p", res(16), 1.0, set(), set(), "") == 0  # 0.4
    assert "WARN" not in capsys.readouterr().out
    assert sc.report(args(), "p", res(24), 1.0, set(), set(), "") == 0  # 0.6 warns
    assert "WARN raise share" in capsys.readouterr().out
    assert sc.report(args(), "p", res(40), 1.0, set(), set(), "") == 1  # 1.0 fails
    assert "FAIL raise share" in capsys.readouterr().out
    ok = args("--raise-warn", "0.9", "--raise-fail", "0.99")
    assert sc.report(ok, "p", res(36), 1.0, set(), set(), "") == 0  # 0.9 is not above 0.9
    assert "WARN" not in capsys.readouterr().out
    assert sc.report(ok, "p", res(38), 1.0, set(), set(), "") == 0
    assert "WARN" in capsys.readouterr().out
    # a function allowed to raise on every call leaves the share
    allowed = result(calls=10, raised={"a": [10, 10, "E: x"]})
    assert sc.report(args(), "p", allowed, 1.0, set(), {"a"}, "") == 0
    assert "WARN" not in capsys.readouterr().out


# --- N4: shapes and dtypes of the Outcome and the actions ---------------------------------------


def test_check_actions():
    sc.check_actions(np.zeros(2, np.int32), 2)
    for bad in (np.zeros(2, np.int64), np.zeros(4, np.int32), np.zeros((2, 1), np.int32)):
        with pytest.raises(ValueError, match="actions"):
            sc.check_actions(bad, 2)


def test_seed_steps_checks_the_saved_actions(tmp_path, monkeypatch):
    """Saved int64 actions fail the part (seed_steps runs check_actions on each input)."""
    import gzip
    import pickle
    from types import SimpleNamespace

    state = SimpleNamespace(active=np.zeros(2, np.int32))
    monkeypatch.setattr(sc, "rebuild_state", lambda saved: state)
    monkeypatch.setattr(sc, "rebuild_outcome", lambda saved: None)
    monkeypatch.setattr(sc, "group_outputs", lambda *a: {})

    def steps(actions):
        inputs = [("seed 0 turn 0 segment 0", {"in_state": {}, "in_actions": actions,
                                               "in_outcome": {}})]  # fmt: skip
        with gzip.open(tmp_path / "0-0.inputs.pkl.gz", "wb") as f:
            pickle.dump(inputs, f)
        return sc.seed_steps(None, tmp_path, "compare", 0, 0, "rest")

    assert len(steps(np.zeros(2, np.int32))) == 1
    with pytest.raises(ValueError, match="actions"):
        steps(np.zeros(2, np.int64))


def _saved_outcome():
    import jax

    from nuzline.engine.outcome import empty_outcome

    return sc.plain(jax.device_get(empty_outcome()))


def test_outcome_shape_and_dtype_changes_are_errors(monkeypatch):
    saved = _saved_outcome()
    monkeypatch.setitem(sc.INPUT_ONE_SIDED, "base", set())
    monkeypatch.setitem(sc.INPUT_ONE_SIDED, "new", set())
    sc.rebuild_outcome(saved)
    for name, bad in (
        ("roll", saved["roll"].astype(np.int32)),
        ("hit", saved["hit"][:1]),
    ):
        with pytest.raises(ValueError, match=name):
            sc.rebuild_outcome({**saved, name: bad})


# --- N5: NaN payloads ---------------------------------------------------------------------


def test_floats_compare_by_ieee_bytes():
    """Python and np.float64 scalars compare by their bytes, as arrays do: one value in either
    type is equal; NaN payloads and signed zeros differ."""
    nan1 = float("nan")
    nan2 = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]  # another payload
    assert np.isnan(nan2)
    for a, b in ((1.5, np.float64(1.5)), (nan1, np.float64(nan1)), (nan2, np.float64(nan2))):
        assert sc.canon({"x": a, "y": [a]}) == sc.canon({"x": b, "y": [b]})
        assert sc.first_diff(sc.canon({"x": a}), sc.canon({"x": b})) is None
    for a, b in ((nan1, nan2), (np.float64(nan1), np.float64(nan2)), (0.0, -0.0), (nan1, 1.0)):
        diff = sc.first_diff(sc.canon({"x": a}), sc.canon({"x": b}))
        assert diff is not None and diff.startswith(".x:"), (a, b)


# --- N6: raise against a result ---------------------------------------------------------------


def _step(entries):
    return [("k", "d", entries)]


def test_raise_against_structure_is_a_divergence_allow_new_cannot_pass(capsys):
    boom = sc.canon({"f": ValueError("boom")})
    tree = sc.canon({"f": {"new_field": np.zeros(2, np.int8), "other": 1}})
    value = sc.canon({"f": np.zeros(2, np.int8)})
    for want, got in ((boom, tree), (tree, boom), (boom, value), (value, boom)):
        diverged, _, _ = sc.compare_dumps(_step(want), [("k", got)])
        assert diverged and "raised" in diverged
    assert sc.compare_dumps(_step(boom), [("k", boom)]) == (None, set(), set())
    res = result(diverged=diverged, only_new=[".f.new_field"])
    assert sc.report(args(), "p", res, 1.0, {"new_field", "f"}, set(), "") == 1
    assert "DIVERGED" in capsys.readouterr().out


def test_one_sided_entries_still_pass_with_allow_new():
    a = sc.canon({"f": np.zeros(2, np.int8)})
    b = sc.canon({"f": np.zeros(2, np.int8), "new_field": np.zeros(1, np.int8)})
    assert sc.compare_dumps(_step(a), [("k", b)]) == (None, set(), {".new_field"})


def test_step_list_difference_is_a_divergence():
    diverged, _, _ = sc.compare_dumps(_step({}), [("other", {})])
    assert "step list differs" in diverged


# --- N8: a crashing child, a SKIPPED part, a one-sided input field, escape flags -----------------


def test_crashing_child_fails_and_marks_the_part(run, monkeypatch, capsys):
    snap = str(run.snap)
    monkeypatch.setattr(
        sc, "run_part", lambda m, d, p: (2, "", "Traceback (most recent call last):\nBoom")
    )
    assert sc.parent(sc.build_parser().parse_args(["capture", snap, "0-1", "--commit", BASE])) == 1
    out = capsys.readouterr().out
    assert "ERROR (exit 2" in out and "Boom" in out
    assert set(json.loads((run.snap / "capture.json").read_text())["parts"].values()) == {"failed"}


def test_a_child_that_cannot_import_the_repo_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sc, "ROOT", sc.ROOT)
    argv = ["_child", "capture", str(tmp_path / "d"), "0-0.step", str(tmp_path)]
    assert sc.main(argv) == 2
    assert "nuzline imported from" in capsys.readouterr().err


def test_skipped_part_fails(run, monkeypatch, capsys):
    monkeypatch.setattr(sc, "missing", lambda part: ["data/x/tables.npz"])
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE]) == 1
    out = capsys.readouterr().out
    assert "SKIPPED: needs data/x/tables.npz" in out and "skipped (each a failure): 0-1.step" in out
    assert set(json.loads((run.snap / "capture.json").read_text())["parts"].values()) == {"failed"}


def test_a_one_sided_input_field_is_reported_and_fails(monkeypatch):
    saved = _saved_outcome()
    old = dict(saved)
    old.pop("roll")  # the capture lacks a field this checkout has
    old["legacy_field"] = np.zeros(1, np.int8)  # ... and has one this checkout lacks
    monkeypatch.setitem(sc.INPUT_ONE_SIDED, "base", set())
    monkeypatch.setitem(sc.INPUT_ONE_SIDED, "new", set())
    out = sc.rebuild_outcome(old)
    assert out.roll.shape == saved["roll"].shape and not hasattr(out, "legacy_field")
    assert sc.INPUT_ONE_SIDED["new"] == {"in_outcome.roll"}
    assert sc.INPUT_ONE_SIDED["base"] == {"in_outcome.legacy_field"}
    res = result(only_base=["in_outcome.legacy_field"], only_new=["in_outcome.roll"])
    assert sc.report(args(), "p", res, 1.0, set(), set(), "") == 1
    assert sc.report(args(), "p", res, 1.0, {"legacy_field", "roll"}, set(), "") == 0


def test_every_escape_flag_defaults_off():
    ns = sc.build_parser().parse_args(["compare", "d", "0-1"])
    assert ns.allow_new is None and ns.allow_all_raise is None
    assert not (ns.subset_ok or ns.same_ok or ns.dirty_ok or ns.no_roark)
    assert (ns.raise_warn, ns.raise_fail) == (sc.RAISE_WARN, sc.RAISE_FAIL)
    assert sc.RAISE_WARN < sc.RAISE_FAIL < 1
    for action in sc.build_parser()._actions:  # no switch is on by default
        if action.nargs == 0 and action.dest != "help":
            assert action.default is False, action.dest
