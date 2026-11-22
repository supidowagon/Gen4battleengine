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
    """``run(argv, res=...) -> exit code`` with fake parts at a git-less checkout."""
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
    assert "seeds: 0-24 (25)" in last


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
    monkeypatch.setattr(sc, "rebuild_state", lambda saved, tables: state)
    monkeypatch.setattr(sc, "rebuild_outcome", lambda saved: None)
    monkeypatch.setattr(sc, "group_outputs", lambda *a: {})

    def steps(actions):
        inputs = [("seed 0 turn 0 segment 0", {"in_state": {}, "in_actions": actions,
                                               "in_outcome": {}})]  # fmt: skip
        with gzip.open(tmp_path / "0-0.inputs.pkl.gz", "wb") as f:
            pickle.dump(inputs, f)
        return sc.seed_steps(SimpleNamespace(tables=None), tmp_path, "compare", 0, 0, "rest")

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
    new = result(only_new=["in_outcome.roll"])
    assert sc.report(args(), "p", new, 1.0, set(), set(), "") == 1
    assert sc.report(args(), "p", new, 1.0, {"roll"}, set(), "") == 0


def test_a_dropped_field_without_a_row_fails_even_with_allow_new(capsys):
    """Rule 3: --allow-new admits only fields new in this checkout; a captured field this
    checkout lacks needs an UPGRADES row (or a "dropped, no successor" one)."""
    res = result(only_base=["in_outcome.legacy_field", ".jit_step.legacy_field"])
    assert sc.report(args(), "p", res, 1.0, {"legacy_field"}, set(), "") == 1
    assert "FAIL dropped field without an upgrade row: in_outcome.legacy_field" in (
        capsys.readouterr().out
    )
    rows = {u.name: u for u in sc.UPGRADES}
    acid = rows["state.acid_rain (): dropped, no successor"]
    assert acid.fill is None and acid.writes == () and "G-13" in acid.ledger
    assert "acid_rain" not in sc.upgrade_fields("state", {"acid_rain": np.int8(2)})
    assert all(u.reason and u.ledger for u in sc.UPGRADES)


def test_learn_pin_and_exp_refused_leave_by_their_rows(monkeypatch, capsys):
    """E3-B2-21 (review b2 #6): main's learn_pin and exp_refused have "dropped, no successor"
    rows, so a captured state holding them upgrades with no "dropped field" failure."""
    monkeypatch.setattr(sc, "UPGRADED", {})
    hp = np.ones(2, np.int16)
    old = {"species": hp, "active": np.zeros(2, np.int8), "hp": hp,
           "learn_pin": np.int8(2), "exp_refused": np.int8(2)}  # fmt: skip
    ((_, got),) = sc.upgrade_inputs([("k", {"in_state": old})], None)
    assert "learn_pin" not in got["in_state"] and "exp_refused" not in got["in_state"]
    got = sc.upgrade_entries(sc.canon({"jit_step": old}), None)
    assert not any(p.endswith(("learn_pin", "exp_refused")) for p in got)
    assert {".jit_step.species", ".jit_step.active", ".jit_step.hp"} <= set(got)
    assert {
        "state.learn_pin (): dropped, no successor",
        "state.exp_refused (): dropped, no successor",
    } <= set(sc.UPGRADED)
    assert sc.report(args(), "p", result(), 1.0, set(), set(), "") == 0
    assert "dropped field without an upgrade row" not in capsys.readouterr().out


# --- layout upgrades (an older commit's field shapes) ---------------------------------------

STARF_ROW, STARF_COLS = sc.UPGRADES[0].name, sc.UPGRADES[1].name


def test_old_shapes_upgrade_on_load(monkeypatch):
    """F-O3: an old int8[B] starf_roll and int8[B, 2] Outcome.starf load as int8[B, 3] (the draw
    in entry 0, then 0) and int8[B, 4] (columns 2-3 copy column 0), in the saved inputs and in
    the captured entries alike; a field already in the new shape is left alone."""
    monkeypatch.setattr(sc, "UPGRADED", {})
    roll, starf = np.array([7, -1], np.int8), np.array([[1, 2], [3, 4]], np.int8)
    want_roll = np.array([[7, 0, 0], [-1, 0, 0]], np.int8)
    want_starf = np.array([[1, 2, 1, 1], [3, 4, 3, 3]], np.int8)
    hp = np.ones(2, np.int16)
    inputs = [("k", {"in_state": {"starf_roll": roll, "hp": hp}, "in_actions": hp,
                     "in_outcome": {"starf": starf}}), ("a", {"in_ai_state": {"starf_roll": roll}})]  # fmt: skip
    (_, got), (_, ai) = sc.upgrade_inputs(inputs, None)
    np.testing.assert_array_equal(got["in_state"]["starf_roll"], want_roll)
    np.testing.assert_array_equal(got["in_outcome"]["starf"], want_starf)
    np.testing.assert_array_equal(ai["in_ai_state"]["starf_roll"], want_roll)
    assert got["in_state"]["hp"] is hp and got["in_actions"] is hp
    marks = {"hit": hp, "roll": hp, "crit": hp}
    entries = sc.canon({"outcome": {"starf": starf, **marks}, "x": hp})
    assert sc.upgrade_entries(entries, None) == sc.canon(
        {"outcome": {"starf": want_starf, **marks}, "x": hp}
    )
    assert sc.UPGRADED == {STARF_ROW: 2, STARF_COLS: 2}
    assert sc.upgrade_fields("state", {"starf_roll": want_roll})["starf_roll"] is want_roll
    # no Outcome marks: not an Outcome, left alone (the kind is part of the match)
    assert sc.upgrade_entries(sc.canon({"starf": starf}), None) == sc.canon({"starf": starf})
    assert sc.UPGRADED == {STARF_ROW: 2, STARF_COLS: 2}


def test_an_unknown_shape_mismatch_still_raises(monkeypatch):
    """Only a row's exact old dtype and shape upgrades: any other mismatch stays the error."""
    monkeypatch.setattr(sc, "UPGRADED", {})
    layout = {"starf_roll": ((2, 3), np.int8), "starf": ((2, 4), np.int8)}
    for name, kind, bad in (
        ("starf_roll", "state", np.zeros((2, 2), np.int8)),
        ("starf_roll", "state", np.zeros(2, np.int16)),
        ("starf", "outcome", np.zeros((2, 3), np.int8)),
    ):
        saved = sc.upgrade_fields(kind, {name: bad})
        assert saved[name] is bad
        with pytest.raises(ValueError, match=name):
            sc.check_layout(saved, layout)
        entries = sc.canon({name: bad})
        assert sc.upgrade_entries(entries, None) == entries
    assert sc.UPGRADED == {}


def test_compare_header_lists_the_upgrades_and_their_counts(run, capsys):
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "step"]) == 0
    capsys.readouterr()
    run.code = "code-b"
    res = {**result(), "upgraded": {STARF_ROW: 4, STARF_COLS: 2}}
    assert run(["compare", snap, "0-1", "--commit", NEW, "--parts", "step"], res=res) == 0
    out = capsys.readouterr().out
    assert f"layout upgrade: {STARF_ROW} (F-O3, G-27): the old draw is entry 0" in out
    assert f"layout upgrade: {STARF_COLS} (F-O3, G-27..G-30): columns 0-1 kept" in out
    assert f"  layout upgrades: {STARF_COLS} x 2, {STARF_ROW} x 4" in out
    assert f"layout upgrades run: {STARF_COLS} x 2, {STARF_ROW} x 4" in out


def test_every_escape_flag_defaults_off():
    ns = sc.build_parser().parse_args(["compare", "d", "0-1"])
    assert ns.allow_new is None and ns.allow_all_raise is None
    assert not (ns.subset_ok or ns.same_ok or ns.dirty_ok or ns.no_roark)
    assert (ns.raise_warn, ns.raise_fail) == (sc.RAISE_WARN, sc.RAISE_FAIL)
    assert sc.RAISE_WARN < sc.RAISE_FAIL < 1
    for action in sc.build_parser()._actions:  # no switch is on by default
        if action.nargs == 0 and action.dest != "help":
            assert action.default is False, action.dest


# --- lossless rebuild of an older capture (rules 1-4) ----------------------------------------


def _old_state(batch=()):
    """An older commit's state fields (0b12b1a's names), ``batch`` leading axes."""

    def z(*shape, dtype=np.int8):
        return np.zeros((*batch, *shape), dtype)

    s = {"species": z(2, 6, dtype=np.int16), "form": z(2, 6), "ability_form": z(2, 6),
         "ability_slot": z(2, 6), "item": z(2, 6, dtype=np.int16), "types": z(2, 6, 2),
         "active": z(2), "transformed": z(2), "hp": z(2, 6, dtype=np.int16), "uproar": z(2),
         "starf_roll": z(2), "weather": z(), "weather_timer": z(), "gravity": z(),
         "trick_room": z()}  # fmt: skip
    s["species"][..., 0] = 25
    return s


@pytest.fixture
def rows(monkeypatch):
    """No tables: Pikachu's personal types are (13, 13)."""
    monkeypatch.setattr(sc, "UPGRADED", {})
    monkeypatch.setattr(sc, "_personal_types", lambda tables, *mon: (13, 13))
    return sc


def test_field_rows_rebuild_the_mask_weather_turns_and_party_ability(rows):
    """Rule 1 (G-4, G-11): a TEMP weather keeps its timer as weather_turns and takes the TEMP
    bit, a permanent one (timer 0) the PERM bit; Gravity << 12, Trick Room << 16, Uproar bit
    8 + b; party_ability / nightmare / starf_taken 0; the old fields leave."""
    from nuzline.engine.ids import FieldCondition as F
    from nuzline.engine.ids import Weather

    s = _old_state()
    s["types"][:, 0] = 13
    s.update(weather=np.int8(Weather.RAIN), weather_timer=np.int8(3), gravity=np.int8(2),
             trick_room=np.int8(4), uproar=np.array([0, 2], np.int8))  # fmt: skip
    new = sc.upgrade_fields("state", s)
    assert int(new["field_mask"]) == F.RAINING_TEMP | 2 << 12 | 4 << 16 | 1 << 9
    assert new["field_mask"].dtype == np.int32 and int(new["weather_turns"]) == 3
    assert not {"weather", "weather_timer", "gravity", "trick_room"} & set(new)
    assert "uproar" in new
    for name, shape in (("party_ability", (2, 6)), ("nightmare", (2,)), ("starf_taken", (2,))):
        assert new[name].shape == shape and not new[name].any(), name
    assert new["party_ability"].dtype == np.int16
    s.update(weather_timer=np.int8(0), gravity=np.int8(0), trick_room=np.int8(0))
    s["uproar"][:] = 0
    new = sc.upgrade_fields("state", s)
    assert (int(new["field_mask"]), int(new["weather_turns"])) == (F.RAINING_PERM, 0)
    assert sc.upgrade_fields("state", new) == new  # a newer state: no row runs
    s.update(weather=np.int8(Weather.FOG), gravity=np.int8(1))  # deep fog: bit 15, one bit
    new = sc.upgrade_fields("state", s)
    assert int(new["field_mask"]) == 1 << 15 | 1 << 12 and F.DEEP_FOG == 1 << 15
    assert int(new["weather_turns"]) == 0


def test_camouflage_is_the_old_inference_but_not_on_a_transformed_battler(rows):
    """G-20: types off the personal row read as camouflaged in the old engine; a Transformed
    battler's copy no longer does."""
    s = _old_state()
    s["types"][:, 0] = 13
    s["types"][0, 0] = (0, 2)  # the player's active: Camouflaged
    s["types"][1, 0] = (0, 2)
    s["transformed"][1] = 1  # the foe: a Transformed copy
    assert sc.upgrade_fields("state", s)["camouflage"].tolist() == [1, 0]


def test_leech_seed_recipient_is_the_opposite_battler_where_seeded(rows):
    """D3-11b: an older capture has no ``leech_seed_recipient``; every older Leech Seed was set
    by the opposite battler (1 + (b ^ 1)), 0 where not seeded; ``leech_seed`` stays; a newer
    state (the field present) runs no row; batched states upgrade per row."""
    s = {"leech_seed": np.array([0, 1], np.int8)}
    new = sc.upgrade_fields("state", s)
    assert new["leech_seed_recipient"].tolist() == [0, 1] and new["leech_seed"].tolist() == [0, 1]
    assert new["leech_seed_recipient"].dtype == np.int8
    both = sc.upgrade_fields("state", {"leech_seed": np.array([1, 1], np.int8)})
    assert both["leech_seed_recipient"].tolist() == [2, 1]
    assert sc.upgrade_fields("state", new) == new
    batch = sc.upgrade_fields(
        "state", {"leech_seed": np.array([[1, 0], [0, 1]], np.int8)}, batched=True
    )
    assert batch["leech_seed_recipient"].tolist() == [[2, 0], [0, 1]]
    assert any("leech_seed_recipient" in name for name in sc.UPGRADED)


def _old_record(batch=()):
    """An older capture's AI record by player party slot, with ``active``, ``hp`` and
    ``ts_alive``."""
    s = {
        "ai_seen_moves": np.full((*batch, 6, 4), 3, np.int16),
        "ai_item_known": np.full((*batch, 6), 3, np.int16),
        "ai_ability_known": np.full((*batch, 6), 3, np.int16),
        "active": np.zeros((*batch, 2), np.int8),
        "hp": np.zeros((*batch, 2, 6), np.int16),
        "ts_alive": np.zeros((*batch, 2), np.int8),
    }
    return s


def test_ai_record_by_battler_upgrade(rows):
    """D45-1: row 0 = the old row of active[0] while it has HP or its faint is pending at a pivot
    pause (0 HP, ``ts_alive`` bit set: no CleanupFaintedMon yet), row 1 = 0; batched states row by
    row; a B = 4 state raises; a new-layout state runs no row."""
    row = sc.UPGRADES[-1]
    assert row.ledger == "D45-1"
    s = _old_record()
    s["active"][:] = (2, 1)
    s["hp"][0, 2] = 10
    s["ai_seen_moves"][2] = (5, 6, 0, 0)
    s["ai_item_known"][2] = 9
    s["ai_ability_known"][2] = 7
    new = sc.upgrade_fields("state", s)
    assert new["ai_seen_moves"].tolist() == [[5, 6, 0, 0], [0, 0, 0, 0]]
    assert new["ai_item_known"].tolist() == [9, 0] and new["ai_ability_known"].tolist() == [7, 0]
    assert new["ai_seen_moves"].dtype == np.int16 and "hp" in new and "active" in new
    s["hp"][0, 2] = 0
    new = sc.upgrade_fields("state", s)
    assert not any(np.asarray(new[f]).any() for f in sc.UPGRADES[-1].writes)
    s["ts_alive"][0] = 1 << 2  # an enemy U-turn fainted it; the faint check comes after the list
    pending = sc.upgrade_fields("state", s)
    assert pending["ai_item_known"].tolist() == [9, 0]
    assert pending["ai_seen_moves"].tolist() == [[5, 6, 0, 0], [0, 0, 0, 0]]
    s["ts_alive"][0] = 1 << 1  # another slot's bit: the active one was cleaned up
    assert not sc.upgrade_fields("state", s)["ai_item_known"].any()
    batch = _old_record((3,))
    batch["active"][:] = (1, 0)
    batch["hp"][:, 0, 1] = (4, 0, 4)
    batch["ai_item_known"][:, 1] = (11, 12, 13)
    got = sc.upgrade_fields("state", batch, batched=True)
    assert got["ai_item_known"].tolist() == [[11, 0], [0, 0], [13, 0]]
    assert got["ai_seen_moves"].shape == (3, 2, 4)
    assert row.lead(new) is None  # a new-layout dict ((2, 4)): no row
    four = _old_record()
    four["active"] = np.zeros(4, np.int8)
    with pytest.raises(ValueError, match="B = 2"):
        sc._ai_record_by_battler(four, (), None)


def test_camouflage_reads_the_tables_of_the_states_game(monkeypatch):
    """Review G MUST: the camouflage row compares a battler's types with the personal row of
    the game its state belongs to: a species whose types a game changed (here Kaizo's Pikachu
    as (13, 2); no built Kaizo row differs from vanilla today) is camouflaged on the vanilla
    tables, not on its own; no tables, no inference (it raises)."""
    monkeypatch.setattr(sc, "UPGRADED", {})
    games = {"vanilla": {25: (13, 13)}, "kaizo": {25: (13, 2)}}
    monkeypatch.setattr(sc, "_personal_types", lambda tables, species, *rest: tables[species])
    s = _old_state()
    s["types"][:, 0] = (13, 2)  # Kaizo's Pikachu as the trace saves it
    assert sc.upgrade_fields("state", s, tables=games["kaizo"])["camouflage"].tolist() == [0, 0]
    assert sc.upgrade_fields("state", s, tables=games["vanilla"])["camouflage"].tolist() == [1, 1]
    entries = sc.canon({"rb": s})
    got = sc.upgrade_entries(entries, games["kaizo"])
    assert np.frombuffer(got[".rb.camouflage"][3], np.int8).tolist() == [0, 0]
    monkeypatch.undo()
    monkeypatch.setattr(sc, "UPGRADED", {})
    with pytest.raises(RuntimeError, match="tables"):
        sc.upgrade_fields("state", s)


def test_each_part_upgrades_on_its_own_games_tables():
    """roark.trace (kaizo_roark_02) reads the Kaizo tables, loaded once; every other part the
    vanilla ones (roark.rank: roark_real_state)."""
    ctx = object.__new__(sc.Ctx)
    loads = []
    ctx.tables, ctx.kdir, ctx._kaizo = "vanilla", "kaizo-dir", None
    ctx._load = lambda d: loads.append(d) or f"tables of {d}"
    for part in ("0-9.step", "0-9.gen", "0-9.ai", "roark.rank"):
        assert ctx.tables_for(part) == "vanilla", part
    assert ctx.tables_for("roark.trace") == ctx.tables_for("roark.trace") == "tables of kaizo-dir"
    assert loads == ["kaizo-dir"]


def test_rows_run_on_batched_outputs_by_kind(rows):
    """Rule 2: leading axes (a batch of states, enumerate_turn's outcomes) upgrade; rows match
    a group of entries by its kind's marks, so an Outcome's starf never meets a state row."""
    from nuzline.engine.ids import FieldCondition as F

    s = _old_state((3,))
    s["types"][..., 0, :] = 13
    s["weather"][1], s["weather_timer"][1] = 1, 0  # sun, permanent
    starf = np.arange(3 * 2 * 2, dtype=np.int8).reshape(3, 2, 2)
    o = {"starf": starf, "hit": np.zeros((3, 2, 6), np.int8), "roll": np.zeros(3, np.int8),
         "crit": np.zeros(3, np.int8)}  # fmt: skip
    inv = {}
    got = sc.upgrade_entries(sc.canon({"rb": [s], "et": [o]}), None, inv)
    mask = np.frombuffer(got[".rb[0].field_mask"][3], np.int32)
    assert mask.tolist() == [0, F.SUNNY_PERM, 0]
    assert ".rb[0].weather" not in got and got[".rb[0].starf_roll"][2] == "(3, 2, 3)"
    assert got[".et[0].starf"][2] == "(3, 2, 4)" and inv == {".et[0].starf": (2, 3)}
    assert got[".rb[0].camouflage"][2] == "(3, 2)"
    # a saved input is one state: no leading axes there
    assert sc.upgrade_fields("state", {"starf_roll": np.zeros((2, 2), np.int8)})[
        "starf_roll"
    ].shape == (2, 2)


# main 08a57000's from_ref output entries (``ref_step``'s state), exactly as its code emitted
# them through ``canon``: from_ref stores every field in LAYOUT's dtype and shape
# (08a57000 reference/convert.py:53-75; state.py:741 _FIELD = (), :868-870 weather /
# weather_timer / trick_room int8, :801 uproar int8[B]), and canon writes a 0-d array as (1,)
# (np.ascontiguousarray, ndim >= 1). Generated with 08a57000's src on a singles state with
# weather 2, timer 3, gravity 4, Trick Room 1, uproar (0, 2). jit(step) / step_events write
# the same dtypes (08a57000 step.py:4133-4134 and :4326 astype; :4157 jnp.where of the int8
# counter and the weak Python int GRAVITY_TURNS stays int8), root_belief and the Roark parts
# return single states: every output state stores its 0-d fields as (1,).
MAIN_OLD_FIELDS = {
    "weather": ("arr", "|i1", "(1,)", b"\x02"),
    "weather_timer": ("arr", "|i1", "(1,)", b"\x03"),
    "gravity": ("arr", "|i1", "(1,)", b"\x04"),
    "trick_room": ("arr", "|i1", "(1,)", b"\x01"),
    "uproar": ("arr", "|i1", "(2,)", b"\x00\x02"),
    "acid_rain": ("arr", "|i1", "(1,)", b"\x00"),
    "species": ("arr", "<i2", "(2, 6)", bytes(24)),
    "active": ("arr", "|i1", "(2,)", bytes(2)),
    "hp": ("arr", "<i2", "(2, 6)", bytes(24)),
}
OUTPUT_PREFIXES = (".ref_step", ".step_events[0]", ".jit_step", ".root_belief[0][0]",
                   ".ref_replay", ".replay_turn", ".state", ".in_state", ".in_ai_state")  # fmt: skip


@pytest.mark.parametrize("prefix", OUTPUT_PREFIXES)
def test_field_row_runs_on_main_output_records(rows, prefix):
    """Task 1 (pod compare of 56cd9f4): the field row ran on the saved inputs only. A captured
    output state's 0-d fields are (1,) entries (canon) while uproar is (2,): the reads' leads
    differed ((1,) against ()), so Upgrade.lead returned None on every output (ref_step,
    step_events, jit_step, root_belief, ref_replay / replay_turn / state, the record's
    in_state / in_ai_state). Read flat, they are the 0-d fields and the row runs."""
    from nuzline.engine.ids import FieldCondition as F

    entries = {f"{prefix}.{k}": v for k, v in MAIN_OLD_FIELDS.items()}
    got = sc.upgrade_entries(entries, None)
    assert not {
        f"{prefix}.{k}" for k in ("weather", "weather_timer", "gravity", "trick_room")
    } & set(got)
    mask = got[f"{prefix}.field_mask"]
    want = F.RAINING_TEMP | 4 << 12 | 1 << 16 | 1 << 9  # Weather 2 = rain, timer 3
    assert mask[:3] == ("arr", "<i4", "(1,)") and struct.unpack("<i", mask[3])[0] == want
    assert got[f"{prefix}.weather_turns"] == ("arr", "<i4", "(1,)", struct.pack("<i", 3))
    assert got[f"{prefix}.uproar"] == MAIN_OLD_FIELDS["uproar"]
    assert sc.UPGRADED[sc.UPGRADES[2].name] == 1


def test_flat_scalars_keep_batches_and_newer_captures(rows):
    """A (1,) read is a batch of one where the other reads say so; a newer capture (a
    field_mask already there) and a scalar of another dtype still run no row."""
    one = {k: v for k, v in MAIN_OLD_FIELDS.items()}
    one.update(uproar=("arr", "|i1", "(1, 2)", b"\x00\x02"), species=("arr", "<i2", "(1, 2, 6)", bytes(24)),
               active=("arr", "|i1", "(1, 2)", bytes(2)), hp=("arr", "<i2", "(1, 2, 6)", bytes(24)))  # fmt: skip
    got = sc.upgrade_entries({f".rb.{k}": v for k, v in one.items()}, None)
    assert got[".rb.field_mask"][2] == "(1,)" and ".rb.weather" not in got
    newer = {f".s.{k}": v for k, v in MAIN_OLD_FIELDS.items()}
    newer[".s.field_mask"] = ("arr", "<i4", "(1,)", bytes(4))
    assert ".s.weather" in sc.upgrade_entries(newer, None)
    wide = {f".s.{k}": v for k, v in MAIN_OLD_FIELDS.items()}
    wide[".s.gravity"] = ("arr", "<i4", "(1,)", struct.pack("<i", 4))
    assert ".s.weather" in sc.upgrade_entries(wide, None)
    batch3 = {f".s.{k}": v for k, v in MAIN_OLD_FIELDS.items()}  # (1,) scalars, lead (3,): no
    batch3[".s.uproar"] = ("arr", "|i1", "(3, 2)", bytes(6))
    assert ".s.weather" in sc.upgrade_entries(batch3, None)


def test_sampler_outputs_drift_in_every_starf_column_saved_outcomes_in_2_3(capsys):
    """Task 2: main drew Outcome.starf as randint(fold_in(key, 48), (B, 2)) and this checkout
    draws (B, 4) from that key (outcome.py sample_outcome), so every column of a fresh draw
    (.sample_outcome) changes: all four are sampler drift. A saved Outcome (.in_outcome)
    invents only columns 2-3; its columns 0-1 stay strict."""
    old = np.array([[5, 1], [7, 2]], np.int8)
    marks = {"hit": np.zeros((2, 6), np.int8), "roll": np.zeros(2, np.int8),
             "crit": np.zeros(2, np.int8)}  # fmt: skip
    entries = sc.canon(
        {"sample_outcome": {"starf": old, **marks}, "in_outcome": {"starf": old, **marks}}
    )
    inv: dict = {}
    want = sc.upgrade_entries(entries, None, inv)
    assert inv == {".sample_outcome.starf": (0, 1, 2, 3), ".in_outcome.starf": (2, 3)}
    drawn = np.array([[9, 8, 7, 6], [5, 4, 3, 2]], np.int8)
    saved = np.array([[5, 1, 5, 5], [7, 2, 7, 7]], np.int8)
    got = sc.canon(
        {"sample_outcome": {"starf": drawn, **marks}, "in_outcome": {"starf": saved, **marks}}
    )
    assert sc.settle_new_columns(want, got, inv) == 1
    assert sc.compare_dumps(_step(want), [("k", got)])[0] is None
    saved[1, 0] = 0  # a saved Outcome's column 0 still diverges
    want = sc.upgrade_entries(entries, None, inv := {})
    got = sc.canon(
        {"sample_outcome": {"starf": drawn, **marks}, "in_outcome": {"starf": saved, **marks}}
    )
    sc.settle_new_columns(want, got, inv)
    diverged = sc.compare_dumps(_step(want), [("k", got)])[0]
    assert diverged and ".in_outcome.starf[1, 0]" in diverged


def test_new_starf_columns_are_sampler_drift_not_divergence(capsys):
    """Rule 4: Outcome.starf columns 2-3 of an old capture are invented: a rerun's own draws
    there are sampler drift; columns 0-1 stay strict."""
    old = np.array([[5, 1, 5, 5], [7, 2, 7, 7]], np.int8)
    new = old.copy()
    new[:, 2:] = 9
    want, got = sc.canon({"s": {"starf": old}}), sc.canon({"s": {"starf": new}})
    assert sc.settle_new_columns(want, got, {".s.starf": (2, 3)}) == 1
    assert sc.compare_dumps(_step(want), [("k", got)])[0] is None
    new[1, 0] = 0
    want, got = sc.canon({"s": {"starf": old}}), sc.canon({"s": {"starf": new}})
    sc.settle_new_columns(want, got, {".s.starf": (2, 3)})
    diverged = sc.compare_dumps(_step(want), [("k", got)])[0]
    assert diverged and ".s.starf[1, 0]" in diverged
    res = {**result(), "sampler_drift": 3}
    assert sc.report(args(), "0-0.sample", res, 1.0, set(), set(), "") == 0
    assert "sampler drift (new columns): 3 steps" in capsys.readouterr().out
    assert sc.part_class("0-0.sample", res, 0) == "sampler drift"


def test_starf_columns_0_1_diverge_even_under_sampler_drift(capsys):
    """Only Outcome.starf columns 2-3 are sampler drift: a step whose columns 2-3 drifted and
    whose column 0 or 1 differs is a divergence, and the part is diverged (a failure), not
    sampler drift."""
    old = np.array([[5, 1, 5, 5], [7, 2, 7, 7]], np.int8)
    for col in (0, 1):
        new = old.copy()
        new[:, 2:] = 9
        new[0, col] += 1
        want, got = sc.canon({"s": {"starf": old}}), sc.canon({"s": {"starf": new}})
        assert sc.settle_new_columns(want, got, {".s.starf": (2, 3)}) == 1
        diverged = sc.compare_dumps(_step(want), [("k", got)])[0]
        assert diverged and f".s.starf[0, {col}]" in diverged, col
    res = {**result(diverged=diverged), "sampler_drift": 1}
    assert sc.report(args(), "0-0.sample", res, 1.0, set(), set(), "") == 1
    assert sc.part_class("0-0.sample", res, 1) == "diverged"


def test_gen_divergence_is_generator_drift_and_the_closing_line_counts(run, capsys):
    """Rule 4: the "gen" part's divergence is counted as generator drift, not a failure; the
    final line counts ok / diverged / error / sampler drift / generator drift."""
    snap = str(run.snap)
    assert run(["capture", snap, "0-1", "--commit", BASE, "--parts", "gen,rest"]) == 0
    run.code = "code-b"
    capsys.readouterr()
    drift = {**result(diverged="seed 1 turn 3 segment 0: .in_actions[0]: 0 -> 2"),
             "generator_drift": True}  # fmt: skip
    argv = ["compare", snap, "0-1", "--commit", NEW, "--parts", "gen,rest"]
    assert run(argv, res=drift) == 1  # rest diverged with the same canned result
    out = capsys.readouterr().out
    assert "generator drift: first at seed 1 turn 3" in out
    assert "parts: 0 ok, 1 diverged, 0 error, 0 sampler drift, 1 generator drift" in out
    assert sc.report(args(), "0-1.gen", drift, 1.0, set(), set(), "") == 0
    hard = {**drift, "generator_drift": False}  # classify_gen found an entry no strategy draws
    assert sc.report(args(), "0-1.gen", hard, 1.0, set(), set(), "") == 1
    assert sc.part_class("0-1.gen", hard, 1) == "diverged"


def test_gen_drift_only_in_inputs_the_strategies_draw():
    """Generator drift is informational only when every differing "gen" entry is a drawn input
    (a segment's actions and Outcome, the AI state, segment 0's state); a later segment's
    state (read from the saved inputs, not drawn), any other entry, a raise or a different step
    list is a divergence."""
    a, b = np.zeros(2, np.int32), np.ones(2, np.int32)
    s0, s1, ai = "seed 1 turn 0 segment 0", "seed 1 turn 0 segment 1", "seed 1 turn 0 ai"

    def gen(changed):
        """(record, dump, diverged) of three steps; ``changed``: {(key, path)} that differ."""
        steps = {s0: (".in_actions", ".in_outcome.roll", ".in_state.hp"),
                 s1: (".in_actions", ".in_outcome.roll", ".in_state.hp"), ai: (".in_ai_state.hp",)}  # fmt: skip
        record = [(k, "d", {p: sc.canon(a)[""] for p in ps}) for k, ps in steps.items()]
        dump = [(k, {p: sc.canon(b if (k, p) in changed else a)[""] for p in ps})
                for k, ps in steps.items()]  # fmt: skip
        return record, dump, sc.compare_dumps(record, dump)[0]

    for changed in ({(s0, ".in_actions")}, {(s0, ".in_state.hp")}, {(s1, ".in_outcome.roll")},
                    {(ai, ".in_ai_state.hp")}, {(s0, ".in_state.hp"), (s1, ".in_actions")}):  # fmt: skip
        record, dump, diverged = gen(changed)
        assert diverged and sc.classify_gen(record, dump, diverged) == (diverged, True), changed
    record, dump, diverged = gen({(s0, ".in_actions"), (s1, ".in_actions"), (s1, ".in_state.hp")})
    hard, drift = sc.classify_gen(record, dump, diverged)
    assert not drift and hard.startswith(f"{s1}: .in_state.hp[0]") and "not an input" in hard
    record, dump, _ = gen(set())
    record[0][2][".other"], dump[0][1][".other"] = sc.canon(a)[""], sc.canon(b)[""]
    hard, drift = sc.classify_gen(record, dump, sc.compare_dumps(record, dump)[0])
    assert not drift and hard.startswith(f"{s0}: .other")
    record, dump, _ = gen(set())
    dump[2][1][".in_ai_state.hp"] = sc.canon(ValueError("x"))[""]
    hard, drift = sc.classify_gen(record, dump, sc.compare_dumps(record, dump)[0])
    assert not drift and "raised on" in hard
    record, dump, _ = gen({(s0, ".in_actions")})
    diverged = sc.compare_dumps(record, dump[:2])[0]
    assert sc.classify_gen(record, dump[:2], diverged) == (diverged, False)
    assert sc.classify_gen(record, dump, None) == (None, False)
