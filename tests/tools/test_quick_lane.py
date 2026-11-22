"""The quick lane (tools/ops/day/DAYBOX.md "Quick lane"): pack.sh --delta, pod/lane.sh, pod/quick.sh.
Tiny temp git repos driven with bash and stub uv, flock and ps; no engine import, no network."""

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.tools.test_pod_gate import BASH, _clean_env

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / "tools" / "ops" / "day"
POD = DAY / "pod"
PY = Path(sys.executable).as_posix()

pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _p(path: Path | str) -> str:
    """A path as Git Bash writes it (/c/...): GNU tar reads 'C:/...' as a remote host."""
    s = Path(path).resolve().as_posix()
    return f"/{s[0].lower()}{s[2:]}" if re.match(r"^[A-Za-z]:/", s) else s


def _pat(name: str, text: str) -> str:
    m = re.search(rf"^{name}='(.*)'$", text, re.MULTILINE)
    assert m, name
    return m.group(1)


PATTERNS = {
    "FORBID": _pat("FORBID", (DAY / "pack.sh").read_text(encoding="utf-8")),
    "BADPATH": _pat("BADPATH", (DAY / "pack.sh").read_text(encoding="utf-8")),
    "TABLE_DENY": _pat("TABLE_DENY", (ROOT / "tools/ops/common.sh").read_text(encoding="utf-8")),
}
STUBS = {
    "flock": "#!/bin/sh\nexit 0\n",
    "ps": "#!/bin/sh\nexit 0\n",
    # uv sync: nothing; uv run python: the test's python; uv run pytest: FAKE_PYTEST
    "uv": '#!/bin/bash\ncase "$1" in sync) exit 0;; run) shift; [ "$1" = --frozen ] && shift\n'
    '  case "$1" in python) shift; exec "$PYEXE" "$@";; pytest) shift; exec bash "$FAKE_PYTEST" "$@";; esac;; esac\nexit 9\n',
}
FAKE_PYTEST = (
    'echo "$@" > "$(dirname "$0")/pytest.args"\n[ "${FAKE_PYTEST_RC:-0}" = 0 ] && { echo "1 passed in 0.01s"; exit 0; }\n'
    'echo "_____ test_a _____"; echo "E   assert 1 == 2"; echo "FAILED tests/x.py::test_a - assert 1 == 2"\n'
    'echo "1 failed in 0.02s"; exit "$FAKE_PYTEST_RC"\n'
)


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@localhost",
            "-c",
            "core.autocrlf=false",
            *args,
        ],
        check=check,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _bash(
    script: str, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, "-c", script],
        check=check,
        capture_output=True,
        text=True,
        env={**_clean_env(), **(env or {})},
    )


def _build_delta(repo: Path, old: str, new: str, out: Path) -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    script = (
        f"set -euo pipefail; HERE='{_p(DAY)}'; RENT='{_p(out)}'; . \"$HERE/../common.sh\"; . \"$HERE/pack_lib.sh\"\n"
        f"pack_delta_build '{_p(repo)}' {old} {new} '{_p(out)}'; echo \"ADD=$DELTA_ADD DEL=$DELTA_DEL\""
    )
    return _bash(script, PATTERNS | {"NUZLINE_REPO": _p(repo)}, check=False)


def _tar_names(tar: Path) -> list[str]:
    r = subprocess.run(["tar", "-tf", _p(tar)], check=True, capture_output=True, text=True)
    return sorted(x for x in r.stdout.split("\n") if x and not x.endswith("/"))


@pytest.fixture
def src(tmp_path):
    """A source repo with three commits: c1, c2 (every kind of change), c3 (one file)."""
    r = tmp_path / "src"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _w(r / ".gitignore", "data/\nexternal/\n")
    _w(r / "a.txt", "a1\n")
    _w(r / "b.txt", "b1\n")
    _w(r / "run.sh", "echo 1\n")
    _w(r / "d/e.txt", "e1\n")
    _w(r / "ren_old.txt", "".join(f"line {i}\n" for i in range(40)))
    c1 = _commit(r, "c1")
    _w(r / "a.txt", "a2\n")
    _w(r / "f.txt", "f1\n")
    (r / "b.txt").unlink()
    (r / "d/e.txt").unlink()
    (r / "ren_old.txt").rename(r / "ren_new.txt")
    c2 = _commit(r, "c2")
    _git(r, "update-index", "--chmod=+x", "run.sh")
    _git(r, "commit", "-qm", "mode")  # a mode change alone
    c2m = _git(r, "rev-parse", "HEAD")
    _w(r / "a.txt", "a3\n")
    _w(r / "g/h.txt", "h\n")
    c3 = _commit(r, "c3")
    return {"repo": r, "c1": c1, "c2": c2, "c2m": c2m, "c3": c3}


# --- the delta list ---------------------------------------------------------------------------------


def test_delta_holds_added_modified_deleted_and_a_rename_as_delete_plus_add(src, tmp_path):
    out = tmp_path / "o1"
    r = _build_delta(src["repo"], src["c1"], src["c2"], out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tar_names(out / "delta.tar") == ["a.txt", "f.txt", "ren_new.txt"]
    assert sorted((out / "deletions.txt").read_text(encoding="utf-8").split()) == [
        "b.txt",
        "d/e.txt",
        "ren_old.txt",
    ]
    assert "ADD=3 DEL=3" in r.stdout


def test_delta_carries_a_file_mode_change_and_an_empty_delta_is_an_empty_tar(src, tmp_path):
    out = tmp_path / "o2"
    assert _build_delta(src["repo"], src["c2"], src["c2m"], out).returncode == 0
    assert _tar_names(out / "delta.tar") == ["run.sh"]
    listing = subprocess.run(
        ["tar", "-tvf", _p(out / "delta.tar")], check=True, capture_output=True, text=True
    ).stdout
    assert listing.startswith("-rwx"), listing
    out = tmp_path / "o3"
    assert _build_delta(src["repo"], src["c2"], src["c2"], out).returncode == 0
    assert (
        _tar_names(out / "delta.tar") == []
        and (out / "deletions.txt").read_text(encoding="utf-8") == ""
    )


def test_delta_refuses_a_forbidden_file(src, tmp_path):
    _w(src["repo"] / "rom/game.nds", "x")
    c = _commit(src["repo"], "rom")
    r = _build_delta(src["repo"], src["c3"], c, tmp_path / "o4")
    assert r.returncode == 3 and "forbidden files in the delta" in r.stdout


# --- pack.sh --delta refusals (before any pod script is read) ----------------------------------------


def _pack(src, tmp_path, *args: str, ids: str = "tests/a.py\n") -> subprocess.CompletedProcess:
    scratch = tmp_path / "scratch"
    (scratch / "rent" / "tables").mkdir(parents=True, exist_ok=True)
    _w(tmp_path / "ids.txt", ids)
    env = {"NUZLINE_REPO": _p(src["repo"]), "NUZLINE_SCRATCH": _p(scratch)}
    return subprocess.run(
        [BASH, "tools/ops/day/pack.sh", "--delta", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**_clean_env(), **env},
    )


def test_pack_delta_refuses_more_than_five_ids(src, tmp_path):
    ids = "".join(f"tests/t{i}.py\n" for i in range(6))
    r = _pack(
        src, tmp_path, src["c1"], src["c2"], "q-e4-aaaaaaa", _p(tmp_path / "ids.txt"), ids=ids
    )
    assert r.returncode == 3 and "at most 5 ids, the list has 6" in r.stdout, r.stdout + r.stderr
    r = _pack(
        src,
        tmp_path,
        src["c1"],
        src["c2"],
        "q-e4-aaaaaaa",
        _p(tmp_path / "ids.txt"),
        ids="tests/a.py 1\n",
    )
    assert r.returncode == 3 and "ids outside the format" in r.stdout
    r = _pack(src, tmp_path, src["c1"], src["c2"], "e4-aaaaaaa", _p(tmp_path / "ids.txt"))
    assert r.returncode == 3 and "q-<stream>-<label>" in r.stdout


def test_pack_delta_refuses_a_changed_tables_key_without_tables(src, tmp_path):
    scratch = tmp_path / "scratch" / "rent" / "tables"
    scratch.mkdir(parents=True)
    _w(
        scratch / "index.txt",
        f"{src['c1']} {'a' * 64}\n{src['c2']} {'b' * 64}\n{src['c3']} {'a' * 64}\n",
    )
    _w(tmp_path / "ids.txt", "tests/a.py\n")
    ids = _p(tmp_path / "ids.txt")
    r = _pack(src, tmp_path, src["c1"], src["c2"], "q-e4-bbbbbbb", ids)
    assert (
        r.returncode == 3
        and "tables key changed (old aaaaaaaaaaaaaaaa, new bbbbbbbbbbbbbbbb) and no --tables"
        in r.stdout
    )
    # an unchanged key passes this check (and stops at the pod-script checks of this checkout)
    r = _pack(src, tmp_path, src["c1"], src["c3"], "q-e4-ccccccc", ids)
    assert "tables key" not in r.stdout
    # no key at all for the new commit
    r = _pack(src, tmp_path, src["c1"], src["c2m"], "q-e4-ddddddd", ids)
    assert r.returncode == 3 and "no table key for" in r.stdout


# --- lane.sh -----------------------------------------------------------------------------------------


def _lane_env(tmp_path: Path) -> dict[str, str]:
    stubs = tmp_path / "stubs"
    for n, body in STUBS.items():
        _w(stubs / n, body)
        os.chmod(stubs / n, 0o755)
    _w(stubs / "pytest_stub.sh", FAKE_PYTEST)
    return {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "PYEXE": PY,
        "FAKE_PYTEST": _p(stubs / "pytest_stub.sh"),
        "LANE_ROOT": _p(tmp_path / "work/_lane"),
        "WORK": _p(tmp_path / "work"),
        "QUICK_POLL_S": "1",
        "QUICK_LOCK_S": "5",
    }


@pytest.mark.parametrize(
    ("run", "want"),
    [
        ("e4-a0m-d1ab9b7", "e4"),
        ("q-e4-d1ab9b7", "e4"),
        ("nightly-20261118-235f2f8", "nightly"),
        ("q-e1_x-aaa", "e1_x"),
        ("E4-x", None),
        ("q--x", None),
    ],
)
def test_stream_is_the_first_field_after_an_optional_q(run, want, tmp_path):
    r = _bash(f". '{_p(POD / 'lane.sh')}'; lane_stream '{run}'", check=False)
    assert (r.stdout.strip() or None) == want


def _run_repo(
    tmp_path: Path,
    src,
    name: str = "e4-a0m-aaaaaaa",
    commit: str | None = None,
    key: str = "k" * 16,
) -> Path:
    """What setup.sh leaves: <work>/<run>/repo = git archive of the commit, committed, with git-ignored built tables."""
    commit = commit or src["c1"]
    repo = tmp_path / "work" / name / "repo"
    repo.mkdir(parents=True)
    subprocess.run(
        f"git -C '{_p(src['repo'])}' -c core.autocrlf=false archive {commit} | tar -x -C '{_p(repo)}'",
        shell=True,
        executable=BASH,
        check=True,
    )
    _git(repo, "init", "-q")
    _commit(repo, f"snapshot {commit}")
    _w(repo / "data/gen4_platinum_kaizo/manifest.json", "old tables\n")
    return repo


def _marker(
    tmp_path: Path, repo: Path, commit: str, key: str = "k" * 16, stream: str = "e4"
) -> subprocess.CompletedProcess:
    env = _lane_env(tmp_path)
    return _bash(
        f". '{_p(POD / 'lane.sh')}'; lane_write_marker {stream} {commit} '{_p(repo)}' '{key}'",
        env,
        check=False,
    )


def _kv(path: Path) -> dict[str, str]:
    return dict(x.split("=", 1) for x in path.read_text(encoding="utf-8").split("\n") if "=" in x)


def test_marker_has_commit_tree_key_repo_and_time_and_is_skipped_when_tables_are_not_ignored(
    src, tmp_path
):
    repo = _run_repo(tmp_path, src)
    r = _marker(tmp_path, repo, src["c1"])
    assert r.returncode == 0, r.stdout + r.stderr
    m = _kv(tmp_path / "work/_lane/e4.sha")
    assert m["COMMIT"] == src["c1"] and m["TREE"] == _git(
        src["repo"], "rev-parse", f"{src['c1']}^{{tree}}"
    )
    assert m["TABLES_KEY"] == "k" * 16 and m["REPO"] == _p(repo) and int(m["TIME"]) > 0
    assert not list((tmp_path / "work/_lane").glob(".*"))  # the temp file went by mv
    # a built table that git does not ignore would change every later tree: no marker
    _w(repo / "visible.json", "{}")
    (tmp_path / "work/_lane/e4.sha").unlink()
    r = _marker(tmp_path, repo, src["c1"])
    assert r.returncode == 1 and "LANE_MARKER_SKIPPED" in r.stdout
    assert not (tmp_path / "work/_lane/e4.sha").exists()


# --- quick.sh ----------------------------------------------------------------------------------------


def _tree(tmp_path: Path, src, c: str) -> str:
    return _git(src["repo"], "rev-parse", f"{c}^{{tree}}")


def _make_run(
    tmp_path,
    src,
    name: str,
    old: str,
    new: str,
    *,
    ids: str = "tests/test_x.py::test_a\n",
    okey: str = "k" * 16,
    nkey: str = "k" * 16,
    new_tree: str | None = None,
    tables: bool = False,
    deletions: str | None = None,
    weights: str = "",
) -> Path:
    d = tmp_path / "work" / name
    assert _build_delta(src["repo"], old, new, d).returncode == 0
    if deletions is not None:
        _w(d / "deletions.txt", deletions)
    for n in ("gate.sh", "lane.sh", "quick.sh", "weights.py", "weights.txt"):
        _w(d / "bin" / n, (POD / n).read_bytes().replace(b"\r", b"").decode("utf-8"))
    if weights:
        _w(d / "bin/weights.txt", weights)
    _w(d / "ids.txt", ids)
    decomp = "d" * 40
    (tmp_path / "work/_decomp" / decomp).mkdir(parents=True, exist_ok=True)
    _w(
        d / "meta.env",
        f"MODE=delta\nOLD={old}\nNEW={new}\nOLD_TREE={_tree(tmp_path, src, old)}\nNEW_TREE={new_tree or _tree(tmp_path, src, new)}\n"
        f"OLD_TABLES_KEY={okey}\nTABLES_KEY={nkey}\nCOMMIT={new}\nDECOMP={decomp}\nWATCH_F=secretf\nWATCH_D=secretd\n",
    )
    if tables:
        (d / "t").mkdir()
        _w(d / "t/data/gen4_platinum_kaizo/manifest.json", "new tables\n")
        subprocess.run(
            f"tar -cJf '{_p(d / 'tables.tar.xz')}' -C '{_p(d / 't')}' data",
            shell=True,
            executable=BASH,
            check=True,
        )
    return d


def _cgroup(tmp_path: Path, limit: int = 100_000_000_000, cur: int = 1_000_000_000) -> Path:
    cg = tmp_path / "cg"
    _w(cg / "memory.max", f"{limit}\n")
    _w(cg / "memory.current", f"{cur}\n")
    return cg


def _quick(tmp_path: Path, run: Path, cg: Path, **env: str) -> subprocess.CompletedProcess:
    e = _lane_env(tmp_path) | {"CGROOT": _p(cg)} | env
    return _bash(f"bash '{_p(run / 'bin/quick.sh')}'", e, check=False)


def _log(run: Path) -> str:
    return (run / "quick.log").read_text(encoding="utf-8")


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git/" not in p.as_posix()
    }


def _setup_lane(tmp_path, src) -> Path:
    repo = _run_repo(tmp_path, src)
    assert _marker(tmp_path, repo, src["c1"]).returncode == 0
    return repo


def test_a_correct_delta_updates_the_lane_and_the_marker_and_chains(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    r = _quick(tmp_path, run, cg)
    log = _log(run)
    assert r.returncode == 0, log
    assert (
        re.search(r"^QUICK_START .* stream e4 ", log, re.MULTILINE)
        and "QUICK_PASS 1 passed" in log
        and log.rstrip().endswith("QUICK_DONE rc=0")
    )
    assert "secret" not in log
    lane = tmp_path / "work/_lane/e4/repo"
    assert (
        (lane / "a.txt").read_text(encoding="utf-8") == "a2\n"
        and (lane / "f.txt").exists()
        and (lane / "ren_new.txt").exists()
    )
    assert not (lane / "b.txt").exists() and not (lane / "ren_old.txt").exists()
    assert not (lane / "d").exists(), "the directory the deletion emptied is removed"
    assert (lane / "data/gen4_platinum_kaizo/manifest.json").read_text(
        encoding="utf-8"
    ) == "old tables\n"  # copied, git-ignored
    assert (lane / "external/pokeplatinum").exists()
    m = _kv(tmp_path / "work/_lane/e4.sha")
    assert (m["COMMIT"], m["TREE"], m["REPO"]) == (
        src["c2"],
        _tree(tmp_path, src, src["c2"]),
        _p(lane),
    )
    assert _git(lane, "rev-parse", "HEAD^{tree}") == m["TREE"]
    assert (run / "quick.rc").read_text(encoding="utf-8").strip() == "0" and (
        run / "agg.txt"
    ).read_text(encoding="utf-8").startswith("RESULT: PASS")
    args = (tmp_path / "stubs/pytest.args").read_text(encoding="utf-8")
    assert "-p no:cacheprovider" in args and args.strip().endswith("tests/test_x.py::test_a")
    # the next delta starts from the lane repo itself
    run2 = _make_run(tmp_path, src, "q-e4-ccccccc", src["c2"], src["c3"])
    assert _quick(tmp_path, run2, cg).returncode == 0, _log(run2)
    assert (lane / "g/h.txt").exists() and _kv(tmp_path / "work/_lane/e4.sha")["COMMIT"] == src[
        "c3"
    ]


def test_a_failing_test_prints_the_summary_and_the_first_failure_block(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    assert _quick(tmp_path, run, _cgroup(tmp_path), FAKE_PYTEST_RC="1").returncode == 1
    log = _log(run)
    assert (
        "QUICK_FAIL rc=1 1 failed in 0.02s" in log
        and "_____ test_a _____" in log
        and "FAILED tests/x.py::test_a" in log
    )
    assert log.rstrip().endswith("QUICK_DONE rc=1") and (run / "agg.txt").read_text(
        encoding="utf-8"
    ).startswith("RESULT: FAIL")


def test_no_marker_and_a_wrong_old_fall_back_and_change_nothing(src, tmp_path):
    cg = _cgroup(tmp_path)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    r = _quick(tmp_path, run, cg)
    assert (
        r.returncode == 3
        and "FALLBACK_FULL no marker" in _log(run)
        and _log(run).rstrip().endswith("QUICK_DONE rc=3")
    )
    assert not (tmp_path / "work/_lane/e4").exists()
    repo = _setup_lane(tmp_path, src)
    before, marker = _digest(tmp_path / "work"), (tmp_path / "work/_lane/e4.sha").read_bytes()
    run2 = _make_run(
        tmp_path, src, "q-e4-ccccccc", src["c2"], src["c3"]
    )  # the marker says c1, not c2
    before = {k: v for k, v in before.items() if not k.startswith("q-e4-ccccccc/")}
    assert _quick(tmp_path, run2, cg).returncode == 3
    assert re.search(
        rf"FALLBACK_FULL marker commit {src['c1'][:8]} is not OLD {src['c2'][:8]}", _log(run2)
    )
    after = {
        k: v
        for k, v in _digest(tmp_path / "work").items()
        if not k.startswith("q-e4-ccccccc/") and not k.startswith("q-e4-bbbbbbb/")
    }
    before = {k: v for k, v in before.items() if not k.startswith("q-e4-bbbbbbb/")}
    assert (
        after == before
        and (tmp_path / "work/_lane/e4.sha").read_bytes() == marker
        and repo.exists()
    )
    assert not (tmp_path / "work/_lane/e4").exists(), "no lane repo was created for a refused check"


def test_a_lane_repo_that_differs_from_old_tree_falls_back_without_changes(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    assert (
        _quick(
            tmp_path, _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"]), cg
        ).returncode
        == 0
    )
    lane = tmp_path / "work/_lane/e4/repo"
    _w(lane / "stray.txt", "someone edited the lane\n")
    run = _make_run(tmp_path, src, "q-e4-ccccccc", src["c2"], src["c3"])
    assert _quick(tmp_path, run, cg).returncode == 3 and "FALLBACK_FULL lane tree" in _log(run)
    assert (lane / "stray.txt").exists() and not (lane / "g").exists()


def test_a_wrong_tree_after_the_apply_removes_the_lane_and_marks_it_broken(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    marker = (tmp_path / "work/_lane/e4.sha").read_bytes()
    run = _make_run(
        tmp_path,
        src,
        "q-e4-bbbbbbb",
        src["c1"],
        src["c2"],
        new_tree=_tree(tmp_path, src, src["c3"]),
    )
    assert _quick(tmp_path, run, cg).returncode == 3
    assert "FALLBACK_FULL tree mismatch" in _log(run)
    assert (
        not (tmp_path / "work/_lane/e4/repo").exists()
        and (tmp_path / "work/_lane/e4/BROKEN").exists()
    )
    assert (tmp_path / "work/_lane/e4.sha").read_bytes() == marker, (
        "the marker still names OLD, never the broken state"
    )
    # the marker's REPO is the lane repo (not there): a later quick run falls back until a normal run rewrites the marker
    repo2 = _run_repo(tmp_path, src, "e4-b0m-ccccccc")
    run2 = _make_run(tmp_path, src, "q-e4-ddddddd", src["c1"], src["c2"])
    _w(
        tmp_path / "work/_lane/e4.sha",
        marker.decode("utf-8").replace(
            _p(tmp_path / "work/e4-a0m-aaaaaaa/repo"), _p(tmp_path / "work/_lane/e4/repo")
        ),
    )
    assert _quick(
        tmp_path, run2, cg
    ).returncode == 3 and "FALLBACK_FULL lane repo missing or broken" in _log(run2)
    assert (
        _marker(tmp_path, repo2, src["c1"]).returncode == 0
    )  # a normal run: the marker names its repo again
    assert (
        _quick(
            tmp_path, _make_run(tmp_path, src, "q-e4-eeeeeee", src["c1"], src["c2"]), cg
        ).returncode
        == 0
    )


def test_unsafe_paths_are_refused_before_anything_is_applied(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    (tmp_path / "work/outside.txt").write_text("keep", encoding="utf-8")
    for bad in ("../outside.txt\n", "/etc/passwd\n", "a.txt\n../../x\n", ".git/config\n"):
        run = _make_run(
            tmp_path, src, f"q-e4-b{abs(hash(bad)) % 10**6}", src["c1"], src["c2"], deletions=bad
        )
        assert _quick(tmp_path, run, cg).returncode == 3, bad
        assert "FALLBACK_FULL unsafe path in deletions.txt" in _log(run), bad
        lane = tmp_path / "work/_lane/e4/repo"
        assert (lane / "a.txt").read_text(encoding="utf-8") == "a1\n" and (
            lane / "b.txt"
        ).exists()  # c1 content, nothing applied
        assert (tmp_path / "work/outside.txt").read_text(encoding="utf-8") == "keep"


def test_tables_travel_only_with_a_changed_key(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    # no tables.tar.xz but the lane's key is not the delta's OLD key
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"], okey="z" * 16)
    assert _quick(
        tmp_path, run, cg
    ).returncode == 3 and "FALLBACK_FULL tables key of the lane" in _log(run)
    # tables.tar.xz replaces the built tables and the marker takes the new key
    run = _make_run(
        tmp_path,
        src,
        "q-e4-ccccccc",
        src["c1"],
        src["c2"],
        okey="z" * 16,
        nkey="n" * 16,
        tables=True,
    )
    assert _quick(tmp_path, run, cg).returncode == 0, _log(run)
    assert (tmp_path / "work/_lane/e4/repo/data/gen4_platinum_kaizo/manifest.json").read_text(
        encoding="utf-8"
    ) == "new tables\n"
    assert _kv(tmp_path / "work/_lane/e4.sha")["TABLES_KEY"] == "n" * 16


def test_the_run_waits_while_free_memory_is_not_over_weight_plus_reserve(src, tmp_path):
    _setup_lane(tmp_path, src)
    # tests/test_x.py weighs 10 GB: it needs more than 10 GB free (the reserve is 0: no unit is live)
    cg = _cgroup(
        tmp_path, limit=100_000_000_000, cur=90_000_000_000
    )  # exactly 10 GB free: not over
    run = _make_run(
        tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"], weights="tests/test_x.py 10 test\n"
    )
    e = _lane_env(tmp_path) | {"CGROOT": _p(cg)}
    p = subprocess.Popen(
        [BASH, "-c", f"bash '{_p(run / 'bin/quick.sh')}'"],
        env={**_clean_env(), **e},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(120):
            if (run / "quick.log").exists() and _log(run).count("QUICK_WAIT") >= 2:
                break
            time.sleep(0.5)
        log = _log(run)
        assert (
            log.count("QUICK_WAIT") >= 2
            and "free 10 GB, need more than 10 GB (weight 10 + reserve 0)" in log
        )
        assert "QUICK_RUN" not in log and p.poll() is None
        _w(cg / "memory.current", "80000000000\n")  # 20 GB free
        assert p.wait(timeout=60) == 0
    finally:
        if p.poll() is None:
            p.kill()
    log = _log(run)
    assert (
        log.index("QUICK_WAIT")
        < log.index("QUICK_RUN free 20 GB, weight 10 GB, reserve 0 GB")
        < log.index("QUICK_PASS")
    )


def test_the_memory_wait_has_a_limit(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])  # no weights line: 20 GB
    r = _quick(tmp_path, run, _cgroup(tmp_path, cur=95_000_000_000), QUICK_WAIT_S="2")
    assert r.returncode == 4 and "QUICK_FAIL memory wait over 2 s" in _log(run)
