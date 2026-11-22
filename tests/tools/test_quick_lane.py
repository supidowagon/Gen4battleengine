"""The quick lane (tools/ops/day/DAYBOX.md "Quick lane"): pack.sh --delta, pod/lane.sh, pod/quick.sh,
and notify.sh / fetch.sh in delta mode. Tiny temp git repos driven with bash and stub uv, flock and
ps; the tests marked with _linux_tools use the real flock, setsid and procps ps (the pod). No engine
import, no network."""

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from tests.tools.test_pod_gate import BASH, _alive, _clean_env, _linux_tools

ROOT = Path(__file__).resolve().parents[2]
DAY = ROOT / "tools" / "ops" / "day"
POD = DAY / "pod"
PY = Path(sys.executable).as_posix()

pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")
linux = pytest.mark.skipif(
    not _linux_tools(), reason="needs setsid, flock and a procps ps (the pod)"
)


def _has(tool: str) -> bool:
    return (
        BASH is not None
        and subprocess.run([BASH, "-c", f"command -v {tool} > /dev/null"], check=False).returncode
        == 0
    )


needs_setsid = pytest.mark.skipif(not _has("setsid"), reason="quick.sh starts its unit with setsid")


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
# FAKE_PYTEST_HOLD=<file>: writes <file>.pid, then waits until <file> exists (a running unit).
# FAKE_PYTEST_SKIP=1: every test skipped, rc 0. FAKE_PYTEST_RC=<n>: one failure, rc n.
FAKE_PYTEST = (
    'echo "$@" > "$(dirname "$0")/pytest.args"\n'
    'if [ -n "${FAKE_PYTEST_HOLD:-}" ]; then echo $$ > "$FAKE_PYTEST_HOLD.pid"\n'
    '  while [ ! -f "$FAKE_PYTEST_HOLD" ]; do sleep 0.2; done; fi\n'
    '[ "${FAKE_PYTEST_SKIP:-}" = 1 ] && { echo "SKIPPED [1] tests/x.py:1: no rom"; echo "1 skipped in 0.01s"; exit 0; }\n'
    '[ "${FAKE_PYTEST_RC:-0}" = 0 ] && { echo "1 passed in 0.01s"; exit 0; }\n'
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


def _commit(repo: Path, msg: str, force: tuple[str, ...] = ()) -> str:
    _git(repo, "add", "-A")
    if force:  # files under an ignored path: tracked but ignored
        _git(repo, "add", "-f", "--", *force)
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


def _build_delta(
    repo: Path, old: str, new: str, out: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    script = (
        f"set -euo pipefail; HERE='{_p(DAY)}'; RENT='{_p(out)}'; . \"$HERE/../common.sh\"; . \"$HERE/pack_lib.sh\"\n"
        f"pack_delta_build '{_p(repo)}' {old} {new} '{_p(out)}'; echo \"ADD=$DELTA_ADD DEL=$DELTA_DEL\""
    )
    return _bash(script, PATTERNS | {"NUZLINE_REPO": _p(repo)} | (env or {}), check=False)


def _tar_names(tar: Path) -> list[str]:
    r = subprocess.run(["tar", "-tf", _p(tar)], check=True, capture_output=True, text=True)
    return sorted(x for x in r.stdout.split("\n") if x and not x.endswith("/"))


@pytest.fixture
def src(tmp_path):
    """A source repo with c1, c2 (every kind of change), c2m (a mode change), c3 (one file).
    ign/ is git-ignored, and ign/old.md (c1) and ign/new.md (c2) are tracked anyway."""
    r = tmp_path / "src"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _w(r / ".gitignore", "data/\nexternal/\nign/\n")
    _w(r / "a.txt", "a1\n")
    _w(r / "b.txt", "b1\n")
    _w(r / "run.sh", "echo 1\n")
    _w(r / "d/e.txt", "e1\n")
    _w(r / "ign/old.md", "old1\n")
    _w(r / "ren_old.txt", "".join(f"line {i}\n" for i in range(40)))
    c1 = _commit(r, "c1", force=("ign/old.md",))
    _w(r / "a.txt", "a2\n")
    _w(r / "f.txt", "f1\n")
    _w(r / "ign/old.md", "old2\n")
    _w(r / "ign/new.md", "new\n")
    (r / "b.txt").unlink()
    (r / "d/e.txt").unlink()
    (r / "ren_old.txt").rename(r / "ren_new.txt")
    c2 = _commit(r, "c2", force=("ign/new.md",))
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
    assert _tar_names(out / "delta.tar") == [
        "a.txt",
        "f.txt",
        "ign/new.md",
        "ign/old.md",
        "ren_new.txt",
    ]
    assert sorted((out / "deletions.txt").read_text(encoding="utf-8").split()) == [
        "b.txt",
        "d/e.txt",
        "ren_old.txt",
    ]
    assert "ADD=5 DEL=3" in r.stdout


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


def test_a_delta_longer_than_one_command_line_is_archived_in_batches(src, tmp_path):
    names = [f"many/{'x' * 60}_{i:04d}.txt" for i in range(300)]  # about 20000 characters
    for n in names:
        _w(src["repo"] / n, n + "\n")
    c = _commit(src["repo"], "many")
    out = tmp_path / "o5"
    r = _build_delta(src["repo"], src["c3"], c, out, {"DELTA_ARG_CHARS": "2000"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert _tar_names(out / "delta.tar") == sorted(names) and "ADD=300 DEL=0" in r.stdout
    assert not (out / "delta.part.tar").exists()
    with tarfile.open(out / "delta.tar") as t:
        assert t.extractfile(names[123]).read() == (names[123] + "\n").encode()


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


def _lane_env(tmp_path: Path, stubs: tuple[str, ...] = ("flock", "ps", "uv")) -> dict[str, str]:
    # one folder per stub set: a folder holding a flock stub would shadow the real flock
    d = tmp_path / ("stubs" if set(stubs) == set(STUBS) else "stubs-" + "-".join(stubs))
    for n in stubs:
        _w(d / n, STUBS[n])
        os.chmod(d / n, 0o755)
    _w(d / "pytest_stub.sh", FAKE_PYTEST)
    return {
        "PATH": f"{d}{os.pathsep}{os.environ['PATH']}",
        "PYEXE": PY,
        "FAKE_PYTEST": _p(d / "pytest_stub.sh"),
        "LANE_ROOT": _p(tmp_path / "work/_lane"),
        "WORK": _p(tmp_path / "work"),
        "RUNNER_LOCK": _p(tmp_path / "work/.runner.lock"),
        "CLAIMS": _p(tmp_path / "work/.claims"),
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
) -> Path:
    """What setup.sh leaves: <work>/<run>/repo = git archive of the commit, committed by
    lane.sh lane_snapshot (setup.sh's snapshot step), with git-ignored built tables."""
    commit = commit or src["c1"]
    repo = tmp_path / "work" / name / "repo"
    repo.mkdir(parents=True)
    subprocess.run(
        f"git -C '{_p(src['repo'])}' -c core.autocrlf=false archive {commit} | tar -x -C '{_p(repo)}'",
        shell=True,
        executable=BASH,
        check=True,
    )
    r = _bash(f". '{_p(POD / 'lane.sh')}'; lane_snapshot '{_p(repo)}' {commit}", check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    _w(repo / "data/gen4_platinum_kaizo/manifest.json", "old tables\n")
    return repo


def _marker(
    tmp_path: Path,
    repo: Path,
    commit: str,
    tree: str,
    key: str = "k" * 16,
    stream: str = "e4",
    env: dict[str, str] | None = None,
    stubs: tuple[str, ...] = ("flock", "ps", "uv"),
) -> subprocess.CompletedProcess:
    e = _lane_env(tmp_path, stubs) | (env or {})
    return _bash(
        f". '{_p(POD / 'lane.sh')}'; lane_write_marker {stream} {commit} '{_p(repo)}' '{key}' '{tree}'",
        e,
        check=False,
    )


def _kv(path: Path) -> dict[str, str]:
    return dict(x.split("=", 1) for x in path.read_text(encoding="utf-8").split("\n") if "=" in x)


def _tree(tmp_path: Path, src, c: str) -> str:
    return _git(src["repo"], "rev-parse", f"{c}^{{tree}}")


def test_marker_has_commit_tree_key_repo_and_time_and_is_skipped_when_tables_are_not_ignored(
    src, tmp_path
):
    repo = _run_repo(tmp_path, src)
    tree = _tree(tmp_path, src, src["c1"])
    r = _marker(tmp_path, repo, src["c1"], tree)
    assert r.returncode == 0, r.stdout + r.stderr
    m = _kv(tmp_path / "work/_lane/e4.sha")
    assert m["COMMIT"] == src["c1"] and m["TREE"] == tree
    assert m["TABLES_KEY"] == "k" * 16 and m["REPO"] == _p(repo) and int(m["TIME"]) > 0
    assert not list((tmp_path / "work/_lane").glob(".*"))  # the temp file went by mv
    # a built table that git does not ignore would change every later tree: no marker
    _w(repo / "visible.json", "{}")
    (tmp_path / "work/_lane/e4.sha").unlink()
    r = _marker(tmp_path, repo, src["c1"], tree)
    assert r.returncode == 1 and "LANE_MARKER_SKIPPED the working tree differs" in r.stdout
    assert not (tmp_path / "work/_lane/e4.sha").exists()


def test_the_snapshot_holds_a_tracked_but_ignored_file_and_the_marker_needs_the_commit_tree(
    src, tmp_path
):
    repo = _run_repo(tmp_path, src)
    assert "ign/old.md" in _git(repo, "ls-files").split("\n")
    assert _git(repo, "rev-parse", "HEAD^{tree}") == _tree(tmp_path, src, src["c1"])
    # a snapshot that is not the commit (meta.env's TREE) gets no marker
    r = _marker(tmp_path, repo, src["c1"], _tree(tmp_path, src, src["c2"]))
    assert r.returncode == 1 and "LANE_MARKER_SKIPPED the snapshot tree" in r.stdout
    r = _marker(tmp_path, repo, src["c1"], "")
    assert r.returncode == 1 and "LANE_MARKER_SKIPPED no TREE in meta.env" in r.stdout
    assert not (tmp_path / "work/_lane/e4.sha").exists()


# --- quick.sh ----------------------------------------------------------------------------------------


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
    for n in (
        "gate.sh",
        "lane.sh",
        "quick.sh",
        "weights.py",
        "weights.txt",
        "notify.sh",
        "fetch.sh",
    ):
        _w(d / "bin" / n, (POD / n).read_bytes().replace(b"\r", b"").decode("utf-8"))
    if weights:
        _w(d / "bin/weights.txt", weights)
    _w(d / "ids.txt", ids)
    decomp = "d" * 40
    (tmp_path / "work/_decomp" / decomp).mkdir(parents=True, exist_ok=True)
    _w(
        d / "meta.env",
        f"MODE=delta\nOLD={old}\nNEW={new}\nOLD_TREE={_tree(tmp_path, src, old)}\nNEW_TREE={new_tree or _tree(tmp_path, src, new)}\n"
        f"OLD_TABLES_KEY={okey}\nTABLES_KEY={nkey}\nCOMMIT={new}\nDECOMP={decomp}\nGATE_VERSION=2\nPOD_SCRIPTS=ps1\n"
        "POD_WEIGHTS=pw1\nWATCH_F=secretf\nWATCH_D=secretd\n",
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


def _qenv(tmp_path: Path, cg: Path, stubs=("flock", "ps", "uv"), **env: str) -> dict[str, str]:
    return _lane_env(tmp_path, stubs) | {"CGROOT": _p(cg)} | env


def _quick(tmp_path: Path, run: Path, cg: Path, **env: str) -> subprocess.CompletedProcess:
    if not _has("setsid"):
        pytest.skip("quick.sh starts its unit with setsid")
    return _bash(f"bash '{_p(run / 'bin/quick.sh')}'", _qenv(tmp_path, cg, **env), check=False)


def _popen_quick(run: Path, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [BASH, "-c", f"bash '{_p(run / 'bin/quick.sh')}'"],
        env={**_clean_env(), **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _log(run: Path) -> str:
    p = run / "quick.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _until(cond, seconds: float = 60) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return cond()


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git/" not in p.as_posix()
    }


def _setup_lane(tmp_path, src, commit: str | None = None) -> Path:
    commit = commit or src["c1"]
    repo = _run_repo(tmp_path, src, commit=commit)
    assert _marker(tmp_path, repo, commit, _tree(tmp_path, src, commit)).returncode == 0
    return repo


@needs_setsid
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
    # a new file under an ignored path is in the tree and the commit (force-added)
    assert {"ign/old.md", "ign/new.md"} <= set(_git(lane, "ls-files").split("\n"))
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
    assert (run / "gate.env").read_text(encoding="utf-8") == (
        "GATE_VERSION=2\nPOD_SCRIPTS=ps1\nPOD_WEIGHTS=pw1\n"
    )
    assert not (run / "out/run/q").exists() and not (run / "out/pending.txt").exists()
    args = (tmp_path / "stubs/pytest.args").read_text(encoding="utf-8")
    assert "-rfEs" in args and "-p no:cacheprovider" in args
    assert args.strip().endswith("tests/test_x.py::test_a")
    # the next delta starts from the lane repo itself
    run2 = _make_run(tmp_path, src, "q-e4-ccccccc", src["c2"], src["c3"])
    assert _quick(tmp_path, run2, cg).returncode == 0, _log(run2)
    assert (lane / "g/h.txt").exists() and _kv(tmp_path / "work/_lane/e4.sha")["COMMIT"] == src[
        "c3"
    ]


@needs_setsid
def test_a_file_that_becomes_a_directory_and_the_reverse_apply(src, tmp_path):
    r = src["repo"]
    _w(r / "fd", "a file\n")
    _w(r / "dd/inner.txt", "in a dir\n")
    c4 = _commit(r, "c4")
    (r / "fd").unlink()
    _w(r / "fd/inner.txt", "now a dir\n")
    shutil.rmtree(r / "dd")
    _w(r / "dd", "now a file\n")
    c5 = _commit(r, "c5")
    _setup_lane(tmp_path, src, c4)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", c4, c5)
    assert _quick(tmp_path, run, _cgroup(tmp_path)).returncode == 0, _log(run)
    lane = tmp_path / "work/_lane/e4/repo"
    assert (lane / "fd/inner.txt").read_text(encoding="utf-8") == "now a dir\n"
    assert (lane / "dd").is_file() and (lane / "dd").read_text(encoding="utf-8") == "now a file\n"
    assert _git(lane, "rev-parse", "HEAD^{tree}") == _tree(tmp_path, src, c5)


@needs_setsid
def test_a_type_change_to_a_symlink_applies(src, tmp_path):
    r = src["repo"]
    (r / "a.txt").unlink()
    try:
        os.symlink("f.txt", r / "a.txt")
    except OSError:
        pytest.skip("no symlinks here")
    c4 = _commit(r, "file to link")
    changes = _git(r, "diff", "--no-renames", "--name-status", src["c3"], c4)
    assert changes.split() == ["T", "a.txt"]
    _setup_lane(tmp_path, src, src["c3"])
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c3"], c4)
    assert _quick(tmp_path, run, _cgroup(tmp_path)).returncode == 0, _log(run)
    lane = tmp_path / "work/_lane/e4/repo"
    assert (lane / "a.txt").is_symlink() and os.readlink(lane / "a.txt") == "f.txt"
    assert _kv(tmp_path / "work/_lane/e4.sha")["COMMIT"] == c4


@needs_setsid
def test_a_failing_test_prints_the_summary_and_the_first_failure_block(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    assert _quick(tmp_path, run, _cgroup(tmp_path), FAKE_PYTEST_RC="1").returncode == 11
    log = _log(run)
    assert (
        "QUICK_FAIL rc=11 (pytest rc 1) 1 failed in 0.02s" in log
        and "_____ test_a _____" in log
        and "FAILED tests/x.py::test_a" in log
    )
    assert log.rstrip().endswith("QUICK_DONE rc=11") and (run / "agg.txt").read_text(
        encoding="utf-8"
    ).startswith("RESULT: FAIL")
    # pytest's own exit codes 3-5 never read as the lane's fallback, gate or broken codes
    run = _make_run(tmp_path, src, "q-e4-ccccccc", src["c2"], src["c3"])
    assert _quick(tmp_path, run, _cgroup(tmp_path), FAKE_PYTEST_RC="5").returncode == 15


@needs_setsid
def test_a_run_where_every_test_skips_is_not_a_pass(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    assert _quick(tmp_path, run, _cgroup(tmp_path), FAKE_PYTEST_SKIP="1").returncode == 6
    log = _log(run)
    assert (
        "QUICK_FAIL no test passed: 1 skipped in 0.01s" in log and "SKIPPED [1] tests/x.py:1" in log
    )
    assert (run / "agg.txt").read_text(encoding="utf-8").startswith("RESULT: FAIL (no test passed")


@needs_setsid
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


@needs_setsid
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


@needs_setsid
def test_a_failed_copy_of_the_built_tables_fails_the_lane_creation(src, tmp_path):
    repo = _setup_lane(tmp_path, src)
    (repo / "data/gen4_platinum_kaizo/manifest.json").unlink()  # the run repo lost its tables
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    assert _quick(tmp_path, run, _cgroup(tmp_path)).returncode == 3
    assert "FALLBACK_FULL lane repo creation failed" in _log(run)
    assert not (tmp_path / "work/_lane/e4/repo").exists()


@needs_setsid
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
        _marker(tmp_path, repo2, src["c1"], _tree(tmp_path, src, src["c1"])).returncode == 0
    )  # a normal run: the marker names its repo again
    assert (
        _quick(
            tmp_path, _make_run(tmp_path, src, "q-e4-eeeeeee", src["c1"], src["c2"]), cg
        ).returncode
        == 0
    )


@needs_setsid
def test_unsafe_paths_are_refused_before_anything_is_applied(src, tmp_path):
    _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    (tmp_path / "work/outside.txt").write_text("keep", encoding="utf-8")
    for i, bad in enumerate(
        ("../outside.txt\n", "/etc/passwd\n", "a.txt\n../../x\n", ".git/config\n")
    ):
        run = _make_run(tmp_path, src, f"q-e4-b{i}", src["c1"], src["c2"], deletions=bad)
        assert _quick(tmp_path, run, cg).returncode == 3, bad
        assert "FALLBACK_FULL unsafe path in deletions.txt" in _log(run), bad
        lane = tmp_path / "work/_lane/e4/repo"
        assert (lane / "a.txt").read_text(encoding="utf-8") == "a1\n" and (
            lane / "b.txt"
        ).exists()  # c1 content, nothing applied
        assert (tmp_path / "work/outside.txt").read_text(encoding="utf-8") == "keep"


def _tar(path: Path, members: list[tuple[str, str]]) -> None:
    """members: (name, "file:<text>" | "link:<target>")."""
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as t:
        for name, what in members:
            kind, val = what.split(":", 1)
            ti = tarfile.TarInfo(name)
            if kind == "link":
                ti.type, ti.linkname = tarfile.SYMTYPE, val
                t.addfile(ti)
            else:
                data = val.encode()
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))


@needs_setsid
@pytest.mark.parametrize(
    ("members", "want"),
    [
        (
            [("a.txt", "file:a2\n"), ("../evil.txt", "file:x")],
            "unsafe path in delta.tar: ../evil.txt",
        ),
        (
            [(".git/hooks/post-commit", "file:x")],
            "unsafe path in delta.tar: .git/hooks/post-commit",
        ),
        ([("lnk", "link:../../outside.txt")], "delta.tar holds a link that leaves the repo"),
        ([("lnk", "link:/etc")], "delta.tar holds a link that leaves the repo"),
        (
            [("lnk", "link:g"), ("lnk/x.txt", "file:x")],
            "unsafe path in delta.tar: lnk/x.txt lies under a link of the same tar",
        ),
        (
            [("deep/x.txt", "file:x"), ("lnk/y/z.txt", "file:z"), ("lnk", "link:d")],
            "unsafe path in delta.tar: lnk/y/z.txt lies under a link of the same tar",
        ),
    ],
)
def test_unsafe_tar_members_and_links_are_refused_before_anything_is_applied(
    src, tmp_path, members, want
):
    _setup_lane(tmp_path, src)
    (tmp_path / "work/outside.txt").write_text("keep", encoding="utf-8")
    lane = tmp_path / "work/_lane/e4/repo"
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    _tar(run / "delta.tar", members)
    assert _quick(tmp_path, run, _cgroup(tmp_path)).returncode == 3, _log(run)
    assert f"FALLBACK_FULL {want}" in _log(run)
    # the lane repo was created (first quick run) but nothing of the delta reached it
    assert (lane / "a.txt").read_text(encoding="utf-8") == "a1\n" and (lane / "b.txt").exists()
    assert not (lane / "lnk").exists() and not (lane / "lnk").is_symlink()
    assert (
        not (tmp_path / "work/evil.txt").exists()
        and not (tmp_path / "work/_lane/e4/evil.txt").exists()
    )
    assert (tmp_path / "work/outside.txt").read_text(encoding="utf-8") == "keep"
    assert _kv(tmp_path / "work/_lane/e4.sha")["COMMIT"] == src["c1"]


@needs_setsid
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


@needs_setsid
def test_the_unit_waits_for_the_runners_gate(src, tmp_path):
    _setup_lane(tmp_path, src)
    # memory at GATE_GB (88) or over: no unit starts (runner.sh: hold-mem)
    cg = _cgroup(tmp_path, cur=90_000_000_000)
    run = _make_run(
        tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"], weights="tests/test_x.py 10 test\n"
    )
    p = _popen_quick(run, _qenv(tmp_path, cg))
    try:
        assert _until(lambda: _log(run).count("QUICK_WAIT") >= 2), _log(run)
        log = _log(run)
        assert (
            "hold-mem: mem 90 + reserve 0 + weight 10 GB; GATE_GB 88 KILL_GB 118, 0 units live"
            in log
        )
        assert "QUICK_RUN" not in log and p.poll() is None
        _w(cg / "memory.current", "80000000000\n")  # 80 GB: an idle pod, 80 + 10 under KILL_GB
        assert p.wait(timeout=60) == 0
    finally:
        if p.poll() is None:
            p.kill()
    log = _log(run)
    assert (
        log.index("QUICK_WAIT")
        < log.index("(start) mem 80 GB, reserve 0 GB, weight 10 GB, pod units 1")
        < log.index("QUICK_PASS")
    )


@needs_setsid
def test_an_idle_pod_holds_a_unit_that_would_pass_kill_gb_for_idle_wait_s(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(
        tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"], weights="tests/test_x.py 50 test\n"
    )
    r = _quick(tmp_path, run, _cgroup(tmp_path, cur=80_000_000_000), IDLE_WAIT_S="2")
    log = _log(run)
    assert r.returncode == 0, log
    assert "QUICK_WAIT" in log and "wait-idle: mem 80 + reserve 0 + weight 50 GB" in log
    assert "(start-idle-wait) mem 80 GB" in log


@needs_setsid
def test_the_gate_wait_has_a_limit(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    r = _quick(tmp_path, run, _cgroup(tmp_path, cur=95_000_000_000), QUICK_WAIT_S="2")
    assert r.returncode == 4 and "QUICK_FAIL gate wait over 2 s" in _log(run)
    assert not (run / "out/run/q").exists()


# --- with the real flock, setsid and ps (the pod) -----------------------------------------------------


@linux
def test_the_running_unit_counts_in_every_runners_podgate_and_goes_after(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(
        tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"], weights="tests/test_x.py 10 test\n"
    )
    hold = tmp_path / "hold"
    p = _popen_quick(run, _qenv(tmp_path, _cgroup(tmp_path), FAKE_PYTEST_HOLD=_p(hold)))
    gate = (
        f"NO_W_FILE_GB=12; . '{_p(POD / 'gate.sh')}'; podgate '{_p(tmp_path / 'work')}'/*/out/run/*"
    )
    try:
        assert _until(lambda: (run / "out/run/q").exists() and Path(f"{hold}.pid").exists()), _log(
            run
        )
        assert (run / "out/q.w").read_text(encoding="utf-8").strip() == "10"
        pg = int((run / "out/run/q").read_text(encoding="utf-8"))
        assert int(Path(f"{hold}.pid").read_text(encoding="utf-8")) == pg, (
            "pytest runs as its own process group, the pid in out/run/q"
        )
        live, res = _bash(gate).stdout.split()
        assert live == "1" and int(res) > 0, "another runner sees the unit and reserves its gap"
        hold.write_text("go", encoding="utf-8")
        assert p.wait(timeout=60) == 0, _log(run)
    finally:
        hold.write_text("go", encoding="utf-8")
        if p.poll() is None:
            p.kill()
    assert not (run / "out/run/q").exists() and _bash(gate).stdout.split() == ["0", "0"]


@linux
def test_pytest_over_quick_max_s_is_killed_with_its_process_group(src, tmp_path):
    _setup_lane(tmp_path, src)
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    hold = tmp_path / "never"
    r = _quick(tmp_path, run, _cgroup(tmp_path), FAKE_PYTEST_HOLD=_p(hold), QUICK_MAX_S="2")
    log = _log(run)
    assert r.returncode == 124 and "QUICK_FAIL timeout after 2 s" in log, log
    assert not _alive(int(Path(f"{hold}.pid").read_text(encoding="utf-8")))
    assert not (run / "out/run/q").exists()


@linux
def test_real_flock_the_stream_lock_and_the_gate_lock(src, tmp_path):
    repo = _setup_lane(tmp_path, src)
    cg = _cgroup(tmp_path)
    no_flock_stub = ("ps", "uv")
    lane_lock = tmp_path / "work/_lane/e4.lock"
    gate_lock = tmp_path / "work/.runner.lock"

    def holder(lock: Path) -> subprocess.Popen:
        h = subprocess.Popen([BASH, "-c", f"exec flock '{_p(lock)}' sleep 60"], env=_clean_env())
        assert _until(lambda: _bash(f"flock -n '{_p(lock)}' true", check=False).returncode == 1, 10)
        return h

    # another quick run of the stream holds the stream lock: lane busy
    h = holder(lane_lock)
    try:
        run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
        r = _bash(
            f"bash '{_p(run / 'bin/quick.sh')}'",
            _qenv(tmp_path, cg, no_flock_stub, QUICK_LOCK_S="1"),
            check=False,
        )
        assert r.returncode == 3 and "FALLBACK_FULL lane busy" in _log(run)
        # a normal send's marker does not wait long for it either
        t0 = time.time()
        m = _marker(
            tmp_path,
            repo,
            src["c1"],
            _tree(tmp_path, src, src["c1"]),
            env={"LANE_LOCK_S": "1"},
            stubs=no_flock_stub,
        )
        assert m.returncode == 1 and "lock of stream e4 busy" in m.stdout and time.time() - t0 < 20
    finally:
        h.kill()
        h.wait()
    # a runner holds RUNNER_LOCK: the unit does not start until it is released
    h = holder(gate_lock)
    run = _make_run(tmp_path, src, "q-e4-ccccccc", src["c1"], src["c2"])
    p = _popen_quick(run, _qenv(tmp_path, cg, no_flock_stub))
    try:
        assert _until(lambda: "QUICK_APPLIED" in _log(run), 60), _log(run)
        time.sleep(3)
        assert "QUICK_RUN" not in _log(run) and p.poll() is None
        h.kill()
        h.wait()
        assert p.wait(timeout=60) == 0, _log(run)
    finally:
        if h.poll() is None:
            h.kill()
        if p.poll() is None:
            p.kill()
    assert "QUICK_PASS" in _log(run)


# --- notify.sh and fetch.sh in delta mode -------------------------------------------------------------


def test_notify_offers_the_quick_results_once_quick_rc_is_written(src, tmp_path):
    run = _make_run(tmp_path, src, "q-e4-bbbbbbb", src["c1"], src["c2"])
    got = tmp_path / "got"
    got.mkdir()
    rpc = tmp_path / "rpc"
    _w(
        rpc,
        '#!/bin/bash\n[ "$1 $2" = "send --code" ] || exit 1\n'
        'echo "$3" >> "$GOT/codes"; cp "$4" "$GOT/" && exit 0\n',
    )
    os.chmod(rpc, 0o755)
    for n in ("notify.sh", "fetch.sh"):  # the pod's runpodctl path, pointed at the stub
        t = (run / "bin" / n).read_text(encoding="utf-8").replace("/work/_bin/runpodctl", _p(rpc))
        _w(run / "bin" / n, t)
    _w(run / "quick.log", "QUICK_START x\nQUICK_PASS 1 passed\nQUICK_DONE rc=0\n")
    _w(run / "agg.txt", "RESULT: PASS (1 passed)\nQUICK_START x\n")
    p = subprocess.Popen(
        [BASH, "-c", f"cd '{_p(run)}' && bash bin/notify.sh > notify.log 2>&1"],
        env={**_clean_env(), "GOT": _p(got), "WATCH_HOURS": "0.05"},
    )
    try:
        time.sleep(2)
        assert p.poll() is None and not (got / "results.tgz").exists(), "no result before quick.rc"
        _w(run / "quick.rc", "0\n")
        assert p.wait(timeout=60) == 0
    finally:
        if p.poll() is None:
            p.kill()
    assert (got / "codes").read_text(encoding="utf-8").split() == ["secretd"]
    assert (run / "notify/taken_d").exists() and (run / "notify/stop").exists()
    with tarfile.open(got / "results.tgz") as t:
        names = set(t.getnames())
        assert {"agg.txt", "quick.log", "ids.txt", "meta.env"} <= names
        assert t.extractfile("agg.txt").read().decode().startswith("RESULT: PASS"), (
            "fetch.sh keeps quick.sh's agg.txt"
        )
        assert b"WATCH_" not in t.extractfile("meta.env").read()
    log = (run / "notify.log").read_text(encoding="utf-8")
    assert "quick run over" in log and "secret" not in log
