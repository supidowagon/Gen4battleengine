"""tools/ops/build_tables.sh: the decomp step (ensure_decomp, decomp_state, link_decomp, unlink_decomp),
sourced from a copy of the script in a throwaway git checkout whose fake decomp is a folder holding
src/. No build runs. On Windows the link is a junction (powershell), elsewhere a symlink."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "tools" / "ops"


def _bash() -> str | None:
    """A POSIX bash: on Windows Git's own usr/bin/bash.exe (the bash.exe in System32 or WindowsApps is
    the WSL launcher, which cannot see these paths), elsewhere the bash on PATH."""
    found = shutil.which("bash")
    if found and not any(d in found.lower() for d in ("system32", "windowsapps", "\\git\\bin\\")):
        return found
    git = shutil.which("git")
    if git:
        exec_path = subprocess.run([git, "--exec-path"], check=False, capture_output=True, text=True).stdout.strip()
        for b in (Path(exec_path).parents[2] / "usr" / "bin" / "bash.exe", Path(git).parents[1] / "usr" / "bin" / "bash.exe"):
            if b.is_file():
                return str(b)
    return None


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("git") is None, reason="no POSIX bash or git")


@pytest.fixture
def repo(tmp_path):
    """A main checkout: git repo with the two scripts and a fake decomp (external/pokeplatinum/src)."""
    r = tmp_path / "repo"
    (r / "tools" / "ops").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    for f in ("build_tables.sh", "common.sh"):
        (r / "tools" / "ops" / f).write_bytes((OPS / f).read_bytes().replace(b"\r\n", b"\n"))
    (r / "external" / "pokeplatinum" / "src").mkdir(parents=True)
    (r / "external" / "pokeplatinum" / "src" / "battle.c").write_text("x\n", encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    yield r, b
    _drop_link(b / "external" / "pokeplatinum")
    assert (r / "external" / "pokeplatinum" / "src" / "battle.c").is_file()


def _drop_link(p: Path) -> None:
    """Removes a link left by a test without touching its target (a junction is removed by rmdir)."""
    if p.is_symlink() or p.is_junction():
        if os.name == "nt":
            os.rmdir(p)
        else:
            os.unlink(p)


def _run(repo_b, body: str) -> subprocess.CompletedProcess:
    r, b = repo_b
    env = dict(os.environ)
    env.pop("NUZLINE_REPO", None)
    env.pop("NUZLINE_SCRATCH", None)
    if os.name == "nt":
        env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
    script = (r / "tools" / "ops" / "build_tables.sh").as_posix()
    return subprocess.run([BASH, "-c", 'set -euo pipefail; . "$1"; B=$2; ' + body, "t", script, b.as_posix()],
                          capture_output=True, text=True, env=env, timeout=120)


def test_no_decomp_links_the_main_checkouts_and_unlink_leaves_it(repo):
    p = _run(repo, 'ensure_decomp "$B"; echo "LINKED=${LINKED##*/}"; decomp_state "$B/external/pokeplatinum"; '
                   'ls "$B/external/pokeplatinum/src"; unlink_decomp; decomp_state "$B/external/pokeplatinum"')
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split() == ["LINKED=pokeplatinum", "link", "battle.c", "none"]


def test_an_empty_folder_is_removed_and_linked(repo):
    _, b = repo
    (b / "external" / "pokeplatinum").mkdir(parents=True)
    p = _run(repo, 'decomp_state "$B/external/pokeplatinum"; ensure_decomp "$B"; '
                   'decomp_state "$B/external/pokeplatinum"; ls "$B/external/pokeplatinum/src"')
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split() == ["empty", "link", "battle.c"]


def test_a_folder_with_content_but_no_src_stops_the_build(repo):
    _, b = repo
    (b / "external" / "pokeplatinum").mkdir(parents=True)
    (b / "external" / "pokeplatinum" / "notes.txt").write_text("keep\n", encoding="utf-8")
    p = _run(repo, 'ensure_decomp "$B"; echo reached')
    assert p.returncode == 2
    assert "is not empty and has no src/" in p.stdout and "reached" not in p.stdout
    assert (b / "external" / "pokeplatinum" / "notes.txt").read_text(encoding="utf-8") == "keep\n"


def test_an_own_decomp_is_kept_and_not_linked(repo):
    _, b = repo
    (b / "external" / "pokeplatinum" / "src").mkdir(parents=True)
    p = _run(repo, 'ensure_decomp "$B"; echo "LINKED=$LINKED"; decomp_state "$B/external/pokeplatinum"')
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split() == ["LINKED=", "decomp"]


def test_an_existing_link_is_used_and_never_removed_by_this_run(repo):
    first = _run(repo, 'ensure_decomp "$B"')   # leaves its link: the run ends without unlink_decomp
    assert first.returncode == 0, first.stdout + first.stderr
    p = _run(repo, 'ensure_decomp "$B"; echo "LINKED=$LINKED"; unlink_decomp; decomp_state "$B/external/pokeplatinum"')
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.split() == ["LINKED=", "link"]
