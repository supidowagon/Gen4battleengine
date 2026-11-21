"""tools/ops/offsite.sh: the state-file selection (--list) and the pre-commit hard-exclusion check
(--check) on a temp tree. No git remote is touched."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "ops" / "offsite.sh"


def _bash() -> str | None:
    """A POSIX bash: on Windows Git's bash (System32's bash.exe is WSL, which cannot see these
    paths), elsewhere the bash on PATH."""
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    git = shutil.which("git")
    if git:
        exec_path = subprocess.run([git, "--exec-path"], check=False, capture_output=True, text=True).stdout.strip()
        for b in (Path(exec_path).parents[2] / "bin" / "bash.exe", Path(git).parents[1] / "bin" / "bash.exe"):
            if b.is_file():
                return str(b)
    return None


BASH = _bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")

# Token-shaped strings are assembled at run time so this file holds none.
TOKENS = {
    "runpod": "rp" + "a_" + "A1b2C3d4E5f6G7h8I9j0K1",
    "github": "gh" + "p_" + "x" * 36,
    "openai": "s" + "k-" + "y" * 40,
    "pem": "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n",
    "assign": 'api' + 'key = "' + "z" * 32 + '"',
}


def _write(root: Path, rel: str, data: bytes | str = "note\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def _env(scratch: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["NUZLINE_SCRATCH"] = str(scratch)
    return env


def _run(*args: str, scratch: Path) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, SCRIPT.as_posix(), *args], capture_output=True, text=True, env=_env(scratch), cwd=ROOT)


KEPT = [
    "handoff/MANAGER.md",
    "manager/podlist_x.txt",
    "manager/streams.json",
    "manager/plain_notes",  # no extension, text
    "e1/family_list.md",
    "e1/rules/A-1.md",
    "e1/ok.log",  # a log under 1 MB
    "e4/sheets/x.md",
    "rent/DAYBOX.md",
    "rent/POD_ID",
    "rent/day3/POD_ID_3",
    "rent/lists/e1-h1.txt",
    "rent/make_snap.sh",
]

EXCLUDED = {
    # nested repos and worktrees
    "manager/wt_x/notes.md": "dir:nested-repo",
    "e2/wt/inner.md": "dir:nested-repo",
    # caches, envs, code and table copies, backups
    "e1/.cache/x.md": "dir:pruned",
    "e1/.venv/x.txt": "dir:pruned",
    "e1/gfx-cache/x.txt": "dir:pruned",
    "e1/fbak/src/a.py": "dir:pruned",
    "e1/backups/n.md": "dir:pruned",
    "e3/Backup2/n.md": "dir:pruned",
    "e2/seg/base/data/intermediate/x.json": "dir:pruned",
    "rent/out/run1/a.md": "dir:pruned",
    "rent/home/.runpod/config.toml": "dir:pruned",
    # ROMs, saves, archives, arrays
    "manager/wb/maylene.sav": "rom-save-archive-or-binary",
    "e1/game.nds": "rom-save-archive-or-binary",
    "e1/game.DSV": "rom-save-archive-or-binary",
    "e1/state.ds1": "rom-save-archive-or-binary",
    "e1/t.tar.xz": "rom-save-archive-or-binary",
    "e1/a.npz": "rom-save-archive-or-binary",
    "e1/s.pkl": "rom-save-archive-or-binary",
    "e1/b.bin": "rom-save-archive-or-binary",
    "e1/z.zip": "rom-save-archive-or-binary",
    # keys and credentials
    "e1/id.pem": "key-or-credential",
    "e1/k.key": "key-or-credential",
    "e1/.env.local": "key-or-credential",
    "e1/meta.env": "key-or-credential",
    "e1/local.toml": "key-or-credential",
    # git control files (a copied .gitignore would hide files from the state clone)
    "e2/seg/.gitignore": "git-control-file",
    "e2/seg/.gitattributes": "git-control-file",
    # content
    "e1/tok_runpod.txt": "secret-like-content",
    "e1/tok_github.md": "secret-like-content",
    "e1/tok_openai.py": "secret-like-content",
    "e1/tok_pem.txt": "secret-like-content",
    "e1/tok_assign.toml": "secret-like-content",
    # size
    "e1/big.txt": "over-2MB",
    "e1/big.log": "log-over-1MB",
    # not text
    "e2/blob.dat": "binary",
    # allow-list per root
    "manager/podcheck_body.json": "exec-pod-body",
    "rent/build_x.log": "rent-run-file",
    "rent/day/out.tar": "rom-save-archive-or-binary",
    "rent/gb1/results.txt": "rent-run-file",
    "misc/top.md": None,  # outside the allow-list roots: never listed
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    s = tmp_path / "scratch"
    for rel in KEPT:
        _write(s, rel)
    _write(s, "e1/ok.log", "x" * 500_000)
    for rel in EXCLUDED:
        _write(s, rel)
    _write(s, "manager/wt_x/.git", "gitdir: elsewhere\n")  # a worktree's .git file
    (s / "e2" / "wt" / ".git").mkdir(parents=True)  # a nested repo
    for name, tok in TOKENS.items():
        ext = {"runpod": "txt", "github": "md", "openai": "py", "pem": "txt", "assign": "toml"}[name]
        _write(s, f"e1/tok_{name}.{ext}", f"line\n{tok}\n")
    _write(s, "e1/big.txt", "y" * (2 * 1024 * 1024 + 1))
    _write(s, "e1/big.log", "y" * (1024 * 1024 + 1))
    _write(s, "e2/blob.dat", b"ab\x00cd")
    return s


def _parse(out: str) -> tuple[set[str], dict[str, str]]:
    kept, excl = set(), {}
    for line in out.splitlines():
        sign, mid, path = line.split("\t")
        if sign == "+":
            kept.add(path)
        else:
            excl[path] = mid
    return kept, excl


def _excluded_reason(excl: dict[str, str], rel: str) -> str | None:
    """The reason for rel, or for the pruned or nested-repo folder that holds it."""
    if rel in excl:
        return excl[rel]
    for path, reason in excl.items():
        if path.endswith("/") and rel.startswith(path):
            return reason
    return None


def test_list_keeps_the_allow_list_and_nothing_else(tree: Path):
    r = _run("--list", scratch=tree)
    assert r.returncode == 0, r.stderr
    kept, excl = _parse(r.stdout)
    assert kept == set(KEPT)
    for rel, reason in EXCLUDED.items():
        if reason is None:
            assert rel not in kept and _excluded_reason(excl, rel) is None
        else:
            assert _excluded_reason(excl, rel) == reason, rel


def test_check_passes_a_clean_tree_and_names_each_offender(tmp_path: Path):
    state = tmp_path / "state"
    _write(state, "handoff/MANAGER.md")
    _write(state, "e1/rules/A-1.md")
    (state / ".git").mkdir()  # the state clone's own repo is allowed
    r = _run("--check", str(state), scratch=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr

    _write(state, "manager/wb/maylene.sav")
    _write(state, "rent/out/x.md")
    _write(state, "e1/backups/n.md")
    _write(state, "e1/wt/.git", "gitdir: elsewhere\n")
    _write(state, "e1/tok.txt", TOKENS["runpod"] + "\n")
    _write(state, "e1/big.txt", "y" * (2 * 1024 * 1024 + 1))
    _write(state, "e2/.gitattributes", "* filter=lfs\n")
    r = _run("--check", str(state), scratch=tmp_path)
    assert r.returncode == 1
    for frag in ("rom-save-archive-or-binary manager/wb/maylene.sav", "run-output rent/out/x.md",
                 "backup-folder e1/backups/n.md", "nested-repo e1/wt/.git", "secret-like-content e1/tok.txt",
                 "over-2MB e1/big.txt", "git-control-file e2/.gitattributes"):
        assert f"OFFENDER {frag}" in r.stdout, frag
