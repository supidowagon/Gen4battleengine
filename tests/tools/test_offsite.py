"""tools/ops/offsite.sh: the state-file selection (--list), the pre-commit hard-exclusion check
(--check), the remote guards and full runs against local throwaway bare repos under tmp_path.
No network remote is ever contacted: every network-URL case is refused before any push."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "ops" / "offsite.sh"


def _bash() -> str | None:
    """A POSIX bash: on Windows Git's own usr/bin/bash.exe (the bash.exe in System32 or WindowsApps is
    the WSL launcher, which cannot see these paths; Git's bin/bash.exe wrapper puts usr/bin ahead of
    any PATH the test sets, so a stand-in command would never run), elsewhere the bash on PATH."""
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
pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _with_bash_tools(env: dict[str, str]) -> dict[str, str]:
    """On Windows, bash's own folder (find, sort, tar, ...) first on PATH, ahead of System32's find.exe
    and sort.exe; a test that prepends a stand-in folder still comes first."""
    if os.name == "nt" and BASH:
        env["PATH"] = str(Path(BASH).parent) + os.pathsep + env.get("PATH", "")
    return env

# Token-shaped strings are assembled at run time so this file holds none.
TOKENS = {
    "runpod": "rp" + "a_" + "A1b2C3d4E5f6G7h8I9j0K1",
    "github": "gh" + "p_" + "x" * 36,
    "openai": "s" + "k-" + "y" * 40,
    "pem": "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n",
    "assign": 'api' + 'key = "' + "z" * 32 + '"',
    "envkey": "RUNPOD" + "_API_KEY=" + "q" * 24,
    "password": "PASS" + "WORD=" + "hunter2hunter2",
    "yamlkey": "api" + "Key: " + "w" * 24,
    "urlcred": "https://" + "user:" + "t0kenvalue" + "@" + "example.com/x",
    "bearer": "Authorization: " + "Bearer " + "a" * 30,
    "hf": "h" + "f_" + "b" * 34,
    "google": "AI" + "za" + "c" * 35,
    "jwt": "ey" + "J" + "d" * 12 + "." + "e" * 12 + "." + "f" * 12,
}
TOKEN_EXT = {"runpod": "txt", "github": "md", "openai": "py", "pem": "txt", "assign": "toml", "envkey": "sh",
             "password": "txt", "yamlkey": "yaml", "urlcred": "md", "bearer": "txt", "hf": "py", "google": "json",
             "jwt": "md"}


def _write(root: Path, rel: str, data: bytes | str = "note\n") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def _env(scratch: Path) -> dict[str, str]:
    env = _with_bash_tools(dict(os.environ))
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
    "handoff/my notes.md",  # names with spaces
    "e1/sub dir/a b.md",
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
    "e1/ROM2.NDS": "rom-save-archive-or-binary",  # upper-case extensions
    "e1/Slot.SAV": "rom-save-archive-or-binary",
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
    "e1/tok_envkey.sh": "secret-like-content",
    "e1/tok_password.txt": "secret-like-content",
    "e1/tok_yamlkey.yaml": "secret-like-content",
    "e1/tok_urlcred.md": "secret-like-content",
    "e1/tok_bearer.txt": "secret-like-content",
    "e1/tok_hf.py": "secret-like-content",
    "e1/tok_google.json": "secret-like-content",
    "e1/tok_jwt.md": "secret-like-content",
    # size
    "e1/big.txt": "over-2MB",
    "e1/big.log": "log-over-1MB",
    # not text
    "e2/blob.dat": "binary",
    "e2/fake.txt": "binary",  # a NUL byte under a text extension
    # a stream folder in upper case is reported, never entered
    "E5/notes.md": "dir:not-lower-case",
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
        _write(s, f"e1/tok_{name}.{TOKEN_EXT[name]}", f"line\n{tok}\n")
    _write(s, "e1/big.txt", "y" * (2 * 1024 * 1024 + 1))
    _write(s, "e1/big.log", "y" * (1024 * 1024 + 1))
    _write(s, "e2/blob.dat", b"ab\x00cd")
    _write(s, "e2/fake.txt", b"text\x00more\n")
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
    _write(state, "e2/fake.md", b"a\x00b\n")
    _write(state, "e2/Slot.SAV")
    r = _run("--check", str(state), scratch=tmp_path)
    assert r.returncode == 1
    for frag in ("rom-save-archive-or-binary manager/wb/maylene.sav", "run-output rent/out/x.md",
                 "backup-folder e1/backups/n.md", "nested-repo e1/wt/.git", "secret-like-content e1/tok.txt",
                 "over-2MB e1/big.txt", "git-control-file e2/.gitattributes", "binary e2/fake.md",
                 "rom-save-archive-or-binary e2/Slot.SAV"):
        assert f"OFFENDER {frag}" in r.stdout, frag


# ---- full runs against local throwaway bare repos ----

ORIGIN_GH = "https://github.com/Owner/Repo.git"


def _genv(tmp_path: Path) -> dict[str, str]:
    """Git with no system or user config but a throwaway identity."""
    cfg = tmp_path / "gitconfig"
    if not cfg.exists():
        cfg.write_text("[user]\n\tname = t\n\temail = t@example.invalid\n[init]\n\tdefaultBranch = main\n")
    env = _with_bash_tools(dict(os.environ))
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(cfg), GIT_TERMINAL_PROMPT="0")
    return env


def _git(*args: str, cwd: Path, env: dict[str, str]) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (args, r.stderr)
    return r.stdout


@pytest.fixture
def proj(tmp_path: Path):
    """A checkout with a commit on main and a feature branch, a bare origin and a bare offsite."""
    env = _genv(tmp_path)
    repo = tmp_path / "my proj"
    repo.mkdir()
    _git("init", "-q", cwd=repo, env=env)
    _write(repo, "a.txt", "a\n")
    _git("add", "a.txt", cwd=repo, env=env)
    _git("commit", "-q", "-m", "a", cwd=repo, env=env)
    _git("branch", "feature", cwd=repo, env=env)
    for name in ("origin.git", "offsite.git"):
        _git("init", "-q", "--bare", str(tmp_path / name), cwd=tmp_path, env=env)
    _git("remote", "add", "origin", (tmp_path / "origin.git").as_posix(), cwd=repo, env=env)
    _git("remote", "add", "offsite", (tmp_path / "offsite.git").as_posix(), cwd=repo, env=env)
    scratch = repo / "_scratch"
    _write(scratch, "handoff/MANAGER.md")
    _write(scratch, "handoff/my notes.md")
    _write(scratch, "e1/sub dir/a b.md")
    _write(scratch, "misc/keep.md")  # outside the allow-list: never copied, never touched
    env.update(NUZLINE_REPO=str(repo), NUZLINE_SCRATCH=str(scratch), NUZLINE_OFFSITE_TEST_LOCAL="1")
    return repo, tmp_path / "offsite.git", env


def _shim(folder: Path, name: str, body: str, executable: bool = True) -> None:
    """A stand-in command <folder>/<name>: LF line endings, mode 755 (Linux runs no file without the
    execute bit; Git Bash on Windows ignores it)."""
    p = folder / name
    p.write_text(body, newline="\n")
    p.chmod(0o755 if executable else 0o644)


def _off(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, SCRIPT.as_posix(), *args], capture_output=True, text=True, env=env, cwd=ROOT)


def _refs(bare: Path, env: dict[str, str]) -> dict[str, str]:
    out = _git("for-each-ref", "--format=%(refname) %(objectname)", cwd=bare, env=env)
    return dict(line.split(" ") for line in out.splitlines())


def _state_files(bare: Path, env: dict[str, str]) -> set[str]:
    out = _git("-c", "core.quotepath=off", "ls-tree", "-r", "--name-only", "refs/heads/scratch-state", cwd=bare, env=env)
    return set(out.splitlines())


@pytest.mark.parametrize("offsite_url, extra", [
    (ORIGIN_GH + "/", {}),
    ("https://github.com/Owner/Repo", {}),
    ("https://GitHub.com/OWNER/repo.git", {}),
    ("git@github.com:Owner/Repo.git", {}),
    ("ssh://git@github.com/Owner/Repo", {}),
    ("https://github.com/someone/private.git", {"remote.offsite.pushurl": "git@github.com:owner/repo"}),
    ("https://github.com/someone/private.git",
     {"url." + ORIGIN_GH + ".pushInsteadOf": "https://github.com/someone/private.git"}),
    ("https://github.com/someone/private.git", {"url." + ORIGIN_GH + ".insteadOf": "https://github.com/someone/private.git"}),
])
def test_refuses_any_push_url_that_names_origin(proj, offsite_url, extra):
    repo, off, env = proj
    _git("remote", "set-url", "origin", ORIGIN_GH, cwd=repo, env=env)
    _git("remote", "set-url", "offsite", offsite_url, cwd=repo, env=env)
    for k, v in extra.items():
        _git("config", k, v, cwd=repo, env=env)
    for mode in ("--dry-run", None):
        r = _off(*([mode] if mode else []), env=env)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "is origin's repo" in r.stderr
        assert "== branches" not in r.stdout


def test_refuses_a_local_offsite_path_that_is_origin(proj, tmp_path):
    repo, _, env = proj
    for form in [(tmp_path / "origin.git").as_posix() + "/", str(tmp_path / "origin"), str(tmp_path / "ORIGIN.git")]:
        _git("remote", "set-url", "offsite", form, cwd=repo, env=env)
        r = _off(env=env)
        assert r.returncode == 2 and "is origin's repo" in r.stderr, (form, r.stderr)


def test_refuses_unless_gh_confirms_private(proj, tmp_path):
    repo, _, env = proj
    _git("remote", "set-url", "offsite", "https://github.com/someone/private.git", cwd=repo, env=env)
    shim = tmp_path / "shim"
    shim.mkdir()
    env = dict(env, PATH=str(shim) + os.pathsep + env["PATH"])
    for body, frag in [("echo false", "not confirmed private"), ("echo 'not logged in' >&2; exit 1", "failed (exit 1;"),
                       ("echo", "not confirmed private"), ("echo true; exit 3", "failed (exit 3;")]:
        _shim(shim, "gh", f"#!/bin/sh\n{body}\n")
        r = _off("--dry-run", env=env)  # the test switch never covers a network URL
        assert r.returncode == 2 and frag in r.stderr, (body, r.stderr)
        assert "== branches" not in r.stdout
    # A gh the shell finds but cannot run (no execute bit) is named as such, not as a login problem.
    # POSIX only, and only with no other gh on PATH: the shell would run that one instead.
    if os.name != "nt" and shutil.which("gh", path=os.environ.get("PATH", "")) is None:
        _shim(shim, "gh", "#!/bin/sh\necho true\n", executable=False)
        r = _off("--dry-run", env=env)
        assert r.returncode == 2 and "cannot be run (exit 126" in r.stderr, r.stderr
        assert "== branches" not in r.stdout
    _git("remote", "set-url", "offsite", (tmp_path / "offsite.git").as_posix(), cwd=repo, env=env)
    env.pop("NUZLINE_OFFSITE_TEST_LOCAL")
    r = _off("--dry-run", env=env)
    assert r.returncode == 2 and "local path" in r.stderr


def test_refuses_mirror_and_push_refspec_and_a_scratch_state_branch(proj):
    repo, off, env = proj
    _git("config", "remote.offsite.mirror", "true", cwd=repo, env=env)
    r = _off(env=env)
    assert r.returncode == 2 and "mirror" in r.stderr
    _git("config", "--unset", "remote.offsite.mirror", cwd=repo, env=env)
    _git("config", "remote.offsite.push", "+refs/heads/*:refs/heads/*", cwd=repo, env=env)
    r = _off(env=env)
    assert r.returncode == 2 and "remote.offsite.push" in r.stderr
    _git("config", "--unset", "remote.offsite.push", cwd=repo, env=env)
    _git("branch", "scratch-state", cwd=repo, env=env)
    r = _off(env=env)
    assert r.returncode == 2 and "scratch-state" in r.stderr
    assert _refs(off, env) == {}


def test_no_force_no_delete_and_exit_4_on_a_rejected_branch(proj, tmp_path):
    repo, off, env = proj
    # offsite holds a branch that is gone locally and a main that diverged from local main
    other = tmp_path / "other"
    _git("init", "-q", str(other), cwd=tmp_path, env=env)
    _write(other, "b.txt", "b\n")
    _git("add", "b.txt", cwd=other, env=env)
    _git("commit", "-q", "-m", "b", cwd=other, env=env)
    _git("push", "-q", off.as_posix(), "HEAD:refs/heads/main", "HEAD:refs/heads/gone", cwd=other, env=env)
    before = _refs(off, env)
    r = _off(env=env)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REJECTED" in r.stdout
    after = _refs(off, env)
    assert after["refs/heads/main"] == before["refs/heads/main"]
    assert after["refs/heads/gone"] == before["refs/heads/gone"]
    assert "refs/heads/feature" in after and "refs/heads/scratch-state" in after
    assert "FORCED" not in r.stdout and "DELETED" not in r.stdout


def test_run_mirrors_state_and_deletions_stay_inside_the_clone(proj):
    repo, off, env = proj
    scratch = repo / "_scratch"
    r = _off(env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _state_files(off, env) == {"handoff/MANAGER.md", "handoff/my notes.md", "e1/sub dir/a b.md"}
    state = scratch / "offsite" / "state"
    assert _git("config", "core.longpaths", cwd=state, env=env).strip() == "true"
    assert _git("config", "core.autocrlf", cwd=state, env=env).strip() == "false"
    assert _git("remote", "get-url", "--push", "--all", "offsite", cwd=state, env=env).splitlines() == [off.as_posix()]
    assert not any((scratch / "offsite" / "tmp").iterdir())

    (scratch / "e1" / "sub dir" / "a b.md").unlink()
    _write(scratch, "handoff/new file.md", "new\n")
    _write(scratch, "handoff/MANAGER.md", "changed\n")
    r = _off("--dry-run", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "vs last state commit: 1 new, 1 changed, 1 deleted" in r.stdout
    r = _off(env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _state_files(off, env) == {"handoff/MANAGER.md", "handoff/my notes.md", "handoff/new file.md"}
    assert not (state / "e1").exists()
    for rel in ("misc/keep.md", "handoff/MANAGER.md", "handoff/my notes.md"):
        assert (scratch / rel).is_file(), rel  # the source is never touched
    assert (repo / "a.txt").is_file()

    # a state clone without its remote (restored by hand) gets it back
    _git("remote", "remove", "offsite", cwd=state, env=env)
    r = _off(env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    # a leftover index.lock is refused
    (state / ".git" / "index.lock").write_text("")
    r = _off(env=env)
    assert r.returncode == 2 and "index.lock" in r.stderr


def test_run_aborts_on_an_offender_in_the_copy(proj, tmp_path):
    """A file that changes after selection (here: a tar shim that writes a token into the clone after
    extracting) is caught by the re-check; nothing is committed or pushed."""
    repo, off, env = proj
    shim = tmp_path / "shim"
    shim.mkdir()
    _shim(shim, "tar",
          '#!/bin/bash\n/usr/bin/tar "$@"; rc=$?\n'
          'if [ "$3" = -xf ]; then printf "x\\n%s\\n" "$LEAK" > "$2/handoff/MANAGER.md"; fi\nexit $rc\n')
    env = dict(env, PATH=str(shim) + os.pathsep + env["PATH"], LEAK=TOKENS["github"])
    r = _off(env=env)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "OFFENDER secret-like-content handoff/MANAGER.md" in r.stdout and "ABORTED" in r.stdout
    assert "refs/heads/scratch-state" not in _refs(off, env)


def test_run_aborts_on_an_offender_left_in_the_clone(proj):
    """No stand-in command: a symlink planted in the state clone is not a selected file, so the mirror
    step neither removes nor overwrites it; the pre-commit re-check catches it (exit 3), nothing is
    left staged and the state branch on offsite stays where it was."""
    repo, off, env = proj
    scratch = repo / "_scratch"
    state = scratch / "offsite" / "state"
    r = _off(env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    before = _refs(off, env)["refs/heads/scratch-state"]
    try:
        (state / "handoff" / "link.md").symlink_to(scratch / "misc" / "keep.md")
    except OSError as exc:  # Windows without the symlink privilege
        pytest.skip(f"cannot create a symlink here: {exc}")
    _write(scratch, "handoff/MANAGER.md", "changed\n")
    r = _off(env=env)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "OFFENDER not-a-regular-file handoff/link.md" in r.stdout and "ABORTED" in r.stdout
    assert _refs(off, env)["refs/heads/scratch-state"] == before
    assert _git("diff", "--cached", "--name-only", cwd=state, env=env) == ""
