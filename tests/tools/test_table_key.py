"""tools/ops/table_key.py: the inputs that key a table build (tools/ops/build_tables.sh)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("table_key", ROOT / "tools" / "ops" / "table_key.py")
tk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tk)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def _git_repo(root: Path, files: dict[str, str], untracked: dict[str, str] = {}) -> None:
    _write(root, files)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    _write(root, untracked)


def test_walk_follows_local_function_level_and_relative_imports(tmp_path):
    _write(tmp_path, {
        "tools/__init__.py": "",
        "tools/a.py": "import numpy\nfrom tools.b import x\ndef f():\n    from .c import y\n",
        "tools/b.py": "from nuzline.data import tables\nx = 1\n",
        "tools/c.py": "y = 2\n",
        "tools/unused.py": "",
        "src/nuzline/__init__.py": "",
        "src/nuzline/data/__init__.py": "",
        "src/nuzline/data/tables.py": "MANIFEST_VERSION = 1\n",
    })
    got = set(tk.trace(tmp_path, ("tools.a",)))
    assert got == {"tools", "tools.a", "tools.b", "tools.c", "nuzline", "nuzline.data", "nuzline.data.tables"}


def test_walk_refuses_a_run_time_import(tmp_path):
    _write(tmp_path, {"tools/__init__.py": "", "tools/a.py": "import importlib\n"})
    with pytest.raises(tk.KeyError_):
        tk.trace(tmp_path, ("tools.a",))


def test_missing_entry_is_refused(tmp_path):
    with pytest.raises(tk.KeyError_):
        tk.trace(tmp_path, ("tools.nothing",))


def test_entry_modules_are_every_module_build_tables_runs():
    assert tk.entry_modules(ROOT / "tools" / "ops" / "build_tables.sh") == (
        "tools.build_tables", "tools.extract_decomp", "tools.extract_kaizo_docs",
        "tools.extract_pkcalc", "tools.extract_rom", "tools.kaizo.overrides", "tools.rom.symbols",
    )


def test_entry_modules_skip_comments(tmp_path):
    s = tmp_path / "b.sh"
    s.write_text("# python -m tools.old\n\"${PY[@]}\" -m tools.new --x\n", encoding="utf-8")
    assert tk.entry_modules(s) == ("tools.new",)


def test_repo_walk_holds_the_manifest_version_and_no_tests():
    got = tk.trace(ROOT, tk.entry_modules(ROOT / "tools" / "ops" / "build_tables.sh"))
    assert "nuzline.data.tables" in got  # MANIFEST_VERSION
    assert not any(n.startswith(("tests", "tools.ops")) for n in got)


def test_tracked_inputs_select_data_and_non_python_tool_and_src_files(tmp_path):
    _git_repo(tmp_path, {
        "data/a.json": "1", "data/kaizo/b.toml": "1", "tools/x.py": "", "tools/y.json": "1",
        "tools/ops/z.sh": "", "tools/ops/w.txt": "", "src/p/q.py": "", "src/p/r.json": "1",
        "src/p/STATE.md": "", "uv.lock": "", ".python-version": "3.12", "tests/t.py": "",
        "docs/d.md": "", "pyproject.toml": "",
    }, untracked={"data/u.json": "1"})
    assert tk.tracked_inputs(tmp_path) == [
        ".python-version", "data/a.json", "data/kaizo/b.toml", "src/p/r.json", "tools/y.json", "uv.lock",
    ]


def test_local_inputs_take_the_environment_first_and_only_build_inputs(tmp_path, monkeypatch):
    for env in tk.LOCAL_INPUTS.values():
        monkeypatch.delenv(env, raising=False)
    cfg = tmp_path / "local.toml"
    cfg.write_text("vanilla_rom = 'roms/v.nds'\nkaizo_patch = 'p.xdelta'\nkaizo_save = 's.sav'\n", encoding="utf-8")
    monkeypatch.setenv("NUZLINE_KAIZO_PATCH", str(tmp_path / "env.xdelta"))
    monkeypatch.setenv("NUZLINE_KAIZO_DOCS", str(tmp_path / "d.xlsx"))
    assert tk.local_inputs(tmp_path, cfg) == {
        "vanilla_rom": tmp_path / "roms" / "v.nds",
        "kaizo_patch": tmp_path / "env.xdelta",
        "kaizo_docs": tmp_path / "d.xlsx",
    }
    assert tk.local_inputs(tmp_path, None) == {"kaizo_patch": tmp_path / "env.xdelta", "kaizo_docs": tmp_path / "d.xlsx"}


def test_key_follows_the_build_script_common_sh_and_gitignore(tmp_path):
    root, ops = tmp_path / "r", tmp_path / "ops"
    _git_repo(root, {"tools/__init__.py": "", "tools/m.py": "", ".gitignore": "data/x/\n"})
    _write(ops, {"build_tables.sh": "\"$PY\" -m tools.m\n", "common.sh": "TABLE_ALLOW='a'\n"})
    key = lambda: tk.key_of(tk.entries(root, None, None, None, ops))  # noqa: E731
    k0 = key()
    assert key() == k0
    (ops / "common.sh").write_text("TABLE_ALLOW='b'\n", encoding="utf-8")
    k1 = key()
    (root / ".gitignore").write_text("data/y/\n", encoding="utf-8")
    k2 = key()
    (ops / "build_tables.sh").write_text("\"$PY\" -m tools.m --game kaizo\n", encoding="utf-8")
    k3 = key()
    (root / "tools" / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert len({k0, k1, k2, k3, key()}) == 5


def test_key_depends_on_every_entry():
    e = [("module a", "1"), ("input b", "2")]
    assert tk.key_of(e) == tk.key_of(list(e))
    assert tk.key_of(e) != tk.key_of([("module a", "1"), ("input b", "3")])
