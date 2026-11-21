"""tools/ops/table_key.py: the import walk that keys a table build (tools/ops/build_tables.sh)."""

import importlib.util
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


def test_repo_walk_holds_the_table_tools_and_the_manifest_version():
    got = tk.trace(ROOT)
    assert set(tk.ENTRY) <= set(got)
    assert "nuzline.data.tables" in got  # MANIFEST_VERSION
    assert not any(n.startswith(("tests", "tools.ops")) for n in got)


def test_key_depends_on_every_entry():
    e = [("module a", "1"), ("input b", "2")]
    assert tk.key_of(e) == tk.key_of(list(e))
    assert tk.key_of(e) != tk.key_of([("module a", "1"), ("input b", "3")])
