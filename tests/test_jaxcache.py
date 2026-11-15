"""The persistent caches (docs/design.md §6): cache root, table and code versions, and the
serialized-trace cache ``cached_export``."""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline import jaxcache
from nuzline.jaxcache import (
    cache_root,
    cached_export,
    code_version,
    prune_other_versions,
    table_version,
)

TABLES_DIR = Path(__file__).resolve().parents[1] / "data" / "gen4_platinum"


def test_cache_root_env_override(tmp_path):
    assert cache_root({"NUZLINE_CACHE_DIR": str(tmp_path)}) == tmp_path
    # From a source checkout the default is the repo's .cache directory.
    repo = Path(jaxcache.__file__).resolve().parents[2]
    assert cache_root({}) == repo / ".cache"


def test_cache_root_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(jaxcache, "PACKAGE_DIR", tmp_path / "site-packages" / "nuzline")
    assert cache_root({}) == Path.home() / ".cache" / "nuzline"


def test_table_version_changes_with_tables(tmp_path):
    if not (TABLES_DIR / "tables.npz").exists():
        pytest.skip("tables not built (run tools/build_tables.py)")
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        for f in ("manifest.json", "tables.npz"):
            shutil.copyfile(TABLES_DIR / f, d / f)
    v = table_version(a)
    assert len(v) == 16 and int(v, 16) >= 0
    assert table_version(b) == v == table_version(TABLES_DIR)
    data = bytearray((b / "tables.npz").read_bytes())
    data[len(data) // 2] ^= 1
    (b / "tables.npz").write_bytes(bytes(data))
    assert table_version(b) != v


@pytest.fixture
def fresh_code_version():
    """``code_version`` is memoized: recompute it in the test and again after it, so a
    monkeypatched package directory never leaks into later tests."""
    code_version.cache_clear()
    yield
    code_version.cache_clear()


def test_code_version_changes_with_source(tmp_path, monkeypatch, fresh_code_version):
    pkg = tmp_path / "nuzline"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "mod.py").write_text("X = 1\n")
    (pkg / "notes.txt").write_text("not source")
    monkeypatch.setattr(jaxcache, "PACKAGE_DIR", pkg)
    v = code_version()
    assert len(v) == 16
    (pkg / "notes.txt").write_text("still not source")
    code_version.cache_clear()
    assert code_version() == v
    (pkg / "sub" / "mod.py").write_text("X = 2\n")
    assert code_version() == v  # memoized: one hash per process
    code_version.cache_clear()
    assert code_version() != v


def _version_dirs(parent, *names):
    for n in names:
        (parent / n).mkdir(parents=True)
        (parent / n / "entry.bin").write_bytes(b"x")


def test_prune_other_versions_keeps_current_and_non_version_entries(tmp_path):
    old16, old32, current = "0123456789abcdef", "fedcba9876543210" * 2, "aaaaaaaaaaaaaaaa"
    parent = tmp_path / "jax"
    _version_dirs(parent, old16, old32, current, "notes", "0123456789abcdeg", "0123")
    (parent / "fedcba9876543210").write_bytes(b"a file, not a directory")
    prune_other_versions(parent / current, root=tmp_path)
    assert sorted(p.name for p in parent.iterdir()) == sorted(
        [current, "notes", "0123456789abcdeg", "0123", "fedcba9876543210"]
    )
    assert (parent / current / "entry.bin").is_file()


def test_prune_other_versions_never_leaves_the_cache_root(tmp_path):
    outside = tmp_path / "elsewhere"
    _version_dirs(outside, "0123456789abcdef", "aaaaaaaaaaaaaaaa")
    prune_other_versions(outside / "aaaaaaaaaaaaaaaa", root=tmp_path / "cache")
    assert (outside / "0123456789abcdef" / "entry.bin").is_file()
    # The cache root's own siblings are outside it too.
    root = tmp_path / "bbbbbbbbbbbbbbbb"
    _version_dirs(tmp_path, "cccccccccccccccc")
    prune_other_versions(root, root=root)
    assert (tmp_path / "cccccccccccccccc" / "entry.bin").is_file()


def test_prune_other_versions_defaults_to_the_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NUZLINE_CACHE_DIR", str(tmp_path / "cache"))
    _version_dirs(tmp_path / "cache" / "jax", "0123456789abcdef", "aaaaaaaaaaaaaaaa")
    _version_dirs(tmp_path / "other", "0123456789abcdef", "aaaaaaaaaaaaaaaa")
    prune_other_versions(tmp_path / "cache" / "jax" / "aaaaaaaaaaaaaaaa")
    prune_other_versions(tmp_path / "other" / "aaaaaaaaaaaaaaaa")
    assert not (tmp_path / "cache" / "jax" / "0123456789abcdef").exists()
    assert (tmp_path / "other" / "0123456789abcdef").is_dir()


@pytest.fixture
def export_dir(tmp_path, monkeypatch):
    """An empty cache root, and no programs remembered from earlier calls (a new run)."""
    monkeypatch.setenv("NUZLINE_CACHE_DIR", str(tmp_path))
    jaxcache._LOADED.clear()
    yield tmp_path / "export" / "v1"
    jaxcache._LOADED.clear()


class _Counted:
    """``x * scale + y["b"]`` that counts how often its Python body runs (traces)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, y, scale):
        self.calls += 1
        return {"out": x * scale + y["b"], "sum": jnp.sum(x)}


def _args(n=4):
    return (jnp.arange(n, dtype=jnp.float32), {"b": jnp.ones(n, jnp.float32)})


def test_cached_export_roundtrip(export_dir):
    from functools import partial

    fn = _Counted()
    args = _args()
    first = cached_export(
        partial(fn, scale=3.0), args, name="f", static={"scale": 3.0}, version="v1"
    )
    got = first(*args)
    assert fn.calls == 1
    (written,) = export_dir.glob("f-*.bin")
    assert len(written.stem.removeprefix("f-")) == 16
    want = jax.jit(partial(fn, scale=3.0))(*args)
    calls = fn.calls

    jaxcache._LOADED.clear()  # as in a new process: only the file remains
    again = cached_export(
        partial(fn, scale=3.0), args, name="f", static={"scale": 3.0}, version="v1"
    )
    out = again(*args)
    assert fn.calls == calls  # loaded, not traced
    assert list(export_dir.glob("f-*.bin")) == [written]
    for k in ("out", "sum"):
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(want[k]))
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(want[k]))


def test_cached_export_key_changes_with_static_and_shape(export_dir):
    from functools import partial

    fn = _Counted()
    cached_export(partial(fn, scale=1.0), _args(), name="f", static={"scale": 1.0}, version="v1")
    cached_export(partial(fn, scale=2.0), _args(), name="f", static={"scale": 2.0}, version="v1")
    cached_export(partial(fn, scale=1.0), _args(8), name="f", static={"scale": 1.0}, version="v1")
    assert fn.calls == 3
    assert len(list(export_dir.glob("f-*.bin"))) == 3
    out = cached_export(
        partial(fn, scale=2.0), _args(), name="f", static={"scale": 2.0}, version="v1"
    )(*_args())
    np.testing.assert_array_equal(np.asarray(out["out"]), np.arange(4) * 2.0 + 1)


def test_cached_export_corrupt_file_recovers(export_dir):
    from functools import partial

    fn = _Counted()

    def f():  # a fresh callable each time, as in a new process (no in-memory trace cache)
        return partial(fn, scale=3.0)

    cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")
    (written,) = export_dir.glob("f-*.bin")
    written.write_bytes(b"not a serialized program")
    jaxcache._LOADED.clear()
    out = cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")(*_args())
    assert fn.calls == 2  # exported again
    np.testing.assert_array_equal(np.asarray(out["out"]), np.arange(4) * 3.0 + 1)
    assert written.read_bytes() != b"not a serialized program"
    jaxcache._LOADED.clear()
    cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")
    assert fn.calls == 2  # the rewritten file loads


def test_cached_export_program_failing_its_first_call_recovers(export_dir):
    from functools import partial

    fn = _Counted()

    def f():
        return partial(fn, scale=3.0)

    cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")
    (written,) = export_dir.glob("f-*.bin")
    # A program that deserializes fine but fails when called: one for other shapes.
    other = jax.export.export(jax.jit(f()))(*_args(8))
    written.write_bytes(other.serialize())
    jaxcache._LOADED.clear()
    out = cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")(*_args())
    assert fn.calls == 3  # the first export, the stand-in, and the re-export
    np.testing.assert_array_equal(np.asarray(out["out"]), np.arange(4) * 3.0 + 1)
    jaxcache._LOADED.clear()
    again = cached_export(f(), _args(), name="f", static={"scale": 3.0}, version="v1")
    np.testing.assert_array_equal(np.asarray(again(*_args())["out"]), np.arange(4) * 3.0 + 1)
    assert fn.calls == 3  # the rewritten file loads and runs


def test_cached_export_write_failure_still_returns_the_program(export_dir, monkeypatch):
    from functools import partial

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr(jaxcache.os, "replace", fail)
    fn = _Counted()
    program = cached_export(
        partial(fn, scale=2.0), _args(), name="f", static={"scale": 2.0}, version="v1"
    )
    np.testing.assert_array_equal(np.asarray(program(*_args())["out"]), np.arange(4) * 2.0 + 1)
    assert list(export_dir.iterdir()) == []  # no file, and no temporary left behind


def test_cached_export_prunes_other_versions(export_dir):
    from functools import partial

    old = export_dir.parent / "0123456789abcdef"
    old.mkdir(parents=True)
    (old / "f-0000000000000000.bin").write_bytes(b"stale")
    fn = _Counted()
    cached_export(partial(fn, scale=1.0), _args(), name="f", static={"scale": 1.0}, version="v1")
    assert not old.exists()
    assert len(list(export_dir.glob("f-*.bin"))) == 1


def test_export_key_includes_the_backend(monkeypatch):
    leaves, treedef = jax.tree.flatten(_args())
    specs = [jaxcache._spec(leaf) for leaf in leaves]
    key = jaxcache._export_key({"scale": 1.0}, treedef, specs)
    assert jaxcache._export_key({"scale": 1.0}, treedef, specs) == key
    monkeypatch.setattr(jaxcache.jax, "default_backend", lambda: "gpu")
    assert jaxcache._export_key({"scale": 1.0}, treedef, specs) != key


@pytest.fixture
def game_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NUZLINE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(jaxcache, "set_compile_cache_dir", lambda directory: None)
    return tmp_path


def test_game_dirs_do_not_prune_each_other(game_cache):
    a = jaxcache.enable_compile_cache("a" * 32, "platinum_us")
    b = jaxcache.enable_compile_cache("b" * 32, "platinum_kaizo_nightmare_final")
    assert a.is_dir() and b.is_dir() and a.parent != b.parent
    assert a.parent == game_cache / "jax" / "platinum_us"


def test_other_versions_of_same_game_pruned(game_cache):
    old = jaxcache.enable_compile_cache("a" * 32, "platinum_us")
    new = jaxcache.enable_compile_cache("c" * 32, "platinum_us")
    assert not old.exists() and new.is_dir()


def test_legacy_flat_dirs_pruned(game_cache):
    legacy = game_cache / "jax" / ("d" * 32)
    legacy.mkdir(parents=True)
    kept = jaxcache.enable_compile_cache("b" * 32, "platinum_kaizo_nightmare_final")
    jaxcache.enable_compile_cache("a" * 32, "platinum_us")
    assert not legacy.exists() and kept.is_dir()


def test_default_game_is_vanilla(game_cache):
    assert jaxcache.enable_compile_cache("a" * 32).parent.name == "platinum_us"


def test_cap_directory_deletes_oldest_first(tmp_path):
    import os

    for i, name in enumerate(("old", "mid", "new")):
        path = tmp_path / "sub" / name if name == "mid" else tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x" * 100)
        os.utime(path, (1000 + i, 1000 + i))
    jaxcache.cap_directory(tmp_path, 200)
    assert not (tmp_path / "old").exists()
    assert (tmp_path / "sub" / "mid").exists() and (tmp_path / "new").exists()
    jaxcache.cap_directory(tmp_path, 1000)  # under the cap: nothing more goes
    assert (tmp_path / "sub" / "mid").exists()


def test_source_edit_changes_the_trace_cache_key(export_dir, monkeypatch, fresh_code_version):
    """The serialized-trace cache loads a program by name, so its key must follow the
    source: every caller passes ``version=code_version()`` (prep/states.py, search/expand.py,
    search/leaves.py), and an engine source edit gives a new export directory, so the old
    program is never loaded. Only the XLA compilation cache is keyed by ``jax_version``."""
    from functools import partial

    root = export_dir.parent.parent
    for module in ("prep/states.py", "search/expand.py", "search/leaves.py"):
        text = (jaxcache.PACKAGE_DIR / module).read_text(encoding="utf-8")
        assert "version=code_version()" in text, module
    pkg = root / "pkg" / "nuzline"
    (pkg / "engine").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "engine" / "damage.py").write_text("POWER = 1\n")
    monkeypatch.setattr(jaxcache, "PACKAGE_DIR", pkg)
    fn = _Counted()
    before = code_version()
    cached_export(partial(fn, scale=1.0), _args(), name="f", static={}, version=before)
    (pkg / "engine" / "damage.py").write_text("POWER = 2\n")
    code_version.cache_clear()
    after = code_version()
    assert after != before
    jaxcache._LOADED.clear()  # a new process
    cached_export(partial(fn, scale=1.0), _args(), name="f", static={}, version=after)
    assert fn.calls == 2  # traced again: the first program was not loaded
    assert {p.name for p in (root / "export").iterdir()} >= {before, after}
    assert jaxcache.jax_version() not in {before, after}
