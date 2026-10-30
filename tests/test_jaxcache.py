"""The persistent caches (docs/design.md §6): cache root, table and code versions, and the
serialized-trace cache ``cached_export``."""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline import jaxcache
from nuzline.jaxcache import cache_root, cached_export, code_version, table_version

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


def test_code_version_changes_with_source(tmp_path, monkeypatch):
    pkg = tmp_path / "nuzline"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "mod.py").write_text("X = 1\n")
    (pkg / "notes.txt").write_text("not source")
    monkeypatch.setattr(jaxcache, "PACKAGE_DIR", pkg)
    v = code_version()
    assert len(v) == 16
    (pkg / "notes.txt").write_text("still not source")
    assert code_version() == v
    (pkg / "sub" / "mod.py").write_text("X = 2\n")
    assert code_version() != v


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
