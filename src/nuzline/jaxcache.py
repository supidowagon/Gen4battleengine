"""Persistent caches that make repeat runs start fast (docs/design.md §6).

* The XLA compilation cache (``enable_compile_cache``): compiled executables survive
  restarts. Its directory is keyed by the table and code versions (spec §6: "cache keys
  include the table version").
* The serialized-trace cache (``cached_export``): the traced program of a function at
  given argument shapes, dtypes and shardings is saved with ``jax.export`` and loaded by
  later runs, so they skip tracing. The file name hashes everything the program depends
  on besides argument values (static arguments, avals, shardings, device count and
  ``code_version()``), so a source edit never loads an old program.

Both live under ``cache_root()``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import jax

__all__ = [
    "cache_root",
    "cached_export",
    "code_version",
    "enable_compile_cache",
    "set_compile_cache_dir",
    "table_version",
]

PACKAGE_DIR = Path(__file__).resolve().parent

_LOADED: dict[Path, Callable] = {}  # programs exported or loaded by this process


def _sha16(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()[:16]


def cache_root(environ: Mapping[str, str] = os.environ) -> Path:
    """``$NUZLINE_CACHE_DIR`` if set; else ``<repo>/.cache`` when running from a source
    checkout (``src/nuzline`` below a ``pyproject.toml``); else ``~/.cache/nuzline``."""
    if environ.get("NUZLINE_CACHE_DIR"):
        return Path(environ["NUZLINE_CACHE_DIR"])
    repo = PACKAGE_DIR.parents[1]
    if PACKAGE_DIR.parent.name == "src" and (repo / "pyproject.toml").is_file():
        return repo / ".cache"
    return Path.home() / ".cache" / "nuzline"


def table_version(tables_dir: Path) -> str:
    """16 hex chars of the sha256 of ``manifest.json`` and ``tables.npz`` in
    ``tables_dir``."""
    d = Path(tables_dir)
    return _sha16((d / "manifest.json").read_bytes(), (d / "tables.npz").read_bytes())


def code_version() -> str:
    """16 hex chars of the sha256 of every ``*.py`` under the package (sorted relative
    paths and their bytes) and the jax and jaxlib versions."""
    import jaxlib

    chunks = [jax.__version__.encode(), jaxlib.__version__.encode()]
    for path in sorted(PACKAGE_DIR.rglob("*.py"), key=lambda p: p.relative_to(PACKAGE_DIR)):
        chunks += [path.relative_to(PACKAGE_DIR).as_posix().encode(), b"\0", path.read_bytes()]
    return _sha16(*chunks)


def set_compile_cache_dir(directory: Path) -> None:
    """Point JAX's persistent compilation cache at ``directory``, caching every compile
    that takes at least a second. Takes effect only before the first compile of the
    process (JAX opens the cache once)."""
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def enable_compile_cache(version: str) -> Path:
    """Enable the XLA compilation cache in ``cache_root()/"jax"/version``; returns it."""
    directory = cache_root() / "jax" / version
    set_compile_cache_dir(directory)
    return directory


def _spec(leaf) -> jax.ShapeDtypeStruct:
    # Only a committed array (one placed with ``device_put``) pins its sharding; an
    # uncommitted one follows the others, and exporting it with its default
    # single-device sharding next to arrays sharded over several devices fails.
    committed = isinstance(leaf, jax.Array) and leaf.committed
    sharding = leaf.sharding if committed else None
    aval = jax.typeof(leaf)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=sharding, weak_type=aval.weak_type)


def cached_export(fn: Callable, args: tuple, *, name: str, static: dict, version: str) -> Callable:
    """A callable equal to ``jax.jit(fn)`` on arguments shaped like ``args``.

    ``fn``'s program at ``args``' shapes, dtypes and shardings (``static`` names
    whatever else ``fn`` closes over that changes it) is exported to
    ``cache_root()/"export"/version/<name>-<key>.bin`` on first use and deserialized
    by later runs, which then skip tracing ``fn``. A file that fails to deserialize is
    deleted and the program exported again."""
    leaves, treedef = jax.tree.flatten(args)
    specs = [_spec(leaf) for leaf in leaves]
    key = _sha16(
        repr(sorted(static.items())).encode(),
        str(treedef).encode(),
        repr([(s.shape, str(s.dtype), s.weak_type, repr(s.sharding)) for s in specs]).encode(),
        str(jax.device_count()).encode(),
        code_version().encode(),
    )
    path = cache_root() / "export" / version / f"{name}-{key}.bin"
    if path in _LOADED:
        return _LOADED[path]
    exported = None
    if path.is_file():
        try:
            exported = jax.export.deserialize(bytearray(path.read_bytes()))
        except Exception:  # noqa: BLE001 - any bad file: re-export, never crash
            path.unlink(missing_ok=True)
    if exported is None:
        exported = jax.export.export(jax.jit(fn))(*jax.tree.unflatten(treedef, specs))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(exported.serialize())
        os.replace(tmp, path)
    _LOADED[path] = jax.jit(exported.call)
    return _LOADED[path]
