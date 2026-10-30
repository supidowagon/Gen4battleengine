"""Persistent caches that make repeat runs start fast (docs/design.md §6).

* The XLA compilation cache (``enable_compile_cache``): compiled executables survive
  restarts. Its directory is keyed by the table and code versions (spec §6: "cache keys
  include the table version").
* The serialized-trace cache (``cached_export``): the traced program of a function at
  given argument shapes, dtypes and shardings is saved with ``jax.export`` and loaded by
  later runs, so they skip tracing. The file name hashes everything the program depends
  on besides argument values (static arguments, avals, shardings, device count, JAX
  backend and ``code_version()``), so a source edit never loads an old program.

Both live under ``cache_root()``, one directory per version; enabling or writing one
deletes the other versions' sibling directories (``prune_other_versions``), so source
edits do not pile up stale caches.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import jax

__all__ = [
    "cache_root",
    "cached_export",
    "code_version",
    "enable_compile_cache",
    "prune_other_versions",
    "set_compile_cache_dir",
    "table_version",
]

PACKAGE_DIR = Path(__file__).resolve().parent

_LOADED: dict[Path, Callable] = {}  # programs exported or loaded by this process

# A version directory name: ``code_version()`` (16 hex chars), optionally after
# ``table_version()`` (another 16).
_VERSION_DIR = re.compile(r"[0-9a-f]{16}(?:[0-9a-f]{16})?")


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


@functools.cache
def code_version() -> str:
    """16 hex chars of the sha256 of every ``*.py`` under the package (sorted relative
    paths and their bytes) and the jax and jaxlib versions. Computed once per process
    (``code_version.cache_clear()`` recomputes it)."""
    import jaxlib

    chunks = [jax.__version__.encode(), jaxlib.__version__.encode()]
    for path in sorted(PACKAGE_DIR.rglob("*.py"), key=lambda p: p.relative_to(PACKAGE_DIR)):
        chunks += [path.relative_to(PACKAGE_DIR).as_posix().encode(), b"\0", path.read_bytes()]
    return _sha16(*chunks)


def prune_other_versions(directory: Path, root: Path | None = None) -> None:
    """Delete the siblings of ``directory`` named like a version directory (16 or 32 hex
    chars), keeping ``directory``. Does nothing unless ``directory``'s parent is inside
    ``root`` (default ``cache_root()``), so nothing outside the cache is touched. Errors
    (say, a file another process holds open) are ignored."""
    directory = Path(directory).resolve()
    root = Path(cache_root() if root is None else root).resolve()
    parent = directory.parent
    if parent != root and root not in parent.parents:
        return
    try:
        siblings = list(parent.iterdir())
    except OSError:
        return
    for sibling in siblings:
        if (
            sibling.name != directory.name
            and _VERSION_DIR.fullmatch(sibling.name)
            and sibling.is_dir()
            and not sibling.is_symlink()
        ):
            shutil.rmtree(sibling, ignore_errors=True)


def set_compile_cache_dir(directory: Path) -> None:
    """Point JAX's persistent compilation cache at ``directory``, caching every compile
    that takes at least a second. Takes effect only before the first compile of the
    process (JAX opens the cache once)."""
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def enable_compile_cache(version: str) -> Path:
    """Enable the XLA compilation cache in ``cache_root()/"jax"/version``, deleting the
    other versions' directories; returns it."""
    directory = cache_root() / "jax" / version
    prune_other_versions(directory)
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


def _export_key(static: dict, treedef, specs: list[jax.ShapeDtypeStruct]) -> str:
    """16 hex chars naming the program: everything it depends on besides argument
    values."""
    return _sha16(
        repr(sorted(static.items())).encode(),
        str(treedef).encode(),
        repr([(s.shape, str(s.dtype), s.weak_type, repr(s.sharding)) for s in specs]).encode(),
        str(jax.device_count()).encode(),
        jax.default_backend().encode(),
        code_version().encode(),
    )


def _export_and_save(fn: Callable, treedef, specs: list, path: Path) -> Callable:
    """Export ``fn`` at ``specs`` and save it to ``path``; the program, even when the
    save fails (the cache is an optimisation: a failed write only costs the next run a
    trace)."""
    exported = jax.export.export(jax.jit(fn))(*jax.tree.unflatten(treedef, specs))
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        prune_other_versions(path.parent)
        tmp.write_bytes(exported.serialize())
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return jax.jit(exported.call)


def cached_export(fn: Callable, args: tuple, *, name: str, static: dict, version: str) -> Callable:
    """A callable equal to ``jax.jit(fn)`` on arguments shaped like ``args``.

    ``fn``'s program at ``args``' shapes, dtypes and shardings (``static`` names
    whatever else ``fn`` closes over that changes it) is exported to
    ``cache_root()/"export"/version/<name>-<key>.bin`` on first use and deserialized
    by later runs, which then skip tracing ``fn``. A file that fails to deserialize, or
    whose program fails on its first call, is deleted and the program exported again.
    A failed write is ignored: the program is still returned."""
    leaves, treedef = jax.tree.flatten(args)
    specs = [_spec(leaf) for leaf in leaves]
    path = cache_root() / "export" / version / f"{name}-{_export_key(static, treedef, specs)}.bin"
    if path in _LOADED:
        return _LOADED[path]
    loaded = None
    if path.is_file():
        try:
            loaded = jax.jit(jax.export.deserialize(bytearray(path.read_bytes())).call)
        except Exception:  # noqa: BLE001 - any bad file: re-export, never crash
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
    if loaded is None:
        _LOADED[path] = _export_and_save(fn, treedef, specs, path)
        return _LOADED[path]

    def first_call(*call_args):
        # The loaded program is trusted only once it has run: one that deserializes but
        # fails to compile or run (say, saved by an incompatible jaxlib) is re-exported.
        try:
            out = loaded(*call_args)
        except Exception:  # noqa: BLE001 - any bad program: re-export, never crash
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            _LOADED[path] = _export_and_save(fn, treedef, specs, path)
            return _LOADED[path](*call_args)
        _LOADED[path] = loaded
        return out

    _LOADED[path] = first_call
    return first_call
