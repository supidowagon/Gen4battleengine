"""Content key of a table build (tools/ops/build_tables.sh <commit>).

Usage: python tools/ops/table_key.py --root <checkout> [--decomp <dir>] [--symbols <file>]
                                     [--config <local.toml>] [--list]
Prints one line: the 64-hex key. --list prints every input and its digest instead.

The key is a sha256 over exactly what the build reads:
- the Python modules the build runs (ENTRY) and every repo module they import, found by
  walking their import statements (module level and function level) from the files of
  <root>/tools and <root>/src; package __init__.py files of every imported module count.
  MANIFEST_VERSION (src/nuzline/data/tables.py) is in that set. No traced module imports by
  name at run time (importlib, __import__): the walk refuses a file that does (exit 2).
- tracked data: every tracked file under data/, every tracked non-.py file under tools/
  (tools/ops excluded) and src/ (.md excluded), .python-version, uv.lock;
- the interpreter: sys.version and every installed distribution with its version (run this
  script with the build's python);
- `node --version` (tools.extract_pkcalc runs node);
- the decomp: HEAD of <decomp> and its `git diff HEAD`;
- the symbols map and the local inputs vanilla_rom, kaizo_patch, pkcalc_zip, kaizo_docs
  (environment NUZLINE_* first, then <config>, as tools/rom/inputs.py resolves them), by content;
  the text of <config> itself.
Paths of inputs never enter the key; only contents do.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ENTRY = (
    "tools.extract_decomp",
    "tools.build_tables",
    "tools.extract_rom",
    "tools.extract_pkcalc",
    "tools.extract_kaizo_docs",
    "tools.kaizo.overrides",
    "tools.rom.symbols",
)
LOCAL_INPUTS = {  # key of data/local.toml -> environment override (tools/rom/inputs.py _ENV)
    "vanilla_rom": "NUZLINE_VANILLA_ROM",
    "kaizo_patch": "NUZLINE_KAIZO_PATCH",
    "pkcalc_zip": "NUZLINE_PKCALC_ZIP",
    "kaizo_docs": "NUZLINE_KAIZO_DOCS",
}
DYNAMIC = ("importlib", "__import__")


class KeyError_(Exception):
    pass


def _module_file(name: str, bases: list[Path]) -> Path | None:
    parts = name.split(".")
    for b in bases:
        f = b.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            return f
        f = b.joinpath(*parts, "__init__.py")
        if f.is_file():
            return f
    return None


def _imports(path: Path, name: str) -> set[str]:
    """Module names a file imports (absolute), plus `from X import y` candidates X.y."""
    tree = ast.parse(path.read_bytes(), str(path))
    is_pkg = path.name == "__init__.py"
    pkg = name if is_pkg else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in DYNAMIC:
                    raise KeyError_(f"{path}: imports {a.name}; the import walk cannot trace it")
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod.split(".")[0] in DYNAMIC:
                raise KeyError_(f"{path}: imports from {mod}; the import walk cannot trace it")
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names if a.name != "*")
        elif isinstance(node, ast.Name) and node.id == "__import__":
            raise KeyError_(f"{path}: calls __import__; the import walk cannot trace it")
    return out


def trace(root: Path, entry: tuple[str, ...] = ENTRY) -> dict[str, Path]:
    """Repo modules reached from `entry`: module name -> file."""
    bases = [root, root / "src"]
    seen: dict[str, Path] = {}
    todo = list(entry)
    while todo:
        name = todo.pop()
        if not name or name in seen:
            continue
        f = _module_file(name, bases)
        if f is None:
            if name in entry:
                raise KeyError_(f"entry module {name} not found under {root}")
            continue  # third-party, stdlib, or a name imported from a module (not a module)
        seen[name] = f
        parts = name.split(".")
        todo.extend(".".join(parts[:i]) for i in range(1, len(parts)))  # parent packages
        todo.extend(_imports(f, name))
    return seen


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def tracked_inputs(root: Path) -> list[str]:
    files = _git(root, "ls-files", "-z", "--", "data", "tools", "src", ".python-version", "uv.lock")
    out = []
    for p in sorted(x for x in files.decode().split("\0") if x):
        if p.startswith("data/") or p in (".python-version", "uv.lock"):
            out.append(p)
        elif p.startswith("tools/ops/") or p.endswith(".py"):
            continue
        elif p.startswith("tools/") or (p.startswith("src/") and not p.endswith(".md")):
            out.append(p)
    return out


def local_inputs(root: Path, config: Path | None) -> dict[str, Path]:
    toml: dict[str, object] = {}
    if config is not None and config.is_file():
        toml = tomllib.loads(config.read_text(encoding="utf-8"))
    found = {}
    for key, env in LOCAL_INPUTS.items():
        if os.environ.get(env):
            found[key] = Path(os.environ[env])
        elif key in toml:
            found[key] = root / str(toml[key])
    return found


def entries(root: Path, decomp: Path | None, symbols: Path | None, config: Path | None) -> list[tuple[str, str]]:
    """(label, digest) pairs, in a fixed order; the key hashes them."""
    e: list[tuple[str, str]] = []
    for name, f in sorted(trace(root).items()):
        e.append((f"module {name}", _sha(f.read_bytes().replace(b"\r\n", b"\n"))))
    for p in tracked_inputs(root):
        e.append((f"tracked {p}", _sha((root / p).read_bytes().replace(b"\r\n", b"\n"))))
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    e.append(("python", _sha(sys.version.encode())))
    e.append(("distributions", _sha("\n".join(dists).encode())))
    try:
        node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        node = "none"
    e.append(("node", _sha(node.encode())))
    if decomp is not None:
        e.append(("decomp HEAD", _git(decomp, "rev-parse", "HEAD").decode().strip()))
        e.append(("decomp diff", _sha(_git(decomp, "diff", "HEAD"))))
    if symbols is not None:
        e.append(("symbols", _file_sha(symbols)))
    if config is not None and config.is_file():
        e.append(("local.toml", _sha(config.read_bytes())))
    for key, p in sorted(local_inputs(root, config).items()):
        e.append((f"input {key}", _file_sha(p) if p.is_file() else "missing"))
    return e


def key_of(e: list[tuple[str, str]]) -> str:
    return _sha("".join(f"{a}\0{b}\n" for a, b in e).encode())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--decomp", type=Path)
    p.add_argument("--symbols", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    try:
        e = entries(a.root.resolve(), a.decomp, a.symbols, a.config)
    except (KeyError_, subprocess.CalledProcessError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if a.list:
        for label, d in e:
            print(d, label)
    print(key_of(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
