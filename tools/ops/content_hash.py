"""Content hash of a built tables archive (tools/ops/build_tables.sh, tools/ops/day/pack.sh).

Usage: python tools/ops/content_hash.py <tables.tar.xz>
       python tools/ops/content_hash.py --root <checkout> --list <file of paths relative to root>
Prints one line: the 64-hex content hash.

The hash is a sha256 over the sorted per-file lines "<sha256 of the file's content>  <path>"
(one per regular file, path with '/'), where a file's content is read as tools/snap_compare.py
file_content reads it (the same notion of "equal tables" as its content_hash): a .npz is its
arrays (sorted names, dtype, shape, bytes; not the zip's timestamps), a .json its parsed value,
any other file its bytes with CRLF read as LF. So two builds with the same table files hash
equal although their table keys (tools/ops/table_key.py, a hash of the build's inputs) differ.
The key stays the cache identity (capture folders, the lane marker's TABLES_KEY); the content
hash only decides whether a quick-lane delta must ship tables.
Plain Python plus numpy (for .npz): no nuzline or jax import.
"""

import hashlib
import importlib.util
import sys
import tarfile
import tempfile
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "snap_compare.py"


def _file_content():
    spec = importlib.util.spec_from_file_location("_snap_compare_for_hash", SNAP)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses in the module look themselves up here
    spec.loader.exec_module(mod)
    return mod.file_content


def combine(lines: list[str]) -> str:
    """sha256 over the sorted lines, one per line."""
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def line(path: Path, name: str, file_content) -> str:
    return f"{hashlib.sha256(file_content(path)).hexdigest()}  {name}"


def hash_files(root: Path, names: list[str]) -> str:
    fc = _file_content()
    return combine([line(root / n, n, fc) for n in names if (root / n).is_file()])


def hash_archive(archive: Path) -> str:
    """Streams the archive: each regular member goes to a temp file (suffix kept) for reading."""
    fc = _file_content()
    lines = []
    with tarfile.open(archive, "r:*") as t, tempfile.TemporaryDirectory() as tmp:
        for m in t:
            if not m.isfile():
                continue
            dst = Path(tmp) / ("member" + Path(m.name).suffix)
            src = t.extractfile(m)
            with src, dst.open("wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            lines.append(line(dst, m.name, fc))
            dst.unlink()
    return combine(lines)


def main(argv: list[str]) -> int:
    if len(argv) == 1 and not argv[0].startswith("-"):
        print(hash_archive(Path(argv[0])))
        return 0
    if len(argv) == 4 and argv[0] == "--root" and argv[2] == "--list":
        names = [x.strip() for x in Path(argv[3]).read_text(encoding="utf-8").splitlines()]
        print(hash_files(Path(argv[1]), [n for n in names if n]))
        return 0
    print(__doc__.split("\n\n")[0].strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
