"""tools/ops/content_hash.py and the meta file build_tables.sh writes with it. No engine import:
plain Python plus numpy, and bash for the meta() step."""

import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tests.tools.test_pod_gate import BASH, _clean_env, _posix
from tools.ops import content_hash

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "tools" / "ops"


def _npz(arrays: dict[str, np.ndarray], stamp: tuple = (1980, 1, 1, 0, 0, 0)) -> bytes:
    """An .npz with its zip entries stamped <stamp>: np.savez_compressed stamps the build time, so
    two builds of the same arrays differ in their bytes."""
    import zipfile

    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.save(buf, a, allow_pickle=False)
            z.writestr(zipfile.ZipInfo(name + ".npy", date_time=stamp), buf.getvalue())
    return raw.getvalue()


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:xz") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 12345 + len(name)
            t.addfile(ti, io.BytesIO(data))
    return path


def _tables(**over: bytes) -> dict[str, bytes]:
    base = {
        "data/gen4_platinum/tables.npz": _npz({"a": np.arange(6, dtype=np.int16), "b": np.ones((2, 2), np.int8)}),
        "data/gen4_platinum/manifest.json": json.dumps({"version": 3, "arrays": ["a", "b"]}).encode(),
        "data/intermediate/species.json": b'{"x": [1, 2]}',
        "data/other.bin": b"line\r\nline\r\n",
    }
    return {**base, **over}


def test_equal_tables_hash_equal_whatever_the_zip_stamps_json_layout_and_line_endings(tmp_path):
    a = content_hash.hash_archive(_archive(tmp_path / "a.tar.xz", _tables()))
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    other = _tables(
        **{
            "data/gen4_platinum/tables.npz": _npz(
                {"a": np.arange(6, dtype=np.int16), "b": np.ones((2, 2), np.int8)}, stamp=(2026, 11, 22, 9, 30, 0)
            ),
            "data/gen4_platinum/manifest.json": b'{\n  "arrays": ["a", "b"],\n  "version": 3\n}\n'.replace(b'"arrays": ["a", "b"],\n  "version": 3', b'"version": 3,\n  "arrays": ["a", "b"]'),
            "data/other.bin": b"line\nline\n",
        }
    )
    assert content_hash.hash_archive(_archive(tmp_path / "b.tar.xz", other)) == a


@pytest.mark.parametrize(
    "change",
    [
        {"data/gen4_platinum/tables.npz": _npz({"a": np.arange(6, dtype=np.int16) + 1, "b": np.ones((2, 2), np.int8)})},
        {"data/gen4_platinum/tables.npz": _npz({"a": np.arange(6, dtype=np.int32), "b": np.ones((2, 2), np.int8)})},
        {"data/gen4_platinum/tables.npz": _npz({"a": np.arange(6, dtype=np.int16), "c": np.ones((2, 2), np.int8)})},
        {"data/gen4_platinum/manifest.json": b'{"version": 4, "arrays": ["a", "b"]}'},
        {"data/intermediate/species.json": b'{"x": [2, 1]}'},
        {"data/other.bin": b"line\nline\nmore\n"},
    ],
)
def test_a_changed_table_file_changes_the_hash(tmp_path, change):
    a = content_hash.hash_archive(_archive(tmp_path / "a.tar.xz", _tables()))
    assert content_hash.hash_archive(_archive(tmp_path / "b.tar.xz", _tables(**change))) != a


def test_a_moved_added_or_removed_file_changes_the_hash(tmp_path):
    base = _tables()
    a = content_hash.hash_archive(_archive(tmp_path / "a.tar.xz", base))
    moved = {("data/intermediate/forms.json" if k == "data/intermediate/species.json" else k): v for k, v in base.items()}
    assert content_hash.hash_archive(_archive(tmp_path / "m.tar.xz", moved)) != a
    assert content_hash.hash_archive(_archive(tmp_path / "p.tar.xz", {**base, "data/new.json": b"{}"})) != a
    assert content_hash.hash_archive(_archive(tmp_path / "r.tar.xz", {k: v for k, v in base.items() if k != "data/other.bin"})) != a
    # the order of the members does not matter
    rev = dict(reversed(list(base.items())))
    assert content_hash.hash_archive(_archive(tmp_path / "o.tar.xz", rev)) == a


def test_a_checkout_and_its_archive_hash_alike_and_the_cli_prints_one_line(tmp_path):
    files = _tables()
    arch = _archive(tmp_path / "a.tar.xz", files)
    root = tmp_path / "co"
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join([*files, "data/not-built-yet.json"]) + "\n", encoding="utf-8")
    want = content_hash.hash_archive(arch)
    assert content_hash.hash_files(root, list(files)) == want
    r = subprocess.run([sys.executable, str(OPS / "content_hash.py"), str(arch)], check=True, capture_output=True, text=True)
    assert r.stdout.strip() == want and len(r.stdout.splitlines()) == 1
    r = subprocess.run([sys.executable, str(OPS / "content_hash.py"), "--root", str(root), "--list", str(lst)],
                       check=True, capture_output=True, text=True)  # fmt: skip
    assert r.stdout.strip() == want  # a listed file that does not exist is skipped
    r = subprocess.run([sys.executable, str(OPS / "content_hash.py")], check=False, capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run([sys.executable, str(OPS / "content_hash.py"), "--version"], check=True, capture_output=True, text=True)
    assert r.stdout.strip() == str(content_hash.VERSION)


def test_the_hash_reads_files_as_snap_compare_does(tmp_path):
    """The definition is tools/snap_compare.py file_content (its own content_hash of the tables)."""
    spec = content_hash._file_content()
    p = tmp_path / "t.json"
    p.write_text('{"b": 1,\n "a": [1, 2]}', encoding="utf-8")
    assert spec(p) == json.dumps({"b": 1, "a": [1, 2]}, ensure_ascii=False, separators=(",", ":")).encode()
    assert "file_content" in (ROOT / "tools/snap_compare.py").read_text(encoding="utf-8")


def _meta_step(tmp_path: Path, archive: Path, key: str) -> subprocess.CompletedProcess:
    """build_tables.sh's meta() function, run on <archive> with the script's own variables."""
    text = (OPS / "build_tables.sh").read_text(encoding="utf-8")
    m = re.search(r"^  meta\(\) \{.*?^  record\(\)", text, re.S | re.M)
    assert m, "meta() is not in build_tables.sh"
    fn = m.group(0).rsplit("  record()", 1)[0]
    rent = tmp_path / "rent"
    (rent / "tables").mkdir(parents=True, exist_ok=True)
    script = (
        f"set -euo pipefail; RENT='{_posix(rent)}'; C=abc1234; KEY={key}; "
        f"OPS_DIR='{_posix(OPS)}'; PY=('{Path(sys.executable).as_posix()}'); KA='{_posix(archive)}'\n"
        "wp() { if command -v cygpath >/dev/null; then cygpath -w \"$1\"; else echo \"$1\"; fi; }\n"
        f"{fn}\nmeta\nmeta\ncat \"${{KA%.tar.xz}}.meta\"; cat \"$RENT/tables_$C.meta\""
    )
    return subprocess.run([BASH, "-c", script], check=False, capture_output=True, text=True, env=_clean_env())


@pytest.mark.skipif(BASH is None, reason="no POSIX bash")
def test_build_tables_meta_writes_key_and_content_hash_once_and_read_only(tmp_path):
    arch = _archive(tmp_path / "tables_kaaaaaaaaaaaaaaaa.tar.xz", _tables())
    key = "a" * 64
    r = _meta_step(tmp_path, arch, key)
    assert r.returncode == 0, r.stdout + r.stderr
    want = content_hash.hash_archive(arch)
    one = f"KEY={key}\nCONTENT_HASH_V={content_hash.VERSION}\nCONTENT_HASH={want}\n"
    assert r.stdout == one + one  # the keyed meta file and tables_<short>.meta, the second meta call kept it
    # the meta file sits beside the archive, under its name (KA minus .tar.xz), where table_key16.sh
    # table_chash reads it for a keyed archive in $RENT/tables; it is read-only, its copy writable
    keyed = arch.with_suffix("").with_suffix(".meta")
    assert keyed.is_file() and not keyed.stat().st_mode & 0o222  # mode bits: os.access says True for root
    assert (tmp_path / "rent" / "tables_abc1234.meta").stat().st_mode & 0o200
    assert not list((tmp_path / "rent").glob("*tmp*"))
    # a meta file of another hash version (or none) is rewritten, read-only again; the copy follows
    for old in (f"KEY={key}\nCONTENT_HASH_V=0\nCONTENT_HASH={'0' * 64}\n", f"KEY={key}\nCONTENT_HASH={'0' * 64}\n"):
        os.chmod(keyed, 0o644)
        keyed.write_text(old, encoding="utf-8", newline="\n")
        os.chmod(keyed, 0o444)
        r = _meta_step(tmp_path, arch, key)
        assert r.returncode == 0 and r.stdout == one + one, r.stdout + r.stderr
        assert not keyed.stat().st_mode & 0o222


def test_build_tables_documents_the_meta_file():
    text = (OPS / "build_tables.sh").read_text(encoding="utf-8")
    assert "CONTENT_HASH" in text and "content_hash.py" in text and "tables_$C.meta" in text
    assert text.count("meta\n") >= 1 and "    meta\n    rm -f" in text  # record() writes it on a reuse too
