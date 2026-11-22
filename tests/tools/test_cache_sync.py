"""tools/ops/day/pod/cache_sync.sh (the network volume: compile cache sync and the capture store)
and the volume parts of snap.sh, with a temp folder as the volume (CACHE_VOL_MOUNTCHECK=0)."""

import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import zlib
from pathlib import Path

import pytest

from tests.tools.test_pod_gate import BASH, _clean_env

ROOT = Path(__file__).resolve().parents[2]
SYNC = ROOT / "tools" / "ops" / "day" / "pod" / "cache_sync.sh"
SNAP = ROOT / "tools" / "ops" / "day" / "snap.sh"
PY = Path(sys.executable).as_posix()
VER = "0123456789abcdef"
CPUINFO = ("processor\t: 0\nmodel name\t: Test CPU 9000\nflags\t\t: sse2 avx2  fpu\n\n"
           "processor\t: 1\nmodel name\t: Test CPU 9000\nflags\t\t: sse2 avx2 fpu\n")


def _tag(model: str, flags: str) -> str:
    """cache_sync.sh host_tag: 12 hex of sha256 over the model name and the sorted flags."""
    import hashlib
    return hashlib.sha256(f"{model}\n{' '.join(sorted(set(flags.split())))} \n".encode()).hexdigest()[:12]


HOST = _tag("Test CPU 9000", "sse2 avx2 fpu")
VT = f"{VER}-{HOST}"   # the volume folder of VER compiled on this (fake) host
KEY = "a" * 40 + "-" + "b" * 16

pytestmark = pytest.mark.skipif(BASH is None, reason="no POSIX bash")


def _has(cmd: str) -> bool:
    return BASH is not None and subprocess.run([BASH, "-c", f"command -v {cmd}"], capture_output=True).returncode == 0


def _env(tmp: Path, **kw: str) -> dict[str, str]:
    (tmp / "vol").mkdir(exist_ok=True)   # the mount point; a pod with the volume always has it
    if not (tmp / "cpuinfo").exists():
        (tmp / "cpuinfo").write_text(CPUINFO, encoding="utf-8", newline="\n")
    env = {k: v for k, v in _clean_env().items()
           if not k.startswith("CACHE_") and k not in ("NUZLINE_TEST_COMPILE_CACHE", "RUNPOD_POD_ID")}
    env.update(CACHE_VOL=(tmp / "vol").as_posix(), CACHE_VOL_MOUNTCHECK="0", CACHE_PY=PY, CACHE_POD="poda",
               NUZLINE_TEST_COMPILE_CACHE=(tmp / "local").as_posix(), CACHE_CPUINFO=(tmp / "cpuinfo").as_posix(),
               CACHE_SEED_LOCK=(tmp / "seed.lock").as_posix())
    env.update(kw)
    return env


def _sync(tmp: Path, *args: str, **kw: str) -> tuple[int, str]:
    r = subprocess.run([BASH, SYNC.as_posix(), *args], env=_env(tmp, **kw), capture_output=True, text=True, check=False)
    return r.returncode, r.stdout + r.stderr


def _popen(tmp: Path, *args: str, **kw: str) -> subprocess.Popen:
    return subprocess.Popen([BASH, SYNC.as_posix(), *args], env=_env(tmp, **kw), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def _blob(seed: int, n: int = 20000) -> bytes:
    """A zlib stream about n bytes long (random data does not compress), as JAX writes entries."""
    return zlib.compress(random.Random(seed).randbytes(n))


def _entry(d: Path, name: str, data: bytes, age: float = 3600) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(data)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def _today() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d")


def _days_ago(n: int) -> str:
    return (dt.datetime.now(dt.UTC) - dt.timedelta(days=n)).strftime("%Y%m%d")


def _temps(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if ".tmp" in p.name or ".snapfetch" in p.name]


def test_without_a_mounted_volume_every_command_is_a_noop(tmp_path):
    (tmp_path / "vol").mkdir()
    _entry(tmp_path / "local" / VER, "e1-cache", _blob(1))
    for cmd, args, want_rc in (("seed", (), 0), ("publish", (), 0), ("prune", (), 0),
                               ("snap-publish", (str(tmp_path / "w"), KEY), 0), ("snap-fetch", (KEY, str(tmp_path / "d")), 1)):
        rc, out = _sync(tmp_path, cmd, *args, CACHE_VOL_MOUNTCHECK="1", CACHE_JAXVER=VER)
        assert rc == want_rc, out
        assert out.strip() == f"cache_sync: no volume mounted at {(tmp_path / 'vol').as_posix()}: {cmd} skipped"
    rc, out = _sync(tmp_path, "publish", CACHE_VOL=(tmp_path / "none").as_posix())   # no folder at all
    assert rc == 0 and "no volume mounted" in out
    assert list((tmp_path / "vol").iterdir()) == []
    assert not (tmp_path / "none").exists() and not (tmp_path / "d").exists()


def test_publish_copies_only_old_complete_entries_by_temp_name_then_rename(tmp_path):
    loc = tmp_path / "local" / VER
    good = _entry(loc, "good-cache", _blob(1))
    _entry(loc, "young-cache", _blob(2), age=0)                       # JAX may still write it
    _entry(loc, "trunc-cache", _blob(3)[:10000])                        # a unit killed mid-write
    _entry(loc, "empty-cache", b"")
    _entry(loc, ".x.tmp.1-cache", _blob(4))                           # someone's temp name
    _entry(loc, "other.bin", _blob(5))                                # not an entry
    _entry(tmp_path / "local" / "notaversion", "z-cache", _blob(6))   # not a version folder
    rc, out = _sync(tmp_path, "publish")
    assert rc == 0, out
    vol = tmp_path / "vol" / "jax" / VT
    assert sorted(p.name for p in vol.iterdir()) == ["good-cache", f"used-{_today()}"]
    assert (vol / "good-cache").read_bytes() == good.read_bytes()
    assert "skip trunc-cache: zlib: incomplete or truncated stream" in out
    assert "skip empty-cache: empty" in out
    assert f"publish {VT}: 1 entries published, 2 skipped" in out
    assert out.rstrip().endswith("cache_sync: publish done")
    assert not (tmp_path / "vol" / "jax" / "notaversion").exists()
    assert not (tmp_path / "vol" / "jax" / ".prune.lock").exists()
    assert _temps(tmp_path / "vol") == []
    rc, out = _sync(tmp_path, "publish")                              # nothing new: nothing copied
    assert rc == 0 and f"publish {VT}: 0 entries published, 2 skipped" in out
    rc, out = _sync(tmp_path, "publish", CACHE_MIN_AGE_S="0")         # age 0: the young one goes too
    assert (vol / "young-cache").is_file()


@pytest.mark.skipif(sys.version_info < (3, 14), reason="compression.zstd needs Python 3.14")
def test_publish_checks_zstd_entries_too(tmp_path):
    from compression import zstd
    blob = zstd.compress(b"program" * 5000)
    loc = tmp_path / "local" / VER
    _entry(loc, "z-cache", blob)
    _entry(loc, "zt-cache", blob[: len(blob) // 2])
    rc, out = _sync(tmp_path, "publish")
    assert rc == 0, out
    assert (tmp_path / "vol" / "jax" / VT / "z-cache").read_bytes() == blob
    assert not (tmp_path / "vol" / "jax" / VT / "zt-cache").exists()
    assert "skip zt-cache: zstd:" in out


@pytest.mark.skipif(not _has("flock"), reason="seed takes a flock; no flock here (Git Bash): runs on the pod")
def test_seed_copies_complete_volume_entries_once_and_keeps_local_ones(tmp_path):
    vol = tmp_path / "vol" / "jax" / VT
    _entry(vol, "a-cache", _blob(1))
    _entry(vol, "b-cache", _blob(2))
    _entry(vol, "bad-cache", _blob(3)[:10000])                          # a bad entry on the volume
    _entry(vol, ".tmp-podb-9-c-cache", _blob(4))                      # another pod's copy in progress
    _entry(vol, "b-cache", _blob(2))
    local = _entry(tmp_path / "local" / VER, "b-cache", _blob(9))     # JAX's own copy: kept
    rc, out = _sync(tmp_path, "seed", CACHE_JAXVER=VER)
    assert rc == 0, out
    loc = tmp_path / "local" / VER
    assert sorted(p.name for p in loc.iterdir()) == ["a-cache", "b-cache"]
    assert (loc / "a-cache").read_bytes() == (vol / "a-cache").read_bytes()
    assert local.read_bytes() == _blob(9)
    assert "seed: skip bad-cache: zlib: incomplete or truncated stream" in out
    assert f"seed done: {VER} 1 entries copied, 1 skipped" in out
    assert (tmp_path / "local" / f".seeded-{VER}").is_file() and (vol / f"used-{_today()}").is_file()
    _entry(vol, "d-cache", _blob(5))
    rc, out = _sync(tmp_path, "seed", CACHE_JAXVER=VER)               # once per pod and version
    assert rc == 0 and "already seeded" in out and not (loc / "d-cache").exists()
    assert _sync(tmp_path, "seed", CACHE_JAXVER="nothex")[0] == 2


@pytest.mark.skipif(not _has("flock"), reason="seed takes a flock; no flock here (Git Bash): runs on the pod")
def test_concurrent_seeds_seed_once(tmp_path):
    vol = tmp_path / "vol" / "jax" / VT
    for i in range(40):
        _entry(vol, f"e{i:02d}-cache", _blob(i))
    procs = [_popen(tmp_path, "seed", CACHE_JAXVER=VER) for _ in range(2)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert sum("seed done" in o for o in outs) == 1 and sum("already seeded" in o for o in outs) == 1
    assert len(list((tmp_path / "local" / VER).iterdir())) == 40
    assert _temps(tmp_path / "local") == []


def test_two_pods_publishing_the_same_entries_at_once_leave_one_complete_copy(tmp_path):
    for pod in ("a", "b"):
        for i in range(30):
            _entry(tmp_path / pod / VER, f"e{i:02d}-cache", _blob(i))
    procs = [_popen(tmp_path, "publish", CACHE_POD=f"pod{pod}", NUZLINE_TEST_COMPILE_CACHE=(tmp_path / pod).as_posix())
             for pod in ("a", "b")]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    vol = tmp_path / "vol" / "jax" / VT
    names = sorted(p.name for p in vol.iterdir() if p.name.endswith("-cache"))
    assert names == [f"e{i:02d}-cache" for i in range(30)]
    assert all((vol / n).read_bytes() == _blob(int(n[1:3])) for n in names)
    assert _temps(tmp_path / "vol") == []
    assert not (tmp_path / "vol" / "jax" / ".prune.lock").exists()


def _old(p: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(p, (t, t))


def test_prune_skips_while_another_pod_holds_the_lock(tmp_path):
    jax = tmp_path / "vol" / "jax"
    (jax / ".prune.lock").mkdir(parents=True)
    old = jax / f"1111111111111111-{HOST}"
    _entry(old, "x-cache", _blob(1))
    (old / f"used-{_days_ago(30)}").touch()
    (jax / VT).mkdir()
    (jax / VT / f"used-{_today()}").touch()
    rc, out = _sync(tmp_path, "prune")
    assert rc == 0 and "prune: lock" in out and "held" in out and "prune skipped" in out
    assert (old / "x-cache").is_file() and (jax / ".prune.lock").is_dir()


def test_prune_takes_over_a_stale_lock_and_prunes_versions_temps_size_and_captures(tmp_path):
    vol = tmp_path / "vol"
    jax = vol / "jax"
    (jax / ".prune.lock").mkdir(parents=True)
    _old(jax / ".prune.lock", 7200)                                   # a pod died mid-prune
    new, recent, stale = jax / VT, jax / f"2222222222222222-{HOST}", jax / f"3333333333333333-{HOST}"
    for d, used in ((new, [_days_ago(2), _today()]), (recent, [_days_ago(3)]), (stale, [_days_ago(10)])):
        d.mkdir()
        for u in used:
            (d / f"used-{u}").touch()
    for i in range(3):
        _entry(new, f"e{i}-cache", _blob(i), age=3600 * (10 - i))   # e0 is the oldest
    _entry(stale, "s-cache", _blob(7))
    size = (new / "e0-cache").stat().st_size
    tmp_old = _entry(new, ".tmp-podb-7-e9-cache", _blob(9), age=7200)
    tmp_new = _entry(new, ".tmp-podb-8-e8-cache", _blob(8), age=10)
    snaps = vol / "snap"
    for i in range(3):
        (snaps / f"{'c' * 40}-{i:016x}" / "snap").mkdir(parents=True)
        _old(snaps / f"{'c' * 40}-{i:016x}", 3600 * (3 - i))          # capture 0 is the oldest
    rc, out = _sync(tmp_path, "prune", CACHE_CAP_BYTES=str(2 * size + size // 2), CACHE_SNAP_KEEP="2")
    assert rc == 0, out
    assert "lock was" in out and "taken over" in out
    assert not stale.exists() and recent.is_dir() and new.is_dir()
    assert sorted(p.name for p in new.iterdir() if p.name.startswith("used-")) == [f"used-{_today()}"]
    assert not tmp_old.exists() and tmp_new.exists()
    assert sorted(p.name for p in new.glob("e*-cache")) == ["e1-cache", "e2-cache"]
    assert "prune: 1 oldest entries deleted" in out
    assert sorted(p.name[-1] for p in snaps.iterdir()) == ["1", "2"]
    assert not (jax / ".prune.lock").exists() and not list(jax.glob(".prune.lock.stale*"))


def test_prune_keeps_the_most_recently_used_version_however_old(tmp_path):
    only = tmp_path / "vol" / "jax" / VT
    _entry(only, "x-cache", _blob(1))
    (only / f"used-{_days_ago(40)}").touch()
    rc, out = _sync(tmp_path, "prune")
    assert rc == 0 and (only / "x-cache").is_file(), out


def _capture(w: Path, parts: dict[str, str], rc: str | None = "0", payload: bytes = b"states") -> Path:
    (w / "snap").mkdir(parents=True)
    (w / "snap" / "capture.json").write_text(json.dumps({"commit": "a" * 40, "parts": parts}), encoding="utf-8")
    for part in parts:   # snap_compare writes <part>.pkl.gz and <part>.sha256 for each part
        (w / "snap" / f"{part}.pkl.gz").write_bytes(payload)
        (w / "snap" / f"{part}.sha256").write_text(f"{'0' * 64}  {part}\n", encoding="utf-8")
    (w / "snap" / "step.0.pkl.gz").write_bytes(payload)
    lines = ["START 1", "SYNC_RC=0 2", "CAPTURE_START 3"] + ([f"CAPTURE_RC={rc} 4"] if rc is not None else [])
    (w / "times.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return w


def test_capture_publish_stores_complete_captures_once_and_fetch_brings_them_back(tmp_path):
    w = _capture(tmp_path / "work" / "snapmain-aaaaaaa", {"step.0": "ok", "ai.0": "ok"})
    stored = tmp_path / "vol" / "snap" / KEY
    rc, out = _sync(tmp_path, "snap-publish", w.as_posix(), KEY)
    assert rc == 0 and f"snap-publish: {KEY} stored" in out, out
    assert (stored / "snap" / "step.0.pkl.gz").read_bytes() == b"states"
    assert (stored / "KEY").read_text().strip() == KEY and "CAPTURE_RC=0" in (stored / "times.txt").read_text()
    other = _capture(tmp_path / "work" / "snapmain-other", {"step.0": "ok"}, payload=b"different")
    rc, out = _sync(tmp_path, "snap-publish", other.as_posix(), KEY)
    assert rc == 0 and "already on the volume: kept, not overwritten" in out
    assert (stored / "snap" / "step.0.pkl.gz").read_bytes() == b"states"
    k2 = "a" * 40 + "-" + "c" * 16
    for bad in (_capture(tmp_path / "work" / "part-failed", {"step.0": "ok", "ai.0": "failed"}),
                _capture(tmp_path / "work" / "no-rc", {"step.0": "ok"}, rc=None),
                _capture(tmp_path / "work" / "rc1", {"step.0": "ok"}, rc="1"),
                _capture(tmp_path / "work" / "no-parts", {})):
        rc, out = _sync(tmp_path, "snap-publish", bad.as_posix(), k2)
        assert rc == 1 and "is not a complete capture" in out, (bad, out)
    assert not (tmp_path / "vol" / "snap" / k2).exists()
    assert _sync(tmp_path, "snap-publish", w.as_posix(), "short-key")[0] == 2
    # fetch into a missing folder (a compare's capture lost with its pod): a finished capture again
    dest = tmp_path / "work2" / "snapmain-aaaaaaa"
    rc, out = _sync(tmp_path, "snap-fetch", KEY, dest.as_posix())
    assert rc == 0 and f"{KEY} fetched into" in out, out
    assert (dest / "snap" / "step.0.pkl.gz").read_bytes() == b"states"
    times = (dest / "times.txt").read_text().splitlines()
    assert times[-1] == "ALL_DONE" and times[-2].startswith(f"FETCHED_FROM_VOLUME {KEY} ") and "CAPTURE_RC=0 4" in times
    assert (dest / "store_key").read_text().strip() == KEY
    rc, out = _sync(tmp_path, "snap-fetch", KEY, dest.as_posix())
    assert rc == 0 and "already on this pod: not fetched" in out
    # fetch into a capture job's own folder: its times.txt is the job's and stays
    job = tmp_path / "work3" / "snapmain-aaaaaaa"
    job.mkdir(parents=True)
    (job / "times.txt").write_text("START 9\nSYNC_RC=0 9\n", encoding="utf-8", newline="\n")
    assert _sync(tmp_path, "snap-fetch", KEY, job.as_posix(), "own")[0] == 0
    assert (job / "times.txt").read_text() == "START 9\nSYNC_RC=0 9\n" and (job / "snap" / "capture.json").is_file()
    unfinished = tmp_path / "work4" / "x"
    (unfinished / "snap").mkdir(parents=True)
    rc, out = _sync(tmp_path, "snap-fetch", KEY, unfinished.as_posix())
    assert rc == 1 and "without capture.json" in out
    rc, out = _sync(tmp_path, "snap-fetch", k2, (tmp_path / "work5" / "y").as_posix())
    assert rc == 1 and "is not on the volume" in out
    assert _temps(tmp_path) == []


def test_a_stored_capture_that_is_incomplete_is_not_fetched(tmp_path):
    stored = tmp_path / "vol" / "snap" / KEY
    _capture(stored, {"step.0": "pending"})
    (stored / "KEY").write_text(KEY + "\n")
    rc, out = _sync(tmp_path, "snap-fetch", KEY, (tmp_path / "d").as_posix())
    assert rc == 1 and "not a complete capture" in out
    assert not (tmp_path / "d" / "snap").exists() and _temps(tmp_path) == []


def test_two_pods_storing_one_capture_key_at_once_store_it_once(tmp_path):
    ws = [_capture(tmp_path / "work" / f"w{i}", {"step.0": "ok"}, payload=bytes([65 + i]) * 100000) for i in range(2)]
    procs = [_popen(tmp_path, "snap-publish", w.as_posix(), KEY, CACHE_POD=f"pod{i}") for i, w in enumerate(ws)]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    snaps = tmp_path / "vol" / "snap"
    assert [p.name for p in snaps.iterdir()] == [KEY]
    assert (snaps / KEY / "snap" / "step.0.pkl.gz").read_bytes() in (b"A" * 100000, b"B" * 100000)
    assert sum("stored" in o and "kept" not in o for o in outs) == 1


# Host tag (the cache holds compiled XLA CPU programs; JAX's key does not hold the CPU)

def test_seed_and_publish_use_only_the_folder_of_this_hosts_cpu(tmp_path):
    other = tmp_path / "vol" / "jax" / f"{VER}-{_tag('Other CPU', 'sse2')}"
    _entry(other, "o-cache", _blob(1))                                 # compiled on another CPU
    _entry(tmp_path / "local" / VER, "mine-cache", _blob(2))
    rc, out = _sync(tmp_path, "publish")
    assert rc == 0, out
    assert (tmp_path / "vol" / "jax" / VT / "mine-cache").is_file() and not (other / "mine-cache").exists()
    if _has("flock"):
        rc, out = _sync(tmp_path, "seed", CACHE_JAXVER=VER, NUZLINE_TEST_COMPILE_CACHE=(tmp_path / "fresh").as_posix())
        assert rc == 0 and f"seed: host tag {HOST}" in out, out
        assert sorted(p.name for p in (tmp_path / "fresh" / VER).iterdir()) == ["mine-cache"]   # not o-cache
    # flags in another order, doubled spaces, CRLF: the same CPU, the same tag
    (tmp_path / "cpuinfo").write_text("model name : Test CPU 9000\r\nflags : fpu  sse2 avx2\r\n", encoding="utf-8", newline="")
    _entry(tmp_path / "local" / VER, "mine2-cache", _blob(3))
    assert _sync(tmp_path, "publish")[0] == 0 and (tmp_path / "vol" / "jax" / VT / "mine2-cache").is_file()


def test_without_a_cpu_model_or_flags_seed_and_publish_skip(tmp_path):
    (tmp_path / "cpuinfo").write_text("processor : 0\nFeatures : neon\n", encoding="utf-8")
    _entry(tmp_path / "local" / VER, "e-cache", _blob(1))
    rc, out = _sync(tmp_path, "publish")
    assert rc == 0 and "no host tag, nothing published" in out and out.rstrip().endswith("publish done"), out
    assert not list((tmp_path / "vol" / "jax").glob(f"{VER}*"))
    rc, out = _sync(tmp_path, "seed", CACHE_JAXVER=VER)
    assert rc == 0 and "no host tag, seed skipped" in out, out


def test_prune_keeps_the_newest_version_per_host_tag_and_drops_untagged_folders(tmp_path):
    jax = tmp_path / "vol" / "jax"
    h2 = _tag("Other CPU", "sse2")
    keep_a, old_a = jax / VT, jax / f"{'1' * 16}-{HOST}"
    keep_b, untagged = jax / f"{'2' * 16}-{h2}", jax / ("3" * 16)
    for d, used in ((keep_a, _today()), (old_a, _days_ago(10)), (keep_b, _days_ago(30)), (untagged, _today())):
        _entry(d, "x-cache", _blob(1))
        (d / f"used-{used}").touch()
    rc, out = _sync(tmp_path, "prune")
    assert rc == 0, out
    assert keep_a.is_dir() and keep_b.is_dir() and not old_a.exists() and not untagged.exists()
    assert f"jax folder {'3' * 16} has no host tag" in out


# The prune lock: takeover re-check and release by owner

def _shim(tmp: Path, name: str, body: str) -> str:
    """A PATH folder holding the bash script <name>; returns the PATH to pass."""
    d = tmp / "shim"
    d.mkdir(exist_ok=True)
    (d / name).write_text("#!/bin/bash\n" + body, encoding="utf-8", newline="\n")
    os.chmod(d / name, 0o755)
    path = subprocess.run([BASH, "-c", 'cygpath -u "$0" 2>/dev/null || echo "$0"', str(d)],
                          capture_output=True, text=True, check=True).stdout.strip()
    rest = subprocess.run([BASH, "-c", 'echo "$PATH"'], env=_env(tmp), capture_output=True, text=True).stdout.strip()
    return f"{path}:{rest}"


def _real(cmd: str) -> str:
    return subprocess.run([BASH, "-c", f"command -v {cmd}"], capture_output=True, text=True).stdout.strip()


def test_prune_releases_the_lock_only_while_it_names_this_pod_and_pid(tmp_path):
    jax = tmp_path / "vol" / "jax"
    _entry(jax / VT, "x-cache", _blob(1))
    # The cap step's helper runs while prune holds the lock: a python that first rewrites the
    # owner file, as a pod taking the lock over mid-prune would.
    wrap = tmp_path / "py.sh"
    wrap.write_text(f'#!/bin/bash\n[ "$2" = cap ] && echo "podz 1 0" > "$LOCKDIR/owner"\nexec "{PY}" "$@"\n',
                    encoding="utf-8", newline="\n")
    os.chmod(wrap, 0o755)
    rc, out = _sync(tmp_path, "prune", CACHE_PY=wrap.as_posix(), LOCKDIR=(jax / ".prune.lock").as_posix())
    assert rc == 0 and "no longer names this pod and pid (owner: podz 1): left to its holder" in out, out
    assert (jax / ".prune.lock" / "owner").read_text().split()[:2] == ["podz", "1"]
    shutil.rmtree(jax / ".prune.lock")
    rc, out = _sync(tmp_path, "prune")                                # its own lock: released
    assert rc == 0 and not (jax / ".prune.lock").exists(), out


def test_a_stale_lock_that_changes_before_the_takeover_is_left_alone(tmp_path):
    lock = tmp_path / "vol" / "jax" / ".prune.lock"
    lock.mkdir(parents=True)
    (lock / "owner").write_text("podb 7 1\n")
    _old(lock, 7200)
    real = _real("stat")
    # stat of the lock answers its old mtime once, then a new one: another pod took it over
    # between the first read and the re-check
    body = ('case "$*" in *.prune.lock)\n'
            f'  if [ -f "$SHIMN" ]; then date +%s; else touch "$SHIMN"; exec "{real}" "$@"; fi; exit 0;; esac\n'
            f'exec "{real}" "$@"\n')
    path = _shim(tmp_path, "stat", body)
    rc, out = _sync(tmp_path, "prune", PATH=path, SHIMN=(tmp_path / "n").as_posix())
    assert rc == 0 and "changed while being taken over: prune skipped" in out, out
    assert (lock / "owner").read_text() == "podb 7 1\n"
    rc, out = _sync(tmp_path, "prune")                                # unchanged: taken over
    assert rc == 0 and "taken over" in out and "owner podb 7 1" in out and not lock.exists(), out


# Capture integrity

def test_a_capture_with_a_part_file_missing_or_empty_is_not_stored_or_fetched(tmp_path):
    for i, (gone, empty) in enumerate((("ai.0.pkl.gz", None), ("ai.0.sha256", None), (None, "ai.0.pkl.gz"))):
        w = _capture(tmp_path / "work" / f"w{i}", {"step.0": "ok", "ai.0": "ok"})
        if gone:
            (w / "snap" / gone).unlink()
        if empty:
            (w / "snap" / empty).write_bytes(b"")
        rc, out = _sync(tmp_path, "snap-publish", w.as_posix(), KEY)
        assert rc == 1 and "is not a complete capture (part ai.0: ai.0." in out, out
    assert not (tmp_path / "vol" / "snap" / KEY).exists()
    w = _capture(tmp_path / "work" / "ok", {"step.0": "ok", "ai.0": "ok"})
    assert _sync(tmp_path, "snap-publish", w.as_posix(), KEY)[0] == 0
    (tmp_path / "vol" / "snap" / KEY / "snap" / "ai.0.sha256").unlink()   # lost on the volume
    rc, out = _sync(tmp_path, "snap-fetch", KEY, (tmp_path / "d").as_posix())
    assert rc == 1 and "on the volume is not a complete capture (part ai.0: ai.0.sha256 missing)" in out, out
    assert not (tmp_path / "d" / "snap").exists() and _temps(tmp_path) == []


def test_fetch_refuses_a_stored_capture_whose_key_file_names_another_key(tmp_path):
    w = _capture(tmp_path / "work" / "w", {"step.0": "ok"})
    assert _sync(tmp_path, "snap-publish", w.as_posix(), KEY)[0] == 0
    k2 = "a" * 40 + "-" + "e" * 16
    os.rename(tmp_path / "vol" / "snap" / KEY, tmp_path / "vol" / "snap" / k2)   # a hand rename
    rc, out = _sync(tmp_path, "snap-fetch", k2, (tmp_path / "d").as_posix())
    assert rc == 1 and f"KEY names '{KEY}', not {k2}: not fetched" in out, out
    assert not (tmp_path / "d" / "snap").exists()


def test_prune_keeps_a_temp_capture_copy_still_being_written(tmp_path):
    snaps = tmp_path / "vol" / "snap"
    live, dead = snaps / f".tmp-podb-1-{KEY}", snaps / ".tmp-podb-2-x"
    for d in (live, dead):
        _entry(d / "snap", "old.pkl.gz", b"x", age=7200)
    (live / "snap" / "new.pkl.gz").write_bytes(b"y")                  # the copy goes on
    for d in (live / "snap", live, dead / "snap", dead):              # folder mtimes: all old
        _old(d, 7200)
    rc, out = _sync(tmp_path, "prune")
    assert rc == 0 and "prune: 1 stale temp names deleted" in out, out
    assert live.is_dir() and not dead.exists()


def test_a_compares_fetch_replaces_a_stale_failed_times_file(tmp_path):
    w = _capture(tmp_path / "work" / "w", {"step.0": "ok"})
    assert _sync(tmp_path, "snap-publish", w.as_posix(), KEY)[0] == 0
    dest = tmp_path / "work2" / "snapmain-aaaaaaa"
    dest.mkdir(parents=True)
    (dest / "times.txt").write_text("START 1\nSYNC_RC=0 2\nCAPTURE_START 3\nCAPTURE_RC=1 4\nALL_DONE\n", newline="\n")
    rc, out = _sync(tmp_path, "snap-fetch", KEY, dest.as_posix())
    assert rc == 0, out
    times = (dest / "times.txt").read_text().splitlines()
    assert "CAPTURE_RC=1 4" not in times and "CAPTURE_RC=0 4" in times and times[-1] == "ALL_DONE"
    assert times[-2].startswith(f"FETCHED_FROM_VOLUME {KEY} ")
    assert "CAPTURE_RC=1 4" in (dest / "times.txt.before-fetch").read_text()
    assert _sync(tmp_path, "snap-fetch", KEY, dest.as_posix(), "mine")[0] == 2   # only "own"


@pytest.mark.skipif(sys.platform == "win32",
                    reason="timeout signals a bash shim's process group: Linux only (WSL runner, the pod)")
def test_a_store_stopped_by_timeout_leaves_no_key_and_no_temp(tmp_path):
    w = _capture(tmp_path / "work" / "w", {"step.0": "ok"})
    path = _shim(tmp_path, "cp", f'case "$*" in *times.txt*) sleep 60;; esac\nexec "{_real("cp")}" "$@"\n')
    r = subprocess.run(["timeout", "-k", "5", "2", BASH, SYNC.as_posix(), "snap-publish", w.as_posix(), KEY],
                       env=_env(tmp_path, PATH=path), capture_output=True, text=True, check=False)
    assert r.returncode == 124, r.stdout + r.stderr
    assert "stopped by a signal: temp copies removed" in r.stdout
    assert not (tmp_path / "vol" / "snap" / KEY).exists() and _temps(tmp_path / "vol") == []


@pytest.mark.skipif(sys.platform == "win32",
                    reason="timeout signals a bash shim's process group: Linux only (WSL runner, the pod)")
@pytest.mark.skipif(not _has("flock"), reason="seed takes a flock; no flock here (Git Bash): runs on the pod")
def test_a_seed_stopped_partway_keeps_the_renamed_batches_and_the_next_seed_finishes(tmp_path):
    vol = tmp_path / "vol" / "jax" / VT
    names = [f"e{i:02d}-cache" for i in range(10)]
    for i, n in enumerate(names):
        _entry(vol, n, _blob(i))
    # the copy of e05 hangs (a volume that stops answering) until the timeout stops the seed
    path = _shim(tmp_path, "cp", f'case "$*" in *e05-cache*) [ -f "$SHIMOFF" ] || sleep 60;; esac\nexec "{_real("cp")}" "$@"\n')
    env = _env(tmp_path, PATH=path, SHIMOFF=(tmp_path / "off").as_posix(), CACHE_JAXVER=VER, CACHE_SEED_BATCH="2")
    r = subprocess.run(["timeout", "-k", "5", "3", BASH, SYNC.as_posix(), "seed"], env=env,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 124, r.stdout + r.stderr
    loc = tmp_path / "local" / VER
    assert sorted(p.name for p in loc.iterdir()) == names[:4]   # the two batches renamed before it stay
    assert not (tmp_path / "local" / f".seeded-{VER}").exists() and _temps(tmp_path / "local") == []
    (tmp_path / "off").touch()
    r = subprocess.run([BASH, SYNC.as_posix(), "seed"], env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0 and f"seed done: {VER} 6 entries copied, 0 skipped" in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in loc.iterdir()) == names
    assert all((loc / n).read_bytes() == (vol / n).read_bytes() for n in names)
    assert (tmp_path / "local" / f".seeded-{VER}").is_file()
    assert _sync(tmp_path, "seed", CACHE_JAXVER=VER, CACHE_SEED_BATCH="0")[0] == 2


@pytest.mark.skipif(sys.platform == "win32",
                    reason="timeout signals a bash shim's process group: Linux only (WSL runner, the pod)")
@pytest.mark.parametrize("earlier", [True, False])
def test_a_compares_fetch_stopped_before_the_rename_leaves_the_folder_as_before(tmp_path, earlier):
    """times.txt is written before snap/ is renamed into place; a stop in between puts the earlier
    one back, so no complete snap/ ever sits beside a times.txt without ALL_DONE."""
    w = _capture(tmp_path / "work" / "w", {"step.0": "ok"})
    assert _sync(tmp_path, "snap-publish", w.as_posix(), KEY)[0] == 0
    dest = tmp_path / "work2" / "snapmain-aaaaaaa"
    dest.mkdir(parents=True)
    old = "START 1\nCAPTURE_RC=1 4\nALL_DONE\n"
    if earlier:
        (dest / "times.txt").write_text(old, newline="\n")
    path = _shim(tmp_path, "mv", f'case "$*" in "-T "*.snapfetch.*/snap*) sleep 60;; esac\nexec "{_real("mv")}" "$@"\n')
    r = subprocess.run(["timeout", "-k", "5", "3", BASH, SYNC.as_posix(), "snap-fetch", KEY, dest.as_posix()],
                       env=_env(tmp_path, PATH=path), capture_output=True, text=True, check=False)
    assert r.returncode == 124 and "stopped by a signal" in r.stdout, r.stdout + r.stderr
    assert not (dest / "snap").exists() and _temps(tmp_path / "work2") == []
    if earlier:
        assert (dest / "times.txt").read_text() == old
    else:
        assert not (dest / "times.txt").exists()
    assert not (dest / "times.txt.before-fetch").exists()
    rc, out = _sync(tmp_path, "snap-fetch", KEY, dest.as_posix())     # the next use fetches it whole
    assert rc == 0 and (dest / "snap" / "capture.json").is_file(), out
    assert (dest / "times.txt").read_text().splitlines()[-1] == "ALL_DONE"
    assert (dest / "times.txt.before-fetch").exists() == earlier


def test_seed_and_publish_run_under_their_timeout_knobs():
    pod = ROOT / "tools" / "ops" / "day" / "pod"
    setup = (pod / "setup.sh").read_text(encoding="utf-8")
    assert 'SEED_T="${CACHE_SEED_TIMEOUT:-900}"' in setup
    assert 'timeout -k 30 "$SEED_T" bash "$D/bin/cache_sync.sh" seed' in setup
    assert "WARNING: cache seed timed out after $SEED_T s" in setup
    names = next(x for x in setup.splitlines() if x.startswith("RUN_ENV_NAMES=")).strip('"').split()
    assert "CACHE_SEED_TIMEOUT" in names and "CACHE_PUBLISH_TIMEOUT" in names   # kept in run.env
    runner = (pod / "runner.sh").read_text(encoding="utf-8")
    assert 'PUB_T="${CACHE_PUBLISH_TIMEOUT:-900}"' in runner


@pytest.mark.skipif(sys.platform == "win32",
                    reason="timeout signals a bash shim's process group: Linux only (WSL runner, the pod)")
@pytest.mark.parametrize("hang", [True, False])
def test_runners_publish_warns_in_its_log_when_its_timeout_fires(tmp_path, hang):
    """The publish lines at the end of runner.sh, run as written (in the foreground), with a
    cache_sync.sh that hangs or ends."""
    runner = (ROOT / "tools" / "ops" / "day" / "pod" / "runner.sh").read_text(encoding="utf-8")
    block = runner[runner.index('if mountpoint -q "${CACHE_VOL:-/vol}"'):]
    block = block[block.index("\n") + 1:block.index("\nfi")]
    d = tmp_path / "run"
    (d / "bin").mkdir(parents=True)
    (d / "bin" / "cache_sync.sh").write_text("sleep 60\n" if hang else "echo 'cache_sync: publish done'\n",
                                             encoding="utf-8", newline="\n")
    script = block.replace("setsid nohup ", "").rstrip().removesuffix("&").rstrip()
    r = subprocess.run([BASH, "-c", f"D='{d.as_posix()}'; CACHE_PUBLISH_TIMEOUT=2\n{script}\necho rc=$?"],
                       env=_clean_env(), capture_output=True, text=True, check=False, timeout=60)
    log = (d / "cache_sync.log").read_text()
    if hang:
        assert "cache_sync: WARNING: publish timed out after 2 s (rc 124)" in log and "rc=124" in r.stdout, (log, r.stdout)
    else:
        assert "WARNING" not in log and "publish done" in log and "rc=0" in r.stdout, (log, r.stdout)


# snap.sh: the store key and the run.sh lines (dry runs on a throwaway repo)

def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fake_repo(tmp_path):
    if shutil.which("git") is None or not _has("xz"):
        pytest.skip("needs git and xz")
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "snap_compare.py").write_text('p.add_argument("--workers")\n', encoding="utf-8", newline="\n")
    (repo / ".gitignore").write_text("external/\n", encoding="utf-8", newline="\n")
    _git("init", "-q", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-qm", "x", cwd=repo)
    dec = repo / "external" / "pokeplatinum"
    dec.mkdir(parents=True)
    (dec / "f").write_text("d", encoding="utf-8")
    _git("init", "-q", cwd=dec)
    _git("add", "-A", cwd=dec)
    _git("-c", "user.name=t", "-c", "user.email=t@localhost", "commit", "-qm", "d", cwd=dec)
    sha = _git("rev-parse", "HEAD", cwd=repo)
    rent = tmp_path / "scratch" / "rent"
    rent.mkdir(parents=True)
    src = tmp_path / "tsrc" / "data" / "gen4_platinum"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text("{}", encoding="utf-8")
    tb = rent / f"tables_{sha[:8]}.tar.xz"
    with tarfile.open(tb, "w:xz") as t:
        t.add(src / "manifest.json", arcname="data/gen4_platinum/manifest.json")
    return repo, sha, rent, tb


def _snap(tmp_path: Path, repo: Path, *args: str, **kw: str) -> tuple[int, str]:
    env = {k: v for k, v in _clean_env().items() if not k.startswith("CACHE_")}
    env.update(NUZLINE_REPO=repo.as_posix(), NUZLINE_SCRATCH=(tmp_path / "scratch").as_posix(), **kw)
    r = subprocess.run([BASH, SNAP.as_posix(), *args, "--dry-run"], env=env, capture_output=True, text=True, check=False)
    return r.returncode, r.stdout + r.stderr


def _committed_cache_sync() -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "HEAD:tools/ops/day/pod/cache_sync.sh"],
                          capture_output=True).returncode == 0


def test_snap_capture_keys_by_sha_and_table_key_and_writes_fetch_and_store(tmp_path, fake_repo):
    if not _committed_cache_sync():
        pytest.skip("cache_sync.sh is not committed at HEAD of this checkout")
    repo, sha, rent, tb = fake_repo
    k = "d" * 64
    tb.with_suffix("").with_suffix(".key").write_text(k + "\n", encoding="utf-8")
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix())
    assert rc == 0, out
    skey = f"{sha}-{'d' * 16}"
    assert f"capture store key {skey}" in out
    o = rent / "out" / f"snapmain-{sha[:7]}"
    run = (o / "run.sh").read_text()
    assert f'snap-fetch {skey} "$W"' in run and f'snap-publish "$W" {skey}' in run
    assert run.count("mountpoint -q /vol") == 2   # both only on a pod with the volume
    assert "CAPTURE_FROM_VOLUME" in run and "STORE_RC=" in run
    # Order: the fetch under the pod lock and before the memory gate; the store after the job and
    # snapstop, before ALL_DONE (the lock is released on exit).
    lines = run.splitlines()
    at = lambda pre: next(i for i, x in enumerate(lines) if x.startswith(pre))   # noqa: E731
    assert (at('snaplock "$L"') < at("if mountpoint -q /vol") < at('snapgate 88 "$W') < at("uv run --frozen")
            < at('snapstop "$W/times.txt"') < at('if [ "$rc" = 0 ] && mountpoint') < at("echo ALL_DONE"))
    with tarfile.open(o / "payload.tar") as t:
        assert "cache_sync.sh" in t.getnames()
    assert (o / "cache_sync.sh").read_bytes() == SYNC.read_bytes().replace(b"\r", b"")
    shutil.rmtree(o)
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix(), f"snap2-{sha[:7]}", "--tool", sha)
    assert rc == 0 and "is not stored on the network volume" in out
    assert "snap-publish" not in (rent / "out" / f"snap2-{sha[:7]}" / "run.sh").read_text()


def test_snap_capture_without_a_table_key_stores_nothing(tmp_path, fake_repo):
    if not _committed_cache_sync():
        pytest.skip("cache_sync.sh is not committed at HEAD of this checkout")
    repo, sha, rent, tb = fake_repo
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix())
    assert rc == 0 and "NOTE: no table key" in out, out
    run = (rent / "out" / f"snapmain-{sha[:7]}" / "run.sh").read_text()
    assert "snap-fetch" not in run and "snap-publish" not in run and "mountpoint" not in run


def test_snap_compare_fetches_a_missing_main_capture_by_its_index_key(tmp_path, fake_repo):
    if not _committed_cache_sync():
        pytest.skip("cache_sync.sh is not committed at HEAD of this checkout")
    repo, sha, rent, tb = fake_repo
    (rent / "tables").mkdir()
    (rent / "tables" / "index.txt").write_text(f"{sha} {'1' * 64}\n{sha} {'e' * 64}\n", encoding="utf-8")
    cap = f"/work/snapmain-{sha[:7]}/snap"
    rc, out = _snap(tmp_path, repo, "compare", sha, tb.as_posix(), cap, "snapcmp-x-1")
    assert rc == 0, out
    run = (rent / "out" / "snapcmp-x-1" / "run.sh").read_text()
    assert f'snap-fetch {sha}-{"e" * 16} "/work/snapmain-{sha[:7]}"' in run
    assert f'[ ! -f "{cap}/capture.json" ]' in run and "FETCH_RC=" in run
    rc, out = _snap(tmp_path, repo, "compare", sha, tb.as_posix(), "/work/snap2-abcdef0/snap", "snapcmp-x-2")
    assert rc == 0 and "NOTE: no store key" in out
    assert "snap-fetch" not in (rent / "out" / "snapcmp-x-2" / "run.sh").read_text()


def test_snap_capture_key_never_comes_from_index_txt(tmp_path, fake_repo):
    """A capture made with a rebuilt or hand-made archive must not be stored under the official
    key of its sha (every later compare would fetch it): the key comes from the archive only."""
    if not _committed_cache_sync():
        pytest.skip("cache_sync.sh is not committed at HEAD of this checkout")
    repo, sha, rent, tb = fake_repo
    (rent / "tables").mkdir()
    (rent / "tables" / "index.txt").write_text(f"{sha} {'e' * 64}\n", encoding="utf-8")
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix())
    assert rc == 0 and "NOTE: no table key" in out, out
    run = (rent / "out" / f"snapmain-{sha[:7]}" / "run.sh").read_text()
    assert "snap-fetch" not in run and "snap-publish" not in run
    # a .key beside an archive that is not the build of that key: no key either
    official = rent / "tables" / f"tables_k{'e' * 16}.tar.xz"
    official.write_bytes(b"the official build")
    tb.with_suffix("").with_suffix(".key").write_text("e" * 64 + "\n", encoding="utf-8")
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix(), "snapmain-x2")
    assert rc == 0 and "differs from " in out and official.name in out and "NOTE: no table key" in out, out
    assert "snap-publish" not in (rent / "out" / "snapmain-x2" / "run.sh").read_text()
    # the archive is that build: its key
    shutil.copyfile(tb, official)
    rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix(), "snapmain-x3")
    assert rc == 0 and f"capture store key {sha}-{'e' * 16}" in out, out
    # the build itself, by its tables_k<key16> name, no .key file
    rc, out = _snap(tmp_path, repo, "capture", sha, official.as_posix(), "snapmain-x4")
    assert rc == 0 and f"capture store key {sha}-{'e' * 16}" in out, out


def test_snap_run_sh_puts_every_volume_call_under_a_timeout(tmp_path, fake_repo):
    if not _committed_cache_sync():
        pytest.skip("cache_sync.sh is not committed at HEAD of this checkout")
    repo, sha, rent, tb = fake_repo
    tb.with_suffix("").with_suffix(".key").write_text("d" * 64 + "\n", encoding="utf-8")
    assert _snap(tmp_path, repo, "capture", sha, tb.as_posix())[0] == 0
    run = (rent / "out" / f"snapmain-{sha[:7]}" / "run.sh").read_text()
    skey = f"{sha}-{'d' * 16}"
    assert f'timeout -k 30 900 bash "$W/cache_sync.sh" snap-fetch {skey} "$W" own' in run
    assert f'timeout -k 30 900 bash "$W/cache_sync.sh" snap-publish "$W" {skey}' in run
    assert run.count("VOLUME_TIMEOUT") == 2 and 'rm -rf "$W/snap" "$W"/.snapfetch.*' in run
    cap = f"/work/snapmain-{sha[:7]}/snap"
    assert _snap(tmp_path, repo, "compare", sha, tb.as_posix(), cap, "snapcmp-t-1")[0] == 0
    run = (rent / "out" / "snapcmp-t-1" / "run.sh").read_text()
    assert f'timeout -k 30 900 bash "$W/cache_sync.sh" snap-fetch {skey} "/work/snapmain-{sha[:7]}"' in run
    assert f'NO_CAPTURE {cap} ' in run and "VOLUME_TIMEOUT fetch" in run
    assert run.count("cache_sync.sh") == run.count("timeout -k 30 900 bash")
    # the knobs: a fetch limit and a store limit of their own; a bad value is refused
    assert _snap(tmp_path, repo, "capture", sha, tb.as_posix(), "snapmain-t2",
                 CACHE_FETCH_TIMEOUT="50", CACHE_STORE_TIMEOUT="70")[0] == 0
    run = (rent / "out" / "snapmain-t2" / "run.sh").read_text()
    assert f'timeout -k 30 50 bash "$W/cache_sync.sh" snap-fetch {skey} "$W" own' in run
    assert f'timeout -k 30 70 bash "$W/cache_sync.sh" snap-publish "$W" {skey}' in run
    assert "timed out after 50 s" in run and "timed out after 70 s" in run
    for i, bad in enumerate(("CACHE_FETCH_TIMEOUT", "CACHE_STORE_TIMEOUT")):
        rc, out = _snap(tmp_path, repo, "capture", sha, tb.as_posix(), f"snapmain-t3{i}", **{bad: "15m"})
        assert rc == 3 and f"REFUSED: {bad} '15m' is not a whole number" in out, out
