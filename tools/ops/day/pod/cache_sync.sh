#!/usr/bin/env bash
# POD SIDE. The shared network volume (DAYBOX.md "Network volume"; docs/process.md section 5):
# a sync of the pod's local XLA compile cache with the volume, and a store of complete snapshot
# captures of main. JAX keeps writing the local cache exactly as without a volume; no test
# process ever reads or writes the volume.
# Usage:
#   cache_sync.sh seed                  copy volume entries this pod lacks into the local cache
#   cache_sync.sh publish               copy local entries the volume lacks onto it, then prune
#   cache_sync.sh prune                 prune the volume (under its lock)
#   cache_sync.sh snap-publish <capture folder> <key>   store a complete capture under <key>
#   cache_sync.sh snap-fetch <key> <capture folder>     fetch a stored capture into <folder>/snap
# Every command is a no-op that prints "cache_sync: no volume mounted at <vol>: <cmd> skipped"
# (exit 0; snap-fetch exits 1, as nothing was fetched) when <vol> is not a mountpoint.
#
# Layout on the volume (<vol>, default /vol):
#   jax/<ver>/<name>-cache   compile cache entries; <ver> = the 16-hex jax_version() folder
#                            (src/nuzline/jaxcache.py), as in the local cache; JAX keys each entry
#                            by program, so merging pods' caches is a plain union
#   jax/<ver>/used-<YYYYMMDD> a pod seeded from or published to <ver> that UTC day
#   jax/.prune.lock/         the prune lock (mkdir; flock is not relied on over the network mount)
#   snap/<sha40>-<key16>/    a complete capture of main commit <sha40> with tables of table key
#                            <key16> (tools/ops/table_key.py, first 16 hex): snap/ (what compares
#                            read), times.txt of the capture, KEY
#   .tmp-* names             a copy in progress; every reader skips them
# Writes are temp name then rename: an entry is copied to <dir>/.tmp-<pod>-<pid>-<name>, its size
# checked, then renamed. Two pods publishing one entry write different temp names; the second
# rename replaces an equal file (same program key). A capture is renamed with mv -T, which
# fails on an existing capture: a stored key is never overwritten.
# Checks: publish copies only entries older than CACHE_MIN_AGE_S (600 s; JAX may still be writing
# a younger one) that decompress completely (zstd when they start with its magic, else zlib, as
# jax/_src/compilation_cache.py wrote them); seed checks its local copy the same way before the
# rename. A failed check skips the entry and prints "skip <name>: <reason>". snap-publish
# stores only a capture whose times.txt has CAPTURE_RC=0 and whose capture.json lists every part
# "ok"; snap-fetch checks the copy the same way and against the volume (diff -r).
# seed: once per pod and <ver> (marker <cache>/.seeded-<ver>), under the local flock
#   CACHE_SEED_LOCK (/work/.cache_seed.lock), so concurrent setups seed once.
# prune (also run by publish and snap-publish): under jax/.prune.lock (a lock older than
#   CACHE_LOCK_STALE_S, 3600 s, is taken over; a held one skips the prune): deletes .tmp-* names
#   older than CACHE_TMP_STALE_S (3600 s), every jax/<ver> except the most recently used one
#   once its newest used-<date> is CACHE_KEEP_DAYS (7) days old, then the oldest entries (by
#   mtime) until jax/ is under CACHE_CAP_BYTES (20 GiB, the cap tests/conftest.py keeps locally),
#   and every capture but the CACHE_SNAP_KEEP (10) newest. It warns when the volume has under
#   CACHE_FREE_WARN_GB (3) GB free: grow it (needs the owner).
# Knobs (env): CACHE_VOL (/vol), NUZLINE_TEST_COMPILE_CACHE (/work/_jaxcache/jax-tests, the
#   folder holding the <ver> folders, as unit.sh), CACHE_PY (python3; setup.sh and runner.sh pass
#   the run's repo/.venv/bin/python), CACHE_JAXVER (seed: the <ver> to seed; setup.sh passes
#   jax_version() of the run's checkout), CACHE_POD (temp-name tag, default $RUNPOD_POD_ID, else
#   the hostname), and the ones named above. CACHE_VOL_MOUNTCHECK=0 accepts a plain folder as
#   the volume (tests only).
set -u
VOL="${CACHE_VOL:-/vol}"; LC="${NUZLINE_TEST_COMPILE_CACHE:-/work/_jaxcache/jax-tests}"
PY="${CACHE_PY:-python3}"; POD="${CACHE_POD:-${RUNPOD_POD_ID:-$(hostname 2>/dev/null || echo pod)}}"
POD=$(printf '%s' "$POD" | tr -c 'A-Za-z0-9_' '_')
MIN_AGE_S="${CACHE_MIN_AGE_S:-600}"; KEEP_DAYS="${CACHE_KEEP_DAYS:-7}"; CAP_BYTES="${CACHE_CAP_BYTES:-21474836480}"
LOCK_STALE_S="${CACHE_LOCK_STALE_S:-3600}"; TMP_STALE_S="${CACHE_TMP_STALE_S:-3600}"; SNAP_KEEP="${CACHE_SNAP_KEEP:-10}"
FREE_WARN_GB="${CACHE_FREE_WARN_GB:-3}"; SEED_LOCK="${CACHE_SEED_LOCK:-/work/.cache_seed.lock}"
J="$VOL/jax"; S="$VOL/snap"

say() { echo "cache_sync: $*"; }
vol_ok() { [ -d "$VOL" ] || return 1; [ "${CACHE_VOL_MOUNTCHECK:-1}" = 0 ] && return 0; mountpoint -q "$VOL" 2>/dev/null; }
ver_ok() { [[ "$1" =~ ^[0-9a-f]{16}$ ]]; }
key_ok() { [[ "$1" =~ ^[0-9a-f]{40}-[0-9a-f]{16}$ ]]; }
size() { stat -c %s "$1" 2>/dev/null; }

# pyh <cmd> ...: the Python side (stdlib only, any python3).
#   list <src> <dst>          names of <src>'s entries missing in <dst>
#   select <src> <dst> <age>  "OK <name>" / "BAD <name> <reason>" for <src>'s entries missing in
#                             <dst> that are at least <age> s old
#   verify <file>...          "OK<tab><file>" / "BAD<tab><file><tab><reason>"
#   capture <snap dir>        exit 0 when capture.json lists parts and every one is "ok"
#   cap <dir> <bytes>         delete the oldest entries under <dir> until the rest total <bytes>;
#                             prints the count deleted
pyh() { "$PY" - "$@" <<'PY'
import json, os, stat, sys, time, zlib

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
sys.stdout.reconfigure(newline="\n")  # bash reads the lines; no CR on Windows


def zstd():
    try:
        from compression import zstd as z  # Python 3.14+
        return z.decompress
    except ImportError:
        pass
    try:
        import zstandard
        return lambda b: zstandard.ZstdDecompressor().decompress(b)
    except ImportError:
        return None


def check(path):
    """None when the entry decompresses completely, else the reason."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return f"unreadable ({e.strerror})"
    if not data:
        return "empty"
    if data[:4] == ZSTD_MAGIC:
        dz = zstd()
        if dz is None:
            return "zstd entry and no zstd module to check it"
        try:
            dz(data)
        except Exception as e:  # noqa: BLE001 - any failure: not published
            return f"zstd: {e}"
        return None
    try:
        d = zlib.decompressobj()
        d.decompress(data)
        if not d.eof:
            return "zlib: incomplete or truncated stream"
    except zlib.error as e:
        return f"zlib: {e}"
    return None


def entries(src):
    try:
        names = sorted(os.listdir(src))
    except OSError:
        return
    for n in names:
        if n.startswith(".") or not n.endswith("-cache"):
            continue
        p = os.path.join(src, n)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode):
            yield n, p, st


cmd, args = sys.argv[1], sys.argv[2:]
if cmd == "list":
    for n, _, _ in entries(args[0]):
        if not os.path.lexists(os.path.join(args[1], n)):
            print(n)
elif cmd == "select":
    now, age = time.time(), float(args[2])
    for n, p, st in entries(args[0]):
        if now - st.st_mtime < age or os.path.lexists(os.path.join(args[1], n)):
            continue
        why = check(p)
        print(f"OK {n}" if why is None else f"BAD {n} {why}")
elif cmd == "verify":
    for p in args:
        why = check(p)
        print(f"OK\t{p}" if why is None else f"BAD\t{p}\t{why}")
elif cmd == "capture":
    try:
        with open(os.path.join(args[0], "capture.json"), encoding="utf-8") as f:
            parts = json.load(f).get("parts") or {}
    except (OSError, ValueError, AttributeError):
        sys.exit(1)
    sys.exit(0 if parts and all(v == "ok" for v in parts.values()) else 1)
elif cmd == "cap":
    files = []
    for root, _, names in os.walk(args[0]):
        for n in names:
            if n.startswith(".") or not n.endswith("-cache"):
                continue
            p = os.path.join(root, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                files.append((st.st_mtime, st.st_size, p))
    total, gone = sum(s for _, s, _ in files), 0
    for _, s, p in sorted(files):
        if total <= int(args[1]):
            break
        try:
            os.unlink(p)
            total -= s
            gone += 1
        except OSError:
            pass
    print(gone)
else:
    sys.exit(f"unknown helper command {cmd}")
PY
}

seed() {
  vol_ok || { say "no volume mounted at $VOL: seed skipped"; return 0; }
  local ver="${CACHE_JAXVER:-}"; ver_ok "$ver" || { say "seed: CACHE_JAXVER '$ver' is not a 16-hex jax version"; return 2; }
  local src="$J/$ver" dst="$LC/$ver" t0 ok=0 bad=0 names n st p why
  mkdir -p "$dst" || return 1
  exec 7>> "$SEED_LOCK"
  flock -w 1800 7 || { say "seed: lock $SEED_LOCK still held after 1800 s: seed skipped"; return 0; }
  if [ -f "$LC/.seeded-$ver" ]; then say "seed: $ver already seeded on this pod"; return 0; fi
  mkdir -p "$src" && touch "$src/used-$(date -u +%Y%m%d)" || say "seed: cannot write $src (read only?)"
  t0=$(date +%s)
  names=$(pyh list "$src" "$dst") || { say "seed: helper failed ($PY)"; return 1; }
  local tmps=()
  while read -r n; do [ -n "$n" ] || continue
    p="$dst/.$n.tmp.$$"
    if cp "$src/$n" "$p" 2>/dev/null; then tmps+=("$p"); else rm -f "$p"; bad=$((bad + 1)); say "seed: skip $n: copy failed"; fi
  done <<< "$names"
  if [ "${#tmps[@]}" -gt 0 ]; then
    while IFS=$'\t' read -r st p why; do
      n=${p##*/.}; n=${n%.tmp.$$}
      if [ "$st" = OK ] && [ ! -e "$dst/$n" ]; then mv -f "$p" "$dst/$n" && ok=$((ok + 1))
      elif [ "$st" = OK ]; then rm -f "$p"   # JAX wrote it meanwhile: keep the local one
      else rm -f "$p"; bad=$((bad + 1)); say "seed: skip $n: $why (a bad entry on the volume: delete $src/$n while no publish runs)"; fi
    done < <(pyh verify "${tmps[@]}")
  fi
  rm -f "$dst"/.*.tmp.$$
  touch "$LC/.seeded-$ver"
  say "seed done: $ver $ok entries copied, $bad skipped, $(( $(date +%s) - t0 )) s"
}

publish() {
  vol_ok || { say "no volume mounted at $VOL: publish skipped"; return 0; }
  local d ver out ok bad st n why tmp any=0
  for d in "$LC"/*/; do ver=${d%/}; ver=${ver##*/}; ver_ok "$ver" || continue; d=${d%/}; any=1
    mkdir -p "$J/$ver" || { say "publish: cannot create $J/$ver"; continue; }
    touch "$J/$ver/used-$(date -u +%Y%m%d)"
    out=$(pyh select "$d" "$J/$ver" "$MIN_AGE_S") || { say "publish: helper failed ($PY)"; continue; }
    ok=0; bad=0
    while read -r st n why; do [ -n "$n" ] || continue
      if [ "$st" != OK ]; then bad=$((bad + 1)); say "publish: skip $n: $why"; continue; fi
      tmp="$J/$ver/.tmp-$POD-$$-$n"
      if cp "$d/$n" "$tmp" 2>/dev/null && [ "$(size "$tmp")" = "$(size "$d/$n")" ] && mv -f "$tmp" "$J/$ver/$n"; then ok=$((ok + 1))
      else rm -f "$tmp"; bad=$((bad + 1)); say "publish: skip $n: copy to the volume failed"; fi
    done <<< "$out"
    say "publish $ver: $ok entries published, $bad skipped"
  done
  [ "$any" = 1 ] || say "publish: no local cache folder under $LC"
  prune
  say "publish done"
}

# The prune lock: mkdir is atomic on the mount. A lock older than LOCK_STALE_S (a pod deleted
# mid-prune) is moved aside by one rename, so only one pod takes it over.
lock_take() { local L="$J/.prune.lock" age
  mkdir -p "$J" || return 1
  mkdir "$L" 2>/dev/null && return 0
  age=$(( $(date +%s) - $(stat -c %Y "$L" 2>/dev/null || date +%s) ))
  if [ "$age" -lt "$LOCK_STALE_S" ]; then say "prune: lock $L held ($age s old): prune skipped"; return 1; fi
  if mv -T "$L" "$L.stale.$POD.$$" 2>/dev/null; then rm -rf "$L.stale.$POD.$$"; say "prune: lock was $age s old: taken over"; fi
  mkdir "$L" 2>/dev/null && return 0
  say "prune: lock $L taken by another pod: prune skipped"; return 1; }

prune() {
  vol_ok || { say "no volume mounted at $VOL: prune skipped"; return 0; }
  lock_take || return 0
  local L="$J/.prune.lock" now cut d v u newest="" newu="" last t gone n=0 free
  echo "$POD $$ $(date +%s)" > "$L/owner"
  now=$(date +%s)
  # 1. stale temp names (a copy whose pod died)
  while IFS= read -r t; do [ -n "$t" ] || continue
    [ $(( now - $(stat -c %Y "$t" 2>/dev/null || echo "$now") )) -ge "$TMP_STALE_S" ] && { rm -rf "$t"; n=$((n + 1)); }
  done < <(find "$J" "$S" -mindepth 1 -maxdepth 2 -name '.tmp-*' 2>/dev/null)
  [ "$n" -gt 0 ] && say "prune: $n stale temp names deleted"
  # 2. jax versions: keep the most recently used; drop the others once unused for KEEP_DAYS
  cut=$(date -u -d "@$(( now - KEEP_DAYS * 86400 ))" +%Y%m%d)
  for d in "$J"/*/; do v=${d%/}; v=${v##*/}; ver_ok "$v" || continue
    last=$(ls "$J/$v" 2>/dev/null | sed -n 's/^used-\([0-9]\{8\}\)$/\1/p' | sort | tail -1)
    [ -n "$last" ] || last=$(date -u -d "@$(stat -c %Y "$J/$v")" +%Y%m%d)
    if [ -z "$newest" ] || [[ "$last" > "$newu" ]]; then newest=$v; newu=$last; fi
    for u in "$J/$v"/used-*; do [ -e "$u" ] && [ "${u##*/used-}" != "$last" ] && rm -f "$u"; done   # older markers
  done
  for d in "$J"/*/; do v=${d%/}; v=${v##*/}; ver_ok "$v" && [ "$v" != "$newest" ] || continue
    last=$(ls "$J/$v" 2>/dev/null | sed -n 's/^used-\([0-9]\{8\}\)$/\1/p' | sort | tail -1)
    [ -n "$last" ] || last=$(date -u -d "@$(stat -c %Y "$J/$v")" +%Y%m%d)
    if [[ "$last" < "$cut" ]] || [ "$last" = "$cut" ]; then rm -rf "${J:?}/$v"; say "prune: jax version $v deleted (last used $last; newest is $newest)"; fi
  done
  # 3. size cap, oldest entries first
  gone=$(pyh cap "$J" "$CAP_BYTES") && [ "${gone:-0}" -gt 0 ] && say "prune: $gone oldest entries deleted (cap $CAP_BYTES bytes)"
  # 4. captures: the SNAP_KEEP newest stay
  n=0
  while IFS= read -r d; do [ -n "$d" ] || continue; n=$((n + 1))
    [ "$n" -gt "$SNAP_KEEP" ] && { rm -rf "${d%/}"; say "prune: capture ${d%/} deleted (keep $SNAP_KEEP)"; }
  done < <(ls -dt "$S"/*/ 2>/dev/null)
  free=$(df -Pk "$VOL" 2>/dev/null | awk 'NR == 2 {printf "%d", $4 / 1048576}')
  [ -n "$free" ] && [ "$free" -lt "$FREE_WARN_GB" ] && say "WARNING: volume $VOL has $free GB free (under $FREE_WARN_GB GB): grow it (owner approval)"
  rm -rf "$L"
  say "prune done"
}

capture_ok() { grep -q '^CAPTURE_RC=0 ' "$1/times.txt" 2>/dev/null && pyh capture "$1/snap"; }

snap_publish() {
  local W="${1:-}" KEY="${2:-}" tmp
  vol_ok || { say "no volume mounted at $VOL: snap-publish skipped"; return 0; }
  key_ok "$KEY" || { say "snap-publish: key '$KEY' is not <sha40>-<key16>"; return 2; }
  capture_ok "$W" || { say "snap-publish: $W is not a complete capture (CAPTURE_RC=0 in times.txt and every part ok in snap/capture.json): not stored"; return 1; }
  mkdir -p "$S" || return 1
  [ -e "$S/$KEY" ] && { say "snap-publish: $KEY is already on the volume: kept, not overwritten"; return 0; }
  tmp="$S/.tmp-$POD-$$-$KEY"; rm -rf "$tmp"; mkdir "$tmp" || return 1
  if ! { cp -r "$W/snap" "$tmp/snap" && cp "$W/times.txt" "$tmp/times.txt" && diff -rq "$W/snap" "$tmp/snap" > /dev/null && echo "$KEY" > "$tmp/KEY"; }; then
    rm -rf "$tmp"; say "snap-publish: copy to the volume failed"; return 1; fi
  if mv -T "$tmp" "$S/$KEY" 2>/dev/null; then say "snap-publish: $KEY stored"
  else rm -rf "$tmp"
    [ -d "$S/$KEY/snap" ] || { say "snap-publish: rename to $S/$KEY failed"; return 1; }
    say "snap-publish: $KEY was stored by another pod meanwhile: kept, not overwritten"; fi
  prune
}

snap_fetch() {
  local KEY="${1:-}" DEST="${2:-}" tmp
  vol_ok || { say "no volume mounted at $VOL: snap-fetch skipped"; return 1; }
  key_ok "$KEY" || { say "snap-fetch: key '$KEY' is not <sha40>-<key16>"; return 2; }
  [ -n "$DEST" ] || { say "snap-fetch: no capture folder given"; return 2; }
  [ -f "$DEST/snap/capture.json" ] && { say "snap-fetch: $DEST/snap is already on this pod: not fetched"; return 0; }
  [ -e "$DEST/snap" ] && { say "snap-fetch: $DEST/snap exists without capture.json (an unfinished capture): not fetched"; return 1; }
  [ -d "$S/$KEY/snap" ] || { say "snap-fetch: $KEY is not on the volume"; return 1; }
  mkdir -p "$DEST" || return 1
  tmp="$DEST/.snapfetch.$$"; rm -rf "$tmp"; mkdir "$tmp" || return 1
  if ! { cp -r "$S/$KEY/snap" "$tmp/snap" && diff -rq "$S/$KEY/snap" "$tmp/snap" > /dev/null && pyh capture "$tmp/snap"; }; then
    rm -rf "$tmp"; say "snap-fetch: the copy of $KEY failed or is not a complete capture"; return 1; fi
  if ! mv -T "$tmp/snap" "$DEST/snap" 2>/dev/null; then rm -rf "$tmp"
    [ -f "$DEST/snap/capture.json" ] && { say "snap-fetch: $DEST/snap was fetched by another job meanwhile"; return 0; }
    say "snap-fetch: rename to $DEST/snap failed"; return 1; fi
  if [ ! -f "$DEST/times.txt" ]; then   # a compare's fetch makes the folder a finished capture
    { cat "$S/$KEY/times.txt" 2>/dev/null; echo "FETCHED_FROM_VOLUME $KEY $(date +%s)"; echo ALL_DONE; } > "$tmp/times.txt" \
      && mv -f "$tmp/times.txt" "$DEST/times.txt"; fi
  echo "$KEY" > "$DEST/store_key"; rm -rf "$tmp"
  say "snap-fetch: $KEY fetched into $DEST/snap"
}

case "${1:-}" in
  seed) seed ;;
  publish) publish ;;
  prune) prune ;;
  snap-publish) shift; snap_publish "$@" ;;
  snap-fetch) shift; snap_fetch "$@" ;;
  *) sed -n '6,11p' "$0" | sed 's/^# *//'; exit 2 ;;
esac
