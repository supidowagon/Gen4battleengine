#!/usr/bin/env bash
# POD SIDE. The shared network volume (DAYBOX.md "Network volume"; docs/process.md section 5):
# a sync of the pod's local XLA compile cache with the volume, and a store of complete snapshot
# captures of main. JAX keeps writing the local cache exactly as without a volume; no test
# process ever reads or writes the volume.
# Usage:
#   cache_sync.sh seed                  copy volume entries this pod lacks into the local cache
#   cache_sync.sh publish               copy local entries the volume lacks onto it, then prune (also
#                                       before the copies when the volume is under CACHE_FREE_WARN_GB left)
#   cache_sync.sh prune                 prune the volume (under its lock)
#   cache_sync.sh snap-publish <capture folder> <key>   store a complete capture under <key>
#   cache_sync.sh snap-fetch <key> <capture folder> [own]  fetch a stored capture into <folder>/snap
# Every command is a no-op that prints "cache_sync: no volume mounted at <vol>: <cmd> skipped"
# (exit 0; snap-fetch exits 1, as nothing was fetched) when <vol> is not a mountpoint.
#
# Layout on the volume (<vol>, default /vol):
#   jax/<ver>-<host>/<name>-cache  compile cache entries; <ver> = the 16-hex jax_version() folder
#                            (src/nuzline/jaxcache.py), as in the local cache; <host> = the host
#                            tag (16 hex: sha256 of /proc/cpuinfo's first processor's numeric
#                            "vendor_id", "cpu family", "model" and its sorted "flags"; not the
#                            "model name" string, not the stepping). The entries are compiled XLA
#                            CPU executables, and JAX's cache key does not hold the host CPU (jax
#                            0.12.0.1, jax/_src/cache_key.py get(), lines 93-133: the program, the
#                            jaxlib version, platform "cpu" and platform_version "cpu" (378-380),
#                            XLA_FLAGS, compile options, the device topology (270-283),
#                            compression), so a pod seeds and publishes only entries of its own
#                            tag. Why these fields: XLA's CPU compiler builds for the host's target
#                            features (jaxlib's CpuCompiler passes TargetMachineFeatures) and the
#                            LLVM host CPU name, which LLVM derives on x86 from CPUID's vendor,
#                            family, model and feature bits, not from the brand string (LLVM's
#                            Host.cpp; not in this repo, not re-checked here). Two hosts whose
#                            names differ only in marketing ("AMD EPYC 9655" / "9655P") share a
#                            tag. JAX keys each entry by program, so merging pods' caches of one
#                            tag is a plain union. A pod whose cpuinfo has no vendor_id or flags,
#                            or no numeric cpu family or model, has no tag: seed and publish skip.
#                            Folders of the earlier 12-hex tag (model name + flags) are pruned.
#   jax/<ver>-<host>/used-<YYYYMMDD> a pod seeded from or published to it that UTC day
#   jax/.prune.lock/         the prune lock (mkdir; flock is not relied on over the network mount);
#                            its owner file holds "<pod> <pid> <epoch>"
#   snap/<sha40>-<key16>/    a complete capture of main commit <sha40> with tables of table key
#                            <key16> (tools/ops/table_key.py, first 16 hex): snap/ (what compares
#                            read), times.txt of the capture, KEY (= the folder name)
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
# "ok" with both its files (<part>.pkl.gz and <part>.sha256, as tools/snap_compare.py writes
# them) present and non-empty; snap-fetch checks the stored capture the same way, its KEY file
# against the folder name, and the copy against the volume (diff -r) and the same way again.
# snap-fetch into a job's own folder ("own": the capture's run.sh) leaves its times.txt alone;
# into another folder (a compare's) it writes times.txt anew (the volume's lines, then
# FETCHED_FROM_VOLUME <key> <epoch>, ALL_DONE) and keeps an earlier one as times.txt.before-fetch,
# before it renames snap/ into place: a stop (timeout) in between puts the earlier times.txt back
# while snap/ has no capture.json, so a complete snap/ never sits beside a times.txt without
# ALL_DONE (the snapshot guard's FAILED).
# A TERM or INT (timeout: setup.sh, runner.sh and run.sh run every call under timeout -k 30 <s>,
# the limits CACHE_SEED_TIMEOUT, CACHE_PUBLISH_TIMEOUT, CACHE_FETCH_TIMEOUT, CACHE_STORE_TIMEOUT
# of DAYBOX.md "Network volume") removes this call's temp copies and releases the prune lock it
# holds, then exits 143.
# seed: once per pod and <ver> (marker <cache>/.seeded-<ver>), under the local flock
#   CACHE_SEED_LOCK (/work/.cache_seed.lock, waited for at most 600 s), so concurrent setups seed
#   once; under it the temp copies of a seed stopped earlier are removed first. Entries go in
#   batches of CACHE_SEED_BATCH (50): each batch is copied, checked and renamed before the next,
#   so a seed stopped partway (its timeout) keeps every batch it renamed and the next seed copies
#   only what is still missing; the marker is written only after a complete pass.
# prune (also run by publish and snap-publish): under jax/.prune.lock (a lock older than
#   CACHE_LOCK_STALE_S, 3600 s, or, 60 s (CACHE_LOCK_EMPTY_S), one whose owner file is empty or missing, is taken over: re-checked to still have the owner and age read
#   before it is renamed aside, and the renamed one checked again; a held one skips the prune;
#   released only while its owner file still names this pod and pid): deletes .tmp-* names whose
#   newest file is older than CACHE_TMP_STALE_S (3600 s; a copy in progress keeps writing
#   files), every jax/<ver> folder without a host tag (written before tags) and every
#   jax/<ver>-<12 hex> folder (the earlier model name tag, which no pod seeds), per host tag every
#   jax/<ver>-<host> except its most recently used one once its newest used-<date> is
#   CACHE_KEEP_DAYS (7) days old, then the oldest entries (by mtime) until jax/ is under
#   CACHE_CAP_BYTES (20 GiB, the cap tests/conftest.py keeps locally), and every capture but the
#   CACHE_SNAP_KEEP (10) newest. It warns when the volume has under CACHE_FREE_WARN_GB (3) GB
#   left (GB = 10^9 bytes, the unit of the volume's size; the pod check, podcheck.sh, uses the
#   same knob and unit): left = CACHE_VOL_QUOTA_GB (30, the volume's size; the pod cannot read it,
#   and df of the mount shows the whole storage cluster) x 10^9 minus du -sk of <vol> in bytes;
#   "used" is shown rounded up, "left" rounded down. du runs under timeout -k 5
#   CACHE_PRUNE_DU_TIMEOUT (60 s); a timed-out or failed du prints "volume used ?" and no warning.
#   Grow the volume only with the owner.
# Knobs (env): CACHE_VOL (/vol), NUZLINE_TEST_COMPILE_CACHE (/work/_jaxcache/jax-tests, the
#   folder holding the <ver> folders, as unit.sh), CACHE_PY (python3; setup.sh and runner.sh pass
#   the run's repo/.venv/bin/python), CACHE_JAXVER (seed: the <ver> to seed; setup.sh passes
#   jax_version() of the run's checkout), CACHE_POD (temp-name tag, default $RUNPOD_POD_ID, else
#   the hostname), CACHE_CPUINFO (/proc/cpuinfo, the host tag's source), and the ones named
#   above. CACHE_VOL_MOUNTCHECK=0 accepts a plain folder as the volume (tests only).
set -u
VOL="${CACHE_VOL:-/vol}"; LC="${NUZLINE_TEST_COMPILE_CACHE:-/work/_jaxcache/jax-tests}"
PY="${CACHE_PY:-python3}"; POD="${CACHE_POD:-${RUNPOD_POD_ID:-$(hostname 2>/dev/null || echo pod)}}"
POD=$(printf '%s' "$POD" | tr -c 'A-Za-z0-9_' '_')
MIN_AGE_S="${CACHE_MIN_AGE_S:-600}"; KEEP_DAYS="${CACHE_KEEP_DAYS:-7}"; CAP_BYTES="${CACHE_CAP_BYTES:-21474836480}"
LOCK_STALE_S="${CACHE_LOCK_STALE_S:-3600}"; LOCK_EMPTY_S="${CACHE_LOCK_EMPTY_S:-60}"; TMP_STALE_S="${CACHE_TMP_STALE_S:-3600}"; SNAP_KEEP="${CACHE_SNAP_KEEP:-10}"
FREE_WARN_GB="${CACHE_FREE_WARN_GB:-3}"; SEED_LOCK="${CACHE_SEED_LOCK:-/work/.cache_seed.lock}"; SEED_BATCH="${CACHE_SEED_BATCH:-50}"
[[ "$SEED_BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "cache_sync: CACHE_SEED_BATCH '$SEED_BATCH' is not a whole number above 0"; exit 2; }
J="$VOL/jax"; S="$VOL/snap"

say() { echo "cache_sync: $*"; }
vol_ok() { [ -d "$VOL" ] || return 1; [ "${CACHE_VOL_MOUNTCHECK:-1}" = 0 ] && return 0; mountpoint -q "$VOL" 2>/dev/null; }
ver_ok() { [[ "$1" =~ ^[0-9a-f]{16}$ ]]; }
vt_ok() { [[ "$1" =~ ^[0-9a-f]{16}-[0-9a-f]{16}$ ]]; }   # <ver>-<host>
vt_old() { [[ "$1" =~ ^[0-9a-f]{16}-[0-9a-f]{12}$ ]]; }  # <ver>-<12-hex tag of the model name scheme>
key_ok() { [[ "$1" =~ ^[0-9a-f]{40}-[0-9a-f]{16}$ ]]; }
size() { stat -c %s "$1" 2>/dev/null; }
# cpu_field <name>: the value of CACHE_CPUINFO's first "<name> : <value>" line (the first
# processor), CR and surrounding blanks dropped; "model" never matches "model name".
cpu_field() { sed -n "s/^$1[[:space:]]*:[[:space:]]*//p" "${CACHE_CPUINFO:-/proc/cpuinfo}" 2>/dev/null | head -1 | tr -d '\r' | sed 's/[[:space:]]*$//'; }
# host_tag: 16 hex of sha256("vendor_id=<v>\ncpu family=<n>\nmodel=<n>\nflags=<sorted flags>\n")
# from CACHE_CPUINFO's first processor; the marketing "model name" and the stepping are left out.
# Fails when vendor_id or flags is empty or cpu family or model is not a number.
host_tag() { local v fa mo fl
  v=$(cpu_field vendor_id); fa=$(cpu_field 'cpu family'); mo=$(cpu_field model)
  fl=$(cpu_field flags | tr -s ' \t' '\n\n' | sed '/^$/d' | LC_ALL=C sort -u | tr '\n' ' ')
  [ -n "$v" ] && [ -n "$fl" ] && [[ "$fa" =~ ^[0-9]+$ ]] && [[ "$mo" =~ ^[0-9]+$ ]] || return 1
  printf 'vendor_id=%s\ncpu family=%s\nmodel=%s\nflags=%s\n' "$v" "$fa" "$mo" "$fl" | sha256sum | cut -c1-16; }
# newest <path>: the newest mtime (whole seconds) of <path> and everything under it.
newest() { find "$1" -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1; }
# A TERM or INT (timeout) removes this call's temp copies (TMPS) and releases a prune lock it
# holds (LOCKED), then exits 143.
TMPS=(); LOCKED=""; UNDO_TIMES=""; UNDO_TMP=""
# undo_times: a compare's snap-fetch wrote <dest>/times.txt (UNDO_TIMES) before renaming snap/
# into place; while <dest>/snap has no capture.json it removes that times.txt and puts back the
# one it replaced (kept in UNDO_TMP until the rename), so the folder is as before the fetch.
undo_times() { [ -n "$UNDO_TIMES" ] || return 0
  if [ ! -f "$UNDO_TIMES/snap/capture.json" ]; then rm -f "$UNDO_TIMES/times.txt"
    [ ! -f "$UNDO_TMP/times.txt.before-fetch" ] || mv -f "$UNDO_TMP/times.txt.before-fetch" "$UNDO_TIMES/times.txt"; fi
  UNDO_TIMES=""; }
on_term() { trap '' TERM INT; undo_times; [ "${#TMPS[@]}" -eq 0 ] || rm -rf -- "${TMPS[@]}"; [ -z "$LOCKED" ] || lock_release
  say "stopped by a signal: temp copies removed"; exit 143; }
trap on_term TERM INT

# pyh <cmd> ...: the Python side (stdlib only, any python3).
#   list <src> <dst>          names of <src>'s entries missing in <dst>
#   select <src> <dst> <age>  "OK <name>" / "BAD <name> <reason>" for <src>'s entries missing in
#                             <dst> that are at least <age> s old
#   verify <file>...          "OK<tab><file>" / "BAD<tab><file><tab><reason>"
#   capture <snap dir>        exit 0 when capture.json lists parts, every one is "ok" and has a
#                             non-empty <part>.pkl.gz and <part>.sha256; else exit 1, the reason
#                             on stderr
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
        sys.exit("capture.json missing or unreadable")
    if not isinstance(parts, dict) or not parts:
        sys.exit("capture.json lists no parts")
    for part, status in sorted(parts.items()):
        if status != "ok":
            sys.exit(f"part {part} is {status}")
        if not part or "/" in part or "\\" in part or part.startswith("."):
            sys.exit(f"part name {part!r} is not a file name")
        for suffix in (".pkl.gz", ".sha256"):
            try:
                st = os.stat(os.path.join(args[0], part + suffix))
            except OSError:
                sys.exit(f"part {part}: {part}{suffix} missing")
            if not stat.S_ISREG(st.st_mode) or st.st_size == 0:
                sys.exit(f"part {part}: {part}{suffix} empty or not a file")
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
  local ht; ht=$(host_tag) || { say "seed: no vendor_id, cpu family, model or flags in ${CACHE_CPUINFO:-/proc/cpuinfo}: no host tag, seed skipped"; return 0; }
  local src="$J/$ver-$ht" dst="$LC/$ver" t0 ok=0 bad=0 names n st p why
  mkdir -p "$dst" || return 1
  exec 7>> "$SEED_LOCK"
  flock -w 600 7 || { say "seed: lock $SEED_LOCK still held after 600 s: seed skipped"; return 0; }
  if [ -f "$LC/.seeded-$ver" ]; then say "seed: $ver already seeded on this pod"; return 0; fi
  rm -f "$dst"/.*.tmp.*   # copies of a seed stopped earlier (only one seed runs: the lock)
  say "seed: host tag $ht"
  mkdir -p "$src" && touch "$src/used-$(date -u +%Y%m%d)" || say "seed: cannot write $src (read only?)"
  t0=$(date +%s)
  names=$(pyh list "$src" "$dst") || { say "seed: helper failed ($PY)"; return 1; }
  # Batches of SEED_BATCH entries: each batch is copied, checked and renamed before the next, so
  # a seed stopped partway (timeout) keeps every batch it renamed, and the next seed copies only
  # what is still missing. The marker is written only after a complete pass.
  local tmps=() i=0
  seed_batch() { [ "${#tmps[@]}" -gt 0 ] || return 0
    while IFS=$'\t' read -r st p why; do
      n=${p##*/.}; n=${n%.tmp.$$}
      if [ "$st" = OK ] && [ ! -e "$dst/$n" ]; then mv -f "$p" "$dst/$n" && ok=$((ok + 1))
      elif [ "$st" = OK ]; then rm -f "$p"   # JAX wrote it meanwhile: keep the local one
      else rm -f "$p"; bad=$((bad + 1)); say "seed: skip $n: $why (a bad entry on the volume: delete $src/$n while no publish runs)"; fi
    done < <(pyh verify "${tmps[@]}")
    rm -f "${tmps[@]}"; tmps=(); TMPS=(); }
  while read -r n; do [ -n "$n" ] || continue
    p="$dst/.$n.tmp.$$"
    TMPS+=("$p")
    if cp "$src/$n" "$p" 2>/dev/null; then tmps+=("$p"); else rm -f "$p"; bad=$((bad + 1)); say "seed: skip $n: copy failed"; fi
    i=$((i + 1)); [ $((i % SEED_BATCH)) -ne 0 ] || seed_batch
  done <<< "$names"
  seed_batch
  rm -f "$dst"/.*.tmp.$$; TMPS=()
  touch "$LC/.seeded-$ver"
  say "seed done: $ver $ok entries copied, $bad skipped, $(( $(date +%s) - t0 )) s"
}

# vol_left <quota GB> <du timeout s>: prints the bytes left on the volume (quota minus du -sk of
# <vol> in bytes; prune step 5 and publish read it); prints nothing and fails when du fails or
# times out. A bad quota or timeout falls back silently to 30 / 60.
vol_left() { local q="$1" dt="$2" k G=1000000000
  [[ "$q" =~ ^[1-9][0-9]*$ ]] || q=30; [[ "$dt" =~ ^[1-9][0-9]*$ ]] || dt=60
  k=$(timeout -k 5 "$dt" du -sk "$VOL" 2>/dev/null | awk 'NR == 1 {print $1}')
  [[ "$k" =~ ^[0-9]+$ ]] || return 1
  echo $(( q * G - k * 1024 )); }

publish() {
  vol_ok || { say "no volume mounted at $VOL: publish skipped"; return 0; }
  local d ver out ok bad st n why tmp any=0 ht V t0 left fw="$FREE_WARN_GB"; t0=$(date +%s)
  [[ "$fw" =~ ^[0-9]+$ ]] || fw=3
  # Over the warning line (under fw GB left): prune before the copies, so a full volume frees
  # space first; the prune after the copies still runs. This pod's used- markers are touched
  # first, so the early prune cannot delete this pod's own version folder as unused and have the
  # copies put it straight back. (The cap step may still delete current entries that the copies
  # then replace: acceptable.)
  if left=$(vol_left "${CACHE_VOL_QUOTA_GB:-30}" "${CACHE_PRUNE_DU_TIMEOUT:-60}") && [ "$left" -lt $(( fw * 1000000000 )) ]; then
    say "publish: volume under $fw GB left: prune before the copies"
    if ht=$(host_tag); then
      for d in "$LC"/*/; do ver=${d%/}; ver=${ver##*/}; ver_ok "$ver" || continue
        mkdir -p "$J/$ver-$ht" && touch "$J/$ver-$ht/used-$(date -u +%Y%m%d)"; done; fi
    prune; fi
  if ht=$(host_tag); then
  for d in "$LC"/*/; do ver=${d%/}; ver=${ver##*/}; ver_ok "$ver" || continue; d=${d%/}; any=1; V="$J/$ver-$ht"
    mkdir -p "$V" || { say "publish: cannot create $V"; continue; }
    touch "$V/used-$(date -u +%Y%m%d)"
    out=$(pyh select "$d" "$V" "$MIN_AGE_S") || { say "publish: helper failed ($PY)"; continue; }
    ok=0; bad=0
    while read -r st n why; do [ -n "$n" ] || continue
      if [ "$st" != OK ]; then bad=$((bad + 1)); say "publish: skip $n: $why"; continue; fi
      tmp="$V/.tmp-$POD-$$-$n"; TMPS=("$tmp")
      if cp "$d/$n" "$tmp" 2>/dev/null && [ "$(size "$tmp")" = "$(size "$d/$n")" ] && mv -f "$tmp" "$V/$n"; then ok=$((ok + 1))
      else rm -f "$tmp"; bad=$((bad + 1)); say "publish: skip $n: copy to the volume failed"; fi
      TMPS=()
    done <<< "$out"
    say "publish $ver-$ht: $ok entries published, $bad skipped"
  done
  [ "$any" = 1 ] || say "publish: no local cache folder under $LC"
  else say "publish: no vendor_id, cpu family, model or flags in ${CACHE_CPUINFO:-/proc/cpuinfo}: no host tag, nothing published"; fi
  prune
  say "publish wall $(( $(date +%s) - t0 )) s"
  say "publish done"
}

# The prune lock: mkdir is atomic on the mount; the owner file ("<pod> <pid> <epoch>") is written
# at once. A lock older than LOCK_STALE_S (a pod deleted mid-prune) is moved aside by one rename,
# so only one pod takes it over; right before the rename its owner and mtime are read again and
# must be the ones that made it stale (else another pod took it over meanwhile: skip), and the
# renamed folder is checked once more (a fresh lock renamed by a race is put back).
# A lock whose owner file is empty or missing (its pod died between mkdir and the owner write)
# is stale once it is LOCK_EMPTY_S (60 s) old: a live holder is never inside its mkdir-to-write
# gap that long, and a pod that took the folder in that gap makes the holder's lock_mine fail.
# lock_mine fails unless the owner file reads back as this pod and pid: another pod may have
# taken the folder between mkdir and the write (a write into a missing folder fails too).
lock_mine() { local o="" p=""
  echo "$POD $$ $(date +%s)" > "$1/owner" 2>/dev/null || return 1
  { read -r o p _ < "$1/owner"; } 2>/dev/null
  [ "$o $p" = "$POD $$" ] || return 1
  LOCKED=1; }
lock_take() { local L="$J/.prune.lock" A="$J/.prune.lock.stale.$POD.$$" m own age
  mkdir -p "$J" || return 1
  if mkdir "$L" 2>/dev/null; then lock_mine "$L" && return 0; say "prune: lock $L taken by another pod before its owner file was written: prune skipped"; return 1; fi
  m=$(stat -c %Y "$L" 2>/dev/null) || m=$(date +%s); own=$(cat "$L/owner" 2>/dev/null)
  age=$(( $(date +%s) - m ))
  if [ "$age" -lt "$LOCK_STALE_S" ] && { [ -n "$own" ] || [ "$age" -lt "$LOCK_EMPTY_S" ]; }; then say "prune: lock $L held ($age s old): prune skipped"; return 1; fi
  if [ "$(stat -c %Y "$L" 2>/dev/null)" != "$m" ] || [ "$(cat "$L/owner" 2>/dev/null)" != "$own" ]; then
    say "prune: lock $L changed while being taken over: prune skipped"; return 1; fi
  if mv -T "$L" "$A" 2>/dev/null; then
    if [ "$(stat -c %Y "$A" 2>/dev/null)" != "$m" ] || [ "$(cat "$A/owner" 2>/dev/null)" != "$own" ]; then
      mv -T "$A" "$L" 2>/dev/null || rm -rf "$A"
      say "prune: lock $L was taken by another pod meanwhile: prune skipped"; return 1; fi
    rm -rf "$A"; say "prune: lock was $age s old (owner ${own:-none}): taken over"; fi
  if mkdir "$L" 2>/dev/null; then lock_mine "$L" && return 0; say "prune: lock $L taken by another pod before its owner file was written: prune skipped"; return 1; fi
  say "prune: lock $L taken by another pod: prune skipped"; return 1; }
# lock_release: remove the lock only while its owner file still names this pod and pid.
lock_release() { local L="$J/.prune.lock" o="" p=""
  { read -r o p _ < "$L/owner"; } 2>/dev/null
  if [ "$o $p" = "$POD $$" ]; then rm -rf "$L"
  else say "prune: lock $L no longer names this pod and pid (owner: ${o:-none} ${p}): left to its holder"; fi
  LOCKED=""; }

prune() {
  vol_ok || { say "no volume mounted at $VOL: prune skipped"; return 0; }
  lock_take || return 0
  local now cut d v u h last t gone n=0 free; local -A NEWV=() NEWU=()
  now=$(date +%s)
  # 1. stale temp names (a copy whose pod died): judged by the newest file inside, as a capture
  #    copy in progress keeps writing new files while its folder's own mtime may stay old
  while IFS= read -r t; do [ -n "$t" ] || continue
    u=$(newest "$t"); [ $(( now - ${u:-$now} )) -ge "$TMP_STALE_S" ] && { rm -rf "$t"; n=$((n + 1)); }
  done < <(find "$J" "$S" -mindepth 1 -maxdepth 2 -name '.tmp-*' 2>/dev/null)
  [ "$n" -gt 0 ] && say "prune: $n stale temp names deleted"
  # 2. jax folders: an untagged <ver> (written before host tags; no pod seeds it) goes; per host
  #    tag the most recently used <ver>-<host> stays, the others go once unused for KEEP_DAYS
  cut=$(date -u -d "@$(( now - KEEP_DAYS * 86400 ))" +%Y%m%d)
  lastused() { local l; l=$(ls "$J/$1" 2>/dev/null | sed -n 's/^used-\([0-9]\{8\}\)$/\1/p' | sort | tail -1)
    [ -n "$l" ] || l=$(date -u -d "@$(stat -c %Y "$J/$1")" +%Y%m%d); echo "$l"; }
  for d in "$J"/*/; do v=${d%/}; v=${v##*/}
    if ver_ok "$v"; then rm -rf "${J:?}/$v"; say "prune: jax folder $v has no host tag (written before host tags): deleted"; continue; fi
    if vt_old "$v"; then rm -rf "${J:?}/$v"; say "prune: jax folder $v has a 12-hex host tag (the model name scheme, no pod seeds it): deleted"; continue; fi
    vt_ok "$v" || continue
    last=$(lastused "$v"); h=${v#*-}
    if [ -z "${NEWV[$h]:-}" ] || [[ "$last" > "${NEWU[$h]}" ]]; then NEWV[$h]=$v; NEWU[$h]=$last; fi
    for u in "$J/$v"/used-*; do [ -e "$u" ] && [ "${u##*/used-}" != "$last" ] && rm -f "$u"; done   # older markers
  done
  for d in "$J"/*/; do v=${d%/}; v=${v##*/}; vt_ok "$v" || continue; h=${v#*-}
    [ "$v" != "${NEWV[$h]:-}" ] || continue
    last=$(lastused "$v")
    if [[ "$last" < "$cut" ]] || [ "$last" = "$cut" ]; then rm -rf "${J:?}/$v"; say "prune: jax version $v deleted (last used $last; newest of host $h is ${NEWV[$h]:-})"; fi
  done
  # 3. size cap, oldest entries first
  gone=$(pyh cap "$J" "$CAP_BYTES") && [ "${gone:-0}" -gt 0 ] && say "prune: $gone oldest entries deleted (cap $CAP_BYTES bytes)"
  # 4. captures: the SNAP_KEEP newest stay
  n=0
  while IFS= read -r d; do [ -n "$d" ] || continue; n=$((n + 1))
    [ "$n" -gt "$SNAP_KEEP" ] && { rm -rf "${d%/}"; say "prune: capture ${d%/} deleted (keep $SNAP_KEEP)"; }
  done < <(ls -dt "$S"/*/ 2>/dev/null)
  # 5. space left: the quota minus du (df of the mount shows the whole storage cluster)
  local q="${CACHE_VOL_QUOTA_GB:-30}" dt="${CACHE_PRUNE_DU_TIMEOUT:-60}" k ub G=1000000000 fw="$FREE_WARN_GB"
  [[ "$q" =~ ^[1-9][0-9]*$ ]] || { say "prune: CACHE_VOL_QUOTA_GB '$q' is not a whole number: 30 used"; q=30; }
  [[ "$fw" =~ ^[0-9]+$ ]] || { say "prune: CACHE_FREE_WARN_GB '$fw' is not a whole number: 3 used"; fw=3; }
  [[ "$dt" =~ ^[1-9][0-9]*$ ]] || dt=60
  if free=$(vol_left "$q" "$dt"); then ub=$(( q * G - free ))
    say "prune: volume used $(( (ub + G - 1) / G )) GB of $q GB"
    [ "$free" -lt $(( fw * G )) ] && say "WARNING: volume $VOL has $(( free / G )) GB left of $q GB (under $fw GB): grow it (owner approval)"
  else say "prune: volume used ? (du failed or took over $dt s)"; fi
  lock_release
  say "prune done"
}

# capture_ok <folder>: CAPTURE_RC=0 in <folder>/times.txt and <folder>/snap a complete capture
# (pyh capture); else exit 1 and prints the reason.
capture_ok() { local why
  grep -q '^CAPTURE_RC=0 ' "$1/times.txt" 2>/dev/null || { echo "no CAPTURE_RC=0 in times.txt"; return 1; }
  why=$(pyh capture "$1/snap" 2>&1) || { echo "${why:-capture check failed}"; return 1; }; }

snap_publish() {
  local W="${1:-}" KEY="${2:-}" tmp why t0; t0=$(date +%s)
  vol_ok || { say "no volume mounted at $VOL: snap-publish skipped"; return 0; }
  key_ok "$KEY" || { say "snap-publish: key '$KEY' is not <sha40>-<key16>"; return 2; }
  why=$(capture_ok "$W") || { say "snap-publish: $W is not a complete capture ($why): not stored"; return 1; }
  mkdir -p "$S" || return 1
  [ -e "$S/$KEY" ] && { say "snap-publish: $KEY is already on the volume: kept, not overwritten"; return 0; }
  tmp="$S/.tmp-$POD-$$-$KEY"; TMPS=("$tmp"); rm -rf "$tmp"; mkdir "$tmp" || return 1
  if ! { cp -r "$W/snap" "$tmp/snap" && cp "$W/times.txt" "$tmp/times.txt" && diff -rq "$W/snap" "$tmp/snap" > /dev/null \
         && why=$(capture_ok "$tmp") && echo "$KEY" > "$tmp/KEY"; }; then
    rm -rf "$tmp"; TMPS=(); say "snap-publish: copy to the volume failed${why:+ ($why)}"; return 1; fi
  if mv -T "$tmp" "$S/$KEY" 2>/dev/null; then TMPS=(); say "snap-publish: $KEY stored ($(( $(date +%s) - t0 )) s)"
  else rm -rf "$tmp"; TMPS=()
    [ -d "$S/$KEY/snap" ] || { say "snap-publish: rename to $S/$KEY failed"; return 1; }
    say "snap-publish: $KEY was stored by another pod meanwhile: kept, not overwritten"; fi
  prune
}

snap_fetch() {
  local KEY="${1:-}" DEST="${2:-}" OWN="${3:-}" tmp why k t0; t0=$(date +%s)
  vol_ok || { say "no volume mounted at $VOL: snap-fetch skipped"; return 1; }
  key_ok "$KEY" || { say "snap-fetch: key '$KEY' is not <sha40>-<key16>"; return 2; }
  [ -n "$DEST" ] || { say "snap-fetch: no capture folder given"; return 2; }
  case "$OWN" in ''|own) ;; *) say "snap-fetch: third argument '$OWN' is not 'own'"; return 2;; esac
  [ -f "$DEST/snap/capture.json" ] && { say "snap-fetch: $DEST/snap is already on this pod: not fetched"; return 0; }
  [ -e "$DEST/snap" ] && { say "snap-fetch: $DEST/snap exists without capture.json (an unfinished capture): not fetched"; return 1; }
  [ -d "$S/$KEY/snap" ] || { say "snap-fetch: $KEY is not on the volume"; return 1; }
  k=$(tr -d ' \r\n' < "$S/$KEY/KEY" 2>/dev/null)
  [ "$k" = "$KEY" ] || { say "snap-fetch: $S/$KEY/KEY names '${k:-nothing}', not $KEY: not fetched"; return 1; }
  why=$(capture_ok "$S/$KEY") || { say "snap-fetch: $KEY on the volume is not a complete capture ($why): not fetched"; return 1; }
  mkdir -p "$DEST" || return 1
  tmp="$DEST/.snapfetch.$$"; TMPS=("$tmp"); rm -rf "$tmp"; mkdir "$tmp" || return 1
  if ! { cp -r "$S/$KEY/snap" "$tmp/snap" && diff -rq "$S/$KEY/snap" "$tmp/snap" > /dev/null && why=$(pyh capture "$tmp/snap" 2>&1); }; then
    rm -rf "$tmp"; TMPS=(); say "snap-fetch: the copy of $KEY failed or is not a complete capture${why:+ ($why)}"; return 1; fi
  # A compare's fetch makes the folder a finished capture: its times.txt (a stale one is kept
  # aside) is written before snap/ is renamed into place, so a stop between the two never leaves
  # a complete snap/ beside a times.txt without ALL_DONE (the snapshot guard's FAILED). A stop
  # or failed rename after it puts the earlier times.txt back (UNDO_TIMES, also run by on_term).
  if [ -z "$OWN" ]; then
    { cat "$S/$KEY/times.txt" 2>/dev/null; echo "FETCHED_FROM_VOLUME $KEY $(date +%s)"; echo ALL_DONE; } > "$tmp/times.txt" || {
      rm -rf "$tmp"; TMPS=(); say "snap-fetch: cannot write $tmp/times.txt"; return 1; }
    UNDO_TMP="$tmp"; UNDO_TIMES="$DEST"; [ ! -f "$DEST/times.txt" ] || mv -f "$DEST/times.txt" "$tmp/times.txt.before-fetch"
    mv -f "$tmp/times.txt" "$DEST/times.txt"; fi
  if ! mv -T "$tmp/snap" "$DEST/snap" 2>/dev/null; then
    if [ -f "$DEST/snap/capture.json" ]; then UNDO_TIMES=""; [ ! -f "$tmp/times.txt.before-fetch" ] || mv -f "$tmp/times.txt.before-fetch" "$DEST/times.txt.before-fetch"
      rm -rf "$tmp"; TMPS=(); say "snap-fetch: $DEST/snap was fetched by another job meanwhile"; return 0; fi
    undo_times; rm -rf "$tmp"; TMPS=(); say "snap-fetch: rename to $DEST/snap failed"; return 1; fi
  UNDO_TIMES=""; [ ! -f "$tmp/times.txt.before-fetch" ] || mv -f "$tmp/times.txt.before-fetch" "$DEST/times.txt.before-fetch"
  echo "$KEY" > "$DEST/store_key"; rm -rf "$tmp"; TMPS=()
  say "snap-fetch: $KEY fetched into $DEST/snap ($(( $(date +%s) - t0 )) s)"
}

case "${1:-}" in
  seed) seed ;;
  publish) publish ;;
  prune) prune ;;
  snap-publish) shift; snap_publish "$@" ;;
  snap-fetch) shift; snap_fetch "$@" ;;
  *) sed -n '6,11p' "$0" | sed 's/^# *//'; exit 2 ;;
esac
