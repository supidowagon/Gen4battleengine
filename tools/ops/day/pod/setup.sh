#!/usr/bin/env bash
# POD SIDE. Run from /work/<run> after payload.tar is unpacked there (code.tar, tables.tar.xz,
# meta.env, files.txt, bin/). Builds <run>/repo from the commit, unpacks the built tables, links
# the decomp, syncs the env, collects every entry of files.txt, cuts units (one test file per
# unit, at most <ids per chunk> ids each; an id already queued by an earlier line is skipped)
# and starts bin/runner.sh detached.
# files.txt lines: "<test file, folder or node id> <ids per chunk> [A]"; A = no -m filter.
set -euxo pipefail
export PATH="/root/.local/bin:/work/_bin:$PATH" UV_CACHE_DIR=/work/_uvcache UV_PYTHON_INSTALL_DIR=/work/_python
D=$(cd "$(dirname "$0")/.." && pwd); cd "$D"; . ./meta.env; MARK="${MARK:-not slow}"
echo "SETUP_START $(date +%s)"
[ -d "/work/_decomp/$DECOMP" ] || { git -C /work/_decomp/repo fetch -q origin || true; git -C /work/_decomp/repo worktree add -q --detach "/work/_decomp/$DECOMP" "$DECOMP"; }
R="$D/repo"; rm -rf "$R"; mkdir "$R"; tar --no-same-owner -xf code.tar -C "$R"
( cd "$R" && git init -q && git add -A && git -c user.name=ci -c user.email=ci@localhost commit -qm "snapshot $COMMIT" )
tar --no-same-owner -xJf tables.tar.xz -C "$R"; [ -f "$R/data/gen4_platinum_kaizo/manifest.json" ]
mkdir -p "$R/external"; ln -sfn "/work/_decomp/$DECOMP" "$R/external/pokeplatinum"
( cd "$R" && uv sync --frozen -q )
# The damage oracle needs node + @smogon/calc; node is installed once per pod (DAYBOX.md).
if command -v npm >/dev/null && [ -f "$R/tools/smogon_calc/package-lock.json" ]; then ( cd "$R/tools/smogon_calc" && npm ci -s ); fi
rm -rf col out; mkdir -p col out
grep -v '^#' files.txt | grep . | awk '{print NR "|" $1 "|" ($3 == "A" ? "A" : "N")}' > col/jobs.txt
collect_one() { IFS='|' read -r n f mk <<< "$1"; cd "$R"; ma=(-m "$MARK"); [ "$mk" = A ] && ma=()
  uv run --frozen pytest --collect-only -q "${ma[@]}" -p no:cacheprovider "$f" > "$D/col/$n.out" 2>&1
  echo $? > "$D/col/$n.rc"; grep '::' "$D/col/$n.out" > "$D/col/$n.ids" || true; }
export -f collect_one; export D R MARK
xargs -d '\n' -P 8 -I{} bash -c 'collect_one "$@"' _ {} < col/jobs.txt
(cd "$R" && uv run --frozen python - "$D") <<'PY'
import sys, pathlib, collections
D = pathlib.Path(sys.argv[1]); q = []; summ = []; n = 0; seen = set()
for k, line in enumerate([l for l in (D / "files.txt").read_text().splitlines() if l.strip() and not l.startswith("#")], 1):
    p = line.split(); f, chunk = p[0], int(p[1]); mk = "A" if len(p) > 2 and p[2] == "A" else "N"
    rc = (D / "col" / f"{k}.rc").read_text().strip()
    by = collections.OrderedDict()
    for i in (D / "col" / f"{k}.ids").read_text().splitlines():
        if i.strip() and i not in seen: seen.add(i); by.setdefault(i.split("::")[0], []).append(i)
    if not by: summ.append(f"{f} 0 collect_rc={rc}")
    for tf, ids in by.items():
        summ.append(f"{tf} {len(ids)} collect_rc={rc}")
        for s in range(0, len(ids), chunk):
            n += 1; u = f"u_{n:03d}"
            (D / "out" / f"{u}.ids").write_text("\n".join(ids[s:s + chunk]) + "\n")
            (D / "out" / f"{u}.m").write_text(mk + "\n")
            q.append(f"{u} {tf}")
(D / "out" / "queue.txt").write_text("\n".join(q) + "\n")
(D / "collected.txt").write_text("\n".join(summ) + "\n")
print("units", n, "ids", len(seen))
PY
cat collected.txt
echo "SETUP_END $(date +%s)"
nohup bash "$D/bin/runner.sh" > "$D/runner.log" 2>&1 < /dev/null &
echo RUNNER_STARTED
