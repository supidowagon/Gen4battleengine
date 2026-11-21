#!/usr/bin/env bash
# Off-site copy to the private git remote named `offsite` (docs/process.md section 12, Off-site copy).
#   bash tools/ops/offsite.sh            push every local branch, mirror the state files, push them
#   bash tools/ops/offsite.sh --dry-run  print what would be pushed and copied; change nothing
#   bash tools/ops/offsite.sh --list     print the selection under the scratch folder (+ kept, - excluded); no git
#   bash tools/ops/offsite.sh --check D  hard-exclusion check of every file under folder D; exit 1 on an offender
# 1. Branches: `git push offsite --all` (no force, no delete); rejected branches are reported, the run goes on.
# 2. State: the allow-listed git-ignored files under _scratch/ (paths relative to _scratch/) are mirrored into
#    the clone _scratch/offsite/state/ (branch scratch-state, no common history with the code), checked
#    again, committed when something changed and pushed (no force). Deletions are mirrored.
# Never pushes to any remote but `offsite`; refuses when `offsite` is missing or has origin's URL.
set -u
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$OPS/common.sh"

MODE=run; CHECK_DIR=
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --list) MODE=list ;;
  --check) MODE=check; CHECK_DIR=${2:?"--check needs a folder"} ;;
  *) echo "usage: offsite.sh [--dry-run | --list | --check <dir>]" >&2; exit 2 ;;
esac

BRANCH=scratch-state
STATE="$SCRATCH/offsite/state"
MAX_FILE=2097152   # 2 MB: any file over this is excluded
MAX_LOG=1048576    # 1 MB: any *.log over this is excluded
# Folders never entered (name, case-insensitive): caches, envs, code and table copies, emulator folders.
PRUNE_DIRS='^(\.git|\.venv|venv|node_modules|__pycache__|\.pytest_cache|\.hypothesis|\.ruff_cache|\.mypy_cache|\.cache|[^/]*cache[^/]*|external|out|src|tests|tools|docs|data|intermediate[^/]*|gen4_platinum[^/]*|emu|shots|roms|stateslots|battery|home|bin|\.runpod|backup[^/]*)$'
# Text extensions kept without a content sniff; any other file is kept only when it has no NUL byte.
TEXT_EXT='^(md|txt|py|sh|ps1|psm1|json|jsonl|toml|ya?ml|csv|tsv|log|out|list|patch|diff|err|lua|js|spec|summary)$'
# Content that marks a key or token (extended regex, grep -E).
SECRET_RE='-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY|rpa_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|(api_?key|apikey|secret|password|passwd|token)["'"'"']?[[:space:]]*[=:][[:space:]]*["'"'"']?[A-Za-z0-9_+/=-]{20,}'

# hard_reason <size> <path relative to _scratch>: sets REASON to why the file must never leave the PC,
# or to "". The hard exclusions; checked at selection and again over the whole state tree before every
# commit. Sets a variable instead of printing, so no subshell per file.
hard_reason() {
  local size=$1 p=${2,,} d b
  REASON=
  b=${p##*/}; case "$p" in */*) d="/${p%/*}/" ;; *) d=/ ;; esac
  case "$d" in
    */.git/*) REASON=nested-repo; return ;;
    */.cache/*|*/.venv/*|*/external/*) REASON=cache-or-external; return ;;
    */backup*) REASON=backup-folder; return ;;
    */.runpod/*) REASON=runpod-config; return ;;
  esac
  case "/$p" in /rent/out/*) REASON=run-output; return ;; esac
  case "$b" in
    *.nds|*.sav|*.dsv|*.ds[0-9]|*.srm|*.gba|*.gbc|*.gb|*.sgm|*.zip|*.7z|*.rar|*.tar|*.tar.*|*.tgz|*.txz|*.gz|*.xz|*.npz|*.npy|*.pkl|*.pickle|*.bin|*.exe|*.dll)
      REASON=rom-save-archive-or-binary; return ;;
    *.key|*.pem|*.p12|*.pfx|*.crt|*.asc|*.gpg|.env|.env.*|*.env|local.toml|id_rsa*|id_ed25519*|id_ecdsa*|.netrc|.git-credentials|credentials*|*.b64)
      REASON=key-or-credential; return ;;
    .gitignore|.gitattributes|.gitmodules)
      REASON=git-control-file; return ;;  # would change what the state clone tracks or how
  esac
  [ "$size" -gt "$MAX_FILE" ] && { REASON=over-2MB; return; }
  case "$b" in *.log) [ "$size" -gt "$MAX_LOG" ] && REASON=log-over-1MB ;; esac
  return 0
}

# soft_reason <path>: the allow-list per root; sets REASON like hard_reason. handoff, manager, e<n>: text
# files except exec-pod bodies; rent: notes (*.md), pod id files, pod lists and its top-level helper
# scripts, never its run outputs.
soft_reason() {
  local p=$1 b=${1##*/}; local bl=${b,,}
  REASON=
  case "$p" in
    handoff/*|manager/*|e[0-9]*/*)
      case "$bl" in *body*.json|*body*.out) REASON=exec-pod-body ;; esac ;;
    rent/lists/*.txt) ;;
    rent/*/*) case "$bl" in *.md|pod_id*|*pod*.txt) ;; *) REASON=rent-run-file ;; esac ;;
    rent/*) case "$bl" in *.md|pod_id*|*pod*.txt|*.sh|*.py) ;; *) REASON=rent-run-file ;; esac ;;
    *) REASON=not-allow-listed ;;
  esac
  return 0
}

# select_files: run in $SCRATCH. Prints "+<TAB>size<TAB>path" for each kept file and
# "-<TAB>reason<TAB>path" for each excluded file or pruned folder (folders end in /).
select_files() {
  local roots=() r kind a b ext tmp
  for r in handoff manager e[0-9]* rent; do [ -d "$r" ] && roots+=("$r"); done
  [ ${#roots[@]} -gt 0 ] || return 0
  tmp=$(mktemp -d) || return 1
  find "${roots[@]}" -regextype posix-extended \
    \( -type d -iregex "(.*/)?${PRUNE_DIRS#^}" -printf 'D\t%p\n' -prune \) -o \
    \( -type d -exec test -e '{}/.git' \; -printf 'G\t%p\n' -prune \) -o \
    \( -type f -printf 'F\t%s\t%p\n' \) -o \( ! -type d -printf 'L\t%p\n' \) > "$tmp/all"
  : > "$tmp/cand"; : > "$tmp/sniff"
  while IFS=$'\t' read -r kind a b; do
    case "$kind" in
      D) printf -- '-\tdir:pruned\t%s/\n' "$a" ;;
      G) printf -- '-\tdir:nested-repo\t%s/\n' "$a" ;;
      L) printf -- '-\tnot-a-regular-file\t%s\n' "$a" ;;
      F)
        hard_reason "$a" "$b"; [ -z "$REASON" ] && soft_reason "$b"
        if [ -n "$REASON" ]; then printf -- '-\t%s\t%s\n' "$REASON" "$b"; continue; fi
        printf '%s\t%s\n' "$a" "$b" >> "$tmp/cand"
        ext=; case "${b##*/}" in *.*) ext=${b##*.}; ext=${ext,,} ;; esac
        if [ "$a" -gt 0 ] && ! [[ $ext =~ $TEXT_EXT ]]; then echo "$b" >> "$tmp/sniff"; fi ;;
    esac
  done < "$tmp/all"
  # Files with an unknown extension: kept only when grep sees text (no NUL byte).
  cut -f2 "$tmp/cand" > "$tmp/paths"
  : > "$tmp/text"
  [ -s "$tmp/sniff" ] && LC_ALL=C xargs -d '\n' grep -lI '' -- < "$tmp/sniff" > "$tmp/text" 2>/dev/null
  # Any kept file whose content looks like a key or token.
  : > "$tmp/secret"
  [ -s "$tmp/paths" ] && LC_ALL=C xargs -d '\n' grep -lE -e "$SECRET_RE" -- < "$tmp/paths" > "$tmp/secret" 2>/dev/null
  awk -F'\t' -v S="$tmp/sniff" -v T="$tmp/text" -v X="$tmp/secret" '
    BEGIN { while ((getline l < S) > 0) sn[l] = 1; while ((getline l < T) > 0) tx[l] = 1; while ((getline l < X) > 0) se[l] = 1 }
    { if ($2 in se) print "-\tsecret-like-content\t" $2
      else if (($2 in sn) && !($2 in tx)) print "-\tbinary\t" $2
      else print "+\t" $1 "\t" $2 }' "$tmp/cand"
  rm -rf "$tmp"
}

# check_tree <dir>: every file under <dir> (its own top-level .git aside) against the hard exclusions and
# the content scan. Prints one line per offender; returns 1 when there is one.
check_tree() {
  local dir=$1 bad=0 kind a b tmp
  tmp=$(mktemp -d) || return 1
  (cd "$dir" && find . -mindepth 1 \( -path ./.git -prune \) -o \( -name .git -printf 'G\t%P\n' -prune \) -o \
     \( -type f -printf 'F\t%s\t%P\n' \) -o \( ! -type d -printf 'L\t%P\n' \)) > "$tmp/all"
  : > "$tmp/paths"
  while IFS=$'\t' read -r kind a b; do
    case "$kind" in
      G) echo "OFFENDER nested-repo $a"; bad=1 ;;
      L) echo "OFFENDER not-a-regular-file $a"; bad=1 ;;
      F) hard_reason "$a" "$b"
         if [ -n "$REASON" ]; then echo "OFFENDER $REASON $b"; bad=1; else echo "$b" >> "$tmp/paths"; fi ;;
    esac
  done < "$tmp/all"
  if [ -s "$tmp/paths" ]; then
    (cd "$dir" && LC_ALL=C xargs -d '\n' grep -lE -e "$SECRET_RE" -- < "$tmp/paths" 2>/dev/null) > "$tmp/secret"
    while read -r b; do echo "OFFENDER secret-like-content $b"; bad=1; done < "$tmp/secret"
  fi
  rm -rf "$tmp"
  return $bad
}

human() { awk -v n="$1" 'BEGIN { if (n >= 1048576) printf "%.1f MB", n/1048576; else printf "%.0f KB", n/1024 }'; }

if [ "$MODE" = check ]; then check_tree "$CHECK_DIR"; exit $?; fi
if [ "$MODE" = list ]; then (cd "$SCRATCH" && select_files); exit $?; fi

URL=$(git -C "$REPO" remote get-url offsite 2>/dev/null) \
  || { echo "ERROR: no git remote named offsite in $REPO; add the private off-site repo as remote offsite first" >&2; exit 2; }
ORIGIN=$(git -C "$REPO" remote get-url origin 2>/dev/null || true)
[ -n "$ORIGIN" ] && [ "$URL" = "$ORIGIN" ] && { echo "REFUSED: remote offsite has origin's URL" >&2; exit 2; }

# ---- 1. Branches ----
echo "== branches -> offsite"
PUSH_ARGS=(--porcelain); [ "$MODE" = dry ] && PUSH_ARGS+=(--dry-run)
out=$(git -C "$REPO" push "${PUSH_ARGS[@]}" offsite --all 2>&1); prc=$?
n_new=0; n_ff=0; n_up=0; n_rej=0; rejected=""
while IFS= read -r line; do
  case "$line" in
    "*"$'\t'*) n_new=$((n_new + 1)); echo "  new      ${line#*$'\t'}" ;;
    " "$'\t'*) n_ff=$((n_ff + 1)); echo "  moved    ${line#*$'\t'}" ;;
    "="$'\t'*) n_up=$((n_up + 1)) ;;
    "!"$'\t'*) n_rej=$((n_rej + 1)); rejected="$rejected ${line#*$'\t'}"; echo "  REJECTED ${line#*$'\t'}" ;;
  esac
done <<< "$out"
if [ $((n_new + n_ff + n_up + n_rej)) -eq 0 ] && [ $prc -ne 0 ]; then
  echo "  ERROR: branch push failed:"; echo "$out" | grep -v '^To ' | sed 's/^/    /'
fi
BR_SUM="branches: $n_new new, $n_ff moved, $n_up up to date, $n_rej rejected"

# ---- 2. State files ----
echo "== state files -> offsite $BRANCH"
SEL=$(mktemp) || exit 1
trap 'rm -f "$SEL" "$SEL".*' EXIT
(cd "$SCRATCH" && select_files) > "$SEL"
grep '^+' "$SEL" | cut -f3 > "$SEL.keep"
n_keep=$(wc -l < "$SEL.keep" | tr -d ' ')
bytes=$(grep '^+' "$SEL" | awk -F'\t' '{ s += $2 } END { print s + 0 }')
grep '^-' "$SEL" | grep -v $'\tdir:pruned\t' | grep -v $'\trent-run-file\t' > "$SEL.excl" || true
for r in "$SCRATCH"/handoff "$SCRATCH"/manager "$SCRATCH"/e[0-9]* "$SCRATCH"/rent; do
  [ -d "$r" ] || continue; r=${r##*/}
  awk -F'\t' -v r="$r/" 'index($3, r) == 1 { n++; s += $2 } END { printf "  %-10s %5d files %10d bytes\n", r, n, s }' <(grep '^+' "$SEL")
done
echo "  excluded inside the allow-list (reason counts; rent run files and pruned cache/code folders not counted):"
cut -f2 "$SEL.excl" | sort | uniq -c | sed 's/^/   /'

if [ "$MODE" = dry ]; then
  echo "  --- files that would be mirrored (size path) ---"
  grep '^+' "$SEL" | cut -f2,3 | sed 's/^/  /'
  echo "  --- excluded files and folders inside the allow-list (reason path) ---"
  cut -f2,3 "$SEL.excl" | sed 's/^/  /'
  if [ -d "$STATE/.git" ] && git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
    git -C "$STATE" ls-tree -r HEAD | awk '{ split($0, t, "\t"); split(t[1], m, " "); print m[3] "\t" t[2] }' | sort -t$'\t' -k2 > "$SEL.old"
    (cd "$SCRATCH" && git hash-object --no-filters --stdin-paths < "$SEL.keep") | paste - "$SEL.keep" | sort -t$'\t' -k2 > "$SEL.new"
    n_add=$(join -t$'\t' -1 2 -2 2 -v 2 "$SEL.old" "$SEL.new" | wc -l | tr -d ' ')
    n_del=$(join -t$'\t' -1 2 -2 2 -v 1 "$SEL.old" "$SEL.new" | wc -l | tr -d ' ')
    n_chg=$(join -t$'\t' -1 2 -2 2 "$SEL.old" "$SEL.new" | awk -F'\t' '$2 != $3' | wc -l | tr -d ' ')
    echo "  vs last state commit: $n_add new, $n_chg changed, $n_del deleted"
  else
    echo "  no state clone yet: all $n_keep files are new"
  fi
  echo "== DRY RUN, nothing changed. $BR_SUM; state: $n_keep files, $(human "$bytes")"
  exit 0
fi

# Real run: clone (once), mirror, re-check, commit, push.
if [ ! -d "$STATE/.git" ]; then
  mkdir -p "$STATE" && git init -q "$STATE" && git -C "$STATE" symbolic-ref HEAD "refs/heads/$BRANCH" \
    && git -C "$STATE" remote add offsite "$URL" || { echo "ERROR: cannot create $STATE" >&2; exit 1; }
fi
git -C "$STATE" remote set-url offsite "$URL"
git -C "$STATE" config core.autocrlf false; git -C "$STATE" config core.quotepath off
if ! git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  # A fresh clone adopts the remote branch, if there is one, so history continues.
  if git -C "$STATE" fetch -q offsite "$BRANCH" 2>/dev/null; then git -C "$STATE" reset -q FETCH_HEAD; fi
fi
# Delete what is no longer selected, then copy the selection in.
(cd "$STATE" && find . -mindepth 1 -path ./.git -prune -o -type f -printf '%P\n') | sort > "$SEL.have"
sort "$SEL.keep" > "$SEL.want"
comm -23 "$SEL.have" "$SEL.want" | (cd "$STATE" && xargs -d '\n' -r rm -f --)
find "$STATE" -mindepth 1 -depth -type d -empty ! -path "$STATE/.git" ! -path "$STATE/.git/*" -delete
if [ -s "$SEL.keep" ]; then
  tar -C "$SCRATCH" -cf - --no-recursion -T "$SEL.keep" | tar -C "$STATE" -xf - \
    || { echo "ERROR: copy into $STATE failed" >&2; exit 1; }
fi
git -C "$STATE" add -A --force
echo "  re-check of the whole state tree before commit:"
if ! check_tree "$STATE"; then
  git -C "$STATE" reset -q
  echo "ABORTED: offender in the state tree; nothing committed or pushed. $BR_SUM"; exit 3
fi
echo "  clean"
commit="no change"
if ! git -C "$STATE" diff --cached --quiet 2>/dev/null || ! git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  ns=$(git -C "$STATE" diff --cached --name-status 2>/dev/null || git -C "$STATE" ls-files | sed 's/^/A\t/')
  na=$(echo "$ns" | grep -c '^A'); nm=$(echo "$ns" | grep -c '^M'); nd=$(echo "$ns" | grep -c '^D')
  if [ "$((na + nm + nd))" -gt 0 ]; then
    git -C "$STATE" commit -q -m "state $(date '+%Y-%m-%d %H:%M'): $n_keep files, $(human "$bytes"); +$na ~$nm -$nd" \
      || { echo "ERROR: commit failed. $BR_SUM" >&2; exit 1; }
    commit="$(git -C "$STATE" rev-parse --short HEAD) (+$na ~$nm -$nd)"
  fi
fi
if git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  if pout=$(git -C "$STATE" push --porcelain offsite "$BRANCH:$BRANCH" 2>&1); then push="pushed"
  else push="PUSH FAILED: $(echo "$pout" | grep -v '^To ' | tr '\n' ' ')"; fi
else push="nothing to push"; fi
echo "== $BR_SUM${rejected:+ (rejected:$rejected)}"
echo "== state: $n_keep files, $(human "$bytes"); commit $commit; $push"
