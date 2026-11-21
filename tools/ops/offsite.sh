#!/usr/bin/env bash
# Off-site copy to the private git remote named `offsite` (docs/process.md section 12, Off-site copy).
#   bash tools/ops/offsite.sh            push every local branch, mirror the state files, push them
#   bash tools/ops/offsite.sh --dry-run  print what would be pushed and copied; change nothing
#   bash tools/ops/offsite.sh --list     print the selection under the scratch folder (+ kept, - excluded); no git
#   bash tools/ops/offsite.sh --check D  hard-exclusion check of every file under folder D; exit 1 on an offender
# 1. Branches: `git push offsite --all` (no force, no delete); a rejected branch is reported, the run goes on.
# 2. State: the allow-listed git-ignored files under _scratch/ (paths relative to _scratch/) are mirrored into
#    the clone _scratch/offsite/state/ (branch scratch-state, no common history with the code), checked
#    again, committed when something changed and pushed (no force). Deletions are mirrored inside the clone.
# Guards (exit 2): every push URL of `offsite` (after insteadOf/pushInsteadOf) must name a repo other than
# every URL of `origin` (compared as lower-case host/owner/repo) and `gh repo view` must report it private;
# remote.offsite.mirror and remote.offsite.push must be unset; no local branch may be named scratch-state.
# NUZLINE_OFFSITE_TEST_LOCAL=1 lets tests use local-path remotes (it never skips the check for a network URL).
# Exit: 0 done, 1 error, 2 refused, 3 offender in the state tree (nothing committed), 4 a branch rejected or
# a push failed.
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
OFF="$SCRATCH/offsite"
STATE="$OFF/state"
MAX_FILE=2097152   # 2 MB: any file over this is excluded
MAX_LOG=1048576    # 1 MB: any *.log over this is excluded
# Top-level folders of _scratch that hold state files. A folder that matches only in other letter case
# is reported (dir:not-lower-case), never entered.
ROOT_RE='^(handoff|manager|e[0-9][^/]*|rent)$'
# Folders never entered, at any depth (name, case-insensitive): caches, envs, code and table copies,
# emulator folders, backups.
PRUNE_DIRS='^(\.git|\.venv|venv|node_modules|__pycache__|\.pytest_cache|\.hypothesis|\.ruff_cache|\.mypy_cache|\.cache|[^/]*cache[^/]*|external|out|src|tests|tools|docs|data|intermediate[^/]*|gen4_platinum[^/]*|emu|shots|roms|stateslots|battery|home|bin|\.runpod|backup[^/]*)$'

# ci <text>: <text> with every ASCII letter turned into a [xX] class (case-insensitive part of an ERE).
ci() { local s=$1 o= c i; for ((i = 0; i < ${#s}; i++)); do c=${s:i:1}
  case "$c" in [a-zA-Z]) o+="[${c,,}${c^^}]" ;; *) o+=$c ;; esac; done; printf '%s' "$o"; }
# Content that marks a key or token (extended regex, grep -E). Keyword parts are case-insensitive.
Q="[\"']?"
SECRET_RE="-----BEGIN ([A-Z0-9]+ )*PRIVATE KEY|rpa_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
SECRET_RE+="|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}"
SECRET_RE+="|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
SECRET_RE+="|[A-Za-z][A-Za-z0-9+.-]*://[^/@:[:space:]]+:[^/@[:space:]]+@"
SECRET_RE+="|$(ci bearer)[[:space:]]+[A-Za-z0-9._~+/=-]{16,}"
SECRET_RE+="|$(ci '(api[_-]?key|secret|token|access[_-]?key|private[_-]?key)')$Q[[:space:]]*[=:][[:space:]]*$Q[A-Za-z0-9_+/=-]{16,}"
SECRET_RE+="|$(ci '(password|passwd|pwd)')$Q[[:space:]]*[=:][[:space:]]*$Q[^[:space:]\"']{8,}"

# Temp files: under _scratch/offsite/tmp/, removed on exit.
mkdir -p "$OFF/tmp" && TMPD=$(mktemp -d "$OFF/tmp/run.XXXXXX") || { echo "ERROR: cannot create a temp folder under $OFF/tmp" >&2; exit 1; }
trap 'rm -rf "$TMPD"' EXIT

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

# grep_files <out> <grep args...> < path list: grep -l over the listed files (relative to the cwd) into
# <out>. Fails closed: returns 1 (and prints grep's errors) on any grep error or stderr output.
grep_files() {
  local out=$1 rc; shift
  LC_ALL=C xargs -d '\n' -r grep "$@" -- > "$out" 2> "$out.err"; rc=$?
  if [ -s "$out.err" ] || { [ $rc -ne 0 ] && [ $rc -ne 123 ]; }; then
    echo "ERROR: grep failed (status $rc):" >&2; sed 's/^/  /' "$out.err" >&2; return 1
  fi
  return 0
}

# scan <tmp> <path list>: in the cwd, writes <tmp>/binary (non-empty files with a NUL byte, whatever
# their extension) and <tmp>/secret (files whose content looks like a key or token). Returns 1 on a
# grep error.
scan() {
  local t=$1 list=$2
  grep_files "$t/text" -lI '' < "$list" || return 1
  LC_ALL=C sort -u "$t/text" > "$t/text.s"
  # Non-empty files that grep does not see as text.
  while IFS= read -r p; do [ -s "$p" ] && printf '%s\n' "$p"; done < "$list" | LC_ALL=C sort -u > "$t/nonempty"
  LC_ALL=C comm -23 "$t/nonempty" "$t/text.s" > "$t/binary"
  grep_files "$t/secret" -lE -e "$SECRET_RE" < "$list" || return 1
}

# select_files: run in $SCRATCH. Prints "+<TAB>size<TAB>path" for each kept file and
# "-<TAB>reason<TAB>path" for each excluded file or pruned folder (folders end in /). Returns 1 on an error.
select_files() {
  local roots=() r kind a b t
  for r in *; do
    [ -d "$r" ] || continue
    if [[ $r =~ $ROOT_RE ]]; then roots+=("$r")
    elif [[ ${r,,} =~ $ROOT_RE ]]; then printf -- '-\tdir:not-lower-case\t%s/\n' "$r"; fi
  done
  [ ${#roots[@]} -gt 0 ] || return 0
  t=$(mktemp -d "$TMPD/sel.XXXXXX") || return 1
  find "${roots[@]}" -regextype posix-extended \
    \( -type d -iregex "(.*/)?${PRUNE_DIRS#^}" -printf 'D\t%p\n' -prune \) -o \
    \( -type d -exec test -e '{}/.git' \; -printf 'G\t%p\n' -prune \) -o \
    \( -type f -printf 'F\t%s\t%p\n' \) -o \( ! -type d -printf 'L\t%p\n' \) > "$t/all" || return 1
  : > "$t/cand"
  while IFS=$'\t' read -r kind a b; do
    case "$kind" in
      D) printf -- '-\tdir:pruned\t%s/\n' "$a" ;;
      G) printf -- '-\tdir:nested-repo\t%s/\n' "$a" ;;
      L) printf -- '-\tnot-a-regular-file\t%s\n' "$a" ;;
      F)
        hard_reason "$a" "$b"; [ -z "$REASON" ] && soft_reason "$b"
        if [ -n "$REASON" ]; then printf -- '-\t%s\t%s\n' "$REASON" "$b"; continue; fi
        printf '%s\t%s\n' "$a" "$b" >> "$t/cand" ;;
    esac
  done < "$t/all"
  cut -f2 "$t/cand" > "$t/paths"
  scan "$t" "$t/paths" || return 1
  awk -F'\t' -v N="$t/binary" -v X="$t/secret" '
    BEGIN { while ((getline l < N) > 0) bn[l] = 1; while ((getline l < X) > 0) se[l] = 1 }
    { if ($2 in se) print "-\tsecret-like-content\t" $2
      else if ($2 in bn) print "-\tbinary\t" $2
      else print "+\t" $1 "\t" $2 }' "$t/cand"
}

# check_tree <dir>: every file under <dir> (its own top-level .git aside) against the hard exclusions,
# the NUL-byte check and the content scan. Prints one line per offender; returns 1 when there is one
# or when the scan cannot run.
check_tree() {
  local dir=$1 bad=0 kind a b t
  t=$(mktemp -d "$TMPD/chk.XXXXXX") || return 1
  (cd "$dir" && find . -mindepth 1 \( -path ./.git -prune \) -o \( -name .git -printf 'G\t%P\n' -prune \) -o \
     \( -type f -printf 'F\t%s\t%P\n' \) -o \( ! -type d -printf 'L\t%P\n' \)) > "$t/all" \
    || { echo "OFFENDER unreadable-tree $dir"; return 1; }
  : > "$t/paths"
  while IFS=$'\t' read -r kind a b; do
    case "$kind" in
      G) echo "OFFENDER nested-repo $a"; bad=1 ;;
      L) echo "OFFENDER not-a-regular-file $a"; bad=1 ;;
      F) hard_reason "$a" "$b"
         if [ -n "$REASON" ]; then echo "OFFENDER $REASON $b"; bad=1; else echo "$b" >> "$t/paths"; fi ;;
    esac
  done < "$t/all"
  if [ -s "$t/paths" ]; then
    if (cd "$dir" && scan "$t" "$t/paths"); then
      while IFS= read -r b; do echo "OFFENDER binary $b"; bad=1; done < "$t/binary"
      while IFS= read -r b; do echo "OFFENDER secret-like-content $b"; bad=1; done < "$t/secret"
    else
      echo "OFFENDER scan-failed (grep error; see above)"; bad=1
    fi
  fi
  return $bad
}

human() { awk -v n="$1" 'BEGIN { if (n >= 1048576) printf "%.1f MB", n/1048576; else printf "%.0f KB", n/1024 }'; }

if [ "$MODE" = check ]; then check_tree "$CHECK_DIR"; exit $?; fi
if [ "$MODE" = list ]; then (cd "$SCRATCH" && select_files); exit $?; fi

# ---- Guards ----
# url_key <url>: sets KEY, the remote's identity in lower case (host/owner/repo for a network URL,
# local:<absolute folder> for a local path; trailing / and .git dropped), and LOCAL (1 for a local path).
url_key() {
  local u=$1 rest host path
  LOCAL=0; u=${u%"${u##*[![:space:]]}"}; u=${u#"${u%%[![:space:]]*}"}
  case "$u" in
    file://*) LOCAL=1; path=${u#file://} ;;
    [A-Za-z]:[\\/]*|/*|.|..|./*|../*) LOCAL=1; path=$u ;;
    *://*) rest=${u#*://}; host=${rest%%/*}; path=${rest#"$host"} ;;
    *:*) host=${u%%:*}; path=${u#*:}
         case "$host" in */*) LOCAL=1; path=$u ;; esac ;;  # a colon after a slash: a local path
    *) LOCAL=1; path=$u ;;
  esac
  path=${path//\\//}
  if [ "$LOCAL" = 1 ]; then
    case "$path" in [A-Za-z]:/*) path=$(cygpath -u "$path") ;; esac
    path=$( (cd "$REPO" && cd "$path" 2>/dev/null && pwd -P) || printf '%s' "$path")
  else
    host=${host##*@}; host=${host%%:*}; host=${host,,}
    case "$host" in ssh.github.com|www.github.com) host=github.com ;; esac
  fi
  path=${path,,}
  while :; do case "$path" in */) path=${path%/} ;; *.git) path=${path%.git} ;; *) break ;; esac; done
  while [[ $path == *//* ]]; do path=${path//\/\//\/}; done
  if [ "$LOCAL" = 1 ]; then KEY="local:$path"; else path=${path#/}; KEY="$host/$path"; fi
}

# check_private <url>: after url_key; returns 1 (with a REFUSED line) unless gh confirms the repo private.
check_private() {
  local h o r x ans
  if [ "$LOCAL" = 1 ]; then
    [ "${NUZLINE_OFFSITE_TEST_LOCAL:-}" = 1 ] && return 0
    echo "REFUSED: offsite push URL $1 is a local path, not a private GitHub repo" >&2; return 1
  fi
  IFS=/ read -r h o r x <<< "$KEY"
  if [ -z "$o" ] || [ -z "$r" ] || [ -n "$x" ]; then
    echo "REFUSED: cannot read owner/repo from offsite push URL $1" >&2; return 1; fi
  command -v gh > /dev/null 2>&1 \
    || { echo "REFUSED: gh (GitHub CLI) is missing; cannot confirm that $h/$o/$r is private" >&2; return 1; }
  ans=$(gh repo view "$h/$o/$r" --json isPrivate --jq .isPrivate 2>&1) \
    || { echo "REFUSED: gh repo view $h/$o/$r failed (gh missing a login, or no access): $(printf '%s' "$ans" | head -c 300)" >&2; return 1; }
  [ "$ans" = true ] || { echo "REFUSED: $h/$o/$r is not confirmed private (gh: isPrivate=$ans)" >&2; return 1; }
}

# guard_remote <git dir>: refuses (returns 1) a mirror or push-refspec setting on offsite, a push URL
# that names origin's repo, or one not confirmed private. Sets PUSH_URLS to the checked push URLs.
guard_remote() {
  local dir=$1 u o v
  git -C "$dir" remote | grep -qx offsite \
    || { echo "ERROR: no git remote named offsite in $dir; add the private off-site repo as remote offsite first" >&2; return 1; }
  v=$(git -C "$dir" config --get-all remote.offsite.mirror)
  [ -z "$v" ] || { echo "REFUSED: remote.offsite.mirror is set ($v) in $dir: a push would force and delete; git -C '$dir' config --unset-all remote.offsite.mirror" >&2; return 1; }
  v=$(git -C "$dir" config --get-all remote.offsite.push)
  [ -z "$v" ] || { echo "REFUSED: remote.offsite.push is set ($v) in $dir; git -C '$dir' config --unset-all remote.offsite.push" >&2; return 1; }
  mapfile -t PUSH_URLS < <(git -C "$dir" remote get-url --push --all offsite)
  [ ${#PUSH_URLS[@]} -gt 0 ] || { echo "ERROR: remote offsite in $dir has no URL" >&2; return 1; }
  for u in "${PUSH_URLS[@]}"; do
    url_key "$u"
    for o in "${ORIGIN_KEYS[@]}"; do
      [ "$KEY" = "$o" ] && { echo "REFUSED: offsite push URL $u is origin's repo ($KEY); never push the off-site copy to origin" >&2; return 1; }
    done
    check_private "$u" || return 1
  done
  return 0
}

ORIGIN_KEYS=()
while IFS= read -r u; do [ -n "$u" ] && { url_key "$u"; ORIGIN_KEYS+=("$KEY"); }; done < <(
  git -C "$REPO" remote get-url --all origin 2>/dev/null; git -C "$REPO" remote get-url --push --all origin 2>/dev/null)
guard_remote "$REPO" || exit 2
git -C "$REPO" show-ref -q --verify "refs/heads/$BRANCH" \
  && { echo "REFUSED: $REPO has a local branch named $BRANCH; push --all would overwrite the state branch. Rename it first." >&2; exit 2; }
FAIL=0

# ---- 1. Branches ----
echo "== branches -> offsite"
PUSH_ARGS=(--porcelain); [ "$MODE" = dry ] && PUSH_ARGS+=(--dry-run)
out=$(git -C "$REPO" -c remote.offsite.mirror=false push "${PUSH_ARGS[@]}" offsite --all 2>&1); prc=$?
n_new=0; n_ff=0; n_up=0; n_rej=0; n_bad=0; rejected=""
while IFS= read -r line; do
  case "$line" in
    "*"$'\t'*) n_new=$((n_new + 1)); echo "  new      ${line#*$'\t'}" ;;
    " "$'\t'*) n_ff=$((n_ff + 1)); echo "  moved    ${line#*$'\t'}" ;;
    "="$'\t'*) n_up=$((n_up + 1)) ;;
    "!"$'\t'*) n_rej=$((n_rej + 1)); rejected="$rejected ${line#*$'\t'}"; echo "  REJECTED ${line#*$'\t'}" ;;
    "+"$'\t'*) n_bad=$((n_bad + 1)); echo "  ERROR: FORCED ${line#*$'\t'}" ;;
    "-"$'\t'*) n_bad=$((n_bad + 1)); echo "  ERROR: DELETED ${line#*$'\t'}" ;;
  esac
done <<< "$out"
if [ $prc -ne 0 ] && [ $n_rej -eq 0 ] || [ $n_bad -gt 0 ]; then
  echo "  ERROR: branch push failed:"; echo "$out" | grep -v '^To ' | sed 's/^/    /'
fi
[ $prc -ne 0 ] || [ $n_rej -gt 0 ] || [ $n_bad -gt 0 ] && FAIL=1
BR_SUM="branches: $n_new new, $n_ff moved, $n_up up to date, $n_rej rejected"
[ $n_bad -gt 0 ] && BR_SUM="$BR_SUM, $n_bad FORCED OR DELETED"

# ---- 2. State files ----
echo "== state files -> offsite $BRANCH"
SEL="$TMPD/sel.txt"
(cd "$SCRATCH" && select_files) > "$SEL" || { echo "ERROR: state-file selection failed. $BR_SUM" >&2; exit 1; }
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
    git -C "$STATE" -c core.quotepath=off ls-tree -r HEAD | awk '{ split($0, t, "\t"); split(t[1], m, " "); print m[3] "\t" t[2] }' | LC_ALL=C sort -t$'\t' -k2 > "$SEL.old"
    # Absolute paths (git's own form, G:/... on Windows): inside a checkout, hash-object --stdin-paths
    # reads relative paths from the checkout's root, not the cwd.
    awk -v p="$(cygpath -m "$SCRATCH" 2>/dev/null || printf '%s' "$SCRATCH")/" '{ print p $0 }' "$SEL.keep" | git hash-object --no-filters --stdin-paths > "$SEL.hash" \
      && [ "$(wc -l < "$SEL.hash")" = "$(wc -l < "$SEL.keep")" ] \
      || { echo "ERROR: cannot hash the selected files. $BR_SUM" >&2; exit 1; }
    paste "$SEL.hash" "$SEL.keep" | LC_ALL=C sort -t$'\t' -k2 > "$SEL.new"
    n_add=$(LC_ALL=C join -t$'\t' -1 2 -2 2 -v 2 "$SEL.old" "$SEL.new" | wc -l | tr -d ' ')
    n_del=$(LC_ALL=C join -t$'\t' -1 2 -2 2 -v 1 "$SEL.old" "$SEL.new" | wc -l | tr -d ' ')
    n_chg=$(LC_ALL=C join -t$'\t' -1 2 -2 2 "$SEL.old" "$SEL.new" | awk -F'\t' '$2 != $3' | wc -l | tr -d ' ')
    echo "  vs last state commit: $n_add new, $n_chg changed, $n_del deleted"
  else
    echo "  no state clone yet: all $n_keep files are new"
  fi
  echo "== DRY RUN, nothing changed. $BR_SUM${rejected:+ (rejected:$rejected)}; state: $n_keep files, $(human "$bytes")"
  [ $FAIL = 0 ] || exit 4
  exit 0
fi

# Real run: clone (once), point it at the checked URLs, mirror, re-check, commit, push.
if [ ! -d "$STATE/.git" ]; then
  mkdir -p "$STATE" && git init -q "$STATE" && git -C "$STATE" symbolic-ref HEAD "refs/heads/$BRANCH" \
    || { echo "ERROR: cannot create $STATE. $BR_SUM" >&2; exit 1; }
fi
[ -e "$STATE/.git/index.lock" ] && { echo "REFUSED: $STATE/.git/index.lock exists: a git command there died or still runs. Make sure none runs, delete the lock, run again. $BR_SUM" >&2; exit 2; }
git -C "$STATE" remote | grep -qx offsite || git -C "$STATE" remote add offsite "${PUSH_URLS[0]}" \
  || { echo "ERROR: cannot add remote offsite in $STATE. $BR_SUM" >&2; exit 1; }
# The clone pushes to exactly the push URLs checked above (checked again below in the clone).
cfg_ok=1
git -C "$STATE" remote set-url offsite "${PUSH_URLS[0]}" || cfg_ok=0
git -C "$STATE" config --unset-all remote.offsite.pushurl
for u in "${PUSH_URLS[@]}"; do git -C "$STATE" remote set-url --add --push offsite "$u" || cfg_ok=0; done
for kv in core.autocrlf=false core.quotepath=off core.longpaths=true; do git -C "$STATE" config "${kv%%=*}" "${kv#*=}" || cfg_ok=0; done
[ $cfg_ok = 1 ] || { echo "ERROR: cannot configure $STATE. $BR_SUM" >&2; exit 1; }
guard_remote "$STATE" || { echo "$BR_SUM" >&2; exit 2; }
if ! git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  # A fresh clone adopts the remote branch, if there is one, so history continues.
  if git -C "$STATE" -c remote.offsite.mirror=false fetch -q offsite "$BRANCH" 2>/dev/null; then git -C "$STATE" reset -q FETCH_HEAD; fi
fi
# Delete what is no longer selected (paths found inside the clone only), then copy the selection in.
(cd "$STATE" && find . -mindepth 1 -path ./.git -prune -o -type f -printf '%P\n') | LC_ALL=C sort > "$SEL.have"
LC_ALL=C sort "$SEL.keep" > "$SEL.want"
LC_ALL=C comm -23 "$SEL.have" "$SEL.want" | (cd "$STATE" && xargs -d '\n' -r rm -f --)
find "$STATE" -mindepth 1 -depth -type d -empty ! -path "$STATE/.git" ! -path "$STATE/.git/*" -delete
if [ -s "$SEL.keep" ]; then
  # A file that changes while tar reads it is a warning (tar exit 1); the re-check below covers the copy.
  tar -C "$SCRATCH" --warning=no-file-changed --no-recursion --verbatim-files-from -cf - -T "$SEL.keep" \
    | tar -C "$STATE" -xf -
  st=("${PIPESTATUS[@]}")
  if [ "${st[0]}" -gt 1 ] || [ "${st[1]}" -ne 0 ]; then
    echo "ERROR: copy into $STATE failed (tar ${st[0]}/${st[1]}). $BR_SUM" >&2; exit 1; fi
fi
git -C "$STATE" add -A --force || { echo "ERROR: git add in $STATE failed. $BR_SUM" >&2; exit 1; }
echo "  re-check of the whole state tree before commit:"
if ! check_tree "$STATE"; then
  git -C "$STATE" reset -q
  echo "ABORTED: offender in the state tree; nothing committed or pushed. $BR_SUM"; exit 3
fi
echo "  clean"
commit="no change"
if ! git -C "$STATE" diff --cached --quiet 2>/dev/null || ! git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  ns=$(git -C "$STATE" diff --cached --no-renames --name-status 2>/dev/null || git -C "$STATE" ls-files | sed 's/^/A\t/')
  na=$(echo "$ns" | grep -c '^A'); nm=$(echo "$ns" | grep -c '^M'); nd=$(echo "$ns" | grep -c '^D')
  if [ "$((na + nm + nd))" -gt 0 ]; then
    git -C "$STATE" commit -q -m "state $(date '+%Y-%m-%d %H:%M'): $n_keep files, $(human "$bytes"); +$na ~$nm -$nd" \
      || { echo "ERROR: commit failed. $BR_SUM" >&2; exit 1; }
    commit="$(git -C "$STATE" rev-parse --short HEAD) (+$na ~$nm -$nd)"
  fi
fi
if git -C "$STATE" rev-parse -q --verify HEAD >/dev/null; then
  pout=$(git -C "$STATE" -c remote.offsite.mirror=false push --porcelain offsite "refs/heads/$BRANCH:refs/heads/$BRANCH" 2>&1); src=$?
  if [ $src -eq 0 ] && ! grep -q $'^[+-]\t' <<< "$pout"; then push="pushed"
  else push="PUSH FAILED: $(echo "$pout" | grep -v '^To ' | tr '\n' ' ')"; FAIL=1; fi
else push="nothing to push"; fi
echo "== $BR_SUM${rejected:+ (rejected:$rejected)}"
echo "== state: $n_keep files, $(human "$bytes"); commit $commit; $push"
[ $FAIL = 0 ] || exit 4
exit 0
