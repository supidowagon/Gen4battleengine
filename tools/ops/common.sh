# Sourced by the PC-side scripts in tools/ops. Sets REPO (main checkout), SCRATCH, RENT, RPC
# and the table upload rules. Run from any checkout or worktree of the repo.
# REPO: $NUZLINE_REPO, else the main checkout (parent of `git rev-parse --git-common-dir`).
_ops=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_common=$(git -C "$_ops" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) \
  || { echo "ERROR: $_ops is not inside a git checkout; set NUZLINE_REPO and NUZLINE_SCRATCH" >&2; exit 2; }
# NUZLINE_REPO / NUZLINE_SCRATCH may be Windows paths; both are normalised by cd + pwd.
if [ -n "${NUZLINE_REPO:-}" ]; then
  REPO=$(cd "$NUZLINE_REPO" 2>/dev/null && pwd) || { echo "ERROR: NUZLINE_REPO=$NUZLINE_REPO: no such folder" >&2; exit 2; }
else REPO=$(cd "$_common/.." && pwd); fi
if [ -n "${NUZLINE_SCRATCH:-}" ]; then
  SCRATCH=$(cd "$NUZLINE_SCRATCH" 2>/dev/null && pwd) || { echo "ERROR: NUZLINE_SCRATCH=$NUZLINE_SCRATCH: no such folder" >&2; exit 2; }
else SCRATCH="$REPO/_scratch"; fi
RENT="$SCRATCH/rent"
RPC="${RUNPODCTL:-$RENT/bin/runpodctl-windows-amd64.exe}"
[ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || { echo "ERROR: NUZLINE_REPO=$REPO is not a checkout" >&2; exit 2; }

# Built tables that may go to a pod: the built game folders, the vanilla decomp extraction,
# the resolved Kaizo intermediates (top level) and the Kaizo ROM extraction. Nothing else.
TABLE_ALLOW='^data/(gen4_platinum|gen4_platinum_kaizo|intermediate)/.*\.(json|npz)$|^data/intermediate_kaizo/[^/]+\.json$|^data/intermediate_kaizo/rom/[^/]+\.json$'
# Never: the calc's dumped globals and normalised extraction, the docs-sheet extraction.
TABLE_DENY='^data/intermediate_kaizo/(pkcalc|docs)/|(^|/)backups?/'

# Run names: [a-z0-9_-], used once, never containing "backup".
check_run_name() {
  echo "$1" | grep -qE '^[a-z0-9][a-z0-9_-]*$' || { echo "REFUSED: run name must be [a-z0-9_-]" >&2; exit 2; }
  echo "$1" | grep -qi 'backup' && { echo "REFUSED: run name contains backup" >&2; exit 2; }
  return 0
}

# Windows path quoted for PowerShell.
psq() { printf "'%s'" "$(cygpath -w "$1" | sed "s/'/''/g")"; }

# pod_bin <checkout> <commit> <dir>: tools/ops/day/pod's *.sh, weights.py and weights.txt as
# committed at <commit> of <checkout>, LF, into <dir> (the payload's bin/).
pod_bin() { local s; s=$(mktemp -d "$RENT/podsrc.XXXXXX") || return 1; mkdir -p "$3"
  git -C "$1" -c core.autocrlf=false archive --format=tar "$2" tools/ops/day/pod | tar -xf - -C "$s" || { rm -rf "$s"; return 1; }
  for f in "$s"/tools/ops/day/pod/*.sh "$s"/tools/ops/day/pod/weights.py "$s"/tools/ops/day/pod/weights.txt; do
    tr -d '\r' < "$f" > "$3/$(basename "$f")" || { rm -rf "$s"; return 1; }; done
  rm -rf "$s"; }
# pod_sum <dir>: POD_SCRIPTS, sha256 over the sha256sum lines of <dir>'s *.sh and weights.py (the
# files that decide gate behaviour) sorted by name. pod_wsum <dir>: POD_WEIGHTS, sha256 of weights.txt.
pod_sum() { (cd "$1" && LC_ALL=C ls -- *.sh weights.py | LC_ALL=C sort | while read -r f; do sha256sum "$f"; done | sha256sum | cut -d' ' -f1); }
pod_wsum() { sha256sum "$1/weights.txt" | cut -d' ' -f1; }
# pod_main_check <checkout>: the pod scripts a payload carries come from the checkout that runs
# pack.sh or podcheck_body.sh (its HEAD), not from the commit under test; refuse (exit 3) unless
# that HEAD's tools/ops/day/pod equals local main's. PACK_OFF_MAIN_POD=1: a road test of an ops
# branch only.
pod_main_check() { [ "${PACK_OFF_MAIN_POD:-}" = 1 ] && return 0
  git -C "$1" diff --quiet main HEAD -- tools/ops/day/pod 2>/dev/null && return 0
  echo "REFUSED: tools/ops/day/pod at this checkout's HEAD differs from main's (or there is no main)."
  echo "The pod scripts come from the checkout that runs this script, not from the commit under test:"
  echo "run it from a checkout at main (the main checkout). PACK_OFF_MAIN_POD=1 overrides, for a road test of an ops branch only."
  git -C "$1" diff --stat main HEAD -- tools/ops/day/pod 2>/dev/null; exit 3; }
