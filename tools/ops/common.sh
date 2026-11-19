# Sourced by the PC-side scripts in tools/ops. Sets REPO (main checkout), SCRATCH, RENT, RPC
# and the table upload rules. Run from any checkout or worktree of the repo.
# REPO: $NUZLINE_REPO, else the main checkout (parent of `git rev-parse --git-common-dir`).
_ops=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_common=$(git -C "$_ops" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) \
  || { echo "ERROR: $_ops is not inside a git checkout; set NUZLINE_REPO and NUZLINE_SCRATCH" >&2; exit 2; }
REPO="${NUZLINE_REPO:-$(cd "$_common/.." && pwd)}"
SCRATCH="${NUZLINE_SCRATCH:-$REPO/_scratch}"
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
