# Build vanilla and Kaizo game tables in the rent worktree.
set -e
NUZLINE_SCRATCH="${NUZLINE_SCRATCH:?set NUZLINE_SCRATCH}"
export UV_CACHE_DIR="$NUZLINE_SCRATCH/rent/uvcache"
cd "$NUZLINE_SCRATCH/rent/wt"
date
uv sync --group rom
uv run python -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate
uv run python -m tools.build_tables
uv run --group rom python -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom
uv run --group rom python -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
uv run --group rom python -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
uv run --group rom python -m tools.kaizo.overrides check
uv run --group rom python -m tools.build_tables --game kaizo
date
echo BUILD_OK
