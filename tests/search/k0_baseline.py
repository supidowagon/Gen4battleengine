"""Records the ``k = 0`` Stage A / Stage B numbers on ``roark_real_state`` before the faint
budget (plan 4b) changes anything: ``uv run python -m tests.search.k0_baseline`` writes
``tests/fixtures/k0_baseline.json``, which ``test_k0_regression`` compares against."""

import json
import subprocess
from pathlib import Path

import jax

from nuzline.data.tables import Names, Tables, load_names, load_tables
from nuzline.search.stage_a import rank_actions
from nuzline.search.stage_b import SolveSettings, solve
from tests.engine.helpers import roark_real_state

BASELINE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "k0_baseline.json"
STAGE_A_N = 2000
STAGE_B_SETTINGS = SolveSettings(max_depth=1, budget_playouts=4000, n_depth0=200)


def stage_a(tables: Tables, names: Names) -> dict:
    """Stage A at the root: ``{str(action): {"wins0": int, "p_win": float}}``."""
    state = roark_real_state(tables, names)
    ranking = rank_actions(tables, names, state, STAGE_A_N, jax.random.PRNGKey(0))
    return {
        str(r.action): {"wins0": round(r.p_win0 * STAGE_A_N), "p_win": r.p_win} for r in ranking
    }


def stage_b(tables: Tables, names: Names) -> dict:
    """Stage B at the root: ``{str(action): {"estimate", "ci_low", "ci_high"}}``."""
    state = roark_real_state(tables, names)
    result = solve(tables, names, state, jax.random.PRNGKey(0), STAGE_B_SETTINGS)
    return {
        str(r.action): {"estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high}
        for r in result.ranking
    }


def baseline(tables: Tables, names: Names) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "commit": commit,
        "stage_a": {"n": STAGE_A_N, "actions": stage_a(tables, names)},
        "stage_b": {"actions": stage_b(tables, names)},
    }


if __name__ == "__main__":
    from tests.conftest import TABLES_DIR

    data = baseline(load_tables(TABLES_DIR), load_names(TABLES_DIR))
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {BASELINE} ({len(data['stage_a']['actions'])} actions)")
