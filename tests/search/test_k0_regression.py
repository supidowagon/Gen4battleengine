"""The ``k = 0`` numbers of Stage A and Stage B at the Roark root stay statistically
unchanged by the faint-budget work (spec section 4); ``tests/search/k0_baseline.py``
recorded them before it."""

import json
from math import sqrt

import jax
import pytest

from nuzline.search.stage_a import rank_actions
from nuzline.search.stage_b import solve
from tests.engine.helpers import roark_real_state
from tests.search.k0_baseline import BASELINE, STAGE_A_N, STAGE_B_SETTINGS


def _recorded() -> dict:
    if not BASELINE.exists():
        pytest.skip(f"{BASELINE} missing (run uv run python -m tests.search.k0_baseline)")
    return json.loads(BASELINE.read_text())


@pytest.mark.slow
def test_stage_a_k0_unchanged(tables, names):
    old = _recorded()["stage_a"]
    assert old["n"] == STAGE_A_N
    state = roark_real_state(tables, names)
    ranking = rank_actions(tables, names, state, STAGE_A_N, jax.random.PRNGKey(0))
    new = {str(r.action): round(r.p_win0 * STAGE_A_N) for r in ranking}
    assert set(new) == set(old["actions"])
    for action, wins_new in new.items():
        wins_old = old["actions"][action]["wins0"]
        p = (wins_old + wins_new) / (2 * STAGE_A_N)
        bound = 4 * sqrt(2 * p * (1 - p) / STAGE_A_N) + 1 / STAGE_A_N
        assert abs(wins_new - wins_old) / STAGE_A_N <= bound, action


@pytest.mark.slow
def test_stage_b_k0_unchanged(tables, names):
    old = _recorded()["stage_b"]["actions"]
    state = roark_real_state(tables, names)
    result = solve(tables, names, state, jax.random.PRNGKey(0), STAGE_B_SETTINGS)
    new = {str(r.action): r for r in result.ranking}
    assert set(new) == set(old)
    for action, row in new.items():
        half_old = (old[action]["ci_high"] - old[action]["ci_low"]) / 2
        half_new = (row.ci_high - row.ci_low) / 2
        assert abs(row.estimate - old[action]["estimate"]) <= half_old + half_new + 1e-9, action
