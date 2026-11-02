"""PP-stall fights (plan 4c Task 11, spec section 10.2): Roark's Cranidos alone, with Rock Throw
on a few PP, against a tank that cannot be hurt, an attacker and a decoy.

The only deathless lines stall Rock Throw's PP with the tank (Defense Curl), then bring in an
attacker. A short stall (5 PP) is required to be found by the default funnel; longer stalls are
reported by ``benchmarks/pp_stall.py``, not tested.
"""

import jax
import pytest

from nuzline.prep.search import PrepSettings, run_prep
from nuzline.prep.setups import Setup
from nuzline.search import deaths
from tests.prep.fixtures import pp_stall_max_turns, pp_stall_problem, recorded_screens

STALL_PP = 5
TANK = 0  # candidate index of the tank; 1 and 2 are the attacker and the decoy
UNFINISHED = deaths.BUCKETS.index("unfinished")


def settings(stall_pp: int) -> PrepSettings:
    return PrepSettings(
        screen_n=16,
        team_keep=4,
        variant_rounds=0,
        finalists=2,
        max_depth=1,
        finalist_playouts=2000,
        max_team=3,
        max_turns=pp_stall_max_turns(stall_pp),
    )


@pytest.fixture(scope="module")
def stall(tables, names):
    """One run on the 5-PP fight, with every setup's accumulated screen
    stats recorded (the funnel keeps them private)."""
    prob = pp_stall_problem(tables, names, STALL_PP)
    with recorded_screens() as total:
        result = run_prep(tables, names, prob, settings(STALL_PP), jax.random.PRNGKey(0))
    return result, total


def has(setup: Setup, candidate: int) -> bool:
    return any(m.candidate == candidate for m in setup.members)


def test_short_stall_ranked_first(stall):
    result, total = stall
    chosen = result.chosen
    assert has(chosen, TANK)
    assert has(chosen, 1) or has(chosen, 2)
    rung0 = result.ladder[0]
    assert rung0.source == "plan" and rung0.setup == chosen
    assert result.plan is not None and result.plan.ranking[0].estimate >= 0.9
    first = result.plan.ranking[0].label
    assert first == ("Defense Curl" if chosen.members[0].candidate == TANK else "Switch -> Geodude")
    without = [s for s in total if not has(s, TANK)]
    assert without
    for setup in without:
        assert total[setup].interval(0)[1] < rung0.ci_low, setup


def test_tank_alone_does_not_win(stall):
    _, total = stall
    alone = [s for s in total if len(s.members) == 1 and s.members[0].candidate == TANK]
    assert len(alone) == 1
    stats = total[alone[0]]
    assert stats.n > 0
    assert stats.rung(0) == 0
    assert stats.buckets[UNFINISHED] == stats.n  # Cranidos never faints: the turn cap
