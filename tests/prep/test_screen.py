"""The all-rung screen and the prep tie rule (plan 4c Task 6)."""

import jax
import numpy as np

from nuzline.prep.screen import EMPTY, Scored, ScreenStats, prep_order, scored, screen
from nuzline.search.leaves import evaluate_leaves
from nuzline.search.stage_a import wilson
from tests.engine.helpers import roark_real_state
from tests.search.fixtures import switch_refusal_root
from tests.search.test_expand import batch

KEY = jax.random.PRNGKey(5)


def test_screen_pools_both_halves(tables, names):
    a = roark_real_state(tables, names)
    b = roark_real_state(tables, names, player_species="Machop")
    states = batch(a, b)
    stats = screen(tables, states, 64, KEY)
    direct = evaluate_leaves(tables, states, [64, 64], KEY)
    for s, v in zip(stats, direct, strict=True):
        assert s.n == 64
        assert s.buckets == tuple(x + y for x, y in zip(v.buckets_a, v.buckets_b, strict=True))
        assert s.deaths == v.deaths_a + v.deaths_b
        assert s.rung(3) >= s.rung(0)
        assert s.interval(0) == wilson(s.buckets[0], 64)
    merged = stats[0].merge(stats[1])
    assert merged.n == 128
    assert EMPTY.merge(stats[0]) == stats[0]


def test_screen_flags_refused(tables, names):
    states = batch(switch_refusal_root(tables, names), roark_real_state(tables, names))
    refusing, fine = screen(tables, states, 256, KEY)
    assert refusing.refused
    assert not fine.refused


def test_prep_order():
    a = Scored(0.90, 0.85, 0.95, 0.30, 0, 0.5, 0.5)
    b = Scored(0.88, 0.83, 0.93, 0.10, 3, 0.5, 0.5)
    c = Scored(0.87, 0.82, 0.92, 0.12, 0, 0.4, 0.5)
    d = Scored(0.60, 0.55, 0.65, 0.0, 0, 1.0, 1.0)
    assert prep_order([a, b, c, d]) == [2, 0, 1, 3]
    c3 = Scored(0.87, 0.82, 0.92, 0.12, 3, 0.4, 0.5)
    assert prep_order([a, b, c3, d]) == [1, 0, 2, 3]
    assert prep_order([]) == []


def test_prep_order_never_trades_a_whiteout_for_fewer_deaths():
    """Spec section 2: a whiteout is worse than any number of deaths. A solo team and the
    same lead with a bench tie on rung 0 (the bench only acts after the lead faints); the
    solo team whites out where the bench wins with deaths, so its plain expected deaths
    are lower, yet the whiteout-free team leads."""
    full = ScreenStats((1100, 0, 900, 0, 0, 0, 0, 0), 1800, 0.0, 0.0, False)
    solo = ScreenStats((1100, 0, 0, 0, 0, 0, 900, 0), 900, 0.0, 0.0, False)
    assert solo.e_deaths < full.e_deaths
    for cost_full, cost_solo in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
        rows = [scored(full, 0, cost_full), scored(solo, 0, cost_solo)]
        assert prep_order(rows)[0] == 0
    # A milder case: the lead sweeps 90 %; when it dies the bench wins, 60 % of those with a
    # second death. Plain expected deaths 0.16 against 0.10.
    full = ScreenStats((900, 40, 60, 0, 0, 0, 0, 0), 160, 0.0, 0.0, False)
    solo = ScreenStats((900, 0, 0, 0, 0, 0, 100, 0), 100, 0.0, 0.0, False)
    assert prep_order([scored(solo, 0, 0.0), scored(full, 0, 1.0)])[0] == 1


def test_rung_k_score():
    stats = ScreenStats((10, 6, 0, 0, 0, 0, 4, 0), 7, 5.0, 10.0, False)
    s = scored(stats, 1, 2.0)
    assert s.score == stats.rung(1) == 0.8
    assert (s.ci_low, s.ci_high) == stats.interval(1)
    assert (s.e_deaths, s.cost, s.hp_left, s.pp_left) == (7 / 20, 2.0, 0.25, 0.5)
    assert s.loss == 4 / 20
    assert np.isclose(sum(stats.distribution.probs), 1.0)


def test_rung_outside_ladder_raises():
    import pytest

    s = ScreenStats((1, 1, 1, 1, 1, 1, 1, 1), 0, 0.0, 0.0, False)
    assert s.rung(3) == 0.5
    for k in (4, 5, -1):
        with pytest.raises(ValueError):
            s.rung(k)
        with pytest.raises(ValueError):
            s.interval(k)
