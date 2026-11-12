"""Scoring players on shared playout keys (spec
docs/superpowers/specs/2026-11-12-l1-line-player-design.md): every line plays the same
keys, the padding never leaks into the odds, and a refused playout raises."""

from functools import partial

import jax
import numpy as np
import pytest

from nuzline.ai.policy import UnsupportedAi
from nuzline.search.deaths import DeathDistribution
from nuzline.search.line import empty_line, line_playout
from nuzline.search.player import GREEDY
from nuzline.search.player_eval import (
    PlayerOdds,
    evaluate_lines,
    evaluate_players,
    line_keys,
)
from tests.engine.helpers import roark_state

N, MAX_TURNS = 64, 40
KEY = jax.random.PRNGKey(7)


def _state(tables, names, **overrides):
    return roark_state(tables, names, player_count=2, opp_count=2, **overrides)


def _rule_line():
    """``empty_line()`` with Geodude (opposing slot 0) answered by party slot 0's first
    move."""
    line = empty_line()
    use, move = line.use.copy(), line.move.copy()
    use[0], move[0] = 0, 0
    return line._replace(use=use, move=move)


@pytest.mark.slow
def test_matches_line_playout(tables, names):
    s = _state(tables, names)
    lines = [empty_line(), _rule_line()]
    odds = evaluate_lines(tables, s, lines, N, KEY, MAX_TURNS)
    keys = line_keys(KEY, N)
    play = jax.vmap(partial(line_playout, max_turns=MAX_TURNS), in_axes=(None, None, None, 0))
    for line, got in zip(lines, odds, strict=True):
        out = play(tables, s, line, keys)
        assert int(np.asarray(out.refused).sum()) == 0
        assert got.n == N
        assert got.wins_any == int(np.sum(np.asarray(out.won) == 1))
        assert got.deaths_total == int(np.asarray(out.player_faints).astype(np.int64).sum())


@pytest.mark.slow
def test_greedy_matches_the_empty_line(tables, names):
    s = _state(tables, names)
    greedy = evaluate_players(tables, s, GREEDY, [()], N, KEY, MAX_TURNS)[0]
    line = evaluate_lines(tables, s, [empty_line()], N, KEY, MAX_TURNS)[0]
    assert (greedy.wins, greedy.wins_any, greedy.deaths_total) == (
        line.wins,
        line.wins_any,
        line.deaths_total,
    )


@pytest.mark.slow
def test_same_line_same_odds(tables, names):
    s = _state(tables, names)
    rule = _rule_line()
    twice = evaluate_lines(tables, s, [rule, rule], N, KEY, MAX_TURNS)
    assert twice[0] == twice[1]
    one = evaluate_lines(tables, s, [rule], N, KEY, MAX_TURNS)[0]
    three = evaluate_lines(tables, s, [empty_line(), rule, empty_line()], N, KEY, MAX_TURNS)[1]
    assert one == three == twice[0]


@pytest.mark.slow
def test_ladder_is_monotone(tables, names):
    s = _state(tables, names)
    for odds in evaluate_lines(tables, s, [empty_line(), _rule_line()], N, KEY, MAX_TURNS):
        w = odds.wins
        assert w[0] <= w[1] <= w[2] <= w[3] <= odds.wins_any <= odds.n
        for k in range(4):
            lo, hi = odds.interval(k)
            assert lo <= odds.rung(k) <= hi
        lo, hi = odds.interval(None)
        assert lo <= odds.wins_any / odds.n <= hi
        assert 0.0 <= odds.fallback_share <= 1.0
        assert odds.distribution.n == N


@pytest.mark.slow
def test_refused_playout_raises(tables, names):
    # test_playout.py::test_refused_playout_cannot_be_counted_as_a_loss's state: the AI's
    # Levitate refusal stops every playout on its first turn.
    s = _state(tables, names, player_species="Gastly", player_ability="Levitate")
    with pytest.raises(UnsupportedAi, match="line playouts refused"):
        evaluate_lines(tables, s, [empty_line()], N, KEY, MAX_TURNS)


def _odds(wins, wins_any, deaths_total) -> PlayerOdds:
    return PlayerOdds(
        n=10,
        wins=wins,
        wins_any=wins_any,
        deaths_total=deaths_total,
        distribution=DeathDistribution.from_counts([1, 0, 0, 0, 0, 0, 0, 0]),
        mean_turns=5.0,
        fallback_share=0.0,
    )


def test_score_orders_by_rung_then_wins():
    a = _odds((3, 5, 5, 5), 5, 4)
    b = _odds((4, 4, 4, 4), 4, 0)  # higher rung 0, fewer wins
    c = _odds((4, 4, 4, 6), 6, 9)  # same rung 0 as b, more wins
    d = _odds((4, 4, 4, 6), 6, 2)  # as c with fewer deaths
    assert sorted([a, b, c, d], key=lambda o: o.score(0), reverse=True) == [d, c, b, a]
    assert a.score(1) > b.score(1)
    assert b.score(0) == (4, 4, 0) and c.score(3) == (6, 6, -9)
    assert b.rung(0) == 0.4
