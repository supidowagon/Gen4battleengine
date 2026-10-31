"""Stage B solver (``search/stage_b.py``; Stage B spec §3-4 and §6, plan rulings 4-10):
depth 0 is Stage A, exact agreement with an exhaustive solver on small endgames, no
peeking, the interval over transpositions, the pruning bound, dropped root actions, the
time budget, refused mass, an all-terminal root and reproducibility."""

from __future__ import annotations

from collections import defaultdict

import jax
import numpy as np
import pytest

from nuzline.engine.step import WINNER_PLAYER
from nuzline.search import stage_b
from nuzline.search.expand import ai_rows, expand
from nuzline.search.leaves import LeafValue
from nuzline.search.stage_a import ActionResult, Ranking
from nuzline.search.stage_b import PlanNode, SolveSettings, solve
from tests.engine.helpers import roark_real_state, roark_state
from tests.search.exhaustive import children, exhaustive
from tests.search.fixtures import (
    all_terminal_root,
    empty_bag,
    low_hp_endgame,
    pixel_endgame,
    shinx_vs_cranidos,
    spore_endgame,
    switch_refusal_root,
)

KEY = jax.random.PRNGKey(0)
SMALL = SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200)
CAP = 2  # the endgames' turn cap
TURNS = 3  # playout length where real playouts run (short: a test-runtime budget)
ENDGAMES = {"pixel": pixel_endgame, "spore": spore_endgame, "low_hp": low_hp_endgame}


def constant_leaf(wins: int, n: int, wins_a: int | None = None):
    """A ``leaf_fn`` giving every evaluation ``wins`` of ``n`` (half A: ``wins_a``, by
    default the same)."""
    wins_a = wins if wins_a is None else wins_a

    def leaf_fn(states, weights, pairs=None):
        return [LeafValue(wins_a, n, wins, n, False) for _ in range(len(states.hp))]

    return leaf_fn


ZERO = constant_leaf(0, 1)


def batch(*states):
    return jax.tree.map(lambda *fs: np.stack([np.asarray(f) for f in fs]), *states)


def plan_nodes(nodes: list[PlanNode]):
    """Every node under ``nodes`` that the plan keeps."""
    for node in nodes:
        yield node
        for kids in node.children.values():
            yield from plan_nodes(kids)


def row(result, action: int):
    (out,) = [r for r in result.ranking if r.action == action]
    return out


@pytest.fixture(scope="module")
def endgame_solves(tables, names):
    """Per endgame: the state and its depth-``CAP`` solve with no pruning and a zero
    leaf evaluator (exact over ``CAP`` turns)."""
    settings = SolveSettings(max_depth=CAP, epsilon=0.0, n_depth0=8)
    out = {}
    for name, make in ENDGAMES.items():
        state = make(tables, names)
        out[name] = state, solve(tables, names, state, KEY, settings, leaf_fn=ZERO)
    return out


def test_depth0_is_stage_a(tables, names, monkeypatch):
    # Ruling 7: depth 0 returns ``rank_actions(n=n_depth0)``'s numbers unchanged. A spy
    # stands in for ``rank_actions`` (whose numbers tests/search/test_stage_a.py checks),
    # so this file compiles no Stage A playout program (test-runtime rule, notes/critic.md
    # B.3).
    state = roark_state(tables, names)
    canned = Ranking.of(
        [
            ActionResult(0, "Ember", 0.625, 0.55, 0.69, 0.75, 0.4, 200),
            ActionResult(2, "Scratch", 0.5, 0.43, 0.57, 0.6, 0.9, 200),
            ActionResult(5, "Switch -> Starly", 0.125, 0.08, 0.18, 0.3, 1.2, 200),
        ]
    )
    calls = []

    def spy(*args):
        calls.append(args)
        return canned

    monkeypatch.setattr(stage_b, "rank_actions", spy)
    settings = SolveSettings(max_depth=0, n_depth0=200, max_turns=TURNS)
    result = solve(tables, names, state, KEY, settings)
    ((t, n, s, n_depth0, key, max_turns),) = calls
    assert t is tables and n is names and s is state
    assert (n_depth0, max_turns) == (200, TURNS)
    assert np.array_equal(np.asarray(key), np.asarray(KEY))
    assert result.depth == 0 and result.dropped == {}
    got = [
        (r.action, r.label, r.estimate, r.ci_low, r.ci_high, r.tied_with_best)
        for r in result.ranking
    ]
    want = [(r.action, r.label, r.p_win0, r.ci_low, r.ci_high, r.tied_with_best) for r in canned]
    assert got == want
    for r in result.ranking:
        assert r.lower_bound == r.estimate
        assert r.unexplored == r.sampled == r.refused == 0
    assert set(result.plan.children) == {0, 2, 5}
    assert result.playouts == 200 * len(canned)


def test_matches_exhaustive_on_endgames(tables, endgame_solves):
    nontrivial = False
    for name, (state, result) in endgame_solves.items():
        want, _ = exhaustive(tables, state, CAP)
        assert result.depth == CAP and result.dropped == {}, name
        got = {r.action: r.estimate for r in result.ranking}
        assert got.keys() == want.keys(), name
        for a, v in want.items():
            assert got[a] == pytest.approx(v, abs=1e-6), (name, a)
        for r in result.ranking:
            assert r.unexplored == r.sampled == r.refused == 0, (name, r)
        nontrivial |= any(0 < v < 1 for v in want.values())
    assert nontrivial


def test_no_peeking(tables, names, endgame_solves):
    state, result = endgame_solves["pixel"]
    grouped, choose = exhaustive(tables, state, CAP, group=True)
    peeking, _ = exhaustive(tables, state, CAP, group=False)
    assert max(peeking.values()) - max(grouped.values()) >= 1e-3
    hydro = names.move_slot(state, 0, "Hydro Pump")
    hps: dict[bytes, set[int]] = defaultdict(set)
    for after, refused, obs, _ in children(tables, state, hydro, 0):
        if not refused and int(after.over) == 0:
            hps[obs].add(int(after.hp[1, 0]))
    shared = {obs for obs, values in hps.items() if len(values) >= 2}
    nodes = [n for n in result.plan.children[hydro] if n.obs in shared]
    assert nodes and all(n.kind == "decision" for n in nodes)
    for node in nodes:
        assert node.action == choose((node.obs,))


def test_interval_aggregates_transpositions(tables, names):
    # Cranidos at 1 HP with no Potion: Spark KOs it with or without a crit, so one state
    # (after Roark's replacement) is reached through two observations.
    state = shinx_vs_cranidos(tables, names, slower_opp=True)
    state = empty_bag(state.replace_hp(1, int(state.active[1]), 1))
    spark = names.move_slot(state, 0, "Spark")
    value = LeafValue(3, 10, 3, 10, False)
    settings = SolveSettings(max_depth=1, epsilon=0.0, budget_playouts=2000, n_depth0=200)
    result = solve(tables, names, state, KEY, settings, leaf_fn=constant_leaf(3, 10))
    q = ai_rows(tables, batch(state))[0].astype(np.float64)
    opp = np.flatnonzero(q > 0).tolist()
    store, kids = expand(tables, batch(state), [(0, spark, o) for o in opp])
    by_state, by_node = defaultdict(float), defaultdict(float)
    for r, o in enumerate(opp):
        for c in kids[r]:
            if not c.refused and int(store.over[c.state_row]) == 0:
                by_state[c.state_row] += q[o] * c.prob
                by_node[(c.state_row, c.obs)] += q[o] * c.prob
    assert len(by_node) > len(by_state), "no transposition"
    summed = sum(w * w for w in by_state.values())
    assert summed > sum(w * w for w in by_node.values()) + 1e-6
    half = 1.96 * np.sqrt(summed * value.var_b)
    r = row(result, spark)
    assert 0 < r.ci_low and r.ci_high < 1
    assert r.estimate - r.ci_low == pytest.approx(half, rel=1e-9)
    assert r.ci_high - r.estimate == pytest.approx(half, rel=1e-9)


def test_pruning_bound(tables, names):
    eps = 0.05
    state = roark_real_state(tables, names)
    settings = SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200, epsilon=eps)
    # Half A says 0, half B 1/2: reported values come from half B only (ruling 5).
    result = solve(tables, names, state, KEY, settings, leaf_fn=constant_leaf(1, 2, wins_a=0))
    q = ai_rows(tables, batch(state))[0].astype(np.float64)
    opp = np.flatnonzero(q > 0).tolist()
    pairs = [(0, r.action, o) for r in result.ranking for o in opp]
    store, kids = expand(tables, batch(state), pairs)
    pruned = defaultdict(float)
    groups = defaultdict(float)
    for i, (_, a, o) in enumerate(pairs):
        if q[o] < eps:
            pruned[a] += q[o]
            continue
        for c in kids[i]:
            if not c.refused and int(store.over[c.state_row]) == 0:
                groups[(a, c.obs)] += q[o] * c.prob
    for (a, _), g in groups.items():
        if g < eps:
            pruned[a] += g
    assert sum(pruned.values()) > 0
    for r in result.ranking:
        nodes = list(plan_nodes(result.plan.children[r.action]))
        assert r.unexplored == pytest.approx(
            sum(n.prob for n in nodes if n.kind == "pruned"), abs=1e-9
        )
        assert r.unexplored == pytest.approx(pruned[r.action], abs=1e-9)
        assert r.lower_bound <= r.estimate
        # Every half-B value is 1/2: the bound drops exactly half the pruned mass.
        assert r.estimate - r.lower_bound == pytest.approx(r.unexplored / 2, abs=1e-9)


def test_dropped_root_action(tables, names):
    # spike_cannon_root's opponent also knows Spike Cannon, whose AI script is not
    # transcribed (the AI refuses the root); here only the player knows it.
    state = roark_state(tables, names, player_moves=["Spike Cannon"])
    cannon = names.move_slot(state, 0, "Spike Cannon")
    result = solve(tables, names, state, KEY, SMALL, leaf_fn=constant_leaf(1, 2))
    assert set(result.dropped) == {cannon}
    assert "Spike Cannon" in result.dropped[cannon]
    assert "width cap" in result.dropped[cannon]
    legal = {r.action for r in result.ranking}
    assert cannon not in legal and len(legal) == 8
    assert set(result.plan.children) == legal


def test_time_budget_stops_deepening(tables, names):
    state = roark_real_state(tables, names)
    settings = SolveSettings(max_depth=2, budget_seconds=1e-9, budget_playouts=2000, n_depth0=200)
    result = solve(tables, names, state, KEY, settings, leaf_fn=constant_leaf(1, 2))
    assert result.depth == 1
    assert result.ranking


def test_refused_replacement_mass(tables, names):
    state = switch_refusal_root(tables, names)
    tackle = names.move_slot(state, 0, "Tackle")
    to_geodude = 4 + 0  # switch to party slot 0
    settings = SolveSettings(max_depth=1, epsilon=0.0, budget_playouts=2000, n_depth0=200)
    result = solve(tables, names, state, KEY, settings, leaf_fn=constant_leaf(1, 1))
    q = ai_rows(tables, batch(state))[0].astype(np.float64)
    opp = np.flatnonzero(q > 0).tolist()
    store, kids = expand(tables, batch(state), [(0, tackle, o) for o in opp])
    refused, lost = {}, 0.0
    for r, o in enumerate(opp):
        refused[o] = sum(c.prob for c in kids[r] if c.refused)
        for c in kids[r]:
            over = int(store.over[c.state_row]) != 0
            if not c.refused and over and int(store.winner[c.state_row]) != WINNER_PLAYER:
                lost += q[o] * c.prob
    assert q[to_geodude] > 0 and refused[to_geodude] > 0
    assert all(v == 0 for o, v in refused.items() if o != to_geodude)
    r = row(result, tackle)
    assert r.refused == pytest.approx(q[to_geodude] * refused[to_geodude], abs=1e-6)
    # Every leaf is worth 1, so the estimate is all the mass but the refused and the lost.
    assert r.estimate == pytest.approx(1 - r.refused - lost, abs=1e-6)


def test_all_terminal_root(tables, names):
    state = all_terminal_root(tables, names)
    result = solve(tables, names, state, KEY, SMALL)
    assert result.playouts == 0
    assert [r.action for r in sorted(result.ranking, key=lambda r: r.action)] == [0, 1, 2, 3]
    for r in result.ranking:
        assert r.estimate == pytest.approx(1.0, abs=1e-6)
        assert r.ci_low == r.ci_high == r.estimate == r.lower_bound
        assert r.unexplored == r.sampled == r.refused == 0


def test_reproducible(tables, names, monkeypatch):
    def no_stage_a(*args):
        raise AssertionError("Stage A runs only at depth 0 (ruling 7)")

    monkeypatch.setattr(stage_b, "rank_actions", no_stage_a)
    state = roark_real_state(tables, names)
    settings = SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200, max_turns=TURNS)
    a = solve(tables, names, state, KEY, settings)
    b = solve(tables, names, state, KEY, settings)
    assert a == b
    assert a.playouts >= 2000
    assert a.ranking and all(r.ci_low <= r.estimate <= r.ci_high for r in a.ranking)
