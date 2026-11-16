"""Stage B solver (``search/stage_b.py``; Stage B spec §3-4 and §6, plan rulings 4-10):
depth 0 is Stage A, exact agreement with an exhaustive solver on small endgames, no
peeking, the interval over transpositions, the pruning bound, dropped root actions, the
time budget, refused mass, an all-terminal root and reproducibility."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace

import jax
import numpy as np
import pytest

from nuzline.engine.damage import damage_table
from nuzline.engine.step import WINNER_PLAYER
from nuzline.search import deaths, stage_b
from nuzline.search import expand as expand_mod
from nuzline.search import leaves as leaves_mod
from nuzline.search.expand import ai_rows, expand
from nuzline.search.leaves import LeafValue, allocate
from nuzline.search.plan_tree import render
from nuzline.search.stage_a import ActionResult, Ranking
from nuzline.search.stage_b import HiddenLegality, PlanNode, SolveSettings, solve
from tests.engine.helpers import drop_programs, roark_real_state, roark_state
from tests.search import exhaustive as exhaustive_mod
from tests.search.exhaustive import children, exhaustive
from tests.search.fixtures import (
    all_terminal_root,
    dead_monferno,
    empty_bag,
    low_hp_endgame,
    pixel_endgame,
    shinx_vs_cranidos,
    spike_cannon_player_root,
    spore_endgame,
    switch_refusal_root,
)

pytestmark = pytest.mark.slow

KEY = jax.random.PRNGKey(0)
SMALL = SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200)
CAP = 2  # the endgames' turn cap
TURNS = 3  # playout length where real playouts run (short: a test-runtime budget)
ENDGAMES = {"pixel": pixel_endgame, "spore": spore_endgame, "low_hp": low_hp_endgame}


@pytest.fixture(scope="module", autouse=True)
def _small_chunk():
    """``expand``'s programs in 1024-row calls instead of ``CHUNK`` (16384), the exhaustive
    solver's in 1024-row calls instead of ``exhaustive.P`` (8192) and the leaf evaluators'
    playouts in 512-row calls instead of ``leaves.LEAF_CHUNK`` (4096): every call pads to
    its full size, and this file reached ~15 GB. A row's result depends on neither the call
    size nor the device count (each row plays its own key: the three modules' docstrings),
    so the solves are unchanged."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(expand_mod, "CHUNK", 1024)
        mp.setattr(exhaustive_mod, "P", 1024)
        mp.setattr(leaves_mod, "LEAF_CHUNK", 512)
        yield


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
    # No leaf budget at depth 0: the footer names Stage A's playouts per action.
    assert result.leaf_budget is None
    footer = render(result, names, 1)[-1]
    assert "Stage A (depth 0): 200 playouts per action" in footer and "leaf budget" not in footer


@pytest.mark.slow
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
    failed = np.asarray(deaths.over_budget(store))
    for i, (_, a, o) in enumerate(pairs):
        if q[o] < eps:
            pruned[a] += q[o]
            continue
        for c in kids[i]:
            # (a failed child, over the death budget, is not pruned: it is worth 0 and played
            # out only for the distribution)
            if not c.refused and int(store.over[c.state_row]) == 0 and not failed[c.state_row]:
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


@pytest.mark.slow
def test_dropped_root_action(tables, names):
    # spike_cannon_root's opponent also knows Spike Cannon, whose AI script is not
    # transcribed (the AI refuses the root); here only the player knows it.
    state = spike_cannon_player_root(tables, names)
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
    # Every leaf is worth 1 and every failed node (over the death budget) 0, so the estimate
    # is all the mass but the refused, the lost and the failed.
    assert r.failed > 0
    assert r.estimate == pytest.approx(1 - r.refused - lost - r.failed, abs=1e-6)


def test_all_terminal_root(tables, names):
    state = all_terminal_root(tables, names)
    result = solve(tables, names, state, KEY, SMALL)
    assert result.playouts == 0
    assert [r.action for r in sorted(result.ranking, key=lambda r: r.action)] == [0, 1, 2, 3]
    for r in result.ranking:
        assert r.estimate == pytest.approx(1.0, abs=1e-6)
        assert r.ci_low == r.ci_high == r.estimate == r.lower_bound
        assert r.unexplored == r.sampled == r.refused == 0


@pytest.mark.slow
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


def test_locked_player_solves(tables, names):
    """Review Focus 2: a 1v1 depth-1 solve from a state where the player is locked into
    Bide ranks the one legal action and finds no hidden legality (the lock is on screen)."""
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_moves=("Bide", "Growl"),
        opp_species="Geodude",
        opp_moves=("Tackle",),
        player_count=1,
        opp_count=1,
    )
    s = s._replace(
        locked_move=s.locked_move.at[0].set(names.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(2),
        bide_damage=s.bide_damage.at[0].set(10),
    )
    settings = SolveSettings(max_depth=1, budget_playouts=200, n_depth0=8, max_turns=TURNS)
    result = solve(tables, names, s, KEY, settings, leaf_fn=ZERO)
    assert [r.action for r in result.ranking] == [names.move_slot(s, 0, "Bide")]
    assert result.ranking[0].label == "Bide"


def test_hidden_legality_raises(tables, names):
    # Spec section 3 assumes the player's legality is observable; a constructed belief
    # node whose two states differ in it (the per-sid cache seeded) must stop the solve.
    state = roark_state(tables, names)
    solver = stage_b._Solver(tables, names, batch(state), SMALL, None)
    solver.legal.update({1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1]})
    same = stage_b._Node("decision", b"", 1.0, states={1: 0.5, 2: 0.5})
    assert solver._node_legal(same) == [0, 1, 2]
    mixed = stage_b._Node("decision", b"", 1.0, states={1: 0.5, 3: 0.5})
    with pytest.raises(HiddenLegality, match="legal actions"):
        solver._node_legal(mixed)


@pytest.mark.slow
def test_default_forced_evaluations_get_the_floor(tables, names, monkeypatch):
    """The default evaluator's forced path inside ``solve``: every pruned opponent action
    is evaluated with exactly ``2 * leaf_floor`` playouts (ruling 8), and the default leaf
    budget is ``n_depth0`` per legal root action. The real ``evaluate_forced`` runs with a
    one-turn cap (test-runtime rule), its program compiled after ``expand``'s are dropped
    (``drop_programs``: the solve's whole-engine programs stacked passed 11 GB);
    ``evaluate_leaves``, which test_reproducible runs inside ``solve``, is a stub that
    honours ``counts``."""
    calls = []
    real = stage_b.evaluate_forced

    def spy(tables, states, pairs, counts, key, max_turns):
        drop_programs()
        values = real(tables, states, pairs, counts, key, max_turns)
        calls.append((np.asarray(pairs), np.asarray(counts), values))
        return values

    def leaves(tables, states, counts, key, max_turns):
        return [LeafValue(0, int(c) // 2, 0, int(c) // 2, False) for c in counts]

    monkeypatch.setattr(stage_b, "evaluate_forced", spy)
    monkeypatch.setattr(stage_b, "evaluate_leaves", leaves)
    state = roark_real_state(tables, names)
    # Roark's AI row here is 3/4, 1/4: epsilon 0.3 prunes one opponent action per action.
    settings = SolveSettings(max_depth=1, epsilon=0.3, n_depth0=20, max_turns=1)
    result = solve(tables, names, state, KEY, settings)
    ((pairs, counts, values),) = calls
    assert len(pairs) == len(counts) == len(values) > 0
    assert np.all(counts == 2 * settings.leaf_floor)
    assert all(v.n_a == v.n_b == settings.leaf_floor for v in values)
    n_legal = len(result.ranking) + len(result.dropped)
    assert result.leaf_budget == settings.n_depth0 * n_legal
    pruned = [n for n in plan_nodes([result.plan]) if n.kind == "pruned" and n.obs is None]
    assert pruned and all(n.opp_action is not None and n.opp_label for n in pruned)


@pytest.mark.slow
def test_sampled_nodes_below_the_root(tables, names):
    # The player alone knows Spike Cannon (as in test_dropped_root_action): dropped at the
    # root, sampling-only at every depth-1 node, where every value 1 makes it the chosen
    # action (the lowest of tied actions), so the plan keeps sampled nodes.
    state = spike_cannon_player_root(tables, names)
    cannon = names.move_slot(state, 0, "Spike Cannon")
    settings = SolveSettings(max_depth=2, epsilon=0.05, budget_playouts=2000, n_depth0=200)
    result = solve(tables, names, state, KEY, settings, leaf_fn=constant_leaf(1, 1))
    assert set(result.dropped) == {cannon}
    counted = 0
    for r in result.ranking:
        nodes = [n for n in plan_nodes(result.plan.children[r.action]) if n.kind == "sampled"]
        assert r.sampled_nodes == len(nodes)
        assert r.sampled == pytest.approx(sum(n.prob for n in nodes), abs=1e-12)
        assert all(n.obs is None and " uses " in n.opp_label for n in nodes)
        counted += len(nodes)
    assert counted > 0
    best = result.ranking[0]
    assert f"({best.sampled_nodes})" in render(result, names, 0, root=state)[2]


def test_default_leaf_floor_is_one():
    # 2026-10-31 amendment: floor 1 per half (same inner-decision quality, cheaper).
    assert SolveSettings().leaf_floor == 1


# -- precision target (2026-10-31 amendment) ----------------------------------------------

# Roark's AI row at roark_real_state is 3/4, 1/4: epsilon 0.3 prunes one opponent action
# per root action, so every pass has forced evaluations as well as leaves.
PRECISE = SolveSettings(max_depth=1, epsilon=0.3, n_depth0=20, max_turns=1)


def stub_evaluators(monkeypatch, wins_a=lambda h: h // 2, wins_b=lambda h: h // 2):
    """Replace both evaluators with stubs honouring ``counts`` (``h = counts // 2`` per
    half, ``wins_a(h)``/``wins_b(h)`` wins); returns the call log ``(kind, counts,
    key)`` in call order."""
    calls = []

    def values(counts):
        out = []
        for c in np.asarray(counts):
            h = int(c) // 2
            out.append(LeafValue(wins_a(h), h, wins_b(h), h, False))
        return out

    def leaves(tables, states, counts, key, max_turns):
        assert len(counts) == len(states.hp) and np.all(np.asarray(counts) >= 2)
        calls.append(("L", np.asarray(counts).copy(), np.asarray(key).copy()))
        return values(counts)

    def forced(tables, states, pairs, counts, key, max_turns):
        assert len(counts) == len(states.hp) == len(pairs) and np.all(np.asarray(counts) >= 2)
        calls.append(("F", np.asarray(counts).copy(), np.asarray(key).copy()))
        return values(counts)

    monkeypatch.setattr(stage_b, "evaluate_leaves", leaves)
    monkeypatch.setattr(stage_b, "evaluate_forced", forced)
    return calls


def round_keys(r: int):
    """The (leaf, forced) keys of round ``r`` of the depth-1 pass under ``KEY``."""
    k_pass = jax.random.fold_in(KEY, 1)
    base = k_pass if r == 0 else jax.random.fold_in(jax.random.fold_in(k_pass, 2), r)
    return tuple(np.asarray(jax.random.fold_in(base, h)) for h in (0, 1))


def test_precision_none_is_one_fixed_round(tables, names, monkeypatch):
    shares = []

    def spy(weights, budget, floor):
        shares.append((np.asarray(weights, np.float64).copy(), budget, floor))
        return allocate(weights, budget, floor)

    monkeypatch.setattr(stage_b, "allocate", spy)
    calls = stub_evaluators(monkeypatch)
    state = roark_real_state(tables, names)
    result = solve(tables, names, state, KEY, PRECISE)
    assert [c[0] for c in calls] == ["L", "F"]
    # Round 0 is ruling 8: B0 = n_depth0 per legal root action, split by leaf-node reach
    # only (pruned nodes add nothing; a state reached only through them gets the floor).
    (leaf_w, b0, floor), _ = shares
    assert b0 == PRECISE.n_depth0 * len(result.ranking) == result.leaf_budget
    assert floor == PRECISE.leaf_floor
    assert np.array_equal(calls[0][1], allocate(leaf_w, b0, floor))
    kids = [kid for a in result.plan.children for kid in result.plan.children[a]]
    leaf_mass = sum(kid.prob for kid in kids if kid.kind == "leaf")
    pruned_mass = sum(kid.prob for kid in kids if kid.kind == "pruned")
    assert pruned_mass > 0 and leaf_w.sum() == pytest.approx(leaf_mass)
    assert np.any(leaf_w == 0) and np.all(calls[0][1][leaf_w == 0] == 2 * floor)
    k_leaf, k_forced = round_keys(0)
    assert np.array_equal(calls[0][2], k_leaf) and np.array_equal(calls[1][2], k_forced)
    assert np.all(calls[1][1] == 2 * PRECISE.leaf_floor)
    assert result.rounds == 0 and result.half_width_a is None and result.precision_met is None
    # A target met at once runs exactly the same round 0.
    again = stub_evaluators(monkeypatch)
    loose = solve(tables, names, state, KEY, replace(PRECISE, precision=1.0))
    assert [(k, c.tolist(), key.tolist()) for k, c, key in again] == [
        (k, c.tolist(), key.tolist()) for k, c, key in calls
    ]
    assert loose.rounds == 0 and loose.precision_met is True
    assert 0 < loose.half_width_a <= 1.0
    assert loose.ranking == result.ranking and loose.playouts == result.playouts


def test_precision_caps_rounds_and_doubles_playouts(tables, names, monkeypatch):
    calls = stub_evaluators(monkeypatch)
    state = roark_real_state(tables, names)
    result = solve(tables, names, state, KEY, replace(PRECISE, precision=1e-9))
    assert result.rounds == stage_b.MAX_ROUNDS and result.precision_met is False
    assert result.half_width_a > 1e-9
    assert len(calls) == 2 * (stage_b.MAX_ROUNDS + 1)
    per_round = []
    for r in range(stage_b.MAX_ROUNDS + 1):
        got = calls[2 * r : 2 * r + 2]
        assert [c[0] for c in got] == ["L", "F"], r
        assert [c[2].tolist() for c in got] == [k.tolist() for k in round_keys(r)], r
        per_round.append(sum(int(c[1].sum()) for c in got))
    assert result.playouts == sum(per_round)
    n_evals = len(calls[0][1]) + len(calls[1][1])
    spent = per_round[0]
    for r, added in enumerate(per_round[1:], 1):
        # Each top-up matches everything spent so far, up to the rounding of the shares.
        assert abs(added - spent) <= n_evals, r
        spent += added


def test_precision_stops_at_the_first_round_that_meets_it(tables, names, monkeypatch):
    state = roark_real_state(tables, names)
    widths = {}
    for k in (1, 2):
        stub_evaluators(monkeypatch)
        monkeypatch.setattr(stage_b, "MAX_ROUNDS", k)
        widths[k] = solve(tables, names, state, KEY, replace(PRECISE, precision=1e-9)).half_width_a
    monkeypatch.setattr(stage_b, "MAX_ROUNDS", 8)
    assert widths[2] < widths[1]
    stub_evaluators(monkeypatch)
    target = (widths[1] + widths[2]) / 2
    result = solve(tables, names, state, KEY, replace(PRECISE, precision=target))
    assert result.rounds == 2 and result.precision_met is True
    assert result.half_width_a == pytest.approx(widths[2])


def test_precision_reads_half_a_only(tables, names, monkeypatch):
    state = roark_real_state(tables, names)
    sure, coin = (lambda h: 0), (lambda h: h // 2)

    def run(wins_a, wins_b, precision):
        calls = stub_evaluators(monkeypatch, wins_a=wins_a, wins_b=wins_b)
        result = solve(tables, names, state, KEY, replace(PRECISE, precision=precision))
        return result, [(k, c.tolist()) for k, c, _ in calls]

    # Round-0 half-A widths (a target of 1 is met at once): half A sure is narrower.
    narrow = run(sure, coin, 1.0)[0].half_width_a
    wide = run(coin, sure, 1.0)[0].half_width_a
    assert narrow < wide
    # Target = the sure half A's width: met at round 0 however noisy half B is ...
    calm, calm_calls = run(sure, coin, narrow)
    assert calm.rounds == 0 and calm.precision_met is True
    same, same_calls = run(sure, sure, narrow)
    assert same_calls == calm_calls and same.rounds == 0
    # ... and missed by the coin-flip half A although every half-B value is exact.
    noisy, _ = run(coin, sure, narrow)
    assert noisy.rounds >= 1 and noisy.half_width_a < wide


def test_top_up_keys_are_fresh(tables, names, monkeypatch):
    calls = stub_evaluators(monkeypatch)
    monkeypatch.setattr(stage_b, "MAX_ROUNDS", 2)
    state = roark_real_state(tables, names)
    solve(tables, names, state, KEY, replace(PRECISE, precision=1e-9))
    keys = [c[2].tobytes() for c in calls]
    assert len(keys) == 6 and len(set(keys)) == 6


def test_forced_evaluations_get_top_ups_by_reach(tables, names, monkeypatch):
    calls = stub_evaluators(monkeypatch)
    monkeypatch.setattr(stage_b, "MAX_ROUNDS", 1)
    state = roark_real_state(tables, names)
    result = solve(tables, names, state, KEY, replace(PRECISE, precision=1e-9))
    assert result.rounds == 1
    q = ai_rows(tables, batch(state))[0].astype(np.float64)
    pruned = q[(q > 0) & (q < PRECISE.epsilon)]
    assert len(pruned) == 1
    # Depth 1 keeps every child in the plan: the evaluated reach is everything but the
    # terminal and refused mass of each root action.
    reach = 0.0
    for r in result.ranking:
        nodes = list(plan_nodes(result.plan.children[r.action]))
        reach += 1 - sum(n.prob for n in nodes if n.kind in ("terminal", "refused"))
    round0_l, round0_f, _, top_f = calls
    batch_size = int(round0_l[1].sum() + round0_f[1].sum())
    want = 2 * np.rint(batch_size * pruned[0] / reach / 2)
    assert want >= 2
    assert len(top_f[1]) == len(round0_f[1])  # every forced evaluation gets a share
    assert np.all(top_f[1] == want)


def test_precision_stops_topping_up_at_the_time_budget(tables, names, monkeypatch):
    calls = stub_evaluators(monkeypatch)
    state = roark_real_state(tables, names)
    settings = replace(PRECISE, precision=1e-9, budget_seconds=1e-9)
    result = solve(tables, names, state, KEY, settings)
    assert result.depth == 1 and result.rounds == 0 and result.precision_met is False
    assert len(calls) == 2


def test_precision_with_a_leaf_fn_tops_up_subsets(tables, names):
    state = roark_real_state(tables, names)
    seen = []

    def leaf_fn(states, weights, pairs=None):
        seen.append((len(states.hp), np.asarray(weights).copy(), pairs is None))
        return [LeafValue(1, 2, 1, 2, False) for _ in range(len(states.hp))]

    result = solve(tables, names, state, KEY, replace(PRECISE, precision=1e-9), leaf_fn=leaf_fn)
    assert result.rounds == stage_b.MAX_ROUNDS
    assert len(seen) == 2 * (stage_b.MAX_ROUNDS + 1)
    # Round 0 as before (forced weights 0); top-ups pass each evaluation's summed reach.
    assert np.all(seen[1][1] == 0)
    assert all(np.all(w > 0) for _, w, _ in seen[2:])


def test_precision_must_be_positive(tables, names):
    state = roark_real_state(tables, names)
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError, match="invalid settings"):
            solve(tables, names, state, KEY, replace(PRECISE, precision=bad), leaf_fn=ZERO)


def test_store_terminal_is_success(tables, names):
    base = dead_monferno(tables, names)._replace(over=np.int8(1), winner=np.int8(WINNER_PLAYER))
    protected = base.with_budget(3)._replace(
        protected=base.protected.at[int(np.flatnonzero(np.asarray(base.deaths))[0])].set(1)
    )
    lost = base.with_budget(3)._replace(winner=np.int8(1 - WINNER_PLAYER))
    store = stage_b._Store()
    store.add(batch(base.with_budget(0), base.with_budget(1), protected, lost))
    assert store.won == [False, True, False, False]
    assert store.over == [True] * 4


def test_terminal_win_after_a_death_scores_zero_at_k0(tables, names):
    state = empty_bag(dead_monferno(tables, names, player_moves=["Bubble"], opp_moves=["Tackle"]))
    state = state.with_pp(0, 0, [int(p) if i == 0 else 0 for i, p in enumerate(state.pp[0, 0])])
    bubble = names.move_slot(state, 0, "Bubble")
    table = np.asarray(damage_table(tables, state, 0, names.move("Bubble"), 0))
    assert table.min() >= int(state.hp[1, 0])  # one Bubble always KOs
    settings = SolveSettings(max_depth=1, epsilon=0.0, n_depth0=8, max_turns=TURNS)
    exact, _ = exhaustive(tables, state.with_budget(0), 1)
    assert exact[bubble] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="death budget"):  # Monferno's death is over budget
        solve(tables, names, state.with_budget(0), KEY, settings, leaf_fn=ZERO)
    s = state.with_budget(1)
    result = solve(tables, names, s, KEY, settings, leaf_fn=ZERO)
    exact, _ = exhaustive(tables, s, 1)
    assert exact[bubble] == pytest.approx(1.0, abs=1e-9)
    assert all(r.estimate == pytest.approx(1.0, abs=1e-9) for r in result.ranking)


def test_win_after_a_new_death_scores_zero_at_k0(tables, names):
    # The lead's Struggle KOs the last Geodude and its recoil KOs the lead; a living bench
    # Pokemon makes it a win reached through a death made inside the search.
    state = empty_bag(
        roark_state(tables, names, player_count=2, opp_count=1)
        .replace_hp(0, 0, 1)
        .replace_hp(1, 0, 1)
        .with_pp(0, 0, [0, 0, 0, 0])
    )
    settings = SolveSettings(max_depth=1, epsilon=0.0, n_depth0=8, max_turns=TURNS)
    for k, want in ((0, 0.0), (1, 1.0)):
        s = state.with_budget(k)
        result = solve(tables, names, s, KEY, settings, leaf_fn=ZERO)
        exact, _ = exhaustive(tables, s, 1)
        assert exact[0] == pytest.approx(want, abs=1e-9)
        assert row(result, 0).estimate == pytest.approx(want, abs=1e-9)


@pytest.mark.slow
def test_result_dict_round_trip(tables, names):
    # Plan 4c Task 8: a checkpoint holds finished solves as JSON; nothing may be lost.
    from tests.search.fixtures import certain_zero_leaf, sacrifice_endgame

    pixel = pixel_endgame(tables, names)
    sacrifice = sacrifice_endgame(tables, names).with_budget(1)
    solves = [
        solve(tables, names, pixel, KEY, SolveSettings(max_depth=2, epsilon=0.0, n_depth0=8), ZERO),
        solve(
            tables, names, sacrifice, KEY,
            SolveSettings(max_depth=2, epsilon=0.0, n_depth0=8), certain_zero_leaf,
        ),
    ]  # fmt: skip
    for r in solves:
        assert any(n.children for n in plan_nodes(r.plan.children[r.ranking[0].action]))
        assert any(a.distribution is not None for a in r.ranking)
        back = stage_b.result_from_dict(json.loads(json.dumps(stage_b.result_to_dict(r))))
        assert back == r
        assert back.seconds == r.seconds


def quick_claw_endgame(tables, names):
    """Monferno at 1 HP holding a Quick Claw (Scratch) against a faster Tackle-only Geodude
    that any Scratch knocks out (no crits: Lucky Chant on both sides): Monferno wins when the
    battle-start draw (the root's ``qc_roll``) moves it first, or when Tackle misses."""
    from tests.search.fixtures import endgame

    s = endgame(tables, names, ["Scratch"], ["Tackle"], player_hp=1)
    s = s._replace(item=s.item.at[0, 0].set(names.item("Quick Claw")))
    spe = 4  # stats[..., 0..4] are Atk..Spe
    s = s._replace(stats=s.stats.at[1, 0, spe].set(int(s.stats[0, 0, spe]) + 1))
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    scratch = np.asarray(damage_table(tables, s, 0, names.move("Scratch"), False))
    return s.replace_hp(1, 0, int(scratch.min()))


def test_root_chance_node_for_the_battle_start_quick_claw(tables, names):
    # Rules §5(b), §11 item 1: the lead's Quick Claw is decided by the battle-start draw
    # (BattleContext_InitCounters, battle_lib.c:2019-2025) before turn 1's order. The root is
    # a chance node over qc_roll (1/5, 4/5); the player sees neither value, so the two states
    # form one belief node. Exact against the exhaustive solver over that belief.
    from nuzline.engine.step import root_belief

    state = quick_claw_endgame(tables, names)
    belief = root_belief(tables, state)
    assert [(int(s.qc_roll[0]), p) for s, p in belief] == [(0, 0.8), (1, 0.2)]
    settings = SolveSettings(max_depth=CAP, epsilon=0.0, n_depth0=8)
    result = solve(tables, names, state, KEY, settings, leaf_fn=ZERO)
    want, _ = exhaustive(tables, state, CAP, belief=belief)
    scratch = names.move_slot(state, 0, "Scratch")
    assert row(result, scratch).estimate == pytest.approx(want[scratch], abs=1e-9)
    # Each fixed root value alone gives another answer; the belief is their 4:1 mix.
    alone = [exhaustive(tables, s, CAP)[0][scratch] for s, _ in belief]
    assert alone[1] - alone[0] > 0.1
    assert want[scratch] == pytest.approx(0.8 * alone[0] + 0.2 * alone[1], abs=1e-12)
    # The given state's own qc_roll is hidden: solving it with either value is the same.
    other = state._replace(qc_roll=state.qc_roll.at[0].set(1))
    again = solve(tables, names, other, KEY, settings, leaf_fn=ZERO)
    assert row(again, scratch).estimate == pytest.approx(want[scratch], abs=1e-9)


def test_root_belief_states_share_the_root_node(tables, names):
    state = quick_claw_endgame(tables, names)
    solver = stage_b._Solver(tables, names, batch(state), SMALL, None)
    assert solver.root_states == {0: 1.0}
    from nuzline.engine.step import root_belief

    belief = root_belief(tables, state)
    solver = stage_b._Solver(
        tables, names, batch(*(s for s, _ in belief)), SMALL, None, weights=[p for _, p in belief]
    )
    assert solver.root_states == {0: 0.8, 1: 0.2}
