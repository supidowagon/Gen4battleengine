"""Stage B with the faint budget (plan 4b, Task 9): replacement decision nodes, failed nodes
and the death distribution, checked against the exhaustive solver on the constructed
endgames of ``tests/search/fixtures.py``."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nuzline.ai.policy import UnsupportedAi
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import damage_table
from nuzline.search import expand as expand_mod
from nuzline.search import leaves as leaves_mod
from nuzline.search import stage_b
from nuzline.search.deaths import DeathDistribution
from nuzline.search.expand import ai_rows
from nuzline.search.leaves import LeafValue
from nuzline.search.rows import row_bytes
from nuzline.search.stage_b import ActionValue, Backup, PlanNode, SolveSettings, solve
from tests.engine.helpers import roark_real_state
from tests.search import exhaustive as exhaustive_mod
from tests.search.exhaustive import exhaustive
from tests.search.fixtures import (
    _facing,
    certain_zero_leaf,
    faint_pending_root,
    sacrifice_endgame,
)

pytestmark = pytest.mark.slow

KEY = jax.random.PRNGKey(0)
EXACT = SolveSettings(max_depth=1, epsilon=0.0, n_depth0=8)
TO_PRINPLUP, TO_STARLY = MOVE_ACTIONS + 1, MOVE_ACTIONS + 3
_SPE = 4  # stats[..., 0..4] are Atk..Spe


@pytest.fixture(scope="module", autouse=True)
def _small_chunk():
    """``expand``'s programs in 1024-row calls instead of ``CHUNK`` (16384), the exhaustive
    solver's in 1024-row calls instead of ``exhaustive.P`` (8192) and the leaf evaluators'
    playouts in 512-row calls instead of ``leaves.LEAF_CHUNK`` (4096), as
    ``tests/search/test_stage_b.py``'s ``_small_chunk``: every call pads to its full size, and
    at the full sizes each solve here peaked at 70-77 GB on the pod (an XLA abort, rc 134,
    in a leaf program's compile once the pod's other chunks filled the rest). A row's result
    depends on neither the call size nor the device count (each row plays its own key: the
    three modules' docstrings), so the solves are unchanged."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(expand_mod, "CHUNK", 1024)
        mp.setattr(exhaustive_mod, "P", 1024)
        mp.setattr(leaves_mod, "LEAF_CHUNK", 512)
        yield


def plan_nodes(nodes: list[PlanNode]):
    """Every node under ``nodes`` that the plan keeps."""
    for node in nodes:
        yield node
        for kids in node.children.values():
            yield from plan_nodes(kids)


def plan_paths(nodes: list[PlanNode], path: tuple[bytes, ...] = ()):
    """``(node, observation path)`` for every node under ``nodes`` that the plan keeps: the
    path ``exhaustive``'s ``choose`` takes."""
    for node in nodes:
        here = path if node.obs is None else (*path, node.obs)
        yield node, here
        for kids in node.children.values():
            yield from plan_paths(kids, here)


def row(result, action: int) -> ActionValue:
    (out,) = [r for r in result.ranking if r.action == action]
    return out


def estimates(result) -> dict[int, float]:
    return {r.action: r.estimate for r in result.ranking}


def all_win_leaf(states, weights, pairs=None):
    """Every evaluation worth 1 with a negligible interval: a mishandled failed node would
    show as value."""
    return [LeafValue(10**6, 10**6, 10**6, 10**6, False) for _ in range(len(states.hp))]


@pytest.mark.slow
def test_pending_root_matches_exhaustive(tables, names):
    state = faint_pending_root(tables, names)
    result = solve(tables, names, state, KEY, EXACT, leaf_fn=certain_zero_leaf)
    exact, _ = exhaustive(tables, state, 1)
    assert sorted(estimates(result)) == [TO_PRINPLUP, TO_STARLY]
    for action, value in exact.items():
        assert row(result, action).estimate == pytest.approx(value, abs=1e-6)
    assert row(result, TO_PRINPLUP).estimate == pytest.approx(1.0, abs=1e-9)
    assert row(result, TO_STARLY).estimate == pytest.approx(0.0, abs=1e-9)
    assert result.ranking[0].action == TO_PRINPLUP  # decided by value, not by the tie rule
    assert result.plan.replacement and result.plan.kind == "decision"
    assert (result.deaths_allowed, result.protected) == (1, ())


def starly_refused(monkeypatch, everything: bool = False):
    """``stage_b.ai_rows`` with the rows of every state whose player active is slot 3 (or
    of every state) refused; returns the list of batches it was asked for."""
    seen = []

    def fake(tables, states):
        seen.append(jax.tree.map(np.asarray, states))
        rows = np.array(ai_rows(tables, states))
        rows[np.ones(len(rows), bool) if everything else np.asarray(states.active)[:, 0] == 3] = (
            np.nan
        )
        return rows

    monkeypatch.setattr(stage_b, "ai_rows", fake)
    return seen


def test_pending_root_drops_only_an_ai_refused_replacement(tables, names, monkeypatch):
    state = faint_pending_root(tables, names)
    seen = starly_refused(monkeypatch)
    result = solve(tables, names, state, KEY, EXACT, leaf_fn=certain_zero_leaf)
    assert [r.action for r in result.ranking] == [TO_PRINPLUP]
    assert row(result, TO_PRINPLUP).estimate == pytest.approx(1.0, abs=1e-9)
    assert list(result.dropped) == [TO_STARLY]
    assert "refused" in result.dropped[TO_STARLY]
    # The pending root's own AI row is never requested (its active has 0 HP).
    root = row_bytes(jax.tree.map(lambda f: np.asarray(f)[None], state), 1)[0].tobytes()
    for batch in seen:
        assert root not in [r.tobytes() for r in row_bytes(batch, len(batch.hp))]
    starly_refused(monkeypatch, everything=True)
    with pytest.raises(UnsupportedAi):
        solve(tables, names, state, KEY, EXACT, leaf_fn=certain_zero_leaf)


@pytest.mark.parametrize("k", [0, 1])
def test_sacrifice_matches_exhaustive(tables, names, k):
    state = sacrifice_endgame(tables, names).with_budget(k)
    settings = SolveSettings(max_depth=2, epsilon=0.0, n_depth0=8)
    result = solve(tables, names, state, KEY, settings, leaf_fn=certain_zero_leaf)
    exact, choose = exhaustive(tables, state, 2)
    assert sorted(estimates(result)) == sorted(exact)
    for action, value in exact.items():
        assert row(result, action).estimate == pytest.approx(value, abs=1e-6)
    if k == 0:
        return
    leer = names.move_slot(state, 0, "Leer")
    assert result.ranking[0].action == leer
    replacements = [
        (n, path)
        for n, path in plan_paths(result.plan.children[leer])
        if n.kind == "decision" and n.replacement
    ]
    assert any(node.action == TO_PRINPLUP for node, _ in replacements)
    for node, path in replacements:  # the exhaustive choice on the same path
        assert node.action == choose(path)


def test_failed_nodes_value_zero(tables, names):
    state = sacrifice_endgame(tables, names).with_budget(0)
    leer = names.move_slot(state, 0, "Leer")
    calls = []

    def recording(states, weights, pairs=None):
        calls.append((jax.tree.map(np.asarray, states), np.asarray(weights), pairs))
        return all_win_leaf(states, weights, pairs)

    result = solve(tables, names, state, KEY, EXACT, leaf_fn=recording)
    r = row(result, leer)
    failed = [n for n in plan_nodes(result.plan.children[leer]) if n.kind == "failed"]
    assert failed
    assert r.failed == pytest.approx(sum(n.prob for n in failed), abs=1e-9)
    assert r.failed == pytest.approx(0.95, abs=1e-9)  # Tackle hits: Monferno is the first death
    # Leaves are worth 1 here, failed nodes 0: the estimate is the rest of the mass.
    assert r.estimate == pytest.approx(1 - r.failed - r.refused, abs=1e-9)
    assert r.ci_high - r.ci_low < 1e-2
    failed_calls = [
        c for c in calls if len(c[1]) and not c[1].any() and c[2] is None
        and (c[0].player_faints > c[0].faint_budget).all()
    ]  # fmt: skip
    assert len(failed_calls) == 1


@pytest.mark.slow
def test_distribution_sums(tables, names):
    state = sacrifice_endgame(tables, names).with_budget(1)
    settings = SolveSettings(max_depth=1, budget_playouts=2000, n_depth0=200)
    result = solve(tables, names, state, KEY, settings)
    assert result.ranking
    for r in result.ranking:
        assert r.distribution.n is None
        assert sum(r.distribution.probs) == pytest.approx(1 - r.refused, abs=1e-9)
        assert r.distribution.rung(1) == pytest.approx(r.estimate, abs=1e-9)


def protected_root(tables, names):
    """``faint_pending_root`` with a protected-or-not choice at stake: Geodude at 3 HP and
    faster than everyone (Starly's Wing Attack KOs it), Prinplup at 4 HP (a Tackle that hits
    KOs it) and Starly at 12 HP without its Oran Berry: one ordinary Tackle leaves it standing, two always KO it.
    That last number closes the escape the 30 HP Starly would give (a switch to Starly on
    turn 1 takes the Tackle instead of Prinplup and wins on turn 2 all the same); with 12 HP
    the free entry after Prinplup's death costs Starly one Tackle, the switch two."""
    s = faint_pending_root(tables, names)
    s = s.replace_hp(1, 0, 3).replace_hp(0, 1, 4).replace_hp(0, 3, 12)
    s = s._replace(item=s.item.at[0, 3].set(0))  # no Oran Berry for Starly
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(25))
    assert int(s.stats[1, 0, _SPE]) > max(int(s.stats[0, 1, _SPE]), int(s.stats[0, 3, _SPE]))
    wing = damage_table(tables, _facing(s, 3), 0, names.move("Wing Attack"), 0)
    assert int(np.asarray(wing).min()) >= 3
    tackle = np.asarray(damage_table(tables, _facing(s, 1), 1, names.move("Tackle"), 0))
    assert int(tackle.min()) >= 4
    on_starly = np.asarray(damage_table(tables, _facing(s, 3), 1, names.move("Tackle"), 0))
    assert int(on_starly.max()) < 12 <= 2 * int(on_starly.min())
    return s


@pytest.mark.slow
def test_protected_line_matches_exhaustive(tables, names):
    base = protected_root(tables, names)
    settings = SolveSettings(max_depth=2, epsilon=0.0, n_depth0=8)
    got = {}
    for label, state in (
        ("open", base.with_budget(k=2)),
        ("protected", base.with_budget(k=2, protected=[1])),
    ):
        result = solve(tables, names, state, KEY, settings, leaf_fn=certain_zero_leaf)
        exact, _ = exhaustive(tables, state, 2)
        for action, value in exact.items():
            assert row(result, action).estimate == pytest.approx(value, abs=1e-6), label
        got[label] = row(result, TO_PRINPLUP)
    # Prinplup's death is within budget unless it is protected: then the line is failed.
    assert got["open"].estimate > 0.9
    assert got["open"].estimate - got["protected"].estimate > 0.5


def test_second_replacement_decision_in_plan(tables, names):
    state = faint_pending_root(tables, names).with_budget(2)
    state = state._replace(stealth_rock=state.stealth_rock.at[0].set(1)).replace_hp(0, 1, 1)
    result = solve(tables, names, state, KEY, EXACT, leaf_fn=certain_zero_leaf)
    exact, _ = exhaustive(tables, state, 1)
    for action, value in exact.items():
        assert row(result, action).estimate == pytest.approx(value, abs=1e-6)
    nodes = list(plan_nodes(result.plan.children[TO_PRINPLUP]))
    second = [n for n in nodes if n.kind == "decision" and n.replacement]
    assert len(second) == 1 and second[0].action == TO_STARLY


def backup(a: float, deaths_a: float) -> Backup:
    return Backup(a, a, 0.0, {}, [], deaths_a, deaths_a, np.zeros(8))


def test_inner_tie_prefers_fewer_deaths():
    assert stage_b._leader({4: backup(0.5, 1.0), 7: backup(0.5, 0.2)}) == 7
    assert stage_b._leader({4: backup(0.5, 1.0), 7: backup(0.5 + 1e-9, 5.0)}) == 7
    assert stage_b._leader({4: backup(0.5 + 1e-9, 5.0), 7: backup(0.5, 0.0)}) == 4
    assert stage_b._leader({7: backup(0.5, 1.0), 4: backup(0.5, 1.0)}) == 4

    def value(action, estimate, e_deaths, lo, hi):
        return ActionValue(action, str(action), estimate, lo, hi, estimate, 0, 0, 0, False,
                           e_deaths=e_deaths)  # fmt: skip

    rows = [
        value(0, 0.60, 2.0, 0.55, 0.65),  # best estimate, many deaths
        value(1, 0.58, 0.5, 0.53, 0.63),  # overlaps it, few deaths: the leader
        value(2, 0.20, 0.0, 0.15, 0.25),  # clearly worse: follows by score
    ]
    ranked = stage_b._rank(rows)
    assert [r.action for r in ranked] == [1, 0, 2]
    assert [r.tied_with_best for r in ranked] == [False, True, False]


def test_ties_never_trade_a_whiteout_for_fewer_deaths():
    """Spec section 2: inner nodes (``_leader``) and the root (``_rank``) weigh a lost
    playout above any number of deaths."""
    risky = Backup(0.5, 0.5, 0.0, {}, [], 0.75, 0.75, np.zeros(8), 0.05)
    safe = Backup(0.5, 0.5, 0.0, {}, [], 1.0, 1.0, np.zeros(8), 0.0)
    assert stage_b._leader({4: risky, 7: safe}) == 7

    def value(action, e_deaths, probs):
        return ActionValue(action, str(action), 0.5, 0.45, 0.55, 0.5, 0, 0, 0, False,
                           e_deaths=e_deaths, distribution=DeathDistribution(probs))  # fmt: skip

    rows = [
        value(0, 0.75, (0.5, 0.45, 0, 0, 0, 0, 0.05, 0)),
        value(1, 1.0, (0.5, 0, 0.5, 0, 0, 0, 0, 0)),
    ]
    assert [r.action for r in stage_b._rank(rows)] == [1, 0]


def test_budget_does_not_change_k0(tables, names):
    state = roark_real_state(tables, names)
    settings = SolveSettings(max_depth=1, budget_playouts=64, n_depth0=8)
    result = solve(tables, names, state, KEY, settings, leaf_fn=all_win_leaf)
    assert result.ranking and result.deaths_allowed == 0
    for r in result.ranking:
        nodes = list(plan_nodes(result.plan.children[r.action]))
        assert not any(n.replacement for n in nodes)
        # Failed nodes are worth 0: the estimate is at most all the mass that is not failed.
        assert r.estimate <= 1 - r.failed - r.refused + 1e-9
