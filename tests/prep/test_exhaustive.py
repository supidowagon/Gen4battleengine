"""The exhaustive check of the funnel (plan 4c Task 12, spec section 8): restricted and pruned
setups, the verdict, and the funnel's pick against every setup's value on small pools."""

import itertools
import os
from dataclasses import replace
from math import comb

import jax
import pytest

from nuzline.prep.exhaustive import (
    Value,
    exhaustive_values,
    pruned_setups,
    recheck_verdict,
    restricted_setups,
    stored_value,
    verdict,
)
from nuzline.prep.problem import Resources
from nuzline.prep.search import PrepSettings, run_prep
from nuzline.prep.variants import RELEARNER, designated_change, member_spec
from tests.engine.helpers import sample_party
from tests.prep.fixtures import candidate, problem

KEY = jax.random.PRNGKey(3)
LEVEL = 14
SLOW = os.environ.get("NUZLINE_SLOW_TESTS") == "1"


def pool(names, species_slots, relearner=(), starly_moves=None):
    """Sample-party members at level 14 as candidates (``levelled`` empty)."""
    party = sample_party(names, {3: starly_moves} if starly_moves else None)
    specs = [replace(party[i], level=LEVEL) for i in species_slots]
    return [candidate(names, s, i, relearner=relearner) for i, s in enumerate(specs)]


def small_problem(tables, names, slots, *, deaths=0, scales=0, **kw):
    cands = pool(names, slots, **kw)
    return problem(
        tables,
        names,
        cands,
        resources=Resources(scales, (), (), ()),
        deaths=deaths,
        cap=LEVEL,
        levelled=[()] * len(cands),
    )


def test_restricted_setups_count(tables, names):
    bite = names.move("Bite")
    cands = pool(names, (0, 1, 2, 3))
    cands = [replace(c, relearner=(bite,)) if i < 3 else c for i, c in enumerate(cands)]
    prob = problem(
        tables,
        names,
        cands,
        resources=Resources(1, (), (), ()),
        cap=LEVEL,
        levelled=[()] * 4,
    )
    designated = [designated_change(tables, prob, i) for i in range(4)]
    assert [d is not None for d in designated] == [True, True, True, False]
    setups = restricted_setups(tables, prob, max_team=3)
    expected = 0
    for k in (1, 2, 3):
        for team in itertools.combinations(range(4), k):
            expected += k * (1 + sum(1 for i in team if i < 3))
    assert len(setups) == expected
    assert len(set(setups)) == len(setups)
    for setup in setups:
        changed = [m for m in setup.members if m.moves]
        assert len(changed) <= 1
        for m in changed:
            assert m.moves == (designated[m.candidate],)
    # without a Heart Scale only the base members remain
    bare = replace(prob, resources=Resources(0, (), (), ()))
    assert len(restricted_setups(tables, bare, max_team=3)) == sum(
        k * comb(4, k) for k in (1, 2, 3)
    )


def test_verdict():
    v = Value
    values = {
        0: v(0.50, 0.40, 0.60, 0.5),
        1: v(0.70, 0.60, 0.80, 0.3),  # best restricted
        2: v(0.65, 0.55, 0.75, 0.3),  # overlaps the best
        3: v(0.30, 0.20, 0.40, 0.7),  # disjoint from the best
    }
    ok = verdict(values, 2, ())
    assert (ok.best, ok.pick, ok.pick_ties_best, ok.beaten_by_pruned, ok.passed) == (
        1,
        2,
        True,
        (),
        True,
    )
    bad = verdict(values, 3, ())
    assert not bad.pick_ties_best and not bad.passed
    # a pruned setup (index 4) clearly above the pick fails the check, and is not "best"
    values[4] = v(0.99, 0.95, 1.0, 0.0)
    out = verdict(values, 2, [4])
    assert out.best == 1 and out.pick_ties_best
    assert out.beaten_by_pruned == (4,) and not out.passed
    # a pruned setup whose interval reaches the pick's does not beat it
    values[4] = v(0.70, 0.60, 0.80, 0.0)
    assert verdict(values, 2, [4]).passed
    # ties on the estimate go to the lower index
    tie = {0: v(0.5, 0.4, 0.6, 0.0), 1: v(0.5, 0.4, 0.6, 0.0)}
    assert verdict(tie, 1, ()).best == 0


@pytest.mark.slow
def test_pruned_setups_are_valued(tables, names):
    starly = ("Growl", "Quick Attack", "Wing Attack", "Double Team")
    prob = small_problem(tables, names, (0, 1, 3), scales=1, starly_moves=starly)
    tackle, scratch = names.move("Tackle"), names.move("Scratch")
    cands = list(prob.candidates)
    cands[2] = replace(cands[2], relearner=(tackle, scratch))
    prob = replace(prob, candidates=tuple(cands))

    setups, pairs = pruned_setups(tables, prob, max_team=2)
    assert setups and pairs
    for dropped, by in pairs:
        assert dropped.candidate == by.candidate == 2
        assert [c.move for c in dropped.moves] == [tackle]
        assert [c.move for c in by.moves] == [scratch]
        assert all(c.source == RELEARNER for c in dropped.moves)
    restricted = restricted_setups(tables, prob, max_team=2)
    assert not set(setups) & set(restricted)
    for setup in setups:
        starly_member = next(m for m in setup.members if m.candidate == 2)
        assert tackle in member_spec(prob, starly_member, tables).moves
        assert starly_member in {d for d, _ in pairs}
        assert all(not m.moves for m in setup.members if m.candidate != 2)

    everything = restricted + setups
    values = exhaustive_values(tables, names, prob, everything, KEY, depth=0, n_screen=64)
    assert sorted(values) == list(range(len(everything)))
    assert all(0.0 <= v.ci_low <= v.estimate <= v.ci_high <= 1.0 for v in values.values())

    pruned = range(len(restricted), len(everything))
    pick = 0
    boosted = {**values, len(restricted): Value(1.0, 0.9, 1.0, 0.0)}
    boosted[pick] = Value(0.2, 0.1, 0.3, 0.0)
    out = verdict(boosted, pick, pruned)
    assert len(restricted) in out.beaten_by_pruned and not out.passed


def _settings(**kw):
    return PrepSettings(
        screen_n=16, team_keep=2, variant_rounds=0, finalists=2, max_depth=0, max_team=2, **kw
    )


@pytest.mark.slow
@pytest.mark.parametrize("k", [0, 1])
def test_funnel_matches_exhaustive_small(tables, names, k):
    prob = small_problem(tables, names, (0, 1, 3), deaths=k)
    setups = restricted_setups(tables, prob, max_team=2)
    assert len(setups) == 9
    values = exhaustive_values(tables, names, prob, setups, KEY, depth=0, n_screen=400)
    pick = run_prep(tables, names, prob, _settings(), jax.random.PRNGKey(0)).chosen
    out = verdict(values, setups.index(pick), ())
    assert out.passed, (out, values)


STARLY = ("Growl", "Quick Attack", "Wing Attack", "Double Team")


def _ci_pool(tables, names, k, max_team, budget):
    """Monferno, Prinplup and Grotle may relearn Bite; Starly (``STARLY`` moves) Tackle or
    Scratch, so the prune drops Tackle (dominated by Scratch) and the pruned half of the
    pass condition has setups to check."""
    bite, tackle, scratch = (names.move(m) for m in ("Bite", "Tackle", "Scratch"))
    prob = small_problem(
        tables, names, (0, 1, 2, 3), deaths=k, scales=1, relearner=(bite,), starly_moves=STARLY
    )
    cands = list(prob.candidates)
    cands[3] = replace(cands[3], relearner=(tackle, scratch))
    prob = replace(prob, candidates=tuple(cands))
    restricted = restricted_setups(tables, prob, max_team)
    extra, _ = pruned_setups(tables, prob, max_team)
    assert extra, "the prune must drop a variant for the pruned check to mean anything"
    everything = restricted + extra
    values = exhaustive_values(
        tables, names, prob, everything, KEY, depth=1, budget_playouts=budget
    )
    settings = PrepSettings(
        screen_n=8,
        team_keep=4,
        variant_rounds=2,
        variant_keep=2,
        finalists=2,
        max_depth=1,
        finalist_playouts=budget,
        max_team=max_team,
        relearner="designated",
        tms=False,
        items=False,
        entry_hp=False,
        levelled=False,
    )
    pick = run_prep(tables, names, prob, settings, jax.random.PRNGKey(0)).chosen
    pruned = range(len(restricted), len(everything))
    return restricted, values, verdict(values, everything.index(pick), pruned), len(extra)


@pytest.mark.slow
def test_funnel_matches_exhaustive_ci(tables, names):
    restricted, values, out, n_pruned = _ci_pool(tables, names, 0, 2, 100)
    assert len(restricted) == 4 * 2 + 6 * 2 * 3
    assert n_pruned > 0  # the prune dropped a Starly relearner variant: setups to check
    assert out.passed, (out, values)


@pytest.mark.skipif(not SLOW, reason="NUZLINE_SLOW_TESTS=1 only")
@pytest.mark.parametrize("k", [0, 1])
def test_funnel_matches_exhaustive_ci_full(tables, names, k):
    _, values, out, n_pruned = _ci_pool(tables, names, k, 4, 500)
    assert n_pruned > 0
    assert out.passed, (out, values)


RUNG0 = {
    "phase1": {
        "891": [0.9985, 0.996, 1.0, 0.0015],
        "3357": [0.99, 0.985, 0.995, 0.02],
        "510": [0.996, 0.993, 0.999, 0.004],
        "77": [1.0, 0.998, 1.0, 0.0],
    },
    "phase2": {
        "891": [0.99900, 0.99624, 1.0, 0.0010],
        "3357": [0.98903, 0.98231, 0.99575, 0.0199],
        "510": [0.99602, 0.98970, 1.0, 0.0040],
    },
}


def test_stored_value_prefers_phase2_then_phase1():
    assert stored_value(RUNG0, 510) == ("phase2", Value(0.99602, 0.98970, 1.0, 0.0040))
    assert stored_value(RUNG0, 77) == ("phase1", Value(1.0, 0.998, 1.0, 0.0))
    assert stored_value(RUNG0, 12345) is None


def test_recheck_verdict_on_the_stored_rung0_values():
    old = recheck_verdict(RUNG0, 3357, Value(0.98903, 0.98231, 0.99575, 0.0199))
    assert old.best == 891 and not old.passed  # the gate's original failure
    new = recheck_verdict(RUNG0, 510, Value(0.99602, 0.98970, 1.0, 0.0040))
    assert new.best == 891 and new.passed
    coarse = recheck_verdict(RUNG0, 77, Value(1.0, 0.998, 1.0, 0.0))
    assert coarse.best == 891  # a value put in from phase 1 never becomes the best
