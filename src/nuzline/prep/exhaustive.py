"""The exhaustive check of the prep funnel (plan 4c Task 12; spec section 8).

The funnel's pick is compared with the value of every setup of the restricted space (every team
of 1..``max_team`` candidates with every lead, each member its ``base_member``, plus at most one
member on its designated Move Relearner change when a Heart Scale is spare; ruling 14) and of the
*pruned* setups: the restricted-shape setups holding one member on a variant that the
conservative prune (ruling 8, a heuristic) dropped. ``verdict`` passes when the pick's interval
overlaps the best restricted setup's and no pruned setup is clearly above the pick. The funnel
under test runs with ``levelled=False``: free level-up slot replacements are outside this space.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import jax
import numpy as np

from nuzline.data.tables import Names, Tables
from nuzline.prep.problem import PrepProblem
from nuzline.prep.screen import screen
from nuzline.prep.setups import Setup, _arranged, knows_field_move, setup_states
from nuzline.prep.variants import (
    RELEARNER,
    Member,
    base_member,
    designated_change,
    prune,
    single_changes,
)
from nuzline.search.stage_a import tied_with
from nuzline.search.stage_b import SolveSettings, solve

__all__ = [
    "Value",
    "Verdict",
    "exhaustive_values",
    "pruned_setups",
    "recheck_verdict",
    "restricted_setups",
    "stored_value",
    "verdict",
]


def _teams(n: int, max_team: int, must_hold: int | None = None):
    for size in range(1, min(max_team, n) + 1):
        for team in itertools.combinations(range(n), size):
            if must_hold is None or must_hold in team:
                yield team


def _designated_member(tables: Tables, problem: PrepProblem, candidate: int) -> Member | None:
    """The candidate's base member with its designated change, if the funnel's neighbours
    would offer it (``relearner="designated"``: the change's slot is still free)."""
    base = base_member(problem, candidate)
    change = designated_change(tables, problem, candidate)
    if (
        change is None
        or change.slot in {c.slot for c in base.moves}
        or any(c.source == RELEARNER for c in base.moves)
    ):
        return None
    return dataclasses.replace(base, moves=(*base.moves, change))


def restricted_setups(tables: Tables, problem: PrepProblem, max_team: int = 6) -> list[Setup]:
    """Every team of ``1..max_team`` and lead, each member its ``base_member``, and (with a
    Heart Scale) each such setup again with exactly one member on its designated change. Count:
    ``sum(comb(n, k) * k * (1 + d_k))``, ``d_k`` a team's members with a designated change;
    with a ``problem.field_move`` only the setups that know it (``knows_field_move``)."""
    n = len(problem.candidates)
    base = [base_member(problem, i) for i in range(n)]
    designated: list[Member | None] = [None] * n
    if problem.resources.heart_scales >= 1:
        designated = [_designated_member(tables, problem, i) for i in range(n)]
    out = []
    for team in _teams(n, max_team):
        for lead in team:
            members = {i: base[i] for i in team}
            out.append(_arranged(members[lead], [members[i] for i in team if i != lead]))
            for i in team:
                if designated[i] is not None:
                    changed = {**members, i: designated[i]}
                    out.append(_arranged(changed[lead], [changed[j] for j in team if j != lead]))
    return [s for s in out if knows_field_move(problem, s)]


def pruned_setups(
    tables: Tables, problem: PrepProblem, max_team: int = 6
) -> tuple[list[Setup], list[tuple[Member, Member]]]:
    """The restricted-shape setups of every variant the prune dropped from a candidate's
    ``relearner="all"`` single changes (no TMs, items or entry-HP regions), with the (dropped,
    dominated by) pairs. Setups already in ``restricted_setups`` (the dropped variant is a
    designated one) are left out."""
    n = len(problem.candidates)
    base = [base_member(problem, i) for i in range(n)]
    seen = set(restricted_setups(tables, problem, max_team))
    pairs: list[tuple[Member, Member]] = []
    setups: list[Setup] = []
    for c in range(n):
        variants = single_changes(
            tables,
            problem,
            c,
            relearner="all",
            tms=False,
            items=False,
            entry_hp=False,
            levelled=False,
        )
        pairs.extend(prune(tables, problem, variants)[1])
    for dropped, _ in pairs:
        c = dropped.candidate
        for team in _teams(n, max_team, c):
            members = {i: base[i] for i in team}
            members[c] = dropped
            for lead in team:
                setup = _arranged(members[lead], [members[i] for i in team if i != lead])
                if setup not in seen and knows_field_move(problem, setup):
                    seen.add(setup)
                    setups.append(setup)
    return setups, pairs


@dataclass(frozen=True)
class Value:
    """A setup's value at the problem's rung: the estimate, its 95 % interval and the expected
    deaths."""

    estimate: float
    ci_low: float
    ci_high: float
    e_deaths: float


def exhaustive_values(
    tables: Tables,
    names: Names,
    problem: PrepProblem,
    setups: Sequence[Setup],
    key,
    *,
    depth: int,
    n_screen: int = 400,
    budget_playouts: int | None = None,
    precision: float | None = None,
    done: Mapping[int, Value] | None = None,
    on_value: Callable[[int, Value], None] | None = None,
) -> dict[int, Value]:
    """Value every setup (index -> :class:`Value`) at rung ``problem.deaths``: depth 0 by
    ``n_screen`` Stage A playouts (``screen``, Wilson interval), depth 1 by a Stage B solve at
    ``max_depth=1`` (the best ranked action's numbers; no ranking: all 0). Setups in ``done``
    are kept as they are; ``on_value(index, value)`` follows each new one. Setup ``i`` uses
    ``fold_in(key, i)``."""
    if depth not in (0, 1):
        raise ValueError(f"depth must be 0 or 1, not {depth}")
    out: dict[int, Value] = dict(done or {})
    for i, setup in enumerate(setups):
        if i in out:
            continue
        states = setup_states(tables, problem, [setup])
        k = jax.random.fold_in(key, i)
        if depth == 0:
            stats = screen(tables, states, n_screen, k)[0]
            low, high = stats.interval(problem.deaths)
            value = Value(stats.rung(problem.deaths), low, high, stats.e_deaths)
        else:
            row = jax.tree.map(lambda f: np.asarray(f)[0], states)
            result = solve(
                tables,
                names,
                row,
                k,
                SolveSettings(max_depth=1, budget_playouts=budget_playouts, precision=precision),
            )
            top = result.ranking[0] if result.ranking else None
            value = (
                Value(top.estimate, top.ci_low, top.ci_high, top.e_deaths)
                if top
                else Value(0.0, 0.0, 0.0, 0.0)
            )
        out[i] = value
        if on_value is not None:
            on_value(i, value)
    return out


@dataclass(frozen=True)
class Verdict:
    """``best``: the best restricted setup; ``pick``: the funnel's; ``pick_ties_best``: their
    intervals overlap; ``beaten_by_pruned``: pruned setups clearly above the pick."""

    best: int
    pick: int
    pick_ties_best: bool
    beaten_by_pruned: tuple[int, ...]
    passed: bool


def verdict(values: Mapping[int, Value], pick: int, pruned: Collection[int]) -> Verdict:
    """The check's result. ``best`` is the highest estimate among the non-pruned indices (ties:
    the lower index); ``pick_ties_best`` the pick's interval overlapping the best's;
    ``beaten_by_pruned`` the pruned setups whose ``ci_low`` exceeds the pick's ``ci_high``."""
    pruned_set = set(pruned)
    best = min((i for i in values if i not in pruned_set), key=lambda i: (-values[i].estimate, i))
    ties = tied_with(values[pick], values[best])
    beaten = tuple(sorted(i for i in pruned_set if values[i].ci_low > values[pick].ci_high))
    return Verdict(best, pick, ties, beaten, ties and not beaten)


def stored_value(rung: Mapping, pick: int) -> tuple[str, Value] | None:
    """The pick's value in a stored exhaustive rung (``{"phase1": .., "phase2": ..}``, values
    ``[estimate, ci_low, ci_high, e_deaths]`` keyed by str index): phase 2, else phase 1."""
    for phase in ("phase2", "phase1"):
        if str(pick) in rung.get(phase, {}):
            return phase, Value(*rung[phase][str(pick)])
    return None


def recheck_verdict(
    rung: Mapping, pick: int, value: Value, pruned: Collection[int] = ()
) -> Verdict:
    """``verdict`` for a new funnel pick against a stored rung: the best is the stored phase-2
    best (the pick's own ``value`` never becomes it); the pick ties when its interval overlaps
    the best's."""
    stored = {int(i): Value(*v) for i, v in rung["phase2"].items()}
    pruned_set = set(pruned)
    best = min((i for i in stored if i not in pruned_set), key=lambda i: (-stored[i].estimate, i))
    ties = tied_with(value, stored[best])
    beaten = tuple(
        sorted(i for i in pruned_set if i in stored and stored[i].ci_low > value.ci_high)
    )
    return Verdict(best, pick, ties, beaten, ties and not beaten)
