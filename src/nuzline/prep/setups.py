"""Setups for the prep search: a team with its lead, the one-change neighbours of a setup, the
batched turn-0 states and the dedupe of byte-equal states (plan 4c Task 5; rulings 4 and 9)."""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.state import BattleState
from nuzline.prep.problem import PrepProblem
from nuzline.prep.states import make_states
from nuzline.prep.variants import (
    LEVEL_UP,
    RELEARNER,
    Cost,
    Member,
    _free_changes,
    _relearner_options,
    ai_kill_damage,
    base_member,
    designated_change,
    hp_regions,
    max_hp_of,
    member_cost,
    member_spec,
    prestatus_options,
)
from nuzline.search.rows import row_bytes

__all__ = [
    "MAX_TEAM_SETUPS",
    "Setup",
    "check_team_space",
    "dedupe",
    "neighbours",
    "protected_slots",
    "setup_cost",
    "setup_specs",
    "setup_states",
    "team_count",
    "team_setups",
]

# The default limit on the team phase's (team, lead) setups. Round 0 screens every one of them
# at once and keeps each one's turn-0 state (about 13 KB a setup), so the time and memory grow
# with this count: the user's 11-candidate Roark pool (7,018 setups) spends about 740 s in the
# team phase; 25,000 is about 13 candidates at a team size of 6, roughly the spec's 60-minute
# target for a gym.
MAX_TEAM_SETUPS = 25_000


@dataclass(frozen=True)
class Setup:
    """A team as brought: ``members[0]`` leads (party slot 0), the others follow in pool order
    (ruling 9); the candidates are distinct."""

    members: tuple[Member, ...]


def _arranged(lead: Member, rest: Sequence[Member]) -> Setup:
    return Setup((lead, *sorted(rest, key=lambda m: m.candidate)))


def team_setups(problem: PrepProblem, max_team: int = 6) -> list[Setup]:
    """Every team of ``1..min(max_team, n)`` candidates (``itertools.combinations`` in pool
    order, by size) with every lead, each member its ``base_member``."""
    n = len(problem.candidates)
    base = [base_member(problem, i) for i in range(n)]
    out = []
    for size in range(1, min(max_team, n) + 1):
        for team in itertools.combinations(range(n), size):
            for lead in team:
                out.append(_arranged(base[lead], [base[i] for i in team if i != lead]))
    return out


def team_count(n: int, max_team: int = 6) -> int:
    """How many setups ``team_setups`` lists for ``n`` candidates: ``sum s * C(n, s)`` over
    team sizes ``s`` in ``1..min(max_team, n)``."""
    return sum(s * math.comb(n, s) for s in range(1, min(max_team, n) + 1))


def check_team_space(n: int, max_team: int, limit: int = MAX_TEAM_SETUPS) -> None:
    """``ValueError`` when ``n`` candidates at teams of up to ``max_team`` give more than
    ``limit`` setups (``team_count``), naming the ways to shrink the search."""
    count = team_count(n, max_team)
    if count > limit:
        raise ValueError(
            f"{n} candidates in teams of up to {max_team} give {count:,} (team, lead) setups, "
            f"over the limit of {limit:,} (the search would take hours and many GB of memory): "
            "leave Pokemon out with --exclude NAME, lower --max-team, or raise --max-setups"
        )


def setup_specs(
    problem: PrepProblem, setup: Setup, tables: Tables | None = None
) -> list[PokemonSpec]:
    """The party specs of ``setup``, lead first (``tables`` as ``member_spec``)."""
    return [member_spec(problem, m, tables) for m in setup.members]


def protected_slots(problem: PrepProblem, setup: Setup) -> tuple[int, ...]:
    """The party slots of ``setup`` holding a protected candidate."""
    return tuple(i for i, m in enumerate(setup.members) if m.candidate in problem.protected)


def setup_cost(problem: PrepProblem, setup: Setup) -> Cost:
    """What the whole setup spends."""
    total = Cost()
    for member in setup.members:
        total = total + member_cost(problem, member)
    return total


def _member_changes(
    tables: Tables,
    problem: PrepProblem,
    member: Member,
    *,
    relearner: str,
    tms: bool,
    items: bool,
    entry_hp: bool,
    levelled: bool = True,
) -> list[Member]:
    """Every one-change variant of ``member`` (ruling 4): a move into a slot no change has
    taken, a bag item or the item taken off (if the item is unchanged), an entry-HP region
    (if the HP is unchanged) and a declared prestatus (if there is none)."""
    if relearner not in ("all", "designated", "none"):
        raise ValueError(f"relearner must be 'all', 'designated' or 'none', not {relearner!r}")
    cand = problem.candidates[member.candidate]
    spec = member_spec(problem, member, tables)
    out: list[Member] = []

    def add_moves(move: int, source: int) -> None:
        for change in _free_changes(spec, member, move, source):
            out.append(dataclasses.replace(member, moves=(*member.moves, change)))

    if levelled:
        for _, move in problem.levelled[member.candidate]:
            add_moves(move, LEVEL_UP)
    if relearner == "designated":
        change = designated_change(tables, problem, member.candidate)
        if (
            change is not None
            and change.slot not in {c.slot for c in member.moves}
            and not any(c.source == RELEARNER for c in member.moves)
        ):
            out.append(dataclasses.replace(member, moves=(*member.moves, change)))
    elif relearner == "all":
        for move in _relearner_options(problem, member.candidate):
            add_moves(move, RELEARNER)
    if tms:
        stock = dict(problem.resources.tms)
        for item, move in cand.tms:
            if item in problem.resources.hms or item in stock:
                add_moves(move, item)
    if items and member.item == 0:
        if spec.item != 0:
            out.append(dataclasses.replace(member, item=-1))
        out.extend(
            dataclasses.replace(member, item=item)
            for item, _ in problem.resources.held
            if item != spec.item
        )
    if entry_hp and member.hp is None:
        kill = ai_kill_damage(
            tables, spec, problem.opp, problem.start_condition, problem.trainer.items
        )
        out.extend(
            dataclasses.replace(member, hp=top)
            for top in hp_regions(tables, spec, max_hp_of(tables, spec), kill)[1:]
        )
    if member.status == 0:
        out.extend(
            dataclasses.replace(member, status=s)
            for s in prestatus_options(tables, spec, problem.prestatus, problem.blocking_abilities)
        )
    return out


def neighbours(
    tables: Tables,
    problem: PrepProblem,
    setup: Setup,
    *,
    pruned: Collection[Member] = (),
    relearner: str = "all",
    tms: bool = True,
    items: bool = True,
    entry_hp: bool = True,
    levelled: bool = True,
) -> list[Setup]:
    """The setups one change from ``setup`` (ruling 4): one change to one member, or another
    member as lead. Only setups that fit the unreserved resources, none whose changed member is
    in ``pruned``, no duplicates and not ``setup`` itself."""
    seen = {setup}
    out: list[Setup] = []

    def add(candidate: Setup) -> None:
        if candidate not in seen and setup_cost(problem, candidate).fits(problem.resources):
            seen.add(candidate)
            out.append(candidate)

    for i, member in enumerate(setup.members):
        for variant in _member_changes(
            tables,
            problem,
            member,
            relearner=relearner,
            tms=tms,
            items=items,
            entry_hp=entry_hp,
            levelled=levelled,
        ):
            if variant not in pruned:
                members = list(setup.members)
                members[i] = variant
                add(Setup(tuple(members)))
    for j in range(1, len(setup.members)):
        rest = [m for k, m in enumerate(setup.members) if k != j]
        add(_arranged(setup.members[j], rest))
    return out


def setup_states(tables: Tables, problem: PrepProblem, setups: Sequence[Setup]) -> BattleState:
    """The turn-0 states of ``setups`` (``make_states``) with ``budget=problem.deaths`` and each
    setup's protected slots."""
    return make_states(
        tables,
        [setup_specs(problem, s, tables) for s in setups],
        problem.opp,
        problem.start_condition,
        problem.trainer.items,
        budget=problem.deaths,
        protected=[protected_slots(problem, s) for s in setups],
        terrain=problem.trainer.terrain,
    )


def dedupe(states: BattleState, setups: Sequence[Setup], costs: Sequence[float]) -> list[int]:
    """The indices kept when several setups give byte-equal ``states`` rows: of each group the
    cheapest, then the lowest index; ascending."""
    n = len(setups)
    if len(costs) != n:
        raise ValueError(f"{len(costs)} costs for {n} setups")
    rows = row_bytes(states, n) if n else np.zeros((0, 0), np.uint8)
    best: dict[bytes, int] = {}
    for i in range(n):
        key = rows[i].tobytes()
        if key not in best or costs[i] < costs[best[key]]:
            best[key] = i
    return sorted(best.values())
