"""The prep search funnel and its anytime budget (plan 4c Task 7; spec section 6).

``run_prep`` screens every (team, lead) of base members in one batch (the team phase), halves
them, expands the survivors by every one-change neighbour (``variant_rounds`` expansions, each
screened in one batch and halved over the neighbours plus their parents), then solves the
finalists of the requested rung with Stage B and deepens the leader (rulings 1-3).

Halving (ruling 2): round ``r`` tops every setup of the round up to ``T_r = screen_n *
(2**(r + 1) - 1)`` playouts (grouped by how many each still needs, one ``screen`` call per
group). A phase keeps one survivor set ``S_j`` per rung ``j`` in ``0..3``; the round's set is
their union; afterwards ``S_k`` (``k`` = the requested rung, ``problem.deaths``) keeps its top
``max(keep, ceil(|S_k| / 2))`` by rung-``k`` point estimate and every other ``S_j`` its top
``max(ladder_keep, ceil(|S_j| / 2))`` by rung ``j``, ties to the lower setup index (the order
the setups were first screened in). The phase ends once ``|S_k| <= keep``.

The answer at any moment is the best so far: the ``prep_order`` leader of ``S_k`` as the last
completed round screened it (before that round's halving), or the best finalist. A unit of work
is a halving round or a Stage B solve; ``progress`` is called after each, a ``KeyboardInterrupt``
raised in one (or by ``progress``) ends the run with ``stopped="interrupted"``, and
``budget_seconds`` is checked before every unit but the very first (ruling 11).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import jax
import numpy as np

from nuzline.ai.policy import UnsupportedAi
from nuzline.data.tables import Names, Tables
from nuzline.prep.problem import PrepProblem
from nuzline.prep.screen import Scored, ScreenStats, prep_order, scored, screen
from nuzline.prep.setups import (
    Setup,
    dedupe,
    neighbours,
    setup_cost,
    setup_states,
    team_setups,
)
from nuzline.prep.variants import DEFAULT_WEIGHTS, Member, prune, single_changes
from nuzline.search import deaths
from nuzline.search.rows import row_bytes
from nuzline.search.stage_b import HiddenLegality, SolveResult, SolveSettings, solve

__all__ = [
    "Finalist",
    "PrepResult",
    "PrepSettings",
    "Progress",
    "Rung",
    "SearchStats",
    "phase_id",
    "run_prep",
    "setup_label",
]

RUNGS = deaths.MAX_DEATH_BUDGET + 1  # the ladder: 0..3 deaths


@dataclass(frozen=True)
class PrepSettings:
    """The funnel's knobs. ``screen_n``: playouts of round 0 (even); ``team_keep`` /
    ``variant_keep`` / ``ladder_keep``: the survivor floors of ruling 2; ``variant_rounds``:
    expansions; ``finalists``: Stage B solves at depth 1; ``max_depth``: the leader's deepest
    solve (0 = screen only); ``precision``: the Stage B target half-width, or
    ``finalist_playouts`` a fixed ``budget_playouts`` instead; ``max_team`` .. ``entry_hp``:
    the variant space; ``budget_seconds``: the anytime limit (ruling 11); ``weights``: the
    resource cost weights of the tie rule (ruling 7)."""

    screen_n: int = 8
    team_keep: int = 32
    variant_rounds: int = 2
    variant_keep: int = 8
    ladder_keep: int = 4
    finalists: int = 3
    max_depth: int = 2
    precision: float | None = 0.015
    finalist_playouts: int | None = None
    max_team: int = 6
    relearner: str = "all"
    tms: bool = True
    items: bool = True
    entry_hp: bool = True
    max_turns: int = 100
    budget_seconds: float | None = None
    weights: tuple[tuple[str, float], ...] = tuple(DEFAULT_WEIGHTS.items())

    def __post_init__(self) -> None:
        if self.screen_n < 2 or self.screen_n % 2:
            raise ValueError(f"screen_n must be even and positive, not {self.screen_n}")
        for name in ("team_keep", "variant_keep", "ladder_keep"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, not {getattr(self, name)}")
        if self.variant_rounds < 0:
            raise ValueError(f"variant_rounds must not be negative, not {self.variant_rounds}")
        if self.finalists < 1:
            raise ValueError(f"finalists must be at least 1, not {self.finalists}")
        if self.max_depth < 0:
            raise ValueError(f"max_depth must not be negative, not {self.max_depth}")
        if self.max_team < 1:
            raise ValueError(f"max_team must be at least 1, not {self.max_team}")


@dataclass(frozen=True)
class Finalist:
    """A Stage B solve of one setup."""

    setup: Setup
    result: SolveResult


@dataclass(frozen=True)
class Rung:
    """The best setup for ``k`` allowed deaths: its probability and 95 % interval, and whether
    they come from the Stage B ``"plan"`` or the Stage A ``"screen"``."""

    k: int
    setup: Setup
    p: float
    ci_low: float
    ci_high: float
    source: str


@dataclass
class SearchStats:
    """What the run did. ``phase``: the last phase finished (``"teams"``, ``"variants"``,
    ``"finalists"``, ``""`` for none); ``stopped``: ``None``, ``"budget"`` or
    ``"interrupted"``; ``seconds`` and ``phase_seconds`` are timings and not compared."""

    setups_screened: int
    playouts: int
    team_rounds: int
    variant_rounds: int
    variant_halvings: int
    phase: str
    stopped: str | None
    seconds: float = field(compare=False)
    phase_seconds: tuple[tuple[str, float], ...] = field(default=(), compare=False)


@dataclass
class PrepResult:
    """The answer: ``chosen`` setup, its ``screen`` numbers, the Stage B ``plan`` (``None``
    for a screen-only answer), the odds ``ladder`` for 0-3 deaths, the death ``distribution``
    (the plan leader's, else the screen's), the ``finalists``, the setups left out
    (``excluded``, named) and the variants the prune dropped with what dominated them."""

    chosen: Setup
    screen: ScreenStats
    plan: SolveResult | None
    ladder: tuple[Rung, ...]
    distribution: deaths.DeathDistribution
    finalists: tuple[Finalist, ...]
    excluded: tuple[str, ...]
    pruned: tuple[tuple[Member, Member], ...]
    stats: SearchStats


@dataclass(frozen=True)
class Progress:
    """After one unit of work. ``phase``: ``"teams"``, ``"variants <v>"`` (``v`` 1-based) or
    ``"finalists"``; ``round``: the halving round, or the finalist index; ``alive``: ``|S_k|``
    (the number of finalists in ``"finalists"``); ``leader``: the best setup so far."""

    phase: str
    round: int
    alive: int
    elapsed: float
    leader: Setup


_PHASES = {"finalists": 0, "teams": 1, "deep": 2}


def phase_id(phase: str, v: int = 0) -> int:
    """The key-derivation id of a phase: ``finalists`` 0, ``teams`` 1, ``deep`` 2, variant
    expansion ``v`` (0-based) ``3 + v``; disjoint for every ``v >= 0``."""
    if phase == "variants":
        if v < 0:
            raise ValueError(f"expansion {v} is negative")
        return 3 + v
    try:
        return _PHASES[phase]
    except KeyError:
        raise ValueError(f"unknown phase {phase!r}") from None


def setup_label(names: Names, problem: PrepProblem, setup: Setup) -> str:
    """The setup in words, lead first: each Pokemon with the moves, item, HP and status it
    was changed to."""
    parts = []
    for m in setup.members:
        changes = [names.move_names[c.move] for c in m.moves]
        if m.item > 0:
            changes.append(names.item_names[m.item])
        elif m.item < 0:
            changes.append("item off")
        if m.hp is not None:
            changes.append(f"{m.hp} HP")
        if m.status:
            changes.append(f"status {m.status}")
        label = problem.candidates[m.candidate].label
        parts.append(f"{label} ({', '.join(changes)})" if changes else label)
    return " / ".join(parts)


class _Stop(Exception):
    """The time budget ran out before a unit."""


class _Funnel:
    def __init__(
        self,
        tables: Tables,
        names: Names,
        problem: PrepProblem,
        settings: PrepSettings,
        key,
        progress: Callable[[Progress], None] | None,
        clock: Callable[[], float],
    ) -> None:
        self.tables, self.names, self.problem, self.settings = tables, names, problem, settings
        self.key, self.progress, self.clock = key, progress, clock
        self.k = problem.deaths
        self.weights = dict(settings.weights)
        self.start = clock()
        self.mark = self.start
        self.setups: list[Setup] = []
        self.index: dict[Setup, int] = {}
        self.known: set[Setup] = set()  # registered or dropped as a duplicate
        self.by_bytes: dict[bytes, int] = {}
        self.rows: list = []  # turn-0 state per registered setup (numpy, one row)
        self.cost: list[float] = []
        self.stats: dict[int, ScreenStats] = {}
        self.excluded: dict[int, str] = {}
        self.playouts = 0
        self.team_rounds = self.variant_rounds = self.variant_halvings = 0
        self.units = 0
        self.phase = ""
        self.seconds: dict[str, float] = {}
        self.last_pools: list[list[int]] = [[] for _ in range(RUNGS)]
        self.finalists: list[Finalist] = []
        self.pruned: list[tuple[Member, Member]] = []
        self.dropped: set[Member] = set()

    # ------------------------------------------------------------------ registry

    def register(self, setups: Sequence[Setup]) -> list[int]:
        """Add the not yet known ``setups`` (byte-equal turn-0 states: the cheapest only) and
        return their indices."""
        new = [s for s in dict.fromkeys(setups) if s not in self.known]
        if not new:
            return []
        self.known.update(new)
        states = setup_states(self.tables, self.problem, new)
        costs = [setup_cost(self.problem, s).weighted(self.weights) for s in new]
        keys = row_bytes(states, len(new))
        out = []
        for i in dedupe(states, new, costs):
            blob = keys[i].tobytes()
            if blob in self.by_bytes:
                continue
            idx = len(self.setups)
            self.by_bytes[blob] = idx
            self.setups.append(new[i])
            self.index[new[i]] = idx
            self.rows.append(jax.tree.map(lambda f, i=i: np.asarray(f)[i], states))
            self.cost.append(costs[i])
            self.stats[idx] = ScreenStats((0,) * len(deaths.BUCKETS), 0, 0.0, 0.0, False)
            out.append(idx)
        return out

    # ------------------------------------------------------------------ ranking

    def row(self, i: int, j: int) -> Scored:
        return scored(self.stats[i], j, self.cost[i])

    def ranked(self, indices: Sequence[int], j: int) -> list[int]:
        """``indices`` best first on rung ``j`` by the tie rule."""
        order = prep_order([self.row(i, j) for i in indices])
        return [indices[o] for o in order]

    def top(self, indices: Sequence[int], j: int, count: int) -> list[int]:
        """The ``count`` best by rung-``j`` point estimate (ties: the lower index), ascending."""
        best = sorted(indices, key=lambda i: (-self.stats[i].rung(j), i))
        return sorted(best[:count])

    # ------------------------------------------------------------------ units

    def elapsed(self) -> float:
        return self.clock() - self.start

    def before_unit(self) -> None:
        budget = self.settings.budget_seconds
        if self.units and budget is not None and self.elapsed() >= budget:
            raise _Stop

    def time_left(self) -> float | None:
        budget = self.settings.budget_seconds
        return None if budget is None else max(0.0, budget - self.elapsed())

    def unit_done(self, phase: str, bucket: str, round_: int, alive: int) -> None:
        self.units += 1
        now = self.clock()
        self.seconds[bucket] = self.seconds.get(bucket, 0.0) + now - self.mark
        self.mark = now
        if self.progress is not None:
            self.progress(Progress(phase, round_, alive, now - self.start, self.leader()))

    def leader(self) -> Setup:
        if self.finalists:
            return self.finalists[0].setup
        return self.setups[self.ranked(self.last_pools[self.k], self.k)[0]]

    def top_up(self, members: Sequence[int], target: int, key) -> None:
        """Screen every member up to ``target`` playouts, one call per distinct shortfall; the
        stats are updated only when all calls are done."""
        groups: dict[int, list[int]] = {}
        for i in members:
            need = target - self.stats[i].n
            if need > 0:
                groups.setdefault(need, []).append(i)
        results: dict[int, ScreenStats] = {}
        for need in sorted(groups):
            idxs = groups[need]
            states = jax.tree.map(lambda *fs: np.stack(fs), *[self.rows[i] for i in idxs])
            out = screen(self.tables, states, need, key, self.settings.max_turns)
            results.update(zip(idxs, out, strict=True))
            self.playouts += need * len(idxs)
        for i, s in results.items():
            self.stats[i] = self.stats[i].merge(s)
            if self.stats[i].refused and i not in self.excluded:
                self.excluded[i] = (
                    f"{setup_label(self.names, self.problem, self.setups[i])}: "
                    "a playout was refused (unsupported mechanic)"
                )

    # ------------------------------------------------------------------ phases

    def halving(
        self, phase: str, bucket: str, pid: int, pools: list[list[int]], keep: int
    ) -> list[list[int]]:
        """Successive halving over the survivor sets ``pools`` (module docstring); returns
        the final sets."""
        k, st = self.k, self.settings
        r = 0
        while True:
            self.before_unit()
            members = sorted(set().union(*pools))
            rkey = jax.random.fold_in(jax.random.fold_in(self.key, pid), r)
            self.top_up(members, st.screen_n * (2 ** (r + 1) - 1), rkey)
            pools = [[i for i in p if i not in self.excluded] for p in pools]
            if not pools[k]:
                self.raise_all_refused()
            self.last_pools = [list(p) for p in pools]
            pools = [
                self.top(p, j, max(keep if j == k else st.ladder_keep, ceil(len(p) / 2)))
                for j, p in enumerate(pools)
            ]
            if bucket == "teams":
                self.team_rounds += 1
            else:
                self.variant_halvings += 1
            self.unit_done(phase, bucket, r, len(pools[k]))
            r += 1
            if len(pools[k]) <= keep:
                return pools

    def raise_all_refused(self) -> None:
        reasons = "; ".join(list(self.excluded.values())[:3])
        raise ValueError(f"every setup refused: {reasons}")

    def team_phase(self) -> list[list[int]]:
        st = self.settings
        everyone = self.register(team_setups(self.problem, st.max_team))
        pools = self.halving(
            "teams", "teams", phase_id("teams"), [list(everyone)] * RUNGS, st.team_keep
        )
        self.phase = "teams"
        return pools

    def expansion(self, v: int, pools: list[list[int]]) -> list[list[int]] | None:
        """One variant expansion; ``None`` when the survivors have no new neighbour."""
        st = self.settings
        fresh: list[Setup] = []
        for i in pools[self.k]:
            fresh.extend(
                neighbours(
                    self.tables,
                    self.problem,
                    self.setups[i],
                    pruned=self.dropped,
                    relearner=st.relearner,
                    tms=st.tms,
                    items=st.items,
                    entry_hp=st.entry_hp,
                )
            )
        new = self.register(fresh)
        if not new:
            return None
        out = self.halving(
            f"variants {v + 1}",
            "variants",
            phase_id("variants", v),
            [sorted({*p, *new}) for p in pools],
            st.variant_keep,
        )
        self.variant_rounds += 1
        return out

    def stage_b(self, setup: Setup, key, max_depth: int) -> SolveResult | None:
        st = self.settings
        row = self.rows[self.index[setup]]
        settings = SolveSettings(
            max_depth=max_depth,
            budget_playouts=st.finalist_playouts,
            budget_seconds=self.time_left(),
            max_turns=st.max_turns,
            precision=None if st.finalist_playouts is not None else st.precision,
        )
        try:
            return solve(self.tables, self.names, row, key, settings)
        except (UnsupportedAi, HiddenLegality) as e:
            idx = self.index[setup]
            self.excluded[idx] = f"{setup_label(self.names, self.problem, setup)}: {e}"
            return None

    def finalist_row(self, f: Finalist) -> Scored:
        top = f.result.ranking[0]
        i = self.index[f.setup]
        stats = self.stats[i]
        return Scored(
            top.estimate,
            top.ci_low,
            top.ci_high,
            top.e_deaths,
            self.cost[i],
            stats.mean_hp,
            stats.mean_pp,
        )

    def finalists_phase(self, pool: list[int]) -> None:
        st = self.settings
        picks = self.ranked(pool, self.k)[: st.finalists]
        fkey = jax.random.fold_in(self.key, phase_id("finalists"))
        for n, i in enumerate(picks):
            self.before_unit()
            result = self.stage_b(self.setups[i], jax.random.fold_in(fkey, n), 1)
            if result is not None and result.ranking:
                self.playouts += result.playouts
                self.finalists.append(Finalist(self.setups[i], result))
                rows = [self.finalist_row(f) for f in self.finalists]
                self.finalists = [self.finalists[o] for o in prep_order(rows)]
            self.unit_done("finalists", "finalists", n, len(picks))
        if st.max_depth >= 2 and self.finalists:
            self.before_unit()
            lead = self.finalists[0]
            dkey = jax.random.fold_in(jax.random.fold_in(self.key, phase_id("deep")), 0)
            deep = self.stage_b(lead.setup, dkey, st.max_depth)
            if deep is not None and deep.ranking:
                self.playouts += deep.playouts
                self.finalists[0] = Finalist(lead.setup, deep)
            self.unit_done("finalists", "finalists", 0, len(picks))
        self.phase = "finalists"

    # ------------------------------------------------------------------ result

    def result(self, stopped: str | None) -> PrepResult:
        k = self.k
        pools = self.last_pools
        if self.finalists:
            lead = self.finalists[0]
            chosen = lead.setup
            plan = lead.result
            top = plan.ranking[0]
        else:
            chosen = self.setups[self.ranked(pools[k], k)[0]]
            plan, top = None, None
        ci = self.index[chosen]
        stats = self.stats[ci]
        ladder = []
        for j in range(RUNGS):
            if j == k and top is not None:
                ladder.append(Rung(j, chosen, top.estimate, top.ci_low, top.ci_high, "plan"))
                continue
            best = self.ranked(pools[j] or pools[k], j)[0]
            low, high = self.stats[best].interval(j)
            ladder.append(Rung(j, self.setups[best], self.stats[best].rung(j), low, high, "screen"))
        dist = top.distribution if top is not None and top.distribution is not None else None
        seconds = self.elapsed()
        run_stats = SearchStats(
            setups_screened=sum(1 for s in self.stats.values() if s.n),
            playouts=self.playouts,
            team_rounds=self.team_rounds,
            variant_rounds=self.variant_rounds,
            variant_halvings=self.variant_halvings,
            phase=self.phase,
            stopped=stopped,
            seconds=seconds,
            phase_seconds=tuple(self.seconds.items()),
        )
        return PrepResult(
            chosen=chosen,
            screen=stats,
            plan=plan,
            ladder=tuple(ladder),
            distribution=dist if dist is not None else stats.distribution,
            finalists=tuple(self.finalists),
            excluded=tuple(self.excluded.values()),
            pruned=tuple(self.pruned),
            stats=run_stats,
        )


def run_prep(
    tables: Tables,
    names: Names,
    problem: PrepProblem,
    settings: PrepSettings,
    key,
    *,
    checkpoint: Path | None = None,
    resume: Path | None = None,
    progress: Callable[[Progress], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> PrepResult:
    """Run the funnel (module docstring) and return the best answer reached.
    ``ValueError`` when every setup is refused ("every setup refused: ...")."""
    if checkpoint is not None or resume is not None:
        raise NotImplementedError("checkpoint and resume arrive with plan 4c Task 8")
    funnel = _Funnel(tables, names, problem, settings, key, progress, clock)
    st = settings
    for c in range(len(problem.candidates)):
        variants = single_changes(
            tables,
            problem,
            c,
            relearner=st.relearner,
            tms=st.tms,
            items=st.items,
            entry_hp=st.entry_hp,
        )
        funnel.pruned.extend(prune(tables, problem, variants)[1])
    funnel.dropped = {d for d, _ in funnel.pruned}
    stopped = None
    try:
        pools = funnel.team_phase()
        for v in range(st.variant_rounds):
            nxt = funnel.expansion(v, pools)
            if nxt is None:
                break
            pools = nxt
        if st.variant_rounds:
            funnel.phase = "variants"
        if st.max_depth >= 1:
            funnel.finalists_phase(pools[funnel.k])
    except _Stop:
        stopped = "budget"
    except KeyboardInterrupt:
        if not funnel.units:
            raise
        stopped = "interrupted"
    return funnel.result(stopped)
