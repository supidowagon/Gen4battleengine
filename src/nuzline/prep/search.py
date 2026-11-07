"""The prep search funnel and its anytime budget (plan 4c Task 7; spec section 6).

``run_prep`` screens every (team, lead) of base members in one batch (the team phase), halves
them, expands the survivors by every one-change neighbour (``variant_rounds`` expansions, each
screened in one batch and halved over the neighbours plus their parents), then solves the
finalists of the requested rung with Stage B and deepens the leader (rulings 1-3). The
finalists come from a fresh race (``race_playouts`` new playouts per final survivor of ``S_k``,
ranked by those alone and never merged into the halving stats), not from the accumulated
stats, which carry the luck that let a setup survive every cut.

Halving (ruling 2): round ``r`` tops every competing setup up to ``T_r = screen_n *
(2**(r + 1) - 1)`` playouts (grouped by how many each still needs, one ``screen`` call per
group). A phase keeps one survivor set ``S_j`` per rung ``j`` in ``0..3``; the round's set is
their union; afterwards ``S_k`` (``k`` = the requested rung, ``problem.deaths``) keeps its top
``max(keep, ceil(|S_k| / 2))`` by rung ``k`` and every other ``S_j`` its top
``max(ladder_keep, ceil(|S_j| / 2))`` by rung ``j``. The order (``halving_key``): the point
estimate first, then, among setups the playouts cannot tell apart (an easy fight saturates at
``n/n`` for the first rounds), ``deaths.tie_deaths``, then the lower cost, the seeded tie rank
(``tie_rank``) last. Mean HP and PP left are reported, not ranked: measured over the whole
party they favour bigger teams. The rounds end
once ``|S_k| <= keep``. In the team phase every setup competes at the same count. In a variant
expansion the parents (the previous survivors, already screened to hundreds or thousands of
playouts) are not halved at first: the new neighbours halve alone, each rung's neighbour set
as above. Then the *merged stage*: rung ``k``'s parents and surviving neighbours are topped up
together to the first halving count at or above the parents' largest count
(``ladder_round``; round keys salted by ``MERGED``) and cut to the floor there, with the exact
ties at the edge kept and screened further while ``T_r < tie_playouts``; the ladder rungs (screen-only
answers) are cut to their floors in the first merged round at the counts they have. Every cut
of rung ``k`` thus compares setups at one playout count: a parent's count would otherwise
decide a cut against a neighbour's handful (a lower confidence bound keeps tight parents over
a better ``n/n`` neighbour; the point estimate drops them for lucky ``n/n`` neighbours).

The answer at any moment is the best so far: the ``prep_order`` leader of ``S_k`` as the last
completed round screened it (before that round's halving), after the race its first pick, or
the best finalist. A unit of work is a halving round, the race or a Stage B solve;
``progress`` is called after each, a ``KeyboardInterrupt`` raised in one (or by ``progress``)
ends the run with ``stopped="interrupted"``, and ``budget_seconds`` is checked before every
unit but the first of this invocation (ruling 11).

Checkpoint (ruling 12; ``prep.checkpoint``): after every unit, before ``progress`` is called, the
whole funnel is written: the registered setups with their accumulated stats, the exclusions,
the counters, the finalists with their Stage B results and a *position*: ``{"phase": "teams"}``
or ``{"phase": "variants", "v": v}`` with ``"r"`` (the next halving round), ``"pools"`` (each
rung's surviving neighbours) and ``"parents"`` (each rung's parents); in the merged stage
``"stage": "merged"`` with ``"pools"`` the merged sets and no ``"parents"``; or
``{"phase": "finalists"}`` with ``"pool"``, ``"picks"`` (the finalists chosen from it, solved
as saved, never re-ranked), ``"next"`` (the next pick to solve) and ``"deep"`` (the leader's
deepening is done). Every random key comes from
the seed, the phase and the round, so a resumed run does the remaining units exactly as the
uninterrupted run would. The turn-0 states are rebuilt from the setups, not stored.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import jax
import numpy as np

from nuzline.ai.policy import UnsupportedAi
from nuzline.data.tables import Names, Tables
from nuzline.prep.checkpoint import (
    CHECKPOINT_VERSION,
    fingerprint,
    read_checkpoint,
    setup_from_json,
    setup_to_json,
    stats_from_json,
    stats_to_json,
    write_checkpoint,
)
from nuzline.prep.problem import PrepProblem
from nuzline.prep.screen import (
    Scored,
    ScreenStats,
    cheaper_alternative,
    prep_order,
    scored,
    screen,
)
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
from nuzline.search.stage_b import (
    HiddenLegality,
    SolveResult,
    SolveSettings,
    result_from_dict,
    result_to_dict,
    solve,
)

__all__ = [
    "Finalist",
    "PrepResult",
    "PrepSettings",
    "Progress",
    "Rung",
    "SearchStats",
    "halving_key",
    "phase_id",
    "run_prep",
    "setup_label",
    "tie_rank",
]

RUNGS = deaths.MAX_DEATH_BUDGET + 1  # the ladder: 0..3 deaths
MERGED = 1 << 20  # key salt of a variant expansion's merged stage (disjoint from round keys)


def ladder_round(screen_n: int, n: int) -> int:
    """The first halving round ``r >= 1`` whose ``T_r = screen_n * (2**(r + 1) - 1)`` reaches
    ``n`` playouts."""
    r = 1
    while screen_n * (2 ** (r + 1) - 1) < n:
        r += 1
    return r


def tie_rank(seed: bytes, row: bytes) -> int:
    """A setup's last tie-break: 8 bytes of blake2b over the run seed and its turn-0 state
    bytes. Unbiased by enumeration order, cost or HP, and the same after a resume (rebuilt from
    the setups, never stored)."""
    return int.from_bytes(hashlib.blake2b(seed + row, digest_size=8).digest(), "little")


def halving_key(stats: ScreenStats, j: int, cost: float, tie: int) -> tuple:
    """A halving round's sort key on rung ``j``, best first (module docstring): the point
    estimate, ``deaths.tie_deaths``, the cost, the seeded ``tie`` rank. HP and PP left are not
    keys: measured over the whole party they favour bigger teams."""
    row = scored(stats, j, cost)
    return (-row.score, row.tie_deaths, cost, tie)


@dataclass(frozen=True)
class PrepSettings:
    """The funnel's knobs. ``screen_n``: playouts of round 0 (even); ``team_keep`` /
    ``variant_keep`` / ``ladder_keep``: the survivor floors of ruling 2; ``variant_rounds``:
    expansions; ``finalists``: Stage B solves at depth 1; ``max_depth``: the leader's deepest
    solve (0 = screen only); ``precision``: the Stage B target half-width, or
    ``finalist_playouts`` a fixed ``budget_playouts`` instead; ``max_team`` .. ``levelled``:
    the variant space (``levelled=False``: no free level-up slot replacements);
    ``budget_seconds``: the anytime limit (ruling 11); ``weights``: the resource cost weights,
    the last tie-break, after win chance and deaths (spec section 4, user decision
    2026-11-04); ``tie_playouts``: at a cut reaching a survivor floor, setups tied exactly with
    the last survivor are kept and screened further until the tie breaks or they reach this
    many playouts (0: off; 1016 = ``screen_n`` 8's round 6); ``race_playouts``: before Stage
    B, rung ``k``'s final survivors are screened afresh with this many playouts each and the
    finalists are the top ``finalists`` of that fresh screen alone by ``prep_order`` (0: off,
    the finalists are then picked from the accumulated stats)."""

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
    levelled: bool = True
    max_turns: int = 100
    budget_seconds: float | None = None
    weights: tuple[tuple[str, float], ...] = tuple(DEFAULT_WEIGHTS.items())
    tie_playouts: int = 1016
    race_playouts: int = 2040

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
        if self.tie_playouts < 0:
            raise ValueError(f"tie_playouts must not be negative, not {self.tie_playouts}")
        if self.race_playouts < 0 or self.race_playouts % 2:
            raise ValueError(
                f"race_playouts must be even and not negative, not {self.race_playouts}"
            )


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
    ``"race"``, ``"finalists"``, ``""`` for none); ``stopped``: ``None``, ``"budget"`` or
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
    (``excluded``, named), the variants the prune dropped with what dominated them and
    ``cheaper``: a cheaper finalist tied within noise with the answer (display only)."""

    chosen: Setup
    screen: ScreenStats
    plan: SolveResult | None
    ladder: tuple[Rung, ...]
    distribution: deaths.DeathDistribution
    finalists: tuple[Finalist, ...]
    excluded: tuple[str, ...]
    pruned: tuple[tuple[Member, Member], ...]
    stats: SearchStats
    cheaper: Finalist | None = None


@dataclass(frozen=True)
class Progress:
    """After one unit of work. ``phase``: ``"teams"``, ``"variants <v>"`` (``v`` 1-based),
    ``"race"`` or ``"finalists"``; ``round``: the halving round, or the finalist index;
    ``alive``: ``|S_k|`` (with the parents in a variant expansion; the number of setups raced
    in ``"race"``; the number of finalists in ``"finalists"``); ``leader``: the best setup so
    far."""

    phase: str
    round: int
    alive: int
    elapsed: float
    leader: Setup


_PHASES = {"finalists": 0, "teams": 1, "deep": 2, "race": 1 << 21}


def phase_id(phase: str, v: int = 0) -> int:
    """The key-derivation id of a phase: ``finalists`` 0, ``teams`` 1, ``deep`` 2, ``race``
    ``1 << 21``, variant expansion ``v`` (0-based) ``3 + v``; disjoint for every ``v >= 0``
    below ``(1 << 21) - 3``."""
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
        sink: Path | None = None,
        print_: dict[str, str] | None = None,
    ) -> None:
        self.tables, self.names, self.problem, self.settings = tables, names, problem, settings
        self.key, self.progress, self.clock = key, progress, clock
        self.sink, self.fingerprint = sink, print_
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
        self.seed = _seed_bytes(key)
        self.tie: list[int] = []  # seeded tie rank per registered setup (``tie_rank``)
        self.stats: dict[int, ScreenStats] = {}
        self.excluded: dict[int, str] = {}
        self.playouts = 0
        self.team_rounds = self.variant_rounds = self.variant_halvings = 0
        self.units = 0  # of this invocation (the budget exempts the first)
        self.prior = 0.0  # seconds spent by earlier invocations (resume)
        self.pos: dict | None = None  # where to continue (module docstring)
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
            self.tie.append(tie_rank(self.seed, blob))
            self.stats[idx] = ScreenStats((0,) * len(deaths.BUCKETS), 0, 0.0, 0.0, False)
            out.append(idx)
        return out

    # ------------------------------------------------------------------ ranking

    def row(self, i: int, j: int) -> Scored:
        return scored(self.stats[i], j, self.cost[i], self.tie[i])

    def ranked(self, indices: Sequence[int], j: int) -> list[int]:
        """``indices`` best first on rung ``j`` by the tie rule."""
        order = prep_order([self.row(i, j) for i in indices])
        return [indices[o] for o in order]

    def top(self, indices: Sequence[int], j: int, count: int, ties: bool = False) -> list[int]:
        """The ``count`` best on rung ``j`` by ``halving_key``, ascending. With ``ties``, every
        setup tied with the ``count``-th on (score, deaths) is kept as well."""

        def key(i):
            return halving_key(self.stats[i], j, self.cost[i], self.tie[i])

        best = sorted(indices, key=key)
        end = count
        if ties and 0 < count < len(best):
            edge = key(best[count - 1])[:2]
            while end < len(best) and key(best[end])[:2] == edge:
                end += 1
        return sorted(best[:end])

    # ------------------------------------------------------------------ units

    def elapsed(self) -> float:
        """Seconds of this invocation (the time budget limits these)."""
        return self.clock() - self.start

    def total(self) -> float:
        """Seconds over all invocations of this run."""
        return self.prior + self.elapsed()

    def before_unit(self) -> None:
        budget = self.settings.budget_seconds
        if self.units and budget is not None and self.elapsed() >= budget:
            raise _Stop

    def time_left(self) -> float | None:
        budget = self.settings.budget_seconds
        return None if budget is None else max(0.0, budget - self.elapsed())

    def unit_done(self, phase: str, bucket: str, round_: int, alive: int, pos: dict) -> None:
        self.units += 1
        now = self.clock()
        self.seconds[bucket] = self.seconds.get(bucket, 0.0) + now - self.mark
        self.mark = now
        self.pos = pos
        if self.sink is not None:
            write_checkpoint(self.sink, self.snapshot())
        if self.progress is not None:
            self.progress(Progress(phase, round_, alive, self.total(), self.leader()))

    # ------------------------------------------------------------------ checkpoint

    def snapshot(self) -> dict:
        """The whole funnel as a checkpoint document (module docstring)."""
        funnel = {
            "setups": [setup_to_json(s) for s in self.setups],
            "stats": [stats_to_json(self.stats[i]) for i in range(len(self.setups))],
            "excluded": {str(i): why for i, why in self.excluded.items()},
            "playouts": self.playouts,
            "team_rounds": self.team_rounds,
            "variant_rounds": self.variant_rounds,
            "variant_halvings": self.variant_halvings,
            "phase": self.phase,
            "seconds": dict(self.seconds),
            "elapsed": self.total(),
            "last_pools": self.last_pools,
            "finalists": [
                {"setup": setup_to_json(f.setup), "result": result_to_dict(f.result)}
                for f in self.finalists
            ],
            "pos": self.pos,
        }
        return {"version": CHECKPOINT_VERSION, "fingerprint": self.fingerprint, "funnel": funnel}

    def restore(self, funnel: dict) -> None:
        """Load a checkpoint's ``funnel``: re-register the saved setups in order (their turn-0
        states are rebuilt in one batch) and set every counter and the position."""
        setups = [setup_from_json(s) for s in funnel["setups"]]
        if setups:
            states = setup_states(self.tables, self.problem, setups)
            keys = row_bytes(states, len(setups))
            for i, setup in enumerate(setups):
                self.by_bytes[keys[i].tobytes()] = i
                self.rows.append(jax.tree.map(lambda f, i=i: np.asarray(f)[i], states))
                self.cost.append(setup_cost(self.problem, setup).weighted(self.weights))
                self.tie.append(tie_rank(self.seed, keys[i].tobytes()))
        self.setups = setups
        self.index = {s: i for i, s in enumerate(setups)}
        self.known = set(setups)
        self.stats = {i: stats_from_json(d) for i, d in enumerate(funnel["stats"])}
        self.excluded = {int(i): why for i, why in funnel["excluded"].items()}
        self.playouts = funnel["playouts"]
        self.team_rounds = funnel["team_rounds"]
        self.variant_rounds = funnel["variant_rounds"]
        self.variant_halvings = funnel["variant_halvings"]
        self.phase = funnel["phase"]
        self.seconds = dict(funnel["seconds"])
        self.prior = funnel["elapsed"]
        self.last_pools = [list(p) for p in funnel["last_pools"]]
        self.finalists = [
            Finalist(setup_from_json(f["setup"]), result_from_dict(f["result"]))
            for f in funnel["finalists"]
        ]
        self.pos = funnel["pos"]

    def screen_leader(self) -> int:
        """The screen-only answer's index: after the race (the position holds its ``picks``)
        its first pick still in the run, else the ``prep_order`` leader of ``S_k`` on the
        accumulated stats."""
        pos = self.pos
        if pos is not None and pos.get("phase") == "finalists" and "picks" in pos:
            picks = [i for i in pos["picks"] if i not in self.excluded]
            if picks:
                return picks[0]
        return self.ranked(self.last_pools[self.k], self.k)[0]

    def leader(self) -> Setup:
        if self.finalists:
            return self.finalists[0].setup
        return self.setups[self.screen_leader()]

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
            if self.stats[i].refused:
                self.refuse(i)

    def refuse(self, i: int) -> None:
        """Exclude setup ``i``: one of its playouts was refused."""
        if i not in self.excluded:
            self.excluded[i] = (
                f"{setup_label(self.names, self.problem, self.setups[i])}: "
                "a playout was refused (unsupported mechanic)"
            )

    def race(self, pool: list[int]) -> list[int]:
        """The finalist race (one unit): every non-excluded member of ``pool`` is screened with
        ``race_playouts`` fresh playouts in one call, and the picks are the top ``finalists`` of
        that fresh screen alone by ``prep_order``. The fresh stats are counted in the playouts
        but never merged into the halving stats (they would carry the luck that let a setup
        survive the cuts); the position saves the picks, so a resume never re-races."""
        self.before_unit()
        st = self.settings
        members = [i for i in pool if i not in self.excluded]
        fresh: dict[int, ScreenStats] = {}
        if members:
            states = jax.tree.map(lambda *fs: np.stack(fs), *[self.rows[i] for i in members])
            rkey = jax.random.fold_in(self.key, phase_id("race"))
            out = screen(self.tables, states, st.race_playouts, rkey, st.max_turns)
            self.playouts += st.race_playouts * len(members)
            for i, s in zip(members, out, strict=True):
                if s.refused:
                    self.refuse(i)
                else:
                    fresh[i] = s
        alive = [i for i in pool if i in fresh]
        if not alive:
            self.raise_all_refused()
        rows = [scored(fresh[i], self.k, self.cost[i], self.tie[i]) for i in alive]
        picks = [alive[o] for o in prep_order(rows)][: st.finalists]
        self.last_pools = [[i for i in p if i not in self.excluded] for p in self.last_pools]
        self.phase = "race"
        pos = {"phase": "finalists", "pool": pool, "next": 0, "deep": False, "picks": picks}
        self.unit_done("race", "race", 0, len(pool), pos)
        return picks

    # ------------------------------------------------------------------ phases

    def halving(
        self,
        phase: str,
        bucket: str,
        pid: int,
        pools: list[list[int]],
        keep: int,
        v: int = 0,
        r: int = 0,
        parents: list[list[int]] | None = None,
        merged: bool = False,
    ) -> list[list[int]]:
        """Successive halving over the survivor sets ``pools`` from round ``r`` (module
        docstring); returns the final sets. ``v``: the expansion (variant phases). With
        ``parents`` (a variant expansion: each rung's parents) ``pools`` are the competing
        neighbours, and the result is each rung's final pick from parents and survivors. Without
        ``parents``, a cut of rung ``k`` to its floor keeps the exact ties at the edge and screens
        them further while the round's target is below ``tie_playouts``. With ``merged`` (a
        variant expansion's merged stage) only rung ``k``'s set is topped up, every set is cut
        to its floor, and the round keys are salted by ``MERGED``."""
        assert not (merged and parents is not None)
        k, st = self.k, self.settings
        floors = [keep if j == k else st.ladder_keep for j in range(RUNGS)]

        def whole(sets: list[list[int]]) -> list[list[int]]:
            if parents is None:
                return sets
            return [sorted({*p, *q}) for p, q in zip(parents, sets, strict=True)]

        while not (r and len(pools[k]) <= keep):  # resumed after its last round: none left
            self.before_unit()
            # the merged stage tops up rung k only: ladder rungs are screen-only answers
            members = sorted(pools[k]) if merged else sorted(set().union(*pools))
            base = jax.random.fold_in(self.key, pid)
            if merged:
                base = jax.random.fold_in(base, MERGED)
            rkey = jax.random.fold_in(base, r)
            t_r = st.screen_n * (2 ** (r + 1) - 1)
            self.top_up(members, t_r, rkey)
            pools = [[i for i in p if i not in self.excluded] for p in pools]
            if not whole(pools)[k]:
                self.raise_all_refused()
            self.last_pools = [list(p) for p in whole(pools)]
            counts = [
                floors[j] if merged else max(floors[j], ceil(len(p) / 2))
                for j, p in enumerate(pools)
            ]
            pools = [
                self.top(
                    p,
                    j,
                    counts[j],
                    ties=(
                        j == k
                        and parents is None
                        and counts[j] == floors[j]
                        and t_r < st.tie_playouts
                    ),
                )
                for j, p in enumerate(pools)
            ]
            if bucket == "teams":
                self.team_rounds += 1
            else:
                self.variant_halvings += 1
            pos = {"phase": bucket, "v": v, "r": r + 1, "pools": pools}
            if parents is not None:
                pos["parents"] = parents
            if merged:
                pos["stage"] = "merged"
            self.unit_done(phase, bucket, r, len(whole(pools)[k]), pos)
            r += 1
        if parents is None:
            return pools
        union = whole(pools)
        if len(union[k]) <= keep:
            return [self.top(p, j, floors[j]) for j, p in enumerate(union)]
        start = ladder_round(st.screen_n, max(self.stats[i].n for i in union[k]))
        return self.halving(phase, bucket, pid, union, keep, v, start, merged=True)

    def raise_all_refused(self) -> None:
        reasons = "; ".join(list(self.excluded.values())[:3])
        raise ValueError(f"every setup refused: {reasons}")

    def team_phase(self, resume: dict | None) -> list[list[int]]:
        st = self.settings
        if resume is None:
            everyone = self.register(team_setups(self.problem, st.max_team))
            pools, r = [list(everyone)] * RUNGS, 0
        else:
            pools, r = resume["pools"], resume["r"]
        pools = self.halving("teams", "teams", phase_id("teams"), pools, st.team_keep, r=r)
        self.phase = "teams"
        return pools

    def expansion(
        self, v: int, pools: list[list[int]], resume: dict | None = None
    ) -> list[list[int]] | None:
        """One variant expansion; ``None`` when the survivors have no new neighbour. With
        ``resume`` (the position saved inside this expansion) it continues at its next round
        over ``resume["pools"]``."""
        st = self.settings
        if resume is not None:
            out = self.halving(
                f"variants {v + 1}",
                "variants",
                phase_id("variants", v),
                resume["pools"],
                st.variant_keep,
                v,
                resume["r"],
                None if resume.get("stage") == "merged" else resume["parents"],
                merged=resume.get("stage") == "merged",
            )
            self.variant_rounds += 1
            return out
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
                    levelled=st.levelled,
                )
            )
        new = self.register(fresh)
        if not new:
            return None
        out = self.halving(
            f"variants {v + 1}",
            "variants",
            phase_id("variants", v),
            [list(new)] * RUNGS,
            st.variant_keep,
            v,
            parents=[list(p) for p in pools],
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
            # the answer is built from last_pools, which were filtered before Stage B ran
            self.last_pools = [[i for i in p if i != idx] for p in self.last_pools]
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
            top.distribution.p_loss if top.distribution is not None else 0.0,
            tie=self.tie[i],
        )

    def finalists_phase(self, pool: list[int], resume: dict | None = None) -> None:
        st = self.settings
        if resume is not None and "picks" in resume:
            picks = list(resume["picks"])
        else:
            picks = self.ranked(pool, self.k)[: st.finalists]
        fkey = jax.random.fold_in(self.key, phase_id("finalists"))
        first = 0 if resume is None else resume["next"]
        deepened = resume is not None and resume["deep"]
        for n in range(first, len(picks)):
            i = picks[n]
            self.before_unit()
            result = self.stage_b(self.setups[i], jax.random.fold_in(fkey, n), 1)
            if result is not None and result.ranking:
                self.playouts += result.playouts
                self.finalists.append(Finalist(self.setups[i], result))
                rows = [self.finalist_row(f) for f in self.finalists]
                self.finalists = [self.finalists[o] for o in prep_order(rows)]
            pos = {"phase": "finalists", "pool": pool, "next": n + 1, "deep": False, "picks": picks}
            self.unit_done("finalists", "finalists", n, len(picks), pos)
        if st.max_depth >= 2 and self.finalists and not deepened:
            self.before_unit()
            lead = self.finalists[0]
            dkey = jax.random.fold_in(jax.random.fold_in(self.key, phase_id("deep")), 0)
            deep = self.stage_b(lead.setup, dkey, st.max_depth)
            if deep is not None and deep.ranking:
                self.playouts += deep.playouts
                self.finalists[0] = Finalist(lead.setup, deep)
            elif self.index[lead.setup] in self.excluded:
                self.finalists.pop(0)  # excluded and named, so no longer the answer
            pos = {
                "phase": "finalists",
                "pool": pool,
                "next": len(picks),
                "deep": True,
                "picks": picks,
            }
            self.unit_done("finalists", "finalists", 0, len(picks), pos)
        self.phase = "finalists"

    def drive(self, pos: dict | None) -> None:
        """Run the phases from the saved position ``pos`` (``None``: from the start)."""
        st = self.settings
        stage = "teams" if pos is None else pos["phase"]
        if stage == "finalists":
            if st.max_depth >= 1:
                self.finalists_phase(pos["pool"], pos)
            return
        if stage == "teams":
            pools = self.team_phase(pos)
            first, mid = 0, None
        else:
            pools, first, mid = pos["pools"], pos["v"], pos
        for v in range(first, st.variant_rounds):
            nxt = self.expansion(v, pools, mid if v == first else None)
            if nxt is None:
                break
            pools = nxt
        if st.variant_rounds:
            self.phase = "variants"
        if st.max_depth >= 1:
            pool = pools[self.k]
            if st.race_playouts > 0 and len(pool) > st.finalists:
                picks = self.race(pool)
                self.finalists_phase(pool, {"next": 0, "deep": False, "picks": picks})
            else:
                self.finalists_phase(pool)

    # ------------------------------------------------------------------ result

    def result(self, stopped: str | None) -> PrepResult:
        k = self.k
        pools = self.last_pools
        if not self.finalists and not pools[k]:
            self.raise_all_refused()
        if self.finalists:
            lead = self.finalists[0]
            chosen = lead.setup
            plan = lead.result
            top = plan.ranking[0]
        else:
            chosen = self.setups[self.screen_leader()]
            plan, top = None, None
        ci = self.index[chosen]
        stats = self.stats[ci]
        ladder = []
        for j in range(RUNGS):
            if j == k and top is not None:
                ladder.append(Rung(j, chosen, top.estimate, top.ci_low, top.ci_high, "plan"))
                continue
            best = ci if j == k else self.ranked(pools[j] or pools[k], j)[0]
            low, high = self.stats[best].interval(j)
            ladder.append(Rung(j, self.setups[best], self.stats[best].rung(j), low, high, "screen"))
        dist = top.distribution if top is not None and top.distribution is not None else None
        cheaper = None
        if self.finalists:
            c = cheaper_alternative([self.finalist_row(f) for f in self.finalists])
            cheaper = self.finalists[c] if c is not None else None
        seconds = self.total()
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
            cheaper=cheaper,
        )


def _seed_bytes(key) -> bytes:
    """The run's seed as bytes: the key's own bits (a legacy or a typed key)."""
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError):
        data = key
    return np.asarray(data).tobytes()


def _seed_of(key) -> int:
    """The run's seed as an integer (``_seed_bytes``)."""
    return int.from_bytes(_seed_bytes(key), "little")


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
    table_version: str | None = None,
    code_version: str | None = None,
    game: str = "platinum_us",
) -> PrepResult:
    """Run the funnel (module docstring) and return the best answer reached.
    ``ValueError`` when every setup is refused ("every setup refused: ...") or when ``resume``
    names a checkpoint of another run (``checkpoint.read_checkpoint``; nothing is written then).

    ``checkpoint``: write the whole state there after every unit. ``resume``: first read that
    file, continue where it stopped (its elapsed time is added to ``stats.seconds``;
    ``settings.budget_seconds`` limits this invocation only) and keep writing to it unless
    ``checkpoint`` names another file. ``table_version`` and ``code_version`` (default: the
    shipped tables' and the package's, ``jaxcache``) and ``game`` (the game id) are what a
    checkpoint is tied to."""
    sink = checkpoint if checkpoint is not None else resume
    print_ = None
    saved = None
    if sink is not None:
        if table_version is None or code_version is None:
            from nuzline import jaxcache
            from nuzline.data.games import get_game

            table_version = table_version or jaxcache.table_version(get_game(game).tables_dir)
            code_version = code_version or jaxcache.code_version()
        print_ = fingerprint(problem, settings, _seed_of(key), table_version, code_version, game)
    if resume is not None:
        saved = read_checkpoint(resume, print_)["funnel"]
    funnel = _Funnel(tables, names, problem, settings, key, progress, clock, sink, print_)
    if saved is not None:
        funnel.restore(saved)
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
            levelled=st.levelled,
        )
        funnel.pruned.extend(prune(tables, problem, variants)[1])
    funnel.dropped = {d for d, _ in funnel.pruned}
    stopped = None
    try:
        funnel.drive(funnel.pos)
    except _Stop:
        stopped = "budget"
    except KeyboardInterrupt:
        if not funnel.units and saved is None:
            raise
        stopped = "interrupted"
    return funnel.result(stopped)
