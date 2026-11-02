"""The all-rung Stage A screen and the prep tie rule (plan 4c Task 6).

``screen`` plays ``n`` playouts from each setup's turn-0 state (``evaluate_leaves``, both
halves pooled; ruling 10) and keeps the 8 ``deaths.BUCKETS`` counts, so one batch scores
every rung. ``ScreenStats`` accumulate across halving rounds (``merge``). ``prep_order`` is
the spec section 4 tie rule over scored setups (ruling 7): it reuses ``stage_a.tied_with``
for the interval overlap, so the whole search has one overlap definition, but its key chain
(deaths within ``DEATHS_TOL``, cost, HP, PP) is its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nuzline.data.tables import Tables
from nuzline.engine.state import BattleState
from nuzline.search import deaths
from nuzline.search.leaves import evaluate_leaves
from nuzline.search.stage_a import tied_with, wilson

__all__ = ["DEATHS_TOL", "EMPTY", "Scored", "ScreenStats", "prep_order", "scored", "screen"]

DEATHS_TOL = 0.05  # expected deaths within this of the fewest count as equal (ruling 7)


@dataclass(frozen=True)
class ScreenStats:
    """Accumulated screen playouts of one setup: ``buckets`` counts per ``deaths.BUCKETS``,
    the summed player faints (``deaths``), summed final HP and PP fractions (``hp_left``,
    ``pp_left``; sums, the ``mean_*`` properties divide by ``n``) and whether any playout
    was refused (the setup is then left out and named)."""

    buckets: tuple[int, ...]
    deaths: int
    hp_left: float
    pp_left: float
    refused: bool

    @property
    def n(self) -> int:
        return sum(self.buckets)

    @property
    def distribution(self) -> deaths.DeathDistribution:
        return deaths.DeathDistribution.from_counts(self.buckets)

    @property
    def e_deaths(self) -> float:
        return self.deaths / self.n if self.n else 0.0

    @property
    def mean_hp(self) -> float:
        return self.hp_left / self.n if self.n else 0.0

    @property
    def mean_pp(self) -> float:
        return self.pp_left / self.n if self.n else 0.0

    def _count(self, k: int) -> int:
        if not 0 <= k < len(deaths.BUCKETS):
            raise ValueError(f"rung {k} outside the death buckets")
        return sum(self.buckets[: k + 1])

    def rung(self, k: int) -> float:
        """P(win with at most ``k`` deaths, no protected loss), the point estimate (0 with
        no playouts)."""
        return self._count(k) / self.n if self.n else 0.0

    def interval(self, k: int) -> tuple[float, float]:
        """The Wilson interval of rung ``k``'s count (all of [0, 1] with no playouts)."""
        return wilson(self._count(k), self.n) if self.n else (0.0, 1.0)

    def merge(self, other: ScreenStats) -> ScreenStats:
        """Both sets of playouts together: every count added, ``refused`` or-ed."""
        return ScreenStats(
            tuple(a + b for a, b in zip(self.buckets, other.buckets, strict=True)),
            self.deaths + other.deaths,
            self.hp_left + other.hp_left,
            self.pp_left + other.pp_left,
            self.refused or other.refused,
        )


EMPTY = ScreenStats((0,) * len(deaths.BUCKETS), 0, 0.0, 0.0, False)


def screen(
    tables: Tables, states: BattleState, n: int, key, max_turns: int = 100
) -> list[ScreenStats]:
    """``n`` playouts (even) from each of the ``N`` numpy ``states`` (each with its own
    ``faint_budget`` and protected slots), both halves pooled."""
    values = evaluate_leaves(tables, states, [n] * len(states.hp), key, max_turns)
    return [
        ScreenStats(
            tuple(a + b for a, b in zip(v.buckets_a, v.buckets_b, strict=True)),
            v.deaths_a + v.deaths_b,
            v.hp_left,
            v.pp_left,
            v.refused,
        )
        for v in values
    ]


@dataclass(frozen=True)
class Scored:
    """One setup's numbers for the tie rule: rung score and interval, expected deaths,
    weighted resource cost and mean HP and PP fractions left."""

    score: float
    ci_low: float
    ci_high: float
    e_deaths: float
    cost: float
    hp_left: float
    pp_left: float


def scored(stats: ScreenStats, k: int, cost: float) -> Scored:
    """``stats`` as rung ``k``'s row for ``prep_order``."""
    low, high = stats.interval(k)
    return Scored(stats.rung(k), low, high, stats.e_deaths, cost, stats.mean_hp, stats.mean_pp)


def prep_order(rows: Sequence[Scored]) -> list[int]:
    """Row indices best first (plan ruling 7). Among the rows whose interval overlaps the
    top-scoring row's, the leader has expected deaths within ``DEATHS_TOL`` of the fewest,
    then the lowest cost, the highest mean HP left, the highest mean PP left, the higher
    score, the lower index. The rest follow by ``(-score, e_deaths, cost, index)``."""
    if not rows:
        return []

    def by_score(i: int):
        return (-rows[i].score, rows[i].e_deaths, rows[i].cost, i)

    ranked = sorted(range(len(rows)), key=by_score)
    top = rows[ranked[0]]
    overlap = [i for i in ranked if tied_with(rows[i], top)]
    fewest = min(rows[i].e_deaths for i in overlap)
    near = [i for i in overlap if rows[i].e_deaths <= fewest + DEATHS_TOL]
    leader = min(
        near,
        key=lambda i: (rows[i].cost, -rows[i].hp_left, -rows[i].pp_left, -rows[i].score, i),
    )
    return [leader] + [i for i in ranked if i != leader]
