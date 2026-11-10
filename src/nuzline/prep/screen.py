"""The all-rung Stage A screen and the prep tie rule (plan 4c Task 6).

``screen`` plays ``n`` playouts from each setup's turn-0 state (``evaluate_leaves`` from
roots, so each playout draws the battle-start Quick Claw roll; both halves pooled; ruling 10) and keeps the 8 ``deaths.BUCKETS`` counts, so one batch scores
every rung. ``ScreenStats`` accumulate across halving rounds (``merge``). ``prep_order`` is
the spec section 4 tie rule over scored setups (user decisions 2026-11-03 and 2026-11-04,
superseding ruling 7): the battle tie rule (``stage_a.tie_ordered``, ruling 5) at every
position, with ``stage_a.tied_with`` for the interval overlap, so the whole search has one
overlap definition. Inside an overlap fewer deaths decide, then the higher score, then cost,
then the seeded tie rank; HP and PP left are not ranked. Its deaths key is
``deaths.tie_deaths``: a lost playout (whiteout or unfinished) counts ``deaths.LOSS_DEATHS``
more, so a setup that risks a whiteout never leads for fewer deaths (spec section 2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nuzline.data.tables import Tables
from nuzline.engine.state import BattleState
from nuzline.search import deaths
from nuzline.search.leaves import evaluate_leaves
from nuzline.search.stage_a import tied_with, wilson

__all__ = [
    "EMPTY",
    "Scored",
    "ScreenStats",
    "cheaper_alternative",
    "prep_order",
    "scored",
    "screen",
]


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
        if not 0 <= k <= deaths.MAX_DEATH_BUDGET:
            raise ValueError(f"rung {k} outside [0, {deaths.MAX_DEATH_BUDGET}]")
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
    values = evaluate_leaves(tables, states, [n] * len(states.hp), key, max_turns, root=True)
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
    weighted resource cost, mean HP and PP fractions left (reported, not ranked),
    P(whiteout or unfinished) (``loss``) and ``tie``: the setup's seeded tie rank
    (``search.tie_rank``), the last real key."""

    score: float
    ci_low: float
    ci_high: float
    e_deaths: float
    cost: float
    hp_left: float
    pp_left: float
    loss: float = 0.0
    tie: int = 0

    @property
    def tie_deaths(self) -> float:
        """The deaths key of ``prep_order`` (``deaths.tie_deaths``)."""
        return deaths.tie_deaths(self.e_deaths, self.loss)


def scored(stats: ScreenStats, k: int, cost: float, tie: int = 0) -> Scored:
    """``stats`` as rung ``k``'s row for ``prep_order``, with tie rank ``tie``."""
    low, high = stats.interval(k)
    loss = deaths.loss_mass(stats.buckets) / stats.n if stats.n else 0.0
    return Scored(
        stats.rung(k), low, high, stats.e_deaths, cost, stats.mean_hp, stats.mean_pp, loss, tie
    )


def prep_order(rows: Sequence[Scored]) -> list[int]:
    """Row indices best first (spec section 4, user decisions 2026-11-03 and 2026-11-04): the
    battle tie rule (``stage_a.tie_ordered``, ruling 5) at every position. Among the remaining
    rows whose interval overlaps the top-scoring remaining row's (``stage_a.tied_with``), the
    next has the lower deaths key (``Scored.tie_deaths``: a lost playout counts
    ``deaths.LOSS_DEATHS``), then the higher score, the lower cost, the lower tie rank, the
    lower index. An allowed resource never outranks win chance or deaths. HP and PP left are
    not ranked."""
    left = sorted(range(len(rows)), key=lambda i: (-rows[i].score, rows[i].tie_deaths, i))
    order = []
    while left:
        top = rows[left[0]]
        nxt = min(
            (i for i in left if tied_with(rows[i], top)),
            key=lambda i: (rows[i].tie_deaths, -rows[i].score, rows[i].cost, rows[i].tie, i),
        )
        order.append(nxt)
        left.remove(nxt)
    return order


def cheaper_alternative(rows: Sequence[Scored]) -> int | None:
    """``rows`` in ``prep_order`` (row 0 the answer): the cheapest later row whose interval
    overlaps the answer's and that costs less (ties: the earlier row), else ``None``. Shown on
    the card only; it never reorders (spec section 4)."""
    if not rows:
        return None
    cheaper = [
        i
        for i in range(1, len(rows))
        if rows[i].cost < rows[0].cost and tied_with(rows[i], rows[0])
    ]
    return min(cheaper, key=lambda i: (rows[i].cost, i)) if cheaper else None
