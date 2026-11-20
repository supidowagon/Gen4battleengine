"""Stage A ranking (docs/design.md §4.3): every legal player action at a node is
scored by ``n`` Monte Carlo playouts (``playout_from_action``) that all actions share,
the same ``n`` keys for each (common random numbers), batched as one ``[NUM_ACTIONS, n]``
``vmap`` (padded) so the playout compiles once per ``n``.

Each action reports the whole faint-budget ladder from one batch of playouts: P(win
with at most ``r`` deaths and no protected Pokemon lost) for rungs ``r`` = 0..3
(``ladder``; ``p_win0`` is rung 0), ``p_success`` (the state's own rung ``k =
faint_budget``) with its 95 % Wilson interval, P(win), the expected number of player
faints and the death distribution (``deaths.BUCKETS``). Playout odds are a lower bound
on optimal play (the playout policy is a heuristic). At a pending player replacement the
ranked actions are the replacements (``decision_actions``).

The ranking follows ``tie_ordered`` (plan ruling 5): among the actions whose interval
overlaps the best score's, the leader has the fewest expected deaths, a lost playout
counting ``deaths.LOSS_DEATHS`` more (``deaths.tie_deaths``: a whiteout is never traded for
fewer deaths); the other actions
follow by ``(-score, e_faints, action)``. ``ActionResult.tied_with_best`` marks the
actions whose interval overlaps the leader's (never the leader itself). A playout that meets a refused mechanic
(``PlayoutResult.refused``) has no result; any such playout makes ``rank_actions``
raise ``UnsupportedAi`` rather than count it, naming the refused mechanic of the first
refused playout (``refusal_reason``).

The root's hidden draws (the battle-start Quick Claw ``speedRand``, ``BattleState.qc_roll``)
are a chance node: each playout draws them (``playout.root_state``) from its own key, the
same for every action (common random numbers).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from nuzline.ai.layout import AiLayout, layout_of
from nuzline.ai.policy import UnsupportedAi
from nuzline.data.tables import Names, Tables
from nuzline.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, decision_actions, usable_moves
from nuzline.engine.coverage import check_acid_rain
from nuzline.engine.outcome import replacement_known
from nuzline.engine.state import MAX_DEATH_BUDGET, BattleState
from nuzline.jaxcache import cached_export, code_version
from nuzline.search import deaths
from nuzline.search.playout import (
    PlayoutResult,
    playout_from_action,
    refusal_reason,
    root_state,
)

__all__ = [
    "ActionResult",
    "Ranking",
    "action_label",
    "rank_actions",
    "tie_ordered",
    "tied_with",
    "wilson",
]


@dataclass(frozen=True)
class ActionResult:
    action: int
    label: str  # "Ember" / "Switch -> Starly"
    p_win0: float  # P(win with zero player faints, no protected loss) = ladder[0]
    ci_low: float  # 95 % Wilson interval of ``score`` (p_win0 when p_success is None)
    ci_high: float
    p_win: float
    e_faints: float  # expected player faints
    n: int  # playouts
    tied_with_best: bool = False  # not the leader, and the interval overlaps the leader's
    p_success: float | None = None  # P(win, deaths <= k, no protected death) for the state's k
    ladder: tuple[float, ...] = ()  # rungs 0..MAX_DEATH_BUDGET with the protected set
    distribution: deaths.DeathDistribution | None = None

    @property
    def score(self) -> float:
        """The objective: ``p_success``, or ``p_win0`` when the row has none."""
        return self.p_win0 if self.p_success is None else self.p_success


def tied_with(row, best) -> bool:
    """Whether ``row``'s 95 % interval (``ci_low``, ``ci_high``) overlaps ``best``'s: the
    statistical tie a ranking marks (``tied_with_best``), never reorders."""
    return row.ci_high >= best.ci_low and row.ci_low <= best.ci_high


def tie_ordered[R](
    rows: Sequence[R], score: Callable[[R], float], deaths: Callable[[R], float]
) -> list[R]:
    """``rows`` (``action``, ``ci_low``, ``ci_high``, ``tied_with_best``) in the tie rule's
    order (plan ruling 5): among the rows whose interval overlaps the top-scoring row's,
    the leader has the fewest ``deaths``, then the higher ``score``, then the lower action;
    the other rows follow by ``(-score, deaths, action)``. ``tied_with_best`` is set for
    the rows whose interval overlaps the leader's, never for the leader."""
    rows = sorted(rows, key=lambda r: (-score(r), deaths(r), r.action))
    if not rows:
        return []
    top = rows[0]
    contenders = [r for r in rows if tied_with(r, top)]
    leader = min(contenders, key=lambda r: (deaths(r), -score(r), r.action))
    rest = [r for r in rows if r is not leader]
    return [replace(leader, tied_with_best=False)] + [
        replace(r, tied_with_best=tied_with(r, leader)) for r in rest
    ]


def _tie_deaths(r: ActionResult) -> float:
    loss = r.distribution.p_loss if r.distribution is not None else 0.0
    return deaths.tie_deaths(r.e_faints, loss)


class Ranking(list):
    """``ActionResult``s, best first; ``all_zero`` when every ``score`` is 0."""

    all_zero: bool

    @classmethod
    def of(cls, results: Iterable[ActionResult]) -> Ranking:
        """``results`` in ``tie_ordered`` order (``score``, ``e_faints`` with a lost playout
        weighted by ``deaths.tie_deaths``)."""
        rows = tie_ordered(list(results), lambda r: r.score, _tie_deaths)
        ranking = cls(rows)
        ranking.all_zero = all(r.score == 0 for r in rows)
        return ranking


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of ``k`` successes in ``n`` trials, clipped to [0, 1]."""
    if n <= 0:
        raise ValueError("wilson needs n > 0")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


# The pytrees of ``_playouts``' arguments and result, for ``jax.export`` serialization.
for _cls in (Tables, BattleState, PlayoutResult):
    jax.export.register_namedtuple_serialization(_cls, serialized_name=f"nuzline.{_cls.__name__}")


def _playouts(
    tables: Tables,
    state: BattleState,
    actions,
    keys,
    max_turns: int,
    layout: AiLayout,
    replacement: bool = True,
) -> PlayoutResult:
    """``PlayoutResult`` of shape ``[A, n]``: every action against every key. Run
    through ``cached_export`` (traced once per shape, sharding, AI layout and source
    version). ``replacement``: ``sample_outcome``'s static flag (``outcome.replacement_known``)."""

    def one(action, k):
        root = root_state(tables, state, k)
        return playout_from_action(tables, root, action, k, max_turns, layout, replacement)

    def per_action(action):
        return jax.vmap(lambda k: one(action, k))(keys)

    return jax.vmap(per_action)(actions)


def action_label(names: Names, state: BattleState, action: int, struggle: bool) -> str:
    """The player action's name at ``state``: the move, ``Struggle`` or ``Switch -> X``."""
    mon = int(state.active[0])
    if action >= MOVE_ACTIONS:
        return f"Switch -> {names.species_names[int(state.species[0, action - MOVE_ACTIONS])]}"
    if struggle:
        return "Struggle"
    return names.move_names[int(state.moves[0, mon, action])]


def rank_actions(
    tables: Tables, names: Names, state: BattleState, n: int, key, max_turns: int = 100
) -> Ranking:
    """Rank the player's legal actions at ``state`` by ``n`` shared playouts each
    (module docstring). Raises ``UnsupportedAi`` when any playout is refused and
    ``ValueError`` on a root with learn work (Stage B answers a learn prompt; K3d-exp)."""
    check_acid_rain(state)  # a refused state is never played on (docs/kaizo FCM.md section 5)
    if np.any(np.asarray(state.learn_level)):
        raise ValueError("rank_actions: a Stage A root carries no learn work")
    legal = np.flatnonzero(np.asarray(decision_actions(tables, state)))
    k = int(state.faint_budget)
    protected = np.asarray(state.protected)
    struggle = not bool(np.any(np.asarray(usable_moves(tables, state, 0))))
    keys = jax.random.split(key, n)
    devices = jax.device_count()
    # With several host devices the key axis is padded to a multiple of the device count
    # (copies of keys[0]) and sharded; the padded columns are dropped below, so the
    # statistics, and the result, do not depend on the device count.
    padded_n = -(-n // devices) * devices
    if devices > 1:
        pad = jnp.broadcast_to(keys[:1], (padded_n - n, *keys.shape[1:]))
        sharding = NamedSharding(Mesh(jax.devices(), ("d",)), PartitionSpec("d"))
        sharded_keys = jax.device_put(jnp.concatenate([keys, pad]), sharding)
    else:
        sharded_keys = keys
    # Padded to NUM_ACTIONS rows (repeating a legal action) so the batch shape, and the
    # compile, does not depend on how many actions are legal; the padding is dropped.
    padded = np.full(NUM_ACTIONS, legal[0], np.int32)
    padded[: legal.size] = legal
    args = (tables, state, jnp.asarray(padded), sharded_keys)
    layout = layout_of(tables)
    replacement = replacement_known(tables, state)
    playouts = cached_export(
        partial(_playouts, max_turns=max_turns, layout=layout, replacement=replacement),
        args,
        name="playouts",
        static={"max_turns": max_turns, "layout": layout.key, "replacement": replacement},
        version=code_version(),
    )
    out = jax.device_get(playouts(*args))
    out = jax.tree.map(lambda x: x[:, :n], out)
    refused = np.asarray(out.refused).astype(bool)
    results = []
    for i, action in enumerate(legal.tolist()):
        label = action_label(names, state, action, struggle)
        if refused[i].any():
            first = int(np.flatnonzero(refused[i])[0])
            root = root_state(tables, state, keys[first])
            reason = refusal_reason(tables, root, action, keys[first], max_turns, layout)
            raise UnsupportedAi(
                f"{int(refused[i].sum())} of {n} playouts from {label!r} refused; "
                f"playout {first}: {reason or 'reason not reproduced outside jit'}"
            )
        won = np.asarray(out.won[i]) == 1
        faints = np.asarray(out.player_faints[i]).astype(np.int32)
        dead = np.asarray(out.deaths[i])
        wins = [
            int(np.sum(deaths.success(won, faints, dead, rung, protected)))
            for rung in range(MAX_DEATH_BUDGET + 1)
        ]
        lo, hi = wilson(wins[k], n)
        buckets = deaths.bucket(won, np.asarray(out.ended[i]), faints, dead, protected)
        counts = np.bincount(buckets, minlength=len(deaths.BUCKETS))
        results.append(
            ActionResult(
                action=action,
                label=label,
                p_win0=wins[0] / n,
                ci_low=lo,
                ci_high=hi,
                p_win=float(np.mean(won)),
                e_faints=float(np.mean(faints)),
                n=n,
                p_success=wins[k] / n,
                ladder=tuple(w / n for w in wins),
                distribution=deaths.DeathDistribution.from_counts(counts),
            )
        )
    return Ranking.of(results)
