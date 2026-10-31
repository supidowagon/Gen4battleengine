"""Stage A ranking (docs/design.md §4.3): every legal player action at a node is
scored by ``n`` Monte Carlo playouts (``playout_from_action``) that all actions share,
the same ``n`` keys for each (common random numbers), batched as one ``[NUM_ACTIONS, n]``
``vmap`` (padded) so the playout compiles once per ``n``.

Each action reports P(win with zero player faints) with a 95 % Wilson interval,
P(win) and the expected number of player faints. Playout odds are a lower bound on
optimal play (the playout policy is a heuristic).

The ranking is sorted strictly by ``(-p_win0, -p_win, action)`` (plan ruling);
statistical ties are reported, not reordered: ``ActionResult.tied_with_best`` marks the
other actions whose interval overlaps the leader's (never the leader itself). A playout that meets a refused mechanic
(``PlayoutResult.refused``) has no result; any such playout makes ``rank_actions``
raise ``UnsupportedAi`` rather than count it, naming the refused mechanic of the first
refused playout (``refusal_reason``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from nuzline.ai.policy import UnsupportedAi
from nuzline.data.tables import Names, Tables
from nuzline.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, legal_actions, usable_moves
from nuzline.engine.state import BattleState
from nuzline.jaxcache import cached_export, code_version
from nuzline.search.playout import PlayoutResult, playout_from_action, refusal_reason

__all__ = ["ActionResult", "Ranking", "action_label", "rank_actions", "tied_with", "wilson"]


@dataclass(frozen=True)
class ActionResult:
    action: int
    label: str  # "Ember" / "Switch -> Starly"
    p_win0: float  # P(win with zero player faints)
    ci_low: float  # 95 % Wilson interval of p_win0
    ci_high: float
    p_win: float
    e_faints: float  # expected player faints
    n: int  # playouts
    tied_with_best: bool = False  # not the leader, and the interval overlaps the leader's


def tied_with(row, best) -> bool:
    """Whether ``row``'s 95 % interval (``ci_low``, ``ci_high``) overlaps ``best``'s: the
    statistical tie a ranking marks (``tied_with_best``), never reorders."""
    return row.ci_high >= best.ci_low and row.ci_low <= best.ci_high


class Ranking(list):
    """``ActionResult``s, best first; ``all_zero`` when every ``p_win0`` is 0."""

    all_zero: bool

    @classmethod
    def of(cls, results: Iterable[ActionResult]) -> Ranking:
        """``results`` sorted by ``(-p_win0, -p_win, action)``, ``tied_with_best`` set."""
        rows = sorted(results, key=lambda r: (-r.p_win0, -r.p_win, r.action))
        if rows:
            best, *others = rows
            rows = [replace(best, tied_with_best=False)] + [
                replace(r, tied_with_best=tied_with(r, best)) for r in others
            ]
        ranking = cls(rows)
        ranking.all_zero = all(r.p_win0 == 0 for r in rows)
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


def _playouts(tables: Tables, state: BattleState, actions, keys, max_turns: int) -> PlayoutResult:
    """``PlayoutResult`` of shape ``[A, n]``: every action against every key. Run
    through ``cached_export`` (traced once per shape, sharding and source version)."""

    def per_action(action):
        return jax.vmap(lambda k: playout_from_action(tables, state, action, k, max_turns))(keys)

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
    (module docstring). Raises ``UnsupportedAi`` when any playout is refused."""
    legal = np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))
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
    playouts = cached_export(
        partial(_playouts, max_turns=max_turns),
        args,
        name="playouts",
        static={"max_turns": max_turns},
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
            reason = refusal_reason(tables, state, action, keys[first], max_turns)
            raise UnsupportedAi(
                f"{int(refused[i].sum())} of {n} playouts from {label!r} refused; "
                f"playout {first}: {reason or 'reason not reproduced outside jit'}"
            )
        won = np.asarray(out.won[i]) == 1
        faints = np.asarray(out.player_faints[i]).astype(np.int32)
        wins0 = int(np.sum(won & (faints == 0)))
        lo, hi = wilson(wins0, n)
        results.append(
            ActionResult(
                action=action,
                label=label,
                p_win0=wins0 / n,
                ci_low=lo,
                ci_high=hi,
                p_win=float(np.mean(won)),
                e_faints=float(np.mean(faints)),
                n=n,
            )
        )
    return Ranking.of(results)
