"""Who died: the success test of the faint budget and the death distribution.

A line succeeds on rung ``k`` when the player wins with at most ``k`` deaths and no
protected Pokemon fainted (spec §4, §10.1); a loss is never traded for fewer deaths.
``BUCKETS`` partitions every playout so one batch scores every rung: rung ``k`` is the
first ``k + 1`` buckets (plan ruling 8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nuzline.engine.state import MAX_DEATH_BUDGET

__all__ = ["BUCKETS", "DeathDistribution", "bucket", "over_budget", "success"]

BUCKETS = (
    "win_0",
    "win_1",
    "win_2",
    "win_3",
    "win_4plus",
    "win_protected_lost",
    "whiteout",
    "unfinished",
)
_WIN_PROTECTED, _WHITEOUT, _UNFINISHED = 5, 6, 7
_LAST_COUNTED = 4  # win_4plus


def over_budget(state):
    """Failed state (plan ruling 3): more player faints than ``faint_budget`` or a
    protected slot in ``deaths``. One state or a batch (leading axes), numpy or jax."""
    xp = np if isinstance(state.deaths, np.ndarray) else _jnp()
    protected_died = xp.any((state.protected != 0) & (state.deaths != 0), axis=-1)
    return (state.player_faints > state.faint_budget) | protected_died


def _jnp():
    import jax.numpy as jnp

    return jnp


def success(won, faints, deaths, k, protected) -> np.ndarray:
    """``won & (faints <= k) & ~any(deaths & protected)``; ``deaths`` and ``protected``
    are ``[..., 6]``, the rest broadcast over the leading axes."""
    lost_protected = np.any((np.asarray(deaths) != 0) & (np.asarray(protected) != 0), axis=-1)
    return (np.asarray(won) != 0) & (np.asarray(faints) <= k) & ~lost_protected


def bucket(won, ended, faints, deaths, protected) -> np.ndarray:
    """Index into ``BUCKETS``: a win with ``faints`` deaths (4+ folded) and no protected
    loss, a win that lost a protected Pokemon, a finished non-win, or a playout that hit
    the turn cap (``ended`` false)."""
    won, ended = np.asarray(won) != 0, np.asarray(ended) != 0
    lost_protected = np.any((np.asarray(deaths) != 0) & (np.asarray(protected) != 0), axis=-1)
    counted = np.minimum(np.asarray(faints), _LAST_COUNTED)
    out = np.where(won, np.where(lost_protected, _WIN_PROTECTED, counted), _WHITEOUT)
    return np.where(won | ended, out, _UNFINISHED).astype(np.int64)


@dataclass(frozen=True)
class DeathDistribution:
    """Probabilities of the ``BUCKETS`` (same order) and the playouts behind them."""

    probs: tuple[float, ...]
    n: int | None = None

    def rung(self, k: int) -> float:
        """P(win with at most ``k`` deaths, no protected loss)."""
        if not 0 <= k <= MAX_DEATH_BUDGET:
            raise ValueError(f"rung {k} outside [0, {MAX_DEATH_BUDGET}]")
        return float(sum(self.probs[: k + 1]))

    @property
    def p_win(self) -> float:
        """P(win), however many deaths and including protected losses."""
        return float(sum(self.probs[: _WIN_PROTECTED + 1]))

    def as_dict(self) -> dict:
        return {**dict(zip(BUCKETS, self.probs, strict=True)), "n": self.n}

    @classmethod
    def from_counts(cls, counts: Sequence[float]) -> DeathDistribution:
        total = float(sum(counts))
        if total <= 0:
            raise ValueError("a death distribution needs at least one playout")
        return cls(tuple(float(c) / total for c in counts), int(total))
