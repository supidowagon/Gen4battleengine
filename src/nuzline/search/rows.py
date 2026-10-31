"""Row helpers for stacked numpy pytrees (a batch of ``BattleState``s, ``Outcome``s,
...), shared by ``expand`` and ``stage_b``."""

from __future__ import annotations

import jax
import numpy as np

__all__ = ["concat", "row_bytes", "take"]


def take(tree, idx):
    """The rows ``idx`` (an index array or a slice) of every field, as numpy."""
    return jax.tree.map(lambda f: np.asarray(f)[idx], tree)


def concat(parts):
    """The pytrees ``parts`` stacked along their leading axis."""
    return jax.tree.map(lambda *fs: np.concatenate(fs), *parts)


def row_bytes(tree, n: int) -> np.ndarray:
    """``uint8[n, bytes]``: each row of a stacked numpy pytree, flattened."""
    leaves = jax.tree.leaves(tree)
    return np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in leaves], axis=1
    )
