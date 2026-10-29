"""Party stat calculation (plain numpy, used when a battle state is built).

Follows ``Pokemon_CalcStats`` (``src/pokemon.c:564``) and
``Pokemon_GetNatureStatValue`` (``src/pokemon.c:2593``) with every vector in
``ids.Stat`` order (HP, Atk, Def, SpA, SpD, Spe). Stat stages are never folded in
here: the battle engine applies them when a stat is used. Shedinja's fixed 1 HP
is not special-cased (it is outside the implemented species set).
"""

from __future__ import annotations

import numpy as np


def calc_stats(
    base: np.ndarray,
    ivs: np.ndarray,
    evs: np.ndarray,
    level: int,
    nature_mods: np.ndarray,
) -> np.ndarray:
    """Return the six stats as ``int32``. ``nature_mods`` holds -1/0/+1 per stat;
    its HP entry is ignored, as in the decomp."""
    base, ivs, evs, mods = (np.asarray(a, dtype=np.int32) for a in (base, ivs, evs, nature_mods))
    level = np.int32(level)
    core = (2 * base + ivs + evs // 4) * level // 100
    stats = core + 5
    stats = np.where(mods > 0, stats * 110 // 100, stats)
    stats = np.where(mods < 0, stats * 90 // 100, stats)
    stats[0] = core[0] + level + 10
    return stats.astype(np.int32)
