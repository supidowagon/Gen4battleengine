"""Party stat calculation (plain numpy, used when a battle state is built).

Follows ``Pokemon_CalcStats`` (``src/pokemon.c:564``) and
``Pokemon_GetNatureStatValue`` (``src/pokemon.c:2593``) with every vector in
``ids.Stat`` order (HP, Atk, Def, SpA, SpD, Spe). Stat stages are never folded in
here: the battle engine applies them when a stat is used. Shedinja's max HP is
pinned at 1 (``src/pokemon.c:598`` and ``:634``).
"""

from __future__ import annotations

import numpy as np

SPECIES_SHEDINJA = 292


def calc_stats(
    base: np.ndarray,
    ivs: np.ndarray,
    evs: np.ndarray,
    level: int,
    nature_mods: np.ndarray,
    species: int = 0,
) -> np.ndarray:
    """Return the six stats as ``int32``. ``nature_mods`` holds -1/0/+1 per stat;
    its HP entry is ignored, as in the decomp. HP is 1 for Shedinja (``species``)."""
    base, ivs, evs, mods = (np.asarray(a, dtype=np.int32) for a in (base, ivs, evs, nature_mods))
    level = np.int32(level)
    core = (2 * base + ivs + evs // 4) * level // 100
    stats = core + 5
    stats = np.where(mods > 0, stats * 110 // 100, stats)
    stats = np.where(mods < 0, stats * 90 // 100, stats)
    stats[0] = 1 if species == SPECIES_SHEDINJA else core[0] + level + 10
    return stats.astype(np.int32)
