"""Party stat calculation: ``calc_stats`` (plain numpy, used when a battle state is built)
and ``calc_stats_jnp`` (its jnp port, for the level-up recompute in ``engine.levelup``).

Follows ``Pokemon_CalcStats`` (``src/pokemon.c:564``) and
``Pokemon_GetNatureStatValue`` (``src/pokemon.c:2593``) with every vector in
``ids.Stat`` order (HP, Atk, Def, SpA, SpD, Spe). Stat stages are never folded in
here: the battle engine applies them when a stat is used. Shedinja's max HP is
pinned at 1 (``src/pokemon.c:598`` and ``:634``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp

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


def calc_stats_jnp(base, ivs, evs, level, nature_mods, species) -> jnp.ndarray:
    """``calc_stats`` in jnp, batched over leading axes: ``base``, ``ivs``, ``evs`` and
    ``nature_mods`` are ``[..., 6]`` in ``ids.Stat`` order, ``level`` and ``species`` are
    ``[...]``. Returns ``int32[..., 6]``. ``Pokemon_CalcStats`` (src/pokemon.c:564-648):
    max HP ``(2B + IV + EV/4) * L / 100 + L + 10``, 1 for Shedinja (:597-601); every other
    stat ``(2B + IV + EV/4) * L / 100 + 5`` (:605-628), then ``Pokemon_GetNatureStatValue``
    (:2593-2615): ``* 110 / 100`` raised, ``* 90 / 100`` lowered."""
    import jax.numpy as jnp  # local: nuzline.saves imports this module and must not load jax

    i32 = jnp.int32
    base, ivs, evs, mods = (jnp.asarray(a).astype(i32) for a in (base, ivs, evs, nature_mods))
    level = jnp.asarray(level).astype(i32)[..., None]
    core = (2 * base + ivs + evs // 4) * level // 100
    stats = core + 5
    stats = jnp.where(mods > 0, stats * 110 // 100, stats)
    stats = jnp.where(mods < 0, stats * 90 // 100, stats)
    hp = jnp.where(jnp.asarray(species) == SPECIES_SHEDINJA, 1, core[..., 0] + level[..., 0] + 10)
    return stats.at[..., 0].set(hp)
