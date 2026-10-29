"""Throwaway M0 spike: one masked JAX step with N_EFFECTS generated effects."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

N_EFFECTS = 100
_DELTAS = (-3, -2, -1, 1, 2, 3)  # never 0
_HP_MAX, _STAT_MAX, _STAGE_MAX, _STATUS_MAX = 999, 999, 12, 7


class SpikeState(NamedTuple):
    hp: jax.Array  # int16[2, 6]
    stats: jax.Array  # int16[2, 6, 6]
    stages: jax.Array  # int8[2, 8]
    status: jax.Array  # int8[2, 6]
    active: jax.Array  # int8[2]
    turn: jax.Array  # int16[]
    field: jax.Array  # int8[8]


def make_state(seed: int, batch: int) -> SpikeState:
    """Batched synthetic state; every value is mid-range so no effect saturates."""
    rng = np.random.default_rng(seed)
    return SpikeState(
        hp=jnp.asarray(rng.integers(50, 200, (batch, 2, 6)), jnp.int16),
        stats=jnp.asarray(rng.integers(50, 200, (batch, 2, 6, 6)), jnp.int16),
        stages=jnp.asarray(rng.integers(4, 9, (batch, 2, 8)), jnp.int8),
        status=jnp.full((batch, 2, 6), 3, jnp.int8),
        active=jnp.asarray(rng.integers(0, 6, (batch, 2)), jnp.int8),
        turn=jnp.zeros((batch,), jnp.int16),
        field=jnp.zeros((batch, 8), jnp.int8),
    )


def _effect(k: int) -> tuple[int, int]:
    """Effect k -> (field index 0..3 = hp/stats/stages/status, delta)."""
    return k % 4, _DELTAS[(k // 4) % len(_DELTAS)]


def _apply_effect(state: SpikeState, side: int, k: int) -> SpikeState:
    """State after effect k hits the acting side's active slot (unmasked)."""
    which, delta = _effect(k)
    a = state.active[side].astype(jnp.int32)
    if which == 0:
        v = jnp.clip(state.hp[side, a] + delta, 0, _HP_MAX).astype(jnp.int16)
        return state._replace(hp=state.hp.at[side, a].set(v))
    if which == 1:
        v = jnp.clip(state.stats[side, a, k % 6] + delta, 1, _STAT_MAX).astype(jnp.int16)
        return state._replace(stats=state.stats.at[side, a, k % 6].set(v))
    if which == 2:
        v = jnp.clip(state.stages[side, k % 8] + delta, 0, _STAGE_MAX).astype(jnp.int8)
        return state._replace(stages=state.stages.at[side, k % 8].set(v))
    v = jnp.clip(state.status[side, a] + delta, 0, _STATUS_MAX).astype(jnp.int8)
    return state._replace(status=state.status.at[side, a].set(v))


def _select(hit: jax.Array, new: SpikeState, old: SpikeState) -> SpikeState:
    return jax.tree.map(lambda n, o: jnp.where(hit, n, o), new, old)


def step(state: SpikeState, effect_id: jax.Array) -> SpikeState:
    """Unbatched step: masked effects for both sides, end-of-turn, turn + 1."""
    for side in range(2):
        for k in range(N_EFFECTS):
            hit = effect_id[side] == k
            modified = _apply_effect(state, side, k)
            state = _select(hit, modified, state)
    hp = jnp.where(state.status != 0, jnp.maximum(state.hp - 1, 0), state.hp).astype(jnp.int16)
    return state._replace(hp=hp, turn=(state.turn + 1).astype(jnp.int16))


batched_step = jax.jit(jax.vmap(step))
