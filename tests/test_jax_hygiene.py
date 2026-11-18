"""``step`` and ``sample_outcome`` are jit- and vmap-able with fixed shapes and no
64-bit types (spec §4.2, §5 JAX hygiene)."""

import jax
import jax.numpy as jnp
import pytest

from nuzline.engine.outcome import MAX_HITS, empty_outcome, sample_outcome
from nuzline.engine.step import step
from tests.engine.helpers import roark_state


@pytest.mark.slow
def test_step_jits_and_vmaps_with_stable_shapes(tables, names):
    s = roark_state(tables, names)
    B = 16
    bs = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s)
    bo = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), empty_outcome())
    f = jax.jit(jax.vmap(step, in_axes=(None, 0, 0, 0)))
    out = f(tables, bs, jnp.zeros((B, 2), jnp.int32), bo)
    for a, b in zip(bs, out):
        assert a.shape == b.shape and a.dtype == b.dtype


def test_no_64bit_in_jaxpr(tables, names):
    s = roark_state(tables, names)
    jaxpr = jax.make_jaxpr(step)(tables, s, jnp.zeros(2, jnp.int32), empty_outcome())
    assert "int64" not in str(jaxpr) and "float64" not in str(jaxpr)


def test_sample_outcome_is_jittable(tables, names):
    s = roark_state(tables, names)
    o = jax.jit(sample_outcome, static_argnums=())(
        jax.random.PRNGKey(0), tables, s, jnp.zeros(2, jnp.int32)
    )
    assert o.roll.shape == (2, MAX_HITS) and int(o.roll.max()) <= 15
