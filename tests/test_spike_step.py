import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.spike_step import N_EFFECTS, batched_step, make_state


def test_step_keeps_shapes_and_dtypes():
    s = make_state(0, 4)
    out = batched_step(s, jnp.zeros((4, 2), jnp.int8))
    for a, b in zip(s, out):
        assert a.shape == b.shape and a.dtype == b.dtype


def test_no_64bit_anywhere():
    s = make_state(0, 4)
    out = batched_step(s, jnp.zeros((4, 2), jnp.int8))
    assert all(x.dtype.itemsize <= 4 for x in out)


def test_batch_of_one_matches_element_of_batch():
    big = make_state(1, 8)
    ids = jnp.full((8, 2), 7, jnp.int8)
    out_big = batched_step(big, ids)
    one = jax.tree.map(lambda x: x[3:4], big)
    out_one = batched_step(one, ids[3:4])
    for a, b in zip(out_big, out_one):
        np.testing.assert_array_equal(np.asarray(a[3]), np.asarray(b[0]))


def test_unknown_effect_id_is_noop_except_end_of_turn():
    s = make_state(2, 1)
    s = s._replace(status=jnp.zeros_like(s.status))
    out = batched_step(s, jnp.full((1, 2), N_EFFECTS + 5, jnp.int8))
    np.testing.assert_array_equal(np.asarray(out.hp), np.asarray(s.hp))
    assert int(out.turn[0]) == int(s.turn[0]) + 1


def test_each_effect_changes_something():
    s = make_state(3, 1)
    for k in range(N_EFFECTS):
        out = batched_step(s, jnp.array([[k, k]], jnp.int8))
        assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(s, out))
