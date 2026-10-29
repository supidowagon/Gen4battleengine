"""Hypothesis differential test: the JAX ``step`` against the plain-Python reference
``ref_step`` on random supported states, legal actions and chance outcomes. Every
disagreement found here is settled by the decomp and logged in docs/discrepancies.md."""

import jax
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.outcome import outcome_to_py
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.strategies import any_outcome, legal_action_pair, supported_state

# Compiled once: every generated state has the same shapes and dtypes (plan ruling).
jit_step = jax.jit(step)


@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_jax_matches_reference(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    jax_out = jit_step(tables, s, acts, o)
    ref_out = from_ref(
        ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(o))
    )
    for name, a, b in zip(jax_out._fields, jax_out, ref_out):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
