"""D3-8 on the reference: the DOUBLES sandstorm double faint through the reference's field pass
(``_field_conditions``: ``_faint_check`` queues ``(battler, slot)``, ``_pay_exp`` pops the lowest
battler first) equals the engine's ``field_conditions``, field by field."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import field_conditions
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _field_conditions, _Turn
from tests.engine.test_d3_payout_order import HOLDER, sandstorm_double_faint


def test_ref_double_faint_pays_battler_1_first_and_equals_the_engine(tables, names):
    s = sandstorm_double_faint(tables, names)
    ref = to_ref(s)
    turn = _Turn(
        to_numpy(tables), outcome_to_py(empty_outcome(DOUBLES)), entry_turn=1, battler_count=4
    )
    _field_conditions(turn, ref, [3, 1])
    got = from_ref(ref)
    assert np.asarray(got.evs[0, HOLDER]).tolist() == [100, 101, 100, 100, 100, 9]
    want = jax.jit(field_conditions)(tables, s, jnp.asarray([3, 1], jnp.int32))[0]
    for name, x, y in zip(want._fields, want, got, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
