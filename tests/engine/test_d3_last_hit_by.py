"""D3-1 note: Pursuit's hit on a switching battler does not write ``last_hit_by``. Its damage goes
through BATTLE_SUBSCRIPT_UPDATE_HP (subscript_pursuit.s:36), not the controller's UpdateHP, whose
bcp.c:3357 is the only ``lastHitByBattler`` write. B = 4, JAX against reference (pod failure on
457d605, test_jax_matches_reference_four)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.coverage import four_turn_refusals
from nuzline.engine.format import MULTI
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.d2_states import four, party_of, with_moves


def test_pursuit_on_a_switcher_leaves_last_hit_by(tables, names):
    s = with_moves(four(tables, names, MULTI, bench=2), names, 0, ["Pursuit"])
    for b in (2, 3):  # battlers 2 and 3 absent (D3-1c): their parties have no HP
        p = party_of(s, b)
        s = s._replace(hp=s.hp.at[p].set(0))
    acts = jnp.asarray([0, MOVE_ACTIONS + 1, 0, 0], jnp.int32)  # 0 Pursuit, 1 switches to slot 1
    targets = [1, 1, 2, 3]
    assert four_turn_refusals(tables, s, acts) == []
    o = empty_outcome(MULTI)
    out = step(tables, s, acts, o, jnp.asarray(targets, jnp.int8))
    ref = from_ref(
        ref_step(
            to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o), targets
        )
    )
    p1 = party_of(s, 1)
    assert int(out.hp[p1, 0]) < int(s.hp[p1, 0])  # the Pursuit hit landed on the switcher
    assert int(out.active[1]) == 1
    assert np.asarray(out.last_hit_by).tolist() == [0, 0, 0, 0]
    for name, a, b in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
