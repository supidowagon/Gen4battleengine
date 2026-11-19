"""Pinned cases the nightly (fast tier) must run: the two Transformed-learner draws of G-69,
which also sit as @example rows on the slow ``test_learn_resolve_matches_reference``, and the
raise of ``initial_arrays`` for a four-battler fight without battler 3's bag (D1-5)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS
from nuzline.engine.levelup import needs_learn
from nuzline.engine.state import initial_arrays
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import (
    ref_advance_learns,
    ref_learn,
    ref_learn_options,
)
from tests.engine.d2_states import mon
from tests.test_differential import (
    _assert_same,
    _transformed_learner,
    jit_advance,
    jit_learn,
    jit_options,
)


@pytest.mark.parametrize("cursor", [13, 1])  # the terminator, and one row in (G-69)
def test_transformed_learner_matches_reference(tables, names, cursor):
    np_tables = to_numpy(tables)
    s = _transformed_learner(tables, names, cursor)
    got = jit_advance(tables, s)
    _assert_same(got, from_ref(ref_advance_learns(np_tables, to_ref(s))))
    for _ in range(6):
        if not bool(needs_learn(got)):
            break
        options = [bool(x) for x in np.asarray(jit_options(tables, got))]
        assert options == ref_learn_options(np_tables, to_ref(got)) and options[4]
        want = from_ref(ref_learn(np_tables, to_ref(got), 4))
        got = jit_learn(tables, got, jnp.int32(4))
        _assert_same(got, want)


@pytest.mark.parametrize("fmt", [DOUBLES, TWO_TRAINERS, MULTI])
def test_four_battler_fight_needs_battler_3s_bag(tables, names, fmt):
    parties = [[mon(names)] * 2 for _ in range(fmt.P)]
    with pytest.raises(ValueError, match="opp2_items"):
        initial_arrays(fmt, tables, parties)
    with pytest.raises(ValueError, match="opp2_items"):
        initial_arrays(fmt, tables, parties, opp2_items=None)
    assert initial_arrays(fmt, tables, parties, opp2_items=())["trainer_item_count"][3] == 0
