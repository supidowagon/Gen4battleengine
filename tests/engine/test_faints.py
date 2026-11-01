"""The death record and traced budget fields (plan 4b task 2): ``deaths`` per player party
slot, ``faint_budget``, ``protected``, ``replace_mul``, ``with_budget`` and
``decision_actions``."""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import NUM_ACTIONS, decision_actions, legal_actions
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import LAYOUT, MAX_DEATH_BUDGET, BattleState
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.engine.helpers import roark_real_state, roark_state

NEW_FIELDS = ("deaths", "faint_budget", "protected", "replace_mul")


def test_new_fields_default_to_zero(tables, names):
    s = roark_state(tables, names)
    for name in NEW_FIELDS:
        assert not np.any(np.asarray(getattr(s, name))), name
    assert BattleState._fields[-4:] == NEW_FIELDS
    assert LAYOUT["deaths"] == ((6,), np.int8)
    assert LAYOUT["faint_budget"] == ((), np.int8)
    assert LAYOUT["protected"] == ((6,), np.int8)
    assert LAYOUT["replace_mul"] == ((), np.int8)
    assert MAX_DEATH_BUDGET == 3


def test_with_budget(tables, names):
    s = roark_state(tables, names).with_budget(k=2, protected=[1, 3])
    assert int(s.faint_budget) == 2
    assert np.asarray(s.protected).tolist() == [0, 1, 0, 1, 0, 0]
    assert int(roark_state(tables, names).with_budget().faint_budget) == 0
    base = roark_state(tables, names)
    for bad in (4, -1):
        with pytest.raises(ValueError, match="budget"):
            base.with_budget(k=bad)
    with pytest.raises(ValueError, match="slot 6"):
        base.with_budget(protected=[6])
    with pytest.raises(ValueError, match="slot 3 is empty"):
        roark_state(tables, names, player_count=2).with_budget(protected=[3])
    with pytest.raises(ValueError, match="slot 2 has already fainted"):
        base.replace_hp(0, 2, 0).with_budget(protected=[2])


def test_faint_records_the_slot(tables, names):
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1)
    tackle = names.move_slot(s, 1, "Tackle")
    actions = jnp.array([0, tackle])
    out = step(tables, s, actions, empty_outcome())
    assert np.asarray(out.deaths).tolist() == [1, 0, 0, 0, 0, 0]
    assert int(out.player_faints) == 1
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(s),
            tuple(int(x) for x in actions),
            outcome_to_py(empty_outcome()),
        )
    )
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out, ref, strict=True))

    won = roark_state(tables, names).replace_hp(1, 0, 1)
    out = step(tables, won, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and not np.any(np.asarray(out.deaths))
    assert int(out.player_faints) == 0


def test_decision_actions(tables, names):
    s = roark_real_state(tables, names)
    assert decision_actions(tables, s).tolist() == legal_actions(tables, s, 0).tolist()
    down = s._replace(trapped=s.trapped.at[0].set(3)).replace_hp(0, 0, 0)
    mask = decision_actions(tables, down).tolist()
    assert len(mask) == NUM_ACTIONS
    assert mask == [False] * 4 + [False] + [True] * 5
