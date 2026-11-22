"""``outcome.future_thresholds`` returns ``future_values: bool[B, 101]`` (the possible thresholds
of ``_future_grid``, clipped to 0-100) in place of the grid itself (``int32[B, K]`` and
``bool[B, K]``). Old against new, for seeded states (the ``gate`` Hypothesis profile is
derandomized): the value set, the Future Sight hit's class law (``enumerate._future_hit``) and the
sampler's class (``outcome.hit_class_mask`` against ``hit_class`` over the grid) are the same.
``_future_grid`` is the old computation, kept as the inner helper; ``_old_future_hit`` is the
old ``_future_hit`` over the grid."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.enumerate import _future_hit, hit_class_law
from nuzline.engine.outcome import (
    _future_grid,
    future_thresholds,
    hit_class,
    hit_class_mask,
)
from tests.engine.d2_states import four
from tests.engine.test_batch8 import _future_due
from tests.strategies import four_state, legal_action_pair, legal_actions_four, supported_state

ROLLS = np.arange(1, 101, dtype=np.int32)
_grid = jax.jit(_future_grid)
_values = jax.jit(future_thresholds)


def _old_future_hit(laws, side):
    """``enumerate._future_hit`` as it read the grid before ``future_values``."""
    if bool(laws.future_loose[side]):
        return list(range(1, 101)), [1] * 100, 100
    possible = np.asarray(laws.future_possible[side], bool)
    thresholds = np.asarray(laws.future_thresholds[side])[possible]
    if thresholds.size == 0:
        return [1], [1], 1
    return hit_class_law(thresholds)


def _same(tables, state, actions):
    """Old against new on one state; returns how many sides had a possible threshold."""
    thr, ok, loose = (np.asarray(x) for x in _grid(tables, state, actions))
    values, loose_new = (np.asarray(x) for x in _values(tables, state, actions))
    assert values.shape == (thr.shape[0], 101) and values.dtype == bool
    assert np.array_equal(loose, loose_new)
    old = SimpleNamespace(future_thresholds=thr, future_possible=ok, future_loose=loose)
    new = SimpleNamespace(future_values=values, future_loose=loose_new)
    live = 0
    for d in range(thr.shape[0]):
        want = {int(min(max(t, 0), 100)) for t in thr[d][ok[d]]}
        assert want == set(np.flatnonzero(values[d]).tolist()), d
        assert _future_hit(new, d) == _old_future_hit(old, d), d
        a = np.asarray(hit_class(ROLLS, jnp.asarray(thr[d]), jnp.asarray(ok[d])))
        b = np.asarray(hit_class_mask(ROLLS, jnp.asarray(values[d])))
        assert np.array_equal(a, b), d
        live += bool(want)
    return live


@pytest.mark.parametrize("foe_move", ["Growl", "Haze", "Gravity", "Embargo", "Lock-On", "Roar"])
def test_hand_states_with_a_pending_future_sight(tables, names, foe_move):
    """B = 2, a Future Sight due on battler 1 (the cases of tests/engine/test_batch8.py)."""
    s = _future_due(tables, names, foe_move)
    assert _same(tables, s, jnp.asarray([0, 0], jnp.int32)) >= 1


def test_hand_state_with_no_future_sight(tables, names):
    """B = 2, none pending: the same classes (the grid still has its entries)."""
    s = _future_due(tables, names, "Growl")
    s = s._replace(
        future_timer=jnp.zeros_like(s.future_timer),
        future_move=jnp.zeros_like(s.future_move),
        future_attacker=jnp.zeros_like(s.future_attacker),
        future_sight=jnp.zeros_like(s.future_sight),
        future_damage=jnp.zeros_like(s.future_damage),
    )
    _same(tables, s, jnp.asarray([0, 0], jnp.int32))


def test_four_battlers_pending_and_not(tables, names):
    """B = 4 (``future_thresholds`` without ``dsets``): a Future Sight of battler 2's on
    battler 0, and none."""
    s = four(tables, names)
    acts = jnp.zeros(4, jnp.int32)
    _same(tables, s, acts)
    pending = s._replace(
        future_timer=s.future_timer.at[0].set(1),
        future_damage=s.future_damage.at[0].set(5),
        future_move=s.future_move.at[0].set(names.move("Future Sight")),
        future_attacker=s.future_attacker.at[0].set(3),
        future_sight=s.future_sight.at[0].set(1),
    )
    _same(tables, pending, acts)


SEEDED = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


@SEEDED
@given(data=st.data())
def test_drawn_states_b2(tables, names, data):
    """Drawn singles states (a pending Future Sight in about a third, ``_draw_future_sight``)."""
    s = data.draw(supported_state(tables, names))
    _same(tables, s, data.draw(legal_action_pair(tables, s)))


@settings(SEEDED, max_examples=12)
@given(data=st.data())
def test_drawn_states_b4(tables, names, data):
    """Drawn four-battler states, the commands' actions."""
    s = data.draw(four_state(tables, names))
    acts, _ = data.draw(legal_actions_four(tables, s))
    _same(tables, s, jnp.asarray(acts))
