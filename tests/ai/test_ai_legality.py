"""The opponent-legality invariant (tests/ai_legality.py) as a property over generated states:
the AI's exact action distribution gives weight only to ``legal_opponent_actions``. Roark's
states (``test_distribution.roark_states``, which the vanilla layout supports) with the gates on
top: the AI's battler trapped (Bind, Mean Look), ingrained, Disabled, taunted, or its moves
sealed by the player's Imprison."""

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.ai.policy import UnsupportedAi, ai_action_distribution
from nuzline.engine.state import BattleState
from tests import ai_legality
from tests.ai.test_distribution import roark_states


@st.composite
def gated_states(draw, tables, names) -> BattleState:
    s = draw(roark_states(tables, names))
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    mon = int(a["active"][1])
    own = [int(m) for m in a["moves"][1, mon] if m]
    a["mean_look"][1] = draw(st.sampled_from([0, 0, 1]))
    a["ingrain"][1] = draw(st.sampled_from([0, 0, 1]))
    a["taunt"][1] = draw(st.sampled_from([0, 0, 3]))
    disabled = draw(st.sampled_from([0, *own]))
    a["disabled_move"][1], a["disable_turns"][1] = disabled, 4 if disabled else 0
    if draw(st.booleans()):  # the player imprisons, sharing one of the AI's moves
        a["imprison"][0] = 1
        a["moves"][0, int(a["active"][0]), 3] = draw(st.sampled_from(own))
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_ai_weights_only_legal_actions(tables, names, data):
    s = data.draw(gated_states(tables, names))
    try:
        probs = ai_action_distribution(tables, s)
    except UnsupportedAi:
        assume(False)
    ai_legality.check_distribution(tables, s, probs)


def test_the_check_catches_an_illegal_weight(tables, names):
    """The checker itself: weight on a switch while the AI's battler is ingrained fails."""
    import pytest

    from nuzline.engine.actions import MOVE_ACTIONS, NUM_OPP_ACTIONS
    from tests.ai.test_distribution import versus

    s = versus(tables, names, 0, "Starly")
    s = s._replace(ingrain=s.ingrain.at[1].set(1))
    probs = np.zeros(NUM_OPP_ACTIONS, np.float32)
    probs[MOVE_ACTIONS + 1] = 1.0
    with pytest.raises(ai_legality.IllegalAiAction):
        ai_legality.check_distribution(tables, s, probs)


def test_the_property_reaches_unrefused_states(tables, names):
    """One generated state with Ingrain and Mean Look on the AI's side is supported (no NaN in
    its action distribution), so the property reaches real picks there (B8-8)."""
    from hypothesis import find

    s = find(gated_states(tables, names), lambda s: s.ingrain[1] == 1 and s.mean_look[1] == 1)
    probs = np.asarray(ai_action_distribution(tables, s))
    assert not np.any(np.isnan(probs))
