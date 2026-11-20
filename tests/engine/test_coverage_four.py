"""D3-1 amendment: the coverage gate of the four-battler turn (``coverage.four_turn_refusals``)."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.coverage import FOUR_TURN_CLAUSES, four_turn_refusals
from tests.strategies import legal_action_pair, supported_state

OWED = [
    "D3-14",
    "D3-16",
]


@pytest.mark.fullset
def test_four_turn_clauses_are_owed():
    """Each D3 group deletes its own clause; the tuple is empty after D3-16."""
    assert [c.group for c in FOUR_TURN_CLAUSES] == OWED


@settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_two_battler_turn_is_never_refused(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    assert four_turn_refusals(tables, s, acts) == []
