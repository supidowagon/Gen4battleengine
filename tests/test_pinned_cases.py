"""Pinned cases the nightly (fast tier) must run: the two Transformed-learner draws of G-69,
which also sit as @example rows on the slow ``test_learn_resolve_matches_reference``, and the
raise of ``initial_arrays`` for a four-battler fight without battler 3's bag (D1-5)."""

import pytest

from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS
from nuzline.engine.state import initial_arrays
from tests.engine.d2_states import mon
from tests.test_differential import _learns_match_reference, _transformed_learner


@pytest.mark.parametrize("cursor", [13, 1])  # the terminator, and one row in (G-69)
def test_transformed_learner_matches_reference(tables, names, cursor):
    _learns_match_reference(tables, _transformed_learner(tables, names, cursor))


@pytest.mark.parametrize("fmt", [DOUBLES, TWO_TRAINERS, MULTI])
def test_four_battler_fight_needs_battler_3s_bag(tables, names, fmt):
    parties = [[mon(names)] * 2 for _ in range(fmt.P)]
    with pytest.raises(ValueError, match="opp2_items"):
        initial_arrays(fmt, tables, parties)
    with pytest.raises(ValueError, match="opp2_items"):
        initial_arrays(fmt, tables, parties, opp2_items=None)
    assert initial_arrays(fmt, tables, parties, opp2_items=())["trainer_item_count"][3] == 0
