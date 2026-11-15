"""E2 batch 7 follow-ups (rulings B7-12.., docs/kaizo/e2-mechanics-ledger.md), each hand case
checked against the reference engine (``assert_matches_reference``)."""

from __future__ import annotations

import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.ids import AbilityId
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()


@pytest.mark.parametrize("pivot", ["Baton Pass", "U-turn"])
def test_a_switch_never_runs_the_pivot_in_its_slot(tables, names, pivot):
    """B7-12 (the differential divergence found at f22b818, from 470fbef): a switch action to
    party slot 1 clips to move slot 3, and a Baton Pass or U-turn held there paused the action, so
    the switch ran again on the newcomer (its Magic Guard restored away, the hazards twice). The
    turn branches on the command picked (bcp.c:893): a switch runs no move script."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Pikachu", ["Tackle"], 19)],
        [
            _mon(n, "Chansey", ["Bide", "Disable", "Pursuit", pivot], 30),
            _mon(n, "Carvanha", ["Bite"], 20),
        ],
    )
    s = s._replace(
        ability=s.ability.at[1, 1].set(AbilityId.MAGIC_GUARD),
        stealth_rock=s.stealth_rock.at[1].set(1),
        spikes=s.spikes.at[1].set(2),
    )
    hp = int(s.hp[1, 1])
    out = assert_matches_reference(tables, s, (0, MOVE_ACTIONS + 1), O)
    assert int(out.pivot_battler) == 0 and int(out.active[1]) == 1
    assert int(out.ability[1, 1]) == AbilityId.MAGIC_GUARD  # never re-entered
    assert int(out.hp[1, 1]) < hp  # Pikachu's Tackle found its target
