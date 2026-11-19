"""The coverage-dependent refusals the search and playout tests rely on, checked directly on
the same states without compiling a playout, so the fullset tier stays fast. The playout
tests that exercise these refusals end to end stay in their slow modules, unmarked."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.ai import policy
from nuzline.search.playout import replacement_refusal, send_replacements
from tests.engine.helpers import roark_state


@pytest.mark.fullset
def test_a_replacement_after_an_ai_switch_is_refused(tables, names):
    # roark.md section 14 (3): after the AI switched this turn, Task_TrainerShowPartyMenu
    # reads the stale aiSwitchedPartySlot; the engine refuses that path (the refusal behind
    # tests/search/fixtures.py::switch_refusal_root).
    s = roark_state(tables, names).replace_hp(1, 0, 0)
    _, refused = send_replacements(tables, s, jnp.bool_(True), 10)
    assert int(refused) == 1
    assert "aiSwitchedPartySlot" in replacement_refusal(tables, s, True, 10)
    assert replacement_refusal(tables, s, False, 10) is None


@pytest.mark.fullset
def test_a_levitate_lead_is_refused_by_the_ai(tables, names):
    # The AI refusal the playout tests' Gastly / Levitate lead stops on (policy
    # _DEFENDER_ABILITIES), read on the same state.
    s = roark_state(tables, names, player_species="Gastly", player_ability="Levitate")
    assert any("Levitate" in r for r in policy.ai_refusals(tables, s, names))
