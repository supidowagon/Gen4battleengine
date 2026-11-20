"""D3-6: ``Battler_MovedThisTurn`` per battler (Fact A), the actor is the last to act (Fact B) and
the defender's turn-order item flags (Fact C), at B = 2. The step carries ``moved`` bool[B] and
``last_to_act`` in place of ``target_moved`` (``position == 1``); at B = 2 the values are today's
(sheet D3-6 4.5). The DOUBLES cases (sheet 4.6) are D3-5's, the Pursuit / Roar Fact C case D3-9's.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_pivot import _mon
from tests.strategies import any_outcome, legal_action_pair, supported_state


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@given(data=st.data())
def test_drawn_singles_turns_equal_the_reference(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    assert_matches_reference(tables, s, acts, o)


def test_protect_second_fails_whatever_the_roll(tables, names):
    """Fact B: the actor is last (the foe is MOVE_END), TryProtection fails (script.c:5161)."""
    s = roark_state(tables, names, player_moves=["Protect"], opp_moves=["Protect"], opp_faster=True)
    acts = A("Protect", "Protect", s, names)
    for roll in (0, 1):
        o = empty_outcome()._replace(protect=empty_outcome().protect.at[0].set(roll))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.protect_count[0]) == 0 and int(out.protect_count[1]) == 1
    first = roark_state(tables, names, player_moves=["Protect"])
    out = assert_matches_reference(
        tables, first, A("Protect", "Tackle", first, names), empty_outcome()
    )
    assert int(out.protect_count[0]) == 1  # first: it succeeds


def test_payback_second_doubles_its_power(tables, names):
    """Fact A of the defender: the faster Geodude has acted (script.c:7198-7208)."""
    n = names
    move = n.move("Payback")
    s = make_state(
        tables, [_mon(n, "Slowpoke", ["Payback"], 30)], [_mon(n, "Geodude", ["Leer"], 30)]
    )
    out = assert_matches_reference(tables, s, (0, 0), empty_outcome())
    doubled = int(damage_for_roll(tables, s, 0, move, False, 15, power_mul=20))
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == min(doubled, int(s.hp[1, 0]))
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == DOUBLED_PAYBACK


def test_zoom_lens_second_lifts_the_accuracy_class(tables, names):
    """Fact A: Tackle's 95 becomes 114 only when the foe has moved (bcp.c:2963-2964)."""
    for opp_faster, dealt in ((True, True), (False, False)):
        s = roark_state(tables, names, player_moves=["Tackle"], opp_faster=opp_faster)
        s = s._replace(item=s.item.at[0, 0].set(names.item("Zoom Lens")))
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(96))
        out = assert_matches_reference(tables, s, A("Tackle", "Leer", s, names), o)
        assert (int(out.hp[1, 0]) < int(s.hp[1, 0])) is dealt


DOUBLED_PAYBACK = 21  # measured at the base (D3-2, 5ac321c), before D3-6
