"""D3-6: ``Battler_MovedThisTurn`` per battler (Fact A), the actor is the last to act (Fact B) and
the defender's turn-order item flags (Fact C), at B = 2. The step carries ``moved`` bool[B] and
``last_to_act`` in place of ``target_moved`` (``position == 1``); at B = 2 the values are today's
(sheet D3-6 4.5). The DOUBLES cases (sheet 4.6) are D3-5's, the Pursuit / Roar Fact C case D3-9's.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.format import DOUBLES
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


# --- D3-6 sheet 4.6 (DOUBLES), carried by D3-5's four-position loop ---------------------------


def _four_play(tables, s, acts, order, targets, outcome=None):
    from tests.engine.test_d3_action_loop import play

    return play(tables, s, acts, order, targets, outcome)


def _hp4(s, b):
    from tests.engine.test_d3_action_loop import hp

    return hp(s, b)


def test_payback_doubles_against_a_defender_that_moved(tables, names):
    from tests.engine.d2_states import four, with_moves

    s = four(tables, names)
    s = with_moves(s, names, 0, ["Payback"])
    for b in (1, 2, 3):
        s = with_moves(s, names, b, ["Swords Dance"])
    moved, _ = _four_play(tables, s, [0, 0, 0, 0], (1, 0, 3, 2), (1, 0, 0, 0))
    waiting, _ = _four_play(tables, s, [0, 0, 0, 0], (1, 0, 3, 2), (3, 0, 0, 0))
    d1, d3 = _hp4(s, 1) - _hp4(moved, 1), _hp4(s, 3) - _hp4(waiting, 3)
    assert d3 > 0 and abs(d1 - 2 * d3) <= 2


def test_zoom_lens_lifts_the_threshold_only_against_a_mover(tables, names):
    from tests.engine.d2_states import four, with_moves
    from tests.engine.test_d3_action_loop import _set_item

    s = four(tables, names)
    s = with_moves(s, names, 0, ["Tackle"])
    for b in (1, 2, 3):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = _set_item(s, 0, names.item("Zoom Lens"))
    o = empty_outcome(DOUBLES)._replace(hit=empty_outcome(DOUBLES).hit.at[0, 0].set(96))
    hit, _ = _four_play(tables, s, [0, 0, 0, 0], (1, 0, 3, 2), (1, 0, 0, 0), o)
    miss, _ = _four_play(tables, s, [0, 0, 0, 0], (1, 0, 3, 2), (3, 0, 0, 0), o)
    assert _hp4(hit, 1) < _hp4(s, 1)
    assert _hp4(miss, 3) == _hp4(s, 3)


def test_protect_last_fails_third_of_four_succeeds(tables, names):
    from tests.engine.d2_states import four, with_moves

    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Protect" if b == 2 else "Tackle"])
    for roll in (0, 1):
        o = empty_outcome(DOUBLES)
        o = o._replace(protect=o.protect.at[2].set(roll))
        _, shown = _four_play(tables, s, [0, 0, 0, 0], (0, 1, 3, 2), (1, 0, 0, 2), o)
        assert int(shown["protect_ok"][2]) == 0  # last: fails whatever the roll
    out, shown = _four_play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 0, 2))
    assert int(shown["protect_ok"][2]) == 1  # third of four, battler 3 still to act
    assert _hp4(out, 2) == _hp4(s, 2)  # battler 3's Tackle at 2 is blocked


def test_fake_out_flinches_only_a_defender_that_has_not_moved(tables, names):
    from tests.engine.d2_states import four, with_moves

    s = four(tables, names)
    s = with_moves(s, names, 0, ["Fake Out"])
    s = s._replace(fake_out_turn=s.fake_out_turn.at[0].set(s.turn))  # Fake Out works now
    for b in (1, 2, 3):
        s = with_moves(s, names, b, ["Swords Dance"])
    o = empty_outcome(DOUBLES)
    o = o._replace(secondary=o.secondary.at[0, 0].set(1))
    early, _ = _four_play(tables, s, [0, 0, 0, 0], (0, 3, 1, 2), (3, 0, 0, 0), o)
    late, _ = _four_play(tables, s, [0, 0, 0, 0], (3, 0, 1, 2), (3, 0, 0, 0), o)
    # the flinch costs battler 3 its action (it moves after the Fake Out); a mover is not flinched
    assert int(early.last_move[3]) == int(s.last_move[3])
    assert int(late.last_move[3]) == names.move("Swords Dance")


def test_roar_into_slot_three_clears_its_quick_claw_flag(tables, names):
    from tests.engine.d2_states import four, with_moves
    from tests.engine.test_d3_action_loop import _set_item

    lum = names.item("Lum Berry")
    s = four(tables, names, bench=3)
    s = with_moves(s, names, 0, ["Roar"])
    s = with_moves(s, names, 2, ["Knock Off"])
    s = s._replace(claw_flag=s.claw_flag.at[3].set(1))
    s = _set_item(_set_item(s, 3, lum), 1, 0)
    s = s._replace(item=s.item.at[1, 2].set(lum))  # the newcomer (party 1 slot 2)
    o = empty_outcome(DOUBLES)
    o = o._replace(roar_target=o.roar_target.at[0].set(1))
    out, _ = _four_play(tables, s, [0, 0, 0, 0], (0, 2, 1, 3), (3, 0, 3, 0), o)
    assert int(out.active[3]) == 2
    assert int(out.item[1, 2]) == 0  # knocked off: the flag went with the old Pokemon
    # control: no Roar, the flag stands and Knock Off fails
    kept = with_moves(s, names, 0, ["Tackle"])
    kept = _set_item(kept, 3, lum)
    ctl, _ = _four_play(tables, kept, [0, 0, 0, 0], (0, 2, 1, 3), (1, 0, 3, 0))
    assert int(ctl.item[1, 1]) == lum
