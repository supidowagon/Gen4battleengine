"""D3-2: the one absence predicate (``actions.absent_now`` / ``replacement_bench``, reference
``_absent_now`` / ``_replacement_bench``) and the skip of an actor that fainted before its turn.
The four-position step cases (sheet D3-2 4.5, bullets 1-4) belong to D3-5's loop."""

from __future__ import annotations

import numpy as np

from nuzline.engine.actions import absent_now, replacement_bench
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS, party
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _absent_now, _replacement_bench
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference


def _both(state, b: int) -> bool:
    """``absent_now`` of the JAX state and of the reference state, asserted equal."""
    jax_val = bool(absent_now(state, b))
    ref = to_ref(state)
    assert jax_val == _absent_now(ref, b)
    assert np.asarray(replacement_bench(state, b)).tolist() == _replacement_bench(ref, b) + [
        False
    ] * (6 - len(_replacement_bench(ref, b)))
    return jax_val


def _party_down(state, p: int, keep=()):
    """Every Pokemon of party ``p`` at 0 HP except the slots in ``keep``."""
    hp = state.hp.at[p].set(0)
    for slot in keep:
        hp = hp.at[p, slot].set(int(state.hp[p, slot]))
    return state._replace(hp=hp)


def test_doubles_battler_two_with_only_battler_zero_left_is_absent(tables, names):
    s = four(tables, names, DOUBLES, bench=2)  # party 0: slot 0 (battler 0), slot 1 (battler 2)
    s = with_hp(s, 2, 0)
    assert _both(s, 2)  # the one other living Pokemon is the partner's active


def test_doubles_battler_two_with_a_third_living_is_not_absent(tables, names):
    s = with_hp(four(tables, names, DOUBLES, bench=3), 2, 0)
    assert not _both(s, 2)  # a replacement is owed (D3-12)
    assert np.asarray(replacement_bench(s, 2)).tolist() == [False, False, True, False, False, False]


def test_a_living_battler_is_not_absent(tables, names):
    s = four(tables, names, DOUBLES, bench=2)
    assert not any(_both(s, b) for b in range(4))


def test_multi_battler_two_is_judged_on_party_two_only(tables, names):
    s = four(tables, names, MULTI, bench=2)
    s = with_hp(s, 2, 0)
    assert party(MULTI, 2) == 2
    assert not _both(s, 2)  # party 2's slot 1 lives (party 0's HP is not its business)
    absent = _party_down(s, 2)
    assert _both(absent, 2)
    # party 0 all down changes nothing for battler 2; party 2 living does not help battler 0
    assert not _both(_party_down(s, 0, keep=(1,)), 2)
    assert _both(_party_down(with_hp(s, 0, 0), 0), 0)


def test_two_trainers_battler_three_is_judged_on_party_two_only(tables, names):
    s = four(tables, names, TWO_TRAINERS, bench=2)
    s = with_hp(s, 3, 0)
    assert party_of(s, 3) == 2
    assert not _both(s, 3)  # party 2 slot 1 lives
    assert _both(_party_down(s, 2), 3)
    # party 0 down and battler 2 (party 0) at 0 HP: battler 3 is still not absent
    assert not _both(_party_down(s, 0), 3)


def test_two_trainers_player_side_shares_a_party_enemy_side_does_not(tables, names):
    s = four(tables, names, TWO_TRAINERS, bench=2)
    # player side (party 0): battler 2 down, its only other living Pokemon is battler 0's: absent
    assert _both(with_hp(s, 2, 0), 2)
    # enemy side: battler 1 (party 1, slot 0) down; battler 3 (party 2) sits on slot 1 of its
    # own party and must not be subtracted from party 1's bench
    s1 = s._replace(active=s.active.at[3].set(1))
    s1 = with_hp(s1, 1, 0)
    assert not _both(s1, 1)
    assert np.asarray(replacement_bench(s1, 1)).tolist() == [False, True, False, False, False, False]


def test_singles_replacement_bench_has_no_partner(tables, names):
    s = four(tables, names, SINGLES, bench=2)
    s = with_hp(s, 0, 0)
    assert not _both(s, 0)
    assert _both(_party_down(s, 0), 0)


def test_b2_second_actor_ko_by_the_first_is_skipped(tables, names):
    """B = 2: battler 1 is KO'd by faster battler 0; its move never runs. ``ref_step`` skips the
    action outright; the step's state equals it (the engine already played nothing, D3-2 4.4)."""
    s = four(tables, names, SINGLES, bench=2)
    s = with_moves(s, names, 0, ["Tackle"])
    s = with_moves(s, names, 1, ["Tackle"])
    s = with_hp(s, 1, 1)
    pp_before = np.asarray(s.pp).copy()
    out = assert_matches_reference(tables, s, [0, 0], empty_outcome(SINGLES))
    assert int(out.hp[1, 0]) == 0
    p0 = party_of(s, 0)
    assert int(out.hp[p0, 0]) == int(s.hp[p0, 0])  # battler 1's Tackle did not hit
    p1 = party_of(s, 1)
    assert np.asarray(out.pp[p1, 0]).tolist() == pp_before[p1, 0].tolist()  # no PP spent
    assert int(out.last_move[1]) == int(s.last_move[1])
