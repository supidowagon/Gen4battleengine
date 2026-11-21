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


# --- D3-2 sheet 4.5 bullets 1-4: the four-position step (D3-5's loop) -------------------------


def _play(tables, s, acts, order, targets, outcome=None):
    from tests.engine.test_d3_action_loop import play

    return play(tables, s, acts, order, targets, outcome)


def test_absent_battler_two_runs_nothing(tables, names):
    """Battler 2 at 0 HP with nothing to send in: PP, ``last_move``, ``chosen_target`` and
    ``protect_count`` unchanged, ``act_kind`` ACT_NONE."""
    s = four(tables, names, DOUBLES, bench=2)
    s = with_hp(s, 2, 0)
    assert bool(absent_now(s, 2))
    out, shown = _play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 2, 2))
    p, slot = party_of(s, 2), int(s.active[2])
    assert np.asarray(out.pp[p, slot]).tolist() == np.asarray(s.pp[p, slot]).tolist()
    assert int(out.last_move[2]) == int(s.last_move[2])
    assert int(out.chosen_target[2]) == int(s.chosen_target[2])
    assert int(out.protect_count[2]) == int(s.protect_count[2])
    assert int(shown["act_kind"][2]) == 0


def test_battler_one_koed_by_battler_zero_does_not_act(tables, names):
    s = four(tables, names, DOUBLES, bench=2)
    s = with_hp(s, 1, 1)
    out, _ = _play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 3, 2))
    p0, p1, slot1 = party_of(s, 0), party_of(s, 1), int(s.active[1])
    assert int(out.hp[p1, slot1]) == 0
    assert int(out.hp[p0, int(s.active[0])]) == int(s.hp[p0, int(s.active[0])])
    assert np.asarray(out.pp[p1, slot1]).tolist() == np.asarray(s.pp[p1, slot1]).tolist()
    assert int(out.last_move[1]) == int(s.last_move[1])


def test_battler_three_locked_into_rollout_runs_it_without_pp(tables, names):
    s = four(tables, names, DOUBLES, bench=2)
    s = with_moves(s, names, 3, ["Tackle", "Swords Dance", "Rollout"])
    rollout = names.move("Rollout")
    s = s._replace(
        locked_move=s.locked_move.at[3].set(rollout),
        rollout_count=s.rollout_count.at[3].set(3),
        chosen_target=s.chosen_target.at[3].set(2),
    )
    out, _ = _play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 3, 2))
    p3, slot3 = party_of(s, 3), int(s.active[3])
    assert int(out.pp[p3, slot3, 2]) == int(s.pp[p3, slot3, 2])  # no PP
    assert int(out.last_move[3]) == rollout
    assert int(out.hp[party_of(s, 2), int(s.active[2])]) < int(
        s.hp[party_of(s, 2), int(s.active[2])]
    )


def test_battler_one_under_uproar_runs_it_without_pp(tables, names):
    s = four(tables, names, DOUBLES, bench=2)
    s = with_moves(s, names, 1, ["Tackle", "Uproar"])
    uproar = names.move("Uproar")
    s = s._replace(
        locked_move=s.locked_move.at[1].set(uproar), uproar=s.uproar.at[1].set(2)
    )
    out, _ = _play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 3, 2))
    p1, slot1 = party_of(s, 1), int(s.active[1])
    assert int(out.pp[p1, slot1, 1]) == int(s.pp[p1, slot1, 1])
    assert int(out.last_move[1]) == uproar


def test_reference_plays_battler_threes_action(tables, names):
    s = four(tables, names, DOUBLES, bench=2)
    s = with_moves(s, names, 3, ["Tackle"])
    out, _ = _play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), (1, 0, 3, 2))
    p3, slot3 = party_of(s, 3), int(s.active[3])
    assert int(out.pp[p3, slot3, 0]) == int(s.pp[p3, slot3, 0]) - 1
