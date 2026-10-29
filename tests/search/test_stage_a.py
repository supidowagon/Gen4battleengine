"""Stage A ranking: Wilson intervals, common random numbers and the ranking order
(docs/design.md §4.3)."""

import jax
import pytest

from nuzline.ai.policy import UnsupportedAi
from nuzline.search.stage_a import ActionResult, Ranking, rank_actions, wilson
from tests.engine.helpers import roark_state


def test_wilson_bounds():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_ranking_covers_legal_actions_and_is_sorted(tables, names):
    s = roark_state(tables, names)
    r = rank_actions(tables, names, s, n=200, key=jax.random.PRNGKey(0))
    assert len(r) == 9 and all(a.ci_low <= a.p_win0 <= a.ci_high for a in r)
    assert [a.p_win0 for a in r] == sorted([a.p_win0 for a in r], reverse=True)


def test_common_random_numbers_are_deterministic(tables, names):
    s = roark_state(tables, names)
    a = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    b = rank_actions(tables, names, s, n=100, key=jax.random.PRNGKey(7))
    assert [(x.action, x.p_win0) for x in a] == [(x.action, x.p_win0) for x in b]


def test_all_zero_flag(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    s = s._replace(trapped=s.trapped.at[0].set(3))
    r = rank_actions(tables, names, s, n=50, key=jax.random.PRNGKey(1))
    assert r.all_zero is True


def test_sort_key_is_strict_and_ties_with_the_leader_are_flagged():
    # Plan ruling: sorted strictly by (-p_win0, -p_win, action); an action whose CI
    # overlaps the leader's is flagged, not reordered.
    rows = [
        ActionResult(3, "c", 0.50, 0.40, 0.60, 0.7, 0.3, 100),
        ActionResult(1, "a", 0.50, 0.40, 0.60, 0.8, 0.2, 100),
        ActionResult(0, "z", 0.10, 0.05, 0.18, 0.1, 0.9, 100),
        ActionResult(2, "b", 0.52, 0.42, 0.62, 0.6, 0.4, 100),
        ActionResult(4, "d", 0.50, 0.40, 0.60, 0.8, 0.2, 100),
    ]
    r = Ranking.of(rows)
    assert [a.action for a in r] == [2, 1, 4, 3, 0]
    assert [a.tied_with_best for a in r] == [True, True, True, True, False]
    assert r.all_zero is False


def test_labels_name_the_move_or_the_switch_target(tables, names):
    s = roark_state(tables, names, player_count=2)
    r = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(2))
    labels = {a.action: a.label for a in r}
    moves = [names.move_names[int(m)] for m in s.moves[0, 0]]
    target = names.species_names[int(s.species[0, 1])]
    assert labels == {0: moves[0], 1: moves[1], 2: moves[2], 3: moves[3], 5: f"Switch → {target}"}


def test_struggle_is_labelled(tables, names):
    s = roark_state(tables, names, player_count=1).with_pp(0, 0, [0, 0, 0, 0])
    r = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(3))
    assert [(a.action, a.label) for a in r] == [(0, "Struggle")]


def test_stats_are_counts_over_the_shared_playouts(tables, names):
    # One Pokemon at 1 HP with no PP: Struggle's recoil faints it on turn 1 in every
    # playout, a certain loss.
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    (only,) = rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(4))
    assert (only.p_win0, only.p_win, only.e_faints, only.n) == (0.0, 0.0, 1.0, 8)
    assert (only.ci_low, only.ci_high) == wilson(0, 8)


def test_refused_playouts_are_refused_loudly(tables, names):
    # Levitate on the player's lead is refused by the AI: the ranking must not count
    # those playouts as losses.
    s = roark_state(tables, names, player_species="Gastly", player_ability="Levitate")
    with pytest.raises(UnsupportedAi, match=r"(?s)refused.*Levitate"):
        rank_actions(tables, names, s, n=8, key=jax.random.PRNGKey(5))
