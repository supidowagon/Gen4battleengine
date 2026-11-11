"""Disable (``effect_script_0086.s``, ``subscript_disable_start.s``,
``BtlCmd_TryDisable`` battle_script.c:4581): the target's last move is disabled for
``RandNext() % 4 + 3`` turns. A disabled move cannot be picked
(``CHECK_INVALID_DISABLED``, battle_lib.c:2255), fails if it was chosen before the
disabling (``CHECK_STATUS_STATE_DISABLE``, battle_controller_player.c:2543, before the PP
decrement), and the counter runs down at ``MON_COND_CHECK_STATE_DISABLE`` (:1565).

The player is a Psyduck with Disable against a Geodude (Rock Throw, Tackle) whose
``last_move`` is Rock Throw unless a test says otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.policy import ai_action_distribution
from nuzline.engine.actions import legal_opponent_actions
from nuzline.engine.ids import MOVE_STRUGGLE
from nuzline.engine.observe import ACT_DISABLED, OBS_FIELDS, observe
from nuzline.engine.order import selected_move
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import ACT_MOVE, step, step_events
from tests.engine.helpers import A, assert_matches_reference, roark_state

FIELDS = dict(OBS_FIELDS)
SPE = 4  # stats[..., 0..4] are Atk..Spe


@pytest.fixture
def state(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Psyduck",
        player_moves=("Disable", "Scratch"),
        opp_species="Geodude",
        opp_moves=("Rock Throw", "Tackle"),
        opp_count=1,
    )
    return s._replace(last_move=s.last_move.at[1].set(names.move("Rock Throw")))


def outcome(duration=4, hit=1):
    return empty_outcome()._replace(
        duration=jnp.full((2,), duration, jnp.int8),
        hit=jnp.full((2, 5), hit, jnp.int8),
    )


def disabled(s, names, move="Rock Throw", turns=4):
    return s._replace(
        disabled_move=s.disabled_move.at[1].set(names.move(move)),
        disable_turns=s.disable_turns.at[1].set(turns),
    )


def faster(s):
    return s._replace(stats=s.stats.at[0, 0, SPE].set(500))


@pytest.mark.slow
def test_disable_disables_last_move(tables, names, state):
    rock = names.move("Rock Throw")
    out = step(tables, state, A("Disable", "Tackle", state, names), outcome(4))
    # The counter drops at the end of the Disable turn itself.
    assert int(out.disabled_move[1]) == rock and int(out.disable_turns[1]) == 3
    assert int(out.disabled_move[0]) == 0
    legal = legal_opponent_actions(tables, out)
    assert not bool(legal[names.move_slot(out, 1, "Rock Throw")])
    assert bool(legal[names.move_slot(out, 1, "Tackle")])


def _no_last_move(s, names):
    return s._replace(last_move=s.last_move.at[1].set(0))


def _struggle(s, names):
    return s._replace(last_move=s.last_move.at[1].set(MOVE_STRUGGLE))


def _no_pp(s, names):
    return s._replace(pp=s.pp.at[1, 0, names.move_slot(s, 1, "Rock Throw")].set(0))


def _already_disabled(s, names):
    return disabled(s, names, "Tackle", 5)


def _unknown_move(s, names):
    return s._replace(last_move=s.last_move.at[1].set(names.move("Water Gun")))


def _same(s, names):
    return s


FAIL_SETUPS = (
    pytest.param(_no_last_move, 1, id="no-last-move"),
    pytest.param(_struggle, 1, id="struggle"),
    pytest.param(_no_pp, 1, id="no-pp"),
    pytest.param(_already_disabled, 1, id="already-disabled"),
    pytest.param(_unknown_move, 1, id="not-a-known-move"),
    pytest.param(_same, 0, id="missed"),
)


@pytest.mark.slow
@pytest.mark.parametrize(("setup", "hit"), FAIL_SETUPS)
def test_disable_fails(tables, names, state, setup, hit):
    s = setup(state, names)
    before = int(s.disabled_move[1]), int(s.disable_turns[1])
    out = step(tables, s, A("Disable", "Tackle", s, names), outcome(4, hit))
    # Nothing new is disabled; an earlier Disable only ticks down at the end of the turn.
    expected = (before[0], before[1] - 1) if before[0] else (0, 0)
    assert (int(out.disabled_move[1]), int(out.disable_turns[1])) == expected


@pytest.mark.slow
def test_disable_countdown(tables, names, state):
    rock = names.move("Rock Throw")
    s = step(tables, state, A("Disable", "Tackle", state, names), outcome(3))
    seen = [(int(s.disabled_move[1]), int(s.disable_turns[1]))]
    for _ in range(3):
        s = step(tables, s, A("Scratch", "Tackle", s, names), outcome(3))
        seen.append((int(s.disabled_move[1]), int(s.disable_turns[1])))
    # 3 -> 2 on the Disable turn, 2 -> 1, 1 -> 0, then the move is cleared.
    assert seen == [(rock, 2), (rock, 1), (rock, 0), (0, 0)]


def test_unknown_disabled_move_is_cleared_at_once(tables, names, state):
    # The counter is zeroed and the move cleared in the same check (bcp.c:1568-1581).
    s = disabled(state, names, "Water Gun", 5)
    out = step(tables, s, A("Scratch", "Tackle", s, names), outcome())
    assert (int(out.disabled_move[1]), int(out.disable_turns[1])) == (0, 0)


@pytest.mark.slow
def test_countdown_end_is_observed(tables, names, state):
    rock = names.move("Rock Throw")
    acts = A("Scratch", "Tackle", state, names)
    for turns, shown in ((0, 0), (1, rock)):
        s = disabled(state, names, "Rock Throw", turns)
        out = step(tables, s, acts, outcome())
        obs = np.asarray(observe(tables, s, acts, outcome(), out))
        assert obs[FIELDS["disabled_move"]].tolist() == [0, shown]


@pytest.mark.slow
def test_chosen_move_disabled_mid_turn(tables, names, state):
    # The faster Psyduck disables the Rock Throw Geodude chose this turn.
    s = faster(state)
    acts = A("Disable", "Rock Throw", s, names)
    rock = names.move("Rock Throw")
    slot = names.move_slot(s, 1, "Rock Throw")
    out, events = step_events(tables, s, acts, outcome(4))
    assert int(events["act_kind"][1]) == ACT_DISABLED and int(events["act_id"][1]) == rock
    assert int(events["act_kind"][0]) == ACT_MOVE
    assert int(out.pp[1, 0, slot]) == int(s.pp[1, 0, slot])  # no PP spent
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])  # Rock Throw never hit
    assert int(out.last_move[1]) == 0  # a disrupted move is not a successful one
    obs = np.asarray(observe(tables, s, acts, outcome(4), out))
    assert obs[FIELDS["act_kind"]].tolist() == [ACT_MOVE, ACT_DISABLED]
    assert obs[FIELDS["act_id"]].tolist()[1] == rock
    assert obs[FIELDS["disabled_move"]].tolist() == [0, rock]


def test_ai_avoids_disabled_slot(tables, names, state):
    rock = names.move_slot(state, 1, "Rock Throw")
    assert np.asarray(ai_action_distribution(tables, state))[rock] > 0
    dist = np.asarray(ai_action_distribution(tables, disabled(state, names)))
    assert dist[rock] == 0 and abs(dist.sum() - 1) < 1e-5
    assert dist[names.move_slot(state, 1, "Stealth Rock")] > 0


def test_ai_struggles_when_all_disabled(tables, names):
    # Review Focus 3: every slot invalid (STRUGGLING_ALL, bcp.c:433) -- the whole AI mass is
    # on action 0 and the stepped move is Struggle.
    s = roark_state(
        tables,
        names,
        player_species="Psyduck",
        player_moves=("Scratch",),
        opp_species="Geodude",
        opp_moves=("Rock Throw",),
        opp_count=1,
    )
    rock = names.move("Rock Throw")
    s = s._replace(moves=s.moves.at[1, 0].set(jnp.asarray([rock, 0, 0, 0], s.moves.dtype)))
    s = disabled(s, names, "Rock Throw", 4)
    dist = np.asarray(ai_action_distribution(tables, s))
    assert dist[0] == pytest.approx(1.0) and dist[1:].sum() == 0
    move, struggling = selected_move(tables, s, 1, 0)
    assert int(move) == MOVE_STRUGGLE and bool(struggling)
    _, events = step_events(tables, s, A("Scratch", 0, s, names), outcome())
    assert int(events["act_kind"][1]) == ACT_MOVE and int(events["act_id"][1]) == MOVE_STRUGGLE


@pytest.mark.slow
def test_disable_matches_reference(tables, names, state):
    for hit in (1, 0):
        for duration in (3, 6):
            o = outcome(duration, hit)
            for setup in (
                _same,
                _no_last_move,
                _struggle,
                _no_pp,
                _already_disabled,
                _unknown_move,
            ):
                s = setup(state, names)
                assert_matches_reference(tables, s, A("Disable", "Tackle", s, names), o)
    # The countdown and the move-time check.
    for turns in (0, 1, 3):
        s = disabled(state, names, "Rock Throw", turns)
        for opp in ("Rock Throw", "Tackle"):
            assert_matches_reference(tables, s, A("Scratch", opp, s, names), outcome())
    s = faster(state)
    assert_matches_reference(tables, s, A("Disable", "Rock Throw", s, names), outcome())
    gone = disabled(state, names, "Water Gun", 5)
    assert_matches_reference(tables, gone, A("Scratch", "Tackle", gone, names), outcome())
