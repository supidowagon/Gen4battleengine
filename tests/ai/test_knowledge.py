"""What the AI knows about the player (docs/ai/roark.md section 9), kept in
``BattleState.ai_seen_moves`` / ``ai_ability_known`` by both engines.

* Seen moves: ``movePrevByBattler`` (written in ``UpdateMoveBuffers``,
  battle_controller_player.c:3945-3949) holds the player's move once it got past the
  status checks, the PP decrement and the no-target check (``SYSCTL_MOVE_SUCCEEDED``,
  :3212; a miss still counts), else ``MOVE_NONE``; ``BtlCmd_TryPursuit``
  (battle_script.c:6930) also writes it for a pre-switch Pursuit. It is
  ``BattleState.last_move``. ``TrainerAI_RecordLastMove`` (trainer_ai.c:2705) copies it
  into the AI's list at the start of the AI's FIGHT decision only (trainer_ai.c:295), so
  ``step`` commits it to ``ai_seen_moves`` at the start of a turn on which the
  opponent's action is a move, and the scores of that decision include it.
* Ability: ``BattleAI_SetAbility`` (battle_script.c:12106), from a script message that
  tags the battler's ability (``BattleMessage_AbilityTag``, :11798). Messages the C code
  builds in ``msgBuffer`` (Intimidate's "cuts", Keen Eye's "prevents") reveal nothing.
* Both are cleared when the player's battler switches in (``BattleSystem_UpdateAfterSwitch``,
  battle_lib.c:2151-2153).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.outcome import enumerate_ai_outcomes, sample_ai_outcome
from nuzline.ai.policy import ai_scores, resolve_player_ability
from nuzline.data.tables import to_numpy
from nuzline.engine.ids import MOVE_STRUGGLE, Status
from nuzline.engine.outcome import MAX_HITS, empty_outcome, outcome_to_py
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step
from tests.ai.fixtures import trace_c
from tests.engine.helpers import A, roark_state

SWITCH_TO_1 = 4 + 1


def _both(tables, s, acts, o=None):
    """``step`` and ``ref_step`` agree on every field; returns the JAX result."""
    o = empty_outcome() if o is None else o
    out = step(tables, s, jnp.asarray(acts, jnp.int32), o)
    ref = from_ref(
        ref_step(to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o))
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


# --- Seen moves -------------------------------------------------------------------


def _seen(s) -> list[int]:
    """The ``ai_seen_moves`` row of the player's active Pokemon."""
    return np.asarray(s.ai_seen_moves[int(s.active[0])]).tolist()


@pytest.mark.slow
def test_used_move_waits_in_last_move_until_the_ai_fights(tables, names):
    s = roark_state(tables, names)
    used = _both(tables, s, A("Scratch", "Leer", s, names))
    assert int(used.last_move[0]) == names.move("Scratch")
    assert _seen(used) == [0, 0, 0, 0]  # TrainerAI_RecordLastMove has not run yet
    fought = _both(tables, used, A("Leer", "Tackle", used, names))
    assert _seen(fought) == [1, 0, 0, 0]  # Scratch, committed at the AI's FIGHT decision
    assert int(fought.last_move[0]) == names.move("Leer")  # pending again


@pytest.mark.slow
def test_move_replaced_while_the_ai_heals_is_never_recorded(tables, names):
    s = roark_state(tables, names)
    potion, second_potion = 10, 11
    s = s._replace(hp=s.hp.at[1, 0].set(5))
    t1 = _both(tables, s, A("Scratch", potion, s, names))
    assert int(t1.last_move[0]) == names.move("Scratch")
    t2 = _both(tables, t1, A("Leer", second_potion, t1, names))
    assert _seen(t2) == [0, 0, 0, 0]  # no FIGHT decision, nothing recorded
    assert int(t2.last_move[0]) == names.move("Leer")
    t3 = _both(tables, t2, A("Leer", "Tackle", t2, names))
    assert _seen(t3) == [0, 1, 0, 0]  # Leer only: Scratch was lost


def test_ai_switch_records_nothing(tables, names):
    s = roark_state(tables, names)
    s = s._replace(last_move=s.last_move.at[0].set(names.move("Scratch")))
    out = _both(tables, s, A("Leer", SWITCH_TO_1, s, names))
    assert np.asarray(out.ai_seen_moves).sum() == 0


@pytest.mark.slow
def test_flinched_or_fully_paralysed_turn_records_nothing(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Headbutt"], opp_faster=True)
    flinch = empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8))
    flinched = _both(tables, s, A("Scratch", "Headbutt", s, names), flinch)
    assert int(flinched.last_move[0]) == 0
    assert _seen(_both(tables, flinched, A("Scratch", "Tackle", flinched, names))) == [0] * 4
    para = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    full_para = empty_outcome()._replace(full_para=jnp.asarray([1, 0], jnp.int8))
    stopped = _both(tables, para, A("Scratch", "Tackle", s, names), full_para)
    assert int(stopped.last_move[0]) == 0
    assert _seen(_both(tables, stopped, A("Scratch", "Tackle", stopped, names))) == [0] * 4


def test_disrupted_turn_overwrites_an_earlier_move(tables, names):
    """UpdateMoveBuffers writes MOVE_NONE for a disrupted move, so last turn's move is
    gone too (battle_controller_player.c:3948)."""
    s = roark_state(tables, names)
    para = s._replace(
        status=s.status.at[0, 0].set(Status.PARALYSIS),
        last_move=s.last_move.at[0].set(names.move("Leer")),
    )
    full_para = empty_outcome()._replace(full_para=jnp.asarray([1, 0], jnp.int8))
    # The AI fights this turn, so the pending Leer is committed first.
    out = _both(tables, para, A("Scratch", "Leer", s, names), full_para)
    assert _seen(out) == [0, 1, 0, 0]
    assert int(out.last_move[0]) == 0


def test_a_miss_is_seen(tables, names):
    s = roark_state(tables, names)
    missed = _both(
        tables,
        s,
        A("Scratch", "Leer", s, names),
        empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8)),
    )
    assert int(missed.last_move[0]) == names.move("Scratch")


@pytest.mark.slow
def test_opponent_moves_are_not_recorded(tables, names):
    s = roark_state(tables, names)
    t1 = _both(tables, s, A("Leer", "Leer", s, names))
    assert int(t1.last_move[1]) == names.move("Leer")  # the opponent's movePrevByBattler
    out = _both(tables, t1, A("Scratch", "Leer", t1, names))
    assert np.asarray(out.ai_seen_moves).sum() == 1  # the player's Leer only
    assert int(out.ai_seen_moves[0, names.move_slot(s, 0, "Leer")]) == 1


def test_forced_struggle_skips_the_record(tables, names):
    """A struggling opponent never reaches the move menu, so ``TrainerAI_Main`` and its
    ``RecordLastMove`` do not run (battle_controller_player.c:433)."""
    s = roark_state(tables, names)
    s = s._replace(pp=s.pp.at[1, 0].set(0), last_move=s.last_move.at[0].set(names.move("Scratch")))
    out = _both(tables, s, A("Leer", 0, s, names))
    assert _seen(out) == [0, 0, 0, 0]


@pytest.mark.slow
def test_player_struggle_is_written_but_not_recorded(tables, names):
    s = roark_state(tables, names)
    s = s._replace(pp=s.pp.at[0, 0].set(0))
    struggled = _both(tables, s, A(0, "Leer", s, names))
    assert int(struggled.last_move[0]) == MOVE_STRUGGLE
    assert _seen(_both(tables, struggled, A(0, "Leer", struggled, names))) == [0] * 4


def test_pursuit_before_a_switch_is_written(tables, names):
    s = roark_state(tables, names, player_moves=["Pursuit"])
    out = _both(tables, s, A("Pursuit", SWITCH_TO_1, s, names))
    assert int(out.last_move[0]) == names.move("Pursuit")


def test_player_switch_in_resets_last_move(tables, names):
    s = roark_state(tables, names)
    s = s._replace(last_move=s.last_move.at[0].set(names.move("Scratch")))
    out = _both(tables, s, A(SWITCH_TO_1, "Leer", s, names))
    # Committed for the Pokemon that was active when the AI decided, then reset.
    assert np.asarray(out.ai_seen_moves[0]).tolist() == [1, 0, 0, 0]
    assert int(out.last_move[0]) == 0


@pytest.mark.slow
def test_scores_include_the_pending_move(tables, names):
    """``TrainerAI_MainSingles`` runs ``RecordLastMove`` before scoring (trainer_ai.c:295),
    so the scores on the AI's FIGHT turn already know the player's last move."""
    base = trace_c(tables, names)  # Cranidos vs Starly
    u_turn = names.move("U-turn")
    knows = base._replace(moves=base.moves.at[0, 0, 3].set(u_turn))
    seen = knows._replace(ai_seen_moves=knows.ai_seen_moves.at[0, 3].set(1))
    pending = knows._replace(last_move=knows.last_move.at[0].set(u_turn))
    outs, _ = enumerate_ai_outcomes(tables, base)
    for i in range(int(outs.tiebreak.shape[0])):
        o = jax.tree.map(lambda x, i=i: x[i], outs)
        expected = np.asarray(ai_scores(tables, seen, o)).tolist()
        assert np.asarray(ai_scores(tables, pending, o)).tolist() == expected


def test_switch_in_clears_the_newcomers_knowledge(tables, names):
    s = roark_state(tables, names)
    s = s._replace(
        ai_seen_moves=s.ai_seen_moves.at[:2].set(1),
        ai_ability_known=s.ai_ability_known.at[:2].set(1),
    )
    out = _both(tables, s, A(SWITCH_TO_1, "Leer", s, names))
    assert np.asarray(out.ai_seen_moves[1]).tolist() == [0, 0, 0, 0]
    assert int(out.ai_ability_known[1]) == 0
    assert np.asarray(out.ai_seen_moves[0]).tolist() == [1, 1, 1, 1]  # the one that left
    assert int(out.ai_ability_known[0]) == 1


@pytest.mark.slow
def test_scores_ignore_an_unseen_move(tables, names):
    """``IfMoveNotKnown DEFENDER, MOVE_U_TURN`` (Expert_Pursuit) reads the seen list:
    a U-turn the player knows but has not used changes no score."""
    base = trace_c(tables, names)  # Cranidos vs Starly
    u_turn = names.move("U-turn")
    knows = base._replace(moves=base.moves.at[0, 0, 3].set(u_turn))
    seen = knows._replace(ai_seen_moves=knows.ai_seen_moves.at[0, 3].set(1))
    outs, _ = enumerate_ai_outcomes(tables, base)
    n = int(outs.tiebreak.shape[0])
    differs = False
    for i in range(n):
        o = jax.tree.map(lambda x, i=i: x[i], outs)
        unseen_scores = np.asarray(ai_scores(tables, knows, o))
        assert unseen_scores.tolist() == np.asarray(ai_scores(tables, base, o)).tolist()
        differs |= unseen_scores.tolist() != np.asarray(ai_scores(tables, seen, o)).tolist()
    assert differs


# --- Ability ----------------------------------------------------------------------


def test_ability_guess_is_chance_until_revealed(tables, names):
    s = roark_state(tables, names, player_species="Shinx", player_ability="Rivalry")
    o0 = sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    assert int(s.ai_ability_known[0]) == 0
    guessed = {
        int(resolve_player_ability(tables, s, o0._replace(ability_guess=jnp.int8(g))))
        for g in (0, 1)
    }
    assert guessed == {names.ability("Rivalry"), names.ability("Intimidate")}
    revealed = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))
    for g in (0, 1):
        o = o0._replace(ability_guess=jnp.int8(g))
        assert int(resolve_player_ability(tables, revealed, o)) == names.ability("Rivalry")


def test_single_ability_species_needs_no_guess(tables, names):
    s = roark_state(tables, names, player_species="Cranidos")  # Mold Breaker only
    s = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(0))
    o0 = sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    for g in (0, 1):
        o = o0._replace(ability_guess=jnp.int8(g))
        assert int(resolve_player_ability(tables, s, o)) == names.ability("Mold Breaker")


def test_intimidate_does_not_reveal_ability(tables, names):
    """subscript_intimidate.s prints through BATTLE_SUBSCRIPT_UPDATE_STAT_STAGE, whose
    "cuts" message ``BtlCmd_ChangeStatStage`` builds in C (battle_script.c:2955-2961)
    with the ability id itself, not through ``BattleMessage_AbilityTag``."""
    s = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert int(s.stages[1, 1]) == -1  # Intimidate did run
    assert int(s.ai_ability_known[0]) == 0


def test_mold_breaker_reveals_on_switch_in(tables, names):
    """subscript_mold_breaker.s: ``PrintMessage ..., TAG_NICKNAME_ABILITY, BTLSCR_MSG_TEMP,
    BTLSCR_MSG_BATTLER_TEMP`` tags the announcing battler's ability."""
    s = roark_state(tables, names, player_species="Cranidos")
    assert int(s.ai_ability_known[0]) == 1
    assert int(s.ai_ability_known[1]) == 0


@pytest.mark.slow
def test_poison_point_reveals_when_it_poisons(tables, names):
    """subscript_poison.s:57 tags ``BTLSCR_MSG_BATTLER_TEMP``'s (the holder's) ability."""
    s = roark_state(tables, names, player_species="Budew", player_ability="Poison Point")
    contact = empty_outcome()._replace(contact_proc=jnp.ones((2, MAX_HITS), jnp.int8))
    acts = A("Growth", "Tackle", s, names)
    poisoned = _both(tables, s, acts, contact)
    assert int(poisoned.status[1, 0]) == Status.POISON
    assert int(poisoned.ai_ability_known[0]) == 1
    not_triggered = _both(tables, s, acts)
    assert int(not_triggered.ai_ability_known[0]) == 0
    # A target that is already poisoned gets no message (subscript_poison.s, _322).
    already = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    assert int(_both(tables, already, acts, contact).ai_ability_known[0]) == 0
