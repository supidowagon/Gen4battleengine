"""What the AI knows about the player (docs/ai/roark.md section 9), kept in
``BattleState.ai_seen_moves`` / ``ai_ability_known`` by both engines.

* Seen moves: ``movePrevByBattler`` (written in ``UpdateMoveBuffers``,
  battle_controller_player.c:3945-3949) holds the player's move once it got past the
  status checks, the PP decrement and the no-target check (``SYSCTL_MOVE_SUCCEEDED``,
  :3212; a miss still counts); ``BtlCmd_TryPursuit`` (battle_script.c:6930) also writes
  it for a pre-switch Pursuit. ``TrainerAI_RecordLastMove`` (trainer_ai.c:2705) copies it
  into the AI's list.
* Ability: ``BattleAI_SetAbility`` (battle_script.c:12106), from a script message that
  tags the battler's ability (``BattleMessage_AbilityTag``, :11798). Messages the C code
  builds in ``msgBuffer`` (Intimidate's "cuts", Keen Eye's "prevents") reveal nothing.
* Both are cleared when the player's battler switches in (``BattleSystem_UpdateAfterSwitch``,
  battle_lib.c:2151-2153).
"""

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai.outcome import enumerate_ai_outcomes, sample_ai_outcome
from nuzline.ai.policy import ai_scores, resolve_player_ability
from nuzline.data.tables import to_numpy
from nuzline.engine.ids import Status
from nuzline.engine.outcome import empty_outcome, outcome_to_py
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


def test_seen_move_recorded_only_when_executed(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Headbutt"], opp_faster=True)
    flinch = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))
    flinched = _both(tables, s, A("Scratch", "Headbutt", s, names), flinch)
    assert int(flinched.ai_seen_moves[0, 0]) == 0
    executed = _both(tables, s, A("Scratch", "Tackle", s, names))
    assert int(executed.ai_seen_moves[0, 0]) == 1
    assert np.asarray(executed.ai_seen_moves).sum() == 1  # only Scratch's slot


def test_a_miss_is_seen_a_full_paralysis_is_not(tables, names):
    s = roark_state(tables, names)
    missed = _both(
        tables,
        s,
        A("Scratch", "Leer", s, names),
        empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8)),
    )
    assert int(missed.ai_seen_moves[0, 0]) == 1
    para = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    full_para = empty_outcome()._replace(full_para=jnp.asarray([1, 0], jnp.int8))
    stopped = _both(tables, para, A("Scratch", "Leer", s, names), full_para)
    assert int(stopped.ai_seen_moves[0, 0]) == 0


def test_opponent_moves_are_not_recorded(tables, names):
    s = roark_state(tables, names)
    out = _both(tables, s, A("Leer", "Leer", s, names))
    assert np.asarray(out.ai_seen_moves).sum() == 1  # the player's Leer only
    assert int(out.ai_seen_moves[0, names.move_slot(s, 0, "Leer")]) == 1


def test_pursuit_before_a_switch_is_seen(tables, names):
    s = roark_state(tables, names, player_moves=["Pursuit"])
    out = _both(tables, s, A("Pursuit", SWITCH_TO_1, s, names))
    assert int(out.ai_seen_moves[0, 0]) == 1


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


def test_poison_point_reveals_when_it_poisons(tables, names):
    """subscript_poison.s:57 tags ``BTLSCR_MSG_BATTLER_TEMP``'s (the holder's) ability."""
    s = roark_state(tables, names, player_species="Budew", player_ability="Poison Point")
    contact = empty_outcome()._replace(contact_proc=jnp.ones((2, 5), jnp.int8))
    acts = A("Growth", "Tackle", s, names)
    poisoned = _both(tables, s, acts, contact)
    assert int(poisoned.status[1, 0]) == Status.POISON
    assert int(poisoned.ai_ability_known[0]) == 1
    not_triggered = _both(tables, s, acts)
    assert int(not_triggered.ai_ability_known[0]) == 0
    # A target that is already poisoned gets no message (subscript_poison.s, _322).
    already = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    assert int(_both(tables, already, acts, contact).ai_ability_known[0]) == 0
