"""Stage A playouts: the player's one-ply policy, the opponent's AI, the turn loop and
the opponent's faint replacement (docs/design.md §4.3)."""

import jax
import jax.numpy as jnp

from nuzline.ai.outcome import sample_ai_outcome
from nuzline.ai.policy import ai_policy
from nuzline.ai.replacement import post_ko_switch_in
from nuzline.data.party import load_party
from nuzline.engine.actions import MOVE_ACTIONS, NUM_OPP_ACTIONS, legal_actions
from nuzline.engine.ids import Status
from nuzline.engine.state import SLOTS, make_state
from nuzline.search import playout as playout_module
from nuzline.search.playout import (
    REFUSED_WON,
    batched_playout,
    playout,
    playout_from_action,
    playout_policy,
    refusal_reason,
    replacement_refusal,
    send_replacements,
)
from tests.engine.helpers import SAMPLE_PARTY, _inter, _lead, _roark, _roark_party, roark_state


def test_policy_returns_legal_action(tables, names):
    s = roark_state(tables, names)
    a = int(playout_policy(tables, s, jax.random.PRNGKey(0)))
    assert bool(legal_actions(tables, s, 0)[a])


def test_playout_terminates_with_result(tables, names):
    s = roark_state(tables, names)
    r = playout(tables, s, jax.random.PRNGKey(1))
    assert int(r.won) in (0, 1) and 0 < int(r.turns) <= 100


def test_batched_playout_shapes(tables, names):
    s = roark_state(tables, names)
    B = 64
    bs = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s)
    r = batched_playout(tables, bs, jax.random.split(jax.random.PRNGKey(2), B))
    assert r.won.shape == (B,) and r.turns.dtype == jnp.int16


def _replay(tables, state, key, max_turns=100):
    """The states of ``playout(tables, state, key)`` turn by turn (its key splits) and
    the opponent's action each turn: ``[(state before, opponent action, state after)]``."""
    out = []
    for _ in range(max_turns):
        if int(state.over) != 0:
            break
        key, k_policy, k_turn = jax.random.split(key, 3)
        action = playout_module._policy_jit(tables, state, k_policy)[0]
        played, refused = playout_module._turn_jit(tables, state, action, k_turn)
        assert not bool(refused)
        opp = int(
            ai_policy(tables, state, sample_ai_outcome(jax.random.split(k_turn)[0], tables, state))
        )
        out.append((state, opp, played))
        state = played
    return out


def _short_party_trace_f(tables, names):
    """roark.md 13, trace F (Cranidos 40/43 in party slot 2, moveHit = Wing Attack, turn
    2; its AI switches with p = 1 - (43690/65536)^4) against a two-Pokemon player party
    led by Starly: both parties are short (slots 2-5 and 3-5 empty)."""
    player = _lead(load_party(SAMPLE_PARTY, names), names.species("Starly"), _inter())[:2]
    s = make_state(tables, player, list(_roark_party()), opp_items=_roark().items)
    return s._replace(
        active=s.active.at[1].set(2),
        hp=s.hp.at[1, 2].set(40),
        turn=jnp.asarray(1, s.turn.dtype),
        move_hit=s.move_hit.at[1].set(names.move("Wing Attack")),
    )


def _replay(tables, state, key, max_turns=100):
    """``playout(tables, state, key)`` turn by turn (the same key splits):
    ``[(state before, the opponent's action, state after)]``."""
    out = []
    for _ in range(max_turns):
        if int(state.over) != 0:
            break
        key, k_policy, k_turn = jax.random.split(key, 3)
        action = playout_module._policy_jit(tables, state, k_policy)[0]
        played, refused = playout_module._turn_jit(tables, state, action, k_turn)
        assert not bool(refused)
        k_ai = jax.random.split(k_turn)[0]  # as _play splits it
        opp = int(ai_policy(tables, state, sample_ai_outcome(k_ai, tables, state)))
        out.append((state, opp, played))
        state = played
    return out


def test_short_party_and_opponent_switch_do_not_index_empty_slot(tables, names):
    # JAX clamps an out-of-range index silently, so a switch or replacement into an
    # empty slot (species 0) or a fainted one would not raise: check every turn.
    s = _short_party_trace_f(tables, names)
    counts = (2, 3)
    switches = replacements = 0
    for seed in range(4):
        for before, opp_action, after in _replay(tables, s, jax.random.PRNGKey(seed)):
            if MOVE_ACTIONS <= opp_action < MOVE_ACTIONS + SLOTS:
                switches += 1
                target = opp_action - MOVE_ACTIONS
                assert target < counts[1] and int(before.species[1, target]) > 0
                assert int(before.hp[1, target]) > 0
            elif int(after.active[1]) != int(before.active[1]):
                replacements += 1
            for side in (0, 1):
                slot = int(after.active[side])
                assert 0 <= slot < counts[side] and int(after.species[side, slot]) > 0
                if not int(after.over):
                    assert int(after.hp[side, slot]) > 0
    assert switches > 0 and replacements > 0


def test_player_faint_ends_playout_as_loss(tables, names):
    # No switch (one Pokemon); with no PP it must Struggle at 1 HP: the recoil faints it.
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1).with_pp(0, 0, [0, 0, 0, 0])
    r = playout(tables, s, jax.random.PRNGKey(4))
    assert int(r.won) == 0 and int(r.player_faints) == 1 and int(r.turns) == 1


def test_policy_prefers_a_damaging_move_to_a_status_move(tables, names):
    # Monferno's Scratch, Leer, Ember, Mach Punch against Geodude: Mach Punch (Fighting,
    # super effective on Rock) removes the most HP, Leer none.
    s = roark_state(tables, names, player_count=1)
    a = int(playout_policy(tables, s, jax.random.PRNGKey(5)))
    assert a == names.move_slot(s, 0, "Mach Punch")


def test_illegal_actions_are_never_chosen(tables, names):
    # Only Ember has PP and switching is impossible (one Pokemon): Ember is the only legal action.
    s = roark_state(tables, names, player_count=1).with_pp(0, 0, [0, 0, 5, 0])
    for seed in range(4):
        assert int(playout_policy(tables, s, jax.random.PRNGKey(seed))) == 2


def test_playout_from_action_plays_that_action_first(tables, names):
    # The lead at 1 HP is slower than Geodude, whose only move with PP is Tackle (95 %
    # accurate; +6 accuracy makes it sure) and which has no party to switch to: Leer
    # loses on turn 1, a switch to slot 1 survives it.
    s = (
        roark_state(tables, names, opp_count=1, opp_faster=True)
        .replace_hp(0, 0, 1)
        .with_pp(1, 0, [35, 0, 0, 0])
        .with_stages(1, acc=6)
    )
    leer = names.move_slot(s, 0, "Leer")
    switch = MOVE_ACTIONS + 1
    for seed in range(3):
        key = jax.random.PRNGKey(seed)
        lost = playout_from_action(tables, s, jnp.int32(leer), key, 100)
        assert (int(lost.won), int(lost.player_faints), int(lost.turns)) == (0, 1, 1)
        kept = playout_from_action(tables, s, jnp.int32(switch), key, 100)
        assert int(kept.turns) > 1 and int(kept.refused) == 0


def test_first_actions_share_the_draws_after_turn_1(tables, names):
    # Common random numbers: asleep (5 turns), Scratch and Leer both leave the same
    # state after turn 1 (neither runs), so one key gives both the same playout.
    s = roark_state(tables, names, opp_count=1).with_status(0, 0, Status.SLEEP, 5)
    scratch, leer = names.move_slot(s, 0, "Scratch"), names.move_slot(s, 0, "Leer")
    results = []
    for seed in range(4):
        key = jax.random.PRNGKey(seed)
        a = playout_from_action(tables, s, jnp.int32(scratch), key, 100)
        b = playout_from_action(tables, s, jnp.int32(leer), key, 100)
        assert [int(x) for x in a] == [int(x) for x in b]
        results.append(tuple(int(x) for x in a))
    assert len(set(results)) > 1  # the draws do change with the key


def test_playout_is_deterministic_in_its_key(tables, names):
    s = roark_state(tables, names)
    a = playout(tables, s, jax.random.PRNGKey(8))
    b = playout(tables, s, jax.random.PRNGKey(8))
    assert [int(x) for x in a] == [int(x) for x in b]


def test_turn_cap_stops_the_playout(tables, names):
    s = roark_state(tables, names)
    r = playout(tables, s, jax.random.PRNGKey(9), max_turns=1)
    assert int(r.turns) == 1


def test_opponent_replacement_follows_post_ko_switch_in(tables, names):
    # Geodude has fainted (the battle goes on): the opponent sends in the Pokemon
    # BattleAI_PostKOSwitchIn picks, not "the first legal slot".
    s = roark_state(tables, names, player_species="Starly").replace_hp(1, 0, 0)
    pick = int(post_ko_switch_in(tables, s))
    out, refused = send_replacements(tables, s, jnp.bool_(False), 10)
    assert int(refused) == 0 and int(out.active[1]) == pick


def test_replacement_after_an_ai_switch_is_refused(tables, names):
    # roark.md section 14 (3): after the AI switched this turn, Task_TrainerShowPartyMenu
    # reads the stale aiSwitchedPartySlot; the engine refuses that path.
    s = roark_state(tables, names).replace_hp(1, 0, 0)
    _, refused = send_replacements(tables, s, jnp.bool_(True), 10)
    assert int(refused) == 1


def test_no_replacement_needed_changes_nothing(tables, names):
    s = roark_state(tables, names)
    out, refused = send_replacements(tables, s, jnp.bool_(True), 10)
    assert int(refused) == 0 and int(out.active[1]) == 0


def test_refused_playout_cannot_be_counted_as_a_loss(tables, names):
    # Levitate on the player's lead is refused by ai_refusals: traced, the first turn
    # stops the playout, and won = -1 so it cannot be summed as a loss.
    s = roark_state(tables, names, player_species="Gastly", player_ability="Levitate")
    r = playout(tables, s, jax.random.PRNGKey(10))
    assert (int(r.won), int(r.refused), int(r.turns)) == (REFUSED_WON, 1, 0)


def test_refusal_reason_names_the_refused_mechanic(tables, names):
    # The traced playout only says "refused"; the eager replay of the same key names
    # the mechanic (here the AI's Levitate refusal).
    s = roark_state(tables, names, player_species="Gastly", player_ability="Levitate")
    reason = refusal_reason(tables, s, 0, jax.random.PRNGKey(10))
    assert reason is not None and "Levitate" in reason


def test_refusal_reason_is_none_for_a_playout_that_is_not_refused(tables, names):
    s = roark_state(tables, names)
    assert int(playout_from_action(tables, s, 0, jax.random.PRNGKey(11)).refused) == 0
    assert refusal_reason(tables, s, 0, jax.random.PRNGKey(11)) is None


def test_replacement_refusal_names_the_stale_switch_slot(tables, names):
    s = roark_state(tables, names).replace_hp(1, 0, 0)
    assert "aiSwitchedPartySlot" in replacement_refusal(tables, s, True, 10)
    assert replacement_refusal(tables, s, False, 10) is None


def test_a_refused_distribution_refuses_the_playout_before_the_player_chooses(
    tables, names, monkeypatch
):
    # Traced, a refused AI distribution is NaN. The player's choice depends on it, so the
    # playout must stop refused there rather than choose against an all-zero
    # distribution (a uniformly random pick). ``_run`` is called unjitted so that the
    # patched distribution is traced in.
    s = roark_state(tables, names)
    nan = jnp.full(NUM_OPP_ACTIONS, jnp.nan, jnp.float32)
    monkeypatch.setattr(playout_module, "ai_action_distribution", lambda tables, state: nan)
    key = jax.random.PRNGKey(12)
    r = playout_module._run(tables, s, key, 100, jnp.int32(-1))
    assert (int(r.won), int(r.refused), int(r.turns)) == (REFUSED_WON, 1, 0)
    # A fixed first action does not depend on it: turn 1 is played, turn 2 is refused.
    r = playout_module._run(tables, s, key, 100, jnp.int32(0))
    assert (int(r.won), int(r.refused), int(r.turns)) == (REFUSED_WON, 1, 1)
