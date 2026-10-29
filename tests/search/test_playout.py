"""Stage A playouts: the player's one-ply policy, the opponent's AI, the turn loop and
the opponent's faint replacement (docs/design.md §4.3)."""

import jax
import jax.numpy as jnp

from nuzline.ai.replacement import post_ko_switch_in
from nuzline.engine.actions import legal_actions
from nuzline.search.playout import (
    batched_playout,
    playout,
    playout_from_action,
    playout_policy,
    send_replacements,
)
from tests.engine.helpers import roark_state


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


def test_short_party_and_opponent_switch_do_not_index_empty_slot(tables, names):
    s = roark_state(tables, names, player_count=2, opp_count=3)
    r = playout(tables, s, jax.random.PRNGKey(3))
    assert int(r.turns) > 0


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
    s = roark_state(tables, names)
    leer = names.move_slot(s, 0, "Leer")
    r = playout_from_action(tables, s, jnp.int32(leer), jax.random.PRNGKey(6), 100)
    assert 0 < int(r.turns) <= 100 and int(r.refused) == 0


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
