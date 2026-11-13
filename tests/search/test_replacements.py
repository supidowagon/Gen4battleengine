"""Set-mode player replacements in the playout (plan 4b Task 3): ``send_replacements``
stops at a pending player replacement, ``replacement_pass`` sends the player in first
against the AI's pick on the fainted battler, ``replacement_policy`` chooses the slot,
and ``playout`` resolves a pending state before its first turn. Pending states are built
by hand (``at_replacement``): the terminal rule still ends a battle at a player faint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.ai.replacement import post_ko_switch_in
from nuzline.engine.actions import MOVE_ACTIONS, legal_replacements
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.step import replace
from nuzline.search import playout as playout_module
from nuzline.search.playout import (
    playout,
    playout_from_action,
    replacement_pass,
    replacement_policy,
    send_replacements,
)
from tests.engine.helpers import roark_state
from tests.search.fixtures import at_replacement
from tests.strategies import supported_state

pytestmark = pytest.mark.slow


def assert_same(a, b):
    """Two states (or results) equal field by field."""
    for name in a._fields:
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name


def _both_down(tables, names, **kw):
    return at_replacement(roark_state(tables, names, **kw), opp_down=True)


def _newcomer(tables, s):
    """A legal player slot whose entry changes the AI's pick (the test asserts one exists)."""
    plain = int(post_ko_switch_in(tables, s, 10))
    for slot in np.flatnonzero(np.asarray(legal_replacements(s, 0))):
        if int(post_ko_switch_in(tables, replace(tables, s, 0, int(slot)), 10)) != plain:
            return int(slot), plain
    raise AssertionError("no newcomer changes the AI's pick")


def test_send_replacements_stops_at_a_player_replacement(tables, names):
    s = _both_down(tables, names)
    for mul in (10, 20):
        after, refused = send_replacements(tables, s, False, mul)
        assert not bool(refused)
        assert np.array_equal(after.active, s.active)
        assert np.array_equal(after.hp, s.hp)
        assert int(after.replace_mul) == mul
    # Only the player is down: nothing is held for the AI.
    only_player = at_replacement(roark_state(tables, names))
    after, refused = send_replacements(tables, only_player, False, 20)
    assert not bool(refused) and int(after.replace_mul) == 0
    assert np.array_equal(after.active, only_player.active)


def test_ai_chooses_against_the_fainted_battler(tables, names):
    s = _both_down(tables, names)
    slot, against_fainted = _newcomer(tables, s)
    after, refused = replacement_pass(tables, s, slot)
    assert not bool(refused)
    assert int(after.active[0]) == slot
    assert int(after.active[1]) == against_fainted


def test_player_enters_first(tables, names):
    s = _both_down(tables, names)
    s = s._replace(stealth_rock=jnp.ones_like(s.stealth_rock))
    o = int(post_ko_switch_in(tables, s, 10))
    expected = replace(tables, replace(tables, s, 0, 1), 1, o)
    after, refused = replacement_pass(tables, s, 1)
    assert not bool(refused)
    assert int(after.replace_mul) == 0
    assert_same(after, expected)
    assert int(after.hp[0, 1]) < int(s.hp[0, 1])  # Stealth Rock hurt the newcomer


def _intimidate_at(s, slot):
    return s._replace(ability=s.ability.at[0, slot].set(AbilityId.INTIMIDATE))


def test_player_intimidate_fires_on_the_opposing_newcomer_after_a_double_ko(tables, names):
    """Both actives down: the player's Intimidate newcomer enters against a fainted foe,
    and ShowBattleMon's switch-in check (battle_lib.c:3795) fires it later against the
    opponent's newcomer, which then has -1 Attack."""
    s = _intimidate_at(_both_down(tables, names), 4)  # Shinx in slot 4
    after, refused = replacement_pass(tables, s, 4)
    assert not bool(refused)
    assert int(after.active[0]) == 4 and int(after.hp[1, after.active[1]]) > 0
    assert int(after.stages[1, Stat.ATK]) == -1
    assert int(after.stages[0, Stat.ATK]) == 0


def test_player_intimidate_fires_once_at_a_single_replacement(tables, names):
    """Only the player is down: Intimidate fires on entry against the living foe, once."""
    s = _intimidate_at(at_replacement(roark_state(tables, names)), 4)
    after, refused = replacement_pass(tables, s, 4)
    assert not bool(refused)
    assert int(after.stages[1, Stat.ATK]) == -1


def test_replace_mul_reaches_the_ai(tables, names):
    s = _both_down(tables, names)._replace(replace_mul=jnp.int8(20))
    o = int(post_ko_switch_in(tables, s, 20))
    expected = replace(tables, replace(tables, s, 0, 1), 1, o)
    after, refused = replacement_pass(tables, s, 1)
    assert not bool(refused)
    assert_same(after, expected._replace(replace_mul=jnp.int8(0)))
    assert int(after.active[1]) == o


def test_refused_opponent_replacement_at_a_player_replacement(tables, names):
    s = _both_down(tables, names)
    _, refused = send_replacements(tables, s, True, 10)
    assert bool(refused)


def test_replacement_policy_picks_the_counter(tables, names):
    # Geodude (Rock) against Prinplup (Water, super effective) and Starly (Normal/Flying,
    # resisted): the Water type goes in, whatever the slot order.
    s = roark_state(tables, names, player_count=4, opp_count=1)
    s = s._replace(species=s.species.at[0, 2].set(0)).replace_hp(0, 2, 0)
    s = at_replacement(s)
    assert [bool(x) for x in legal_replacements(s, 0)] == [False, True, False, True, False, False]
    assert int(replacement_policy(tables, s)) == 1


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_replacement_policy_is_legal(tables, names, data):
    s = data.draw(supported_state(tables, names))
    s = s.replace_hp(0, int(s.active[0]), 0)
    assume(bool(np.any(np.asarray(legal_replacements(s, 0)))))
    slot = int(replacement_policy(tables, s))
    assert bool(legal_replacements(s, 0)[slot])


@pytest.mark.slow
def test_playout_from_a_pending_state(tables, names):
    s = at_replacement(roark_state(tables, names))
    key = jax.random.PRNGKey(3)
    chosen = int(replacement_policy(tables, s))
    resolved = replacement_pass(tables, s, chosen)[0]
    assert int(resolved.active[0]) == chosen
    assert_same(playout(tables, s, key), playout(tables, resolved, key))
    forced = playout_from_action(tables, s, MOVE_ACTIONS + 3, key)
    assert_same(forced, playout(tables, replacement_pass(tables, s, 3)[0], key))


def test_pending_state_with_both_down_resolves_in_a_playout(tables, names):
    s = _both_down(tables, names)
    r = playout(tables, s, jax.random.PRNGKey(4))
    assert int(r.refused) == 0 and int(r.ended) == 1
    assert int(r.deaths[0]) == 1  # the record the pending state carried


@pytest.mark.slow
def test_result_records_deaths_and_end(tables, names):
    s = roark_state(tables, names)
    key = jax.random.PRNGKey(1)
    r = playout(tables, s, key)
    state = s
    for _ in range(int(r.turns)):
        key, k_policy, k_turn = jax.random.split(key, 3)
        action = playout_module._policy_jit(tables, state, k_policy)[0]
        state, refused = playout_module._turn_jit(tables, state, action, k_turn)
        assert not bool(refused)
    assert np.array_equal(r.deaths, state.deaths)
    assert int(r.ended) == 1 and int(state.over) == 1
    cut = playout(tables, s, key, max_turns=1)
    assert int(cut.turns) == 1 and int(cut.ended) == 0 and int(cut.refused) == 0


def test_replace_mul_is_the_ai_power_mul(tables, names, monkeypatch):
    # On this data the pick is the same at powerMul 10 and 20, so a stand-in whose pick
    # depends on the value shows what the pass hands the AI.
    def by_mul(_tables, _state, power_mul=10):
        return jnp.where(power_mul == 20, 2, 1).astype(jnp.int32)

    monkeypatch.setattr(playout_module, "post_ko_switch_in", by_mul)
    fresh = jax.jit(lambda t, st, slot: replacement_pass.__wrapped__(t, st, slot))  # new trace
    s = _both_down(tables, names)
    assert int(fresh(tables, s, 1)[0].active[1]) == 1
    assert int(fresh(tables, s._replace(replace_mul=jnp.int8(20)), 1)[0].active[1]) == 2


def test_refusal_reason_replays_a_pending_state(tables, names):
    s = _both_down(tables, names)
    key = jax.random.PRNGKey(5)
    assert not int(playout(tables, s, key).refused)
    assert playout_module.refusal_reason(tables, s, MOVE_ACTIONS + 1, key) is None
    assert playout_module.replacement_refusal(tables, s, True, 10) is not None
    assert playout_module.replacement_refusal(tables, s, False, 10) is None
