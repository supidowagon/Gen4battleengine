"""The pluggable player (spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md
§7a): ``GREEDY`` is ``playout``, a new player plugs into the loop unchanged, and players
are hashable (jit static arguments)."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.search.line import LINE
from nuzline.search.player import GREEDY, Player, _player_run, player_playout
from nuzline.search.playout import REFUSED_WON, playout, replacement_policy, root_state
from tests.engine.helpers import roark_state
from tests.search.fixtures import at_replacement


@pytest.mark.slow
def test_greedy_equals_playout(tables, names):
    s = roark_state(tables, names)
    keys = jax.random.split(jax.random.PRNGKey(11), 8)
    for k in keys:
        got = player_playout(tables, s, GREEDY, (), k, 60)
        want = playout(tables, root_state(tables, s, k), k, 60)
        assert int(got.won) == int(want.won)
        assert int(got.player_faints) == int(want.player_faints)
        assert int(got.turns) == int(want.turns)
        assert np.array_equal(np.asarray(got.deaths), np.asarray(want.deaths))


def _first_move(tables, state, params, key, layout):
    return jnp.int32(0), jnp.bool_(False), jnp.bool_(False)


def _replace(tables, state, params, key, layout):
    return replacement_policy(tables, state), jnp.bool_(False)


def _refuse_replace(tables, state, params, key, layout):
    return replacement_policy(tables, state), jnp.bool_(True)


@pytest.mark.slow
def test_a_new_player_plugs_in(tables, names):
    s = roark_state(tables, names, player_count=1, opp_count=1)
    player = Player("first-move", _first_move, _replace)
    run = jax.jit(partial(_player_run, player=player, max_turns=3))
    result, final = run(tables, s, params=(), key=jax.random.PRNGKey(2))
    assert int(result.refused) == 0 and int(result.turns) >= 1
    assert int(result.fallback_turns) == 0
    spent = np.asarray(s.pp[0, 0], np.int32) - np.asarray(final.pp[0, 0], np.int32)
    assert spent[0] >= 1 and not spent[1:].any()
    public = player_playout(tables, s, player, (), jax.random.PRNGKey(2), 3)
    assert int(public.fallback_turns) == 0 and int(public.refused) == 0


@pytest.mark.slow
def test_a_refused_replacement_refuses(tables, names):
    s = at_replacement(roark_state(tables, names))
    player = Player("refuses-replacement", _first_move, _refuse_replace)
    result = player_playout(tables, s, player, (), jax.random.PRNGKey(6), 3)
    assert int(result.refused) == 1 and int(result.won) == REFUSED_WON
    assert int(result.turns) == 0


def test_players_are_hashable():
    assert isinstance(hash(GREEDY), int) and isinstance(hash(LINE), int)
    assert GREEDY != LINE
