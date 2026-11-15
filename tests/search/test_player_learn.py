"""Learn-move prompts in the pluggable player's loop (``nuzline.search.learn``, K3d-exp
design §3.5): ``player_playout`` resolves a turn's learns before the replacements and
settles a pending learn at its root, as ``playout`` does, and ``refusal_reason`` names a
refused learn (window (c)). The states are ``tests/search/test_learn.py``'s."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import Stat
from nuzline.engine.state import make_state
from nuzline.search.line import LINE, empty_line
from nuzline.search.player import GREEDY, Player, _player_run, player_playout, refusal_reason
from nuzline.search.playout import playout, playout_from_action, replacement_policy, root_state
from tests.engine.helpers import roark_real_state
from tests.engine.test_levelup import _base, _mon, _with
from tests.search.test_learn import CRANIDOS, GEODUDE, ONIX, _window_c, _with_prompt


def _fixed_act(tables, state, params, key, layout):
    return jnp.asarray(params, jnp.int32), jnp.bool_(False), jnp.bool_(False)


def _policy_replace(tables, state, params, key, layout):
    return replacement_policy(tables, state), jnp.bool_(False)


FIXED = Player("fixed", _fixed_act, _policy_replace)

# _player_run (the final state as well as the result) compiled once per player and turn
# count, as player_playout is, for the pending-root runs (no turn): called eagerly, its
# while_loop was traced and compiled anew at every call.
_run = jax.jit(_player_run, static_argnames=("player", "max_turns", "layout"))


def _level_up_turn(tables, names):
    # test_learn.py::test_finish_turn_gives_up_before_the_opponent_replacement: Bulbasaur (one
    # short of L13, moves full) KOs Geodude (1 HP) with Tackle; L13 prompts PoisonPowder and
    # Sleep Powder, both given up, then the AI sends Bonsly in.
    bulbasaur = _mon(names, "Bulbasaur", 12, moves=("Tackle", "Growl", "Leech Seed", "Vine Whip"))
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = make_state(tables, [bulbasaur], opp).replace_hp(1, 0, 1)
    return _with(s, exp={0: _base(tables, names, "Bulbasaur", 13) - 1})


def _pending_learn_root(tables, names, mul=0):
    # test_learn.py::test_a_pending_learn_settles_the_opponent_replacement_with_the_carried_mul:
    # at Geodude Atk 400 the AI sends Cranidos at powerMul 10 and Onix at the carried 20.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    s = s._replace(stats=s.stats.at[1, GEODUDE, Stat.ATK - 1].set(jnp.int16(400)))
    return _with_prompt(tables, names, s)[0]._replace(replace_mul=jnp.int8(mul))


def _pending_learn_root_pursuit(tables, names):
    return _pending_learn_root(tables, names, 20)


def _window_c_root(tables, names):
    # test_learn.py::test_refusal_reason_settles_a_pending_learn_root
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    s = s.replace_hp(1, ONIX, 1).replace_hp(1, CRANIDOS, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    level = int(s.level[0, 0])
    s = _with(s, exp={0: _base(tables, names, "Machop", level + 1) - 1})
    return _with_prompt(tables, names, s)[0]


def _same(got, want):
    for field in ("won", "player_faints", "turns", "refused", "ended", "hp_left", "pp_left"):
        assert float(getattr(got, field)) == float(getattr(want, field)), field
    assert np.array_equal(np.asarray(got.deaths), np.asarray(want.deaths))


def test_greedy_and_the_empty_line_equal_playout_through_learns(tables, names):
    # One test over the four roots: each case traces the same three programs (playout and the
    # GREEDY and LINE player_playout), and as separate tests each grew the process past the
    # conftest cache-clear threshold, so every case traced them again.
    builds = (_level_up_turn, _pending_learn_root, _pending_learn_root_pursuit, _window_c_root)
    for build in builds:
        s = build(tables, names)
        for k in jax.random.split(jax.random.PRNGKey(5), 4):
            want = playout(tables, root_state(tables, s, k), k, 8)
            _same(player_playout(tables, s, GREEDY, (), k, 8), want)
            _same(player_playout(tables, s, LINE, empty_line(), k, 8), want)


# Slow: the one-turn FIXED player loop is a program of its own that grows the process by
# about 9.6 GB from a cold cache. The fast tier keeps the property through
# test_greedy_and_the_empty_line_equal_playout_through_learns, whose _level_up_turn root
# requires player_playout to equal playout through this same turn, learns resolved.
@pytest.mark.slow
def test_a_turn_resolves_its_learns(tables, names):
    # The final state of a one-turn run: Bulbasaur reached L13 and its prompts were answered
    # (learn_level 0, the moves kept) before Bonsly came in.
    s = _level_up_turn(tables, names)
    tackle = names.move_slot(s, 0, "Tackle")
    levelled = 0
    for k in jax.random.split(jax.random.PRNGKey(2), 6):
        # eager: jitted whole, this one-turn FIXED run grew the process by 9.5 GB
        out = _player_run(tables, s, FIXED, tackle, k, 1)[1]
        if int(out.level[0, 0]) != 13:
            continue
        levelled += 1
        assert int(out.learn_level[0]) == 0 and int(out.active[1]) == 1
        np.testing.assert_array_equal(np.asarray(out.moves[0, 0]), np.asarray(s.moves[0, 0]))
    assert levelled, "no key levelled Bulbasaur"


@pytest.mark.parametrize("mul, newcomer", [(0, CRANIDOS), (20, ONIX)])
def test_a_pending_learn_root_sends_with_the_carried_mul(tables, names, mul, newcomer):
    s = _pending_learn_root(tables, names, mul)
    out = _run(tables, s, GREEDY, (), jax.random.PRNGKey(0), 0)[1]
    assert int(out.active[1]) == newcomer and int(out.learn_level[0]) == 0


def test_a_pending_learn_root_window_c_is_refused_and_named(tables, names):
    s = _window_c_root(tables, names)
    key = jax.random.PRNGKey(7)
    assert int(player_playout(tables, s, GREEDY, (), key, 0).refused) == 1
    assert "window (c)" in refusal_reason(tables, s, GREEDY, (), key, 0)


def test_a_turn_window_c_is_refused_and_named(tables, names):
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    want = playout_from_action(tables, root_state(tables, s, key), action, key, 5)
    got = player_playout(tables, s, FIXED, action, key, 5)
    assert int(want.refused) == 1 and int(got.refused) == 1
    assert "window (c)" in refusal_reason(tables, s, FIXED, action, key, 5)
