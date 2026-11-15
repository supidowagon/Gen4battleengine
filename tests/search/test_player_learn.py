"""Learn-move prompts in the pluggable player's loop (``nuzline.search.learn``, K3d-exp
design §3.5): ``player_playout`` resolves a turn's learns before the replacements and
settles a pending learn at its root, as ``playout`` does, and both loops'
``refusal_reason`` name a refused learn (window (c)). The states are
``tests/search/test_learn.py``'s.

Compile cost shapes this module (each loop program is the whole engine): the free-policy
runs share one turn count and the fixed-action runs another (a root refused at its settle
plays no turn, so its count is free), the playouts the players are compared with run once
per module, and no test traces more than two loop programs of its own."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import Stat
from nuzline.engine.levelup import needs_learn
from nuzline.engine.state import make_state
from nuzline.search import player as player_module
from nuzline.search import playout as playout_module
from nuzline.search.line import LINE, empty_line
from nuzline.search.player import GREEDY, Player, _player_run, player_playout
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
# count, as player_playout is: called eagerly, its while_loop was traced and compiled anew
# at every call.
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


def _same(got, want, root=""):
    for field in ("won", "player_faints", "turns", "refused", "ended", "hp_left", "pp_left"):
        assert float(getattr(got, field)) == float(getattr(want, field)), f"{root}: {field}"
    assert np.array_equal(np.asarray(got.deaths), np.asarray(want.deaths)), f"{root}: deaths"


TURNS = 8  # the free-policy runs' one turn count (playout, GREEDY, LINE)
FIXED_TURNS = 5  # the fixed-action runs' one turn count (playout_from_action, FIXED)
_ROOTS = (_level_up_turn, _pending_learn_root, _pending_learn_root_pursuit, _window_c_root)
_KEYS = jax.random.split(jax.random.PRNGKey(5), 4)


@pytest.fixture(scope="module")
def roots(tables, names):
    """``[(label, root, key, playout result)]`` over the four roots and four keys: the
    ``playout`` every player run is compared with, computed once for the module."""
    out = []
    for build in _ROOTS:
        s = build(tables, names)
        for k in _KEYS:
            want = jax.device_get(playout(tables, root_state(tables, s, k), k, TURNS))
            out.append((f"{build.__name__}, key {np.asarray(k).tolist()}", s, k, want))
    return out


def test_greedy_equals_playout_through_learns(tables, roots):
    # The GREEDY run's final state carries no learn prompt: the loop resolves each turn's
    # learns (player.py, resolve_learns before send_replacements) and the root's pending one.
    # The result fields alone cannot see a skipped resolve: every move is given up anyway.
    # The pending roots also pin the root's opponent replacement and its carried powerMul
    # (Cranidos at 0, Onix at 20): a run settling them otherwise plays a different battle.
    for root, s, k, want in roots:
        got, out = _run(tables, s, GREEDY, (), k, TURNS)
        _same(got, want, root)
        if not int(got.refused):
            assert not bool(needs_learn(out)), f"{root}: a learn prompt was left pending"


def test_a_pending_learn_root_window_c_is_refused_and_named(tables, names):
    # Refused at the root's settle, before any turn: both loops on the programs of the test
    # above (the turn count is free there: no turn is played).
    s = _window_c_root(tables, names)
    key = jax.random.PRNGKey(7)
    assert int(_run(tables, s, GREEDY, (), key, TURNS)[0].refused) == 1
    assert int(playout(tables, s, key, TURNS).refused) == 1
    # Both refusal_reason replays name it with no turn to replay (max_turns 0), so the name
    # can only come from the root's settle (player._replay's and playout.refusal_reason's
    # settle_reason, as _resolve_pending's settle_learns); with turns, the first replayed
    # turn names the same window. Formerly test_learn.py::
    # test_refusal_reason_settles_a_pending_learn_root. Python-level replays: no new program.
    for turns in (0, TURNS):
        reason = player_module.refusal_reason(tables, s, GREEDY, (), key, turns) or ""
        assert "window (c)" in reason, ("player", turns)
        reason = playout_module.refusal_reason(tables, s, 0, key, turns) or ""
        assert "window (c)" in reason, ("playout", turns)


def test_the_empty_line_equals_playout_through_learns(tables, roots):
    for root, s, k, want in roots:
        _same(player_playout(tables, s, LINE, empty_line(), k, TURNS), want, root)


def test_a_turn_resolves_its_learns(tables, names):
    # Bulbasaur's Tackle KOs Geodude in the first turn: it reaches L13 and its prompts
    # (PoisonPowder, Sleep Powder) are answered by giving both up, and Bonsly comes in. The
    # FIXED program of the next test (a one-turn run was a program of its own, 9.5 GB
    # jitted, 9.8 GB eager); its later turns teach nothing (Bulbasaur's next move is at 15).
    s = _level_up_turn(tables, names)
    tackle = names.move_slot(s, 0, "Tackle")
    levelled = 0
    for k in jax.random.split(jax.random.PRNGKey(2), 6):
        got, out = _run(tables, s, FIXED, tackle, k, FIXED_TURNS)
        if int(got.refused) or int(out.level[0, 0]) < 13:
            continue
        levelled += 1
        assert int(out.learn_level[0]) == 0 and int(out.active[1]) == 1
        np.testing.assert_array_equal(np.asarray(out.moves[0, 0]), np.asarray(s.moves[0, 0]))
    assert levelled, "no key levelled Bulbasaur"


def test_a_turn_window_c_refuses_the_fixed_player_and_is_named(tables, names):
    # The playout's window (c) turn (the test below) in the player loop, on the FIXED
    # program of the test above (its refusal_reason shares playout's _play_jit).
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    assert int(_run(tables, s, FIXED, action, key, FIXED_TURNS)[0].refused) == 1
    reason = player_module.refusal_reason(tables, s, FIXED, action, key, FIXED_TURNS)
    assert "window (c)" in reason


def test_a_turn_window_c_refuses_the_playout_and_is_named(tables, names):
    # Refused in the first turn's payout (test_learn.py::_window_c). Formerly
    # test_learn.py::test_window_c_refuses_the_playout_and_names_it.
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    assert int(playout_from_action(tables, s, action, key, FIXED_TURNS).refused) == 1
    assert "window (c)" in playout_module.refusal_reason(tables, s, action, key, FIXED_TURNS)


# Slow: a no-turn GREEDY program of its own (about 2.3 GB). The fast tier keeps the property
# through test_greedy_equals_playout_through_learns' two pending roots (it fails with the
# root's settle_learns removed from _player_run, and with its carried powerMul replaced by
# the normal one).
@pytest.mark.slow
@pytest.mark.parametrize("mul, newcomer", [(0, CRANIDOS), (20, ONIX)])
def test_a_pending_learn_root_sends_with_the_carried_mul(tables, names, mul, newcomer):
    s = _pending_learn_root(tables, names, mul)
    out = _run(tables, s, GREEDY, (), jax.random.PRNGKey(0), 0)[1]
    assert int(out.active[1]) == newcomer and int(out.learn_level[0]) == 0
