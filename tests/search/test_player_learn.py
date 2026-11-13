"""Learn-move prompts in the pluggable player's loop (``nuzline.search.learn``, K3d-exp
design §3.5): ``player_playout`` resolves a turn's learns before the replacements and
settles a pending learn at its root, as ``playout`` does, and ``refusal_reason`` names a
refused learn (window (c)). The states are ``tests/search/test_learn.py``'s."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import Status
from nuzline.engine.state import make_state
from nuzline.search.line import LINE, empty_line
from nuzline.search.player import GREEDY, Player, player_playout, refusal_reason
from nuzline.search.playout import playout, playout_from_action, replacement_policy, root_state
from tests.engine.helpers import roark_real_state
from tests.engine.test_levelup import _base, _mask, _with
from tests.search.test_learn import GEODUDE, ONIX, CRANIDOS, _mon, _window_c, _with_prompt


def _fixed_act(tables, state, params, key, layout):
    return jnp.asarray(params, jnp.int32), jnp.bool_(False), jnp.bool_(False)


def _policy_replace(tables, state, params, key, layout):
    return replacement_policy(tables, state), jnp.bool_(False)


FIXED = Player("fixed", _fixed_act, _policy_replace)


def _learn_then_replace(tables, names):
    # test_learn.py::test_the_player_replacement_reads_the_moves_after_the_learn: Machop KOs
    # Magikarp and faints to poison; the bench Magikarp in slot 2 learns Tackle on the way.
    machop = _mon(names, "Machop", 20, current_hp=1, status=int(Status.POISON))
    karp = _mon(names, "Magikarp", 14, moves=("Splash",))
    opp = [_mon(names, "Magikarp", 5, moves=("Splash",))] * 2
    s = make_state(tables, [machop, karp, karp], opp).replace_hp(1, 0, 1)
    return _mask(_with(s, exp={2: _base(tables, names, "Magikarp", 15) - 1}), 0, 2)


def _pending_learn_root(tables, names):
    # test_learn.py::test_a_pending_learn_settles_the_opponent_replacement_with_the_carried_mul
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    return _with_prompt(tables, names, s)[0]


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


@pytest.mark.parametrize("build", [_learn_then_replace, _pending_learn_root, _window_c_root])
def test_greedy_and_the_empty_line_equal_playout_through_learns(tables, names, build):
    s = build(tables, names)
    for k in jax.random.split(jax.random.PRNGKey(5), 4):
        want = playout(tables, root_state(tables, s, k), k, 8)
        _same(player_playout(tables, s, GREEDY, (), k, 8), want)
        _same(player_playout(tables, s, LINE, empty_line(), k, 8), want)


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
