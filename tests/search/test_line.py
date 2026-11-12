"""The line player (spec docs/superpowers/specs/2026-11-12-l1-line-player-design.md §3):
one test per step of the turn rule and the replacement rule, the fallback's action mask,
and the line playout loop."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.replacement import post_ko_switch_in
from nuzline.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, NUM_OPP_ACTIONS
from nuzline.engine.ids import Status
from nuzline.engine.state import SLOTS
from nuzline.search import playout as playout_module
from nuzline.search.line import (
    MODE_FALLBACK,
    MODE_MOVES_ONLY,
    MODE_RULE,
    Line,
    _line_run,
    empty_line,
    line_action,
    line_playout,
    line_replacement,
)
from nuzline.search.playout import (
    REFUSED_WON,
    _carried_mul,
    _policy,
    playout,
    replacement_policy,
    root_state,
)
from tests.engine.helpers import roark_state
from tests.search.fixtures import at_replacement


def _line(foe: int = 0, **fields: int) -> Line:
    """``empty_line()`` with ``foe``'s rule set to ``fields``."""
    line = empty_line()
    return line._replace(**{k: _set(getattr(line, k), foe, v) for k, v in fields.items()})


def _set(arr, i, v):
    out = np.array(arr, copy=True)
    out[i] = v
    return out


def _act(tables, state, line):
    action, mode = line_action(tables, state, line)
    return int(action), int(mode)


_MOVES_ONLY = jnp.arange(NUM_ACTIONS) < MOVE_ACTIONS
_RUN_1 = jax.jit(partial(_line_run, max_turns=1))  # one compile for the one-turn tests


def _active(state, slot):
    return state._replace(active=state.active.at[0].set(slot))


def test_no_rule_is_fallback(tables, names):
    s = roark_state(tables, names)
    assert _act(tables, s, empty_line())[1] == MODE_FALLBACK


def test_answer_active_plays_move(tables, names):
    s = roark_state(tables, names)
    assert _act(tables, s, _line(use=0, move=2)) == (2, MODE_RULE)


def test_first_until_status(tables, names):
    s = roark_state(tables, names)
    line = _line(use=0, move=2, first=1)
    assert int(s.status[1, 0]) == 0
    assert _act(tables, s, line) == (1, MODE_RULE)
    assert _act(tables, s.with_status(1, 0, Status.POISON), line) == (2, MODE_RULE)


def test_first_without_pp_plays_move(tables, names):
    s = roark_state(tables, names)
    pp = [int(x) for x in s.pp[0, 0]]
    pp[1] = 0
    assert _act(tables, s.with_pp(0, 0, pp), _line(use=0, move=2, first=1)) == (2, MODE_RULE)


def test_answer_on_bench_switches_in(tables, names):
    s = roark_state(tables, names)
    assert _act(tables, s, _line(use=1, move=0)) == (MOVE_ACTIONS + 1, MODE_RULE)


def test_via_then_answer(tables, names):
    s = roark_state(tables, names)
    line = _line(use=1, move=0, via=2)
    assert _act(tables, s, line) == (MOVE_ACTIONS + 2, MODE_RULE)
    assert _act(tables, _active(s, 2), line) == (MOVE_ACTIONS + 1, MODE_RULE)


def test_via_from_narrows(tables, names):
    s = roark_state(tables, names)
    line = _line(use=1, move=0, via=2, via_from=3)
    assert _act(tables, s, line) == (MOVE_ACTIONS + 1, MODE_RULE)
    assert _act(tables, _active(s, 3), line) == (MOVE_ACTIONS + 2, MODE_RULE)


def test_fainted_via_is_skipped(tables, names):
    s = roark_state(tables, names).replace_hp(0, 2, 0)
    assert _act(tables, s, _line(use=1, move=0, via=2)) == (MOVE_ACTIONS + 1, MODE_RULE)


def test_fainted_answer_is_fallback(tables, names):
    s = roark_state(tables, names).replace_hp(0, 1, 0)
    assert _act(tables, s, _line(use=1, move=0))[1] == MODE_FALLBACK


def test_wait_keeps_the_active_in(tables, names):
    s = roark_state(tables, names)
    assert _act(tables, s, _line(use=1, move=0, wait=1))[1] == MODE_MOVES_ONLY


def test_illegal_rule_action_is_fallback(tables, names):
    s = roark_state(tables, names)
    trapped = s._replace(trapped=s.trapped.at[0].set(2))
    assert _act(tables, trapped, _line(use=1, move=0))[1] == MODE_FALLBACK
    pp = [int(x) for x in s.pp[0, 0]]
    pp[2] = 0
    assert _act(tables, s.with_pp(0, 0, pp), _line(use=0, move=2))[1] == MODE_FALLBACK


def test_replacement_sends_the_answer(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 0)
    assert int(line_replacement(tables, s, _line(use=2, move=0))) == 2
    gone = s.replace_hp(0, 2, 0)
    want = int(replacement_policy(tables, gone))
    assert int(line_replacement(tables, gone, _line(use=2, move=0))) == want


def test_replacement_answers_the_incoming_foe(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 0).replace_hp(1, 0, 0)
    incoming = int(post_ko_switch_in(tables, s, _carried_mul(s)))
    assert incoming in (1, 2)
    line = _line(0, use=2, move=0)
    line = line._replace(use=_set(line.use, incoming, 3), move=_set(line.move, incoming, 0))
    got = int(line_replacement(tables, s, line))
    assert got == 3 and got != int(line.use[0])


def test_empty_line_fields_are_separate_arrays():
    line = empty_line()
    line.use[0] = 1
    assert int(line.via[0]) == -1 and int(line.move[0]) == -1


def _switching(tables, names):
    """Monferno at 1 HP against Geodude: the unmasked fallback switches out."""
    return roark_state(tables, names).replace_hp(0, 0, 1)


def test_choose_mask_default_unchanged(tables, names):
    key = jax.random.PRNGKey(5)
    base = roark_state(tables, names)
    states = [base, _active(base, 2), base.replace_hp(1, 0, 5), _switching(tables, names)]
    every = jnp.ones(NUM_ACTIONS, bool)
    for s in states:
        a, r = _policy(tables, s, key)
        b, r2 = _policy(tables, s, key, allowed=every)
        assert int(a) == int(b) and bool(r) == bool(r2)


def test_choose_mask_moves_only(tables, names):
    key = jax.random.PRNGKey(5)
    s = _switching(tables, names)
    assert int(_policy(tables, s, key)[0]) >= MOVE_ACTIONS
    assert int(_policy(tables, s, key, allowed=_MOVES_ONLY)[0]) < MOVE_ACTIONS


def test_choose_mask_ignored_when_nothing_is_left(tables, names):
    key = jax.random.PRNGKey(5)
    s = _switching(tables, names)
    want = int(_policy(tables, s, key)[0])
    assert int(_policy(tables, s, key, allowed=jnp.zeros(NUM_ACTIONS, bool))[0]) == want
    trapped = s._replace(trapped=s.trapped.at[0].set(2))
    switches = ~_MOVES_ONLY
    assert int(_policy(tables, trapped, key, allowed=switches)[0]) == int(
        _policy(tables, trapped, key)[0]
    )


@pytest.mark.slow
def test_empty_line_equals_playout(tables, names):
    s = roark_state(tables, names)
    keys = jax.random.split(jax.random.PRNGKey(11), 8)
    line = jax.tree.map(jnp.asarray, empty_line())
    for k in keys:
        got = line_playout(tables, s, line, k, 60)
        want = playout(tables, root_state(tables, s, k), k, 60)
        assert int(got.won) == int(want.won)
        assert int(got.player_faints) == int(want.player_faints)
        assert int(got.turns) == int(want.turns)
        assert np.array_equal(np.asarray(got.deaths), np.asarray(want.deaths))
        assert int(got.fallback_turns) == int(got.turns)


def _greedy_first(tables, s, key):
    """The unmasked fallback's turn-1 action in ``_line_run(tables, s, line, key)``."""
    k_policy = jax.random.split(key, 3)[1]
    return int(_policy(tables, root_state(tables, s, key), k_policy)[0])


@pytest.mark.slow
def test_line_changes_the_first_turn(tables, names):
    s = roark_state(tables, names, player_count=1, opp_count=1)
    key = jax.random.PRNGKey(3)
    leer = 1  # Monferno's Leer: a move the greedy fallback does not pick here
    assert _greedy_first(tables, s, key) != leer
    result, final = _RUN_1(tables, s, _line(use=0, move=leer), key)
    full = int(tables.move_pp[s.moves[0, 0, leer]])
    assert int(result.turns) == 1
    assert int(final.pp[0, 0, leer]) == full - 1
    assert int(result.fallback_turns) == 0


@pytest.mark.slow
def test_wait_turn_does_not_switch(tables, names):
    # The unmasked fallback would switch out; in MODE_MOVES_ONLY the lead (faster than
    # Geodude) spends a move instead.
    s = _switching(tables, names)
    key = jax.random.PRNGKey(5)
    assert _greedy_first(tables, s, key) >= MOVE_ACTIONS
    line = _line(use=1, move=0, wait=1)
    assert _act(tables, s, line)[1] == MODE_MOVES_ONLY
    result, final = _RUN_1(tables, s, line, key)
    assert (int(result.turns), int(result.fallback_turns), int(result.refused)) == (1, 1, 0)
    spent = int(jnp.sum(s.pp[0, 0].astype(jnp.int32) - final.pp[0, 0].astype(jnp.int32)))
    assert spent == 1


@pytest.mark.slow
def test_rule_turn_ignores_a_refused_distribution(tables, names, monkeypatch):
    # As test_playout's refused-distribution test: unjitted, so the patch is traced in.
    s = roark_state(tables, names)
    nan = jnp.full(NUM_OPP_ACTIONS, jnp.nan, jnp.float32)
    monkeypatch.setattr(playout_module, "ai_action_distribution", lambda tables, state, layout: nan)
    key = jax.random.PRNGKey(12)
    ruled, _ = _line_run(tables, s, _line(use=0, move=2), key, 1)
    assert (int(ruled.refused), int(ruled.turns), int(ruled.fallback_turns)) == (0, 1, 0)
    fallback, _ = _line_run(tables, s, empty_line(), key, 1)
    assert (int(fallback.won), int(fallback.refused), int(fallback.turns)) == (REFUSED_WON, 1, 0)


@pytest.mark.slow
def test_pending_replacement_sends_the_answer(tables, names):
    s = at_replacement(roark_state(tables, names))
    greedy = int(replacement_policy(tables, s))
    answer = next(c for c in range(1, SLOTS) if c != greedy)
    run = jax.jit(partial(_line_run, max_turns=0))
    result, final = run(tables, s, _line(use=answer, move=0), jax.random.PRNGKey(6))
    assert int(result.refused) == 0 and int(result.turns) == 0
    assert int(final.active[0]) == answer


@pytest.mark.slow
def test_unstatusable_foe_still_ends(tables, names):
    # Ground is immune to Thunder Wave (tests/engine/test_effects.py, the type chart), so
    # `first` would be kept until its PP runs out; the playout must still stop.
    s = roark_state(tables, names, player_moves=["Thunder Wave"])
    slot = [int(m) for m in s.moves[0, 0]]
    first = slot.index(names.move("Thunder Wave"))
    move = slot.index(names.move("Ember"))
    max_turns = 30
    r = line_playout(
        tables, s, _line(use=0, move=move, first=first), jax.random.PRNGKey(4), max_turns
    )
    assert int(r.refused) == 0 and int(r.turns) <= max_turns
    assert int(r.ended) == 1 or int(r.turns) == max_turns
