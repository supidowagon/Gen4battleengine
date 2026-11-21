"""Learn-move prompts in the pluggable player's loop (``nuzline.search.learn``, K3d-exp
design §3.5): ``player_playout`` resolves a turn's learns before the replacements and
settles a pending learn at its root, as ``playout`` does, and both loops'
``refusal_reason`` name a refused learn (none since E3-B2-21: no window (c)). The states are
``tests/search/test_learn.py``'s.

Compile cost shapes this module (each loop or turn program is the whole engine, several GB
resident once compiled): the free-policy runs share one turn count and the fixed-action runs
another, a root refused at its settle runs the no-turn programs (max_turns 0: the settle
alone), the playouts the players are compared with run once per module, and a test that
needs two whole-engine programs of its own drops the first before the second
(``drop_programs``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.layout import VANILLA
from nuzline.engine.ids import Stat
from nuzline.engine.levelup import needs_learn
from nuzline.engine.state import make_state
from nuzline.search import player as player_module
from nuzline.search import playout as playout_module
from nuzline.search.line import LINE, empty_line
from nuzline.search.player import GREEDY, Player, _player_run, player_playout
from nuzline.search.playout import playout, playout_from_action, replacement_policy, root_state
from tests.engine.helpers import drop_programs, roark_real_state
from tests.engine.test_levelup import _base, _mon, _prinplup_state, _with
from tests.search.test_learn import (
    CRANIDOS,
    GEODUDE,
    HAZARD_SETTLED_MOVES,
    ONIX,
    _moves,
    _window_c,
    _with_prompt,
    hazard_settle_root,
    perish_root,
)

# Slow: each test compiles or loads a whole-engine loop or turn program (about 8 GB resident
# at its compile, 10 GB at a load from the persistent cache, minutes either way), past the
# fast tier's 60 s and 4 GB.
pytestmark = pytest.mark.slow


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


def _bonsly_last(tables, names):
    # test_learn.py::test_learn_work_of_a_battle_ending_entry_ko_stays: Prinplup KOs Geodude
    # (L17, no move); Bonsly L40 (1 HP), the last foe, faints on Stealth Rock in the replacement
    # pass and ends the battle: its payout's L18 work stays (D8, the loop's skipped second
    # resolve and _replay's).
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 40, current_hp=1)]
    s = _prinplup_state(tables, names, opp=opp).replace_hp(1, 0, 1)
    return s._replace(stealth_rock=s.stealth_rock.at[1].set(1))


def _same(got, want, root=""):
    for field in ("won", "player_faints", "turns", "refused", "ended", "hp_left", "pp_left"):
        assert float(getattr(got, field)) == float(getattr(want, field)), f"{root}: {field}"
    assert np.array_equal(np.asarray(got.deaths), np.asarray(want.deaths)), f"{root}: deaths"


TURNS = 8  # the free-policy runs' one turn count (playout, GREEDY, LINE)
FIXED_TURNS = 5  # the fixed-action runs' one turn count (playout_from_action, FIXED)
_ROOTS = (
    _level_up_turn,
    _pending_learn_root,
    _pending_learn_root_pursuit,
    _window_c_root,
    _window_c,  # the hazard faint's work answered after the passes, the battle going on
    _bonsly_last,  # the hazard faint ends the battle: the work stays (D8)
)
_KEYS = jax.random.split(jax.random.PRNGKey(5), 4)


@pytest.fixture(scope="module")
def roots(tables, names):
    """``[(label, root, key, playout result)]`` over the roots and four keys: the
    ``playout`` every player run is compared with, computed once for the module; its program
    is dropped before the players' compile (``drop_programs``)."""
    out = []
    for build in _ROOTS:
        s = build(tables, names)
        for k in _KEYS:
            want = jax.device_get(playout(tables, root_state(tables, s, k), k, TURNS))
            out.append((f"{build.__name__}, key {np.asarray(k).tolist()}", s, k, want))
    drop_programs()
    return out


def test_greedy_equals_playout_through_learns(tables, roots):
    # The GREEDY run answers each turn's learn work (player.py: resolve_learns after the turn,
    # before send_replacements, and again after them and the replacement passes) and the root's
    # (after its settle, whose send can leave a hazard faint's work, and its passes). Work
    # survives only at a finished battle (D8): from the turn itself, from a newcomer's hazard
    # faint in send_replacements or the passes, or from the root's settle and passes. The result
    # fields alone cannot see a skipped resolve (every move is given up anyway), so each
    # unrefused run's final state is checked: (1) the host replay of the same run
    # (player._replay, the same key splits) to its turn count equals it field by field, which
    # ties the run to the replay (every root has such a run); (2) a run that did not end holds
    # no learn work; (3) work at a finished battle is the final turn's: the replay one turn
    # shorter holds none. The pending roots also pin the root's opponent replacement and its
    # carried powerMul (Cranidos at 0, Onix at 20): a run settling them otherwise plays a
    # different battle.
    runs = []
    for root, s, k, want in roots:
        got, out = jax.device_get(_run(tables, s, GREEDY, (), k, TURNS))
        _same(got, want, root)
        if root.startswith(_window_c_root.__name__):
            # E3-B2-21: no window (c) any more; the settle refuses nothing.
            assert int(got.refused) == 0 and int(want.refused) == 0, root
        runs.append((root, s, k, got, out))
    drop_programs()  # the loop program before the replay's turn programs (module docstring)
    checks = {build.__name__: 0 for build in _ROOTS}
    sites = {"window_c": 0, "bonsly_last": 0}  # runs that reach the two hazard-faint sites
    for root, s, k, got, out in runs:
        if int(got.refused):
            continue
        turns = int(got.turns)
        reason, final, n = player_module._replay(tables, s, GREEDY, (), k, TURNS, VANILLA)
        assert reason is None and n == turns, root
        for name, a, b in zip(out._fields, out, final, strict=True):
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"{root}: replay {name}"
        name = root.split(",")[0]
        checks[name] += 1
        if name == _window_c.__name__:
            # Bonsly's hazard payout (L17 -> L18) in turn 1's passes, the battle going on: the
            # resolve after the passes (player.py, the second _select) answers its work.
            _, first, n = player_module._replay(tables, s, GREEDY, (), k, 1, VANILLA)
            sites["window_c"] += n == 1 and int(first.over) == 0 and int(first.level[0, 0]) == 18
        if name == _bonsly_last.__name__:
            sites["bonsly_last"] += bool(int(got.ended) and needs_learn(out) and turns)
        if not int(got.ended):
            assert not bool(needs_learn(out)), f"{root}: learn work was left pending"
        elif bool(needs_learn(out)) and turns:  # turns 0: the root's settle ended the battle
            reason, before, n = player_module._replay(tables, s, GREEDY, (), k, turns - 1, VANILLA)
            assert reason is None and n == turns - 1 and int(before.over) == 0, root
            assert not bool(needs_learn(before)), f"{root}: learn work left before the final turn"
    assert all(checks.values()), f"roots with no unrefused run: {checks}"
    assert all(sites.values()), f"hazard-faint sites not reached: {sites}"


def test_the_root_settle_answers_a_hazard_faints_learn_work(tables, names):
    # The root's settle sends Bonsly into Stealth Rock (test_learn.py::hazard_settle_root); the
    # resolve after the root's passes (player._player_run, player._replay) scans its L19 work
    # before turn 1, so turn 1's policy reads BubbleBeam in the empty slot. Without it the work
    # stays unscanned (learn_level 19, slot 3 empty) until turn 1's MoveEnd. The no-turn
    # program (max_turns 0) of test_a_pending_learn_root_sends_with_the_carried_mul.
    s = hazard_settle_root(tables, names)
    key = jax.random.PRNGKey(0)
    got, out = _run(tables, s, GREEDY, (), key, 0)
    reason, replayed, n = player_module._replay(tables, s, GREEDY, (), key, 0, VANILLA)
    assert reason is None and n == 0 and int(got.refused) == 0
    for state in (out, replayed):
        assert int(state.over) == 0 and int(state.hp[1, 1]) == 0
        assert int(state.level[0, 0]) == 19
        assert np.asarray(state.learn_level).tolist() == [0] * 6
        assert _moves(names, state, 0) == HAZARD_SETTLED_MOVES


def test_a_battle_ending_turn_end_payout_keeps_its_learn_work(tables, names):
    # Turn 1's Perish Song faint ends the battle at TurnEnd's payout, which step leaves unscanned
    # (test_learn.py::perish_root): the loop's first resolve after the turn (player._player_run)
    # and the replay's (player._replay) keep it (ruling D8), so no move fills the empty slots.
    s = perish_root(tables, names)
    key = jax.random.PRNGKey(0)
    got, out = _run(tables, s, GREEDY, (), key, TURNS)
    assert int(got.ended) == 1 and int(got.won) == 1 and int(got.turns) == 1
    drop_programs()  # the loop program before the replay's turn programs (module docstring)
    reason, replayed, n = player_module._replay(tables, s, GREEDY, (), key, TURNS, VANILLA)
    assert reason is None and n == 1
    for state in (out, replayed):
        assert int(state.over) == 1 and int(state.level[0, 0]) == 13
        assert int(state.learn_level[0]) == 13
        assert _moves(names, state, 0) == ["Growl", "", "", ""]


@pytest.mark.fullset
def test_a_pending_learn_root_is_not_refused(tables, names):
    # E3-B2-21: the root's settle refuses nothing (no window (c)): both loops run with no turn to
    # play (max_turns 0, the no-turn programs: the root's settle alone).
    s = _window_c_root(tables, names)
    key = jax.random.PRNGKey(7)
    assert int(_run(tables, s, GREEDY, (), key, 0)[0].refused) == 0
    assert int(playout(tables, s, key, 0).refused) == 0
    # Both refusal_reason replays name it with no turn to replay (max_turns 0), so the name
    # can only come from the root's settle (player._replay's and playout.refusal_reason's
    # settle_reason, as _resolve_pending's settle_learns); with turns, the first replayed
    # turn names the same window. Formerly test_learn.py::
    # test_refusal_reason_settles_a_pending_learn_root. Python-level replays: no new program.
    for turns in (0, TURNS):
        reason = player_module.refusal_reason(tables, s, GREEDY, (), key, turns) or ""
        assert not reason, ("player", turns)
        reason = playout_module.refusal_reason(tables, s, 0, key, turns) or ""
        assert not reason, ("playout", turns)


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


@pytest.mark.fullset
def test_a_hazard_ko_turn_does_not_refuse_the_fixed_player(tables, names):
    # The playout's window (c) turn (the test below) in the player loop, on the FIXED
    # program of the test above; its refusal_reason replays the turn in playout's _play_jit,
    # compiled after the loop program is dropped.
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    assert int(_run(tables, s, FIXED, action, key, FIXED_TURNS)[0].refused) == 0
    drop_programs()
    assert not player_module.refusal_reason(tables, s, FIXED, action, key, FIXED_TURNS)


@pytest.mark.fullset
def test_a_hazard_ko_turn_does_not_refuse_the_playout(tables, names):
    # Refused in the first turn's payout (test_learn.py::_window_c). Formerly
    # test_learn.py::test_window_c_refuses_the_playout_and_names_it. The replay's _play_jit
    # and finish_turn programs, then the loop program after they are dropped.
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    assert not playout_module.refusal_reason(tables, s, action, key, FIXED_TURNS)
    drop_programs()
    assert int(playout_from_action(tables, s, action, key, FIXED_TURNS).refused) == 0


# test_greedy_equals_playout_through_learns' two pending roots hold the property too (it
# fails with the root's settle_learns removed from _player_run, and with its carried powerMul
# replaced by the normal one).
@pytest.mark.parametrize("mul, newcomer", [(0, CRANIDOS), (20, ONIX)])
def test_a_pending_learn_root_sends_with_the_carried_mul(tables, names, mul, newcomer):
    s = _pending_learn_root(tables, names, mul)
    out = _run(tables, s, GREEDY, (), jax.random.PRNGKey(0), 0)[1]
    assert int(out.active[1]) == newcomer and int(out.learn_level[0]) == 0
