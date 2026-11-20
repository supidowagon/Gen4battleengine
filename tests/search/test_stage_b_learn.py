"""K3d-exp T8: the Stage B learn node (design §3.5, ``expand._turn`` / ``expand._learn``,
``stage_b`` learn decisions).

A payout runs its learn-move prompts inside it (E3-B2-21: MoveEnd bcp.c:3968, the field
and battler checks :941, :1296, TurnEnd :1846, before ``ReplaceFainted`` :1848): a turn's
child is paused at the prompt (``BattleState.learn_site``) before the opponent's replacement,
the turn's ``powerMul`` kept in ``ts_power_mul``; an answer resumes the turn through
``expand``. Learn work at a finished battle is no prompt (D8).

Bulbasaur learns PoisonPowder (index 4) and Sleep Powder (index 5) at level 13 (vanilla
learnset); one exp short of 13, any payout levels it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.ids import Stat, Status
from nuzline.engine.levelup import (
    GIVE_UP,
    LEARN_MON,
    LEARN_MOVE_END,
    advance_learns,
    learn,
    learn_options,
    needs_learn,
)
from nuzline.engine.observe import OBS_FIELDS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import PURSUIT_POWER_MUL, replacement_power_mul, step
from nuzline.search import expand, stage_b
from nuzline.search.leaves import LeafValue
from nuzline.search.plan_tree import render, to_json
from nuzline.search.stage_b import (
    SolveSettings,
    learn_label,
    result_from_dict,
    result_to_dict,
    solve,
)
from tests.engine.helpers import A, roark_real_state
from tests.engine.test_levelup import _base, _mon, _with
from tests.search.test_learn import _with_prompt

GEODUDE, ONIX, CRANIDOS = 0, 1, 2
KEY = jax.random.PRNGKey(0)
_jit_turn = jax.jit(expand._turn)


def _turn(tables, before, actions, outcome):
    """``expand._turn`` with no takeover variant (``taken`` all False, B11-1)."""
    return _jit_turn(tables, before, actions, outcome, jnp.zeros(2, bool))


_learn = jax.jit(expand._learn)
_answer = jax.jit(learn)
_advance = jax.jit(advance_learns)
_options = jax.jit(learn_options)
_OBS = dict(OBS_FIELDS)


def _batch(*states):
    return jax.tree.map(lambda *fs: np.stack([np.asarray(f) for f in fs]), *states)


def _bulbasaur(names):
    return _mon(names, "Bulbasaur", 12, moves=("Tackle", "Growl", "Leech Seed", "Vine Whip"))


def _short(tables, names, s, slot: int):
    """``s`` with the Bulbasaur in player ``slot`` one exp short of level 13."""
    return _with(s, exp={slot: _base(tables, names, "Bulbasaur", 13) - 1})


def _vs(tables, names, opp):
    return _short(tables, names, make_state(tables, [_bulbasaur(names)], opp), 0)


def _geodude_1hp(names, **kw):
    return _mon(names, "Geodude", 12, current_hp=1, **kw)


def _move(names, name: str) -> int:
    return names.move(name)


# --- the learn stop and the answers (expand._turn / expand._learn) ----------------------------


def _resume(tables, state, option):
    """The one child of answering the paused ``state``'s last prompt through ``expand`` (the
    turn resumes, E3-B2-21): ``(child state, child)``."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(expand, "CHUNK", 1024)  # as test_expand's small_chunk: no 16384-row padding
        store, kids = expand.expand(tables, _batch(state), [(0, option, 0)])
    (child,) = kids[0]
    return jax.tree.map(lambda x: x[child.state_row], store), child


# Slow: the resume runs expand's whole-engine laws and step programs (11.7 GB at a compile).
@pytest.mark.slow
def test_a_turn_child_stops_at_the_learn_prompt(tables, names):
    # E3-B2-21: the payout and its prompts run inside MoveEnd (bcp.c:3968), so the child is
    # paused there (``LEARN_MOVE_END``), before the opponent's replacement.
    s = _vs(tables, names, [_geodude_1hp(names), _mon(names, "Bonsly", 15)])
    after, refused, obs, keep = _turn(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert bool(keep)  # expand._turn's call_consistent row flag (E2 batch 5)
    assert not bool(refused)
    assert bool(needs_learn(after)) and int(after.learn_level[0]) == 13
    assert int(after.learn_site) == LEARN_MOVE_END
    assert stage_b._offered_move(tables, after) == _move(names, "PoisonPowder")
    # Stopped before the opponent's replacement: Geodude is still down.
    assert (int(after.active[1]), int(after.hp[1, GEODUDE])) == (0, 0)
    assert int(np.asarray(obs)[_OBS["learn_level"]][0]) == 13  # observed after the scan
    assert stage_b._pending(_batch(after)) == ["learn"]
    # Giving PoisonPowder up leaves the Sleep Powder prompt (one child, no turn played);
    # giving that up resumes the turn: the opponent's Bonsly comes in.
    second, child = _resume(tables, after, GIVE_UP)
    assert not child.refused and bool(needs_learn(second)) and int(second.learn_site) != 0
    assert stage_b._offered_move(tables, second) == _move(names, "Sleep Powder")
    assert int(second.active[1]) == 0
    done, child = _resume(tables, second, GIVE_UP)
    assert not child.refused and not bool(needs_learn(done))
    assert int(done.learn_site) == 0 and int(done.active[1]) == 1
    assert stage_b._pending(_batch(done)) == [""]


# Slow: the resume runs expand's whole-engine laws and step programs (11.7 GB at a compile).
@pytest.mark.slow
def test_the_learn_stop_holds_the_turns_pursuit_power_mul(tables, names):
    # Steelix switches to Bulbasaur; Geodude's Pursuit hits the switcher (powerMul 20,
    # BtlCmd_TryPursuit, battle_script.c:6884), then faints to poison at the end of the turn:
    # the payout levels the newcomer (a participant: it entered while Geodude had HP), whose
    # prompt pauses the turn (``LEARN_MON``) with the 20 kept in ``ts_power_mul`` for the
    # opponent's replacement after the resume.
    steelix = _mon(names, "Steelix", 30)
    geodude = _geodude_1hp(names, moves=("Pursuit",), status=int(Status.POISON))
    opp = [geodude, _mon(names, "Onix", 12), _mon(names, "Cranidos", 14)]
    s = _short(tables, names, make_state(tables, [steelix, _bulbasaur(names)], opp), 1)
    actions = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    assert int(replacement_power_mul(tables, s, actions)) == PURSUIT_POWER_MUL
    after, refused, _, keep = _turn(tables, s, actions, empty_outcome())
    assert bool(keep)  # expand._turn's call_consistent row flag (E2 batch 5)
    assert not bool(refused)
    assert int(after.hp[1, GEODUDE]) == 0 and int(after.active[1]) == GEODUDE
    assert bool(needs_learn(after)) and int(after.learn_level[1]) == 13
    assert int(after.learn_site) == LEARN_MON
    assert int(after.ts_power_mul) == PURSUIT_POWER_MUL
    second, _ = _resume(tables, after, GIVE_UP)
    assert bool(needs_learn(second)) and int(second.ts_power_mul) == PURSUIT_POWER_MUL
    done, child = _resume(tables, second, GIVE_UP)
    assert not child.refused and int(done.learn_site) == 0
    assert int(done.active[1]) != GEODUDE and int(done.replace_mul) == 0


def test_a_learn_answer_sends_the_opponent_replacement_with_the_carried_mul(tables, names):
    # roark.md section 14 data (tests/ai/test_replacement.py::test_power_mul_scales_stage_2):
    # at Geodude Atk 400 the AI sends Onix at powerMul 20 and Cranidos at 10.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    s = s._replace(stats=s.stats.at[1, GEODUDE, Stat.ATK - 1].set(jnp.int16(400)))
    s = _advance(tables, _with_prompt(tables, names, s)[0])
    for mul, newcomer in ((PURSUIT_POWER_MUL, ONIX), (0, CRANIDOS)):
        held = s._replace(replace_mul=jnp.int8(mul))
        for _ in range(21):
            held, refused, _ = _learn(tables, held, jnp.int32(GIVE_UP))
            assert not bool(refused)
            if not bool(needs_learn(held)):
                break
        assert int(held.active[1]) == newcomer and int(held.replace_mul) == 0


# Fast after the module's other tests compile its program; under -m fullset alone it pays
# that compile (pod 172-190 s), which only the fullset run sees. Fullset: it asserts the
# coverage refusal of an opponent replacement after its AI switched (roark.md 14, item 3).
@pytest.mark.fullset
def test_an_ai_switch_with_a_needed_replacement_pauses_then_refuses_at_the_resume(tables, names):
    # The foe switches to Onix (1 HP); Tackle KOs it. E3-B2-21: the payout and Bulbasaur's
    # prompts run in MoveEnd (bcp.c:3968), before the tail (:3973) and the opponent's
    # replacement, so the child pauses at the prompt unrefused, the AI switch kept in
    # ``ts_switched``. Answering both prompts resumes the turn; its replacement after the
    # AI switch is then refused (``send_replacements`` through ``turn_muls``).
    s = _vs(tables, names, [_mon(names, "Geodude", 12), _mon(names, "Onix", 12, current_hp=1)])
    actions = jnp.asarray([names.move_slot(s, 0, "Tackle"), MOVE_ACTIONS + 1], jnp.int32)
    after, refused, _, keep = _turn(tables, s, actions, empty_outcome())
    assert bool(keep)  # expand._turn's call_consistent row flag (E2 batch 5)
    assert not bool(refused)
    assert int(after.learn_site) == LEARN_MOVE_END and bool(needs_learn(after))
    assert (int(after.active[1]), int(after.hp[1, ONIX])) == (ONIX, 0)
    assert int(after.ts_switched[1]) != 0
    answered = _answer(tables, _answer(tables, after, GIVE_UP), GIVE_UP)
    assert not bool(needs_learn(answered)) and int(answered.learn_site) != 0
    # expand's resumed row: the answer's request actions (option, 0) are not read.
    done, refused, _, keep = _turn(tables, answered, jnp.asarray([GIVE_UP, 0]), empty_outcome())
    assert bool(keep) and bool(refused)
    assert int(done.learn_site) == 0 and int(done.active[1]) == ONIX


def test_learn_work_at_a_battle_ending_ko_is_terminal(tables, names):
    # D8: the last foe faints; the prompt stays in the state but is no decision.
    s = _vs(tables, names, [_geodude_1hp(names)])
    after, refused, obs, keep = _turn(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert bool(keep)  # expand._turn's call_consistent row flag (E2 batch 5)
    assert not bool(refused) and int(after.over) == 1
    assert int(after.learn_level[0]) == 13
    assert int(np.asarray(obs)[_OBS["learn_level"]][0]) == 0
    assert stage_b._pending(_batch(after)) == [""]


# --- Stage B ------------------------------------------------------------------------------

SETTINGS = SolveSettings(max_depth=1, budget_playouts=64, epsilon=0.0)


def _sleep_powder_leaf(names):
    """A ``leaf_fn``: a state is won iff the player's slot 0 knows Sleep Powder."""
    sp = _move(names, "Sleep Powder")

    def leaf_fn(states, weights, pairs=None):
        knows = (np.asarray(states.moves)[:, 0, 0] == sp).any(axis=1)
        return [LeafValue(10 * int(k), 10, 10 * int(k), 10, False) for k in knows]

    return leaf_fn


def _plan_nodes(nodes):
    for node in nodes:
        yield node
        for kids in node.children.values():
            yield from _plan_nodes(kids)


@pytest.mark.slow
def test_stage_b_branches_into_the_learn_options(tables, names):
    s = _vs(tables, names, [_geodude_1hp(names), _mon(names, "Bonsly", 15)])
    result = solve(tables, names, s, KEY, SETTINGS, leaf_fn=_sleep_powder_leaf(names))
    sp = _move(names, "Sleep Powder")
    learns = [n for n in _plan_nodes([result.plan]) if n.decision == "learn"]
    assert learns, "no learn decision in the plan"
    for node in learns:
        assert node.kind == "decision" and GIVE_UP in node.values
        assert max(node.values.values()) >= node.values[GIVE_UP]
        assert node.values[node.action] == max(node.values.values())
    chosen = [n for n in learns if n.learn_move == sp]
    assert chosen
    for node in chosen:  # learning Sleep Powder wins, giving it up does not
        assert node.action != GIVE_UP
        assert node.values[node.action] > node.values[GIVE_UP]
    text = "\n".join(render(result, names, 3, root=s))
    assert "learn Sleep Powder over" in text


@pytest.mark.slow
def test_a_root_with_a_pending_learn(tables, names):
    # The raw state after the KO, before the scan: solve scans it (advance_learns) and the
    # root is the PoisonPowder prompt.
    s = _vs(tables, names, [_geodude_1hp(names), _mon(names, "Bonsly", 15)])
    played = jax.jit(step)(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert int(played.learn_level[0]) == 13 and int(played.learn_cursor[0]) == 0
    result = solve(tables, names, played, KEY, SETTINGS, leaf_fn=_sleep_powder_leaf(names))
    assert result.plan.decision == "learn"
    pp = _move(names, "PoisonPowder")
    assert result.plan.learn_move == pp
    root = _advance(tables, played)
    legal = np.flatnonzero(np.asarray(_options(tables, root))).tolist()
    assert sorted(r.action for r in result.ranking) == legal
    moves = np.asarray(root.moves)[0, 0]
    want = {learn_label(names, pp, 0 if a == GIVE_UP else int(moves[a]), a) for a in legal}
    assert {r.label for r in result.ranking} == want
    # Each answer leads to the Sleep Powder prompt, where learning it is chosen.
    best = result.ranking[0]
    assert best.estimate == pytest.approx(1.0)
    (second,) = result.plan.children[best.action]
    assert second.decision == "learn" and second.learn_move == _move(names, "Sleep Powder")
    assert second.action != GIVE_UP
    assert result_from_dict(result_to_dict(result)) == result
    data = to_json(result, names, root=played)
    assert data["plan"]["decision"] == "learn"


@pytest.mark.slow
def test_a_learn_at_a_terminal_state_stays_terminal(tables, names):
    s = _vs(tables, names, [_geodude_1hp(names)])
    zero = _sleep_powder_leaf(names)
    result = solve(tables, names, s, KEY, SETTINGS, leaf_fn=zero)
    tackle = names.move_slot(s, 0, "Tackle")
    kids = result.plan.children[tackle]
    assert any(k.kind == "terminal" for k in kids)
    assert not any(n.decision == "learn" for n in _plan_nodes([result.plan]))
