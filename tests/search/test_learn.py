"""K3d-exp T6: learn-move prompts in the playout (design §3.5, ``nuzline.search.learn``).

``TurnEnd`` pays exp and runs its prompts (battle_controller_player.c:1846) before
``ReplaceFainted`` (:1848): ``finish_turn`` resolves the learns (the playout gives every
move up, ruling D3) before any replacement, and a playout started at a pending learn settles
it, then the opponent's replacement with the turn's carried ``powerMul``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.ids import Stat, Status
from nuzline.engine.levelup import advance_learns, needs_learn
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import EXP_REFUSED_WINDOW_C, step
from nuzline.search import playout as playout_module
from nuzline.search.learn import learn_reason, resolve_learns
from nuzline.search.playout import finish_turn, playout_from_action, refusal_reason
from nuzline.search.stage_a import rank_actions
from tests.engine.helpers import A, roark_real_state
from tests.engine.test_levelup import _base, _mask, _mon, _prinplup_state, _with

GEODUDE, ONIX, CRANIDOS = 0, 1, 2
_step = jax.jit(step)
_resolve_pending = jax.jit(playout_module._resolve_pending)


def _moves(names, s, slot: int) -> list[str]:
    return [names.move_names[int(m)] if int(m) else "" for m in np.asarray(s.moves[0, slot])]


# Four moves Machop never learns by level (its learnset is Low Kick, Leer, Focus Energy,
# Karate Chop, Foresight, Seismic Toss, Revenge, ...).
_FOREIGN = ("Splash", "Harden", "Tail Whip", "Sand-Attack")


def _with_prompt(tables, names, s):
    """``(s, slot)``: the lead Machop with four moves outside its learnset and learnset work
    from level 2 up to its level, so the scan stops at a prompt (its first learnable entry)."""
    assert int(s.species[0, 0]) == names.species("Machop")
    ids = jnp.asarray([names.move(m) for m in _FOREIGN], s.moves.dtype)
    pp = jnp.asarray([int(tables.move_pp[int(m)]) for m in ids], s.pp.dtype)
    s = s._replace(moves=s.moves.at[0, 0].set(ids), pp=s.pp.at[0, 0].set(pp))
    out = _with(s, learn_level={0: 2}, learn_cursor={0: 0})
    assert bool(needs_learn(advance_learns(tables, out)))
    return out, 0


def test_resolve_learns_gives_every_move_up(tables, names):
    s, slot = _with_prompt(tables, names, roark_real_state(tables, names, player_species="Machop"))
    out = resolve_learns(tables, s)
    assert not bool(needs_learn(out)) and int(out.learn_level[slot]) == 0
    np.testing.assert_array_equal(np.asarray(out.moves), np.asarray(s.moves))


def test_finish_turn_gives_up_before_the_opponent_replacement(tables, names):
    # Bulbasaur (one short of L13, moves full) KOs Geodude: L13 prompts PoisonPowder and
    # Sleep Powder. finish_turn gives both up, then the AI sends Bonsly in.
    bulbasaur = _mon(names, "Bulbasaur", 12, moves=("Tackle", "Growl", "Leech Seed", "Vine Whip"))
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = make_state(tables, [bulbasaur], opp).replace_hp(1, 0, 1)
    s = _with(s, exp={0: _base(tables, names, "Bulbasaur", 13) - 1})
    played = _step(tables, s, A("Tackle", "Tackle", s, names), empty_outcome())
    assert int(played.level[0, 0]) == 13 and int(played.learn_level[0]) == 13
    out, refused = finish_turn(tables, played, False, 10)
    assert not bool(refused)
    assert int(out.learn_level[0]) == 0 and int(out.active[1]) == 1
    assert _moves(names, out, 0) == ["Tackle", "Growl", "Leech Seed", "Vine Whip"]


def test_the_player_replacement_reads_the_moves_after_the_learn(tables, names):
    # Machop (1 HP, poisoned) KOs Magikarp with Tackle and faints to poison at the end of the
    # turn: both sides replace. Participants {0, 2}: E = 20 * 5 / 7 = 14, g = 7, t = 10, so
    # the bench Magikarp in slot 2 (one short of L15) reaches L15 and learns Tackle into its
    # empty slot. replacement_policy then prefers it (Tackle scores damage) over slot 1, a
    # Splash-only Magikarp (a tie without the learn, which goes to the lower slot).
    machop = _mon(names, "Machop", 20, current_hp=1, status=int(Status.POISON))
    karp = _mon(names, "Magikarp", 14, moves=("Splash",))
    opp = [_mon(names, "Magikarp", 5, moves=("Splash",))] * 2
    s = make_state(tables, [machop, karp, karp], opp).replace_hp(1, 0, 1)
    s = _mask(_with(s, exp={2: _base(tables, names, "Magikarp", 15) - 1}), 0, 2)
    played = _step(tables, s, A("Tackle", "Splash", s, names), empty_outcome())
    assert int(played.hp[0, 0]) == 0 and int(played.hp[1, 0]) == 0
    assert int(played.level[0, 2]) == 15 and int(played.learn_level[2]) == 15
    out, refused = finish_turn(tables, played, False, 10)
    assert not bool(refused)
    assert _moves(names, out, 2) == ["Splash", "Tackle", "", ""]
    assert int(out.active[0]) == 2 and int(out.active[1]) == 1


def test_learn_work_of_a_battle_ending_entry_ko_stays(tables, names):
    # Prinplup KOs Geodude: E = 125, t = 187, 3119 -> 3306, L17 (no move at 17), resolved
    # before the replacement. Bonsly L40 (1 HP) then faints on Stealth Rock, the last foe:
    # E = 68 * 40 / 7 = 388, t = 582, 3888 >= 3798: L18, a new marker that stays (the battle
    # is over, ruling D8). Resolving after the replacement would have kept the L17 marker
    # through the second payout and scanned 17-18 to the end (learn_level 0).
    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 40, current_hp=1)]
    s = _prinplup_state(tables, names, opp=opp).replace_hp(1, 0, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    played = _step(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    assert (int(played.level[0, 0]), int(played.learn_level[0])) == (17, 17)
    out, refused = finish_turn(tables, played, False, 10)
    assert not bool(refused) and int(out.over) == 1
    assert int(out.exp[0]) == 3888 and int(out.level[0, 0]) == 18
    assert int(out.learn_level[0]) == 18


def _window_c(tables, names):
    # Prinplup KOs Geodude (L17); the AI sends Bonsly L40 (1 HP) into Stealth Rock: its
    # payout levels Prinplup to 18 and the battle goes on (a third Pokemon): window (c).
    opp = [
        _mon(names, "Geodude", 12),
        _mon(names, "Bonsly", 40, current_hp=1),
        _mon(names, "Geodude", 12),
    ]
    s = _prinplup_state(tables, names, opp=opp).replace_hp(1, 0, 1)
    return s._replace(stealth_rock=s.stealth_rock.at[1].set(1))


def test_window_c_refuses_finish_turn(tables, names):
    s = _window_c(tables, names)
    played = _step(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    out, refused = finish_turn(tables, played, False, 10)
    assert bool(refused) and int(out.exp_refused) == EXP_REFUSED_WINDOW_C
    assert "window (c)" in learn_reason(out)
    assert learn_reason(played) is None


def test_window_c_refuses_the_playout_and_names_it(tables, names):
    s = _window_c(tables, names)
    action = names.move_slot(s, 0, "Bubble")
    key = jax.random.PRNGKey(3)
    r = playout_from_action(tables, s, action, key, 5)
    assert int(r.refused) == 1
    assert "window (c)" in refusal_reason(tables, s, action, key, 5)


def test_a_pending_learn_settles_the_opponent_replacement_with_the_carried_mul(tables, names):
    # roark.md section 14 data (tests/ai/test_replacement.py::test_power_mul_scales_stage_2):
    # at Geodude Atk 400 the AI sends Cranidos at powerMul 10 and Onix at 20. A playout
    # rooted at a pending learn (after a turn whose Pursuit hit, replace_mul 20) gives the
    # move up, then the opponent's replacement reads 20.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    s = s._replace(stats=s.stats.at[1, GEODUDE, Stat.ATK - 1].set(jnp.int16(400)))
    s, slot = _with_prompt(tables, names, s)
    for mul, newcomer in ((20, ONIX), (0, CRANIDOS)):
        held = s._replace(replace_mul=jnp.int8(mul))
        out, first, refused = _resolve_pending(tables, held, jnp.int32(-1))
        assert not bool(refused) and int(first) == -1
        assert int(out.active[1]) == newcomer and int(out.replace_mul) == 0
        assert int(out.learn_level[slot]) == 0


def test_a_stage_a_root_carries_no_learn_work(tables, names):
    s = _with(_prinplup_state(tables, names), learn_level={0: 17})
    with pytest.raises(AssertionError, match="no learn work"):
        rank_actions(tables, names, s, 2, jax.random.PRNGKey(0), max_turns=1)
