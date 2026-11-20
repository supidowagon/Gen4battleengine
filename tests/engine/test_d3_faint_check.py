"""D3-7: the faint check's side-out test and ``_blacked_out`` per format (sheet D3-7 section 5).
CheckBattleOver (bcp.c:4181-4262) branches a / b / c, CheckBlackOut (script.c:8846-8915). The
format cases are the action phase of a turn (``play``: engine equal to reference, every field)
where one Tackle faints the battler left with HP; ``deaths`` and the draw are direct calls."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.format import (
    DOUBLES,
    MULTI,
    SINGLES,
    TWO_TRAINERS,
    blackout_parties,
    side_out_parties,
)
from nuzline.engine.step import (
    WINNER_OPPONENT,
    WINNER_PLAYER,
    _blacked_out,
    _faint_check,
    _sides_out,
)
from tests.engine.d2_states import four
from tests.engine.test_d3_action_loop import play

TACKLE, SWORDS = 0, 1  # action ids of d2_states.DEFAULT_MOVES


def only_alive(state, p: int, slot=None):
    """Party ``p`` with every Pokemon at 0 HP except ``slot`` (at 1 HP); ``None``: all at 0."""
    hp = state.hp.at[p].set(0)
    if slot is not None:
        hp = hp.at[p, slot].set(1)
    return state._replace(hp=hp)


def empty(state, p: int):
    return only_alive(state, p, None)


def tackle(tables, state, attacker: int, target: int):
    """The action phase where ``attacker`` Tackles ``target`` first and the others Swords Dance."""
    acts = [SWORDS] * 4
    acts[attacker] = TACKLE
    order = [attacker] + [b for b in range(4) if b != attacker]
    targets = [0] * 4
    targets[attacker] = target
    out, _ = play(tables, state, acts, order, targets=tuple(targets))
    return out


def result(out):
    return int(out.over), int(out.winner)


# ---------------------------------------------------------------- the per-format tables


def test_side_out_parties_table():
    assert [side_out_parties(f, 0) for f in (SINGLES, DOUBLES, TWO_TRAINERS, MULTI)] == [(0,)] * 4
    assert side_out_parties(SINGLES, 1) == (1,)
    assert side_out_parties(DOUBLES, 1) == (1,)
    assert side_out_parties(TWO_TRAINERS, 1) == (1, 2)
    assert side_out_parties(MULTI, 1) == (1, 3)


BLACKOUT = {  # rule 4: the parties CheckBlackOut sums for battler b
    SINGLES: {0: (0,), 1: (1,)},
    DOUBLES: {0: (0,), 1: (1,), 2: (0,), 3: (1,)},
    TWO_TRAINERS: {0: (0,), 1: (1, 2), 2: (0,), 3: (1, 2)},
    MULTI: {0: (0,), 1: (1, 3), 2: (2, 0), 3: (1, 3)},
}


@pytest.mark.parametrize("fmt", [SINGLES, DOUBLES, TWO_TRAINERS, MULTI])
def test_blackout_parties_table(fmt):
    for b, want in BLACKOUT[fmt].items():
        assert blackout_parties(fmt, b) == want, (fmt, b)


@pytest.mark.parametrize("fmt", [DOUBLES, TWO_TRAINERS, MULTI])
def test_blacked_out_per_format(tables, names, fmt):
    """Out exactly when every named party is empty: a state with only party p alive is blacked out
    for b iff p is not among b's parties (MULTI battler 0: party 0 only, battler 2: parties 2, 0)."""
    s = four(tables, names, fmt)
    for b, parties in BLACKOUT[fmt].items():
        for p in range(fmt.P):
            alive = s._replace(hp=jnp.zeros_like(s.hp).at[p, 0].set(1))
            assert bool(_blacked_out(alive, jnp.int32(b))) == (p not in parties), (b, p)


# ---------------------------------------------------------------- side-out cases (play)


def test_doubles_opponent_party_empty_is_a_win(tables, names):
    s = only_alive(four(tables, names, DOUBLES), 1, 0)
    assert result(tackle(tables, s, 0, 1)) == (1, WINNER_PLAYER)


def test_doubles_opponent_second_battler_alive_goes_on(tables, names):
    s = only_alive(four(tables, names, DOUBLES), 1, 0)
    s = s._replace(hp=s.hp.at[1, 1].set(5))  # battler 3 (party 1's slot 1) still stands
    assert result(tackle(tables, s, 0, 1)) == (0, 0)


def test_doubles_player_party_empty_is_a_loss(tables, names):
    s = only_alive(four(tables, names, DOUBLES), 0, 0)
    assert result(tackle(tables, s, 1, 0)) == (1, WINNER_OPPONENT)


def test_doubles_both_empty_in_one_faint_check_is_a_draw_played_as_a_loss(tables, names):
    s = four(tables, names, DOUBLES)
    after = s._replace(hp=jnp.zeros_like(s.hp))
    out = _faint_check(tables, s, after)
    assert (int(out.over), int(out.winner)) == (1, WINNER_OPPONENT)  # ruling 2


def test_two_trainers_one_opponent_party_empty_is_not_over(tables, names):
    s = only_alive(four(tables, names, TWO_TRAINERS), 1, 0)  # party 2 is whole
    assert result(tackle(tables, s, 0, 1)) == (0, 0)


def test_two_trainers_both_opponent_parties_empty_is_a_win(tables, names):
    # (the faint is battler 1's: ``_move_end``'s foe is still ``_foe(side)``, D3-15)
    s = only_alive(empty(four(tables, names, TWO_TRAINERS), 2), 1, 0)
    assert result(tackle(tables, s, 0, 1)) == (1, WINNER_PLAYER)


def test_two_trainers_player_party_empty_is_a_loss(tables, names):
    s = only_alive(four(tables, names, TWO_TRAINERS), 0, 0)
    assert result(tackle(tables, s, 1, 0)) == (1, WINNER_OPPONENT)


def test_multi_player_party_empty_while_the_partner_stands_is_a_loss(tables, names):
    """Branch a: battler 0's own party decides; battler 2's party 2 is whole."""
    s = only_alive(four(tables, names, MULTI), 0, 0)
    assert result(tackle(tables, s, 1, 0)) == (1, WINNER_OPPONENT)


def test_multi_partner_party_empty_alone_does_not_end_it(tables, names):
    """The partner's battler 2 never decides (rule 1a)."""
    s = four(tables, names, MULTI)
    out = _faint_check(tables, s, only_alive(s, 2, None))
    assert (int(out.over), int(out.winner)) == (0, 0)


def test_multi_both_enemy_parties_empty_is_a_win(tables, names):
    s = empty(only_alive(four(tables, names, MULTI), 1, 0), 3)
    assert result(tackle(tables, s, 0, 1)) == (1, WINNER_PLAYER)


def test_multi_one_enemy_party_alone_is_not_a_win(tables, names):
    s = only_alive(four(tables, names, MULTI), 1, 0)  # party 3 whole
    assert result(tackle(tables, s, 0, 1)) == (0, 0)
    s = four(tables, names, MULTI)  # party 3 emptied alone: party 1 whole
    out = _faint_check(tables, s, only_alive(s, 3, None))
    assert (int(out.over), int(out.winner)) == (0, 0)


# ---------------------------------------------------------------- deaths stay party 0


def test_multi_deaths_count_party_zero_only(tables, names):
    s = four(tables, names, MULTI)
    partner = _faint_check(tables, s, s._replace(hp=s.hp.at[2, 0].set(0)))
    assert int(partner.player_faints) == 0 and not np.any(np.asarray(partner.deaths))
    own = _faint_check(tables, s, s._replace(hp=s.hp.at[0, 0].set(0)))
    assert int(own.player_faints) == 1 and np.asarray(own.deaths).tolist()[0] == 1


# ---------------------------------------------------------------- B = 2 is unchanged


def _old_sides_out(state):
    return (
        jnp.all((state.species[0] == 0) | (state.hp[0] == 0)),
        jnp.all((state.species[1] == 0) | (state.hp[1] == 0)),
    )


def _old_blacked_out(state, side):
    return ~jnp.any((state.species[side] != 0) & (state.hp[side] > 0))


def test_singles_side_out_and_blackout_trace_the_old_ops(tables, names):
    s = four(tables, names, SINGLES)
    assert str(jax.make_jaxpr(_sides_out)(s)) == str(jax.make_jaxpr(_old_sides_out)(s))
    assert str(jax.make_jaxpr(_blacked_out)(s, 1)) == str(jax.make_jaxpr(_old_blacked_out)(s, 1))


def test_singles_side_out_and_blackout_equal_the_old_ones_on_20_states(tables, names):
    s = four(tables, names, SINGLES)
    rng = np.random.default_rng(7)
    for _ in range(20):
        alive = rng.random(s.hp.shape) < 0.3
        x = s._replace(hp=jnp.where(jnp.asarray(alive), s.hp, 0))
        for a, b in zip(_sides_out(x), _old_sides_out(x), strict=True):
            assert bool(a) == bool(b)
        for side in (0, 1):
            assert bool(_blacked_out(x, jnp.int32(side))) == bool(_old_blacked_out(x, side))
        new = _faint_check(tables, s, x)
        assert bool(new.over) == any(bool(v) for v in _old_sides_out(x))
