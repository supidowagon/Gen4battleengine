import jax
import jax.numpy as jnp

from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import needs_replacement, replace, step

# roark_state: the sample party vs the built Roark party, both leads in slot 0.
from tests.engine.helpers import roark_state


def test_faster_side_moves_first_and_deals_damage(tables, names):
    s = roark_state(tables, names)  # Monferno (Spe 30) vs Geodude (Spe 10)
    o = empty_outcome()
    scratch = names.move_slot(s, 0, "Scratch")
    out = step(tables, s, jnp.array([scratch, names.move_slot(s, 1, "Tackle")]), o)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])
    assert int(out.turn) == 1 and int(out.pp[0, 0, scratch]) == int(s.pp[0, 0, scratch]) - 1


def test_speed_tie_uses_outcome(tables, names):
    s = roark_state(tables, names, equal_speed=True)
    o = empty_outcome()
    first = action_order(tables, s, jnp.array([0, 0]), o)
    second = action_order(tables, s, jnp.array([0, 0]), o._replace(speed_tie=jnp.int8(1)))
    assert int(first[0]) == 0 and int(second[0]) == 1


def test_miss_deals_no_damage(tables, names):
    s = roark_state(tables, names)
    o = empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8))
    out = step(tables, s, jnp.array([0, 0]), o)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_player_faint_is_terminal(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1)  # player's active at 1 HP
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.over) == 1 and int(out.winner) == 1
    assert int(out.player_faints) == 1


def test_opponent_faint_needs_replacement_then_replace(tables, names):
    s = roark_state(tables, names).replace_hp(1, 0, 1)
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert needs_replacement(out).tolist() == [0, 1]
    out2 = replace(tables, out, 1, 1)
    assert int(out2.active[1]) == 1 and needs_replacement(out2).tolist() == [0, 0]


def test_all_opponents_fainted_is_win(tables, names):
    s = roark_state(tables, names, opp_count=1).replace_hp(1, 0, 1)
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2


def test_switch_goes_before_moves_and_resets_stages(tables, names):
    s = roark_state(tables, names).with_stages(0, atk=2)
    out = step(tables, s, jnp.array([4 + 1, 0]), empty_outcome())
    assert int(out.active[0]) == 1 and int(out.stages[0, 1]) == 0
    assert int(out.hp[0, 1]) < int(s.hp[0, 1])


def test_struggle_when_no_pp(tables, names):
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_struggle_recoil_is_quarter_max_hp_and_spends_no_pp(tables, names):
    # subscript_struggle.s: recoil = BattleSystem_Divide(-maxHP, 4); Struggle has no PP to spend.
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    miss = empty_outcome()._replace(hit=jnp.array([[1] * 5, [0] * 5], jnp.int8))
    out = step(tables, s, jnp.array([0, 0]), miss)  # Geodude's Tackle misses
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 4
    assert out.pp[0, 0].tolist() == [0, 0, 0, 0]


def test_terminal_state_is_left_unchanged(tables, names):
    s = roark_state(tables, names)._replace(over=jnp.int8(1), winner=jnp.int8(2))
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(out, s, strict=True))


def test_step_jits_and_vmaps_like_eager(tables, names):
    s = roark_state(tables, names)
    acts = jnp.array([0, 0], jnp.int32)
    o = empty_outcome()
    eager = step(tables, s, acts, o)
    jitted = jax.jit(step)(tables, s, acts, o)
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(eager, jitted, strict=True))
    assert all(a.dtype == b.dtype for a, b in zip(eager, s, strict=True))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), s)
    batched = jax.vmap(step, in_axes=(None, 0, None, None))(tables, batch, acts, o)
    assert batched.hp.shape == (2, 2, 6) and bool(jnp.array_equal(batched.hp[1], eager.hp))
