import jax
import jax.numpy as jnp
import pytest

from nuzline.engine.ids import Status
from nuzline.engine.order import action_order
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.step import needs_replacement, replace, step

# roark_state: the sample party vs the built Roark party, both leads in slot 0.
from tests.engine.helpers import jit_step, roark_state


@pytest.mark.slow
def test_faster_side_moves_first_and_deals_damage(tables, names):
    s = roark_state(tables, names)  # Monferno (Spe 30) vs Geodude (Spe 10)
    o = empty_outcome()
    scratch = names.move_slot(s, 0, "Scratch")
    out = jit_step(tables, s, jnp.array([scratch, names.move_slot(s, 1, "Tackle")]), o)
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
    o = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = jit_step(tables, s, jnp.array([0, 0]), o)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_last_player_faint_is_a_whiteout(tables, names):
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1)  # the only Pokemon
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.over) == 1 and int(out.winner) == 1
    assert int(out.player_faints) == 1


def test_opponent_faint_needs_replacement_then_replace(tables, names):
    s = roark_state(tables, names).replace_hp(1, 0, 1)
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert needs_replacement(out).tolist() == [0, 1]
    out2 = replace(tables, out, 1, 1)
    assert int(out2.active[1]) == 1 and needs_replacement(out2).tolist() == [0, 0]


def test_all_opponents_fainted_is_win(tables, names):
    s = roark_state(tables, names, opp_count=1).replace_hp(1, 0, 1)
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2


def test_switch_goes_before_moves_and_resets_stages(tables, names):
    s = roark_state(tables, names).with_stages(0, atk=2)
    out = jit_step(tables, s, jnp.array([4 + 1, 0]), empty_outcome())
    assert int(out.active[0]) == 1 and int(out.stages[0, 1]) == 0
    assert int(out.hp[0, 1]) < int(s.hp[0, 1])


def test_struggle_when_no_pp(tables, names):
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_struggle_recoil_is_quarter_max_hp_and_spends_no_pp(tables, names):
    # subscript_struggle.s: recoil = BattleSystem_Divide(-maxHP, 4); Struggle has no PP to spend.
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    miss = empty_outcome()._replace(hit=jnp.array([[1] * MAX_HITS, [0] * MAX_HITS], jnp.int8))
    out = jit_step(tables, s, jnp.array([0, 0]), miss)  # Geodude's Tackle misses
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 4
    assert out.pp[0, 0].tolist() == [0, 0, 0, 0]


def test_terminal_state_is_left_unchanged(tables, names):
    s = roark_state(tables, names)._replace(over=jnp.int8(1), winner=jnp.int8(2))
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(out, s, strict=True))


@pytest.mark.slow
def test_step_jits_and_vmaps_like_eager(tables, names):
    s = roark_state(tables, names)
    acts = jnp.array([0, 0], jnp.int32)
    o = empty_outcome()
    eager = jit_step(tables, s, acts, o)
    jitted = jax.jit(step)(tables, s, acts, o)
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(eager, jitted, strict=True))
    assert all(a.dtype == b.dtype for a, b in zip(eager, s, strict=True))
    batch = jax.tree.map(lambda x: jnp.stack([x, x]), s)
    batched = jax.vmap(step, in_axes=(None, 0, None, None))(tables, batch, acts, o)
    assert batched.hp.shape == (2, 2, 6) and bool(jnp.array_equal(batched.hp[1], eager.hp))


def test_a_fainted_battler_loses_its_stages_and_volatiles(tables, names):
    # BattleSystem_CleanupFaintedMon (battle_lib.c:2156, from PlayFaintAnimation,
    # battle_script.c:1803) clears the fainted battler's stat stages, statusVolatile,
    # moveEffectsMask and moveEffectsData when it faints, not when it is replaced.
    s = roark_state(tables, names).replace_hp(1, 0, 1).with_stages(1, atk=-2, spe=1)
    s = s._replace(
        confusion=s.confusion.at[1].set(3),
        taunt=s.taunt.at[1].set(3),
        leech_seed=s.leech_seed.at[1].set(1),
        water_sport=s.water_sport.at[1].set(1),
    )
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())  # Scratch faints Geodude
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert out.stages[1].tolist() == [0] * 8
    assert [int(out.confusion[1]), int(out.taunt[1]), int(out.leech_seed[1])] == [0, 0, 0]
    assert int(out.water_sport[1]) == 0


def test_fainted_pokemon_has_no_status(tables, names):
    # battle_controller.c:1482-1486: a Pokemon with 0 HP keeps no status condition.
    s = roark_state(tables, names).replace_hp(1, 0, 1).with_status(1, 0, Status.BURN)
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())  # Scratch faints Geodude
    assert int(out.hp[1, 0]) == 0
    assert int(out.status[1, 0]) == 0 and int(out.status_counter[1, 0]) == 0
