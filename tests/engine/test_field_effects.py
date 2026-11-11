"""Stealth Rock, Bind, Pursuit, Taunt, Leech Seed, Water Sport and confusion
(decomp citations in ``effects.py`` and ``step.py``)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.damage import damage_table
from nuzline.engine.ids import Status
from nuzline.engine.outcome import empty_outcome, sample_outcome
from nuzline.engine.step import (
    END_OF_TURN_ORDER,
    PURSUIT_POWER_MUL,
    replacement_power_mul,
    step,
)
from tests.engine.helpers import A, roark_state


@pytest.mark.slow
def test_stealth_rock_sets_and_damages_on_switch(tables, names):
    s = roark_state(tables, names)  # Geodude leads with Stealth Rock
    out = step(tables, s, A("Scratch", "Stealth Rock", s, names), empty_outcome())
    assert int(out.stealth_rock[0]) == 1
    # Without its Oran Berry, which Geodude's Tackle would otherwise trigger (hooks.py).
    out = out._replace(item=out.item.at[0, 3].set(0))
    out2 = step(tables, out, A(4 + 3, "Tackle", out, names), empty_outcome())  # Starly (Flying: 2x)
    assert int(out2.hp[0, 3]) <= int(out.max_hp[0, 3]) - int(out.max_hp[0, 3]) // 4


@pytest.mark.slow
def test_stealth_rock_damage_is_max_hp_over_8_scaled_by_rock_effectiveness(tables, names):
    # BtlCmd_CheckStealthRock (battle_script.c:8302): maxHP / 8 at neutral, / 4 at 2x.
    s = roark_state(tables, names)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    starly = step(tables, s, A(4 + 3, "Leer", s, names), empty_outcome())  # Normal/Flying: 2x
    assert int(starly.max_hp[0, 3]) - int(starly.hp[0, 3]) == int(s.max_hp[0, 3]) // 4
    prinplup = step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())  # Water: neutral
    assert int(prinplup.max_hp[0, 1]) - int(prinplup.hp[0, 1]) == int(s.max_hp[0, 1]) // 8


def test_stealth_rock_can_faint_incoming_and_needs_a_replacement(tables, names):
    s = roark_state(tables, names).replace_hp(0, 3, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    out = step(tables, s, A(4 + 3, "Tackle", s, names), empty_outcome())
    assert int(out.hp[0, 3]) == 0 and int(out.over) == 0
    assert int(out.deaths[3]) == 1 and int(out.player_faints) == 1


@pytest.mark.slow
def test_bind_traps_and_ticks(tables, names):
    s = roark_state(tables, names, opp_species="Onix", opp_moves=["Bind"])
    o = empty_outcome()._replace(duration=jnp.array([0, 4], jnp.int8))
    out = step(tables, s, A("Scratch", "Bind", s, names), o)
    assert int(out.trapped[0]) > 0
    # Onix with Bind first has no Leer left (four slots: Bind, Tackle, Rock Throw, Stealth
    # Rock), so the brief's "Leer" is Stealth Rock here: no damage except the Bind tick.
    tick = step(tables, out, A("Leer", "Stealth Rock", out, names), empty_outcome())
    assert int(out.hp[0, 0]) - int(tick.hp[0, 0]) == int(s.max_hp[0, 0]) // 16
    assert int(tick.trapped[0]) == int(out.trapped[0]) - 1


@pytest.mark.slow
def test_bind_counter_is_the_rolled_duration_and_last_decrement_deals_no_damage(tables, names):
    # subscript_bind_start.s stores the roll; the end-of-turn check
    # (battle_controller_player.c:1462) decrements first and only damages while it stays > 0.
    s = roark_state(tables, names, opp_species="Onix", opp_moves=["Bind"])
    o = empty_outcome()._replace(duration=jnp.array([0, 3], jnp.int8))
    out = step(tables, s, A("Leer", "Bind", s, names), o)
    assert int(out.trapped[0]) == 2 and int(out.trap_move[0]) == names.move("Bind")
    last = out._replace(trapped=out.trapped.at[0].set(1))
    end = step(tables, last, A("Leer", "Stealth Rock", last, names), empty_outcome())
    assert int(end.trapped[0]) == 0 and int(end.hp[0, 0]) == int(last.hp[0, 0])


def test_bind_ends_when_the_binder_switches_out(tables, names):
    # BattleSystem_UpdateAfterSwitch (battle_lib.c:2094) clears Bind sourced from it.
    s = roark_state(tables, names)
    s = s._replace(
        trapped=s.trapped.at[0].set(4), trap_move=s.trap_move.at[0].set(names.move("Bind"))
    )
    out = step(tables, s, A("Leer", 4 + 1, s, names), empty_outcome())
    assert int(out.trapped[0]) == 0 and int(out.hp[0, 0]) == int(s.hp[0, 0])


@pytest.mark.slow
def test_pursuit_hits_switching_target_double(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Pursuit"])
    stay = step(tables, s, A("Leer", "Pursuit", s, names), empty_outcome())
    go = step(tables, s, A(4 + 1, "Pursuit", s, names), empty_outcome())
    assert (int(s.hp[0, 0]) - int(go.hp[0, 0])) > (int(s.hp[0, 0]) - int(stay.hp[0, 0]))
    assert int(go.active[0]) == 1


def test_replacement_power_mul_is_pursuits_only_after_a_pursued_switch(tables, names):
    # The opponent's forced replacement after the turn sees powerMul = 20 only when the
    # player's switch met Pursuit (BtlCmd_TryPursuit, battle_script.c:6884), else 10.
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Pursuit", "Leer"])
    go = A(4 + 1, "Pursuit", s, names)
    assert int(replacement_power_mul(tables, s, go)) == PURSUIT_POWER_MUL
    assert int(replacement_power_mul(tables, s, A("Leer", "Pursuit", s, names))) == 10
    assert int(replacement_power_mul(tables, s, A(4 + 1, "Leer", s, names))) == 10


def test_pursuit_on_a_switch_ignores_accuracy_spends_pp_once_and_ends_the_turn(tables, names):
    # subscript_pursuit.s has no accuracy check; BtlCmd_TryPursuit (battle_script.c:6884)
    # spends the PP and ends the pursuer's action, so the incoming Pokemon is not hit.
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Pursuit"])
    miss = empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8))
    go = step(tables, s, A(4 + 1, "Pursuit", s, names), miss)
    assert int(go.hp[0, 0]) < int(s.hp[0, 0])
    assert int(go.hp[0, 1]) == int(s.hp[0, 1])
    assert int(go.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1


@pytest.mark.slow
def test_pursuit_with_its_last_pp_still_pursues(tables, names):
    # The move is fixed when chosen: spending Pursuit's last PP in BtlCmd_TryPursuit
    # does not turn the hit into Struggle (no recoil, same damage).
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Pursuit"])
    last = s.with_pp(1, 0, [1, 10, 10, 10])
    go = step(tables, s, A(4 + 1, "Pursuit", s, names), empty_outcome())
    go_last = step(tables, last, A(4 + 1, "Pursuit", last, names), empty_outcome())
    assert int(go_last.hp[0, 0]) == int(go.hp[0, 0])
    assert int(go_last.hp[1, 0]) == int(s.hp[1, 0])
    assert int(go_last.pp[1, 0, 0]) == 0


def test_taunt_sets_counter(tables, names):
    s = roark_state(tables, names, player_moves=["Taunt"])
    o = empty_outcome()._replace(duration=jnp.array([3, 0], jnp.int8))
    out = step(tables, s, A("Taunt", "Tackle", s, names), o)
    assert int(out.taunt[1]) > 0


@pytest.mark.slow
def test_taunt_counts_down_at_end_of_turn_and_blocks_status_moves(tables, names):
    # subscript_taunt_start.s stores the roll; battle_controller_player.c:1635 decrements
    # it at the end of the turn; :2557 fails a power-0 move before its PP is spent.
    s = roark_state(tables, names, player_moves=["Taunt"])
    o = empty_outcome()._replace(duration=jnp.array([3, 0], jnp.int8))
    taunted = s._replace(taunt=s.taunt.at[1].set(3))
    out = step(tables, taunted, A("Scratch", "Leer", taunted, names), o)
    assert int(out.taunt[1]) == 2
    assert int(out.stages[0, 2]) == 0  # Leer failed
    assert int(out.pp[1, 0, 3]) == int(s.pp[1, 0, 3])  # and spent no PP
    hit = step(tables, out, A("Leer", "Tackle", out, names), empty_outcome())
    assert int(hit.hp[0, 0]) < int(out.hp[0, 0])  # damaging moves still work


@pytest.mark.slow
def test_leech_seed_drains_and_grass_immune(tables, names):
    s = roark_state(tables, names, player_species="Budew", player_moves=["Leech Seed"]).replace_hp(
        0, 0, 5
    )
    out = step(tables, s, A("Leech Seed", "Leer", s, names), empty_outcome())
    assert int(out.leech_seed[1]) == 1 and int(out.hp[0, 0]) > 5
    g = roark_state(
        tables, names, player_species="Budew", player_moves=["Leech Seed"], opp_species="Budew"
    )
    # The brief's opponent action was "Absorb", but an opp_species lead without opp_moves
    # gets the fixture moveset (plan ruling), which has no Absorb; Leer is used instead.
    assert (
        int(step(tables, g, A("Leech Seed", "Leer", g, names), empty_outcome()).leech_seed[1]) == 0
    )


def test_leech_seed_drains_an_eighth_and_heals_by_the_hp_lost(tables, names):
    # subscript_leech_seed_effect.s: maxHP / 8 off the seeded mon, the HP it lost to the other.
    s = roark_state(tables, names, player_species="Budew", player_moves=["Leech Seed"]).replace_hp(
        0, 0, 5
    )
    out = step(tables, s, A("Leech Seed", "Leer", s, names), empty_outcome())
    drained = int(s.max_hp[1, 0]) // 8
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == drained
    assert int(out.hp[0, 0]) == 5 + drained


def test_water_sport_sets_flag_and_halves_fire(tables, names):
    # effect_script_0210.s sets the user's flag; battle_lib.c:6815 halves Fire power.
    s = roark_state(tables, names, player_species="Prinplup", opp_species="Cranidos")
    out = step(tables, s, A("Water Sport", "Leer", s, names), empty_outcome())
    assert int(out.water_sport[0]) == 1
    ember = names.move("Ember")
    flat = out._replace(water_sport=jnp.zeros_like(out.water_sport))
    halved = damage_table(tables, out, 0, ember, False)
    full = damage_table(tables, flat, 0, ember, False)
    assert int(halved[-1]) < int(full[-1])


@pytest.mark.slow
def test_confusion_self_hit_replaces_move(tables, names):
    s = roark_state(tables, names, player_species="Zubat", player_moves=["Supersonic"])
    # Plan ruling: Outcome.duration is indexed by the acting (inflicting) side, so the
    # player's Supersonic reads duration[0] (the brief had [0, 3]).
    o = empty_outcome()._replace(duration=jnp.array([3, 0], jnp.int8))
    out = step(tables, s, A("Supersonic", "Leer", s, names), o)
    assert int(out.confusion[1]) > 0
    hit_self = empty_outcome()._replace(confusion_hit=jnp.array([0, 1], jnp.int8))
    out2 = step(tables, out, A("Supersonic", "Tackle", out, names), hit_self)
    assert int(out2.hp[1, 0]) < int(out.hp[1, 0]) and int(out2.hp[0, 0]) == int(out.hp[0, 0])


def test_confusion_counter_decrements_before_the_roll_and_snaps_out_at_zero(tables, names):
    # battle_controller_player.c:2614: decrement; at 0 the battler snaps out and moves.
    s = roark_state(tables, names)
    s = s._replace(confusion=s.confusion.at[1].set(1))
    hit_self = empty_outcome()._replace(confusion_hit=jnp.array([0, 1], jnp.int8))
    out = step(tables, s, A("Leer", "Tackle", s, names), hit_self)
    assert int(out.confusion[1]) == 0
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])


@pytest.mark.slow
def test_confusion_self_hit_is_40_power_typeless_physical_with_the_roll(tables, names):
    # CALC_SELF_HIT(MOVE_STRUGGLE, 40) (battle_controller_player.c:2633, common.h:35):
    # own Atk against own Def, no STAB or type chart, then the damage variance.
    s = roark_state(tables, names)
    s = s._replace(confusion=s.confusion.at[1].set(3))
    hit_self = empty_outcome()._replace(confusion_hit=jnp.array([0, 1], jnp.int8))
    # The player switches so that nothing changes Geodude's stages before it moves.
    out = step(tables, s, A(4 + 1, "Tackle", s, names), hit_self)
    atk, def_ = int(s.stats[1, 0, 0]), int(s.stats[1, 0, 1])
    level = int(s.level[1, 0])
    base = atk * 40 * (level * 2 // 5 + 2) // def_ // 50 + 2
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == base  # roll 15: x100/100
    low = step(
        tables,
        s,
        A(4 + 1, "Tackle", s, names),
        hit_self._replace(roll=jnp.zeros((2, 5), jnp.int8)),
    )
    assert int(s.hp[1, 0]) - int(low.hp[1, 0]) == max(base * 85 // 100, 1)


def test_end_of_turn_order_lists_the_implemented_checks_in_decomp_order():
    # enum MonCondCheckState (battle_controller_player.c:1259-1284): AQUA_RING, then ABILITY,
    # USE_ITEM before LEECH_SEED, DISABLE between BIND and TAUNT, ITEM_CONDITION after TAUNT.
    assert [name for name, _ in END_OF_TURN_ORDER] == [
        "aqua_ring",
        "ability",
        "use_item",
        "leech_seed",
        "poison",
        "toxic",
        "burn",
        "bind",
        "disable",
        "taunt",
        "item_condition",
    ]


@pytest.mark.slow
def test_end_of_turn_runs_the_faster_battler_first(tables, names):
    # Both seeded at 1 HP: the faster battler's seed drains it first and the battle ends
    # there; the slower one's check never runs (battle_controller_player.c:1289).
    s = roark_state(tables, names, player_count=1)  # Monferno outspeeds Geodude
    s = s._replace(leech_seed=jnp.ones_like(s.leech_seed)).replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    s = s._replace(species=s.species.at[1, 1:].set(0), hp=s.hp.at[1, 1:].set(0))
    out = step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 1 and int(out.hp[1, 0]) == 2  # healed 1
    slow = s._replace(stats=s.stats.at[0, 0, 4].set(1))
    out = step(tables, slow, A("Leer", "Leer", slow, names), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2 and int(out.hp[0, 0]) == 2


@pytest.mark.slow
def test_end_of_turn_speed_tie_uses_its_own_outcome(tables, names):
    s = roark_state(tables, names, equal_speed=True, player_count=1)
    s = s._replace(leech_seed=jnp.ones_like(s.leech_seed)).replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    s = s._replace(species=s.species.at[1, 1:].set(0), hp=s.hp.at[1, 1:].set(0))
    first = step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    second = step(
        tables, s, A("Leer", "Leer", s, names), empty_outcome()._replace(end_speed_tie=jnp.int8(1))
    )
    assert int(first.winner) == 1 and int(second.winner) == 2


@pytest.mark.slow
def test_sampled_durations_follow_each_effects_random_command(tables, names):
    # Random 3, 2 (confusion, sleep): 2-5; Random 3, 3 (Bind): 3-6; Random 2, 3 (Taunt): 3-5.
    s = roark_state(tables, names, player_moves=["Taunt"], opp_species="Onix", opp_moves=["Bind"])
    keys = jax.random.split(jax.random.key(3), 4000)
    sample = jax.vmap(sample_outcome, in_axes=(0, None, None, None))
    o = sample(keys, tables, s, A("Taunt", "Bind", s, names))
    assert sorted(np.unique(np.asarray(o.duration[:, 0])).tolist()) == [3, 4, 5]
    assert sorted(np.unique(np.asarray(o.duration[:, 1])).tolist()) == [3, 4, 5, 6]
    assert abs(float(np.mean(np.asarray(o.end_speed_tie))) - 0.5) < 0.05


def test_stealth_rock_lands_after_the_foe_fainted_from_its_own_recoil(tables, names):
    # RANGE_OPPONENT_SIDE: BattleSystem_Defender (battle_lib.c:1663) takes the foe's
    # battler slot whatever its HP (BattleSystem_RandomOpponent, :4194), so the
    # no-target check (BattleControllerPlayer_HasNoTarget, battle_controller_player.c:
    # 2306) passes and Stealth Rock is laid although Cranidos fainted to its own recoil.
    s = roark_state(
        tables,
        names,
        player_moves=["Stealth Rock"],
        opp_species="Cranidos",
        opp_moves=["Take Down"],
        opp_faster=True,
    ).replace_hp(1, 0, 1)
    out = step(tables, s, A("Stealth Rock", "Take Down", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert int(out.stealth_rock[1]) == 1


def test_natural_cure_comes_before_a_pursuit_that_faints_the_switcher(tables, names):
    # subscript_switch_pokemon.s:5-6 restores the status (BtlCmd_TryRestoreStatusOnSwitch,
    # battle_script.c:9139) before the Pursuit hit at :11, so Budew faints unparalyzed.
    s = roark_state(
        tables, names, player_species="Budew", opp_species="Cranidos", opp_moves=["Pursuit"]
    )
    s = s.replace_hp(0, 0, 1).with_status(0, 0, Status.PARALYSIS)
    out = step(tables, s, A(4 + 1, "Pursuit", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == 0
    assert int(out.status[0, 0]) == Status.NONE
