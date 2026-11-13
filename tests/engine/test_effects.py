"""Move effects: stat stages, status moves and secondaries (with their decomp
immunities), flinch, drain, recoil and multi-hit. Decomp citations are in
``src/nuzline/engine/effects.py``; test-specific ones are inline.

Deviations from the plan's test text (plan rulings and the decomp win):

* Opponents know the fixture moveset (Tackle, Rock Throw, Stealth Rock, Leer)
  unless ``opp_moves`` says otherwise, so the Headbutt tests pass
  ``opp_moves=["Headbutt"]``.
* Cranidos has Mold Breaker, which ignores Inner Focus (``subscript_flinch_mon.s``
  checks it with ``CheckIgnorableAbility``), so the Inner Focus test uses the
  default Geodude lead with Headbutt; a separate test covers Mold Breaker.
* Simple doubles the effective stage at calculation time, not the stored change
  (``BtlCmd_ChangeStatStage`` has no Simple branch; ``battle_lib.c:6850-6884``).
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.damage import damage_table
from nuzline.engine.effects import can_have_status
from nuzline.engine.ids import AbilityId, Stat, Status, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import step
from tests.engine.helpers import A, roark_state

PROC = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))


def dealt(before, after, side=1):
    return int(before.hp[side, 0]) - int(after.hp[side, 0])


# --- stat stages -------------------------------------------------------------


@pytest.mark.slow
def test_leer_lowers_defense_and_clamps(tables, names):
    s = roark_state(tables, names).with_stages(1, def_=-6)
    out = step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(out.stages[1, 2]) == -6
    fresh = roark_state(tables, names)
    assert (
        int(step(tables, fresh, A("Leer", "Tackle", fresh, names), empty_outcome()).stages[1, 2])
        == -1
    )


def test_rock_polish_raises_speed_two(tables, names):
    s = roark_state(tables, names, opp_moves=["Rock Polish"])
    out = step(tables, s, A("Scratch", "Rock Polish", s, names), empty_outcome())
    assert int(out.stages[1, 5]) == 2


def test_stat_raise_clamps_at_six(tables, names):
    s = roark_state(tables, names, opp_moves=["Rock Polish"]).with_stages(1, spe=5)
    out = step(tables, s, A("Scratch", "Rock Polish", s, names), empty_outcome())
    assert int(out.stages[1, 5]) == 6


def test_self_targeting_move_works_with_the_foe_active_fainted(tables, names):
    # RANGE_USER: the decomp's defender is the user itself, so the no-target check
    # (battle_controller_player.c:3190) cannot fail it.
    s = roark_state(tables, names, opp_moves=["Rock Polish"]).replace_hp(0, 0, 0)
    out = step(tables, s, A("Scratch", "Rock Polish", s, names), empty_outcome())
    assert int(out.stages[1, 5]) == 2


def test_self_boost_after_foe_faints_from_its_own_recoil(tables, names):
    # Cranidos (1 HP, no Rock Head) moves first and Take Down's recoil faints it;
    # Grotle's Withdraw still raises its Defense.
    s = roark_state(
        tables,
        names,
        player_species="Grotle",
        opp_species="Cranidos",
        opp_moves=["Take Down"],
        opp_faster=True,
    ).replace_hp(1, 0, 1)
    out = step(tables, s, A("Withdraw", "Take Down", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0
    assert int(out.over) == 0
    assert int(out.stages[0, 2]) == 1


@pytest.mark.slow
def test_simple_doubles_stage_change(tables, names):
    # Gen 4 Simple doubles the stage when a stat is used, not the stored change.
    s = roark_state(
        tables,
        names,
        player_species="Bidoof",
        player_ability="Simple",
        player_moves=["Defense Curl"],
    )
    out = step(tables, s, A("Defense Curl", "Leer", s, names), empty_outcome())
    assert int(out.stages[0, 2]) == 0  # +1 from Defense Curl, -1 from Leer
    curl = step(tables, s, A("Defense Curl", "Stealth Rock", s, names), empty_outcome())
    assert int(curl.stages[0, 2]) == 1
    plain = curl.with_stages(0, def_=2)._replace(ability=curl.ability.at[0, 0].set(AbilityId.NONE))
    tackle = names.move("Tackle")
    assert (
        damage_table(tables, curl, 1, tackle, False).tolist()
        == damage_table(tables, plain, 1, tackle, False).tolist()
    )


@pytest.mark.slow
def test_keen_eye_blocks_accuracy_drop(tables, names):
    s = roark_state(tables, names, player_species="Starly", opp_moves=["Sand-Attack"])
    out = step(tables, s, A("Growl", "Sand-Attack", s, names), empty_outcome())
    assert int(out.stages[0, Stat.ACC]) == 0
    # Keen Eye is an ignorable ability (AbilityBlocksSpecificStatReduction): Mold Breaker pierces it.
    mb = roark_state(
        tables, names, player_species="Starly", opp_species="Cranidos", opp_moves=["Sand-Attack"]
    )
    out = step(tables, mb, A("Growl", "Sand-Attack", mb, names), empty_outcome())
    assert int(out.stages[0, Stat.ACC]) == -1


def test_certain_secondary_self_drop_needs_no_proc(tables, names):
    # Hammer Arm: MOVE_SIDE_EFFECT_ON_HIT (effect_script_0218.s), chance 0, applies on every hit.
    s = roark_state(tables, names, player_moves=["Hammer Arm"])
    out = step(tables, s, A("Hammer Arm", "Leer", s, names), empty_outcome())
    assert int(out.stages[0, Stat.SPE]) == -1


@pytest.mark.slow
def test_secondary_stat_effects_only_on_proc(tables, names):
    s = roark_state(
        tables, names, player_species="Prinplup", opp_species="Cranidos"
    )  # survives Bubble
    no = step(tables, s, A("Bubble", "Leer", s, names), empty_outcome())
    yes = step(tables, s, A("Bubble", "Leer", s, names), PROC)
    assert int(no.stages[1, Stat.SPE]) == 0 and int(yes.stages[1, Stat.SPE]) == -1
    mc = roark_state(tables, names, player_moves=["Metal Claw"])
    assert int(step(tables, mc, A("Metal Claw", "Leer", mc, names), PROC).stages[0, Stat.ATK]) == 1


def test_no_secondary_on_fainted_target(tables, names):
    s = roark_state(tables, names, player_species="Prinplup").replace_hp(1, 0, 1)
    out = step(tables, s, A("Bubble", "Leer", s, names), PROC)
    assert int(out.hp[1, 0]) == 0 and int(out.stages[1, Stat.SPE]) == 0


# --- status ------------------------------------------------------------------


@pytest.mark.slow
def test_ember_burn_only_on_secondary_proc(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos")
    no = step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    yes = step(tables, s, A("Ember", "Leer", s, names), PROC)
    assert int(no.status[1, 0]) == Status.NONE and int(yes.status[1, 0]) == Status.BURN


@pytest.mark.slow
def test_burn_halves_physical_damage(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Headbutt"])
    burned = s.with_status(1, 0, Status.BURN)
    a, b = (
        step(tables, s, A("Scratch", "Headbutt", s, names), empty_outcome()),
        step(tables, burned, A("Scratch", "Headbutt", s, names), empty_outcome()),
    )
    assert (int(s.hp[0, 0]) - int(b.hp[0, 0])) < (int(s.hp[0, 0]) - int(a.hp[0, 0]))


def test_fire_types_cannot_be_burned(tables, names):
    # subscript_burn.s:40-41: BATTLEMON_TYPE_1/2 == TYPE_FIRE fails.
    s = roark_state(tables, names, opp_species="Chimchar")
    assert int(step(tables, s, A("Ember", "Leer", s, names), PROC).status[1, 0]) == Status.NONE


@pytest.mark.slow
def test_ice_types_cannot_be_frozen_nor_anyone_in_sun(tables, names):
    # subscript_freeze.s:21-22 (Ice), :7 (sun fails unless Cloud Nine / Air Lock).
    s = roark_state(tables, names, player_moves=["Powder Snow"], opp_species="Cranidos")
    assert (
        int(step(tables, s, A("Powder Snow", "Leer", s, names), PROC).status[1, 0]) == Status.FREEZE
    )
    ice = roark_state(tables, names, player_moves=["Powder Snow"], opp_species="Snorunt")
    assert (
        int(step(tables, ice, A("Powder Snow", "Leer", ice, names), PROC).status[1, 0])
        == Status.NONE
    )
    sun = s._replace(weather=jnp.int8(Weather.SUN))
    assert (
        int(step(tables, sun, A("Powder Snow", "Leer", sun, names), PROC).status[1, 0])
        == Status.NONE
    )


@pytest.mark.slow
def test_poison_and_steel_types_cannot_be_poisoned(tables, names):
    # subscript_poison.s:34-37: TYPE_POISON or TYPE_STEEL fails.
    for opp, expected in (
        ("Geodude", Status.POISON),
        ("Zubat", Status.NONE),
        ("Bronzor", Status.NONE),
    ):
        s = roark_state(
            tables, names, player_species="Budew", player_moves=["PoisonPowder"], opp_species=opp
        )
        out = step(tables, s, A("PoisonPowder", "Leer", s, names), empty_outcome())
        assert int(out.status[1, 0]) == expected, opp


def test_toxic_immunities(tables, names):
    # subscript_badly_poison.s:20-26 (indirect) and 47-53 (direct): an existing
    # status, a Poison type or a Steel type fails Toxic.
    def ok(s):
        return bool(can_have_status(tables, s, 1, 0, Status.TOXIC))

    assert not ok(roark_state(tables, names, opp_species="Zubat"))
    assert not ok(roark_state(tables, names, opp_species="Bronzor"))
    geodude = roark_state(tables, names)
    assert ok(geodude)
    assert not ok(geodude.with_status(1, 0, Status.BURN))


@pytest.mark.slow
def test_stun_spore_paralyzes_and_never_overwrites(tables, names):
    s = roark_state(tables, names, player_species="Budew")
    assert (
        int(step(tables, s, A("Stun Spore", "Leer", s, names), empty_outcome()).status[1, 0])
        == Status.PARALYSIS
    )
    burned = s.with_status(1, 0, Status.BURN)
    assert (
        int(
            step(tables, burned, A("Stun Spore", "Leer", burned, names), empty_outcome()).status[
                1, 0
            ]
        )
        == Status.BURN
    )


def test_status_move_misses_with_the_accuracy_roll(tables, names):
    s = roark_state(tables, names, player_species="Budew")
    miss = empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8))
    assert int(step(tables, s, A("Stun Spore", "Leer", s, names), miss).status[1, 0]) == Status.NONE


def test_sleep_counter_comes_from_duration(tables, names):
    # subscript_fall_asleep.s:59-60: Random 3, 2 turns stored with the status. The
    # opponent switches (Onix comes in and is put to sleep), so no move attempt of the
    # sleeper lowers the counter this turn (step's sleep check, Task 8d).
    s = roark_state(tables, names, player_species="Budew", player_moves=["Sleep Powder"])
    out = step(
        tables,
        s,
        A("Sleep Powder", 4 + 1, s, names),
        empty_outcome()._replace(duration=jnp.array([3, 0], jnp.int8)),
    )
    assert int(out.status[1, 1]) == Status.SLEEP and int(out.status_counter[1, 1]) == 3


@pytest.mark.slow
def test_thunder_wave_respects_the_type_chart(tables, names):
    # BattleControllerPlayer_CheckTypeChart runs Thunder Wave through the chart: Ground is immune.
    s = roark_state(tables, names, player_moves=["Thunder Wave"])
    assert (
        int(step(tables, s, A("Thunder Wave", "Leer", s, names), empty_outcome()).status[1, 0])
        == Status.NONE
    )
    rock = roark_state(tables, names, player_moves=["Thunder Wave"], opp_species="Cranidos")
    assert (
        int(
            step(tables, rock, A("Thunder Wave", "Leer", rock, names), empty_outcome()).status[1, 0]
        )
        == Status.PARALYSIS
    )


# --- flinch ------------------------------------------------------------------


def test_headbutt_flinch_blocks_slower_target(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Headbutt"], opp_faster=True)
    out = step(tables, s, A("Scratch", "Headbutt", s, names), PROC)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])  # Monferno flinched
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0])  # and spent no PP
    assert int(out.flinched[0]) == 0  # flinch lasts only for the turn


def test_inner_focus_ignores_flinch(tables, names):
    s = roark_state(tables, names, player_species="Zubat", opp_moves=["Headbutt"], opp_faster=True)
    out = step(tables, s, A("Bite", "Headbutt", s, names), PROC)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])


def test_mold_breaker_pierces_inner_focus(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Zubat",
        opp_species="Cranidos",
        opp_moves=["Headbutt"],
        opp_faster=True,
    )
    out = step(tables, s, A("Bite", "Headbutt", s, names), PROC)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_steadfast_raises_speed_on_flinch(tables, names):
    # subscript_flinched.s:10-14: Steadfast raises the flinched Pokemon's Speed by 1.
    s = roark_state(
        tables,
        names,
        player_species="Riolu",
        player_ability="Steadfast",
        opp_moves=["Headbutt"],
        opp_faster=True,
    )
    out = step(tables, s, A(0, "Headbutt", s, names), PROC)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) and int(out.stages[0, Stat.SPE]) == 1


def test_no_flinch_without_proc(tables, names):
    s = roark_state(tables, names, opp_moves=["Headbutt"], opp_faster=True)
    out = step(tables, s, A("Scratch", "Headbutt", s, names), empty_outcome())
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])


# --- drain and recoil ----------------------------------------------------------


def _no_exp(s):
    """No participant: a KO pays no exp, so no level-up changes the HP these tests read
    (K3d-exp: the payout at MoveEnd, after the drain). The game never reaches this state (its
    active facing a living foe is always flagged, design §1.2); these tests read only the
    drain, which comes before the payout (bcp.c:3965-3969), so it is the drain alone that is
    checked, and the payout's own HP rule is tested in test_levelup.py."""
    return s._replace(exp_mask=s.exp_mask * 0)


def test_absorb_heals_half_damage_dealt(tables, names):
    s = _no_exp(roark_state(tables, names, player_species="Budew").replace_hp(0, 0, 5))
    out = step(tables, s, A("Absorb", "Leer", s, names), empty_outcome())
    hit = dealt(s, out)
    assert hit > 0 and int(out.hp[0, 0]) == min(int(s.max_hp[0, 0]), 5 + max(1, hit // 2))


def test_drain_uses_damage_capped_at_target_hp(tables, names):
    # BtlCmd_UpdateHealthBarValue caps hitDamage at the target's HP (battle_script.c:1701).
    s = roark_state(tables, names, player_species="Budew").replace_hp(0, 0, 5).replace_hp(1, 0, 3)
    s = _no_exp(s)
    out = step(tables, s, A("Absorb", "Leer", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) == 5 + 1  # 3 // 2


def test_take_down_recoil_is_quarter_of_damage_dealt(tables, names):
    s = roark_state(tables, names, player_moves=["Take Down"])
    out = step(tables, s, A("Take Down", "Leer", s, names), empty_outcome())
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(1, dealt(s, out) // 4)


def test_rock_head_prevents_recoil(tables, names):
    s = roark_state(tables, names, player_moves=["Take Down"])
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.ROCK_HEAD))
    out = step(tables, s, A("Take Down", "Leer", s, names), empty_outcome())
    assert dealt(s, out) > 0 and int(out.hp[0, 0]) == int(s.hp[0, 0])


# --- multi-hit -----------------------------------------------------------------


@pytest.mark.slow
def test_fury_swipes_hits_n_times(tables, names):
    s = roark_state(tables, names, player_moves=["Fury Swipes"])
    two = step(
        tables,
        s,
        A("Fury Swipes", "Leer", s, names),
        empty_outcome()._replace(n_hits=jnp.array([2, 2], jnp.int8)),
    )
    five = step(
        tables,
        s,
        A("Fury Swipes", "Leer", s, names),
        empty_outcome()._replace(n_hits=jnp.array([5, 5], jnp.int8)),
    )
    assert (int(s.hp[1, 0]) - int(five.hp[1, 0])) > (int(s.hp[1, 0]) - int(two.hp[1, 0]))


@pytest.mark.slow
def test_multi_hit_checks_accuracy_once(tables, names):
    # SYSCTL_MULTI_HIT_MOVE sets SYSCTL_SKIP_ACCURACY_CHECK for hits after the first.
    s = roark_state(tables, names, player_moves=["Fury Swipes"])
    three = empty_outcome()._replace(n_hits=jnp.array([3, 2], jnp.int8))
    first_only = three._replace(hit=jnp.array([[1, 0, 0, 0, 0], [1] * 5], jnp.int8))
    a = step(tables, s, A("Fury Swipes", "Leer", s, names), three)
    b = step(tables, s, A("Fury Swipes", "Leer", s, names), first_only)
    assert dealt(s, a) == dealt(s, b) > 0
    miss = three._replace(hit=jnp.array([[0, 1, 1, 1, 1], [1] * 5], jnp.int8))
    assert dealt(s, step(tables, s, A("Fury Swipes", "Leer", s, names), miss)) == 0


@pytest.mark.slow
def test_multi_hit_rolls_crit_per_hit(tables, names):
    s = roark_state(tables, names, player_moves=["Fury Swipes"])
    base = empty_outcome()._replace(n_hits=jnp.array([3, 2], jnp.int8))
    crit2 = base._replace(crit=jnp.array([[0, 1, 0, 0, 0], [0] * 5], jnp.int8))
    crit4 = base._replace(crit=jnp.array([[0, 0, 0, 1, 0], [0] * 5], jnp.int8))  # beyond the 3 hits
    plain = dealt(s, step(tables, s, A("Fury Swipes", "Leer", s, names), base))
    assert dealt(s, step(tables, s, A("Fury Swipes", "Leer", s, names), crit2)) > plain
    assert dealt(s, step(tables, s, A("Fury Swipes", "Leer", s, names), crit4)) == plain
    one = damage_table(tables, s, 0, names.move("Fury Swipes"), False)[15]
    assert plain == 3 * int(one)


def test_multi_hit_stops_when_target_faints(tables, names):
    s = roark_state(tables, names, player_moves=["Fury Swipes"]).replace_hp(1, 0, 1)
    out = step(
        tables,
        s,
        A("Fury Swipes", "Leer", s, names),
        empty_outcome()._replace(n_hits=jnp.array([5, 2], jnp.int8)),
    )
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
