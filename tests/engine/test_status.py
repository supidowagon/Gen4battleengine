"""Major status: sleep, paralysis, burn, poison, toxic, freeze, the contact abilities
(Static, Poison Point), Natural Cure, and their end-of-turn damage. Decomp citations
are in ``src/nuzline/engine/step.py`` and ``effects.py``; test-specific ones are inline.

Deviation from the plan's test text: Cranidos as an ``opp_species`` lead without
``opp_moves`` knows the fixture moveset (plan ruling), which has no Headbutt, so the
paralysis test's second turn uses Tackle; the player uses Leer on that turn because
a second Spark knocks Cranidos out before it moves, which made the check vacuous.
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.ids import Stat, Status
from nuzline.engine.order import effective_speed
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from tests.engine.helpers import A, jit_step, roark_state

NO_PROC = jnp.zeros((2, MAX_HITS), jnp.int8)


@pytest.mark.slow
def test_paralysis_quarters_speed_and_full_para_skips(tables, names):
    s = roark_state(
        tables, names, player_species="Shinx", player_moves=["Spark"], opp_species="Cranidos"
    )
    hit = jit_step(
        tables,
        s,
        A("Spark", "Leer", s, names),
        empty_outcome()._replace(secondary=jnp.ones((2, MAX_HITS), jnp.int8)),
    )
    assert int(hit.status[1, 0]) == Status.PARALYSIS
    # Shinx outspeeds the paralyzed Cranidos (Speed / 4), so it moves first.
    assert int(effective_speed(tables, hit, 1)) == int(hit.stats[1, 0, Stat.SPE - 1]) // 4
    skipped = jit_step(
        tables,
        hit,
        A("Leer", "Tackle", hit, names),
        empty_outcome()._replace(full_para=jnp.array([0, 1], jnp.int8)),
    )
    assert int(skipped.hp[0, 0]) == int(hit.hp[0, 0])
    moved = jit_step(tables, hit, A("Leer", "Tackle", hit, names), empty_outcome())
    assert int(moved.hp[0, 0]) < int(hit.hp[0, 0])


@pytest.mark.slow
def test_full_paralysis_spends_no_pp_and_otherwise_the_move_runs(tables, names):
    # CHECK_STATUS_STATE_PARALYSIS (battle_controller_player.c:2652) runs before
    # BattleControllerPlayer_DecrementPP (:3182).
    s = roark_state(tables, names).with_status(1, 0, Status.PARALYSIS)
    para = empty_outcome()._replace(full_para=jnp.array([0, 1], jnp.int8))
    skipped = jit_step(tables, s, A("Leer", "Tackle", s, names), para)
    assert int(skipped.pp[1, 0, 0]) == int(s.pp[1, 0, 0])
    moved = jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert (
        int(moved.hp[0, 0]) < int(s.hp[0, 0]) and int(moved.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1
    )


def test_burn_ticks_one_eighth(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos").with_status(1, 0, Status.BURN)
    out = jit_step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == int(s.max_hp[1, 0]) // 8


def test_poison_ticks_one_eighth(tables, names):
    s = roark_state(tables, names).with_status(1, 0, Status.POISON)
    out = jit_step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == int(s.max_hp[1, 0]) // 8


@pytest.mark.slow
def test_toxic_counter_rises_before_each_tick(tables, names):
    # MON_COND_CHECK_STATE_TOXIC (battle_controller_player.c:1405): the counter is
    # incremented first, then the damage is BattleSystem_Divide(maxHP, 16) * counter.
    s = roark_state(tables, names).with_status(1, 0, Status.TOXIC)
    sixteenth = max(int(s.max_hp[1, 0]) // 16, 1)
    t1 = jit_step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(s.hp[1, 0]) - int(t1.hp[1, 0]) == sixteenth and int(t1.status_counter[1, 0]) == 1
    t2 = jit_step(tables, t1, A("Leer", "Leer", t1, names), empty_outcome())
    assert int(t1.hp[1, 0]) - int(t2.hp[1, 0]) == 2 * sixteenth
    assert int(t2.status_counter[1, 0]) == 2


def test_toxic_counter_caps_at_fifteen(tables, names):
    # The counter is the 4-bit MON_CONDITION_TOXIC_COUNTER: it stops at 15.
    s = roark_state(tables, names).with_status(1, 0, Status.TOXIC, counter=15)
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]))
    out = jit_step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    sixteenth = max(int(s.max_hp[1, 0]) // 16, 1)
    assert int(out.status_counter[1, 0]) == 15
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == min(15 * sixteenth, int(s.hp[1, 0]))


def test_toxic_counter_resets_on_switch_out(tables, names):
    # battle_controller.c:1482 writes the status back to the party without
    # MON_CONDITION_TOXIC_COUNTER, so the counter restarts when the Pokemon returns.
    s = roark_state(tables, names).with_status(0, 0, Status.TOXIC, counter=3)
    out = jit_step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.TOXIC and int(out.status_counter[0, 0]) == 0


@pytest.mark.slow
def test_sleep_counts_down_and_wakes(tables, names):
    s = roark_state(tables, names).with_status(1, 0, Status.SLEEP, counter=2)
    t1 = jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(t1.hp[0, 0]) == int(s.hp[0, 0])
    t2 = jit_step(tables, t1, A("Leer", "Tackle", t1, names), empty_outcome())
    assert int(t2.status[1, 0]) == Status.NONE


@pytest.mark.slow
def test_sleep_spends_no_pp_and_the_wake_up_turn_moves(tables, names):
    # CHECK_STATUS_STATE_SLEEP (battle_controller_player.c:2430): the counter drops
    # first; still asleep disrupts the move, while waking (subscript_wake_up) goes
    # on with it.
    s = roark_state(tables, names).with_status(1, 0, Status.SLEEP, counter=2)
    t1 = jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(t1.status[1, 0]) == Status.SLEEP and int(t1.status_counter[1, 0]) == 1
    assert int(t1.pp[1, 0, 0]) == int(s.pp[1, 0, 0])
    t2 = jit_step(tables, t1, A("Leer", "Tackle", t1, names), empty_outcome())
    assert int(t2.hp[0, 0]) < int(t1.hp[0, 0])
    assert int(t2.status_counter[1, 0]) == 0


def test_sleep_counter_only_drops_on_a_move(tables, names):
    # The sleep check is part of the move's status checks; a switch skips them.
    s = roark_state(tables, names).with_status(0, 0, Status.SLEEP, counter=2)
    out = jit_step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.SLEEP and int(out.status_counter[0, 0]) == 2


@pytest.mark.slow
def test_freeze_thaw_outcome(tables, names):
    s = roark_state(tables, names).with_status(1, 0, Status.FREEZE)
    frozen = jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    thawed = jit_step(
        tables,
        s,
        A("Leer", "Tackle", s, names),
        empty_outcome()._replace(thaw=jnp.array([0, 1], jnp.int8)),
    )
    assert int(frozen.hp[0, 0]) == int(s.hp[0, 0]) and int(thawed.hp[0, 0]) < int(s.hp[0, 0])
    assert int(frozen.status[1, 0]) == Status.FREEZE and int(thawed.status[1, 0]) == Status.NONE
    assert int(frozen.pp[1, 0, 0]) == int(s.pp[1, 0, 0])


@pytest.mark.slow
def test_fire_hit_thaws_a_frozen_target(tables, names):
    # AFTER_MOVE_EFFECT_THAW_DEFENDER (battle_controller_player.c:3710): a Fire move that
    # damages a frozen defender thaws it; other types do not.
    s = roark_state(tables, names, opp_species="Cranidos").with_status(1, 0, Status.FREEZE)
    ember = jit_step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    scratch = jit_step(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert int(ember.status[1, 0]) == Status.NONE
    assert int(scratch.status[1, 0]) == Status.FREEZE


def test_flame_wheel_thaws_its_frozen_user(tables, names):
    # BATTLE_EFFECT_THAW_AND_BURN_HIT skips the frozen disruption (:2481) and
    # CHECK_STATUS_STATE_SELF_THAW (:2728) clears the freeze before the move.
    s = roark_state(tables, names, player_moves=["Flame Wheel"], opp_species="Cranidos")
    s = s.with_status(0, 0, Status.FREEZE)
    out = jit_step(tables, s, A("Flame Wheel", "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.hp[1, 0]) < int(s.hp[1, 0])


@pytest.mark.slow
def test_poison_point_procs_on_contact(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Budew",
        player_ability="Poison Point",
        opp_species="Cranidos",
        opp_moves=["Headbutt"],
    )
    o = empty_outcome()._replace(
        contact_proc=jnp.array([[1] + [0] * (MAX_HITS - 1), [0] * MAX_HITS], jnp.int8)
    )
    out = jit_step(tables, s, A("Absorb", "Headbutt", s, names), o)
    assert int(out.status[1, 0]) == Status.POISON
    no = jit_step(tables, s, A("Absorb", "Headbutt", s, names), empty_outcome())
    assert int(no.status[1, 0]) == Status.NONE


@pytest.mark.slow
def test_static_paralyzes_contact_attackers_only(tables, names):
    # BattleSystem_TriggerAbilityOnHit (battle_lib.c:4200): a contact move that dealt
    # damage, an attacker with no status; subscript_paralyze.s has no type check, so
    # a Ground-type attacker is paralyzed too.
    s = roark_state(
        tables, names, player_species="Pikachu", player_moves=["Growl"], player_ability="Static"
    )
    proc = empty_outcome()._replace(
        contact_proc=jnp.array([[1] * MAX_HITS, [0] * MAX_HITS], jnp.int8)
    )
    tackle = jit_step(tables, s, A("Growl", "Tackle", s, names), proc)
    assert int(tackle.status[1, 0]) == Status.PARALYSIS
    rock_throw = jit_step(tables, s, A("Growl", "Rock Throw", s, names), proc)
    assert int(rock_throw.status[1, 0]) == Status.NONE
    burned = s.with_status(1, 0, Status.BURN)
    assert int(jit_step(tables, burned, A("Growl", "Tackle", s, names), proc).status[1, 0]) == (
        Status.BURN
    )
    missed = proc._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    assert int(jit_step(tables, s, A("Growl", "Tackle", s, names), missed).status[1, 0]) == (
        Status.NONE
    )


def test_natural_cure_on_switch(tables, names):
    s = roark_state(
        tables, names, player_species="Budew", player_ability="Natural Cure"
    ).with_status(0, 0, Status.BURN)
    out = jit_step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE


def test_status_stays_on_switch_without_natural_cure(tables, names):
    s = roark_state(
        tables, names, player_species="Budew", player_ability="Poison Point"
    ).with_status(0, 0, Status.BURN)
    out = jit_step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.BURN


def test_no_contact_proc_without_its_roll(tables, names):
    s = roark_state(
        tables, names, player_species="Pikachu", player_moves=["Growl"], player_ability="Static"
    )
    out = jit_step(
        tables, s, A("Growl", "Tackle", s, names), empty_outcome()._replace(contact_proc=NO_PROC)
    )
    assert int(out.status[1, 0]) == Status.NONE
