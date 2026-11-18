"""E2 batch 13 (rulings B13-n, docs/kaizo/e2-mechanics-ledger.md): power formulas, secondary-effect
hits and held-item moves, by family; hand cases each checked against the reference engine
(``assert_matches_reference``), and the enumerated law of each new shape against the brute
force and the sampler (``tests/engine/test_enumerate.py``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import (
    PLATE_TYPES,
    damage_for_roll,
    falloff_power,
    gyro_ball_power,
    move_type_of,
    punishment_power,
    trump_card_power,
    weather_ball_params,
)
from nuzline.engine.effects import assurance_mark
from nuzline.engine.ids import AbilityId, EffectId, MoveFlag, SemiKind, Status, Type, Weather
from nuzline.engine.order import effective_speed
from nuzline.engine.outcome import (
    MAX_HITS,
    Outcome,
    crit_odds,
    empty_outcome,
    live_hit_threshold,
)
from nuzline.engine.state import make_state
from tests.engine.helpers import A, assert_matches_reference
from tests.engine.test_enumerate import (
    assert_equals_bruteforce,
    check_frequencies,
    children,
    from_bytes_like,
    sampled,
)
from tests.engine.test_pivot import _mon, _resume

O = empty_outcome()


def _state(tables, names, mine, theirs, level=(40, 40)):
    n = names
    return make_state(
        tables,
        [_mon(n, mine[0], mine[1:], level[0])],
        [_mon(n, theirs[0], theirs[1:], level[1])],
    )


def _ability(s, side: int, ability):
    return s._replace(ability=s.ability.at[side, 0].set(ability))


def _plain(s):
    """Both actives without an ability (no Swarm, Simple, Shield Dust, ... in the way)."""
    return _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)


def _secondary(*hits: int):
    row = list(hits) + [0] * (MAX_HITS - len(hits))
    return O._replace(secondary=jnp.asarray([row, [0] * MAX_HITS], jnp.int8))


# --- B13-1: Twineedle (effect_script_0077.s: effect 2's poison, SetMultiHit 2) ---------------


def test_twineedle_tables_both_games(tables, names, kaizo_tables, kaizo_names):
    """effect 77 is SECONDARY_STATUS poison with MoveFlag.HIT_TWICE; vanilla 25 power 20 %,
    Kaizo 50 power 30 % (ROM move data)."""
    for t, n, power, chance in ((tables, names, 25, 20), (kaizo_tables, kaizo_names, 50, 30)):
        m = n.move("Twineedle")
        assert int(t.move_effect[m]) == EffectId.SECONDARY_STATUS
        assert int(np.asarray(t.move_param)[m, 0]) == Status.POISON
        assert int(t.move_flags[m]) & MoveFlag.HIT_TWICE
        assert (int(t.move_power[m]), int(t.move_effect_chance[m])) == (power, chance)


def test_twineedle_hits_twice_and_rolls_poison_on_each_hit(tables, names):
    """SetMultiHit 2 (effect_script_0077.s:6): two hits; the reloaded script sets the poison
    again (:5, bcp.c:3789) and MULTI_HIT_TRIGGER_SECONDARY rolls it per hit (bcp.c:3544-3554,
    battle_lib.c:1570-1586): a proc on either hit poisons."""
    s = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Snorlax", "Growl")))
    hit = int(damage_for_roll(tables, s, 0, names.move("Twineedle"), False, 15))
    for procs, poisoned in (((0, 0), False), ((1, 0), True), ((0, 1), True)):
        out = assert_matches_reference(tables, s, (0, 0), _secondary(*procs))
        tick = int(s.max_hp[1, 0]) // 8 if poisoned else 0  # the turn end's poison damage
        assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == 2 * hit + tick
        assert (int(out.status[1, 0]) == Status.POISON) == poisoned, procs


def test_twineedle_stops_and_skips_its_poison_when_the_target_faints(tables, names):
    """LoopMultiHit stops once faintedMon is set (bcp.c:3780-3782); the fainted defender's roll
    gives nothing (curHP == 0, battle_lib.c:1580)."""
    s = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Bidoof", "Growl")))
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), _secondary(1, 1))
    assert int(out.hp[1, 0]) == 0 and int(out.status[1, 0]) == Status.NONE


def test_twineedle_second_hit_cannot_poison_after_breaking_a_substitute(tables, names):
    """Hit 1 breaks the substitute (bcp.c:3361-3371 sets SELF_TURN_FLAG_SUBSTITUTE_HIT, which
    BattleSystem_SetupLoop does not clear, battle_lib.c:3357-3362); hit 2 hurts the Pokemon but
    its rolled poison is stopped by Battler_SubstituteWasHit (battle_lib.c:1582, :6259-6268) and
    subscript_poison.s's CheckSubstitute (battle_script.c:9173-9186)."""
    s = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Snorlax", "Growl")))
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(1))
    out = assert_matches_reference(tables, s, (0, 0), _secondary(1, 1))
    assert int(out.substitute_hp[1]) == 0
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # the second hit went through
    assert int(out.status[1, 0]) == Status.NONE


def test_twineedle_poison_meets_shield_dust_and_poison_types(tables, names):
    """Shield Dust stops the indirect poison (subscript_poison.s:20) and a Poison type cannot be
    poisoned (subscript_poison.s), on both hits; the damage stays."""
    s = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Snorlax", "Growl")))
    dusty = _ability(s, 1, AbilityId.SHIELD_DUST)
    dust = assert_matches_reference(tables, dusty, (0, 0), _secondary(1, 1))
    assert int(dust.status[1, 0]) == Status.NONE and int(dust.hp[1, 0]) < int(s.hp[1, 0])
    p = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Grimer", "Growl")))
    out = assert_matches_reference(tables, p, (0, 0), _secondary(1, 1))
    assert int(out.status[1, 0]) == Status.NONE


def test_twineedle_poison_is_synchronized_back_after_its_hit(tables, names):
    """AfterMoveEffects' Synchronize step runs after each hit (bcp.c:3665-3670, in the multi-hit
    loop; BattleSystem_SynchronizeStatus, battle_lib.c:4493-4534): a poison on either hit of a
    non-Poison user's Twineedle comes back from a Synchronize holder."""
    s = _plain(_state(tables, names, ("Pikachu", "Twineedle"), ("Snorlax", "Growl")))
    s = _ability(s, 1, AbilityId.SYNCHRONIZE)
    for procs in ((1, 0), (0, 1)):
        out = assert_matches_reference(tables, s, (0, 0), _secondary(*procs))
        assert int(out.status[1, 0]) == Status.POISON, procs
        assert int(out.status[0, 0]) == Status.POISON, procs


@pytest.mark.slow  # the brute force's and the sampler's compiles: 140 s, 5.9 GB
def test_twineedle_law_exact_and_sampled(tables, names):
    """The enumerated turn equals the brute force and the sampler: two crits, two rolls and two
    poison rolls, Serene Grace doubling each (battle_lib.c:1571-1575)."""
    s = _plain(_state(tables, names, ("Beedrill", "Twineedle"), ("Snorlax", "Growl"), (50, 30)))
    s = _ability(s, 0, AbilityId.SERENE_GRACE)
    acts = A("Twineedle", "Growl", s, names)
    assert_equals_bruteforce(tables, s, acts)
    expected = children(tables, s, acts)
    poisoned = [from_bytes_like(s, k)["status"][1, 0] == Status.POISON for k in expected]
    assert any(poisoned) and not all(poisoned)
    n = 20_000
    check_frequencies(expected, sampled(tables, s, acts, n), n)


# --- B13-2: Blaze Kick, Poison Tail, Cross Poison (_0200.s, _0209.s) -------------------------


def test_high_crit_status_hits_tables_and_crit_stage(tables, names, kaizo_tables, kaizo_names):
    """BTLVAR_CRITICAL_BOOSTS + 1 (effect_script_0200.s:5, _0209.s:5) is MoveFlag.CRIT_UP: stage 1
    is 1 in 8, + Scope Lens 1 in 4 (sCriticalStageRates, battle_lib.c:7097, :7124-7130); the
    effect is effect 4's burn / effect 2's poison (:6)."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for move, status in (
            ("Blaze Kick", Status.BURN),
            ("Poison Tail", Status.POISON),
            ("Cross Poison", Status.POISON),
        ):
            m = n.move(move)
            assert int(t.move_effect[m]) == EffectId.SECONDARY_STATUS, move
            assert int(np.asarray(t.move_param)[m, 0]) == status, move
            assert int(t.move_flags[m]) & MoveFlag.CRIT_UP, move
    s = _plain(_state(tables, names, ("Blaziken", "Blaze Kick", "Tackle"), ("Snorlax", "Growl")))
    assert crit_odds(tables, s, jnp.asarray([0, 0], jnp.int32)).tolist()[0] == 8
    assert crit_odds(tables, s, jnp.asarray([1, 0], jnp.int32)).tolist()[0] == 16
    lens = make_state(
        tables,
        [_mon(names, "Blaziken", ["Blaze Kick"], 40, "Scope Lens")],
        [_mon(names, "Snorlax", ["Growl"], 40)],
    )
    assert crit_odds(tables, lens, jnp.asarray([0, 0], jnp.int32)).tolist()[0] == 4


def test_blaze_kick_crits_at_its_stage_and_burns_on_a_proc(tables, names):
    """Crit value 9 lands at 1 in 8 ((9 - 1) % 8 == 0, outcome.crit_lands), not at Tackle's 1 in
    16; the burn needs its roll and a target that can burn (subscript_burn.s: not a Fire
    type)."""
    s = _plain(_state(tables, names, ("Blaziken", "Blaze Kick"), ("Snorlax", "Growl")))
    m = names.move("Blaze Kick")
    normal = int(damage_for_roll(tables, s, 0, m, False, 15))
    crit = int(damage_for_roll(tables, s, 0, m, True, 15))
    o = O._replace(crit=jnp.asarray([[9] + [0] * (MAX_HITS - 1), [0] * MAX_HITS], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == crit > normal
    out = assert_matches_reference(tables, s, (0, 0), _secondary(1))
    assert int(out.status[1, 0]) == Status.BURN
    fire = _plain(_state(tables, names, ("Blaziken", "Blaze Kick"), ("Ponyta", "Growl")))
    out = assert_matches_reference(tables, fire, (0, 0), _secondary(1))
    assert int(out.status[1, 0]) == Status.NONE


@pytest.mark.slow  # the brute force's and the sampler's compiles: 140 s, 5.9 GB
def test_blaze_kick_law_exact_and_sampled(tables, names):
    """The crit draw at 1 in 8 and the burn roll in the enumeration equal the brute force and the
    sampler."""
    s = _plain(_state(tables, names, ("Blaziken", "Blaze Kick"), ("Snorlax", "Growl"), (50, 30)))
    acts = A("Blaze Kick", "Growl", s, names)
    assert_equals_bruteforce(tables, s, acts)
    n = 20_000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


# --- B13-3, B13-4: Kaizo Hurricane (be_seq 141, overlay16's two effect-141 compares) ----------


def _hurricane(t, n, weather=Weather.NONE, theirs=("Snorlax", "Growl")):
    s = _plain(_state(t, n, ("Pidgeot", "Hurricane"), theirs, (50, 40)))
    return s._replace(weather=jnp.int8(weather))


def _cloud_nine(t, s, side: int):
    """``s`` with ``side``'s active Pokemon on the first personal row and ability slot whose
    decomp ability is Cloud Nine or Air Lock (``damage.weather_suppressed`` reads the decomp
    ability; Kaizo has no Cloud Nine species)."""
    from nuzline.engine.ids import WEATHER_ABILITY_IDS

    grid = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    species, slot = next(
        (sp, k)
        for sp, row in enumerate(rows)
        if row >= 0
        for k in range(grid.shape[1])
        if grid[row, k] in (WEATHER_ABILITY_IDS["Cloud Nine"], WEATHER_ABILITY_IDS["Air Lock"])
    )
    return s._replace(
        species=s.species.at[side, 0].set(species),
        form=s.form.at[side, 0].set(0),
        ability_slot=s.ability_slot.at[side, 0].set(slot),
    )


def test_kaizo_hurricane_tables(kaizo_tables, kaizo_names):
    """B13-3: Kaizo's effect 141 is effect 76's confusion (SECONDARY_CONFUSE) with
    SYSCTL_HIT_DURING_FLY (HIT_AIR, no x2: no DOUBLE_SEMI) and the Thunder weather branches
    (MoveFlag.THUNDER); 120 power, 70 accuracy, 30 % (ROM move data)."""
    t, n = kaizo_tables, kaizo_names
    m = n.move("Hurricane")
    assert int(t.move_effect[m]) == EffectId.SECONDARY_CONFUSE
    flags = int(t.move_flags[m])
    assert flags & MoveFlag.THUNDER and flags & MoveFlag.HIT_AIR
    assert not flags & MoveFlag.DOUBLE_SEMI
    assert (int(t.move_power[m]), int(t.move_accuracy[m]), int(t.move_effect_chance[m])) == (
        120,
        70,
        30,
    )


def test_kaizo_hurricane_weather_accuracy(kaizo_tables, kaizo_names):
    """B13-4: in sun the effect-141 compare beside BATTLE_EFFECT_THUNDER gives hitRate 50
    (bcp.c:2915-2917); in rain it clears MOVE_STATUS_MISSED (:3024-3027); both inside
    NO_CLOUD_NINE (an Air Lock foe keeps 70 and the miss)."""
    t, n = kaizo_tables, kaizo_names
    m = n.move("Hurricane")
    assert int(live_hit_threshold(t, _hurricane(t, n, Weather.SUN), 0, m)) == 50
    assert int(live_hit_threshold(t, _hurricane(t, n), 0, m)) == 70
    cloud = _cloud_nine(t, _hurricane(t, n, Weather.SUN), 1)
    assert int(live_hit_threshold(t, cloud, 0, m)) == 70
    miss = O._replace(hit=jnp.full((2, MAX_HITS), 100, jnp.int8))
    rain = _hurricane(t, n, Weather.RAIN)
    out = assert_matches_reference(t, rain, (0, 0), miss)
    assert int(out.hp[1, 0]) < int(rain.hp[1, 0])
    plain = _hurricane(t, n)
    out = assert_matches_reference(t, plain, (0, 0), miss)
    assert int(out.hp[1, 0]) == int(plain.hp[1, 0])
    out = assert_matches_reference(t, _cloud_nine(t, rain, 1), (0, 0), miss)
    assert int(out.hp[1, 0]) == int(rain.hp[1, 0])


def test_kaizo_hurricane_hits_a_flyer_without_doubling_and_confuses(kaizo_tables, kaizo_names):
    """B13-3: SYSCTL_HIT_DURING_FLY lets it hit MOVE_EFFECT_AIRBORNE (bcp.c:3034-3040) at its
    plain power (the script sets no BTLVAR_POWER_MULTI); a proc confuses for the drawn
    Random 3, 2 (subscript_confuse.s), unless Shield Dust (:15)."""
    t, n = kaizo_tables, kaizo_names
    s = _hurricane(t, n, theirs=("Snorlax", "Fly"))
    up = s._replace(
        semi_invulnerable=s.semi_invulnerable.at[1].set(SemiKind.AIRBORNE),
        locked_move=s.locked_move.at[1].set(n.move("Fly")),
    )
    m = n.move("Hurricane")
    plain = int(damage_for_roll(t, s, 0, m, False, 15))
    assert int(damage_for_roll(t, up, 0, m, False, 15)) == plain
    o = _secondary(1)._replace(duration=jnp.asarray([3, 2], jnp.int8))
    out = assert_matches_reference(t, up, (0, 0), o)
    assert int(up.hp[1, 0]) - int(out.hp[1, 0]) == plain
    assert int(out.confusion[1]) > 0
    dust = assert_matches_reference(t, _ability(up, 1, AbilityId.SHIELD_DUST), (0, 0), o)
    assert int(dust.confusion[1]) == 0


@pytest.mark.slow  # the brute force's and the sampler's compiles: 140 s, 5.9 GB
def test_kaizo_hurricane_law_exact_and_sampled(kaizo_tables, kaizo_names):
    """The enumeration in sun (threshold 50) and rain (no miss) equals the brute force and the
    sampler: the hit class, crit, roll, confusion roll and its duration."""
    t, n = kaizo_tables, kaizo_names
    for weather in (Weather.SUN, Weather.RAIN):
        s = _hurricane(t, n, weather)
        acts = A("Hurricane", "Growl", s, n)
        assert_equals_bruteforce(t, s, acts)
        num = 20_000
        check_frequencies(children(t, s, acts), sampled(t, s, acts, num), num)


# --- B13-5..B13-8: the Triple Kick family (effect_script_0104.s; Kaizo be_seq 104 and 88) -----


def _hits(*classes: int):
    """Hit 0's class and the later hits' classes (100 misses any rate below 100)."""
    row = list(classes) + [1] * (MAX_HITS - len(classes))
    return O._replace(hit=jnp.asarray([row, [1] * MAX_HITS], jnp.int8))


def test_triple_kick_tables_both_games(tables, names, kaizo_tables, kaizo_names):
    """B13-5: vanilla effect 104 adds 10 per hit (effect_script_0104.s:7: step 10); Kaizo's be_seq
    104 (Triple Kick, Rock Wrecker) and 88 (Triple Axel, Fury Cutter) add the move's power
    (rom_diff.toml: step 0); powers from the ROM move data."""
    m = names.move("Triple Kick")
    assert int(tables.move_effect[m]) == EffectId.TRIPLE_KICK
    assert np.asarray(tables.move_param)[m].tolist() == [10, 0, 0]
    assert int(tables.move_power[m]) == 10
    for move, power in (
        ("Triple Kick", 30),
        ("Rock Wrecker", 30),
        ("Triple Axel", 20),
        ("Fury Cutter", 20),
    ):
        k = kaizo_names.move(move)
        assert int(kaizo_tables.move_effect[k]) == EffectId.TRIPLE_KICK, move
        assert np.asarray(kaizo_tables.move_param)[k].tolist() == [0, 0, 0], move
        assert int(kaizo_tables.move_power[k]) == power, move


def test_triple_kick_power_steps_and_technician(kaizo_tables, kaizo_names):
    """B13-5: hit k has power k x step (BTLVAR_MOVE_POWER kept between hits, battle_lib.c:1979,
    :3357-3362), read as inPower (:6673-6677) before Technician's movePower <= 60 test
    (:6698-6702): Kaizo's 30-power Triple Kick boosts hits 1-2 (30, 60), not hit 3 (90)."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_state(t, n, ("Hitmontop", "Triple Kick"), ("Snorlax", "Growl"), (40, 60)))
    m = n.move("Triple Kick")
    plain = [int(damage_for_roll(t, s, 0, m, False, 15, hit=k)) for k in range(3)]
    assert plain[0] < plain[1] < plain[2]
    tech = _ability(s, 0, AbilityId.TECHNICIAN)
    boosted = [int(damage_for_roll(t, tech, 0, m, False, 15, hit=k)) for k in range(3)]
    assert boosted[0] > plain[0] and boosted[1] > plain[1] and boosted[2] == plain[2]
    out = assert_matches_reference(t, s, (0, 0), O)
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == sum(plain)


def test_triple_kick_vanilla_steps_by_ten(tables, names):
    """B13-5: vanilla's script adds 10 (effect_script_0104.s:7): hits of power 10, 20, 30, as a
    10-, 20- and 30-power hit of the same move would deal."""
    s = _plain(_state(tables, names, ("Hitmontop", "Triple Kick"), ("Snorlax", "Growl"), (40, 60)))
    m = names.move("Triple Kick")
    plain = [int(damage_for_roll(tables, s, 0, m, False, 15, hit=k)) for k in range(3)]
    for k in range(3):
        power = tables.move_power.at[m].set(10 * (k + 1))
        param = tables.move_param.at[m, 0].set(10 * (k + 1))
        scaled = tables._replace(move_power=power, move_param=param)
        assert int(damage_for_roll(scaled, s, 0, m, False, 15)) == plain[k], k
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == sum(plain)


def test_triple_kick_later_miss_ends_the_move(kaizo_tables, kaizo_names):
    """B13-6: SYSCTL_TRIPLE_KICK has no SYSCTL_SKIP_ACCURACY_CHECK (system_control.h:81-86): hit k
    rolls again (bcp.c:3255-3259), and a miss is MOVE_STATUS_MULTI_HIT_DISRUPTED (:3320-3323),
    which ends the loop (:3782): no later hit, whatever its roll."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_state(t, n, ("Hitmontop", "Triple Kick"), ("Snorlax", "Growl"), (40, 60)))
    m = n.move("Triple Kick")
    per = [int(damage_for_roll(t, s, 0, m, False, 15, hit=k)) for k in range(3)]
    for classes, landed in (((1, 1, 100), 2), ((1, 100, 1), 1), ((100, 1, 1), 0)):
        out = assert_matches_reference(t, s, (0, 0), _hits(*classes))
        assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == sum(per[:landed]), classes


def test_triple_kick_later_hits_skip_the_overrides(kaizo_tables, kaizo_names):
    """B13-6: a later hit skips CheckMoveHitOverrides (SYSCTL_SKIP_ACCURACY_OVERRIDES,
    bcp.c:3264-3268): the first hit of a No Guard user cannot miss (:3016-3021), its later
    hits roll against the plain rate (outcome.live_hit_threshold, overrides=False)."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_state(t, n, ("Machamp", "Triple Kick"), ("Snorlax", "Growl"), (40, 60)))
    guard = _ability(s, 0, AbilityId.NO_GUARD)
    m = n.move("Triple Kick")
    assert int(live_hit_threshold(t, guard, 0, m)) == 100
    assert int(live_hit_threshold(t, guard, 0, m, overrides=False)) == 95
    per = [int(damage_for_roll(t, guard, 0, m, False, 15, hit=k)) for k in range(3)]
    out = assert_matches_reference(t, guard, (0, 0), _hits(100, 96, 1))
    assert int(guard.hp[1, 0]) - int(out.hp[1, 0]) == per[0]
    out = assert_matches_reference(t, guard, (0, 0), _hits(100, 95, 95))
    assert int(guard.hp[1, 0]) - int(out.hp[1, 0]) == sum(per)


def test_triple_kick_micle_boosts_the_first_check_only(kaizo_tables, kaizo_names):
    """B13-6 with G-59: every Triple Kick check runs CheckMoveHitAccuracy, which reads and clears
    the attacker's micleBerry (bcp.c:2967-2970): the first hit's roll is boosted x120/100, the
    later hits' are not (the flag is gone); hit_threshold_pair's later column reads it cleared."""
    from nuzline.engine.outcome import hit_threshold_pair

    t, n = kaizo_tables, kaizo_names
    s = _plain(_state(t, n, ("Hitmontop", "Triple Kick"), ("Snorlax", "Growl"), (40, 60)))
    m = n.move("Triple Kick")
    plain = int(live_hit_threshold(t, s, 0, m))
    micled = s._replace(micle=s.micle.at[0].set(1))
    boosted = int(live_hit_threshold(t, micled, 0, m))
    assert boosted == plain * 120 // 100 > plain
    acts = A("Triple Kick", "Growl", micled, n)
    first, _, later = hit_threshold_pair(t, micled, acts)
    assert int(first[0, 0]) == boosted and int(later[0, 0]) == plain
    per = [int(damage_for_roll(t, micled, 0, m, False, 15, hit=k)) for k in range(3)]
    out = assert_matches_reference(t, micled, (0, 0), _hits(boosted, plain + 1, 1))
    assert int(out.micle[0]) == 0
    assert int(micled.hp[1, 0]) - int(out.hp[1, 0]) == per[0]
    out = assert_matches_reference(t, micled, (0, 0), _hits(boosted, plain, plain))
    assert int(micled.hp[1, 0]) - int(out.hp[1, 0]) == sum(per)


def test_triple_kick_stops_when_the_target_faints(tables, names):
    """LoopMultiHit stops on faintedMon (bcp.c:3780-3782), as for every multi-hit move."""
    s = _plain(_state(tables, names, ("Hitmontop", "Triple Kick"), ("Bidoof", "Growl")))
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[1, 0]) == 0


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_triple_kick_law_exact_and_sampled(kaizo_tables, kaizo_names):
    """B13-6: the later hits' own accuracy classes (hit[s, 1], hit[s, 2], read once the hit
    before can have landed) and the per-hit damage classes (OutcomeLaws.damage[side, hit]) in
    the enumeration equal the brute force and the sampler for a No Guard user (hit 0 sure, the
    later ones at 95), and the sampler at -1 accuracy (95 x 75 / 100 = 71 for every hit; its
    brute force is past the width cap). The foe's Battle Armor keeps the brute force's crit-free
    rolls (16^3) under that cap."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_state(t, n, ("Hitmontop", "Triple Kick"), ("Snorlax", "Growl"), (50, 50)))
    s = _ability(s, 1, AbilityId.BATTLE_ARMOR)
    guard = _ability(s, 0, AbilityId.NO_GUARD)
    acts = A("Triple Kick", "Growl", guard, n)
    assert_equals_bruteforce(t, guard, acts)
    n_ = 20_000
    check_frequencies(children(t, guard, acts), sampled(t, guard, acts, n_), n_)
    low = s._replace(stages=s.stages.at[0, 6].set(-1))
    acts = A("Triple Kick", "Growl", low, n)
    expected = children(t, low, acts)
    assert int(s.hp[1, 0]) in {int(from_bytes_like(low, k)["hp"][1, 0]) for k in expected}
    check_frequencies(expected, sampled(t, low, acts, n_), n_)


@pytest.mark.parametrize("later", [1, 2])
def test_triple_kick_later_classes_equal_raw_rolls(tables, names, later):
    """B13-6: a later hit's class over hit_thresholds(..., overrides=False) (enumerate's
    ``hit_later_thresholds``) steps like every raw roll 1-100 it stands for, after a faster
    Sand-Attack (the check reads the stage of its moment, bcp.c:2873-2874); the hits before it
    land (class 1)."""
    from fractions import Fraction

    from nuzline.engine.enumerate import hit_class_law
    from tests.engine.test_accuracy_classes import laws_of, merged, state_keys, stepped

    s = _plain(
        _state(tables, names, ("Hitmontop", "Triple Kick"), ("Snorlax", "Sand-Attack"), (30, 70))
    )
    acts = A("Triple Kick", "Sand-Attack", s, names)
    laws = laws_of(tables, s, acts)
    assert bool(laws.relevant.hit[0, later])
    thresholds = laws.hit_later_thresholds[0][laws.hit_later_possible[0]]
    assert {90, 67} <= set(thresholds.tolist())
    law = dict(zip(*hit_class_law(thresholds)[:2]))

    def rows(values):
        base = empty_outcome()
        fields = [np.repeat(np.asarray(f)[None], len(values), axis=0).copy() for f in base]
        out = Outcome(*fields)
        out.hit[:, 0, later] = values
        return out

    want = merged(
        state_keys(stepped(tables, s, acts, rows(list(law)))),
        [Fraction(w, 100) for w in law.values()],
    )
    raw = list(range(1, 101))
    got = merged(state_keys(stepped(tables, s, acts, rows(raw))), [Fraction(1, 100)] * 100)
    assert got == want


# --- B13-9..B13-11: Beat Up (effect_script_0154.s, BtlCmd_BeatUp) ---------------------------


def _beat_up_party(t, n, members, foe=("Snorlax", "Growl"), level=40, item=""):
    return make_state(
        t,
        [_mon(n, sp, ["Beat Up"], lv) for sp, lv in members],
        [_mon(n, foe[0], foe[1:], level, item)],
    )


def _beat_up_hit(t, s, slot, crit, roll):
    """battle_script.c:6206-6218 by hand: base Attack x power x (level x 2 / 5 + 2) / base
    Defense / 50 + 2, x2 on a crit, then the variance (at least 1)."""
    base = np.asarray(t.species_base_stats)
    rows = np.asarray(t.species_form_row)
    row = rows[int(s.species[0, slot]), int(s.form[0, slot])]
    foe = rows[int(s.species[1, 0]), int(s.form[1, 0])]
    power = int(t.move_power[int(s.moves[0, 0, 0])])
    level = int(s.level[0, slot])
    d = int(base[row, 1]) * power * (level * 2 // 5 + 2) // int(base[foe, 2]) // 50 + 2
    d = d * (2 if crit else 1)
    return max(d * (85 + roll) // 100, 1)


def test_beat_up_tables_both_games(tables, names, kaizo_tables, kaizo_names):
    """B13-9: effect 154 is BEAT_UP in both games (10 power vanilla, 30 Kaizo; ROM move data)."""
    for t, n, power in ((tables, names, 10), (kaizo_tables, kaizo_names, 30)):
        m = n.move("Beat Up")
        assert int(t.move_effect[m]) == EffectId.BEAT_UP
        assert int(t.move_power[m]) == power


def test_beat_up_hits_once_per_eligible_member(kaizo_tables, kaizo_names):
    """B13-9: BeatUpEligibleMon (battle_script.c:6157-6164): the user always (poisoned here), every
    other slot with HP, a species and no status, in party order, each hit its member's own
    damage (B13-10). A fainted and a poisoned bench member give no hit."""
    t, n = kaizo_tables, kaizo_names
    members = [("Sneasel", 40), ("Machop", 30), ("Absol", 35), ("Geodude", 20), ("Gengar", 45)]
    s = _plain(_beat_up_party(t, n, members))
    s = s._replace(
        hp=s.hp.at[0, 1].set(0),
        status=s.status.at[0, 3].set(Status.POISON).at[0, 0].set(Status.POISON),
    )
    out = assert_matches_reference(t, s, (0, 0), O)
    want = sum(_beat_up_hit(t, s, slot, False, 15) for slot in (0, 2, 4))
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == want


def test_beat_up_full_party_hits_six_times_with_crits_per_hit(kaizo_tables, kaizo_names):
    """B13-9, B13-10: a healthy party of six gives six hits (MAX_HITS 6), each with its own crit
    (CalcCrit in the reloaded script) and roll."""
    t, n = kaizo_tables, kaizo_names
    members = [("Sneasel", 40), ("Machop", 30), ("Absol", 35), ("Geodude", 20), ("Gengar", 45),
               ("Bidoof", 10)]  # fmt: skip
    s = _plain(_beat_up_party(t, n, members, level=60))
    rolls = [[3, 7, 0, 15, 9, 12], [15] * MAX_HITS]
    crits = [[0, 1, 0, 0, 0, 1], [0] * MAX_HITS]
    o = O._replace(roll=jnp.asarray(rolls, jnp.int8), crit=jnp.asarray(crits, jnp.int8))
    out = assert_matches_reference(t, s, (0, 0), o)
    want = sum(_beat_up_hit(t, s, k, crits[0][k], rolls[0][k]) for k in range(6))
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == want


def test_beat_up_skips_the_type_chart(kaizo_tables, kaizo_names):
    """B13-10: SYSCTL_IGNORE_IMMUNITIES (effect_script_0154.s:5) skips CheckTypeChart
    (bcp.c:2355): no Wonder Guard block (Shedinja takes the hit) and no resist berry
    (subscript_type_resist_berry.s:5: a Colbur Berry stays)."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_beat_up_party(t, n, [("Sneasel", 40)], foe=("Shedinja", "Growl")))
    s = _ability(s, 1, AbilityId.WONDER_GUARD)
    out = assert_matches_reference(t, s, (0, 0), O)
    assert int(out.hp[1, 0]) == 0
    berry = _plain(
        _beat_up_party(t, n, [("Sneasel", 40)], foe=("Gengar", "Growl"), item="Colbur Berry")
    )
    out = assert_matches_reference(t, berry, (0, 0), O)
    assert int(out.item[1, 0]) == int(berry.item[1, 0]) != 0
    assert int(berry.hp[1, 0]) - int(out.hp[1, 0]) == _beat_up_hit(t, berry, 0, False, 15)


def test_beat_up_life_orb_recoil_without_the_boost(kaizo_tables, kaizo_names):
    """B13-10: no CalcDamage, so no Life Orb boost (battle_script.c:1346 is CalcDamage's); its
    recoil still follows a hit that set SYSCTL_MOVE_HIT (bcp.c:4758-4765): maxHP / 10."""
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t,
        [_mon(n, "Sneasel", ["Beat Up"], 40, "Life Orb"), _mon(n, "Machop", ["Beat Up"], 30)],
        [_mon(n, "Snorlax", ["Growl"], 40)],
    )
    s = _plain(s)
    out = assert_matches_reference(t, s, (0, 0), O)
    want = sum(_beat_up_hit(t, s, slot, False, 15) for slot in (0, 1))
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == want
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == int(s.max_hp[0, 0]) // 10


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_beat_up_law_exact_and_sampled(kaizo_tables, kaizo_names):
    """B13-10: three members' crits and rolls as per-hit damage classes
    (OutcomeLaws.damage[side, hit]) equal the brute force and the sampler."""
    t, n = kaizo_tables, kaizo_names
    s = _plain(_beat_up_party(t, n, [("Sneasel", 40), ("Machop", 30), ("Absol", 35)], level=50))
    acts = A("Beat Up", "Growl", s, n)
    assert_equals_bruteforce(t, s, acts)
    n_ = 20_000
    check_frequencies(children(t, s, acts), sampled(t, s, acts, n_), n_)


# --- B13-12..B13-17: single-hit variable power and type (agent C) ---------------------------

DECOMP_SCRIPTS = "external/pokeplatinum/res/battle/scripts/effects"


def _weather(s, weather):
    return s._replace(
        weather=jnp.asarray(weather, s.weather.dtype),
        weather_timer=jnp.zeros_like(s.weather_timer),
    )


def _loss(s, out, side=1, slot=0):
    return int(s.hp[side, slot]) - int(out.hp[side, slot])


def _dfr(t, s, move, in_power=0, side=0):
    return int(damage_for_roll(t, s, side, move, False, 15, in_power=in_power))


def test_variable_power_tables_both_games(tables, names, kaizo_tables, kaizo_names):
    """The family's effects and the ROM move data (Kaizo numbers: Weather Ball 50, Water Spout
    150, Gyro Ball 1, Assurance 70, Trump Card 1, Punishment 1, Wake-Up Slap 60 at chance 100,
    Judgment 120); Kaizo's Eruption is effect 253, not 190 (B13-12..B13-17)."""
    from nuzline.engine.ids import FLAIL_FALLOFF, POWER_IF_TARGET_ASLEEP

    want = {
        "Weather Ball": (EffectId.WEATHER_BALL, [0, 0, 0], 50, 50),
        "Water Spout": (EffectId.FLAIL, [FLAIL_FALLOFF, 0, 0], 150, 150),
        "Gyro Ball": (EffectId.GYRO_BALL, [0, 0, 0], 1, 1),
        "Assurance": (EffectId.ASSURANCE, [0, 0, 0], 50, 70),
        "Trump Card": (EffectId.TRUMP_CARD, [0, 0, 0], 1, 1),
        "Punishment": (EffectId.PUNISHMENT, [0, 0, 0], 1, 1),
        "Wake-Up Slap": (EffectId.DOUBLE_POWER, [POWER_IF_TARGET_ASLEEP, 1, 0], 60, 60),
        "Judgment": (EffectId.JUDGMENT, [0, 0, 0], 100, 120),
    }
    for move, (effect, params, vanilla, kaizo) in want.items():
        for t, n, power in ((tables, names, vanilla), (kaizo_tables, kaizo_names, kaizo)):
            m = n.move(move)
            assert int(t.move_effect[m]) == effect, move
            assert np.asarray(t.move_param)[m].tolist() == params, move
            assert int(t.move_power[m]) == power, move
    m = names.move("Eruption")
    assert int(tables.move_effect[m]) == EffectId.FLAIL
    assert int(kaizo_tables.move_effect[kaizo_names.move("Eruption")]) != EffectId.FLAIL


def test_variable_power_scripts_read_back_from_the_decomp():
    """B13-12..B13-17: each effect's script runs its power command before CalcCrit / CalcDamage,
    and Judgment's script maps each HOLD_EFFECT_ARCEUS_* to the type PLATE_TYPES holds, the same
    as sTypeBoostingItems' (tools.effect_map.TYPE_BOOST_HOLD_EFFECTS)."""
    import re
    from pathlib import Path

    from tools.effect_map import TYPE_BOOST_HOLD_EFFECTS

    root = Path(DECOMP_SCRIPTS)
    if not root.is_dir():
        pytest.skip("decomp not checked out")

    def body(e):
        return (root / f"effect_script_{e:04d}.s").read_text(encoding="utf-8")

    for e, cmd in (
        (190, "CalcHPFalloffPower"),
        (203, "CalcWeatherBallParams"),
        (219, "CalcGyroBallPower"),
        (235, "CalcTrumpCardPower"),
        (245, "CalcPunishmentPower"),
    ):
        text = body(e)
        assert text.index(cmd) < text.index("CalcCrit") < text.index("CalcDamage"), e
    assert "BTLVAR_DEFENDER_ASSURANCE_DAMAGE_MASK" in body(231)
    assert "OPCODE_MUL, BTLVAR_MOVE_POWER, 2" in body(231)
    slap = body(217)
    assert "CheckSubstitute" in slap and "MON_CONDITION_SLEEP" in slap
    assert "MOVE_SUBSCRIPT_PTR_HEAL_TARGET_SLEEP" in slap
    text = body(268)
    jumps = dict(re.findall(r"HOLD_EFFECT_ARCEUS_(\w+), (_\d+)", text))
    sets = dict(re.findall(r"(_\d+):\s+UpdateVar OPCODE_SET, BTLVAR_MOVE_TYPE, TYPE_(\w+)", text))
    by_name = {name: h for h, (name, _) in TYPE_BOOST_HOLD_EFFECTS.items()}
    assert len(jumps) == 16
    for arceus, label in jumps.items():
        hold = by_name[f"HOLD_EFFECT_ARCEUS_{arceus}"]
        assert PLATE_TYPES[hold] == Type[sets[label]] == TYPE_BOOST_HOLD_EFFECTS[hold][1]
    assert set(PLATE_TYPES) == {by_name[f"HOLD_EFFECT_ARCEUS_{a}"] for a in jumps}


# B13-12: Weather Ball (CalcWeatherBallParams, battle_script.c:6843-6872) and Judgment.


def test_weather_ball_type_and_power_per_weather(tables, names):
    """x2 and Water / Rock / Fire / Ice in rain / sand / sun / hail; fog doubles the power and
    keeps Normal (FIELD_CONDITION_WEATHER includes DEEP_FOG, condition.h:131-135); Cloud Nine
    on the field (NO_CLOUD_NINE) leaves the table power and type."""
    s = _plain(_state(tables, names, ("Castform", "Weather Ball"), ("Snorlax", "Tail Whip")))
    m = names.move("Weather Ball")
    for weather, kind, power in (
        (Weather.NONE, Type.NORMAL, 50),
        (Weather.RAIN, Type.WATER, 100),
        (Weather.SANDSTORM, Type.ROCK, 100),
        (Weather.SUN, Type.FIRE, 100),
        (Weather.HAIL, Type.ICE, 100),
        (Weather.FOG, Type.NORMAL, 100),
    ):
        w = _weather(s, weather)
        _, got_power, got_type = weather_ball_params(tables, w, m)
        assert (int(got_power), int(got_type)) == (power, kind), weather
        assert int(move_type_of(tables, w, 0, m)) == kind
        cloudy = _cloud_nine(tables, w, 1)
        _, got_power, got_type = weather_ball_params(tables, cloudy, m)
        assert (int(got_power), int(got_type)) == (50, Type.NORMAL), weather


def test_weather_ball_type_reaches_the_chart_and_stab(tables, names):
    """The script's moveType is what the chart (CheckTypeChart, bcp.c:2358-2361) and STAB
    (ApplyTypeChart's inType, battle_lib.c:2582-2597) read: Normal Weather Ball does nothing to
    Gengar, the rain's Water one hits it; a Water-type user gets x1.5 STAB on it in rain."""
    s = _plain(_state(tables, names, ("Castform", "Weather Ball"), ("Gengar", "Tail Whip")))
    acts = A("Weather Ball", "Tail Whip", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert _loss(s, out) == 0
    rain = _weather(s, Weather.RAIN)
    out = assert_matches_reference(tables, rain, acts, O)
    assert _loss(rain, out) > 0
    m = names.move("Weather Ball")
    plain = rain._replace(types=rain.types.at[0, 0].set(jnp.asarray([Type.NORMAL] * 2, jnp.int8)))
    water = rain._replace(types=rain.types.at[0, 0].set(jnp.asarray([Type.WATER] * 2, jnp.int8)))
    assert _dfr(tables, water, m) == _dfr(tables, plain, m) * 15 // 10


def test_judgment_takes_its_plate_type(tables, names, kaizo_tables, kaizo_names):
    """effect_script_0268.s: a held HOLD_EFFECT_ARCEUS_* item (Flame Plate) makes Judgment Fire,
    which hits Gengar; no item or a type item that is no plate (Charcoal) keeps Normal, which
    does not. Any holder, not only Arceus (CheckItemHoldEffect on the attacker)."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        m = n.move("Judgment")
        for user, item, kind in (
            ("Arceus", "Flame Plate", Type.FIRE),
            ("Snorlax", "Splash Plate", Type.WATER),
            ("Arceus", "", Type.NORMAL),
            ("Arceus", "Charcoal", Type.NORMAL),
        ):
            s = make_state(
                t, [_mon(n, user, ["Judgment"], 40, item)], [_mon(n, "Gengar", ["Tail Whip"], 40)]
            )
            s = _plain(s)
            assert int(move_type_of(t, s, 0, m)) == kind, (user, item)
            out = assert_matches_reference(t, s, A("Judgment", "Tail Whip", s, n), O)
            assert (_loss(s, out) > 0) == (kind != Type.NORMAL), (user, item)


# B13-13: Water Spout / Eruption (CalcHPFalloffPower, battle_script.c:6645-6657).


def test_water_spout_power_falls_with_the_users_hp(tables, names):
    """power x curHP / maxHP, at least 1; read before the hit (its only one)."""
    assert int(falloff_power(150, 100, 100)) == 150
    assert int(falloff_power(150, 51, 100)) == 76
    assert int(falloff_power(150, 1, 300)) == 1
    s = _plain(_state(tables, names, ("Wailmer", "Water Spout"), ("Snorlax", "Tail Whip")))
    m = names.move("Water Spout")
    for hp in (int(s.max_hp[0, 0]), int(s.max_hp[0, 0]) // 2, 1):
        h = s.replace_hp(0, 0, hp)
        power = int(falloff_power(150, hp, int(s.max_hp[0, 0])))
        out = assert_matches_reference(tables, h, A("Water Spout", "Tail Whip", h, names), O)
        assert _loss(h, out) == _dfr(tables, h, m, power), hp


def test_water_spout_hp_regions_cut_every_power():
    """prep.variants.falloff_cuts: every HP of a region has its top's power (B13-13)."""
    from nuzline.prep.variants import falloff_cuts

    tops = sorted(set(falloff_cuts(100, 150)) | {100})
    for hp in range(1, 101):
        top = min(t for t in tops if t >= hp)
        assert int(falloff_power(150, hp, 100)) == int(falloff_power(150, top, 100)), hp


# B13-14: Gyro Ball (CalcGyroBallPower, battle_script.c:7123-7131).


def test_gyro_ball_power_formula():
    assert int(gyro_ball_power(50, 100)) == 51
    assert int(gyro_ball_power(100, 50)) == 13
    assert int(gyro_ball_power(10, 100)) == 150  # capped (:7128-7129)
    assert int(gyro_ball_power(100, 3)) == 1


def test_gyro_ball_reads_both_speeds(tables, names):
    """monSpeedValues of both battlers (CompareBattlerSpeed: stages, paralysis, Choice Scarf);
    a paralyzed user's quartered Speed raises the power."""
    s = _plain(_state(tables, names, ("Bronzor", "Gyro Ball"), ("Snorlax", "Tail Whip")))
    m = names.move("Gyro Ball")
    for state in (s, s.with_status(0, 0, Status.PARALYSIS), s.with_stages(1, spe=2)):
        power = int(
            gyro_ball_power(effective_speed(tables, state, 0), effective_speed(tables, state, 1))
        )
        out = assert_matches_reference(tables, state, A("Gyro Ball", "Tail Whip", state, names), O)
        assert _loss(state, out) == _dfr(tables, state, m, power)


def test_gyro_ball_speeds_are_the_action_starts(tables, names):
    """BranchActions' SortMonSpeedOrder (bcp.c:888) writes monSpeedValues before the user's
    status checks: a sleeping Quick Feet user (x1.5 with any status, battle_lib.c:1257-1258)
    that wakes this turn uses its sleeping Speed."""
    s = _plain(_state(tables, names, ("Bronzor", "Gyro Ball"), ("Snorlax", "Tail Whip")))
    s = _ability(s, 0, AbilityId.QUICK_FEET).with_status(0, 0, Status.SLEEP, counter=1)
    m = names.move("Gyro Ball")
    asleep = int(gyro_ball_power(effective_speed(tables, s, 0), effective_speed(tables, s, 1)))
    awake = s.with_status(0, 0, Status.NONE)
    woken = int(
        gyro_ball_power(effective_speed(tables, awake, 0), effective_speed(tables, awake, 1))
    )
    assert asleep != woken
    out = assert_matches_reference(tables, s, A("Gyro Ball", "Tail Whip", s, names), O)
    assert int(out.status[0, 0]) == Status.NONE
    assert _loss(s, out) == _dfr(tables, awake, m, asleep)


def test_gyro_ball_zero_speed_is_refused(tables, names):
    """coverage.gyro_speed_line: below GYRO_SPEED_MIN the -6 stage and paralysis bring the
    divisor to 0 (B13-14)."""
    from nuzline.engine.coverage import GYRO_SPEED_MIN, gyro_speed_line

    slow = _mon(names, "Bronzor", ["Gyro Ball"], 5)
    fast = _mon(names, "Bronzor", ["Gyro Ball"], 60)
    m = names.move("Gyro Ball")
    from nuzline.engine.ids import Stat
    from nuzline.engine.state import spec_stats

    assert int(spec_stats(tables, slow)[Stat.SPE]) < GYRO_SPEED_MIN
    assert gyro_speed_line(tables, m, slow) is not None
    assert gyro_speed_line(tables, m, fast) is None
    assert gyro_speed_line(tables, names.move("Tackle"), slow) is None
    # An Iron Ball holder's Speed is halved before paralysis (battle_lib.c:1240-1246, G-56): it
    # reaches 0 below twice the bound.
    mid = next(
        mon
        for lv in range(5, 101)
        if GYRO_SPEED_MIN
        <= int(spec_stats(tables, mon := _mon(names, "Bronzor", ["Gyro Ball"], lv))[Stat.SPE])
        < 2 * GYRO_SPEED_MIN
    )
    assert gyro_speed_line(tables, m, mid) is None
    assert gyro_speed_line(tables, m, _mon(names, "Bronzor", ["Gyro Ball"], mid.level, "Iron Ball"))


# B13-15: Trump Card (CalcTrumpCardPower, :7233-7245) and Punishment (:7354-7371).


def test_trump_card_reads_the_pp_after_its_decrement(tables, names):
    """ppCur of moveSlot after BattleControllerPlayer_DecrementPP (bcp.c:3182): 3 PP spends to 2
    (60), 1 to 0 (200); a Pressure foe takes 2 (bcp.c:2236-2270)."""
    assert [int(trump_card_power(p)) for p in range(7)] == [200, 80, 60, 50, 40, 40, 40]
    s = _plain(_state(tables, names, ("Eevee", "Trump Card"), ("Snorlax", "Tail Whip")))
    m = names.move("Trump Card")
    for pp, pressure, power in ((3, False, 60), (1, False, 200), (3, True, 80), (9, False, 40)):
        h = s.with_pp(0, 0, [pp, 0, 0, 0])
        if pressure:
            h = _ability(h, 1, AbilityId.PRESSURE)
        out = assert_matches_reference(tables, h, A("Trump Card", "Tail Whip", h, names), O)
        assert _loss(h, out) == _dfr(tables, h, m, power), (pp, pressure)


def test_punishment_counts_the_targets_raised_stages(tables, names):
    """60 + 20 x the stages above neutral (drops ignored), at most 200."""
    assert int(punishment_power(jnp.asarray([0, 2, -1, 0, 0, 1, 0, 0]))) == 120
    assert int(punishment_power(jnp.asarray([0, 6, 6, 0, 0, 0, 0, 0]))) == 200
    s = _plain(_state(tables, names, ("Sneasel", "Punishment"), ("Snorlax", "Tail Whip")))
    m = names.move("Punishment")
    for stages, power in (({}, 60), ({"atk": 2, "spe": 1, "def_": -1}, 120), ({"eva": 6}, 180)):
        h = s.with_stages(1, **stages)
        out = assert_matches_reference(tables, h, A("Punishment", "Tail Whip", h, names), O)
        assert _loss(h, out) == _dfr(tables, h, m, power), stages


# B13-16: Wake-Up Slap (effect_script_0217.s, SmellingSalt's script on sleep).


def test_wake_up_slap_doubles_and_wakes_a_sleeper(tables, names, kaizo_tables, kaizo_names):
    """Against MON_CONDITION_SLEEP (any count) powerMul 20 and an ON_HIT
    subscript_heal_target_sleep.s (wakes a target with HP left); behind a substitute
    (CheckSubstitute, :5) a plain hit and no wake-up; Kaizo's effect chance 100 changes
    nothing (an ON_HIT effect rolls no chance)."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = _plain(_state(t, n, ("Hitmonchan", "Wake-Up Slap"), ("Snorlax", "Tail Whip"), (30, 50)))
        m = n.move("Wake-Up Slap")
        plain = _dfr(t, s, m)
        sleeper = s.with_status(1, 0, Status.SLEEP, counter=3)
        out = assert_matches_reference(t, sleeper, A("Wake-Up Slap", "Tail Whip", sleeper, n), O)
        assert _loss(sleeper, out) == _dfr(t, sleeper, m) == _dfr(t, s, m, 2 * int(t.move_power[m]))
        assert _loss(sleeper, out) > plain
        assert int(out.status[1, 0]) == Status.NONE and int(out.status_counter[1, 0]) == 0
        subbed = sleeper._replace(substitute_hp=sleeper.substitute_hp.at[1].set(200))
        out = assert_matches_reference(t, subbed, A("Wake-Up Slap", "Tail Whip", subbed, n), O)
        assert 200 - int(out.substitute_hp[1]) == plain
        assert int(out.status[1, 0]) == Status.SLEEP


# B13-17: Assurance (effect_script_0231.s, turnFlags.assuranceDamageMask).


def test_assurance_mark_follows_update_hp():
    """subscript_update_hp.s:18-19: hpCalcTemp <= 0 is stored: a loss sets, a zero update
    clears, a heal or no update keeps."""
    from nuzline.engine.state import BattleState

    s = BattleState(*[jnp.zeros(1)] * len(BattleState._fields))._replace(
        ts_assurance=jnp.zeros(2, jnp.int8)
    )
    s = assurance_mark(s, 1, 5, True)
    assert np.asarray(s.ts_assurance).tolist() == [0, 1]
    assert np.asarray(assurance_mark(s, 1, 0, False).ts_assurance).tolist() == [0, 1]
    assert np.asarray(assurance_mark(s, 1, -3, True).ts_assurance).tolist() == [0, 1]
    assert np.asarray(assurance_mark(s, 1, 0, True).ts_assurance).tolist() == [0, 0]


def _assurance(t, n, foe_moves, levels=(25, 70), foe_item="", bench=False):
    foes = [_mon(n, "Snorlax", foe_moves, levels[1], foe_item)]
    if bench:
        foes.append(_mon(n, "Snorlax", ["Tail Whip"], levels[1]))
    return _plain(make_state(t, [_mon(n, "Absol", ["Assurance"], levels[0])], foes))


def test_assurance_doubles_after_the_targets_own_hp_loss(tables, names):
    """A faster foe's Belly Drum (UPDATE_HP, subscript_belly_drum.s:8-12) or Substitute
    (subscript_try_substitute.s:12-13) sets its flag: Assurance's movePower x2 (:7-9), the
    substitute taking it; a Growl leaves it plain; Assurance first sees no flag."""
    t, n = tables, names
    m = n.move("Assurance")
    s = _assurance(t, n, ["Tail Whip", "Belly Drum", "Substitute"])
    plain, double = _dfr(t, s, m, 50), _dfr(t, s, m, 100)
    out = assert_matches_reference(t, s, A("Assurance", "Tail Whip", s, n), O)
    assert _loss(s, out) == plain
    out = assert_matches_reference(t, s, A("Assurance", "Belly Drum", s, n), O)
    assert _loss(s, out) == int(s.max_hp[1, 0]) // 2 + double
    out = assert_matches_reference(t, s, A("Assurance", "Substitute", s, n), O)
    quarter = int(s.max_hp[1, 0]) // 4
    assert int(out.substitute_hp[1]) == max(quarter - double, 0)
    slow = _assurance(t, n, ["Belly Drum"], levels=(45, 40))
    out = assert_matches_reference(t, slow, A("Assurance", "Belly Drum", slow, n), O)
    assert _loss(slow, out) == int(slow.max_hp[1, 0]) // 2 + _dfr(t, slow, m, 50)
    for name in ("ts_assurance",):
        assert not np.any(np.asarray(getattr(out, name)))  # 0 between turns


def test_assurance_doubles_on_a_newcomer_hurt_by_the_hazards(tables, names):
    """The hazards' UPDATE_HP (subscript_hazards_check.s) sets the switched-in battler's flag;
    the switch keeps turnFlags (UpdateAfterSwitch clears no turnFlags)."""
    t, n = tables, names
    s = _assurance(t, n, ["Tail Whip"], bench=True)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    out = assert_matches_reference(t, s, (A("Assurance", "Tail Whip", s, n)[0], 4 + 1), O)
    rock = int(s.max_hp[1, 1]) // 8
    newcomer = s._replace(active=s.active.at[1].set(1))
    assert _loss(s, out, slot=1) == rock + _dfr(t, newcomer, n.move("Assurance"), 100)


def test_assurance_flag_survives_the_u_turn_pause(tables, names):
    """A faster foe's U-turn with Life Orb (subscript_lose_hp_from_item.s: UPDATE_HP) sets its
    battler's flag and the hit sets the user's (bcp.c:3437-3438), kept in the pause's scratch;
    the newcomer behind it takes the x2 (the switch keeps the battler's turnFlags)."""
    t, n = tables, names
    s = _plain(
        make_state(
            t,
            [_mon(n, "Snorlax", ["Assurance"], 40)],
            [
                _mon(n, "Scyther", ["U-turn"], 35, "Life Orb"),
                _mon(n, "Snorlax", ["Tail Whip"], 40),
            ],
        )
    )
    paused = assert_matches_reference(t, s, A("Assurance", "U-turn", s, n), O)
    assert int(paused.pivot_battler) == 2
    assert np.asarray(paused.ts_assurance).tolist() == [1, 1]
    after = _resume(t, paused, 1)
    newcomer = paused._replace(active=paused.active.at[1].set(1))
    assert _loss(paused, after, slot=1) == _dfr(t, newcomer, n.move("Assurance"), 100)


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_wake_up_slap_law_exact_and_sampled(tables, names):
    """B13-16: a faster Wake-Up Slap wakes the sleeper, whose own move then draws its hit, crit
    and roll this turn; the enumeration equals the brute force and the sampler."""
    s = _plain(
        _state(tables, names, ("Hitmonchan", "Wake-Up Slap"), ("Snorlax", "Tackle"), (30, 50))
    )
    s = s.with_status(1, 0, Status.SLEEP, counter=3)
    acts = A("Wake-Up Slap", "Tackle", s, names)
    assert_equals_bruteforce(tables, s, acts)
    n_ = 20_000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n_), n_)


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_assurance_law_exact_and_sampled(tables, names):
    """B13-17: a faster foe's Take Down sets its flag through its recoil only when it lands, so
    Assurance's damage follows the foe's hit roll (a NON_TABLE_DAMAGE_EFFECTS move: its crit and
    roll stay); the enumeration equals the brute force and the sampler."""
    s = _assurance(tables, names, ["Take Down"])
    s = s.with_stages(1, acc=-1)
    acts = A("Assurance", "Take Down", s, names)
    assert_equals_bruteforce(tables, s, acts)
    n_ = 20_000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n_), n_)


# --- B13-19..B13-20: the held-item moves (Recycle, Natural Gift) ---------------------------


def _held(s, side: int, item: int, slot: int = 0):
    return s._replace(item=s.item.at[side, slot].set(item))


def _record(s, side: int, item: int):
    return s._replace(recycle_item=s.recycle_item.at[side].set(item))


def test_held_item_move_tables_both_games(tables, names, kaizo_tables, kaizo_names):
    """effect_script_0184.s (Recycle) and _0222.s (Natural Gift) map to their own ids in both
    games (Kaizo keeps both scripts and Natural Gift's power 1)."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Recycle")]) == EffectId.RECYCLE
        assert int(t.move_effect[n.move("Natural Gift")]) == EffectId.NATURAL_GIFT
        assert int(t.move_power[n.move("Natural Gift")]) == 1


def test_remove_item_writes_the_battlers_recycle_item(tables, names):
    """BtlCmd_RemoveItem (battle_script.c:8977): an Oran Berry eaten after a hit goes to its
    battler's recycleItem (and the slot's item_consumed); a switch keeps the record, which
    belongs to the battler (nothing but RemoveItem and TryRecycle write it, B13-19)."""
    oran = names.item("Oran Berry")
    s = make_state(
        tables,
        [_mon(names, "Machop", ["Tackle"], 40)],
        [
            _mon(names, "Geodude", ["Defense Curl"], 20, "Oran Berry"),
            _mon(names, "Onix", ["Harden"], 20),
        ],
    )
    s = _plain(s).replace_hp(1, 0, 20)
    out = assert_matches_reference(tables, s, A("Tackle", "Defense Curl", s, names), O)
    assert int(out.item[1, 0]) == 0 and int(out.recycle_item[1]) == oran
    assert int(out.item_consumed[1, 0]) == oran
    switched = assert_matches_reference(tables, out, A("Tackle", MOVE_ACTIONS + 1, out, names), O)
    assert int(switched.active[1]) == 1 and int(switched.recycle_item[1]) == oran


def test_recycle_gives_back_the_record_and_clears_it(tables, names):
    """effect_script_0184.s: with no held item and a record, the user holds the item again (:7)
    and BtlCmd_TryRecycle clears the record (battle_script.c:9005-9007); the message names
    msgItemTemp (BTLSCR_MSG_TEMP), so the AI learns nothing (B13-19)."""
    sitrus = names.item("Sitrus Berry")
    s = _plain(_state(tables, names, ("Snorlax", "Recycle"), ("Geodude", "Defense Curl")))
    s = _record(s, 0, sitrus)
    out = assert_matches_reference(tables, s, A("Recycle", "Defense Curl", s, names), O)
    assert int(out.item[0, 0]) == sitrus and int(out.recycle_item[0]) == 0
    assert int(out.ai_item_known[0]) == 0


def test_recycled_berry_is_eaten_at_the_move_end(tables, names):
    """The restored Sitrus Berry meets AfterMoveEffects' held-item check (a holder at half HP
    or less eats it), whose RemoveItem writes the record again (B13-19)."""
    sitrus = names.item("Sitrus Berry")
    s = _plain(_state(tables, names, ("Snorlax", "Recycle"), ("Geodude", "Defense Curl")))
    s = _record(s, 0, sitrus).replace_hp(0, 0, 20)
    out = assert_matches_reference(tables, s, A("Recycle", "Defense Curl", s, names), O)
    assert int(out.item[0, 0]) == 0 and int(out.recycle_item[0]) == sitrus
    assert int(out.hp[0, 0]) == 20 + int(s.max_hp[0, 0]) // 4


def test_recycle_fails_holding_an_item_or_without_a_record(tables, names):
    """effect_script_0184.s:5 (a held item) and BtlCmd_TryRecycle's empty record
    (battle_script.c:9005, :9010-9011): MOVE_STATUS_FAILED, nothing changes but the PP."""
    oran, sitrus = names.item("Oran Berry"), names.item("Sitrus Berry")
    base = _plain(_state(tables, names, ("Snorlax", "Recycle"), ("Geodude", "Defense Curl")))
    for s in (_record(_held(base, 0, oran), 0, sitrus), base):
        out = assert_matches_reference(tables, s, A("Recycle", "Defense Curl", s, names), O)
        assert int(out.item[0, 0]) == int(s.item[0, 0])
        assert int(out.recycle_item[0]) == int(s.recycle_item[0])
        slot = names.move_slot(s, 0, "Recycle")
        assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot]) - 1


def test_natural_gift_takes_its_berrys_power_and_type_and_eats_it(tables, names):
    """BtlCmd_CalcNaturalGiftParams (battle_script.c:8088-8102): an Oran Berry gives power 60
    and Poison (res/items/data/oran_berry.json), so Natural Gift does nothing to a Steel foe;
    RemoveItem (effect_script_0222.s:8) eats the berry before TryMove either way, into
    recycleItem (B13-20)."""
    oran = names.item("Oran Berry")
    m = names.move("Natural Gift")
    for foe, hurts in (("Machop", True), ("Steelix", False)):
        s = _plain(_state(tables, names, ("Snorlax", "Natural Gift"), (foe, "Tail Whip")))
        s = _held(s, 0, oran)
        assert int(move_type_of(tables, s, 0, m)) == Type.POISON
        want = _dfr(tables, s, m)
        out = assert_matches_reference(tables, s, A("Natural Gift", "Tail Whip", s, names), O)
        assert int(out.item[0, 0]) == 0 and int(out.recycle_item[0]) == oran
        assert int(out.item_consumed[0, 0]) == oran
        assert _loss(s, out) == (want if hurts else 0), foe
        # damage.gift_item: after the removal the record names the berry, so the type holds.
        assert int(move_type_of(tables, out, 0, m)) == Type.POISON


def test_natural_gift_misses_still_eat_the_berry(tables, names):
    """The script's RemoveItem runs before TryMove's accuracy roll: a miss loses the berry."""
    oran = names.item("Oran Berry")
    s = _plain(_state(tables, names, ("Snorlax", "Natural Gift"), ("Machop", "Tail Whip")))
    s = _held(s, 0, oran)
    miss = O._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))  # class 0 misses
    out = assert_matches_reference(tables, s, A("Natural Gift", "Tail Whip", s, names), miss)
    assert int(out.item[0, 0]) == 0 and int(out.recycle_item[0]) == oran
    assert _loss(s, out) == 0


def test_natural_gift_fails_without_a_gift_item(tables, names):
    """naturalGiftPower 0 (no item; a Potion, not a berry) fails it (battle_script.c:8094-8099,
    effect_script_0222.s:11-13): the item stays, no damage, no record."""
    potion = names.item("Potion")
    base = _plain(_state(tables, names, ("Snorlax", "Natural Gift"), ("Machop", "Tail Whip")))
    for s in (base, _held(base, 0, potion)):
        out = assert_matches_reference(tables, s, A("Natural Gift", "Tail Whip", s, names), O)
        assert int(out.item[0, 0]) == int(s.item[0, 0]) and _loss(s, out) == 0
        assert int(out.recycle_item[0]) == 0


def test_natural_gift_user_unburdened_after_its_berry(tables, names):
    """An Unburden holder that loses its berry to Natural Gift is doubled afterwards
    (order.effective_speed, battle_lib.c:1268-1272)."""
    oran = names.item("Oran Berry")
    s = _plain(_state(tables, names, ("Drifblim", "Natural Gift"), ("Machop", "Tail Whip")))
    s = _ability(_held(s, 0, oran), 0, AbilityId.UNBURDEN)
    s = s._replace(can_unburden=s.can_unburden.at[0].set(1))
    out = assert_matches_reference(tables, s, A("Natural Gift", "Tail Whip", s, names), O)
    assert int(effective_speed(tables, out, 0)) == 2 * int(effective_speed(tables, s, 0))


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_natural_gift_law_exact_and_sampled(tables, names):
    """B13-20: Natural Gift's hit, crit and roll with its berry's power and type, the berry gone
    before the foe's move; the enumeration equals the brute force and the sampler."""
    s = _plain(_state(tables, names, ("Snorlax", "Natural Gift"), ("Machop", "Karate Chop")))
    s = _held(s, 0, names.item("Sitrus Berry")).replace_hp(0, 0, 60)
    acts = A("Natural Gift", "Karate Chop", s, names)
    assert_equals_bruteforce(tables, s, acts)
    n_ = 20_000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n_), n_)


@pytest.mark.slow  # the brute force's and the sampler's compiles
def test_recycle_law_exact_and_sampled(tables, names):
    """B13-19: a faster Recycle gives an Unburden holder its Salac Berry back (no longer doubled;
    eaten at the move end in a pinch, Speed +1) before the foe's hit; the enumeration equals the
    brute force and the sampler (the end-of-turn speeds: _end_speeds' toggled Unburden point)."""
    s = _plain(_state(tables, names, ("Drifblim", "Recycle"), ("Machop", "Karate Chop"), (40, 30)))
    s = _ability(_record(s, 0, names.item("Salac Berry")), 0, AbilityId.UNBURDEN)
    s = s._replace(can_unburden=s.can_unburden.at[0].set(1)).replace_hp(0, 0, 30)
    acts = A("Recycle", "Karate Chop", s, names)
    assert_equals_bruteforce(tables, s, acts)
    n_ = 20_000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n_), n_)
