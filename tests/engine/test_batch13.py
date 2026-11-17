"""E2 batch 13 (rulings B13-n, docs/kaizo/e2-mechanics-ledger.md): power formulas, secondary-effect
hits and held-item moves, by family; hand cases each checked against the reference engine
(``assert_matches_reference``), and the enumerated law of each new shape against the brute
force and the sampler (``tests/engine/test_enumerate.py``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.ids import AbilityId, EffectId, MoveFlag, SemiKind, Status, Weather
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
from tests.engine.test_pivot import _mon

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
