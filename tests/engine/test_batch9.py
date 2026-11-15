"""E2 batch 9 (rulings B9-n, docs/kaizo/e2-mechanics-ledger.md): the Gardenia split's mechanics,
hand cases each checked against the reference engine (``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.ids import AbilityId, Status, Weather
from nuzline.engine.order import effective_speed
from nuzline.engine.outcome import empty_outcome, live_hit_threshold
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()


def _roll(u: int):
    return O._replace(hit=jnp.full((2, 5), u, jnp.int8))


def test_thunder_and_blizzard_weather_accuracy(tables, names):
    """B9-1: Thunder's hitRate is 50 in sun (bcp.c:2915-2917); rain clears Thunder's miss and
    hail Blizzard's (CheckMoveHitOverrides, :3024-3031)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Pikachu", ["Thunder", "Blizzard"], 40)],
        [_mon(n, "Psyduck", ["Growl"], 20)],
    )
    sun = s._replace(weather=jnp.int8(Weather.SUN))
    assert int(live_hit_threshold(tables, sun, 0, n.move("Thunder"))) == 50
    assert int(live_hit_threshold(tables, s, 0, n.move("Thunder"))) == 70
    for weather, slot in ((Weather.RAIN, 0), (Weather.HAIL, 1)):
        w = s._replace(weather=jnp.int8(weather))
        out = assert_matches_reference(tables, w, (slot, 0), _roll(100))
        assert int(out.hp[1, 0]) < int(s.hp[1, 0])
    missed = assert_matches_reference(tables, s, (1, 0), _roll(100))
    assert int(missed.hp[1, 0]) == int(s.hp[1, 0])


def test_fire_fang_burns_and_flinches(tables, names):
    """B9-2: subscript_burn_or_flinch.s is subscript_paralyze_or_flinch.s with the burn: the
    status roll then the flinch roll."""
    n = names
    s = make_state(
        tables, [_mon(n, "Growlithe", ["Fire Fang"], 40)], [_mon(n, "Geodude", ["Tackle"], 20)]
    )
    o = O._replace(
        secondary=jnp.ones((2, 5), jnp.int8), fang_flinch=jnp.ones((2,), jnp.int8)
    )
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.status[1, 0]) == Status.BURN


def test_unburden_doubles_speed_after_its_berry(tables, names):
    """B9-3: RecoverStatusByAbility arms canUnburden for an Unburden holder with an item
    (battle_lib.c:4433-4437); with no item it doubles its speed (:1268-1272)."""
    n = names
    from dataclasses import replace

    drifloon = replace(_mon(n, "Drifloon", ["Growl"], 30, "Oran Berry"), ability_slot=1)
    s = make_state(tables, [drifloon], [_mon(n, "Geodude", ["Tackle"], 30)])
    assert int(s.ability[0, 0]) == AbilityId.UNBURDEN and int(s.can_unburden[0]) == 1
    s = s._replace(hp=s.hp.at[0, 0].set(int(s.max_hp[0, 0]) // 2 - 1))
    before = int(effective_speed(tables, s, 0))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.item[0, 0]) == 0 and int(out.can_unburden[0]) == 1
    assert int(effective_speed(tables, out, 0)) == 2 * before


def test_huge_power_doubles_attack_before_the_choice_band(tables, names):
    """B9-3: Huge Power's x2 (battle_lib.c:6706-6708) comes before Choice Band's x1.5 (:6724)."""
    from nuzline.engine.damage import damage_table

    n = names
    s = make_state(
        tables,
        [_mon(n, "Marill", ["Tackle"], 30, "Choice Band")],
        [_mon(n, "Geodude", ["Tackle"], 30)],
    )
    plain = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    huge = s._replace(ability=s.ability.at[0, 0].set(AbilityId.HUGE_POWER))
    move = n.move("Tackle")
    assert int(damage_table(tables, huge, 0, move, False)[15]) > int(
        damage_table(tables, plain, 0, move, False)[15]
    )


def test_double_edge_recoil_is_a_third(tables, names):
    """B9-4: subscript_recoil_1_3.s: DivideVarByValue(hitDamage, 3), at least 1."""
    n = names
    s = make_state(
        tables, [_mon(n, "Bidoof", ["Double-Edge"], 40)], [_mon(n, "Psyduck", ["Growl"], 40)]
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert dealt > 0
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(dealt // 3, 1)


def test_kaizo_recoil_scripts(kaizo_tables, kaizo_names):
    """B9-4: Kaizo's effect 48 is 1/2 recoil, its 182 the vanilla 1/4 recoil (rom_diff.toml
    be_seq 48, 182 modelled; effect_map.KAIZO_EFFECTS)."""
    import numpy as np

    from nuzline.engine.ids import EffectId

    t, n = kaizo_tables, kaizo_names
    for move, divisor in (("Hyper Beam", 2), ("Take Down", 4), ("Double-Edge", 3)):
        m = n.move(move)
        assert int(t.move_effect[m]) == EffectId.RECOIL, move
        assert int(np.asarray(t.move_param)[m, 0]) == divisor, move


def test_helping_hand_fails_and_follow_me_changes_nothing_in_singles(tables, names):
    """B9-5: BtlCmd_TryHelpingHand fails outside a double battle (battle_script.c:6309-6311);
    Follow Me's followMe names the side's one battler, the foe's target anyway."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Togepi", ["Helping Hand", "Follow Me"], 30)],
        [_mon(n, "Geodude", ["Growl"], 30)],
    )
    for slot in (0, 1):
        out = assert_matches_reference(tables, s, (slot, 0), O)
        assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_volt_absorb_takes_thunder_wave_and_clear_body_keeps_its_attack(tables, names):
    """B9-6: Volt Absorb heals Divide(maxHP, 4) from an Electric move whatever its power
    (battle_lib.c:3500-3505); Clear Body blocks a foe's drop (battle_script.c:2884-2895)."""
    n = names
    s = make_state(
        tables, [_mon(n, "Pikachu", ["Thunder Wave"], 30)], [_mon(n, "Psyduck", ["Growl"], 30)]
    )
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.VOLT_ABSORB))
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[1, 0]) == Status.NONE
    assert int(out.hp[1, 0]) == 1 + max(int(s.max_hp[1, 0]) // 4, 1)
    g = make_state(
        tables, [_mon(n, "Bidoof", ["Growl"], 30)], [_mon(n, "Geodude", ["Rock Polish"], 30)]
    )
    g = g._replace(ability=g.ability.at[1, 0].set(AbilityId.CLEAR_BODY))
    out = assert_matches_reference(tables, g, (0, 0), O)
    from nuzline.engine.ids import Stat

    assert int(out.stages[1, Stat.ATK]) == 0


def test_poison_heal_and_bad_dreams(tables, names):
    """B9-8: subscript_poison_damage.s:4-12 heals a Poison Heal holder Divide(maxHP, 8);
    Bad Dreams takes Divide(maxHP, 8) from a sleeping foe (bcp.c:1484-1500)."""
    n = names
    s = make_state(tables, [_mon(n, "Shroomish", ["Growl"], 30)], [_mon(n, "Geodude", ["Growl"], 30)])
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.POISON_HEAL))
    s = s.with_status(0, 0, Status.POISON)._replace(hp=s.hp.at[0, 0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[0, 0]) == 1 + max(int(s.max_hp[0, 0]) // 8, 1)
    d = make_state(tables, [_mon(n, "Shroomish", ["Growl"], 30)], [_mon(n, "Geodude", ["Growl"], 30)])
    d = d._replace(ability=d.ability.at[0, 0].set(AbilityId.BAD_DREAMS)).with_status(1, 0, Status.SLEEP, 3)
    out = assert_matches_reference(tables, d, (0, 0), O)
    assert int(out.hp[1, 0]) == int(d.hp[1, 0]) - max(int(d.max_hp[1, 0]) // 8, 1)


def test_dragon_rage_deals_forty(tables, names):
    """B9-9: effect_script_0041.s sets BTLVAR_DAMAGE -40 under SYSCTL_IGNORE_TYPE_CHECKS."""
    n = names
    s = make_state(tables, [_mon(n, "Gyarados", ["Dragon Rage"], 30)], [_mon(n, "Onix", ["Growl"], 40)])
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == 40


def test_belly_drum_minimize_stomp_bulk_up(tables, names):
    """B9-10: subscript_belly_drum.s (Attack +6, Divide(maxHP, 2)); subscript_minimize.s (the flag
    and Evasion +1); Stomp's powerMul 20 against it (effect_script_0150.s:6-8); Bulk Up's Attack
    and Defense +1."""
    from nuzline.engine.damage import damage_table
    from nuzline.engine.ids import Stat

    n = names
    s = make_state(
        tables,
        [_mon(n, "Snorlax", ["Belly Drum", "Minimize", "Bulk Up"], 30)],
        [_mon(n, "Geodude", ["Harden"], 30)],
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.stages[0, Stat.ATK]) == 6
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) - max(int(s.max_hp[0, 0]) // 2, 1)
    m = assert_matches_reference(tables, s, (1, 0), O)
    assert int(m.minimized[0]) == 1 and int(m.stages[0, Stat.EVA]) == 1
    b = assert_matches_reference(tables, s, (2, 0), O)
    assert int(b.stages[0, Stat.ATK]) == 1 and int(b.stages[0, Stat.DEF]) == 1
    t = make_state(tables, [_mon(n, "Tauros", ["Stomp"], 30)], [_mon(n, "Snorlax", ["Growl"], 30)])
    stomp = n.move("Stomp")
    plain = int(damage_table(tables, t, 0, stomp, False)[15])
    small = t._replace(minimized=t.minimized.at[1].set(1))
    assert int(damage_table(tables, small, 0, stomp, False)[15]) > plain


def test_dream_eater_needs_a_sleeper_and_payback_doubles_after_the_target(tables, names):
    """B9-12: effect_script_0008.s fails on an awake defender (MOVE_STATUS_NO_MORE_WORK, :5);
    BtlCmd_CalcPaybackPower doubles once the defender acted (battle_script.c:7198-7209)."""
    n = names
    s = make_state(tables, [_mon(n, "Drowzee", ["Dream Eater"], 30)], [_mon(n, "Geodude", ["Harden"], 30)])
    awake = assert_matches_reference(tables, s, (0, 0), O)
    assert int(awake.hp[1, 0]) == int(s.hp[1, 0])
    asleep = s.with_status(1, 0, Status.SLEEP, 3)._replace(hp=s.hp.at[0, 0].set(1))
    out = assert_matches_reference(tables, asleep, (0, 0), O)
    dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert dealt > 0 and int(out.hp[0, 0]) == 1 + max(dealt // 2, 1)
    slow = make_state(tables, [_mon(n, "Slowpoke", ["Payback"], 30)], [_mon(n, "Geodude", ["Harden"], 30)])
    assert_matches_reference(tables, slow, (0, 0), O)
    # Payback doubles once the faster Geodude has acted (Leer touches neither damage input).
    move = n.move("Payback")
    lee = make_state(tables, [_mon(n, "Slowpoke", ["Payback"], 30)], [_mon(n, "Geodude", ["Leer"], 30)])
    out = assert_matches_reference(tables, lee, (0, 0), O)
    from nuzline.engine.damage import damage_for_roll

    doubled = int(damage_for_roll(tables, lee, 0, move, False, 15, power_mul=20))
    assert doubled > int(damage_for_roll(tables, lee, 0, move, False, 15))
    assert int(lee.hp[1, 0]) - int(out.hp[1, 0]) == min(doubled, int(lee.hp[1, 0]))


# --- batch 9 review follow-ups (B9-13 onward) --------------------------------------------


def _set_ability(s, side, ability, slot=0):
    return s._replace(ability=s.ability.at[side, slot].set(ability))


def test_unburden_speed_axis_of_a_newcomer(tables, names):
    """B9-13: the end-of-turn speed grid's incoming Pokemon has its own canUnburden: zeroed by
    UpdateAfterSwitch (battle_lib.c:2101-2104) and armed only for a holder with an item
    (:4433-4437), not the outgoing battler's flag."""
    from nuzline.engine.enumerate import _end_speeds
    from nuzline.engine.order import effective_speed

    n = names
    s = make_state(
        tables,
        [_mon(n, "Drifloon", ["Growl"], 30), _mon(n, "Drifloon", ["Growl"], 30)],
        [_mon(n, "Geodude", ["Growl"], 30)],
    )
    s = _set_ability(_set_ability(s, 0, AbilityId.UNBURDEN, 0), 0, AbilityId.UNBURDEN, 1)
    s = s._replace(can_unburden=s.can_unburden.at[0].set(1))  # the outgoing one lost its item
    actions = jnp.asarray([5, 0], jnp.int32)  # the player switches to slot 1 (no item)
    speeds, possible = _end_speeds(tables, s, actions, 0, False, False, False, False)
    point = (0 + 6) * 2  # stage 0, not paralyzed
    assert bool(possible[point])
    fresh = s._replace(active=s.active.at[0].set(1), can_unburden=s.can_unburden.at[0].set(0))
    assert int(speeds[point]) == int(effective_speed(tables, fresh, 0))


def test_trapping_abilities(tables, names):
    """B9-11: Battler_IsTrapped (battle_lib.c:5514-5536): Shadow Tag unless the battler has it,
    Arena Trap on a grounded battler (Flying and Levitate are not, Gravity grounds them), Magnet
    Pull on Steel only, and only a living trapper."""
    from nuzline.engine.actions import trapped

    n = names

    def pair(mine: str, foe_ability):
        s = make_state(tables, [_mon(n, mine, ["Growl"], 30)], [_mon(n, "Geodude", ["Growl"], 30)])
        return _set_ability(_set_ability(s, 1, foe_ability), 0, AbilityId.NONE)

    tag = pair("Bidoof", AbilityId.SHADOW_TAG)
    assert bool(trapped(tag, 0))
    assert not bool(trapped(_set_ability(tag, 0, AbilityId.SHADOW_TAG), 0))
    assert not bool(trapped(tag._replace(hp=tag.hp.at[1, 0].set(0)), 0))  # a fainted trapper
    arena = pair("Bidoof", AbilityId.ARENA_TRAP)
    assert bool(trapped(arena, 0))
    assert not bool(trapped(_set_ability(arena, 0, AbilityId.LEVITATE), 0))
    bird = pair("Pidgey", AbilityId.ARENA_TRAP)
    assert not bool(trapped(bird, 0))
    assert bool(trapped(bird._replace(gravity=jnp.int8(5)), 0))
    assert bool(trapped(pair("Magnemite", AbilityId.MAGNET_PULL), 0))
    assert not bool(trapped(pair("Bidoof", AbilityId.MAGNET_PULL), 0))


def test_belly_drum_fails_at_plus_six_and_at_half_hp(tables, names):
    """B9-10: subscript_belly_drum.s:3-6: Attack +6, or curHP <= Divide(maxHP, 2), fails."""
    from nuzline.engine.ids import Stat

    n = names
    s = make_state(tables, [_mon(n, "Snorlax", ["Belly Drum"], 30)], [_mon(n, "Geodude", ["Harden"], 30)])
    top = s.with_stages(0, atk=6)
    out = assert_matches_reference(tables, top, (0, 0), O)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    half = s._replace(hp=s.hp.at[0, 0].set(int(s.max_hp[0, 0]) // 2))
    out = assert_matches_reference(tables, half, (0, 0), O)
    assert int(out.hp[0, 0]) == int(half.hp[0, 0]) and int(out.stages[0, Stat.ATK]) == 0


def test_chlorophyll_moves_first_in_sun(tables, names):
    """B9-3: Chlorophyll doubles speed in sun (battle_lib.c:1228-1238): the slower Oddish KOs the
    1-HP Pidgey before it attacks."""
    n = names
    s = make_state(tables, [_mon(n, "Oddish", ["Absorb"], 30)], [_mon(n, "Pidgey", ["Tackle"], 30)])
    s = _set_ability(_set_ability(s, 0, AbilityId.CHLOROPHYLL), 1, AbilityId.NONE)
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    plain = assert_matches_reference(tables, s, (0, 0), O)
    assert int(plain.hp[0, 0]) < int(s.hp[0, 0])
    sunny = s._replace(weather=jnp.int8(Weather.SUN))
    out = assert_matches_reference(tables, sunny, (0, 0), O)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.hp[1, 0]) == 0


def _dealt(tables, s, actions=(0, 0)):
    out = assert_matches_reference(tables, s, actions, O)
    return int(s.hp[1, 0]) - int(out.hp[1, 0])


def test_swarm_and_flower_gift_raise_the_hit(tables, names):
    """B9-3: Swarm x1.5 power on a Bug move at curHP <= maxHP / 3 (battle_lib.c:6834-6838); Flower
    Gift x1.5 Attack in sun for its holder's side (:6930-6941)."""
    n = names
    s = make_state(tables, [_mon(n, "Kricketot", ["Bug Bite"], 30)], [_mon(n, "Snorlax", ["Leer"], 40)])
    s = _set_ability(_set_ability(s, 0, AbilityId.SWARM), 1, AbilityId.NONE)
    low = s._replace(hp=s.hp.at[0, 0].set(max(int(s.max_hp[0, 0]) // 3, 1)))
    assert _dealt(tables, low) > _dealt(tables, s)
    f = make_state(tables, [_mon(n, "Cherrim", ["Tackle"], 30)], [_mon(n, "Snorlax", ["Leer"], 40)])
    f = _set_ability(f, 1, AbilityId.NONE)._replace(weather=jnp.int8(Weather.SUN))
    gift = _set_ability(f, 0, AbilityId.FLOWER_GIFT)
    assert _dealt(tables, gift) > _dealt(tables, _set_ability(f, 0, AbilityId.NONE))


def test_water_absorb_rain_dish_aftermath_sonicboom(tables, names):
    """B9-6..B9-9: Water Absorb heals Divide(maxHP, 4) from a Water move; Rain Dish
    Divide(maxHP, 16) in rain at the turn end; Aftermath takes Divide(maxHP, 4) from the contact
    attacker that KOs its holder; SonicBoom deals 20."""
    n = names
    s = make_state(tables, [_mon(n, "Psyduck", ["Water Gun"], 30)], [_mon(n, "Lotad", ["Leer"], 30)])
    s = _set_ability(_set_ability(s, 1, AbilityId.WATER_ABSORB), 0, AbilityId.NONE)
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[1, 0]) == 1 + max(int(s.max_hp[1, 0]) // 4, 1)
    r = make_state(tables, [_mon(n, "Lotad", ["Leer"], 30)], [_mon(n, "Geodude", ["Leer"], 30)])
    r = _set_ability(_set_ability(r, 0, AbilityId.RAIN_DISH), 1, AbilityId.NONE)
    r = r._replace(weather=jnp.int8(Weather.RAIN), weather_timer=jnp.int8(0))
    r = r._replace(hp=r.hp.at[0, 0].set(1))
    out = assert_matches_reference(tables, r, (0, 0), O)
    assert int(out.hp[0, 0]) == 1 + max(int(r.max_hp[0, 0]) // 16, 1)
    a = make_state(tables, [_mon(n, "Bidoof", ["Tackle"], 30)], [_mon(n, "Stunky", ["Leer"], 30)])
    a = _set_ability(_set_ability(a, 1, AbilityId.AFTERMATH), 0, AbilityId.NONE)
    a = a._replace(hp=a.hp.at[1, 0].set(1))
    out = assert_matches_reference(tables, a, (0, 0), O)
    assert int(out.hp[1, 0]) == 0
    assert int(a.hp[0, 0]) - int(out.hp[0, 0]) == max(int(a.max_hp[0, 0]) // 4, 1)
    b = make_state(tables, [_mon(n, "Magnemite", ["SonicBoom"], 30)], [_mon(n, "Snorlax", ["Leer"], 40)])
    assert _dealt(tables, b) == 20


def test_hyper_cutter_keeps_attack_from_intimidate(tables, names):
    """B9-6: subscript_intimidate.s runs BtlCmd_ChangeStatStage, where Hyper Cutter blocks an
    Attack drop (battle_script.c:2884-2902)."""
    from nuzline.engine.ids import Stat

    n = names
    s = make_state(
        tables,
        [_mon(n, "Krabby", ["Leer"], 30)],
        [_mon(n, "Geodude", ["Leer"], 30), _mon(n, "Growlithe", ["Leer"], 30)],
    )
    s = _set_ability(s, 1, AbilityId.INTIMIDATE, 1)
    plain = assert_matches_reference(tables, _set_ability(s, 0, AbilityId.NONE), (0, 5), O)
    assert int(plain.stages[0, Stat.ATK]) == -1
    cutter = assert_matches_reference(tables, _set_ability(s, 0, AbilityId.HYPER_CUTTER), (0, 5), O)
    assert int(cutter.stages[0, Stat.ATK]) == 0


def test_kaizo_recoil_divisors_in_battle(kaizo_tables, kaizo_names):
    """B9-4: Kaizo's effect 48 (Hyper Beam) recoils Divide(hitDamage, 2), its 182 (Take Down)
    Divide(hitDamage, 4)."""
    t, n = kaizo_tables, kaizo_names
    for move, divisor in (("Hyper Beam", 2), ("Take Down", 4)):
        s = make_state(t, [_mon(n, "Snorlax", [move], 40)], [_mon(n, "Snorlax", ["Leer"], 40)])
        s = _set_ability(_set_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)
        out = assert_matches_reference(t, s, (0, 0), O)
        dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
        assert dealt > 0, move
        assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == max(dealt // divisor, 1), move
