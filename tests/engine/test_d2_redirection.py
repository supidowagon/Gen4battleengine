"""D2-12: Lightning Rod and Storm Drain redirection (``BattleSystem_CheckRedirectionAbilities``,
lib.c:1732-1790): ``targets.redirect_type`` and ``targets.redirect`` at B = 4 with explicit speed
orders, held equal to the reference."""

from __future__ import annotations

import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.damage import hidden_power
from nuzline.engine.hooks import REDIRECTED, reveal_ability
from nuzline.engine.ids import MOVE_STRUGGLE, AbilityId, Stat, Type, Weather
from nuzline.engine.state import make_state
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, mon, with_hp

NONE = targets.NO_DEFENDER
ORDER = [0, 1, 2, 3]
LR, SD = AbilityId.LIGHTNINGROD, AbilityId.STORM_DRAIN


def _slot(s, b):
    return int(targets.party(targets.format_of(s), b)), int(s.active[b])


def ability(s, b, value):
    p, slot = _slot(s, b)
    return s._replace(ability=s.ability.at[p, slot].set(int(value)))


def item(s, b, value):
    p, slot = _slot(s, b)
    return s._replace(item=s.item.at[p, slot].set(value))


def type_of(tables, s, move, names, attacker=0):
    mid = names.move(move) if isinstance(move, str) else move
    got = int(targets.redirect_type(tables, s, attacker, mid))
    want = ref.redirect_type(to_numpy(tables), to_ref(s), attacker, mid)
    assert got == int(want), (move, got, want)
    return got


def red(tables, names, s, attacker, move, defender, order=ORDER):
    mid = names.move(move)
    got, flag = targets.redirect(tables, s, attacker, mid, defender, np.asarray(order))
    want = ref.redirect(
        to_numpy(tables), to_ref(s), attacker, mid, None if defender == NONE else defender, order
    )
    want = (NONE if want[0] is None else want[0], want[1])
    assert (int(got), bool(flag)) == want, (move, attacker, defender, (int(got), bool(flag)), want)
    return want


def _electric_ivs():
    stats = [Stat.HP, Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE]
    for mask in range(64):
        ivs = np.zeros(6, np.int8)
        for k, stat in enumerate(stats):
            ivs[int(stat)] = 30 | ((mask >> k) & 1)
        if int(hidden_power(ivs)[0]) == Type.ELECTRIC:
            return ivs
    raise AssertionError("no Electric Hidden Power IV pattern")


def test_redirect_type(tables, names):
    s = four(tables, names)
    assert type_of(tables, s, "Thunderbolt", names) == Type.ELECTRIC
    assert type_of(tables, s, MOVE_STRUGGLE, names) == Type.NORMAL
    assert type_of(tables, s, "Water Gun", names) == Type.WATER
    p, slot = _slot(s, 0)
    hp = s._replace(ivs=s.ivs.at[p, slot].set(_electric_ivs()))
    assert type_of(tables, hp, "Hidden Power", names) == Type.ELECTRIC
    # Natural Gift with a Water berry
    water = int(
        ((tables.item_gift_power > 0) & (tables.item_gift_type == Type.WATER)).nonzero()[0][0]
    )
    assert type_of(tables, item(s, 0, water), "Natural Gift", names) == Type.WATER
    # Judgment with a Zap Plate; without a plate Normal
    assert type_of(tables, item(s, 0, names.item("Zap Plate")), "Judgment", names) == Type.ELECTRIC
    assert type_of(tables, s, "Judgment", names) == Type.NORMAL
    # Weather Ball in rain; with no weather the data type (Normal)
    assert type_of(tables, s._replace(weather=Weather.RAIN), "Weather Ball", names) == Type.WATER
    assert type_of(tables, s, "Weather Ball", names) == Type.NORMAL


def test_electric_move_goes_to_a_lightning_rod_holder(tables, names):
    s = ability(four(tables, names), 2, LR)
    assert red(tables, names, s, 1, "Thunderbolt", 0) == (2, True)  # a foe's move at the left one
    assert red(tables, names, s, 0, "Thunderbolt", 1) == (2, True)  # the attacker's own partner
    assert red(tables, names, s, 0, "Thunderbolt", NONE) == (NONE, False)
    assert red(tables, names, s, 0, "Tackle", 1) == (1, False)


def test_first_holder_in_the_speed_order(tables, names):
    s = ability(ability(four(tables, names), 2, LR), 3, LR)
    assert red(tables, names, s, 0, "Thunderbolt", 1, order=[3, 0, 1, 2]) == (3, True)
    assert red(tables, names, s, 0, "Thunderbolt", 1, order=[2, 3, 1, 0]) == (2, True)


def test_no_change_cases(tables, names):
    s = ability(four(tables, names), 2, LR)
    # holder == defender: no change, no flag
    one = ability(four(tables, names), 1, LR)
    assert red(tables, names, one, 0, "Thunderbolt", 1) == (1, False)
    # Mold Breaker attacker
    assert red(tables, names, ability(s, 0, AbilityId.MOLD_BREAKER), 0, "Thunderbolt", 1) == (
        1,
        False,
    )
    # a living Follow Me user on the target side
    fm = s._replace(ts_follow_me=s.ts_follow_me.at[1].set(4))
    assert red(tables, names, fm, 0, "Thunderbolt", 3) == (3, False)
    # spread moves are not single-target
    assert red(tables, names, s, 0, "Discharge", 1) == (1, False)
    sd = ability(four(tables, names), 2, SD)
    assert red(tables, names, sd, 0, "Surf", 1) == (1, False)
    # a fainted holder
    assert red(tables, names, with_hp(s, 2, 0), 0, "Thunderbolt", 1) == (1, False)


def test_storm_drain_takes_water_moves(tables, names):
    s = ability(four(tables, names), 2, SD)
    assert red(tables, names, s, 1, "Water Gun", 0) == (2, True)
    assert red(tables, names, s, 1, "Thunderbolt", 0) == (0, False)  # not its type
    both = ability(s, 3, LR)
    assert red(tables, names, both, 0, "Water Gun", 1) == (2, True)
    assert red(tables, names, both, 0, "Thunderbolt", 1) == (3, True)


def test_singles_never_redirects(tables, names):
    s = ability(make_state(tables, [mon(names)], [mon(names)]), 1, LR)
    for defender in (1, NONE):  # the resolver never gives the user to a single-target Electric move
        assert red(tables, names, s, 0, "Thunderbolt", defender, order=[0, 1])[1] is False
    s = ability(s, 1, SD)
    assert red(tables, names, s, 0, "Water Gun", 1, order=[0, 1]) == (1, False)


def test_reveal_names_the_player_holder(tables, names):
    for holder, decomp in ((LR, 31), (SD, 114)):
        s = ability(four(tables, names), 2, holder)
        slot = int(s.active[2])
        out = reveal_ability(s, 2, slot, REDIRECTED, True)
        assert int(out.ai_ability_known[slot]) == decomp
        r = to_ref(s)
        ref._reveal_redirect(r, to_numpy(tables), 2)
        assert r.ai_ability_known[slot] == decomp
        off = reveal_ability(s, 2, slot, REDIRECTED, False)
        assert np.array_equal(off.ai_ability_known, s.ai_ability_known)
    # a foe's holder is not tracked (the AI only follows the player's Pokemon)
    foe = ability(four(tables, names), 3, LR)
    out = reveal_ability(foe, 3, int(foe.active[3]), REDIRECTED, True)
    assert np.array_equal(out.ai_ability_known, foe.ai_ability_known)


def test_storm_drain_is_a_supported_ability():
    from nuzline.engine.coverage import IMPLEMENTED_ABILITIES
    from nuzline.engine.ids import ABILITY_DECOMP_IDS

    assert AbilityId.STORM_DRAIN in IMPLEMENTED_ABILITIES
    assert ABILITY_DECOMP_IDS[AbilityId.STORM_DRAIN] == 114
