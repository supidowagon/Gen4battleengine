"""Intimidate, pinch abilities, Rivalry, Oran and Sitrus Berries (decomp citations in
``hooks.py``)."""

from nuzline.engine.hooks import after_damage_items
from nuzline.engine.ids import AbilityId, ItemId, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import step
from tests.engine.helpers import A, roark_state


def test_intimidate_on_lead_and_on_switch_in(tables, names):
    s = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert int(s.stages[1, 1]) == -1  # make_state runs on_switch_in for leads
    out = step(tables, s, A(4 + 1, "Tackle", s, names), empty_outcome())
    assert int(out.stages[1, 1]) == -1  # Prinplup (slot 1) has no Intimidate: unchanged


def test_intimidate_on_a_switch_action(tables, names):
    s = roark_state(tables, names)  # Monferno leads; Shinx (slot 4) gets Intimidate
    s = s._replace(ability=s.ability.at[0, 4].set(AbilityId.INTIMIDATE))
    assert int(s.stages[1, 1]) == 0
    out = step(tables, s, A(4 + 4, "Leer", s, names), empty_outcome())
    assert int(out.stages[1, 1]) == -1


def test_blaze_boosts_fire_at_third_hp(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos")
    low = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 3)
    normal = step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    boosted = step(tables, low, A("Ember", "Leer", low, names), empty_outcome())
    assert (int(s.hp[1, 0]) - int(boosted.hp[1, 0])) > (int(s.hp[1, 0]) - int(normal.hp[1, 0]))


def test_oran_berry_triggers_and_is_consumed(tables, names):
    s = roark_state(tables, names, player_species="Starly")  # holds Oran Berry
    half = int(s.max_hp[0, 0]) // 2 + 1
    s = s.replace_hp(0, 0, half)
    out = step(tables, s, A("Growl", "Tackle", s, names), empty_outcome())
    dealt = half - (int(out.hp[0, 0]) - 10)
    assert (
        dealt > 0
        and int(out.item[0, 0]) == names.item("")
        and int(out.item_consumed[0, 0]) == names.item("Oran Berry")
    )


def test_sitrus_heals_quarter(tables, names):
    s = roark_state(tables, names, player_species="Grotle")  # holds Sitrus Berry
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2 + 1)
    out = step(tables, s, A("Withdraw", "Tackle", s, names), empty_outcome())
    assert int(out.item[0, 0]) == names.item("")
    assert int(out.hp[0, 0]) > int(s.hp[0, 0]) - (int(s.max_hp[0, 0]) // 4)


def test_berry_item_effects_come_from_the_tables(tables, names):
    assert int(tables.item_effect[names.item("Oran Berry")]) == ItemId.ORAN_BERRY
    assert int(tables.item_effect[names.item("Sitrus Berry")]) == ItemId.SITRUS_BERRY


def test_berry_does_not_trigger_above_half(tables, names):
    s = roark_state(tables, names, player_species="Starly")
    out = step(tables, s, A("Growl", "Leer", s, names), empty_outcome())
    assert int(out.item[0, 0]) == names.item("Oran Berry")
    assert int(out.item_consumed[0, 0]) == 0


def test_berry_triggers_at_end_of_turn_after_poison(tables, names):
    # MON_COND_CHECK_STATE_ITEM_CONDITION (battle_controller_player.c:1699) runs after
    # the poison check: poison takes Starly to half, then the Oran Berry heals 10.
    s = roark_state(tables, names, player_species="Starly")
    max_hp = int(s.max_hp[0, 0])
    start = max_hp // 2 + 1
    s = s.replace_hp(0, 0, start).with_status(0, 0, Status.POISON)
    out = step(tables, s, A("Growl", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == min(start - max(max_hp // 8, 1) + 10, max_hp)
    assert int(out.item[0, 0]) == 0


def test_sitrus_berry_triggers_after_stealth_rock_on_switch_in(tables, names):
    # Stealth Rock (subscript_hazards_check.s), then SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS
    # (battle_lib.c:4151): Grotle (Grass, neutral to Rock) drops to half and heals 1/4.
    s = roark_state(tables, names)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    max_hp = int(s.max_hp[0, 2])
    start = max_hp // 2 + 1
    s = s.replace_hp(0, 2, start)
    out = step(tables, s, A(4 + 2, "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 2]) == start - max_hp // 8 + max_hp * 25 // 100
    assert int(out.item[0, 2]) == 0 and int(out.item_consumed[0, 2]) == names.item("Sitrus Berry")


def test_rivalry_same_gender_boost(tables, names):
    male = roark_state(
        tables,
        names,
        player_species="Shinx",
        player_ability="Rivalry",
        player_gender="male",
        opp_species="Cranidos",
        opp_gender="male",
    )
    female = roark_state(
        tables,
        names,
        player_species="Shinx",
        player_ability="Rivalry",
        player_gender="male",
        opp_species="Cranidos",
        opp_gender="female",
    )
    a = step(tables, male, A("Tackle", "Leer", male, names), empty_outcome())
    b = step(tables, female, A("Tackle", "Leer", female, names), empty_outcome())
    assert (int(male.hp[1, 0]) - int(a.hp[1, 0])) > (int(female.hp[1, 0]) - int(b.hp[1, 0]))


def test_after_damage_items_cond_gates_the_check(tables, names):
    # ``cond`` is step's ``connected`` gate on the after-move item check
    # (AFTER_MOVE_EFFECT_ATTACKER_ITEM / _DEFENDER_ITEM, battle_controller_player.c:3684,
    # :3691): with cond=False nothing changes; the default (True) eats the berry.
    s = roark_state(tables, names, player_species="Starly")  # holds Oran Berry
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2)
    gated = after_damage_items(tables, s, 0, cond=False)
    assert int(gated.hp[0, 0]) == int(s.hp[0, 0])
    assert int(gated.item[0, 0]) == names.item("Oran Berry")
    eaten = after_damage_items(tables, s, 0)
    assert int(eaten.hp[0, 0]) == int(s.hp[0, 0]) + 10
    assert int(eaten.item_consumed[0, 0]) == names.item("Oran Berry")
