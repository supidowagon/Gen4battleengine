"""E1 batch F (rulings F-n, docs/kaizo/e1-mechanics-ledger.md): Heatproof, Flash Fire, Multitype,
Intimidate's substitute skip, Hydration, Forecast, the held items of hold effects 36, 41, 42, 50,
92, 0, 1 and 12, the party friendship wrap and Shaymin-Sky's revert, each case checked against the
reference engine."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table, personal_types
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.hooks import battle_start
from nuzline.engine.ids import (
    SPECIES_FARFETCHD,
    STARF_DRAWS,
    AbilityId,
    ItemId,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.order import effective_speed
from nuzline.engine.outcome import CRIT_ODDS_EFFECTS, crit_variants, empty_outcome
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_e1_batch_e import _bag, _use_both
from tests.engine.test_pivot import _mon, _resume
from tests.engine.test_protect import _state

O = empty_outcome()


def _ability(s, side, ability):
    """``ability`` on ``side``'s active slot."""
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _both(tables, s, move):
    """The 16 damage rolls of ``move`` from side 0, JAX and reference (asserted equal)."""
    got = damage_table(tables, s, 0, move, False).tolist()
    assert got == ref_damage_table(to_numpy(tables), to_ref(s), 0, move, False)
    return got


# --- Heatproof (F-1) -----------------------------------------------------------------------


def test_heatproof_is_covered():
    assert IMPLEMENTED_ABILITIES[AbilityId.HEATPROOF] == Support.EXACT


def test_heatproof_halves_fire_power_like_thick_fat_and_mold_breaker_ignores_it(tables, names):
    """battle_lib.c:6841-6844 halves a Fire move's power for a Heatproof defender
    (Battler_IgnorableAbility), the same cut as Thick Fat's Fire half (:6785-6788); a Water move
    is untouched; a Mold Breaker attacker ignores it."""
    s = roark_state(tables, names, player_moves=["Ember", "Water Gun"])
    ember, water = names.move("Ember"), names.move("Water Gun")
    bare = _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)
    heat = _ability(bare, 1, AbilityId.HEATPROOF)
    thick = _ability(bare, 1, AbilityId.THICK_FAT)
    assert _both(tables, heat, ember) == _both(tables, thick, ember)
    assert max(_both(tables, heat, ember)) < max(_both(tables, bare, ember))
    assert _both(tables, heat, water) == _both(tables, bare, water)
    breaker = _ability(heat, 0, AbilityId.MOLD_BREAKER)
    assert _both(tables, breaker, ember) == _both(
        tables, _ability(bare, 0, AbilityId.MOLD_BREAKER), ember
    )


@pytest.mark.parametrize(
    ("max_hp", "loss"),
    [(40, 2), (9, 1)],  # Divide(40, 8) = 5, then Divide(5, 2) = 2; Divide(9, 8) = 1, then 1 (floor)
)
def test_heatproof_halves_burn_damage(tables, names, max_hp, loss):
    """subscript_burn_damage.s:6-9: Divide(maxHP, 8), then Divide(loss, 2) for Heatproof
    (BattleSystem_Divide floors at 1, battle_lib.c:3599-3618); other holders lose the eighth."""
    s = _state(tables, names, mine=["Protect"])
    s = s._replace(hp=s.hp.at[1, 0].set(max_hp), max_hp=s.max_hp.at[1, 0].set(max_hp))
    s = s.with_status(1, 0, Status.BURN)
    acts = A("Protect", "Tackle", s, names)
    plain = assert_matches_reference(tables, _ability(s, 1, AbilityId.NONE), acts, O)
    assert int(plain.hp[1, 0]) == max_hp - max(max_hp // 8, 1)
    heat = assert_matches_reference(tables, _ability(s, 1, AbilityId.HEATPROOF), acts, O)
    assert int(heat.hp[1, 0]) == max_hp - loss


# --- Flash Fire (F-2) ----------------------------------------------------------------------


def _flash(tables, names, moves=("Ember", "Will-O-Wisp")):
    """Monferno (acts first) against a Flash Fire Geodude."""
    s = _state(tables, names, mine=list(moves))
    return _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.FLASH_FIRE)


def test_flash_fire_is_covered_and_its_flag_is_volatile_only():
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    assert IMPLEMENTED_ABILITIES[AbilityId.FLASH_FIRE] == Support.EXACT
    assert "flash_fire" in VOLATILE_FIELDS and "flash_fire" not in BATON_PASSED_FIELDS


def test_a_fire_move_is_absorbed_and_sets_the_flag(tables, names):
    """battle_lib.c:3515-3521 + subscript_absorb_and_boost_fire_type_moves.s:13: no damage, no
    heal, flashFire := 1; the hit is taken whole (NO_MORE_WORK), so no burn from Ember either."""
    s = _flash(tables, names)
    out = assert_matches_reference(tables, s, A("Ember", "Tackle", s, names), O)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) and int(out.flash_fire[1]) == 1
    assert int(out.flash_fire[0]) == 0 and int(out.status[1, 0]) == Status.NONE


def test_will_o_wisp_is_absorbed_though_it_has_no_power(tables, names):
    s = _flash(tables, names)
    acts = A("Will-O-Wisp", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.flash_fire[1]) == 1 and int(out.status[1, 0]) == Status.NONE
    bare = assert_matches_reference(tables, _ability(s, 1, AbilityId.NONE), acts, O)
    assert int(bare.status[1, 0]) == Status.BURN and int(bare.flash_fire[1]) == 0


def test_a_frozen_holder_and_a_mold_breaker_attacker_are_not_absorbing(tables, names):
    """:3517 (frozen holder takes the hit), :3108-3123 (Battler_IgnorableAbility)."""
    s = _flash(tables, names)
    acts = A("Ember", "Tackle", s, names)
    for t in (s.with_status(1, 0, Status.FREEZE), _ability(s, 0, AbilityId.MOLD_BREAKER)):
        out = assert_matches_reference(tables, t, acts, O)
        assert int(out.flash_fire[1]) == 0 and int(out.hp[1, 0]) < int(t.hp[1, 0])


def test_the_flag_boosts_fire_moves_only_and_outlives_the_ability(tables, names):
    """battle_lib.c:7074-7076: x15/10 on the attacker's Fire damage from the flag alone."""
    s = _state(tables, names, mine=["Ember", "Water Gun"])
    s = _ability(_ability(s, 0, AbilityId.NONE), 1, AbilityId.NONE)
    ember, water = names.move("Ember"), names.move("Water Gun")
    flagged = s._replace(flash_fire=s.flash_fire.at[0].set(1))
    assert max(_both(tables, flagged, ember)) > max(_both(tables, s, ember))
    assert _both(tables, flagged, water) == _both(tables, s, water)
    # The other side's flag does not boost this attacker.
    other = s._replace(flash_fire=s.flash_fire.at[1].set(1))
    assert _both(tables, other, ember) == _both(tables, s, ember)


# --- Multitype (F-3) -----------------------------------------------------------------------

PLATES = {"Flame Plate": Type.FIRE, "Splash Plate": Type.WATER, "Iron Plate": Type.STEEL}


def test_multitype_is_covered_and_arceus_has_it(tables, names):
    assert IMPLEMENTED_ABILITIES[AbilityId.MULTITYPE] == Support.EXACT
    row = int(tables.species_form_row[names.species("Arceus"), 0])
    assert int(tables.species_abilities[row, 0]) == AbilityId.MULTITYPE


def test_an_arceus_has_its_raw_plates_type_whatever_its_form(tables, names):
    """Battler_MonType (battle_lib.c:7618-7703) / MON_DATA_TYPE_1/2 (pokemon.c:1219-1226): both
    types of the HOLD_EFFECT_ARCEUS_* of the held item, Normal for any other item or none; the
    form's row types are ignored; other species keep theirs."""
    arceus = names.species("Arceus")
    for form in (0, 7):
        for plate, kind in PLATES.items():
            got = personal_types(tables, arceus, form, 0, 0, names.item(plate)).tolist()
            assert got == [kind, kind], (form, plate)
        for item in (0, names.item("Lum Berry"), names.item("Charcoal")):
            assert personal_types(tables, arceus, form, 0, 0, item).tolist() == [Type.NORMAL] * 2
    bidoof = names.species("Bidoof")
    row = int(tables.species_form_row[bidoof, 0])
    assert (
        personal_types(tables, bidoof, 0, 0, 0, names.item("Flame Plate")).tolist()
        == tables.species_types[row].tolist()
    )


def _transform_into(tables, names, *, plate, foe_types=None):
    """Monferno (holding ``plate``) Transforms into a Multitype Arceus."""
    s = _state(tables, names, mine=["Transform"])
    s = _ability(s, 0, AbilityId.NONE)
    s = s._replace(
        species=s.species.at[1, 0].set(names.species("Arceus")),
        ability=s.ability.at[1, 0].set(AbilityId.MULTITYPE),
        item=s.item.at[0, 0].set(names.item(plate) if plate else 0),
    )
    if foe_types is not None:
        s = s._replace(types=s.types.at[1, 0].set(jnp.asarray(foe_types, s.types.dtype)))
    return assert_matches_reference(tables, s, A("Transform", "Tackle", s, names), O)


def test_a_transform_into_a_multitype_arceus_takes_the_users_own_plate_type(tables, names):
    """The copy gives the user species Arceus and Multitype, so Battler_MonType reads its own
    heldItem (battle_lib.c:7629-7699), not the target's types."""
    out = _transform_into(tables, names, plate="Flame Plate", foe_types=[Type.WATER] * 2)
    assert out.types[0, 0].tolist() == [Type.FIRE] * 2
    out = _transform_into(tables, names, plate="", foe_types=[Type.WATER] * 2)
    assert out.types[0, 0].tolist() == [Type.NORMAL] * 2


def test_a_transform_into_another_species_still_copies_its_types(tables, names):
    s = _state(tables, names, mine=["Transform"])
    s = s._replace(types=s.types.at[1, 0].set(jnp.asarray([Type.WATER] * 2, s.types.dtype)))
    out = assert_matches_reference(tables, s, A("Transform", "Tackle", s, names), O)
    assert out.types[0, 0].tolist() == [Type.WATER] * 2


# --- Intimidate and a substitute the action hit (F-4) --------------------------------------


def _u_turn_into_intimidate(tables, names, substitute):
    """Scyther's U-turn hits Bulbasaur (a 1 HP substitute up, or none), then Gyarados comes in."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Gyarados", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 35)],
    )
    if substitute:
        s = s._replace(substitute_hp=s.substitute_hp.at[1].set(1))
    paused = assert_matches_reference(tables, s, (0, 0), O)
    return _resume(tables, paused, 1)


def test_a_u_turn_newcomer_skips_a_foe_whose_substitute_the_action_hit(tables, names):
    """subscript_intimidate.s:13 / BtlCmd_CheckSubstitute (battle_script.c:9172-9184): the flag
    SELF_TURN_FLAG_SUBSTITUTE_HIT (bcp.c:3359-3371) outlives the broken substitute until the
    action ends, so the foe keeps its Attack; with no substitute it drops one stage."""
    ATK = Stat.ATK
    plain = _u_turn_into_intimidate(tables, names, substitute=False)
    assert int(plain.stages[1, ATK]) == -1
    broken = _u_turn_into_intimidate(tables, names, substitute=True)
    assert int(broken.substitute_hp[1]) == 0 and int(broken.stages[1, ATK]) == 0
    assert np.asarray(broken.ts_sub_hit).tolist() == [0, 0]  # the action ended


# --- Hydration (F-5) and Forecast (F-6) ----------------------------------------------------


def _become(tables, names, s, side, species, ability_slot=0):
    """``side``'s active slot is ``species`` with its ``ability_slot`` ability and personal types
    (the decomp ability is read from the species table, ``damage.decomp_abilities``)."""
    row = int(tables.species_form_row[names.species(species), 0])
    slot = int(s.active[side])
    return s._replace(
        species=s.species.at[side, slot].set(names.species(species)),
        ability_slot=s.ability_slot.at[side, slot].set(ability_slot),
        ability_form=s.ability_form.at[side, slot].set(0),
        ability=s.ability.at[side, slot].set(int(tables.species_abilities[row, ability_slot])),
        types=s.types.at[side, slot].set(jnp.asarray(tables.species_types[row], s.types.dtype)),
    )


def _weather(s, weather, timer=0):
    return s._replace(
        weather=jnp.asarray(weather, s.weather.dtype),
        weather_timer=jnp.asarray(timer, s.weather_timer.dtype),
    )


def _hydration(tables, names, status, weather=Weather.RAIN, timer=0, foe="Phione"):
    """Monferno protects; ``foe`` (Phione: Hydration) has ``status``, in ``weather``."""
    s = _state(tables, names, mine=["Protect"])
    s = _become(tables, names, s, 1, foe)
    s = s.with_status(1, 0, status)
    if status in (Status.TOXIC, Status.SLEEP):
        s = s._replace(status_counter=s.status_counter.at[1, 0].set(2))
    s = _weather(s, weather, timer)
    return s, assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), O)


def test_hydration_and_forecast_are_covered_and_their_species_have_them(tables, names):
    for ability, species in ((AbilityId.HYDRATION, "Phione"), (AbilityId.FORECAST, "Castform")):
        assert IMPLEMENTED_ABILITIES[ability] == Support.EXACT
        row = int(tables.species_form_row[names.species(species), 0])
        assert int(tables.species_abilities[row, 0]) == ability


@pytest.mark.parametrize("status", [Status.BURN, Status.POISON, Status.TOXIC, Status.SLEEP])
def test_rain_cures_a_hydration_holders_status_before_the_turns_damage(tables, names, status):
    """battle_script.c:5626-5639 + subscript_ability_restore_status.s:5: status and its counter
    go at the end of turn; the weather step precedes the poison / burn damage (bcp.c:1113-1136,
    :1252), so the holder takes none."""
    s, out = _hydration(tables, names, status)
    assert int(out.status[1, 0]) == Status.NONE and int(out.status_counter[1, 0]) == 0
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_hydration_needs_rain_that_goes_on_and_no_cloud_nine_and_the_ability(tables, names):
    """The rain block (:5613) inside NO_CLOUD_NINE (:5573); timed rain that runs out this turn
    cures nothing (bcp.c:1121-1128); other weathers and abilities do not cure."""
    burn = Status.BURN
    assert int(_hydration(tables, names, burn, Weather.SUN)[1].status[1, 0]) == burn
    assert int(_hydration(tables, names, burn, Weather.RAIN, timer=1)[1].status[1, 0]) == burn
    assert int(_hydration(tables, names, burn, Weather.RAIN, timer=2)[1].status[1, 0]) == 0
    assert int(_hydration(tables, names, burn, foe="Geodude")[1].status[1, 0]) == burn
    s = _hydration(tables, names, burn)[0]
    s = _become(tables, names, s, 0, "Golduck", ability_slot=1)  # Cloud Nine
    out = assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), O)
    assert int(out.status[1, 0]) == burn


@pytest.mark.parametrize(
    ("weather", "kind"),
    [
        (Weather.SUN, Type.FIRE),
        (Weather.RAIN, Type.WATER),
        (Weather.HAIL, Type.ICE),
        (Weather.SANDSTORM, Type.NORMAL),
        (Weather.NONE, Type.NORMAL),
    ],
)
def test_forecast_takes_the_weathers_type_at_the_turn_start(tables, names, weather, kind):
    """battle_lib.c:6318-6352: Fire / Water / Ice for sun / rain / hail, Normal for none (sandstorm
    counts as none); the turn start's ShowBattleMon runs it (bcp.c:256); types only."""
    s = _state(tables, names, mine=["Protect"])
    s = _become(tables, names, s, 1, "Castform")
    start = Type.WATER if kind != Type.WATER else Type.FIRE
    s = _weather(
        s._replace(types=s.types.at[1, 0].set(jnp.asarray([start] * 2, s.types.dtype))), weather
    )
    out = assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), O)
    assert out.types[1, 0].tolist() == [kind] * 2
    assert int(out.form[1, 0]) == int(s.form[1, 0])  # the party's form is not written (:6446)


def test_a_weather_move_changes_forecast_before_the_next_battler_acts(tables, names):
    """AfterMoveEffects / MoveEnd's TriggerEffectOnSwitch (bcp.c:3672-3680, :3973): Rain Dance
    makes a Castform Water at once (the type Ember then meets is resisted)."""
    s = _state(tables, names, mine=["Rain Dance"], order="player")
    s = _become(tables, names, s, 1, "Castform")
    out = assert_matches_reference(tables, s, A("Rain Dance", "Tackle", s, names), O)
    assert out.types[1, 0].tolist() == [Type.WATER] * 2 and int(out.weather) == Weather.RAIN


def test_an_alive_cloud_nine_forces_normal_and_a_lead_castform_changes_at_battle_start(
    tables, names
):
    """:6355-6363 (Cloud Nine / Air Lock: Normal whatever the weather); the battle start's
    ShowBattleMon runs the form change after the weather (:3662, :3739; hooks.battle_start)."""
    s = _state(tables, names, mine=["Protect"])
    s = _weather(_become(tables, names, s, 1, "Castform"), Weather.RAIN)
    started = battle_start(tables, s)
    assert started.types[1, 0].tolist() == [Type.WATER] * 2
    locked = battle_start(tables, _become(tables, names, s, 0, "Golduck", ability_slot=1))
    assert locked.types[1, 0].tolist() == [Type.NORMAL] * 2


# --- Held-item hold effects (F-7 .. F-14) --------------------------------------------------


def _slot_tables(tables, names, item, effect, param=(0, 0), gift=None):
    """``tables`` with the vanilla slot ``item`` (a name) given ``effect`` and ``param``: the
    Kaizo item the build maps to ``effect``, on a slot the vanilla state builders know."""
    i = names.item(item)
    out = tables._replace(
        item_effect=jnp.asarray(tables.item_effect).at[i].set(int(effect)),
        item_param=jnp.asarray(tables.item_param).at[i].set(jnp.asarray(param, jnp.int16)),
    )
    if gift is not None:
        out = out._replace(
            item_gift_power=jnp.asarray(out.item_gift_power).at[i].set(gift[0]),
            item_gift_type=jnp.asarray(out.item_gift_type).at[i].set(gift[1]),
        )
    return out, i


def _hold(s, side, item, hp_div=None):
    """``side``'s active slot holds ``item``, at ``max_hp // hp_div`` HP when given."""
    slot = int(s.active[side])
    s = s._replace(item=s.item.at[side, slot].set(item))
    if hp_div:
        s = s._replace(hp=s.hp.at[side, slot].set(max(int(s.max_hp[side, slot]) // hp_div, 1)))
    return s


def test_every_item_of_the_group_is_covered_and_only_the_berries_are_pluckable():
    from nuzline.engine.coverage import IMPLEMENTED_ITEMS
    from nuzline.engine.hooks import PLUCKABLE

    new = (
        ItemId.MACHO_BRACE,
        ItemId.BERSERK_GENE,
        ItemId.LANSAT_BERRY,
        ItemId.STARF_BERRY,
        ItemId.LAVA_COOKIE,
    )
    assert all(IMPLEMENTED_ITEMS[i] == Support.EXACT for i in new)
    # Item_IsBerry (src/item.c:357-365) and PLUCK_EFFECT_NONE: Lansat / Starf are berries with a
    # pluck effect; the gene, the cookies and the brace are no berries (battle_lib.c:5832-5835).
    assert {i for i in new if i in PLUCKABLE} == {ItemId.LANSAT_BERRY, ItemId.STARF_BERRY}


def test_the_tables_map_each_game_to_its_hold_effect(tables, names, kaizo_tables, kaizo_names):
    """Hold effects 41 / 42 are Lansat / Starf in both games; 50 is Macho Brace and, in Kaizo,
    the six Power items (vanilla's 117-122); Kaizo's renamed or re-purposed slots take the id of
    the hold effect they now have (docs/kaizo/discrepancies.md:31-34, :1336-1348)."""
    both = {"Lansat Berry": ItemId.LANSAT_BERRY, "Starf Berry": ItemId.STARF_BERRY}
    both |= {"Macho Brace": ItemId.MACHO_BRACE, "Power Weight": ItemId.MACHO_BRACE}
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for name, want in both.items():
            assert int(t.item_effect[n.item(name)]) == want, name
    kaizo_only = {
        "Berserk Gene": ItemId.BERSERK_GENE,
        "Leek": ItemId.CRIT_ITEM,
        "Big Nugget": ItemId.NONE,
        "RageCandyBar": ItemId.BERRY_JUICE,
        "Lava Cookie": ItemId.LAVA_COOKIE,
        "Old Gateau": ItemId.LAVA_COOKIE,
        "Power Bracer": ItemId.MACHO_BRACE,
        "Power Anklet": ItemId.MACHO_BRACE,
    }
    for name, want in kaizo_only.items():
        assert int(kaizo_tables.item_effect[kaizo_names.item(name)]) == want, name
    k, n = kaizo_tables, kaizo_names
    assert k.item_param[n.item("Berserk Gene")].tolist() == [0, Stat.ATK]
    assert int(k.item_param[n.item("Leek"), 1]) == SPECIES_FARFETCHD
    assert int(k.item_param[n.item("RageCandyBar"), 0]) == 100
    assert int(k.item_gift_power[n.item("RageCandyBar")]) == 250
    assert int(k.item_gift_type[n.item("RageCandyBar")]) == 31


def test_macho_brace_halves_speed_from_the_raw_hold_effect(tables, names):
    """sSpeedHalvingItemEffects (battle_lib.c:1160-1169, :1240-1246): hold effect 50 and the
    Power items' 117-122 halve Speed, Klutz or not."""
    s = _state(tables, names)
    base = int(effective_speed(tables, s, 0))
    for item in ("Macho Brace", "Power Bracer"):
        held = _hold(s, 0, names.item(item))
        assert int(effective_speed(tables, held, 0)) == base // 2, item
        klutz = _ability(held, 0, AbilityId.KLUTZ)
        assert int(effective_speed(tables, klutz, 0)) == base // 2, item


def test_a_berserk_gene_raises_attack_at_the_first_check_and_pluck_leaves_it(tables, names):
    """Hold 36 with param 0: ``maxHP / 0`` is ``maxHP`` (G-53), so it fires at full HP
    (battle_lib.c:4773-4784); +1 Attack through the stage routine, then the item is gone
    (subscript_held_item_raise_stat.s:11-12). Pluck: PLUCK_EFFECT_NONE and no berry
    (:5832-5835): its user eats nothing and gets no Attack."""
    t, gene = _slot_tables(tables, names, "Shoal Salt", ItemId.BERSERK_GENE, (0, Stat.ATK))
    s = _hold(_state(t, names, mine=["Protect"]), 0, gene)
    out = assert_matches_reference(t, s, A("Protect", "Tackle", s, names), O)
    assert int(out.stages[0, Stat.ATK]) == 1 and int(out.item[0, 0]) == 0
    s = _hold(_state(t, names, mine=["Pluck"]), 1, gene)
    out = assert_matches_reference(t, s, A("Pluck", "Tackle", s, names), O)
    assert int(out.stages[0, Stat.ATK]) == 0 and int(out.stages[1, Stat.ATK]) == 1


def test_a_lansat_berry_sets_focus_energy_at_a_quarter_of_hp_unless_it_has_it(tables, names):
    """battle_lib.c:4838-4848: curHP <= maxHP / 4 and no VOLATILE_CONDITION_FOCUS_ENERGY; the
    subscript sets it (subscript_held_item_raise_crit.s:11) and the berry is eaten."""
    lansat = names.item("Lansat Berry")
    base = _state(tables, names, mine=["Protect"])
    acts = A("Protect", "Tackle", base, names)
    low = _hold(base, 0, lansat, hp_div=4)
    out = assert_matches_reference(tables, low, acts, O)
    assert int(out.focus_energy[0]) == 1 and int(out.item[0, 0]) == 0
    high = _hold(base, 0, lansat, hp_div=2)
    out = assert_matches_reference(tables, high, acts, O)
    assert int(out.focus_energy[0]) == 0 and int(out.item[0, 0]) == lansat
    focused = low._replace(focus_energy=low.focus_energy.at[0].set(1))
    out = assert_matches_reference(tables, focused, acts, O)
    assert int(out.item[0, 0]) == lansat  # already pumped: no trigger, the berry is kept


def test_a_plucked_lansat_berry_gives_its_eater_focus_energy(tables, names):
    """PLUCK_EFFECT_CRIT_UP (battle_lib.c:5819-5825): the eater gains the volatile unless it has
    it, at any HP; the berry is eaten either way."""
    lansat = names.item("Lansat Berry")
    s = _hold(_state(tables, names, mine=["Pluck"]), 1, lansat)
    acts = A("Pluck", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.focus_energy[0]) == 1 and int(out.item[1, 0]) == 0
    focused = s._replace(focus_energy=s.focus_energy.at[0].set(1))
    out = assert_matches_reference(tables, focused, acts, O)
    assert int(out.item[1, 0]) == 0


def test_a_lansat_holder_can_have_focus_energy_before_its_own_hit_after_a_foes(tables, names):
    """The crit odds of a Lansat holder's hit (outcome.crit_variants): the pinch can come from the
    foe's damaging hit first (the last variant bit: 1 in 4 with +2 stages, not 1 in 16); a foe's
    status move, or a holder with no berry, adds no variant."""
    lansat = names.item("Lansat Berry")
    s = _hold(_state(tables, names, mine=["Tackle"], opp="Tackle"), 0, lansat)
    odds, possible = crit_variants(tables, s, A("Tackle", "Tackle", s, names))
    bit = 1 << len(CRIT_ODDS_EFFECTS)
    assert bool(possible[0, bit]) and int(odds[0, 0]) == 16 and int(odds[0, bit]) == 4
    assert not bool(possible[1, bit])  # the foe holds nothing
    calm = _hold(_state(tables, names, mine=["Tackle"], opp="Growl"), 0, lansat)
    assert not bool(crit_variants(tables, calm, A("Tackle", "Growl", calm, names))[1][0, bit])
    bare = _state(tables, names, mine=["Tackle"], opp="Tackle")
    assert not bool(crit_variants(tables, bare, A("Tackle", "Tackle", bare, names))[1][0, bit])


def _starf(tables, names, roll, stages=None, side=0):
    """Side ``side``'s battler holds a Starf Berry at a quarter HP (``stages``: stat -> stage) and
    the turn's draw is ``roll``; Protect makes the turn-start check the only trigger."""
    s = _hold(_state(tables, names, mine=["Protect"]), side, names.item("Starf Berry"), hp_div=4)
    for stat, stage in (stages or {}).items():
        s = s._replace(stages=s.stages.at[side, stat].set(stage))
    draw = np.zeros(2, np.int8)
    draw[side] = roll
    o = O._replace(starf=jnp.asarray(draw))
    return assert_matches_reference(tables, s, A("Protect", "Tackle", s, names), o)


@pytest.mark.parametrize(
    ("roll", "stat"),
    [(0, Stat.ATK), (1, Stat.DEF), (2, Stat.SPE), (3, Stat.SPA), (4, Stat.SPD), (5, Stat.ATK)],
)
def test_a_starf_berry_raises_the_drawn_stat_by_two(tables, names, roll, stat):
    """battle_lib.c:4850-4872: ``RandNext() % 5`` over Attack, Defense, Speed, Sp. Atk, Sp. Def
    (never accuracy or evasion), +2 by a direct stage write; the berry is eaten (F-10)."""
    out = _starf(tables, names, roll)
    expected = np.zeros(8, np.int8)
    expected[stat] = 2
    assert out.stages[0].tolist() == expected.tolist() and int(out.item[0, 0]) == 0


def test_a_starf_berry_redraws_past_stats_at_plus_six_and_keeps_when_all_are(tables, names):
    """The rejection loop (:4864-4866): the draw is uniform over the stats below +6 (its rank
    modulo their count); the +2 is capped at +6 (:14-18 of the subscript); with all five at +6
    nothing triggers and the berry stays."""
    out = _starf(tables, names, 0, {Stat.ATK: 6})
    assert int(out.stages[0, Stat.DEF]) == 2 and int(out.stages[0, Stat.ATK]) == 6
    out = _starf(tables, names, 3, {Stat.ATK: 6, Stat.DEF: 5})
    assert out.stages[0].tolist()[1:6] == [6, 5, 0, 2, 0]  # rank 3 of Def, Spe, Spa, Spd: Spd
    maxed = dict.fromkeys((Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD), 6)
    out = _starf(tables, names, 1, maxed)
    assert int(out.item[0, 0]) == names.item("Starf Berry")


def test_a_plucked_starf_berry_gives_its_eater_the_same_draw(tables, names):
    """PLUCK_EFFECT_RANDOM_UP2 (battle_lib.c:5798-5817): the eater's stat, at any HP; eaten even
    with every stat at +6."""
    s = _hold(_state(tables, names, mine=["Pluck"]), 1, names.item("Starf Berry"))
    o = O._replace(starf=jnp.asarray([2, 0], jnp.int8))
    acts = A("Pluck", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.stages[0, Stat.SPE]) == 2 and int(out.item[1, 0]) == 0
    maxed = s._replace(stages=s.stages.at[0, 1:6].set(6))
    out = assert_matches_reference(tables, maxed, acts, o)
    assert out.stages[0].tolist()[1:6] == [6] * 5 and int(out.item[1, 0]) == 0


def test_the_starf_draw_is_an_outcome_variable_with_a_uniform_law(tables, names):
    """``Outcome.starf`` is read when a living Starf holder is in a party; its law is uniform over
    ``STARF_DRAWS`` values (every count 1-5 divides them)."""
    from nuzline.engine.enumerate import _LAWS

    s = _hold(_state(tables, names, mine=["Protect"]), 0, names.item("Starf Berry"))
    acts = jnp.asarray(A("Protect", "Tackle", s, names))
    laws = jax.tree.map(np.asarray, outcome_laws(tables, s, acts))
    assert np.asarray(laws.relevant.starf).tolist() == [True, False]
    values, weights, total = _LAWS["starf"](laws, 0)
    assert len(values) == total == STARF_DRAWS and set(weights) == {1}
    assert all(STARF_DRAWS % c == 0 for c in range(1, 6))


def test_a_lava_cookie_cures_like_a_lum_berry_and_pluck_leaves_it(tables, names):
    """Hold 12 (battle_lib.c:4662-4697): any status or confusion cleared with no HP test, then
    the item is gone. Pluck: PLUCK_EFFECT_NONE and no berry (:5832-5835), unlike Lum: its user
    is not cured by it and the foe keeps it."""
    t, cookie = _slot_tables(tables, names, "Lava Cookie", ItemId.LAVA_COOKIE)
    s = _hold(_state(t, names, mine=["Protect"]), 0, cookie).with_status(0, 0, Status.BURN)
    out = assert_matches_reference(t, s, A("Protect", "Tackle", s, names), O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.item[0, 0]) == 0
    s = _hold(_state(t, names, mine=["Pluck"]), 1, cookie).with_status(0, 0, Status.BURN)
    out = assert_matches_reference(t, s, A("Pluck", "Tackle", s, names), O)
    assert int(out.status[0, 0]) == Status.BURN and int(out.item[1, 0]) == cookie


def test_natural_gift_with_type_31_is_neutral_and_typeless(tables, names):
    """Kaizo's RageCandyBar: Natural Gift power 250, type 31 (no type id): no chart row, no STAB
    (battle_lib.c:2592, :2611-2645), as a ??? hit; every table read stays in range (F-14)."""
    t, cake = _slot_tables(tables, names, "Shoal Shell", ItemId.BERRY_JUICE, (100, 0), (250, 31))
    ng = names.move("Natural Gift")
    s = _hold(_state(t, names, mine=["Natural Gift"]), 0, cake)
    got = _both(t, s, ng)
    t9 = t._replace(item_gift_type=jnp.asarray(t.item_gift_type).at[cake].set(Type.MYSTERY))
    assert got == _both(t9, s, ng) and max(got) > 0


# --- The party friendship wrap (F-15) ---------------------------------------------------------

SWITCH = 4  # order.MOVE_ACTIONS + party slot


@pytest.mark.parametrize(
    ("party", "battler", "want_party", "want_battler"),
    [
        (254, 254, 2, 255),  # 254 + 3 > 255: (255 + 3) & 0xFF, pokemon.c:1859-1874
        (255, 255, 2, 255),  # the sheet's Christine case: party 2, battler stays 255
        (250, 250, 253, 253),  # no crossing: both add
        (0, 0, 3, 3),  # a Frustration holder: the low band
        (99, 255, 102, 255),  # the band is the party's (battle_system.c:716-720); battler clamps
        (200, 10, 203, 13),  # the high band by the party value, though the battler reads low
    ],
)
def test_a_poke_balls_friendship_wraps_the_party_value_not_the_battlers(
    tables, names, party, battler, want_party, want_battler
):
    """UseBagItem bands by the party Pokemon's friendship and adds through
    ``Pokemon_IncreaseValue`` (u8: a sum past 255 stores ``255 + gain - 256``, pokemon.c:1859-1874);
    the battler's ``BattleMon_AddVal`` clamps (battle_lib.c:1032-1043). Item 4 has ``partyUse``
    false, so every party param reads its dummy 3 (item.c:186-195, bs:716-755)."""
    s = _bag(roark_state(tables, names), names, "Poké Ball")
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]) // 5)
    s = s._replace(
        friendship=s.friendship.at[1, 0].set(battler),
        party_friendship=s.party_friendship.at[1, 0].set(party),
    )
    out = _use_both(tables, s, names.item("Poké Ball"))
    assert (int(out.party_friendship[1, 0]), int(out.friendship[1, 0])) == (
        want_party,
        want_battler,
    )


def test_a_switch_in_rereads_the_party_friendship(tables, names):
    """``BattleSystem_InitBattleMon`` loads the battler's friendship from the party Pokemon
    (battle_lib.c:136): the wrapped 2 replaces the battler's clamped 255 when it comes back."""
    s = _bag(roark_state(tables, names), names, "Poké Ball")
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]) // 5)
    s = s._replace(
        friendship=s.friendship.at[1, 0].set(255),
        party_friendship=s.party_friendship.at[1, 0].set(255),
    )
    s = assert_matches_reference(tables, s, [SWITCH + 1, FIRST_ITEM_ACTION], O)
    assert (int(s.party_friendship[1, 0]), int(s.friendship[1, 0])) == (2, 255)
    s = assert_matches_reference(tables, s, [SWITCH, SWITCH + 1], O)
    s = assert_matches_reference(tables, s, [SWITCH + 1, SWITCH], O)
    assert int(s.active[1]) == 0
    assert (int(s.party_friendship[1, 0]), int(s.friendship[1, 0])) == (2, 2)


# --- Shaymin-Sky reverts to Land Forme when frozen (F-16) -------------------------------------


def _shaymin_state(tables, names, *, status=Status.FREEZE, substitute=0):
    """Side 0: a Shaymin-Sky (form 1) and a bench Geodude; side 1: a Geodude (Tackle), slower
    than the Shaymin's frozen action."""
    sky = dataclasses.replace(_mon(names, "Shaymin", ["Air Slash"], 50), form=1)
    s = make_state(
        tables,
        [sky, _mon(names, "Geodude", ["Tackle"], 30)],
        [_mon(names, "Geodude", ["Tackle"], 30)],
    )
    s = s._replace(stats=s.stats.at[1, 0, Stat.SPE].set(500))  # the foe moves first
    if status:
        s = s.with_status(0, 0, status)
    return s._replace(substitute_hp=s.substitute_hp.at[0].set(substitute))


def test_a_frozen_shaymin_sky_becomes_land_forme_after_any_hit(tables, names):
    """subscript_check_shaymin_form.s:5-30 after the secondary effect (bcp.c:3497-3505): form 0,
    Land's ability slot (Natural Cure), types, recalculated stats and the AI's knowledge of the
    new ability; HP, status and stages stay (the freeze goes at the thaw checks, not here)."""
    sky = _shaymin_state(tables, names)
    assert int(sky.form[0, 0]) == 1 and int(sky.ability[0, 0]) == AbilityId.SERENE_GRACE
    out = assert_matches_reference(tables, sky, A("Air Slash", "Tackle", sky, names), O)
    land = make_state(
        tables,
        [
            dataclasses.replace(_mon(names, "Shaymin", ["Air Slash"], 50), form=0),
            _mon(names, "Geodude", ["Tackle"], 30),
        ],
        [_mon(names, "Geodude", ["Tackle"], 30)],
    )
    assert int(out.form[0, 0]) == 0
    assert int(out.ability[0, 0]) == int(land.ability[0, 0]) == AbilityId.NATURAL_CURE
    assert int(out.ability_form[0, 0]) == 0
    assert out.types[0, 0].tolist() == land.types[0, 0].tolist() == [Type.GRASS, Type.GRASS]
    assert out.stats[0, 0].tolist() == land.stats[0, 0].tolist() != sky.stats[0, 0].tolist()
    assert int(out.max_hp[0, 0]) == int(sky.max_hp[0, 0])
    assert int(out.ai_ability_known[0]) != 0


@pytest.mark.parametrize(
    "case", ["not frozen", "substitute", "land forme already"], ids=lambda c: c
)
def test_the_revert_needs_a_frozen_sky_shaymin_without_a_substitute(tables, names, case):
    """chk:6-9: the form is not 0, no Substitute, status exactly Freeze."""
    s = _shaymin_state(
        tables,
        names,
        status=Status.BURN if case == "not frozen" else Status.FREEZE,
        substitute=40 if case == "substitute" else 0,
    )
    if case == "land forme already":
        s = s._replace(form=s.form.at[0, 0].set(0))
    out = assert_matches_reference(tables, s, A("Air Slash", "Tackle", s, names), O)
    assert int(out.form[0, 0]) == int(s.form[0, 0])
    assert int(out.ability[0, 0]) == int(s.ability[0, 0])
