"""E1 batch F (rulings F-n, docs/kaizo/e1-mechanics-ledger.md): Heatproof, Flash Fire, Multitype,
Intimidate's substitute skip, Hydration and Forecast, each case checked against the reference engine."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table, personal_types
from nuzline.engine.hooks import battle_start
from nuzline.engine.ids import AbilityId, Stat, Status, Type, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state
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
