"""E1 batch G (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): the per-party ability override
(F-O2, G-1..G-3): a Transform user's copy of a frozen Sky Shaymin reverts to Land Forme and
Land's ability is written into the user's own party Pokemon (subscript_check_shaymin_form.s:5-30,
battle_script.c:3401-3407, battle_io_command.c:597-604). Each case is checked against the
reference engine."""

from __future__ import annotations

import dataclasses

import pytest

from nuzline.ai.switching import switching_refusals
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import decomp_abilities
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Stat, Status, Type
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

O = empty_outcome()
SHAYMIN = 492


def _acts(names, s, d, mine, theirs):
    """Move names as actions: side ``d`` uses ``mine``, the other side ``theirs``."""
    acts = [0, 0]
    acts[d] = names.move_slot(s, d, mine)
    acts[1 - d] = names.move_slot(s, 1 - d, theirs)
    return acts


def _fight(tables, names, d, *, personality_slot=0):
    """Turn 1 of a Transform: side ``d`` (the Transform user, a level-50 Ditto and a bench
    Geodude) copies the other side's level-5 Sky Shaymin (form 1); the Ditto acts first. Returns the
    plain (untransformed) start state and the state after the turn."""
    ditto = dataclasses.replace(
        _mon(names, "Ditto", ["Transform"], 50), ability_slot=personality_slot
    )
    sky = dataclasses.replace(_mon(names, "Shaymin", ["Air Slash"], 5), form=1)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    parties = {d: [ditto, geo], 1 - d: [sky, geo]}
    s = make_state(tables, parties[0], parties[1])
    plain = s
    s = s._replace(stats=s.stats.at[d, 0, Stat.SPE].set(500))
    out = assert_matches_reference(tables, s, _acts(names, s, d, "Transform", "Air Slash"), O)
    return plain, out


def _revert(tables, names, d, *, status=Status.FREEZE, substitute=0, personality_slot=0):
    """Turn 2: the copy has ``status`` and the Shaymin's Air Slash hits it first."""
    plain, s = _fight(tables, names, d, personality_slot=personality_slot)
    assert int(s.transformed[d]) == 1 and int(s.species[d, 0]) == SHAYMIN and int(s.form[d, 0]) == 1
    s = s.with_status(d, 0, status)
    s = s._replace(
        substitute_hp=s.substitute_hp.at[d].set(substitute),
        stats=s.stats.at[d, 0, Stat.SPE].set(1).at[1 - d, 0, Stat.SPE].set(500),
    )
    acts = _acts(names, s, d, "Air Slash", "Air Slash")
    return plain, s, assert_matches_reference(tables, s, acts, O)


@pytest.mark.parametrize("d", [0, 1], ids=["player Ditto", "AI Ditto"])
def test_a_transformed_copy_of_a_frozen_sky_shaymin_reverts_into_its_party_pokemon(
    tables, names, d
):
    """shay:5-9 pass for the copy; :10-30 and BtlIOCmd_UpdatePartyMon (io:597-604): the copy is
    form 0 with Natural Cure and Land's types, the party Pokemon holds Natural Cure (override)
    and form 0, and RefreshMonData gives the copy the Ditto's OWN stats (lib:169-193)."""
    plain, _, out = _revert(tables, names, d)
    assert int(out.form[d, 0]) == 0 and int(out.species[d, 0]) == SHAYMIN
    assert int(out.transformed[d]) == 1 and int(out.shadow_form[d]) == 0
    assert int(out.ability[d, 0]) == int(out.party_ability[d, 0]) == AbilityId.NATURAL_CURE
    assert out.types[d, 0].tolist() == [Type.GRASS, Type.GRASS]
    assert out.stats[d, 0].tolist() == plain.stats[d, 0].tolist() == out.shadow_stats[d].tolist()
    assert int(out.max_hp[d, 0]) == int(plain.max_hp[d, 0])
    if d == 0:  # the AI learns the written ability (battle_script.c:3401-3403)
        assert int(out.ai_ability_known[0]) != 0


@pytest.mark.parametrize("case", ["not frozen", "substitute"])
def test_the_copy_is_left_alone_without_a_freeze_or_with_a_substitute(tables, names, case):
    """shay:8-9: a Substitute (one the hit does not break), or a status other than Freeze, ends
    the script at :32."""
    _, before, out = _revert(
        tables,
        names,
        0,
        status=Status.BURN if case == "not frozen" else Status.FREEZE,
        substitute=200 if case == "substitute" else 0,
    )
    assert int(out.form[0, 0]) == 1 and int(out.party_ability[0, 0]) == 0
    assert int(out.ability[0, 0]) == int(before.ability[0, 0])


def test_land_slot_one_falls_back_to_slot_zero(tables, names):
    """shay:12-19: the personality bit picks Land's ability 2, which is empty (decomp id 0, whose
    AbilityId is NONE, not 0): ability 1 is taken."""
    _, _, out = _revert(tables, names, 0, personality_slot=1)
    assert int(out.ability[0, 0]) == int(out.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    assert int(out.ability_slot[0, 0]) == 0


def test_the_written_ability_outlives_the_copy_and_cures_at_the_next_switch_out(tables, names):
    """lib:124 (InitBattleMon reads MON_DATA_ABILITY): after the switch-out the Ditto is itself
    with Natural Cure (the bench slot reads it, ``decomp_abilities``); sent out again and
    burned, its next switch-out cures the status (the party override, not a copy, scr:9146-9154)."""
    plain, _, s = _revert(tables, names, 0)
    leave = MOVE_ACTIONS + 1
    foe = names.move_slot(s, 1, "Air Slash")
    s = assert_matches_reference(tables, s, [leave, foe], O)
    assert int(s.species[0, 0]) == int(plain.species[0, 0]) and int(s.transformed[0]) == 0
    assert s.types[0, 0].tolist() == plain.types[0, 0].tolist()
    assert s.stats[0, 0].tolist() == plain.stats[0, 0].tolist()
    assert int(s.ability[0, 0]) == int(s.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    assert int(decomp_abilities(tables, s)[0, 0]) == ABILITY_DECOMP_IDS[AbilityId.NATURAL_CURE]
    s = assert_matches_reference(tables, s, [MOVE_ACTIONS, foe], O)  # the Ditto comes back in
    assert int(s.active[0]) == 0 and int(s.transformed[0]) == 0
    s = s.with_status(0, 0, Status.BURN)
    s = assert_matches_reference(tables, s, [leave, foe], O)
    assert int(s.status[0, 0]) == 0


def test_a_new_transform_copies_over_the_override_and_ending_it_restores_the_override(
    tables, names
):
    """scr:5346-5348: the copy takes the target's ability; the party's stays (lib:124)."""
    ditto = _mon(names, "Ditto", ["Transform"], 50)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    s = make_state(tables, [ditto, geo], [geo, geo])
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.NATURAL_CURE).at[1, 0].set(AbilityId.INTIMIDATE),
        party_ability=s.party_ability.at[0, 0].set(AbilityId.NATURAL_CURE),
        stats=s.stats.at[0, 0, Stat.SPE].set(500),
    )
    foe = names.move_slot(s, 1, "Tackle")
    s = assert_matches_reference(tables, s, [names.move_slot(s, 0, "Transform"), foe], O)
    assert int(s.ability[0, 0]) == AbilityId.INTIMIDATE
    assert int(s.party_ability[0, 0]) == AbilityId.NATURAL_CURE
    s = assert_matches_reference(tables, s, [MOVE_ACTIONS + 1, foe], O)
    assert int(s.ability[0, 0]) == AbilityId.NATURAL_CURE


def test_the_ais_bench_ditto_with_the_override_is_seen_as_natural_cure(tables, names):
    """trainer_ai.c:3451-3528 read the AI's own party abilities: ``decomp_abilities`` gives the
    override, and ``switching_refusals`` names AI_IsAsleepWithNaturalCure."""
    ditto = _mon(names, "Ditto", ["Transform"], 50)
    geo = _mon(names, "Geodude", ["Tackle"], 30)
    s = make_state(tables, [geo, geo], [geo, ditto])
    assert not any("Natural Cure" in line for line in switching_refusals(tables, s))
    s = s._replace(
        ability=s.ability.at[1, 1].set(AbilityId.NATURAL_CURE),
        party_ability=s.party_ability.at[1, 1].set(AbilityId.NATURAL_CURE),
    )
    assert int(decomp_abilities(tables, s)[1, 1]) == ABILITY_DECOMP_IDS[AbilityId.NATURAL_CURE]
    assert any("Natural Cure" in line for line in switching_refusals(tables, s))


# --- G-14: Air Lock is Cloud Nine's lock (include/battle/common.h:63-67) ----------------------


def _weather_state(tables, names, weather, lock_ability):
    """Roark's lead (side 1) with ``lock_ability`` written as its ability, under ``weather``;
    the player's lead knows Ember and Leer."""
    from nuzline.engine.field import with_weather
    from tests.engine.helpers import roark_state

    s = roark_state(tables, names, player_count=1, opp_count=1, player_moves=("Ember", "Leer"))
    s = s._replace(ability=s.ability.at[1, 0].set(lock_ability))
    return with_weather(s, weather)


def test_air_lock_is_supported_and_maps_to_decomp_76():
    from nuzline.engine.coverage import IMPLEMENTED_ABILITIES

    assert AbilityId.AIR_LOCK in IMPLEMENTED_ABILITIES
    assert ABILITY_DECOMP_IDS[AbilityId.AIR_LOCK] == 76  # generated/abilities.txt:77


@pytest.mark.parametrize("weather", ["RAIN", "SUN"], ids=["rain", "sun"])
def test_air_lock_cancels_the_weather_damage_boost_like_cloud_nine(tables, names, weather):
    """lib.c:7046-7072 sits inside NO_CLOUD_NINE: Ember is halved in rain / x1.5 in sun only
    without a living Cloud Nine / Air Lock holder; Air Lock and Cloud Nine read alike."""
    from nuzline.engine.damage import damage_for_roll
    from nuzline.engine.ids import Weather

    w = getattr(Weather, weather)
    ember = names.move("Ember")

    def dmg(weather_id, ability):
        s = _weather_state(tables, names, weather_id, ability)
        return int(damage_for_roll(tables, s, 0, ember, False, 15))

    free = dmg(Weather.NONE, AbilityId.NONE)
    open_weather = dmg(w, AbilityId.NONE)
    assert open_weather != free
    assert dmg(w, AbilityId.AIR_LOCK) == free
    assert dmg(w, AbilityId.CLOUD_NINE) == free


def test_air_lock_stops_the_sand_chip_and_matches_the_reference(tables, names):
    """EndOfTurnWeatherEffect (battle_script.c:5573-5645) is gated by NO_CLOUD_NINE: the
    player's lead takes the sand chip only with no Air Lock holder on the field."""
    from nuzline.engine.ids import Weather

    def player_hp(ability):
        s = _weather_state(tables, names, Weather.SANDSTORM, ability)
        acts = [names.move_slot(s, 0, "Leer"), names.move_slot(s, 1, "Leer")]
        out = assert_matches_reference(tables, s, acts, O)
        return int(s.hp[0, 0]), int(out.hp[0, 0])

    before, after = player_hp(AbilityId.NONE)
    assert after < before
    before, after = player_hp(AbilityId.AIR_LOCK)
    assert after == before


def test_field_defog_start_is_a_run_input(kaizo_trainers, kaizo_names):
    """MapHeader_GetMapType (rom_diff.md:277; G-15-G-17): a fight on a fog header (14) starts
    in fog (subscript_overworld_fog, battle_lib.c:3684-3688) unless the run used field Defog on
    that map visit (FLAG_DEFOG_ACTIVE, field_map_change.c:275-280; scrcmd.c:3667-3674), which
    needs the Relic Badge (field_move_tasks.c:504) and a party member with Defog; a save on the
    fight's header gives the overworld weather itself (field_battle_data_transfer.c:267,
    field_map_change.c:521-525). Deep fog (15, Hearthome Gym) is never cleared."""
    from types import SimpleNamespace

    from nuzline.cli import RELIC_BADGE, fight_start
    from nuzline.engine.ids import Weather

    luna, names = kaizo_trainers[66], kaizo_names
    defog = names.move("Defog")
    holder, other = SimpleNamespace(moves=(defog, 0, 0, 0)), SimpleNamespace(moves=(33, 0, 0, 0))

    def save(header=0, weather=0, badges=1 << RELIC_BADGE):
        return SimpleNamespace(map_header=header, overworld_weather=weather, badges=badges)

    assert luna.area_condition.fog_headers == (363,)
    assert fight_start(luna, names) == (Weather.FOG, 0)
    assert fight_start(luna, names, save(), True, [other, holder]) == (Weather.NONE, defog)
    assert fight_start(luna, names, save(), True) == (Weather.NONE, defog)  # prep picks the team
    with pytest.raises(ValueError, match="no team member knows Defog"):
        fight_start(luna, names, save(), True, [other])
    with pytest.raises(ValueError, match="Relic Badge"):
        fight_start(luna, names, save(badges=0xEF), True, [holder])
    assert fight_start(luna, names, save(363, 0)) == (Weather.NONE, 0)
    assert fight_start(luna, names, save(363, 14)) == (Weather.FOG, 0)
    with pytest.raises(ValueError, match="overworld weather 2"):
        fight_start(luna, names, save(363, 2))
    assert fight_start(kaizo_trainers[913], names, save(449, 0)) == (Weather.NONE, 0)
    hearthome = kaizo_trainers[34]
    assert hearthome.start_condition == "fog" and not hearthome.area_condition.fog_clearable
    with pytest.raises(ValueError, match="OVERWORLD_WEATHER_FOG"):
        fight_start(hearthome, names, None, True, [holder])
