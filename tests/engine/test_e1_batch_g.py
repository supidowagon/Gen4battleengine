"""E1 batch G (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): the per-party ability override
(F-O2, G-1..G-3): a Transform user's copy of a frozen Sky Shaymin reverts to Land Forme and
Land's ability is written into the user's own party Pokemon (subscript_check_shaymin_form.s:5-30,
battle_script.c:3401-3407, battle_io_command.c:597-604). Each case is checked against the
reference engine."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.switching import switching_refusals
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import decomp_abilities
from nuzline.engine.hooks import battle_start
from nuzline.engine.ids import ABILITY_DECOMP_IDS, STARF_DRAWS, AbilityId, Stat, Status, Type
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


@pytest.mark.fullset
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
    assert fight_start(luna, names, save(363, 0), False, [other]) == (Weather.NONE, 0)
    assert fight_start(luna, names, save(363, 14)) == (Weather.FOG, 0)
    # The save decides on the fight's map: a --field-defog its fog contradicts is refused.
    with pytest.raises(ValueError, match="the save says otherwise"):
        fight_start(luna, names, save(363, 14), True, [holder])
    # Prep may visit a PC: the warp clears FLAG_DEFOG_ACTIVE (field_map_change_flags.c:80-85)
    # and the header's fog returns (field_map_change.c:273-280), so a Defog holder is kept.
    assert fight_start(luna, names, save(363, 0)) == (Weather.NONE, defog)
    with pytest.raises(ValueError, match="overworld weather 2"):
        fight_start(luna, names, save(363, 2))
    assert fight_start(kaizo_trainers[913], names, save(449, 0), False, [other]) == (
        Weather.NONE,
        0,
    )
    hearthome = kaizo_trainers[34]
    assert hearthome.start_condition == "fog" and not hearthome.area_condition.fog_clearable
    with pytest.raises(ValueError, match="OVERWORLD_WEATHER_FOG"):
        fight_start(hearthome, names, None, True, [holder])


def test_mixed_battle_terrain_is_a_run_input(kaizo_trainers, kaizo_tables, kaizo_names):
    """Battle terrain per fight (G-18, manager ruling 1): the terrain is the tile under the
    player at the start (CalcTerrain, field_battle_data_transfer.c:467-494; battle_main.c:1052),
    so a trainer whose start tiles give several (Ace Trainer Laura 134, Route 217: GRASS / SNOW)
    is refused while a terrain move would read it, until the run names one of them; the row
    records the three sources (ROM several; calc repo, docs sheet none). Nature Power then
    calls that terrain's move (to_move.h; battle_script.c:8036-8048). A single-terrain fight
    needs no input."""
    from nuzline.cli import fight_terrain, terrain_id
    from nuzline.data.trainer import TERRAIN_NAMES, TERRAIN_UNKNOWN, run_terrain
    from nuzline.engine.coverage import terrain_calls, terrain_refusal

    tables, names = kaizo_tables, kaizo_names
    nature_power = names.move("Nature Power")
    laura = kaizo_trainers[134]
    grass, snow = terrain_id("grass"), terrain_id("TERRAIN_SNOW")
    assert (laura.terrain, laura.terrains) == (TERRAIN_UNKNOWN, (grass, snow))
    assert any(nature_power in m.moves for m in laura.party)
    assert "several: pkcalc none (move texts only), sheet none" in laura.sources["terrain"]
    line = terrain_refusal(tables, names, laura, [nature_power])
    assert "give TERRAIN_GRASS, TERRAIN_SNOW (ROM; the calc repo and the docs sheet" in line
    assert line.endswith("would read it: name it with --terrain")
    assert fight_terrain(laura) is laura and not run_terrain(laura)
    on_snow = fight_terrain(laura, "snow")
    assert on_snow.terrain == snow and run_terrain(on_snow)
    assert terrain_refusal(tables, names, on_snow, [nature_power]) is None
    assert terrain_calls(tables, on_snow, [nature_power]) == [names.move("Blizzard")]
    on_grass = fight_terrain(laura, "Grass")
    assert terrain_calls(tables, on_grass, [nature_power]) == [names.move("Seed Bomb")]
    with pytest.raises(ValueError, match="start tiles give TERRAIN_GRASS, TERRAIN_SNOW"):
        fight_terrain(laura, "water")
    with pytest.raises(ValueError, match="not a battle terrain"):
        terrain_id("lava")
    single = next(t for t in kaizo_trainers if t.split and len(t.terrains) == 1)
    assert fight_terrain(single) is single and not run_terrain(single)
    named = TERRAIN_NAMES[single.terrain]
    assert fight_terrain(single, named).terrain == single.terrain
    other = TERRAIN_NAMES[grass if single.terrain != grass else snow]
    with pytest.raises(ValueError, match="start tiles give"):
        fight_terrain(single, other)


def test_script_weather_before_a_battle_is_the_start_condition(kaizo_trainers):
    """gFieldScriptCommands 0x9C (rom_diff.md:292; G-23): scr_seq 329 sets VAR_0x4016 = 4 and
    runs 0x9C at 0x01df before StartTagBattle 609, 835, 836 at 0x0202 (no 0C3 / 0C4 / Warp
    between), so Triton #2 and Neptune #2 start in the overworld weather 4, rain in the Kaizo
    switch (battle_lib.c:3665-3669; field_battle_data_transfer.c:267), as PKCalc says
    (manager ruling 5). The tag partner is not an opponent: Riley keeps the header's none."""
    from nuzline.engine.coverage import start_weather
    from nuzline.engine.ids import Weather

    for tid in (835, 836):
        t = kaizo_trainers[tid]
        assert t.start_condition == "rain" and start_weather(t) == Weather.RAIN, tid
        assert "scr_seq 329 script weather at 0x01df" in t.sources["start_condition"], tid
    assert kaizo_trainers[609].start_condition == "none"


def test_route_213_script_rain_is_a_run_input(kaizo_trainers, kaizo_names):
    """scr_seq 459 entry 12 (coord event VAR 0x40FA == 7) sets overworld weather 4 at 0x0264,
    then VAR 0x40FA := 8; the rain lasts until ScrCmd_0C4 after the Chelle win (0x02b7,
    scrcmd.c:3667-3674) or a map change (field_map_change.c:272-280). G-24, manager ruling 3:
    the trainers that can meet it start in it or not (--script-weather), the start condition
    the default; a save past VAR 0x40FA = 8 decides (off the map: none; on it: its overworld
    weather). Coord battles at VAR 0x40FA 3-5 (692-695, 466) come before the rain."""
    from types import SimpleNamespace

    from nuzline.cli import fight_start
    from nuzline.engine.ids import Weather

    names = kaizo_names
    sheltin, chelle = kaizo_trainers[103], kaizo_trainers[491]
    window = (373, 4, 0x40FA, 8)
    for t in (sheltin, chelle):
        assert t.area_condition.script_weather == "rain"
        assert t.area_condition.script_window == window
    for tid in (466, 692, 693, 694, 695, 835):
        assert not kaizo_trainers[tid].area_condition.script_window, tid

    def save(var, header=0, weather=0):
        vars_ = [0] * 288
        vars_[0xFA] = var
        return SimpleNamespace(
            map_header=header, overworld_weather=weather, badges=0, var=lambda v: vars_[v - 0x4000]
        )

    def start(t, s=None, flag=None, team=None):
        return fight_start(t, names, s, team=team, script_weather=flag)[0]

    assert start(sheltin) == Weather.RAIN  # the user's answer (2026-11-07) is the default
    assert start(sheltin, flag=False) == Weather.NONE
    assert start(sheltin, save(5), False) == Weather.NONE  # before the cutscene: the route
    assert start(sheltin, save(7)) == Weather.RAIN
    assert start(sheltin, save(8)) == Weather.NONE  # left Route 213 after the rain
    assert start(chelle, save(8, 373, 4)) == Weather.RAIN
    brought = [SimpleNamespace(moves=(33, 0, 0, 0))]
    with pytest.raises(ValueError, match="the save says otherwise"):
        start(chelle, save(8, 373, 4), False, brought)
    # Prep (no team yet): a PC visit's warp rewrites the weather from header 373 (clear,
    # field_map_change.c:273-280) and VAR 0x40FA = 8 keeps the script from running again.
    assert start(chelle, save(8, 373, 4), False) == Weather.NONE
    assert start(chelle, save(9, 373, 0)) == Weather.NONE
    with pytest.raises(ValueError, match="the save says otherwise"):
        start(sheltin, save(8), True)
    with pytest.raises(ValueError, match="overworld weather 2"):
        start(sheltin, save(8, 373, 2))
    with pytest.raises(ValueError, match="no map script sets the weather"):
        start(kaizo_trainers[466], flag=True)


@pytest.mark.fullset
def test_a_status_word_with_several_conditions_is_refused():
    """0x3F (rom_diff.md:292; G-25): scr_seq 363 writes 0x7B to party slot 0 before the Lady
    Emi battle (sleep 3 | poison | burn | freeze | paralysis, condition.h:7-13); the battle
    tests each bit and clears them one by one (battle_controller_player.c:2449,
    subscript_thaw_out.s), so the word reaches later fights. The engine keeps one status:
    the save's Pokemon is refused for a fight, and prep (which heals) takes it."""
    from types import SimpleNamespace

    from nuzline.saves.pkm import RecordError, SaveMon
    from nuzline.saves.status import status_conditions

    assert status_conditions(0x7B) == ["sleep 3", "poison", "burn", "freeze", "paralysis"]
    assert status_conditions(0x10) == ["burn"] and status_conditions(0) == []
    mon = SimpleNamespace(location=("party", 0, 0), status_raw=0x7B)
    with pytest.raises(RecordError, match="sleep 3, poison, burn, freeze, paralysis"):
        SaveMon.to_spec(mon, None)


# --- F-O3 (G-27..G-30): every Starf Berry eat on one battler in one turn draws its own -------


_SPE = 4  # stats[..., 0..4] are Atk, Def, Sp. Atk, Sp. Def, Speed (state.py, ``stats``)


def _starf_fight(names, tables, mine, theirs, *, foe_item="", bench=False, foe_level=10):
    """Side 0: a level-50 Abra (``mine``) holding a Starf Berry at a quarter HP, and with
    ``bench`` a second one at a quarter HP; side 1: a Rattata (``theirs``, ``foe_item``) at
    ``foe_level``, faster than the Abra."""
    abra = _mon(names, "Abra", mine, 50, "Starf Berry")
    party = [abra, _mon(names, "Abra", ["Tackle"], 50, "Starf Berry")] if bench else [abra]
    s = make_state(tables, party, [_mon(names, "Rattata", theirs, foe_level, foe_item)])
    for slot in range(len(party)):
        s = s.replace_hp(0, slot, int(s.max_hp[0, slot]) // 4)
    return s._replace(stats=s.stats.at[1, 0, _SPE].set(500))


def _draws(first, second=0, third=0):
    """``Outcome.starf`` with side 0's eats drawing ``first``, ``second``, ``third`` (columns 0, 2,
    3; ``ids.STARF_QUEUE``)."""
    return O._replace(
        starf=jnp.asarray([[first, 0, second, third], [0, 0, 0, 0]], jnp.int8),
        roar_level=jnp.ones(2, jnp.int8),
    )


def _boosts(s, side=0):
    return {Stat(i).name: int(v) for i, v in enumerate(np.asarray(s.stages[side])) if v}


def test_a_recycled_starf_berry_draws_again(tables, names):
    """(c) The foe's faster Tackle puts the holder's Starf Berry check at the defender item check
    (bcp:3691-3696, lib:4865: Attack, draw 0); its Recycle gives the berry back and it is eaten
    again with its own draw (lib:4865 / :5288: rank 3 of five, Sp. Atk), not a repeat (G-27)."""
    s = _starf_fight(names, tables, ["Recycle"], ["Tackle"])
    out = assert_matches_reference(tables, s, _acts(names, s, 0, "Recycle", "Tackle"), _draws(0, 3))
    assert _boosts(out) == {"ATK": 2, "SPA": 2} and int(out.item[0, 0]) == 0
    assert np.asarray(out.starf_roll).tolist() == [[0, 0, 0]] * 2
    assert np.asarray(out.starf_taken).tolist() == [0, 0]


def test_a_roar_newcomer_eats_with_its_own_draw(tables, names):
    """(b) The holder eats at its Tackle's attacker item check (bcp:3684-3689: Attack, draw 0);
    the foe's Roar drags in the bench holder at its pinch, whose switch-in check (lib:4151-4165,
    :5288) takes the second draw: 4 of five, Sp. Def, on fresh stages (lib:97-101), G-27."""
    s = _starf_fight(names, tables, ["Tackle"], ["Roar"], bench=True, foe_level=60)
    out = assert_matches_reference(tables, s, _acts(names, s, 0, "Tackle", "Roar"), _draws(0, 4))
    assert int(out.active[0]) == 1 and int(out.item[0, 0]) == int(out.item[0, 1]) == 0
    assert _boosts(out) == {"SPD": 2}


@pytest.mark.parametrize("theirs", ["Tackle", "Roar"])
def test_bug_bite_of_a_starf_berry_then_its_own_then_a_roar_newcomer(tables, names, theirs):
    """(d) Bug Bite eats the foe's Starf Berry for the attacker (lib:5798-5817, draw :5808:
    Attack, draw 0); the attacker's own one goes at its item check (bcp:3684-3689: rank 3 of five,
    Sp. Atk); a slower foe's Roar then drags in the bench holder, the third draw (Defense, 1),
    G-28."""
    level = 60 if theirs == "Roar" else 10  # Roar's level check; a weak Tackle
    s = _starf_fight(
        names, tables, ["Bug Bite"], [theirs], foe_item="Starf Berry", bench=True, foe_level=level
    )
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(1))
    out = assert_matches_reference(
        tables, s, _acts(names, s, 0, "Bug Bite", theirs), _draws(0, 3, 1)
    )
    assert int(out.item[1, 0]) == 0 and int(out.item[0, 0]) == 0
    if theirs == "Roar":
        assert int(out.active[0]) == 1 and _boosts(out) == {"DEF": 2}
    else:
        assert int(out.active[0]) == 0 and _boosts(out) == {"ATK": 2, "SPA": 2}


def test_a_pluck_flag_left_set_eats_the_attackers_starf_twice(tables, names):
    """A Bug Bite of a Cheri Berry that cures nothing leaves SELF_TURN_FLAG_PLUCK_BERRY set
    (lib:5848; subscript_pluck_check.s): the attacker's Starf Berry is kept at the attacker item
    check (bcp:3684-3689, draw 0: Attack) and eaten at _HELD_ITEM_STATUS (bcp:3730-3761, lib:5288:
    rank 2 of five, Speed), two draws, G-28."""
    s = _starf_fight(names, tables, ["Bug Bite"], ["Growl"], foe_item="Cheri Berry")
    s = s._replace(stats=s.stats.at[1, 0, _SPE].set(1))
    acts = _acts(names, s, 0, "Bug Bite", "Growl")
    out = assert_matches_reference(tables, s, acts, _draws(0, 2))
    assert int(out.item[0, 0]) == 0 and _boosts(out) == {"ATK": 2, "SPE": 2}


def test_a_battle_start_berry_takes_entry_zero_and_a_recycled_one_the_next(tables, names):
    """A lead's battle-start Starf Berry waits open for the turn's first draw (F-10: Sp. Atk, 3);
    its Recycle then eats it again from queue entry 2 (Attack, 0), G-27."""
    s = _starf_fight(names, tables, ["Recycle"], ["Tackle"])
    started = battle_start(tables, s)
    assert int(started.starf_open[0]) == 1
    out = assert_matches_reference(
        tables, started, _acts(names, started, 0, "Recycle", "Tackle"), _draws(3, 0)
    )
    assert _boosts(out) == {"SPA": 2, "ATK": 2} and int(out.starf_open[0]) == 0


@pytest.mark.slow
def test_the_later_starf_draws_are_read_where_eats_can_meet(tables, names):
    """``enumerate._starf_reads`` (G-29): ``starf[d, 2]`` where two eats can meet on battler
    ``d``, ``[d, 3]`` where three can; five values for a fresh Roar newcomer, all 60 for the same
    battler's Recycle (``_STARF_STAGE_JUMPS``); none with one eat source."""
    from nuzline.engine.enumerate import _starf, outcome_laws

    def laws(s, mine, theirs):
        acts = jnp.asarray(_acts(names, s, 0, mine, theirs))
        return jax.tree.map(np.asarray, outcome_laws(tables, s, acts))

    one = laws(_starf_fight(names, tables, ["Tackle"], ["Tackle"]), "Tackle", "Tackle")
    assert one.relevant.starf[0].tolist() == [True, False, False, False]
    rec = laws(_starf_fight(names, tables, ["Recycle"], ["Tackle"]), "Recycle", "Tackle")
    assert rec.relevant.starf[0].tolist() == [True, False, True, False]
    assert _starf(rec, 0, 2)[2] == STARF_DRAWS
    roar = _starf_fight(names, tables, ["Tackle"], ["Roar"], bench=True, foe_level=60)
    roar_laws = laws(roar, "Tackle", "Roar")
    assert roar_laws.relevant.starf[0].tolist() == [True, True, True, False]
    assert _starf(roar_laws, 0, 2) == (list(range(5)), [1] * 5, 5)
    bite = _starf_fight(
        names, tables, ["Bug Bite"], ["Roar"], foe_item="Starf Berry", bench=True, foe_level=60
    )
    assert laws(bite, "Bug Bite", "Roar").relevant.starf[0].tolist() == [True] * 4


# --- Nightmare (F-O5, G-31..G-34) -----------------------------------------------------------


def _nm(tables, names, mine, opp="Tackle", *, counter=3, bit=0, foe="Bidoof", first=0):
    """Monferno (``mine``) against a level-matched ``foe`` (Bidoof: Normal) asleep with
    ``counter`` turns and Nightmare bit ``bit``; side ``first`` moves first."""
    from tests.engine.test_e1_batch_f import _become
    from tests.engine.test_protect import _state

    s = _become(tables, names, _state(tables, names, mine=mine, opp=opp), 1, foe)
    spe = (200, 10) if first == 0 else (10, 200)
    s = s._replace(stats=s.stats.at[0, 0, Stat.SPE].set(spe[0]).at[1, 0, Stat.SPE].set(spe[1]))
    if counter:
        s = s.with_status(1, 0, Status.SLEEP, counter)
    return s._replace(nightmare=s.nightmare.at[1].set(bit))


def _quarter(s, side=1):
    return max(int(s.max_hp[side, 0]) // 4, 1)


def _step(tables, names, s, mine, theirs, outcome=O):
    return assert_matches_reference(tables, s, _acts(names, s, 0, mine, theirs), outcome)


def test_nightmare_on_a_sleeper_costs_a_quarter_at_the_turn_end(tables, names):
    """effect_script_0107.s:8 -> subscript_nightmare_start.s:9 sets the bit on a Normal-type
    sleeper (power 0: no type chart, bcp.c:2350-2357); the turn end runs
    subscript_nightmare_effect.s: Divide(maxHP, 4) (:8-10, bcp.c:1436-1448)."""
    s = _nm(tables, names, ["Nightmare"])
    out = _step(tables, names, s, "Nightmare", "Tackle")
    assert int(out.nightmare[1]) == 1 and int(out.status[1, 0]) == Status.SLEEP
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - _quarter(s)


@pytest.mark.parametrize("case", ["awake", "substitute", "already"])
def test_nightmare_fails_on_an_awake_substituted_or_haunted_target(tables, names, case):
    """effect_script_0107.s:5-7, :11-12: MOVE_STATUS_FAILED; an existing bit costs one quarter."""
    s = _nm(tables, names, ["Nightmare"], counter=0 if case == "awake" else 3)
    if case == "substitute":
        s = s._replace(substitute_hp=s.substitute_hp.at[1].set(5))
    if case == "already":
        s = s._replace(nightmare=s.nightmare.at[1].set(1))
    out = _step(tables, names, s, "Nightmare", "Tackle")
    assert int(out.nightmare[1]) == (case == "already")
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - (_quarter(s) if case == "already" else 0)


def test_a_wake_up_at_its_own_move_ends_the_nightmare(tables, names):
    """The sleeper's counter runs out at its move: subscript_wake_up.s:20 clears the bit (G-32),
    so the turn end costs nothing."""
    s = _nm(tables, names, ["Nightmare"], counter=1)
    out = _step(tables, names, s, "Nightmare", "Tackle")
    assert int(out.nightmare[1]) == 0 and int(out.status[1, 0]) == Status.NONE
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_wake_up_slap_keeps_the_bit_for_a_rest_after_it(tables, names):
    """subscript_heal_target_sleep.s:6 clears only the sleep: the bit stays, the foe's Rest puts
    it back to sleep, and the turn end costs a quarter of the full HP Rest gave (G-32)."""
    s = _nm(tables, names, ["Wake-Up Slap"], opp="Rest", bit=1)
    s = s._replace(
        hp=s.hp.at[1, 0].set(300), max_hp=s.max_hp.at[1, 0].set(300)
    )  # survives the slap
    out = _step(tables, names, s, "Wake-Up Slap", "Rest")
    assert int(out.status[1, 0]) == Status.SLEEP and int(out.nightmare[1]) == 1
    assert int(out.hp[1, 0]) == int(out.max_hp[1, 0]) - _quarter(s)


def test_magic_guard_keeps_the_bit_and_loses_nothing(tables, names):
    """subscript_nightmare_effect.s:5: CheckAbility Magic Guard jumps to the end."""
    s = _nm(tables, names, ["Protect"], bit=1)
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.MAGIC_GUARD))
    out = _step(tables, names, s, "Protect", "Tackle")
    assert int(out.nightmare[1]) == 1 and int(out.hp[1, 0]) == int(s.hp[1, 0])


@pytest.mark.parametrize("case", ["chesto", "lum", "shed skin", "no shed skin", "hydration"])
def test_the_foes_cures_end_the_nightmare_before_it_bites(tables, names, case):
    """Chesto / Lum at the turn end's USE_ITEM (bcp.c:1362; subscript_held_item_slp_restore.s:13),
    Shed Skin at its ABILITY check (:1355, 30 %: subscript_ability_restore_status.s:6) and Hydration
    in rain (battle_script.c:5626-5639, the same subscript) all run before NIGHTMARE (:1436)."""
    from nuzline.engine.field import with_weather
    from nuzline.engine.ids import Weather

    foe = "Phione" if case == "hydration" else "Bidoof"
    s = _nm(tables, names, ["Protect"], bit=1, foe=foe)
    outcome = O
    if case in ("chesto", "lum"):
        item = names.item("Chesto Berry" if case == "chesto" else "Lum Berry")
        s = s._replace(item=s.item.at[1, 0].set(item))
    if "shed skin" in case:
        s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.SHED_SKIN))
        outcome = O._replace(shed_skin=jnp.asarray([0, case == "shed skin"], jnp.int8))
    if case == "hydration":
        s = with_weather(s, Weather.RAIN)
    out = _step(tables, names, s, "Protect", "Tackle", outcome)
    cured = case != "no shed skin"
    assert int(out.nightmare[1]) == (not cured)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - (0 if cured else _quarter(s))


def test_insomnia_gained_by_skill_swap_ends_the_nightmare(tables, names):
    """The move end's RecoverStatusByAbility (bcp.c:3965-3967) on the defender, now Insomnia:
    subscript_ability_forbids_status.s:16-18 clears the sleep and the bit (G-32)."""
    s = _nm(tables, names, ["Skill Swap"], bit=1)
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.INSOMNIA))
    out = _step(tables, names, s, "Skill Swap", "Tackle")
    assert int(out.status[1, 0]) == Status.NONE and int(out.nightmare[1]) == 0
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])


def test_heal_bell_from_sleep_talk_ends_the_users_nightmare(tables, names):
    """BtlCmd_TryPartyStatusRefresh (battle_script.c:5039-5042): the user's status and bit."""
    s = _nm(tables, names, ["Sleep Talk", "Heal Bell"], counter=0)
    s = s.with_status(0, 0, Status.SLEEP, 3)._replace(nightmare=s.nightmare.at[0].set(1))
    out = _step(tables, names, s, "Sleep Talk", "Tackle")
    assert int(out.status[0, 0]) == Status.NONE and int(out.nightmare[0]) == 0


@pytest.mark.parametrize("baton", [False, True])
def test_a_switch_or_baton_pass_leaves_no_nightmare(tables, names, baton):
    """BattleSystem_UpdateAfterSwitch: statusVolatile cleared, Baton Pass keeping only
    VOLATILE_CONDITION_BATON_PASSED (battle_lib.c:2070, :2075; condition.h:72-76: no Nightmare)."""
    from nuzline.engine.step import _reset_battler
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _reset_battler as ref_reset

    s = _nm(tables, names, ["Protect"], bit=1)
    assert int(_reset_battler(s, 1, True, baton).nightmare[1]) == 0
    sd = to_ref(s).battlers[1]
    ref_reset(sd, baton)
    assert sd.nightmare == 0
