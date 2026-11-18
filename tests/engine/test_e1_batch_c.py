"""E1 Batch C (rulings G-44 .. G-52, docs/kaizo/e1-mechanics-ledger.md): hand cases, each step
checked against the reference engine (``assert_matches_reference``).

Deep fog (G-44); the abilities Ice Body (G-45), White Smoke (G-46), Vital Spirit (G-47), Speed
Boost (G-48) and Iron Fist (G-49); the Kaizo ROM's overlay16 accuracy change for effect 141
(G-50); ``IntermediateData`` keyed on the game id (G-51).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.games import GAMES, KAIZO, VANILLA
from nuzline.data.tables import to_numpy
from nuzline.data.trainer import IntermediateData
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, IMPLEMENTED_WEATHER, Support
from nuzline.engine.damage import damage_table
from nuzline.engine.hooks import intimidate
from nuzline.engine.ids import AbilityId, MoveFlag, Stat, Status, Weather
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome, hit_threshold
from nuzline.engine.state import make_state
from nuzline.engine.step import replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tools.build_tables import _move_flags
from tools.effect_map import KAIZO_THUNDER_EFFECTS, THUNDER_EFFECT

O = empty_outcome()
BATCH_C = (
    AbilityId.ICE_BODY,
    AbilityId.WHITE_SMOKE,
    AbilityId.VITAL_SPIRIT,
    AbilityId.SPEED_BOOST,
    AbilityId.IRON_FIST,
)


def _weather(s, weather, timer=0):
    return s._replace(
        weather=jnp.asarray(weather, s.weather.dtype),
        weather_timer=jnp.asarray(timer, s.weather_timer.dtype),
    )


def _ability(s, side, slot, ability):
    return s._replace(ability=s.ability.at[side, slot].set(ability))


def _hp(s, side, slot, hp, max_hp):
    return s._replace(hp=s.hp.at[side, slot].set(hp), max_hp=s.max_hp.at[side, slot].set(max_hp))


def test_the_family_is_covered(kaizo_names):
    """Fog and the five abilities are graded EXACT and mapped from both games' ability names."""
    assert IMPLEMENTED_WEATHER[Weather.FOG] == Support.EXACT
    for ability in BATCH_C:
        assert IMPLEMENTED_ABILITIES[ability] == Support.EXACT
    labels = [c.label for c in kaizo_names.rom_change_conditions]
    assert not any("CheckMoveHit" in label for label in labels), labels


# --- G-44 deep fog ----------------------------------------------------------------------


def test_fog_cuts_the_hit_rate_and_never_ends(tables, names):
    """bcp.c:2934-2936 x6/10 inside NO_CLOUD_NINE; :1213-1225 only prints a message: no
    countdown, no HP change. A roll of 61 hits in clear weather and misses in fog."""
    s = roark_state(tables, names)
    leer = A("Leer", "Leer", s, names)
    fog = _weather(s, Weather.FOG)
    assert hit_threshold(tables, fog, leer).tolist() == [60, 60]
    o = O._replace(hit=O.hit.at[0, 0].set(61))
    clear = assert_matches_reference(tables, s, leer, o)
    foggy = assert_matches_reference(tables, fog, leer, o)
    assert int(clear.stages[1, Stat.DEF]) == -1  # the player's Leer lands
    assert int(foggy.stages[1, Stat.DEF]) == 0  # and misses in fog
    assert int(foggy.weather) == Weather.FOG and int(foggy.weather_timer) == 0
    assert foggy.hp.tolist() == fog.hp.tolist()


def test_a_weather_move_and_a_weather_ability_replace_fog(tables, names):
    """effect_script_0115.s:8 and subscript_drizzle.s:11 clear FIELD_CONDITION_WEATHER (fog
    included, condition.h:131-135) before they set their own weather."""
    s = _weather(roark_state(tables, names, player_moves=["Rain Dance"]), Weather.FOG)
    out = assert_matches_reference(tables, s, A("Rain Dance", "Leer", s, names), O)
    assert int(out.weather) == Weather.RAIN and int(out.weather_timer) > 0
    s = _ability(_weather(roark_state(tables, names), Weather.FOG), 1, 1, AbilityId.DRIZZLE)
    out = assert_matches_reference(tables, s, A("Leer", MOVE_ACTIONS + 1, s, names), O)
    assert int(out.weather) == Weather.RAIN and int(out.weather_timer) == 0


# --- G-45 Ice Body ------------------------------------------------------------------------


def _snorunt(tables, names, hp):
    s = roark_state(tables, names, player_species="Snorunt", player_moves=["Leer"],
                    player_ability="Ice Body")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.ICE_BODY
    return _hp(_weather(s, Weather.HAIL), 0, 0, hp, 40)


def test_ice_body_heals_in_hail_and_shows_its_ability(tables, names):
    """battle_script.c:5597-5604: Divide(maxHP, 16) below full HP, no chip at full HP;
    subscript_weather_continues.s:_076 tags the healed battler (the AI learns it)."""
    s = _snorunt(tables, names, 30)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.hp[0, 0]) == 32 and int(out.ai_ability_known[0]) != 0
    full = _snorunt(tables, names, 40)
    out = assert_matches_reference(tables, full, A("Leer", "Leer", full, names), O)
    assert int(out.hp[0, 0]) == 40 and int(out.ai_ability_known[0]) == 0
    clear = _weather(s, Weather.NONE)
    out = assert_matches_reference(tables, clear, A("Leer", "Leer", clear, names), O)
    assert int(out.hp[0, 0]) == 30


# --- G-46 White Smoke ---------------------------------------------------------------------


def _torkoal(tables, names):
    s = roark_state(tables, names, player_species="Torkoal", player_moves=["Leer"],
                    player_ability="White Smoke")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.WHITE_SMOKE
    return s


def test_white_smoke_blocks_a_foes_drop_and_intimidate(tables, names):
    """battle_script.c:2883-2884: White Smoke is Clear Body's twin (a drop by another battler,
    Battler_IgnorableAbility); its message is built in C, so the AI learns nothing."""
    s = _torkoal(tables, names)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.stages[0, Stat.DEF]) == 0 and int(out.ai_ability_known[0]) == 0
    bare = _ability(s, 0, 0, AbilityId.NONE)
    out = assert_matches_reference(tables, bare, A("Leer", "Leer", bare, names), O)
    assert int(out.stages[0, Stat.DEF]) == -1
    # subscript_intimidate.s through BtlCmd_ChangeStatStage: the opponent's newcomer.
    s = _ability(s, 1, 1, AbilityId.INTIMIDATE)
    out = assert_matches_reference(tables, s, A("Leer", MOVE_ACTIONS + 1, s, names), O)
    assert int(out.stages[0, Stat.ATK]) == 0
    bare = _ability(s, 0, 0, AbilityId.NONE)
    out = assert_matches_reference(tables, bare, A("Leer", MOVE_ACTIONS + 1, bare, names), O)
    assert int(out.stages[0, Stat.ATK]) == -1
    lead = _ability(s, 1, 0, AbilityId.INTIMIDATE)
    assert int(intimidate(lead, 1).stages[0, Stat.ATK]) == 0


# --- G-47 Vital Spirit --------------------------------------------------------------------


def _mankey(tables, names, opp_moves=None):
    s = roark_state(tables, names, player_species="Mankey", player_moves=["Leer"],
                    player_ability="Vital Spirit", opp_moves=opp_moves)  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.VITAL_SPIRIT
    return s


def test_vital_spirit_stops_sleep_and_shows_its_ability(tables, names):
    """subscript_fall_asleep.s:24 (an ignorable Vital Spirit, beside Insomnia) -> :_202/:_221
    "stayed awake" with TAG_NICKNAME_ABILITY: the AI learns it."""
    s = _mankey(tables, names, ["Hypnosis"])
    out = assert_matches_reference(tables, s, A("Leer", "Hypnosis", s, names), O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0
    bare = _ability(s, 0, 0, AbilityId.NONE)
    out = assert_matches_reference(tables, bare, A("Leer", "Hypnosis", bare, names), O)
    assert int(out.status[0, 0]) == Status.SLEEP


def test_vital_spirit_wakes_its_sleeping_holder(tables, names):
    """BattleSystem_RecoverStatusByAbility (battle_lib.c:4404-4405), at the move end: an asleep
    Vital Spirit holder is cured (subscript_ability_forbids_status.s tags it)."""
    s = _mankey(tables, names)
    s = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
    )
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0


# --- G-48 Speed Boost ---------------------------------------------------------------------


def _yanma(tables, names):
    s = roark_state(tables, names, player_species="Yanma", player_moves=["Leer"],
                    player_ability="Speed Boost")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.SPEED_BOOST
    return s


def test_speed_boost_at_the_turn_end(tables, names):
    """battle_lib.c:3555-3564: +1 Speed at MON_COND_CHECK_STATE_ABILITY for a holder with HP
    below +6; its message is built in C (no reveal)."""
    s = _yanma(tables, names)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.stages[0, Stat.SPE]) == 1 and int(out.ai_ability_known[0]) == 0
    top = s._replace(stages=s.stages.at[0, Stat.SPE].set(6))
    out = assert_matches_reference(tables, top, A("Leer", "Leer", top, names), O)
    assert int(out.stages[0, Stat.SPE]) == 6


def test_speed_boost_skips_the_turn_its_holder_entered(tables, names):
    """fakeOutTurnNumber != totalTurns + 1 (:3558): a holder switched in this turn waits a turn."""
    s = _ability(roark_state(tables, names), 0, 1, AbilityId.SPEED_BOOST)
    out = assert_matches_reference(tables, s, A(MOVE_ACTIONS + 1, "Leer", s, names), O)
    assert int(out.active[0]) == 1 and int(out.stages[0, Stat.SPE]) == 0
    move = int(np.flatnonzero(np.asarray(out.moves[0, 1]))[0])
    out = assert_matches_reference(tables, out, A(move, "Leer", out, names), O)
    assert int(out.stages[0, Stat.SPE]) == 1


# --- G-49 Iron Fist -----------------------------------------------------------------------


def test_iron_fist_powers_up_punching_moves(tables, names):
    """battle_lib.c:6913-6918: a sPunchingMoves move (MoveFlag.PUNCH) x12/10 after Rivalry; any
    other move as before."""
    s = roark_state(tables, names, player_species="Hitmonchan",
                    player_moves=["Mach Punch", "Tackle"], player_ability="Iron Fist")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.IRON_FIST
    punch, tackle = names.move("Mach Punch"), names.move("Tackle")
    assert int(tables.move_flags[punch]) & MoveFlag.PUNCH
    assert not int(tables.move_flags[tackle]) & MoveFlag.PUNCH
    bare = _ability(s, 0, 0, AbilityId.NONE)
    power = np.asarray(tables.move_power).copy()
    power[punch] = power[punch] * 12 // 10
    boosted = tables._replace(move_power=jnp.asarray(power, jnp.asarray(tables.move_power).dtype))
    got = damage_table(tables, s, 0, punch, False).tolist()
    assert got == damage_table(boosted, bare, 0, punch, False).tolist()
    assert got != damage_table(tables, bare, 0, punch, False).tolist()
    assert (
        damage_table(tables, s, 0, tackle, False).tolist()
        == damage_table(tables, bare, 0, tackle, False).tolist()
    )
    assert_matches_reference(tables, s, A("Mach Punch", "Leer", s, names), O)


# --- G-50 the Kaizo accuracy change -------------------------------------------------------


def test_kaizo_effect_141_is_thunder_for_the_weather_accuracy(kaizo_tables, kaizo_names):
    """The patched CheckMoveHitAccuracy / CheckMoveHitOverrides compare effect 152 or 141
    (rom:overlay16@0x0224F068, @0x0224F384): Kaizo's Hurricane gets 50 in sun and a sure hit in
    rain, as Thunder; vanilla's builder keeps the flag on effect 152 alone."""
    move = {"flags": [], "effect": 141}
    lists = {"sound": [], "punch": []}
    assert KAIZO_THUNDER_EFFECTS == {141} and THUNDER_EFFECT == 152
    assert _move_flags(move, 1, lists, kaizo=True) & MoveFlag.THUNDER
    assert not _move_flags(move, 1, lists) & MoveFlag.THUNDER
    hurricane = kaizo_names.move("Hurricane")
    flags = np.asarray(kaizo_tables.move_flags)
    assert flags[hurricane] & MoveFlag.THUNDER
    effect = np.asarray(kaizo_tables.move_ai_effect)
    assert {int(m) for m in np.flatnonzero(flags & MoveFlag.THUNDER)} == {
        int(m) for m in np.flatnonzero(np.isin(effect, [152, 141]))
    }


def test_kaizo_hurricane_hit_rate_in_sun_and_rain(kaizo_tables, kaizo_names, kaizo_roark_party):
    """outcome._rate on a Kaizo state (Kaizo tables, Kaizo Roark's party on both sides): accuracy
    70 at the neutral stage sum; in sun hitRate = 50 before the stage scaling (bcp.c:2915-2917),
    so 50; in rain CheckMoveHitOverrides' Thunder branch clears the miss (bcp.c:3023-3027), a sure
    hit: threshold exactly 100."""
    hurricane = kaizo_names.move("Hurricane")
    assert int(kaizo_tables.move_accuracy[hurricane]) == 70
    party = list(kaizo_roark_party)
    s = make_state(kaizo_tables, party, party)
    s = s._replace(moves=s.moves.at[0, 0, 0].set(hurricane))
    acts = jnp.array([0, 0], jnp.int32)
    rate = {
        w: int(hit_threshold(kaizo_tables, _weather(s, w), acts)[0])
        for w in (Weather.NONE, Weather.SUN, Weather.RAIN)
    }
    assert rate == {Weather.NONE: 70, Weather.SUN: 50, Weather.RAIN: 100}


# --- G-51 the intermediates' game ---------------------------------------------------------


def test_intermediate_data_keys_on_the_game_id():
    """``patched_builder`` follows the game id (by default the game whose intermediate_dir it
    is), never a file in the directory."""
    vanilla = GAMES[VANILLA].intermediate_dir
    assert IntermediateData(vanilla).game == VANILLA
    assert not IntermediateData(vanilla).patched_builder
    assert IntermediateData(vanilla, KAIZO).patched_builder
    if (GAMES[KAIZO].intermediate_dir / "species.json").exists():
        assert IntermediateData(GAMES[KAIZO].intermediate_dir).patched_builder
    with pytest.raises(ValueError, match="no game's intermediate directory"):
        IntermediateData(vanilla / "..")


# --- review cases -------------------------------------------------------------------------


def test_speed_boost_on_a_faint_replacement(tables, names):
    """A faint replacement enters after the turn end (ReplaceFainted, bcp.c:1848) with
    ``fakeOutTurnNumber = totalTurns + 1`` (battle_lib.c:2195-2197): nothing on entry, and the
    next turn's end is not its entry turn (:3558), so it boosts there."""
    s = _ability(roark_state(tables, names), 0, 1, AbilityId.SPEED_BOOST)
    s = s._replace(hp=s.hp.at[0, 0].set(0))
    out = replace(tables, s, 0, 1)
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(s), 0, 1))
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert int(out.active[0]) == 1 and int(out.stages[0, Stat.SPE]) == 0
    move = int(np.flatnonzero(np.asarray(out.moves[0, 1]))[0])
    out = assert_matches_reference(tables, out, A(move, "Leer", out, names), O)
    assert int(out.stages[0, Stat.SPE]) == 1


def test_white_smoke_does_not_block_its_own_drop(tables, names):
    """battle_script.c:2876: the Clear Body / White Smoke test runs only when the attacker
    is not ``sideEffectMon``, so the holder's own move still lowers its stats."""
    move = "Hammer Arm"  # the supported self drop (Speed -1 on every hit, effect 218)
    s = roark_state(tables, names, player_species="Torkoal", player_moves=[move],
                    player_ability="White Smoke")  # fmt: skip
    out = assert_matches_reference(tables, s, A(move, "Leer", s, names), O)
    assert int(out.stages[0, Stat.SPE]) == -1


def test_vital_spirit_blocks_yawn(tables, names):
    """subscript_yawn.s:6: an ignorable Vital Spirit ends Yawn before it starts its counter."""
    s = _mankey(tables, names, ["Yawn"])
    out = assert_matches_reference(tables, s, A("Leer", "Yawn", s, names), O)
    assert int(out.yawn[0]) == 0
    # Yawn reads the decomp ability (effects.yawn_blocks): Mankey's other slot, Anger Point.
    bare = _ability(s, 0, 0, AbilityId.NONE)
    bare = bare._replace(ability_slot=bare.ability_slot.at[0, 0].set(1))
    out = assert_matches_reference(tables, bare, A("Leer", "Yawn", bare, names), O)
    assert int(out.yawn[0]) != 0
