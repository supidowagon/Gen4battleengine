"""The end-of-turn field-condition pass, ``BattleControllerPlayer_CheckFieldConditions``
(battle_controller_player.c:932-1254): side-condition countdowns, the weather countdown and
the sandstorm / hail chip (``BtlCmd_EndOfTurnWeatherEffect``, battle_script.c:5561-5647,
from ``subscript_weather_continues.s``), one faint check after the weather step, then
Gravity."""

import jax.numpy as jnp
import pytest

from nuzline.engine.ids import Status, Type, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import WINNER_OPPONENT, WINNER_PLAYER
from tests.engine.helpers import A, assert_matches_reference, jit_step, roark_state

_SIDE_TIMERS = ("reflect", "light_screen", "mist", "safeguard", "tailwind", "lucky_chant")


def _types(s, side, *types):
    return s._replace(types=s.types.at[side, s.active[side]].set(jnp.array(types, jnp.int8)))


def _normal_foe(s):
    """Roark's Geodude made Normal-type, so it is not sandstorm-immune."""
    return _types(s, 1, Type.NORMAL, Type.NORMAL)


def _weather(s, weather, timer=0):
    return s._replace(
        weather=jnp.asarray(weather, s.weather.dtype),
        weather_timer=jnp.asarray(timer, s.weather_timer.dtype),
    )


def _max_hp(s, side, value, hp=None):
    mon = int(s.active[side])
    return s._replace(
        max_hp=s.max_hp.at[side, mon].set(value),
        hp=s.hp.at[side, mon].set(value if hp is None else hp),
    )


def _leer(tables, names, s, o=None):
    return jit_step(tables, s, A("Leer", "Leer", s, names), o or empty_outcome())


def _lost(before, after, side):
    mon = int(before.active[side])
    return int(before.hp[side, mon]) - int(after.hp[side, mon])


@pytest.mark.slow
def test_screens_count_down_and_end(tables, names):
    # bcp.c:947-1083: each side's Reflect, Light Screen, Mist, Safeguard, Tailwind and
    # Lucky Chant turns drop by 1; at 0 the condition ends. Gravity likewise (:1227-1238).
    s = roark_state(tables, names)
    for name in _SIDE_TIMERS:
        s = s._replace(**{name: jnp.array([1, 5], jnp.int8)})
    s = s._replace(gravity=jnp.asarray(2, s.gravity.dtype))
    out = _leer(tables, names, s)
    for name in _SIDE_TIMERS:
        assert getattr(out, name).tolist() == [0, 4], name
    assert int(out.gravity) == 1
    assert int(_leer(tables, names, out).gravity) == 0


def test_trick_room_is_not_counted_in_the_field_pass(tables, names):
    # Trick Room counts down in the side-condition pass at turn end (bcp.c:1823-1826),
    # which is not implemented: the field pass leaves it alone.
    s = roark_state(tables, names)._replace(trick_room=jnp.asarray(3, jnp.int8))
    assert int(_leer(tables, names, s).trick_room) == 3


@pytest.mark.slow
def test_permanent_sand_never_ends(tables, names):
    s = _weather(roark_state(tables, names), Weather.SANDSTORM)
    for _ in range(3):
        s = _leer(tables, names, s)
        assert int(s.weather) == Weather.SANDSTORM and int(s.weather_timer) == 0


@pytest.mark.slow
def test_temporary_sand_ends_at_zero(tables, names):
    # bcp.c:1138-1161: --weatherTurns; at 0 subscript_sandstorm_end.s clears the weather
    # and no chip runs that turn (only subscript_weather_continues.s chips).
    s = _weather(roark_state(tables, names), Weather.SANDSTORM, timer=2)
    mid = _leer(tables, names, s)
    assert int(mid.weather) == Weather.SANDSTORM and int(mid.weather_timer) == 1
    assert _lost(s, mid, 0) == max(int(s.max_hp[0, 0]) // 16, 1)
    end = _leer(tables, names, mid)
    assert int(end.weather) == Weather.NONE and int(end.weather_timer) == 0
    assert _lost(mid, end, 0) == 0


@pytest.mark.slow
@pytest.mark.parametrize("weather", [Weather.RAIN, Weather.SUN, Weather.HAIL])
def test_temporary_weather_counts_down(tables, names, weather):
    s = _weather(roark_state(tables, names), weather, timer=1)
    out = _leer(tables, names, s)
    assert int(out.weather) == Weather.NONE and int(out.weather_timer) == 0


def test_fog_never_counts_down(tables, names):
    # FIELD_COND_CHECK_STATE_DEEP_FOG (bcp.c:1213-1225) only prints a message.
    s = _weather(roark_state(tables, names), Weather.FOG, timer=1)
    out = _leer(tables, names, s)
    assert int(out.weather) == Weather.FOG and int(out.weather_timer) == 1


@pytest.mark.slow
@pytest.mark.parametrize(("max_hp", "chip"), [(46, 2), (17, 1), (15, 1)])
def test_sand_chip_values(tables, names, max_hp, chip):
    # BattleSystem_Divide(maxHP * -1, 16): truncated, at least 1 (battle_lib.c:3599).
    s = _max_hp(_weather(roark_state(tables, names), Weather.SANDSTORM), 0, max_hp)
    out = _leer(tables, names, s)  # Monferno, Fire/Fighting
    assert _lost(s, out, 0) == chip
    assert _lost(s, out, 1) == 0  # Geodude, Rock/Ground


@pytest.mark.slow
@pytest.mark.parametrize("immune", [Type.ROCK, Type.GROUND, Type.STEEL])
def test_sand_type_immunities(tables, names, immune):
    s = _weather(roark_state(tables, names), Weather.SANDSTORM)
    s = _types(s, 0, Type.NORMAL, immune)
    assert _lost(s, _leer(tables, names, s), 0) == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("species", "ability"),
    [("Cacnea", "Sand Veil"), ("Clefairy", "Magic Guard")],
)
def test_sand_ability_immunities(tables, names, species, ability):
    # script.c:5576 (Sand Veil); subscript_weather_continues.s CheckAbility Magic Guard.
    s = roark_state(tables, names, player_species=species, player_moves=["Leer"],
                    player_ability=ability)  # fmt: skip
    s = _normal_foe(_weather(s, Weather.SANDSTORM))
    out = _leer(tables, names, s)
    assert _lost(s, out, 0) == 0
    assert _lost(s, out, 1) == max(int(s.max_hp[1, 0]) // 16, 1)


def test_cloud_nine_alive_stops_the_chip_but_not_the_countdown(tables, names):
    # NO_CLOUD_NINE (include/battle/common.h:63): no alive battler with Cloud Nine.
    s = roark_state(tables, names, player_species="Psyduck", player_moves=["Leer"],
                    player_ability="Cloud Nine")  # fmt: skip
    s = _normal_foe(_weather(s, Weather.SANDSTORM, timer=3))
    out = _leer(tables, names, s)
    assert _lost(s, out, 0) == 0 and _lost(s, out, 1) == 0
    assert int(out.weather_timer) == 2


@pytest.mark.slow
def test_hail_chip_and_immunities(tables, names):
    # script.c, the hail branch: Ice types and Snow Cloak take nothing; Ice Body heals
    # Divide(maxHP, 16) below full HP instead.
    s = _normal_foe(_weather(roark_state(tables, names), Weather.HAIL))
    out = _leer(tables, names, s)
    assert _lost(s, out, 0) == max(int(s.max_hp[0, 0]) // 16, 1)
    ice = _types(s, 0, Type.ICE, Type.ICE)
    assert _lost(ice, _leer(tables, names, ice), 0) == 0
    cloak = roark_state(tables, names, player_species="Swinub", player_moves=["Leer"],
                        player_ability="Snow Cloak")  # fmt: skip
    cloak = _types(_normal_foe(_weather(cloak, Weather.HAIL)), 0, Type.NORMAL, Type.NORMAL)
    assert _lost(cloak, _leer(tables, names, cloak), 0) == 0
    body = roark_state(tables, names, player_species="Snorunt", player_moves=["Leer"],
                       player_ability="Ice Body")  # fmt: skip
    body = _types(_normal_foe(_weather(body, Weather.HAIL)), 0, Type.NORMAL, Type.NORMAL)
    body = _max_hp(body, 0, 40, hp=30)
    assert _lost(body, _leer(tables, names, body), 0) == -2


@pytest.mark.slow
@pytest.mark.parametrize("tie", [0, 1])
def test_sand_double_faint_is_a_loss(tables, names, tie):
    # Both chips run inside one subscript; the battle end is checked once the field loop
    # resumes (bcp.c:939-944): both last Pokemon down is a draw, played as a loss
    # (_faint_check), whichever battler chipped first. Gravity is not reached.
    s = roark_state(tables, names, player_count=1, opp_count=1)
    s = _normal_foe(_weather(s, Weather.SANDSTORM))
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    s = s._replace(gravity=jnp.asarray(3, s.gravity.dtype))
    o = empty_outcome()._replace(end_speed_tie=jnp.asarray(tie, jnp.int8))
    s = s._replace(stats=s.stats.at[1, 0, 4].set(s.stats[0, 0, 4]))  # equal speed
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), o)
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.over) == 1 and int(out.winner) == WINNER_OPPONENT
    assert int(out.player_faints) == 1 and int(out.gravity) == 3


def _fixed_states(tables, names):
    """``(state, (player move, opponent move))`` pairs."""
    leer = ("Leer", "Leer")
    base = roark_state(tables, names)
    for weather in Weather:
        for timer in (0, 1, 3):
            s = _normal_foe(_weather(base, weather, timer))
            for name in _SIDE_TIMERS:
                s = s._replace(**{name: jnp.array([timer, 1], jnp.int8)})
            yield s._replace(gravity=jnp.asarray(timer, jnp.int8)), leer
            yield _max_hp(s, 0, 17, hp=1), leer  # the chip faints the player lead
            yield _types(s, 0, Type.ICE, Type.STEEL), leer
    for species, ability in (
        ("Cacnea", "Sand Veil"),
        ("Clefairy", "Magic Guard"),
        ("Psyduck", "Cloud Nine"),
        ("Swinub", "Snow Cloak"),
        ("Snorunt", "Ice Body"),
    ):
        s = roark_state(tables, names, player_species=species, player_moves=["Leer"],
                        player_ability=ability)  # fmt: skip
        s = _types(_normal_foe(s), 0, Type.NORMAL, Type.NORMAL)
        s = _max_hp(s, 0, 40, hp=30)
        for weather in (Weather.SANDSTORM, Weather.HAIL):
            yield _weather(s, weather), leer
            # A holder the foe's Tackle faints is no longer alive (COUNT_ALIVE_BATTLERS):
            # a fainted Cloud Nine holder does not stop the foe's chip.
            yield _weather(s, weather).replace_hp(0, 0, 1), ("Leer", "Tackle")


def test_field_pass_matches_reference(tables, names):
    for s, (player, opp) in _fixed_states(tables, names):
        assert_matches_reference(tables, s, A(player, opp, s, names), empty_outcome())


def test_fainted_cloud_nine_holder_does_not_stop_the_chip(tables, names):
    s = roark_state(tables, names, player_species="Psyduck", player_moves=["Leer"],
                    player_ability="Cloud Nine")  # fmt: skip
    s = _normal_foe(_weather(s, Weather.SANDSTORM)).replace_hp(0, 0, 1)
    out = jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == 0
    assert _lost(s, out, 1) == max(int(s.max_hp[1, 0]) // 16, 1)


def test_a_tailwind_that_ends_still_orders_the_end_of_turn(tables, names):
    # monSpeedOrder is sorted once after the last action (bcp.c:887), before the field
    # pass ends the opponent's Tailwind: the opponent (faster only with Tailwind) still
    # runs its poison check first, faints first, and the player wins.
    s = roark_state(tables, names, player_count=1, opp_count=1)
    spe = int(s.stats[0, 0, 4])
    s = s._replace(
        stats=s.stats.at[1, 0, 4].set(spe // 2 + 1),
        tailwind=s.tailwind.at[1].set(1),
        status=s.status.at[:, 0].set(Status.POISON),
    )
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.tailwind[1]) == 0
    assert int(out.over) == 1 and int(out.winner) == WINNER_PLAYER
    assert int(out.hp[0, 0]) == 1
