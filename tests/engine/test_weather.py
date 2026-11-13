"""Weather inside a turn, and the abilities that read it.

* Sandstorm raises a Rock-type defender's SpD x1.5 (battle_lib.c:6925-6928); rain and sun
  scale Water and Fire damage (:7046-7075). Both sit inside ``NO_CLOUD_NINE``
  (include/battle/common.h:63-64): an alive Cloud Nine / Air Lock battler turns them off,
  like the end-of-turn chip (``test_field_pass.py``).
* Sand Veil (in sand) and Snow Cloak (in hail) cut the hit rate x80/100 after
  Compoundeyes and before deep fog, all inside ``NO_CLOUD_NINE``; they are read through
  ``Battler_IgnorableAbility``, so Mold Breaker pierces them
  (battle_controller_player.c:2922-2936).
* Swift Swim doubles Speed in rain (battle_lib.c:1228-1238, ``NO_CLOUD_NINE``), after the
  stage and before paralysis (:1260) and Tailwind (:1274).
* Sun stops a freeze unless a battler with HP left has Cloud Nine or Air Lock:
  ``subscript_freeze.s:6`` runs ``CheckIgnoreWeather`` (battle_script.c:9195-9206) before the
  sun check at :7 (K3b2 weather lock audit).
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.damage import damage_table
from nuzline.engine.ids import AbilityId, Stat, Status, Weather
from nuzline.engine.order import effective_speed
from nuzline.engine.outcome import empty_outcome, hit_threshold
from tests.engine.helpers import A, assert_matches_reference, roark_state

SPD, SPE = Stat.SPD - 1, Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe


def _weather(s, weather):
    return s._replace(weather=jnp.asarray(weather, s.weather.dtype))


def _table(tables, s, move, side=0):
    return damage_table(tables, s, side, move, False).tolist()


def _psyduck(tables, names, moves):
    """Psyduck with Cloud Nine (its second ability slot) against Roark's Geodude."""
    s = roark_state(tables, names, player_species="Psyduck", player_moves=moves,
                    player_ability="Cloud Nine")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.CLOUD_NINE
    return s


def test_rock_spdef_boost_in_sand(tables, names):
    # Monferno's Ember into Geodude (Rock/Ground): sand is SpD x1.5 before the stage.
    s = roark_state(tables, names, player_moves=["Ember"])
    ember = names.move("Ember")
    boosted = s._replace(stats=s.stats.at[1, 0, SPD].set(s.stats[1, 0, SPD] * 15 // 10))
    sand = _table(tables, _weather(s, Weather.SANDSTORM), ember)
    assert sand == _table(tables, boosted, ember)
    assert sand != _table(tables, s, ember)


@pytest.mark.slow
def test_cloud_nine_suppresses_sand_boost_and_chip(tables, names):
    s = _psyduck(tables, names, ["Water Gun", "Leer"])
    gun = names.move("Water Gun")
    plain = _table(tables, s, gun)
    for weather in (Weather.SANDSTORM, Weather.RAIN, Weather.SUN):
        assert _table(tables, _weather(s, weather), gun) == plain, weather
    sand = _weather(s, Weather.SANDSTORM)
    out = assert_matches_reference(tables, sand, A("Leer", "Leer", sand, names), empty_outcome())
    assert int(out.hp[0, 0]) == int(sand.hp[0, 0])  # Water-type Psyduck takes no chip
    # A fainted holder no longer counts (COUNT_ALIVE_BATTLERS): with no Cloud Nine the
    # boost applies and rain raises Water Gun.
    bare = s._replace(ability_slot=s.ability_slot.at[0, 0].set(0))  # Damp
    assert _table(tables, _weather(bare, Weather.SANDSTORM), gun) != plain
    assert _table(tables, _weather(bare, Weather.RAIN), gun) != plain


def _cacnea_foe(tables, names, **kw):
    """Roark's lead replaced by Cacnea (Sand Veil, its only ability)."""
    s = roark_state(tables, names, opp_species="Cacnea", **kw)
    assert int(s.ability[1, 0]) == AbilityId.SAND_VEIL
    return s


def test_sand_veil_accuracy_80_percent(tables, names):
    s = _cacnea_foe(tables, names)
    leer = A("Leer", "Leer", s, names)  # Leer: accuracy 100
    assert int(hit_threshold(tables, s, leer)[0]) == 100
    sand = _weather(s, Weather.SANDSTORM)
    assert int(hit_threshold(tables, sand, leer)[0]) == 80
    assert int(hit_threshold(tables, _weather(s, Weather.HAIL), leer)[0]) == 100
    # After Compoundeyes: Tackle 95 * 130 // 100 = 123, then * 80 // 100 = 98.
    eyes = _cacnea_foe(tables, names, player_species="Venonat", player_ability="Compoundeyes",
                       player_moves=["Tackle"])  # fmt: skip
    eyes = _weather(eyes, Weather.SANDSTORM)
    assert int(hit_threshold(tables, eyes, A("Tackle", "Leer", eyes, names))[0]) == 98
    # Mold Breaker pierces it (Battler_IgnorableAbility); Cloud Nine turns it off.
    cranidos = _weather(_cacnea_foe(tables, names, player_species="Cranidos",
                                    player_moves=["Leer"]), Weather.SANDSTORM)  # fmt: skip
    assert int(cranidos.ability[0, 0]) == AbilityId.MOLD_BREAKER
    assert int(hit_threshold(tables, cranidos, A("Leer", "Leer", cranidos, names))[0]) == 100
    cloud = _weather(_cacnea_foe(tables, names, player_species="Psyduck", player_moves=["Leer"],
                                 player_ability="Cloud Nine"), Weather.SANDSTORM)  # fmt: skip
    assert int(hit_threshold(tables, cloud, A("Leer", "Leer", cloud, names))[0]) == 100


def test_sand_veil_of_the_incoming_pokemon(tables, names):
    # A switch goes first, so the move meets the incoming Pokemon's ability.
    s = _weather(roark_state(tables, names, player_species="Cacnea"), Weather.SANDSTORM)
    assert int(s.ability[0, 0]) == AbilityId.SAND_VEIL
    swap = s._replace(active=s.active.at[0].set(1))  # Prinplup out, Cacnea in slot 0
    to_cacnea = jnp.array([4 + 0, names.move_slot(swap, 1, "Leer")], jnp.int32)
    assert int(hit_threshold(tables, swap, to_cacnea)[1]) == 80


def test_snow_cloak_in_hail(tables, names):
    s = roark_state(tables, names, player_species="Swinub", player_moves=["Leer"],
                    player_ability="Snow Cloak")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.SNOW_CLOAK
    leer = A("Leer", "Leer", s, names)
    assert int(hit_threshold(tables, _weather(s, Weather.HAIL), leer)[1]) == 80
    assert int(hit_threshold(tables, _weather(s, Weather.SANDSTORM), leer)[1]) == 100
    assert int(hit_threshold(tables, s, leer)[1]) == 100
    assert int(hit_threshold(tables, _weather(s, Weather.FOG), leer)[1]) == 60


def test_fog_is_gated_on_cloud_nine(tables, names):
    s = _weather(_psyduck(tables, names, ["Leer"]), Weather.FOG)
    leer = A("Leer", "Leer", s, names)
    assert hit_threshold(tables, s, leer).tolist() == [100, 100]
    bare = s._replace(ability_slot=s.ability_slot.at[0, 0].set(0))  # Damp
    assert hit_threshold(tables, bare, leer).tolist() == [60, 60]


def _goldeen(tables, names, speed):
    s = roark_state(tables, names, player_species="Goldeen", player_moves=["Leer"],
                    player_ability="Swift Swim")  # fmt: skip
    assert int(s.ability[0, 0]) == AbilityId.SWIFT_SWIM
    return s._replace(stats=s.stats.at[0, 0, SPE].set(speed))


def test_swift_swim_doubles_speed_in_rain_unless_cloud_nine(tables, names):
    s = _goldeen(tables, names, 7)
    assert int(effective_speed(tables, s, 0)) == 7
    assert int(effective_speed(tables, _weather(s, Weather.SANDSTORM), 0)) == 7
    rain = _weather(s, Weather.RAIN)
    assert int(effective_speed(tables, rain, 0)) == 14
    # Before paralysis: 7 * 2 // 4 = 3, not 7 // 4 * 2 = 2.
    assert int(effective_speed(tables, rain.with_status(0, 0, Status.PARALYSIS), 0)) == 3
    # Cloud Nine on the foe (Psyduck's second slot) turns rain off for the speed check.
    psyduck = names.species("Psyduck")
    cloudy = rain._replace(
        species=rain.species.at[1, 0].set(psyduck),
        ability_slot=rain.ability_slot.at[1, 0].set(1),
        ability=rain.ability.at[1, 0].set(AbilityId.CLOUD_NINE),
    )
    assert int(effective_speed(tables, cloudy, 0)) == 7
    # A fainted Cloud Nine holder does not count.
    assert int(effective_speed(tables, cloudy.replace_hp(1, 0, 0), 0)) == 14


def _fixed_states(tables, names):
    """``(state, (player move, opponent move))``: the holders, the Cloud Nine holder on
    either side, in every weather, with speeds that make Swift Swim decide the order."""
    leer = ("Leer", "Leer")
    holders = (
        ("Cacnea", "Sand Veil", ["Leer", "Absorb"]),
        ("Swinub", "Snow Cloak", ["Leer", "Powder Snow"]),
        ("Goldeen", "Swift Swim", ["Leer", "Water Gun"]),
        ("Psyduck", "Cloud Nine", ["Leer", "Water Gun"]),
        ("Monferno", None, ["Leer", "Ember"]),
    )
    for species, ability, moves in holders:
        base = roark_state(tables, names, player_species=species, player_moves=moves,
                           player_ability=ability)  # fmt: skip
        spe = int(base.stats[0, 0, SPE])
        # The foe is one point faster: Swift Swim in rain makes the player first.
        base = base._replace(stats=base.stats.at[1, 0, SPE].set(spe + 1))
        for weather in Weather:
            s = _weather(base, weather)
            yield s, leer
            yield s, (moves[1], "Rock Throw")
            yield s.with_status(0, 0, Status.PARALYSIS), (moves[1], "Tackle")
            # Cloud Nine on the foe's side too (Psyduck's second slot).
            cloudy = s._replace(
                species=s.species.at[1, 0].set(names.species("Psyduck")),
                ability_slot=s.ability_slot.at[1, 0].set(1),
                ability=s.ability.at[1, 0].set(AbilityId.CLOUD_NINE),
            )
            yield cloudy, (moves[1], "Tackle")


@pytest.mark.parametrize("hit", [0, 1])
def test_weather_abilities_match_reference(tables, names, hit):
    o = empty_outcome()._replace(hit=jnp.full_like(empty_outcome().hit, hit))
    for s, (player, opp) in _fixed_states(tables, names):
        assert_matches_reference(tables, s, A(player, opp, s, names), o)


# --- freeze in sun under a weather lock (subscript_freeze.s:6-7) -----------------------

FREEZE = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))  # every proc


def _freeze_case(tables, names, holder):
    """Powder Snow (10 % freeze) into Roark's Geodude in sun. ``holder``: who has Cloud
    Nine: "user" (the player's Psyduck), "target" (the opponent's lead made a Psyduck) or "none" (the Psyduck's Damp slot)."""
    s = roark_state(tables, names, player_species="Psyduck", player_moves=["Powder Snow"],
                    player_ability="Cloud Nine")  # fmt: skip
    if holder != "user":
        s = s._replace(ability_slot=s.ability_slot.at[0, 0].set(0))  # Damp
    if holder == "target":
        psyduck, slot = int(s.species[0, 0]), 1  # Psyduck's Cloud Nine slot
        s = s._replace(
            species=s.species.at[1, 0].set(psyduck),
            form=s.form.at[1, 0].set(0),
            ability_slot=s.ability_slot.at[1, 0].set(slot),
            types=s.types.at[1, 0].set(s.types[0, 0]),
        )
    # Enough HP that the hit never faints the target before its secondary effect.
    s = s._replace(max_hp=s.max_hp.at[1, 0].set(999), hp=s.hp.at[1, 0].set(999))
    return _weather(s, Weather.SUN)


@pytest.mark.parametrize(("holder", "frozen"), [("user", True), ("target", True), ("none", False)])
def test_freeze_in_sun_needs_no_weather_lock(tables, names, holder, frozen):
    s = _freeze_case(tables, names, holder)
    out = assert_matches_reference(tables, s, A("Powder Snow", "Leer", s, names), FREEZE)
    assert (int(out.status[1, 0]) == Status.FREEZE) == frozen


def test_a_fainted_holder_keeps_no_lock(tables, names):
    # COUNT_ALIVE_BATTLERS (battle_lib.c:2951-2957): a holder with no HP does not count. The
    # opponent's Psyduck at 0 HP is no longer on the field as a battler with HP: its own
    # freeze cannot happen (it has fainted), so test the read itself.
    from nuzline.engine.effects import can_have_status

    s = _freeze_case(tables, names, "user")
    assert bool(can_have_status(tables, s, 1, 0, Status.FREEZE))
    fainted = s._replace(hp=s.hp.at[0, 0].set(0))
    assert not bool(can_have_status(tables, fainted, 1, 0, Status.FREEZE))


@pytest.mark.slow
@pytest.mark.parametrize("holder", ["user", "target", "none"])
@pytest.mark.parametrize("order", ["freezer_first", "freezer_second"])
def test_freeze_in_sun_equals_bruteforce(tables, names, holder, order):
    # The secondary is read only when the target can be frozen (enumerate's status gate,
    # judged on the turn-start field: the lock can only leave, never come, before the first
    # mover's hit); a freeze by the first mover makes the target's thaw roll relevant.
    from tests.engine.test_enumerate import assert_equals_bruteforce

    s = _freeze_case(tables, names, holder)
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))  # no crits: a smaller product
    spe = s.stats[0, 0, SPE]
    faster = spe + 1 if order == "freezer_second" else max(int(spe) - 1, 1)
    s = s._replace(stats=s.stats.at[1, 0, SPE].set(faster))
    width, brute = assert_equals_bruteforce(tables, s, A("Powder Snow", "Tackle", s, names))
    assert width <= brute
