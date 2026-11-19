"""FCM step 1 (FCM.md 4.4): ``field_mask`` (fieldConditionsMask) and ``weather_turns``
(FieldConditions.weatherTurns) written beside the old field fields by every writer. On each cited
writer path the mask is clean (4.4: it equals the old fields) and the engine equals the
reference. Step 2 (4.6): the weather readers read the mask, so a corrupted mask (3.1: 0x7F, 0x5,
0x1 with a stale count) reads as the game reads it (3.2), engine equal to reference; the Gravity
readers read bits 12-14 (0x7FFF: Gravity 7, Kaizo re-arm -> permanent; 0x5000 - 3); the Uproar
readers read bits 8-11 (0xFFF: a bit of every battler, none uproaring)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS, gravity_banned
from nuzline.engine.damage import grounded, weather_ball_params
from nuzline.engine.effects import weather_fails, weather_heal_amount
from nuzline.engine.ids import (
    GRAVITY_TURNS,
    TRICK_ROOM_TURNS,
    WEATHER_TURNS,
    FieldCondition,
    Status,
    Weather,
    clean_field_mask,
)
from nuzline.engine.outcome import MAX_HITS, empty_outcome, live_hit_threshold
from nuzline.engine.state import make_state
from tests.engine.helpers import A, assert_matches_reference


def _spec(n, species, level, moves, ability_slot=0):
    return PokemonSpec(
        species=n.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(n.move(m) for m in moves) + (0,) * (4 - len(moves)),
        item=0,
        ability_slot=ability_slot,
        gender=0,
        friendship=70,
    )


def _fight(t, n, mine, opp=("Growl",), foe="Geodude", bench=(), foes=(), **start):
    """A level 20 Monferno (``mine``; ``bench`` species behind it) against a level 40 ``foe``
    (``opp``; ``foes`` species behind it); ``start``: make_state's start condition, gravity,
    trick_room."""
    player = [_spec(n, "Monferno", 20, mine)] + [_spec(n, b, 20, ("Growl",)) for b in bench]
    opponent = [_spec(n, foe, 40, opp)] + [_spec(n, f, 40, ("Growl",)) for f in foes]
    return make_state(t, player, opponent, **start)


def _masked(s, mask, turns):
    """``s`` with a corrupted fieldConditionsMask ``mask`` and weatherTurns ``turns`` (FCM 3.1);
    the old weather fields cleared (no weather reader reads them since step 2)."""
    return s._replace(
        field_mask=jnp.asarray(mask, s.field_mask.dtype),
        weather_turns=jnp.asarray(turns, s.weather_turns.dtype),
        weather=jnp.zeros_like(s.weather),
        weather_timer=jnp.zeros_like(s.weather_timer),
    )


def _hits():
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))


def clean_mask(s) -> int:
    """fieldConditionsMask of the old fields (FCM 4.4, ``ids.clean_field_mask``)."""
    return clean_field_mask(
        int(s.weather), int(s.weather_timer), int(s.gravity), int(s.trick_room), s.uproar.tolist()
    )


def _step(t, s, acts, outcome=None):
    """One turn, engine equal to reference (every field), then the clean-mask invariant."""
    out = assert_matches_reference(t, s, acts, _hits() if outcome is None else outcome)
    assert int(out.field_mask) == clean_mask(out)
    if int(out.weather_timer):
        assert int(out.weather_turns) == int(out.weather_timer)
    return out


def test_the_start_conditions_set_their_bits(tables, names):
    """subscript_overworld_hail.s:11 (PERM), _fog.s:11 (bit 15), Kaizo's area Gravity (sub_seq
    297, GRAVITY_INIT) and Trick Room (subscript_overworld_trick_room.s:14, TRICK_ROOM_INIT)."""
    cases = (
        ({"start_condition": Weather.HAIL}, FieldCondition.HAILING_PERM),
        ({"start_condition": Weather.FOG}, FieldCondition.DEEP_FOG),
        ({"gravity": GRAVITY_TURNS}, FieldCondition.GRAVITY_INIT),
        ({"trick_room": TRICK_ROOM_TURNS}, FieldCondition.TRICK_ROOM_INIT),
        ({}, 0),
    )
    for start, mask in cases:
        s = _fight(tables, names, ("Growl",), **start)
        assert (int(s.field_mask), int(s.weather_turns)) == (mask, 0)
        assert int(s.field_mask) == clean_mask(s)


def test_a_weather_move_sets_its_temp_bit_and_the_countdown_ends_it(tables, names):
    """Rain Dance under a permanent hail (effect_script_0136.s:8-10): WEATHER off, RAINING_TEMP,
    weatherTurns 5; each turn end counts it down (bcp.c:1121) and at 0 subscript_raining_end.s:5
    clears the bit."""
    s = _fight(tables, names, ("Rain Dance", "Growl"), start_condition=Weather.HAIL)
    s = _step(tables, s, A("Rain Dance", "Growl", s, names))
    assert int(s.field_mask) == FieldCondition.RAINING_TEMP
    assert int(s.weather_turns) == WEATHER_TURNS - 1
    for left in range(WEATHER_TURNS - 2, -1, -1):
        s = _step(tables, s, A("Growl", "Growl", s, names))
        assert int(s.weather_turns) == left
    assert (int(s.field_mask), int(s.weather)) == (0, Weather.NONE)


def test_a_weather_ability_sets_its_perm_bit_and_a_move_replaces_it(tables, names):
    """Sand Stream at battle start (subscript_sand_stream.s:11-12: SANDSTORM_PERM, weatherTurns
    untouched); Sunny Day then clears WEATHER and sets SUNNY_TEMP (effect_script_0137.s:8-10)."""
    s = _fight(tables, names, ("Sunny Day", "Growl"), foe="Hippopotas")
    assert (int(s.field_mask), int(s.weather_turns)) == (FieldCondition.SANDSTORM_PERM, 0)
    s = _step(tables, s, A("Growl", "Growl", s, names))
    assert int(s.field_mask) == FieldCondition.SANDSTORM_PERM
    s = _step(tables, s, A("Sunny Day", "Growl", s, names))
    assert int(s.field_mask) == FieldCondition.SUNNY_TEMP


def _rearming(t, flag):
    """``t`` with Kaizo's end-script flag ``flag`` set (``Tables.gravity_rearm`` /
    ``trick_room_rearm``, 1 in the Kaizo tables): same shapes, so no second compile."""
    return t._replace(**{flag: jnp.ones_like(getattr(t, flag))})


def test_gravity_bits_count_down_and_kaizo_rearms_them(tables, names):
    """subscript_gravity_start.s:9 ORs GRAVITY_INIT; bcp.c:1229 takes 1 << 12 a turn; at 0 Kaizo's
    sub_seq 238 ORs GRAVITY_INIT again (G-67), vanilla's end only prints."""
    s = _fight(tables, names, ("Gravity",))
    s = _step(tables, s, A("Gravity", "Growl", s, names))
    assert int(s.field_mask) == (GRAVITY_TURNS - 1) << 12
    for t, after in ((tables, 0), (_rearming(tables, "gravity_rearm"), GRAVITY_TURNS)):
        s = _fight(t, names, ("Growl",), gravity=1)
        s = _step(t, s, A("Growl", "Growl", s, names))
        assert int(s.field_mask) == after << 12


def test_trick_room_bits_toggle_and_kaizo_rearms_them(tables, names):
    """effect_script_0259.s: TRICK_ROOM_INIT (:6), or every Trick Room bit off while one is set
    (:12); bcp.c:1824 takes 1 << 16 a turn; at 0 Kaizo's sub_seq 251 (the move's script) sets
    TRICK_ROOM_INIT again (ruling 8, E3-2-4), vanilla's end only prints."""
    s = _fight(tables, names, ("Trick Room",))
    s = _step(tables, s, A("Trick Room", "Growl", s, names))
    assert int(s.field_mask) == (TRICK_ROOM_TURNS - 1) << 16
    s = _step(tables, s, A("Trick Room", "Growl", s, names))
    assert int(s.field_mask) == 0
    for t, after in ((tables, 0), (_rearming(tables, "trick_room_rearm"), TRICK_ROOM_TURNS)):
        s = _fight(t, names, ("Growl",), trick_room=1)
        s = _step(t, s, A("Growl", "Growl", s, names))
        assert int(s.field_mask) == after << 16


def test_the_uproar_bit_is_set_with_the_count_and_cleared_when_it_ends_or_faints(tables, names):
    """subscript_uproar.s:9-11 sets battler 0's bit (8); bcp.c:1528 / :1534 clear it when its
    Uproar ends; BattleSystem_CleanupFaintedMon clears it on a faint (battle_lib.c:2213)."""
    o = _hits()._replace(duration=jnp.asarray([3, 0], jnp.int8))
    s = _fight(tables, names, ("Uproar",), bench=("Chimchar",))
    s = _step(tables, s, A("Uproar", "Growl", s, names), o)
    assert int(s.field_mask) == 1 << 8
    s = _step(tables, s, A("Uproar", "Growl", s, names), o)
    assert int(s.field_mask) == 1 << 8
    s = _step(tables, s, A("Uproar", "Growl", s, names), o)
    assert (int(s.uproar[0]), int(s.field_mask)) == (0, 0)
    # The faster Monferno uproars at 1 HP, then the Geodude's Tackle faints it.
    s = _fight(tables, names, ("Uproar",), ("Tackle",), bench=("Chimchar",)).replace_hp(0, 0, 1)
    s = _step(tables, s, A("Uproar", "Tackle", s, names), o)
    assert (int(s.hp[0, 0]), int(s.uproar[0]), int(s.field_mask)) == (0, 0, 0)


# --- FCM step 2, weather readers (FCM 3.2, 4.6) ----------------------------------------------


@pytest.mark.parametrize(
    "mask, ball, rain_dance, hail, heal",
    [
        # all four TEMP+PERM but hail PERM: hail is the last Weather Ball writer, sun gives 2/3
        (0x7F, "ICE", True, True, (20, 30)),
        (0x5, "ROCK", True, False, (1, 4)),  # rain T + sand T: sand after rain
        (0x1, "WATER", True, False, (1, 4)),  # rain T with a stale count
        (0x3F, "FIRE", True, False, (20, 30)),  # rain, sand, sun T+P: Hail can be used
        (0x8000, None, False, False, (1, 4)),  # deep fog alone: x2, the move's own type
    ],
)
def test_weather_readers_read_each_group(tables, names, mask, ball, rain_dance, hail, heal):
    """Weather Ball (battle_script.c:6847-6865: power x2 on any WEATHER bit, the type of the last
    set group of rain, sand, sun, hail), the weather moves' fail tests (effect_script_0136.s:5
    the RAINING group, effect_script_0164.s:5 HAILING_TEMP only) and WeatherHPRecovery
    (battle_script.c:5970-5976: any SUNNY bit 2/3, other weather 1/4) on corrupted masks."""
    from nuzline.engine.ids import Type

    s = _masked(_fight(tables, names, ("Growl",)), mask, 0)
    move = names.move("Weather Ball")
    on, power, kind = (int(x) for x in weather_ball_params(tables, s, move))
    assert (on, power) == (1, 2 * int(tables.move_power[move]))
    assert kind == (int(tables.move_type[move]) if ball is None else int(Type[ball]))
    param = tables.move_param.astype(jnp.int32)
    assert bool(weather_fails(s, param[names.move("Rain Dance")])) is rain_dance
    assert bool(weather_fails(s, param[names.move("Hail")])) is hail
    max_hp = int(s.max_hp[0, 0])
    num, den = heal
    want = max_hp // 2 if mask == 0 else max(max_hp * num // den, 1)
    assert int(weather_heal_amount(tables, s, 0)) == want


def test_thunder_reads_sun_then_rain(tables, names):
    """bcp.c:2915-2917 (any SUNNY bit: hitRate 50) and :3025 (any RAINING bit: a sure hit, after
    the roll): with both set Thunder is sure; sun alone 50. Kaizo's effect 141 (Hurricane) shares
    both through MoveFlag.THUNDER (rom_diff.md:366-367, B13-3)."""
    s = _fight(tables, names, ("Thunder",), foe="Bidoof")
    move = names.move("Thunder")
    sun = FieldCondition.SUNNY_PERM
    both = sun | FieldCondition.RAINING_PERM
    assert int(live_hit_threshold(tables, _masked(s, sun, 0), 0, move)) == 50
    assert int(live_hit_threshold(tables, _masked(s, both, 0), 0, move)) == 100


def test_each_set_group_runs_the_turn_end_weather_effect(tables, names):
    """0x7F with a stale count 3: RAINING, SANDSTORM, SUNNY (PERM bits) go on, HAILING (TEMP only)
    counts weatherTurns down to 2 and goes on (bcp.c:1113-1211): EndOfTurnWeatherEffect runs four
    times per battler on the whole mask (script.c:5561-5645), the last writer a hail chip of
    1/16 each time. Ember's damage reads rain then sun (lib.c:7047-7070)."""
    s = _masked(_fight(tables, names, ("Ember",)), 0x7F, 3)
    out = assert_matches_reference(tables, s, A("Ember", "Growl", s, names), _hits())
    chip = max(int(s.max_hp[0, 0]) // 16, 1)
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == 4 * chip
    assert (int(out.field_mask), int(out.weather_turns)) == (0x7F, 2)


def test_two_temp_only_groups_count_down_twice_and_wrap(tables, names):
    """0x5 (rain T + sand T), weatherTurns 2: RAINING counts 2 -> 1 and goes on (a sand chip: the
    effect reads the whole mask), SANDSTORM 1 -> 0 ends (subscript_sandstorm_end.s:5): 0x1 with a
    stale count 0. The next turn RAINING counts 0 -> -1 (u32 wrap): it never ends (FCM 3.2)."""
    s = _masked(_fight(tables, names, ("Growl",)), 0x5, 2)
    out = assert_matches_reference(tables, s, A("Growl", "Growl", s, names), _hits())
    chip = max(int(s.max_hp[0, 0]) // 16, 1)
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == chip
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])  # Geodude: Rock/Ground, no sand chip
    assert (int(out.field_mask), int(out.weather_turns)) == (0x1, 0)
    again = assert_matches_reference(tables, out, A("Growl", "Growl", out, names), _hits())
    assert (int(again.field_mask), int(again.weather_turns)) == (0x1, -1)
    assert int(again.hp[0, 0]) == int(out.hp[0, 0])


def test_snow_warning_cleans_a_corrupted_mask(tables, names):
    """Snow Warning switching in under 0x7F: its own HAILING_PERM is clear (battle_lib.c:3776), so
    subscript_snow_warning.s:12-13 clears FIELD_CONDITION_WEATHER and sets HAILING_PERM: 0x80;
    weatherTurns stays (3). Then one hail state: a chip on Monferno, none on the Ice-type Snover."""
    s = _masked(_fight(tables, names, ("Growl",), foes=("Snover",)), 0x7F, 3)
    out = assert_matches_reference(tables, s, A("Growl", MOVE_ACTIONS + 1, s, names), _hits())
    assert (int(out.field_mask), int(out.weather_turns)) == (FieldCondition.HAILING_PERM, 3)
    chip = max(int(s.max_hp[0, 0]) // 16, 1)
    assert int(s.hp[0, 0]) - int(out.hp[0, 0]) == chip


def test_rain_dance_fails_under_0x7f_and_hail_succeeds_under_0x3f(tables, names):
    """effect_script_0136.s:5: Rain Dance fails while any RAINING bit is set (0x7F unchanged but
    the hail countdown); effect_script_0164.s:5: Hail tests HAILING_TEMP only, so under 0x3F it
    clears WEATHER and sets HAILING_TEMP with weatherTurns 5 (:8-10), counted to 4 at the turn
    end."""
    s = _masked(_fight(tables, names, ("Rain Dance", "Hail")), 0x7F, 3)
    out = assert_matches_reference(tables, s, A("Rain Dance", "Growl", s, names), _hits())
    assert (int(out.field_mask), int(out.weather_turns)) == (0x7F, 2)
    s = _masked(s, 0x3F, 3)
    out = assert_matches_reference(tables, s, A("Hail", "Growl", s, names), _hits())
    assert int(out.field_mask) == FieldCondition.HAILING_TEMP
    assert int(out.weather_turns) == WEATHER_TURNS - 1


# --- FCM step 2, Gravity readers (FCM 3.2 "Gravity", 4.6) -------------------------------------


@pytest.mark.parametrize("mask, after", [(0x7FFF, 0x6FFF), (0x5000 - 3, 0x3FFD), (0xFFF, 0x4FFF)])
def test_gravity_readers_read_the_counter_bits(tables, names, mask, after):
    """Any FIELD_CONDITION_GRAVITY bit (12-14) is Gravity up for every reader: the move ban
    (Move_FailsInHighGravity, battle_lib.c:3398), a Flying type grounded (BattlerIsGrounded,
    :5506), accuracy x10/6 (bcp.c:2972); 0xFFF (every weather and Uproar bit) has none. The
    Gravity move fails while one is set (effect_script_0215.s:5), else ORs GRAVITY_INIT
    (subscript_gravity_start.s:9); the turn end takes 1 << 12 (bcp.c:1229)."""
    s = _masked(_fight(tables, names, ("Gravity", "Scratch"), foe="Pidgey"), mask, 0)
    up = bool(mask & FieldCondition.GRAVITY)
    assert bool(gravity_banned(s, jnp.asarray([names.move("Fly")]))[0]) is up
    assert bool(grounded(tables, s, 1, 0)) is up  # Pidgey: Normal / Flying
    assert int(live_hit_threshold(tables, s, 0, names.move("Scratch"))) == (166 if up else 100)
    out = assert_matches_reference(tables, s, A("Gravity", "Growl", s, names), _hits())
    assert int(out.field_mask) == after


def test_a_borrowed_gravity_counter_runs_seven_turns_and_kaizo_rearms_it(tables, names):
    """0x7000 (FCM 3.1: a borrow leaves Gravity 7): bcp.c:1229 takes 1 << 12 a turn, and at 0
    Kaizo's sub_seq 238 ORs GRAVITY_INIT back in (G-67): Gravity never ends."""
    t = _rearming(tables, "gravity_rearm")
    s = _masked(_fight(t, names, ("Growl",)), 7 << 12, 0)
    for left in (6, 5, 4, 3, 2, 1, GRAVITY_TURNS):
        s = assert_matches_reference(t, s, A("Growl", "Growl", s, names), _hits())
        assert int(s.field_mask) == left << 12


# --- FCM step 2, Uproar readers (FCM 3.2 "Status scripts", "Spurious Uproar bit", 4.6) --------


def test_spurious_uproar_bits_keep_everyone_awake_until_their_battler_leaves(tables, names):
    """0xFFF: every Uproar bit (8-11) with no battler uproaring (FCM 3.1). Any FIELD_CONDITION_UPROAR
    bit wakes the sleeping Monferno at its move (bcp.c:2432, subscript_wake_up.s:6), fails its Rest
    (subscript_rest.s:11) and the foe's Hypnosis (subscript_fall_asleep.s:16); the turn end's Uproar
    check runs only for a battler with the volatile (bcp.c:1506), so the bits stay; a switch clears
    the switcher's bit only (battle_lib.c:2135)."""
    s = _fight(tables, names, ("Rest",), ("Hypnosis",), bench=("Chimchar",))
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2).with_status(0, 0, Status.SLEEP, 3)
    s = _masked(s, 0xFFF, 0)
    out = assert_matches_reference(tables, s, A("Rest", "Hypnosis", s, names), _hits())
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # no Rest heal, the weather chips
    assert int(out.field_mask) == 0xFFF
    out = assert_matches_reference(
        tables, out, A(MOVE_ACTIONS + 1, "Hypnosis", out, names), _hits()
    )
    assert int(out.status[0, 1]) == Status.NONE  # Chimchar: Hypnosis still fails
    assert int(out.field_mask) == 0xEFF
