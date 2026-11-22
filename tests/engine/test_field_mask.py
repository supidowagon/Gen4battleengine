"""FCM (FCM.md section 4): ``field_mask`` (fieldConditionsMask) and ``weather_turns``
(FieldConditions.weatherTurns), the only field-condition state since step 3. On each cited
writer path the mask holds the decomp's bits and the engine equals the reference. Step 2 (4.6): the weather readers read the mask, so a corrupted mask (3.1: 0x7F, 0x5,
0x1 with a stale count) reads as the game reads it (3.2), engine equal to reference; the Gravity
readers read bits 12-14 (0x7FFF: Gravity 7, Kaizo re-arm -> permanent; 0x5000 - 3); the Uproar
readers read bits 8-11 (0xFFF: a bit of every battler, none uproaring); the Trick Room readers
bits 16-18 (0x40000 - 3: Trick Room 3; 0x1FFFF: a counter of 1, Kaizo re-arm). Step 4
(4.6): each row of 3.1 through ``sub_to_zero``; Acid Rain on the Pursuit KO paths of 1.3
(fcm-verify (c), (d))."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import MOVE_ACTIONS, gravity_banned
from nuzline.engine.damage import grounded, weather_ball_params
from nuzline.engine.effects import weather_fails, weather_heal_amount
from nuzline.engine.field import acid_rain, sub_to_zero
from nuzline.engine.format import SINGLES
from nuzline.engine.ids import (
    GRAVITY_TURNS,
    TRICK_ROOM_TURNS,
    WEATHER_TURNS,
    FieldCondition,
    Status,
    Weather,
)
from nuzline.engine.outcome import MAX_HITS, empty_outcome, live_hit_threshold
from nuzline.engine.state import BattleState, layout, make_state
from nuzline.engine.step import first_pursuer
from nuzline.reference.engine import _acid_rain as ref_acid_rain
from tests.engine.d2_states import four, with_moves
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
    """``s`` with a corrupted fieldConditionsMask ``mask`` and weatherTurns ``turns`` (FCM 3.1)."""
    return s._replace(
        field_mask=jnp.asarray(mask, s.field_mask.dtype),
        weather_turns=jnp.asarray(turns, s.weather_turns.dtype),
    )


def _hits():
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))


def _step(t, s, acts, outcome=None):
    """One turn, engine equal to reference (every field)."""
    return assert_matches_reference(t, s, acts, _hits() if outcome is None else outcome)


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
    assert int(s.field_mask) == 0


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


# --- FCM step 2, Trick Room readers (FCM 3.2 "Trick Room", 4.6) --------------------------------


@pytest.mark.parametrize("mask, up", [(0x40000 - 3, True), (0xFFFF, False)])
def test_trick_room_bits_reverse_the_order(tables, names, mask, up):
    """Any FIELD_CONDITION_TRICK_ROOM bit (16-18) reverses CompareBattlerSpeed (battle_lib.c:1432):
    the slower Geodude hits the 1 HP Monferno first, which never moves; 0xFFFF (every bit below
    them) has none."""
    s = _fight(tables, names, ("Tackle",), ("Tackle",))
    s = s._replace(hp=s.hp.at[0, 0].set(1), stats=s.stats.at[0, 0, 4].set(500))
    s = _masked(s, mask, 0)
    out = assert_matches_reference(tables, s, A("Tackle", "Tackle", s, names), _hits())
    assert (int(out.hp[1, 0]) == int(s.hp[1, 0])) is up


@pytest.mark.parametrize(
    "mask, move, rearm, after",
    [
        # (bits 12-14 hold Gravity 7 in each: the field pass takes it to 6, bcp.c:1229)
        (0x40000 - 3, "Trick Room", False, 0xEFFD),  # any bit: the move clears them all
        (0x1FFFF, "Growl", False, 0xEFFF),  # a counter of 1 ends at the turn end
        (0x1FFFF, "Growl", True, 0x5EFFF),  # Kaizo's sub_seq 251 sets it again
    ],
)
def test_trick_room_bits_toggle_and_count_down(tables, names, mask, move, rearm, after):
    """effect_script_0259.s:5 tests any Trick Room bit (:12 clears them, a borrowed counter's
    too); bcp.c:1824 takes 1 << 16 a turn and at 0 Kaizo's sub_seq 251 ORs TRICK_ROOM_INIT
    (fcm-verify (a), ruling 8: E3's one ``_trick_room_check``)."""
    t = _rearming(tables, "trick_room_rearm") if rearm else tables
    s = _masked(_fight(t, names, ("Trick Room", "Growl")), mask, 0)
    out = assert_matches_reference(t, s, A(move, "Growl", s, names), _hits())
    assert int(out.field_mask) == after


# --- FCM step 4, Acid Rain (FCM 1, 3.1, 4.6; ledger G-12) -----------------------------------

# FCM 3.1, each row: (start m, (m - 1, m - 2, m - 3) clamped at 0).
_SUB_ROWS = [
    (0, (0, 0, 0)),
    (0x1, (0, 0, 0)),
    (0x2, (0x1, 0, 0)),
    (0x4, (0x3, 0x2, 0x1)),
    (0x8, (0x7, 0x6, 0x5)),
    (0x10, (0xF, 0xE, 0xD)),
    (0x20, (0x1F, 0x1E, 0x1D)),
    (0x40, (0x3F, 0x3E, 0x3D)),
    (0x80, (0x7F, 0x7E, 0x7D)),
    (0x8000, (0x7FFF, 0x7FFE, 0x7FFD)),
    *((0x100 << b, tuple((0x100 << b) - k for k in (1, 2, 3))) for b in range(4)),
    *((c << 12, tuple(((c - 1) << 12) | (0x1000 - k) for k in (1, 2, 3))) for c in range(1, 8)),
    *((t << 16, tuple(((t - 1) << 16) | (0x10000 - k) for k in (1, 2, 3))) for t in range(1, 8)),
    (0x80 | 5 << 12, (0x507F, 0x507E, 0x507D)),
    (0x1 | 5 << 12, (0x5000, 0x4FFF, 0x4FFE)),
]


@pytest.mark.parametrize("start, results", _SUB_ROWS)
def test_sub_to_zero_gives_each_row_of_3_1(start, results):
    """OPCODE_SUB_TO_ZERO (battle_script.c:3282-3287) on fieldConditionsMask with faintedMon
    1-3 (subscript_pursuit.s:115): ``field.sub_to_zero``, ``field.acid_rain`` and the
    reference's ``_acid_rain`` give FCM 3.1's table."""
    blank = {k: jnp.zeros(shape, dtype) for k, (shape, dtype) in layout(SINGLES).items()}
    state = BattleState(**blank)._replace(field_mask=jnp.asarray(start, jnp.int32))
    ref = SimpleNamespace(field_mask=start)
    for k, want in zip((1, 2, 3), results, strict=True):
        assert int(sub_to_zero(start, k)) == want
        assert int(acid_rain(state, k, True).field_mask) == want
        ref.field_mask = start
        ref_acid_rain(ref, (k, 0))
        assert ref.field_mask == want


def _pursuit_fight(
    t, n, mine, theirs, foe="Geodude", my_item="", foe_item="", my_level=20, **start
):
    """A Monferno (``my_level``, ``mine``, ``my_item``) with a Chimchar behind it against a level
    40 ``foe`` (``theirs``, ``foe_item``) with a Geodude behind it; ``start``: make_state's."""
    me = replace(_spec(n, "Monferno", my_level, mine), item=n.item(my_item) if my_item else 0)
    them = replace(_spec(n, foe, 40, theirs), item=n.item(foe_item) if foe_item else 0)
    player = [me, _spec(n, "Chimchar", 20, ("Growl",))]
    return make_state(t, player, [them, _spec(n, "Geodude", 40, ("Growl",))], **start)


def test_the_players_pursuit_ko_subtracts_the_switcher(tables, names):
    """Permanent hail (0x80); the foe (battler 1) switches and the player's Pursuit KOs it
    (subscript_pursuit.s:39): faintedMon = 1 (:100, battle_script.c:1769-1770), 0x80 - 1 = 0x7F
    (:115). The turn end then runs rain, sand, sun (PERM bits) and hail (TEMP only: the stale
    weatherTurns 0 counts to -1 and never ends, FCM 3.2)."""
    s = _pursuit_fight(tables, names, ("Pursuit",), ("Growl",), start_condition=Weather.HAIL)
    s = s.replace_hp(1, 0, 1)
    out = _step(tables, s, A("Pursuit", MOVE_ACTIONS + 1, s, names))
    assert int(out.hp[1, 0]) == 0
    assert (int(out.field_mask), int(out.weather_turns)) == (0x7F, -1)


@pytest.mark.parametrize("item, mask", [("", 0x80), ("Life Orb", 0x7F)])
def test_the_foes_pursuit_ko_subtracts_battler_0_or_the_fallen_pursuer(tables, names, item, mask):
    """The player's 1 HP Monferno (battler 0) switches and the foe's Pursuit KOs it: faintedMon
    0, the mask unchanged (FCM verdict, B = 2). Holding Life Orb at 1 HP the pursuer falls to its
    recoil (TriggerHeldItemOnPivotMove, subscript_pursuit.s:94-95; battle_lib.c:7357-7365) after
    the KO: UPDATE_HP's TryFaintMon makes it faintedMon (FCM 1.3): 0x80 - 1."""
    s = _pursuit_fight(
        tables, names, ("Growl",), ("Pursuit",), foe_item=item, start_condition=Weather.HAIL
    )
    s = s.replace_hp(0, 0, 1)
    s = s.replace_hp(1, 0, 1) if item else s  # the recoil faints it (the hail chip would too)
    out = _step(tables, s, A(MOVE_ACTIONS + 1, "Pursuit", s, names))
    assert int(out.hp[0, 0]) == 0 and (int(out.hp[1, 0]) == 0) is bool(item)
    assert int(out.field_mask) == mask


def test_destiny_bond_clears_the_pursuers_bit_before_the_subtraction(tables, names):
    """fcm-verify (c): the foe's switcher holds Destiny Bond, the player's 1 HP pursuer with a
    spurious Uproar bit (0x100, FCM 3.1) the only condition up KOs it. Destiny Bond
    (subscript_pursuit.s:85) faints the pursuer at once (subscript_faint_check_destiny_bond.s:16
    -> battle_script.c:1803 -> battle_lib.c:2213 clears bit 8) and restores faintedMon = 1 (:25):
    0 - 1 clamps to 0."""
    s = _pursuit_fight(tables, names, ("Pursuit",), ("Growl",))
    s = _masked(s.replace_hp(0, 0, 1).replace_hp(1, 0, 1), 0x100, 0)
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(1))
    out = _step(tables, s, A("Pursuit", MOVE_ACTIONS + 1, s, names))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.field_mask) == 0


def test_a_recoil_fainted_pursuer_keeps_its_bit_for_the_subtraction(tables, names):
    """Against (c): the foe's 1 HP pursuer (battler 1, Life Orb) with a spurious bit 0x200 the
    only condition up KOs the player's switcher and falls to its recoil: UPDATE_HP's TryFaintMon
    only (battle_script.c:1769-1772), so its bit is still set at :115: 0x200 - 1 = 0x1FF, and its
    later CleanupFaintedMon (battle_lib.c:2213) finds bit 9 clear already. The switch then goes
    on (subscript_switch_pokemon.s:22, SwitchAndUpdateMon -> BattleSystem_UpdateAfterSwitch,
    battle_lib.c:2135) and clears the switcher's bit 8, which the borrow set: 0xFF."""
    s = _pursuit_fight(tables, names, ("Growl",), ("Pursuit",), foe_item="Life Orb")
    s = _masked(s.replace_hp(0, 0, 1).replace_hp(1, 0, 1), 0x200, 0)
    out = _step(tables, s, A(MOVE_ACTIONS + 1, "Pursuit", s, names))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.field_mask) == 0xFF


def test_pursuit_chasing_a_u_turn_subtracts_too(tables, names):
    """Under Gravity (0x5000) the foe's 1 HP Scyther U-turns and the slower player's Pursuit KOs it
    before the list (subscript_attack_then_switch_out.s:54, the same subscript_pursuit.s): 0x5000
    - 1 = 0x4FFF; the turn end takes 1 << 12 (bcp.c:1229): 0x3FFF. (A Monferno that the four
    weather chips of 0xFF faint loses its own bit 8 too, battle_lib.c:2213: level 35 survives.)"""
    s = _pursuit_fight(
        tables,
        names,
        ("Pursuit",),
        ("U-turn",),
        foe="Scyther",
        my_level=35,
        gravity=GRAVITY_TURNS,
    )
    s = s.replace_hp(1, 0, 1)
    out = _step(tables, s, A("Pursuit", "U-turn", s, names))
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) > 0 and int(out.pivot_battler) == 0
    assert int(out.field_mask) == 0x3FFF


def test_aftermath_fells_the_foes_pursuer_into_the_operand(tables, names):
    """The contact ability's recoil (from the merged guard test, G-13): the player's 1 HP Drifloon
    (Aftermath) switches, the foe's 1 HP pursuer KOs it and falls to Aftermath (subscript_pursuit.s
    :86-87, battle_lib.c:4358-4371): UPDATE_HP's TryFaintMon makes it faintedMon (FCM 1.3):
    0x80 - 1, as for the Life Orb case."""
    drifloon = _spec(names, "Drifloon", 5, ("Tackle",))
    slot = names.species_abilities(names.species("Drifloon")).index(names.ability("Aftermath"))
    s = make_state(
        tables,
        [replace(drifloon, ability_slot=slot), _spec(names, "Chimchar", 20, ("Growl",))],
        [_spec(names, "Houndour", 35, ("Pursuit",)), _spec(names, "Geodude", 40, ("Growl",))],
        start_condition=Weather.HAIL,
    )
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    out = _step(tables, s, A(MOVE_ACTIONS + 1, "Pursuit", s, names))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.field_mask) == 0x7F


def test_the_players_pursuer_felled_by_its_life_orb_subtracts_nothing(tables, names):
    """The player's 1 HP pursuer (battler 0, Life Orb) KOs the foe's switcher and falls to its
    recoil (battle_lib.c:7357-7365): faintedMon 0, the mask unchanged (merged guard test, G-13)."""
    s = _pursuit_fight(
        tables, names, ("Pursuit",), ("Growl",), my_item="Life Orb", start_condition=Weather.HAIL
    )
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    out = _step(tables, s, A("Pursuit", MOVE_ACTIONS + 1, s, names))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.field_mask) == 0x80


def test_destiny_bond_keeps_the_switcher_as_operand_before_rough_skin(tables, names):
    """The foe's 1 HP Carvanha (Rough Skin) holds Destiny Bond and switches; the player's 1 HP
    pursuer KOs it. Destiny Bond (subscript_pursuit.s:85) takes the pursuer down before Rough
    Skin (:86-87, which then fails on the attacker's HP, battle_lib.c:4258) and restores
    faintedMon = 1 (subscript_faint_check_destiny_bond.s:25): 0x80 - 1 (merged guard test, G-13)."""
    s = _pursuit_fight(
        tables, names, ("Pursuit",), ("Growl",), foe="Carvanha", start_condition=Weather.HAIL
    )
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    s = s._replace(destiny_bond=s.destiny_bond.at[1].set(1))
    out = _step(tables, s, A("Pursuit", MOVE_ACTIONS + 1, s, names))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.field_mask) == 0x7F


def test_the_first_pursuer_is_the_first_ready_battler_by_index(tables, names):
    """fcm-verify (d): BtlCmd_TryPursuit loops the battlers in index order (battle_script.c:6899)
    and breaks at the first that is ready (:6934), whatever the speeds: battler 0 switches and
    battlers 1 and 3 (battler 3 the faster) both chose Pursuit; battler 1 is taken."""
    s = four(tables, names)
    for b in (1, 3):
        s = with_moves(s, names, b, ("Pursuit",))
    s = s._replace(stats=s.stats.at[1, 1, 4].set(999))  # battler 3: party 1, slot 1
    acts = jnp.asarray([MOVE_ACTIONS + 2, 0, 0, 0], jnp.int32)
    moves = jnp.asarray([0, names.move("Pursuit"), 0, names.move("Pursuit")], jnp.int32)
    choices = (moves, jnp.zeros(4, bool), jnp.zeros(4, bool))
    who, found = first_pursuer(tables, s, 0, acts, choices)
    assert (int(who), bool(found)) == (1, True)
    assert not bool(first_pursuer(tables, s, 1, acts, choices)[1])  # battlers 0, 2: none ready


def test_future_sight_law_covers_a_corrupted_mask_that_loses_a_group(tables, names):
    """G-6's owed item: 0x8044 (sand T, hail T, deep fog) with weatherTurns 1 ends the turn as
    0x8040 (SANDSTORM counts 1 -> 0 and ends, subscript_sandstorm_end.s:5; HAILING counts to -1
    and goes on, bcp.c:1138-1211). The Future Sight hit (bcp.c:1758-1783) on a Sand Veil
    Sandshrew then reads fog without sand (bcp.c:2927, :2934): its threshold is in the law
    (``outcome.future_thresholds``' none / sand / hail grid, each with and without the start's
    fog), and the step hits at it and misses above it."""
    from nuzline.engine.enumerate import outcome_laws
    from nuzline.engine.outcome import future_threshold

    n = names
    s = make_state(
        tables, [_spec(n, "Abra", 30, ("Future Sight",))], [_spec(n, "Sandshrew", 30, ("Growl",))]
    )
    s = _masked(s, 0x8044, 1)._replace(
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(5),
        future_move=s.future_move.at[1].set(n.move("Future Sight")),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
    )
    live = int(future_threshold(tables, _masked(s, 0x8040, -1), 1))
    assert live != int(future_threshold(tables, s, 1))  # the start's (sand and fog) differs
    laws = outcome_laws(tables, s, jnp.asarray([0, 0]))
    assert bool(laws.future_values[1][min(live, 100)])
    left = []
    for roll in (live, live + 1):
        o = _hits()._replace(future_hit=jnp.asarray([1, roll], jnp.int8))
        out = assert_matches_reference(tables, s, (0, 0), o)
        assert (int(out.field_mask), int(out.weather_turns)) == (0x8040, -1)
        left.append(int(out.hp[1, 0]))
    assert left[0] == left[1] - 5  # the stored 5 HP hit lands at the live threshold only
