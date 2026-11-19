"""``fieldConditionsMask`` and ``FieldConditions.weatherTurns`` as ``BattleState.field_mask`` and
``BattleState.weather_turns`` (FCM build sheet, step 1 of section 4.4): the writers, pure ``jnp``.

Every writer of the old field fields (``weather``, ``weather_timer``, ``gravity``,
``trick_room``, the per-battler ``uproar``) also writes the mask through these, bit for bit as
the decomp writes ``fieldConditionsMask`` (bits ``ids.FieldCondition``, include/constants/
battle/condition.h:110-149). Step 2 (4.4): the weather readers read the weather bits (bits 0-7,
deep fog 15) and ``weather_turns`` through the views below (``weather_bits``, ``rain`` ..
``fog``, ``weather_ball_type``, ``weather_state``), each as its decomp test reads the mask, so a
mask with several groups (FCM 3.2) reads as the game reads it; the Gravity readers read bits
12-14 (``gravity_on``, ``gravity_count``); the Uproar and Trick Room readers still read the old
fields. On every state the engine
reaches the mask is clean: at most one weather group, TEMP-only with ``weather_turns ==
weather_timer`` or PERM-only; the Gravity / Trick Room bits are ``gravity`` / ``trick_room``;
Uproar bit ``8 + b`` is ``uproar[b] != 0``. ``weather_turns`` is written only by the weather
moves and the TEMP-only countdown (stale outside a TEMP-only weather, as the decomp's).

Each writer takes a ``cond`` (``bool``) and leaves the state where it is false. Battlers are
passed in (doubles-state-design section 5): the Uproar bit is ``8 + b`` of battler ``b``."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.ids import (
    FIELD_CONDITION_GRAVITY_SHIFT,
    FIELD_CONDITION_TRICK_ROOM_SHIFT,
    FIELD_CONDITION_UPROAR_SHIFT,
    WEATHER_COUNTDOWN,
    WEATHER_PERM_BIT,
    WEATHER_TEMP_BIT,
    FieldCondition,
    Type,
    Weather,
)
from nuzline.engine.state import BattleState

_I32 = jnp.int32
_WEATHER = int(FieldCondition.WEATHER)
_RAINING = int(FieldCondition.RAINING)
_SANDSTORM = int(FieldCondition.SANDSTORM)
_SUNNY = int(FieldCondition.SUNNY)
_HAILING = int(FieldCondition.HAILING)
_DEEP_FOG = int(FieldCondition.DEEP_FOG)
# FIELD_CONDITION_SOLAR_DOWN (condition.h:136-139): every weather bit but SUNNY's.
_SOLAR_DOWN = _WEATHER & ~_SUNNY
# The weather states of CheckFieldConditions in order (battle_controller_player.c:907-912,
# :1113-1225) as (TEMP, PERM) bits: RAINING .. HAILING count a TEMP-only group down; DEEP_FOG
# (:1213-1225) has one bit and never counts down (as a PERM bit: it always goes on).
WEATHER_STATES = (
    *((int(temp), int(perm)) for temp, perm in WEATHER_COUNTDOWN),
    (0, _DEEP_FOG),
)
# The groups an observer sees, by ``Weather`` id (1 sun .. 5 fog): one message per set group.
_SEEN_GROUPS = (_SUNNY, _RAINING, _SANDSTORM, _HAILING, _DEEP_FOG)
_GRAVITY = int(FieldCondition.GRAVITY)
_TRICK_ROOM = int(FieldCondition.TRICK_ROOM)


def _mask(state: BattleState) -> jnp.ndarray:
    return state.field_mask.astype(_I32)


def _put(state: BattleState, mask, cond) -> BattleState:
    value = jnp.where(jnp.asarray(cond), mask, state.field_mask)
    return state._replace(field_mask=value.astype(state.field_mask.dtype))


def _bit(table, weather) -> jnp.ndarray:
    return jnp.asarray([int(b) for b in table], _I32)[jnp.asarray(weather, _I32)]


def uproar_bit(b) -> jnp.ndarray:
    """``int32``: battler ``b``'s ``FIELD_CONDITION_UPROAR`` bit (``FlagIndex(b) << 8``,
    condition.h:118, :147)."""
    return jnp.left_shift(jnp.int32(1), FIELD_CONDITION_UPROAR_SHIFT + jnp.asarray(b, _I32))


def weather_move(state: BattleState, weather, turns, cond) -> BattleState:
    """A weather move's success (effect_script_0115.s, _0136.s, _0137.s, _0164.s): ``FLAG_OFF
    FIELD_CONDITION_WEATHER`` (every weather bit and deep fog, :8), ``FLAG_ON`` its TEMP bit (:9)
    and ``weatherTurns = turns`` (5, :10, plus the user's rock param, :12-14)."""
    mask = (_mask(state) & ~_WEATHER) | _bit(WEATHER_TEMP_BIT, weather)
    state = _put(state, mask, cond)
    value = jnp.where(jnp.asarray(cond), jnp.asarray(turns, _I32), state.weather_turns)
    return state._replace(weather_turns=value.astype(state.weather_turns.dtype))


def weather_ability(state: BattleState, weather, cond) -> BattleState:
    """A switch-in weather ability (``SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES``, battle_lib.c:
    3740-3790): skipped while its own PERM bit is set (:3749, :3758, :3767, :3776); else its
    script (subscript_drizzle.s, _sand_stream.s, _drought.s :11-12, _snow_warning.s :12-13)
    clears ``FIELD_CONDITION_WEATHER`` and sets the PERM bit. ``weatherTurns`` is untouched."""
    perm = _bit(WEATHER_PERM_BIT, weather)
    mask = _mask(state)
    runs = jnp.asarray(cond) & ((mask & perm) == 0)
    return _put(state, (mask & ~_WEATHER) | perm, runs)


def weather_state(state: BattleState, temp, perm, live) -> tuple[BattleState, jnp.ndarray]:
    """One weather state of CheckFieldConditions (battle_controller_player.c:1113-1225) where
    ``live``, for the group of bits ``temp | perm`` (``WEATHER_STATES``): ``(state, goes_on)``.
    A set group with its PERM bit goes on (subscript_weather_continues.s); a TEMP-only one counts
    the one shared ``weatherTurns`` down (``--weatherTurns == 0``, :1121, :1146, :1171, :1196):
    at 0 its end script clears its TEMP bit (subscript_raining_end.s, _sandstorm_end.s,
    _sunny_end.s, _hailing_end.s :5) and nothing else of it runs that turn, else it goes on. Two
    TEMP-only groups count it down twice a turn; past 0 it goes negative (the u32 wraps) and
    ``--x == 0`` never holds again. Deep fog (``temp`` 0) always goes on (:1213-1225)."""
    mask, turns = _mask(state), state.weather_turns.astype(_I32)
    temp, perm = jnp.asarray(temp, _I32), jnp.asarray(perm, _I32)
    on = jnp.asarray(live) & ((mask & (temp | perm)) != 0)
    counts = on & ((mask & perm) == 0)
    turns = jnp.where(counts, turns - 1, turns)
    ends = counts & (turns == 0)
    state = state._replace(weather_turns=turns.astype(state.weather_turns.dtype))
    return _put(state, mask & ~temp, ends), on & ~ends


def weather_bits(state: BattleState) -> jnp.ndarray:
    """``int32``: the mask's ``FIELD_CONDITION_WEATHER`` bits (condition.h:131-135: the four
    weathers, TEMP and PERM, and deep fog), what the weather readers test. A what-if view of the
    weather is such a value too (``outcome``, ``enumerate``, ``order.effective_speed``)."""
    return _mask(state) & _WEATHER


def rain(w) -> jnp.ndarray:
    """``WEATHER_IS_RAIN`` (include/battle/common.h:68) of the weather bits ``w``."""
    return (jnp.asarray(w, _I32) & _RAINING) != 0


def sand(w) -> jnp.ndarray:
    """``WEATHER_IS_SAND`` (common.h:69)."""
    return (jnp.asarray(w, _I32) & _SANDSTORM) != 0


def sun(w) -> jnp.ndarray:
    """``WEATHER_IS_SUN`` (common.h:70)."""
    return (jnp.asarray(w, _I32) & _SUNNY) != 0


def hail(w) -> jnp.ndarray:
    """``WEATHER_IS_HAIL`` (common.h:71)."""
    return (jnp.asarray(w, _I32) & _HAILING) != 0


def fog(w) -> jnp.ndarray:
    """``WEATHER_IS_FOG`` (common.h:72)."""
    return (jnp.asarray(w, _I32) & _DEEP_FOG) != 0


def any_weather(w) -> jnp.ndarray:
    """Any ``FIELD_CONDITION_WEATHER`` bit, fog included (``NO_WEATHER``'s first test,
    common.h:65; CalcWeatherBallParams' power test, battle_script.c:6848)."""
    return (jnp.asarray(w, _I32) & _WEATHER) != 0


def solar_down(w) -> jnp.ndarray:
    """Any ``FIELD_CONDITION_SOLAR_DOWN`` bit (condition.h:136-139: rain, sand, hail, fog)."""
    return (jnp.asarray(w, _I32) & _SOLAR_DOWN) != 0


def weather_ball_type(w, default) -> jnp.ndarray:
    """``int32``: the type Weather Ball's three readers write (``BtlCmd_CalcWeatherBallParams``,
    battle_script.c:6851-6865; ``CalcMoveType``, battle_lib.c:7896-7911; ``TrainerAI_MoveType``,
    trainer_ai.c:3223-3238): ``if`` after ``if``, so the last set group wins: rain Water < sand
    Rock < sun Fire < hail Ice; ``default`` (the move's own) when none is set (fog alone)."""
    out = jnp.asarray(default, _I32)
    for test, kind in ((rain, Type.WATER), (sand, Type.ROCK), (sun, Type.FIRE), (hail, Type.ICE)):
        out = jnp.where(test(w), int(kind), out)
    return out


def ability_bits(w, weather) -> jnp.ndarray:
    """``int32``: the weather bits ``w`` after a weather ability that sets ``weather`` (an
    ``ids.Weather``; NONE: no weather ability, ``w``) acts: skipped while its own PERM bit is set
    (battle_lib.c:3749, :3758, :3767, :3776), else ``FIELD_CONDITION_WEATHER`` off and the PERM
    bit on (subscript_drizzle.s etc. :11-12). A what-if view (``damage.entry_weather``,
    ``outcome``)."""
    w = jnp.asarray(w, _I32)
    perm = _bit(WEATHER_PERM_BIT, weather)
    sets = (jnp.asarray(weather, _I32) != Weather.NONE) & ((w & perm) == 0)
    return jnp.where(sets, (w & ~_WEATHER) | perm, w)


def weather_move_fails(w, weather, whole_group) -> jnp.ndarray:
    """``bool``: the weather move of ``weather`` fails on the weather bits ``w``: a bit of its
    group set (``whole_group``: Sandstorm, Rain Dance, Sunny Day, effect_script_0115.s, _0136.s,
    _0137.s :5), else (Hail, effect_script_0164.s:5) its TEMP bit set."""
    weather = jnp.clip(jnp.asarray(weather, _I32), 0, len(WEATHER_TEMP_BIT) - 1)
    temp = _bit(WEATHER_TEMP_BIT, weather)
    bits = jnp.where(jnp.asarray(whole_group), temp | _bit(WEATHER_PERM_BIT, weather), temp)
    return (weather != Weather.NONE) & ((jnp.asarray(w, _I32) & bits) != 0)


def move_bits(w, weather) -> jnp.ndarray:
    """``int32``: the weather bits ``w`` after a weather move of ``weather`` succeeds
    (effect_script_0115.s etc. :8-9: ``FIELD_CONDITION_WEATHER`` off, the TEMP bit on). A what-if
    view (``outcome``)."""
    return (jnp.asarray(w, _I32) & ~_WEATHER) | _bit(WEATHER_TEMP_BIT, weather)


def weather_seen(w) -> jnp.ndarray:
    """``int32``: the weather a player sees (an observation): the ``Weather`` id of the one set
    group (CheckFieldConditions prints one message per set group, :1113-1225; TEMP and PERM read
    alike), 0 with none, and with several groups ``8 +`` a bit per set group (bit ``id - 1``)."""
    w = jnp.asarray(w, _I32)
    sets = jnp.stack([(w & group) != 0 for group in _SEEN_GROUPS])
    ids = jnp.arange(1, len(_SEEN_GROUPS) + 1, dtype=_I32)
    one = jnp.max(jnp.where(sets, ids, 0))
    groups = jnp.sum(jnp.where(sets, jnp.left_shift(1, ids - 1), 0))
    return jnp.where(jnp.sum(sets) > 1, 8 + groups, one)


def weather_text(seen) -> str:
    """A ``weather_seen`` value as text (``search.plan_tree``, ``search.stage_b``)."""
    names = {weather: weather.name.lower() for weather in Weather}
    seen = int(seen)
    if seen < 8:
        return names[Weather(seen)]
    return " + ".join(names[w] for w in Weather if w and (seen - 8) >> (w - 1) & 1)


def gravity_count(state: BattleState) -> jnp.ndarray:
    """``int32``: the Gravity counter, the mask's bits 12-14 (``FIELD_CONDITION_GRAVITY``,
    condition.h:119-121, :142-143), which bcp.c:1229 counts down."""
    return (_mask(state) & _GRAVITY) >> FIELD_CONDITION_GRAVITY_SHIFT


def gravity_on(state: BattleState) -> jnp.ndarray:
    """``bool``: any ``FIELD_CONDITION_GRAVITY`` bit set, the test of every Gravity reader
    (bcp.c:1227, :2972; battle_lib.c:2545, :2701, :2766, :3095, :3215, :3398, :5506;
    effect_script_0215.s:5; subscript_hazards_check.s:6; trainer_ai script.s:1028, :6920)."""
    return (_mask(state) & _GRAVITY) != 0


def gravity_start(state: BattleState, cond) -> BattleState:
    """``FLAG_ON FIELD_CONDITION_GRAVITY_INIT`` (subscript_gravity_start.s:9; condition.h:143)."""
    return _put(state, _mask(state) | int(FieldCondition.GRAVITY_INIT), cond)


def gravity_tick(state: BattleState, live, rearm) -> BattleState:
    """``FIELD_COND_CHECK_STATE_GRAVITY`` (battle_controller_player.c:1227-1238) where ``live``:
    with any Gravity bit, ``-= 1 << 12`` (:1229); at 0 (:1231) Kaizo's subscript_gravity_end
    (sub_seq 238, data/kaizo/rom_diff.toml:615-619; ``rearm``: ``Tables.gravity_rearm``) ORs
    ``FIELD_CONDITION_GRAVITY_INIT`` back in (G-67); vanilla's only prints."""
    mask = _mask(state)
    on = jnp.asarray(live) & ((mask & _GRAVITY) != 0)
    mask = jnp.where(on, mask - (1 << FIELD_CONDITION_GRAVITY_SHIFT), mask)
    again = on & ((mask & _GRAVITY) == 0) & jnp.asarray(rearm)
    return _put(state, jnp.where(again, mask | int(FieldCondition.GRAVITY_INIT), mask), True)


def trick_room_toggle(state: BattleState, cond) -> BattleState:
    """Trick Room's script (effect_script_0259.s): with any Trick Room bit (:5) ``FLAG_OFF
    FIELD_CONDITION_TRICK_ROOM`` (:12), else ``FLAG_ON FIELD_CONDITION_TRICK_ROOM_INIT`` (:6)."""
    mask = _mask(state)
    up = (mask & _TRICK_ROOM) != 0
    toggled = jnp.where(up, mask & ~_TRICK_ROOM, mask | int(FieldCondition.TRICK_ROOM_INIT))
    return _put(state, toggled, cond)


def trick_room_tick(state: BattleState, live, rearm) -> BattleState:
    """``SIDE_COND_CHECK_STATE_TRICK_ROOM`` (battle_controller_player.c:1822-1827) where
    ``live``: with any Trick Room bit, ``-= 1 << 16`` (:1824); at 0 (:1825) Kaizo's
    subscript_trick_room_end (sub_seq 251, byte for byte effect_script_0259.s, rom_diff.toml:
    621-626; ``rearm``: ``Tables.trick_room_rearm``) finds no bit set and ORs
    ``FIELD_CONDITION_TRICK_ROOM_INIT`` (:6), E3-2-4; vanilla's only prints."""
    mask = _mask(state)
    on = jnp.asarray(live) & ((mask & _TRICK_ROOM) != 0)
    mask = jnp.where(on, mask - (1 << FIELD_CONDITION_TRICK_ROOM_SHIFT), mask)
    again = on & ((mask & _TRICK_ROOM) == 0) & jnp.asarray(rearm)
    return _put(state, jnp.where(again, mask | int(FieldCondition.TRICK_ROOM_INIT), mask), True)


def uproar_start(state: BattleState, b, cond) -> BattleState:
    """Battler ``b``'s Uproar bit on (subscript_uproar.s:9-11)."""
    return _put(state, _mask(state) | uproar_bit(b), cond)


def uproar_end(state: BattleState, b, cond) -> BattleState:
    """Battler ``b``'s Uproar bit off: its Uproar ends (battle_controller_player.c:1528, :1534),
    it switches out (``BattleSystem_UpdateAfterSwitch``, battle_lib.c:2135) or faints
    (``BattleSystem_CleanupFaintedMon``, :2213)."""
    return _put(state, _mask(state) & ~uproar_bit(b), cond)


def with_gravity(state: BattleState, count, cond=True) -> BattleState:
    """The Gravity counter bits set to ``count`` where ``cond``, clean (FCM 4.4), and the old
    ``gravity`` field with them until FCM step 3 drops it (a view that puts Gravity up: ``order.
    gravity_state``, ``outcome``'s foe-Gravity variant; a state set by hand: a test)."""
    count = jnp.asarray(count, _I32)
    bits = jnp.left_shift(count, FIELD_CONDITION_GRAVITY_SHIFT)
    state = _put(state, (_mask(state) & ~_GRAVITY) | bits, cond)
    value = jnp.where(jnp.asarray(cond), count, state.gravity)
    return state._replace(gravity=value.astype(state.gravity.dtype))


def with_weather_of(state: BattleState, source: BattleState) -> BattleState:
    """``state`` with ``source``'s weather bits (deep fog included) and ``weatherTurns`` (a
    view: ``playout._sort_view``)."""
    mask = (_mask(state) & ~_WEATHER) | (_mask(source) & _WEATHER)
    return _put(state, mask, True)._replace(weather_turns=source.weather_turns)


def with_weather(state: BattleState, weather, timer=0) -> BattleState:
    """``state`` holding the one ``Weather`` ``weather`` (an int), clean (FCM 4.4): a weather
    move's TEMP bit with ``timer`` turns left (``weather_turns``), or with ``timer`` 0 a
    permanent PERM bit (deep fog: bit 15; ``weather_turns`` kept). The old ``weather`` /
    ``weather_timer`` fields too, until FCM step 3 drops them. For a state set by hand: a test,
    a what-if view."""
    timer = jnp.asarray(timer, _I32)
    temp = timer != 0
    bits = jnp.where(temp, _bit(WEATHER_TEMP_BIT, weather), _bit(WEATHER_PERM_BIT, weather))
    state = _put(state, (_mask(state) & ~_WEATHER) | bits, True)
    return state._replace(
        weather=jnp.asarray(weather).astype(state.weather.dtype),
        weather_timer=timer.astype(state.weather_timer.dtype),
        weather_turns=jnp.where(temp, timer, state.weather_turns).astype(state.weather_turns.dtype),
    )


def without_weather(state: BattleState) -> BattleState:
    """``state`` with no weather bit (a view: the battle-start speed sort before the
    start condition is set, ``hooks.battle_start``)."""
    return _put(state, _mask(state) & ~_WEATHER, True)
