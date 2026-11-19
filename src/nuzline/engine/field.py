"""``fieldConditionsMask`` and ``FieldConditions.weatherTurns`` as ``BattleState.field_mask`` and
``BattleState.weather_turns`` (FCM build sheet, step 1 of section 4.4): the writers, pure ``jnp``.

Every writer of the old field fields (``weather``, ``weather_timer``, ``gravity``,
``trick_room``, the per-battler ``uproar``) also writes the mask through these, bit for bit as
the decomp writes ``fieldConditionsMask`` (bits ``ids.FieldCondition``, include/constants/
battle/condition.h:110-149); the readers still read the old fields. On every state the engine
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
)
from nuzline.engine.state import BattleState

_I32 = jnp.int32
_WEATHER = int(FieldCondition.WEATHER)
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


def weather_countdown(state: BattleState, live) -> BattleState:
    """The weather states of CheckFieldConditions (battle_controller_player.c:1113-1211, in that
    order) where ``live``: a set group with its PERM bit goes on; a TEMP-only one counts the one
    shared ``weatherTurns`` down (``--weatherTurns == 0``, :1121, :1146, :1171, :1196) and at 0
    its end script clears its TEMP bit (subscript_raining_end.s, _sandstorm_end.s, _sunny_end.s,
    _hailing_end.s :5). Deep fog never counts down (:1213-1225)."""
    mask, turns = _mask(state), state.weather_turns.astype(_I32)
    live = jnp.asarray(live)
    for temp, perm in WEATHER_COUNTDOWN:
        group = int(temp) | int(perm)
        counts = live & ((mask & group) != 0) & ((mask & int(perm)) == 0)
        turns = jnp.where(counts, turns - 1, turns)
        mask = jnp.where(counts & (turns == 0), mask & ~int(temp), mask)
    state = state._replace(weather_turns=turns.astype(state.weather_turns.dtype))
    return _put(state, mask, True)


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
    """The Gravity counter bits set to ``count`` (a view that puts Gravity up: ``order.
    gravity_state``, ``outcome``'s foe-Gravity variant)."""
    bits = jnp.left_shift(jnp.asarray(count, _I32), FIELD_CONDITION_GRAVITY_SHIFT)
    return _put(state, (_mask(state) & ~_GRAVITY) | bits, cond)


def with_weather_of(state: BattleState, source: BattleState) -> BattleState:
    """``state`` with ``source``'s weather bits (deep fog included) and ``weatherTurns`` (a
    view: ``playout._sort_view``)."""
    mask = (_mask(state) & ~_WEATHER) | (_mask(source) & _WEATHER)
    return _put(state, mask, True)._replace(weather_turns=source.weather_turns)


def without_weather(state: BattleState) -> BattleState:
    """``state`` with no weather bit (a view: the battle-start speed sort before the
    start condition is set, ``hooks.battle_start``)."""
    return _put(state, _mask(state) & ~_WEATHER, True)
