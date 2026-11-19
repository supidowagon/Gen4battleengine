"""FCM step 1 (FCM.md 4.4): ``field_mask`` (fieldConditionsMask) and ``weather_turns``
(FieldConditions.weatherTurns) written beside the old field fields by every writer; the readers
still read the old fields. On each cited writer path the mask is clean (4.4: it equals the old
fields) and the engine equals the reference. The reader tests of steps 2-3 come with them."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.spec import PokemonSpec
from nuzline.engine.ids import (
    GRAVITY_TURNS,
    TRICK_ROOM_TURNS,
    WEATHER_TURNS,
    FieldCondition,
    Weather,
    clean_field_mask,
)
from nuzline.engine.outcome import MAX_HITS, empty_outcome
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


def _fight(t, n, mine, opp=("Growl",), foe="Geodude", bench=(), **start):
    """A level 20 Monferno (``mine``; ``bench`` species behind it) against a level 40 ``foe``
    (``opp``); ``start``: make_state's start condition, gravity, trick_room."""
    player = [_spec(n, "Monferno", 20, mine)] + [_spec(n, b, 20, ("Growl",)) for b in bench]
    return make_state(t, player, [_spec(n, foe, 40, opp)], **start)


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
