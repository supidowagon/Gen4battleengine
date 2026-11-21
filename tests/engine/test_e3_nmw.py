"""E3 nmw (rulings E3-B2-32, E3-B2-33, docs/kaizo/e3-ledger.md), in both engines: the Toxic Spikes
block flag `entry_hazards` returns (MOVE_STATUS_NO_MORE_WORK of subscript_poison.s:5-10 -> :127,
subscript_badly_poison.s:5-10 -> :140), and Thunder Wave on a 0x chart result (a Ground defender):
MOVE_STATUS_INEFFECTIVE (battle_lib.c:7551-7552) takes it out of AfterMoveEffects
(BattleControllerPlayer_CheckTypeChart, battle_controller_player.c:2357; :3325-3328)."""

import jax
import jax.numpy as jnp

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine import step as step_mod
from nuzline.engine.field import with_weather
from nuzline.engine.ids import AbilityId, Status, Type, Weather
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _enter, _entry_hazards, _send_out_exp, _Turn
from tests.engine.helpers import A, assert_matches_reference

SPE = 4

# Called eagerly the engine compiles each conditional as a program of its own (GBs of memory).
_send_out = jax.jit(step_mod._send_out, static_argnames=("effects",))


def _spec(n, species, level, moves, slot=0):
    return PokemonSpec(
        species=n.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(n.move(m) for m in moves) + (0,) * (4 - len(moves)),
        item=0,
        ability_slot=slot,
        gender=0,
        friendship=70,
    )


def _thunder_wave_fight(t, n, defender):
    """Side 0: ``defender`` (Conversion 2, Speed 1) with a stale Growl record from battler 1;
    side 1: a Pikachu with Thunder Wave. No abilities."""
    s = make_state(
        t,
        [_spec(n, defender, 50, ("Conversion 2",))],
        [_spec(n, "Pikachu", 50, ("Thunder Wave",))],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(
        stats=s.stats.at[0, 0, SPE].set(1),
        conv2_move=s.conv2_move.at[0].set(n.move("Growl")),
        conv2_battler=s.conv2_battler.at[0].set(2),
        conv2_type=s.conv2_type.at[0].set(Type.NORMAL),
    )
    out = empty_outcome()
    return s, out._replace(conv2=out.conv2.at[0, Type.ELECTRIC].set(Type.GRASS).at[0, 0].set(5))


def test_thunder_wave_into_a_ground_defender_has_no_effects(tables, names):
    """bcp.c:2357: Thunder Wave meets the chart; Ground is 0x to Electric (battle_lib.c:2419), so
    MOVE_STATUS_INEFFECTIVE is in NO_EFFECTS: UpdateFlagsWhenHit clears the defender's conversion2
    record (bcp.c:4531-4535) and its Conversion 2 fails (battle_script.c:4775-4776)."""
    s, out = _thunder_wave_fight(tables, names, "Geodude")
    before = [int(x) for x in s.types[0, 0]]
    got = assert_matches_reference(tables, s, A("Conversion 2", "Thunder Wave", s, names), out)
    assert int(got.status[0, 0]) == Status.NONE
    assert [int(x) for x in got.types[0, 0]] == before
    assert [int(x) for x in (got.conv2_move[0], got.conv2_battler[0], got.conv2_type[0])] == [
        0,
        0,
        0,
    ]


def test_thunder_wave_that_lands_records_for_conversion_2(tables, names):
    """Control: on a Monferno (Fire / Fighting) Thunder Wave lands and the Electric record gives
    Conversion 2's Grass pick."""
    s, out = _thunder_wave_fight(tables, names, "Monferno")
    got = assert_matches_reference(tables, s, A("Conversion 2", "Thunder Wave", s, names), out)
    assert int(got.status[0, 0]) == Status.PARALYSIS
    assert [int(x) for x in got.types[0, 0]] == [Type.GRASS, Type.GRASS]


def test_thunder_wave_into_a_ground_defender_takes_back_the_metronome_count(tables, names):
    """E3-B2-33: the Thunder Wave user holds Metronome (the item) and has used the move twice;
    BeforeMove's update (battle_lib.c:6160-6178) counts the third use, then the 0x chart result
    (MOVE_STATUS_INEFFECTIVE, in NO_EFFECTS) makes BattleSystem_VerifyMetronomeCount take it
    back (battle_lib.c:6181-6197, at UpdateMoveBuffers bcp.c:3957): the count ends at 2."""
    s, out = _thunder_wave_fight(tables, names, "Geodude")
    s = s._replace(
        item=s.item.at[1, 0].set(names.item("Metronome")),
        metronome_move=s.metronome_move.at[1].set(names.move("Thunder Wave")),
        metronome_turns=s.metronome_turns.at[1].set(2),
    )
    got = assert_matches_reference(tables, s, A("Conversion 2", "Thunder Wave", s, names), out)
    assert int(got.metronome_turns[1]) == 2


def _hazard_state(t, n, newcomer, slot, weather, layers, foe, foe_slot, burned, poison_typed):
    s = make_state(
        t,
        [_spec(n, "Rattata", 30, ("Tackle",)), _spec(n, newcomer, 30, ("Tackle",), slot=slot)],
        [_spec(n, foe, 30, ("Growl",), slot=foe_slot)],
    )
    s = with_weather(s, weather)._replace(toxic_spikes=s.toxic_spikes.at[0].set(layers))
    if burned:
        s = s._replace(status=s.status.at[0, 1].set(Status.BURN))
    if poison_typed:
        s = s._replace(types=s.types.at[0, 1, :].set(Type.POISON))
    return s


# (newcomer, ability slot, weather, layers, foe, foe slot, burned, poison typed,
#  want stopped, want status)
_HAZARD_CASES = (
    ("Tangela", 1, Weather.SUN, 1, "Pikachu", 0, False, False, True, Status.NONE),
    ("Snorlax", 0, Weather.NONE, 2, "Pikachu", 0, False, False, True, Status.NONE),
    ("Tangela", 1, Weather.SUN, 1, "Pikachu", 0, True, False, True, Status.BURN),
    ("Snorlax", 0, Weather.NONE, 1, "Pikachu", 0, False, True, False, Status.NONE),
    ("Tangela", 1, Weather.NONE, 1, "Pikachu", 0, False, False, False, Status.POISON),
    ("Tangela", 1, Weather.SUN, 1, "Psyduck", 1, False, False, False, Status.POISON),
)


def test_entry_hazards_airborne_immunity_newcomer_is_not_stopped(tables, names):
    """The `down &` term of the Toxic Spikes block: an Immunity Snorlax made Flying type is off
    the ground, so subscript_hazards_check.s skips the spikes and nothing sets NO_MORE_WORK
    (subscript_poison.s:5-10 is never reached); it stays unpoisoned and the layer stays."""
    t, n = tables, names
    s = _hazard_state(t, n, "Snorlax", 0, Weather.NONE, 1, "Pikachu", 0, False, False)
    s = s._replace(types=s.types.at[0, 1, :].set(Type.FLYING))
    out, stopped = _send_out(t, s, 0, 1, effects=False)
    assert not bool(stopped)
    assert int(out.status[0, 1]) == Status.NONE
    assert int(out.toxic_spikes[0]) == 1

    r = to_ref(s)
    turn = _Turn(to_numpy(t), None, entry_turn=r.turn, battler_count=2)
    _enter(turn, r, 0, 1)
    _send_out_exp(r, 0)
    assert _entry_hazards(turn, r, 0) is False

    sw = assert_matches_reference(t, s, jnp.array([5, 0], jnp.int32), empty_outcome())
    assert int(sw.active[0]) == 1
    assert int(sw.status[0, 1]) == Status.NONE


def test_entry_hazards_return_the_toxic_spikes_block(tables, names):
    """subscript_poison.s:5-10 -> :127 / subscript_badly_poison.s:5-10 -> :140: the plain
    Immunity or Leaf Guard in sun (no Cloud Nine) sets NO_MORE_WORK before any status test; a
    Poison type absorbs the spikes first (battle_script.c:7643-7647)."""
    t, n = tables, names
    for case in _HAZARD_CASES:
        *args, want_stopped, want_status = case
        s = _hazard_state(t, n, *args)
        out, stopped = _send_out(t, s, 0, 1, effects=False)
        assert bool(stopped) == want_stopped, case
        assert int(out.status[0, 1]) == want_status, case
        if args[-1]:  # a Poison type absorbs the layers
            assert int(out.toxic_spikes[0]) == 0, case

        r = to_ref(s)
        turn = _Turn(to_numpy(t), None, entry_turn=r.turn, battler_count=2)
        _enter(turn, r, 0, 1)
        _send_out_exp(r, 0)
        assert _entry_hazards(turn, r, 0) is want_stopped, case

        sw = assert_matches_reference(t, s, jnp.array([5, 0], jnp.int32), empty_outcome())
        assert int(sw.active[0]) == 1, case
        assert int(sw.status[0, 1]) == want_status, case
