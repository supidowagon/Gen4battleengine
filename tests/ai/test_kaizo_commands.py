"""The script commands Kaizo Roark's labels need (K3c Task 4; docs/ai/script_commands.md).

One fixed-state test per command row against its handler. Decomp paths are relative to
``external/pokeplatinum``: ``trainer_ai.c`` is ``src/battle/trainer_ai/trainer_ai.c``,
``battle_lib.c`` is ``src/battle/battle_lib.c``, ``bcp.c`` is
``src/battle/battle_controller_player.c``.

The speed compare's gate (plan K3c Ruling 3): the exact command passes no tie draw, because
at equal speed ``BattleSystem_CompareBattlerSpeed`` returns FASTER or TIE, never SLOWER
(battle_lib.c:1401-1446), and every compare on Roark's paths tests SLOWER. The loose oracle
draws the tie bit as its own read (``ReadKind.SPEED_TIE``) and compares the decomp's
three-valued result. ``test_speed_tie_never_slower`` checks the gate in every branch that
reads the tie.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai.commands import (
    AiWeather,
    Battler,
    FieldCondition,
    LoadType,
    MonCondition,
    ReadKind,
    SideCondition,
)
from nuzline.engine.damage import ai_damage_for_roll, weather_suppressed
from nuzline.engine.ids import WEATHER_ABILITY_IDS, Stat, Status, Type, Weather
from nuzline.engine.order import (
    COMPARE_SPEED_FASTER,
    COMPARE_SPEED_SLOWER,
    COMPARE_SPEED_TIE,
    battler_slower,
    compare_speed,
    player_second,
)
from nuzline.engine.state import MOVES, make_state
from nuzline.engine.step import step
from tests.ai.fixtures import GEODUDE, outcome, versus
from tests.engine.helpers import A, roark_state

DECOMP = Path(__file__).resolve().parents[2] / "external/pokeplatinum"
_D, _A = Battler.DEFENDER, Battler.ATTACKER
_SPE = Stat.SPE - 1  # stats[..., 0..4] are Atk..Spe
_step = jax.jit(step)


def _ctx(tables, s, slot: int = 0, loose_bits=None):
    """The AI context of ``s`` at move ``slot`` (a fixed outcome: no command here reads a
    draw, except the loose speed compare, which reads ``loose_bits``)."""
    loose = None if loose_bits is None else c.loose_draws(loose_bits)
    return c.at_slot(c.new_context(tables, s, outcome(), loose=loose), slot)


def _jump(ctx) -> bool:
    return bool(ctx.jump)


def _temp(ctx) -> int:
    return int(ctx.calc_temp)


def _speeds(s, player: int, opp: int):
    return s._replace(
        stats=s.stats.at[0, int(s.active[0]), _SPE]
        .set(player)
        .at[1, int(s.active[1]), _SPE]
        .set(opp)
    )


def _species_with(tables, s, side: int, species: int, ability: int):
    """``s`` with ``side``'s active Pokemon made ``species`` with the species ability slot
    of the decomp ability ``ability`` (``Battler_Ability``: ``damage.decomp_abilities``).
    Its stats and types stay as they were."""
    a = int(s.active[side])
    slots = [int(x) for x in np.asarray(tables.species_abilities_decomp)[species]]
    return s._replace(
        species=s.species.at[side, a].set(species),
        form=s.form.at[side, a].set(0),
        ability_slot=s.ability_slot.at[side, a].set(slots.index(ability)),
    )


# --- AddToMoveScore (trainer_ai.c:576-586) -------------------------------------------------


@pytest.mark.parametrize(
    "start, value, expected",
    [(100, -10, 90), (5, -10, 0), (120, 7, 127), (127, 2, 0), (126, 4, 0), (100, 4, 104)],
)
def test_add_to_move_score_wraps_the_s8(tables, names, start, value, expected):
    # moveScore is s8 (include/battle/ai_context.h:14): `moveScore += val` wraps to
    # -128..127, then a negative score becomes 0 (trainer_ai.c:581-585). K3c final review.
    assert "s8 moveScore[4];" in (DECOMP / "include/battle/ai_context.h").read_text()
    ctx = _ctx(tables, versus(tables, names, GEODUDE))
    ctx = ctx._replace(scores=ctx.scores.at[0].set(start))
    assert int(c.add_to_move_score(ctx, value).scores[0]) == expected


def test_a_score_past_the_s8_is_refused(monkeypatch, kaizo_tables, kaizo_roark_party):
    # A score above 127 cannot come out of the wrapped AddToMoveScore; if one ever did, the
    # factored distribution is NaN instead of ranking it (policy._slot_move_distribution).
    from nuzline.ai import policy
    from tests.ai.test_kaizo_outcome import KAIZO  # Roark's fight (E1 batch 8)

    s = make_state(kaizo_tables, [kaizo_roark_party[2]], list(kaizo_roark_party))
    real = policy._slot_histograms
    dist = np.asarray(policy._slot_move_distribution(kaizo_tables, s, KAIZO))
    assert not np.isnan(dist).any()

    def high(tables, state, layout):
        scores, hist = real(tables, state, layout)
        return scores.at[..., 0].set(128), hist

    monkeypatch.setattr(policy, "_slot_histograms", high)
    dist = np.asarray(policy._slot_move_distribution(kaizo_tables, s, KAIZO))
    assert np.isnan(dist).all()


# --- HP percent (trainer_ai.c:618, :633) ---------------------------------------------------


def test_if_hp_percent_equal_and_not_equal(tables, names):
    # u32 hpPercent = curHP * 100 / maxHP (trainer_ai.c:626, :641): integer division.
    s = versus(tables, names, GEODUDE)  # Geodude 32/32
    assert int(s.max_hp[1, GEODUDE]) == 32
    ctx = _ctx(tables, s)
    assert _jump(c.if_hp_percent_equal_to(ctx, _A, 100))
    assert not _jump(c.if_hp_percent_not_equal_to(ctx, _A, 100))
    s = versus(tables, names, GEODUDE, opp_hp=31)  # 3100 / 32 = 96 (96.875 rounded down)
    ctx = _ctx(tables, s)
    assert _jump(c.if_hp_percent_equal_to(ctx, _A, 96))
    assert not _jump(c.if_hp_percent_equal_to(ctx, _A, 97))
    assert _jump(c.if_hp_percent_not_equal_to(ctx, _A, 100))
    assert not _jump(c.if_hp_percent_not_equal_to(ctx, _A, 96))
    # The defender's: Monferno 44/44.
    assert _jump(c.if_hp_percent_equal_to(ctx, _D, 100))


# --- IfStatus (trainer_ai.c:648) -----------------------------------------------------------


def test_mon_condition_masks_are_the_decomps():
    # include/constants/battle/condition.h:7-14, :22, :25.
    assert MonCondition.SLEEP == 0b111
    assert (MonCondition.POISON, MonCondition.BURN, MonCondition.FREEZE) == (8, 16, 32)
    assert (MonCondition.PARALYSIS, MonCondition.TOXIC) == (64, 128)
    assert MonCondition.ANY == 0xFF


@pytest.mark.parametrize("status", list(Status))
def test_if_status_any(tables, names, status):
    # Basic_CheckCannotPoison (script.s:545): IfStatus DEFENDER, MON_CONDITION_ANY. The
    # defender's true status (battleMons[battler].status, trainer_ai.c:657).
    s = versus(tables, names, GEODUDE)
    s = s._replace(status=s.status.at[0, 0].set(status))
    ctx = _ctx(tables, s)
    assert _jump(c.if_status(ctx, _D, MonCondition.ANY)) == (status != Status.NONE)
    assert not _jump(c.if_status(ctx, _A, MonCondition.ANY))  # Geodude is healthy


def test_if_status_narrow_mask(tables, names):
    s = versus(tables, names, GEODUDE)
    para = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    burn = s._replace(status=s.status.at[0, 0].set(Status.BURN))
    toxic = s._replace(status=s.status.at[0, 0].set(Status.TOXIC))
    assert _jump(c.if_status(_ctx(tables, para), _D, MonCondition.PARALYSIS))
    assert not _jump(c.if_status(_ctx(tables, burn), _D, MonCondition.PARALYSIS))
    # MON_CONDITION_POISON and MON_CONDITION_TOXIC are different bits (condition.h:10, :14).
    assert not _jump(c.if_status(_ctx(tables, toxic), _D, MonCondition.POISON))
    assert _jump(c.if_status(_ctx(tables, toxic), _D, MonCondition.TOXIC))


def test_if_status_refuses_a_partial_group():
    # A mask that splits the sleep counter is not a whole engine status.
    with pytest.raises(AssertionError):
        c._condition_statuses(1)


# --- IfStatStageLessThan (trainer_ai.c:1465) -----------------------------------------------


def test_if_stat_stage_less_than(tables, names):
    s = versus(tables, names, GEODUDE)
    s = s._replace(stages=s.stages.at[1, Stat.SPE].set(-1))  # decomp 5
    ctx = _ctx(tables, s)
    assert _jump(c.if_stat_stage_less_than(ctx, _A, Stat.SPE, 6))
    assert not _jump(c.if_stat_stage_less_than(ctx, _A, Stat.SPE, 5))
    assert not _jump(c.if_stat_stage_less_than(ctx, _D, Stat.SPE, 6))  # player at decomp 6
    assert _jump(c.if_stat_stage_less_than(ctx, _D, Stat.SPE, 7))


# --- IfSideCondition (trainer_ai.c:732) ----------------------------------------------------


def test_side_condition_masks_are_the_decomps():
    # include/constants/battle/condition.h:88-95.
    assert (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN) == (1, 2)
    assert (SideCondition.SAFEGUARD, SideCondition.STEALTH_ROCK) == (8, 128)


@pytest.mark.parametrize(
    "field, mask",
    [
        ("reflect", SideCondition.REFLECT),
        ("light_screen", SideCondition.LIGHT_SCREEN),
        ("safeguard", SideCondition.SAFEGUARD),
        ("stealth_rock", SideCondition.STEALTH_ROCK),
    ],
)
def test_if_side_condition(tables, names, field, mask):
    # Basic_CheckAlreadyUnderLightScreen (script.s:550) reads the ATTACKER's side;
    # Basic_CheckCannotPoison (:546) the DEFENDER's Safeguard.
    s = versus(tables, names, GEODUDE)
    for side, battler in ((0, _D), (1, _A)):
        on = s._replace(**{field: getattr(s, field).at[side].set(3)})
        assert _jump(c.if_side_condition(_ctx(tables, on), battler, mask))
        other = Battler(1 - int(battler))
        assert not _jump(c.if_side_condition(_ctx(tables, on), other, mask))
    ctx = _ctx(tables, s)
    assert not any(_jump(c.if_side_condition(ctx, b, mask)) for b in (_D, _A))
    # Another condition does not match the mask.
    others = [m for m in SideCondition if m != mask]
    on = s._replace(**{field: getattr(s, field).at[0].set(3)})
    assert not any(_jump(c.if_side_condition(_ctx(tables, on), _D, m)) for m in others)


# --- LoadTypeFrom (trainer_ai.c:937) -------------------------------------------------------


def test_load_type_targets_are_the_decomps():
    # generated/ai_load_type_targets.txt, line n + 1 is value n.
    lines = (DECOMP / "generated/ai_load_type_targets.txt").read_text().split()
    for target in LoadType:
        assert lines[int(target)] == f"LOAD_{target.name}"


def test_load_type_from_attacker_types(tables, names):
    # Basic_CheckCurse (script.s:681-684): the attacker's true types.
    s = versus(tables, names, GEODUDE)  # Geodude: Rock / Ground
    ctx = _ctx(tables, s)
    assert _temp(c.load_type_from(ctx, LoadType.ATTACKER_TYPE_1)) == Type.ROCK
    assert _temp(c.load_type_from(ctx, LoadType.ATTACKER_TYPE_2)) == Type.GROUND
    ghost = s._replace(types=s.types.at[1, GEODUDE, 1].set(Type.GHOST))
    assert _temp(c.load_type_from(_ctx(tables, ghost), LoadType.ATTACKER_TYPE_2)) == Type.GHOST
    # The defender's and the move's are unchanged: Monferno Fire / Fighting, Tackle Normal.
    assert _temp(c.load_type_from(ctx, LoadType.DEFENDER_TYPE_1)) == Type.FIRE
    assert _temp(c.load_type_from(ctx, LoadType.DEFENDER_TYPE_2)) == Type.FIGHTING
    assert _temp(c.load_type_from(ctx, LoadType.MOVE_TYPE)) == int(
        tables.move_type[int(s.moves[1, GEODUDE, 0])]
    )


# --- IfFieldConditionsMask (trainer_ai.c:1936-1946) ----------------------------------------


def test_if_field_conditions_mask_trick_room(tables, names):
    # Basic_CheckHighStatStage_Speed (script.s:364).
    # FIELD_CONDITION_TRICK_ROOM = bits 16-18 (include/constants/battle/condition.h:123-125, :144).
    assert FieldCondition.TRICK_ROOM == (1 << 16) | (1 << 17) | (1 << 18)
    s = versus(tables, names, GEODUDE)
    assert not _jump(c.if_field_conditions_mask(_ctx(tables, s), FieldCondition.TRICK_ROOM))
    tr = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    assert _jump(c.if_field_conditions_mask(_ctx(tables, tr), FieldCondition.TRICK_ROOM))


# --- LoadCurrentWeather (trainer_ai.c:1414-1439) -------------------------------------------


def test_ai_weather_ids_are_the_decomps():
    # generated/ai_weather_types.txt, line n + 1 is value n.
    lines = (DECOMP / "generated/ai_weather_types.txt").read_text().split()
    for weather in AiWeather:
        assert lines[int(weather)] == f"AI_WEATHER_{weather.name}"


@pytest.mark.parametrize(
    "weather, expected",
    [
        (Weather.NONE, AiWeather.CLEAR),
        (Weather.SUN, AiWeather.SUNNY),
        (Weather.RAIN, AiWeather.RAINING),
        (Weather.SANDSTORM, AiWeather.SANDSTORM),
        (Weather.HAIL, AiWeather.HAILING),
        (Weather.FOG, AiWeather.DEEP_FOG),
    ],
)
def test_load_current_weather(tables, names, weather, expected):
    s = versus(tables, names, GEODUDE)
    s = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
    assert _temp(c.load_current_weather(_ctx(tables, s))) == expected


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("lock", ["Cloud Nine", "Air Lock"])
def test_load_current_weather_ignores_cloud_nine(tables, names, side, lock):
    # LoadCurrentWeather reads the raw fieldConditionsMask through WEATHER_IS_* with no
    # NO_CLOUD_NINE (trainer_ai.c:1414-1439; include/battle/common.h:68-72): a permanent
    # sandstorm (FIELD_CONDITION_SANDSTORM_PERM, condition.h:113, :128) with a Cloud Nine
    # or Air Lock holder on the field still loads AI_WEATHER_SANDSTORM.
    holder = {"Cloud Nine": "Psyduck", "Air Lock": "Rayquaza"}[lock]
    s = versus(tables, names, GEODUDE)
    s = s._replace(
        weather=jnp.asarray(Weather.SANDSTORM, s.weather.dtype),
        weather_timer=jnp.asarray(0, s.weather_timer.dtype),  # permanent
    )
    s = _species_with(tables, s, side, names.species(holder), WEATHER_ABILITY_IDS[lock])
    assert bool(weather_suppressed(tables, s))  # the battle's weather effects are off
    assert _temp(c.load_current_weather(_ctx(tables, s))) == AiWeather.SANDSTORM


# --- The AI's damage in sand (trainer_ai.c:3081-3091; battle_lib.c:6921-6929) --------------


def test_ai_damage_sand_spdef_gated(tables, names):
    # TrainerAI_CalcDamage passes fieldConditionsMask to CalcMoveDamage (trainer_ai.c:3085);
    # the Rock-type defender's SpD x15/10 in sand is inside NO_CLOUD_NINE
    # (battle_lib.c:6921, :6926-6929), which counts Cloud Nine and Air Lock on every living
    # battler (include/battle/common.h:63-64; COUNT_ALIVE_BATTLERS, battle_lib.c:2952-2959).
    s = versus(tables, names, GEODUDE, "Geodude")  # a Rock-type player
    assert Type.ROCK in [int(x) for x in s.types[0, 0]]
    water_gun = names.move("Water Gun")
    assert int(tables.move_category[water_gun]) == 1  # special
    s = s._replace(moves=s.moves.at[1, GEODUDE, 0].set(water_gun))
    psyduck = names.species("Psyduck")
    abilities = np.asarray(tables.species_abilities_decomp)[psyduck]
    damp = (
        int(abilities[0])
        if int(abilities[0]) != WEATHER_ABILITY_IDS["Cloud Nine"]
        else int(abilities[1])
    )

    def damage(st):
        return int(ai_damage_for_roll(tables, st, 1, water_gun, 15))

    def weather(st, w):
        return st._replace(
            weather=jnp.asarray(w, st.weather.dtype),
            weather_timer=jnp.asarray(0, st.weather_timer.dtype),
        )

    no_lock = _species_with(tables, s, 1, psyduck, damp)
    clear, sand = (
        damage(weather(no_lock, Weather.NONE)),
        damage(weather(no_lock, Weather.SANDSTORM)),
    )
    # The boost exactly: the player's SpD x15/10 and no weather gives the sand damage.
    spd = int(no_lock.stats[0, 0, Stat.SPD - 1])
    boosted = no_lock._replace(stats=no_lock.stats.at[0, 0, Stat.SPD - 1].set(spd * 15 // 10))
    assert sand == damage(weather(boosted, Weather.NONE)) < clear
    # A Cloud Nine (the AI's side) or Air Lock (the player's side) holder turns it off.
    nine = _species_with(tables, s, 1, psyduck, WEATHER_ABILITY_IDS["Cloud Nine"])
    assert damage(weather(nine, Weather.SANDSTORM)) == damage(weather(nine, Weather.NONE))
    lock = _species_with(
        tables, no_lock, 0, names.species("Rayquaza"), WEATHER_ABILITY_IDS["Air Lock"]
    )
    assert damage(weather(lock, Weather.SANDSTORM)) == damage(weather(lock, Weather.NONE))
    # A fainted holder does not count (curHP, battle_lib.c:2956): only the active can be on
    # the field, so fainting the AI's own active is the case to show.
    gone = weather(nine, Weather.SANDSTORM).replace_hp(1, GEODUDE, 0)
    assert not bool(weather_suppressed(tables, gone))


# --- IfSpeedCompareEqualTo (trainer_ai.c:1102-1112) ----------------------------------------


def test_compare_speed_constants_are_the_decomps():
    # include/constants/battle.h:163-165.
    assert (COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER, COMPARE_SPEED_TIE) == (0, 1, 2)


def _speed_state(tables, names, player: int, opp: int, **fields):
    s = versus(tables, names, GEODUDE)
    s = _speeds(s, player, opp)
    return s._replace(**{k: jnp.asarray(v, getattr(s, k).dtype) for k, v in fields.items()})


def _slower(tables, s) -> bool:
    return _jump(c.if_speed_compare_equal_to(_ctx(tables, s), COMPARE_SPEED_SLOWER))


def test_speed_compare_plain_and_trick_room(tables, names):
    # CompareBattlerSpeed(attacker = the AI, defender = the player, TRUE) == SLOWER
    # (trainer_ai.c:1110): the AI's Pokemon is slower (battle_lib.c:1440-1447), or faster
    # under Trick Room (:1432-1439).
    assert _slower(tables, _speed_state(tables, names, 50, 40))
    assert not _slower(tables, _speed_state(tables, names, 40, 50))
    assert not _slower(tables, _speed_state(tables, names, 50, 40, trick_room=3))
    assert _slower(tables, _speed_state(tables, names, 40, 50, trick_room=3))


def test_speed_compare_ignores_priority(tables, names):
    # ignoreQuickClaw TRUE skips the priority read (battle_lib.c:1375-1399): both
    # priorities stay 0, whatever the battlers' moves. Roark's Geodude knows Tackle; a
    # +1 move in the player's slot changes nothing.
    s = _speed_state(tables, names, 40, 50)
    s = s._replace(moves=s.moves.at[0, 0, 0].set(names.move("Quick Attack")))
    assert not _slower(tables, s)


def test_speed_compare_reads_player_quick_claw(tables, names):
    # Review Focus 2: the speedRand read happens with ignoreQuickClaw TRUE too; only the
    # moveEffectsData write is skipped (battle_lib.c:1278-1286 for battler1, :1344-1352 for
    # battler2). A lone activated Quick Claw goes first (:1408-1411). The AI decides before
    # the turn's speedRand redraw (bcp.c:853-857), so qc_roll is the previous turn's draw.
    s = _speed_state(tables, names, 40, 50)  # Roark faster
    s = s._replace(item=s.item.at[0, 0].set(names.item("Quick Claw")))
    assert _slower(tables, s._replace(qc_roll=jnp.asarray([1, 0], s.qc_roll.dtype)))
    assert not _slower(tables, s._replace(qc_roll=jnp.asarray([0, 0], s.qc_roll.dtype)))
    # The AI's own activated claw beats a faster player (battler1's read, :1278-1286).
    t = _speed_state(tables, names, 50, 40)
    t = t._replace(item=t.item.at[1, GEODUDE].set(names.item("Quick Claw")))
    assert not _slower(tables, t._replace(qc_roll=jnp.asarray([0, 1], t.qc_roll.dtype)))
    assert _slower(tables, t._replace(qc_roll=jnp.asarray([0, 0], t.qc_roll.dtype)))


def test_speed_compare_lagging_tail(tables, names):
    # A lone Lagging Tail holder is SLOWER (battle_lib.c:1418-1421), whatever the speeds.
    s = _speed_state(tables, names, 40, 50)
    s = s._replace(item=s.item.at[1, GEODUDE].set(names.item("Lagging Tail")))
    assert _slower(tables, s)


def _tie_cases(tables, names):
    """Equal speeds in every branch of battle_lib.c:1401-1446 that reads the tie:
    both Quick Claws activated (:1402-1407), both Lagging Tails (:1412-1417), Trick Room
    (:1432-1439) and plain speed (:1440-1447). Stall is unsupported (order.py docstring)."""
    claw, tail = names.item("Quick Claw"), names.item("Lagging Tail")
    base = _speed_state(tables, names, 45, 45)
    both = lambda item: base._replace(item=base.item.at[0, 0].set(item).at[1, GEODUDE].set(item))
    return {
        "plain": base,
        "trick_room": base._replace(trick_room=jnp.asarray(3, base.trick_room.dtype)),
        "quick_claws": both(claw)._replace(qc_roll=jnp.asarray([1, 1], base.qc_roll.dtype)),
        "lagging_tails": both(tail),
    }


@pytest.mark.parametrize("case", ["plain", "trick_room", "quick_claws", "lagging_tails"])
def test_speed_tie_never_slower(tables, names, case):
    # Ruling 3: at equal speed every branch gives FASTER (bit 0) or TIE (bit 1), never
    # SLOWER (battle_lib.c:1401-1447; the ties at :1405-1406, :1415-1416, :1437-1438
    # and :1445-1446 set TIE, and no equal-speed path sets SLOWER). So the AI's SLOWER test does not depend
    # on the tie's RandNext() & 1, and the exact command draws none.
    s = _tie_cases(tables, names)[case]
    for side in (0, 1):
        results = {int(compare_speed(tables, s, side, bit, s.qc_roll)) for bit in (0, 1)}
        assert results == {COMPARE_SPEED_FASTER, COMPARE_SPEED_TIE}, (side, results)
    assert not _slower(tables, s)
    # The loose oracle draws the tie as its own read and agrees for both bits.
    for bit in (0, 1):
        bits = np.zeros((MOVES, c.LOOSE_READS), np.int8)
        bits[0, 0] = bit
        ctx = c.if_speed_compare_equal_to(_ctx(tables, s, loose_bits=bits), COMPARE_SPEED_SLOWER)
        assert not _jump(ctx)
        assert int(ctx.loose.count[0]) == 1
        assert int(ctx.loose.kind[0, 0]) == ReadKind.SPEED_TIE
        assert bool(ctx.loose.live[0, 0])  # the decomp reads RandNext here
        assert float(ctx.loose.p[0, 0]) == 0.5


def test_loose_speed_compare_reads_no_tie_at_unequal_speed(tables, names):
    # Unequal speeds read no RandNext (battle_lib.c:1440-1447): the loose read is off the path.
    s = _speed_state(tables, names, 50, 40)
    bits = np.zeros((MOVES, c.LOOSE_READS), np.int8)
    ctx = c.if_speed_compare_equal_to(_ctx(tables, s, loose_bits=bits), COMPARE_SPEED_SLOWER)
    assert _jump(ctx)
    assert int(ctx.loose.count[0]) == 1 and not bool(ctx.loose.live[0, 0])


def test_flat_speed_compare_refuses_faster(tables, names):
    # The flat (vanilla) layout draws no tie, which is exact only for SLOWER; FASTER's tie is a
    # read of the per-slot layout only (E1 Ruling R8).
    ctx = _ctx(tables, _speed_state(tables, names, 45, 45))
    with pytest.raises(AssertionError):
        c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_FASTER)


def test_battler_slower_is_compare_speed_not_faster(tables, names):
    # player_second is battler_slower's battler1 = 0 case; battler_slower is
    # "CompareBattlerSpeed(battler1, other) != FASTER".
    claw, tail = names.item("Quick Claw"), names.item("Lagging Tail")
    for player_speed, opp_speed in ((40, 50), (50, 40), (45, 45)):
        for items in ((0, 0), (claw, 0), (0, claw), (claw, claw), (tail, 0), (tail, tail)):
            for qc in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for tr in (0, 3):
                    s = _speed_state(tables, names, player_speed, opp_speed, trick_room=tr)
                    s = s._replace(item=s.item.at[0, 0].set(items[0]).at[1, GEODUDE].set(items[1]))
                    q = jnp.asarray(qc)
                    for tie in (0, 1):
                        r0 = int(compare_speed(tables, s, 0, tie, q))
                        r1 = int(compare_speed(tables, s, 1, tie, q))
                        assert bool(player_second(tables, s, tie, q)) == (
                            r0 != COMPARE_SPEED_FASTER
                        )
                        assert bool(battler_slower(tables, s, 0, tie, q)) == (r0 != 0)
                        assert bool(battler_slower(tables, s, 1, tie, q)) == (r1 != 0)
                        if player_speed != opp_speed:  # no tie: one SLOWER, the other FASTER
                            assert {r0, r1} == {COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER}


@pytest.mark.parametrize(
    ("holder", "item", "expected"),
    [
        # Exactly one activated Quick Claw (speedRand % (100 / 20) == 0, qc_roll 1):
        # battler1 (the AI) without, battler2 (the player) with: SLOWER (:1408-1409).
        ("player", "Quick Claw", COMPARE_SPEED_SLOWER),
        # battler1 with, battler2 without: FASTER (:1410-1411).
        ("ai", "Quick Claw", COMPARE_SPEED_FASTER),
        # No Quick Claw; battler1 alone holds Lagging Tail: SLOWER (:1418-1419).
        ("ai", "Lagging Tail", COMPARE_SPEED_SLOWER),
        # battler2 alone holds it: FASTER (:1420-1421).
        ("player", "Lagging Tail", COMPARE_SPEED_FASTER),
    ],
)
def test_ai_speed_compare_one_claw_or_tail_at_equal_speed(tables, names, holder, item, expected):
    """Task 4 review m1, written from battle_lib.c ``BattleSystem_CompareBattlerSpeed``
    rather than from ``compare_speed``: the AI's compare is battler1 = the AI, battler2 = the
    player (trainer_ai.c:1110). At equal raw speed (45/45) these branches return before any
    speed compare, so neither tie bit matters and no ``RandNext`` is read (the loose read is
    off the path). ``result`` starts as FASTER (:1190)."""
    s = _speed_state(tables, names, 45, 45)
    side, slot = (0, 0) if holder == "player" else (1, GEODUDE)
    s = s._replace(item=s.item.at[side, slot].set(names.item(item)))
    qc = [0, 0]
    if item == "Quick Claw":
        qc[side] = 1
    s = s._replace(qc_roll=jnp.asarray(qc, s.qc_roll.dtype))
    for tie in (0, 1):
        assert int(compare_speed(tables, s, 1, tie, s.qc_roll)) == expected, tie
    assert _slower(tables, s) == (expected == COMPARE_SPEED_SLOWER)
    bits = np.ones((MOVES, c.LOOSE_READS), np.int8)
    ctx = c.if_speed_compare_equal_to(_ctx(tables, s, loose_bits=bits), COMPARE_SPEED_SLOWER)
    assert _jump(ctx) == (expected == COMPARE_SPEED_SLOWER)
    assert not bool(ctx.loose.live[0, 0])


# --- IfMoveEffectKnown / IfMoveEffectNotKnown (trainer_ai.c:1739, :1781) -------------------


def _kaizo_bonsly(kaizo_tables, kaizo_roark_party, moves):
    import dataclasses

    opp = [dataclasses.replace(kaizo_roark_party[0], moves=tuple(moves), pp=None)]
    return make_state(kaizo_tables, [kaizo_roark_party[2]], opp)


def test_if_move_effect_known_attacker_true_moveset(kaizo_tables, kaizo_names, kaizo_roark_party):
    # ATTACKER: battleMons[attacker].moves, nonzero, MOVE_DATA(move).effect (trainer_ai.c:1752-1753),
    # the per-game effect (tables.move_ai_effect). Kaizo Sand Tomb is WHIRLPOOL (261), vanilla
    # BIND_HIT (42) (plan K3c Ruling 5).
    t, n = kaizo_tables, kaizo_names
    sand_tomb = n.move("Sand Tomb")
    assert int(t.move_ai_effect[sand_tomb]) == 261
    s = _kaizo_bonsly(t, kaizo_roark_party, (n.move("Selfdestruct"), sand_tomb, 0, 0))
    s = s._replace(pp=s.pp.at[1, 0, 1].set(0))  # no PP: still in the true moveset
    ctx = _ctx(t, s)
    assert _jump(c.if_move_effect_known(ctx, _A, 261))
    assert not _jump(c.if_move_effect_known(ctx, _A, 42))
    assert _jump(c.if_move_effect_known(ctx, _A, 7))  # Selfdestruct, HALVE_DEFENSE
    assert _jump(c.if_move_effect_not_known(ctx, _A, 42))
    assert not _jump(c.if_move_effect_not_known(ctx, _A, 261))
    # An empty slot is skipped (moves[i] != 0, :1752): MOVE_NONE's effect 0 is not known.
    assert int(t.move_ai_effect[0]) == 0
    assert not _jump(c.if_move_effect_known(ctx, _A, 0))
    assert _jump(c.if_move_effect_not_known(ctx, _A, 0))


def test_if_move_effect_known_defender_seen_moves(tables, names):
    # DEFENDER: AI_CONTEXT.battlerMoves, the moves the AI has seen (trainer_ai.c:1765-1766).
    s = versus(tables, names, GEODUDE)  # Monferno; nothing seen
    moves = [int(m) for m in s.moves[0, 0]]
    slot = next(i for i, m in enumerate(moves) if m and int(tables.move_ai_effect[m]) != 0)
    effect = int(tables.move_ai_effect[moves[slot]])
    assert not _jump(c.if_move_effect_known(_ctx(tables, s), _D, effect))
    assert _jump(c.if_move_effect_not_known(_ctx(tables, s), _D, effect))
    seen = s._replace(ai_seen_moves=s.ai_seen_moves.at[0, slot].set(1))
    assert _jump(c.if_move_effect_known(_ctx(tables, seen), _D, effect))
    assert not _jump(c.if_move_effect_not_known(_ctx(tables, seen), _D, effect))
    # The pending last move counts as seen (TrainerAI_RecordLastMove first, :295).
    pending = s._replace(last_move=s.last_move.at[0].set(moves[slot]))
    assert _jump(c.if_move_effect_known(_ctx(tables, pending), _D, effect))


# --- IfAttackerHasNoDamagingMoves (trainer_ai.c:913-929) -----------------------------------


def test_if_attacker_has_no_damaging_moves(tables, names):
    # moves[i] != MOVE_NONE && MOVE_DATA(moves[i]).power: power != 0, not > 1 (:920-921).
    s = versus(tables, names, GEODUDE)
    status = [names.move(m) for m in ("Stealth Rock", "Screech", "Leer")]

    def with_moves(moves):
        row = jnp.asarray(list(moves) + [0] * (4 - len(moves)), s.moves.dtype)
        return s._replace(moves=s.moves.at[1, GEODUDE].set(row))

    assert _jump(c.if_attacker_has_no_damaging_moves(_ctx(tables, with_moves(status))))
    counter = names.move("Counter")
    assert int(tables.move_power[counter]) == 1  # damaging here, not for FlagMoveDamageScore
    ctx = _ctx(tables, with_moves([*status, counter]))
    assert not _jump(c.if_attacker_has_no_damaging_moves(ctx))
    # The true moveset: a damaging move with no PP still counts.
    empty = with_moves([*status, names.move("Tackle")])
    empty = empty._replace(pp=empty.pp.at[1, GEODUDE, 3].set(0))
    assert not _jump(c.if_attacker_has_no_damaging_moves(_ctx(tables, empty)))


# --- LoadDefenderLastUsedMoveClass (trainer_ai.c:2050-2054) --------------------------------


def test_load_defender_last_used_move_class(tables, names):
    # MOVE_DATA(movePrevByBattler[defender]).class: 0 physical, 1 special, 2 status
    # (generated/move_classes.txt).
    s = versus(tables, names, GEODUDE)
    for move, cls in (("Scratch", 0), ("Ember", 1), ("Leer", 2)):
        last = s._replace(last_move=s.last_move.at[0].set(names.move(move)))
        assert _temp(c.load_defender_last_used_move_class(_ctx(tables, last))) == cls
    # MOVE_NONE is CLASS_PHYSICAL (res/moves/none/data.json:9).
    assert _temp(c.load_defender_last_used_move_class(_ctx(tables, s))) == 0


def test_last_used_move_class_nature_power(tables, names):
    # movePrevByBattler = moveTemp on success, else MOVE_NONE (bcp.c:3944-3949). moveTemp is set
    # at command selection (bcp.c:1861-1898); Nature Power's script (effect_script_0173.s) writes
    # msgMoveTemp (GetTerrainMove, battle_script.c:8045) and moveCur (GoToMoveScript, :2118),
    # never moveTemp; no battle script writes BTLVAR_MOVE_TEMP (battle_script.c:9585-9586: the
    # one reader is effect_script_0092.s:6's compare, and subscript_pursuit.s:81, :117's
    # OPCODE_GET copies moveTemp into moveCur, battle_script.c:3278-3279). So the class is
    # Nature Power's (status), not Tri Attack's (special, the BUILDING call).
    from tests.engine.test_terrain_moves import BUILDING, NP, TRI
    from tests.engine.test_terrain_moves import _o as terrain_outcome

    s = roark_state(tables, names, player_moves=[NP, "Tackle"], opp_species="Cranidos")
    s = s._replace(terrain=jnp.asarray(BUILDING, s.terrain.dtype))
    assert int(tables.move_category[names.move(NP)]) == 2
    assert int(tables.move_category[names.move(TRI)]) == 1
    out = _step(tables, s, A(NP, "Tackle", s, names), terrain_outcome())
    assert int(out.last_move[0]) == names.move(NP)
    assert _temp(c.load_defender_last_used_move_class(_ctx(tables, out))) == 2
    # A failed move (asleep: the status checks stop it) leaves MOVE_NONE: class 0.
    asleep = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
        last_move=s.last_move.at[0].set(names.move(NP)),
    )
    out = _step(tables, asleep, A(NP, "Tackle", asleep, names), terrain_outcome())
    assert int(out.last_move[0]) == 0
    assert _temp(c.load_defender_last_used_move_class(_ctx(tables, out))) == 0


# --- LoadCurrentMoveEffect, IfLoadedNotInTable (trainer_ai.c:1164, :876-891) ---------------


def test_load_current_move_effect(kaizo_tables, kaizo_names, kaizo_roark_party):
    t, n = kaizo_tables, kaizo_names
    s = _kaizo_bonsly(t, kaizo_roark_party, (n.move("Selfdestruct"), n.move("Sand Tomb"), 0, 0))
    assert _temp(c.load_current_move_effect(_ctx(t, s, 0))) == 7
    assert _temp(c.load_current_move_effect(_ctx(t, s, 1))) == 261


def test_if_loaded_not_in_table(tables, names):
    ctx = _ctx(tables, versus(tables, names, GEODUDE))
    loaded = ctx._replace(calc_temp=jnp.asarray(7, jnp.int32))
    assert not _jump(c.if_loaded_not_in_table(loaded, (1, 7, 140)))
    assert _jump(c.if_loaded_not_in_table(loaded, (1, 8, 140)))
    assert _jump(c.if_loaded_not_in_table(loaded, ()))


# --- The command table ---------------------------------------------------------------------


def test_script_commands_table_matches_the_module():
    # Every row of docs/ai/script_commands.md names a function of nuzline.ai.commands, and the
    # count check at its end holds.
    text = Path("docs/ai/script_commands.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("|")]
    body = rows[2:]
    for row in body:
        name = row.strip().strip("|").split("|")[-1].strip().strip("`")
        assert callable(getattr(c, name, None)), name
    count = re.search(r"gives (\d+): (\d+) command rows", text)
    assert count is not None
    assert int(count.group(1)) == len(rows) and int(count.group(2)) == len(body)


# --- IfHeldItemEqualTo (trainer_ai.c:1915-1934) --------------------------------------------


def test_if_held_item_equal_to(tables, names):
    # The AI's own battler: its true held item (:1925-1926), 0 once consumed; the player's:
    # AI_CONTEXT.battlerHeldItems, only what a message named (:1927-1928).
    s = versus(tables, names, GEODUDE)
    herb = c.ITEM_POWER_HERB
    held = s._replace(item=s.item.at[1, int(s.active[1])].set(herb))
    assert _jump(c.if_held_item_equal_to(_ctx(tables, held), _A, herb))
    gone = held._replace(item=held.item.at[1, int(s.active[1])].set(0))
    assert not _jump(c.if_held_item_equal_to(_ctx(tables, gone), _A, herb))
    hidden = s._replace(item=s.item.at[0, int(s.active[0])].set(herb))
    assert not _jump(c.if_held_item_equal_to(_ctx(tables, hidden), _D, herb))
    named = hidden._replace(ai_item_known=hidden.ai_item_known.at[int(s.active[0])].set(herb))
    assert _jump(c.if_held_item_equal_to(_ctx(tables, named), _D, herb))
