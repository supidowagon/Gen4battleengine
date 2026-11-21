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
from nuzline.engine.field import with_trick_room, with_weather
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
    slot = slots.index(ability)
    # The engine ability too: one that differs from the species row reads as a script's write
    # (``decomp_abilities``), which no Pokemon sent out has.
    own = int(np.asarray(tables.species_abilities)[species, slot])
    return s._replace(
        ability=s.ability.at[side, a].set(own),
        species=s.species.at[side, a].set(species),
        form=s.form.at[side, a].set(0),
        ability_form=s.ability_form.at[side, a].set(0),
        ability_slot=s.ability_slot.at[side, a].set(slot),
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
    lines = _decomp_text("generated/ai_load_type_targets.txt").split()
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
    tr = with_trick_room(s, 3)
    assert _jump(c.if_field_conditions_mask(_ctx(tables, tr), FieldCondition.TRICK_ROOM))


@pytest.mark.parametrize("mask, up", [(0x40000 - 3, True), (0xFFFF, False)])
def test_if_field_conditions_mask_trick_room_reads_the_mask(tables, names, mask, up):
    # script.s:364 etc.: fieldConditionsMask & FIELD_CONDITION_TRICK_ROOM (trainer_ai.c:
    # 1936-1946), any of bits 16-18, a borrowed counter's too (FCM 3.1).
    s = versus(tables, names, GEODUDE)
    s = s._replace(field_mask=jnp.asarray(mask, s.field_mask.dtype))
    assert _jump(c.if_field_conditions_mask(_ctx(tables, s), FieldCondition.TRICK_ROOM)) is up


@pytest.mark.parametrize("mask, up", [(0x7FFF, True), (0x5000 - 3, True), (0xFFF, False)])
def test_if_field_conditions_mask_gravity_reads_the_mask(tables, names, mask, up):
    # Basic_CheckGravityActive (script.s:1028): fieldConditionsMask & FIELD_CONDITION_GRAVITY
    # (trainer_ai.c:1936-1946), any of bits 12-14, a borrowed counter's too (FCM 3.1).
    s = versus(tables, names, GEODUDE)
    s = s._replace(field_mask=jnp.asarray(mask, s.field_mask.dtype))
    assert _jump(c.if_field_conditions_mask(_ctx(tables, s), FieldCondition.GRAVITY)) is up


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
    s = with_weather(s, weather)
    assert _temp(c.load_current_weather(_ctx(tables, s))) == expected


@pytest.mark.parametrize(
    "mask, expected",
    [
        (0x7F, AiWeather.HAILING),
        (0x807F, AiWeather.DEEP_FOG),
        (0x5, AiWeather.SANDSTORM),
        (0x3D, AiWeather.SUNNY),
    ],
)
def test_load_current_weather_takes_the_last_set_group(tables, names, mask, expected):
    # LoadCurrentWeather (trainer_ai.c:1414-1438) tests WEATHER_IS_RAIN, _SAND, _SUN, _HAIL,
    # _FOG one after another, each overwriting calcTemp: on a mask with several groups (FCM
    # 3.1-3.2) the last set one wins.
    s = versus(tables, names, GEODUDE)
    s = s._replace(field_mask=jnp.asarray(mask, s.field_mask.dtype))
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
    s = with_weather(s, Weather.SANDSTORM)
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
        return with_weather(st, w)

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


def _speed_state(tables, names, player: int, opp: int, trick_room: int = 0):
    s = versus(tables, names, GEODUDE)
    return with_trick_room(_speeds(s, player, opp), trick_room)


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
        "trick_room": with_trick_room(base, 3),
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
    seen = s._replace(ai_seen_moves=s.ai_seen_moves.at[0, 0].set(moves[slot]))
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


@pytest.mark.fullset
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
    named = hidden._replace(ai_item_known=hidden.ai_item_known.at[0].set(herb))
    assert _jump(c.if_held_item_equal_to(_ctx(tables, named), _D, herb))


# --- Partner arguments, GetPartner and the Tag Strategy commands (E4-A0, A-10 sections 0-6) --
#
# At B = 2 the partner arguments name battlers 2 and 3, whose BattleMon is all zero: the
# context is cleared (bcp.c:180-182) and only ``i < maxBattlers`` is initialized (:223-227).

_DP, _AP = Battler.DEFENDER_PARTNER, Battler.ATTACKER_PARTNER


def _decomp_text(relative: str) -> str:
    """A decomp file's text (the submodule is empty in a worktree that has not checked it out)."""
    path = DECOMP / relative
    if not path.exists():
        pytest.skip(f"external/pokeplatinum/{relative} is not checked out")
    return path.read_text()


_PARTNERS = (_AP, _DP)


def _slot_ctx(tables, s, slot: int = 0, rolls=(), reads: int = 8):
    """A per-slot-layout context at move ``slot``: the slot's reads in order are ``rolls``
    (``READ_DRAWS``: 1 below every N and odd, 254 above and even), then 254."""
    from nuzline.ai.outcome import READ_DRAWS, no_calc_draws

    script = np.full((4, reads), READ_DRAWS[0], np.uint8)
    for k, r in enumerate(rolls):
        script[slot, k] = r
    o = outcome()._replace(script_rolls=jnp.asarray(script), calc_draws=no_calc_draws((), reads))
    return c.at_slot(c.new_context(tables, s, o), slot)


def test_battler_args_are_the_decomps_and_resolve_by_xor_2(tables, names):
    # include/constants/battle/trainer_ai.h:28-31; AIScript_Battler (trainer_ai.c:2765-2782).
    header = _decomp_text("include/constants/battle/trainer_ai.h")
    for name in ("DEFENDER", "ATTACKER", "DEFENDER_PARTNER", "ATTACKER_PARTNER"):
        assert f"AI_BATTLER_{name}" in header
    assert [int(b) for b in (_D, _A, _DP, _AP)] == [0, 1, 0 ^ 2, 1 ^ 2]
    ctx = c.new_context(tables, versus(tables, names, GEODUDE), outcome())
    assert [int(c._battler(ctx, b)) for b in (_D, _A, _DP, _AP)] == [0, 1, 2, 3]


def test_cleared_battler_is_the_zeroed_battle_mon(tables, kaizo_tables):
    # battleMons[2] and [3] at B = 2: hp 0 / 0 (ruling G-53: the division returns the dividend),
    # ability 0, species 0 whose two abilities are 0 and 0 in both ROMs (A-8 section 9.2).
    z = c.CLEARED
    assert (z.cur_hp, z.max_hp, z.hp_percent, z.ability, z.species) == (0, 0, 0, 0, 0)
    assert z.types == (Type.NORMAL, Type.NORMAL) and z.held_item == 0 and z.moves == (0,) * 4
    for t in (tables, kaizo_tables):
        assert not np.asarray(t.species_abilities_decomp)[0].any()


def test_partner_args_b2_read_zero(tables, names):
    s = versus(tables, names, GEODUDE)
    ctx = c.at_slot(_ctx(tables, s)._replace(calc_temp=jnp.asarray(77, jnp.int32)), 0)
    for p in _PARTNERS:
        # hp% = 0 * 100 / 0 = 0 (G-53)
        assert _jump(c.if_hp_percent_equal_to(ctx, p, 0))
        assert not _jump(c.if_hp_percent_equal_to(ctx, p, 100))
        assert _jump(c.if_hp_percent_less_than(ctx, p, 30))
        assert not _jump(c.if_hp_percent_less_than(ctx, p, 0))
        assert not _jump(c.if_hp_percent_greater_than(ctx, p, 0))
        assert _jump(c.if_hp_percent_not_equal_to(ctx, p, 100))
        # the knowledge branch for DEFENDER_PARTNER (species 0: abilities 0, 0) and the raw
        # ability 0 for ATTACKER_PARTNER both read ABILITY_NONE: AI_UNKNOWN (:1255-1260)
        for expected in (0, 26, 100):
            assert _temp(c.check_battler_ability(ctx, p, expected)) == c.AI_UNKNOWN
        assert _temp(c.load_battler_ability(ctx, p)) == 0  # :1179-1210, no coin (:1202-1206)
        # masks 0: nothing set, IfNot* always jumps
        assert not _jump(c.if_status(ctx, p, MonCondition.ANY))
        assert _jump(c.if_not_status(ctx, p, MonCondition.ANY))
        for mask in c.MoveEffectMask:
            assert not _jump(c.if_move_effect(ctx, p, mask))
            assert _jump(c.if_not_move_effect(ctx, p, mask))
        # statBoosts stored 0, not the neutral 6
        assert _jump(c.if_stat_stage_equal_to(ctx, p, Stat.ATK, 0))
        assert not _jump(c.if_stat_stage_equal_to(ctx, p, Stat.ATK, 6))
        assert not _jump(c.if_stat_stage_equal_to(ctx, p, Stat.ATK, 12))
        assert not _jump(c.if_stat_stage_greater_than(ctx, p, Stat.SPE, 7))
        assert _jump(c.if_stat_stage_less_than(ctx, p, Stat.SPE, 6))
        # no moves; IfMoveKnown: ATTACKER_PARTNER no jump with curHP 0 (:1650-1653), the
        # DEFENDER_PARTNER has no case (:1678-1679), for IfMoveNotKnown too
        for move in (0, 33):
            assert not _jump(c.if_move_known(ctx, p, move))
            assert not _jump(c.if_move_not_known(ctx, p, move))
        # item 0 on both sides of the record (ai_record_item is 0 for b >= B)
        assert _jump(c.if_held_item_equal_to(ctx, p, 0))
        assert not _jump(c.if_held_item_equal_to(ctx, p, c.ITEM_POWER_HERB))


def test_partner_args_b2_do_not_follow_the_real_battlers(tables, names):
    # The cleared battler is not the attacker's or the defender's: a held item, status or stat
    # stage on either real battler changes nothing at battlers 2 and 3.
    s = versus(tables, names, GEODUDE)
    s = s._replace(
        item=s.item.at[1, GEODUDE].set(c.ITEM_POWER_HERB).at[0, 0].set(c.ITEM_POWER_HERB),
        status=s.status.at[0, 0].set(Status.BURN).at[1, GEODUDE].set(Status.BURN),
        stages=s.stages.at[:, 0].set(3),
    )
    ctx = c.at_slot(_ctx(tables, s), 0)
    for p in _PARTNERS:
        assert not _jump(c.if_held_item_equal_to(ctx, p, c.ITEM_POWER_HERB))
        assert not _jump(c.if_status(ctx, p, MonCondition.ANY))
        assert _jump(c.if_stat_stage_equal_to(ctx, p, Stat.ATK, 0))
    # the real battlers: ATTACKER reads its own item, DEFENDER the AI's record (nothing named)
    assert _jump(c.if_held_item_equal_to(ctx, _A, c.ITEM_POWER_HERB))
    assert not _jump(c.if_held_item_equal_to(ctx, _D, c.ITEM_POWER_HERB))


def test_check_battler_ability_branches_on_the_argument(tables, names):
    # CheckBattlerAbility takes the knowledge branch for the DEFENDER and DEFENDER_PARTNER
    # arguments and the raw ability for the other two (trainer_ai.c:1223, :1255-1256): the
    # defender's real ability is hidden (species of two abilities, none expected: ability 1).
    geodude = names.species("Geodude")
    base = np.asarray(tables.species_abilities_decomp)[geodude]
    assert base[0] != 0 and base[1] != 0 and base[0] != base[1]
    s = _species_with(tables, versus(tables, names, GEODUDE), 0, geodude, int(base[0]))
    ctx = c.at_slot(_ctx(tables, s), 0)
    assert _temp(c.check_battler_ability(ctx, _D, int(base[0]))) == c.AI_UNKNOWN
    assert _temp(c.check_battler_ability(ctx, _D, 255)) == c.AI_NOT_HAVE  # ability 1 != 255
    raw = int(c.attacker_ability(tables, s))
    assert _temp(c.check_battler_ability(ctx, _A, raw)) == c.AI_HAVE


def test_load_type_attacker_partner_b2(tables, names):
    # LOAD_ATTACKER_PARTNER_TYPE_1 / _2 (generated/ai_load_type_targets.txt: line n + 1 = n) read
    # BattleSystem_GetPartner(attacker) (:965-978), the attacker itself in singles (sys:400-402).
    lines = _decomp_text("generated/ai_load_type_targets.txt").split()
    assert lines.index("LOAD_ATTACKER_PARTNER_TYPE_1") == LoadType.ATTACKER_PARTNER_TYPE_1 == 6
    assert lines.index("LOAD_ATTACKER_PARTNER_TYPE_2") == LoadType.ATTACKER_PARTNER_TYPE_2 == 8
    s = versus(tables, names, GEODUDE)
    ctx = c.at_slot(_ctx(tables, s), 0)
    own = np.asarray(s.types[1, GEODUDE])
    assert own[0] != own[1]  # Rock, Ground
    assert _temp(c.load_type_from(ctx, LoadType.ATTACKER_PARTNER_TYPE_1)) == own[0]
    assert _temp(c.load_type_from(ctx, LoadType.ATTACKER_PARTNER_TYPE_2)) == own[1]
    assert _temp(c.load_type_from(ctx, LoadType.ATTACKER_TYPE_1)) == own[0]
    assert c._get_partner(s, 1) == 1 and c._get_partner(s, 0) == 0


def test_flag_battler_is_type(tables, names):
    # FlagBattlerIsType (:991-1004): calcTemp 1 iff type 1 or 2 of the battler is the type.
    s = versus(tables, names, GEODUDE)
    ctx = c.at_slot(_ctx(tables, s), 0)
    for battler, side in ((_A, 1), (_D, 0)):
        types = {int(t) for t in np.asarray(s.types[side, int(s.active[side])])}
        for t in Type:
            got = _temp(c.flag_battler_is_type(ctx, battler, t))
            assert got == int(int(t) in types), (battler, t)
    # Partner args at B = 2: types 0, 0 = Normal only
    for p in _PARTNERS:
        assert _temp(c.flag_battler_is_type(ctx, p, Type.NORMAL)) == 1
        assert _temp(c.flag_battler_is_type(ctx, p, Type.FLYING)) == 0
        assert _temp(c.flag_battler_is_type(ctx, p, Type.ROCK)) == 0


def test_flag_battler_is_type_arceus_plate(tables, names):
    # Battler_MonType (battle_lib.c:7618-7703): a Multitype Arceus has its plate's type twice.
    from nuzline.engine.damage import personal_types

    s = versus(tables, names, GEODUDE)
    arceus = names.species("Arceus")
    for plate, kind in (("Flame Plate", Type.FIRE), (None, Type.NORMAL)):
        item = names.item(plate) if plate else 0
        types = personal_types(tables, arceus, 0, 0, 0, item)
        t = s._replace(types=s.types.at[1, GEODUDE].set(types.astype(s.types.dtype)))
        ctx = c.at_slot(_ctx(tables, t), 0)
        assert _temp(c.flag_battler_is_type(ctx, _A, kind)) == 1
        assert _temp(c.flag_battler_is_type(ctx, _A, Type.ROCK)) == 0


def test_if_activated_flash_fire(tables, names):
    # IfActivatedFlashFire (:2641-2652): moveEffectsData.flashFire of the battler.
    s = versus(tables, names, GEODUDE)
    assert not _jump(c.if_activated_flash_fire(c.at_slot(_ctx(tables, s), 0), _A))
    on = s._replace(flash_fire=s.flash_fire.at[1].set(1))
    ctx = c.at_slot(_ctx(tables, on), 0)
    assert _jump(c.if_activated_flash_fire(ctx, _A))
    assert not _jump(c.if_activated_flash_fire(ctx, _D))
    for p in _PARTNERS:  # the zeroed battlers never have it
        assert not _jump(c.if_activated_flash_fire(ctx, p))
    other = s._replace(flash_fire=s.flash_fire.at[0].set(1))
    assert _jump(c.if_activated_flash_fire(c.at_slot(_ctx(tables, other), 0), _D))


def test_if_battler_fainted_b2(tables, names):
    # IfBattlerFainted (:2438-2452): battlersSwitchingMask has battlers 2 and 3 from the start in
    # a single battle (battle_lib.c:2031-2034); the assertions refuse the real battlers' args.
    s = versus(tables, names, GEODUDE)
    ctx = c.at_slot(_ctx(tables, s), 0)
    for p in _PARTNERS:
        assert _jump(c.if_battler_fainted(ctx, p))
    for b in (_A, _D):
        with pytest.raises(AssertionError):
            c.if_battler_fainted(ctx, b)


def test_if_held_item_equal_to_partner_sides(tables, names):
    # Same side as the attacker -> its true heldItem (:1925-1926): ATTACKER_PARTNER is battler 3,
    # whose item is 0 whatever the attacker holds; DEFENDER_PARTNER takes the record (:1927-1928).
    s = versus(tables, names, GEODUDE)
    held = s._replace(item=s.item.at[1, GEODUDE].set(c.ITEM_POWER_HERB))
    ctx = c.at_slot(_ctx(tables, held), 0)
    assert _jump(c.if_held_item_equal_to(ctx, _A, c.ITEM_POWER_HERB))
    assert not _jump(c.if_held_item_equal_to(ctx, _AP, c.ITEM_POWER_HERB))
    named = held._replace(ai_item_known=held.ai_item_known.at[0].set(c.ITEM_POWER_HERB))
    ctx = c.at_slot(_ctx(tables, named), 0)
    assert not _jump(c.if_held_item_equal_to(ctx, _DP, c.ITEM_POWER_HERB))  # row 2: 0 at B = 2


# --- LoadBattlerSpeedRank (trainer_ai.c:2056-2094), B = 2 -----------------------------------


def _rank(ctx, battler=_A):
    out = c.load_battler_speed_rank(ctx._replace(calc_temp=jnp.asarray(77, jnp.int32)), battler)
    return out, _temp(out)


def test_load_battler_speed_rank_b2(tables, names):
    s = versus(tables, names, GEODUDE)
    # distinct speeds: the one pair, no draw. The faster battler is at position 0.
    for player, opp, attacker_rank in ((40, 90, 0), (90, 40, 1)):
        fast = _speeds(s, player, opp)
        out, rank = _rank(_slot_ctx(tables, fast))
        assert rank == attacker_rank
        assert int(out.reads[0]) == 0 and float(out.weight[0]) == 1.0
        assert _rank(_slot_ctx(tables, fast), _D)[1] == 1 - attacker_rank
        # the compare wrote both speeds to monSpeedValues (battle_lib.c:1372-1373)
        assert bool(out.written) and int(out.first_cmp[0]) == int(out.flag)


def test_load_battler_speed_rank_tie_draw_law_b2(tables, names):
    from nuzline.ai.outcome import READ_DRAWS

    s = _speeds(versus(tables, names, GEODUDE), 60, 60)
    # equal speed: RandNext() & 1 decides TIE (a swap: the attacker, battler 1, is first) or
    # FASTER (no swap) (battle_lib.c:1440-1446); one read of probability 1/2 either way
    for draw, attacker_rank in ((READ_DRAWS[1], 0), (READ_DRAWS[0], 1)):
        out, rank = _rank(_slot_ctx(tables, s, rolls=(draw,)))
        assert rank == attacker_rank
        assert int(out.reads[0]) == 1 and float(out.weight[0]) == 0.5


def test_load_battler_speed_rank_partner_arg_leaves_calc_temp(tables, names):
    s = _speeds(versus(tables, names, GEODUDE), 60, 60)
    for p in _PARTNERS:
        out, rank = _rank(_slot_ctx(tables, s), p)
        assert rank == 77  # not in the order (:2088-2093)
        assert int(out.reads[0]) == 0  # its result goes unused: no read of the exact layout
        assert bool(out.written)  # the compare still wrote monSpeedValues


def test_load_battler_speed_rank_dead_battler_b2(tables, names):
    # One battler without HP is SLOWER at once (:1204-1211): no draw, no monSpeedValues write.
    s = _speeds(versus(tables, names, GEODUDE), 60, 60)
    dead = s.replace_hp(0, 0, 0)
    out, rank = _rank(_slot_ctx(tables, dead))
    assert rank == 0 and int(out.reads[0]) == 0 and not bool(out.written)
    dead = s.replace_hp(1, GEODUDE, 0)
    out, rank = _rank(_slot_ctx(tables, dead))
    assert rank == 1 and int(out.reads[0]) == 0 and not bool(out.written)


def test_load_battler_speed_rank_trick_room_and_quick_claw(tables, names):
    s = _speeds(versus(tables, names, GEODUDE), 40, 90)
    assert _rank(_slot_ctx(tables, s))[1] == 0
    assert _rank(_slot_ctx(tables, with_trick_room(s, 3)))[1] == 1  # the slower acts first
    # a lone Quick Claw holder whose draw activated it is first (:1408-1411), from the previous
    # turn's draw ``qc_roll`` (``ignoreQuickClaw`` only skips the write)
    slow = _speeds(versus(tables, names, GEODUDE), 90, 40)
    claw = slow._replace(
        item=slow.item.at[1, GEODUDE].set(names.item("Quick Claw")),
        qc_roll=slow.qc_roll.at[1].set(1),
    )
    assert _rank(_slot_ctx(tables, slow))[1] == 1
    assert _rank(_slot_ctx(tables, claw))[1] == 0


def test_load_battler_speed_rank_loose_draws_the_tie(tables, names):
    # The loose oracle draws the tie as its own read: SPEED_TIE_DRAWN where the result is used,
    # SPEED_TIE for a partner argument (its draw changes nothing the exact layout reads).
    s = _speeds(versus(tables, names, GEODUDE), 60, 60)
    for bit, attacker_rank in ((1, 0), (0, 1)):
        bits = np.zeros((MOVES, c.LOOSE_READS), np.int8)
        bits[0, 0] = bit
        ctx = c.at_slot(_ctx(tables, s, loose_bits=bits), 0)
        out, rank = _rank(ctx)
        assert rank == attacker_rank
        assert int(out.loose.count[0]) == 1
        assert int(out.loose.kind[0, 0]) == c.ReadKind.SPEED_TIE_DRAWN
        assert float(out.loose.p[0, 0]) == 0.5 and bool(out.loose.live[0, 0])
        out, rank = _rank(ctx, _AP)
        assert rank == 77 and int(out.loose.kind[0, 0]) == c.ReadKind.SPEED_TIE


# --- CheckIfHighestDamageWithPartner (trainer_ai.c:2369-2436), B = 2 -------------------------

TRIPLE_AXEL, BULLDOZE = 149, 222  # Kaizo's ids of Psywave's and Magnitude's draws (G-39)


def _tag_fight(kaizo_tables, kaizo_names, kaizo_roark_party, moves):
    """Kaizo Roark's Geodude (level 15) with ``moves`` (move names or ids) against his
    Cranidos-led sample: the AI's four moves are the estimate's sweeps."""
    s = make_state(kaizo_tables, [kaizo_roark_party[2]], list(kaizo_roark_party))
    ids = [m if isinstance(m, int) else kaizo_names.move(m) for m in moves]
    return s._replace(
        moves=s.moves.at[1, 0].set(jnp.asarray(ids, s.moves.dtype)),
        pp=s.pp.at[1, 0].set(jnp.full(4, 20, s.pp.dtype)),
    )


def _p_highest(est, slot: int, sweeps: int) -> float:
    """P(HIGHEST) by enumerating the joint law of every estimate of ``sweeps`` independent
    sweeps of ``est`` (``CalcEstimate [4, C]``): the slot's sweep-1 value must be >= every
    value of every sweep."""
    import itertools

    value, prob = np.asarray(est.value), np.asarray(est.prob, np.float64)
    cells = [(i, k) for k in range(sweeps) for i in range(4)]
    ranges = [[cl for cl in range(prob.shape[1]) if prob[i, cl] > 0] for i, _ in cells]
    total = 0.0
    for combo in itertools.product(*ranges):
        p = float(np.prod([prob[i, cl] for (i, _), cl in zip(cells, combo)]))
        d = [value[i, cl] for (i, _), cl in zip(cells, combo)]
        if all(x <= d[slot] for x in d):
            total += p
    return total


def _highest_read(tables, s, slot: int, highest: bool):
    """The command's ``(calc_temp, weight, reads)`` at ``slot`` when its one read comes out
    ``highest`` (the enumeration's pattern: the read's bit is its result)."""
    from nuzline.ai.outcome import READ_DRAWS

    out = c.check_if_highest_damage_with_partner(
        _slot_ctx(tables, s, slot, rolls=(READ_DRAWS[int(highest)],)), c.USE_MAX_DAMAGE
    )
    return _temp(out), float(out.weight[slot]), int(out.reads[slot])


@pytest.mark.parametrize("slot", range(4))
def test_highest_with_partner_b2_without_a_drawing_id_is_flag_move_damage_score(
    kaizo_tables, kaizo_names, kaizo_roark_party, slot
):
    # Sweep 2 estimates the same moves, so with no Psywave / Magnitude id the result is the
    # one sweep's HIGHEST and no read is made (trainer_ai.c:2369-2436, sys:400-402).
    s = _tag_fight(
        kaizo_tables,
        kaizo_names,
        kaizo_roark_party,
        ["Tackle", "Rock Throw", "Earthquake", "Harden"],
    )
    ctx = _slot_ctx(kaizo_tables, s, slot)
    one = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    two = c.check_if_highest_damage_with_partner(ctx, c.USE_MAX_DAMAGE)
    assert _temp(two) == _temp(one)
    assert int(two.reads[slot]) == int(one.reads[slot]) == 0
    assert float(two.weight[slot]) == 1.0


def test_highest_with_partner_b2_not_damaging_is_no_comparison(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    # The gate is FlagMoveDamageScore's (:2393-2394): Harden (slot 3) is not damaging, and a
    # drawing move elsewhere in the set makes no read for it.
    s = _tag_fight(
        kaizo_tables,
        kaizo_names,
        kaizo_roark_party,
        [TRIPLE_AXEL, "Tackle", "Earthquake", "Harden"],
    )
    out = c.check_if_highest_damage_with_partner(_slot_ctx(kaizo_tables, s, 3), c.USE_MAX_DAMAGE)
    assert _temp(out) == c.AI_NO_COMPARISON_MADE
    assert int(out.reads[3]) == 0 and float(out.weight[3]) == 1.0


@pytest.mark.parametrize(
    "moves",
    [
        [TRIPLE_AXEL, "Tackle", "Rock Throw", "Harden"],  # Psywave's case (Triple Axel)
        [BULLDOZE, "Tackle", "Rock Throw", "Harden"],  # Magnitude's case (Bulldoze)
        [TRIPLE_AXEL, BULLDOZE, "Tackle", "Rock Throw"],  # both, independent in each sweep
    ],
)
def test_highest_with_partner_b2_two_sweeps_exact_law(
    kaizo_tables, kaizo_names, kaizo_roark_party, moves
):
    # Two sweeps of the attacker's moves with independent draws (the partner is the attacker):
    # HIGHEST iff the slot's sweep-1 value >= every value of both sweeps; one read per slot.
    s = _tag_fight(kaizo_tables, kaizo_names, kaizo_roark_party, moves)
    est = c.seen_damage(_slot_ctx(kaizo_tables, s, 0))
    for slot in range(4):
        if int(est.value[slot, 0]) == 0:  # not damaging (Harden): the gate
            continue
        law = _p_highest(est, slot, sweeps=2)
        once = _p_highest(est, slot, sweeps=1)
        temp, weight, reads = _highest_read(kaizo_tables, s, slot, True)
        assert (temp, reads) == (c.AI_MOVE_IS_HIGHEST_DAMAGE, 1)
        assert weight == pytest.approx(law, abs=1e-6), (moves, slot)
        temp, weight, reads = _highest_read(kaizo_tables, s, slot, False)
        assert (temp, reads) == (c.AI_NOT_HIGHEST_DAMAGE, 1)
        assert weight == pytest.approx(1 - law, abs=1e-6), (moves, slot)
        # the second sweep matters: a draw can only make HIGHEST less likely, strictly for a
        # slot some draw can exceed
        assert law <= once + 1e-9
    # a fixed-damage slot (Tackle) beaten by a drawing move is strictly less likely
    slot = moves.index("Tackle")
    assert _p_highest(est, slot, 2) < _p_highest(est, slot, 1) - 1e-3


def test_highest_with_partner_b2_loose_digits(kaizo_tables, kaizo_names, kaizo_roark_party):
    # The loose oracle: each estimate that draws is its own digit read, eight estimates in the
    # decomp's order (sweep 1 then sweep 2), the result computed from the digits.
    s = _tag_fight(
        kaizo_tables,
        kaizo_names,
        kaizo_roark_party,
        [TRIPLE_AXEL, "Tackle", "Rock Throw", "Harden"],
    )
    est = c.seen_damage(_slot_ctx(kaizo_tables, s, 0))
    values = np.asarray(est.value[0])  # Triple Axel's damage per class (Psywave's % 11)
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    assert values[lo] < 12 < values[hi]  # beside Tackle's 12
    for x, y in ((hi, lo), (hi, hi), (lo, hi), (lo, lo)):
        bits = np.zeros((MOVES, c.LOOSE_READS), np.int8)
        bits[0, 0], bits[0, 4] = x, y  # estimates 0 (sweep 1) and 4 (sweep 2) are Triple Axel
        ctx = c.at_slot(_ctx(kaizo_tables, s, loose_bits=bits), 0)
        out = c.check_if_highest_damage_with_partner(ctx, c.USE_MAX_DAMAGE)
        assert int(out.loose.count[0]) == 8
        kinds = np.asarray(out.loose.kind[0, :8])
        assert list(kinds) == [c.ReadKind.CALC_PSYWAVE, 0, 0, 0] * 2
        got = _temp(out) == c.AI_MOVE_IS_HIGHEST_DAMAGE
        # HIGHEST iff Tackle (12), Rock Throw (12) and the second draw are all <= the first
        assert got == (values[x] >= 12 and values[y] <= values[x])


def test_highest_with_partner_b2_sampled_draws_take_sweep_two_from_the_next_raw_draw(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    # A sampled outcome carries the raw RandNext() values: the first drawing estimate (sweep 1's
    # Triple Axel) takes draws[0], sweep 2's takes draws[1] (``_drawn_classes``); Psywave's class
    # is ``RandNext() % 11``.
    from nuzline.ai.outcome import CALC_DRAWS

    s = _tag_fight(
        kaizo_tables,
        kaizo_names,
        kaizo_roark_party,
        [TRIPLE_AXEL, "Tackle", "Rock Throw", "Harden"],
    )
    values = np.asarray(c.seen_damage(_slot_ctx(kaizo_tables, s, 0)).value[0])
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    for x, y in ((hi, lo), (hi, hi), (lo, hi)):
        calc = np.zeros((4, 8, CALC_DRAWS), np.uint16)
        calc[0, 0, 0], calc[0, 0, 1] = x, y  # raw draws: their % 11 is the class
        base = _slot_ctx(kaizo_tables, s, 0)
        o = base.outcome._replace(calc_draws=jnp.asarray(calc))
        out = c.check_if_highest_damage_with_partner(base._replace(outcome=o), c.USE_MAX_DAMAGE)
        assert int(out.reads[0]) == 1
        got = _temp(out) == c.AI_MOVE_IS_HIGHEST_DAMAGE
        assert got == (values[x] >= 12 and values[y] <= values[x])
