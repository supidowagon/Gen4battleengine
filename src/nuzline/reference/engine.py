"""Plain-Python reference engine: one battle turn, written as straight-line, branchy
code that follows the decomp's control flow, one ``if`` per rule.

It is an independent second implementation of the JAX engine (``nuzline.engine``),
used by the differential tests: it imports nothing from ``nuzline.engine`` but
``ids``, and every rule cites the decomp function it follows (paths
relative to ``external/pokeplatinum``; ``bcp.c`` is
``src/battle/battle_controller_player.c``, ``lib.c`` ``src/battle/battle_lib.c``,
``script.c`` ``src/battle/battle_script.c``, ``.s`` files are under
``res/battle/scripts/``).

Entry points:

* ``ref_step(tables, state, actions, outcome) -> RefState``: one turn, like
  ``engine.step.step``; the input state is not modified.
* ``ref_replace(tables, state, side, slot) -> RefState``: a faint replacement,
  like ``engine.step.replace``.
* ``ref_damage_table(tables, state, atk_side, move, crit) -> list[int]``: the 16
  damage rolls, like ``engine.damage.damage_table``.

``tables`` is ``data.tables.to_numpy(tables)``; ``outcome`` is
``engine.outcome.outcome_to_py(outcome)``. Every random event comes from
``outcome`` with the JAX engine's meaning (``engine/outcome.py``). Only the
implemented mechanics exist; anything outside them is an ``UNSUPPORTED`` id in the
tables and does nothing here. Actions are assumed legal (``engine.actions``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from nuzline.engine.ids import (
    ABILITY_MULTITYPE,
    MOVE_FAKE_OUT,
    MOVE_STRUGGLE,
    POTION_HP_RESTORED,
    PROTECT_CHAIN_MOVES,
    RESIST_ANY_NORMAL,
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    TERRAIN_TYPES,
    TRI_ATTACK_STATUSES,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    MoveTarget,
    Stat,
    Status,
    TrainerItemId,
    Type,
    Weather,
)
from nuzline.reference.state import RefPokemon, RefSide, RefState

MOVE_ACTIONS = 4  # actions 0-3 use a move slot, 4-9 switch to party slot action - 4
FIRST_ITEM_ACTION = 10  # opponent actions 10-13 use bag slot action - 10
FIGHT, SWITCH, ITEM = "fight", "switch", "item"  # the decomp's PLAYER_INPUT_* commands
PHYSICAL, SPECIAL, STATUS_CLASS = 0, 1, 2  # tables.move_category (the decomp's CLASS_*)
WINNER_OPPONENT, WINNER_PLAYER = 1, 2
STAGE_MIN, STAGE_MAX = -6, 6
# sStatStageBoosts (lib.c:6550), indexed by stage + 6.
STAGE_RATIO = [
    (10, 40), (10, 35), (10, 30), (10, 25), (10, 20), (10, 15), (10, 10),
    (15, 10), (20, 10), (25, 10), (30, 10), (35, 10), (40, 10),
]  # fmt: skip
# MON_CONDITION_TOXIC_COUNTER is 4 bits (include/constants/battle/condition.h).
TOXIC_COUNTER_MAX = 15
PURSUIT_POWER_MUL = 20  # BtlCmd_TryPursuit (script.c:6884): powerMul = 20
CONFUSION_SELF_HIT_POWER = 40  # CALC_SELF_HIT(MOVE_STRUGGLE, 40), bcp.c:2632
# BtlCmd_CheckStealthRock (script.c:8302): maxHP divisor by BattleSystem_TypeMatchupMultiplier.
STEALTH_ROCK_DIVISOR = {160: 2, 80: 4, 40: 8, 20: 16, 10: 32}
# BattleSystem_Defender (lib.c:1599): ranges whose defender exists whatever the foe's
# HP: the user itself (lib.c:1665-1669) or, for RANGE_OPPONENT_SIDE, the foe's
# battler slot (BattleSystem_RandomOpponent, lib.c:4194).
DEFENDER_ALWAYS = {
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_OPPONENT_SIDE,
}
# ... of which these hit the user, not the foe (no damage to the foe's active).
TARGETS_USER = {MoveTarget.RANGE_USER, MoveTarget.RANGE_USER_SIDE, MoveTarget.RANGE_FIELD}
# BattleSystem_Defender (lib.c:1665-1683): in singles these ranges make the user the
# defender, so the move leaves no moveHit on the foe.
SELF_DEFENDER = TARGETS_USER | {
    MoveTarget.RANGE_SINGLE_TARGET_SPECIAL,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
}
# _status_disrupted's results.
GOES_AHEAD, DISRUPTED, HURT_ITSELF = 0, 1, 2
# sProtectSuccessRate (script.c:5128-5133), indexed by protectSuccessTurns.
PROTECT_SUCCESS_RATE = [0xFFFF, 0x7FFF, 0x3FFF, 0x1FFF]
# HitRateByStage (include/data/hit_rate_stages.h), indexed by the clamped stage sum.
HIT_RATE_BY_STAGE = [
    (33, 100), (36, 100), (43, 100), (50, 100), (60, 100), (75, 100), (1, 1),
    (133, 100), (166, 100), (2, 1), (233, 100), (133, 50), (3, 1),
]  # fmt: skip


# --------------------------------------------------------------------------- helpers


@dataclass
class _Turn:
    """Per-turn scratch the decomp keeps in ``BattleContext``."""

    tables: object
    outcome: object
    # totalTurns + 1, the fakeOutTurnNumber a switch-in or faint writes (lib.c:2117,
    # :2195): turn + 1 during ref_step, turn in ref_replace (after the turn was counted).
    entry_turn: int
    fainted: list[tuple[int, int]] = field(default_factory=list)  # SYSCTL_MON_FAINTED
    done: list[bool] = field(default_factory=lambda: [False, False])  # action is MOVE_END
    # selfTurnFlags, cleared by BattleContext_Init at every action's end (bcp.c:3998,
    # lib.c:2014): the defender's focusItemActivated (bcp.c:3391) and the attacker's
    # SELF_TURN_FLAG_PLUCK_BERRY (lib.c:5848).
    focus_armed: bool = False
    pluck_flag: bool = False
    # turnFlags.protecting (cleared at the turn end, lib.c:2231): TryProtection succeeded.
    protecting: list[bool] = field(default_factory=lambda: [False, False])
    # SYSCTL_ATTACK_MESSAGE_SHOWN of the current action (script.c:1432), for
    # UpdateMoveBuffers' moveProtect (bcp.c:3936-3942).
    attack_message: bool = False


def _active(s: RefState, side: int) -> RefPokemon:
    return s.sides[side].party[s.sides[side].active]


def _t(tables, name: str, *index) -> int:
    return int(getattr(tables, name)[index])


def _terrain_row(s: RefState) -> int | None:
    """The terrain tables' row ``BattleSystem_GetTerrain`` gives ``GetTerrainMove`` /
    ``TryCamouflage`` (``min(terrain, TERRAIN_SPECIAL)``, script.c:8007-8010, :8040-8043), or
    None for an unknown terrain (-1), which no table is indexed with (coverage refuses such a
    fight; its readers do nothing, as the JAX engine's)."""
    return None if s.terrain < 0 else min(s.terrain, TERRAIN_SPECIAL)


def _called_move(tables, s: RefState, move: int) -> int:
    """effect_script_0173.s (Nature Power): ``GetTerrainMove`` (script.c:8036-8048) then
    ``GoToMoveScript`` (:2111-2134) runs ``sTerrainMove[row]`` as ``moveCur``; MOVE_NONE for
    an unknown terrain. Any other move runs itself."""
    if _t(tables, "move_effect", move) != EffectId.NATURE_POWER:
        return move
    row = _terrain_row(s)
    return 0 if row is None else TERRAIN_MOVES[row]


def _restore_types(tables, s: RefState, side: int) -> None:
    """The leaving battler's types go back to its personal row's: Camouflage writes only the
    battler's ``BattleMon.type1/type2``, which ``BattleSystem_InitBattleMon`` re-reads from
    the party (lib.c:114-115)."""
    mon = _active(s, side)
    row = _t(tables, "species_form_row", mon.species, mon.form)
    mon.types = [_t(tables, "species_types", row, i) for i in range(len(mon.types))]


def _hidden_power(ivs: list[int]) -> tuple[int, int]:
    """``(type, power)`` of Hidden Power (``BtlCmd_CalcHiddenPowerParams``,
    script.c:6007-6033; ``ivs`` in ``ids.Stat`` order, the decomp's bit order is HP,
    Atk, Def, Spe, SpA, SpD)."""
    hp, atk, dfn = ivs[Stat.HP], ivs[Stat.ATK], ivs[Stat.DEF]
    spe, spa, spd = ivs[Stat.SPE], ivs[Stat.SPA], ivs[Stat.SPD]
    power = ((hp & 2) >> 1) | (atk & 2) | ((dfn & 2) << 1)
    power |= ((spe & 2) << 2) | ((spa & 2) << 3) | ((spd & 2) << 4)
    kind = (hp & 1) | ((atk & 1) << 1) | ((dfn & 1) << 2)
    kind |= ((spe & 1) << 3) | ((spa & 1) << 4) | ((spd & 1) << 5)
    power = power * 40 // 63 + 30
    kind = kind * 15 // 63 + 1
    if kind >= Type.MYSTERY:
        kind += 1
    return kind, power


def _move_type(tables, s: RefState, side: int, move: int) -> int:
    """The type of ``move`` as ``side``'s active Pokemon uses it (lib.c:6682; Hidden
    Power's from its IVs, script.c:6028)."""
    if _t(tables, "move_effect", move) == EffectId.HIDDEN_POWER:
        return _hidden_power(_active(s, side).ivs)[0]
    return _t(tables, "move_type", move)


# BtlCmd_CalcMagnitudePower (script.c:5820-5852), indexed by level - 4.
MAGNITUDE_POWER = [10, 30, 50, 70, 90, 110, 150]
# BtlCmd_Present (script.c:5789-5806): rnd = RandNext() & 0xFF; < 102 -> 40, < 178 -> 80,
# < 204 -> 120 (Outcome.present 0, 1, 2), else the heal path (PRESENT_HEAL).
PRESENT_POWER = [40, 80, 120]
PRESENT_HEAL = 3
# sHPPixelsToFlailPower (script.c:4941-4948): (largest pixel count, power).
FLAIL_BANDS = [(1, 200), (5, 150), (12, 100), (21, 80), (42, 40), (64, 20)]


def _flail_power(cur: int, max_hp: int) -> int:
    """``BtlCmd_CalcFlailPower`` (script.c:4958-4970) on ``App_PixelCount(cur, max, 64)``
    (src/unk_0208C098.c:41-49)."""
    pixels = cur * 64 // max_hp
    if pixels == 0 and cur > 0:
        pixels = 1
    for bound, power in FLAIL_BANDS:
        if pixels <= bound:
            return power
    return FLAIL_BANDS[-1][1]


def _rollout_multiplier(s: RefState, attacker: int) -> int:
    """``BtlCmd_CalcRolloutPower`` (script.c:5661): a first use starts ``rolloutCount`` at
    5, each use decrements it, and ``movePower`` doubles ``5 - rolloutCount - 1`` times
    (once more when the user is curled, ``VOLATILE_CONDITION_DEFENSE_CURL``)."""
    sd = s.sides[attacker]
    left = max(sd.rollout_count - 1, 0) if sd.locked_move else 4
    power = 1
    for _ in range(1, 5 - left):
        power *= 2
    return power * 2 if sd.curled else power


def _fixed_damage(tables, s: RefState, attacker: int, move: int) -> int:
    """``effect_script_0087.s``: the attacker's level, with ``SYSCTL_IGNORE_TYPE_CHECKS``
    (lib.c:7544): no multiplier, but the chart's immunity still fails the move. Bide's
    unleash is ``2 * storedDamage`` (bcp.c:2702) and ignores the chart (``RANGE_USER``,
    bcp.c:2350-2358); on any other Bide turn nothing is dealt."""
    sd = s.sides[attacker]
    if _t(tables, "move_effect", move) == EffectId.BIDE:
        return 2 * sd.bide_damage if sd.locked_move and sd.bide_turns == 0 else 0
    foe = _active(s, 1 - attacker)
    kind = _move_type(tables, s, attacker, move)
    if _matchup_x40(tables, kind, foe.types, bool(s.sides[1 - attacker].foresight)) == 0:
        return 0
    return _active(s, attacker).level


def _divide(dividend: int, divisor: int) -> int:
    """``BattleSystem_Divide`` (lib.c:3599) for a non-negative dividend: 0 stays 0,
    otherwise at least 1."""
    if dividend == 0:
        return 0
    return max(dividend // divisor, 1)


def _lose_hp(turn: _Turn, s: RefState, side: int, amount: int) -> int:
    """``BATTLE_SUBSCRIPT_UPDATE_HP`` with a negative amount on ``side``'s active
    Pokemon: HP never drops below 0; reaching 0 flags a faint. Returns the HP lost."""
    mon = _active(s, side)
    lost = min(amount, mon.hp)
    mon.hp -= lost
    if lost > 0 and mon.hp == 0:
        turn.fainted.append((side, s.sides[side].active))
    return lost


def _gain_hp(s: RefState, side: int, amount: int) -> None:
    mon = _active(s, side)
    mon.hp = min(mon.hp + amount, mon.max_hp)


def _change_stage(s: RefState, side: int, stat: int, delta: int) -> None:
    """``BtlCmd_ChangeStatStage`` (script.c:2804): clamped to +-6."""
    stages = s.sides[side].stages
    stages[stat] = min(max(stages[stat] + delta, STAGE_MIN), STAGE_MAX)


def _simple(stage: int) -> int:
    """Simple doubles a stage where it is used, clamped to +-6 (lib.c:6850,
    CompareSpeed_ApplySimple lib.c:1171)."""
    return min(max(stage * 2, STAGE_MIN), STAGE_MAX)


def _apply_stage(stat: int, stage: int) -> int:
    num, den = STAGE_RATIO[stage + 6]
    return stat * num // den


def _ignorable(s: RefState, attacker: int, defender: int) -> int:
    """``Battler_IgnorableAbility``: the defender's ability as the attacker sees it;
    Mold Breaker hides it."""
    if _active(s, attacker).ability == AbilityId.MOLD_BREAKER:
        return AbilityId.NONE
    return _active(s, defender).ability


def _chart_rows(tables, move_type: int, identified: bool = False) -> list[tuple[int, int]]:
    """The rows of ``sTypeMatchupMultipliers`` (lib.c:2399) for ``move_type``, in
    table order, as (defending type, multiplier in tenths). ``identified`` (Foresight)
    drops the rows after the 0xFE separator (lib.c:2615)."""
    order = tables.type_chart_order[move_type]
    listed = [
        (int(order[d]), d)
        for d in range(len(order))
        if int(order[d]) >= 0 and not (identified and tables.type_chart_foresight[move_type, d])
    ]
    return [(d, _t(tables, "type_chart", move_type, d)) for _, d in sorted(listed)]


def _matchup_x40(tables, attacking_type: int, types: list[int], identified: bool = False) -> int:
    """``BattleSystem_TypeMatchupMultiplier`` (lib.c:3009): 40 = neutral. ``identified``
    is for the callers that go through ``ApplyTypeChart`` (which reads Foresight)."""
    mul = 40
    for defending, m in _chart_rows(tables, attacking_type, identified):
        if defending == types[0]:
            mul = mul * m // 10
        if defending == types[1] and types[0] != types[1]:
            mul = mul * m // 10
    return mul


def _effective_speed(tables, s: RefState, side: int) -> int:
    """``BattleSystem_CompareBattlerSpeed`` (lib.c:1188), implemented modifiers only."""
    mon = _active(s, side)
    stage = s.sides[side].stages[Stat.SPE]
    if mon.ability == AbilityId.SIMPLE:  # CompareSpeed_ApplySimple, lib.c:1222
        stage = _simple(stage)
    speed = _apply_stage(mon.stats[Stat.SPE - 1], stage)  # lib.c:1225
    # lib.c:1228-1238: Swift Swim in rain, inside NO_CLOUD_NINE.
    if (
        mon.ability == AbilityId.SWIFT_SWIM
        and s.weather == Weather.RAIN
        and not _weather_suppressed(tables, s)
    ):
        speed *= 2
    if mon.status == Status.PARALYSIS:  # lib.c:1260 (Quick Feet is unsupported)
        speed //= 4
    if s.sides[side].tailwind:  # lib.c:1274
        speed *= 2
    return speed


def _item_effect(tables, mon: RefPokemon) -> int:
    return _t(tables, "item_effect", mon.item)


def _player_second(
    tables, s: RefState, priority: tuple[int, int], tie_roll: int, qc: list[int]
) -> bool:
    """``BattleSystem_CompareBattlerSpeed(player, opponent)`` != FASTER, i.e. the two
    are swapped (lib.c:1401-1445). A fainted battler goes after a living one
    (lib.c:1205-1211). ``qc``: each battler's ``speedRand % 5 == 0`` (lib.c:1278-1286)."""
    p_hp, o_hp = _active(s, 0).hp, _active(s, 1).hp
    if p_hp == 0 and o_hp:
        return True
    if p_hp and o_hp == 0:
        return False
    if priority[0] != priority[1]:
        return priority[0] < priority[1]
    speed0, speed1 = _effective_speed(tables, s, 0), _effective_speed(tables, s, 1)
    # Quick Claw (HOLD_EFFECT_SOMETIMES_PRIORITY, lib.c:1278-1286); Custap is unsupported.
    claw = [
        _item_effect(tables, _active(s, side)) == ItemId.QUICK_CLAW and qc[side] != 0
        for side in (0, 1)
    ]
    if claw[0] and claw[1]:  # lib.c:1402-1407: plain speed, no Trick Room, no Lagging Tail
        if speed0 == speed1:
            return tie_roll != 0
        return speed0 < speed1
    if claw[0] != claw[1]:  # lib.c:1408-1411: the lone activated holder goes first
        return claw[1]
    lag = [_item_effect(tables, _active(s, side)) == ItemId.LAGGING_TAIL for side in (0, 1)]
    if lag[0] and lag[1]:  # lib.c:1412-1417: the faster goes second, Trick Room ignored
        if speed0 == speed1:
            return tie_roll != 0
        return speed0 > speed1
    if lag[0] != lag[1]:  # lib.c:1418-1421: the lone holder goes second
        return lag[0]
    # Stall (lib.c:1422-1431) is unsupported.
    if speed0 == speed1:
        return tie_roll != 0  # RandNext() & 1
    if s.trick_room:
        return speed0 > speed1
    return speed0 < speed1


# --------------------------------------------------------------------------- damage


def _calc_move_damage(
    tables,
    s: RefState,
    attacker: int,
    defender: int,
    move: int,
    crit: bool,
    *,
    power_mul: int = 10,
    in_power: int = 0,
    magnitude: int = 4,
    present: int = -1,
    side_conditions: bool = True,
    field_conditions: bool = True,
) -> int:
    """``BattleSystem_CalcMoveDamage`` (lib.c:6601) times ``criticalMul``
    (``BattleScript_CalcMoveDamage``, script.c:1344). ``side_conditions`` /
    ``field_conditions`` False is the confusion self-hit's 0 masks."""
    atk, dfn = _active(s, attacker), _active(s, defender)
    attack, sp_attack = atk.stats[Stat.ATK - 1], atk.stats[Stat.SPA - 1]
    defense, sp_defense = dfn.stats[Stat.DEF - 1], dfn.stats[Stat.SPD - 1]
    attack_stage = s.sides[attacker].stages[Stat.ATK]
    sp_attack_stage = s.sides[attacker].stages[Stat.SPA]
    defense_stage = s.sides[defender].stages[Stat.DEF]
    sp_defense_stage = s.sides[defender].stages[Stat.SPD]
    level = atk.level
    a_ability = atk.ability
    d_ability = _ignorable(s, attacker, defender)

    # lib.c:6674: the input power if given, else the move's.
    power = in_power if in_power else _t(tables, "move_power", move)
    effect = _t(tables, "move_effect", move)
    if not in_power and effect == EffectId.HIDDEN_POWER:  # script.c:6007
        power = _hidden_power(atk.ivs)[1]
    elif not in_power and effect == EffectId.MAGNITUDE:  # script.c:5820
        power = MAGNITUDE_POWER[magnitude - 4]
    elif not in_power and effect == EffectId.PRESENT and 0 <= present < PRESENT_HEAL:
        power = PRESENT_POWER[present]  # script.c:5795-5800; no draw: the table power
    elif not in_power and effect == EffectId.FLAIL:  # script.c:4958
        power = _flail_power(atk.hp, atk.max_hp)
    elif not in_power and effect == EffectId.ROLLOUT:  # script.c:5661
        power *= _rollout_multiplier(s, attacker)
    move_type = _move_type(tables, s, attacker, move)  # lib.c:6680 (Normalize is unsupported)
    power = power * power_mul // 10  # lib.c:6688
    move_class = _t(tables, "move_category", move)

    # lib.c:6793 Guts.
    if a_ability == AbilityId.GUTS and atk.status != Status.NONE:
        attack = attack * 150 // 100
    # lib.c:6811 Mud Sport, :6815 Water Sport (any battler on the field).
    if move_type == Type.ELECTRIC and (s.sides[0].mud_sport or s.sides[1].mud_sport):
        power //= 2
    if move_type == Type.FIRE and (s.sides[0].water_sport or s.sides[1].water_sport):
        power //= 2
    # lib.c:6819-6832 Overgrow, Blaze, Torrent.
    pinch = atk.hp <= atk.max_hp // 3
    if move_type == Type.GRASS and a_ability == AbilityId.OVERGROW and pinch:
        power = power * 150 // 100
    if move_type == Type.FIRE and a_ability == AbilityId.BLAZE and pinch:
        power = power * 150 // 100
    if move_type == Type.WATER and a_ability == AbilityId.TORRENT and pinch:
        power = power * 150 // 100
    # lib.c:6850 Simple attacker; :6868 Simple defender (ignorable).
    if a_ability == AbilityId.SIMPLE:
        attack_stage, sp_attack_stage = _simple(attack_stage), _simple(sp_attack_stage)
    if d_ability == AbilityId.SIMPLE:
        defense_stage, sp_defense_stage = _simple(defense_stage), _simple(sp_defense_stage)
    # lib.c:6886 Unaware defender (ignorable); :6891 Unaware attacker.
    if d_ability == AbilityId.UNAWARE:
        attack_stage = sp_attack_stage = 0
    if a_ability == AbilityId.UNAWARE:
        defense_stage = sp_defense_stage = 0
    # lib.c:6901, :6907 Rivalry.
    if a_ability == AbilityId.RIVALRY and atk.gender != 2 and dfn.gender != 2:
        if atk.gender == dfn.gender:
            power = power * 125 // 100
        else:
            power = power * 75 // 100
    # lib.c:6921 NO_CLOUD_NINE gates the weather here and at lib.c:7046.
    weather = s.weather
    if not field_conditions or _weather_suppressed(tables, s):
        weather = Weather.NONE
    # lib.c:6926 sandstorm boosts a Rock-type defender's SpD.
    if weather == Weather.SANDSTORM and Type.ROCK in dfn.types:
        sp_defense = sp_defense * 15 // 10
    # lib.c:6943-6945 BATTLE_EFFECT_HALVE_DEFENSE (Selfdestruct, Explosion) halves defenseStat.
    if effect == EffectId.SELF_DESTRUCT:
        defense //= 2

    damage = 0
    if move_class in (PHYSICAL, SPECIAL):
        physical = move_class == PHYSICAL
        a_stat, a_stage = (attack, attack_stage) if physical else (sp_attack, sp_attack_stage)
        d_stat, d_stage = (defense, defense_stage) if physical else (sp_defense, sp_defense_stage)
        # lib.c:6948-6958: a crit uses the attacker's stage only when it is positive.
        if crit and a_stage <= 0:
            damage = a_stat
        else:
            damage = _apply_stage(a_stat, a_stage)
        damage *= power  # lib.c:6960
        damage *= level * 2 // 5 + 2  # lib.c:6961
        # lib.c:6963-6973: a crit uses the defender's stage only when it is negative.
        if crit and d_stage >= 0:
            divisor = d_stat
        else:
            divisor = _apply_stage(d_stat, d_stage)
        damage //= divisor  # lib.c:6975
        damage //= 50  # lib.c:6976
        # lib.c:6978 burn halves physical damage unless Guts.
        if physical and atk.status == Status.BURN and a_ability != AbilityId.GUTS:
            damage //= 2
        # lib.c:6982 Reflect, :7023 Light Screen on the defender's side, not on a crit.
        screen = s.sides[defender].reflect if physical else s.sides[defender].light_screen
        if side_conditions and screen and not crit and effect != EffectId.BRICK_BREAK:
            damage //= 2
    # lib.c:7047 rain, :7062 sun (fog does nothing here).
    if weather == Weather.RAIN:
        if move_type == Type.FIRE:
            damage //= 2
        elif move_type == Type.WATER:
            damage = damage * 15 // 10
    if weather == Weather.SUN:
        if move_type == Type.FIRE:
            damage = damage * 15 // 10
        elif move_type == Type.WATER:
            damage //= 2
    damage += 2  # lib.c:7078
    if crit:
        damage *= 2  # script.c:1344, criticalMul (Sniper is unsupported)
    return damage


def _variance(damage: int, roll: int) -> int:
    """``BattleSystem_CalcDamageVariance`` (lib.c:7081): ``* (100 - rand % 16) / 100``,
    at least 1; ``roll`` is ``15 - rand % 16``."""
    if damage == 0:
        return 0
    return max(damage * (85 + roll) // 100, 1)


def _apply_type_chart(tables, s: RefState, attacker: int, defender: int, move: int, damage: int,
                      flags: set[str] | None = None):  # fmt: skip
    """``BattleSystem_ApplyTypeChart`` (lib.c:2560): STAB, then every chart row for
    the move's type in table order, each matching defender type through
    ``ApplyTypeMultiplier`` (lib.c:7542). Struggle returns early (lib.c:2573). ``flags``
    receives the ``moveStatusFlags`` effectiveness bits."""
    if flags is None:
        flags = set()
    if move == MOVE_STRUGGLE:
        return damage
    move_type = _move_type(tables, s, attacker, move)
    atk_types = _active(s, attacker).types
    def_types = _active(s, defender).types
    if move_type in atk_types:  # lib.c:2592-2597
        if _active(s, attacker).ability == AbilityId.ADAPTABILITY:
            damage *= 2
        else:
            damage = damage * 15 // 10
    identified = bool(s.sides[defender].foresight)  # lib.c:2615
    update = _t(tables, "move_power", move) != 0  # ApplyTypeMultiplier's `update`
    # flags: MOVE_STATUS_SUPER_EFFECTIVE / _NOT_VERY_EFFECTIVE
    for defending, mul in _chart_rows(tables, move_type, identified):  # lib.c:2611-2646
        if defending == def_types[0]:
            damage = _apply_type_multiplier(damage, mul, update, flags)
        if defending == def_types[1] and def_types[0] != def_types[1]:
            damage = _apply_type_multiplier(damage, mul, update, flags)
    # lib.c:2657-2660: Solid Rock (ignorable) on a super-effective hit.
    if "super" in flags and update and _ignorable(s, attacker, defender) == AbilityId.SOLID_ROCK:
        damage = _divide(damage * 3, 4)
    return damage


def _apply_type_multiplier(damage: int, mul: int, update: bool, flags: set[str]) -> int:
    """``ApplyTypeMultiplier`` (lib.c:7542): the damage through ``BattleSystem_Divide``
    when non-zero, and the effectiveness flags: an immunity clears both, a x2 entry
    clears a set "not very effective" or else sets "super", and the other way round."""
    if damage:
        damage = _divide(damage * mul, 10)
    if mul == 0:
        flags.clear()
    elif update and mul in (5, 20):
        mine, other = ("super", "not_very") if mul == 20 else ("not_very", "super")
        if other in flags:
            flags.discard(other)
        else:
            flags.add(mine)
    return damage


def _move_damage(
    tables, s: RefState, attacker: int, move: int, crit: bool, roll: int, power_mul=10,
    magnitude: int = 4, present: int = -1,
):  # fmt: skip
    """One hit's damage: the move script's ``CalcDamage`` and variance, then the
    controller's type chart (``TRY_MOVE_STATE_CHECK_TYPE_CHART``, bcp.c:3272). A
    status move never calculates damage; ``CheckTypeChart`` needs base power
    (bcp.c:2354). Present's heal path jumps past ``CalcDamage`` (effect_script_0122.s)."""
    if _t(tables, "move_category", move) == STATUS_CLASS or _t(tables, "move_power", move) == 0:
        return 0
    if _t(tables, "move_effect", move) == EffectId.PRESENT and present == PRESENT_HEAL:
        return 0
    defender = 1 - attacker
    if _t(tables, "move_effect", move) in (EffectId.LEVEL_DAMAGE, EffectId.BIDE):
        return _fixed_damage(tables, s, attacker, move)
    damage = _calc_move_damage(
        tables,
        s,
        attacker,
        defender,
        move,
        crit,
        power_mul=power_mul,
        magnitude=magnitude,
        present=present,
    )
    damage = _variance(damage, roll)
    return _apply_type_chart(tables, s, attacker, defender, move, damage)


def ref_damage_table(tables, state: RefState, atk_side: int, move: int, crit: bool) -> list[int]:
    """Damage of ``move`` by ``atk_side``'s active Pokemon for rolls 0..15."""
    return [_move_damage(tables, state, atk_side, move, bool(crit), roll) for roll in range(16)]


# --------------------------------------------------------------------------- faints


def _cleanup_fainted(turn: _Turn, s: RefState, side: int) -> None:
    """``BattleSystem_CleanupFaintedMon`` (lib.c:2156): the fainted battler's stat
    stages, ``statusVolatile``, ``moveEffectsMask`` and ``moveEffectsData`` are
    cleared, ``fakeOutTurnNumber = totalTurns + 1`` (lib.c:2195), ``moveHit`` and
    ``movePrevByBattler`` reset (lib.c:2199); what it is the source of ends: the Bind it put
    on a foe (lib.c:2179) and the foe's infatuation (:2174-2176)."""
    _restore_types(turn.tables, s, side)
    _reset_battler(s.sides[side])
    s.sides[side].fake_out_turn = turn.entry_turn
    _source_leaves(s, side)


def _reset_battler(sd: RefSide) -> None:
    """The volatile state of a side's active battler back to a fresh battler's."""
    sd.stages = [0] * len(sd.stages)
    for name in (
        "confusion",
        "leech_seed",
        "substitute_hp",
        "taunt",
        "trapped",
        "trap_move",
        "flinched",
        "choice_lock",
        "last_move",
        "protect_count",
        "rollout_count",
        "charge_turn",
        "recharge",
        "pursuit_target",
        "water_sport",
        "move_hit",
        "foresight",
        "disabled_move",
        "disable_turns",
        "locked_move",
        "bide_turns",
        "bide_damage",
        "curled",
        "mud_sport",
        "aqua_ring",
        "infatuated",
    ):
        setattr(sd, name, 0)


def _unlock(sd: RefSide) -> None:
    """``Battler_UnlockMoveChoice`` (lib.c:1834): the move lock, Bide's turns and stored
    damage (0 outside a Bide) and the Rollout count."""
    sd.locked_move = sd.bide_turns = sd.bide_damage = sd.rollout_count = 0


def _release_bind(sd: RefSide) -> None:
    sd.trapped = 0
    sd.trap_move = 0


def _source_leaves(s: RefState, side: int) -> None:
    """``side``'s battler leaves (UpdateAfterSwitch lib.c:2088-2098, CleanupFaintedMon
    :2168-2186): the Bind it put on the foe and the foe's ATTRACT bit sourced from it end."""
    _release_bind(s.sides[1 - side])
    s.sides[1 - side].infatuated = 0


def _faint_check(turn: _Turn, s: RefState) -> None:
    """``BattleControllerPlayer_AnyFainted`` (bcp.c:4409) then ``CheckBattleOver``:
    each flagged Pokemon faints (``subscript_faint_mon.s`` -> ``PlayFaintAnimation``,
    script.c:1795, which ends its action and cleans the battler up), then the battle
    ends when a side has no Pokemon with HP left: a whiteout is a loss, a simultaneous last
    opponent faint too (a draw, ``BATTLE_RESULT_DRAW`` plays as a loss: battle.h:78,
    bcp.c:4015-4016, ruling 2); a player faint otherwise only records a death. A fainted Pokemon keeps no status
    condition (battle_controller.c:1482-1486)."""
    for side, slot in turn.fainted:
        mon = s.sides[side].party[slot]
        mon.status = Status.NONE
        mon.status_counter = 0
    if s.over:
        turn.fainted.clear()
        return
    player_fainted = 0
    for side, slot in turn.fainted:
        turn.done[side] = True  # script.c:1801, battlerActions = MOVE_END
        if s.sides[side].active == slot:
            _cleanup_fainted(turn, s, side)
        if side == 0:
            player_fainted += 1
            s.deaths[slot] = 1
    turn.fainted.clear()
    s.player_faints += player_fainted
    if all(m.species == 0 or m.hp == 0 for m in s.sides[0].party):
        s.over, s.winner = 1, WINNER_OPPONENT
    elif all(m.species == 0 or m.hp == 0 for m in s.sides[1].party):
        s.over, s.winner = 1, WINNER_PLAYER


# --------------------------------------------------------------------------- items, switch-in


def _cure(s: RefState, side: int, effect: int) -> bool:
    """The cure of a status berry or White Herb ``effect`` on ``side``'s active Pokemon,
    as ``BattleSystem_TriggerHeldItem`` (lib.c:4608-4700, :4875-4887) and
    ``BattleSystem_PluckBerry`` (lib.c:5577-5675) choose it, and the restore subscript it
    runs (``subscript_held_item_{slp,psn,cnf,multi,...}_restore.s``). Returns whether it
    did anything."""
    mon, sd = _active(s, side), s.sides[side]
    cured = False
    if effect == ItemId.CHESTO_BERRY and mon.status == Status.SLEEP:
        cured = True
    elif effect == ItemId.PECHA_BERRY and mon.status in (Status.POISON, Status.TOXIC):
        cured = True  # MON_CONDITION_ANY_POISON
    elif effect == ItemId.LUM_BERRY and (mon.status != Status.NONE or sd.confusion):
        cured = True
        sd.confusion = 0
    if cured:
        mon.status = Status.NONE  # sleep turns and the toxic counter are status bits
        mon.status_counter = 0
        return True
    if effect == ItemId.WHITE_HERB and any(x < 0 for x in sd.stages):
        # lib.c:4875-4887: every stage below DEFAULT_STAT_STAGE back to it, in C.
        sd.stages = [max(x, 0) for x in sd.stages]
        return True
    return False


def _held_item(turn: _Turn, s: RefState, side: int, keep: bool = False) -> bool:
    """``BattleSystem_TriggerHeldItem`` (lib.c:4576) / ``..._OnStatus`` (lib.c:4967) for
    ``side``'s active Pokemon with HP left. Oran Berry and Berry Juice
    (``HOLD_EFFECT_HP_RESTORE``) heal their power, Sitrus Berry (``HP_PCT_RESTORE``)
    ``Divide(maxHP * power, 100)``, when ``curHP <= maxHP / 2``
    (``subscript_held_item_hp_restore.s``, capped); Chesto, Pecha, Lum and White Herb cure
    (``_cure``). Every subscript ends in ``BATTLE_SUBSCRIPT_PLUCK_CHECK``, which removes
    the item (``BtlCmd_RemoveItem`` keeps it as the recycle item) unless the attacker's
    pluck flag is set (``keep``). Returns whether the item triggered."""
    tables = turn.tables
    mon = _active(s, side)
    if mon.hp == 0:
        return False
    effect = _item_effect(tables, mon)
    power = _t(tables, "item_param", mon.item, 0)
    if effect in (ItemId.ORAN_BERRY, ItemId.BERRY_JUICE, ItemId.SITRUS_BERRY):
        if mon.hp > mon.max_hp // 2:
            return False
        flat = effect != ItemId.SITRUS_BERRY
        _gain_hp(s, side, power if flat else _divide(mon.max_hp * power, 100))
    elif not _cure(s, side, effect):
        return False
    if not keep:
        mon.item_consumed = mon.item
        mon.item = 0
    return True


def _held_item_hp(turn: _Turn, s: RefState, side: int) -> None:
    """``_held_item`` at a site no pluck flag can reach (switch-in, end of turn)."""
    _held_item(turn, s, side)


def _focus_sash(turn: _Turn, s: RefState, side: int, damage: int, armed: bool) -> int:
    """The HP ``side``'s active Pokemon loses to ``damage``: ``armed`` (a Focus Sash
    activation) and ``curHP + damage <= 0`` leave 1 HP (bcp.c:3396-3404,
    ``BtlCmd_CheckHoldOnWith1HP`` script.c:9119-9122), and
    ``subscript_move_followup_message.s:12-19`` removes the item while it still is a
    Focus Sash."""
    mon = _active(s, side)
    if not (armed and damage > 0 and mon.hp - damage <= 0):
        return damage
    if _item_effect(turn.tables, mon) == ItemId.FOCUS_SASH:
        mon.item_consumed = mon.item
        mon.item = 0
    return mon.hp - 1


def _sash_at_full_hp(turn: _Turn, s: RefState, side: int) -> bool:
    mon = _active(s, side)
    return _item_effect(turn.tables, mon) == ItemId.FOCUS_SASH and mon.hp == mon.max_hp


def _resist_berry(
    turn: _Turn, s: RefState, side: int, move_type: int, super_effective: bool
) -> bool:
    """``subscript_type_resist_berry.s`` (``subscript_update_hp.s:9``) on ``side``'s active
    Pokemon, the battler whose HP changes: a Chilan Berry (``HOLD_EFFECT_WEAKEN_NORMAL``,
    ``RESIST_ANY_NORMAL``) halves any hit whose type (``calcTemp``) is Normal (:_128), a
    Rindo Berry a super-effective Grass hit (:_156); the berry is then removed. Returns
    whether it triggered (the caller halves ``hpCalcTemp``)."""
    mon = _active(s, side)
    if _item_effect(turn.tables, mon) != ItemId.RESIST_BERRY:
        return False
    resisted = _t(turn.tables, "item_param", mon.item, 1)
    if resisted == RESIST_ANY_NORMAL:
        triggered = move_type == Type.NORMAL
    else:
        triggered = super_effective and move_type == resisted
    if triggered:
        mon.item_consumed = mon.item
        mon.item = 0
    return triggered


def _hit_resist_berry(turn: _Turn, s: RefState, attacker: int, move: int) -> bool:
    """``_resist_berry`` for a move hit on the foe of ``attacker``. :5 skips a move whose
    script set ``SYSCTL_IGNORE_TYPE_CHECKS`` / ``_IGNORE_IMMUNITIES`` (Night Shade, Seismic
    Toss: effect_script_0087.s; OHKO moves, :6, are unsupported). ``calcTemp`` is the move's
    type (Hidden Power's ``moveType``; Normalize is unsupported). The super-effective flag
    comes from ``CheckTypeChart`` (bcp.c:2350-2358), which skips ``RANGE_USER`` (Bide) and
    power-0 moves."""
    tables = turn.tables
    effect = _t(tables, "move_effect", move)
    if effect in (EffectId.LEVEL_DAMAGE, EffectId.FORESIGHT):
        return False
    flags: set[str] = set()
    if effect != EffectId.BIDE and _t(tables, "move_power", move) != 0:
        _apply_type_chart(tables, s, attacker, 1 - attacker, move, 0, flags)
    move_type = _move_type(tables, s, attacker, move)
    return _resist_berry(turn, s, 1 - attacker, move_type, "super" in flags)


def _big_root(turn: _Turn, s: RefState, side: int, heal: int) -> int:
    """``subscript_drain_half_damage_dealt.s:_011`` / ``subscript_leech_seed_effect.s``:
    with a Big Root (``HOLD_EFFECT_LEECH_BOOST``) the heal becomes ``heal * (param + 100)
    / 100`` (param 30), no minimum."""
    mon = _active(s, side)
    if _item_effect(turn.tables, mon) != ItemId.BIG_ROOT:
        return heal
    return heal * (_t(turn.tables, "item_param", mon.item, 0) + 100) // 100


def _stealth_rock_damage(turn: _Turn, s: RefState, side: int) -> None:
    """``subscript_hazards_check.s`` -> ``BtlCmd_CheckStealthRock`` (script.c:8302)."""
    mon = _active(s, side)
    if not s.sides[side].stealth_rock or mon.hp == 0:
        return
    mul = _matchup_x40(turn.tables, Type.ROCK, mon.types)
    if mul == 0:  # TYPE_MULTI_IMMUNE
        return
    _lose_hp(turn, s, side, _divide(mon.max_hp, STEALTH_ROCK_DIVISOR[mul]))


def _recover_status_by_ability(s: RefState, side: int) -> None:
    """``BattleSystem_RecoverStatusByAbility`` (lib.c:4378) for ``side``'s battler: an
    asleep Insomnia holder (:4404) or a paralyzed Limber holder (:4397) with HP left is
    cured, and subscript_ability_forbids_status.s tags its ability (BattleAI_SetAbility,
    script.c:11806). Run at each move end for the attacker then the defender
    (bcp.c:3965-3967) and on a switch-in (lib.c:4135-4145)."""
    mon = _active(s, side)
    forbidden = {AbilityId.INSOMNIA: Status.SLEEP, AbilityId.LIMBER: Status.PARALYSIS}
    if forbidden.get(mon.ability) == mon.status and mon.hp:
        mon.status = Status.NONE
        mon.status_counter = 0
        if side == 0:
            s.ai_ability_known[s.sides[0].active] = 1


def _turn_end_ability(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_ABILITY`` (bcp.c:1355): Shed Skin (lib.c:3567-3590) cures a
    status on ``RandNext() % 10 < 3`` (``outcome.shed_skin``) while the holder has HP."""
    mon = _active(s, side)
    if (
        mon.ability == AbilityId.SHED_SKIN
        and mon.status != Status.NONE
        and mon.hp
        and turn.outcome.shed_skin[side]
    ):
        mon.status = Status.NONE
        mon.status_counter = 0
        # subscript_ability_restore_status.s:8 tags the holder (script.c:11806).
        if side == 0:
            s.ai_ability_known[s.sides[0].active] = 1


def _switch_in_effects(turn: _Turn, s: RefState, side: int) -> None:
    """``BattleSystem_TriggerEffectOnSwitch`` (lib.c:3649) after ``side``'s battler
    entered: ``SWITCH_IN_CHECK_STATE_INTIMIDATE`` (lib.c:3795; only the newcomer has
    not announced its Intimidate) then ``SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS``
    (lib.c:4151) for every battler in ``monSpeedOrder``; each battler's check only
    touches that battler, so the order does not matter. (At a double KO the player's
    newcomer enters against a fainted foe and announces nothing here;
    ``playout.replacement_pass`` fires its Intimidate once the opponent's newcomer is in.)"""
    mon, foe = _active(s, side), _active(s, 1 - side)
    # subscript_intimidate.s: ATTACK_DOWN_1_STAGE on a living foe (Clear Body, Hyper
    # Cutter, White Smoke, Mist and Substitute are unsupported). Its "cuts" message is
    # built in C (BtlCmd_ChangeStatStage, script.c:2956): the AI learns nothing.
    if mon.hp and mon.ability == AbilityId.INTIMIDATE and foe.hp:
        _change_stage(s, 1 - side, Stat.ATK, -1)
    # SWITCH_IN_CHECK_STATE_MOLD_BREAKER (lib.c:4075): subscript_mold_breaker.s tags the
    # living newcomer's ability, BattleAI_SetAbility (script.c:11806).
    if side == 0 and mon.hp and mon.ability == AbilityId.MOLD_BREAKER:
        s.ai_ability_known[s.sides[0].active] = 1
    _recover_status_by_ability(s, side)
    for battler in (0, 1):
        _held_item_hp(turn, s, battler)


def _enter(turn: _Turn, s: RefState, side: int, slot: int) -> None:
    """``SwitchAndUpdateMon`` -> ``BattleSystem_InitBattleMon`` (lib.c:70): a fresh
    battler (stages neutral, no volatiles; no Baton Pass in M1), and
    ``BattleSystem_UpdateAfterSwitch`` (lib.c:2094) sets ``fakeOutTurnNumber =
    totalTurns + 1`` (lib.c:2117), clears ``moveHit`` (lib.c:2121) and ends what the outgoing
    battler is the source of (the Bind on the foe, the foe's infatuation, lib.c:2088-2098).
    The caller then runs ``PokemonSendOut`` (``_send_out_exp``) and the entry hazards
    (``_stealth_rock_damage``), in that order."""
    sd = s.sides[side]
    _restore_types(turn.tables, s, side)  # the outgoing battler's Camouflage ends
    sd.active = slot
    _reset_battler(sd)
    sd.fake_out_turn = turn.entry_turn
    # lib.c:2151-2152: the AI forgets the battler's moves and ability (it only tracks
    # the player's).
    if side == 0:
        s.ai_seen_moves[slot] = [0] * len(s.ai_seen_moves[slot])
        s.ai_ability_known[slot] = 0
    _source_leaves(s, side)


def _send_out_exp(s: RefState, side: int) -> None:
    """``BtlCmd_PokemonSendOut`` (script.c:543-635) for ``side``'s newcomer: a player
    send-out flags the enemy (:571-572, :592-593, :607-608, :622-623), an enemy send-out
    clears the mask, then flags (:580-581, :595-596, :610-611, :625-626).
    ``BattleSystem_ClearSideExpGain`` (lib.c:1456-1459) sets ``sideGetExpMask`` to 0;
    ``BattleSystem_FlagBattlerExpGain`` (lib.c:1461-1479) ORs in
    ``selectedPartySlot[PLAYER_1]``'s bit when the enemy's ``curHP`` is nonzero (``PLAYER_2``
    and ``ENEMY_2`` are in ``battlersSwitchingMask`` in singles, lib.c:2031-2033)."""
    if side == 1:
        s.exp_mask = 0
    if _active(s, 1).hp:
        s.exp_mask |= 1 << s.sides[0].active


# --------------------------------------------------------------------------- the move


def _move_target(tables, move: int) -> int:
    return _t(tables, "move_target", move)


def _curse_on_user(tables, move: int) -> bool:
    """A non-Ghost user's Curse (coverage refuses a Ghost-type one) targets its user although
    its data range is RANGE_SINGLE_TARGET: ``BattleControllerPlayer_MustSelectTarget``
    (bcp.c:4344-4346) gives it ``RANGE_USER``, so the chosen target is the user (:4363) and
    ``BattleSystem_Defender`` (lib.c:1709-1718) returns it: like a ``TARGETS_USER`` move."""
    return _t(tables, "move_effect", move) == EffectId.CURSE


def _targets_user(tables, move: int) -> bool:
    return _move_target(tables, move) in TARGETS_USER or _curse_on_user(tables, move)


def _defender_always(tables, move: int) -> bool:
    return _move_target(tables, move) in DEFENDER_ALWAYS or _curse_on_user(tables, move)


def _protected(turn: _Turn, s: RefState, side: int, move: int, unleash: bool) -> bool:
    """``BattleControllerPlayer_CheckMoveHitOverrides`` (bcp.c:3001-3014): the defender,
    ``side``'s foe, is protecting, the move has ``MOVE_FLAG_CAN_PROTECT``, the foe is its
    defender (not a SELF_DEFENDER range, not a non-Ghost Curse), and a multi-turn move
    (``Move_IsMultiTurn``, lib.c:2975: Bide here) is on its last turn (the unleash)."""
    tables = turn.tables
    if not turn.protecting[1 - side]:
        return False
    if not _t(tables, "move_flags", move) & MoveFlag.PROTECT:
        return False
    if _t(tables, "move_effect", move) == EffectId.BIDE:
        return unleash
    return _move_target(tables, move) not in SELF_DEFENDER and not _curse_on_user(tables, move)


def _can_have_status(tables, s: RefState, side: int, status: int) -> bool:
    """No status yet and no immunity: ``subscript_burn.s`` (Fire), ``subscript_poison.s``
    (Poison, Steel), ``subscript_freeze.s`` (Ice, or sun without a weather lock: the
    ``CheckIgnoreWeather`` at :6 skips the sun check at :7, battle_script.c:9195-9206)."""
    mon = _active(s, side)
    if mon.status != Status.NONE:
        return False
    if status == Status.BURN and Type.FIRE in mon.types:
        return False
    if status in (Status.POISON, Status.TOXIC) and (
        Type.POISON in mon.types or Type.STEEL in mon.types
    ):
        return False
    sun = s.weather == Weather.SUN and not _weather_suppressed(tables, s)
    frozen_out = Type.ICE in mon.types or sun
    return not (status == Status.FREEZE and frozen_out)


def _inflict(turn: _Turn, s: RefState, side: int, target: int, status: int,
             direct: bool) -> None:  # fmt: skip
    """The status subscripts (``subscript_fall_asleep.s`` ... ``subscript_paralyze.s``)
    on a living target; sleep lasts ``Random 3, 2`` turns (``outcome.duration`` of the
    inflicting side). ``direct``: a status move (``SIDE_EFFECT_TYPE_DIRECT``), not a
    secondary effect (``_INDIRECT``)."""
    mon = _active(s, target)
    # subscript_fall_asleep.s:_055/_202/_221: an ignorable Insomnia stops the sleep before
    # the other checks and its "stayed awake" message tags the ability.
    if mon.hp and status == Status.SLEEP and _ignorable(s, side, target) == AbilityId.INSOMNIA:
        if target == 0:
            s.ai_ability_known[s.sides[0].active] = 1
        return
    # subscript_paralyze.s:5/_170: an ignorable Limber stops the paralysis first; its
    # "prevents paralysis" message (TAG_NICKNAME_ABILITY) is shown for a direct effect only.
    if mon.hp and status == Status.PARALYSIS and _ignorable(s, side, target) == AbilityId.LIMBER:
        if target == 0 and direct:
            s.ai_ability_known[s.sides[0].active] = 1
        return
    if mon.hp and _can_have_status(turn.tables, s, target, status):
        mon.status = status
        mon.status_counter = turn.outcome.duration[side] if status == Status.SLEEP else 0
        # subscript_fall_asleep.s:_176 and subscript_freeze.s:_073 unlock the new sleeper
        # or ice cube.
        if status in (Status.SLEEP, Status.FREEZE):
            _unlock(s.sides[target])


def _pluck_berry(turn: _Turn, s: RefState, side: int) -> None:
    """``BtlCmd_TryPluck`` (script.c:8116) -> ``BattleSystem_PluckBerry`` (lib.c:5539), by
    the item's pluck effect (res/items/data/*.json): the user eats the foe's Oran
    (``PLUCK_EFFECT_HP_RESTORE``) / Sitrus (``_HP_PCT_RESTORE``) Berry, healing (capped)
    unless at full HP, or its Chesto / Pecha / Lum (``_SLP_RESTORE`` / ``_PSN_RESTORE`` /
    ``_ALL_RESTORE``), curing as the held berry would, or its Rindo / Chilan
    (``PLUCK_EFFECT_NONE`` berries, no effect); the berry is removed
    (``RemoveItem BTLSCR_DEFENDER``) even when the foe fainted. Berry Juice and the other
    implemented items have ``PLUCK_EFFECT_NONE`` and are no berries (lib.c:5833): they
    stay. ``SELF_TURN_FLAG_PLUCK_BERRY`` (lib.c:5848) stays set when the berry did
    nothing for the eater (no restore subscript ran its PLUCK_CHECK)."""
    tables = turn.tables
    user, foe = _active(s, side), _active(s, 1 - side)
    effect = _item_effect(tables, foe)
    power = _t(tables, "item_param", foe.item, 0)
    if effect == ItemId.ORAN_BERRY:
        heal = power
    elif effect == ItemId.SITRUS_BERRY:
        heal = _divide(user.max_hp * power, 100)
    # Rindo, Chilan (RESIST_BERRY): PLUCK_EFFECT_NONE berries, eaten with no effect (:5833).
    elif effect in (ItemId.CHESTO_BERRY, ItemId.PECHA_BERRY, ItemId.LUM_BERRY, ItemId.RESIST_BERRY):
        heal = 0
    else:
        return
    foe.item_consumed = foe.item
    foe.item = 0
    if heal:
        did = user.hp != user.max_hp
        if did:
            _gain_hp(s, side, heal)
    else:
        did = _cure(s, side, effect)
    turn.pluck_flag = not did


def _move_effect(turn: _Turn, s: RefState, side: int, move: int, hit: int, landed: bool,
                 effective: bool, dealt: int, target_moved: bool) -> None:  # fmt: skip
    """The move's own effect for one hit. Status and stat moves are primary effects
    (``BattleSystem_TriggerPrimaryEffect``, lib.c:1481); a damaging move's effect is
    a secondary one (``BattleSystem_TriggerSecondaryEffect``, lib.c:1519): a
    probabilistic one rolls ``outcome.secondary``, an ``ON_HIT`` one (effect chance 0)
    always applies."""
    tables, o = turn.tables, turn.outcome
    target = 1 - side
    effect = _t(tables, "move_effect", move)
    p0, p1 = _t(tables, "move_param", move, 0), _t(tables, "move_param", move, 1)
    chance = _t(tables, "move_effect_chance", move)
    certain = chance == 0
    proc = certain or o.secondary[side][hit] != 0
    target_alive = _active(s, target).hp > 0
    seen = _ignorable(s, side, target)

    if effect == EffectId.STAT_SELF and landed:
        _change_stage(s, side, p0, p1)
        if _t(tables, "move_param", move, 2) == 1:  # Defense Curl (effect_script_0156.s)
            s.sides[side].curled = 1
    elif effect == EffectId.STAT_FOE and landed and target_alive:
        if not (seen == AbilityId.KEEN_EYE and p0 == Stat.ACC and p1 < 0):
            _change_stage(s, target, p0, p1)
    elif effect == EffectId.SECONDARY_STAT_SELF and effective and proc:
        if certain or _active(s, side).hp > 0:
            _change_stage(s, side, p0, p1)
    elif effect == EffectId.SECONDARY_STAT_FOE and effective and proc:
        keen_eye = seen == AbilityId.KEEN_EYE and p0 == Stat.ACC and p1 < 0
        if (certain or target_alive) and not keen_eye:
            _change_stage(s, target, p0, p1)
    elif effect in (EffectId.STATUS_FOE, EffectId.SECONDARY_STATUS) and effective:
        if effect == EffectId.STATUS_FOE or proc:
            _inflict(turn, s, side, target, p0, effect == EffectId.STATUS_FOE)
    elif effect == EffectId.TRI_ATTACK and effective and proc:
        # effect_script_0036.s: Random 2, 3 (script.c:3200-3210) picks burn, freeze or
        # paralysis, an indirect TO_DEFENDER effect at the move's chance (lib.c:1570-1586:
        # the roll, then a living defender) through subscript_burn/freeze/paralyze.s.
        _inflict(turn, s, side, target, TRI_ATTACK_STATUSES[o.tri_attack[side]], False)
    elif effect == EffectId.CAMOUFLAGE and landed:
        # effect_script_0213.s -> BtlCmd_TryCamouflage (script.c:7997-8022): fails for
        # Multitype or when either type already is the terrain's; else both become it.
        user, row = _active(s, side), _terrain_row(s)
        if row is not None and _decomp_ability(tables, user) != ABILITY_MULTITYPE:
            new = int(TERRAIN_TYPES[row])
            if new not in user.types:
                user.types = [new] * len(user.types)
    elif effect == EffectId.FANG_PARALYZE_FLINCH and effective:
        # effect_script_0275.s -> subscript_paralyze_or_flinch.s: CheckEffectActivation
        # (the roll, then the target's HP; script.c:8377-8378) and the indirect paralysis,
        # then a second CheckEffectActivation and subscript_flinch_mon.s (a target that has
        # not moved, no Inner Focus; its message is for Fake Out or a direct flinch only).
        if proc:
            _inflict(turn, s, side, target, p0, False)
        alive = _active(s, target).hp > 0
        if o.fang_flinch[side] and alive and not target_moved and seen != AbilityId.INNER_FOCUS:
            s.sides[target].flinched = 1
    elif effect in (EffectId.FLINCH, EffectId.FAKE_OUT) and effective and proc:
        # subscript_flinch_mon.s: a living target that has not moved yet; Inner Focus,
        # whose message (:18-23, TAG_NICKNAME_ABILITY) is shown for Fake Out only.
        if target_alive and not target_moved:
            if seen != AbilityId.INNER_FOCUS:
                s.sides[target].flinched = 1
            elif move == MOVE_FAKE_OUT and target == 0:
                s.ai_ability_known[s.sides[0].active] = 1
    elif effect == EffectId.RAISE_ALL_STATS_HIT and effective and proc:
        # effect_script_0140.s: indirect TO_ATTACKER at the move's chance; the recipient,
        # the user, must have HP (lib.c:1570-1586). subscript_boost_all_stats.s.
        if _active(s, side).hp > 0:
            for stat in (Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD):
                _change_stage(s, side, stat, 1)
    elif effect == EffectId.CURSE and landed and _active(s, side).hp > 0:
        # effect_script_0109.s (non-Ghost user) -> subscript_curse_normal.s: three
        # ChangeStatStage calls, each skipped on its own at the cap.
        _change_stage(s, side, Stat.SPE, -1)
        _change_stage(s, side, Stat.ATK, 1)
        _change_stage(s, side, Stat.DEF, 1)
    elif effect == EffectId.RECOVER and landed:
        # subscript_recover_half_max_hp.s: Divide(maxHP, 2); subscript_recover_hp.s:6:
        # "HP is full!" at full HP, else the HP update (capped).
        user = _active(s, side)
        if user.hp != user.max_hp:
            _gain_hp(s, side, _divide(user.max_hp, 2))
    elif effect == EffectId.PRESENT and landed and o.present[side] == PRESENT_HEAL:
        # effect_script_0122.s _004: a direct ON_HIT effect under SYSCTL_IGNORE_IMMUNITIES
        # (no type chart: a Ghost is healed). subscript_present_heal.s ->
        # subscript_recover_hp.s on the target: "HP is full!" at full HP, else
        # Divide(maxHP, 4), capped (script.c:5802).
        foe = _active(s, target)
        if foe.hp != foe.max_hp:
            _gain_hp(s, target, _divide(foe.max_hp, 4))
    elif effect == EffectId.ATTRACT and effective:
        # effect_script_0120.s: a direct TO_DEFENDER effect on a living defender ->
        # subscript_infatuate.s -> BtlCmd_TryAttract (script.c:5727-5742).
        if target_alive and _attractable(s, target, side):
            s.sides[target].infatuated = 1
    elif effect == EffectId.DRAIN and effective and dealt > 0:
        # subscript_drain_half_damage_dealt.s: Divide(hitDamage * percent, 100), then
        # Big Root.
        _gain_hp(s, side, _big_root(turn, s, side, _divide(dealt * p0, 100)))
    elif effect == EffectId.RECOIL and effective and dealt > 0:
        # subscript_recoil_1_{2,4}.s; Rock Head prevents it.
        if _active(s, side).ability != AbilityId.ROCK_HEAD:
            _lose_hp(turn, s, side, _divide(dealt * p0, 100))
    elif effect == EffectId.STEALTH_ROCK and landed:
        # effect_script_0266.s: fails when the foe's side already has it.
        if not s.sides[target].stealth_rock:
            s.sides[target].stealth_rock = 1
    elif effect == EffectId.TRAP and effective and proc:
        # subscript_bind_start.s (CHECK_HP_AND_SUBSTITUTE): Random 3, 3 turns.
        if target_alive and not s.sides[target].trapped:
            s.sides[target].trapped = o.duration[side]
            s.sides[target].trap_move = move
    elif effect == EffectId.TAUNT and effective:
        # subscript_taunt_start.s: fails on a taunted target; Random 2, 3 turns.
        if target_alive and not s.sides[target].taunt:
            s.sides[target].taunt = o.duration[side]
    elif effect == EffectId.DISABLE and effective:
        # effect_script_0086.s / BtlCmd_TryDisable (script.c:4581): nothing disabled yet and
        # the target's movePrevByBattler is one of its moves with PP left; RandNext() % 4 + 3.
        victim, last = s.sides[target], s.sides[target].last_move
        mon_t = _active(s, target)
        has_pp = last in mon_t.moves and mon_t.pp[mon_t.moves.index(last)] > 0
        if not victim.disabled_move and last and has_pp:
            victim.disabled_move = last
            victim.disable_turns = o.duration[side]
    elif effect == EffectId.CONFUSE_FOE and effective:
        # subscript_confuse.s: fails on a confused target; Random 3, 2 turns.
        if target_alive and not s.sides[target].confusion:
            s.sides[target].confusion = o.duration[side]
    elif effect == EffectId.SECONDARY_CONFUSE and effective and proc:
        # effect_script_0076.s: a probabilistic effect, cleared for a fainted target
        # (lib.c:1556); subscript_confuse.s fails on a confused target; Random 3, 2 turns.
        if target_alive and not s.sides[target].confusion:
            s.sides[target].confusion = o.duration[side]
    elif effect == EffectId.MUD_SPORT and landed and not s.sides[side].mud_sport:
        # effect_script_0201.s: fails when the user already has it.
        s.sides[side].mud_sport = 1
    elif effect == EffectId.AQUA_RING and landed and not s.sides[side].aqua_ring:
        # effect_script_0251.s: fails when the user already has it.
        s.sides[side].aqua_ring = 1
    elif effect in (EffectId.LIGHT_SCREEN, EffectId.REFLECT) and landed:
        # BtlCmd_TryLightScreen / TryReflect (script.c:4214, :4257): fail when already up,
        # else NUM_SCREEN_TURNS (Light Clay is unmapped, so coverage refuses it).
        name = "light_screen" if effect == EffectId.LIGHT_SCREEN else "reflect"
        if not getattr(s.sides[side], name):
            setattr(s.sides[side], name, 5)
    elif effect == EffectId.BRICK_BREAK and landed:
        # lib.c:1481-1510: applies on an immune target, not on a miss; script.c:6550.
        s.sides[target].reflect = s.sides[target].light_screen = 0
    elif effect == EffectId.EAT_BERRY and effective:
        # effect_script_0224.s / subscript_pluck.s: an ON_HIT effect, no HP check.
        _pluck_berry(turn, s, side)
    elif effect == EffectId.LEECH_SEED and effective:
        # subscript_leech_seed_start.s: Grass types and seeded targets are immune.
        grass = Type.GRASS in _active(s, target).types
        if target_alive and not grass and not s.sides[target].leech_seed:
            s.sides[target].leech_seed = 1
    elif effect == EffectId.FORESIGHT and landed:
        # effect_script_0113.s / subscript_foresight.s: an ON_HIT indirect effect on a
        # living target, no failure check.
        s.sides[target].foresight = 1
    elif effect == EffectId.WATER_SPORT and landed and not s.sides[side].water_sport:
        # effect_script_0210.s: fails when the user already has it.
        s.sides[side].water_sport = 1
    elif effect == EffectId.BIDE and landed and not s.sides[side].locked_move:
        # effect_script_0026.s / subscript_bide_start.s: only a first use (not locked, not
        # the unleash turn) locks in with BIDE_1 (two turns) and storedDamage = 0.
        s.sides[side].locked_move = move
        s.sides[side].bide_turns = 2
        s.sides[side].bide_damage = 0
    elif effect == EffectId.ROLLOUT and landed:
        # BtlCmd_CalcRolloutPower (script.c:5661): rolloutCount starts at 5 on a first use,
        # every use decrements it, the lock ends at 0; a hit that dealt nothing (an
        # immunity) runs subscript_missed.s, which unlocks.
        sd = s.sides[side]
        left = max(sd.rollout_count - 1, 0) if sd.locked_move else 4
        if effective and left > 0:
            sd.locked_move, sd.rollout_count = move, left
        else:
            sd.locked_move = sd.rollout_count = 0


def _attractable(s: RefState, side: int, source: int) -> bool:
    """``BtlCmd_TryAttract`` (script.c:5727-5742): different genders, neither genderless (2),
    and ``side``'s battler not in love yet."""
    mine, theirs = _active(s, side).gender, _active(s, source).gender
    return mine != theirs and 2 not in (mine, theirs) and not s.sides[side].infatuated


def _contact_ability(turn: _Turn, s: RefState, side: int, move: int, hit: int, effective: bool,
                     dealt: int) -> None:  # fmt: skip
    """``BattleSystem_TriggerAbilityOnHit`` (lib.c:4200): the defender's own ability
    (Mold Breaker does not matter) on a contact move that hit and dealt damage, with a
    living attacker. Rough Skin (:4257) has no roll: ``subscript_rough_skin.s`` takes
    ``BattleSystem_Divide(maxHP, 8)`` from the attacker and tags the holder's ability.
    Static / Poison Point / Flame Body (:4214, :4303, :4321) need an attacker with no
    status and ``RandNext() % 10 < 3`` (``outcome.contact_proc`` of the defending side);
    the status goes through ``subscript_paralyze.s`` / ``subscript_poison.s`` /
    ``subscript_burn.s`` (type immunities apply; the attacker's own Limber stops a
    paralysis silently, subscript_paralyze.s:5, :_170)."""
    target = 1 - side
    ability = _active(s, target).ability
    contact = _t(turn.tables, "move_flags", move) & MoveFlag.CONTACT
    attacker = _active(s, side)
    if not (contact and effective and dealt > 0 and attacker.hp > 0):
        return
    if ability == AbilityId.ROUGH_SKIN:
        _lose_hp(turn, s, side, _divide(attacker.max_hp, 8))
        if target == 0:
            s.ai_ability_known[s.sides[0].active] = 1
        return
    if ability == AbilityId.CUTE_CHARM:
        # lib.c:4339-4355: an attacker not in love, a living defender, then the roll;
        # subscript_infatuate.s, whose TryAttract fails silently; its message (:17) tags the
        # holder's ability.
        holder = _active(s, target)
        if holder.hp and turn.outcome.contact_proc[target][hit] and _attractable(s, side, target):
            s.sides[side].infatuated = 1
            if target == 0:
                s.ai_ability_known[s.sides[0].active] = 1
        return
    if ability == AbilityId.STATIC:
        status = Status.PARALYSIS
    elif ability == AbilityId.POISON_POINT:
        status = Status.POISON
    elif ability == AbilityId.FLAME_BODY:
        status = Status.BURN
    else:
        return
    if status == Status.PARALYSIS and attacker.ability == AbilityId.LIMBER:
        return
    if turn.outcome.contact_proc[target][hit] and _can_have_status(turn.tables, s, side, status):
        attacker.status = status
        attacker.status_counter = 0
        # subscript_paralyze.s:44 / subscript_poison.s:57 tag the holder's ability
        # (BattleAI_SetAbility, script.c:11806).
        if target == 0:
            s.ai_ability_known[s.sides[0].active] = 1


def _thaw_target(turn: _Turn, s: RefState, side: int, move: int, dealt: int) -> None:
    """``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (bcp.c:3710): a living frozen defender hit
    for damage by a Fire-type move thaws."""
    mon = _active(s, 1 - side)
    fire = _move_type(turn.tables, s, side, move) == Type.FIRE
    if mon.status == Status.FREEZE and mon.hp and dealt > 0 and fire:
        mon.status = Status.NONE
        mon.status_counter = 0


def _hit(turn: _Turn, s: RefState, side: int, move: int, hit: int, struggling: bool,
         target_moved: bool) -> None:  # fmt: skip
    """One hit: the move script (damage, ``BATTLE_SUBSCRIPT_UPDATE_HP``), then
    ``BattleControllerPlayer_AfterMoveMessage`` (bcp.c:3467): the secondary effect
    (``ONE_HIT_TRIGGER_SECONDARY`` / ``MULTI_HIT_TRIGGER_SECONDARY``), then the
    contact ability (``*_TRIGGER_ABILITY``), then the defender's thaw."""
    tables, o = turn.tables, turn.outcome
    target = 1 - side
    target_mon = _active(s, target)
    # The move reaches its defender (BattleSystem_Defender): the living foe, or the
    # defender a DEFENDER_ALWAYS move always has; a TARGETS_USER move hits its user.
    landed = target_mon.hp > 0 or _defender_always(tables, move)
    # Bide is a RANGE_USER move whose unleash hits the foe (bcp.c:2702-2713).
    hits_foe = not _targets_user(tables, move) or _t(tables, "move_effect", move) == EffectId.BIDE

    damage = _move_damage(
        tables,
        s,
        side,
        move,
        o.crit[side][hit] != 0,
        o.roll[side][hit],
        magnitude=o.magnitude[side],
        present=o.present[side],
    )
    dealt = stored = 0
    if landed and hits_foe:
        # BattleControllerPlayer_UpdateHP (bcp.c:3346-3404) runs for a non-zero damage:
        # a Focus Sash at full HP sets focusItemActivated, which lasts the whole action.
        if damage > 0 and _sash_at_full_hp(turn, s, target):
            turn.focus_armed = True
        taken = _focus_sash(turn, s, target, damage, turn.focus_armed)  # battleCtx->damage
        # subscript_update_hp.s: the resist berry halves hpCalcTemp (BattleSystem_Divide).
        if damage > 0 and _hit_resist_berry(turn, s, side, move):
            dealt = _lose_hp(turn, s, target, _divide(taken, 2))
            stored = taken
        else:
            dealt = stored = _lose_hp(turn, s, target, taken)
    # BattleControllerPlayer_UpdateHP (bcp.c:3407): a Biding target stores battleCtx->damage,
    # unhalved by a resist berry; otherwise what it lost (the same unless it fainted, and a
    # fainted battler's is cleared at the faint check).
    if s.sides[target].bide_turns:
        s.sides[target].bide_damage += stored

    # Whether the move "worked" on the target: a damaging move dealt damage (not
    # MOVE_STATUS_NO_EFFECTS); a type-checked status move (move_param[1]) is not
    # ineffective (BattleControllerPlayer_CheckTypeChart).
    if _t(tables, "move_category", move) == STATUS_CLASS:
        effect = _t(tables, "move_effect", move)
        type_checked = _t(tables, "move_param", move, 1) == 1
        passes = True
        if effect == EffectId.STATUS_FOE and type_checked:
            # Thunder Wave goes through ApplyTypeChart (bcp.c:2350), which reads Foresight.
            passes = (
                _matchup_x40(
                    tables,
                    _move_type(tables, s, side, move),
                    target_mon.types,
                    bool(s.sides[target].foresight),
                )
                > 0
            )
    else:
        passes = damage > 0
    effective = landed and passes

    _move_effect(turn, s, side, move, hit, landed, effective, dealt, target_moved)
    if struggling and landed:
        # effect_script_0254.s: MOVE_SIDE_EFFECT_ON_HIT | subscript_struggle.s, a
        # secondary effect: Divide(maxHP, 4) recoil, before the contact ability.
        _lose_hp(turn, s, side, _divide(_active(s, side).max_hp, 4))
    _contact_ability(turn, s, side, move, hit, effective, dealt)
    _thaw_target(turn, s, side, move, dealt)


def _status_disrupted(turn: _Turn, s: RefState, side: int, move: int) -> int:
    """``BattleControllerPlayer_CheckStatusDisruption`` (bcp.c:2417): ``GOES_AHEAD``,
    ``DISRUPTED`` when the move does not go ahead, or ``HURT_ITSELF`` for a confusion
    self-hit, which happens here."""
    tables, o = turn.tables, turn.outcome
    sd, mon = s.sides[side], _active(s, side)
    effect = _t(tables, "move_effect", move)
    thaw_move = effect == EffectId.SECONDARY_STATUS and _t(tables, "move_param", move, 2) == 1

    # CHECK_STATUS_STATE_SLEEP (bcp.c:2430): the counter drops by 1 (Early Bird is
    # unsupported); still asleep stops the move, waking up lets the checks go on.
    if mon.status == Status.SLEEP:
        if mon.status_counter < 1:
            mon.status = Status.NONE
        else:
            mon.status_counter -= 1
            if mon.status_counter > 0:
                return DISRUPTED
            mon.status = Status.NONE
    # CHECK_STATUS_STATE_FREEZE (bcp.c:2478): RandNext() % 5 == 0 thaws; otherwise the
    # move stops unless it is a thawing move (it thaws at SELF_THAW below).
    still_frozen = False
    if mon.status == Status.FREEZE:
        if o.thaw[side]:
            mon.status = Status.NONE
        elif not thaw_move:
            return DISRUPTED
        else:
            still_frozen = True
    # CHECK_STATUS_STATE_FLINCH (bcp.c:2529): subscript_flinched.s raises Steadfast's Speed.
    if sd.flinched:
        sd.flinched = 0
        if mon.ability == AbilityId.STEADFAST:
            _change_stage(s, side, Stat.SPE, 1)
        _unlock(sd)  # subscript_flinched.s:9
        return DISRUPTED
    # CHECK_STATUS_STATE_DISABLE (bcp.c:2543): the chosen move is the disabled one.
    if sd.disabled_move and sd.disabled_move == move:
        _unlock(sd)  # subscript_move_is_disabled.s
        return DISRUPTED
    # CHECK_STATUS_STATE_TAUNT (bcp.c:2557): a taunted battler's power-0 move fails.
    if sd.taunt and _t(tables, "move_power", move) == 0:
        _unlock(sd)  # subscript_move_fail_taunted.s
        return DISRUPTED
    # CHECK_STATUS_STATE_CONFUSION (bcp.c:2614): the counter drops; while it stays
    # above 0, RandNext() & 1 makes the battler hurt itself (CALC_SELF_HIT, a Normal
    # physical hit on itself with no side or field conditions, then the variance).
    if sd.confusion:
        sd.confusion -= 1
        if sd.confusion and o.confusion_hit[side]:
            damage = _calc_move_damage(
                tables,
                s,
                side,
                side,
                MOVE_STRUGGLE,
                False,
                in_power=CONFUSION_SELF_HIT_POWER,
                side_conditions=False,
                field_conditions=False,
            )
            damage = _variance(damage, o.roll[side][0])
            # subscript_hurt_self_in_confusion.s: CheckHoldOnWith1HP on the user
            # (script.c:9098-9124), then the followup message removes the sash.
            armed = _sash_at_full_hp(turn, s, side)
            taken = _focus_sash(turn, s, side, damage, armed)
            # Its UPDATE_HP runs the resist-berry check on the user: moveType is Normal and
            # moveStatusFlags 0 since BattleContext_Init (lib.c:1981-1982), so calcTemp is the
            # chosen move's table type (moveCur, bcp.c:1898) and nothing is super-effective.
            if _resist_berry(turn, s, side, _t(tables, "move_type", move), False):
                taken = _divide(taken, 2)
            _lose_hp(turn, s, side, taken)
            _unlock(sd)  # subscript_hurt_self_in_confusion.s
            return HURT_ITSELF
    # CHECK_STATUS_STATE_PARALYSIS (bcp.c:2652): RandNext() % 4 == 0 (Magic Guard is
    # unsupported).
    if mon.status == Status.PARALYSIS and o.full_para[side]:
        _unlock(sd)  # subscript_fully_paralyzed.s
        return DISRUPTED
    # CHECK_STATUS_STATE_ATTRACT (bcp.c:2669-2692): in love, RandNext() & 1 == 0 immobilizes
    # (subscript_immobilized_by_love.s: UnlockMoveChoice, UPDATE_MOVE_BUFFERS, no message).
    if sd.infatuated and o.infatuation[side]:
        _unlock(sd)
        return DISRUPTED
    # CHECK_STATUS_STATE_BIDE (bcp.c:2694): the counter drops; at 0 the unleash is
    # 2 * storedDamage at the foe's active Pokemon (the last hitter, a replacement
    # included). Nothing stored (subscript_bide_end.s) or the foe fainted
    # (subscript_bide_no_target.s) ends the move; the lock is released after it (_fight).
    if sd.bide_turns:
        sd.bide_turns -= 1
        if sd.bide_turns == 0 and (sd.bide_damage <= 0 or _active(s, 1 - side).hp == 0):
            _unlock(sd)
            return DISRUPTED
    # CHECK_STATUS_STATE_SELF_THAW (bcp.c:2728).
    if still_frozen:
        mon.status = Status.NONE
    return GOES_AHEAD


def _hit_rate(tables, s: RefState, attacker: int, move: int) -> int:
    """``hitRate`` of ``BattleControllerPlayer_CheckMoveHitAccuracy`` (bcp.c:2865-2979) for
    ``attacker``'s ``move`` against the foe's active Pokemon, from the state as it is now.
    Thunder in sun, Hustle, Tangled Feet and the accuracy items are unsupported."""
    defender = 1 - attacker
    atk = _active(s, attacker)
    seen = _ignorable(s, attacker, defender)  # Battler_IgnorableAbility
    acc_stages = s.sides[attacker].stages[Stat.ACC]  # statBoosts - 6 (:2873)
    eva_stages = -s.sides[defender].stages[Stat.EVA]  # 6 - statBoosts (:2874)
    if atk.ability == AbilityId.SIMPLE:  # :2876
        acc_stages *= 2
    if seen == AbilityId.SIMPLE:  # :2879
        eva_stages *= 2
    if seen == AbilityId.UNAWARE:  # :2882
        acc_stages = 0
    if atk.ability == AbilityId.UNAWARE:  # :2885
        eva_stages = 0
    if s.sides[defender].foresight and eva_stages < 0:  # MON_IS_IDENTIFIED, :2888
        eva_stages = 0
    total = min(max(6 + eva_stages + acc_stages, 0), 12)  # :2892-2898
    rate = _t(tables, "move_accuracy", move)
    num, den = HIT_RATE_BY_STAGE[total]
    rate = rate * num // den  # :2918-2919
    if atk.ability == AbilityId.COMPOUNDEYES:  # :2922
        rate = rate * 130 // 100
    if not _weather_suppressed(tables, s):  # NO_CLOUD_NINE, :2926
        if s.weather == Weather.SANDSTORM and seen == AbilityId.SAND_VEIL:  # :2927
            rate = rate * 80 // 100
        if s.weather == Weather.HAIL and seen == AbilityId.SNOW_CLOAK:  # :2931
            rate = rate * 80 // 100
        if s.weather == Weather.FOG:  # :2935
            rate = rate * 6 // 10
    if s.gravity:  # FIELD_CONDITION_GRAVITY, :2972
        rate = rate * 10 // 6
    return rate


def _fight(turn: _Turn, s: RefState, side: int, action: int, move: int, struggling: bool,
           locked: bool, target_moved: bool) -> int | None:  # fmt: skip
    """``BattleControllerPlayer_FightCommand`` through ``TryMove`` and the move script
    for ``side``'s active Pokemon. Returns what ``UpdateFlagsWhenHit`` (bcp.c:4499-4507)
    writes to the foe's ``moveHit`` (``ref_step`` applies it after the faint check), or
    None when the foe is not the defender."""
    tables, o = turn.tables, turn.outcome
    mon = _active(s, side)
    # FightCommand (bcp.c:1900) picks the defender first: the foe while it is alive (any
    # HP for RANGE_OPPONENT_SIDE), the user for SELF_DEFENDER ranges.
    target_range = _move_target(tables, move)
    foe_defender = (
        target_range not in SELF_DEFENDER
        and not _curse_on_user(tables, move)
        and (_active(s, 1 - side).hp > 0 or target_range == MoveTarget.RANGE_OPPONENT_SIDE)
    )
    # UpdateMoveBuffers (bcp.c:3945-3949), reached on every path of a battler that acts:
    # movePrevByBattler is MOVE_NONE unless the move succeeded (below).
    s.sides[side].last_move = 0
    unleash = s.sides[side].bide_turns == 1  # this check brings the Bide counter to 0
    disruption = _status_disrupted(turn, s, side, move)
    if disruption == HURT_ITSELF:  # the user became the defender (bcp.c:2629)
        return None
    if disruption == DISRUPTED:  # no attack message: moveHit = MOVE_NONE
        return 0 if foe_defender else None
    # From here every path prints the attack message (subscript_no_pp.s,
    # subscript_no_target.s, subscript_missed.s, the move script): moveHit = the move.
    record = move if foe_defender else None
    turn.attack_message = True
    # BattleControllerPlayer_DecrementPP (bcp.c:2235): one PP from the move's slot;
    # none left fails the move (MOVE_STATUS_NO_PP). Struggle spends none, nor does a
    # battler that was locked at command selection (ppDecremented, bcp.c:335).
    if not struggling and not locked:
        slot = action
        if mon.pp[slot] == 0:
            return record
        mon.pp[slot] -= 1
    # BattleControllerPlayer_HasNoTarget (bcp.c:2306): BattleSystem_Defender found no
    # living foe for a move aimed at it. Nature Power's defender is its user
    # (RANGE_SINGLE_TARGET_SPECIAL, lib.c:1665-1669): it never fails here.
    nature_power = _t(tables, "move_effect", move) == EffectId.NATURE_POWER
    if not nature_power and not _defender_always(tables, move) and _active(s, 1 - side).hp == 0:
        _unlock(s.sides[side])  # subscript_no_target.s
        return record
    # SYSCTL_MOVE_SUCCEEDED (bcp.c:3212): UpdateMoveBuffers (bcp.c:3945) keeps the move
    # (Struggle's own id when struggling; Nature Power, moveTemp, for Nature Power) in
    # movePrevByBattler.
    s.sides[side].last_move = move
    if nature_power:
        # effect_script_0173.s: GetTerrainMove, then GoToMoveScript (script.c:2111-2134):
        # moveCur is the called move from here on, and the defender is re-picked for it
        # (randomize: the living foe in singles, else BATTLER_NONE -> subscript_no_target,
        # which unlocks; UpdateFlagsWhenHit then writes no moveHit). With a defender the foe's
        # moveHit becomes moveCur.
        move = _called_move(tables, s, move)
        if not _defender_always(tables, move) and _active(s, 1 - side).hp == 0:
            _unlock(s.sides[side])
            return None
        record = move
    # The move script (bcp.c:3214): effect_script_0158.s (Fake Out) sets MOVE_STATUS_FAILED
    # unless fakeOutTurnNumber == totalTurns; TryMove then ends in subscript_missed.s
    # ("But it failed!").
    effect = _t(tables, "move_effect", move)
    if effect == EffectId.FAKE_OUT and s.sides[side].fake_out_turn != s.turn:
        _unlock(s.sides[side])  # subscript_missed.s
        return record
    # effect_script_0007.s (Selfdestruct, Explosion): Damp (unsupported) aside, the script
    # sets the user's HP to 0 here, before the accuracy check and the type chart. It faints
    # at BATTLE_CONTROL_FAINT_AFTER_SELFDESTRUCT (bcp.c:3889), within this action: the same
    # faint check as the target's, where the order of the flagged faints changes nothing.
    if effect == EffectId.SELF_DESTRUCT:
        mon.hp = 0
        turn.fainted.append((side, s.sides[side].active))
    # CheckMoveHitAccuracy (bcp.c:2865): accuracy 0 never misses; a multi-hit move
    # checks only its first hit (SYSCTL_MULTI_HIT_MOVE). The hit rate is computed here,
    # from the stages of this moment; o.hit[side][0] is the class of the roll
    # RandNext() % 100 + 1 (engine/outcome.py), which misses when it is 0 or above it.
    roll = o.hit[side][0]
    missed = _t(tables, "move_accuracy", move) != 0 and not 0 < roll <= _hit_rate(
        tables, s, side, move
    )
    # CheckMoveHitOverrides (bcp.c:3001-3014) after the accuracy roll: a protecting foe
    # stops the move, Battler_UnlockMoveChoice; subscript_missed.s unlocks too.
    if missed or _protected(turn, s, side, move, unleash):
        _unlock(s.sides[side])  # subscript_missed.s
        # Bide's unleash defender is the last hitter (bcp.c:2702-2706): its moveHit.
        return move if unleash else record
    # effect_script_0111.s (Protect, Detect) -> subscript_protect.s -> BtlCmd_TryProtection
    # (script.c:5149-5190). The moveProtect reset (:5154-5158) is folded into protect_count
    # at the end of each action (ref_step). The roll comes first, then the last-to-act test
    # (waitingBattlers, bcp.c:881-886: the battler acting second is the last).
    if effect == EffectId.PROTECT:
        sd = s.sides[side]
        n = sd.protect_count
        passes = n == 0 or o.protect[side] != 0  # PROTECT_SUCCESS_RATE[n] >= RandNext()
        if passes and not target_moved:
            turn.protecting[side] = True
            sd.protect_count = min(n + 1, len(PROTECT_SUCCESS_RATE) - 1)
        else:
            sd.protect_count = 0
            return record  # MOVE_STATUS_FAILED: subscript_missed.s, no AfterMoveEffects
    # effect_script_0028.s (Roar, Whirlwind): its ON_HIT subscript runs for a move that got
    # past the accuracy check and Protect; a failure ("But it failed!", subscript_missed.s)
    # skips AfterMoveEffects. After a drag the hit below plays against the newcomer: Roar
    # deals no damage and has no effect there, and AfterMoveEffects checks the items of the
    # attacker and the new defender.
    if effect == EffectId.ROAR and not _roar(turn, s, side):
        return record
    # BtlCmd_SetMultiHit (script.c:2634): SetMultiHit 2 (effect_script_0044.s) is fixed,
    # SetMultiHit 0 draws 2-5.
    hits = 1
    if effect == EffectId.MULTI_HIT:
        hits = _t(tables, "move_param", move, 0) or o.n_hits[side]
    turn.focus_armed = turn.pluck_flag = False  # BattleContext_Init ended the last action
    for hit in range(hits):
        # BattleControllerPlayer_LoopMultiHit (bcp.c:3780) stops once a battler fainted
        # (faintedMon): the target, or the attacker by Rough Skin.
        if hit > 0 and (_active(s, 1 - side).hp == 0 or _active(s, side).hp == 0):
            break
        # subscript_recover_hp.s _036 ("HP is full!", for Recover's user or Present's heal
        # target) sets MOVE_STATUS_NO_MORE_WORK: CheckMoveFailure (bcp.c:3313-3315) goes to
        # LOOP_SPREAD_MOVES, past AfterMoveEffects (no item checks). Single-hit moves.
        user, foe = _active(s, side), _active(s, 1 - side)
        heal = effect == EffectId.PRESENT and o.present[side] == PRESENT_HEAL
        full = (effect == EffectId.RECOVER and user.hp == user.max_hp) or (
            heal and foe.hp == foe.max_hp > 0
        )
        _hit(turn, s, side, move, hit, struggling, target_moved)
        if full:
            continue
        # BattleControllerPlayer_AfterMoveEffects (bcp.c:3638), after every hit:
        # AFTER_MOVE_EFFECT_ATTACKER_ITEM (bcp.c:3684), then _DEFENDER_ITEM (:3691), then
        # _HELD_ITEM_STATUS (:3730) for every battler. A restore subscript's PLUCK_CHECK
        # keeps the item while the pluck flag is set, and clears the flag.
        for battler in (side, 1 - side, side, 1 - side):
            if _held_item(turn, s, battler, keep=turn.pluck_flag):
                turn.pluck_flag = False
    if unleash:  # the unleash script spent the Bide counter (bcp.c:2702, subscript_bide_end.s)
        _unlock(s.sides[side])
        record = move
    return record


def _roar(turn: _Turn, s: RefState, side: int) -> bool:
    """``subscript_force_target_to_switch_or_flee.s`` for ``side``'s Roar on the foe; returns
    whether it dragged a Pokemon in. Its semi-invulnerable, Suction Cups and Ingrain tests
    (:5-7) and Soundproof (lib.c:3523-3530) cannot be met: no supported move or ability sets
    them. ``BtlCmd_TryWhirlwind`` (script.c:5241-5320), trainer battle, singles: count the
    foe party's slots with a species, not an egg and HP left (:5285-5296); one or none ->
    fail with no draw (:5299-5300). ``BattleSystem_CanWhirlwind`` (lib.c:3072-3086): a user
    of at least the target's level passes, else ``(((RandNext() & 0xFF) * (La + Ld)) >> 8) + 1
    > Ld / 4`` (``outcome.roar_level``). The loop ``RandNext() % partyCount`` until an
    eligible slot other than the active one (:5303-5311): ``outcome.roar_target`` is its
    rank among them. Then ``TryRestoreStatusOnSwitch`` (Natural Cure), ``SwitchAndUpdateMon``
    (script.c:2488-2525: InitBattleMon, UpdateAfterSwitch), the hazards, and AfterMoveEffects'
    switch-in effects (bcp.c:3672-3680). UpdateAfterSwitch sets the newcomer's action to
    MOVE_END (lib.c:2053): it does not act this turn."""
    o = turn.outcome
    foe = 1 - side
    sd = s.sides[foe]
    eligible = [i for i, m in enumerate(sd.party) if m.species and m.hp and i != sd.active]
    if not eligible:
        return False
    if _active(s, side).level < _active(s, foe).level and not o.roar_level[side]:
        return False
    slot = eligible[min(int(o.roar_target[side]), len(eligible) - 1)]
    out = _active(s, foe)
    if out.hp and out.ability == AbilityId.NATURAL_CURE:  # TryRestoreStatusOnSwitch, :9139
        out.status = Status.NONE
        out.status_counter = 0
    # Back to the party: the status is written back without MON_CONDITION_TOXIC_COUNTER
    # (battle_controller.c:1481-1483), which every write-back strips.
    if out.status == Status.TOXIC:
        out.status_counter = 0
    _enter(turn, s, foe, slot)
    _send_out_exp(s, foe)  # PokemonSendOut BTLSCR_DEFENDER (:22)
    _stealth_rock_damage(turn, s, foe)  # BATTLE_SUBSCRIPT_HAZARDS_CHECK (:31)
    _switch_in_effects(turn, s, foe)
    turn.done[foe] = True
    return True


def _pursuit(turn: _Turn, s: RefState, side: int, actions, choices) -> None:
    """``side`` is switching: ``BtlCmd_TryPursuit`` (script.c:6884, from
    ``subscript_switch_pokemon.s:11``): a foe that has not acted, chose Pursuit with
    PP left, alive and neither asleep nor frozen, spends the PP and hits with
    ``powerMul = 20``; ``subscript_pursuit.s`` runs crit, damage and the type chart
    (no accuracy check, no secondary effect), then the contact ability. The
    pursuer's action is over."""
    tables = turn.tables
    pursuer = 1 - side
    if turn.done[pursuer] or actions[pursuer] >= MOVE_ACTIONS:
        return
    move, struggling, _ = choices[pursuer]
    mon = _active(s, pursuer)
    if struggling or _t(tables, "move_effect", move) != EffectId.PURSUIT:
        return
    if mon.hp == 0 or mon.status in (Status.SLEEP, Status.FREEZE):
        return
    slot = actions[pursuer]
    if mon.pp[slot] == 0:
        return
    mon.pp[slot] -= 1
    turn.done[pursuer] = True
    # script.c:6930: movePrevByBattler[pursuer] = Pursuit.
    s.sides[pursuer].last_move = move
    o = turn.outcome
    damage = _move_damage(tables, s, pursuer, move, o.crit[pursuer][0] != 0, o.roll[pursuer][0],
                          PURSUIT_POWER_MUL)  # fmt: skip
    # subscript_pursuit.s: CheckHoldOnWith1HP on the target (script.c:9098-9124).
    armed = _sash_at_full_hp(turn, s, side)
    taken = _focus_sash(turn, s, side, damage, armed)
    # Its UPDATE_HP runs the resist-berry check on the target (subscript_update_hp.s:9).
    if damage > 0 and _hit_resist_berry(turn, s, pursuer, move):
        taken = _divide(taken, 2)
    dealt = _lose_hp(turn, s, side, taken)
    _contact_ability(turn, s, pursuer, move, 0, damage > 0, dealt)


def _switch(turn: _Turn, s: RefState, side: int, slot: int, actions, choices) -> None:
    """``BattleControllerPlayer_SwitchCommand`` -> ``subscript_switch_pokemon.s``, then
    ``BattleControllerPlayer_MoveEnd`` (bcp.c:3973) runs the switch-in effects."""
    mon = _active(s, side)
    # :5-6 BtlCmd_TryRestoreStatusOnSwitch (script.c:9139): Natural Cure with HP left.
    if mon.hp and mon.ability == AbilityId.NATURAL_CURE:
        mon.status = Status.NONE
        mon.status_counter = 0
    # :11 Pursuit, then its own faint check (the battle may end here).
    _pursuit(turn, s, side, actions, choices)
    _faint_check(turn, s)
    if s.over:
        return
    # The outgoing Pokemon goes back to the party: battle_controller.c:1482 writes its
    # status back without MON_CONDITION_TOXIC_COUNTER.
    if mon.status == Status.TOXIC:
        mon.status_counter = 0
    _enter(turn, s, side, slot)
    _send_out_exp(s, side)  # subscript_switch_pokemon.s:39
    _stealth_rock_damage(turn, s, side)  # :43
    _switch_in_effects(turn, s, side)


def _use_item(tables, s: RefState, side: int, item: int) -> None:
    """``BattleControllerPlayer_ItemCommand`` (bcp.c:1905) -> ``subscript_use_potion.s``:
    ``BattleSystem_UseBagItem`` (battle_system.c:435) restores the Potion's 20 HP to a
    living active Pokemon that is not at full HP; ``BATTLE_SUBSCRIPT_UPDATE_HP`` caps
    it at max HP."""
    mon = _active(s, side)
    if _t(tables, "trainer_item_effect", item) != TrainerItemId.POTION:
        return
    if mon.hp == 0 or mon.hp == mon.max_hp:
        return
    _gain_hp(s, side, POTION_HP_RESTORED)


# --------------------------------------------------------------------------- end of turn


def _speed_order(tables, s: RefState, o) -> list[int]:
    """``BattleSystem_SortMonSpeedOrder`` (lib.c:3364): ``CompareBattlerSpeed`` with
    ``ignoreQuickClaw`` (no priority); a tie swaps on ``outcome.end_speed_tie``."""
    second = _player_second(tables, s, (0, 0), o.end_speed_tie, list(o.quick_claw))
    return [1, 0] if second else [0, 1]


def _decomp_ability(tables, mon: RefPokemon) -> int:
    """``Battler_Ability``: the decomp ability id at the Pokemon's personal row and
    ability slot."""
    row = _t(tables, "species_form_row", mon.species, mon.form)
    return _t(tables, "species_abilities_decomp", row, mon.ability_slot)


def _weather_suppressed(tables, s: RefState) -> bool:
    """Not ``NO_CLOUD_NINE`` (include/battle/common.h:63-64): an active battler with HP
    left (``COUNT_ALIVE_BATTLERS``) has Cloud Nine or Air Lock."""
    locks = (WEATHER_ABILITY_IDS["Cloud Nine"], WEATHER_ABILITY_IDS["Air Lock"])
    return any(
        _active(s, side).hp and _decomp_ability(tables, _active(s, side)) in locks
        for side in (0, 1)
    )


def _weather_effect(turn: _Turn, s: RefState, side: int) -> None:
    """``BtlCmd_EndOfTurnWeatherEffect`` (script.c:5561-5647) and
    ``subscript_weather_continues.s`` for one battler (``NO_CLOUD_NINE`` checked by the
    caller). Rain and sun do something only through unmapped abilities (Dry Skin, Solar
    Power, Rain Dish, Hydration); Dig / Dive's no-weather-damage flag cannot be set."""
    mon = _active(s, side)
    if not mon.hp:
        return
    ability = _decomp_ability(turn.tables, mon)
    hp_calc = 0
    if s.weather == Weather.SANDSTORM:  # script.c:5573-5583
        immune = {Type.ROCK, Type.STEEL, Type.GROUND} & set(mon.types)
        if not immune and mon.ability != AbilityId.SAND_VEIL:
            hp_calc = -_divide(mon.max_hp, 16)
    if s.weather == Weather.HAIL:  # the hail branch
        if ability == WEATHER_ABILITY_IDS["Ice Body"]:
            if mon.hp < mon.max_hp:
                hp_calc = _divide(mon.max_hp, 16)
        elif Type.ICE not in mon.types and mon.ability != AbilityId.SNOW_CLOAK:
            hp_calc = -_divide(mon.max_hp, 16)
    if hp_calc > 0:  # subscript_weather_continues.s _076
        _gain_hp(s, side, hp_calc)
    elif hp_calc < 0 and ability != WEATHER_ABILITY_IDS["Magic Guard"]:
        _lose_hp(turn, s, side, -hp_calc)


def _field_conditions(turn: _Turn, s: RefState, order: list[int]) -> None:
    """``BattleControllerPlayer_CheckFieldConditions`` (bcp.c:932-1254). Every step is
    preceded by ``AnyFainted`` / ``CheckBattleOver`` (:939-944): nothing runs once the
    battle is over, and the chip's faints are checked once, after every battler's."""
    if s.over:
        return
    # REFLECT, LIGHT_SCREEN, MIST, SAFEGUARD, TAILWIND, LUCKY_CHANT (:947-1083), each for
    # side 0 then side 1: the count drops and the condition ends at 0.
    for name in ("reflect", "light_screen", "mist", "safeguard", "tailwind", "lucky_chant"):
        for sd in s.sides:
            if getattr(sd, name):
                setattr(sd, name, getattr(sd, name) - 1)
    # WISH (:1085-1111): not implemented (no supported move sets it).
    # RAINING, SANDSTORM, SUNNY, HAILING (:1113-1211): a permanent weather goes on; a
    # temporary one ends when its count reaches 0 (subscript_*_end.s: no effect that turn).
    # DEEP_FOG (:1213-1225) prints a message only.
    if s.weather in (Weather.RAIN, Weather.SANDSTORM, Weather.SUN, Weather.HAIL):
        goes_on = True
        if s.weather_timer:
            s.weather_timer -= 1
            if s.weather_timer == 0:
                s.weather = Weather.NONE
                goes_on = False
        # subscript_weather_continues.s: each battler in monSpeedOrder.
        if goes_on and not _weather_suppressed(turn.tables, s):
            for side in order:
                _weather_effect(turn, s, side)
        _faint_check(turn, s)
        if s.over:
            return
    # GRAVITY (:1227-1238).
    if s.gravity:
        s.gravity -= 1


def _end_of_turn(turn: _Turn, s: RefState, order: list[int]) -> None:
    """``BattleControllerPlayer_CheckMonConditions`` (bcp.c:1289): every implemented
    check for one battler, then for the next, in ``monSpeedOrder`` (``order``: sorted
    once after the last action, bcp.c:887, before the field pass); each check that ran
    is followed by the faint check, and nothing runs once the battle is over."""
    for battler in order:
        for check in (
            _aqua_ring_check,  # MON_COND_CHECK_STATE_AQUA_RING, bcp.c:1332
            _turn_end_ability,  # MON_COND_CHECK_STATE_ABILITY, bcp.c:1355
            _held_item_hp,  # MON_COND_CHECK_STATE_USE_ITEM, bcp.c:1362
            _leech_seed_check,  # MON_COND_CHECK_STATE_LEECH_SEED, bcp.c:1378
            _poison_check,  # MON_COND_CHECK_STATE_POISON, bcp.c:1393
            _toxic_check,  # MON_COND_CHECK_STATE_TOXIC, bcp.c:1405
            _burn_check,  # MON_COND_CHECK_STATE_BURN, bcp.c:1425
            _bind_check,  # MON_COND_CHECK_STATE_BIND, bcp.c:1462
            _disable_check,  # MON_COND_CHECK_STATE_DISABLE, bcp.c:1565
            _taunt_check,  # MON_COND_CHECK_STATE_TAUNT, bcp.c:1635
            _held_item_hp,  # MON_COND_CHECK_STATE_ITEM_CONDITION, bcp.c:1699
        ):
            if s.over:
                return
            check(turn, s, battler)
            _faint_check(turn, s)


def _aqua_ring_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_AQUA_RING`` (bcp.c:1332-1351): a ringed battler below full HP
    and alive heals ``Divide(maxHP, 16)`` (``subscript_aqua_ring_heal.s``, Big Root applied).
    Heal Block (``healBlockTurns``) is unsupported."""
    mon = _active(s, side)
    if s.sides[side].aqua_ring and mon.hp and mon.hp != mon.max_hp:
        _gain_hp(s, side, _big_root(turn, s, side, _divide(mon.max_hp, 16)))


def _leech_seed_check(turn: _Turn, s: RefState, side: int) -> None:
    """Seeded and alive, with the seeder's battler slot alive:
    ``subscript_leech_seed_effect.s`` drains ``maxHP / 8`` (at least 1, capped at HP)
    and the recipient gains what was lost."""
    if not s.sides[side].leech_seed:
        return
    mon, recipient = _active(s, side), _active(s, 1 - side)
    if mon.hp and recipient.hp:
        lost = _lose_hp(turn, s, side, max(mon.max_hp // 8, 1))
        # subscript_leech_seed_effect.s: the recipient gains hitDamage, Big Root applied.
        _gain_hp(s, 1 - side, _big_root(turn, s, 1 - side, lost))


def _poison_check(turn: _Turn, s: RefState, side: int) -> None:
    mon = _active(s, side)
    if mon.status == Status.POISON and mon.hp:
        _lose_hp(turn, s, side, _divide(mon.max_hp, 8))


def _toxic_check(turn: _Turn, s: RefState, side: int) -> None:
    """The counter rises (up to 15), then ``Divide(maxHP, 16) * counter``."""
    mon = _active(s, side)
    if mon.status == Status.TOXIC and mon.hp:
        if mon.status_counter < TOXIC_COUNTER_MAX:
            mon.status_counter += 1
        _lose_hp(turn, s, side, _divide(mon.max_hp, 16) * mon.status_counter)


def _burn_check(turn: _Turn, s: RefState, side: int) -> None:
    """``subscript_burn_damage.s``: ``DivideVarByValue`` maxHP by 8."""
    mon = _active(s, side)
    if mon.status == Status.BURN and mon.hp:
        _lose_hp(turn, s, side, _divide(mon.max_hp, 8))


def _bind_check(turn: _Turn, s: RefState, side: int) -> None:
    """The counter drops; while it stays above 0, ``Divide(maxHP, 16)`` damage
    (``subscript_bind_effect.s``), at 0 ``subscript_bind_end.s``."""
    sd, mon = s.sides[side], _active(s, side)
    if sd.trapped and mon.hp:
        sd.trapped -= 1
        if sd.trapped:
            _lose_hp(turn, s, side, _divide(mon.max_hp, 16))
        else:
            sd.trap_move = 0


def _disable_check(turn: _Turn, s: RefState, side: int) -> None:
    """bcp.c:1565: a disabled move the Pokemon no longer knows zeroes the counter; the
    counter drops while non-zero, else the move is cleared."""
    sd = s.sides[side]
    if not sd.disabled_move:
        return
    if sd.disabled_move not in _active(s, side).moves:
        sd.disable_turns = 0
    if sd.disable_turns:
        sd.disable_turns -= 1
    else:
        sd.disabled_move = 0


def _taunt_check(turn: _Turn, s: RefState, side: int) -> None:
    sd = s.sides[side]
    if sd.taunt:
        sd.taunt -= 1


# --------------------------------------------------------------------------- the turn


def _usable(tables, s: RefState, side: int) -> list[bool]:
    """``BattleSystem_CheckInvalidMoves`` (lib.c:2255), implemented checks: a move, PP
    left, not the disabled move, not power 0 while taunted."""
    mon, taunted = _active(s, side), s.sides[side].taunt > 0
    disabled = s.sides[side].disabled_move
    lock = _locked_slot(s, side)
    if lock >= 0:  # Battler_CanPickCommand is FALSE (lib.c:6224): only the locked move
        return [k == lock for k in range(MOVE_ACTIONS)]
    return [
        m > 0 and pp > 0 and m != disabled and not (taunted and _t(tables, "move_power", m) == 0)
        for m, pp in zip(mon.moves, mon.pp)
    ]


def _locked_slot(s: RefState, side: int) -> int:
    """The move slot holding ``moveLockedInto``, -1 when not locked."""
    locked = s.sides[side].locked_move
    moves = _active(s, side).moves
    return moves.index(locked) if locked and locked in moves else -1


def _command(action: int) -> str:
    if action >= FIRST_ITEM_ACTION:
        return ITEM
    return SWITCH if action >= MOVE_ACTIONS else FIGHT


def _choose(tables, s: RefState, side: int, action: int) -> tuple[int, bool, bool]:
    """``(move, struggling, locked)`` a move action selects, fixed at command selection:
    the locked move of a battler that picks no command (bcp.c:335, :1885-1888), Struggle
    when no move is usable (``STRUGGLING_ALL``, bcp.c:433), else the slot's move."""
    if _locked_slot(s, side) >= 0:
        return s.sides[side].locked_move, False, True
    if not any(_usable(tables, s, side)):
        return MOVE_STRUGGLE, True, False
    return _active(s, side).moves[min(action, MOVE_ACTIONS - 1)], False, False


def ref_step(tables, state: RefState, actions: tuple[int, int], outcome) -> RefState:
    """Play one turn (``engine.step.step``'s contract)."""
    s = copy.deepcopy(state)
    if s.over:
        return s
    turn = _Turn(tables, outcome, entry_turn=s.turn + 1)
    actions = (int(actions[0]), int(actions[1]))
    choices = [_choose(tables, s, side, actions[side]) for side in (0, 1)]
    commands = [_command(a) for a in actions]
    # TrainerAI_RecordLastMove (trainer_ai.c:2705-2717) runs on the AI's FIGHT path
    # (TrainerAI_MainSingles, :295), which a forced Struggle skips (bcp.c:433): the
    # player's movePrevByBattler joins the AI's list, as the matching move slot of the
    # player's active Pokemon (Struggle matches none and is not recorded).
    if commands[1] == FIGHT and not choices[1][1] and not choices[1][2]:
        player = s.sides[0]
        pending = player.last_move
        for k, m in enumerate(_active(s, 0).moves):
            if pending and m == pending:
                s.ai_seen_moves[player.active][k] = 1
    # TrainerAI_ShouldUseItem (trainer_ai.c:4056) empties the bag slot when it picks the
    # item (:4193), before any action runs. Its loop (:4091) does not break: every later
    # slot i with aliveMons <= trainerItemCounts - i + 1 (:4092) is zeroed too, aliveMons
    # counting the party members with HP left (:4081-4089).
    items = [0, 0]
    for side in (0, 1):
        if commands[side] == ITEM:
            sd = s.sides[side]
            alive = sum(1 for m in sd.party if m.species and m.hp)
            used = actions[side] - FIRST_ITEM_ACTION
            items[side] = sd.trainer_items[used]
            sd.trainer_items[used] = 0
            for i in range(used + 1, len(sd.trainer_items)):
                if alive <= sd.trainer_item_count - i + 1:
                    sd.trainer_items[i] = 0

    # BattleControllerPlayer_CalcTurnOrder (bcp.c:707): items and switches first, in
    # battler order, then moves; two actions of one command are ordered by
    # CompareBattlerSpeed, with priority only between two moves.
    early = [c != FIGHT for c in commands]
    if early[0] != early[1]:
        order = [0, 1] if early[0] else [1, 0]
    elif commands[0] != commands[1]:  # an item and a switch: battler order
        order = [0, 1]
    else:
        priority = (0, 0)
        if not early[0]:
            priority = tuple(_t(tables, "move_priority", choices[side][0]) for side in (0, 1))
        # The order reads the speedRand drawn the turn before (qc_roll): this turn's is
        # drawn after CalcTurnOrder (bcp.c:853-857).
        second = _player_second(tables, s, priority, outcome.speed_tie, _qc_rolls(s))
        order = [1, 0] if second else [0, 1]

    for position, side in enumerate(order):
        if s.over:
            break
        if turn.done[side] or _active(s, side).hp == 0:
            continue
        record = None
        if commands[side] == ITEM:
            _use_item(tables, s, side, items[side])
        elif commands[side] == SWITCH:
            _switch(turn, s, side, actions[side] - MOVE_ACTIONS, actions, choices)
        else:
            move, struggling, locked = choices[side]
            turn.attack_message = False
            record = _fight(
                turn, s, side, actions[side], move, struggling, locked, target_moved=position == 1
            )
            # UpdateMoveBuffers (bcp.c:3936-3942): moveProtect = the move if its attack
            # message showed, else MOVE_NONE; the next TryProtection keeps the chain only for
            # Protect, Detect or Endure (script.c:5154-5158). Folded into protect_count here.
            if not (turn.attack_message and move in PROTECT_CHAIN_MOVES):
                s.sides[side].protect_count = 0
        turn.done[side] = True
        _faint_check(turn, s)  # BATTLE_CONTROL_CHECK_ANY_FAINTED after each action
        # UpdateFlagsWhenHit runs after LOOP_FAINTED (bcp.c:3820-3837): a KO'd defender
        # keeps the record until its replacement clears it.
        if record is not None:
            s.sides[1 - side].move_hit = record
        # BattleControllerPlayer_MoveEnd -> ClearFlags (bcp.c:3986, :4389) clears the
        # actor's own moveHit, unless CheckBattleOver ended the battle first (:3969).
        if not s.over:
            s.sides[side].move_hit = 0
            # BattleControllerPlayer_MoveEnd: RecoverStatusByAbility, attacker then
            # defender (bcp.c:3965-3967).
            _recover_status_by_ability(s, side)
            _recover_status_by_ability(s, 1 - side)

    order = _speed_order(tables, s, outcome)  # BattleSystem_SortMonSpeedOrder, bcp.c:887
    _field_conditions(turn, s, order)
    _end_of_turn(turn, s, order)
    # BattleSystem_SetupNextTurn (lib.c:2234) clears every flinch and unlocks a sleeping
    # battler (lib.c:2240-2243).
    for sd in s.sides:
        sd.flinched = 0
        if sd.party[sd.active].status == Status.SLEEP:
            _unlock(sd)
    s.turn += 1
    # The turn's speedRand (bcp.c:853-857), kept only while it can be read (_qc_kept).
    _keep_qc(tables, s, list(outcome.quick_claw))
    return s


def _qc_rolls(s: RefState) -> list[int]:
    return [sd.qc_roll for sd in s.sides]


def _qc_kept(tables, s: RefState) -> list[bool]:
    """``step.qc_kept``: a living Quick Claw holder is active, or the side waits for a
    replacement and a living holder is on its bench (the next turn's order and the AI read
    the slot's speedRand with the battler on the field then)."""
    out = []
    for sd in s.sides:

        def claw(mon):
            return mon.species and mon.hp and _item_effect(tables, mon) == ItemId.QUICK_CLAW

        active = sd.party[sd.active]
        bench = [m for i, m in enumerate(sd.party) if i != sd.active]
        pending = active.hp == 0 and any(m.species and m.hp for m in bench)
        out.append(not s.over and (bool(claw(active)) or (pending and any(claw(m) for m in bench))))
    return out


def _keep_qc(tables, s: RefState, qc: list[int]) -> None:
    for sd, kept, value in zip(s.sides, _qc_kept(tables, s), qc, strict=True):
        sd.qc_roll = int(bool(value)) if kept else 0


def ref_replace(tables, state: RefState, side: int, slot: int) -> RefState:
    """A faint replacement (``engine.step.replace``'s contract): ``TurnEnd`` ->
    ``ReplaceFainted`` (bcp.c:1848) -> ``subscript_show_party_list.s`` sends the
    Pokemon in and runs the hazards and a faint check (:22-24); the battle-over check
    (bcp.c:1847), then ``ShowBattleMon`` (bcp.c:256) runs the switch-in effects."""
    s = copy.deepcopy(state)
    # ReplaceFainted runs before totalTurns++ (bcp.c:1852), after ref_step counted the
    # turn: the decomp's totalTurns + 1 is s.turn here.
    turn = _Turn(tables, None, entry_turn=s.turn)
    _enter(turn, s, side, slot)
    _send_out_exp(s, side)  # subscript_show_party_list.s:18
    _stealth_rock_damage(turn, s, side)  # :22
    _faint_check(turn, s)
    if not s.over:
        _switch_in_effects(turn, s, side)
    _keep_qc(tables, s, _qc_rolls(s))  # the slot's speedRand stays while it can be read
    return s
