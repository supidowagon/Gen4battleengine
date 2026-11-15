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
* ``ref_grant_exp(tables, state, slot, pending, selected) -> RefState``: the exp payout for
  the opponent's party ``slot`` fainting, like ``engine.levelup.pay_exp``.
* ``ref_advance_learns(tables, state)``, ``ref_learn_options(tables, state)`` and
  ``ref_learn(tables, state, option)``: the deferred learnset scan, like
  ``engine.levelup.advance_learns`` / ``learn_options`` / ``learn``.

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
    ABILITY_DECOMP_IDS,
    ABILITY_MULTITYPE,
    CHOICE_ITEMS,
    FUTURE_SIGHT_TURNS,
    ITEM_ABILITY_IDS,
    ITEM_GRISEOUS_ORB,
    METRONOME_DRAW,
    MINIMIZE_PARAM,
    POWER_IF_TARGET_MOVED,
    MOVE_BATON_PASS,
    MOVE_FAKE_OUT,
    MOVE_SLEEP_TALK,
    MOVE_SNORE,
    MOVE_STRUGGLE,
    MOVE_U_TURN,
    POTION_HP_RESTORED,
    POWER_IF_TARGET_HALF_HP,
    POWER_IF_TARGET_PARALYZED,
    POWER_IF_USER_STATUSED,
    PROTECT_CHAIN_MOVES,
    RESIST_ANY_NORMAL,
    SECRET_POWER_EFFECTS,
    SNORE_PARAM,
    SPECIES_CLAMPERL,
    SPECIES_CUBONE,
    SPECIES_DIALGA,
    SPECIES_DITTO,
    SPECIES_GIRATINA,
    SPECIES_LATIAS,
    SPECIES_LATIOS,
    SPECIES_MAROWAK,
    SPECIES_PALKIA,
    SPECIES_PIKACHU,
    SPIKES_MAX,
    STOMP_PARAM,
    TAILWIND_TURNS,
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    TERRAIN_TYPES,
    TOXIC_SPIKES_MAX,
    TRI_ATTACK_STATUSES,
    UNROLLED_EFFECTS,
    UNTRACEABLE_ABILITIES,
    WEATHER_ABILITIES,
    WEATHER_ABILITY_IDS,
    WEATHER_TURNS,
    YAWN_ABILITY_IDS,
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
# In singles RANGE_ALLY / RANGE_USER_OR_ALLY give the attacker itself (lib.c:1670-1684, B9-5).
DEFENDER_ALWAYS = {
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
    MoveTarget.RANGE_OPPONENT_SIDE,
}
# ... of which these hit the user, not the foe (no damage to the foe's active).
TARGETS_USER = {
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
}
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
    # turnFlags.roosting (subscript_roost.s:11-12, cleared at the turn end): the battler's chart
    # skips its Flying rows (lib.c:2540-2543) for the rest of the turn.
    roosting: list[bool] = field(default_factory=lambda: [False, False])
    # turnFlags damage taken per side this turn, [physical, special, last] and their attackers
    # as 1 + battler (bcp.c:3413-3434), and the current Counter / Mirror Coat / Metal Burst
    # damage (battleCtx->damage).
    taken: list[list[int]] = field(default_factory=lambda: [[0] * 6, [0] * 6])
    counter_damage: int = 0
    # SYSCTL_ATTACK_MESSAGE_SHOWN of the current action (script.c:1432), for
    # UpdateMoveBuffers' moveProtect (bcp.c:3936-3942).
    attack_message: bool = False
    # SYSCTL_PAYOUT_EXP (script.c:1800): the opponent party slots whose faint is still to pay.
    payout: list[int] = field(default_factory=list)
    # SYSCTL_MOVE_HIT of the current action: a hit's UpdateHP ran (bcp.c:3340, :3441).
    move_hit: bool = False
    # The defender's physical / specialDamageTaken this action is nonzero (bcp.c:3411-3424):
    # the Sticky Barb transfer (lib.c:5386-5398) and King's Rock read it.
    damaged: bool = False
    # moveCur of the current action (Nature Power's called move), for Life Orb's class test.
    move_cur: int = 0
    # ATTACKER_SELF_TURN_FLAGS.shellBellDamageDealt of the current action (bcp.c:3427-3431).
    shell_dealt: int = 0
    # moveEffectsData.custapBerry per side, written by CalcTurnOrder (lib.c:1296, :1362).
    custap: list[bool] = field(default_factory=lambda: [False, False])
    # moveFailFlags per side (BattleContext_MoveFailed, lib.c:2808-2820; cleared by
    # SetupNextTurn, lib.c:2232): the Uproar check of the turn's end reads them.
    move_failed: list[bool] = field(default_factory=lambda: [False, False])
    # E2 batch 6: the turn's actions, picks and order as chosen, the position acting, the
    # battler whose action a Pursuit ended (1 + battler), the slot a Roar dragged in per side
    # (1 + slot), powerMul at a U-turn's party list, the search's turn powerMul.
    actions: tuple = (0, 0)
    choices: list = field(default_factory=list)
    items: list[int] = field(default_factory=lambda: [0, 0])
    commands: list[str] = field(default_factory=list)
    order: list[int] = field(default_factory=lambda: [0, 1])
    position: int = 0
    pursued: int = 0
    dragged: list[int] = field(default_factory=lambda: [0, 0])
    pick_mul: int = 10
    power_mul: int = 10
    # The paused action's own inputs (BattleState.ts_used / ts_dealt).
    pivot_dealt: int = 0


class _Pause(Exception):
    """The turn stops at a U-turn's or Baton Pass's party list (``ref_step`` writes the
    scratch, ``_resume`` goes on)."""


def _active(s: RefState, side: int) -> RefPokemon:
    return s.sides[side].party[s.sides[side].active]


def _reveal_item(s: RefState, side: int) -> None:
    """A message names ``side``'s active Pokemon's held item through an item tag of the
    attacker, defender or ``msgBattlerTemp`` (``BattleMessage_ItemTag``, script.c:11726-11745):
    ``BattleAI_SetHeldItem`` (:12119-12122). The AI reads the other side's only, so only the
    player's is kept."""
    if side == 0:
        s.ai_item_known[s.sides[0].active] = _active(s, 0).item


def _forget_item(s: RefState, side: int) -> None:
    """``side``'s active Pokemon lost its item: the removal's ``BattleMon_CopyToParty``
    clears the AI's record (``BattleAI_ClearKnownItem``, lib.c:1819-1823)."""
    if side == 0:
        s.ai_item_known[s.sides[0].active] = 0


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


def _invoked_move(tables, s: RefState, side: int, move: int, call: int) -> int:
    """The move an invoker's script calls (``order.invoked_move``), 0 when it finds none; any
    other move itself. BtlCmd_TrySleepTalk (script.c:4907-4937): the allowed slots (not
    Move_IsInvoker / Focus Punch / Uproar / Chatter / Move_IsMultiTurn, ``NO_SLEEP_TALK``; not
    flagged by CheckInvalidMoves without its NO_PP check: disabled, power 0 while taunted, a
    Choice holder's other moves), the ``call % count``-th of them. BtlCmd_Metronome
    (:4535-4562): the ``call % count``-th of ids 1..467 that the user does not know and
    Move_CanBeMetronomed allows (``NO_METRONOME``). BtlCmd_TryCopycat (:7328-7341): the battle's
    movePrev unless it is MOVE_NONE or not Metronome-able."""
    effect = _t(tables, "move_effect", move)
    mon, sd = _active(s, side), s.sides[side]
    if effect == EffectId.SLEEP_TALK:
        lock = sd.choice_lock
        locked = _holds_choice(tables, s, side) and lock != 0 and lock in mon.moves
        allowed = [
            m
            for m in mon.moves
            if m > 0
            and not _t(tables, "move_flags", m) & MoveFlag.NO_SLEEP_TALK
            and m != sd.disabled_move
            and not (sd.taunt > 0 and _t(tables, "move_power", m) == 0)
            and not (locked and m != lock)
            and not _move_imprisoned(s, side, m)
        ]
        return allowed[call % len(allowed)] if allowed else 0
    if effect == EffectId.METRONOME:
        allowed = [
            m
            for m in range(1, METRONOME_DRAW + 1)
            if m not in mon.moves and not _t(tables, "move_flags", m) & MoveFlag.NO_METRONOME
        ]
        return allowed[call % len(allowed)] if allowed else 0
    if effect == EffectId.COPYCAT:
        prev = s.move_prev
        ok = prev != 0 and not _t(tables, "move_flags", prev) & MoveFlag.NO_METRONOME
        return prev if ok else 0
    return move


def _restore_types(tables, s: RefState, side: int) -> None:
    """The leaving battler's types go back to its personal row's: Camouflage writes only the
    battler's ``BattleMon.type1/type2``, which ``BattleSystem_InitBattleMon`` re-reads from
    the party (lib.c:114-115)."""
    mon = _active(s, side)
    row = _t(tables, "species_form_row", mon.species, mon.form)
    mon.types = [_t(tables, "species_types", row, i) for i in range(len(mon.types))]


def _restore_ability(tables, s: RefState, side: int) -> None:
    """The leaving battler's ability goes back to its personal row's: Trace writes only the
    battler's ``BattleMon.ability``, which ``BattleSystem_InitBattleMon`` re-reads from the
    party (lib.c:124)."""
    mon = _active(s, side)
    row = _t(tables, "species_form_row", mon.species, mon.form)
    mon.ability = _t(tables, "species_abilities", row, mon.ability_slot)


def _armored(s: RefState, attacker: int, defender: int) -> bool:
    """``BattleSystem_CalcCriticalMulti`` (lib.c:7135-7136): no crit against Battle Armor or
    Shell Armor (``Battler_IgnorableAbility``)."""
    return _ignorable(s, attacker, defender) in (AbilityId.SHELL_ARMOR, AbilityId.BATTLE_ARMOR)


def _levitated(tables, s: RefState, attacker: int, defender: int, move_type: int) -> bool:
    """``ApplyTypeChart``'s ``MOVE_STATUS_LEVITATED`` (lib.c:2600-2603): a Ground move against
    Levitate as the attacker sees it; part of ``MOVE_STATUS_IMMUNE``."""
    return move_type == Type.GROUND and _ignorable(s, attacker, defender) == AbilityId.LEVITATE


def _identified(s: RefState, attacker: int, defender: int) -> int:
    """``ApplyTypeChart``'s walk flags (``_chart_rows``): 1 when it skips the Ghost immunities
    (lib.c:2611-2615) for Foresight or the attacker's Scrappy, 2 when ``BasicTypeMulApplies``
    skips an immune Flying row for an ingrained defender (lib.c:2534-2537)."""
    seen = bool(s.sides[defender].foresight) or _active(s, attacker).ability == AbilityId.SCRAPPY
    return int(seen) | (2 if s.sides[defender].ingrain else 0)


def _truant_loafs(s: RefState, side: int) -> bool:
    """``Battler_CheckTruant`` (lib.c:3304-3313): Truant and ``truant != totalTurns & 1``, the
    flag being ``(totalTurns + 1) & 1`` from the entry (lib.c:2119, :2197): ``fake_out_turn``'s
    parity (0 for a lead)."""
    sd = s.sides[side]
    return _active(s, side).ability == AbilityId.TRUANT and (sd.fake_out_turn & 1) != (s.turn & 1)


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
    effect = _t(tables, "move_effect", move)
    if effect == EffectId.FUTURE_SIGHT:  # effect_script_0148.s: no CalcDamage at the use
        return 0
    if effect == EffectId.BIDE:
        return 2 * sd.bide_damage if sd.locked_move and sd.bide_turns == 0 else 0
    foe = _active(s, 1 - attacker)
    kind = _move_type(tables, s, attacker, move)
    if _matchup_x40(tables, kind, foe.types, _identified(s, attacker, 1 - attacker)) == 0:
        return 0
    if _levitated(tables, s, attacker, 1 - attacker, kind):
        return 0
    if effect == EffectId.HALVE_HP:  # effect_script_0040.s: Divide(curHP, 2)
        return _divide(foe.hp, 2)
    if effect == EffectId.OHKO:
        # BtlCmd_TryOHKOMove (script.c:4340-4341): an ignorable Sturdy, MOVE_STATUS_STURDY;
        # else UpdateHP deals maxHP (bcp.c:3343-3345).
        if _ignorable(s, attacker, 1 - attacker) == AbilityId.STURDY:
            return 0
        return foe.max_hp
    flat = _t(tables, "move_param", move, 0)  # Dragon Rage 40, SonicBoom 20 (B9-9)
    return flat if flat else _active(s, attacker).level


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
    ability = _active(s, defender).ability
    # Battler_Ability (lib.c:3100-3103): an ingrained battler's Levitate is ABILITY_NONE.
    if s.sides[defender].ingrain and ability == AbilityId.LEVITATE:
        return AbilityId.NONE
    return ability


def _chart_rows(tables, move_type: int, identified: bool = False) -> list[tuple[int, int]]:
    """The rows of ``sTypeMatchupMultipliers`` (lib.c:2399) for ``move_type``, in
    table order, as (defending type, multiplier in tenths). ``identified`` (Foresight)
    drops the rows after the 0xFE separator (lib.c:2615)."""
    order = tables.type_chart_order[move_type]
    flags = int(identified)
    listed = [
        (int(order[d]), d)
        for d in range(len(order))
        if int(order[d]) >= 0
        and not (flags & 1 and tables.type_chart_foresight[move_type, d])
        # BasicTypeMulApplies (lib.c:2534-2537): an ingrained defender skips the immune Flying row.
        and not (flags & 2 and d == Type.FLYING and _t(tables, "type_chart", move_type, d) == 0)
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
    # lib.c:1228-1238: Swift Swim in rain, Chlorophyll in sun, inside NO_CLOUD_NINE.
    doubled = (mon.ability == AbilityId.SWIFT_SWIM and s.weather == Weather.RAIN) or (
        mon.ability == AbilityId.CHLOROPHYLL and s.weather == Weather.SUN
    )
    if doubled and not _weather_suppressed(tables, s):
        speed *= 2
    # lib.c:1248: Choice Scarf (the speed-halving items before it are unsupported).
    if _item_effect(tables, mon) == ItemId.CHOICE_SCARF:
        speed = speed * 15 // 10
    # lib.c:1257-1261: Quick Feet with any status x15/10, else paralysis quarters it.
    if mon.ability == AbilityId.QUICK_FEET and mon.status != Status.NONE:
        speed = speed * 15 // 10
    elif mon.status == Status.PARALYSIS:
        speed //= 4
    # lib.c:1268-1272: Unburden with canUnburden and no item.
    if mon.ability == AbilityId.UNBURDEN and s.sides[side].can_unburden and not mon.item:
        speed *= 2
    if s.sides[side].tailwind:  # lib.c:1274
        speed *= 2
    return speed


def _item_effect(tables, mon: RefPokemon) -> int:
    return _t(tables, "item_effect", mon.item)


def _pinch_divisor(tables, mon: RefPokemon) -> int:
    """A pinch berry's (or Custap Berry's) param, halved for its holder's own Gluttony
    (``itemPower /= 2``, lib.c:4774-4776, :1289-1291)."""
    power = _t(tables, "item_param", mon.item, 0)
    return power // 2 if mon.ability == AbilityId.GLUTTONY else power


def _custap_active(tables, s: RefState, side: int) -> bool:
    """The active battler holds a Custap Berry at ``curHP <= maxHP / param`` (lib.c:1288-1300)."""
    mon = _active(s, side)
    if _item_effect(tables, mon) != ItemId.CUSTAP_BERRY:
        return False
    return mon.hp <= mon.max_hp // _pinch_divisor(tables, mon)


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
    # Quick Claw (HOLD_EFFECT_SOMETIMES_PRIORITY, lib.c:1278-1286) or a Custap Berry in its
    # pinch (HOLD_EFFECT_PINCH_PRIORITY, lib.c:1288-1300): one flag.
    claw = [
        (_item_effect(tables, _active(s, side)) == ItemId.QUICK_CLAW and qc[side] != 0)
        or _custap_active(tables, s, side)
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
    elif not in_power and effect == EffectId.HP_POWER:  # script.c:7265
        power = 1 + (120 * dfn.hp) // dfn.max_hp
    elif not in_power and effect == EffectId.WEIGHT_POWER:  # script.c:6808-6826
        power = _weight_power(tables, dfn)
    move_type = _move_type(tables, s, attacker, move)  # lib.c:6680 (Normalize is unsupported)
    power = power * power_mul // 10  # lib.c:6688
    # lib.c:6698-6702 Technician: movePower <= 60, not Struggle.
    if a_ability == AbilityId.TECHNICIAN and move != MOVE_STRUGGLE and power <= 60:
        power = power * 15 // 10
    move_class = _t(tables, "move_category", move)

    # lib.c:6706-6708 Huge Power / Pure Power: attackStat x2, before the items.
    if a_ability in (AbilityId.HUGE_POWER, AbilityId.PURE_POWER):
        attack *= 2
    # lib.c:6716-6783: the held items (BattleSystem_GetItemData of Battler_HeldItem,
    # lib.c:6662-6668), the attacker's and the defender's; movePower and the stats are u16,
    # which no value here reaches.
    a_item, d_item = _item_effect(tables, atk), _item_effect(tables, dfn)
    a_power = _t(tables, "item_param", atk.item, 0)
    # lib.c:6716-6722: sTypeBoostingItems.
    if a_item == ItemId.TYPE_BOOST and move_type == _t(tables, "item_param", atk.item, 1):
        power = power * (100 + a_power) // 100
    if a_item == ItemId.CHOICE_BAND:  # lib.c:6724
        attack = attack * 150 // 100
    if a_item == ItemId.CHOICE_SPECS:  # lib.c:6727
        sp_attack = sp_attack * 150 // 100
    # lib.c:6730-6739: Soul Dew (not a Battle Frontier battle).
    if a_item == ItemId.SOUL_DEW and atk.species in (SPECIES_LATIOS, SPECIES_LATIAS):
        sp_attack = sp_attack * 150 // 100
    if d_item == ItemId.SOUL_DEW and dfn.species in (SPECIES_LATIOS, SPECIES_LATIAS):
        sp_defense = sp_defense * 150 // 100
    if a_item == ItemId.DEEP_SEA_TOOTH and atk.species == SPECIES_CLAMPERL:  # lib.c:6740
        sp_attack *= 2
    if d_item == ItemId.DEEP_SEA_SCALE and dfn.species == SPECIES_CLAMPERL:  # lib.c:6744
        sp_defense *= 2
    if a_item == ItemId.LIGHT_BALL and atk.species == SPECIES_PIKACHU:  # lib.c:6748
        power *= 2
    if d_item == ItemId.METAL_POWDER and dfn.species == SPECIES_DITTO:  # lib.c:6752
        defense *= 2
    if a_item == ItemId.THICK_CLUB and atk.species in (SPECIES_CUBONE, SPECIES_MAROWAK):
        attack *= 2  # lib.c:6756
    # lib.c:6760-6775: the orbs (Giratina's Transform test: Transform is unsupported).
    orbs = {
        ItemId.ADAMANT_ORB: (SPECIES_DIALGA, Type.STEEL),
        ItemId.LUSTROUS_ORB: (SPECIES_PALKIA, Type.WATER),
        ItemId.GRISEOUS_ORB: (SPECIES_GIRATINA, Type.GHOST),
    }
    orb = orbs.get(a_item)
    if orb and atk.species == orb[0] and move_type in (Type.DRAGON, orb[1]):
        power = power * (100 + a_power) // 100
    # lib.c:6776-6783: Muscle Band (physical), Wise Glasses (special).
    if (a_item == ItemId.MUSCLE_BAND and move_class == PHYSICAL) or (
        a_item == ItemId.WISE_GLASSES and move_class == SPECIAL
    ):
        power = power * (100 + a_power) // 100

    # lib.c:6785-6788 Thick Fat (ignorable): Fire and Ice power halved.
    if d_ability == AbilityId.THICK_FAT and move_type in (Type.FIRE, Type.ICE):
        power //= 2
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
    if move_type == Type.BUG and a_ability == AbilityId.SWARM and pinch:  # lib.c:6834-6838
        power = power * 150 // 100
    # lib.c:6845-6848 Dry Skin (ignorable): Fire power x125/100.
    if d_ability == AbilityId.DRY_SKIN and move_type == Type.FIRE:
        power = power * 125 // 100
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
    # lib.c:6922-6924 Solar Power in sun.
    if weather == Weather.SUN and a_ability == AbilityId.SOLAR_POWER:
        sp_attack = sp_attack * 15 // 10
    # lib.c:6926 sandstorm boosts a Rock-type defender's SpD.
    if weather == Weather.SANDSTORM and Type.ROCK in dfn.types:
        sp_defense = sp_defense * 15 // 10
    # lib.c:6930-6941 Flower Gift in sun: a living holder on the attacker's side (Atk) or the
    # defender's (SpD, not against Mold Breaker); in singles the side's active battler.
    if weather == Weather.SUN and a_ability == AbilityId.FLOWER_GIFT and atk.hp:
        attack = attack * 15 // 10
    if (
        weather == Weather.SUN
        and dfn.ability == AbilityId.FLOWER_GIFT
        and dfn.hp
        and a_ability != AbilityId.MOLD_BREAKER
    ):
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
        # script.c:1344, criticalMul: x3 for Sniper (Battler_Ability, lib.c:7142-7144).
        damage *= 3 if a_ability == AbilityId.SNIPER else 2
    return damage


def _double_power(tables, s: RefState, attacker: int, defender: int, move: int) -> bool:
    """SmellingSalt against a paralyzed defender (effect_script_0171.s:6), Facade for a user
    with MON_CONDITION_FACADE_BOOST (effect_script_0169.s:5), Brine against
    ``curHP <= maxHP / 2`` (effect_script_0221.s:5-8)."""
    if _t(tables, "move_effect", move) != EffectId.DOUBLE_POWER:
        return False
    cond = _t(tables, "move_param", move, 0)
    foe = _active(s, defender)
    if cond == POWER_IF_TARGET_PARALYZED:
        return foe.status == Status.PARALYSIS
    if cond == POWER_IF_USER_STATUSED:
        return _active(s, attacker).status in (
            Status.POISON,
            Status.TOXIC,
            Status.BURN,
            Status.PARALYSIS,
        )
    return cond == POWER_IF_TARGET_HALF_HP and foe.hp <= foe.max_hp // 2


WEIGHT_TO_POWER = ((100, 20), (250, 40), (500, 60), (1000, 80), (2000, 100))


def _weight_power(tables, mon: RefPokemon) -> int:
    """``BtlCmd_CalcWeightBasedPower`` (script.c:6808-6826, weight_to_power.h): the first
    bound at least the weight gives the power, else 120."""
    weight = _t(tables, "species_weight", mon.species)
    for bound, power in WEIGHT_TO_POWER:
        if bound >= weight:
            return power
    return 120


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
    identified = _identified(s, attacker, defender)  # lib.c:2611-2615 (Scrappy too)
    update = _t(tables, "move_power", move) != 0  # ApplyTypeMultiplier's `update`
    # flags: MOVE_STATUS_SUPER_EFFECTIVE / _NOT_VERY_EFFECTIVE
    for defending, mul in _chart_rows(tables, move_type, identified):  # lib.c:2611-2646
        if defending == def_types[0]:
            damage = _apply_type_multiplier(damage, mul, update, flags)
        if defending == def_types[1] and def_types[0] != def_types[1]:
            damage = _apply_type_multiplier(damage, mul, update, flags)
    # lib.c:2657-2660: Solid Rock (ignorable) on a super-effective hit.
    if (
        "super" in flags
        and update
        and _ignorable(s, attacker, defender) in (AbilityId.SOLID_ROCK, AbilityId.FILTER)
    ):
        damage = _divide(damage * 3, 4)
    # lib.c:2663-2665: then the attacker's Expert Belt.
    atk = _active(s, attacker)
    if "super" in flags and update and _item_effect(tables, atk) == ItemId.EXPERT_BELT:
        damage = damage * (100 + _t(tables, "item_param", atk.item, 0)) // 100
    # lib.c:2668-2672: Tinted Lens on a not-very-effective hit.
    if "not_very" in flags and update and atk.ability == AbilityId.TINTED_LENS:
        damage *= 2
    # lib.c:2600-2603: Levitate takes the move instead of the chart (MOVE_STATUS_LEVITATED).
    if _levitated(tables, s, attacker, defender, move_type):
        return 0
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
    if _t(tables, "move_effect", move) in UNROLLED_EFFECTS:
        return _fixed_damage(tables, s, attacker, move)
    # effect_script_0048.s, _0269.s: a Reckless user's RECOIL move sets powerMul to 12.
    if (
        _t(tables, "move_effect", move) == EffectId.RECOIL
        and _active(s, attacker).ability == AbilityId.RECKLESS
    ):
        power_mul = 12
    # effect_script_0169.s, _0171.s, _0221.s: powerMul 20 when the condition holds; Stomp
    # against MOVE_EFFECT_MINIMIZE (effect_script_0150.s:6-8, B9-10).
    stomp = (
        _t(tables, "move_effect", move) == EffectId.FLINCH
        and _t(tables, "move_param", move, 2) == STOMP_PARAM
    )
    if _double_power(tables, s, attacker, defender, move) or (
        stomp and s.sides[defender].minimized
    ):
        power_mul = 20
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
    # script.c:1346-1348: Life Orb, after criticalMul.
    atk = _active(s, attacker)
    if _item_effect(tables, atk) == ItemId.LIFE_ORB:
        damage = damage * (100 + _t(tables, "item_param", atk.item, 0)) // 100
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
    _source_leaves(s, side, faint=True)


# What Baton Pass keeps (BattleSystem_UpdateAfterSwitch, lib.c:2078-2079, :2107; condition.h:72,
# moves.h:46-48), with the stat stages (InitBattleMon, lib.c:97-101).
BATON_PASSED = (
    "confusion", "leech_seed", "substitute_hp", "water_sport", "mud_sport", "aqua_ring",
    "ingrain", "mean_look", "mean_look_target", "lock_on", "lock_on_target",
    "perish_song", "perish_song_turns", "curse",
)  # fmt: skip


def _reset_battler(sd: RefSide, baton: bool = False) -> None:
    """The volatile state of a side's active battler back to a fresh battler's; a Baton Pass
    keeps ``BATON_PASSED`` and the stages."""
    if not baton:
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
        "destiny_bond",
        "uproar",
        "ingrain",
        "mean_look",
        "mean_look_target",
        "imprison",
        "imprisoned",
        "lock_on",
        "lock_on_target",
        "perish_song",
        "perish_song_turns",
        "yawn",
        "curse",
        "can_unburden",
        "minimized",
    ):
        if not (baton and name in BATON_PASSED):
            setattr(sd, name, 0)


def _unlock(sd: RefSide) -> None:
    """``Battler_UnlockMoveChoice`` (lib.c:1834-1841): the move lock, Bide's turns and stored
    damage (0 outside a Bide) and the Rollout count. It clears VOLATILE_CONDITION_MOVE_LOCKED
    only: an uproaring battler keeps VOLATILE_CONDITION_UPROAR and moveLockedInto, so it stays
    locked (Battler_CanPickCommand, lib.c:6224-6235)."""
    sd.bide_turns = sd.bide_damage = sd.rollout_count = 0
    if not sd.uproar:
        sd.locked_move = 0


def _release_bind(sd: RefSide) -> None:
    sd.trapped = 0
    sd.trap_move = 0


def _source_leaves(s: RefState, side: int, baton: bool = False, faint: bool = False) -> None:
    """``side``'s battler leaves (UpdateAfterSwitch lib.c:2088-2098, CleanupFaintedMon
    :2168-2186): the Bind it put on the foe and the foe's ATTRACT bit sourced from it end, and,
    unless it left by Baton Pass (:2055), the Mean Look of every battler it trapped
    (:2056-2061, :2170-2172)."""
    _release_bind(s.sides[1 - side])
    s.sides[1 - side].infatuated = 0
    if not baton:
        for sd in s.sides:
            if sd.mean_look and sd.mean_look_target == 1 + side:
                sd.mean_look = 0  # meanLookTarget stays (B7-15)
    # The Lock-On of every battler it locked onto: a switch clears it (lib.c:2063-2067), a Baton
    # Pass restarts it at the initial count (:2077-2085); a faint leaves it until the slot's
    # replacement comes in, whose UpdateAfterSwitch clears it (B8-2, B8-7).
    if not faint:
        for sd in s.sides:
            if sd.lock_on and sd.lock_on_target == 1 + side:
                if baton:
                    sd.lock_on = 2
                else:
                    sd.lock_on = sd.lock_on_target = 0


def _faint_check(turn: _Turn, s: RefState, decide: bool = True) -> None:
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
        if side == 1:  # script.c:1800 flags the payout; CalcExpGain pays a SOLO_ENEMY only
            turn.payout.append(slot)
        if s.sides[side].active == slot:
            _cleanup_fainted(turn, s, side)
        if side == 0:
            player_fainted += 1
            s.deaths[slot] = 1
    turn.fainted.clear()
    s.player_faints += player_fainted
    if not decide:  # AnyFainted alone, no CheckBattleOver (the side-condition pass, bcp.c:1749)
        return
    if all(m.species == 0 or m.hp == 0 for m in s.sides[0].party):
        s.over, s.winner = 1, WINNER_OPPONENT
    elif all(m.species == 0 or m.hp == 0 for m in s.sides[1].party):
        s.over, s.winner = 1, WINNER_PLAYER


# --------------------------------------------------------------------------- experience

# Decomp hold effects (generated/item_hold_effects.txt, line N is id N - 1).
HOLD_EFFECT_EVS_UP_SPEED_DOWN = 50  # Macho Brace
HOLD_EFFECT_EXP_SHARE = 51
HOLD_EFFECT_FRIENDSHIP_UP = 53  # Soothe Bell
HOLD_EFFECT_EXP_UP = 66  # Lucky Egg
# The Power item of each stat (HOLD_EFFECT_LVLUP_*_EV_UP, item_hold_effects.txt:118-123).
POWER_ITEM_HOLD_EFFECTS = {
    Stat.ATK: 117,
    Stat.DEF: 118,
    Stat.SPA: 119,
    Stat.SPD: 120,
    Stat.SPE: 121,
    Stat.HP: 122,
}
# STAT_HP, STAT_ATTACK, STAT_DEFENSE, STAT_SPEED, STAT_SPECIAL_ATTACK, STAT_SPECIAL_DEFENSE
# (generated/pokemon_stats.txt:1-6) as ids.Stat.
DECOMP_STATS = (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD)
SPECIES_SHEDINJA = 292  # Pokemon_CalcStats (src/pokemon.c:598, :634)
MAX_POKEMON_LEVEL = 100  # include/constants/pokemon.h:22
MAX_EVS_SINGLE_STAT, MAX_EVS_ALL_STATS = 255, 510  # include/constants/pokemon.h:15-16
MAX_FRIENDSHIP_VALUE = 255  # include/constants/pokemon.h:28
LOW_FRIENDSHIP_LIMIT, MED_FRIENDSHIP_LIMIT = 100, 200  # include/constants/pokemon.h:29-30
# sFriendshipChangeTable[FRIENDSHIP_EVENT_LEVEL_UP] (src/pokemon.c:2623).
LEVEL_UP_FRIENDSHIP = (5, 3, 2)


def _hold_effect(tables, mon: RefPokemon) -> int:
    """``Item_LoadParam(MON_DATA_HELD_ITEM, ITEM_PARAM_HOLD_EFFECT)``: the party item's."""
    return _t(tables, "item_hold_effect", mon.item)


def _pay_exp(turn: _Turn, s: RefState, pending: bool = False, selected: int = 0) -> bool:
    """``BattleControllerPlayer_AnyExpPayout`` (bcp.c:4449-4469), re-entered until no payout
    flag is left: ``subscript_grant_exp.s`` for the lowest flagged battler each time. Returns
    whether a player Pokemon gained a level."""
    leveled = False
    while turn.payout:
        slot = min(turn.payout)
        turn.payout.remove(slot)
        leveled |= _grant_exp(turn.tables, s, slot, pending, selected)
    return leveled


def ref_grant_exp(
    tables, state: RefState, slot: int, pending: bool = False, selected: int = 0
) -> RefState:
    """The payout for the opponent's party ``slot`` fainting (``engine.levelup.pay_exp``'s
    contract); the input state is not modified."""
    s = copy.deepcopy(state)
    _grant_exp(tables, s, slot, pending, selected)
    return s


def _grant_exp(tables, s: RefState, fainted: int, pending: bool, selected: int) -> bool:
    """``subscript_grant_exp.s``: ``BtlCmd_CalcExpGain`` (script.c:2187-2245), then
    ``BattleScript_GetExpTask`` (script.c:9700-10193) for every party slot in turn. The engine
    plays trainer singles only: the fainted battler is a ``SOLO_ENEMY`` (script.c:2196), the
    battle is a trainer battle (:9753-9755) and the payout goes to party 0 (:9716-9720).
    Returns whether a slot gained a level."""
    party = s.sides[0].party
    foe = s.sides[1].party[fainted]
    # CalcExpGain (script.c:2201-2212): the party Pokemon with a species and HP.
    gaining = with_share = 0
    for i, mon in enumerate(party):
        if mon.species and mon.hp:
            if s.exp_mask & (1 << i):
                gaining += 1
            if _hold_effect(tables, mon) == HOLD_EFFECT_EXP_SHARE:
                with_share += 1
    # script.c:2216-2217: the yield by species (SpeciesData_GetSpeciesValue), a u16.
    exp = (_t(tables, "species_exp_yield", foe.species) * foe.level // 7) & 0xFFFF
    # script.c:2219-2238. With no participant left (gaining == 0) the divide is made safe and
    # gainedExp reaches nobody (:9740-9743 pays it only to mask slots with HP), ruling D4.
    if with_share:
        gained = max((exp // 2) // gaining, 1) if gaining else 1
        shared = max((exp // 2) // with_share, 1)
    else:
        gained = max(exp // gaining, 1) if gaining else 1
        shared = 0
    leveled = False
    for slot, mon in enumerate(party):
        effect = _hold_effect(tables, mon)
        # script.c:9703-9712: the next slot in the mask or holding an Exp. Share.
        if not (effect == HOLD_EFFECT_EXP_SHARE or s.exp_mask & (1 << slot)):
            continue
        total = 0
        if mon.hp and mon.level != MAX_POKEMON_LEVEL:  # script.c:9740
            if s.exp_mask & (1 << slot):
                total = gained  # :9741-9743
            if effect == HOLD_EFFECT_EXP_SHARE:
                total += shared  # :9745-9747
            if effect == HOLD_EFFECT_EXP_UP:
                total = total * 150 // 100  # :9749-9751
            total = total * 150 // 100  # BATTLE_TYPE_TRAINER, :9753-9755
            # BattleSystem_PokemonIsOT / MON_DATA_LANGUAGE vs gGameLanguage (:9757-9763),
            # resolved by the host into exp_boost: 100 (OT), 150 or 170.
            if s.exp_boost[slot] != 100:
                total = total * s.exp_boost[slot] // 100
            s.exp[slot] += total  # :9767-9775
            _effort_values(tables, mon, foe, s.pokerus[slot])  # :9776-9779
        if total:  # :9782-9791; SEQ_GET_EXP_CHECK_LEVEL_UP, :9824-9836, once per level
            while _should_level_up(tables, s, slot):
                leveled = True
                # SEQ_GET_EXP_WAIT_LEVEL_UP_EFFECT (:9838-9877): friendship, CalcStats, and
                # ReloadPokemon for the active battler (lib.c:169-194; here party and battler
                # are one object).
                _level_up_friendship(tables, mon, s.friend_bonus[slot])
                _calc_stats(tables, mon)
                # The level's learnset scan (:9879-9883) is deferred to the learn resolve:
                # the first pending level is marked (design §3.2).
                if s.learn_level[slot] == 0:
                    s.learn_level[slot] = mon.level
                    s.learn_cursor[slot] = 0
                if pending and slot == s.sides[0].active:
                    s.learn_pin = selected + 1
        s.exp_mask &= ~(1 << slot)  # SEQ_GET_EXP_CHECK_DONE, :10190-10193
    return leveled


def _should_level_up(tables, s: RefState, slot: int) -> bool:
    """``Pokemon_ShouldLevelUp`` (src/pokemon.c:3501-3526): the growth rate by species
    (:3506); exp is clamped to the level-100 base and written back (:3507-3512); one level
    up when exp reaches the next level's base (:3518-3523)."""
    mon = s.sides[0].party[slot]
    rate = _t(tables, "species_exp_rate", mon.species)
    max_exp = _t(tables, "exp_table", rate, MAX_POKEMON_LEVEL)
    s.exp[slot] = min(s.exp[slot], max_exp)
    next_level = mon.level + 1
    if next_level > MAX_POKEMON_LEVEL:
        return False
    if s.exp[slot] >= _t(tables, "exp_table", rate, next_level):
        mon.level = next_level
        return True
    return False


def _level_up_friendship(tables, mon: RefPokemon, luxury_ball: int) -> None:
    """``Pokemon_UpdateFriendship(mon, FRIENDSHIP_EVENT_LEVEL_UP)`` (src/pokemon.c:
    2635-2689): the tier of the current value (:2651-2662), +1 for a Luxury Ball
    (:2664-2666), ``* 150 / 100`` with Soothe Bell (:2672-2676), clamped (:2680-2686). The
    egg-location +1 (:2668-2670) is not modelled (refused where read, design §4.4)."""
    tier = 0
    if mon.friendship >= LOW_FRIENDSHIP_LIMIT:
        tier += 1
    if mon.friendship >= MED_FRIENDSHIP_LIMIT:
        tier += 1
    change = LEVEL_UP_FRIENDSHIP[tier]
    if change > 0 and luxury_ball:
        change += 1
    if change > 0 and _hold_effect(tables, mon) == HOLD_EFFECT_FRIENDSHIP_UP:
        change = change * 150 // 100
    mon.friendship = min(max(mon.friendship + change, 0), MAX_FRIENDSHIP_VALUE)


def _calc_stats(tables, mon: RefPokemon) -> None:
    """``Pokemon_CalcStats`` (src/pokemon.c:564-648) at the Pokemon's level: base stats of its
    form (:591-595), max HP (1 for Shedinja, :597-601), the other stats with the nature
    (:607-628, ``Pokemon_GetNatureStatValue`` :2593-2615); a nonzero current HP gains the
    max HP's change (Shedinja: 1, :633-645)."""
    row = _t(tables, "species_form_row", mon.species, mon.form)
    level = mon.level

    def core(stat: int) -> int:
        base = _t(tables, "species_base_stats", row, stat)
        return (2 * base + mon.ivs[stat] + mon.evs[stat] // 4) * level // 100

    old_max = mon.max_hp
    new_max = 1 if mon.species == SPECIES_SHEDINJA else core(Stat.HP) + level + 10
    stats = []
    for stat in (Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE):
        value = core(stat) + 5
        affinity = _t(tables, "nature_mods", mon.nature, stat)
        if affinity == 1:
            value = value * 110 // 100
        elif affinity == -1:
            value = value * 90 // 100
        stats.append(value)
    mon.max_hp = new_max
    mon.stats = stats
    if mon.hp != 0 or old_max == 0:
        if mon.species == SPECIES_SHEDINJA:
            mon.hp = 1
        elif mon.hp == 0:
            mon.hp = new_max
        else:
            mon.hp += new_max - old_max


def _effort_values(tables, mon: RefPokemon, foe: RefPokemon, pokerus: int) -> None:
    """``BattleScript_CalcEffortValues`` (script.c:10214-10306) for ``mon`` beating ``foe``:
    the yields of the foe's form (:10224), the receiver's party item (:10226-10228)."""
    row = _t(tables, "species_form_row", foe.species, foe.form)
    effect = _hold_effect(tables, mon)
    power = _t(tables, "item_hold_param", mon.item)
    total = sum(mon.evs)
    for stat in DECOMP_STATS:
        if total >= MAX_EVS_ALL_STATS:  # :10237-10239
            break
        tmp = _t(tables, "species_ev_yield", row, stat)
        if effect == POWER_ITEM_HOLD_EFFECTS[stat]:  # :10244-10282
            tmp += power
        if pokerus:  # Pokemon_HasPokerus, :10285-10287
            tmp *= 2
        if effect == HOLD_EFFECT_EVS_UP_SPEED_DOWN:  # :10289-10291
            tmp *= 2
        if total + tmp > MAX_EVS_ALL_STATS:  # :10294-10296
            tmp -= total + tmp - MAX_EVS_ALL_STATS
        if mon.evs[stat] + tmp > MAX_EVS_SINGLE_STAT:  # :10299-10301
            tmp -= mon.evs[stat] + tmp - MAX_EVS_SINGLE_STAT
        mon.evs[stat] += tmp
        total += tmp


# --------------------------------------------------------------------------- learning a move

LEARNED_MOVES_MAX = 4  # include/constants/moves.h:8
LEARN_GIVE_UP = LEARNED_MOVES_MAX  # the learn option that gives the move up
# include/struct_defs/species.h:11-14.
LEARNSET_NO_MOVE_TO_LEARN = 0
LEARNSET_MOVE_ALREADY_KNOWN = 0xFFFE
LEARNSET_ALL_SLOTS_FILLED = 0xFFFF
LEARNSET_SENTINEL_ENTRY = 0xFFFF


def _learnset(tables, mon: RefPokemon) -> list[int]:
    """``Pokemon_LoadLevelUpMovesOf`` (src/pokemon.c:3943): the u16 entries ``move | level << 9``
    of the Pokemon's species and form, in source order, then the terminator."""
    row = _t(tables, "species_form_row", mon.species, mon.form)
    entries = []
    for i in range(tables.learnset_level.shape[1]):
        level = _t(tables, "learnset_level", row, i)
        if level == 0:  # the tables' end marker (a real entry has level 1..100)
            break
        entries.append(_t(tables, "learnset_move", row, i) | level << 9)
    return [*entries, LEARNSET_SENTINEL_ENTRY]


def _add_move(tables, mon: RefPokemon, move: int) -> int:
    """``BoxPokemon_AddMove`` (src/pokemon.c:3852-3873): the first empty slot learns the move
    (``BoxPokemon_SetMoveSlot`` :3925-3933, ``MoveTable_CalcMaxPP`` with that slot's PP Ups,
    src/move_table.c:25-35; an empty slot's PP Ups are 0, coverage refuses others), a slot
    already holding it stops the walk."""
    for i in range(LEARNED_MOVES_MAX):
        if mon.moves[i] == 0:
            mon.moves[i] = move
            mon.pp[i] = _t(tables, "move_pp", move)
            return move
        if mon.moves[i] == move:
            return LEARNSET_MOVE_ALREADY_KNOWN
    return LEARNSET_ALL_SLOTS_FILLED


def _level_up_move(tables, mon: RefPokemon, level: int, index: int) -> tuple[int, int, int]:
    """``Pokemon_LevelUpMove`` (src/pokemon.c:3935-3966) with the Pokemon at ``level``:
    ``(result, move, index)``. It walks the whole list from ``index`` to the terminator,
    skipping entries of other levels, and moves the index past a match."""
    entries = _learnset(tables, mon)
    if entries[index] == LEARNSET_SENTINEL_ENTRY:
        return LEARNSET_NO_MOVE_TO_LEARN, 0, index
    while (entries[index] & 0xFE00) != level << 9:
        index += 1
        if entries[index] == LEARNSET_SENTINEL_ENTRY:
            return LEARNSET_NO_MOVE_TO_LEARN, 0, index
    move = entries[index] & 0x1FF
    index += 1
    return _add_move(tables, mon, move), move, index


def _scan_learnsets(tables, s: RefState) -> None:
    """``SEQ_GET_EXP_CHECK_LEARN_MOVE`` (script.c:10003-10038) for every slot's deferred
    levels, slots ascending, until a prompt (``LEARNSET_ALL_SLOTS_FILLED``, :10016-10018):
    the cursor is left on the prompted entry. "Already known" goes to the next entry
    (:10008-10010), a learned move too (via ``SEQ_GET_EXP_LEARNED_MOVE_WAIT``, :10183-10187);
    no move left (:10012-10013) ends the level, and the next level's scan restarts at index 0
    (``GET_EXP_LEARNSET_INDEX = 0``, :9882)."""
    for slot, mon in enumerate(s.sides[0].party):
        while s.learn_level[slot]:
            result, _, index = _level_up_move(
                tables, mon, s.learn_level[slot], s.learn_cursor[slot]
            )
            if result == LEARNSET_ALL_SLOTS_FILLED:
                s.learn_cursor[slot] = index - 1
                return
            if result == LEARNSET_NO_MOVE_TO_LEARN:
                if s.learn_level[slot] == mon.level:
                    s.learn_level[slot] = s.learn_cursor[slot] = 0
                else:
                    s.learn_level[slot] += 1
                    s.learn_cursor[slot] = 0
            else:
                s.learn_cursor[slot] = index


def ref_advance_learns(tables, state: RefState) -> RefState:
    """``engine.levelup.advance_learns``' contract: the deferred scans up to the first
    prompt; the input state is not modified."""
    s = copy.deepcopy(state)
    _scan_learnsets(tables, s)
    return s


def _prompted(s: RefState) -> int | None:
    return next((i for i, level in enumerate(s.learn_level) if level), None)


def ref_learn_options(tables, state: RefState) -> list[bool]:
    """``engine.levelup.learn_options``: replacing a slot whose move is an HM is not allowed
    (``CheckSelectedMoveIsHM``, src/battle_sub_menus/battle_party.c:1724-1735;
    ``Item_IsHMMove``, src/item.c:306-315); the search also skips an AI-seen slot and, for the
    active learner, the disabled, locked and choice-locked moves (ruling D2) and the slot
    pinned by ``learn_pin`` (ruling U1), and every replacement when the offered move's effect
    is unsupported (ruling D10). Giving up is always allowed."""
    prompted = _prompted(state)
    slot = 0 if prompted is None else prompted
    side = state.sides[0]
    mon = side.party[slot]
    supported = True
    if prompted is not None:
        entries = _learnset(tables, mon)
        index = state.learn_cursor[slot]
        entry = entries[index] if index < len(entries) else LEARNSET_SENTINEL_ENTRY
        offered = 0 if entry == LEARNSET_SENTINEL_ENTRY else entry & 0x1FF
        supported = _t(tables, "move_effect", offered) != EffectId.UNSUPPORTED
    options = []
    for k in range(LEARNED_MOVES_MAX):
        move = mon.moves[k]
        ok = supported and not _t(tables, "move_is_hm", move) and not state.ai_seen_moves[slot][k]
        if slot == side.active and (
            move in (side.disabled_move, side.locked_move, side.choice_lock)
            or state.learn_pin == k + 1
        ):
            ok = False
        options.append(ok)
    return [*options, True]


def ref_learn(tables, state: RefState, option: int) -> RefState:
    """``engine.levelup.learn``: the answer to the pending prompt. Replacing slot ``k``
    (``SEQ_GET_EXP_LEARNED_MOVE``, script.c:10162-10181): PP Ups 0, then
    ``Pokemon_SetMoveSlot`` (src/pokemon.c:3920-3933) gives the base PP; giving up
    (:10132-10135) changes nothing. The scan then goes on from the next entry."""
    s = copy.deepcopy(state)
    slot = _prompted(s)
    mon = s.sides[0].party[slot]
    index = s.learn_cursor[slot]
    move = _learnset(tables, mon)[index] & 0x1FF
    if option != LEARN_GIVE_UP:
        mon.moves[option] = move
        mon.pp[option] = _t(tables, "move_pp", move)  # MoveTable_CalcMaxPP(move, 0)
    s.learn_cursor[slot] = index + 1
    _scan_learnsets(tables, s)
    return s


# --------------------------------------------------------------------------- items, switch-in


def _cure(s: RefState, side: int, effect: int) -> bool:
    """The cure of a status berry or White Herb ``effect`` on ``side``'s active Pokemon,
    as ``BattleSystem_TriggerHeldItem`` (lib.c:4608-4700, :4875-4887) and
    ``BattleSystem_PluckBerry`` (lib.c:5577-5675) choose it, and the restore subscript it
    runs (``subscript_held_item_{slp,psn,cnf,multi,...}_restore.s``). Returns whether it
    did anything."""
    mon, sd = _active(s, side), s.sides[side]
    # The status each status berry cures (lib.c:4601-4634; Pecha: MON_CONDITION_ANY_POISON).
    cures = {
        ItemId.CHERI_BERRY: (Status.PARALYSIS,),
        ItemId.CHESTO_BERRY: (Status.SLEEP,),
        ItemId.PECHA_BERRY: (Status.POISON, Status.TOXIC),
        ItemId.RAWST_BERRY: (Status.BURN,),
        ItemId.ASPEAR_BERRY: (Status.FREEZE,),
    }
    cured = False
    if mon.status in cures.get(effect, ()):
        cured = True
    elif effect == ItemId.PERSIM_BERRY and sd.confusion:  # lib.c:4655: confusion only
        sd.confusion = 0
        return True
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


def _raise_pinch_stat(turn: _Turn, s: RefState, side: int, item: int) -> bool:
    """A stat pinch berry's ``subscript_held_item_raise_stat.s`` on ``side``'s battler: its
    stat (``item_param[item, 1]``) +1 when below +6 (lib.c:4779, :5754; script.c:2841-2873).
    Returns whether it rose."""
    stat = _t(turn.tables, "item_param", item, 1)
    if s.sides[side].stages[stat] >= STAGE_MAX:
        return False
    _change_stage(s, side, stat, 1)
    return True


def _held_item(turn: _Turn, s: RefState, side: int, keep: bool = False) -> bool:
    """``BattleSystem_TriggerHeldItem`` (lib.c:4576) / ``..._OnStatus`` (lib.c:4967) for
    ``side``'s active Pokemon with HP left. Oran Berry and Berry Juice
    (``HOLD_EFFECT_HP_RESTORE``) heal their power, Sitrus Berry (``HP_PCT_RESTORE``)
    ``Divide(maxHP * power, 100)``, when ``curHP <= maxHP / 2``
    (``subscript_held_item_hp_restore.s``, capped); the status berries, Lum and White Herb
    cure (``_cure``); a stat pinch berry raises its stat at ``curHP <= maxHP / param``
    (``_raise_pinch_stat``). Every subscript ends in ``BATTLE_SUBSCRIPT_PLUCK_CHECK``, which removes
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
    elif effect == ItemId.PINCH_BERRY:
        # HOLD_EFFECT_PINCH_*_UP (lib.c:4773-4836): curHP <= maxHP / param (Gluttony halves it).
        if mon.hp > mon.max_hp // _pinch_divisor(tables, mon):
            return False
        if not _raise_pinch_stat(turn, s, side, mon.item):
            return False
    elif not _cure(s, side, effect):
        return False
    if not keep:
        mon.item_consumed = mon.item
        mon.item = 0
        _forget_item(s, side)
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
        _forget_item(s, side)
    return mon.hp - 1


def _sash_at_full_hp(turn: _Turn, s: RefState, side: int) -> bool:
    mon = _active(s, side)
    return _item_effect(turn.tables, mon) == ItemId.FOCUS_SASH and mon.hp == mon.max_hp


def _resist_berry(
    turn: _Turn, s: RefState, side: int, move_type: int, super_effective: bool
) -> bool:
    """``subscript_type_resist_berry.s`` (``subscript_update_hp.s:9``) on ``side``'s active
    Pokemon, the battler whose HP changes: a Chilan Berry (``HOLD_EFFECT_WEAKEN_NORMAL``,
    ``RESIST_ANY_NORMAL``) halves any hit whose type (``calcTemp``) is Normal (:_128), every
    other resist berry a super-effective hit of its type (:_135-_240); the berry is then
    removed. Returns
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
        _forget_item(s, side)
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
    # :5 SYSCTL_IGNORE_TYPE_CHECKS (Night Shade, Foresight, Super Fang), :6 an OHKO hit.
    if effect in (EffectId.LEVEL_DAMAGE, EffectId.FORESIGHT, EffectId.HALVE_HP, EffectId.OHKO):
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


def _entry_hazards(turn: _Turn, s: RefState, side: int) -> None:
    """``subscript_hazards_check.s``: nothing for Magic Guard (:5); a grounded newcomer (Gravity,
    or neither Levitate nor a Flying type, :6-11) meets Toxic Spikes (a Poison type absorbs them,
    one layer poisons, two badly poison: Immunity, Leaf Guard in sun, a status, Poison / Steel
    stop it; script.c:7629-7651, subscript_poison.s / _badly_poison.s:5-10) and Spikes
    (``Divide(maxHP, (5 - layers) * 2)`` with HP left, script.c:5434-5436); then Stealth Rock."""
    tables = turn.tables
    mon, sd = _active(s, side), s.sides[side]
    if mon.ability == AbilityId.MAGIC_GUARD:
        return
    # CheckAbility reads Battler_Ability: an ingrained Levitate is none (lib.c:3100-3103); a
    # Baton Pass newcomer keeps the Ingrain (lib.c:2079).
    levitate = mon.ability == AbilityId.LEVITATE and not s.sides[side].ingrain
    airborne = levitate or Type.FLYING in mon.types
    if s.gravity or not airborne:
        if sd.toxic_spikes:
            if Type.POISON in mon.types:
                sd.toxic_spikes = 0
            else:
                status = Status.TOXIC if sd.toxic_spikes >= 2 else Status.POISON
                ability = _decomp_ability(tables, mon)
                leaf = (
                    ability == ITEM_ABILITY_IDS["Leaf Guard"]
                    and s.weather == Weather.SUN
                    and not _weather_suppressed(tables, s)
                )
                if ability == ITEM_ABILITY_IDS["Immunity"] or leaf:
                    # "{0}'s {1} prevents poisoning!" (subscript_poison.s:80-82,
                    # _badly_poison.s:93-103): BattleAI_SetAbility (script.c:11806).
                    if side == 0:
                        s.ai_ability_known[sd.active] = 1
                elif _can_have_status(tables, s, side, status):
                    mon.status, mon.status_counter = status, 0
        if sd.spikes and mon.hp:
            _lose_hp(turn, s, side, _divide(mon.max_hp, (5 - sd.spikes) * 2))
    _stealth_rock_damage(turn, s, side)


def _stealth_rock_damage(turn: _Turn, s: RefState, side: int) -> None:
    """``subscript_hazards_check.s`` -> ``BtlCmd_CheckStealthRock`` (script.c:8302)."""
    mon = _active(s, side)
    # subscript_hazards_check.s:5: a Magic Guard newcomer skips every hazard.
    if not s.sides[side].stealth_rock or mon.hp == 0 or mon.ability == AbilityId.MAGIC_GUARD:
        return
    mul = _matchup_x40(turn.tables, Type.ROCK, mon.types)
    if mul == 0:  # TYPE_MULTI_IMMUNE
        return
    _lose_hp(turn, s, side, _divide(mon.max_hp, STEALTH_ROCK_DIVISOR[mul]))


def _arm_unburden(s: RefState, side: int) -> None:
    """RecoverStatusByAbility's ABILITY_UNBURDEN case (lib.c:4433-4437): an Unburden holder
    holding an item gets canUnburden."""
    mon = _active(s, side)
    if mon.ability == AbilityId.UNBURDEN and mon.item:
        s.sides[side].can_unburden = 1


def _drop_blocked(seen: int, stat: int, delta: int) -> bool:
    """BtlCmd_ChangeStatStage (script.c:2884-2902): a drop by another battler meets the target's
    Clear Body, Keen Eye on accuracy or Hyper Cutter on Attack (ignorable, ``seen``)."""
    if delta >= 0:
        return False
    return (
        seen == AbilityId.CLEAR_BODY
        or (seen == AbilityId.KEEN_EYE and stat == Stat.ACC)
        or (seen == AbilityId.HYPER_CUTTER and stat == Stat.ATK)
    )


def _recover_status_by_ability(s: RefState, side: int) -> None:
    """``BattleSystem_RecoverStatusByAbility`` (lib.c:4378) for ``side``'s battler: an
    asleep Insomnia holder (:4404) or a paralyzed Limber holder (:4397) with HP left is
    cured, and subscript_ability_forbids_status.s tags its ability (BattleAI_SetAbility,
    script.c:11806). Run at each move end for the attacker then the defender
    (bcp.c:3965-3967) and on a switch-in (lib.c:4135-4145)."""
    mon = _active(s, side)
    _arm_unburden(s, side)
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
    tables = turn.tables
    other = 1 - side
    mon, foe = _active(s, side), _active(s, other)
    # SWITCH_IN_CHECK_STATE_TRACE (lib.c:3718-3737): the newcomer, then the battler facing it,
    # whose Trace stayed armed while its foe was down; no two Trace holders copy each other.
    _trace(tables, s, side, other)
    traced = _trace(tables, s, other, side)
    # SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES (lib.c:3740-3790).
    for battler in (side, other) if traced else (side,):
        weather = WEATHER_ABILITIES.get(_active(s, battler).ability)
        if weather is not None and _active(s, battler).hp:
            if not (s.weather == weather and s.weather_timer == 0) and battler == 0:
                s.ai_ability_known[s.sides[0].active] = 1  # subscript_drought.s etc.
            s.weather, s.weather_timer = weather, 0
    # subscript_intimidate.s: ATTACK_DOWN_1_STAGE on a living foe (Clear Body, Hyper
    # Cutter, White Smoke, Mist and Substitute are unsupported). Its "cuts" message is
    # built in C (BtlCmd_ChangeStatStage, script.c:2956): the AI learns nothing.
    blockers = (AbilityId.CLEAR_BODY, AbilityId.HYPER_CUTTER)  # script.c:2884-2902 (B9-6)
    if mon.hp and mon.ability == AbilityId.INTIMIDATE and foe.hp and foe.ability not in blockers:
        _change_stage(s, other, Stat.ATK, -1)
    if (
        traced
        and foe.hp
        and foe.ability == AbilityId.INTIMIDATE
        and mon.hp
        and mon.ability not in blockers
    ):
        _change_stage(s, side, Stat.ATK, -1)
    # SWITCH_IN_CHECK_STATE_MOLD_BREAKER (lib.c:4075): subscript_mold_breaker.s tags the
    # living newcomer's ability, BattleAI_SetAbility (script.c:11806).
    if side == 0 and mon.hp and mon.ability == AbilityId.MOLD_BREAKER:
        s.ai_ability_known[s.sides[0].active] = 1
    if traced and other == 0 and foe.ability == AbilityId.MOLD_BREAKER:
        s.ai_ability_known[s.sides[0].active] = 1
    _recover_status_by_ability(s, side)
    if traced:
        _recover_status_by_ability(s, other)
    _arm_unburden(s, other)  # FORBIDDEN_STATUS runs for every battler (lib.c:4135-4145)
    for battler in (0, 1):
        _held_item_hp(turn, s, battler)


def _trace(tables, s: RefState, side: int, foe: int) -> bool:
    """``SWITCH_IN_CHECK_STATE_TRACE`` for ``side``'s battler (lib.c:3718-3737, ``ChooseTraceTarget``
    :7750-7775): with HP, Trace, no Griseous Orb, a living foe whose ability is not Forecast,
    Trace or Multitype; ``subscript_trace.s`` copies the foe's ability to the battler and the
    AI learns both (script.c:3403, :11806)."""
    mon, target = _active(s, side), _active(s, foe)
    if not (
        mon.hp
        and mon.ability == AbilityId.TRACE
        and mon.item != ITEM_GRISEOUS_ORB
        and target.hp
        and _decomp_ability(tables, target) not in UNTRACEABLE_ABILITIES
    ):
        return False
    mon.ability = target.ability
    for battler in (side, foe):
        if battler == 0:
            s.ai_ability_known[s.sides[0].active] = 1
    return True


def _enter(turn: _Turn, s: RefState, side: int, slot: int, baton: bool = False) -> None:
    """``SwitchAndUpdateMon`` -> ``BattleSystem_InitBattleMon`` (lib.c:70): a fresh
    battler (stages neutral, no volatiles; no Baton Pass in M1), and
    ``BattleSystem_UpdateAfterSwitch`` (lib.c:2094) sets ``fakeOutTurnNumber =
    totalTurns + 1`` (lib.c:2117), clears ``moveHit`` (lib.c:2121) and ends what the outgoing
    battler is the source of (the Bind on the foe, the foe's infatuation, lib.c:2088-2098).
    The caller then runs ``PokemonSendOut`` (``_send_out_exp``) and the entry hazards
    (``_stealth_rock_damage``), in that order."""
    sd = s.sides[side]
    _restore_types(turn.tables, s, side)  # the outgoing battler's Camouflage ends
    _restore_ability(turn.tables, s, side)  # and its traced ability
    sd.active = slot
    _reset_battler(sd, baton)
    sd.fake_out_turn = turn.entry_turn
    # lib.c:2151-2153: the AI forgets the battler's moves, ability and held item (it only
    # tracks the player's).
    if side == 0:
        s.ai_seen_moves[slot] = [0] * len(s.ai_seen_moves[slot])
        s.ai_ability_known[slot] = 0
        s.ai_item_known[slot] = 0
    _source_leaves(s, side, baton)


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


def _ghost(s: RefState, side: int) -> bool:
    """``side``'s active battler has the Ghost type (effect_script_0109.s:5-6)."""
    return Type.GHOST in _active(s, side).types


def _curse_on_user(tables, move: int, ghost: bool = False) -> bool:
    """A non-Ghost user's Curse (a Ghost user's targets the foe, B8-5) targets its user although
    its data range is RANGE_SINGLE_TARGET: ``BattleControllerPlayer_MustSelectTarget``
    (bcp.c:4344-4346) gives it ``RANGE_USER``, so the chosen target is the user (:4363) and
    ``BattleSystem_Defender`` (lib.c:1709-1718) returns it: like a ``TARGETS_USER`` move."""
    return _t(tables, "move_effect", move) == EffectId.CURSE and not ghost


def _targets_user(tables, move: int, ghost: bool = False) -> bool:
    return _move_target(tables, move) in TARGETS_USER or _curse_on_user(tables, move, ghost)


def _defender_always(tables, move: int, ghost: bool = False) -> bool:
    return _move_target(tables, move) in DEFENDER_ALWAYS or _curse_on_user(tables, move, ghost)


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
    return _move_target(tables, move) not in SELF_DEFENDER and not _curse_on_user(
        tables, move, _ghost(s, side)
    )


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
    # Shield Dust (ignorable) stops an indirect effect silently (subscript_paralyze.s:13,
    # subscript_burn.s:27, subscript_freeze.s:11, subscript_poison.s:20, ...).
    if not direct and _ignorable(s, side, target) == AbilityId.SHIELD_DUST:
        return
    # subscript_fall_asleep.s:43-45: a target not asleep yet stays awake while any battler is in
    # an uproar (FIELD_CONDITION_UPROAR; Soundproof is unsupported).
    if status == Status.SLEEP and mon.status != Status.SLEEP and _uproar_on(s):
        return
    if mon.hp and _can_have_status(turn.tables, s, target, status):
        mon.status = status
        mon.status_counter = turn.outcome.duration[side] if status == Status.SLEEP else 0
        # subscript_fall_asleep.s:_176 and subscript_freeze.s:_073 unlock the new sleeper
        # or ice cube.
        if status in (Status.SLEEP, Status.FREEZE):
            _unlock(s.sides[target])


def _uproar_on(s: RefState) -> bool:
    """FIELD_CONDITION_UPROAR: a battler's bit is set while it is in an uproar."""
    return any(sd.uproar for sd in s.sides)


def _life_orb_recoil(turn: _Turn, s: RefState, side: int) -> None:
    """Life Orb (``HOLD_EFFECT_HP_DRAIN_ON_ATK``) after a move whose hit dealt damage
    (bcp.c:4758-4772; Pursuit's hit: lib.c:7357-7366): a living holder without Magic Guard
    (decomp ability id; unsupported) loses ``Divide(maxHP, 10)`` through
    ``subscript_lose_hp_from_item.s`` (no resist berry, no Bide memory)."""
    mon = _active(s, side)
    if _item_effect(turn.tables, mon) != ItemId.LIFE_ORB or mon.hp == 0:
        return
    if _decomp_ability(turn.tables, mon) == WEATHER_ABILITY_IDS["Magic Guard"]:
        return
    _lose_hp(turn, s, side, _divide(mon.max_hp, 10))


def _holds_choice(tables, s: RefState, side: int) -> bool:
    return _item_effect(tables, _active(s, side)) in CHOICE_ITEMS


def _update_choice_lock(
    turn: _Turn, s: RefState, side: int, defender: int | None, move: int, exempt: bool = False
) -> None:
    """``UpdateFlagsWhenHit`` (bcp.c:4489-4496) then ``UpdateMoveBuffers`` (bcp.c:3921-3934),
    when the attack message was shown: the defending foe's lock is cleared unless it holds a
    Choice item; the user is locked into ``moveTemp`` (``move``; not Struggle, not ``exempt``: a
    U-turn under SYSCTL_UTURN_ACTIVE or a Baton Pass that succeeded, bcp.c:3927-3928) while it
    holds one, else its lock is cleared. ``defender``:
    the defending side (``DEFENDING_MON``, bcp.c:4488-4496), None when the foe is not it."""
    if not turn.attack_message:
        return
    if defender is not None and not _holds_choice(turn.tables, s, defender):
        s.sides[defender].choice_lock = 0
    if not _holds_choice(turn.tables, s, side):
        s.sides[side].choice_lock = 0
    elif move != MOVE_STRUGGLE and not exempt:
        s.sides[side].choice_lock = move


def _pluck_berry(turn: _Turn, s: RefState, side: int, target: int) -> None:
    """``BtlCmd_TryPluck`` (script.c:8116) -> ``BattleSystem_PluckBerry`` (lib.c:5539), by
    the item's pluck effect (res/items/data/*.json): the user eats the foe's Oran
    (``PLUCK_EFFECT_HP_RESTORE``) / Sitrus (``_HP_PCT_RESTORE``) Berry, healing (capped)
    unless at full HP, or its Chesto / Pecha / Lum (``_SLP_RESTORE`` / ``_PSN_RESTORE`` /
    ``_ALL_RESTORE``), curing as the held berry would, or a resist berry or a berry with no
    hold effect (``PLUCK_EFFECT_NONE`` berries, no effect); the berry is removed
    (``RemoveItem BTLSCR_DEFENDER``) even when the foe fainted. Berry Juice and the other
    implemented items have ``PLUCK_EFFECT_NONE`` and are no berries (lib.c:5833): they
    stay. ``SELF_TURN_FLAG_PLUCK_BERRY`` (lib.c:5848) stays set when the berry did
    nothing for the eater (no restore subscript ran its PLUCK_CHECK). ``target``: the
    defender's side, whose berry is eaten."""
    tables = turn.tables
    user, foe = _active(s, side), _active(s, target)
    # BtlCmd_TryPluck (script.c:8125-8127): nothing while the foe holds an item and its Custap
    # Berry flag (set by this turn's order) is up.
    if foe.item and turn.custap[target]:
        return
    effect = _item_effect(tables, foe)
    power = _t(tables, "item_param", foe.item, 0)
    if effect == ItemId.ORAN_BERRY:
        heal = power
    elif effect == ItemId.SITRUS_BERRY:
        heal = _divide(user.max_hp * power, 100)
    # The resist berries, Custap and the berries with no hold effect: PLUCK_EFFECT_NONE
    # berries, eaten with no effect (Item_IsBerry, :5833). The status berries cure, a stat
    # pinch berry raises the eater's stat (PLUCK_EFFECT_*_UP, lib.c:5753-5796; no HP test).
    elif effect in (
        ItemId.CHERI_BERRY,
        ItemId.CHESTO_BERRY,
        ItemId.PECHA_BERRY,
        ItemId.RAWST_BERRY,
        ItemId.ASPEAR_BERRY,
        ItemId.PERSIM_BERRY,
        ItemId.LUM_BERRY,
        ItemId.PINCH_BERRY,
        ItemId.RESIST_BERRY,
        ItemId.CUSTAP_BERRY,
        ItemId.NO_EFFECT_BERRY,
    ):
        heal = 0
    else:
        return
    item = foe.item
    foe.item_consumed = foe.item
    foe.item = 0
    _forget_item(s, target)  # subscript_pluck.s:7 names it, :10 removes it
    if heal:
        did = user.hp != user.max_hp
        if did:
            _gain_hp(s, side, heal)
    elif effect == ItemId.PINCH_BERRY:
        did = _raise_pinch_stat(turn, s, side, item)
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
    if effect == EffectId.SECRET_POWER:
        # effect_script_0197.s GetTerrainSecondaryEffect (script.c:8063-8075).
        row = _terrain_row(s)
        if row is None:
            effect = EffectId.PLAIN
        else:
            effect, (p0, p1, _) = SECRET_POWER_EFFECTS[row]
    chance = _t(tables, "move_effect_chance", move)
    certain = chance == 0
    proc = certain or o.secondary[side][hit] != 0
    target_alive = _active(s, target).hp > 0
    seen = _ignorable(s, side, target)

    if effect == EffectId.STAT_SELF and landed:
        p2 = _t(tables, "move_param", move, 2)
        if p2 == MINIMIZE_PARAM and p0 == Stat.EVA:  # subscript_minimize.s:5 (B9-10)
            s.sides[side].minimized = 1
        _change_stage(s, side, p0, p1)
        if p2 == 1 and p0 == Stat.DEF:  # Defense Curl (effect_script_0156.s)
            s.sides[side].curled = 1
    elif effect == EffectId.STAT_FOE and landed and target_alive:
        if not _drop_blocked(seen, p0, p1):
            _change_stage(s, target, p0, p1)
    elif effect == EffectId.SECONDARY_STAT_SELF and effective and proc:
        if certain or _active(s, side).hp > 0:
            _change_stage(s, side, p0, p1)
    elif effect == EffectId.SECONDARY_STAT_FOE and effective and proc:
        # BtlCmd_ChangeStatStage (script.c:2918-2921): Shield Dust stops an indirect drop.
        blocked = _drop_blocked(seen, p0, p1) or seen == AbilityId.SHIELD_DUST
        if (certain or target_alive) and not blocked:
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
        blocked = seen in (AbilityId.INNER_FOCUS, AbilityId.SHIELD_DUST)  # flinch_mon.s:7, :9
        if o.fang_flinch[side] and alive and not target_moved and not blocked:
            s.sides[target].flinched = 1
    elif effect in (EffectId.FLINCH, EffectId.FAKE_OUT) and effective and proc:
        # subscript_flinch_mon.s: a living target that has not moved yet; Inner Focus,
        # whose message (:18-23, TAG_NICKNAME_ABILITY) is shown for Fake Out only.
        if target_alive and not target_moved:
            if seen == AbilityId.INNER_FOCUS:
                if move == MOVE_FAKE_OUT and target == 0:
                    s.ai_ability_known[s.sides[0].active] = 1
            elif seen != AbilityId.SHIELD_DUST:  # subscript_flinch_mon.s:8-9, indirect
                s.sides[target].flinched = 1
    elif effect == EffectId.RAISE_ALL_STATS_HIT and effective and proc:
        # effect_script_0140.s: indirect TO_ATTACKER at the move's chance; the recipient,
        # the user, must have HP (lib.c:1570-1586). subscript_boost_all_stats.s.
        if _active(s, side).hp > 0:
            for stat in (Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD):
                _change_stage(s, side, stat, 1)
    elif effect == EffectId.CURSE and landed and _ghost(s, side):
        # effect_script_0109.s:9-17 (a Ghost user) -> subscript_curse_ghost.s on the foe: fails
        # behind a substitute or on a cursed foe (:3-4); else the foe's curse, and the user loses
        # Divide(maxHP, 2) through UPDATE_HP (:6-12).
        foe_sd = s.sides[target]
        if not foe_sd.substitute_hp and not foe_sd.curse:
            foe_sd.curse = 1
            _lose_hp(turn, s, side, _divide(_active(s, side).max_hp, 2))
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
        # subscript_recoil_1_{2,3,4}.s: Divide(hitDamage, 2 / 3 / 4); Rock Head and Magic Guard
        # skip it (:5-6).
        if _active(s, side).ability not in (AbilityId.ROCK_HEAD, AbilityId.MAGIC_GUARD):
            _lose_hp(turn, s, side, _divide(dealt, p0))
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
        # (lib.c:1556); subscript_confuse.s fails on a confused target (Shield Dust stops it,
        # :15); Random 3, 2 turns.
        if target_alive and not s.sides[target].confusion and seen != AbilityId.SHIELD_DUST:
            s.sides[target].confusion = o.duration[side]
    elif effect == EffectId.MUD_SPORT and landed and not s.sides[side].mud_sport:
        # effect_script_0201.s: fails when the user already has it.
        s.sides[side].mud_sport = 1
    elif effect == EffectId.AQUA_RING and landed and not s.sides[side].aqua_ring:
        # effect_script_0251.s: fails when the user already has it.
        s.sides[side].aqua_ring = 1
    elif effect == EffectId.YAWN and landed and _active(s, target).hp:
        # subscript_yawn.s: an ability that blocks it is tagged; else unless it fails the count 2.
        fails, by_ability = _yawn_blocks(tables, s, side, target)
        if by_ability and target == 0:
            s.ai_ability_known[s.sides[0].active] = 1
        if not fails:
            s.sides[target].yawn = 2
    elif effect == EffectId.PERISH_SONG and landed:
        # BtlCmd_TryPerishSong (script.c:5465-5487): every living battler without the effect.
        for k in (0, 1):
            if not s.sides[k].perish_song and _active(s, k).hp:
                s.sides[k].perish_song, s.sides[k].perish_song_turns = 1, 3
    elif effect == EffectId.LOCK_ON and landed:
        # subscript_lock_on.s: a living defender without a substitute gets the counter 2 and
        # lockOnTarget = the attacker (:7-9), whatever it had.
        foe_sd = s.sides[target]
        if _active(s, target).hp and not foe_sd.substitute_hp:
            foe_sd.lock_on, foe_sd.lock_on_target = 2, 1 + side
    elif effect == EffectId.INGRAIN and landed and not s.sides[side].ingrain:
        s.sides[side].ingrain = 1  # effect_script_0181.s:6
    elif effect == EffectId.IMPRISON and landed and not s.sides[side].imprison:
        s.sides[side].imprison = 1  # BtlCmd_TryImprison (script.c:6724)
    elif effect == EffectId.MEAN_LOOK and landed:
        # subscript_mean_look.s (CHECK_HP | TO_DEFENDER): a living defender not trapped by it
        # and without a substitute; meanLookTarget gets the attacker (:9-10).
        foe_sd = s.sides[target]
        if _active(s, target).hp and not foe_sd.mean_look and not foe_sd.substitute_hp:
            foe_sd.mean_look = 1
            # :10 ORs the attacker into the never-cleared 2-bit field (B7-15).
            foe_sd.mean_look_target = 1 + (max(foe_sd.mean_look_target - 1, 0) | side)
    elif effect in (EffectId.LIGHT_SCREEN, EffectId.REFLECT) and landed:
        # BtlCmd_TryLightScreen / TryReflect (script.c:4214, :4257): fail when already up,
        # else NUM_SCREEN_TURNS, plus the user's Light Clay param (script.c:4229, :4272).
        name = "light_screen" if effect == EffectId.LIGHT_SCREEN else "reflect"
        if not getattr(s.sides[side], name):
            user = _active(s, side)
            extra = 0
            if _item_effect(tables, user) == ItemId.LIGHT_CLAY:
                extra = _t(tables, "item_param", user.item, 0)
            setattr(s.sides[side], name, 5 + extra)
    elif effect == EffectId.BRICK_BREAK and landed:
        # lib.c:1481-1510: applies on an immune target, not on a miss; script.c:6550.
        s.sides[target].reflect = s.sides[target].light_screen = 0
    elif effect == EffectId.EAT_BERRY and effective:
        # effect_script_0224.s / subscript_pluck.s: an ON_HIT effect, no HP check.
        _pluck_berry(turn, s, side, target)
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
    elif effect == EffectId.WEATHER and landed:
        # effect_script_0115.s, _0136.s, _0137.s, _0164.s: fails on its weather (Hail only on
        # a temporary one); else that weather for 5 turns plus the user's matching rock.
        if not _weather_fails(s, p0, p1):
            user = _active(s, side)
            extra = 0
            if (
                _item_effect(tables, user) == ItemId.WEATHER_ROCK
                and _t(tables, "item_param", user.item, 1) == p0
            ):
                extra = _t(tables, "item_param", user.item, 0)
            s.weather, s.weather_timer = p0, WEATHER_TURNS + extra
    elif effect == EffectId.DOUBLE_POWER and effective and p1 == 1:
        # effect_script_0171.s:12: an ON_HIT indirect subscript_heal_target_paralysis.s
        # against a defender paralyzed when the script ran: cured when it has HP left (:5-6).
        foe = _active(s, target)
        if foe.hp and foe.status == Status.PARALYSIS:
            foe.status, foe.status_counter = Status.NONE, 0
    elif effect == EffectId.SWAGGER and landed and target_alive:
        # subscript_swagger.s: Attack +2 below +6 (:6), then subscript_confuse.s unless
        # confused (:13; Random 3, 2 turns).
        if s.sides[target].stages[Stat.ATK] < 6:
            _change_stage(s, target, Stat.ATK, 2)
        if not s.sides[target].confusion:
            s.sides[target].confusion = o.duration[side]
    elif effect == EffectId.ROOST and landed:
        # subscript_roost.s: Divide(maxHP, 2) unless at full HP, then turnFlags.roosting.
        user = _active(s, side)
        if user.hp != user.max_hp:
            _gain_hp(s, side, _divide(user.max_hp, 2))
            turn.roosting[side] = True
    elif effect == EffectId.TAILWIND and landed and not s.sides[side].tailwind:
        s.sides[side].tailwind = TAILWIND_TURNS  # subscript_tailwind_start.s:11
    elif effect == EffectId.SPIKES and landed and s.sides[target].spikes < SPIKES_MAX:
        s.sides[target].spikes += 1  # BtlCmd_TrySpikes (script.c:5388-5403)
    elif effect == EffectId.TOXIC_SPIKES and landed:
        if s.sides[target].toxic_spikes < TOXIC_SPIKES_MAX:  # script.c:7592-7605
            s.sides[target].toxic_spikes += 1
    elif effect == EffectId.DESTINY_BOND and landed:
        s.sides[side].destiny_bond = 1  # subscript_destiny_bond.s:9
    elif effect == EffectId.UPROAR and effective and not s.sides[side].uproar:
        # effect_script_0159.s:5-6: an ON_HIT indirect subscript_uproar.s for a user not in an
        # uproar (lib.c:1524-1529: not NO_EFFECTS): moveLockedInto = moveCur, Random 3, 3 turns
        # and its FIELD_CONDITION_UPROAR bit.
        s.sides[side].locked_move = move
        s.sides[side].uproar = o.duration[side]
    elif effect == EffectId.DRAGON_DANCE and landed and _active(s, side).hp > 0:
        # subscript_user_atk_and_{speed,def}_up_1_stage.s:13-16: Attack +1, then Speed / Defense
        # (move_param 0) +1 (B9-10).
        _change_stage(s, side, Stat.ATK, 1)
        _change_stage(s, side, p0, 1)
    elif effect == EffectId.BELLY_DRUM and landed:
        # subscript_belly_drum.s: Attack to +6, then Divide(maxHP, 2) through UPDATE_HP (a failing
        # one ended at the failure check before AfterMoveEffects).
        user = _active(s, side)
        s.sides[side].stages[Stat.ATK] = 6
        _lose_hp(turn, s, side, _divide(user.max_hp, 2))
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


def _weather_fails(s: RefState, weather: int, permanent_too: int) -> bool:
    """A weather move fails: its weather is up, either kind (``permanent_too``) or only a
    temporary one (Hail, effect_script_0164.s)."""
    return s.weather == weather and (permanent_too == 1 or s.weather_timer != 0)


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
    if ability == AbilityId.AFTERMATH:
        # lib.c:4358-4371: the defender fainted to this hit (faintedMon), the attacker has no
        # Magic Guard: subscript_aftermath.s takes Divide(maxHP, 4) and tags the holder (B9-7).
        if _active(s, target).hp == 0 and attacker.ability != AbilityId.MAGIC_GUARD:
            _lose_hp(turn, s, side, _divide(attacker.max_hp, 4))
            if target == 0:
                s.ai_ability_known[s.sides[0].active] = 1
        return
    if ability == AbilityId.ROUGH_SKIN:
        # lib.c:4259: no subscript (and no ability tag) for a Magic Guard attacker.
        if attacker.ability != AbilityId.MAGIC_GUARD:
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
    """``AFTER_MOVE_EFFECT_THAW_DEFENDER`` (bcp.c:3708-3727), after the hit's item checks and
    Sticky Barb transfer: a living frozen defender hit for damage by a Fire-type move thaws."""
    mon = _active(s, 1 - side)
    fire = _move_type(turn.tables, s, side, move) == Type.FIRE
    if mon.status == Status.FREEZE and mon.hp and dealt > 0 and fire:
        mon.status = Status.NONE
        mon.status_counter = 0


def _sticky_barb_transfer(turn: _Turn, s: RefState, side: int, foe: int, move: int) -> None:
    """``BattleSystem_TriggerHeldItemOnHit`` (lib.c:5386-5398) after a hit of ``side``'s move
    once the defender took damage this action (the caller checks it): a defender holding a
    ``HOLD_EFFECT_DMG_USER_CONTACT_XFR`` item gives it to a living attacker holding nothing,
    on a contact move (``subscript_transfer_sticky_barb.s``; no RemoveItem)."""
    tables = turn.tables
    user, holder = _active(s, side), _active(s, foe)
    if _item_effect(tables, holder) != ItemId.STICKY_BARB:
        return
    if user.hp and user.item == 0 and _t(tables, "move_flags", move) & MoveFlag.CONTACT:
        user.item, holder.item = holder.item, 0
        _forget_item(s, foe)  # UpdateMonData's CopyToParty (script.c:3077-3078)


def _shell_bell(turn: _Turn, s: RefState, side: int, dealt: int) -> None:
    """Shell Bell (``HOLD_EFFECT_HP_RESTORE_ON_DMG``) after a move that set
    ``SYSCTL_MOVE_HIT`` (bcp.c:4736-4752; Pursuit: lib.c:7345-7355): with ``dealt``
    (``shellBellDamageDealt``) nonzero, a living holder below full HP regains
    ``Divide(dealt, param)`` (``subscript_restore_a_little_hp.s``, capped)."""
    mon = _active(s, side)
    if _item_effect(turn.tables, mon) != ItemId.SHELL_BELL or not dealt:
        return
    if mon.hp and mon.hp < mon.max_hp:
        _reveal_item(s, side)  # subscript_restore_a_little_hp.s:10
        _gain_hp(s, side, _divide(dealt, _t(turn.tables, "item_param", mon.item, 0)))


def _leftovers_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_LEFTOVERS`` (bcp.c:1369) -> ``BattleSystem_TriggerLeftovers``
    (lib.c:4922-4965): Leftovers, or Black Sludge on a Poison type, restore
    ``Divide(maxHP, 16)`` below full HP; Black Sludge on any other type hurts
    ``Divide(maxHP, 8)`` unless Magic Guard (decomp id). HP left needed."""
    tables = turn.tables
    mon = _active(s, side)
    if not mon.hp:
        return
    effect = _item_effect(tables, mon)
    if effect == ItemId.LEFTOVERS or (effect == ItemId.BLACK_SLUDGE and Type.POISON in mon.types):
        if mon.hp < mon.max_hp:
            _reveal_item(s, side)  # subscript_restore_a_little_hp.s:10
            _gain_hp(s, side, _divide(mon.max_hp, 16))
    elif (
        effect == ItemId.BLACK_SLUDGE
        and _decomp_ability(tables, mon) != WEATHER_ABILITY_IDS["Magic Guard"]
    ):
        _reveal_item(s, side)  # subscript_lose_hp_from_item_with_message.s:10
        _lose_hp(turn, s, side, _divide(mon.max_hp, 8))


def _detrimental_item_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT`` (bcp.c:1710) ->
    ``BattleSystem_TriggerDetrimentalHeldItem`` (lib.c:5307-5350) on a holder with HP left:
    Toxic Orb runs ``subscript_badly_poison.s``, Flame Orb ``subscript_burn.s``, as held-item
    side effects (skipped for Immunity / Water Veil, Leaf Guard in sun without a weather lock,
    any status, a Poison or Steel / a Fire type); Sticky Barb takes ``Divide(maxHP, param)``
    unless Magic Guard."""
    tables = turn.tables
    mon = _active(s, side)
    if not mon.hp:
        return
    effect = _item_effect(tables, mon)
    ability = _decomp_ability(tables, mon)
    sun = s.weather == Weather.SUN and not _weather_suppressed(tables, s)
    leaf_guard = sun and ability == ITEM_ABILITY_IDS["Leaf Guard"]
    if effect == ItemId.TOXIC_ORB:
        if ability == ITEM_ABILITY_IDS["Immunity"] or leaf_guard:
            return
        if _can_have_status(tables, s, side, Status.TOXIC):
            mon.status, mon.status_counter = Status.TOXIC, 0
    elif effect == ItemId.FLAME_ORB:
        if ability == ITEM_ABILITY_IDS["Water Veil"] or leaf_guard:
            return
        if _can_have_status(tables, s, side, Status.BURN):
            mon.status, mon.status_counter = Status.BURN, 0
    elif effect == ItemId.STICKY_BARB:
        if ability != WEATHER_ABILITY_IDS["Magic Guard"]:
            _reveal_item(s, side)  # subscript_lose_hp_from_item_with_message.s:10
            _lose_hp(turn, s, side, _divide(mon.max_hp, _t(tables, "item_param", mon.item, 0)))


def _hit(turn: _Turn, s: RefState, side: int, move: int, hit: int, struggling: bool,
         target_moved: bool) -> tuple[int, bool]:  # fmt: skip
    """One hit: the move script (damage, ``BATTLE_SUBSCRIPT_UPDATE_HP``), then
    ``BattleControllerPlayer_AfterMoveMessage`` (bcp.c:3467): the secondary effect
    (``ONE_HIT_TRIGGER_SECONDARY`` / ``MULTI_HIT_TRIGGER_SECONDARY``), then the
    contact ability (``*_TRIGGER_ABILITY``), then King's Rock (``*_EXTRA_FLINCH``). Returns
    ``(dealt, update_hp)``: the HP the target lost, and whether ``UpdateHP`` ran."""
    tables, o = turn.tables, turn.outcome
    target = 1 - side
    target_mon = _active(s, target)
    # The move reaches its defender (BattleSystem_Defender): the living foe, or the
    # defender a DEFENDER_ALWAYS move always has; a TARGETS_USER move hits its user.
    landed = target_mon.hp > 0 or _defender_always(tables, move, _ghost(s, side))
    # Bide is a RANGE_USER move whose unleash hits the foe (bcp.c:2702-2713).
    counter = _t(tables, "move_effect", move) == EffectId.COUNTER
    hits_foe = (
        not _targets_user(tables, move, _ghost(s, side))
        or _t(tables, "move_effect", move) == EffectId.BIDE
        or counter
    )

    if counter:
        # The command's battleCtx->damage under SYSCTL_IGNORE_TYPE_CHECKS: an immunity still
        # flags (lib.c:7550-7551). No Counter-family move is RANGE_USER in either game (Counter,
        # Mirror Coat, Metal Burst: RANGE_SINGLE_TARGET_SPECIAL; Kaizo's Bide: single target).
        kind = _move_type(tables, s, side, move)
        immune = _matchup_x40(
            tables, kind, target_mon.types, _identified(s, side, target)
        ) == 0 or _levitated(tables, s, side, target, kind)
        damage = 0 if immune else turn.counter_damage
    else:
        damage = _move_damage(
            tables,
            s,
            side,
            move,
            o.crit[side][hit] != 0 and not _armored(s, side, target),
            o.roll[side][hit],
            # Payback: movePower x2 once the target acted (script.c:7198-7209, B9-12).
            power_mul=20
            if _t(tables, "move_effect", move) == EffectId.DOUBLE_POWER
            and _t(tables, "move_param", move, 0) == POWER_IF_TARGET_MOVED
            and target_moved
            else 10,
            magnitude=o.magnitude[side],
            present=o.present[side],
        )
    dealt = stored = 0
    update_hp = landed and hits_foe and damage > 0
    if update_hp:
        turn.move_hit = True  # BattleControllerPlayer_UpdateHP: SYSCTL_MOVE_HIT (bcp.c:3441)
    taken = 0
    if landed and hits_foe:
        # BattleControllerPlayer_UpdateHP (bcp.c:3346-3404) runs for a non-zero damage:
        # a Focus Sash at full HP sets focusItemActivated, which lasts the whole action.
        if damage > 0 and _sash_at_full_hp(turn, s, target):
            turn.focus_armed = True
        # bcp.c:3381-3384, before the endure checks: False Swipe leaves 1 HP.
        swipe = _t(tables, "move_effect", move) == EffectId.FALSE_SWIPE
        if swipe and damage > 0 and damage >= target_mon.hp:
            damage = target_mon.hp - 1
        taken = _focus_sash(turn, s, target, damage, turn.focus_armed)  # battleCtx->damage
        if update_hp:  # shellBellDamageDealt (bcp.c:3427-3431): capped at the target's HP
            turn.shell_dealt += min(taken, target_mon.hp)
        # physical / specialDamageTaken (bcp.c:3411-3424): nonzero damage after the caps,
        # read by the Sticky Barb transfer and King's Rock.
        if update_hp and taken:
            turn.damaged = True
        if update_hp:  # the defender's turnFlags (bcp.c:3413-3434), by moveCur's class
            record = turn.taken[target]
            klass = _t(tables, "move_category", move)
            if klass == PHYSICAL:
                record[0], record[3] = taken, side + 1
            elif klass == SPECIAL:
                record[1], record[4] = taken, side + 1
            record[2], record[5] = taken, side + 1
        # subscript_update_hp.s: the resist berry halves hpCalcTemp (BattleSystem_Divide).
        if taken > 0 and _hit_resist_berry(turn, s, side, move):
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
            kind = _move_type(tables, s, side, move)
            passes = _matchup_x40(
                tables, kind, target_mon.types, _identified(s, side, target)
            ) > 0 and not _levitated(tables, s, side, target, kind)
    else:
        passes = damage > 0
    effective = landed and passes

    _move_effect(turn, s, side, move, hit, landed, effective, dealt, target_moved)
    if struggling and landed:
        # effect_script_0254.s: MOVE_SIDE_EFFECT_ON_HIT | subscript_struggle.s, a
        # secondary effect: Divide(maxHP, 4) recoil, before the contact ability.
        _lose_hp(turn, s, side, _divide(_active(s, side).max_hp, 4))
    effect = _t(tables, "move_effect", move)
    if effect == EffectId.U_TURN and effective and _u_turn(turn, s, side, move, hit, dealt):
        return dealt, update_hp
    _contact_ability(turn, s, side, move, hit, effective, dealt)
    # BattleControllerPlayer_CheckExtraFlinch (bcp.c:4630-4653): King's Rock / Razor Fang on a
    # hit that damaged a living target, RandNext() % 100 < param (outcome.flinch_item), a
    # MOVE_FLAG_TRIGGERS_KINGS_ROCK move; subscript_flinch_mon.s: a target that has not moved,
    # no Inner Focus as the attacker sees it.
    if (
        update_hp
        and taken
        and _item_effect(tables, _active(s, side)) == ItemId.KINGS_ROCK
        and o.flinch_item[side][hit]
        and _t(tables, "move_flags", move) & MoveFlag.KINGS_ROCK
        and _active(s, target).hp > 0
        and not target_moved
        and _ignorable(s, side, target) not in (AbilityId.INNER_FOCUS, AbilityId.SHIELD_DUST)
    ):
        s.sides[target].flinched = 1
    return dealt, update_hp


def _bench(s: RefState, side: int) -> bool:
    """``BattleSystem_AnyReplacementMons`` (lib.c:3125) in singles."""
    sd = s.sides[side]
    return any(m.species and m.hp for i, m in enumerate(sd.party) if i != sd.active)


def _u_turn(turn: _Turn, s: RefState, side: int, move: int, hit: int, dealt: int) -> bool:
    """U-turn's ON_HIT secondary effect, ``subscript_attack_then_switch_out.s``, on a hit that
    worked. Returns whether it ran past its checks (:5 the defender side is not blacked out,
    :6 the user has a replacement), in which case its contact ability (:7) and the King's Rock
    check after it are this function's: the contact ability, the pivot item (:11,
    ``BattleSystem_TriggerHeldItemOnPivotMove``, lib.c:7336-7386: the last match of the
    defender's Sticky Barb, Jaboca (unsupported), the attacker's Life Orb or Shell Bell), Destiny
    Bond (:15-34); a user still alive (:48) meets its foe's Pursuit (:54, then the faint check
    and payout of subscript_pursuit.s), and a user alive after it (:55) gets its Natural Cure
    (:56-57) and the party list (:71): ``_Pause``. A user that fainted switches nothing: the
    action goes on, King's Rock included (bcp.c:3524, no attacker HP test)."""
    tables, o = turn.tables, turn.outcome
    foe = 1 - side
    if all(m.species == 0 or m.hp == 0 for m in s.sides[foe].party) or not _bench(s, side):
        return False
    target_moved = turn.position == 1
    _contact_ability(turn, s, side, move, hit, True, dealt)
    user, holder = _active(s, side), _active(s, foe)
    barb = (
        turn.damaged
        and _item_effect(tables, holder) == ItemId.STICKY_BARB
        and user.hp
        and user.item == 0
        and _t(tables, "move_flags", move) & MoveFlag.CONTACT
    )
    if barb:
        _sticky_barb_transfer(turn, s, side, foe, move)
    elif turn.move_hit:
        _shell_bell(turn, s, side, turn.shell_dealt)
        if _t(tables, "move_category", move) != STATUS_CLASS:
            _life_orb_recoil(turn, s, side)
    if holder.hp == 0 and s.sides[foe].destiny_bond and user.hp > 0:
        _lose_hp(turn, s, side, user.hp)
    if user.hp > 0 and not target_moved:
        before = turn.done[foe]
        _pursuit(turn, s, side, turn.actions, turn.choices)
        if turn.done[foe] and not before:  # the Pursuit hit (subscript_pursuit.s)
            turn.pursued = foe + 1
            turn.pick_mul = PURSUIT_POWER_MUL
            _faint_check(turn, s)
            _pay_exp(turn, s)
    if user.hp > 0 and not s.over:
        if user.ability == AbilityId.NATURAL_CURE:  # TryRestoreStatusOnSwitch, script.c:9139
            user.status = Status.NONE
            user.status_counter = 0
        turn.pivot_dealt = dealt
        raise _Pause
    # No switch: AfterMoveMessage goes on with the extra flinch (the pursuer, if any, has moved).
    if (
        turn.damaged
        and _item_effect(tables, user) == ItemId.KINGS_ROCK
        and o.flinch_item[side][hit]
        and _t(tables, "move_flags", move) & MoveFlag.KINGS_ROCK
        and holder.hp > 0
        and not target_moved
        and not turn.done[foe]
        and _ignorable(s, side, foe) not in (AbilityId.INNER_FOCUS, AbilityId.SHIELD_DUST)
    ):
        s.sides[foe].flinched = 1
    return True


def _status_disrupted(turn: _Turn, s: RefState, side: int, move: int) -> int:
    """``BattleControllerPlayer_CheckStatusDisruption`` (bcp.c:2417): ``GOES_AHEAD``,
    ``DISRUPTED`` when the move does not go ahead, or ``HURT_ITSELF`` for a confusion
    self-hit, which happens here."""
    tables, o = turn.tables, turn.outcome
    sd, mon = s.sides[side], _active(s, side)
    effect = _t(tables, "move_effect", move)
    thaw_move = effect == EffectId.SECONDARY_STATUS and _t(tables, "move_param", move, 2) == 1

    # CHECK_STATUS_STATE_SLEEP (bcp.c:2430): the counter drops by 1 (Early Bird is
    # unsupported); still asleep stops the move, waking up lets the checks go on. In an uproar
    # (FIELD_CONDITION_UPROAR) a sleeper wakes at once (bcp.c:2432-2439, subscript_wake_up.s).
    if mon.status == Status.SLEEP and _uproar_on(s):
        mon.status, mon.status_counter = Status.NONE, 0
    if mon.status == Status.SLEEP:
        if mon.status_counter < 1:
            mon.status = Status.NONE
        else:
            mon.status_counter -= 1
            # bcp.c:2455: a sleeping battler still uses Snore or Sleep Talk.
            if mon.status_counter > 0:
                if move not in (MOVE_SNORE, MOVE_SLEEP_TALK):
                    return DISRUPTED
            else:
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
    # CHECK_STATUS_STATE_TRUANT (bcp.c:2502-2510): subscript_loafing_around.s unlocks.
    if _truant_loafs(s, side):
        _unlock(sd)
        return DISRUPTED
    # CHECK_STATUS_STATE_FLINCH (bcp.c:2529): subscript_flinched.s raises Steadfast's Speed.
    # The checks below that disrupt the move set moveFailFlags (bcp.c:2531, :2545, :2560, :2628,
    # :2656, :2681); sleep, freeze and Truant above do not.
    if sd.flinched:
        turn.move_failed[side] = True
        sd.flinched = 0
        if mon.ability == AbilityId.STEADFAST:
            _change_stage(s, side, Stat.SPE, 1)
        _unlock(sd)  # subscript_flinched.s:9
        return DISRUPTED
    # CHECK_STATUS_STATE_DISABLE (bcp.c:2543): the chosen move is the disabled one.
    if sd.disabled_move and sd.disabled_move == move:
        turn.move_failed[side] = True
        _unlock(sd)  # subscript_move_is_disabled.s
        return DISRUPTED
    # CHECK_STATUS_STATE_TAUNT (bcp.c:2557): a taunted battler's power-0 move fails.
    if sd.taunt and _t(tables, "move_power", move) == 0:
        turn.move_failed[side] = True
        _unlock(sd)  # subscript_move_fail_taunted.s
        return DISRUPTED
    # CHECK_STATUS_STATE_IMPRISON (bcp.c:2572-2583): a foe with Imprison knows moveCur.
    if _move_imprisoned(s, side, move):
        turn.move_failed[side] = True
        _unlock(sd)  # subscript_move_is_imprisoned.s
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
            turn.move_failed[side] = True
            return HURT_ITSELF
    # CHECK_STATUS_STATE_PARALYSIS (bcp.c:2652): RandNext() % 4 == 0 (Magic Guard is
    # unsupported).
    # Magic Guard (Battler_Ability) skips the check before the draw (bcp.c:2652-2654).
    if (
        mon.status == Status.PARALYSIS
        and mon.ability != AbilityId.MAGIC_GUARD
        and o.full_para[side]
    ):
        turn.move_failed[side] = True
        _unlock(sd)  # subscript_fully_paralyzed.s
        return DISRUPTED
    # CHECK_STATUS_STATE_ATTRACT (bcp.c:2669-2692): in love, RandNext() & 1 == 0 immobilizes
    # (subscript_immobilized_by_love.s: UnlockMoveChoice, UPDATE_MOVE_BUFFERS, no message).
    if sd.infatuated and o.infatuation[side]:
        turn.move_failed[side] = True
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


def _hit_rate(tables, s: RefState, attacker: int, move: int, target_moved: bool = False) -> int:
    """``hitRate`` of ``BattleControllerPlayer_CheckMoveHitAccuracy`` (bcp.c:2865-2979) for
    ``attacker``'s ``move`` against the foe's active Pokemon, from the state as it is now;
    ``target_moved``: the foe acted earlier this turn (Zoom Lens). Hustle and the Micle Berry
    are unsupported."""
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
    weather = Weather.NONE if _weather_suppressed(tables, s) else s.weather
    flags = _t(tables, "move_flags", move)
    if flags & MoveFlag.THUNDER and weather == Weather.SUN:  # :2915-2917 (B9-1)
        rate = 50
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
    # :2944-2947 Tangled Feet (ignorable) on a confused target.
    if seen == AbilityId.TANGLED_FEET and s.sides[defender].confusion:
        rate = rate * 50 // 100
    # :2949-2965: the defender's BrightPowder / Lax Incense, the attacker's Wide Lens and, after
    # the target moved, Zoom Lens.
    foe = _active(s, defender)
    if _item_effect(tables, foe) == ItemId.BRIGHT_POWDER:
        rate = rate * (100 - _t(tables, "item_param", foe.item, 0)) // 100
    lens = _item_effect(tables, atk)
    if lens == ItemId.WIDE_LENS or (lens == ItemId.ZOOM_LENS and target_moved):
        rate = rate * (100 + _t(tables, "item_param", atk.item, 0)) // 100
    if s.gravity:  # FIELD_CONDITION_GRAVITY, :2972
        rate = rate * 10 // 6
    no_guard = AbilityId.NO_GUARD in (atk.ability, _active(s, defender).ability)
    # MON_IS_LOCKED_ONTO beside No Guard (bcp.c:3016-3018; TryOHKOMove, script.c:4343-4361).
    no_guard = no_guard or _locked_onto(s, attacker, defender)
    # :3023-3031 Thunder in rain, Blizzard in hail clear the miss (B9-1).
    if (flags & MoveFlag.THUNDER and weather == Weather.RAIN) or (
        flags & MoveFlag.BLIZZARD and weather == Weather.HAIL
    ):
        no_guard = True
    if _t(tables, "move_effect", move) == EffectId.OHKO:
        # BtlCmd_TryOHKOMove (script.c:4334-4386): its own test, nothing above (the
        # nonstandard check): accuracy + level difference at a level not lower, No Guard 100.
        diff = atk.level - foe.level
        if diff < 0:
            return 0
        return 100 if no_guard else _t(tables, "move_accuracy", move) + diff
    # CheckMoveHitOverrides (bcp.c:3016-3021): either battler's No Guard clears the miss.
    if no_guard:
        return 100
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
        and not _curse_on_user(tables, move, _ghost(s, side))
        and (_active(s, 1 - side).hp > 0 or target_range == MoveTarget.RANGE_OPPONENT_SIDE)
    )
    # BEFORE_MOVE_STATE_QUICK_CLAW (bcp.c:3140-3143): subscript_check_quick_claw.s:26-28 eats
    # an activated Custap Berry (RemoveItem: the recycle item) before the status checks.
    if turn.custap[side]:
        turn.custap[side] = False
        mon.item_consumed, mon.item = mon.item, 0
        _forget_item(s, side)
    s.sides[side].destiny_bond = 0  # CHECK_STATUS_START (bcp.c:2425)
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
    # A special-range Counter / Mirror Coat / Metal Burst is its own defender until its command.
    counter = _t(tables, "move_effect", move) == EffectId.COUNTER
    special_counter = counter and _move_target(tables, move) in SELF_DEFENDER
    if (
        not nature_power
        and not special_counter
        and not _defender_always(tables, move, _ghost(s, side))
        and _active(s, 1 - side).hp == 0
    ):
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
        turn.move_cur = move
        if not _defender_always(tables, move, _ghost(s, side)) and _active(s, 1 - side).hp == 0:
            _unlock(s.sides[side])
            return None
        record = move
    if _t(tables, "move_effect", move) in (
        EffectId.SLEEP_TALK,
        EffectId.METRONOME,
        EffectId.COPYCAT,
    ):
        # The invokers (B5-1..): the script picks the move now (Copycat reads the battle's
        # movePrev as the faster foe left it), then GoToMoveScript (script.c:2111-2134) runs it
        # as moveCur with a defender re-picked for it. Sleep Talk fails unless the user is still
        # asleep (effect_script_0097.s:5); a call that finds no move fails (MOVE_STATUS_FAILED,
        # subscript_missed.s); either way the defender stays the user (no moveHit).
        called = _invoked_move(tables, s, side, move, turn.outcome.call[side])
        sleep_talk = _t(tables, "move_effect", move) == EffectId.SLEEP_TALK
        if called == 0 or (sleep_talk and mon.status != Status.SLEEP):
            _unlock(s.sides[side])
            return None
        move = _called_move(tables, s, called)
        turn.move_cur = move
        if not _defender_always(tables, move, _ghost(s, side)) and _active(s, 1 - side).hp == 0:
            _unlock(s.sides[side])  # subscript_no_target.s
            return None
        own_defender = _move_target(tables, move) in SELF_DEFENDER or _curse_on_user(
            tables, move, _ghost(s, side)
        )
        record = None if own_defender else move
        counter = _t(tables, "move_effect", move) == EffectId.COUNTER
        special_counter = counter and _move_target(tables, move) in SELF_DEFENDER
    # TRY_MOVE_STATE_CHECK_TYPE_CHART (bcp.c:3271-3277, after the script, hit or miss): a move
    # with power that is not RANGE_USER / RANGE_USER_SIDE and is ineffective on the foe sets the
    # attacker's moveFailFlags.noEffect (bcp.c:2350-2370), which the Uproar check reads.
    if (
        _t(tables, "move_power", move) != 0
        and not _targets_user(tables, move, _ghost(s, side))
        # Future Sight's script sets SYSCTL_IGNORE_IMMUNITIES (effect_script_0148.s:7).
        and _t(tables, "move_effect", move) != EffectId.FUTURE_SIGHT
        and _matchup_x40(
            tables,
            _move_type(tables, s, side, move),
            _active(s, 1 - side).types,
            _identified(s, side, 1 - side),
        )
        == 0
    ):
        turn.move_failed[side] = True
    # The move script (bcp.c:3214): effect_script_0158.s (Fake Out) sets MOVE_STATUS_FAILED
    # unless fakeOutTurnNumber == totalTurns; TryMove then ends in subscript_missed.s
    # ("But it failed!").
    effect = _t(tables, "move_effect", move)
    if effect == EffectId.FUTURE_SIGHT:
        # effect_script_0148.s -> BtlCmd_TryFutureSight (script.c:6065-6100): with nothing pending
        # on the defender's battler, its side's SIDE_CONDITION_FUTURE_SIGHT, 3 turns, the move,
        # the attacker and CalcMoveDamage (criticalMul 1, the conditions of now) with its variance
        # (o.roll[side][0]); with one pending, "But it failed!" (subscript_but_it_failed.s:
        # MOVE_STATUS_NO_MORE_WORK). Then SYSCTL_NONSTANDARD_ACC_CHECK: no accuracy roll.
        foe_sd = s.sides[1 - side]
        if foe_sd.future_timer:
            return record
        foe_sd.future_sight = 1
        foe_sd.future_timer = FUTURE_SIGHT_TURNS
        foe_sd.future_move = move
        foe_sd.future_attacker = side + 1
        damage = _calc_move_damage(tables, s, side, 1 - side, move, False)
        # An s32 in the decomp; read only through min(damage, curHP) and damage >= curHP, so it is
        # kept capped to BattleState's int16 storage.
        foe_sd.future_damage = min(_variance(damage, o.roll[side][0]), 32767)
    if effect == EffectId.FAKE_OUT and s.sides[side].fake_out_turn != s.turn:
        _unlock(s.sides[side])  # subscript_missed.s
        return record
    # Imprison (effect_script_0192.s): BtlCmd_TryImprison sets the foes' MOVE_EFFECT_IMPRISONED
    # first (script.c:6686-6690), whatever follows.
    if effect == EffectId.IMPRISON:
        s.sides[1 - side].imprisoned = 1
    # The scripts that fail a user that has their effect already (MOVE_STATUS_FAILED, as Fake
    # Out's): Ingrain (effect_script_0181.s:5), Aqua Ring (_0251.s:5), Mud Sport (_0201.s:5),
    # Water Sport (_0210.s:5); Imprison also without a move shared with a foe (script.c:6692-6722).
    if _script_fails(s, side, effect, _t(tables, "move_param", move, 1)):
        _unlock(s.sides[side])  # subscript_missed.s
        return record
    # Snore (effect_script_0092.s:5): MOVE_STATUS_FAILED unless the user is asleep.
    if counter:
        # BtlCmd_Counter / MirrorCoat / TryMetalBurst (script.c:4615-4705, :7153-7184): the
        # user's turnFlags damage and its last attacker (1 + battler); 0 taken, an attacker of
        # the user's side or one with no HP fails (MOVE_STATUS_FAILED, :4619-4625, :4672-4678, :7157-7162).
        source = _t(tables, "move_param", move, 0)
        taken = turn.taken[side][source - 1]
        attacker = turn.taken[side][source + 2] - 1
        if not taken or attacker in (-1, side) or _active(s, attacker).hp == 0:
            _unlock(s.sides[side])
            return None if special_counter else record
        turn.counter_damage = taken * _t(tables, "move_param", move, 1) // 10
        if special_counter:
            record = move
    snore = effect == EffectId.FLINCH and _t(tables, "move_param", move, 2) == SNORE_PARAM
    if snore and mon.status != Status.SLEEP:
        _unlock(s.sides[side])
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
    # An OHKO move against an ignorable Sturdy (script.c:4340-4341, MOVE_STATUS_STURDY):
    # subscript_missed.s:67-69 tags the defender's ability (BattleAI_SetAbility,
    # script.c:11806) unless it is protected (:18), Levitate (:34) or ineffective (:59) first.
    if (
        effect == EffectId.OHKO
        and 1 - side == 0
        and _ignorable(s, side, 1 - side) == AbilityId.STURDY
        and not _protected(turn, s, side, move, unleash)
    ):
        kind = _move_type(tables, s, side, move)
        foe = _active(s, 1 - side)
        if _matchup_x40(tables, kind, foe.types, _identified(s, side, 1 - side)) != 0 and not (
            _levitated(tables, s, side, 1 - side, kind)
        ):
            s.ai_ability_known[s.sides[0].active] = 1
    roll = o.hit[side][0]
    missed = (
        _t(tables, "move_accuracy", move) != 0
        and effect != EffectId.FUTURE_SIGHT  # SYSCTL_NONSTANDARD_ACC_CHECK (bcp.c:2911-2913)
        and not 0 < roll <= _hit_rate(tables, s, side, move, target_moved)
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
    # Baton Pass (effect_script_0127.s, subscript_baton_pass.s): no replacement fails (:5,
    # :19-20: subscript_missed.s, no AfterMoveEffects); else Natural Cure (:7-8) and the list.
    if effect == EffectId.BATON_PASS:
        if not _bench(s, side):
            return record
        if mon.ability == AbilityId.NATURAL_CURE:
            mon.status = Status.NONE
            mon.status_counter = 0
        turn.pivot_dealt = 0
        raise _Pause
    # A failed weather move: MOVE_STATUS_FAILED, subscript_missed.s, no AfterMoveEffects.
    if effect == EffectId.WEATHER and _weather_fails(
        s, _t(tables, "move_param", move, 0), _t(tables, "move_param", move, 1)
    ):
        return record
    # TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES (bcp.c:3281-3286): the foe's Dry Skin
    # (ignorable) takes a Water move with power that hit (lib.c:3538-3544):
    # subscript_ability_restores_hp.s heals Divide(maxHP, 4) below full HP and tags the
    # ability; MOVE_STATUS_NO_MORE_WORK: no hit, no AfterMoveEffects.
    foe_side = 1 - side
    kind = _move_type(tables, s, side, move)
    seen = _ignorable(s, side, foe_side)
    water = (
        seen in (AbilityId.DRY_SKIN, AbilityId.WATER_ABSORB)
        and kind == Type.WATER
        and _t(tables, "move_power", move) != 0
    )
    volt = seen == AbilityId.VOLT_ABSORB and kind == Type.ELECTRIC  # lib.c:3500-3505 (B9-6)
    if (
        record is not None
        and (water or volt)
        and _matchup_x40(tables, kind, _active(s, foe_side).types, _identified(s, side, foe_side))
        > 0
    ):
        foe = _active(s, foe_side)
        if foe.hp != foe.max_hp:
            _gain_hp(s, foe_side, _divide(foe.max_hp, 4))
        if foe_side == 0:
            s.ai_ability_known[s.sides[0].active] = 1
        return record
    # BtlCmd_SetMultiHit (script.c:2634): SetMultiHit 2 (effect_script_0044.s) is fixed,
    # SetMultiHit 0 draws 2-5.
    hits = 1
    if effect == EffectId.MULTI_HIT:
        hits = _t(tables, "move_param", move, 0) or o.n_hits[side]
    foe_mon0 = _active(s, 1 - side)
    foe_hp0 = foe_mon0.hp
    turn.focus_armed = turn.pluck_flag = turn.damaged = False  # BattleContext_Init
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
        # subscript_swagger.s:7-8 (MOVE_STATUS_FAILED), and Dragon Dance with both stages at +6
        # (MOVE_STATUS_NO_MORE_WORK): CheckMoveFailure skips AfterMoveEffects (bcp.c:3312-3331).
        own, theirs = s.sides[side].stages, s.sides[1 - side].stages
        full = full or (
            effect == EffectId.SWAGGER and theirs[Stat.ATK] >= 6 and s.sides[1 - side].confusion
        )
        second = _t(tables, "move_param", move, 0)
        full = full or (
            effect == EffectId.DRAGON_DANCE and own[Stat.ATK] >= 6 and own[second] >= 6
        )
        # subscript_belly_drum.s:3-6 (MOVE_STATUS_FAILED).
        full = full or (
            effect == EffectId.BELLY_DRUM
            and (own[Stat.ATK] >= 6 or user.hp <= _divide(user.max_hp, 2))
        )
        # Roost at full HP (subscript_recover_hp.s), Tailwind up, full Spikes / Toxic Spikes.
        full = full or (effect == EffectId.ROOST and user.hp == user.max_hp)
        full = full or (effect == EffectId.TAILWIND and s.sides[side].tailwind != 0)
        # subscript_mean_look.s:7-8, :48: a trapped or substituted defender (MOVE_STATUS_FAILED).
        foe_sd = s.sides[1 - side]
        full = full or (
            effect == EffectId.MEAN_LOOK and bool(foe_sd.mean_look or foe_sd.substitute_hp)
        )
        full = full or (effect == EffectId.LOCK_ON and bool(foe_sd.substitute_hp))  # :6, :33
        full = full or (effect == EffectId.YAWN and _yawn_blocks(tables, s, side, 1 - side)[0])
        full = full or (
            effect == EffectId.CURSE
            and _ghost(s, side)
            and bool(foe_sd.substitute_hp or foe_sd.curse)
        )  # subscript_curse_ghost.s:3-4, :18-20
        full = full or (
            effect == EffectId.PERISH_SONG
            and not any(not x.perish_song and _active(s, k).hp for k, x in enumerate(s.sides))
        )  # BtlCmd_TryPerishSong's failure (script.c:5482-5484)
        full = full or (effect == EffectId.SPIKES and s.sides[1 - side].spikes >= SPIKES_MAX)
        full = full or (
            effect == EffectId.TOXIC_SPIKES and s.sides[1 - side].toxic_spikes >= TOXIC_SPIKES_MAX
        )
        # A direct sleep on a living foe not asleep, without Insomnia as the user sees it, in an
        # uproar: subscript_fall_asleep.s:43-45 -> :_292 sets MOVE_STATUS_NO_MORE_WORK (:_330).
        full = full or (
            effect == EffectId.STATUS_FOE
            and _t(tables, "move_param", move, 0) == Status.SLEEP
            and foe.hp > 0
            and _ignorable(s, side, 1 - side) != AbilityId.INSOMNIA
            and foe.status != Status.SLEEP
            and _uproar_on(s)
        )
        dealt, _ = _hit(turn, s, side, move, hit, struggling, target_moved)
        if full:
            continue
        # BattleControllerPlayer_AfterMoveEffects (bcp.c:3638), after every hit:
        # AFTER_MOVE_EFFECT_ATTACKER_ITEM (bcp.c:3684), then _DEFENDER_ITEM (:3691), then
        # _TRIGGER_ITEMS_ON_HIT (:3698, the Sticky Barb once the defender took damage this
        # action: SYSCTL_MOVE_HIT), _THAW_DEFENDER (:3708) and _HELD_ITEM_STATUS (:3730) for
        # every battler. A restore subscript's PLUCK_CHECK keeps the item while the pluck flag
        # is set, and clears the flag.
        for battler in (side, 1 - side):
            if _held_item(turn, s, battler, keep=turn.pluck_flag):
                turn.pluck_flag = False
        if turn.damaged:
            _sticky_barb_transfer(turn, s, side, 1 - side, move)
        _thaw_target(turn, s, side, move, dealt)
        for battler in (side, 1 - side):
            if _held_item(turn, s, battler, keep=turn.pluck_flag):
                turn.pluck_flag = False
    # LOOP_FAINTED (bcp.c:3820-3823): subscript_faint_check_destiny_bond.s takes a living
    # attacker down with a defender that fainted holding Destiny Bond.
    foe_now = _active(s, 1 - side)
    if (
        foe_hp0 > 0
        and foe_now is foe_mon0
        and foe_now.hp == 0
        and s.sides[1 - side].destiny_bond
        and mon.hp > 0
    ):
        _lose_hp(turn, s, side, mon.hp)
    if unleash:  # the unleash script spent the Bide counter (bcp.c:2702, subscript_bide_end.s)
        _unlock(s.sides[side])
        record = move
    return record


def _roar(turn: _Turn, s: RefState, side: int) -> bool:
    """``subscript_force_target_to_switch_or_flee.s`` for ``side``'s Roar on the foe; returns
    whether it dragged a Pokemon in. Its semi-invulnerable and Suction Cups tests (:5-6) and
    Soundproof (lib.c:3523-3530) cannot be met: no supported move or ability sets them. Ingrain
    (:7) anchors a defender whose moveEffectsMask is exactly MOVE_EFFECT_INGRAIN
    (``_ingrain_anchors``). ``BtlCmd_TryWhirlwind`` (script.c:5241-5320), trainer battle, singles: count the
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
    if _ingrain_anchors(turn.tables, s, foe):
        return False  # :86-90: "anchored itself with its roots!", MOVE_STATUS_NO_MORE_WORK
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
    turn.dragged[foe] = slot + 1
    _send_out_exp(s, foe)  # PokemonSendOut BTLSCR_DEFENDER (:22)
    _entry_hazards(turn, s, foe)  # BATTLE_SUBSCRIPT_HAZARDS_CHECK (:31)
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
    if _truant_loafs(s, pursuer):  # script.c:6903
        return
    slot = actions[pursuer]
    if mon.pp[slot] == 0:
        return
    mon.pp[slot] -= 1
    turn.done[pursuer] = True
    # script.c:6930: movePrevByBattler[pursuer] = Pursuit; :6943-6950: a Choice holder's
    # choiceLockedMove = Pursuit.
    s.sides[pursuer].last_move = move
    if _holds_choice(tables, s, pursuer):
        s.sides[pursuer].choice_lock = move
    o = turn.outcome
    crit = o.crit[pursuer][0] != 0 and not _armored(s, pursuer, side)
    damage = _move_damage(tables, s, pursuer, move, crit, o.roll[pursuer][0],
                          PURSUIT_POWER_MUL)  # fmt: skip
    # subscript_pursuit.s: CheckHoldOnWith1HP on the target (script.c:9098-9124).
    armed = _sash_at_full_hp(turn, s, side)
    taken = _focus_sash(turn, s, side, damage, armed)
    # Its UPDATE_HP runs the resist-berry check on the target (subscript_update_hp.s:9).
    if damage > 0 and _hit_resist_berry(turn, s, pursuer, move):
        taken = _divide(taken, 2)
    target_hp = _active(s, side).hp
    dealt = _lose_hp(turn, s, side, taken)
    _contact_ability(turn, s, pursuer, move, 0, damage > 0, dealt)
    # subscript_pursuit.s:85: a switcher that fainted holding Destiny Bond takes the living
    # pursuer down (subscript_faint_check_destiny_bond.s), before the items (:90, :94-95).
    if target_hp > 0 and _active(s, side).hp == 0 and s.sides[side].destiny_bond and mon.hp > 0:
        _lose_hp(turn, s, pursuer, mon.hp)
    # subscript_pursuit.s:44 / :90: TriggerHeldItemOnHit (the Sticky Barb); then :48-49
    # (:94-95 after a KO): SYSCTL_MOVE_HIT, TriggerHeldItemOnPivotMove: Shell Bell on the
    # script's shellBellDamageDealt (:21-29: the damage before CheckHoldOnWith1HP, capped at
    # the target's HP) or Life Orb (lib.c:7345-7366); all skipped when the hit did not hit
    # (:17, MOVE_STATUS_DID_NOT_HIT: damage 0).
    if damage > 0:
        _sticky_barb_transfer(turn, s, pursuer, side, move)
        _shell_bell(turn, s, pursuer, min(damage, target_hp))
        _life_orb_recoil(turn, s, pursuer)


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
    _pay_exp(turn, s)  # subscript_pursuit.s:106-108: Pursuit's own payout
    if s.over:
        return
    # The outgoing Pokemon goes back to the party: battle_controller.c:1482 writes its
    # status back without MON_CONDITION_TOXIC_COUNTER.
    if mon.status == Status.TOXIC:
        mon.status_counter = 0
    _enter(turn, s, side, slot)
    _send_out_exp(s, side)  # subscript_switch_pokemon.s:39
    _entry_hazards(turn, s, side)  # :43
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
    traced = _t(tables, "species_abilities", row, mon.ability_slot) == AbilityId.TRACE
    if traced and mon.ability != AbilityId.TRACE:
        return ABILITY_DECOMP_IDS[AbilityId(mon.ability)]  # a traced ability
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
    # script.c:5585-5596: sun hurts Dry Skin and Solar Power by Divide(maxHP, 8).
    if s.weather == Weather.SUN and mon.ability in (AbilityId.DRY_SKIN, AbilityId.SOLAR_POWER):
        hp_calc = -_divide(mon.max_hp, 8)
    if s.weather == Weather.HAIL:  # the hail branch
        if ability == WEATHER_ABILITY_IDS["Ice Body"]:
            if mon.hp < mon.max_hp:
                hp_calc = _divide(mon.max_hp, 16)
        elif Type.ICE not in mon.types and mon.ability != AbilityId.SNOW_CLOAK:
            hp_calc = -_divide(mon.max_hp, 16)
    # script.c:5620-5625: rain heals Dry Skin below full HP by Divide(maxHP, 8).
    if s.weather == Weather.RAIN and mon.ability == AbilityId.RAIN_DISH and mon.hp < mon.max_hp:
        hp_calc = _divide(mon.max_hp, 16)  # script.c:5614-5618 (B9-6)
    if s.weather == Weather.RAIN and mon.ability == AbilityId.DRY_SKIN and mon.hp < mon.max_hp:
        hp_calc = _divide(mon.max_hp, 8)
    sun = s.weather == Weather.SUN and mon.ability in (AbilityId.DRY_SKIN, AbilityId.SOLAR_POWER)
    rain = (
        s.weather == Weather.RAIN
        and mon.ability in (AbilityId.DRY_SKIN, AbilityId.RAIN_DISH)
        and hp_calc > 0
    )
    magic_guard = ability == WEATHER_ABILITY_IDS["Magic Guard"]
    if hp_calc > 0:  # subscript_weather_continues.s _076
        _gain_hp(s, side, hp_calc)
    elif hp_calc < 0 and not magic_guard:
        _lose_hp(turn, s, side, -hp_calc)
    # :_045, :_057, :_076 tag the battler's ability (after the Magic Guard skip).
    if side == 0 and ((sun and not magic_guard) or rain):
        s.ai_ability_known[s.sides[0].active] = 1


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
        _pay_exp(turn, s)  # the loop head's AnyExpPayout, :938-944
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
            _ingrain_check,  # MON_COND_CHECK_STATE_INGRAIN, bcp.c:1311
            _aqua_ring_check,  # MON_COND_CHECK_STATE_AQUA_RING, bcp.c:1332
            _turn_end_ability,  # MON_COND_CHECK_STATE_ABILITY, bcp.c:1355
            _held_item_hp,  # MON_COND_CHECK_STATE_USE_ITEM, bcp.c:1362
            _leftovers_check,  # MON_COND_CHECK_STATE_LEFTOVERS, bcp.c:1369
            _leech_seed_check,  # MON_COND_CHECK_STATE_LEECH_SEED, bcp.c:1378
            _poison_check,  # MON_COND_CHECK_STATE_POISON, bcp.c:1393
            _toxic_check,  # MON_COND_CHECK_STATE_TOXIC, bcp.c:1405
            _burn_check,  # MON_COND_CHECK_STATE_BURN, bcp.c:1425
            _curse_check,  # MON_COND_CHECK_STATE_CURSE, bcp.c:1451
            _bind_check,  # MON_COND_CHECK_STATE_BIND, bcp.c:1462
            _bad_dreams_check,  # MON_COND_CHECK_STATE_BAD_DREAMS, bcp.c:1484
            _uproar_check,  # MON_COND_CHECK_STATE_UPROAR, bcp.c:1505
            _disable_check,  # MON_COND_CHECK_STATE_DISABLE, bcp.c:1565
            _lock_on_check,  # MON_COND_CHECK_STATE_LOCK_ON, bcp.c:1618
            _taunt_check,  # MON_COND_CHECK_STATE_TAUNT, bcp.c:1635
            _yawn_check,  # MON_COND_CHECK_STATE_YAWN, bcp.c:1684
            _held_item_hp,  # MON_COND_CHECK_STATE_ITEM_CONDITION, bcp.c:1699
            _detrimental_item_check,  # MON_COND_CHECK_STATE_ITEM_DETRIMENTAL_EFFECT, :1710
        ):
            if s.over:
                return
            check(turn, s, battler)
            _faint_check(turn, s)
            _pay_exp(turn, s)  # CheckMonConditions' head: AnyFainted, AnyExpPayout (:1295-1299)


def _future_sight(turn: _Turn, s: RefState, order: list[int]) -> None:
    """``SIDE_COND_CHECK_STATE_FUTURE_SIGHT`` (bcp.c:1758-1783): each battler in monSpeedOrder
    with a count drops it; at 0 a battler with HP clears its side's bit and takes
    subscript_future_sight_damage.s: ``CheckMoveHit`` of the stored attacker's battler slot with
    the stored move (bcp.c:217-221, the ordinary check; Zoom Lens: every battler has moved),
    failing on MOVE_STATUS_DID_NOT_HIT ("But it failed!"); ``CheckHoldOnWith1HP`` (Focus Sash at
    full HP, script.c:9098-9124), ``UPDATE_HP`` (no resist berry: waitingBattlers is 0), then the
    endured item's removal (:40-48). ``moveStatusFlags`` lasts the whole pass (only
    BattleContext_Init clears it, lib.c:1983), and ``AnyFainted`` faints a battler between the two
    without CheckBattleOver (:1749); TurnEnd checks the end and pays (bcp.c:1844-1847)."""
    if s.over:
        return
    tables, o = turn.tables, turn.outcome
    flags: set[str] = set()
    for d in order:
        sd = s.sides[d]
        if not sd.future_timer:
            continue
        sd.future_timer -= 1
        mon = _active(s, d)
        if sd.future_timer or not mon.hp:
            continue
        sd.future_sight = 0
        attacker, move = sd.future_attacker - 1, sd.future_move
        # BattleControllerPlayer_CheckMoveHitAccuracy: the roll against the rate of now.
        if not 0 < o.future_hit[d] <= _hit_rate(tables, s, attacker, move, True):
            flags.add("missed")
        # CheckMoveHitOverrides (bcp.c:3001-3021): Protect first (it unlocks the attacker's
        # battler), else either battler's No Guard clears MOVE_STATUS_MISSED.
        if _protected(turn, s, attacker, move, False):
            _unlock(s.sides[attacker])
            flags.add("protected")
        elif AbilityId.NO_GUARD in (_active(s, attacker).ability, mon.ability) or _locked_onto(
            s, attacker, d
        ):
            flags.discard("missed")
        if flags & {"missed", "protected"}:
            continue
        damage = sd.future_damage
        sash = _item_effect(tables, mon) == ItemId.FOCUS_SASH
        if sash and mon.hp == mon.max_hp and mon.hp - damage <= 0:
            damage = mon.hp - 1
            flags.add("endured")
        _lose_hp(turn, s, d, damage)
        if "endured" in flags and sash:
            mon.item_consumed, mon.item = mon.item, 0
            _forget_item(s, d)
        _faint_check(turn, s, decide=False)
    _perish_song_check(turn, s, order)
    _faint_check(turn, s)  # TurnEnd's CheckBattleOver (bcp.c:1845)
    _pay_exp(turn, s)


def _perish_song_check(turn: _Turn, s: RefState, order: list[int]) -> None:
    """``SIDE_COND_CHECK_STATE_PERISH_SONG`` (bcp.c:1791-1820): each battler in monSpeedOrder
    with the effect faints at a count of 0 (subscript_continue_perish_song.s:6-7), else its count
    drops; AnyFainted after each (:1749), no battle-over check."""
    for d in order:
        sd = s.sides[d]
        if not sd.perish_song:
            continue
        if sd.perish_song_turns == 0:
            sd.perish_song = 0
            _lose_hp(turn, s, d, _active(s, d).hp)  # UPDATE_HP of -curHP
        else:
            sd.perish_song_turns -= 1
        _faint_check(turn, s, decide=False)


def _sunny(tables, s: RefState) -> bool:
    return s.weather == Weather.SUN and not _weather_suppressed(tables, s)


def _yawn_blocks(tables, s: RefState, side: int, target: int) -> tuple[bool, bool]:
    """subscript_yawn.s's ``(fails, by_ability)`` on ``target``'s battler (effects.yawn_blocks):
    ignorable Insomnia / Vital Spirit, Leaf Guard in sun (:5-9), a substitute (:14), an uproar
    unless Soundproof (:15-16), any status (:18), Safeguard (:19), a Yawn counting (:21)."""
    mon, sd = _active(s, target), s.sides[target]
    own = _decomp_ability(tables, mon)
    seen = 0 if _active(s, side).ability == AbilityId.MOLD_BREAKER else own
    ids = YAWN_ABILITY_IDS
    by_ability = seen in (ids["Insomnia"], ids["Vital Spirit"]) or (
        _sunny(tables, s) and seen == ids["Leaf Guard"]
    )
    fails = (
        by_ability
        or bool(sd.substitute_hp)
        or (own != ids["Soundproof"] and _uproar_on(s))
        or mon.status != Status.NONE
        or bool(sd.safeguard)
        or bool(sd.yawn)
    )
    return fails, by_ability


def _yawn_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_YAWN`` (bcp.c:1684-1697): the count drops; at 0
    subscript_fall_asleep.s as a move effect (:4-14): its own Insomnia / Vital Spirit / Leaf
    Guard in sun keep it awake (tagged), a sleeper, an uproar (unless Soundproof) or a status stop
    it, else it sleeps ``yawn_sleep`` turns and is unlocked."""
    sd, mon, tables = s.sides[side], _active(s, side), turn.tables
    if not sd.yawn:
        return
    sd.yawn -= 1
    if sd.yawn:
        return
    own = _decomp_ability(tables, mon)
    ids = YAWN_ABILITY_IDS
    if own in (ids["Insomnia"], ids["Vital Spirit"]) or (
        _sunny(tables, s) and own == ids["Leaf Guard"]
    ):
        if side == 0:
            s.ai_ability_known[sd.active] = 1
        return
    if own != ids["Soundproof"] and _uproar_on(s):
        return
    if mon.status != Status.NONE:
        return
    mon.status, mon.status_counter = Status.SLEEP, turn.outcome.yawn_sleep[side]
    _unlock(sd)


def _lock_on_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_LOCK_ON`` (bcp.c:1618-1624): the counter drops by 1."""
    if s.sides[side].lock_on:
        s.sides[side].lock_on -= 1


def _locked_onto(s: RefState, attacker: int, defender: int) -> bool:
    """``MON_IS_LOCKED_ONTO`` (include/battle/common.h:51)."""
    sd = s.sides[defender]
    return bool(sd.lock_on) and sd.lock_on_target == 1 + attacker


def _ingrain_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_INGRAIN`` (bcp.c:1311-1326): an ingrained battler below full HP and
    alive heals ``Divide(maxHP, 16)`` (``subscript_ingrain_heal.s``, Big Root applied). Heal
    Block is unsupported."""
    mon = _active(s, side)
    if s.sides[side].ingrain and mon.hp and mon.hp != mon.max_hp:
        _gain_hp(s, side, _big_root(turn, s, side, _divide(mon.max_hp, 16)))


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
    if mon.hp and recipient.hp and mon.ability != AbilityId.MAGIC_GUARD:  # bcp.c:1381
        lost = _lose_hp(turn, s, side, max(mon.max_hp // 8, 1))
        # subscript_leech_seed_effect.s: the recipient gains hitDamage, Big Root applied.
        _gain_hp(s, 1 - side, _big_root(turn, s, 1 - side, lost))


def _poison_check(turn: _Turn, s: RefState, side: int) -> None:
    mon = _active(s, side)
    # subscript_poison_damage.s:5: not for Magic Guard.
    if mon.status == Status.POISON and mon.hp and mon.ability != AbilityId.MAGIC_GUARD:
        if not _poison_heal(s, side):
            _lose_hp(turn, s, side, _divide(mon.max_hp, 8))


def _poison_heal(s: RefState, side: int) -> bool:
    """subscript_poison_damage.s:4-12: a Poison Heal holder heals Divide(maxHP, 8) below full HP
    (and tags its ability), nothing at full HP; True when it holds the ability (B9-8)."""
    mon = _active(s, side)
    if mon.ability != AbilityId.POISON_HEAL:
        return False
    if mon.hp != mon.max_hp:
        _gain_hp(s, side, _divide(mon.max_hp, 8))
        if side == 0:
            s.ai_ability_known[s.sides[0].active] = 1
    return True


def _bad_dreams_check(turn: _Turn, s: RefState, side: int) -> None:
    """MON_COND_CHECK_STATE_BAD_DREAMS (bcp.c:1484-1500): an asleep battler with HP, no Magic
    Guard, facing a living Bad Dreams holder loses Divide(maxHP, 8); the message tags the holder."""
    mon, foe = _active(s, side), _active(s, 1 - side)
    if (
        foe.ability == AbilityId.BAD_DREAMS
        and foe.hp
        and mon.status == Status.SLEEP
        and mon.ability != AbilityId.MAGIC_GUARD
        and mon.hp
    ):
        _lose_hp(turn, s, side, _divide(mon.max_hp, 8))
        if side == 1:
            s.ai_ability_known[s.sides[0].active] = 1


def _toxic_check(turn: _Turn, s: RefState, side: int) -> None:
    """The counter rises (up to 15), then ``Divide(maxHP, 16) * counter``."""
    mon = _active(s, side)
    if mon.status == Status.TOXIC and mon.hp:
        if mon.status_counter < TOXIC_COUNTER_MAX:
            mon.status_counter += 1
        if mon.ability != AbilityId.MAGIC_GUARD and not _poison_heal(s, side):
            _lose_hp(turn, s, side, _divide(mon.max_hp, 16) * mon.status_counter)


def _burn_check(turn: _Turn, s: RefState, side: int) -> None:
    """``subscript_burn_damage.s``: ``DivideVarByValue`` maxHP by 8."""
    mon = _active(s, side)
    # subscript_burn_damage.s:5: not for Magic Guard.
    if mon.status == Status.BURN and mon.hp and mon.ability != AbilityId.MAGIC_GUARD:
        _lose_hp(turn, s, side, _divide(mon.max_hp, 8))


def _curse_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_CURSE`` (bcp.c:1451-1459), subscript_curse_damage.s: a cursed
    battler with HP and no Magic Guard loses ``Divide(maxHP, 4)``."""
    mon = _active(s, side)
    if s.sides[side].curse and mon.hp and mon.ability != AbilityId.MAGIC_GUARD:
        _lose_hp(turn, s, side, _divide(mon.max_hp, 4))


def _bind_check(turn: _Turn, s: RefState, side: int) -> None:
    """The counter drops; while it stays above 0, ``Divide(maxHP, 16)`` damage
    (``subscript_bind_effect.s``), at 0 ``subscript_bind_end.s``."""
    sd, mon = s.sides[side], _active(s, side)
    if sd.trapped and mon.hp:
        sd.trapped -= 1
        if sd.trapped:
            if mon.ability != AbilityId.MAGIC_GUARD:  # subscript_bind_effect.s:5
                _lose_hp(turn, s, side, _divide(mon.max_hp, 16))
        else:
            sd.trap_move = 0


def _uproar_check(turn: _Turn, s: RefState, side: int) -> None:
    """``MON_COND_CHECK_STATE_UPROAR`` (bcp.c:1505-1541): while the battler is in an uproar,
    the first battler (battler order) asleep with HP left wakes (subscript_wake_up.s; Soundproof
    is unsupported) and the check repeats; then the count drops, and the uproar ends
    (subscript_uproar_end.s) on this turn's moveFailFlags or at 0, which frees the battler's
    command: it never had VOLATILE_CONDITION_MOVE_LOCKED."""
    sd = s.sides[side]
    if not sd.uproar:
        return
    while True:
        sleeper = next(
            (j for j in (0, 1) if _active(s, j).status == Status.SLEEP and _active(s, j).hp),
            None,
        )
        if sleeper is None:
            break
        woken = _active(s, sleeper)
        woken.status, woken.status_counter = Status.NONE, 0
    sd.uproar -= 1
    if turn.move_failed[side] or not sd.uproar:
        sd.uproar = 0
        sd.locked_move = 0


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
    left, not the disabled move, not power 0 while taunted, and the Choice lock
    (lib.c:2307-2315: a Choice holder locked into a move it knows can use only that one; a
    lock it does not know is cleared there, which reads as no lock)."""
    mon, taunted = _active(s, side), s.sides[side].taunt > 0
    disabled = s.sides[side].disabled_move
    lock = _locked_slot(s, side)
    if lock >= 0:  # Battler_CanPickCommand is FALSE (lib.c:6224): only the locked move
        return [k == lock for k in range(MOVE_ACTIONS)]
    choice = s.sides[side].choice_lock
    choice_locked = _holds_choice(tables, s, side) and choice != 0 and choice in mon.moves
    return [
        m > 0
        and pp > 0
        and m != disabled
        and not (taunted and _t(tables, "move_power", m) == 0)
        and not (choice_locked and m != choice)
        and not _move_imprisoned(s, side, m)  # CHECK_INVALID_IMPRISONED (lib.c:2287)
        for m, pp in zip(mon.moves, mon.pp)
    ]


def _move_imprisoned(s: RefState, side: int, move: int) -> bool:
    """``Move_Imprisoned`` (lib.c:3316-3338): a battler of the other side has
    ``MOVE_EFFECT_IMPRISON`` and knows ``move``."""
    return any(
        b != side and s.sides[b].imprison and move and move in _active(s, b).moves for b in (0, 1)
    )


def _script_fails(s: RefState, side: int, effect: int, move_param1: int = 0) -> bool:
    """A user-effect script's MOVE_STATUS_FAILED (``_use_move``)."""
    sd = s.sides[side]
    if effect == EffectId.HELPING_HAND:  # BtlCmd_TryHelpingHand: no partner (script.c:6309-6311)
        return True
    if effect == EffectId.DRAIN and move_param1 == 1:  # Dream Eater, effect_script_0008.s:5
        return _active(s, 1 - side).status != Status.SLEEP
    if effect == EffectId.IMPRISON:
        own = [m for m in _active(s, side).moves if m]
        shared = any(m in _active(s, b).moves for b in (0, 1) if b != side for m in own)
        return bool(sd.imprison) or not shared
    flags = {
        EffectId.INGRAIN: sd.ingrain,
        EffectId.AQUA_RING: sd.aqua_ring,
        EffectId.MUD_SPORT: sd.mud_sport,
        EffectId.WATER_SPORT: sd.water_sport,
    }
    return bool(flags.get(effect, 0))


def _ingrain_anchors(tables, s: RefState, side: int) -> bool:
    """subscript_force_target_to_switch_or_flee.s:7: the defender's ``moveEffectsMask`` is
    exactly ``MOVE_EFFECT_INGRAIN`` (``OPCODE_EQU``): ingrained, no other kept bit, no Camouflage
    (its types are the personal row's)."""
    sd, mon = s.sides[side], _active(s, side)
    others = (
        sd.leech_seed, sd.lock_on, sd.perish_song, sd.yawn, sd.imprison, sd.imprisoned,
        sd.mud_sport, sd.water_sport, sd.aqua_ring, sd.minimized,
    )  # fmt: skip
    row = _t(tables, "species_form_row", mon.species, mon.form)
    party = [_t(tables, "species_types", row, i) for i in range(len(mon.types))]
    return bool(sd.ingrain) and not any(others) and list(mon.types) == party


def _locked_slot(s: RefState, side: int) -> int:
    """The move slot holding ``moveLockedInto``, -1 when not locked; 0 (action 0) for a lock on
    a move the battler does not know (a call's: FightCommand runs moveLockedInto anyway,
    bcp.c:1885-1888)."""
    locked = s.sides[side].locked_move
    moves = _active(s, side).moves
    if not locked:
        return -1
    return moves.index(locked) if locked in moves else 0


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
    """Play one turn (``engine.step.step``'s contract): a state paused at a U-turn's or Baton
    Pass's party list (``pivot_battler``) resumes with ``actions[pivot]`` the switch to the
    picked slot (``_resume``); a turn that reaches such a list stops there and writes the
    scratch (``_write_scratch``)."""
    s = copy.deepcopy(state)
    if s.over:
        return s
    actions = (int(actions[0]), int(actions[1]))
    if s.pivot_battler:
        return _resume(tables, s, actions, outcome)
    s.learn_pin = 0  # the turn's own pin (design §3.3(a))
    turn = _Turn(tables, outcome, entry_turn=s.turn + 1)
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
    # The search's turn powerMul (step.replacement_power_mul): the player's switch meets the
    # opponent's Pursuit, judged on the turn-start state.
    turn.power_mul = PURSUIT_POWER_MUL if _pursues(tables, s, 0, actions, choices) else 10
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
        # Two FIGHT commands are compared with ignoreQuickClaw = FALSE (bcp.c:772-786): each
        # activated Custap Berry sets its holder's custapBerry flag (lib.c:1296, :1362).
        if not early[0]:
            turn.custap = [_custap_active(tables, s, side) for side in (0, 1)]
    turn.actions, turn.choices, turn.items = actions, choices, items
    turn.commands, turn.order = commands, order
    try:
        _actions(turn, s, 0)
    except _Pause:
        _write_scratch(turn, s)
        return s
    return _end_turn(turn, s)


def _pursues(tables, s: RefState, side: int, actions, choices) -> bool:
    """``step.pursuit_fires``: ``side`` switches and the other side, which has not acted, chose
    Pursuit and can use it now (alive, not asleep or frozen, not loafing)."""
    pursuer = 1 - side
    move, struggling, _ = choices[pursuer]
    mon = _active(s, pursuer)
    return (
        _command(actions[side]) == SWITCH
        and _command(actions[pursuer]) == FIGHT
        and not struggling
        and _t(tables, "move_effect", move) == EffectId.PURSUIT
        and mon.hp > 0
        and mon.status not in (Status.SLEEP, Status.FREEZE)
        and not _truant_loafs(s, pursuer)
    )


def _actions(turn: _Turn, s: RefState, start: int) -> None:
    """The actions from ``start`` on in ``turn.order``; each ends in ``_finish_action``. A party
    list raises ``_Pause`` with ``turn.position`` at the paused action."""
    tables = turn.tables
    actions, choices, items, commands = turn.actions, turn.choices, turn.items, turn.commands
    for position in range(start, 2):
        side = turn.order[position]
        turn.position = position
        if s.over:
            break
        if turn.done[side] or _active(s, side).hp == 0:
            continue
        record = None
        if commands[side] == ITEM:
            _use_item(tables, s, side, items[side])
        elif commands[side] == SWITCH:
            pursuer_done = turn.done[1 - side]
            _switch(turn, s, side, actions[side] - MOVE_ACTIONS, actions, choices)
            if turn.done[1 - side] and not pursuer_done:  # its Pursuit hit (_pursuit)
                turn.pursued = 2 - side
        else:
            move, struggling, locked = choices[side]
            turn.attack_message = turn.move_hit = turn.damaged = False
            turn.move_cur = move
            turn.shell_dealt = 0
            # A foe that roosted this turn: the move's chart skips its Flying rows
            # (lib.c:2540-2543); only the chart reads the target's types for Flying.
            foe = _active(s, 1 - side)
            saved = list(foe.types)
            if turn.roosting[1 - side]:
                foe.types = [Type.MYSTERY if t == Type.FLYING else t for t in foe.types]
            try:
                record = _fight(
                    turn, s, side, actions[side], move, struggling, locked,
                    target_moved=position == 1,
                )  # fmt: skip
            finally:
                if _active(s, 1 - side) is foe:
                    foe.types = saved
            # TRIGGER_AFTER_HIT_EFFECTS (bcp.c:3897-3915): Shell Bell (:4736-4752), then Life
            # Orb's recoil, which needs SYSCTL_MOVE_HIT and a non-status moveCur (:4758-4772).
            if turn.move_hit:
                _shell_bell(turn, s, side, turn.shell_dealt)
            if turn.move_hit and _t(tables, "move_category", turn.move_cur) != STATUS_CLASS:
                _life_orb_recoil(turn, s, side)
            # UpdateFlagsWhenHit / UpdateMoveBuffers: the Choice lock (the foe is the defender
            # exactly when _fight returns its moveHit record); a Baton Pass that succeeded
            # (moveTemp, bcp.c:3928) never locks.
            defender = 1 - side if record is not None else None
            exempt = move == MOVE_BATON_PASS and turn.attack_message
            _update_choice_lock(turn, s, side, defender, move, exempt)
            # UpdateMoveBuffers (bcp.c:3936-3942) / UpdateFlagsWhenHit (:4499-4508): the
            # battle's movePrev is moveTemp (the chosen move) when the attack message showed.
            s.move_prev = move if turn.attack_message else 0
            # UpdateMoveBuffers (bcp.c:3936-3942): moveProtect = the move if its attack
            # message showed, else MOVE_NONE; the next TryProtection keeps the chain only for
            # Protect, Detect or Endure (script.c:5154-5158). Folded into protect_count here.
            if not (turn.attack_message and turn.move_cur in PROTECT_CHAIN_MOVES):
                s.sides[side].protect_count = 0
        _finish_action(turn, s, side, position, record)


def _finish_action(turn: _Turn, s: RefState, side: int, position: int, record) -> None:
    """An action's end: its faint check, ``moveHit``, ``MoveEnd`` and its payout."""
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
    # MoveEnd's AnyExpPayout, after the status recovery and before CheckBattleOver
    # (bcp.c:3965-3969): a KO that ends the battle pays too. The player's move is still to
    # run when the opponent acted first and the player chose a move (learn_pin).
    pending = position == 0 and side == 1 and turn.commands[0] == FIGHT
    _pay_exp(turn, s, pending, turn.actions[0])


def _end_turn(turn: _Turn, s: RefState) -> RefState:
    """The end of the turn after the actions."""
    tables, outcome = turn.tables, turn.outcome
    order = _speed_order(tables, s, outcome)  # BattleSystem_SortMonSpeedOrder, bcp.c:887
    _field_conditions(turn, s, order)
    _end_of_turn(turn, s, order)
    _future_sight(turn, s, order)
    # BattleSystem_SetupNextTurn (lib.c:2234) clears every flinch and unlocks a sleeping
    # battler (lib.c:2240-2243).
    for sd in s.sides:
        sd.flinched = 0
        if sd.party[sd.active].status == Status.SLEEP:
            _unlock(sd)
    s.turn += 1
    # The next ShowBattleMon (bcp.c:256) runs BattleSystem_TriggerEffectOnSwitch, whose
    # SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS (lib.c:4151-4165) checks every battler's item; with a
    # replacement pending it comes after it (ref_replace's switch-in effects).
    waiting = any(
        sd.party[sd.active].hp == 0
        and any(m.species and m.hp for i, m in enumerate(sd.party) if i != sd.active)
        for sd in s.sides
    )
    if not s.over and not waiting:
        for side in (0, 1):
            _held_item(turn, s, side)
    # The turn's speedRand (bcp.c:853-857), kept only while it can be read (_qc_kept).
    _keep_qc(tables, s, list(outcome.quick_claw))
    # turnFlags.roosting until SetupNextTurn (lib.c:2229-2231), after the replacements.
    _set_roosting(s, _roosting(s, turn) if waiting and not s.over else [0, 0])
    return s


def _set_roosting(s: RefState, flags: list[int]) -> None:
    for sd, flag in zip(s.sides, flags, strict=True):
        sd.roosting = flag


def _roosting(s: RefState, turn: _Turn) -> list[int]:
    """The turn's ``turnFlags.roosting`` per battler; a fainted battler's are cleared
    (CleanupFaintedMon, lib.c:2189-2192)."""
    return [int(turn.roosting[k] and _active(s, k).hp > 0) for k in (0, 1)]


def _write_scratch(turn: _Turn, s: RefState) -> None:
    """The turn stopped at a party list: ``BattleState``'s scratch (``engine.state``
    docstring), field for field as ``step`` writes it."""
    side = turn.order[turn.position]
    assert not turn.payout, "the payouts before a list are paid"
    s.pivot_battler = side + 1
    s.ts_cursor = turn.position
    s.ts_acted = [int(b in turn.order[: turn.position]) for b in (0, 1)]
    s.ts_order = list(turn.order)
    s.ts_actions = list(turn.actions)
    s.ts_move = [c[0] for c in turn.choices]
    s.ts_struggling = [int(c[1]) for c in turn.choices]
    s.ts_locked = [int(c[2]) for c in turn.choices]
    s.ts_item = list(turn.items)
    s.ts_custap = [int(f) for f in turn.custap]
    s.ts_pursued = turn.pursued
    s.ts_protect_ok = [int(f) for f in turn.protecting]
    _set_roosting(s, _roosting(s, turn))
    s.ts_taken = [list(row) for row in turn.taken]
    s.ts_dragged = list(turn.dragged)
    s.ts_move_failed = [int(f) for f in turn.move_failed]
    s.ts_used = turn.move_cur
    s.ts_damaged = int(turn.damaged)
    s.ts_dealt = turn.pivot_dealt
    fainted = set(turn.fainted)
    s.ts_alive = [
        sum(1 << i for i, m in enumerate(sd.party) if m.hp > 0 or (k, i) in fainted)
        for k, sd in enumerate(s.sides)
    ]
    s.ts_power_mul = turn.power_mul
    s.ts_switched = [int(c == SWITCH) for c in turn.commands]
    s.ts_pick_mul = turn.pick_mul
    for sd in s.sides:  # read by the order, next read at the next turn
        sd.qc_roll = 0


_SCRATCH_ZERO = {
    "pivot_battler": 0, "ts_cursor": 0, "ts_acted": [0, 0], "ts_order": [0, 0], "ts_actions": [0, 0],
    "ts_move": [0, 0], "ts_struggling": [0, 0], "ts_locked": [0, 0], "ts_item": [0, 0],
    "ts_custap": [0, 0], "ts_pursued": 0, "ts_protect_ok": [0, 0],
    "ts_taken": [[0] * 6, [0] * 6], "ts_dragged": [0, 0], "ts_move_failed": [0, 0],
    "ts_used": 0, "ts_damaged": 0, "ts_dealt": 0, "ts_alive": [0, 0], "ts_power_mul": 0,
    "ts_switched": [0, 0], "ts_pick_mul": 0,
}  # fmt: skip


def _resume(tables, s: RefState, actions: tuple[int, int], outcome) -> RefState:
    """Resume a turn paused at a party list: the turn's scratch back into ``_Turn``, the
    switch of ``subscript_show_party_list.s`` (``SwitchAndUpdateMon``, a Baton Pass keeping its
    effects, ``PokemonSendOut``, the hazards, the faint check: :9-24), then the rest of the paused
    action on the newcomer (``AfterMoveMessage``'s King's Rock, ``AfterMoveEffects``: the switch-in
    effects, the item checks, the thaw; the guarded Sticky Barb, Shell Bell and Life Orb skipped;
    ``UpdateMoveBuffers``, ``MoveEnd``), the later actions and the end of the turn."""
    side = s.pivot_battler - 1
    turn = _Turn(tables, outcome, entry_turn=s.turn + 1)
    turn.actions = tuple(s.ts_actions)
    turn.choices = [(s.ts_move[k], bool(s.ts_struggling[k]), bool(s.ts_locked[k])) for k in (0, 1)]
    turn.items = list(s.ts_item)
    turn.commands = [_command(a) for a in turn.actions]
    turn.order = list(s.ts_order)
    turn.position = s.ts_cursor
    turn.custap = [bool(f) for f in s.ts_custap]
    turn.pursued = s.ts_pursued
    turn.protecting = [bool(f) for f in s.ts_protect_ok]
    turn.roosting = [bool(sd.roosting) for sd in s.sides]
    turn.taken = [list(row) for row in s.ts_taken]
    turn.dragged = list(s.ts_dragged)
    turn.move_failed = [bool(f) for f in s.ts_move_failed]
    turn.power_mul, turn.pick_mul = s.ts_power_mul, s.ts_pick_mul
    turn.fainted = [
        (k, i)
        for k, sd in enumerate(s.sides)
        for i, m in enumerate(sd.party)
        if m.hp == 0 and s.ts_alive[k] >> i & 1
    ]
    for position in range(s.ts_cursor):
        turn.done[turn.order[position]] = True
    if s.ts_pursued:
        turn.done[s.ts_pursued - 1] = True
    for k in (0, 1):
        if s.ts_dragged[k]:
            turn.done[k] = True
    move, used = s.ts_move[side], s.ts_used
    damaged, dealt = bool(s.ts_damaged), s.ts_dealt
    position = s.ts_cursor
    for name, value in _SCRATCH_ZERO.items():
        setattr(s, name, copy.deepcopy(value))
    foe = 1 - side
    baton = _t(tables, "move_effect", used) == EffectId.BATON_PASS
    # The list (subscript_show_party_list.s): the outgoing Pokemon goes back to the party
    # without its toxic counter (battle_controller.c:1482).
    out = _active(s, side)
    if out.status == Status.TOXIC:
        out.status_counter = 0
    _enter(turn, s, side, actions[side] - MOVE_ACTIONS, baton)
    _send_out_exp(s, side)  # :18
    _entry_hazards(turn, s, side)  # :22
    _faint_check(turn, s)  # :23-24
    # AfterMoveMessage's King's Rock with the newcomer's item (bcp.c:4630-4650).
    o = turn.outcome
    holder = _active(s, foe)
    if (
        damaged
        and _item_effect(tables, _active(s, side)) == ItemId.KINGS_ROCK
        and o.flinch_item[side][0]
        and _t(tables, "move_flags", used) & MoveFlag.KINGS_ROCK
        and holder.hp > 0
        and position == 0
        and _ignorable(s, side, foe) not in (AbilityId.INNER_FOCUS, AbilityId.SHIELD_DUST)
    ):
        s.sides[foe].flinched = 1
    # AfterMoveEffects (bcp.c:3638): the switch-in effects, the item checks, the thaw.
    _switch_in_effects(turn, s, side)
    for battler in (side, foe):
        _held_item(turn, s, battler)
    _thaw_target(turn, s, side, used, dealt)
    for battler in (side, foe):
        _held_item(turn, s, battler)
    # UpdateMoveBuffers / UpdateFlagsWhenHit on the newcomer (bcp.c:3917-3944): no Choice lock
    # for U-turn under SYSCTL_UTURN_ACTIVE nor Baton Pass (moveTemp), movePrev and
    # movePrevByBattler the paused move's moveTemp, the protect chain reset.
    turn.attack_message = True
    turn.move_cur = used
    record = None if _move_target(tables, used) in SELF_DEFENDER else used
    defender = foe if record is not None else None
    exempt = move in (MOVE_U_TURN, MOVE_BATON_PASS)
    _update_choice_lock(turn, s, side, defender, move, exempt)
    s.move_prev = move
    s.sides[side].last_move = move
    s.sides[side].protect_count = 0
    _finish_action(turn, s, side, position, record)
    try:
        _actions(turn, s, position + 1)
    except _Pause:
        _write_scratch(turn, s)
        return s
    return _end_turn(turn, s)


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
    # subscript_show_party_list.s:18 / :22, and subscript_replace_fainted.s:58 / :62 in shift
    # style (the same order): the exp flag at the send-out, then the hazards.
    _send_out_exp(s, side)
    _entry_hazards(turn, s, side)
    _faint_check(turn, s)
    # TurnEnd's AnyExpPayout (bcp.c:1846) for a newcomer the hazards fainted. Learn work it
    # leaves inside the opponent's replacement loop is refused (window (c), design §3.3).
    if _pay_exp(turn, s) and side == 1 and not s.over:
        s.exp_refused = 2
    if not s.over:
        _switch_in_effects(turn, s, side)
    _keep_qc(tables, s, _qc_rolls(s))  # the slot's speedRand stays while it can be read
    # The turn's roosting flags end with the last replacement (SetupNextTurn, bcp.c:1856).
    if s.over or not any(
        sd.party[sd.active].hp == 0
        and any(m.species and m.hp for i, m in enumerate(sd.party) if i != sd.active)
        for sd in s.sides
    ):
        _set_roosting(s, [0, 0])
    return s
