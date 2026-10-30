"""The trainer-AI script commands Roark's flags execute (``docs/ai/script_commands.md``).

One JAX function per command row, each ``(ctx: AiContext, ...) -> AiContext``. The
decomp's interpreter (``TrainerAI_EvalMoves``, trainer_ai.c:483) walks the script with
jumps; here the scripts are straight-line (``nuzline.ai.policy``): every command runs,
and ``AiContext.live`` masks the commands of a path the decomp would not be on. A jump
command only stores its condition in ``AiContext.jump``; the script splits the mask on
it. Commands that write ``calcTemp`` do so on every path, which is safe because the
scripts always reload ``calcTemp`` before reading it and every value written is a
function of the state and the outcome alone.

Paths are relative to ``external/pokeplatinum``; ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``. The AI reads the true battle state except for
the player's moves and ability, which go through its knowledge (roark.md 6, 9):
``AiContext.seen_moves`` and ``AiContext.player_ability``.

Only the script paths of the moves in ``MOVE_EFFECTS`` are transcribed (roark.md 5);
``nuzline.ai.policy.ai_refusals`` refuses an opponent that knows any other move, and
the mechanics ``effectiveness_class`` does not implement.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.outcome import AiOutcome, script_roll_sites
from nuzline.ai.switching import INEFFECTIVE, apply_type_chart, type_chart_flags
from nuzline.data.tables import Tables
from nuzline.engine.actions import usable_moves
from nuzline.engine.damage import ai_damage_for_roll
from nuzline.engine.ids import Stat
from nuzline.engine.state import MOVES, BattleState

_I32 = jnp.int32

# FlagMoveDamageScore / IfCurrentMoveKills argument (include/constants/battle/trainer_ai.h:45).
USE_MAX_DAMAGE, ROLL_FOR_DAMAGE = 0, 1
# FlagMoveDamageScore results in calcTemp (trainer_ai.h:35-37).
AI_NO_COMPARISON_MADE, AI_NOT_HIGHEST_DAMAGE, AI_MOVE_IS_HIGHEST_DAMAGE = 0, 1, 2
SIM_ROLL_MAX = 15  # engine roll index of variance 100 (USE_MAX_DAMAGE)
STAGE_NEUTRAL = 6  # decomp stat stages run 0..12 with 6 neutral (engine 0)


class Battler(IntEnum):
    """Script battler argument, as the engine side it names (singles)."""

    DEFENDER = 0  # AI_BATTLER_DEFENDER: the player's active Pokemon
    ATTACKER = 1  # AI_BATTLER_ATTACKER: the AI's active Pokemon


class LoadType(IntEnum):
    """``LoadTypeFrom`` targets Roark's scripts use (trainer_ai.c:937)."""

    MOVE_TYPE = 0  # LOAD_MOVE_TYPE
    DEFENDER_TYPE_1 = 1  # LOAD_DEFENDER_TYPE_1
    DEFENDER_TYPE_2 = 2  # LOAD_DEFENDER_TYPE_2


class SideCondition(IntEnum):
    """``IfSideCondition`` masks Roark's scripts use."""

    STEALTH_ROCK = 0  # SIDE_CONDITION_STEALTH_ROCK, BattleState.stealth_rock


class TypeMulti(IntEnum):
    """``IfMoveEffectivenessEquals`` classes (include/constants/battle.h:144-153)."""

    IMMUNE = 0
    QUARTER = 10
    HALF = 20
    BASE = 40
    STAB = 60
    DOUBLE = 80
    QUADRUPLE = 160


class BattleEffect(IntEnum):
    """Decomp ``BATTLE_EFFECT_*`` ids (line - 1 of generated/move_battle_effects.txt)."""

    HIT = 0
    DEF_DOWN = 19
    FLINCH_HIT = 31
    DEF_DOWN_2 = 59
    HIT_BEFORE_SWITCH = 128
    STEALTH_ROCK = 266


# Decomp move id -> battle effect for every move whose script paths are transcribed
# (roark.md 4-5, plus the test fixture's Tackle); the same set as the script-roll site
# table (nuzline.ai.outcome.AI_SITE_MOVES). None of these effects is in
# sNoDamageCalcMoveEffects or sAltPowerMoveEffects (trainer_ai.c:30-62), so a move is
# damaging for the AI exactly when its power is greater than 1.
MOVE_EFFECTS: dict[int, int] = {
    446: BattleEffect.STEALTH_ROCK,  # Stealth Rock
    88: BattleEffect.HIT,  # Rock Throw
    228: BattleEffect.HIT_BEFORE_SWITCH,  # Pursuit
    103: BattleEffect.DEF_DOWN_2,  # Screech
    43: BattleEffect.DEF_DOWN,  # Leer
    29: BattleEffect.FLINCH_HIT,  # Headbutt
    33: BattleEffect.HIT,  # Tackle
}
_EFFECT_MOVES = jnp.array(list(MOVE_EFFECTS), _I32)
_EFFECT_IDS = jnp.array(list(MOVE_EFFECTS.values()), _I32)

# Decomp ids, checked against the manifest by tests/ai/test_policy.py.
ABILITY_IDS = {
    "Mold Breaker": 104,
    "Volt Absorb": 10,
    "Motor Drive": 78,
    "Water Absorb": 11,
    "Flash Fire": 18,
    "Wonder Guard": 25,
    "Levitate": 26,
    "Soundproof": 43,
    "Clear Body": 29,
    "White Smoke": 73,
    "Shadow Tag": 23,
    "Magnet Pull": 42,
    "Arena Trap": 71,
    "Normalize": 96,
    "Adaptability": 91,
    "Scrappy": 113,
    "Tinted Lens": 110,
    "Filter": 111,
    "Solid Rock": 116,
}
MOVE_IDS = {
    "Fissure": 90,
    "Horn Drill": 32,
    "U-turn": 369,
    "Roar": 46,
    "Whirlwind": 18,
    # Basic_CheckSoundproof's IfMoveEqualTo list (script.s:121-131).
    "Growl": 45,
    "Sing": 47,
    "Supersonic": 48,
    "Screech": 103,
    "Snore": 173,
    "Uproar": 253,
    "Metal Sound": 319,
    "GrassWhistle": 320,
    "Bug Buzz": 405,
    "Chatter": 448,
}
SOUND_MOVES = tuple(
    MOVE_IDS[n]
    for n in (
        "Growl",
        "Roar",
        "Sing",
        "Supersonic",
        "Screech",
        "Snore",
        "Uproar",
        "Metal Sound",
        "GrassWhistle",
        "Bug Buzz",
        "Chatter",
    )
)
TYPE_IDS = {
    "Ground": 4,
    "Ghost": 7,
    "Fire": 10,
    "Water": 11,
    "Electric": 13,
    "Psychic": 14,
}
ITEM_EXPERT_BELT = 268  # HOLD_EFFECT_POWER_UP_SE
_TRAPPING = jnp.array([ABILITY_IDS[n] for n in ("Shadow Tag", "Magnet Pull", "Arena Trap")], _I32)


class AiContext(NamedTuple):
    """The decomp's ``AIContext`` for one decision, plus the masks of the straight-line
    scripts."""

    tables: Tables
    state: BattleState
    outcome: AiOutcome
    scores: jnp.ndarray  # int32[4], moveScore
    slot: int  # moveSlot under evaluation (static: the scripts unroll over slots)
    move: jnp.ndarray  # int32, AI_CONTEXT.move: decomp id, 0 = MOVE_NONE (empty or no PP)
    live: jnp.ndarray  # bool: the decomp's script cursor is on the current path
    jump: jnp.ndarray  # bool: the last jump command's condition
    calc_temp: jnp.ndarray  # int32, calcTemp
    seen_moves: jnp.ndarray  # int32[4]: the player's active's moves the AI has seen, else 0
    player_ability: jnp.ndarray  # int32: LoadBattlerAbility DEFENDER (decomp id)
    damage: jnp.ndarray  # int32[4]: ai_damage of the AI's move slots, USE_MAX_DAMAGE
    sites: jnp.ndarray  # int32[4, 1 + MAX_EXPERT_SITES]: script_rolls index per site


def ai_damage(tables: Tables, state: BattleState, move, roll) -> jnp.ndarray:
    """``int32``: ``TrainerAI_CalcDamage`` (trainer_ai.c:2868) of ``move`` by the AI's
    active Pokemon against the player's, with variance ``85 + roll`` percent applied
    after STAB and the type chart (:3109; the battle applies it before them): the
    engine's ``ai_damage_for_roll`` (roark.md 7.1), 0 for a move that is not damaging
    (power <= 1) and for an immunity."""
    move = jnp.asarray(move, _I32)
    damage = ai_damage_for_roll(tables, state, 1, move, roll)
    return jnp.where(is_damaging(tables, move), damage, 0)


def is_damaging(tables: Tables, move) -> jnp.ndarray:
    """``bool``: the ``power > 1`` test of ``FlagMoveDamageScore`` (trainer_ai.c:1034) for
    the transcribed moves, which have no no-calc or alternate-power effect."""
    move = jnp.asarray(move, _I32)
    return (move != 0) & (tables.move_power[move] > 1)


def _slot_damage(tables: Tables, state: BattleState, rolls) -> jnp.ndarray:
    """``int32[4]``: ``TrainerAI_CalcAllDamage`` (trainer_ai.c:2799) over the AI's true
    moveset, whatever the PP, with ``rolls[i]`` for slot i."""
    moves = state.moves[1, state.active[1]].astype(_I32)
    return jax.vmap(lambda m, r: ai_damage(tables, state, m, r))(moves, jnp.asarray(rolls, _I32))


def move_effect(move) -> jnp.ndarray:
    """``int32``: the decomp battle effect of a transcribed move, -1 for any other."""
    hit = jnp.asarray(move, _I32) == _EFFECT_MOVES
    return jnp.where(jnp.any(hit), _EFFECT_IDS[jnp.argmax(hit)], -1)


def effectiveness_class(tables: Tables, state: BattleState, move) -> jnp.ndarray:
    """``int32``: the ``TypeMulti`` class of ``move`` by the AI's active Pokemon against
    the player's (``IfMoveEffectivenessEquals``, trainer_ai.c:1311; roark.md 7.2): 40
    through ``BattleSystem_ApplyTypeChart`` (STAB x15/10, then each chart entry with
    ``BattleSystem_Divide``), mapped 120 -> 80, 240 -> 160, 30 -> 20, 15 -> 10, and 0 when
    the status mask has an immunity bit. Normalize, Adaptability, Scrappy, Tinted Lens,
    Expert Belt, Levitate, Wonder Guard, Filter, Solid Rock, Iron Ball, Foresight,
    Miracle Eye, Roost, Magnet Rise and Ingrain change this value and are refused
    (``policy.ai_refusals``)."""
    move = jnp.asarray(move, _I32)
    d_types = state.types[0, state.active[0]]
    value = apply_type_chart(tables, state, 1, move, TypeMulti.BASE)
    stab_value = int(TypeMulti.STAB)
    mapped = jnp.select(
        [
            value == stab_value * 2,
            value == stab_value * 4,
            value == stab_value // 2,
            value == stab_value // 4,
        ],
        [TypeMulti.DOUBLE, TypeMulti.QUADRUPLE, TypeMulti.HALF, TypeMulti.QUARTER],
        value,
    )
    immune = (type_chart_flags(tables, move, d_types, state.gravity) & INEFFECTIVE) != 0
    # Struggle: ApplyTypeChart returns 40 unchanged and sets no bit (battle_lib.c:2573).
    return jnp.where(immune, TypeMulti.IMMUNE, mapped).astype(_I32)


def attacker_ability(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: the AI's active Pokemon's true ability (decomp id)."""
    a = state.active[1].astype(_I32)
    species = state.species[1, a].astype(_I32)
    return tables.species_abilities_decomp[species, state.ability_slot[1, a].astype(_I32)].astype(
        _I32
    )


def defender_ability(tables: Tables, state: BattleState, ability_guess) -> jnp.ndarray:
    """``int32``: ``LoadBattlerAbility DEFENDER`` (trainer_ai.c:1170; roark.md 9.2): the
    true ability when the AI has learned it or when it is Shadow Tag, Magnet Pull or
    Arena Trap; else the guessed species slot ``ability_guess`` when the species has two
    abilities; else its only one. The engine has no ability suppression (Gastro Acid),
    so the ``ABILITY_NONE`` case never applies."""
    d = state.active[0].astype(_I32)
    species = state.species[0, d].astype(_I32)
    both = tables.species_abilities_decomp[species].astype(_I32)
    true = both[state.ability_slot[0, d].astype(_I32)]
    known = (state.ai_ability_known[d] > 0) | jnp.isin(true, _TRAPPING)
    guess = both[jnp.clip(jnp.asarray(ability_guess, _I32), 0, 1)]
    only = jnp.where(both[0] != 0, both[0], both[1])
    resolved = jnp.where((both[0] != 0) & (both[1] != 0), guess, only)
    return jnp.where(known, true, resolved)


def new_context(tables: Tables, state: BattleState, outcome: AiOutcome) -> AiContext:
    """``TrainerAI_Init`` (trainer_ai.c:211) with ``AI_INIT_SCORE_ALL_MOVES``: every score
    100, then 0 for the slots ``BattleSystem_CheckInvalidMoves`` flags (the engine's
    ``usable_moves``: empty, no PP, Taunt on a power-0 move).

    The seen moves are the stored ``ai_seen_moves`` plus the player's pending
    ``last_move``: ``TrainerAI_MainSingles`` runs ``TrainerAI_RecordLastMove`` (:295)
    before scoring, so the scores already know the move the player used last turn;
    ``step`` commits it to ``ai_seen_moves`` when the AI fights."""
    d = state.active[0]
    moves = state.moves[0, d].astype(_I32)
    pending = state.last_move[0].astype(_I32)
    known = (state.ai_seen_moves[d] > 0) | ((moves == pending) & (pending != 0))
    seen = jnp.where(known, moves, 0).astype(_I32)
    scores = jnp.where(usable_moves(tables, state, 1), 100, 0).astype(_I32)
    false = jnp.asarray(False)
    return AiContext(
        tables=tables,
        state=state,
        outcome=outcome,
        scores=scores,
        slot=0,
        move=jnp.asarray(0, _I32),
        live=false,
        jump=false,
        calc_temp=jnp.asarray(0, _I32),
        seen_moves=seen,
        player_ability=defender_ability(tables, state, outcome.ability_guess),
        damage=_slot_damage(tables, state, jnp.full((MOVES,), SIM_ROLL_MAX, _I32)),
        sites=script_roll_sites(state),
    )


def at_slot(ctx: AiContext, slot: int) -> AiContext:
    """``AI_EVAL_STEP_INIT`` (trainer_ai.c:487): the move of ``slot``, ``MOVE_NONE`` when
    it has no PP; a ``MOVE_NONE`` slot runs no command."""
    state = ctx.state
    a = state.active[1]
    move = jnp.where(state.pp[1, a, slot] > 0, state.moves[1, a, slot], 0).astype(_I32)
    return ctx._replace(
        slot=jnp.asarray(slot, _I32), move=move, live=move != 0, jump=jnp.asarray(False)
    )


def _side(battler: Battler) -> int:
    return int(battler)


def _jump(ctx: AiContext, cond) -> AiContext:
    return ctx._replace(jump=jnp.asarray(cond))


# --- The command functions, one per row of docs/ai/script_commands.md -------------


def if_target_is_partner(ctx: AiContext) -> AiContext:
    """``IfTargetIsPartner`` (trainer_ai.c:2630): never in a single battle."""
    return _jump(ctx, False)


def if_move_equal_to(ctx: AiContext, move: int) -> AiContext:
    """``IfMoveEqualTo`` (trainer_ai.c:834)."""
    return _jump(ctx, ctx.move == move)


def flag_move_damage_score(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``FlagMoveDamageScore`` (trainer_ai.c:1012): ``AI_NO_COMPARISON_MADE`` for a move
    that is not damaging; else ``AI_MOVE_IS_HIGHEST_DAMAGE`` when no slot of the true
    moveset estimates strictly more damage, ``AI_NOT_HIGHEST_DAMAGE`` otherwise."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        damage = _slot_damage(ctx.tables, ctx.state, ctx.outcome.sim_roll)
    else:
        damage = ctx.damage
    highest = ~jnp.any(damage > damage[ctx.slot])
    flag = jnp.where(highest, AI_MOVE_IS_HIGHEST_DAMAGE, AI_NOT_HIGHEST_DAMAGE)
    value = jnp.where(is_damaging(ctx.tables, ctx.move), flag, AI_NO_COMPARISON_MADE)
    return ctx._replace(calc_temp=value.astype(_I32))


def if_loaded_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedEqualTo`` (trainer_ai.c:786)."""
    return _jump(ctx, ctx.calc_temp == value)


def if_move_effectiveness_equals(ctx: AiContext, cls: TypeMulti) -> AiContext:
    """``IfMoveEffectivenessEquals`` (trainer_ai.c:1311)."""
    return _jump(ctx, effectiveness_class(ctx.tables, ctx.state, ctx.move) == int(cls))


def load_battler_ability(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadBattlerAbility`` (trainer_ai.c:1170): the attacker's true ability, or the
    defender's ability as the AI resolves it (``AiContext.player_ability``; its coin
    flip is ``AiOutcome.ability_guess``, one per decision, roark.md 8.2)."""
    if battler == Battler.ATTACKER:
        value = attacker_ability(ctx.tables, ctx.state)
    else:
        value = ctx.player_ability
    return ctx._replace(calc_temp=jnp.asarray(value, _I32))


def load_type_from(ctx: AiContext, target: LoadType) -> AiContext:
    """``LoadTypeFrom`` (trainer_ai.c:937): the move's type, or the defender's true type
    1 or 2."""
    state = ctx.state
    d_types = state.types[0, state.active[0]].astype(_I32)
    if target == LoadType.MOVE_TYPE:
        value = ctx.tables.move_type[ctx.move].astype(_I32)
    elif target == LoadType.DEFENDER_TYPE_1:
        value = d_types[0]
    else:
        value = d_types[1]
    return ctx._replace(calc_temp=value)


def if_temp_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfTempEqualTo`` (trainer_ai.c:1078)."""
    return _jump(ctx, ctx.calc_temp == value)


def go_to(ctx: AiContext) -> AiContext:
    """``GoTo`` (trainer_ai.c:2561): an unconditional jump."""
    return _jump(ctx, True)


def if_loaded_not_equal_to(ctx: AiContext, value: int) -> AiContext:
    """``IfLoadedNotEqualTo`` (trainer_ai.c:798)."""
    return _jump(ctx, ctx.calc_temp != value)


def if_current_move_effect_equal_to(ctx: AiContext, effect: BattleEffect) -> AiContext:
    """``IfCurrentMoveEffectEqualTo`` (trainer_ai.c:1441)."""
    return _jump(ctx, move_effect(ctx.move) == int(effect))


def _stage(ctx: AiContext, battler: Battler, stat: Stat) -> jnp.ndarray:
    """The battler's true stat stage in decomp units (engine + 6)."""
    return ctx.state.stages[_side(battler), stat].astype(_I32) + STAGE_NEUTRAL


def if_stat_stage_equal_to(ctx: AiContext, battler: Battler, stat: Stat, value: int) -> AiContext:
    """``IfStatStageEqualTo`` (trainer_ai.c:1495), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) == value)


def if_side_condition(ctx: AiContext, battler: Battler, condition: SideCondition) -> AiContext:
    """``IfSideCondition`` (trainer_ai.c:732) for ``SIDE_CONDITION_STEALTH_ROCK``."""
    del condition  # SideCondition has only STEALTH_ROCK
    return _jump(ctx, ctx.state.stealth_rock[_side(battler)] > 0)


def count_alive_party_battlers(ctx: AiContext, battler: Battler) -> AiContext:
    """``CountAlivePartyBattlers`` (trainer_ai.c:1126): the side's living party members
    other than the active one."""
    state, side = ctx.state, _side(battler)
    alive = (state.species[side] > 0) & (state.hp[side] > 0)
    alive = alive & (jnp.arange(alive.shape[0]) != state.active[side])
    return ctx._replace(calc_temp=jnp.sum(alive.astype(_I32)))


def add_to_move_score(ctx: AiContext, value: int) -> AiContext:
    """``AddToMoveScore`` (trainer_ai.c:576): add, then raise a negative score to 0 (no
    upper clamp; Roark's largest score, 106, fits the decomp's ``s8``)."""
    current = ctx.scores[ctx.slot]
    new = jnp.maximum(current + value, 0)
    return ctx._replace(scores=ctx.scores.at[ctx.slot].set(jnp.where(ctx.live, new, current)))


def pop_or_end(ctx: AiContext) -> AiContext:
    """``PopOrEnd`` (trainer_ai.c:2568): Roark's scripts never push, so this ends the
    slot for this flag."""
    return ctx._replace(live=jnp.asarray(False))


def if_current_move_kills(ctx: AiContext, use_damage_roll: int) -> AiContext:
    """``IfCurrentMoveKills`` (trainer_ai.c:1525): the move is damaging and the
    defender's true HP is at most its estimated damage."""
    if use_damage_roll == ROLL_FOR_DAMAGE:
        roll = ctx.outcome.sim_roll[ctx.slot]
    else:
        roll = SIM_ROLL_MAX
    damage = ai_damage(ctx.tables, ctx.state, ctx.move, roll)
    state = ctx.state
    hp = state.hp[0, state.active[0]].astype(_I32)
    return _jump(ctx, is_damaging(ctx.tables, ctx.move) & (hp <= damage))


def if_random_less_than(ctx: AiContext, n: int, site) -> AiContext:
    """``IfRandomLessThan`` (trainer_ai.c:528): ``RandNext() % 256 < n``, which is
    ``AiOutcome.script_rolls[site] == 1`` (probability n/256, roark.md 8.3). ``site`` is
    the draw's index from ``script_roll_sites``; ``n`` is the threshold that index's
    probability already carries (``script_roll_thresholds``)."""
    del n  # the draw's probability lives in the outcome's site table
    roll = jnp.take(ctx.outcome.script_rolls, site, mode="fill", fill_value=0)
    return _jump(ctx, roll == 1)


def _knows(ctx: AiContext, battler: Battler, move: int) -> jnp.ndarray:
    state = ctx.state
    if battler == Battler.ATTACKER:
        moves = state.moves[1, state.active[1]].astype(_I32)
    else:
        moves = ctx.seen_moves  # AI_CONTEXT.battlerMoves (trainer_ai.c:1665)
    return jnp.any(moves == move)


def if_move_known(ctx: AiContext, battler: Battler, move: int) -> AiContext:
    """``IfMoveKnown`` (trainer_ai.c:1627): the attacker's true moveset, or the moves the
    AI has seen from the defender."""
    return _jump(ctx, _knows(ctx, battler, move))


def _hp_percent(ctx: AiContext, battler: Battler) -> jnp.ndarray:
    state, side = ctx.state, _side(battler)
    mon = state.active[side]
    return state.hp[side, mon].astype(_I32) * 100 // state.max_hp[side, mon].astype(_I32)


def if_hp_percent_less_than(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentLessThan`` (trainer_ai.c:588): ``curHP * 100 / maxHP < percent``."""
    return _jump(ctx, _hp_percent(ctx, battler) < percent)


def if_stat_stage_greater_than(
    ctx: AiContext, battler: Battler, stat: Stat, value: int
) -> AiContext:
    """``IfStatStageGreaterThan`` (trainer_ai.c:1480), ``value`` in decomp units."""
    return _jump(ctx, _stage(ctx, battler, stat) > value)


def if_hp_percent_greater_than(ctx: AiContext, battler: Battler, percent: int) -> AiContext:
    """``IfHPPercentGreaterThan`` (trainer_ai.c:603): ``curHP * 100 / maxHP > percent``."""
    return _jump(ctx, _hp_percent(ctx, battler) > percent)


def load_is_first_turn_in_battle(ctx: AiContext, battler: Battler) -> AiContext:
    """``LoadIsFirstTurnInBattle`` (trainer_ai.c:2480): TRUE (1) unless
    ``fakeOutTurnNumber < totalTurns`` (``BattleState.fake_out_turn``, ``turn``)."""
    state = ctx.state
    first = state.fake_out_turn[_side(battler)].astype(_I32) >= state.turn.astype(_I32)
    return ctx._replace(calc_temp=first.astype(_I32))


def if_move_not_known(ctx: AiContext, battler: Battler, move: int) -> AiContext:
    """``IfMoveNotKnown`` (trainer_ai.c:1683): the negation of ``IfMoveKnown``."""
    return _jump(ctx, ~_knows(ctx, battler, move))
