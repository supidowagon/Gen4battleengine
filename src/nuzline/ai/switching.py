"""The voluntary-switch draw layout of ``TrainerAI_ShouldSwitch`` (trainer_ai.c:3894).

``docs/ai/roark.md`` 11.1 lists the checks. Three kinds of draw change the result: the
first-success draws of ``AI_CannotDamageWonderGuard`` (step 4, trainer_ai.c:3286-3348; ruling
G-23), the keep draw of each super-effective damaging move of the AI's active Pokemon ``A``
against the player's active ``D`` (step 8), and the first-success draws of the two party
passes (steps 10 and 11). ``switch_sites`` lists
them in the decomp's order as ``SwitchSite`` kinds, one per site, so that
``AiOutcome.switch_stop`` (the first site that stops the check) is one exact
categorical (``nuzline.ai.outcome``). ``switch_action`` maps a stop to the action it
means. Both come from the same layout, so the sampler, the enumerator and Task 3's
``maybe_switch`` cannot disagree on it.

How many sites the outcome encodes is the AI layout's ``switch_sites`` (``ai.layout``),
passed to the functions here as ``sites`` (static): ``SWITCH_SITES`` (9) for vanilla
Roark's three Pokemon, ``KAIZO_SWITCH_SITES`` (60) for a party of six. The candidate draws
are the same for both: one Wonder Guard draw per (bench Pokemon, move slot)
(``AI_CannotDamageWonderGuard``, :3315-3344, the draw at :3338), the keep draw of each of the
active's four move slots (``AI_HasSuperEffectiveMove``, trainer_ai.c:3576-3592, the draw at
:3588), then for each pass one draw per (bench Pokemon, move slot)
(``AI_HasPartyMemberWithSuperEffectiveMove``, :3756-3804, the draw at :3796). The bench excludes the active slot
(``selectedPartySlot``) and ``aiSwitchedPartySlot`` (:3762-3765), which is 6 at every
``TrainerAI_ShouldSwitch`` call: ``BattleContext_Init`` sets it (battle_lib.c:2015) after
every action (battle_controller_player.c:3998) and at the turn's end
(battle_controller_player.c:1855), and inside the check only a branch that returns TRUE
writes it (:3797). A bench Pokemon can be in both passes: ``UpateMoveStatusForTypeMul``
(battle_lib.c:2762-2781) sets ``NOT_VERY_EFFECTIVE`` on a resisted chart row after an immune
row cleared it (Ground on Flying/Bug), and Wonder Guard's ``INEFFECTIVE`` (:2734-2739) never
clears ``NOT_VERY_EFFECTIVE``; so the passes hold up to ``2 * 5 * 4 = 40`` draws. The Wonder
Guard draws (step 4) exist only when no move of ``A`` is super effective (:3300-3311), so never
beside a keep draw: at most ``max(4, 5 * 4) + 40 = 60`` candidates (ruling G-33), and a party
of six never has more sites than ``KAIZO_SWITCH_SITES``
(``tests/ai/test_kaizo_switching.py::test_kaizo_switch_sites_reach_sixty``).

Wonder Guard (ruling G-23). ``AI_CannotDamageWonderGuard`` (step 4) runs when ``D``'s
``battleMons[].ability`` is Wonder Guard (no Mold Breaker test, no knowledge: the true
ability) and no move of ``A`` sets ``MOVE_STATUS_SUPER_EFFECTIVE`` through ``ApplyTypeChart``
(:3300-3311, the keep draws' flags); then each bench Pokemon's moves in party order through
``CalcEffectiveness`` against ``D`` (:3328-3336), and each super-effective one draws
``RandNext() % 3 < 2`` (:3338): the first success switches to that Pokemon. Wonder Guard's
own bit in those calls (``MOVE_STATUS_WONDER_GUARD`` / ``INEFFECTIVE``, battle_lib.c:
2649-2654, :2734-2739) never clears the super-effective bit, so the keep draws and the passes'
second calls are unchanged. The passes' first call (:3769-3777) has the AI's bench Pokemon as
the defender: a Wonder Guard bench Pokemon is ``INEFFECTIVE`` against the player's last move
(``moveHit``) unless the player's active has Mold Breaker, the move is super effective or not
on its damaging turn (``damage.on_damaging_turn``: ``BattleContext_Init`` cleared
``SYSCTL_LAST_OF_MULTI_TURN``, so Fire Fang never is), which makes it a pass-1 candidate.

The gates in steps 1, 2 and 9 are implemented from the state (``_layout``): ``A`` trapped,
``statusVolatile & VOLATILE_CONDITION_TRAPPED`` (trainer_ai.c:3906), which is Bind's
``VOLATILE_CONDITION_BIND`` (``trapped > 0``) or ``VOLATILE_CONDITION_MEAN_LOOK``
(``mean_look``) (condition.h:77-78); ``A`` ingrained, ``moveEffectsMask &
MOVE_EFFECT_INGRAIN`` (``ingrain``, :3907); Shadow Tag or Arena Trap on ``D`` (:3908-3909);
Magnet Pull on ``D`` with ``A`` Steel type (:3910-3911); no other living party member; and
positive stages summing to 4 or more (which ends the check after the keep draws).

Step 3, ``AI_PerishSongKO`` (trainer_ai.c:3263-3272), runs after the trap gate and with a bench
(:3906-3913, :3941), before the Wonder Guard step and without a draw: the AI's active has
``MOVE_EFFECT_PERISH_SONG`` and ``perishSongTurns == 0`` (it faints at this turn's end:
battle_controller_player.c:1799-1809; the comment at :3253-3262 says "does nothing", the code
does not) -> it switches (``aiSwitchedPartySlot = 6``, :3269-3270, TRUE). ``_Layout.perish``;
no site exists then, and ``switch_action`` returns the ``TrainerAI_PickCommand`` slot (:4001-4026,
``BattleAI_PostKOSwitchIn``, battle_lib.c:7923, ``replacement``) whatever the stop (E1 ruling G-64).

Effectiveness is the decomp's status mask, built in chart-row order
(``tables.type_chart_order``): ``BattleSystem_ApplyTypeChart`` (battle_lib.c:2560) for
the keep draws, where only moves with power update the super/not-very-effective bits
(``ApplyTypeMultiplier``, :7542), and ``BattleSystem_CalcEffectiveness`` (:2681) for
the party passes, which updates them for every move (``UpateMoveStatusForTypeMul``,
:2781). Struggle sets no bit (:2573, :2687). Gravity lifts the Ground-on-Flying
immunity in both (``BasicTypeMulApplies`` :2529, ``NoImmunityOverrides`` :2759). Mold
Breaker only matters against Levitate, which is refused, and the AI bench's Wonder Guard
(above).

Refused, never approximated (``switching_refusals`` names each one):

* abilities on either side that change these masks: Levitate, Scrappy, Normalize; and on
  the AI's side the steps 6-7 abilities (Flash Fire, Water Absorb,
  Volt Absorb: ``AI_HasAbsorbAbilityInParty``; Natural Cure:
  ``AI_IsAsleepWithNaturalCure``), whose draws are not modelled;
* moves whose state the engine does not carry: Miracle Eye, and the variable-type
  moves Weather
  Ball, Natural Gift and Judgment (``TrainerAI_MoveType``, :3127; Hidden Power's IV type is
  modelled on both sides: ``damage.ai_move_type``, and a bench member's own,
  ``Move_CalcVariableType``, battle_lib.c:8162-8175);
* a state where ``AI_OnlyIneffectiveMoves`` (step 5, :3361) applies;
* a layout with more than ``sites`` sites (never under ``KAIZO_SWITCH_SITES``, which is the
  six-party maximum).

Called on concrete arrays, ``switch_sites`` and ``switch_action`` raise
``UnsupportedSwitching`` for these. Traced (jit, vmap) they cannot raise, so every
entry becomes ``SwitchSite.REFUSED``: its stop probability is NaN, the sampler gives
``switch_stop = SWITCH_REFUSED`` and ``switch_action`` gives ``REFUSED_ACTION``. The
coverage registry (Task 5) runs ``switching_refusals`` on the fight first.
"""

from __future__ import annotations

from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import (
    ai_move_type,
    decomp_abilities,
    divide,
    hidden_power,
    iron_ball,
    on_damaging_turn,
)
from nuzline.engine.ids import ItemId
from nuzline.engine.state import MOVES, SLOTS, BattleState

SWITCH_SITES = 9  # roark.md 11.1: 1 keep draw + 4 + 4 party-pass draws for Roark
SWITCH_NONE = SWITCH_SITES  # switch_stop when no draw stops the check (vanilla layout)
# Every candidate draw of a party of six (module docstring, ruling G-33): the Wonder Guard draws
# (5 bench Pokemon x 4 moves) or the 4 keep draws, never both, then 5 x 4 in each of the two
# passes: 20 + 40 = 60. A layout's switch_stop is ``sites`` when none stops.
KAIZO_SWITCH_SITES = max(MOVES, (SLOTS - 1) * MOVES) + 2 * (SLOTS - 1) * MOVES
SWITCH_REFUSED = -1  # switch_stop of a refused layout (traced only)
NO_SWITCH = -1  # switch_action: the AI does not switch
REFUSED_ACTION = -2  # switch_action of a refused layout (traced only)
_I8, _I32 = jnp.int8, jnp.int32


class SwitchSite(IntEnum):
    """Kind of one voluntary-switch draw (``AiOutcome.switch_stop``; roark.md 11.1)."""

    NONE = 0  # unused entry: never stops
    # AI_HasSuperEffectiveMove(..., FALSE) (trainer_ai.c:3560): RandNext() % 10 != 0
    # keeps the battler in (:3588), 58982 of 65536 u16 values.
    KEEP = 1
    # AI_HasPartyMemberWithSuperEffectiveMove(..., 0x8 = INEFFECTIVE, 2) (:3975):
    # RandNext() % 2 == 0 switches (:3796), 32768 of 65536.
    PASS_INEFFECTIVE = 2
    # The same with 0x4 = NOT_VERY_EFFECTIVE, 3 (:3981): RandNext() % 3 == 0, 21846 of 65536.
    PASS_NOT_VERY_EFFECTIVE = 3
    # The layout is refused (module docstring); only produced under tracing.
    REFUSED = 4
    # AI_CannotDamageWonderGuard (:3286): RandNext() % 3 < 2 switches (:3338), 43691 of 65536
    # (ruling G-23).
    WONDER_GUARD = 5


# MOVE_STATUS_* bits (include/constants/battle/moves.h:58-60).
SUPER_EFFECTIVE, NOT_VERY_EFFECTIVE, INEFFECTIVE = 1 << 1, 1 << 2, 1 << 3
# MOVE_STATUS_MAGNET_RISE (moves.h:77): part of MOVE_STATUS_IMMUNE (:87); ApplyTypeChart's walk is
# skipped for it, so no other bit of the chart is set (G-66).
MAGNET_RISE = 1 << 20

# Decomp ids, checked against the manifest by tests/ai/test_switching.py.
MOVE_STRUGGLE = 165
_TYPE_FLYING, _TYPE_STEEL, _TYPE_GROUND = 2, 8, 4
ABILITY_IDS = {
    "Levitate": 26,
    "Wonder Guard": 25,
    "Scrappy": 113,
    "Normalize": 96,
    "Shadow Tag": 23,
    "Arena Trap": 71,
    "Magnet Pull": 42,
    "Flash Fire": 18,
    "Water Absorb": 11,
    "Volt Absorb": 10,
    "Natural Cure": 30,
    "Mold Breaker": 104,
}
MOVE_IDS = {
    "Magnet Rise": 393,
    "Roost": 355,
    "Miracle Eye": 357,
    "Ingrain": 275,
    "Perish Song": 195,
    "Hidden Power": 237,
    "Weather Ball": 311,
    "Natural Gift": 363,
    "Judgment": 449,
}
# Refused abilities: name -> what the decomp does with it that is not implemented.
_ANY_SIDE_ABILITIES = {
    "Levitate": "LEVITATED / INEFFECTIVE in ApplyTypeChart and CalcEffectiveness",
    "Scrappy": "skips the Ghost immunities",
    "Normalize": "makes every move Normal",
}
_AI_SIDE_ABILITIES = {
    "Flash Fire": "AI_HasAbsorbAbilityInParty",
    "Water Absorb": "AI_HasAbsorbAbilityInParty",
    "Volt Absorb": "AI_HasAbsorbAbilityInParty",
    "Natural Cure": "AI_IsAsleepWithNaturalCure",
}
_MOVES = {
    "Miracle Eye": "lifts the Dark immunity",
    "Weather Ball": "variable type (TrainerAI_MoveType)",
    "Natural Gift": "variable type (TrainerAI_MoveType)",
    "Judgment": "variable type (TrainerAI_MoveType)",
}
# What the AI refuses on the player's side (side 0): the any-side abilities and every refused
# move (Hidden Power's IV type is modelled on both sides, see ``_layout``). Decomp ids.
PLAYER_ABILITY_REFUSALS = {
    ABILITY_IDS[n]: f"ability {n} ({why}) is not implemented"
    for n, why in _ANY_SIDE_ABILITIES.items()
}
PLAYER_MOVE_REFUSALS = {
    MOVE_IDS[n]: f"move {n} ({why}) is not implemented" for n, why in _MOVES.items()
}
_ANY_SIDE_IDS = jnp.array([ABILITY_IDS[n] for n in _ANY_SIDE_ABILITIES], _I32)
_AI_SIDE_IDS = jnp.array([ABILITY_IDS[n] for n in _AI_SIDE_ABILITIES], _I32)
_MOVE_REFUSED_IDS = jnp.array([MOVE_IDS[n] for n in _MOVES], _I32)
_ABILITY_NAMES = {i: n for n, i in ABILITY_IDS.items()}
_MOVE_NAMES = {i: n for n, i in MOVE_IDS.items()}

# Wonder Guard draws, then keep draws, then pass 1, then pass 2.
_CANDIDATES = MOVES + 3 * SLOTS * MOVES
_CANDIDATE_KINDS = jnp.array(
    [SwitchSite.WONDER_GUARD] * (SLOTS * MOVES)
    + [SwitchSite.KEEP] * MOVES
    + [SwitchSite.PASS_INEFFECTIVE] * (SLOTS * MOVES)
    + [SwitchSite.PASS_NOT_VERY_EFFECTIVE] * (SLOTS * MOVES),
    _I8,
)
_BENCH_TARGETS = [j for j in range(SLOTS) for _ in range(MOVES)]
_CANDIDATE_TARGETS = jnp.array(_BENCH_TARGETS + [-1] * MOVES + _BENCH_TARGETS * 2, _I8)
_WONDER_GUARD, _MOLD_BREAKER = ABILITY_IDS["Wonder Guard"], ABILITY_IDS["Mold Breaker"]


class UnsupportedSwitching(ValueError):
    """The voluntary-switch rule meets a mechanic this module does not implement."""


def _apply_mul(mask, mul, update):
    """One chart entry's ``ApplyTypeMultiplier`` / ``UpateMoveStatusForTypeMul``
    update of the status mask (``mul`` in tenths; 10 leaves it unchanged)."""
    has_se = (mask & SUPER_EFFECTIVE) != 0
    has_nve = (mask & NOT_VERY_EFFECTIVE) != 0
    immune = (mask | INEFFECTIVE) & ~(SUPER_EFFECTIVE | NOT_VERY_EFFECTIVE)
    nve = jnp.where(has_se, mask & ~SUPER_EFFECTIVE, mask | NOT_VERY_EFFECTIVE)
    se = jnp.where(has_nve, mask & ~NOT_VERY_EFFECTIVE, mask | SUPER_EFFECTIVE)
    out = jnp.where(mul == 0, immune, mask)
    out = jnp.where((mul == 5) & update, nve, out)
    return jnp.where((mul == 20) & update, se, out)


def chart_multipliers(
    tables: Tables,
    move,
    def_types,
    gravity,
    identified=False,
    move_type=None,
    grounded=False,
    roosting=False,
    magnet=False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The two ``ApplyTypeMultiplier`` factors (``int32`` tenths, 10 = no entry) of
    ``move``'s type against a defender of ``def_types`` (int8[2]), in the order
    ``BattleSystem_ApplyTypeChart`` meets them (chart-row order, battle_lib.c:2611-2646).
    Type 2 is skipped when it equals type 1 (:2635); Gravity lifts the Ground-on-Flying
    immunity (``BasicTypeMulApplies``, :2529). ``identified`` (the defender's Foresight
    flag) drops the Ghost-immunity rows after the 0xFE separator (:2615); only
    ``BattleSystem_ApplyTypeChart`` reads it, not ``CalcEffectiveness`` (:2681).
    ``move_type`` overrides the move's table type (Hidden Power's IV type,
    ``TrainerAI_MoveType``, trainer_ai.c:3208). ``grounded`` (the defender's
    ``MOVE_EFFECT_INGRAIN``) lifts the Flying immunity and ``roosting`` (its
    ``turnFlags.roosting``) skips every Flying row, ``BasicTypeMulApplies`` (:2534-2543):
    ``ApplyTypeChart`` only; ``CalcEffectiveness``'s ``NoImmunityOverrides`` (:2759) has
    neither. ``magnet`` (the defender's Magnet Rise turns without Ingrain or an Iron Ball,
    ``defender_chart_flags``): a Ground move skips the walk (:2605-2609): both factors are 10,
    the value keeps only STAB."""
    if move_type is None:
        move_type = tables.move_type[jnp.asarray(move, _I32)]
    move_type = jnp.asarray(move_type, _I32)
    t1, t2 = jnp.asarray(def_types, _I32)[0], jnp.asarray(def_types, _I32)[1]
    identified = jnp.asarray(identified, jnp.bool_)
    grounded = (jnp.asarray(gravity) > 0) | jnp.asarray(grounded, jnp.bool_)

    def entry(t):
        mul = tables.type_chart[move_type, t].astype(_I32)
        mul = jnp.where(identified & (tables.type_chart_foresight[move_type, t] != 0), 10, mul)
        lifted = grounded & (t == _TYPE_FLYING) & (mul == 0)
        skipped = jnp.asarray(roosting, jnp.bool_) & (t == _TYPE_FLYING)
        return jnp.where(lifted | skipped, 10, mul)

    ground = jnp.asarray(magnet, jnp.bool_) & (move_type == _TYPE_GROUND)
    m1 = jnp.where(ground, 10, entry(t1))
    m2 = jnp.where((t2 != t1) & ~ground, entry(t2), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    return jnp.where(t2_first, m2, m1), jnp.where(t2_first, m1, m2)


def party_move_types(tables: Tables, state: BattleState, side: int = 1) -> jnp.ndarray:
    """``int32[SLOTS, MOVES]``: ``Move_CalcVariableType`` (battle_lib.c:8097) of each party
    member's moves on ``side``: Hidden Power its own IV type (:8162-8175), any other move
    its table type (Natural Gift, Judgment and Weather Ball are refused)."""
    moves = state.moves[side].astype(_I32)
    iv_types = jax.vmap(lambda ivs: hidden_power(ivs)[0])(state.ivs[side])  # [SLOTS]
    return jnp.where(
        moves == MOVE_IDS["Hidden Power"], iv_types[:, None], tables.move_type[moves].astype(_I32)
    )


def apply_type_chart(
    tables: Tables, state: BattleState, atk_side, move, value, move_type=None
) -> jnp.ndarray:
    """``int32``: ``BattleSystem_ApplyTypeChart`` (battle_lib.c:2560) on ``value`` for
    ``move`` by ``atk_side``'s active Pokemon against the other side's: STAB x15/10
    (:2592), then each ``chart_multipliers`` factor through ``ApplyTypeMultiplier``'s
    ``BattleSystem_Divide(value * mul, 10)`` (:7547); Struggle returns ``value``
    unchanged (:2573). The defender's Foresight, Ingrain / Iron Ball, roosting and Magnet Rise
    flags are read (``chart_multipliers``); Wonder Guard sets a flag only (``wonder_guard_bit``),
    the value is unchanged; the other type-chart modifiers (Adaptability, Normalize, Scrappy,
    Levitate, Miracle Eye) are not implemented: ``policy.ai_refusals`` refuses them.
    ``move_type`` is the caller's type for the move (``party_move_types``); by default the
    attacker's ``TrainerAI_MoveType`` (``damage.ai_move_type``)."""
    move = jnp.asarray(move, _I32)
    atk_side = jnp.asarray(atk_side, _I32)
    a_types = state.types[atk_side, state.active[atk_side]].astype(_I32)
    d_types = state.types[1 - atk_side, state.active[1 - atk_side]]
    if move_type is None:
        move_type = ai_move_type(tables, state, atk_side, move)  # TrainerAI_MoveType
    value = jnp.asarray(value, _I32)
    out = jnp.where(jnp.any(a_types == move_type), value * 15 // 10, value)
    identified, grounded, roosting, magnet = defender_chart_flags(tables, state, 1 - atk_side)
    for mul in chart_multipliers(
        tables, move, d_types, state.gravity, identified, move_type, grounded, roosting, magnet
    ):
        out = divide(out * mul, 10)
    return jnp.where(move == MOVE_STRUGGLE, value, out)


def defender_chart_flags(tables: Tables, state: BattleState, side) -> tuple:
    """``(identified, grounded, roosting, magnet)`` of ``side``'s active battler as
    ``BattleSystem_ApplyTypeChart`` reads them (``chart_multipliers``): Foresight
    (battle_lib.c:2615), ``MOVE_EFFECT_INGRAIN`` or an Iron Ball (``Battler_HeldItemEffect``,
    ruling G-56), ``turnFlags.roosting`` (``BasicTypeMulApplies``, :2534-2543) and Magnet Rise
    (``magnetRiseTurns`` without Ingrain or an Iron Ball, :2605-2606, G-66)."""
    grounded = (state.ingrain[side] != 0) | iron_ball(tables, state, side)
    magnet = (state.magnet_rise_turns[side] != 0) & ~grounded
    return state.foresight[side] != 0, grounded, state.roosting[side] != 0, magnet


def _chart_flags(
    tables: Tables,
    move,
    def_types,
    gravity,
    update,
    identified=False,
    move_type=None,
    grounded=False,
    roosting=False,
    magnet=False,
) -> jnp.ndarray:
    move = jnp.asarray(move, _I32)
    first, second = chart_multipliers(
        tables, move, def_types, gravity, identified, move_type, grounded, roosting, magnet
    )
    mask = _apply_mul(_apply_mul(jnp.zeros((), _I32), first, update), second, update)
    # MOVE_STATUS_MAGNET_RISE instead of the walk (:2605-2609): no other chart bit is set.
    if move_type is None:
        move_type = tables.move_type[move]
    ground = jnp.asarray(magnet, jnp.bool_) & (jnp.asarray(move_type, _I32) == _TYPE_GROUND)
    mask = jnp.where(ground, MAGNET_RISE, mask)
    return jnp.where(move == MOVE_STRUGGLE, 0, mask)


def type_chart_flags(
    tables: Tables,
    move,
    def_types,
    gravity,
    identified=False,
    move_type=None,
    grounded=False,
    roosting=False,
    magnet=False,
) -> jnp.ndarray:
    """``int32``: the ``MOVE_STATUS_*`` bits ``BattleSystem_ApplyTypeChart`` leaves for
    ``move`` against a defender of ``def_types`` (int8[2]), identified by Foresight or
    not, grounded by Ingrain or not, roosting or not, magnet-risen or not
    (``defender_chart_flags``; a Ground move against Magnet Rise leaves ``MAGNET_RISE`` alone);
    only a move with power updates the super/not-very-effective bits. ``move_type`` overrides the
    move's table type (``damage.ai_move_type``)."""
    power = tables.move_power[jnp.asarray(move, _I32)]
    return _chart_flags(
        tables,
        move,
        def_types,
        gravity,
        power > 0,
        identified,
        move_type,
        grounded,
        roosting,
        magnet,
    )


def wonder_guard_bit(
    tables: Tables, state: BattleState, atk_side, def_side, move, flags
) -> jnp.ndarray:
    """``bool``: ``BattleSystem_ApplyTypeChart`` sets ``MOVE_STATUS_WONDER_GUARD`` (battle_lib.c:
    2649-2654) for ``move`` by ``atk_side``'s active battler against ``def_side``'s, given the
    walk's ``flags`` (``type_chart_flags``): the defender's Wonder Guard (``Battler_Ability``:
    ``damage.decomp_abilities``, the true ability) unless the attacker has Mold Breaker
    (``Battler_IgnorableAbility``), a move on its damaging turn (``damage.on_damaging_turn``:
    the AI reads at command selection, after ``BattleContext_Init`` cleared
    ``SYSCTL_LAST_OF_MULTI_TURN``), with table power, not Struggle (:2573), and
    ``MOVE_STATUS_SUPER_EFFECTIVE`` clear. Part of ``MOVE_STATUS_IMMUNE`` (ruling G-23)."""
    atk_side = jnp.asarray(atk_side, _I32)
    def_side = jnp.asarray(def_side, _I32)
    move = jnp.asarray(move, _I32)
    ability = decomp_abilities(tables, state)
    attacker = ability[atk_side, state.active[atk_side]]
    defender = ability[def_side, state.active[def_side]]
    return (
        (attacker != _MOLD_BREAKER)
        & (defender == _WONDER_GUARD)
        & on_damaging_turn(tables, move)
        & ((jnp.asarray(flags, _I32) & SUPER_EFFECTIVE) == 0)
        & (tables.move_power[move] != 0)
        & (move != MOVE_STRUGGLE)
    )


def calc_effectiveness_flags(
    tables: Tables, move, def_types, gravity, move_type=None, iron=False
) -> jnp.ndarray:
    """``int32``: the ``MOVE_STATUS_*`` bits ``BattleSystem_CalcEffectiveness`` leaves
    for ``move`` against ``def_types`` (int8[2]), for any move, power or not.
    ``move_type`` overrides the move's table type (``damage.ai_move_type``). ``iron``: the
    caller's ``defenderItemEffect`` is ``HOLD_EFFECT_SPEED_DOWN_GROUNDED``, which lifts the
    Flying immunity like Gravity (``NoImmunityOverrides``, battle_lib.c:2759-2764; G-56)."""
    return _chart_flags(tables, move, def_types, gravity, True, move_type=move_type, grounded=iron)


class _Layout(NamedTuple):
    kinds: jnp.ndarray  # int8[sites]
    targets: jnp.ndarray  # int8[sites], party slot of a pass site, else -1
    count: jnp.ndarray  # int32, sites found
    only_ineffective: jnp.ndarray  # bool, AI_OnlyIneffectiveMoves applies
    ability_refused: jnp.ndarray  # bool[2, SLOTS]
    move_refused: jnp.ndarray  # bool[2, SLOTS, MOVES]
    refused: jnp.ndarray  # bool
    perish: jnp.ndarray  # bool, AI_PerishSongKO (step 3) switches the AI's active out


@partial(jax.jit, static_argnames="sites")
def _layout(tables: Tables, state: BattleState, sites: int) -> _Layout:
    a = state.active[1].astype(_I32)
    d = state.active[0].astype(_I32)
    gravity = state.gravity
    d_types = state.types[0, d]
    species = state.species.astype(_I32)
    ability = decomp_abilities(tables, state)  # the engine's current abilities
    d_ability = ability[0, d]
    present = species > 0

    # Step 1 (:3906-3913) and step 2 (:3915-3940). VOLATILE_CONDITION_TRAPPED is Bind's
    # ``trapped`` and Mean Look's ``mean_look`` (condition.h:77-78); :3907 MOVE_EFFECT_INGRAIN.
    steel = jnp.any(state.types[1, a] == _TYPE_STEEL)
    gate = (
        (state.trapped[1] > 0)
        | (state.mean_look[1] != 0)
        | (state.ingrain[1] != 0)
        | (d_ability == ABILITY_IDS["Shadow Tag"])
        | (d_ability == ABILITY_IDS["Arena Trap"])
        | ((d_ability == ABILITY_IDS["Magnet Pull"]) & steel)
    )
    bench = (jnp.arange(SLOTS) != a) & present[1] & (state.hp[1] > 0)
    gate = gate | ~jnp.any(bench)

    # Step 3, AI_PerishSongKO (trainer_ai.c:3263-3272): after the trap gate and the bench test,
    # before the Wonder Guard step; no draw. Everything after it (steps 4-11) is not reached.
    perish = ~gate & (state.perish_song[1] != 0) & (state.perish_song_turns[1] == 0)

    # Step 5 precondition and step 8: A's moves against D through ApplyTypeChart.
    a_moves = state.moves[1, a].astype(_I32)
    identified, grounded, roosting, magnet = defender_chart_flags(tables, state, 0)
    a_flags = jax.vmap(
        lambda m: type_chart_flags(
            tables,
            m,
            d_types,
            gravity,
            identified,
            ai_move_type(tables, state, 1, m),
            grounded,
            roosting,
            magnet,
        )
    )(a_moves)
    keep = (a_moves != 0) & ((a_flags & SUPER_EFFECTIVE) != 0)
    damaging = (a_moves != 0) & (tables.move_power[a_moves] > 0)
    only_ineffective = (
        ~gate
        & ~perish
        & (jnp.sum(damaging.astype(_I32)) >= 2)
        & jnp.all(~damaging | ((a_flags & INEFFECTIVE) != 0))
    )

    # Step 9 (:3872): positive stages summing to 4 or more end the check.
    boosted = jnp.sum(jnp.maximum(state.stages[1].astype(_I32), 0)) >= 4

    # Steps 10-11 (:3727): moveHit against each bench member, then its moves against D.
    hit = state.move_hit[1].astype(_I32)
    hit_ok = (hit != 0) & (tables.move_power[hit] > 0)
    # TrainerAI_MoveType (:3767): the type of moveHit as its user (the player's active) used it.
    hit_type = ai_move_type(tables, state, 0, hit)
    # The bench member's own party item hold effect (:3774, BattleSystem_GetItemData).
    bench_iron = tables.item_effect[state.item[1]] == ItemId.IRON_BALL
    hit_flags = jax.vmap(
        lambda t, i: calc_effectiveness_flags(tables, hit, t, gravity, hit_type, i)
    )(state.types[1], bench_iron)
    # CalcEffectiveness's Wonder Guard (battle_lib.c:2734-2739) with the bench Pokemon as the
    # defender (MON_DATA_ABILITY) and Battler_Ability(moveHitBattler), the player's active, as
    # the attacker: INEFFECTIVE unless Mold Breaker, a super-effective moveHit or one not on its
    # damaging turn (ruling G-23).
    bench_guard = (
        (ability[1] == _WONDER_GUARD)
        & (d_ability != _MOLD_BREAKER)
        & on_damaging_turn(tables, hit)
        & ((hit_flags & SUPER_EFFECTIVE) == 0)
    )
    hit_flags = jnp.where(bench_guard, hit_flags | INEFFECTIVE, hit_flags)
    bench_moves = state.moves[1].astype(_I32).reshape(-1)  # [SLOTS * MOVES], slot-major
    # Move_CalcVariableType (:3780): each bench member's own Hidden Power type.
    bench_types = party_move_types(tables, state).reshape(-1)
    # Battler_HeldItemEffect(moveHitBattler), the player's active (:3790).
    d_iron = iron_ball(tables, state, 0)
    bench_flags = jax.vmap(
        lambda m, k: calc_effectiveness_flags(tables, m, d_types, gravity, k, d_iron)
    )(bench_moves, bench_types).reshape(SLOTS, MOVES)
    bench_moves = bench_moves.reshape(SLOTS, MOVES)
    bench_se = (bench_moves != 0) & ((bench_flags & SUPER_EFFECTIVE) != 0)
    passes = [
        ((bench & hit_ok & ~boosted & ((hit_flags & mask) != 0))[:, None] & bench_se).reshape(-1)
        for mask in (INEFFECTIVE, NOT_VERY_EFFECTIVE)
    ]
    # Step 4, AI_CannotDamageWonderGuard (:3286-3348): D's battleMons[].ability (:3298) is
    # Wonder Guard and no move of A is super effective by ApplyTypeChart (:3300-3311, ``keep``'s
    # flags); then each bench Pokemon's super-effective moves by CalcEffectiveness, in party
    # order (``bench_se``), draw (:3338). Doubles return FALSE (:3294), so singles only.
    guard = (d_ability == _WONDER_GUARD) & ~jnp.any(keep)
    guard_draws = (guard & bench[:, None] & bench_se).reshape(-1)
    exists = jnp.concatenate([guard_draws, keep, *passes]) & ~gate & ~perish  # [_CANDIDATES]

    position = jnp.cumsum(exists.astype(_I32)) - 1
    target = jnp.where(exists, position, sites)  # out of range: dropped
    kinds = jnp.zeros(sites, _I8).at[target].set(_CANDIDATE_KINDS, mode="drop")
    targets = jnp.full(sites, -1, _I8).at[target].set(_CANDIDATE_TARGETS, mode="drop")
    count = jnp.sum(exists.astype(_I32))

    side = jnp.arange(2)[:, None]
    ability_refused = present & (
        jnp.isin(ability, _ANY_SIDE_IDS) | ((side == 1) & jnp.isin(ability, _AI_SIDE_IDS))
    )
    move_refused = present[..., None] & jnp.isin(state.moves.astype(_I32), _MOVE_REFUSED_IDS)
    refused = (count > sites) | only_ineffective | jnp.any(ability_refused) | jnp.any(move_refused)
    kinds = jnp.where(refused, jnp.asarray(SwitchSite.REFUSED, _I8), kinds)
    return _Layout(
        kinds,
        targets,
        count,
        only_ineffective,
        ability_refused,
        move_refused,
        refused,
        perish,
    )


def _refusals(lay: _Layout, state: BattleState, tables: Tables, sites: int) -> list[str]:
    out: list[str] = []
    ability = np.asarray(decomp_abilities(tables, state))
    for side, s in zip(*np.nonzero(np.asarray(lay.ability_refused)), strict=True):
        name = _ABILITY_NAMES[int(ability[side, s])]
        why = (_ANY_SIDE_ABILITIES | _AI_SIDE_ABILITIES)[name]
        out.append(f"switching: side {side} slot {s}: ability {name} ({why}) is not implemented")
    moves = np.asarray(state.moves)
    for side, s, k in zip(*np.nonzero(np.asarray(lay.move_refused)), strict=True):
        name = _MOVE_NAMES[int(moves[side, s, k])]
        out.append(
            f"switching: side {side} slot {s}: move {name} ({_MOVES[name]}) is not implemented"
        )
    if bool(lay.only_ineffective):
        out.append(
            "switching: AI_OnlyIneffectiveMoves (trainer_ai.c:3361) applies to the AI's "
            "active Pokemon and is not implemented"
        )
    if int(lay.count) > sites:
        out.append(
            f"switching: {int(lay.count)} voluntary-switch draws exceed the "
            f"{sites} sites AiOutcome.switch_stop encodes"
        )
    return out


def switching_refusals(tables: Tables, state: BattleState, sites: int = SWITCH_SITES) -> list[str]:
    """Every reason the voluntary-switch layout of this (concrete) state is refused under
    a layout of ``sites`` sites, one human-readable line each; empty when it is supported."""
    return _refusals(_layout(tables, state, sites), state, tables, sites)


def _checked_layout(tables: Tables, state: BattleState, sites: int) -> _Layout:
    lay = _layout(tables, state, sites)
    if not isinstance(lay.refused, jax.core.Tracer) and bool(lay.refused):
        raise UnsupportedSwitching("; ".join(_refusals(lay, state, tables, sites)))
    return lay


def switch_sites(tables: Tables, state: BattleState, sites: int = SWITCH_SITES) -> jnp.ndarray:
    """``int8[sites]``: the ``SwitchSite`` kind of each result-changing draw of
    ``TrainerAI_ShouldSwitch`` for the opponent's active Pokemon (the Wonder Guard draws, the
    keep draws, then the two passes), in the decomp's order
    (``SwitchSite.NONE`` for unused entries), under a layout of ``sites`` sites (the AI
    layout's ``switch_sites``; the default is the vanilla 9). Raises
    ``UnsupportedSwitching`` on a refused layout when not traced (module docstring)."""
    return _checked_layout(tables, state, sites).kinds


def switch_action(
    tables: Tables, state: BattleState, stop, sites: int = SWITCH_SITES
) -> jnp.ndarray:
    """``int32``: what ``AiOutcome.switch_stop = stop`` means for this state under a
    layout of ``sites`` sites: the switch action ``4 + j`` when a party-pass site for slot
    j stops (a Wonder Guard draw too: ``aiSwitchedPartySlot = i``, :3339), ``NO_SWITCH`` for a
    keep draw or for ``stop = sites`` (none stops),
    ``REFUSED_ACTION`` for a refused layout."""
    lay = _checked_layout(tables, state, sites)
    stop = jnp.asarray(stop, _I32)
    in_range = (stop >= 0) & (stop < sites)
    index = jnp.clip(stop, 0, sites - 1)
    kind = lay.kinds[index]
    is_pass = in_range & (
        (kind == SwitchSite.PASS_INEFFECTIVE)
        | (kind == SwitchSite.PASS_NOT_VERY_EFFECTIVE)
        | (kind == SwitchSite.WONDER_GUARD)
    )
    action = jnp.where(is_pass, MOVE_ACTIONS + lay.targets[index].astype(_I32), NO_SWITCH)
    # AI_PerishSongKO switches whatever the stop (no draw precedes it): TrainerAI_PickCommand
    # (trainer_ai.c:4001-4026) sends BattleAI_PostKOSwitchIn's pick, or the first living
    # non-active slot (:4014-4022; the pick is always one while a bench exists).
    from nuzline.ai.replacement import post_ko_pick  # circular at module level

    pick, stale = post_ko_pick(tables, state)
    action = jnp.where(lay.perish, jnp.where(stale, REFUSED_ACTION, MOVE_ACTIONS + pick), action)
    return jnp.where(lay.refused | (stop == SWITCH_REFUSED), REFUSED_ACTION, action).astype(_I32)
