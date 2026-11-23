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
Roark's three Pokemon, ``KAIZO_SWITCH_SITES`` (69) for a party of six. The candidate draws
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
beside a keep draw: at most ``max(4, 5 * 4) + 40 = 60`` candidates without the absorb draws
(ruling G-33). The absorb draws (step 6, at most 5, :3681-3708) exist with or without a keep
draw: beside the keep draws that is at most ``5 + 4 + 40 = 49``; with the Wonder Guard draws,
``20 + 5 + 40 = 65`` for a Wonder Guard player, an absorbable ``moveHit`` and bench absorb
holders together. ``KAIZO_SWITCH_SITES`` (69, ruling 19 Nov) is the sum of the four parts, each
at its own bound: the guard draws ``(SLOTS - 1) * MOVES`` (:3315-3344), the keep draws ``MOVES``
(:3576-3592), the absorb draws ``SLOTS - 1`` (:3681-3708) and the two passes
``2 * (SLOTS - 1) * MOVES`` (:3756-3804), so no state of a party of six has more.

Step 6, ``AI_HasAbsorbAbilityInParty`` (trainer_ai.c:3640-3713; ruling G-36). When ``A`` has a
super-effective move (``AI_HasSuperEffectiveMove(..., TRUE)``, no draw inside, :3586-3587) it
first reads ``RandNext() % 3`` and returns FALSE unless it is 0 (:3651-3653): one gate draw
shared by every bench draw below it. Then a ``moveHit`` with power (:3656-3663) of table type
Fire, Water or Electric (``MOVE_DATA().type``, :3665-3674) names Flash Fire, Water Absorb or Volt
Absorb; ``A``'s own ``Battler_Ability`` of that name holds the switch (:3677-3679); each living
bench Pokemon (party order; ``aiSlot2`` is ``A`` itself in singles, battle_system.c:400-402)
whose ``MON_DATA_ABILITY`` is it draws ``RandNext() & 1`` (:3703-3708), the first success
switching to it. Its sites are ``SwitchSite.ABSORB`` without the gate and
``SwitchSite.ABSORB_GATED`` behind it; the gate is no site of its own: ``outcome`` folds it into
the stop law (``switch_stop_probs``) and draws it apart in the sampler (G-37).

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
Magnet Pull on ``D`` with ``A`` Steel type (:3910-3911); the three abilities are read by
``CountAbility`` through ``Battler_Ability`` (battle_lib.c:2896-2903, :2933-2940, :3088-3106), so
a ``D`` under Gastro Acid (``ability_suppressed[0]``, not Multitype) gates nothing; no other living party member; and
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
Breaker hides Levitate (``Battler_IgnorableAbility``) and matters for the AI bench's Wonder Guard
(above).

Refused, never approximated (``switching_refusals`` names each one):

* abilities on either side that change these masks: Normalize (Levitate and Scrappy are read
  since H-6: ``chart_flags``, ``calc_effectiveness_flags``); and on
  the AI's side step 7's Natural Cure (``AI_IsAsleepWithNaturalCure``), whose draws are not
  modelled (step 6, every absorb case, is: above);
* moves whose state the engine does not carry: none now (Miracle Eye's bit is read since E3-B2-10,
  ``chart_flags``); the variable-type moves are
  modelled (``TrainerAI_MoveType``, :3127, ``damage.ai_move_type`` for the AI's active and the
  player's ``moveHit``; ``Move_CalcVariableType``, battle_lib.c:8097, ``party_move_types`` for
  a bench member: Hidden Power, Natural Gift, Judgment, Weather Ball, B13-35), except Weather
  Ball where its case assigns no type (``weather_ball_refused``);
* a state where ``AI_OnlyIneffectiveMoves`` (step 5, :3361) applies.

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
from nuzline.engine import field as fcm
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import (
    AI_JUDGMENT,
    AI_NATURAL_GIFT,
    AI_TYPELESS,
    AI_WEATHER_BALL,
    ai_move_type,
    ai_weather_ball_typed,
    battler_decomp_ability,
    decomp_abilities,
    divide,
    hidden_power,
    ignorable,
    iron_ball,
    judgment_type,
    levitate_off,
    levitated,
    on_damaging_turn,
    weather_ball_params,
)
from nuzline.engine.format import foe, format_of, party
from nuzline.engine.ids import AbilityId, ItemId, Type
from nuzline.engine.state import MOVES, SLOTS, BattleState

SWITCH_SITES = 9  # the vanilla layout's width (roark.md 11.1); the bound below is 8
# Its bound for every vanilla trainer the flat layout accepts (every party move in
# ``commands.MOVE_EFFECTS``: Roark, id 246, and four one-Pokemon parties with Tackle): a draw needs
# a move flagged SUPER_EFFECTIVE (``BattleSystem_CalcEffectiveness``, battle_lib.c:2681-2730, no
# power test, so Stealth Rock counts): Geodude and Onix hold two (Rock), Cranidos one (Pursuit).
# The keep draws (trainer_ai.c:3576-3592) are the active's, at most 2; the Wonder Guard draws
# (:3315-3344) only when the active has none (:3300-3311), at most 2 + 2 from the bench; no bench
# Pokemon can be in both passes (:3756-3804; that needs one attacking type immune on one of its
# types and resisted on the other, battle_lib.c:2782-2805: none for Rock / Ground, and Cranidos
# has one type) and none has an absorb ability (:3681-3708): passes at most 2 + 2. So at most
# max(2, 4) + 4 = 8 draws; not relied on: Roark's every Pokemon has a move super effective on a
# Bug / Ghost Shedinja, which drops the guard draws and leaves 5.
# ``tests/ai/test_switching.py::test_vanilla_switch_sites_bound_every_accepted_trainer``
# recomputes it from the trainer data.
SWITCH_NONE = SWITCH_SITES  # switch_stop when no draw stops the check (vanilla layout)
# Every candidate draw of a party of six (module docstring; rulings G-33 and 19 Nov), the four
# parts each at its bound: the Wonder Guard draws (5 bench Pokemon x 4 moves, trainer_ai.c:
# 3315-3344), the keep draws (4 moves, :3576-3592), the absorb draws (5 bench Pokemon,
# :3681-3708) and the two passes (5 x 4 each, :3756-3804): 20 + 4 + 5 + 40 = 69. A layout's
# switch_stop is ``sites`` when none stops.
_GUARD_SITES = (SLOTS - 1) * MOVES
_KEEP_SITES = MOVES
_ABSORB_SITES = SLOTS - 1
_PASS_SITES = 2 * (SLOTS - 1) * MOVES
KAIZO_SWITCH_SITES = _GUARD_SITES + _KEEP_SITES + _ABSORB_SITES + _PASS_SITES
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
    # AI_HasAbsorbAbilityInParty (:3640-3713), Flash Fire's case (F-2): a bench Pokemon with Flash
    # Fire after a Fire moveHit with power: RandNext() & 1 switches to it (:3706), 32768 of 65536.
    # Water Absorb and Volt Absorb after a Water / Electric moveHit likewise (:3668-3671; G-36).
    ABSORB = 6
    # The same draw behind the function's gate (:3651-3653): A has a super-effective move, so one
    # RandNext() % 3 shared by every ABSORB_GATED site of the layout lets them draw only on residue
    # 0 (21846 of 65536). ``outcome`` folds the gate into the stop law (G-37).
    ABSORB_GATED = 7


# MOVE_STATUS_* bits (include/constants/battle/moves.h:58-60).
SUPER_EFFECTIVE, NOT_VERY_EFFECTIVE, INEFFECTIVE = 1 << 1, 1 << 2, 1 << 3
# MOVE_STATUS_MAGNET_RISE (moves.h:77): part of MOVE_STATUS_IMMUNE (:87); ApplyTypeChart's walk is
# skipped for it, so no other bit of the chart is set (G-66).
MAGNET_RISE = 1 << 20
# MOVE_STATUS_LEVITATED (moves.h:68): ApplyTypeChart's Levitate immunity, also part of
# MOVE_STATUS_IMMUNE (:87) and no part of INEFFECTIVE; the walk is skipped for it as for Magnet
# Rise (battle_lib.c:2600-2603, H-6).
LEVITATED = 1 << 11
# The chart bits of MOVE_STATUS_IMMUNE (moves.h:87) less Wonder Guard's (``wonder_guard_bit``).
IMMUNE_CHART = INEFFECTIVE | LEVITATED | MAGNET_RISE

# Decomp ids, checked against the manifest by tests/ai/test_switching.py.
MOVE_STRUGGLE = 165
_TYPE_FLYING, _TYPE_STEEL, _TYPE_GROUND, _TYPE_DARK = 2, 8, 4, 17
ABILITY_IDS = {
    "Levitate": 26,
    "Wonder Guard": 25,
    "Scrappy": 113,
    "Filter": 111,
    "Solid Rock": 116,
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
    "Ingrain": 275,
    "Perish Song": 195,
    "Hidden Power": 237,
    "Weather Ball": 311,
    "Natural Gift": 363,
    "Judgment": 449,
}
# Refused abilities: name -> what the decomp does with it that is not implemented.
_ANY_SIDE_ABILITIES = {
    "Normalize": "makes every move Normal",
}
_AI_SIDE_ABILITIES = {
    "Natural Cure": "AI_IsAsleepWithNaturalCure",
}
_MOVES: dict[str, str] = {}  # none refused now: Miracle Eye is modelled (E3-B2-10)
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
_CANDIDATES = MOVES + SLOTS + 3 * SLOTS * MOVES
_CANDIDATE_KINDS = jnp.array(
    [SwitchSite.WONDER_GUARD] * (SLOTS * MOVES)
    + [SwitchSite.ABSORB] * SLOTS
    + [SwitchSite.KEEP] * MOVES
    + [SwitchSite.PASS_INEFFECTIVE] * (SLOTS * MOVES)
    + [SwitchSite.PASS_NOT_VERY_EFFECTIVE] * (SLOTS * MOVES),
    _I8,
)
_BENCH_TARGETS = [j for j in range(SLOTS) for _ in range(MOVES)]
_CANDIDATE_TARGETS = jnp.array(
    _BENCH_TARGETS + list(range(SLOTS)) + [-1] * MOVES + _BENCH_TARGETS * 2, _I8
)
_WONDER_GUARD, _MOLD_BREAKER = ABILITY_IDS["Wonder Guard"], ABILITY_IDS["Mold Breaker"]
# AI_HasAbsorbAbilityInParty's checkAbility by moveHit's table type (trainer_ai.c:3665-3674).
_ABSORB_TYPES = jnp.array([Type.FIRE, Type.WATER, Type.ELECTRIC], _I32)
_ABSORB_ABILITIES = jnp.array(
    [ABILITY_IDS[n] for n in ("Flash Fire", "Water Absorb", "Volt Absorb")], _I32
)
_ABSORB_FIRST = SLOTS * MOVES  # the absorb candidates' first index in _CANDIDATE_KINDS


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
    eyed=False,
    levitated=False,
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
    ``chart_flags``): a Ground move skips the walk (:2605-2609): both factors are 10,
    the value keeps only STAB. ``eyed`` (the defender's ``MOVE_EFFECT_MIRACLE_EYE``) skips the
    immune Dark rows (``BasicTypeMulApplies``, :2551-2555, E3-B2-10): ``ApplyTypeChart`` only,
    like ``grounded``; ``CalcEffectiveness`` reads neither. ``levitated`` (the defender's Levitate
    as the attacker sees it, ``chart_flags``): a Ground move skips the walk like Magnet Rise's
    (:2600-2603, H-6)."""
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
        dark = jnp.asarray(eyed, jnp.bool_) & (t == _TYPE_DARK) & (mul == 0)
        return jnp.where(lifted | skipped | dark, 10, mul)

    ground = (jnp.asarray(magnet, jnp.bool_) | jnp.asarray(levitated, jnp.bool_)) & (
        move_type == _TYPE_GROUND
    )
    m1 = jnp.where(ground, 10, entry(t1))
    m2 = jnp.where((t2 != t1) & ~ground, entry(t2), 10)
    t2_first = tables.type_chart_order[move_type, t2] < tables.type_chart_order[move_type, t1]
    return jnp.where(t2_first, m2, m1), jnp.where(t2_first, m1, m2)


def party_move_types(tables: Tables, state: BattleState, side: int = 1) -> jnp.ndarray:
    """``int32[SLOTS, MOVES]``: ``Move_CalcVariableType`` (battle_lib.c:8097-8203) of each party
    member's moves on ``side``, resolved as ``BattleSystem_CalcEffectiveness`` (:2690-2696) and
    ``ApplyTypeChart`` read it (``TYPE_NORMAL``, 0, is the move's table type): Hidden Power its
    own IV type (:8162-8175); Natural Gift the ``naturalGiftType`` of the member's party item,
    ``MON_DATA_HELD_ITEM`` (:8102-8104: no Klutz or Embargo gate, whatever its power; a
    non-berry's 0x1F is no type id, ``AI_TYPELESS``); Judgment the ``HOLD_EFFECT_ARCEUS_*`` type
    of that item, else ``TYPE_NORMAL`` (:8106-8160, ``damage.judgment_type``); Weather Ball the
    weather's type under ``NO_CLOUD_NINE`` and rain, sand, sun or hail (:8177-8195,
    ``damage.weather_ball_params``). Elsewhere Weather Ball's case assigns nothing and the
    function returns its uninitialized ``type``: refused (``weather_ball_refused``), the table
    type stands in. Any other move: its table type (B13-35)."""
    row = party(format_of(state), side)
    moves = state.moves[row].astype(_I32)
    table = tables.move_type[moves].astype(_I32)
    iv_types = jax.vmap(lambda ivs: hidden_power(ivs)[0])(state.ivs[row])  # [SLOTS]
    items = state.item[row].astype(_I32)[:, None]  # the party's own items
    gift = tables.item_gift_type[items].astype(_I32)
    gift = jnp.where(gift == 0, table, gift)
    gift = jnp.where((gift < 0) | (gift > _TYPE_DARK), AI_TYPELESS, gift)
    plate = judgment_type(tables, items, table)
    ball = weather_ball_params(tables, state, AI_WEATHER_BALL)[2]
    out = jnp.where(moves == AI_WEATHER_BALL, ball, table)
    out = jnp.where(moves == AI_JUDGMENT, plate, out)
    out = jnp.where(moves == AI_NATURAL_GIFT, gift, out)
    return jnp.where(moves == MOVE_IDS["Hidden Power"], iv_types[:, None], out)


def weather_ball_refused(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool``: a type read of the voluntary-switch rule or the post-KO replacement meets
    Weather Ball where its case assigns no type (``damage.ai_weather_ball_typed``: no weather,
    fog alone, or Cloud Nine / Air Lock): ``TrainerAI_MoveType`` (trainer_ai.c:3222-3240) of
    the AI's active's moves (:3302, :3387, :3578, :3605) or of the player's ``moveHit`` (:3767),
    and ``Move_CalcVariableType`` (battle_lib.c:8177-8195) of an AI party member's (:3324,
    :3443, :3502, :3782; battle_lib.c:7998, :8051) return an uninitialized register, not
    knowable (B13-26, B13-35). Any present AI party member with Weather Ball, or a
    Weather Ball ``moveHit``, in such a state: refused."""
    ball = AI_WEATHER_BALL
    present = state.species[1] > 0
    held = jnp.any(present[:, None] & (state.moves[1].astype(_I32) == ball))
    hit = state.move_hit[1].astype(_I32) == ball
    return (held | hit) & ~ai_weather_ball_typed(tables, state)


def apply_type_chart(
    tables: Tables, state: BattleState, atk_side, move, value, move_type=None, def_side=None
) -> jnp.ndarray:
    """``int32``: ``BattleSystem_ApplyTypeChart`` (battle_lib.c:2560) on ``value`` for
    ``move`` by ``atk_side``'s active Pokemon against the other side's: STAB x15/10
    (:2592), then each ``chart_multipliers`` factor through ``ApplyTypeMultiplier``'s
    ``BattleSystem_Divide(value * mul, 10)`` (:7547); Struggle returns ``value``
    unchanged (:2573). The defender's Foresight, Ingrain / Iron Ball, roosting, Magnet Rise and
    Levitate flags and the attacker's Scrappy are read (``chart_flags``); Wonder Guard sets a flag
    only (``wonder_guard_bit``), the value is unchanged; a super-effective move with power
    against Filter or Solid Rock (``Battler_IgnorableAbility``, ``damage.ignorable``) is
    ``BattleSystem_Divide(value * 3, 4)`` (:2657-2660, H-6); the other type-chart modifiers
    (Adaptability, Normalize, Tinted Lens, Expert Belt) are not implemented:
    ``policy.ai_refusals`` refuses them.
    ``move_type`` is the caller's type for the move (``party_move_types``); by default the
    attacker's ``TrainerAI_MoveType`` (``damage.ai_move_type``). ``def_side``: the defending
    battler, default the one across (``foe``)."""
    move = jnp.asarray(move, _I32)
    atk_side = jnp.asarray(atk_side, _I32)
    fmt = format_of(state)
    def_side = foe(atk_side) if def_side is None else jnp.asarray(def_side, _I32)
    a_types = state.types[party(fmt, atk_side), state.active[atk_side]].astype(_I32)
    d_types = state.types[party(fmt, def_side), state.active[def_side]]
    if move_type is None:
        move_type = ai_move_type(tables, state, atk_side, move)  # TrainerAI_MoveType
    value = jnp.asarray(value, _I32)
    out = jnp.where(jnp.any(a_types == move_type), value * 15 // 10, value)
    cf = chart_flags(tables, state, atk_side, def_side)
    gravity = fcm.gravity_count(state)
    for mul in chart_multipliers(
        tables, move, d_types, gravity, move_type=move_type, **cf._asdict()
    ):
        out = divide(out * mul, 10)
    # MOVE_STATUS_SUPER_EFFECTIVE && movePower (:2657; ``type_chart_flags`` gates the bit by power)
    # and Battler_IgnorableAbility(Filter) or (Solid Rock) (:2658-2660); no Wonder Guard bit then.
    flags = type_chart_flags(tables, move, d_types, gravity, move_type=move_type, **cf._asdict())
    seen = ignorable(state, atk_side, def_side)
    filtered = (flags & SUPER_EFFECTIVE) != 0
    filtered = filtered & ((seen == AbilityId.FILTER) | (seen == AbilityId.SOLID_ROCK))
    out = jnp.where(filtered, divide(out * 3, 4), out)
    return jnp.where(move == MOVE_STRUGGLE, value, out)


class ChartFlags(NamedTuple):
    """The defender's state ``BattleSystem_ApplyTypeChart`` reads (``chart_flags``), by the
    keyword names of ``chart_multipliers`` and ``type_chart_flags``."""

    identified: jnp.ndarray  # Foresight, or the attacker's Scrappy
    grounded: jnp.ndarray  # Ingrain or an Iron Ball
    roosting: jnp.ndarray
    magnet: jnp.ndarray  # Magnet Rise turns without Ingrain or an Iron Ball
    eyed: jnp.ndarray  # Miracle Eye
    levitated: jnp.ndarray  # Levitate as the attacker sees it, without an Iron Ball


def chart_flags(tables: Tables, state: BattleState, atk_side, def_side) -> ChartFlags:
    """``BattleSystem_ApplyTypeChart``'s reads of ``def_side``'s active battler as ``atk_side``'s
    active battler attacks it (``chart_multipliers``): Foresight (battle_lib.c:2615) or the
    attacker's Scrappy (``Battler_Ability``, :2615-2616: Gastric Juice makes it none, no Mold
    Breaker gate), ``MOVE_EFFECT_INGRAIN`` or an Iron Ball (``Battler_HeldItemEffect``, ruling
    G-56), ``turnFlags.roosting`` (``BasicTypeMulApplies``, :2534-2543), Magnet Rise
    (``magnetRiseTurns`` without Ingrain or an Iron Ball, :2605-2606, G-66),
    ``MOVE_EFFECT_MIRACLE_EYE`` (:2551-2555, E3-B2-10) and Levitate through
    ``Battler_IgnorableAbility`` (:2600-2603, H-6; ``damage.levitated``, which reads Gravity,
    Ingrain, Gastric Juice and Mold Breaker as the decomp's ``Battler_Ability`` does)."""
    grounded = (state.ingrain[def_side] != 0) | iron_ball(tables, state, def_side)
    magnet = (state.magnet_rise_turns[def_side] != 0) & ~grounded
    eyed = state.miracle_eye[def_side] != 0
    scrappy = battler_decomp_ability(tables, state, atk_side) == ABILITY_IDS["Scrappy"]
    return ChartFlags(
        (state.foresight[def_side] != 0) | scrappy,
        grounded,
        state.roosting[def_side] != 0,
        magnet,
        eyed,
        levitated(tables, state, atk_side, def_side, Type.GROUND),
    )


def seen_ability(tables: Tables, state: BattleState, battler) -> jnp.ndarray:
    """``int32``: ``Battler_Ability`` (battle_lib.c:3088-3104) of ``battler`` in decomp ids:
    ``battler_decomp_ability`` (Gastric Juice), and Levitate reads as none under Gravity or the
    holder's Ingrain (:3095-3103, ``damage.levitate_off``, the engine-id twin's rule)."""
    ability = battler_decomp_ability(tables, state, battler)
    off = levitate_off(state, battler) & (ability == ABILITY_IDS["Levitate"])
    return jnp.where(off, 0, ability)


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
    eyed=False,
    levitated=False,
) -> jnp.ndarray:
    move = jnp.asarray(move, _I32)
    first, second = chart_multipliers(
        tables,
        move,
        def_types,
        gravity,
        identified,
        move_type,
        grounded,
        roosting,
        magnet,
        eyed,
        levitated,
    )
    mask = _apply_mul(_apply_mul(jnp.zeros((), _I32), first, update), second, update)
    # MOVE_STATUS_LEVITATED (:2600-2603), else MOVE_STATUS_MAGNET_RISE (:2605-2609), instead of
    # the walk: no other chart bit is set.
    if move_type is None:
        move_type = tables.move_type[move]
    ground = jnp.asarray(move_type, _I32) == _TYPE_GROUND
    mask = jnp.where(jnp.asarray(magnet, jnp.bool_) & ground, MAGNET_RISE, mask)
    mask = jnp.where(jnp.asarray(levitated, jnp.bool_) & ground, LEVITATED, mask)
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
    eyed=False,
    levitated=False,
) -> jnp.ndarray:
    """``int32``: the ``MOVE_STATUS_*`` bits ``BattleSystem_ApplyTypeChart`` leaves for
    ``move`` against a defender of ``def_types`` (int8[2]), identified by Foresight or
    not, grounded by Ingrain or not, roosting or not, magnet-risen or not, under Miracle Eye or
    not, levitating or not (``chart_flags``; a Ground move against Magnet Rise leaves
    ``MAGNET_RISE`` alone, against Levitate ``LEVITATED``);
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
        eyed,
        levitated,
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
    # Battler_IgnorableAbility (battle_lib.c:2649): Battler_Ability of both.
    attacker = battler_decomp_ability(tables, state, atk_side)
    defender = battler_decomp_ability(tables, state, def_side)
    return (
        (attacker != _MOLD_BREAKER)
        & (defender == _WONDER_GUARD)
        & on_damaging_turn(tables, move)
        & ((jnp.asarray(flags, _I32) & SUPER_EFFECTIVE) == 0)
        & (tables.move_power[move] != 0)
        & (move != MOVE_STRUGGLE)
    )


def calc_effectiveness_flags(
    tables: Tables,
    move,
    def_types,
    gravity,
    move_type=None,
    iron=False,
    atk_ability=0,
    def_ability=0,
) -> jnp.ndarray:
    """``int32``: the ``MOVE_STATUS_*`` bits ``BattleSystem_CalcEffectiveness`` leaves
    for ``move`` against ``def_types`` (int8[2]), for any move, power or not.
    ``move_type`` overrides the move's table type (``damage.ai_move_type``). ``iron``: the
    caller's ``defenderItemEffect`` is ``HOLD_EFFECT_SPEED_DOWN_GROUNDED``, which lifts the
    Flying immunity like Gravity (``NoImmunityOverrides``, battle_lib.c:2759-2764; G-56).
    ``atk_ability`` and ``def_ability`` are the caller's ability arguments (decomp ids): the
    attacker's Scrappy breaks the walk at the Ghost immunities (:2708-2714, no Foresight test);
    Levitate on the defender, with no Mold Breaker attacker, a Ground move, no Gravity and no
    Iron Ball, leaves ``INEFFECTIVE`` alone (:2698-2703, H-6)."""
    atk_ability = jnp.asarray(atk_ability, _I32)
    flags = _chart_flags(
        tables,
        move,
        def_types,
        gravity,
        True,
        identified=atk_ability == ABILITY_IDS["Scrappy"],
        move_type=move_type,
        grounded=iron,
    )
    if move_type is None:
        move_type = tables.move_type[jnp.asarray(move, _I32)]
    floats = (
        (atk_ability != _MOLD_BREAKER)
        & (jnp.asarray(def_ability, _I32) == ABILITY_IDS["Levitate"])
        & (jnp.asarray(move_type, _I32) == _TYPE_GROUND)
        & (jnp.asarray(gravity) == 0)
        & ~jnp.asarray(iron, jnp.bool_)
    )
    return jnp.where(floats & (jnp.asarray(move, _I32) != MOVE_STRUGGLE), INEFFECTIVE, flags)


class _Layout(NamedTuple):
    kinds: jnp.ndarray  # int8[sites]
    targets: jnp.ndarray  # int8[sites], party slot of a pass site, else -1
    count: jnp.ndarray  # int32, sites found
    only_ineffective: jnp.ndarray  # bool, AI_OnlyIneffectiveMoves applies
    ability_refused: jnp.ndarray  # bool[2, SLOTS]
    move_refused: jnp.ndarray  # bool[2, SLOTS, MOVES]
    weather_ball: jnp.ndarray  # bool, weather_ball_refused
    refused: jnp.ndarray  # bool
    perish: jnp.ndarray  # bool, AI_PerishSongKO (step 3) switches the AI's active out


@partial(jax.jit, static_argnames="sites")
def _layout(tables: Tables, state: BattleState, sites: int) -> _Layout:
    a = state.active[1].astype(_I32)
    d = state.active[0].astype(_I32)
    gravity = fcm.gravity_count(state)
    d_types = state.types[0, d]
    species = state.species.astype(_I32)
    # The engine's current abilities, raw: the refusal scan below is over every party slot, and
    # AI_CannotDamageWonderGuard reads battleMons[].ability (trainer_ai.c:3298).
    ability = decomp_abilities(tables, state)
    d_ability = ability[0, d]
    # The trap gate reads ``CountAbility`` -> ``Battler_Ability`` (battle_lib.c:2896-2903,
    # :2933-2940; :3088-3106), not the raw field: Gastro Acid (ability suppressed) on the
    # player's active empties it, except Multitype (:3090-3093), so no trap ability gates.
    d_trap = battler_decomp_ability(tables, state, 0)
    present = species > 0

    # Step 1 (:3906-3913) and step 2 (:3915-3940). VOLATILE_CONDITION_TRAPPED is Bind's
    # ``trapped`` and Mean Look's ``mean_look`` (condition.h:77-78); :3907 MOVE_EFFECT_INGRAIN.
    steel = jnp.any(state.types[1, a] == _TYPE_STEEL)
    gate = (
        (state.trapped[1] > 0)
        | (state.mean_look[1] != 0)
        | (state.ingrain[1] != 0)
        | (d_trap == ABILITY_IDS["Shadow Tag"])
        | (d_trap == ABILITY_IDS["Arena Trap"])
        | ((d_trap == ABILITY_IDS["Magnet Pull"]) & steel)
    )
    bench = (jnp.arange(SLOTS) != a) & present[1] & (state.hp[1] > 0)
    gate = gate | ~jnp.any(bench)

    # Step 3, AI_PerishSongKO (trainer_ai.c:3263-3272): after the trap gate and the bench test,
    # before the Wonder Guard step; no draw. Everything after it (steps 4-11) is not reached.
    perish = ~gate & (state.perish_song[1] != 0) & (state.perish_song_turns[1] == 0)

    # Step 5 precondition and step 8: A's moves against D through ApplyTypeChart.
    a_moves = state.moves[1, a].astype(_I32)
    cf = chart_flags(tables, state, 1, 0)
    a_flags = jax.vmap(
        lambda m: type_chart_flags(
            tables, m, d_types, gravity, move_type=ai_move_type(tables, state, 1, m), **cf._asdict()
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
    # ai.c:3769-3777: attacker Battler_Ability(moveHitBattler), defender the bench member's raw
    # MON_DATA_ABILITY (Levitate, H-6).
    d_seen = seen_ability(tables, state, 0)
    hit_flags = jax.vmap(
        lambda t, i, ab: calc_effectiveness_flags(tables, hit, t, gravity, hit_type, i, d_seen, ab)
    )(state.types[1], bench_iron, ability[1])
    # CalcEffectiveness's Wonder Guard (battle_lib.c:2734-2739) with the bench Pokemon as the
    # defender (MON_DATA_ABILITY) and Battler_Ability(moveHitBattler), the player's active, as
    # the attacker: INEFFECTIVE unless Mold Breaker, a super-effective moveHit or one not on its
    # damaging turn (ruling G-23).
    bench_guard = (
        (ability[1] == _WONDER_GUARD)
        & (d_seen != _MOLD_BREAKER)
        & on_damaging_turn(tables, hit)
        & ((hit_flags & SUPER_EFFECTIVE) == 0)
    )
    hit_flags = jnp.where(bench_guard, hit_flags | INEFFECTIVE, hit_flags)
    bench_moves = state.moves[1].astype(_I32).reshape(-1)  # [SLOTS * MOVES], slot-major
    # Move_CalcVariableType (:3780): each bench member's own Hidden Power type.
    bench_types = party_move_types(tables, state).reshape(-1)
    # Battler_HeldItemEffect(moveHitBattler), the player's active (:3790).
    d_iron = iron_ball(tables, state, 0)
    # :3786-3794: attacker the member's raw ability (Scrappy, Mold Breaker), defender
    # Battler_Ability(moveHitBattler) (Levitate: none under Ingrain, Gravity, Gastric Juice).
    bench_abilities = jnp.repeat(ability[1], MOVES)
    bench_flags = jax.vmap(
        lambda m, k, ab: calc_effectiveness_flags(
            tables, m, d_types, gravity, k, d_iron, ab, d_seen
        )
    )(bench_moves, bench_types, bench_abilities).reshape(SLOTS, MOVES)
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
    # Step 6, AI_HasAbsorbAbilityInParty (:3640-3713; F-2, G-36). With a super-effective move of
    # A (flag TRUE, no draw inside, :3586-3587; ``keep``'s flags) it reads RandNext() % 3 first
    # and returns FALSE unless it is 0 (:3651-3653): the sites below are ABSORB_GATED then
    # (G-37). A moveHit with power (:3656-3663) of table type Fire / Water / Electric
    # (``MOVE_DATA().type``, not TrainerAI_MoveType: :3665-3674) names checkAbility, else none
    # (:3672-3674); A's own Battler_Ability equal to it holds the switch (:3677-3679); each living
    # bench Pokemon (the same bench as the passes: selectedPartySlot and aiSwitchedPartySlot of
    # aiSlot1 = aiSlot2 = A in singles, :3681-3686, battle_system.c:400-402; the latter is 6)
    # whose MON_DATA_ABILITY (:3703, party data) is checkAbility draws RandNext() & 1 in party
    # order, the first success switching to it (:3706-3708).
    hit_table_type = tables.move_type[hit].astype(_I32)
    check = jnp.sum(jnp.where(_ABSORB_TYPES == hit_table_type, _ABSORB_ABILITIES, 0))
    absorb_bench = (
        bench
        & hit_ok
        & (check != 0)
        & (battler_decomp_ability(tables, state, 1) != check)
        & (ability[1] == check)
    )
    exists = (
        jnp.concatenate([guard_draws, absorb_bench, keep, *passes]) & ~gate & ~perish
    )  # [_CANDIDATES]
    candidate = jnp.arange(_CANDIDATES)
    is_absorb = (candidate >= _ABSORB_FIRST) & (candidate < _ABSORB_FIRST + SLOTS)
    candidate_kinds = jnp.where(
        is_absorb & jnp.any(keep), jnp.asarray(SwitchSite.ABSORB_GATED, _I8), _CANDIDATE_KINDS
    )

    position = jnp.cumsum(exists.astype(_I32)) - 1
    target = jnp.where(exists, position, sites)  # out of range: dropped
    kinds = jnp.zeros(sites, _I8).at[target].set(candidate_kinds, mode="drop")
    targets = jnp.full(sites, -1, _I8).at[target].set(_CANDIDATE_TARGETS, mode="drop")
    count = jnp.sum(exists.astype(_I32))

    side = jnp.arange(2)[:, None]
    ability_refused = present & (
        jnp.isin(ability, _ANY_SIDE_IDS) | ((side == 1) & jnp.isin(ability, _AI_SIDE_IDS))
    )
    move_refused = present[..., None] & jnp.isin(state.moves.astype(_I32), _MOVE_REFUSED_IDS)
    weather_ball = weather_ball_refused(tables, state)
    refused = only_ineffective | jnp.any(ability_refused) | jnp.any(move_refused)
    refused = refused | weather_ball
    kinds = jnp.where(refused, jnp.asarray(SwitchSite.REFUSED, _I8), kinds)
    return _Layout(
        kinds,
        targets,
        count,
        only_ineffective,
        ability_refused,
        move_refused,
        weather_ball,
        refused,
        perish,
    )


def _refusals(lay: _Layout, state: BattleState, tables: Tables) -> list[str]:
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
    if bool(lay.weather_ball):
        out.append(
            "switching: Weather Ball's type case assigns nothing in this weather "
            "(TrainerAI_MoveType, trainer_ai.c:3222-3240; Move_CalcVariableType, "
            "battle_lib.c:8177-8195): an uninitialized register"
        )
    if bool(lay.only_ineffective):
        out.append(
            "switching: AI_OnlyIneffectiveMoves (trainer_ai.c:3361) applies to the AI's "
            "active Pokemon and is not implemented"
        )
    return out


def switching_refusals(tables: Tables, state: BattleState, sites: int = SWITCH_SITES) -> list[str]:
    """Every reason the voluntary-switch layout of this (concrete) state is refused under
    a layout of ``sites`` sites, one human-readable line each; empty when it is supported."""
    return _refusals(_layout(tables, state, sites), state, tables)


def _checked_layout(tables: Tables, state: BattleState, sites: int) -> _Layout:
    lay = _layout(tables, state, sites)
    if not isinstance(lay.refused, jax.core.Tracer) and bool(lay.refused):
        raise UnsupportedSwitching("; ".join(_refusals(lay, state, tables)))
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
        | (kind == SwitchSite.ABSORB)
        | (kind == SwitchSite.ABSORB_GATED)
    )
    action = jnp.where(is_pass, MOVE_ACTIONS + lay.targets[index].astype(_I32), NO_SWITCH)
    # AI_PerishSongKO switches whatever the stop (no draw precedes it): TrainerAI_PickCommand
    # (trainer_ai.c:4001-4026) sends BattleAI_PostKOSwitchIn's pick, or the first living
    # non-active slot (:4014-4022; the pick is always one while a bench exists).
    from nuzline.ai.replacement import post_ko_pick  # circular at module level

    pick, stale = post_ko_pick(tables, state)
    action = jnp.where(lay.perish, jnp.where(stale, REFUSED_ACTION, MOVE_ACTIONS + pick), action)
    return jnp.where(lay.refused | (stop == SWITCH_REFUSED), REFUSED_ACTION, action).astype(_I32)
