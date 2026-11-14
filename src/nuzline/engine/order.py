"""Turn order and the move each action selects (pure ``jnp``, jit/vmap-able).

Actions are ``int32`` scalars: ``0-3`` use that move slot, ``4-9`` switch to party
slot ``action - 4``, and the opponent's ``10-13`` use bag slot ``action - 10``
(``actions``). ``action_order`` follows ``BattleControllerPlayer_CalcTurnOrder``
(``src/battle/battle_controller_player.c:707``): items and switches first, in battler
order (player first), then moves; two actions of the same kind are swapped when
``BattleSystem_CompareBattlerSpeed`` (``src/battle/battle_lib.c:1188``) says the
player's side does not go first (priority only between two moves, ``:1401``). An item
and a switch are different kinds, so the player's switch precedes the opponent's item.

Speed modifiers of the implemented set only: stages (Simple doubled, ``:1222``),
Swift Swim ``x 2`` in rain unless Cloud Nine / Air Lock (``:1228-1238``), Choice Scarf
``* 15 / 10`` (``:1248``, ``:1314``), paralysis ``/ 4`` (``:1260``), Tailwind ``x 2``
(``:1274``) and Trick Room's reversal (``:1432``). Quick Feet, Chlorophyll, Slow Start,
Unburden, Quick Powder and the speed-halving items belong to unsupported abilities and
items (``UNSUPPORTED`` ids), so they never apply.

Within equal priority (``player_second``, ``:1401-1440``) the decomp decides in this
order: Quick Claw / Custap Berry (``HOLD_EFFECT_SOMETIMES_PRIORITY`` /
``PINCH_PRIORITY``, one flag: ``custap_active``), Lagging Tail (``HOLD_EFFECT_PRIORITY_DOWN``),
Stall, then Trick Room or speed. Stall is unsupported. ``SortMonSpeedOrder`` (``end_of_turn_order``)
uses the same comparison with ``ignoreQuickClaw = TRUE`` (priorities 0; the Quick Claw
flag is still evaluated, only its ``moveEffectsData`` write is skipped, :1278-1300).

Quick Claw activates when ``speedRand[battler] % (100 / param) == 0`` (:1278-1286,
:1344-1352). ``speedRand`` is redrawn in the turn's pre-move step
(battle_controller_player.c:853-857), *after* ``CalcTurnOrder`` (:707-790), so a turn's
order reads the draw of the turn before (or of the battle start, battle_lib.c:2024):
``action_order`` reads ``BattleState.qc_roll``. The same turn's draw
(``Outcome.quick_claw``) orders that turn's end (``SortMonSpeedOrder`` at :888). The flag is
computed for every pair ``CompareBattlerSpeed`` compares, so it also orders two switches
(the outgoing Pokemon's item: the battler on the field) and two bag items.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION, MOVE_ACTIONS, locked_slot, usable_moves
from nuzline.engine.damage import apply_stage, simple_stage, weather_suppressed
from nuzline.engine.ids import (
    MOVE_STRUGGLE,
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    AbilityId,
    EffectId,
    ItemId,
    Stat,
    Status,
    Weather,
)
from nuzline.engine.state import BattleState

_I32 = jnp.int32


def is_switch(action) -> jnp.ndarray:
    action = jnp.asarray(action)
    return (action >= MOVE_ACTIONS) & (action < FIRST_ITEM_ACTION)


def is_item(action) -> jnp.ndarray:
    """An opponent item action (``actions.FIRST_ITEM_ACTION`` and up)."""
    return jnp.asarray(action) >= FIRST_ITEM_ACTION


def selected_move(
    tables: Tables, state: BattleState, side, action
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(move id, struggling)`` for a move action by ``side``'s active Pokemon in the
    state it was chosen in: the slot's move, or Struggle when no slot is usable
    (``actions.usable_moves``: no move, no PP, disabled, or power 0 while taunted), the decomp's
    ``STRUGGLING_ALL`` (battle_controller_player.c:433). The legality mask then only
    allows action 0. A locked battler picks no command and uses ``moveLockedInto``
    (``FightCommand``, battle_controller_player.c:1885-1888), never Struggling. Meaningless
    for a switch or item action (callers mask it)."""
    slot = jnp.clip(jnp.asarray(action, _I32), 0, MOVE_ACTIONS - 1)
    mon = state.active[side]
    struggling = ~jnp.any(usable_moves(tables, state, side))
    move = jnp.where(struggling, MOVE_STRUGGLE, state.moves[side, mon, slot].astype(_I32))
    locked = locked_slot(state, side) >= 0
    return jnp.where(locked, state.locked_move[side].astype(_I32), move), struggling


def terrain_index(state: BattleState) -> jnp.ndarray:
    """``int32``: the row of the terrain tables (``ids.TERRAIN_MOVES``, ``TERRAIN_TYPES``) the
    battle reads: ``min(terrain, TERRAIN_SPECIAL)`` (``BtlCmd_GetTerrainMove``,
    battle_script.c:8040-8043; ``BtlCmd_TryCamouflage``, :8007-8010). An unknown terrain
    (``TERRAIN_UNKNOWN``, -1) is clamped to row 0 so that no table is ever indexed with -1;
    its readers test ``terrain_known`` and do nothing then (coverage and ``trace_to_state``
    refuse such a battle before it reaches the engine)."""
    return jnp.clip(jnp.asarray(state.terrain, _I32), 0, TERRAIN_SPECIAL)


def terrain_known(state: BattleState) -> jnp.ndarray:
    """``bool``: the battle's terrain is a ``TERRAIN_*`` id, not ``TERRAIN_UNKNOWN``."""
    return jnp.asarray(state.terrain, _I32) >= 0


def called_move(tables: Tables, state: BattleState, move) -> jnp.ndarray:
    """``int32``: the move ``move`` runs from its move script on (``battleCtx->moveCur``):
    for Nature Power (``EffectId.NATURE_POWER``), effect_script_0173.s's ``GetTerrainMove``
    (``msgMoveTemp = sTerrainMove[min(terrain, TERRAIN_SPECIAL)]``, battle_script.c:8036-8048)
    then ``GoToMoveScript`` (:2111-2134: ``moveCur = msgMoveTemp``, the defender re-picked
    for it), so the accuracy check, Protect, the type chart, the damage, the secondary
    effect and ``moveHit`` read the called move (``step._use_move``); otherwise ``move``
    itself. Nature Power stays the move for the status checks, the PP, Taunt, Disable,
    priority and ``movePrevByBattler`` (``moveTemp``, set at command selection,
    bcp.c:1861-1898, is never changed by ``GoToMoveScript``). With ``TERRAIN_UNKNOWN`` the
    call is ``MOVE_NONE`` (0, the guard of ``terrain_index``): the move then does nothing."""
    move = jnp.asarray(move, _I32)
    terrain_move = jnp.asarray(TERRAIN_MOVES, _I32)[terrain_index(state)]
    terrain_move = jnp.where(terrain_known(state), terrain_move, 0)
    nature_power = tables.move_effect[move] == EffectId.NATURE_POWER
    return jnp.where(nature_power, terrain_move, move)


def used_move(tables: Tables, state: BattleState, side, action) -> jnp.ndarray:
    """``int32``: ``called_move`` of ``selected_move``: the move whose accuracy, crit odds,
    damage and effect a move action draws for."""
    return called_move(tables, state, selected_move(tables, state, side, action)[0])


def effective_speed(tables: Tables, state: BattleState, side, suppressed=None) -> jnp.ndarray:
    """``int32`` speed as ``BattleSystem_CompareBattlerSpeed`` computes it. ``suppressed``
    (``bool``, default ``damage.weather_suppressed`` of ``state``) overrides the weather
    lock (``NO_CLOUD_NINE``), for a lock that can change before the speed is read
    (``enumerate._end_speeds``)."""
    if suppressed is None:
        suppressed = weather_suppressed(tables, state)
    mon = state.active[side]
    ability = state.ability[side, mon]
    stage = state.stages[side, Stat.SPE].astype(_I32)
    stage = jnp.where(ability == AbilityId.SIMPLE, simple_stage(stage), stage)
    speed = apply_stage(state.stats[side, mon, Stat.SPE - 1].astype(_I32), stage)
    # :1228-1238 Swift Swim (Battler_Ability) in rain, inside NO_CLOUD_NINE.
    swim = (
        (ability == AbilityId.SWIFT_SWIM)
        & (state.weather == Weather.RAIN)
        & ~jnp.asarray(suppressed, bool)
    )
    speed = jnp.where(swim, speed * 2, speed)
    # :1248 Choice Scarf (Battler_HeldItemEffect), after the speed-halving items (:1240-1246,
    # unsupported) and before paralysis.
    scarf = tables.item_effect[state.item[side, mon]] == ItemId.CHOICE_SCARF
    speed = jnp.where(scarf, speed * 15 // 10, speed)
    speed = jnp.where(state.status[side, mon] == Status.PARALYSIS, speed // 4, speed)
    return jnp.where(state.tailwind[side] != 0, speed * 2, speed)


def _holds(tables: Tables, state: BattleState, side, item) -> jnp.ndarray:
    return tables.item_effect[state.item[side, state.active[side]]] == item


def pinch_divisor(tables: Tables, state: BattleState, side, mon, item) -> jnp.ndarray:
    """``int32``: the hold param of a pinch berry or Custap Berry ``item`` held by ``side``'s
    party ``mon``, halved (``itemPower /= 2``) when the holder's own ability is Gluttony
    (``Battler_Ability``: battle_lib.c:4774-4776, :1289-1291). The berry triggers at
    ``curHP <= maxHP / divisor`` (C integer division); build_tables keeps the divisor >= 1
    (``tools.effect_map.PINCH_DIVISOR_ITEMS``)."""
    power = tables.item_param[item, 0].astype(_I32)
    gluttony = state.ability[side, mon] == AbilityId.GLUTTONY
    return jnp.where(gluttony, power // 2, power)


def custap_active(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[2]``: each side's active battler holds a Custap Berry
    (``HOLD_EFFECT_PINCH_PRIORITY``) at ``curHP <= maxHP / pinch_divisor``: the decomp's
    ``battlerNQuickClaw`` flag (battle_lib.c:1288-1300, :1354-1366), whatever
    ``ignoreQuickClaw``."""
    out = []
    for side in (0, 1):
        mon = state.active[side]
        item = state.item[side, mon]
        hp = state.hp[side, mon].astype(_I32)
        max_hp = state.max_hp[side, mon].astype(_I32)
        low = hp <= max_hp // pinch_divisor(tables, state, side, mon, item)
        out.append((tables.item_effect[item] == ItemId.CUSTAP_BERRY) & low)
    return jnp.stack(out)


def custap_flags(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: ``moveEffectsData.custapBerry`` after the turn order: ``CalcTurnOrder``
    compares two FIGHT commands with ``ignoreQuickClaw = FALSE`` (bcp.c:772-786), which writes
    the flag of each activated holder (``custap_active`` on the turn-start ``state``,
    battle_lib.c:1296-1298, :1362-1364). The holder eats the berry when its move starts
    (``subscript_check_quick_claw.s:26-28``, ``step._use_move``); until then Pluck cannot take
    it (``hooks.pluck_berry``)."""
    actions = jnp.asarray(actions, _I32)
    fight = ~is_switch(actions) & ~is_item(actions)
    return custap_active(tables, state) & fight[0] & fight[1]


def quick_claw_active(tables: Tables, state: BattleState, qc) -> jnp.ndarray:
    """``bool[2]``: each side's active battler holds a Quick Claw and its ``speedRand``
    draw ``qc`` (``int[2]``, 1 = ``speedRand % 5 == 0``) activates it (lib.c:1278-1286)."""
    qc = jnp.asarray(qc)
    return jnp.stack(
        [_holds(tables, state, side, ItemId.QUICK_CLAW) & (qc[side] != 0) for side in (0, 1)]
    )


# BattleSystem_CompareBattlerSpeed's results (include/constants/battle.h:163-165).
COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER, COMPARE_SPEED_TIE = 0, 1, 2


def compare_speed(tables: Tables, state: BattleState, battler1: int, tie_roll, qc) -> jnp.ndarray:
    """``int32``: ``BattleSystem_CompareBattlerSpeed(battler1, the other side)`` within equal
    priority (lib.c:1401-1447): ``COMPARE_SPEED_FASTER``, ``SLOWER`` or ``TIE``.
    ``battler1`` is a side (static int). Quick Claw first (``qc``: each side's ``speedRand``
    bit, ``quick_claw_active``): both active compare plain speeds, Trick Room and Lagging
    Tail ignored (:1402-1407); a lone active one is FASTER (:1408-1411), even under Trick
    Room and against a Lagging Tail. Lagging Tail: a lone holder is SLOWER (:1418-1421); two
    holders compare speeds reversed, the faster SLOWER, Trick Room ignored (:1412-1417). Else
    (Stall is unsupported) Trick Room reverses the speeds (:1432-1439). At equal speed each
    of these branches gives TIE when ``tie_roll`` (``RandNext() & 1``) is 1, else FASTER,
    never SLOWER; the decomp draws that bit only there. The dead-battler short-circuit
    (:1205-1211) is not part of this function, as it was not of ``player_second``; the
    trainer AI compares at its decision, with both actives alive. A Custap Berry in its pinch
    (``custap_active``) sets the same flag as an activated Quick Claw (:1288-1300)."""
    other = 1 - battler1
    speed = (effective_speed(tables, state, 0), effective_speed(tables, state, 1))
    claw = quick_claw_active(tables, state, qc) | custap_active(tables, state)
    lag = tuple(_holds(tables, state, side, ItemId.LAGGING_TAIL) for side in (0, 1))
    speed1, speed2 = speed[battler1], speed[other]
    claw1, claw2 = claw[battler1], claw[other]
    lag1, lag2 = lag[battler1], lag[other]
    reversed_ = ~(claw1 & claw2) & ((lag1 & lag2) | ((state.trick_room != 0) & ~lag1 & ~lag2))
    slower = jnp.where(reversed_, speed1 > speed2, speed1 < speed2)
    tie = (speed1 == speed2) & (jnp.asarray(tie_roll) != 0)
    by_speed = jnp.where(
        slower, COMPARE_SPEED_SLOWER, jnp.where(tie, COMPARE_SPEED_TIE, COMPARE_SPEED_FASTER)
    )
    lagging = jnp.where(
        lag1 != lag2, jnp.where(lag1, COMPARE_SPEED_SLOWER, COMPARE_SPEED_FASTER), by_speed
    )
    lone_claw = jnp.where(claw2, COMPARE_SPEED_SLOWER, COMPARE_SPEED_FASTER)
    result = jnp.where(claw1 != claw2, lone_claw, jnp.where(claw1, by_speed, lagging))
    return result.astype(_I32)


def battler_slower(tables: Tables, state: BattleState, battler1: int, tie_roll, qc) -> jnp.ndarray:
    """``bool``: ``compare_speed`` is not FASTER: within equal priority ``battler1`` acts
    after the other side. With ``tie_roll`` 0 no TIE arises, so this is exactly
    ``compare_speed == COMPARE_SPEED_SLOWER`` (the trainer AI's speed compare,
    ``ai.commands.if_speed_compare_equal_to``)."""
    return compare_speed(tables, state, battler1, tie_roll, qc) != COMPARE_SPEED_FASTER


def player_second(tables: Tables, state: BattleState, tie_roll, qc) -> jnp.ndarray:
    """``bool``: ``BattleSystem_CompareBattlerSpeed(player, opponent)`` swaps the two
    within equal priority (lib.c:1401-1447): ``battler_slower`` with ``battler1 = 0``.
    Equal speeds swap on ``tie_roll`` (``RandNext() & 1``), drawn only where the decomp
    compares speeds."""
    return battler_slower(tables, state, 0, tie_roll, qc)


def action_order(tables: Tables, state: BattleState, actions, outcome) -> jnp.ndarray:
    """``int8[2]``: the two sides in execution order. ``outcome.speed_tie`` decides
    equal speeds (0 = player first): the decomp swaps on ``RandNext() & 1``. Quick Claw
    reads ``state.qc_roll``, the draw of the turn before (module docstring), never the
    outcome."""
    actions = jnp.asarray(actions, _I32)
    switch, item = is_switch(actions), is_item(actions)
    early = switch | item  # items and switches go before moves
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    both_moves = ~early[0] & ~early[1]
    priority = jnp.where(both_moves, tables.move_priority[moves].astype(_I32), 0)
    second = player_second(tables, state, outcome.speed_tie, state.qc_roll)
    player_second_ = (priority[0] < priority[1]) | ((priority[0] == priority[1]) & second)
    # Two actions of one kind compare speeds; an item and a switch go in battler order
    # (player first); a switch or item always precedes a move.
    same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
    player_second_ = jnp.where(same_kind, player_second_, early[1] & ~early[0])
    first = player_second_.astype(jnp.int8)
    return jnp.stack([first, 1 - first]).astype(jnp.int8)
