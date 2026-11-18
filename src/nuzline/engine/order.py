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
(``:1274``) and Trick Room's reversal (``:1432``), and Quick Feet ``* 15 / 10`` with any
status instead of the paralysis quartering (``:1257-1261``). Chlorophyll, Slow Start,
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
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    MOVE_ACTIONS,
    choice_locked,
    gravity_banned,
    locked_slot,
    move_imprisoned,
    usable_moves,
)
from nuzline.engine.damage import apply_stage, simple_stage, weather_suppressed
from nuzline.engine.ids import (
    CANNOT_ENCORE_MOVES,
    INVOKER_MOVES,
    METRONOME_DRAW,
    METRONOME_HEAVY,
    MOVE_STRUGGLE,
    NATURE_DISLIKED_FLAVOR,
    SECRET_POWER_EFFECTS,
    SPEED_HALVING_HOLD_EFFECTS,
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
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


# The invokers (ids.EffectId.SLEEP_TALK, METRONOME, COPYCAT): ``call_kind``.
CALL_NONE, CALL_SLEEP_TALK, CALL_METRONOME, CALL_COPYCAT = 0, 1, 2, 3
CALL_ASSIST = 4  # E2 batch 10, B10-11


def call_kind(tables: Tables, move) -> jnp.ndarray:
    """``int32``: which invoker ``move`` is (``CALL_*``), ``CALL_NONE`` for any other move."""
    effect = tables.move_effect[jnp.asarray(move, _I32)]
    return jnp.where(
        effect == EffectId.SLEEP_TALK,
        CALL_SLEEP_TALK,
        jnp.where(
            effect == EffectId.METRONOME,
            CALL_METRONOME,
            jnp.where(
                effect == EffectId.COPYCAT,
                CALL_COPYCAT,
                jnp.where(effect == EffectId.ASSIST, CALL_ASSIST, CALL_NONE),
            ),
        ),
    ).astype(_I32)


def assist_candidates(tables: Tables, state: BattleState, side) -> tuple:
    """``(moves int32[SLOTS * 4], valid bool[...])``: BtlCmd_TryAssist's list (battle_script.c:
    6418-6436) for ``side``'s active battler: in party order, every move of every other party
    slot holding a Pokemon (fainted ones too: only SPECIES_NONE and eggs are skipped) that is
    not an invoker (Move_IsInvoker, ``ids.INVOKER_MOVES``) and that Move_CanBeMetronomed allows
    (``MoveFlag.NO_METRONOME``, and ``actions.gravity_banned`` under Gravity: its Heal Block test
    is owed with that move). A
    move two mates know is listed twice. The call is ``RandNext() % numMoves`` (:6439):
    ``Outcome.call``'s uniform value, ``invoked_move`` taking the ``call % count``-th entry."""
    moves = state.moves[side].astype(_I32)  # [SLOTS, 4]
    slots = jnp.arange(moves.shape[0])[:, None]
    mate = (slots != state.active[side]) & (state.species[side][:, None] != 0)
    invoker = jnp.isin(moves, jnp.asarray(INVOKER_MOVES, _I32))
    allowed = ((tables.move_flags[moves] & MoveFlag.NO_METRONOME) == 0) & ~gravity_banned(
        state, moves
    )
    valid = mate & ~invoker & allowed
    return moves.reshape(-1), valid.reshape(-1)


def _nth(mask, k) -> jnp.ndarray:
    """``int32``: the index of the ``k``-th (0-based) true entry of ``mask`` (0 when none)."""
    rank = jnp.cumsum(mask.astype(_I32)) - 1
    return jnp.argmax(mask & (rank == k)).astype(_I32)


def sleep_talk_slots(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[4]``: the slots ``BtlCmd_TrySleepTalk`` (battle_script.c:4907-4937) can draw for
    ``side``'s active battler, as the battler is now: not ``MoveFlag.NO_SLEEP_TALK``
    (Move_IsInvoker, which holds an empty slot's MOVE_NONE, Focus Punch, Uproar, Chatter,
    Move_IsMultiTurn: :4918-4925), and not flagged by ``BattleSystem_CheckInvalidMoves`` with every
    check but NO_PP (:4927; battle_lib.c:2255-2316): the disabled move, a power-0 move while
    taunted, a Choice holder's moves other than its lock (``actions.choice_locked``: a lock
    it does not know is cleared there and restricts nothing) and a move a foe's Imprison seals
    (``actions.move_imprisoned``, CHECK_INVALID_IMPRISONED, battle_lib.c:2287-2290; ruling
    B7-13) and, under Encore, every move but the encored one (battle_lib.c:2302-2304, B10-9).
    Torment and Heal Block come from unsupported moves; Gravity (``Move_FailsInHighGravity``,
    ``gravity_banned``, CHECK_INVALID_GRAVITY, :2292-2295) is read here (G-67)."""
    mon = state.active[side]
    moves = state.moves[side, mon].astype(_I32)
    ok = (moves > 0) & ((tables.move_flags[moves] & MoveFlag.NO_SLEEP_TALK) == 0)
    ok = ok & (moves != state.disabled_move[side])
    ok = ok & ~((state.taunt[side] > 0) & (tables.move_power[moves] == 0))
    ok = ok & ~move_imprisoned(state, side, moves)
    encore = state.encore_move[side].astype(_I32)
    ok = ok & ((encore == 0) | (moves == encore))
    lock = state.choice_lock[side].astype(_I32)
    return ok & (~choice_locked(tables, state, side) | (moves == lock))


def metronome_candidates(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[N]`` over move ids: what ``BtlCmd_Metronome`` (battle_script.c:4535-4562) can call
    for ``side``'s active battler: ids 1..``METRONOME_DRAW``, none of the battler's own moves
    (:4544-4551), none of ``MoveFlag.NO_METRONOME`` (``Move_CanBeMetronomed``, battle_lib.c:
    7204-7218; Gravity's test is ``actions.gravity_banned`` (:7198-7201), Heal Block's is owed)."""
    ids = jnp.arange(tables.move_effect.shape[0], dtype=_I32)
    own = state.moves[side, state.active[side]].astype(_I32)
    ok = (ids >= 1) & (ids <= METRONOME_DRAW)
    ok = ok & ((tables.move_flags[ids] & MoveFlag.NO_METRONOME) == 0) & ~gravity_banned(state, ids)
    return ok & ~jnp.any(ids[:, None] == own[None, :], axis=1)


def metronome_weights(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``int32[N]``: each move id's weight in Metronome's draw (``ids.METRONOME_HEAVY``): 141 or
    140 of 65536 for a candidate, 0 otherwise. The call is the candidate itself: its
    probability is its weight over the candidates' sum (the redraw loop, :4538-4560)."""
    ids = jnp.arange(tables.move_effect.shape[0], dtype=_I32)
    weight = jnp.where(ids <= METRONOME_HEAVY, 141, 140)
    return jnp.where(metronome_candidates(tables, state, side), weight, 0).astype(_I32)


def copycat_call(tables: Tables, move_prev, state: BattleState | None = None) -> jnp.ndarray:
    """``int32``: the move ``BtlCmd_TryCopycat`` (battle_script.c:7328-7341) calls with the
    battle's ``movePrev`` ``move_prev``, 0 when it fails: MOVE_NONE, an invoker or a move
    ``Move_CanBeMetronomed`` refuses (every invoker is in sCannotMetronomeMoves, so
    ``MoveFlag.NO_METRONOME`` covers both)."""
    move_prev = jnp.asarray(move_prev, _I32)
    ok = (move_prev != 0) & ((tables.move_flags[move_prev] & MoveFlag.NO_METRONOME) == 0)
    if state is not None:  # Move_CanBeMetronomed's Gravity test (battle_lib.c:7198-7201), G-67
        ok = ok & ~gravity_banned(state, move_prev)
    return jnp.where(ok, move_prev, 0).astype(_I32)


def invoked_move(tables: Tables, state: BattleState, side, move, call, move_prev=None):
    """``int32``: the move ``side``'s ``move`` runs through its invoker script, read with
    ``state`` as it is when the script runs; 0 when the invoker finds none. Sleep Talk: the
    allowed slot ``call % count`` (``Outcome.call``, ``ids.SLEEP_TALK_VALUES``); Metronome: the
    candidate of rank ``call % count`` (``metronome_candidates``; ``Outcome.call`` draws it by
    ``metronome_weights``); Copycat: ``copycat_call`` of ``move_prev`` (default
    ``state.move_prev``, the battle's movePrev when the script runs). Any other move: itself.
    Nature Power on the result is ``called_move``'s."""
    move = jnp.asarray(move, _I32)
    call = jnp.asarray(call, _I32)
    kind = call_kind(tables, move)
    slots = sleep_talk_slots(tables, state, side)
    n_slots = jnp.sum(slots.astype(_I32))
    slot = _nth(slots, call % jnp.maximum(n_slots, 1))
    talk = jnp.where(n_slots > 0, state.moves[side, state.active[side], slot].astype(_I32), 0)
    cands = metronome_candidates(tables, state, side)
    n_cands = jnp.sum(cands.astype(_I32))
    metronome = jnp.where(n_cands > 0, _nth(cands, call % jnp.maximum(n_cands, 1)), 0)
    prev = state.move_prev if move_prev is None else move_prev
    copycat = copycat_call(tables, prev, state)
    listed, valid = assist_candidates(tables, state, side)
    n_listed = jnp.sum(valid.astype(_I32))
    assist = jnp.where(n_listed > 0, listed[_nth(valid, call % jnp.maximum(n_listed, 1))], 0)
    return jnp.where(
        kind == CALL_SLEEP_TALK,
        talk,
        jnp.where(
            kind == CALL_METRONOME,
            metronome,
            jnp.where(kind == CALL_COPYCAT, copycat, jnp.where(kind == CALL_ASSIST, assist, move)),
        ),
    ).astype(_I32)


def encore_slot_of(tables: Tables, state: BattleState, target_side) -> tuple:
    """``(fails, slot)``: BtlCmd_TryEncore (battle_script.c:4725-4748) on ``target_side``'s
    battler: the slot of its movePrevByBattler (``last_move``; Battler_SlotForMove), and the
    failure (MOVE_STATUS_FAILED): it is encored already, its last move is none, not in its slots,
    one Move_CanBeEncored refuses (the battle effect of a ``CANNOT_ENCORE_MOVES`` move,
    battle_lib.c:7221-7231) or out of PP."""
    mon = state.active[target_side]
    last = state.last_move[target_side].astype(_I32)
    moves = state.moves[target_side, mon].astype(_I32)
    held = (moves == last) & (last != 0)
    slot = jnp.argmax(held).astype(_I32)
    banned = tables.move_ai_effect[jnp.asarray(CANNOT_ENCORE_MOVES, _I32)]
    refused = jnp.any(tables.move_ai_effect[last] == banned)
    fails = (
        (state.encore_move[target_side] != 0)
        | (last == 0)
        | ~jnp.any(held)
        | refused
        | (state.pp[target_side, mon, slot] == 0)
    )
    return fails, slot


def disabled_state(state: BattleState, sides) -> BattleState:
    """``state`` with ``disabled_move = last_move`` on ``sides`` (``bool[2]``): the only set
    change of Sleep Talk a turn can make. BtlCmd_TryDisable (battle_script.c:4586-4594) disables
    the defender's movePrevByBattler, only while it has PP and nothing is disabled yet; a Sleep
    Talk user that moves second has not moved this turn, so the move is its turn-start
    ``last_move``."""
    value = jnp.where(sides, state.last_move, state.disabled_move)
    return state._replace(disabled_move=value.astype(state.disabled_move.dtype))


def sealed_state(state: BattleState, sides) -> BattleState:
    """``state`` with ``MOVE_EFFECT_IMPRISON`` on every battler of the other side of ``sides``
    (``bool[2]``): a faster foe's Imprison this turn (``BtlCmd_TryImprison``, battle_script.c:
    6724), which then seals its moves for the Sleep Talk user (battle_script.c:4927; battle
    b on side b in singles)."""
    foe_of = jnp.asarray(sides)[::-1]
    value = jnp.where(foe_of, 1, state.imprison)
    return state._replace(imprison=value.astype(state.imprison.dtype))


def encored_state(tables: Tables, state: BattleState, sides) -> BattleState:
    """``state`` with each battler of ``sides`` (``bool[2]``) encored into its last move
    (encoredMove = movePrevByBattler, encoredMoveSlot its slot: BtlCmd_TryEncore, battle_script.c:
    4740-4741), the hypothetical state of a takeover variant (``takeover_possible``; ruling B11-1).
    The count is left as it is: nothing before the turn's end reads it."""
    sides = jnp.asarray(sides, bool)
    slots = jnp.stack([encore_slot_of(tables, state, side)[1] for side in (0, 1)])
    move = jnp.where(sides, state.last_move, state.encore_move).astype(state.encore_move.dtype)
    slot = jnp.where(sides, slots, state.encore_slot).astype(state.encore_slot.dtype)
    return state._replace(encore_move=move, encore_slot=slot)


def can_call_encore(tables: Tables, state: BattleState, side, move) -> jnp.ndarray:
    """``bool``: ``side``'s move ``move`` can run Encore before its foe's action (ruling B11-1):
    Encore itself, or an invoker whose call can be Encore: Sleep Talk when the user knows it
    (BtlCmd_TrySleepTalk draws its own slots, battle_script.c:4907-4937), Metronome when it is a
    candidate (``metronome_candidates``: not NO_METRONOME, not a move the user knows), Assist when
    a listed mate's move is (``assist_candidates``), Copycat when the battle's movePrev is Encore
    (``copycat_call``: its foe's move of this turn comes after the foe's action, too late to
    replace it). Over-approximate (Sleep Talk's sleep and set are the turn's)."""
    move = jnp.asarray(move, _I32)
    effect = tables.move_effect
    kind = call_kind(tables, move)
    own = state.moves[side, state.active[side]].astype(_I32)
    talk = jnp.any((own != 0) & (effect[own] == EffectId.ENCORE))
    ids = jnp.arange(effect.shape[0], dtype=_I32)
    metronome = jnp.any(
        metronome_candidates(tables, state, side) & (effect[ids] == EffectId.ENCORE)
    )
    listed, valid = assist_candidates(tables, state, side)
    assist = jnp.any(valid & (effect[listed] == EffectId.ENCORE))
    copycat = effect[copycat_call(tables, state.move_prev, state)] == EffectId.ENCORE
    return (
        (effect[move] == EffectId.ENCORE)
        | ((kind == CALL_SLEEP_TALK) & talk)
        | ((kind == CALL_METRONOME) & metronome)
        | ((kind == CALL_ASSIST) & assist)
        | ((kind == CALL_COPYCAT) & copycat)
    )


def takeover_possible(tables: Tables, state: BattleState, actions, side, foe) -> jnp.ndarray:
    """``bool``: ``foe``'s Encore can replace ``side``'s chosen move this turn (ruling B11-1).
    BattleControllerPlayer_FightCommand (bcp.c:1866-1873) runs an encored battler's encored move,
    after Struggle, whatever it chose; an Encore landing before its action this turn (a faster
    foe's: BtlCmd_TryEncore, battle_script.c:4725-4748) therefore replaces it. Possible when the
    side uses a move without Struggling, is not encored at the turn start, ``encore_slot_of``
    would not fail it (its last move held, encorable, with PP), that move differs from the chosen
    one, and ``foe`` uses Encore or an invoker that can call it (``can_call_encore``). ``foe`` is
    the battler whose Encore targets ``side`` (doubles rule 1: the caller names it; B11-11).
    Sound, not tight: the call laws drop the rows where it did not happen
    (``enumerate.call_consistent``)."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    chosen, struggling = selected_move(tables, state, side, actions[side])
    foe_chosen = selected_move(tables, state, foe, actions[foe])[0]
    fails = encore_slot_of(tables, state, side)[0]
    foe_can = uses_move[foe] & can_call_encore(tables, state, foe, foe_chosen)
    return (
        uses_move[side]
        & ~struggling
        & ~fails
        & (state.last_move[side].astype(_I32) != chosen)
        & foe_can
    )


def move_effects(tables: Tables, state: BattleState, move) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(effect, param)``: ``tables.move_effect[move]`` and ``tables.move_param[move]`` (``int32``,
    ``int32[..., 3]``), with Secret Power's resolved to the terrain's secondary effect
    (``ids.SECRET_POWER_EFFECTS``; effect_script_0197.s ``GetTerrainSecondaryEffect``,
    battle_script.c:8063-8075: ``min(terrain, TERRAIN_SPECIAL)``), the effect every hit and law
    reads. An unknown terrain (coverage refuses it) resolves to ``PLAIN``. ``move`` may be
    batched."""
    move = jnp.asarray(move, _I32)
    effect = tables.move_effect[move].astype(_I32)
    param = tables.move_param[move].astype(_I32)
    row = terrain_index(state)
    sp_effect = jnp.asarray([e for e, _ in SECRET_POWER_EFFECTS], _I32)[row]
    sp_param = jnp.asarray([p for _, p in SECRET_POWER_EFFECTS], _I32)[row]
    sp_effect = jnp.where(terrain_known(state), sp_effect, EffectId.PLAIN)
    secret = effect == EffectId.SECRET_POWER
    return (
        jnp.where(secret, sp_effect, effect),
        jnp.where(secret[..., None], sp_param, param),
    )


def used_move(
    tables: Tables, state: BattleState, side, action, call=None, move_prev=None
) -> jnp.ndarray:
    """``int32``: ``called_move`` of ``selected_move``: the move whose accuracy, crit odds,
    damage and effect a move action draws for. ``call`` (``int[2]``, ``Outcome.call``) resolves
    an invoker (``invoked_move``; ``move_prev`` for Copycat, default ``state.move_prev``); one
    whose call finds no move uses itself (it fails). Without ``call`` an invoker uses itself:
    the heuristics' view (``search.playout``), never a chance law's."""
    chosen = selected_move(tables, state, side, action)[0]
    if call is None:
        return called_move(tables, state, chosen)
    invoked = invoked_move(tables, state, side, chosen, jnp.asarray(call)[side], move_prev)
    return jnp.where(invoked == 0, chosen, called_move(tables, state, invoked)).astype(_I32)


def used_moves(
    tables: Tables, state: BattleState, actions, call, after_foe, chosen=None
) -> jnp.ndarray:
    """``int32[2]``: ``used_move`` of each side with ``call`` (``Outcome.call``) and Copycat's
    movePrev as the order leaves it: ``after_foe[side]`` (the foe used a move before the side's
    action) makes it the foe's chosen move (moveTemp, UpdateMoveBuffers bcp.c:3936-3942; a foe
    whose attack message did not show leaves MOVE_NONE and Copycat fails, reading nothing),
    else the turn-start ``state.move_prev``. ``chosen`` (``int32[2]``, default each action's
    ``selected_move``) overrides the moveTemp of each side: a takeover variant's encored move
    (ruling B11-1), with ``state`` its ``encored_state``."""
    actions = jnp.asarray(actions, _I32)
    if chosen is None:
        chosen = jnp.stack(
            [selected_move(tables, state, side, actions[side])[0] for side in (0, 1)]
        )
    chosen = jnp.asarray(chosen, _I32)
    after_foe = jnp.asarray(after_foe, bool)
    call = jnp.asarray(call, _I32)
    out = []
    for side in (0, 1):
        prev = jnp.where(after_foe[side], chosen[1 - side], state.move_prev.astype(_I32))
        invoked = invoked_move(tables, state, side, chosen[side], call[side], prev)
        out.append(
            jnp.where(invoked == 0, chosen[side], called_move(tables, state, invoked)).astype(_I32)
        )
    return jnp.stack(out)


def effective_speed(
    tables: Tables, state: BattleState, side, suppressed=None, weather=None
) -> jnp.ndarray:
    """``int32`` speed as ``BattleSystem_CompareBattlerSpeed`` computes it. ``suppressed``
    (``bool``, default ``damage.weather_suppressed`` of ``state``) overrides the weather
    lock (``NO_CLOUD_NINE``) and ``weather`` (default ``state.weather``) the weather, for a
    lock or a weather that can change before the speed is read (``enumerate._end_speeds``)."""
    if suppressed is None:
        suppressed = weather_suppressed(tables, state)
    weather = state.weather if weather is None else weather
    mon = state.active[side]
    ability = state.ability[side, mon]
    stage = state.stages[side, Stat.SPE].astype(_I32)
    stage = jnp.where(ability == AbilityId.SIMPLE, simple_stage(stage), stage)
    speed = apply_stage(state.stats[side, mon, Stat.SPE - 1].astype(_I32), stage)
    # :1228-1238 Swift Swim in rain, Chlorophyll in sun (Battler_Ability), inside NO_CLOUD_NINE.
    swim = (
        ((ability == AbilityId.SWIFT_SWIM) & (weather == Weather.RAIN))
        | ((ability == AbilityId.CHLOROPHYLL) & (weather == Weather.SUN))
    ) & ~jnp.asarray(suppressed, bool)
    speed = jnp.where(swim, speed * 2, speed)
    # :1240-1246 sSpeedHalvingItemEffects (:1160-1169: Iron Ball, Macho Brace, the Power items),
    # read through BattleSystem_GetItemData on the held item (no Klutz / Embargo), G-56.
    hold = tables.item_hold_effect[state.item[side, mon]].astype(_I32)
    halved = jnp.isin(hold, jnp.asarray(SPEED_HALVING_HOLD_EFFECTS, _I32))
    speed = jnp.where(halved, speed // 2, speed)
    # :1248 Choice Scarf (Battler_HeldItemEffect), after the speed-halving items and before
    # paralysis.
    scarf = tables.item_effect[state.item[side, mon]] == ItemId.CHOICE_SCARF
    speed = jnp.where(scarf, speed * 15 // 10, speed)
    # :1257-1261 Quick Feet (Battler_Ability) with any status (MON_CONDITION_ANY): x15/10,
    # and then no paralysis quartering; else paralysis quarters it.
    status = state.status[side, mon]
    quick_feet = (ability == AbilityId.QUICK_FEET) & (status != Status.NONE)
    speed = jnp.where(
        quick_feet,
        speed * 15 // 10,
        jnp.where(status == Status.PARALYSIS, speed // 4, speed),
    )
    # :1268-1272 Unburden (Battler_Ability) with canUnburden and no held item: x2 (Slow Start,
    # just before, is unsupported); then Tailwind (:1274-1276).
    unburdened = (
        (ability == AbilityId.UNBURDEN)
        & (state.can_unburden[side] != 0)
        & (state.item[side, mon] == 0)
    )
    speed = jnp.where(unburdened, speed * 2, speed)
    return jnp.where(state.tailwind[side] != 0, speed * 2, speed)


def _holds(tables: Tables, state: BattleState, side, item) -> jnp.ndarray:
    return tables.item_effect[state.item[side, state.active[side]]] == item


def pinch_divisor(tables: Tables, state: BattleState, side, mon, item) -> jnp.ndarray:
    """``int32``: the hold param of a pinch berry or Custap Berry ``item`` held by ``side``'s
    party ``mon``, halved (``itemPower /= 2``) when the holder's own ability is Gluttony
    (``Battler_Ability``: battle_lib.c:4774-4776, :1289-1291). The berry triggers at
    ``curHP <= maxHP / divisor`` (``pinch_limit``)."""
    power = tables.item_param[item, 0].astype(_I32)
    gluttony = state.ability[side, mon] == AbilityId.GLUTTONY
    return jnp.where(gluttony, power // 2, power)


def pinch_limit(max_hp, divisor) -> jnp.ndarray:
    """``int32``: ``maxHP / divisor`` as the ROM computes it at every pinch-berry and Custap
    site (battle_lib.c:4778 etc., :1294, :5201 etc.): the compiler calls the ARM9 runtime's
    unsigned divide ``_u32_div_f`` (rom:arm9@0x020E2178, e.g. from overlay16@0x02257DA2 for
    Ganlon, @0x02253202 for Custap), whose first instructions ``cmp r1, #0; bxeq lr`` return
    the dividend for a zero divisor (ruling G-53). So a param 0 (Kaizo's Ganlon and Apicot) or
    1 under Gluttony gives ``maxHP``: the berry's HP test always passes."""
    max_hp = jnp.asarray(max_hp, _I32)
    divisor = jnp.asarray(divisor, _I32)
    return jnp.where(divisor == 0, max_hp, max_hp // jnp.maximum(divisor, 1))


def dislikes_berry(tables: Tables, state: BattleState, side, mon, item) -> jnp.ndarray:
    """``bool``: ``item`` is a flavor berry (``ItemId.FLAVOR_BERRY``) whose flavor
    (``item_param[item, 1]``) ``side``'s party ``mon`` dislikes:
    ``Pokemon_GetFlavorAffinityOf(personality, flavor) == -1`` (battle_lib.c:4703 etc., :5683
    etc.; src/pokemon.c:4123-4127, ``ids.NATURE_DISLIKED_FLAVOR`` by its nature), ruling G-4."""
    disliked = jnp.asarray(NATURE_DISLIKED_FLAVOR, _I32)[state.nature[side, mon].astype(_I32)]
    flavor = tables.item_param[item, 1].astype(_I32)
    return (tables.item_effect[item] == ItemId.FLAVOR_BERRY) & (disliked == flavor)


def custap_active(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[2]``: each side's active battler holds a Custap Berry
    (``HOLD_EFFECT_PINCH_PRIORITY``) at ``curHP <= pinch_limit(maxHP, pinch_divisor)``: the decomp's
    ``battlerNQuickClaw`` flag (battle_lib.c:1288-1300, :1354-1366), whatever
    ``ignoreQuickClaw``."""
    out = []
    for side in (0, 1):
        mon = state.active[side]
        item = state.item[side, mon]
        hp = state.hp[side, mon].astype(_I32)
        max_hp = state.max_hp[side, mon].astype(_I32)
        low = hp <= pinch_limit(max_hp, pinch_divisor(tables, state, side, mon, item))
        out.append((tables.item_effect[item] == ItemId.CUSTAP_BERRY) & low)
    return jnp.stack(out)


def custap_flags(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: ``moveEffectsData.custapBerry`` after the turn order: ``CalcTurnOrder``
    compares two FIGHT commands with ``ignoreQuickClaw = FALSE`` (bcp.c:772-786), which writes
    the flag of each activated holder (``custap_active`` on the turn-start ``state``,
    battle_lib.c:1296-1298, :1362-1364). The holder eats the berry when its move starts
    (``subscript_check_quick_claw.s:26-28``, ``step._use_move``); until then Pluck cannot take
    it (``hooks.pluck_berry``). ``BattleSystem_SortMonActionOrder`` (battle_lib.c:7276-7300)
    writes it too, through the same compare with ``ignoreQuickClaw = FALSE``, when a Trick
    Room activation re-sorts the turn (bcp.c:3990-3993); Trick Room is unsupported, and
    whoever adds it must re-set the flag there."""
    actions = jnp.asarray(actions, _I32)
    fight = ~is_switch(actions) & ~is_item(actions)
    return custap_active(tables, state) & fight[0] & fight[1]


def claw_flags(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[2]``: ``moveEffectsData.quickClaw`` after the turn order: ``CalcTurnOrder``'s
    FIGHT-against-FIGHT compare (``ignoreQuickClaw = FALSE``, bcp.c:772-786) sets it for each
    activated Quick Claw (battle_lib.c:1283, :1349; ``quick_claw_active`` on the turn's
    ``qc_roll``). The holder's own BEFORE_MOVE_STATE_QUICK_CLAW clears it
    (subscript_check_quick_claw.s:22-23), so it stands until the holder's move starts; only
    Knock Off reads it (subscript_knock_off.s:8), B10-6. The sort compares every pair of
    battlers whose commands are both FIGHT (bcp.c:766-778), so a FIGHT battler is compared when
    any other battler fights (doubles design section 5 rule 4)."""
    actions = jnp.asarray(actions, _I32)
    fight = ~is_switch(actions) & ~is_item(actions)
    others = jnp.sum(fight.astype(_I32)) - fight.astype(_I32)
    return quick_claw_active(tables, state, state.qc_roll) & fight & (others > 0)


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


def truant_loafs(state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``Battler_CheckTruant`` (battle_lib.c:3304-3313): ``side``'s battler has Truant
    (``Battler_Ability``) and ``moveEffectsData.truant != (totalTurns & 1)``. ``truant`` is
    ``(totalTurns + 1) & 1`` from the switch-in or faint that brought the battler in
    (battle_lib.c:2119, :2197), i.e. ``fake_out_turn & 1``, and 0 for a lead (the zeroed
    context, as ``fake_out_turn``); Transform's write (battle_script.c:5359) is unsupported.
    So a Truant battler loafs on every other turn from its entry, whatever it did."""
    mon = state.active[side]
    parity = (state.fake_out_turn[side].astype(_I32) & 1) != (state.turn.astype(_I32) & 1)
    return (state.ability[side, mon] == AbilityId.TRUANT) & parity
