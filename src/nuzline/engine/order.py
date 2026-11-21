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
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine import field as fcm
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    MOVE_ACTIONS,
    absent_now,
    choice_locked,
    gravity_banned,
    locked_slot,
    move_imprisoned,
    usable_moves,
)
from nuzline.engine.damage import apply_stage, simple_stage, weather_suppressed
from nuzline.engine.format import foe as _foe
from nuzline.engine.format import format_of, party, side_of
from nuzline.engine.ids import (
    CALL_GRAVITY,
    CANNOT_ENCORE_MOVES,
    GRAVITY_TURNS,
    INVOKER_MOVES,
    ITEM_TRANSFER_EFFECTS,
    ME_FIRST_BANNED_MOVES,
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
    MoveTarget,
    Stat,
    Status,
)
from nuzline.engine.state import BattleState, battler_ability, held_item

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
    for a switch or item action (callers mask it). A turn paused at a party list or a learn
    prompt keeps the Struggle and the lock fixed at the selection (bcp.c:335, :433; the scratch
    ``ts_struggling`` / ``ts_locked`` / ``ts_move``, as ``step._execute_actions`` reads them), so
    the laws see the move the step runs, while a free slot is re-read (FightCommand, bcp.c:
    1889-1895; review b2 #7)."""
    fmt = format_of(state)
    slot = jnp.clip(jnp.asarray(action, _I32), 0, MOVE_ACTIONS - 1)
    mon = state.active[side]
    paused = (state.pivot_battler != 0) | (state.learn_site != 0)
    struggling = jnp.where(
        paused, state.ts_struggling[side] != 0, ~jnp.any(usable_moves(tables, state, side))
    )
    move = jnp.where(
        struggling, MOVE_STRUGGLE, state.moves[party(fmt, side), mon, slot].astype(_I32)
    )
    locked = jnp.where(paused, state.ts_locked[side] != 0, locked_slot(state, side) >= 0)
    lock_move = jnp.where(paused, state.ts_move[side], state.locked_move[side]).astype(_I32)
    return jnp.where(locked, lock_move, move), struggling


def terrain_index(state: BattleState) -> jnp.ndarray:
    """``int32``: the row of the terrain tables (``ids.TERRAIN_MOVES``, ``TERRAIN_TYPES``) the
    battle reads: ``min(terrain, TERRAIN_SPECIAL)`` (``BtlCmd_GetTerrainMove``,
    battle_script.c:8040-8043; ``BtlCmd_TryCamouflage``, :8007-8010;
    ``BtlCmd_GetTerrainSecondaryEffect``, :8067-8070). A fight that reads it has a named terrain:
    ``TERRAIN_UNKNOWN`` (-1, the run named none) with a terrain move is refused before the engine
    (``coverage.terrain_refusal``, manager ruling 1; ledger G-21); the lower bound 0 only keeps the
    index in the tables."""
    return jnp.clip(jnp.asarray(state.terrain, _I32), 0, TERRAIN_SPECIAL)


def called_move(tables: Tables, state: BattleState, move) -> jnp.ndarray:
    """``int32``: the move ``move`` runs from its move script on (``battleCtx->moveCur``):
    for Nature Power (``EffectId.NATURE_POWER``), effect_script_0173.s's ``GetTerrainMove``
    (``msgMoveTemp = sTerrainMove[min(terrain, TERRAIN_SPECIAL)]``, battle_script.c:8036-8048)
    then ``GoToMoveScript`` (:2111-2134: ``moveCur = msgMoveTemp``, the defender re-picked
    for it), so the accuracy check, Protect, the type chart, the damage, the secondary
    effect and ``moveHit`` read the called move (``step._use_move``); otherwise ``move``
    itself. Nature Power stays the move for the status checks, the PP, Taunt, Disable,
    priority and ``movePrevByBattler`` (``moveTemp``, set at command selection,
    bcp.c:1861-1898, is never changed by ``GoToMoveScript``)."""
    move = jnp.asarray(move, _I32)
    terrain_move = jnp.asarray(TERRAIN_MOVES, _I32)[terrain_index(state)]
    nature_power = tables.move_effect[move] == EffectId.NATURE_POWER
    return jnp.where(nature_power, terrain_move, move)


# The invokers (ids.EffectId.SLEEP_TALK, METRONOME, COPYCAT): ``call_kind``.
CALL_NONE, CALL_SLEEP_TALK, CALL_METRONOME, CALL_COPYCAT = 0, 1, 2, 3
CALL_ASSIST = 4  # E2 batch 10, B10-11
CALL_ME_FIRST = 5  # E3-2-3: Me First, ``me_first_copy``; no draw, ordered like Copycat
CALL_MIRROR_MOVE = (
    6  # E3-B2-2: Mirror Move, ``mirror_call``; no draw in singles, ordered like Copycat
)


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
                jnp.where(
                    effect == EffectId.ASSIST,
                    CALL_ASSIST,
                    jnp.where(
                        effect == EffectId.ME_FIRST,
                        CALL_ME_FIRST,
                        jnp.where(effect == EffectId.MIRROR_MOVE, CALL_MIRROR_MOVE, CALL_NONE),
                    ),
                ),
            ),
        ),
    ).astype(_I32)


def assist_candidates(tables: Tables, state: BattleState, side) -> tuple:
    """``(moves int32[SLOTS * 4], valid bool[...])``: BtlCmd_TryAssist's list (battle_script.c:
    6418-6436) for ``side``'s active battler: in party order, every move of every other party
    slot holding a Pokemon (fainted ones too: only SPECIES_NONE and eggs are skipped) that is
    not an invoker (Move_IsInvoker, ``ids.INVOKER_MOVES``) and that Move_CanBeMetronomed allows
    (``MoveFlag.NO_METRONOME``, and ``actions.gravity_banned`` under Gravity; Heal Block, its other
    test, comes from an unsupported move). A
    move two mates know is listed twice. The call is ``RandNext() % numMoves`` (:6439):
    ``Outcome.call``'s uniform value, ``invoked_move`` taking the ``call % count``-th entry."""
    fmt = format_of(state)
    moves = state.moves[party(fmt, side)].astype(_I32)  # [SLOTS, 4]
    slots = jnp.arange(moves.shape[0])[:, None]
    mate = (slots != state.active[side]) & (state.species[party(fmt, side)][:, None] != 0)
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
    Torment (CHECK_INVALID_TORMENTED, :2275-2279: the slot holding ``last_move``, E3-2-1) and Heal Block
    come from unsupported moves; Gravity (``Move_FailsInHighGravity``,
    ``gravity_banned``, CHECK_INVALID_GRAVITY, :2292-2295: the opMask ``~CHECK_INVALID_NO_PP``
    keeps it) is read here (G-67)."""
    fmt = format_of(state)
    mon = state.active[side]
    moves = state.moves[party(fmt, side), mon].astype(_I32)
    ok = (moves > 0) & ((tables.move_flags[moves] & MoveFlag.NO_SLEEP_TALK) == 0)
    ok = ok & (moves != state.disabled_move[side])
    ok = ok & ~((state.taunt[side] > 0) & (tables.move_power[moves] == 0))
    ok = ok & ~((state.torment[side] > 0) & (moves == state.last_move[side]))  # :2275, E3-2-1
    ok = ok & ~move_imprisoned(state, side, moves)
    ok = ok & ~gravity_banned(state, moves)
    encore = state.encore_move[side].astype(_I32)
    ok = ok & ((encore == 0) | (moves == encore))
    lock = state.choice_lock[side].astype(_I32)
    return ok & (~choice_locked(tables, state, side) | (moves == lock))


def metronome_candidates(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[N]`` over move ids: what ``BtlCmd_Metronome`` (battle_script.c:4535-4562) can call
    for ``side``'s active battler: ids 1..``METRONOME_DRAW``, none of the battler's own moves
    (:4544-4551), none of ``MoveFlag.NO_METRONOME`` (``Move_CanBeMetronomed``, battle_lib.c:
    7204-7218), nor a move Gravity bans when it is up (``actions.gravity_banned``, :7198-7201;
    Heal Block, the other test there, comes from an unsupported move)."""
    fmt = format_of(state)
    ids = jnp.arange(tables.move_effect.shape[0], dtype=_I32)
    own = state.moves[party(fmt, side), state.active[side]].astype(_I32)
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


# The ranges whose defender stays the user (a Counter-family command re-picks it, so its special
# range is left out: it may be aimed at the foe, over-approximately): such a move does not write the
# foe's ``moveCopied``.
MIRROR_SELF_RANGES = (
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_USER_SIDE,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_ALLY,
    MoveTarget.RANGE_USER_OR_ALLY,
)


def mirror_call(tables: Tables, copied) -> jnp.ndarray:
    """``int32``: the move ``BtlCmd_SetMirrorMove`` (battle_script.c:3483-3525) calls with the
    user's ``moveCopied`` ``copied``, 0 when it fails: MOVE_NONE, or a move ``Move_CanBeEncored``
    refuses (battle_lib.c:7221-7231: the battle effect of a ``CANNOT_ENCORE_MOVES`` move, the
    table Mirror Move shares with Encore). A call is not Metronome-limited (an invoker is never
    recorded: no invoker has ``MoveFlag.MIRROR_MOVE``) and reads nothing of Gravity (E3-B2-2)."""
    copied = jnp.asarray(copied, _I32)
    banned = tables.move_ai_effect[jnp.asarray(CANNOT_ENCORE_MOVES, _I32)]
    refused = jnp.any(tables.move_ai_effect[copied][..., None] == banned, axis=-1)
    return jnp.where((copied != 0) & ~refused, copied, 0).astype(_I32)


def mirror_copied(state: BattleState, side, call) -> jnp.ndarray:
    """``int32``: ``moveCopied`` of ``side``'s battler as ``BtlCmd_SetMirrorMove`` reads it
    (battle_script.c:3490-3503): ``move_copied``, else in a doubles battle (more than two battlers)
    a random nonzero entry of its ``move_copied_hit`` row (``RandNext() % 4`` until nonzero:
    uniform over the nonzero entries, the ``call % count``-th of them), else none (E3-B2-2)."""
    fmt = format_of(state)
    own = state.move_copied[side].astype(_I32)
    if fmt.B <= 2:
        return own
    row = state.move_copied_hit[side].astype(_I32)
    nonzero = row != 0
    count = jnp.sum(nonzero.astype(_I32))
    pick = row[_nth(nonzero, jnp.asarray(call, _I32) % jnp.maximum(count, 1))]
    return jnp.where(own != 0, own, jnp.where(count > 0, pick, 0)).astype(_I32)


def me_first_copy(tables: Tables, state: BattleState, target_side, picked, fights) -> jnp.ndarray:
    """``int32``: the move ``BtlCmd_TryMeFirst`` (battle_script.c:7286-7312) copies from
    ``target_side``'s battler, 0 when it fails, for a target that has not acted yet (the caller's
    test: ``BATTLE_ACTION_PICK_COMMAND`` is not MOVE_END). The move is the target's encored move when
    it still holds it in its encored slot (:7293-7296), else ``Battler_SelectedMove`` (battle_lib.c:
    2858-2868): ``picked`` (its ``selected_move``) when it chose FIGHT (``fights``), else MOVE_NONE.
    It must not be struggling (the target's ``usable_moves`` leave no slot, bcp.c:433), differ in
    battle effect from every ``ME_FIRST_BANNED_MOVES`` move (``Move_MeFirstCanCopy``, battle_lib.c:
    7242-7252, ``tables.move_ai_effect``) and have power (:7300)."""
    fmt = format_of(state)
    mon = state.active[target_side]
    moves = state.moves[party(fmt, target_side), mon].astype(_I32)
    encored = state.encore_move[target_side].astype(_I32)
    held = (encored != 0) & (encored == moves[state.encore_slot[target_side]])
    move = jnp.where(held, encored, jnp.where(fights, jnp.asarray(picked, _I32), 0))
    banned = tables.move_ai_effect[jnp.asarray(ME_FIRST_BANNED_MOVES, _I32)]
    struggling = ~jnp.any(usable_moves(tables, state, target_side))
    ok = (
        ~struggling
        & ~jnp.any(tables.move_ai_effect[move] == banned)
        & (tables.move_power[move] != 0)
    )
    return jnp.where(ok, move, 0).astype(_I32)


def me_first_target(tables: Tables, state: BattleState, side, picked):
    """The battler whose chosen move ``side``'s Me First copies: the action's defender
    (``BtlCmd_TryMeFirst`` reads ``DEFENDER_ACTION`` and ``Battler_SelectedMove(defender)``,
    battle_script.c:7286-7312), ``targets.resolve``'s ``target_side`` for the move ``picked``. At
    B = 2 it is the foe (``_foe(side)``). At B = 4 the resolver's draw bit is 0 and the order the
    identity: the enumeration's stand-in (the draw and the order are the laws' D6-D7, as
    ``enumerate.call_laws`` reads the other call kinds)."""
    fmt = format_of(state)
    if fmt.B == 2:
        return _foe(side)
    from nuzline.engine import targets  # lazy: targets imports effects, which imports this module

    return targets.resolve(
        tables,
        state,
        side,
        picked,
        targets.fight_randomizes(state, side, False),
        state.chosen_target[side],
        jnp.arange(fmt.B),
        0,
    ).target_side


def invoked_move(
    tables: Tables, state: BattleState, side, move, call, move_prev=None, me_first=None, copied=None
):
    """``int32``: the move ``side``'s ``move`` runs through its invoker script, read with
    ``state`` as it is when the script runs; 0 when the invoker finds none. Sleep Talk: the
    allowed slot ``call % count`` (``Outcome.call``, ``ids.SLEEP_TALK_VALUES``); Metronome: the
    candidate of rank ``call % count`` (``metronome_candidates``; ``Outcome.call`` draws it by
    ``metronome_weights``); Copycat: ``copycat_call`` of ``move_prev`` (default
    ``state.move_prev``, the battle's movePrev when the script runs). Any other move: itself.
    Nature Power on the result is ``called_move``'s. A Gravity variant's ``CALL_GRAVITY`` offset
    is dropped (``call % CALL_GRAVITY``). Me First: ``me_first``, the caller's ``me_first_copy`` of
    its target (0 when none is given or the target has acted). Mirror Move: ``mirror_call`` of
    ``copied`` (default ``mirror_copied``, the user's moveCopied when the script runs)."""
    fmt = format_of(state)
    move = jnp.asarray(move, _I32)
    call = jnp.asarray(call, _I32) % CALL_GRAVITY
    kind = call_kind(tables, move)
    slots = sleep_talk_slots(tables, state, side)
    n_slots = jnp.sum(slots.astype(_I32))
    slot = _nth(slots, call % jnp.maximum(n_slots, 1))
    talk = jnp.where(
        n_slots > 0, state.moves[party(fmt, side), state.active[side], slot].astype(_I32), 0
    )
    cands = metronome_candidates(tables, state, side)
    n_cands = jnp.sum(cands.astype(_I32))
    metronome = jnp.where(n_cands > 0, _nth(cands, call % jnp.maximum(n_cands, 1)), 0)
    prev = state.move_prev if move_prev is None else move_prev
    copycat = copycat_call(tables, prev, state)
    listed, valid = assist_candidates(tables, state, side)
    n_listed = jnp.sum(valid.astype(_I32))
    assist = jnp.where(n_listed > 0, listed[_nth(valid, call % jnp.maximum(n_listed, 1))], 0)
    first = jnp.asarray(0 if me_first is None else me_first, _I32)
    held = mirror_copied(state, side, call) if copied is None else jnp.asarray(copied, _I32)
    mirror = mirror_call(tables, held)
    return jnp.where(
        kind == CALL_SLEEP_TALK,
        talk,
        jnp.where(
            kind == CALL_METRONOME,
            metronome,
            jnp.where(
                kind == CALL_COPYCAT,
                copycat,
                jnp.where(
                    kind == CALL_ASSIST,
                    assist,
                    jnp.where(
                        kind == CALL_ME_FIRST,
                        first,
                        jnp.where(kind == CALL_MIRROR_MOVE, mirror, move),
                    ),
                ),
            ),
        ),
    ).astype(_I32)


def encore_slot_of(tables: Tables, state: BattleState, target_side) -> tuple:
    """``(fails, slot)``: BtlCmd_TryEncore (battle_script.c:4725-4748) on ``target_side``'s
    battler: the slot of its movePrevByBattler (``last_move``; Battler_SlotForMove), and the
    failure (MOVE_STATUS_FAILED): it is encored already, its last move is none, not in its slots,
    one Move_CanBeEncored refuses (the battle effect of a ``CANNOT_ENCORE_MOVES`` move,
    battle_lib.c:7221-7231) or out of PP."""
    fmt = format_of(state)
    mon = state.active[target_side]
    last = state.last_move[target_side].astype(_I32)
    moves = state.moves[party(fmt, target_side), mon].astype(_I32)
    held = (moves == last) & (last != 0)
    slot = jnp.argmax(held).astype(_I32)
    banned = tables.move_ai_effect[jnp.asarray(CANNOT_ENCORE_MOVES, _I32)]
    refused = jnp.any(tables.move_ai_effect[last] == banned)
    fails = (
        (state.encore_move[target_side] != 0)
        | (last == 0)
        | ~jnp.any(held)
        | refused
        | (state.pp[party(fmt, target_side), mon, slot] == 0)
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
    b on side b in singles). A battler gets the bit when any battler of the other side is in
    ``sides`` (Move_Imprisoned's side test, battle_lib.c:3323-3324)."""
    fmt = format_of(state)
    sides = jnp.asarray(sides)
    foe_of = jnp.stack(
        [
            jnp.any(sides[np.asarray([x for x in range(fmt.B) if side_of(x) != side_of(b)])])
            for b in range(fmt.B)
        ]
    )
    value = jnp.where(foe_of, 1, state.imprison)
    return state._replace(imprison=value.astype(state.imprison.dtype))


def gravity_state(state: BattleState) -> BattleState:
    """``state`` with Gravity up: a faster foe's Gravity this turn (effect_script_0215.s,
    subscript_gravity_start.s), which then bans ``GRAVITY_MOVES`` from the invokers' calls
    (``Move_CanBeMetronomed``, battle_lib.c:7198-7201; Sleep Talk's CHECK_INVALID_GRAVITY,
    :2292-2295). Only the count is set: the call sets read nothing else of it (G-67)."""
    return fcm.with_gravity(state, GRAVITY_TURNS)


def gravity_possible(tables: Tables, state: BattleState, actions, side, foe) -> jnp.ndarray:
    """``bool``: ``foe``'s Gravity can land before ``side``'s invoker call this turn and change
    its set (G-67): Gravity is down at the turn start (effect_script_0215.s fails while it is
    up, and nothing ends it within a turn), ``side`` uses Sleep Talk, Metronome or Assist (the
    calls with a set; Copycat draws nothing, and under Gravity its one call either stays or
    fails, reading nothing) and ``foe`` uses a move that can run Gravity (``can_call_effect``).
    ``foe`` is named by the caller (doubles rule 1). Sound, not tight: the call laws drop the
    rows where it did not happen (``enumerate.call_consistent``)."""
    actions = jnp.asarray(actions, _I32)
    uses_move = (state.over == 0) & ~is_switch(actions) & ~is_item(actions)
    chosen = selected_move(tables, state, side, actions[side])[0]
    kind = call_kind(tables, chosen)
    calls = (kind == CALL_SLEEP_TALK) | (kind == CALL_METRONOME) | (kind == CALL_ASSIST)
    foe_chosen = selected_move(tables, state, foe, actions[foe])[0]
    foe_can = uses_move[foe] & can_call_effect(tables, state, foe, foe_chosen, EffectId.GRAVITY)
    return ~fcm.gravity_on(state) & uses_move[side] & calls & foe_can


def encored_state(tables: Tables, state: BattleState, sides) -> BattleState:
    """``state`` with each battler of ``sides`` (``bool[2]``) encored into its last move
    (encoredMove = movePrevByBattler, encoredMoveSlot its slot: BtlCmd_TryEncore, battle_script.c:
    4740-4741), the hypothetical state of a takeover variant (``takeover_possible``; ruling B11-1).
    The count is left as it is: nothing before the turn's end reads it."""
    fmt = format_of(state)
    sides = jnp.asarray(sides, bool)
    slots = jnp.stack([encore_slot_of(tables, state, side)[1] for side in range(fmt.B)])
    move = jnp.where(sides, state.last_move, state.encore_move).astype(state.encore_move.dtype)
    slot = jnp.where(sides, slots, state.encore_slot).astype(state.encore_slot.dtype)
    return state._replace(encore_move=move, encore_slot=slot)


def can_call_effect(tables: Tables, state: BattleState, side, move, wanted) -> jnp.ndarray:
    """``bool``: ``side``'s move ``move`` can run a move of effect ``wanted`` before its foe's
    action: the effect itself, or an invoker whose call can be one (``can_call_encore``'s rules).
    Over-approximate."""
    fmt = format_of(state)
    move = jnp.asarray(move, _I32)
    effect = tables.move_effect
    kind = call_kind(tables, move)
    own = state.moves[party(fmt, side), state.active[side]].astype(_I32)
    talk = jnp.any((own != 0) & (effect[own] == wanted))
    ids = jnp.arange(effect.shape[0], dtype=_I32)
    metronome = jnp.any(metronome_candidates(tables, state, side) & (effect[ids] == wanted))
    listed, valid = assist_candidates(tables, state, side)
    assist = jnp.any(valid & (effect[listed] == wanted))
    copycat = effect[copycat_call(tables, state.move_prev, state)] == wanted
    mirror = effect[mirror_call(tables, mirror_copied(state, side, 0))] == wanted
    return (
        (effect[move] == wanted)
        | ((kind == CALL_SLEEP_TALK) & talk)
        | ((kind == CALL_METRONOME) & metronome)
        | ((kind == CALL_ASSIST) & assist)
        | ((kind == CALL_COPYCAT) & copycat)
        | ((kind == CALL_MIRROR_MOVE) & mirror)
    )


def can_call_encore(tables: Tables, state: BattleState, side, move) -> jnp.ndarray:
    """``bool``: ``side``'s move ``move`` can run Encore before its foe's action (ruling B11-1):
    Encore itself, or an invoker whose call can be Encore: Sleep Talk when the user knows it
    (BtlCmd_TrySleepTalk draws its own slots, battle_script.c:4907-4937), Metronome when it is a
    candidate (``metronome_candidates``: not NO_METRONOME, not a move the user knows), Assist when
    a listed mate's move is (``assist_candidates``), Copycat when the battle's movePrev is Encore
    (``copycat_call``: its foe's move of this turn comes after the foe's action, too late to
    replace it). Over-approximate (Sleep Talk's sleep and set are the turn's)."""
    return can_call_effect(tables, state, side, move, EffectId.ENCORE)


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
    reads. ``move`` may be batched."""
    move = jnp.asarray(move, _I32)
    effect = tables.move_effect[move].astype(_I32)
    param = tables.move_param[move].astype(_I32)
    row = terrain_index(state)
    sp_effect = jnp.asarray([e for e, _ in SECRET_POWER_EFFECTS], _I32)[row]
    sp_param = jnp.asarray([p for _, p in SECRET_POWER_EFFECTS], _I32)[row]
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
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if chosen is None:
        chosen = jnp.stack(
            [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
        )
    chosen = jnp.asarray(chosen, _I32)
    after_foe = jnp.asarray(after_foe, bool)
    call = jnp.asarray(call, _I32)
    out = []
    for side in range(fmt.B):
        foe = _foe(side)
        prev = jnp.where(after_foe[side], chosen[foe], state.move_prev.astype(_I32))
        # Me First copies its action's defender's move, a target that has not acted: ``after_foe``
        # says it has (E3-2-3; D3-5: the defender, the foe at B = 2).
        target = me_first_target(tables, state, side, chosen[side])
        fights = (state.over == 0) & ~is_switch(actions[target]) & ~is_item(actions[target])
        first = jnp.where(
            after_foe[side], 0, me_first_copy(tables, state, target, chosen[target], fights)
        )
        # Mirror Move copies what the foe's move this turn recorded for its user (UpdateFlagsWhenHit,
        # bcp.c:4480-4486): the foe's chosen move when it has MoveFlag.MIRROR_MOVE and is aimed at
        # the user, else the turn-start ``move_copied`` (E3-B2-2).
        flagged = (tables.move_flags[chosen[foe]] & MoveFlag.MIRROR_MOVE) != 0
        own_range = jnp.isin(tables.move_target[chosen[foe]], jnp.asarray(MIRROR_SELF_RANGES, _I32))
        aimed = flagged & ~own_range
        held = jnp.where(
            after_foe[side] & aimed, chosen[foe], mirror_copied(state, side, call[side])
        )
        invoked = invoked_move(
            tables, state, side, chosen[side], call[side], prev, first, copied=held
        )
        run = jnp.where(invoked == 0, chosen[side], called_move(tables, state, invoked))
        # A Magic Coat user acting before a foe whose move it can reflect runs that move on the foe
        # (``step._execute_actions``' reflect phase draws it from the holder's own slots): its
        # accuracy, crit odds, duration and secondary roll are the reflected move's (E3-B2-6;
        # over-approximate where the Magic Coat fails: the draw is then unread).
        coat = (tables.move_effect[chosen[side]] == EffectId.MAGIC_COAT) & fights & ~after_foe[side]
        reflected = coat & ((tables.move_flags[chosen[foe]] & MoveFlag.MAGIC_COAT) != 0)
        out.append(jnp.where(reflected, chosen[foe], run).astype(_I32))
    return jnp.stack(out)


def effective_speed(
    tables: Tables, state: BattleState, side, suppressed=None, weather=None
) -> jnp.ndarray:
    """``int32`` speed as ``BattleSystem_CompareBattlerSpeed`` computes it. ``suppressed``
    (``bool``, default ``damage.weather_suppressed`` of ``state``) overrides the weather
    lock (``NO_CLOUD_NINE``) and ``weather`` (weather bits, default ``field.weather_bits`` of
    ``state``) the weather, for a lock or a weather that can change before the speed is read
    (``enumerate._end_speeds``)."""
    fmt = format_of(state)
    if suppressed is None:
        suppressed = weather_suppressed(tables, state)
    weather = fcm.weather_bits(state) if weather is None else weather
    mon = state.active[side]
    ability = battler_ability(state, side)  # CompareBattlerSpeed (battle_lib.c:1213-1214)
    stage = state.stages[side, Stat.SPE].astype(_I32)
    stage = jnp.where(ability == AbilityId.SIMPLE, simple_stage(stage), stage)
    speed = apply_stage(state.stats[party(fmt, side), mon, Stat.SPE - 1].astype(_I32), stage)
    # :1228-1238 Swift Swim in rain, Chlorophyll in sun (Battler_Ability, WEATHER_IS_RAIN /
    # _SUN of the mask), inside NO_CLOUD_NINE.
    swim = (
        ((ability == AbilityId.SWIFT_SWIM) & fcm.rain(weather))
        | ((ability == AbilityId.CHLOROPHYLL) & fcm.sun(weather))
    ) & ~jnp.asarray(suppressed, bool)
    speed = jnp.where(swim, speed * 2, speed)
    # :1240-1246 sSpeedHalvingItemEffects (:1160-1169: Iron Ball, Macho Brace, the Power items),
    # read through BattleSystem_GetItemData on the raw heldItem (no Klutz / Embargo: "not
    # ignored by any negation effect", :1241), G-56, B13-29.
    hold = tables.item_hold_effect[state.item[party(fmt, side), mon]].astype(_I32)
    halved = jnp.isin(hold, jnp.asarray(SPEED_HALVING_HOLD_EFFECTS, _I32))
    speed = jnp.where(halved, speed // 2, speed)
    # :1248 Choice Scarf (Battler_HeldItemEffect, :1215: ``held_item``, B13-29), after the
    # speed-halving items and before paralysis.
    scarf = tables.item_effect[held_item(state, side, mon)] == ItemId.CHOICE_SCARF
    speed = jnp.where(scarf, speed * 15 // 10, speed)
    # :1257-1261 Quick Feet (Battler_Ability) with any status (MON_CONDITION_ANY): x15/10,
    # and then no paralysis quartering; else paralysis quarters it.
    status = state.status[party(fmt, side), mon]
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
        & (state.item[party(fmt, side), mon] == 0)
    )
    speed = jnp.where(unburdened, speed * 2, speed)
    return jnp.where(state.tailwind[side_of(side)] != 0, speed * 2, speed)


def record_speeds(tables: Tables, state: BattleState, cond=True) -> BattleState:
    """``state`` with ``speed_rec`` (``BattleContext.monSpeedValues``) written as a
    ``SortMonSpeedOrder`` writes it where ``cond`` (E3-B2-15): ``BattleSystem_CompareBattlerSpeed``
    stores both battlers' speeds (``effective_speed``, battle_lib.c:1372-1373) unless either has no
    HP (:1204-1211), and the sort compares every pair of battlers (:3364-3383), so a battler's
    value is written when it is alive and another battler is (doubles section 5 rule 4: a loop,
    not "me and the foe"). A fainted battler's slot keeps its last value. Called at the battle
    start after the switch-in effects (bcp.c:263), before each action and after the last one of
    a turn (BranchActions, bcp.c:888)."""
    fmt = format_of(state)
    sides = list(range(fmt.B))
    alive = jnp.stack([state.hp[party(fmt, b), state.active[b]] > 0 for b in sides])
    speeds = jnp.stack([effective_speed(tables, state, b) for b in sides])
    write = jnp.asarray(cond, bool) & alive & (jnp.sum(alive.astype(_I32)) >= 2)
    return state._replace(
        speed_rec=jnp.where(write, speeds, state.speed_rec).astype(state.speed_rec.dtype)
    )


def _holds(tables: Tables, state: BattleState, side, item) -> jnp.ndarray:
    """Battler_HeldItemEffect (battle_lib.c:1215, :1217: ``held_item``, B13-29) is ``item``."""
    return tables.item_effect[held_item(state, side)] == item


def pinch_divisor(tables: Tables, state: BattleState, side, mon, item) -> jnp.ndarray:
    """``int32``: the hold param of a pinch berry or Custap Berry ``item`` held by ``side``'s
    party ``mon``, halved (``itemPower /= 2``) when the holder's own ability is Gluttony
    (``Battler_Ability``: battle_lib.c:4774-4776, :1289-1291). The berry triggers at
    ``curHP <= maxHP / divisor`` (``pinch_limit``)."""
    power = tables.item_param[item, 0].astype(_I32)
    gluttony = battler_ability(state, side, mon) == AbilityId.GLUTTONY
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
    fmt = format_of(state)
    disliked = jnp.asarray(NATURE_DISLIKED_FLAVOR, _I32)[
        state.nature[party(fmt, side), mon].astype(_I32)
    ]
    flavor = tables.item_param[item, 1].astype(_I32)
    return (tables.item_effect[item] == ItemId.FLAVOR_BERRY) & (disliked == flavor)


def custap_active(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[2]``: each side's active battler holds a Custap Berry
    (``HOLD_EFFECT_PINCH_PRIORITY``) at ``curHP <= pinch_limit(maxHP, pinch_divisor)``: the decomp's
    ``battlerNQuickClaw`` flag (battle_lib.c:1288-1300, :1354-1366), whatever
    ``ignoreQuickClaw``."""
    fmt = format_of(state)
    out = []
    for side in range(fmt.B):
        mon = state.active[side]
        item = held_item(state, side, mon)  # battler1ItemEffect / _ItemParam (:1215-1218)
        hp = state.hp[party(fmt, side), mon].astype(_I32)
        max_hp = state.max_hp[party(fmt, side), mon].astype(_I32)
        low = hp <= pinch_limit(max_hp, pinch_divisor(tables, state, side, mon, item))
        out.append((tables.item_effect[item] == ItemId.CUSTAP_BERRY) & low)
    return jnp.stack(out)


def _alive_battlers(state: BattleState) -> jnp.ndarray:
    """``bool[B]``: each battler's active Pokemon has HP (``curHP`` of lib.c:1204-1211)."""
    fmt = format_of(state)
    return jnp.stack([state.hp[party(fmt, b), state.active[b]] > 0 for b in range(fmt.B)])


def _compared_fighters(state: BattleState, actions) -> jnp.ndarray:
    """``bool[B]`` at B = 4: the battlers whose claw / pinch flag ``CalcTurnOrder`` writes
    (rule 9, sheet D3-3): a FIGHT compare (``ignoreQuickClaw == FALSE``, bcp.c:773-786) with both
    battlers alive has the battler on one side. Absent battlers are FIGHT entries that lose every
    compare against a living one (lib.c:1204-1211), so they are never the other side of a write.
    Every living FIGHT battler meets another living FIGHT battler in the position loop whenever
    there is one (swaps stay inside a command group, the pairs cover every position pair, a
    dead battler sinks past a living one): the flag is the closed form, whatever the tie bits
    (checked over all 16 x 81 x 64 cases of alive masks, commands and swap results)."""
    alive = _alive_battlers(state)
    fight = (~is_switch(actions) & ~is_item(actions)) | ~alive  # an absent battler is FIGHT
    eligible = fight & alive
    others = jnp.sum(eligible.astype(_I32)) - eligible.astype(_I32)
    return eligible & (others > 0)


def custap_flags(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[B]``: ``moveEffectsData.custapBerry`` after the turn order: ``CalcTurnOrder``
    compares two FIGHT commands with ``ignoreQuickClaw = FALSE`` (bcp.c:772-786), which writes
    the flag of each activated holder (``custap_active`` on the turn-start ``state``,
    battle_lib.c:1296-1298, :1362-1364). The holder eats the berry when its move starts
    (``subscript_check_quick_claw.s:26-28``, ``step._use_move``); until then Pluck cannot take
    it (``hooks.pluck_berry``). ``BattleSystem_SortMonActionOrder`` (battle_lib.c:7276-7300)
    writes it too, through the same compare with ``ignoreQuickClaw = FALSE``, when a Trick
    Room activation re-sorts the turn (bcp.c:3990-3993): ``resort_action_order`` (at B = 2
    ``resort_claw_flags``, its flag part) re-sets it there."""
    actions = jnp.asarray(actions, _I32)
    if format_of(state).B == 4:
        return custap_active(tables, state) & _compared_fighters(state, actions)
    fight = ~is_switch(actions) & ~is_item(actions)
    return custap_active(tables, state) & fight[0] & fight[1]


def claw_flags(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``bool[B]``: ``moveEffectsData.quickClaw`` after the turn order: ``CalcTurnOrder``'s
    FIGHT-against-FIGHT compare (``ignoreQuickClaw = FALSE``, bcp.c:772-786) sets it for each
    activated Quick Claw (battle_lib.c:1283, :1349; ``quick_claw_active`` on the turn's
    ``qc_roll``). The holder's own BEFORE_MOVE_STATE_QUICK_CLAW clears it
    (subscript_check_quick_claw.s:22-23), so it stands until the holder's move starts; only
    Knock Off reads it (subscript_knock_off.s:8), B10-6. The sort compares every pair of
    battlers whose commands are both FIGHT (bcp.c:766-778), so a FIGHT battler is compared when
    any other battler fights (doubles design section 5 rule 4). A Trick Room's MoveEnd re-sort
    sets it again (``resort_action_order``; at B = 2 ``resort_claw_flags``)."""
    actions = jnp.asarray(actions, _I32)
    if format_of(state).B == 4:
        return quick_claw_active(tables, state, state.qc_roll) & _compared_fighters(state, actions)
    fight = ~is_switch(actions) & ~is_item(actions)
    others = jnp.sum(fight.astype(_I32)) - fight.astype(_I32)
    return quick_claw_active(tables, state, state.qc_roll) & fight & (others > 0)


def resort_claw_flags(tables: Tables, state: BattleState, actions, qc):
    """``(bool[B] claw_set, bool[B] custap_set)``: what ``BattleSystem_SortMonActionOrder``
    (battle_lib.c:7276-7300) sets at a Trick Room's MoveEnd (bcp.c:3991): every pair with equal
    ``SELECTED_COMMAND`` is compared, a FIGHT pair with both alive with ``ignoreQuickClaw =
    FALSE`` (:7289-7293; a dead battler short-circuits, :1204-1211), which sets
    ``moveEffectsData.quickClaw`` for an activated Quick Claw (:1283, :1349; this turn's
    ``speedRand``, ``qc`` = ``Outcome.quick_claw``) and ``.custapBerry`` for a Custap holder in
    its pinch on the current HP (:1297, :1363). It never clears. ``state`` is the state at the
    MoveEnd, ``actions`` the turn's commands."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    fight = ~is_switch(actions) & ~is_item(actions)
    alive = jnp.stack([state.hp[party(fmt, b), state.active[b]] > 0 for b in range(fmt.B)])
    eligible = fight & alive
    others = jnp.sum(eligible.astype(_I32)) - eligible.astype(_I32)
    pair = eligible & (others > 0)
    return (
        quick_claw_active(tables, state, qc) & pair,
        custap_active(tables, state) & pair,
    )


def resort_action_order(tables: Tables, state: BattleState, order, real, qc, ties, struggling=None):
    """``(order int32[B], claw_set bool[B], custap_set bool[B], read bool[6])``:
    ``BattleSystem_SortMonActionOrder`` (battle_lib.c:7276-7300) at B = 4, which a Trick Room's
    MoveEnd runs (battle_controller_player.c:3989-3991; D3-18). Over the CURRENT ``order`` (no
    regrouping into ITEM / PARTY first, unlike CalcTurnOrder), for each position pair of ``PAIRS``
    (i < j, :7283-7286) the battlers now at the two positions are compared only when their
    SELECTED_COMMAND is equal (:7288; ITEM / PARTY / FIGHT from ``real``; a battler absent now
    (``actions.absent_now``) is FIGHT, D3-3 rule 7), with ``ignoreQuickClaw = command != FIGHT``
    (:7289-7293), and swapped on a result other than FASTER (:7295-7298). Every battler takes part,
    MOVE_END or not. The compare (``compare_full``) reads ``state``, the state at the MoveEnd:
    current speeds and Trick Room, Quick Claw from ``qc`` (``Outcome.quick_claw``: this turn's
    ``speedRand``), Custap on the current HP, a FIGHT pair's priorities from the slot's move now
    (``moveSlot``, ``BattleMon_Get(MOVE_1 + slot)``; Struggle where the turn's ``struggling`` says
    so: lib.c:1375-1399). ``ties`` is ``Outcome.resort_tie[battler whose Trick Room it is]``;
    ``read[p]`` whether pair ``p``'s compare drew its bit. ``claw_set`` / ``custap_set`` are
    ``resort_claw_flags``: what the FIGHT-against-FIGHT compares with both alive write (they only
    set)."""
    fmt = format_of(state)
    real = jnp.asarray(real, _I32)
    switch, item = is_switch(real), is_item(real)
    absent = jnp.stack([absent_now(state, b) for b in range(fmt.B)])
    command = jnp.where(absent | (~switch & ~item), 0, jnp.where(item, 1, 2)).astype(_I32)
    slot = jnp.clip(real, 0, MOVE_ACTIONS - 1)
    moves = jnp.stack(
        [state.moves[party(fmt, b), state.active[b], slot[b]].astype(_I32) for b in range(fmt.B)]
    )
    if struggling is not None:
        moves = jnp.where(jnp.asarray(struggling, bool), MOVE_STRUGGLE, moves)
    priority = tables.move_priority[moves].astype(_I32)
    facts = _pair_facts(tables, state, qc)
    order = jnp.asarray(order, _I32)
    reads = []
    for p, (i, j) in enumerate(PAIRS):
        b1, b2 = order[i], order[j]
        same = command[b1] == command[b2]
        skip = command[b1] != 0
        p1 = jnp.where(skip, 0, priority[b1])
        p2 = jnp.where(skip, 0, priority[b2])
        result, read = compare_full(tables, state, b1, b2, ties[p], qc, p1, p2, facts)
        swap = same & (result != COMPARE_SPEED_FASTER)
        order = order.at[i].set(jnp.where(swap, b2, b1)).at[j].set(jnp.where(swap, b1, b2))
        reads.append(same & read)
    claw_set, custap_set = resort_claw_flags(tables, state, real, qc)
    return order, claw_set, custap_set, jnp.stack(reads)


def quick_claw_active(tables: Tables, state: BattleState, qc) -> jnp.ndarray:
    """``bool[2]``: each side's active battler holds a Quick Claw and its ``speedRand``
    draw ``qc`` (``int[2]``, 1 = ``speedRand % 5 == 0``) activates it (lib.c:1278-1286)."""
    fmt = format_of(state)
    qc = jnp.asarray(qc)
    return jnp.stack(
        [_holds(tables, state, side, ItemId.QUICK_CLAW) & (qc[side] != 0) for side in range(fmt.B)]
    )


def qc_transfer(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool``: a held item can move onto another battler in this turn: a party Pokemon (either
    side) knows a move of an ``ITEM_TRANSFER_EFFECTS`` battle effect. Thief / Covet
    (``BtlCmd_TryStealItem``, battle_script.c:5093-5126) and Trick / Switcheroo
    (``BtlCmd_TrySwapItems``, :6336-6360) move it between the attacker and the defender, both on
    the field; the defender may be the attacker's ally (another party in a tag battle): only a
    side-1 attacker is blocked (:5102-5104, :6345-6346). The other paths that touch a held item
    give no battler another's: Knock Off removes one (:6620-6628), ``BtlCmd_RemoveItem``
    consumes the battler's own into its ``recycleItem`` (:8972-8979; Fling's
    ``RemoveItem BTLSCR_ATTACKER``, effect_script_0233.s:18, Natural Gift, the eaten berries, the
    defender's berry Bug Bite / Pluck eat, subscript_pluck_check.s:6), Sticky Barb moves only
    itself (battle_lib.c:5385-5396, :7377-7384), Pickup's item comes after the battle
    (``BtlCmd_GenerateEndOfBattleItem``, battle_script.c:7781-7797). Recycle gives the attacker
    back its own ``recycleItem`` (:9000-9012), a flung Quick Claw included: ``qc_recycle``. A
    battler uses Thief, Covet, Trick or Switcheroo only out of a moveset: Metronome and Assist
    never call them (``sCannotMetronomeMoves``, battle_lib.c:7152-7179, read by
    ``Move_CanBeMetronomed`` :7194-7210 in ``BtlCmd_Metronome``, battle_script.c:4554, and
    ``BtlCmd_TryAssist``, :6430); Mirror Move, Me First, Mimic, Sleep Talk and Transform take a
    move a Pokemon of the battle knows."""
    effect = tables.move_ai_effect[state.moves]  # [P, 6, 4]
    known = (state.moves != 0) & (state.species[..., None] != 0)
    hit = jnp.zeros(effect.shape, bool)
    for e in ITEM_TRANSFER_EFFECTS:
        hit = hit | (effect == e)
    return jnp.any(known & hit)


def qc_recycle(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[B]``: the battler's ``recycleItem`` is a Quick Claw (a flung one: Fling's
    ``RemoveItem BTLSCR_ATTACKER``, effect_script_0233.s:18, stores it, battle_script.c:8978), so
    a Recycle this turn can put it back on the battler (``BtlCmd_TryRecycle``, :9000-9012;
    effect_script_0184.s:5-7) before the turn end's speed sort. ``recycleItem`` is per battler
    and outlives a switch (nothing else writes it), and Recycle needs no moveset: Metronome can
    call it (it is not in ``sCannotMetronomeMoves``, battle_lib.c:7152-7179), as can Assist,
    Sleep Talk, Mimic, Sketch, Copycat and Mirror Move; so no move test (over-keeping only
    splits states). The raw item, as ``qc_kept``."""
    return tables.item_effect[state.recycle_item] == ItemId.QUICK_CLAW


# BattleSystem_CompareBattlerSpeed's results (include/constants/battle.h:163-165).
COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER, COMPARE_SPEED_TIE = 0, 1, 2

# CalcTurnOrder's position pairs (bcp.c:773-774) and their count per battler count: the length of
# ``Outcome.order_tie`` (``TIE_PAIRS[2] = 0``: at B = 2 the one pair's bit is ``speed_tie``).
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TIE_PAIRS = {2: 0, 4: 6}


def compare_speed(
    tables: Tables, state: BattleState, battler1: int, tie_roll, qc, battler2: int | None = None
) -> jnp.ndarray:
    """``int32``: ``BattleSystem_CompareBattlerSpeed(battler1, battler2)`` within equal
    priority (lib.c:1401-1447): ``COMPARE_SPEED_FASTER``, ``SLOWER`` or ``TIE``.
    ``battler1`` and ``battler2`` are battlers (static ints); ``battler2`` defaults to
    ``foe(battler1)``. Every battler's speed, claw and lag are built (``range(fmt.B)``), then
    the pair is read. Quick Claw first (``qc``: each side's ``speedRand``
    bit, ``quick_claw_active``): both active compare plain speeds, Trick Room and Lagging
    Tail ignored (:1402-1407); a lone active one is FASTER (:1408-1411), even under Trick
    Room and against a Lagging Tail. Lagging Tail: a lone holder is SLOWER (:1418-1421); two
    holders compare speeds reversed, the faster SLOWER, Trick Room ignored (:1412-1417). Else
    (Stall is unsupported) Trick Room reverses the speeds (:1432-1439). At equal speed each
    of these branches gives TIE when ``tie_roll`` (``RandNext() & 1``) is 1, else FASTER,
    never SLOWER; the decomp draws that bit only there. Before all of it the dead-battler
    short-circuit (:1204-1211): ``battler1`` with no HP against a ``battler2`` with HP is
    SLOWER, the reverse FASTER, so a fainted battler is last in the sort after the last action
    (``BattleSystem_SortMonSpeedOrder``, :3364-3381; ``step.end_of_turn_order``), whatever its
    Quick Claw or speed. A Custap Berry in its pinch (``custap_active``) sets the same flag as
    an activated Quick Claw (:1288-1300)."""
    fmt = format_of(state)
    other = _foe(battler1) if battler2 is None else battler2
    speed = tuple(effective_speed(tables, state, b) for b in range(fmt.B))
    claw = quick_claw_active(tables, state, qc) | custap_active(tables, state)
    lag = tuple(_holds(tables, state, side, ItemId.LAGGING_TAIL) for side in range(fmt.B))
    speed1, speed2 = speed[battler1], speed[other]
    claw1, claw2 = claw[battler1], claw[other]
    lag1, lag2 = lag[battler1], lag[other]
    reversed_ = ~(claw1 & claw2) & ((lag1 & lag2) | (fcm.trick_room_on(state) & ~lag1 & ~lag2))
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
    alive1 = state.hp[party(fmt, battler1), state.active[battler1]] > 0
    alive2 = state.hp[party(fmt, other), state.active[other]] > 0
    result = jnp.where(
        alive1 == alive2, result, jnp.where(alive1, COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER)
    )
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


def _pair_facts(tables: Tables, state: BattleState, qc):
    """``(speed int32[B], claw bool[B], lag bool[B])``: each battler's speed as
    ``CompareBattlerSpeed`` computes it, its Quick Claw / Custap activation (``qc`` the
    ``speedRand`` bits) and Lagging Tail, built once for every pair of a sort."""
    fmt = format_of(state)
    speed = jnp.stack([effective_speed(tables, state, b) for b in range(fmt.B)])
    claw = quick_claw_active(tables, state, qc) | custap_active(tables, state)
    lag = jnp.stack([_holds(tables, state, b, ItemId.LAGGING_TAIL) for b in range(fmt.B)])
    return speed, claw, lag


def compare_full(
    tables: Tables,
    state: BattleState,
    b1,
    b2,
    tie_bit,
    qc,
    priority1=0,
    priority2=0,
    facts=None,
):
    """``(result int32, read bool)``: ``BattleSystem_CompareBattlerSpeed(b1, b2)`` (lib.c:1188-1454)
    with the dead-battler short-circuit (:1204-1211: ``b1`` dead and ``b2`` alive is SLOWER,
    ``b1`` alive and ``b2`` dead FASTER, nothing computed; two dead go through the compare).
    Within equal priority the ladder is ``compare_speed``'s (Quick Claw / Custap, Lagging Tail,
    Trick Room, speed; Stall is unsupported); a lower priority is SLOWER (:1448-1449). ``b1`` and
    ``b2`` may be traced battler indices; ``priority1`` / ``priority2`` are the moves' priorities
    (0 both where the caller skips priority). ``tie_bit`` is the pair's ``RandNext() & 1``;
    ``read`` is true where the decomp draws it: both battlers alive, equal priority, a branch that
    compares speeds (not a lone Quick Claw or Lagging Tail holder) at equal speed. ``facts``:
    ``_pair_facts`` if the caller has it."""
    speed, claw, lag = _pair_facts(tables, state, qc) if facts is None else facts
    alive = _alive_battlers(state)
    alive1, alive2 = alive[b1], alive[b2]
    speed1, speed2 = speed[b1], speed[b2]
    claw1, claw2 = claw[b1], claw[b2]
    lag1, lag2 = lag[b1], lag[b2]
    reversed_ = ~(claw1 & claw2) & ((lag1 & lag2) | (fcm.trick_room_on(state) & ~lag1 & ~lag2))
    slower = jnp.where(reversed_, speed1 > speed2, speed1 < speed2)
    tie = (speed1 == speed2) & (jnp.asarray(tie_bit) != 0)
    by_speed = jnp.where(
        slower, COMPARE_SPEED_SLOWER, jnp.where(tie, COMPARE_SPEED_TIE, COMPARE_SPEED_FASTER)
    )
    lagging = jnp.where(
        lag1 != lag2, jnp.where(lag1, COMPARE_SPEED_SLOWER, COMPARE_SPEED_FASTER), by_speed
    )
    lone_claw = jnp.where(claw2, COMPARE_SPEED_SLOWER, COMPARE_SPEED_FASTER)
    ladder = jnp.where(claw1 != claw2, lone_claw, jnp.where(claw1, by_speed, lagging))
    p1, p2 = jnp.asarray(priority1, _I32), jnp.asarray(priority2, _I32)
    result = jnp.where(
        p1 == p2,
        ladder,
        jnp.where(p1 < p2, COMPARE_SPEED_SLOWER, COMPARE_SPEED_FASTER),
    )
    result = jnp.where(
        alive1 != alive2,
        jnp.where(alive1, COMPARE_SPEED_FASTER, COMPARE_SPEED_SLOWER),
        result,
    ).astype(_I32)
    compares = (claw1 == claw2) & (claw1 | (lag1 == lag2))
    # Two fainted entries: the decomp falls through to the compare and draws the tie bit
    # (battle_lib.c:1204-1211 short-circuits only one dead; :1440-1446 draws), but the order of
    # two fainted entries is never read by any later step, so not drawing it changes no result.
    read = alive1 & alive2 & (p1 == p2) & compares & (speed1 == speed2)
    return result, read


def _four_order(tables: Tables, state: BattleState, actions, order_tie, qc):
    """``(order int8[4], read bool[6])``: ``BattleControllerPlayer_CalcTurnOrder`` over four
    battlers (bcp.c:754-786, rules 2, 3 and 7 of sheet D3-3). ITEM and PARTY commands first, then
    the rest, each group in battler order (absent battlers are FIGHT entries); then for each
    position pair of ``PAIRS`` (lexicographic, :773-774) the battlers *currently* at the two
    positions, when their commands are equal, are compared (``compare_full``; priority only for
    FIGHT, :779-780) and swapped on a result other than FASTER. ``order_tie[p]`` is pair ``p``'s
    bit; ``read[p]`` whether that compare drew it."""
    alive = _alive_battlers(state)
    actions = jnp.asarray(actions, _I32)
    switch, item = is_switch(actions), is_item(actions)
    fight = (~switch & ~item) | ~alive
    command = jnp.where(fight, 0, jnp.where(item, 1, 2)).astype(_I32)
    moves = jnp.stack([selected_move(tables, state, b, actions[b])[0] for b in range(4)])
    priority = tables.move_priority[moves].astype(_I32)
    facts = _pair_facts(tables, state, qc)
    # Stable sort on "is FIGHT": ITEM / PARTY battlers first, then the rest, each in battler order.
    order = jnp.argsort(fight.astype(_I32), stable=True).astype(_I32)
    reads = []
    for p, (i, j) in enumerate(PAIRS):
        b1, b2 = order[i], order[j]
        same = command[b1] == command[b2]
        skip = command[b1] != 0
        p1 = jnp.where(skip, 0, priority[b1])
        p2 = jnp.where(skip, 0, priority[b2])
        result, read = compare_full(tables, state, b1, b2, order_tie[p], qc, p1, p2, facts)
        swap = same & (result != COMPARE_SPEED_FASTER)
        order = order.at[i].set(jnp.where(swap, b2, b1)).at[j].set(jnp.where(swap, b1, b2))
        reads.append(same & read)
    return order.astype(jnp.int8), jnp.stack(reads)


def sort_mon_speed(tables: Tables, state: BattleState, qc, ties):
    """``(order int32[B], read bool[len(PAIRS)])``: ``BattleSystem_SortMonSpeedOrder``
    (lib.c:3364-3383) at B = 4, the sort before an action (bcp.c:888): ``order[i] = i``, then for
    each position pair of ``PAIRS`` the battlers currently at the two positions are compared with
    ``ignoreQuickClaw`` TRUE (``compare_full``, both priorities 0, the claw bits of ``qc`` and
    Custap's pinch on the current HP still deciding) and swapped on a result other than FASTER.
    Every pair is compared whatever the commands. ``ties``: one row of ``Outcome.sort_tie`` (the
    acting battler's, ledger D3-4); ``read[p]``: whether pair ``p``'s compare drew its bit. Not
    called at B = 2 (``speed_order`` stays ``arange(2)``)."""
    facts = _pair_facts(tables, state, qc)
    order = jnp.arange(4, dtype=_I32)
    reads = []
    for p, (i, j) in enumerate(PAIRS):
        b1, b2 = order[i], order[j]
        result, read = compare_full(tables, state, b1, b2, ties[p], qc, 0, 0, facts)
        swap = result != COMPARE_SPEED_FASTER
        order = order.at[i].set(jnp.where(swap, b2, b1)).at[j].set(jnp.where(swap, b1, b2))
        reads.append(read)
    return order, jnp.stack(reads)


def action_order_reads(tables: Tables, state: BattleState, actions, outcome) -> jnp.ndarray:
    """``bool[TIE_PAIRS[B]]``: the elements of ``outcome.order_tie`` the order reads (the pair's
    compare drew its bit); empty at B = 2, where ``speed_tie`` is the one bit."""
    if format_of(state).B == 2:
        return jnp.zeros((0,), bool)
    return _four_order(tables, state, actions, outcome.order_tie, state.qc_roll)[1]


def action_order(tables: Tables, state: BattleState, actions, outcome) -> jnp.ndarray:
    """``int8[B]``: the battlers in execution order. At B = 2, ``outcome.speed_tie`` decides
    equal speeds (0 = player first): the decomp swaps on ``RandNext() & 1``. Quick Claw
    reads ``state.qc_roll``, the draw of the turn before (module docstring), never the
    outcome. At B = 4 (``_four_order``) the pairs' bits are ``outcome.order_tie``; absent
    battlers (``battlersSwitchingMask``: PICK_COMMAND MOVE_END, bcp.c:330-333) lose every compare
    against a living one (lib.c:1204-1211) and go last."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    if fmt.B == 4:
        return _four_order(tables, state, actions, outcome.order_tie, state.qc_roll)[0]
    switch, item = is_switch(actions), is_item(actions)
    early = switch | item  # items and switches go before moves
    moves = jnp.stack(
        [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
    )
    both_moves = ~early[0] & ~early[1]
    priority = jnp.where(both_moves, tables.move_priority[moves].astype(_I32), 0)
    second = player_second(tables, state, outcome.speed_tie, state.qc_roll)
    player_second_ = (priority[0] < priority[1]) | ((priority[0] == priority[1]) & second)
    # Two actions of one kind compare speeds; an item and a switch go in battler order
    # (player first); a switch or item always precedes a move.
    same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
    player_second_ = jnp.where(same_kind, player_second_, early[1] & ~early[0])
    first = player_second_.astype(jnp.int8)
    return jnp.stack([first, _foe(first)]).astype(jnp.int8)


def truant_loafs(state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``Battler_CheckTruant`` (battle_lib.c:3304-3313): ``side``'s battler has Truant
    (``Battler_Ability``) and ``moveEffectsData.truant != (totalTurns & 1)``. ``truant`` is
    ``(totalTurns + 1) & 1`` from the switch-in or faint that brought the battler in
    (battle_lib.c:2119, :2197; ``step._entered``), 0 for a lead (the zeroed context), and
    ``totalTurns & 1`` after a Transform (battle_script.c:5359, G-69). So a Truant battler loafs on
    every other turn from its entry or its Transform, whatever it did."""
    parity = state.truant[side].astype(_I32) != (state.turn.astype(_I32) & 1)
    return (battler_ability(state, side) == AbilityId.TRUANT) & parity
