"""Legal-action masks for the two decision points: choosing a turn's action and
choosing a replacement after a faint.

Actions are ``int32`` scalars ``0..NUM_ACTIONS-1``: ``0-3`` use the active
Pokemon's move slot, ``4-9`` switch to party slot ``action - 4`` (spec §4.2). The
active slot's own switch is always illegal, so at most 9 actions are legal.

The opponent has four more, ``10-13`` (``0..NUM_OPP_ACTIONS-1``): use the item in bag
slot ``trainer_items[1, action - 10]`` on its active Pokemon (the decomp's
``PLAYER_INPUT_ITEM`` from ``TrainerAI_PickCommand``, trainer_ai.c:4034). A slot is
legal when it holds an item (any: ``engine.bag`` acts by its party params) and the active
Pokemon has HP left. The player's mask stays ``NUM_ACTIONS`` wide: the player's bag
is not modelled.

Move legality follows ``BattleSystem_CheckInvalidMoves`` (battle_lib.c:2255) for the
checks M1 has state for: an empty slot (``CHECK_INVALID_NO_MOVE``), no PP
(``CHECK_INVALID_NO_PP``), Disable (``CHECK_INVALID_DISABLED``: the slot holding
``disabled_move``), Taunt (``CHECK_INVALID_TAUNTED``: a taunted battler
cannot pick a move whose power is 0) and the Choice lock (``CHECK_INVALID_CHOICE_ITEM``,
:2307-2315: a holder of a Choice item whose ``choice_lock`` is a move it knows can pick
only that move; a lock it does not know is cleared there, which the engine reads as no
lock) and Imprison (``CHECK_INVALID_IMPRISONED``, :2287-2290: ``move_imprisoned``). Torment,
Gravity, Heal Block and Encore come with their mechanics. The
trainer AI scores the same mask (``TrainerAI_Init``, trainer_ai.c:234,
``CHECK_INVALID_ALL``). When every slot is invalid the decomp sets ``struggling`` instead
of letting the player pick a move (battle_controller_player.c:433); here action 0 stays legal and means Struggle
(``order.selected_move`` makes the same call from ``usable_moves``).

Switches are legal to a non-active, non-empty, non-fainted slot, and never while the
active Pokemon is trapped (``trapped``: Bind, Mean Look, Ingrain).

A battler locked into a move (``locked_move``: Bide, Rollout, Ice Ball, an uproar) picks no
command (``Battler_CanPickCommand`` is FALSE, battle_controller_player.c:335): its only action
is the slot holding the locked move, whatever its PP, and it has no switch or bag item
(``locked_slot``). A lock on a move the battler does not know (a call's: Copycat calling Bide,
Rollout or Uproar locks ``moveLockedInto`` to moveCur, battle_lib.c:1828-1832,
subscript_uproar.s:5) is still a lock: ``FightCommand`` runs ``moveLockedInto``
(battle_controller_player.c:1885-1888) and spends no PP (``ppDecremented``, :335;
``Battler_SlotForMove`` finds no slot, :2272); its one action is then action 0 (ruling B5-5).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.format import format_of, party
from nuzline.engine.ids import CHOICE_ITEMS, AbilityId, ItemId, Type
from nuzline.engine.state import BAG_SLOTS, SLOTS, BattleState, held_item

MOVE_ACTIONS = 4  # actions 0-3 use a move slot; 4-9 switch
NUM_ACTIONS = MOVE_ACTIONS + SLOTS
FIRST_ITEM_ACTION = NUM_ACTIONS  # opponent actions 10-13 use bag slot action - 10
NUM_OPP_ACTIONS = FIRST_ITEM_ACTION + BAG_SLOTS


def locked_slot(state: BattleState, side) -> jnp.ndarray:
    """``int32``: the move slot of ``side``'s active Pokemon holding ``locked_move``
    (``moveLockedInto``), 0 when it is locked into a move it does not know (module docstring),
    -1 when it is not locked."""
    fmt = format_of(state)
    moves = state.moves[party(fmt, side), state.active[side]]
    lock = state.locked_move[side]
    held = (moves == lock) & (lock != 0)
    return jnp.where(lock == 0, -1, jnp.where(jnp.any(held), jnp.argmax(held), 0)).astype(jnp.int32)


def choice_locked(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``side``'s active Pokemon holds a Choice item and its ``choice_lock`` is a
    move it knows (``CHECK_INVALID_CHOICE_ITEM``, battle_lib.c:2307-2315)."""
    fmt = format_of(state)
    mon = state.active[side]
    lock = state.choice_lock[side]
    # Battler_HeldItemEffect (:2309: ``held_item``, B13-29).
    held = jnp.isin(tables.item_effect[held_item(state, side, mon)], jnp.asarray(CHOICE_ITEMS))
    return held & (lock != 0) & jnp.any(state.moves[party(fmt, side), mon] == lock)


def move_imprisoned(state: BattleState, side, move) -> jnp.ndarray:
    """``bool`` (the shape of ``move``): ``Move_Imprisoned`` (battle_lib.c:3316-3338): a battler
    of the other side has ``MOVE_EFFECT_IMPRISON`` and knows ``move`` (the loop over battlers,
    doubles convention 4; battler b is on side b in singles)."""
    fmt = format_of(state)
    move = jnp.asarray(move, jnp.int32)
    out = jnp.zeros(move.shape, bool)
    for battler in range(state.active.shape[0]):
        known = state.moves[party(fmt, battler), state.active[battler]].astype(jnp.int32)
        holds = jnp.any(move[..., None] == known, axis=-1)
        sealing = (jnp.asarray(side) != battler) & (state.imprison[battler] != 0)
        out = out | (sealing & holds & (move != 0))
    return out


def imprisoning_foe(state: BattleState, side) -> jnp.ndarray:
    """``bool``: a battler of the other side has ``MOVE_EFFECT_IMPRISON`` (the first test of
    ``Move_Imprisoned``, battle_lib.c:3325-3326)."""
    out = jnp.bool_(False)
    for battler in range(state.active.shape[0]):
        out = out | ((jnp.asarray(side) != battler) & (state.imprison[battler] != 0))
    return out


def usable_moves(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[4]``: the active Pokemon's move slots that pass the checks above (a
    move, PP left, not the disabled move, not power 0 while taunted, the Choice-locked move
    when locked). None usable means Struggle. A locked battler's only usable slot is the
    locked one (``locked_slot``): it never Struggles, even with no PP left."""
    fmt = format_of(state)
    mon = state.active[side]
    moves = state.moves[party(fmt, side), mon]
    usable = (
        (moves > 0) & (state.pp[party(fmt, side), mon] > 0) & (moves != state.disabled_move[side])
    )
    taunted = state.taunt[side] > 0
    usable = usable & ~(taunted & (tables.move_power[moves] == 0))
    usable = usable & (~choice_locked(tables, state, side) | (moves == state.choice_lock[side]))
    usable = usable & ~move_imprisoned(state, side, moves)  # CHECK_INVALID_IMPRISONED, :2287
    # Encore (B10-9): every move but the encored one is invalid (battle_lib.c:2302-2304, any
    # opMask); with one valid move left the selection is skipped and the encored slot taken
    # (bcp.c:443-446): the only legal FIGHT action. None valid: Struggle (bcp.c:432-441 first).
    encore = state.encore_move[side]
    encored = encore != 0
    usable = usable & (~encored | (moves == encore))
    forced = jnp.arange(MOVE_ACTIONS) == state.encore_slot[side].astype(jnp.int32)
    usable = jnp.where(encored & jnp.any(usable), forced, usable)
    lock = locked_slot(state, side)
    return jnp.where(lock >= 0, jnp.arange(MOVE_ACTIONS) == lock, usable)


def legal_replacements(state: BattleState, side) -> jnp.ndarray:
    """``bool[6]``: party slots ``side`` may send in, non-active, non-empty and not
    fainted."""
    fmt = format_of(state)
    slots = jnp.arange(SLOTS)
    return (
        (slots != state.active[side])
        & (state.species[party(fmt, side)] > 0)
        & (state.hp[party(fmt, side)] > 0)
    )


def trapped(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool``: ``Battler_IsTrapped`` (battle_lib.c:5514-5536), which the party menu reads
    (battle_controller_player.c:572): ``VOLATILE_CONDITION_TRAPPED`` (the Bind of ``trapped`` and
    ``mean_look``, condition.h:77-78) or ``MOVE_EFFECT_INGRAIN``; a living foe's Shadow Tag unless
    the battler has it too, a living foe's Magnet Pull on a Steel battler, a living foe's Arena Trap
    on a grounded battler (``BattlerIsGrounded``, :5500-5507: no Levitate, no Magnet Rise, not
    Flying, or Iron Ball, or Gravity), all by ``Battler_Ability`` over the other side's living
    battlers (``CountAbilityTheirSide``, :5509-5512; E2 batch 9, B9-11). Shed Shell
    (HOLD_EFFECT_SWITCH, :5518) is an unsupported item."""
    fmt = format_of(state)
    mon = state.active[side]
    own = state.ability[party(fmt, side), mon]
    types = state.types[party(fmt, side), mon]
    # BattlerIsGrounded's Iron Ball (Battler_HeldItemEffect, :5505), G-56.
    iron = tables.item_effect[held_item(state, side, mon)] == ItemId.IRON_BALL
    grounded = (
        ((own != AbilityId.LEVITATE) & ~jnp.any(types == Type.FLYING)) | iron | (state.gravity != 0)
    )
    steel = jnp.any(types == Type.STEEL)
    trapped_by = jnp.asarray(False)
    for foe in range(state.active.shape[0]):
        other = state.active[foe]
        theirs = state.ability[party(fmt, foe), other]
        live = (jnp.asarray(side) != foe) & (state.hp[party(fmt, foe), other] > 0)
        abilities = (
            ((theirs == AbilityId.SHADOW_TAG) & (own != AbilityId.SHADOW_TAG))
            | ((theirs == AbilityId.MAGNET_PULL) & steel)
            | ((theirs == AbilityId.ARENA_TRAP) & grounded)
        )
        trapped_by = trapped_by | (live & abilities)
    return (
        (state.trapped[side] != 0)
        | (state.mean_look[side] != 0)
        | (state.ingrain[side] != 0)
        | trapped_by
    )


def legal_actions(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[NUM_ACTIONS]``: the actions ``side`` may choose this turn."""
    usable = usable_moves(tables, state, side)
    struggle = ~jnp.any(usable)
    usable = usable.at[0].set(usable[0] | struggle)
    switches = legal_replacements(state, side) & ~trapped(tables, state, side)
    switches = switches & (locked_slot(state, side) < 0)
    return jnp.concatenate([usable, switches])


def usable_trainer_items(tables: Tables, state: BattleState, side) -> jnp.ndarray:
    """``bool[BAG_SLOTS]``: ``side``'s bag slots holding an item, while its active Pokemon
    has HP left and is not locked into a move."""
    fmt = format_of(state)
    held = state.trainer_items[party(fmt, side)] != 0
    free = (state.hp[party(fmt, side), state.active[side]] > 0) & (locked_slot(state, side) < 0)
    return held & free


def legal_opponent_actions(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[NUM_OPP_ACTIONS]``: the opponent's moves and switches as
    ``legal_actions(tables, state, 1)``, then its bag slots (``usable_trainer_items``)."""
    return jnp.concatenate(
        [legal_actions(tables, state, 1), usable_trainer_items(tables, state, 1)]
    )


def decision_actions(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[NUM_ACTIONS]``: the player's legal actions at a decision point. At a player
    replacement (the active Pokemon has 0 HP, a living bench Pokemon exists and the battle
    goes on: ``step.needs_replacement(state)[0]``) only the switches to
    ``legal_replacements`` (trapping is ignored: the Pokemon is gone); otherwise
    ``legal_actions``. At a pause at a party list (E2 batch 6, ``BattleState.pivot_battler``):
    the player's own list offers its ``legal_replacements`` (``TryReplaceFaintedMon``, battle_script.c:
    5867-5877; a U-turn or Baton Pass user is not trapped by the list), the opponent's offers only
    action 0, a placeholder: the AI's pick is the search's (``playout.turn_inputs``)."""
    slots = jnp.arange(SLOTS)
    active_down = state.hp[0, state.active[0]] == 0
    bench = (state.species[0] != 0) & (state.hp[0] > 0) & (slots != state.active[0])
    replacing = active_down & jnp.any(bench) & (state.over == 0)
    forced = jnp.concatenate([jnp.zeros(MOVE_ACTIONS, bool), legal_replacements(state, 0)])
    wait = jnp.zeros(NUM_ACTIONS, bool).at[0].set(True)
    out = jnp.where(replacing, forced, legal_actions(tables, state, 0))
    # The pivoting battler (1 + battler) and its side: in singles battler b is side b.
    paused = state.pivot_battler != 0
    pivot_side = state.pivot_battler.astype(jnp.int32) - 1
    out = jnp.where(paused & (pivot_side != 0), wait, out)
    return jnp.where(paused & (pivot_side == 0), forced, out)
