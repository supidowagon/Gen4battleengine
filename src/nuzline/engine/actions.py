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
lock), Imprison (``CHECK_INVALID_IMPRISONED``, :2287-2290: ``move_imprisoned``) and Gravity
(``CHECK_INVALID_GRAVITY``, :2292-2295: ``gravity_banned``, G-67). Torment, Heal Block and
Encore come with their mechanics. The
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
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.format import format_of, party
from nuzline.engine.ids import (
    CHOICE_ITEMS,
    GRAVITY_MOVES,
    MOVE_CURSE,
    AbilityId,
    ItemId,
    MoveTarget,
    Type,
)
from nuzline.engine.state import BAG_SLOTS, SLOTS, BattleState, held_item

MOVE_ACTIONS = 4  # actions 0-3 use a move slot; 4-9 switch
NUM_ACTIONS = MOVE_ACTIONS + SLOTS
FIRST_ITEM_ACTION = NUM_ACTIONS  # opponent actions 10-13 use bag slot action - 10
NUM_OPP_ACTIONS = FIRST_ITEM_ACTION + BAG_SLOTS

# RANGE_MY_SIDE (include/constants/battle/moves.h:91): the ranges whose B = 2 target is the user
# (BattleControllerPlayer_MustSelectTarget, bcp.c:4363-4367).
MY_SIDE_RANGES = (
    MoveTarget.RANGE_SINGLE_TARGET_SPECIAL,
    MoveTarget.RANGE_USER,
    MoveTarget.RANGE_FIELD,
    MoveTarget.RANGE_USER_OR_ALLY,
)

# The target menu's legal set per range (D2-2 rule 5, GetTargetSelectLayout / sMoveTargetSlotFlags,
# battle_subscreen.c:1336-1364, :797-810), as relations of the user: columns user, partner
# (b ^ 2), foe b ^ 1, foe b ^ 3; rows by ``MoveTarget``.
_RELATIONS = np.zeros((len(MoveTarget), 4), bool)
for _range, _row in {
    MoveTarget.RANGE_SINGLE_TARGET: (0, 1, 1, 1),
    MoveTarget.RANGE_SINGLE_TARGET_SPECIAL: (1, 0, 0, 0),
    MoveTarget.RANGE_RANDOM_OPPONENT: (1, 0, 0, 0),
    MoveTarget.RANGE_ADJACENT_OPPONENTS: (0, 0, 1, 1),
    MoveTarget.RANGE_ALL_ADJACENT: (0, 1, 1, 1),
    MoveTarget.RANGE_USER: (1, 0, 0, 0),
    MoveTarget.RANGE_USER_SIDE: (1, 1, 0, 0),
    MoveTarget.RANGE_FIELD: (1, 1, 1, 1),
    MoveTarget.RANGE_OPPONENT_SIDE: (0, 0, 1, 1),
    MoveTarget.RANGE_ALLY: (0, 1, 0, 0),
    MoveTarget.RANGE_USER_OR_ALLY: (1, 1, 0, 0),
    MoveTarget.RANGE_SINGLE_TARGET_ME_FIRST: (0, 0, 1, 1),
}.items():
    _RELATIONS[_range] = _row


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


def gravity_banned(state: BattleState, moves) -> jnp.ndarray:
    """``bool`` per move id of ``moves``: ``Move_FailsInHighGravity`` (battle_lib.c:3394-3407):
    Gravity is up and the move is one of ``GRAVITY_MOVES`` (Fly, Bounce, Jump Kick, Hi Jump Kick,
    Splash, Magnet Rise). Selection (CHECK_INVALID_GRAVITY, :2292), the status check
    (CHECK_STATUS_STATE_GRAVITY, bcp.c:2586) and the invokers' calls read it."""
    moves = jnp.asarray(moves, jnp.int32)
    return (state.gravity != 0) & jnp.isin(moves, jnp.asarray(GRAVITY_MOVES, jnp.int32))


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
    # CHECK_INVALID_GRAVITY (battle_lib.c:2292-2295): Move_FailsInHighGravity (G-67).
    usable = usable & ~gravity_banned(state, moves)
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
    on a grounded battler (``BattlerIsGrounded``, :5500-5507: no Levitate, no Magnet Rise turns, not
    Flying, or Iron Ball, or Gravity), all by ``Battler_Ability`` over the other side's living
    battlers (``CountAbilityTheirSide``, :5509-5512; E2 batch 9, B9-11). Shed Shell
    (HOLD_EFFECT_SWITCH, :5518) is an unsupported item."""
    fmt = format_of(state)
    mon = state.active[side]
    own = state.ability[party(fmt, side), mon]
    types = state.types[party(fmt, side), mon]
    # BattlerIsGrounded's Iron Ball (Battler_HeldItemEffect, :5505), G-56.
    iron = tables.item_effect[held_item(state, side, mon)] == ItemId.IRON_BALL
    # BattlerIsGrounded reads the turns of Magnet Rise (:5503), not the mask bit (G-66).
    grounded = (
        (
            (own != AbilityId.LEVITATE)
            & (state.magnet_rise_turns[side] == 0)
            & ~jnp.any(types == Type.FLYING)
        )
        | iron
        | (state.gravity != 0)
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
    held = state.trainer_items[side] != 0
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


def _present(state: BattleState, b) -> jnp.ndarray:
    """``bool``: battler ``b``'s active Pokemon has HP (the engine's form of "not in
    ``battlersSwitchingMask``" at a command point, bcp.c:4108-4111, :4132-4135; D2-1 rule 5)."""
    fmt = format_of(state)
    return state.hp[party(fmt, b), state.active[b]] > 0


def selection_range(tables: Tables, state: BattleState, b, slot) -> jnp.ndarray:
    """``int32``: the range ``BattleControllerPlayer_MustSelectTarget`` uses for battler ``b``'s
    move slot ``slot`` (bcp.c:4344-4349): ``RANGE_USER`` for Curse (the move id, not its effect)
    on a non-Ghost user, else the move's data range (D2-1 rule 3)."""
    from nuzline.engine.effects import ghost_user  # effects imports this module

    fmt = format_of(state)
    move = state.moves[party(fmt, b), state.active[b], jnp.minimum(slot, MOVE_ACTIONS - 1)]
    own = (move == MOVE_CURSE) & ~ghost_user(state, b)
    return jnp.where(own, MoveTarget.RANGE_USER, tables.move_target[move]).astype(jnp.int32)


def selects_target(tables: Tables, state: BattleState, b, action) -> jnp.ndarray:
    """``bool``: battler ``b`` reaches ``COMMAND_SELECTION_TARGET_SELECT_INIT`` with ``action``
    and writes ``battlerActions[b][1]`` (D2-1 rule 5): a move action of a present battler that
    picks a command (not locked, bcp.c:334-338), has a usable move (not Struggle, :433-441), is
    not encored (:442-453) and, at B = 4, is not using RANGE_ALLY with the partner absent
    (bcp.c:4351-4357)."""
    fmt = format_of(state)
    action = jnp.asarray(action)
    ok = (
        (action < MOVE_ACTIONS)
        & (locked_slot(state, b) < 0)
        & jnp.any(usable_moves(tables, state, b))
        & (state.encore_move[b] == 0)
        & _present(state, b)
    )
    if fmt.B == 4:
        ally_alone = (selection_range(tables, state, b, action) == MoveTarget.RANGE_ALLY) & ~(
            _present(state, b ^ 2)
        )
        ok = ok & ~ally_alone
    return ok


def chosen_targets(tables: Tables, state: BattleState, actions, targets) -> jnp.ndarray:
    """``int8[B]``: ``battlerActions[b][1]`` after the turn's command selection (D2-1): per
    battler the new target when it ``selects_target``, else the stored ``chosen_target``. At
    B = 2 the new target follows from the range (bcp.c:4363-4367: the user for a RANGE_MY_SIDE
    range or a non-Ghost Curse, else ``b ^ 1``) and ``targets`` is ignored; at B = 4 it is
    ``targets[b]`` (the menu's pick, bcp.c:548)."""
    fmt = format_of(state)
    actions = jnp.asarray(actions)
    out = []
    for b in range(fmt.B):
        if fmt.B == 2:
            own = jnp.isin(
                selection_range(tables, state, b, actions[b]), jnp.asarray(MY_SIDE_RANGES)
            )
            new = jnp.where(own, b, b ^ 1)
        else:
            new = jnp.asarray(targets)[b]
        out.append(
            jnp.where(selects_target(tables, state, b, actions[b]), new, state.chosen_target[b])
        )
    return jnp.stack(out).astype(state.chosen_target.dtype)


def legal_targets(tables: Tables, state: BattleState, b, slot) -> jnp.ndarray:
    """``bool[B]``: the battlers battler ``b`` may choose as the target of move slot ``slot``
    (D2-2). B = 2: the one target ``MustSelectTarget`` writes (the user for a RANGE_MY_SIDE
    range, else ``b ^ 1``, bcp.c:4363-4369). B = 4: the menu's row for the range
    (``_RELATIONS``: user ``b``, partner ``b ^ 2``, foes ``b ^ 1`` and ``b ^ 3``), only the
    battlers whose active Pokemon has HP (battle_controller.c:810-833, battle_subscreen.c:2575);
    RANGE_ALLY with the partner absent gives none (bcp.c:4352-4357). Battlers 1 and 3 get the
    same relations: the AI's real pick is D8's."""
    fmt = format_of(state)
    rng = selection_range(tables, state, b, slot)
    if fmt.B == 2:
        own = jnp.isin(rng, jnp.asarray(MY_SIDE_RANGES))
        return jnp.arange(2) == jnp.where(own, b, b ^ 1)
    flags = jnp.asarray(_RELATIONS)[rng]
    who = jnp.stack([jnp.asarray(b), jnp.asarray(b) ^ 2, jnp.asarray(b) ^ 1, jnp.asarray(b) ^ 3])
    out = jnp.zeros(fmt.B, bool).at[who].set(flags)
    present = jnp.stack([_present(state, x) for x in range(fmt.B)])
    return out & present


def switch_targets(tables: Tables, state: BattleState, b) -> jnp.ndarray:
    """``bool[SLOTS]``: the party slots battler ``b`` may switch to (D2-3 rule 7): today's
    ``legal_replacements``, not the active slot of a partner that sends from the same party
    (``CheckCanSwitchPokemon``, party.c:1640-1646), not while trapped, not while locked. At
    B = 2 it is today's switch mask."""
    fmt = format_of(state)
    out = legal_replacements(state, b) & ~trapped(tables, state, b) & (locked_slot(state, b) < 0)
    if fmt.B == 4:
        same = party(fmt, b ^ 2) == party(fmt, b)
        taken = (jnp.arange(SLOTS) == state.active[b ^ 2]) & same
        out = out & ~taken
    return out


def legal_choices(tables: Tables, state: BattleState, b) -> jnp.ndarray:
    """``bool[NUM_ACTIONS, B]``: battler ``b``'s legal (action, target) pairs (D2-3). An absent
    battler has only ``[0, b]``; a move that selects a target (D2-1 ``selects_target``) takes any
    row of ``legal_targets``, one that does not (Struggle, Encore, a lock, RANGE_ALLY with no
    partner) writes none, so ``[action, b]`` is its one placeholder; a switch is ``[action, b]``
    when its slot is in ``switch_targets``. Every selectable target stays: the stored value is
    read later by a move lock (D2-3 rule 6)."""
    fmt = format_of(state)
    legal = legal_actions(tables, state, b)
    placeholder = jnp.arange(fmt.B) == b
    rows = []
    for a in range(MOVE_ACTIONS):
        picks = jnp.where(
            selects_target(tables, state, b, a), legal_targets(tables, state, b, a), placeholder
        )
        rows.append(picks & legal[a])
    switches = switch_targets(tables, state, b)
    for k in range(SLOTS):
        rows.append(placeholder & switches[k])
    out = jnp.stack(rows)
    absent = jnp.zeros_like(out).at[0, :].set(placeholder)
    return jnp.where(_present(state, b), out, absent)


def decision_pairs(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[NUM_ACTIONS, B, NUM_ACTIONS, B]``: the player's joint decision at B = 4 in DOUBLES
    and TWO_TRAINERS, where it commands battlers 0 and 2 (D2-3 rules 1, 2, 8):
    ``legal_choices`` of each, and not two switches to one slot (battler 2's party menu refuses
    battler 0's pending slot, party.c:1653-1661). MULTI commands battler 0 alone
    (``legal_choices(tables, state, 0)``): ValueError there and at B = 2."""
    fmt = format_of(state)
    if fmt.B != 4 or fmt.party_of[0] != fmt.party_of[2]:
        raise ValueError("decision_pairs is the joint decision of DOUBLES and TWO_TRAINERS")
    first = legal_choices(tables, state, 0)
    second = legal_choices(tables, state, 2)
    acts = jnp.arange(NUM_ACTIONS)
    same_slot = (
        (acts[:, None] >= MOVE_ACTIONS)
        & (acts[None, :] >= MOVE_ACTIONS)
        & (acts[:, None] == acts[None, :])
    )
    return first[:, :, None, None] & second[None, None, :, :] & ~same_slot[:, None, :, None]
