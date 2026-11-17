"""A trainer's bag items, generic over their party params (E1 ruling G-43).

``tables.item_party_param[item]`` holds ``Item_Get`` of every party param the two decomp
functions read (``ids.ItemParam``). ``bag_item_category`` is ``TrainerAI_ShouldUseItem``'s
choice for one bag slot (trainer_ai/trainer_ai.c:4056-4197); ``use_bag_item`` runs the
subscript ``BattleControllerPlayer_ItemCommand`` picks for it
(battle_controller_player.c:1905-1944) around ``BattleSystem_UseBagItem``
(battle/battle_system.c:435-758). Both read only the item user's own battler, so they take
``side`` and its active slot (docs/doubles-state-design.md section 5; the decomp's
``targetSlot`` is the user's own slot in singles, battle_system.c:443-454).

Not modelled (``refused_bag_params``, refused by coverage and the AI): an item that sets
Mist (``GUARD_SPEC``, :563-573) or Focus Energy (``CRIT_STAGES``, :617-626), whose
mechanics belong to another stream, and one whose friendship amount can take a band's
friendship past 255, where ``Pokemon_IncreaseValue`` wraps the party's u8 friendship
(pokemon.c:1859-1874) apart from the battler's clamped one (battle_lib.c:1032-1044) and a
switch-in re-reads the party's (battle_lib.c ``BattleSystem_InitBattleMon``).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import (
    HP_RESTORED_ALL,
    HP_RESTORED_HALF,
    HP_RESTORED_QUARTER,
    ITEM_FULL_RESTORE,
    ItemAiCategory,
    ItemParam,
    Stat,
    Status,
)
from nuzline.engine.state import BattleState

_I32 = jnp.int32
MAX_STAGE = 6  # MAX_STAT_STAGE 12 in the decomp's 0-12 encoding; BattleState stages are -6..6
FRIENDSHIP_MAX = 255
# UseBagItem's stage params and the stage each raises (battle_system.c:575-615), in order.
STAGE_PARAMS = (
    (ItemParam.ATK_STAGES, Stat.ATK),
    (ItemParam.DEF_STAGES, Stat.DEF),
    (ItemParam.SPATK_STAGES, Stat.SPA),
    (ItemParam.SPDEF_STAGES, Stat.SPD),
    (ItemParam.SPEED_STAGES, Stat.SPE),
    (ItemParam.ACC_STAGES, Stat.ACC),
)
# TrainerAI_ShouldUseItem's status branches in order (trainer_ai.c:4118-4155): the param, the
# statuses it tests and the usedItemCondition bit it sets (ITEM_RECOVER_*, constants/items.h:
# 38-43). Confusion (bit 0) is a volatile and has no status.
STATUS_BRANCHES = (
    (ItemParam.HEAL_SLEEP, (Status.SLEEP,), 5),
    (ItemParam.HEAL_POISON, (Status.POISON, Status.TOXIC), 4),
    (ItemParam.HEAL_BURN, (Status.BURN,), 3),
    (ItemParam.HEAL_FREEZE, (Status.FREEZE,), 2),
    (ItemParam.HEAL_PARALYSIS, (Status.PARALYSIS,), 1),
)
CONFUSION_BIT = 0  # ITEM_RECOVER_CONFUSION's bit
# UseBagItem's friendship bands (battle_system.c:716-732): give flag, amount, and the band
# [low, top) of the Pokemon's friendship it applies in.
FRIENDSHIP_BANDS = (
    (ItemParam.GIVE_FRIENDSHIP_LOW, ItemParam.FRIENDSHIP_LOW, 100),
    (ItemParam.GIVE_FRIENDSHIP_MED, ItemParam.FRIENDSHIP_MED, 200),
    (ItemParam.GIVE_FRIENDSHIP_HIGH, ItemParam.FRIENDSHIP_HIGH, FRIENDSHIP_MAX + 1),
)
# UseBagItem's status cures (battle_system.c:459-541): each clears its statuses.
STATUS_CURES = (
    (ItemParam.HEAL_SLEEP, (Status.SLEEP,)),
    (ItemParam.HEAL_POISON, (Status.POISON, Status.TOXIC)),
    (ItemParam.HEAL_BURN, (Status.BURN,)),
    (ItemParam.HEAL_FREEZE, (Status.FREEZE,)),
    (ItemParam.HEAL_PARALYSIS, (Status.PARALYSIS,)),
)


# The params TrainerAI_ShouldUseItem has a branch for, besides Full Restore's id
# (trainer_ai.c:4105-4185): an item with none of them is never used (crit-stage-only, PP-only,
# EV- or friendship-only items: Dire Hit, Ether, HP Up, Rare Candy, the EV berries).
AI_BRANCH_PARAMS = (
    ItemParam.HP_RESTORE,
    ItemParam.HEAL_SLEEP,
    ItemParam.HEAL_POISON,
    ItemParam.HEAL_BURN,
    ItemParam.HEAL_FREEZE,
    ItemParam.HEAL_PARALYSIS,
    ItemParam.HEAL_CONFUSION,
    *(param for param, _ in STAGE_PARAMS),
    ItemParam.GUARD_SPEC,
)


def refused_bag_params(params, item: int) -> list[str]:
    """Why bag item ``item`` with party params ``params`` (its ``item_party_param`` row, any
    int sequence) is not modelled: the mechanics ``BattleSystem_UseBagItem`` would apply that
    the engine does not build (module docstring). Empty when it is modelled, and for an item
    the AI never uses (no branch of ``AI_BRANCH_PARAMS`` nor Full Restore's id), whose
    ``UseBagItem`` never runs."""
    if int(item) != ITEM_FULL_RESTORE and not any(int(params[p]) for p in AI_BRANCH_PARAMS):
        return []
    out = []
    if int(params[ItemParam.GUARD_SPEC]):
        out.append("sets Mist on its side (Guard Spec param, battle_system.c:563-573)")
    if int(params[ItemParam.CRIT_STAGES]):
        out.append("sets Focus Energy (crit-stages param, battle_system.c:617-626)")
    for give, amount, top in FRIENDSHIP_BANDS:
        gain = int(params[amount])
        if gain > 0 and int(params[ItemParam.HOLD_FRIENDSHIP_UP]):
            gain = gain * 150 // 100
        if int(params[give]) and top - 1 + gain > FRIENDSHIP_MAX:
            out.append(
                f"can raise a friendship of {top - 1} past {FRIENDSHIP_MAX}, which wraps the "
                "party's friendship (pokemon.c:1859-1874)"
            )
    return out


def _params(tables: Tables, item) -> jnp.ndarray:
    return tables.item_party_param[jnp.asarray(item).astype(_I32)].astype(_I32)


def bag_item_branch(tables: Tables, state: BattleState, side, item) -> tuple:
    """``(kind, condition, use)`` of bag item ``item`` in ``TrainerAI_ShouldUseItem``
    (trainer_ai.c:4099-4189), whose else-if chain tests only the first branch whose param
    test holds: ``kind`` is that branch's ``ids.ItemAiCategory`` (``NONE``: no branch),
    ``condition`` its ``usedItemCondition`` (the ``ITEM_RECOVER_*`` bit index of a status
    branch, the stat of a stat booster) and ``use`` whether the AI uses the item there. The
    branches: Full Restore by id, used below 1/4 HP; an HP item (``HP_RESTORE``), used with a
    nonzero u8 ``HP_RESTORED`` below 1/4 HP or when the whole amount fits; each status cure,
    used when the battler has that status; confusion; then the stage params and Guard Spec,
    used from the battler's first turn on (``fakeOutTurnNumber - totalTurns >= 0``, :4156;
    Guard Spec also without Mist on the enemy side, ``sideConditionsMask[1]``)."""
    p = _params(tables, item)
    mon = state.active[side]
    hp, max_hp = state.hp[side, mon].astype(_I32), state.max_hp[side, mon].astype(_I32)
    status = state.status[side, mon].astype(_I32)
    low = (hp > 0) & (hp < max_hp // 4)
    restored = p[ItemParam.HP_RESTORED] & 0xFF  # u8 hpRestore (trainer_ai.c:4061, :4106)
    heals = (restored > 0) & (hp > 0) & ((hp < max_hp // 4) | (max_hp - hp > restored))
    out = [jnp.int32(ItemAiCategory.NONE), jnp.int32(0), jnp.asarray(False)]
    decided = [jnp.asarray(False)]

    def branch(test, use, category, condition):
        take = ~decided[0] & test
        out[0] = jnp.where(take, category, out[0])
        out[1] = jnp.where(take, condition, out[1])
        out[2] = jnp.where(take, use, out[2])
        decided[0] = decided[0] | take

    branch(jnp.asarray(item).astype(_I32) == ITEM_FULL_RESTORE, low, ItemAiCategory.FULL_RESTORE, 0)
    branch(p[ItemParam.HP_RESTORE] != 0, heals, ItemAiCategory.RECOVER_HP, 0)
    for param, statuses, bit in STATUS_BRANCHES:
        has = jnp.isin(status, jnp.array([int(s) for s in statuses], _I32))
        branch(p[param] != 0, has, ItemAiCategory.RECOVER_STATUS, bit)
    confused = state.confusion[side] > 0
    branch(p[ItemParam.HEAL_CONFUSION] != 0, confused, ItemAiCategory.RECOVER_STATUS, 0)
    settled = state.fake_out_turn[side].astype(_I32) - state.turn.astype(_I32) >= 0
    for param, stat in STAGE_PARAMS:
        branch(p[param] != 0, settled, ItemAiCategory.STAT_BOOSTER, int(stat))
    # Guard Spec: never used in a fight (refused_bag_params), kept for the branch order.
    no_mist = state.mist[1] == 0
    branch(p[ItemParam.GUARD_SPEC] != 0, settled & no_mist, ItemAiCategory.GUARD_SPEC, 0)
    return out[0], out[1], out[2]


def bag_item_category(tables: Tables, state: BattleState, side, item) -> jnp.ndarray:
    """``TrainerAI_ShouldUseItem``'s ``usedItemType`` for bag item ``item``: its branch's
    category when the AI uses it, else ``NONE`` (``bag_item_branch``)."""
    kind, _, use = bag_item_branch(tables, state, side, item)
    return jnp.where(use, kind, ItemAiCategory.NONE).astype(_I32)


def _set(state: BattleState, field: str, index, value, cond) -> BattleState:
    arr = getattr(state, field)
    new = jnp.where(cond, value, arr[index]).astype(arr.dtype)
    return state._replace(**{field: arr.at[index].set(new)})


def use_bag_item(tables: Tables, state: BattleState, side, item, run) -> BattleState:
    """``side``'s active battler uses bag item ``item`` where ``run``: the subscript of its
    ``bag_item_branch`` around ``BattleSystem_UseBagItem`` (see the module docstring).

    * ``subscript_use_full_restore.s`` first sets the status to none and clears confusion,
      then runs ``subscript_use_potion.s``: ``UseBagItem``, then ``BATTLE_SUBSCRIPT_UPDATE_HP``
      with the battler's ``ITEM_HP_RECOVERY`` when it is nonzero (capped at max HP).
    * ``subscript_use_status_recovery.s``: ``UseBagItem``, then for condition bit 0
      (``LowestBit``, pokemon.c:4979) clear confusion, else set the status to none.
    * ``subscript_use_stat_booster.s`` / ``_guard_spec.s``: ``UseBagItem`` only.

    ``UseBagItem`` on the battler (``BtlCmd_UseBagItem``, battle_script.c:8533-8542, move slot
    NULL = 0): each status cure (:459-541; a sleep cure also ends Nightmare, which the engine
    does not build), confusion (:543-551), infatuation (:553-561); each stage param raises
    its stage by 1 below +6 (:575-615); ``PP_RESTORE`` restores ``PP_RESTORED`` to move slot
    0 and ``PP_RESTORE_ALL`` to every slot whose PP is below its max (:628-661), capped at
    ``MoveTable_CalcMaxPP`` (pokemon.c ``Pokemon_IncreaseValue``; battle_lib.c ``Battler_AddVal``;
    the battler's PP is skipped for a Transformed battler or a Mimicked slot, :636-641, never
    the case here: coverage refuses Transform and Mimic, which the engine does not build),
    the move's base PP for a trainer Pokemon (no PP Ups: ``BoxPokemon_SetMoveSlot``); an HP
    item (:663-714) restores ``HP_RESTORED`` (``HP_RESTORED_*`` codes: all, half, a quarter,
    at least 1) to a living battler below max HP (to a fainted one with ``REVIVE``), set as the
    AI battler's ``ITEM_HP_RECOVERY``; then, when any of those applied, the friendship band's
    amount (:716-755): +1 more for a Luxury Ball or an egg met on this map (neither for a
    trainer Pokemon, caught in a Poke Ball, pokemon.c:441-442, with no egg location), times
    1.5 for a ``HOLD_EFFECT_FRIENDSHIP_UP`` bag item, clamped to 0-255. Mist and Focus Energy
    (``refused_bag_params``) are not applied."""
    run = jnp.asarray(run)
    # The item's branch picks the subscript; in a fight the AI used it there, so the use
    # condition held (an item action the AI did not choose is a test's, not the game's).
    cat, cond, _ = bag_item_branch(tables, state, side, item)
    p = _params(tables, item)
    mon = state.active[side]
    full = run & (cat == ItemAiCategory.FULL_RESTORE)
    state = _set(state, "status", (side, mon), Status.NONE, full)
    state = _set(state, "status_counter", (side, mon), 0, full)
    state = _set(state, "confusion", side, 0, full)

    applied = jnp.asarray(False)
    status = state.status[side, mon].astype(_I32)
    for param, statuses in STATUS_CURES:
        has = jnp.isin(status, jnp.array([int(s) for s in statuses], _I32))
        cure = run & (p[param] != 0) & has
        state = _set(state, "status", (side, mon), Status.NONE, cure)
        state = _set(state, "status_counter", (side, mon), 0, cure)
        applied = applied | cure
    cure = run & (p[ItemParam.HEAL_CONFUSION] != 0) & (state.confusion[side] > 0)
    state = _set(state, "confusion", side, 0, cure)
    applied = applied | cure
    cure = run & (p[ItemParam.HEAL_ATTRACT] != 0) & (state.infatuated[side] > 0)
    state = _set(state, "infatuated", side, 0, cure)
    applied = applied | cure
    for param, stat in STAGE_PARAMS:
        stage = state.stages[side, stat].astype(_I32)
        up = run & (p[param] != 0) & (stage < MAX_STAGE)
        state = _set(state, "stages", (side, stat), stage + 1, up)
        applied = applied | up
    moves = state.moves[side, mon].astype(_I32)
    max_pp = tables.move_pp[moves].astype(_I32)
    gain = p[ItemParam.PP_RESTORED]
    slots = jnp.arange(moves.shape[0])
    one = (p[ItemParam.PP_RESTORE] != 0) & (slots == 0)
    every = p[ItemParam.PP_RESTORE_ALL] != 0
    # PP_RESTORE then PP_RESTORE_ALL: slot 0 can take both (:628-661).
    pp = state.pp[side, mon].astype(_I32)
    for restore in (one, every & (slots >= 0)):
        restore = run & restore & (pp != max_pp)
        pp = jnp.where(restore, jnp.minimum(pp + gain, max_pp), pp)
        applied = applied | jnp.any(restore)
    state = _set(state, "pp", (side, mon), pp, run)

    hp, max_hp = state.hp[side, mon].astype(_I32), state.max_hp[side, mon].astype(_I32)
    revive = p[ItemParam.REVIVE] != 0
    target = jnp.where(revive, hp == 0, hp != 0)
    heal = run & (p[ItemParam.HP_RESTORE] != 0) & target & (hp != max_hp)
    code = p[ItemParam.HP_RESTORED] & 0xFF
    amount = jnp.where(
        code == HP_RESTORED_ALL,
        max_hp,
        jnp.where(
            code == HP_RESTORED_HALF,
            jnp.maximum(max_hp // 2, 1),
            jnp.where(code == HP_RESTORED_QUARTER, jnp.maximum(max_hp * 25 // 100, 1), code),
        ),
    )
    applied = applied | heal
    recovery = jnp.where(heal & ~revive, amount, 0)

    friendship = state.friendship[side, mon].astype(_I32)
    gain = jnp.int32(0)
    low = 0
    for give, amt, top in FRIENDSHIP_BANDS:
        band = (p[give] != 0) & (friendship >= low) & (friendship < top) & applied
        gain = jnp.where(band, p[amt], gain)
        low = top
    boosted = jnp.where(p[ItemParam.HOLD_FRIENDSHIP_UP] != 0, gain * 150 // 100, gain)
    gain = jnp.where(gain > 0, boosted, gain)
    friendship = jnp.clip(friendship + gain, 0, FRIENDSHIP_MAX)
    state = _set(state, "friendship", (side, mon), friendship, run & (gain != 0))

    # The subscript after UseBagItem (module docstring).
    potion = (cat == ItemAiCategory.FULL_RESTORE) | (cat == ItemAiCategory.RECOVER_HP)
    healed = jnp.minimum(hp + recovery, max_hp)
    state = _set(state, "hp", (side, mon), healed, run & potion & (recovery != 0))
    status_cat = run & (cat == ItemAiCategory.RECOVER_STATUS)
    state = _set(state, "confusion", side, 0, status_cat & (cond == CONFUSION_BIT))
    clear = status_cat & (cond != CONFUSION_BIT)
    state = _set(state, "status", (side, mon), Status.NONE, clear)
    return _set(state, "status_counter", (side, mon), 0, clear)


def refused_bag_items(tables: Tables, items) -> jnp.ndarray:
    """``bool[...]``: ``refused_bag_params`` is not empty for each of the bag items ``items``
    (the traced form, for the AI's refusals)."""
    items = jnp.asarray(items).astype(_I32)
    p = tables.item_party_param[items].astype(_I32)
    used = items == ITEM_FULL_RESTORE
    for param in AI_BRANCH_PARAMS:
        used = used | (p[..., param] != 0)
    out = (p[..., ItemParam.GUARD_SPEC] != 0) | (p[..., ItemParam.CRIT_STAGES] != 0)
    for give, amount, top in FRIENDSHIP_BANDS:
        gain = p[..., amount]
        boosted = (gain > 0) & (p[..., ItemParam.HOLD_FRIENDSHIP_UP] != 0)
        gain = jnp.where(boosted, gain * 150 // 100, gain)
        out = out | ((p[..., give] != 0) & (top - 1 + gain > FRIENDSHIP_MAX))
    return used & out
