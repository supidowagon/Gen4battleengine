"""A trainer's bag items, generic over their party params (E1 ruling G-43).

``tables.item_party_param[item]`` holds ``Item_Get`` of every party param the two decomp
functions read (``ids.ItemParam``). ``bag_item_category`` is ``TrainerAI_ShouldUseItem``'s
choice for one bag slot (trainer_ai/trainer_ai.c:4056-4197); ``use_bag_item`` runs the
subscript ``BattleControllerPlayer_ItemCommand`` picks for it
(battle_controller_player.c:1905-1944) around ``BattleSystem_UseBagItem``
(battle/battle_system.c:435-758). Both read only the item user's own battler, so they take
``side`` and its active slot (docs/doubles-state-design.md section 5; the decomp's
``targetSlot`` is the user's own slot in singles, battle_system.c:443-454).

Guard Spec sets Mist on the user's side (``BattleState.mist``, :563-573) and the crit-stages
param sets Focus Energy (``BattleState.focus_energy``, :617-626), E1 ruling G-62. The friendship
gain raises the party Pokemon's u8 friendship, which wraps past 255
(``Pokemon_IncreaseValue``, pokemon.c:1859-1874; ``BattleState.party_friendship``), and the
battler's, clamped (battle_lib.c:1032-1044; ``BattleState.friendship``): a switch-in re-reads the
party's (battle_lib.c ``BattleSystem_InitBattleMon``), E1 ruling F-15.
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.effects import MIST_TURNS, clear_nightmare
from nuzline.engine.format import format_of, party, side_of
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
from nuzline.engine.state import BattleState, field_index

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
    Guard Spec also without Mist on side 1, the AI's own, ``sideConditionsMask[1]``)."""
    p = _params(tables, item)
    mon, row = state.active[side], party(format_of(state), side)
    hp, max_hp = state.hp[row, mon].astype(_I32), state.max_hp[row, mon].astype(_I32)
    status = state.status[row, mon].astype(_I32)
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
    # Guard Spec: Mist on the AI's own side, side 1 (sideConditionsMask[1], :4181-4182).
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
    index = field_index(state, field, index)
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
    NULL = 0): each status cure (:459-541; a sleep cure also ends Nightmare, :469-472, G-32;
    Full Restore's own clear runs first and copies to the party, battle_script.c:3077-3078, so
    its UseBagItem finds no sleep and keeps the bit), confusion (:543-551), infatuation
    (:553-561); each stage param raises its stage by 1 below +6 (:575-615); ``PP_RESTORE`` restores ``PP_RESTORED`` to move slot
    0 and ``PP_RESTORE_ALL`` to every slot whose PP is below its max (:628-661), capped at
    ``MoveTable_CalcMaxPP`` (pokemon.c ``Pokemon_IncreaseValue``; battle_lib.c ``Battler_AddVal``;
    the test and the restore read the party Pokemon, so a Transformed battler's or a Mimicked
    slot's restore goes to the party's PP, ``shadow_pp``, and the battler's is skipped,
    :634-639, :651-654; F-21),
    the move's base PP for a trainer Pokemon (no PP Ups: ``BoxPokemon_SetMoveSlot``); an HP
    item (:663-714) restores ``HP_RESTORED`` (``HP_RESTORED_*`` codes: all, half, a quarter,
    at least 1) to a living battler below max HP (to a fainted one with ``REVIVE``), set as the
    AI battler's ``ITEM_HP_RECOVERY``; then, when any of those applied, the friendship band's
    amount (:716-755): +1 more for a Luxury Ball or an egg met on this map (neither for a
    trainer Pokemon, caught in a Poke Ball, pokemon.c:441-442, with no egg location), times
    1.5 for a ``HOLD_EFFECT_FRIENDSHIP_UP`` bag item; the band reads the party Pokemon's friendship
    (``party_friendship``), which takes the amount as ``Pokemon_IncreaseValue`` does (a sum past
    255 stores ``(255 + amount) & 0xFF``, below 0 stores 0, pokemon.c:1859-1874), the battler's
    (``friendship``, the user's own slot: always) clamped to 0-255 (battle_lib.c:1032-1044;
    F-15). Guard Spec (:563-573) sets the user's side's Mist for ``MIST_TURNS`` when its Mist turns are 0, and the
    crit-stages param (:617-626) sets the battler's Focus Energy when it is not set; each
    counts as applied for the friendship band (``result = TRUE``). The crit-stages param, like
    the stage params, needs ``selectedSlot == partySlot || targetSlot == partySlot``: the
    user's own slot in singles (module docstring), so always true here (E1 ruling G-62)."""
    run = jnp.asarray(run)
    # The item's branch picks the subscript; in a fight the AI used it there, so the use
    # condition held (an item action the AI did not choose is a test's, not the game's).
    cat, cond, _ = bag_item_branch(tables, state, side, item)
    p = _params(tables, item)
    mon, row = state.active[side], party(format_of(state), side)
    full = run & (cat == ItemAiCategory.FULL_RESTORE)
    state = _set(state, "status", (side, mon), Status.NONE, full)
    state = _set(state, "status_counter", (side, mon), 0, full)
    state = _set(state, "confusion", side, 0, full)

    applied = jnp.asarray(False)
    status = state.status[row, mon].astype(_I32)
    for param, statuses in STATUS_CURES:
        has = jnp.isin(status, jnp.array([int(s) for s in statuses], _I32))
        cure = run & (p[param] != 0) & has
        state = _set(state, "status", (side, mon), Status.NONE, cure)
        state = _set(state, "status_counter", (side, mon), 0, cure)
        if param == ItemParam.HEAL_SLEEP:  # battle_system.c:469-472 (G-32)
            state = clear_nightmare(state, side, cure)
        applied = applied | cure
    cure = run & (p[ItemParam.HEAL_CONFUSION] != 0) & (state.confusion[side] > 0)
    state = _set(state, "confusion", side, 0, cure)
    applied = applied | cure
    cure = run & (p[ItemParam.HEAL_ATTRACT] != 0) & (state.infatuated[side] > 0)
    state = _set(state, "infatuated", side, 0, cure)
    applied = applied | cure
    mist = run & (p[ItemParam.GUARD_SPEC] != 0) & (state.mist[side_of(side)] == 0)
    state = _set(state, "mist", side, MIST_TURNS, mist)
    applied = applied | mist
    for param, stat in STAGE_PARAMS:
        stage = state.stages[side, stat].astype(_I32)
        up = run & (p[param] != 0) & (stage < MAX_STAGE)
        state = _set(state, "stages", (side, stat), stage + 1, up)
        applied = applied | up
    focus = run & (p[ItemParam.CRIT_STAGES] != 0) & (state.focus_energy[side] == 0)
    state = _set(state, "focus_energy", side, 1, focus)
    applied = applied | focus
    # The test and the restore read the party Pokemon's move and PP (Pokemon_IncreaseValue,
    # :631-632); the battler's PP follows only for a slot that is no Transform or Mimic copy
    # (:635-639, :651-654). A copied slot's party values are the shadow (``mimic_slots``: all
    # four while transformed, G-68 / G-69; F-21).
    slots = jnp.arange(state.moves.shape[-1])
    copied = ((state.mimic_slots[side].astype(_I32) >> slots) & 1) == 1
    battler_pp = state.pp[row, mon].astype(_I32)
    moves = jnp.where(copied, state.shadow_moves[side], state.moves[row, mon]).astype(_I32)
    max_pp = tables.move_pp[moves].astype(_I32)
    gain = p[ItemParam.PP_RESTORED]
    one = (p[ItemParam.PP_RESTORE] != 0) & (slots == 0)
    every = p[ItemParam.PP_RESTORE_ALL] != 0
    # PP_RESTORE then PP_RESTORE_ALL: slot 0 can take both (:628-661).
    pp = jnp.where(copied, state.shadow_pp[side], battler_pp).astype(_I32)
    for restore in (one, every & (slots >= 0)):
        restore = run & restore & (pp != max_pp)
        pp = jnp.where(restore, jnp.minimum(pp + gain, max_pp), pp)
        applied = applied | jnp.any(restore)
    state = _set(state, "pp", (side, mon), jnp.where(copied, battler_pp, pp), run)
    state = _set(state, "shadow_pp", side, jnp.where(copied, pp, state.shadow_pp[side]), run)

    hp, max_hp = state.hp[row, mon].astype(_I32), state.max_hp[row, mon].astype(_I32)
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

    # The band is the party Pokemon's (battle_system.c:716-733); the party's u8 wraps and the
    # battler's clamps (F-15).
    party_value = state.party_friendship[row, mon].astype(_I32)
    gain = jnp.int32(0)
    low = 0
    for give, amt, top in FRIENDSHIP_BANDS:
        band = (p[give] != 0) & (party_value >= low) & (party_value < top) & applied
        gain = jnp.where(band, p[amt], gain)
        low = top
    boosted = jnp.where(p[ItemParam.HOLD_FRIENDSHIP_UP] != 0, gain * 150 // 100, gain)
    gain = jnp.where(gain > 0, boosted, gain)
    total = party_value + gain
    wrapped = jnp.where(total > FRIENDSHIP_MAX, (FRIENDSHIP_MAX + gain) & 0xFF, total)
    wrapped = jnp.where(total < 0, 0, wrapped)
    battler = jnp.clip(state.friendship[row, mon].astype(_I32) + gain, 0, FRIENDSHIP_MAX)
    state = _set(state, "party_friendship", (side, mon), wrapped, run & (gain != 0))
    state = _set(state, "friendship", (side, mon), battler, run & (gain != 0))

    # The subscript after UseBagItem (module docstring).
    potion = (cat == ItemAiCategory.FULL_RESTORE) | (cat == ItemAiCategory.RECOVER_HP)
    healed = jnp.minimum(hp + recovery, max_hp)
    state = _set(state, "hp", (side, mon), healed, run & potion & (recovery != 0))
    status_cat = run & (cat == ItemAiCategory.RECOVER_STATUS)
    state = _set(state, "confusion", side, 0, status_cat & (cond == CONFUSION_BIT))
    clear = status_cat & (cond != CONFUSION_BIT)
    state = _set(state, "status", (side, mon), Status.NONE, clear)
    return _set(state, "status_counter", (side, mon), 0, clear)
