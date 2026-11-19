"""Experience payout and level-up (K3d-exp, design §1.1-1.4, §3.1): ``pay_exp``; the
learnset resolve after ``step`` (design §1.5, §3.2-3.3): ``advance_learns``, ``needs_learn``,
``learn_options``, ``learn``.

Paths are relative to ``external/pokeplatinum``; ``bs.c`` is ``src/battle/battle_script.c``,
``bcp.c`` ``src/battle/battle_controller_player.c``.

When. ``BtlCmd_PlayFaintAnimation`` sets the payout flag of the fainted battler (bs.c:1800);
``BattleControllerPlayer_AnyExpPayout`` (bcp.c:4449-4469) runs ``subscript_grant_exp.s``
(``CalcExpGain``, ``StartGetExpTask``, ``WaitGetExpTask``) for the lowest flagged battler. Its
callers are ``MoveEnd`` after ``RecoverStatusByAbility`` for the attacker and the defender
(bcp.c:3965-3969), the heads of ``CheckFieldConditions`` (:938-944) and ``CheckMonConditions``
(:1295-1299) and ``TurnEnd`` (:1846-1848), each before ``CheckBattleOver``, so a KO that ends
the battle pays too; and Pursuit's own ``BATTLE_SUBSCRIPT_GRANT_EXP``
(``subscript_pursuit.s:106-108``). ``step`` calls ``pay_exp`` after the matching faint
checks (``step._execute_actions``, ``field_conditions``, ``end_of_turn``, ``replace``).

What (``BtlCmd_CalcExpGain``, bs.c:2187-2245; ``BattleScript_GetExpTask``, bs.c:9700-9836;
``BattleScript_CalcEffortValues``, bs.c:10214-10306; ``Pokemon_ShouldLevelUp``,
src/pokemon.c:3501-3526; ``Pokemon_UpdateFriendship``, src/pokemon.c:2635-2689;
``Pokemon_CalcStats``, src/pokemon.c:564-648) is computed in closed form, with no loop over
levels: the reference engine (``nuzline.reference.engine._grant_exp``) runs the decomp's
per-level loop literally, and the differential tests prove the two equal.

Not modelled here, and refused elsewhere (design §4): the egg-location friendship bonus
(src/pokemon.c:2668-2670, read only by the friendship-power moves, §4.4) and a battle item
that differs from the party item (§4.2). The learnset scan of the levels gained is deferred
to the post-step resolve (``learn_level`` / ``learn_cursor``, design §3.2-3.3).

The resolve (``advance_learns``, ``learn``) runs after ``step`` and before any faint
replacement: the decomp's ``TurnEnd`` payout and its prompts (bcp.c:1846) come before
``ReplaceFainted`` (:1848). It is never inlined into ``step``. Only the learnset work is
deferred: auto-learn fills only an empty slot (src/pokemon.c:3857-3862), and the readers of a
slot that holds a move, between a mid-turn payout and the resolve, are exact under the
``learn_options`` masks or refused (design §3.3):

* the player's pending move after the opponent fainted first: the decomp stores the slot at
  selection (bcp.c:525-527) and re-reads ``moves[moveSlot]`` when the move runs
  (bcp.c:1885-1895; the PP slot by ``Battler_SlotForMove``, bcp.c:2272-2283), so only a
  prompt that replaces the selected slot differs: ``learn_pin`` masks that option;
* the end-of-turn Disable and Encore checks end each once ``disabledMove`` / ``encoredMove``
  is no longer known (bcp.c:1565-1576, :1591-1602), while the engine runs them before the
  resolve: masked with the disabled and the encored move (ruling D2);
* the locked move and the Choice lock are read by move id: ``FightCommand`` runs
  ``moveLockedInto`` (bcp.c:1885-1888) and spends no PP for it (``ppDecremented``,
  bcp.c:335; no slot, :2272-2294), as ``order.selected_move`` and ``actions.locked_slot``
  do, and ``CHECK_INVALID_CHOICE_ITEM`` clears a lock the battler no longer knows
  (battle_lib.c:2307-2310), which ``actions.choice_locked`` reads as no lock: replacing
  either move is exact and offered;
* a newcomer the hazards faint inside the opponent's replacement loop: refused
  (``exp_refused``, window (c)).

The converter refuses a recorded answer the masks drop (``tools.trace_to_state._learn_refusal``).

Prompts still pending when the battle ends stay in the state (ruling D8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import EffectId, Stat
from nuzline.engine.state import MOVES, SLOTS, BattleState
from nuzline.engine.stats import SPECIES_SHEDINJA, calc_stats_jnp

_I32 = jnp.int32

# Decomp hold effects (generated/item_hold_effects.txt, line N is id N - 1), read from each
# game's own item data (``tables.item_hold_effect``): Kaizo gives the Power items Macho
# Brace's effect (tests/data/test_tables.py).
HOLD_EFFECT_EVS_UP_SPEED_DOWN = 50  # Macho Brace, :51
HOLD_EFFECT_EXP_SHARE = 51  # :52
HOLD_EFFECT_FRIENDSHIP_UP = 53  # Soothe Bell, :54
HOLD_EFFECT_EXP_UP = 66  # Lucky Egg, :67
# HOLD_EFFECT_LVLUP_{HP,ATK,DEF,SPATK,SPDEF,SPEED}_EV_UP (:123, :118-122), in ids.Stat order.
POWER_ITEM_HOLD_EFFECTS = (122, 117, 118, 119, 120, 121)

# The STAT_* order of BattleScript_CalcEffortValues' loop (generated/pokemon_stats.txt:1-6,
# MON_DATA_HP_EV + stat, generated/pokemon_data_params.txt:14-19), as ids.Stat indices.
DECOMP_STAT_ORDER = (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD)

MAX_LEVEL = 100  # MAX_POKEMON_LEVEL, include/constants/pokemon.h:22
MAX_EVS_SINGLE_STAT = 255  # include/constants/pokemon.h:15
MAX_EVS_ALL_STATS = 510  # include/constants/pokemon.h:16
MAX_FRIENDSHIP = 255  # MAX_FRIENDSHIP_VALUE, include/constants/pokemon.h:28
# LOW_FRIENDSHIP_LIMIT, MED_FRIENDSHIP_LIMIT (include/constants/pokemon.h:29-30): the tiers of
# sFriendshipChangeTable[FRIENDSHIP_EVENT_LEVEL_UP] = {5, 3, 2} (src/pokemon.c:2623).
FRIENDSHIP_TIERS = (100, 200)
LEVEL_UP_FRIENDSHIP = (5, 3, 2)
BOOST_PERCENT = 150  # Lucky Egg, trainer battle, OT-less same language (bs.c:9749-9763)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _friendship(f0, k, bonus, soothe):
    """Friendship after ``k`` level-ups of ``Pokemon_UpdateFriendship(FRIENDSHIP_EVENT_
    LEVEL_UP)`` (src/pokemon.c:2635-2689), in closed form. Each level adds the tier delta of
    the value before it (``< 100``, ``< 200``, else; :2651-2662), +1 for a Luxury Ball
    (``bonus``, :2664-2666), then ``* 150 / 100`` with Soothe Bell (:2672-2676), clamped to
    255 (:2684-2686). Every delta is at least 2 and positive, so the value only rises: tier 0
    takes ``ceil((100 - f)^+ / d0)`` levels, tier 1 ``ceil((200 - f)^+ / d1)``, tier 2 the
    rest, and the clamp can bind only in tier 2 (a tier-1 value stays below 200 + d1)."""

    def delta(base):
        d = base + bonus
        return jnp.where(soothe, d * BOOST_PERCENT // 100, d)

    d0, d1, d2 = (delta(b) for b in LEVEL_UP_FRIENDSHIP)
    low, mid = FRIENDSHIP_TIERS
    n0 = jnp.minimum(k, _ceil_div(jnp.maximum(low - f0, 0), d0))
    f1 = f0 + n0 * d0
    n1 = jnp.minimum(k - n0, _ceil_div(jnp.maximum(mid - f1, 0), d1))
    f2 = f1 + n1 * d1
    return jnp.minimum(MAX_FRIENDSHIP, f2 + (k - n0 - n1) * d2)


def _effort_values(tables: Tables, after: BattleState, paid, foe_row, hold):
    """``int32[6, 6]``: the player's EVs after ``BattleScript_CalcEffortValues`` (bs.c:
    10214-10306) for the ``paid`` slots (bs.c:9776-9779, before any level-up check). Yields
    of the fainted Pokemon's form row (``SpeciesData_FromMonForm``, :10224); the receiver's
    party item (:10226-10228). Per stat in ``DECOMP_STAT_ORDER``: stop once the sum is 510
    (:10236-10239); yield plus ``itemPower`` for the matching Power item (:10244-10282);
    ``* 2`` with Pokerus (:10285-10287) and with Macho Brace (:10289-10291); clip the sum to
    510 (:10294-10296), then the stat to 255 (:10299-10301)."""
    evs = after.evs[0].astype(_I32)
    yields = tables.species_ev_yield[foe_row].astype(_I32)
    power = tables.item_hold_param[after.item[0]].astype(_I32)
    pokerus = after.pokerus != 0
    macho = hold == HOLD_EFFECT_EVS_UP_SPEED_DOWN
    total = jnp.sum(evs, axis=1)
    for stat in DECOMP_STAT_ORDER:
        tmp = yields[stat] + jnp.where(hold == POWER_ITEM_HOLD_EFFECTS[stat], power, 0)
        tmp = jnp.where(pokerus, tmp * 2, tmp)
        tmp = jnp.where(macho, tmp * 2, tmp)
        tmp = jnp.where(total + tmp > MAX_EVS_ALL_STATS, MAX_EVS_ALL_STATS - total, tmp)
        cur = evs[:, stat]
        tmp = jnp.where(cur + tmp > MAX_EVS_SINGLE_STAT, MAX_EVS_SINGLE_STAT - cur, tmp)
        tmp = jnp.where(paid & (total < MAX_EVS_ALL_STATS), tmp, 0)
        evs = evs.at[:, stat].add(tmp)
        total = total + tmp
    return evs


def pay_exp(
    tables: Tables, before: BattleState, after: BattleState, pending=False, selected=0
) -> BattleState:
    """``after`` with the exp payout for an opponent faint between ``before`` and ``after``
    (``after`` is the state after the hook's faint check); unchanged when nothing fainted.
    ``pending``: the player's move ``selected`` (its slot, 0-3) has not run yet this turn, so
    a level-up of the player's active pins that slot (``learn_pin``, design §3.3(a))."""
    return pay_exp_leveled(tables, before, after, pending, selected)[0]


def pay_exp_leveled(
    tables: Tables, before: BattleState, after: BattleState, pending=False, selected=0
) -> tuple[BattleState, jnp.ndarray]:
    """``(pay_exp(...), bool[6])``: the state and the player slots that gained a level.

    Trigger (design §1.1): an opponent party slot ``j`` with ``before.hp > 0`` and
    ``after.hp == 0`` (the per-slot test of ``step._faint_check``): the payout is keyed on the
    fainted battler (bs.c:1800) and reads ``battleMons[faintedMon]`` (bs.c:2216-2217, 9776-9779),
    which holds a Pokemon Roar dragged in. Singles have at most one such slot per hook; the
    lowest is taken, as ``AnyExpPayout`` takes the lowest flag (bcp.c:4452-4458).

    ``CalcExpGain`` (bs.c:2196-2238; the engine plays only trainer singles, so every opponent
    faint pays): ``E = yield[species] * level / 7`` (yield by species, bs.c:2216-2217);
    ``n`` / ``s`` count the player's party Pokemon with a species and HP that are in
    ``exp_mask`` / hold an Exp. Share (party item, bs.c:2201-2212). With ``s > 0``:
    ``g = max(1, (E/2)/n)``, ``h = max(1, (E/2)/s)`` (:2219-2230); else ``g = max(1, E/n)``,
    ``h = 0`` (:2231-2238). ``n = 0`` divides safely and pays no ``g``: ``g`` reaches only
    mask slots with HP (ruling D4).

    ``GetExpTask`` visits the slots in the mask or holding an Exp. Share (bs.c:9703-9712); a
    visited slot with HP below level 100 (:9740) gets ``t = g`` if in the mask (:9741-9743),
    ``+ h`` with an Exp. Share (:9745-9747), ``* 150 / 100`` with a Lucky Egg (:9749-9751),
    ``* 150 / 100`` in a trainer battle (:9753-9755), then ``* exp_boost / 100`` (the OT and
    language test, :9757-9763, resolved by the host). ``exp += t`` (:9767-9775), then its EVs
    (``_effort_values``). The level-up check runs only when ``t != 0`` (:9782-9791):
    ``ShouldLevelUp`` clamps exp to the level-100 base (rate by species, src/pokemon.c:
    3506-3512) and raises the level one at a time while ``exp >= table[level + 1]``
    (:3518-3523; bs.c:9807-9836), so the final level is the last ``L`` with
    ``table[L] <= exp``. Every visited slot's mask bit is cleared (bs.c:10190-10193), and
    every mask slot is visited: the mask is 0 after a payout.

    Per level gained (bs.c:9838-9877): friendship (``_friendship``), then ``Pokemon_CalcStats``
    with the new EVs (base stats of the form row, src/pokemon.c:591-628; ``calc_stats_jnp``),
    whose HP rule ``hp += newMax - oldMax`` (Shedinja 1, :633-645) telescopes over the levels
    to ``hp + M(L1) - M0`` (max HP only grows with the level, HP stays nonzero). A slot that
    does not level keeps its stats, even with new EVs. Party and battler share one row, which
    is ``BattleSystem_ReloadPokemon``'s copy for the active (battle_lib.c:169-194). A leveled
    slot without pending learnset work gets ``learn_level = L0 + 1``, ``learn_cursor = 0``."""
    fainted = (before.hp[1] > 0) & (after.hp[1] == 0)
    paying = jnp.any(fainted)
    j = jnp.argmax(fainted)
    foe_species = after.species[1, j].astype(_I32)  # a Transformed foe's copy, bs.c:2216
    foe_row = tables.species_form_row[foe_species, after.form[1, j]]
    exp_e = tables.species_exp_yield[foe_species].astype(_I32) * after.level[1, j].astype(_I32)
    exp_e = exp_e // 7

    slots = jnp.arange(SLOTS)
    species, party_form, party_ivs = party_identity(after)
    hp = after.hp[0].astype(_I32)
    level = after.level[0].astype(_I32)
    alive = (species != 0) & (hp > 0)
    in_mask = (jnp.right_shift(after.exp_mask.astype(_I32), slots) & 1) == 1
    hold = tables.item_hold_effect[after.item[0]]
    share = hold == HOLD_EFFECT_EXP_SHARE
    n = jnp.sum((alive & in_mask).astype(_I32))
    s = jnp.sum((alive & share).astype(_I32))
    whole = jnp.where(s > 0, exp_e // 2, exp_e)
    g = jnp.maximum(1, whole // jnp.maximum(n, 1))
    h = jnp.where(s > 0, jnp.maximum(1, exp_e // 2 // jnp.maximum(s, 1)), 0)

    paid = paying & (in_mask | share) & alive & (level != MAX_LEVEL)
    t = jnp.where(in_mask, g, 0) + jnp.where(share, h, 0)
    t = jnp.where(hold == HOLD_EFFECT_EXP_UP, t * BOOST_PERCENT // 100, t)
    t = t * BOOST_PERCENT // 100
    boost = after.exp_boost.astype(_I32)
    t = jnp.where(boost != 100, t * boost // 100, t)
    t = jnp.where(paid, t, 0)
    checked = paid & (t != 0)

    table = tables.exp_table[tables.species_exp_rate[species]]  # [6, 101], rate by species
    exp = after.exp.astype(_I32) + t
    exp = jnp.where(checked, jnp.minimum(exp, table[:, MAX_LEVEL]), exp)
    evs = _effort_values(tables, after, paid, foe_row, hold)

    reached = jnp.sum((table[:, 1:] <= exp[:, None]).astype(_I32), axis=1)
    new_level = jnp.where(checked, jnp.maximum(level, reached), level)
    leveled = new_level > level

    soothe = hold == HOLD_EFFECT_FRIENDSHIP_UP
    f0 = after.party_friendship[0].astype(_I32)  # Pokemon_UpdateFriendship reads the party
    bonus = after.friend_bonus.astype(_I32)
    friendship = jnp.where(leveled, _friendship(f0, new_level - level, bonus, soothe), f0)

    row = tables.species_form_row[species, party_form]
    mods = tables.nature_mods[after.nature[0]]
    base = tables.species_base_stats[row]
    new = calc_stats_jnp(base, party_ivs, evs, new_level, mods, species)
    old_max = after.max_hp[0].astype(_I32)
    new_hp = jnp.where(species == SPECIES_SHEDINJA, 1, hp + new[:, Stat.HP] - old_max)

    marked = leveled & (after.learn_level == 0)
    pin = jnp.asarray(pending) & leveled[after.active[0]]

    def put(field, side0):
        full = getattr(after, field)
        return full.at[0].set(side0.astype(full.dtype))

    out = after._replace(
        level=put("level", new_level),
        hp=put("hp", jnp.where(leveled, new_hp, hp)),
        max_hp=put("max_hp", jnp.where(leveled, new[:, Stat.HP], old_max)),
        stats=put("stats", jnp.where(leveled[:, None], new[:, Stat.ATK :], after.stats[0])),
        # BattleSystem_ReloadPokemon (battle_lib.c:169-174) reloads the active battler's Attack
        # and Defense unswapped and keeps MOVE_EFFECT_POWER_TRICK (B12-10).
        power_trick_desync=after.power_trick_desync.at[0].set(
            jnp.where(
                leveled[after.active[0]], after.power_trick[0], after.power_trick_desync[0]
            ).astype(after.power_trick_desync.dtype)
        ),
        # A Transformed active Pokemon: the reload gives the battler its own new stats (:169-174)
        # and the party's shadow too.
        shadow_stats=after.shadow_stats.at[0].set(
            jnp.where(
                (after.transformed[0] != 0) & leveled[after.active[0]],
                new[after.active[0], Stat.ATK :],
                after.shadow_stats[0],
            ).astype(after.shadow_stats.dtype)
        ),
        # The level-up's ReloadPokemon re-reads the party value into the battler (F-15).
        friendship=put("friendship", jnp.where(leveled, friendship, after.friendship[0])),
        party_friendship=put("party_friendship", friendship),
        evs=put("evs", evs),
        exp=exp.astype(after.exp.dtype),
        exp_mask=jnp.where(paying, 0, after.exp_mask).astype(after.exp_mask.dtype),
        learn_level=jnp.where(marked, level + 1, after.learn_level).astype(after.learn_level.dtype),
        learn_cursor=jnp.where(marked, 0, after.learn_cursor).astype(after.learn_cursor.dtype),
        learn_pin=jnp.where(pin, jnp.asarray(selected) + 1, after.learn_pin).astype(
            after.learn_pin.dtype
        ),
    )
    return out, leveled


# --- the learnset resolve (design §1.5, §3.2) ---------------------------------------------------

GIVE_UP = MOVES  # learn option 4 gives the move up; options 0-3 replace that move slot
LEARN_OPTIONS = MOVES + 1


def needs_learn(state: BattleState) -> jnp.ndarray:
    """``bool``: a learn-move prompt is pending. Valid after ``advance_learns``, which leaves
    learnset work only at a prompt; the lowest such player slot holds it (``GetExpTask``
    visits the slots in ascending order, bs.c:9703-9712)."""
    return jnp.any(state.learn_level > 0)


def _learner(state: BattleState) -> jnp.ndarray:
    """The lowest player slot with learnset work."""
    return jnp.argmax(state.learn_level > 0)


def _row(tables: Tables, state: BattleState, slot) -> jnp.ndarray:
    """The learnset's personal row, by species and form (``Pokemon_LoadLevelUpMovesOf``,
    src/pokemon.c:3943)."""
    species, form, _ = party_identity(state)
    return tables.species_form_row[species[slot], form[slot]]


def party_identity(state: BattleState) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(species[6], form[6], ivs[6, 6])`` of the player's party Pokemon: the active slot's own
    (``shadow_*``) while it is Transformed, else the slots' fields. The exp table, the stats, the
    learnset and the Shedinja rule belong to the party Pokemon (``Pokemon_CalcStats``, the
    ``ReloadPokemon`` of battle_lib.c:169-194 copies its own stats back), G-69."""
    slots = jnp.arange(SLOTS)
    swapped = (state.transformed[0] != 0) & (slots == state.active[0])
    species = jnp.where(swapped, state.shadow_species[0], state.species[0])
    form = jnp.where(swapped, state.shadow_form[0], state.form[0])
    ivs = jnp.where(swapped[:, None], state.shadow_ivs[0][None, :], state.ivs[0])
    return species, form, ivs


def _masked(state: BattleState, slot) -> jnp.ndarray:
    """``bool[4]``: the move slots of player party ``slot`` whose battler value is not the party's:
    ``mimic_slots`` of the active Pokemon (G-68). The learn reads and writes the party Pokemon
    (``Pokemon_SetMoveSlot``, battle_script.c:10168-10176; ``BoxPokemon_AddMove``, pokemon.c:3852),
    and ``BattleSystem_ReloadPokemon`` reloads the unflagged slots from it (battle_lib.c:183-191)."""
    bits = (state.mimic_slots[0].astype(_I32) >> jnp.arange(MOVES)) & 1
    return (bits == 1) & (jnp.asarray(slot) == state.active[0])


def _party_moves(state: BattleState, slot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(moves, pp)`` of player party ``slot`` as the party Pokemon holds them: the shadow of a
    flagged slot (``_masked``), else the slot's own."""
    masked = _masked(state, slot)
    moves = jnp.where(masked, state.shadow_moves[0], state.moves[0, slot])
    pp = jnp.where(masked, state.shadow_pp[0], state.pp[0, slot])
    return moves, pp


def _put_party(state: BattleState, slot, k, move, pp, apply) -> BattleState:
    """Party ``slot``'s move slot ``k`` becomes ``move`` with ``pp`` where ``apply``: into the
    shadow when the battler's slot is flagged (``_masked``), else into the slot (the battler reload
    then shows it)."""
    shadowed = _masked(state, slot)[k]
    live, shadow = apply & ~shadowed, apply & shadowed

    def put(arr, index, value, cond):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    return state._replace(
        moves=put(state.moves, (0, slot, k), move, live),
        pp=put(state.pp, (0, slot, k), pp, live),
        shadow_moves=put(state.shadow_moves, (0, k), move, shadow),
        shadow_pp=put(state.shadow_pp, (0, k), pp, shadow),
    )


def _advance_once(tables: Tables, state: BattleState) -> tuple[BattleState, jnp.ndarray]:
    """``(state, prompt)``: one step of the lowest pending slot's learnset scan.

    ``Pokemon_LevelUpMove`` (src/pokemon.c:3935-3966) walks the whole list from the index,
    skipping every entry whose level is not exactly the scanned level, up to the terminator
    (:3945-3956): it does not stop at a higher level, and a list need not be sorted (vanilla
    row 354). On a match it moves the index past it (:3958-3961) and calls
    ``BoxPokemon_AddMove`` (:3852-3873), which walks the move slots in order: the first
    empty slot learns the move (``SetMoveSlot``, :3859-3862, with the slot's PP Ups,
    ``MoveTable_CalcMaxPP``, :3925-3933 and src/move_table.c:25-35: the base PP, since
    coverage refuses PP Ups on an empty slot), a slot holding the move first makes it
    "already known" (:3864-3866; skipped, bs.c:10008-10010), and with neither the player is
    prompted (bs.c:10016-10018): the cursor stays on that entry. No match left
    (``LEARNSET_NO_MOVE_TO_LEARN`` -> ``SEQ_GET_EXP_GAUGE``, bs.c:10012-10013): the work is done
    at the Pokemon's level, else the next level's scan restarts at index 0 (bs.c:9882)."""
    slot = _learner(state)
    level = state.learn_level[slot].astype(_I32)
    cursor = state.learn_cursor[slot].astype(_I32)
    row = _row(tables, state, slot)
    levels = tables.learnset_level[row].astype(_I32)
    hit = (levels == level) & (jnp.arange(levels.shape[0]) >= cursor)
    found = jnp.any(hit)
    at = jnp.argmax(hit)
    move = tables.learnset_move[row, at]
    moves, _ = _party_moves(state, slot)
    stop = (moves == 0) | (moves == move)
    k = jnp.argmax(stop)
    learned = found & jnp.any(stop) & (moves[k] == 0)
    prompt = found & ~jnp.any(stop)
    done = level >= state.level[0, slot].astype(_I32)
    new_level = jnp.where(found, level, jnp.where(done, 0, level + 1))
    new_cursor = jnp.where(found, jnp.where(prompt, at, at + 1), 0)
    out = _put_party(state, slot, k, move, tables.move_pp[move], learned)
    out = out._replace(
        learn_level=state.learn_level.at[slot].set(new_level.astype(state.learn_level.dtype)),
        learn_cursor=state.learn_cursor.at[slot].set(new_cursor.astype(state.learn_cursor.dtype)),
    )
    return out, prompt


def advance_learns(tables: Tables, state: BattleState) -> BattleState:
    """The pending learnset scans, slots ascending, up to the first prompt (``needs_learn``)
    or to the end (every ``learn_level`` 0): auto-learns and "already known" entries are
    applied, a prompt stops the scan (``_advance_once``). Deterministic, and a no-op at a
    prompt. One ``lax.while_loop``, run after ``step``, never inside it."""
    width = tables.learnset_level.shape[1]
    # Each step uses up an entry (at most ``width`` per slot over all levels), moves to the
    # next level (fewer than MAX_LEVEL per slot) or stops at a prompt: never reached.
    bound = SLOTS * (width + MAX_LEVEL) + 1

    def cond(carry):
        s, prompt, n = carry
        return ~prompt & needs_learn(s) & (n < bound)

    def body(carry):
        s, _, n = carry
        s, prompt = _advance_once(tables, s)
        return s, prompt, n + 1

    state, _, _ = jax.lax.while_loop(cond, body, (state, jnp.bool_(False), jnp.int32(0)))
    return state


def learn_options(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[5]``: the search's answers to the pending prompt (``needs_learn``, after
    ``advance_learns``). Option ``k < 4`` replaces move slot ``k``; option 4 (``GIVE_UP``)
    gives the move up and is always legal (a "No" to "Make it forget another move?" leads to
    the give-up prompt, bs.c:10076-10077, :10104-10135).

    Replacing slot ``k`` is offered only if all of these hold:

    * its move is not an HM (``CheckSelectedMoveIsHM``, src/battle_sub_menus/battle_party.c:
      1724-1735; ``Item_IsHMMove``, src/item.c:306-315): a game rule;
    * for the player's active learner: it is not the disabled or the encored move (ruling D2:
      the decomp ends Disable and Encore at the end of a turn whose payout replaced it,
      bcp.c:1565-1576, :1591-1602), and not the slot the player selected this turn
      (``learn_pin``, window (a), ruling U1; re-read when its move runs, bcp.c:1891-1895);
    * the offered move is one the engine supports (its effect is not ``UNSUPPORTED``; ruling
      D10). Coverage refuses an unsupported move only where it could be learned without a
      prompt, into an empty slot (``BoxPokemon_AddMove``, src/pokemon.c:3852-3873;
      battle_script.c:10007-10027); as a prompt the player may always give it up.

    A slot the AI has seen is open: ``ai_seen_moves`` holds move ids, so the replaced move's id
    stays in it as in the decomp's list (``TrainerAI_RecordLastMove``,
    src/battle/trainer_ai/trainer_ai.c:2705-2716, G-68) and the AI does not know the new move.
    The locked and the choice-locked move are offered: both are read by id (module
    docstring). The D2, pin and D10 masks drop legal game choices, so plan values at learn
    nodes are achievable lower bounds, except the mid-turn-prompt case (``STATE.md``,
    Experience)."""
    slot = _learner(state)
    moves, _ = _party_moves(state, slot)
    width = tables.learnset_level.shape[1]
    row = _row(tables, state, slot)
    cursor = state.learn_cursor[slot].astype(_I32)
    col = jnp.minimum(cursor, width - 1)
    real = (cursor < width) & (tables.learnset_level[row, col] > 0)
    offered = jnp.where(real, tables.learnset_move[row, col], 0)
    supported = (tables.move_effect[offered] != EffectId.UNSUPPORTED) | ~needs_learn(state)
    ok = (tables.move_is_hm[moves] == 0) & supported
    live = state.moves[0, slot]
    held = (
        (live == state.disabled_move[0])
        | (live == state.encore_move[0])
        | (jnp.arange(MOVES) + 1 == state.learn_pin)
    )
    ok = ok & ~((slot == state.active[0]) & held)
    return jnp.concatenate([ok, jnp.ones(1, bool)])


def learn(tables: Tables, state: BattleState, option) -> BattleState:
    """The answer ``option`` (``learn_options``' indexing) to the pending prompt, then
    ``advance_learns``. Replacing slot ``k`` (``SEQ_GET_EXP_LEARNED_MOVE``,
    bs.c:10162-10181) sets its PP Ups to 0 and ``SetMoveSlot`` gives the new move's base PP
    (src/pokemon.c:3925-3933); giving up changes nothing. Either way the scan goes on at the
    same level from the next entry (``SEQ_GET_EXP_LEARNED_MOVE_WAIT`` and
    ``SEQ_GET_EXP_GIVE_UP_LEARNING_WAIT`` -> ``SEQ_GET_EXP_CHECK_LEARN_MOVE``,
    bs.c:10132-10135, :10183-10187). Every option is applied as given: ``learn_options`` is
    the search's mask, not a game rule here (the converter fits all non-HM options and
    refuses a fitted answer the mask drops, ``tools.trace_to_state._learn_refusal``). Replacing
    a slot the AI has seen keeps the old move's id in ``ai_seen_moves``, as the decomp's list does
    (``TrainerAI_RecordLastMove``, src/battle/trainer_ai/trainer_ai.c:2705-2716, G-68), and the
    AI does not know the new move. A flagged Mimic slot of the active learner (``_masked``)
    takes the move into the party's shadow."""
    slot = _learner(state)
    cursor = state.learn_cursor[slot]
    move = tables.learnset_move[_row(tables, state, slot), cursor]
    option = jnp.asarray(option, _I32)
    replaced = option < MOVES
    k = jnp.minimum(option, MOVES - 1)
    out = _put_party(state, slot, k, move, tables.move_pp[move], replaced)
    out = out._replace(
        # SEQ_GET_EXP_LEARNED_MOVE sets the slot's PP Ups to 0 (battle_script.c:10169-10170, E1 Batch D).
        pp_ups=state.pp_ups.at[0, slot, k].set(
            jnp.where(replaced, 0, state.pp_ups[0, slot, k]).astype(state.pp_ups.dtype)
        ),
        learn_cursor=state.learn_cursor.at[slot].set((cursor + 1).astype(cursor.dtype)),
    )
    return advance_learns(tables, out)
