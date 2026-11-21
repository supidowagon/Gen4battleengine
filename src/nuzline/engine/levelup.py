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
that differs from the party item (§4.2).

The learnset scan and the prompt run at the payout (E3-B2-21, ruling L1). ``advance_learns``
is called by ``step`` right after each ``pay_exp`` of an in-turn payout site (MoveEnd,
``CheckFieldConditions``, ``CheckMonConditions``; ``step._execute_actions``, ``field_conditions``,
``end_of_turn``), and a prompt that is left pauses the turn there (``BattleState.learn_site``;
the E2 batch 6 pause, ``step.step_events``): the decomp runs the whole ``GetExpTask``, every
level's prompts included, inside the payout, before any later action, check or draw of the turn
(bcp.c:3968, :941, :1296; bs.c:10003-10187). The answer (``learn``) is made at the pause and the
turn resumes as a chance node; the later draws of the turn are unseen when it is made. Pursuit's
payout (``subscript_pursuit.s:106-108``) is taken at the same action's MoveEnd: the pursued
action is over by then (the pursued battler fainted, so its switch or its U-turn's list never
runs), and nothing between them reads the learn state. A payout that is not followed by an
in-turn step before the replacements (``TurnEnd``'s, bcp.c:1846-1848: Future Sight, Perish Song,
the replacement loop) leaves its prompt in the state after ``step`` / ``replace`` (``learn_site``
0); the answer comes before the replacement (``search``).

The state read between the payout and the answer is the decomp's: the pending move of the
learner re-reads its slot when it runs (bcp.c:1889-1895; ``step._execute_actions``), the
end-of-turn Disable and Encore checks run after the answer (bcp.c:1565-1576, :1591-1602), and
the locked and the choice-locked move are read by id (bcp.c:1885-1888; ``order.selected_move``,
``actions.locked_slot``, ``actions.choice_locked`` read a lock the battler no longer knows as no
lock, battle_lib.c:2307-2310). So ``learn_options`` has no mask but the game's (an HM, ruling
D10).

Prompts still pending when the battle ends stay in the state (ruling D8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.format import MULTI, format_of, party, side_of
from nuzline.engine.ids import EffectId, Stat
from nuzline.engine.state import MOVES, SLOTS, BattleState, party_held_item
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


def _effort_values(tables: Tables, after: BattleState, paid, foe_row, item, hold):
    """``int32[6, 6]``: the player's EVs after ``BattleScript_CalcEffortValues`` (bs.c:
    10214-10306) for the ``paid`` slots (bs.c:9776-9779, before any level-up check). Yields
    of the fainted Pokemon's form row (``SpeciesData_FromMonForm``, :10224); the receiver's
    party item (:10226-10228). Per stat in ``DECOMP_STAT_ORDER``: stop once the sum is 510
    (:10236-10239); yield plus ``itemPower`` for the matching Power item (:10244-10282);
    ``* 2`` with Pokerus (:10285-10287) and with Macho Brace (:10289-10291); clip the sum to
    510 (:10294-10296), then the stat to 255 (:10299-10301)."""
    evs = after.evs[0].astype(_I32)
    yields = tables.species_ev_yield[foe_row].astype(_I32)
    power = tables.item_hold_param[item].astype(_I32)
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


def pay_exp(tables: Tables, before: BattleState, after: BattleState, battler=None) -> BattleState:
    """``after`` with the exp payout for an opponent faint between ``before`` and ``after``
    (``after`` is the state after the hook's faint check); unchanged when nothing fainted.
    A level-up of battler 2's active reloads it into battler 2 (``expBattler = 2``,
    battle_script.c:9716-9720, :9860-9864; D3-12). ``battler``: a static Python int, the payout
    for that battler alone (see ``pay_exp_leveled``); ``None`` is the singles payout."""
    return pay_exp_leveled(tables, before, after, battler)[0]


def pay_exp_leveled(
    tables: Tables, before: BattleState, after: BattleState, battler=None
) -> tuple[BattleState, jnp.ndarray]:
    """``(pay_exp(...), bool[6])``: the state and the player slots that gained a level.

    ``battler`` (D3-8): ``None`` is today's payout (the fainted slot of party 1, the lowest
    when several). A static Python int is the payout of that battler alone: nothing for a
    player battler (``CalcExpGain`` pays only a SOLO_ENEMY, script.c:2196); for an enemy
    battler ``b`` the fainted Pokemon is party ``party(fmt, b)`` slot ``active[b]``
    (script.c:2216-2217; ``AnyExpPayout`` pays one battler bit per call, bcp.c:4449-4469).

    Trigger (design §1.1): an opponent party slot ``j`` with ``before.hp > 0`` and
    ``after.hp == 0`` (the per-slot test of ``step._faint_check``): the payout is keyed on the
    fainted battler (bs.c:1800) and reads ``battleMons[faintedMon]`` (bs.c:2216-2217, 9776-9779),
    which holds a Pokemon Roar dragged in. Singles have at most one such slot per hook; the
    lowest is taken, as ``AnyExpPayout`` takes the lowest flag (bcp.c:4452-4458).

    ``CalcExpGain`` (bs.c:2196-2238; the engine plays only trainer singles, so every opponent
    faint pays): ``E = yield[species] * level / 7`` (yield by species, bs.c:2216-2217);
    ``n`` / ``s`` count the player's party Pokemon with a species and HP that are in
    ``exp_mask`` (the entry of the fainted enemy battler, ``(battler >> 1) & 1``, at B = 4;
    D3-8b) / hold an Exp. Share (party item, bs.c:2201-2212). With ``s > 0``:
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
    if battler is None:
        fp = 1
        fainted = (before.hp[1] > 0) & (after.hp[1] == 0)
        paying = jnp.any(fainted)
        j = jnp.argmax(fainted)
    else:
        if side_of(battler) == 0:
            return after, jnp.zeros(SLOTS, dtype=bool)
        fp = party(format_of(after), battler)
        j = after.active[battler]
        paying = (before.hp[fp, j] > 0) & (after.hp[fp, j] == 0)
    foe_species = after.species[fp, j].astype(_I32)  # a Transformed foe's copy, bs.c:2216
    foe_row = tables.species_form_row[foe_species, after.form[fp, j]]
    exp_e = tables.species_exp_yield[foe_species].astype(_I32) * after.level[fp, j].astype(_I32)
    exp_e = exp_e // 7

    slots = jnp.arange(SLOTS)
    species, party_form, party_ivs = party_identity(after)
    hp = after.hp[0].astype(_I32)
    level = after.level[0].astype(_I32)
    alive = (species != 0) & (hp > 0)
    # sideGetExpMask[(faintedMon >> 1) & 1] (bs.c:2205, :9700, :9709, :9741; D3-8b): one entry
    # per enemy battler at B = 4, a scalar at B = 2.
    entry = None if after.exp_mask.ndim == 0 else (battler >> 1) & 1
    mask = after.exp_mask if entry is None else after.exp_mask[entry]
    in_mask = (jnp.right_shift(mask.astype(_I32), slots) & 1) == 1
    item = party_held_item(after, 0)  # the party row, read once (D3-8b)
    hold = tables.item_hold_effect[item]
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
    evs = _effort_values(tables, after, paid, foe_row, item, hold)

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
    # BattleSystem_ReloadPokemon runs for the battler whose ``selectedPartySlot`` is the leveled
    # slot: battler 2 first, in DOUBLES and TWO_TRAINERS (BATTLE_TYPE_AI excluded), else battler 0
    # (battle_script.c:9716-9720, :9860-9864; D3-12 owed from D3-8b). ``reloaders`` are the
    # player battlers of party 0 that can be that battler (MULTI's battler 2 draws from party 2).
    reloaders = _reloaders(format_of(after))

    def put(field, side0):
        full = getattr(after, field)
        return full.at[0].set(side0.astype(full.dtype))

    out = after._replace(
        level=put("level", new_level),
        hp=put("hp", jnp.where(leveled, new_hp, hp)),
        max_hp=put("max_hp", jnp.where(leveled, new[:, Stat.HP], old_max)),
        stats=put("stats", jnp.where(leveled[:, None], new[:, Stat.ATK :], after.stats[0])),
        # The level-up's ReloadPokemon re-reads the party value into the battler (F-15).
        friendship=put("friendship", jnp.where(leveled, friendship, after.friendship[0])),
        party_friendship=put("party_friendship", friendship),
        evs=put("evs", evs),
        exp=exp.astype(after.exp.dtype),
        exp_mask=(
            jnp.where(paying, 0, after.exp_mask)
            if entry is None
            else after.exp_mask.at[entry].set(jnp.where(paying, 0, mask))  # bs.c:10191
        ).astype(after.exp_mask.dtype),
        learn_level=jnp.where(marked, level + 1, after.learn_level).astype(after.learn_level.dtype),
        learn_cursor=jnp.where(marked, 0, after.learn_cursor).astype(after.learn_cursor.dtype),
    )
    for p in reloaders:
        reloaded = leveled[after.active[p]]
        out = out._replace(
            # BattleSystem_ReloadPokemon (battle_lib.c:169-174) reloads the active battler's
            # Attack and Defense unswapped and keeps MOVE_EFFECT_POWER_TRICK (B12-10).
            power_trick_desync=out.power_trick_desync.at[p].set(
                jnp.where(reloaded, after.power_trick[p], after.power_trick_desync[p]).astype(
                    after.power_trick_desync.dtype
                )
            ),
            # A Transformed active Pokemon: the reload gives the battler its own new stats
            # (:169-174) and the party's shadow too.
            shadow_stats=out.shadow_stats.at[p].set(
                jnp.where(
                    (after.transformed[p] != 0) & reloaded,
                    new[after.active[p], Stat.ATK :],
                    after.shadow_stats[p],
                ).astype(after.shadow_stats.dtype)
            ),
        )
    return out, leveled


# --- the learnset resolve (design §1.5, §3.2) ---------------------------------------------------

GIVE_UP = MOVES  # learn option 4 gives the move up; options 0-3 replace that move slot
LEARN_OPTIONS = MOVES + 1

# ``BattleState.learn_site``: where the turn paused for a learn prompt (module docstring). The
# payout sites of the decomp (E3-B2-21): MoveEnd (bcp.c:3968; Pursuit's, subscript_pursuit.s:106,
# is taken here), the head of CheckFieldConditions (:941), the head of CheckMonConditions' next
# re-entry (:1296; the pause is after the battler block whose check fainted the foe).
LEARN_MOVE_END, LEARN_FIELD, LEARN_MON = 1, 2, 3


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
    species, form, ivs = state.species[0], state.form[0], state.ivs[0]
    # At B = 4 the active slot of either party-0 battler (``_reloaders``; D3-12).
    for p in _reloaders(format_of(state)):
        swapped = (state.transformed[p] != 0) & (slots == state.active[p])
        species = jnp.where(swapped, state.shadow_species[p], species)
        form = jnp.where(swapped, state.shadow_form[p], form)
        ivs = jnp.where(swapped[:, None], state.shadow_ivs[p][None, :], ivs)
    return species, form, ivs


def _reloaders(fmt) -> tuple[int, ...]:
    """The player battlers of party 0 that ``BattleSystem_ReloadPokemon`` can reload as
    ``expBattler``: battler 2 when its ``selectedPartySlot`` is the slot, in a doubles battle
    without ``BATTLE_TYPE_AI`` (DOUBLES, TWO_TRAINERS), else battler 0 (battle_script.c:9716-9720,
    :9860-9864, :10173-10178; D3-12). MULTI's battler 2 draws from party 2."""
    return (0, 2) if fmt.B == 4 and fmt != MULTI else (0,)


def _exp_battler(state: BattleState, slot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(active, battler)``: whether player party ``slot`` is the active Pokemon of a reloader
    (``_reloaders``), and that battler (0 when it is none)."""
    slot = jnp.asarray(slot)
    on, battler = jnp.bool_(False), jnp.int32(0)
    for p in _reloaders(format_of(state)):
        here = slot == state.active[p]
        on, battler = on | here, jnp.where(here, p, battler)
    return on, battler


def _masked(state: BattleState, slot) -> jnp.ndarray:
    """``bool[4]``: the move slots of player party ``slot`` whose battler value is not the party's:
    ``mimic_slots`` of the active Pokemon (G-68). The learn reads and writes the party Pokemon
    (``Pokemon_SetMoveSlot``, battle_script.c:10168-10176; ``BoxPokemon_AddMove``, pokemon.c:3852),
    and ``BattleSystem_ReloadPokemon`` reloads the unflagged slots from it (battle_lib.c:183-191)."""
    on, p = _exp_battler(state, slot)
    bits = (state.mimic_slots[p].astype(_I32) >> jnp.arange(MOVES)) & 1
    return (bits == 1) & on


def _party_moves(state: BattleState, slot) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(moves, pp)`` of player party ``slot`` as the party Pokemon holds them: the shadow of a
    flagged slot (``_masked``), else the slot's own."""
    masked = _masked(state, slot)
    p = _exp_battler(state, slot)[1]
    moves = jnp.where(masked, state.shadow_moves[p], state.moves[0, slot])
    pp = jnp.where(masked, state.shadow_pp[p], state.pp[0, slot])
    return moves, pp


def _put_party(state: BattleState, slot, k, move, pp, apply) -> BattleState:
    """Party ``slot``'s move slot ``k`` becomes ``move`` with ``pp`` where ``apply``: into the
    shadow when the battler's slot is flagged (``_masked``), else into the slot (the battler reload
    then shows it)."""
    shadowed = _masked(state, slot)[k]
    live, shadow = apply & ~shadowed, apply & shadowed
    p = _exp_battler(state, slot)[1]

    def put(arr, index, value, cond):
        return arr.at[index].set(jnp.where(cond, value, arr[index]).astype(arr.dtype))

    return state._replace(
        moves=put(state.moves, (0, slot, k), move, live),
        pp=put(state.pp, (0, slot, k), pp, live),
        shadow_moves=put(state.shadow_moves, (p, k), move, shadow),
        shadow_pp=put(state.shadow_pp, (p, k), pp, shadow),
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


def learn_prompt(tables: Tables, state: BattleState) -> tuple[BattleState, jnp.ndarray]:
    """``(state, prompt)``: ``advance_learns`` at a payout site, and whether a prompt is left in a
    battle that goes on (D8: none once it is over). ``step`` pauses the turn on a prompt."""
    state = advance_learns(tables, state)
    return state, needs_learn(state) & (state.over == 0)


def learn_options(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[5]``: the search's answers to the pending prompt (``needs_learn``, after
    ``advance_learns``). Option ``k < 4`` replaces move slot ``k``; option 4 (``GIVE_UP``)
    gives the move up and is always legal (a "No" to "Make it forget another move?" leads to
    the give-up prompt, bs.c:10076-10077, :10104-10135).

    Replacing slot ``k`` is offered only if both hold:

    * its move is not an HM (``CheckSelectedMoveIsHM``, src/battle_sub_menus/battle_party.c:
      1724-1735; ``Item_IsHMMove``, src/item.c:306-315): a game rule;
    * the offered move is one the engine supports (its effect is not ``UNSUPPORTED``; ruling
      D10). Coverage refuses an unsupported move only where it could be learned without a
      prompt, into an empty slot (``BoxPokemon_AddMove``, src/pokemon.c:3852-3873;
      battle_script.c:10007-10027); as a prompt the player may always give it up.

    No move is masked for the active learner (E3-B2-21): the prompt is answered at the payout,
    where the decomp reads the new move list (module docstring), so the selected, the disabled,
    the encored, the locked and the choice-locked move are all offered. A slot the AI has seen is
    open too: ``ai_seen_moves`` holds move ids, so the replaced move's id stays in it as in the
    decomp's list (``TrainerAI_RecordLastMove``, src/battle/trainer_ai/trainer_ai.c:2705-2716,
    G-68) and the AI does not know the new move. The D10 mask drops legal game choices, so plan
    values at learn nodes are achievable lower bounds."""
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
