"""Experience payout and level-up (K3d-exp, design §1.1-1.4, §3.1): ``pay_exp``.

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
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import Tables
from nuzline.engine.ids import Stat
from nuzline.engine.state import SLOTS, BattleState
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
    foe_species = after.species[1, j].astype(_I32)
    foe_row = tables.species_form_row[foe_species, after.form[1, j]]
    exp_e = tables.species_exp_yield[foe_species].astype(_I32) * after.level[1, j].astype(_I32)
    exp_e = exp_e // 7

    slots = jnp.arange(SLOTS)
    species = after.species[0]
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
    f0 = after.friendship[0].astype(_I32)
    bonus = after.friend_bonus.astype(_I32)
    friendship = jnp.where(leveled, _friendship(f0, new_level - level, bonus, soothe), f0)

    row = tables.species_form_row[species, after.form[0]]
    mods = tables.nature_mods[after.nature[0]]
    base = tables.species_base_stats[row]
    new = calc_stats_jnp(base, after.ivs[0], evs, new_level, mods, species)
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
        friendship=put("friendship", friendship),
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
