"""The battle state pytree and its construction from two parties.

``BattleState`` is unbatched: side axis 0 is the player, 1 the opponent; slot axis
is party order. Every field is ``int8``/``int16`` storage (spec §4.2). The field
list is ticked off against the decomp's battle structs in ``STATE.md``.

Encodings:

* ``species``/``moves``/``item``/``item_consumed``/``trainer_items``/``trap_move``/
  ``choice_lock``/``last_move``/``move_hit`` hold decomp ids (0 = none); ``ability``
  holds an ``ids.AbilityId`` (from ``tables.species_abilities``) and ``ability_slot``
  the species slot (0/1) it came from, so ``tables.species_abilities_decomp[species,
  ability_slot]`` is the decomp ability id; ``status`` an ``ids.Status``.
* ``stats[..., 0..4]`` are Atk, Def, SpA, SpD, Spe (``ids.Stat`` minus one); HP is
  ``max_hp``. ``ivs`` are in ``ids.Stat`` order.
* ``stages[side, ids.Stat]`` are the active Pokemon's stat stages, 0 = neutral
  (the decomp stores 6 = neutral); index 0 (HP) is unused.
* ``weather`` is an ``ids.Weather``; ``weather_timer == 0`` means permanent.
* ``winner``: 0 none, 1 player lost (or any player Pokemon fainted), 2 player won.
* ``turn`` is the decomp's ``totalTurns`` while a turn is played and at each decision;
  ``step`` adds 1 at the end, so a faint replacement (``step.replace``, run after
  ``step``) sees ``turn`` one ahead of the decomp's, which increments after it
  (battle_controller_player.c:1844-1852).
* ``fake_out_turn`` is ``fakeOutTurnNumber``: 0 for the leads, ``totalTurns + 1`` after
  every switch-in and faint; the AI's ``LoadIsFirstTurnInBattle`` (trainer_ai.c:2487)
  is ``fake_out_turn >= turn``.
* ``move_hit`` is ``moveHit``: the last move aimed at the side's active battler since its
  own last action ended (see ``step``).
* ``foresight``, ``disabled_move``/``disable_turns``, ``locked_move`` (0 = not locked),
  ``bide_turns``/``bide_damage`` (``bide_damage`` is ``storedDamage`` only while
  ``bide_turns > 0``, else 0), ``curled`` and ``mud_sport`` are the active battler's
  matching volatile / move-effect state; all are reset like the other volatiles
  (``step._VOLATILE``).
* ``trainer_item_count`` is ``aiContext.trainerItemCounts``: the bag's size at battle
  start, never decremented.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.ids import Stat, Weather
from nuzline.engine.stats import calc_stats

SIDES, SLOTS, MOVES = 2, 6, 4
MAX_DEATH_BUDGET = 3  # the ladder's range: k = 0..3 deaths allowed (spec §4)
BAG_SLOTS = 4  # MAX_TRAINER_ITEMS: trainer_items[side] holds the AI's bag
STAGE_MIN, STAGE_MAX = -6, 6
_STAGE_KEYWORDS = {
    "atk": Stat.ATK,
    "def_": Stat.DEF,
    "spa": Stat.SPA,
    "spd": Stat.SPD,
    "spe": Stat.SPE,
    "acc": Stat.ACC,
    "eva": Stat.EVA,
}


class BattleState(NamedTuple):
    # per side x slot [2,6]
    species: jnp.ndarray  # int16
    form: jnp.ndarray  # int8
    level: jnp.ndarray  # int8
    hp: jnp.ndarray  # int16
    max_hp: jnp.ndarray  # int16
    stats: jnp.ndarray  # int16[2,6,5]: Atk, Def, SpA, SpD, Spe
    types: jnp.ndarray  # int8[2,6,2]
    ability: jnp.ndarray  # int16, ids.AbilityId
    ability_slot: jnp.ndarray  # int8, species ability slot 0/1
    item: jnp.ndarray  # int16
    item_consumed: jnp.ndarray  # int16, id of the last item consumed
    moves: jnp.ndarray  # int16[2,6,4]
    pp: jnp.ndarray  # int8[2,6,4]
    status: jnp.ndarray  # int8, ids.Status
    status_counter: jnp.ndarray  # int8, sleep turns / toxic counter
    gender: jnp.ndarray  # int8, 0 male, 1 female, 2 none
    friendship: jnp.ndarray  # int16
    ivs: jnp.ndarray  # int8[2,6,6], ids.Stat order
    # per side, active Pokemon [2]
    active: jnp.ndarray  # int8, party slot
    stages: jnp.ndarray  # int8[2,8], index ids.Stat, 0 = neutral
    confusion: jnp.ndarray  # int8, turns left
    leech_seed: jnp.ndarray  # int8
    substitute_hp: jnp.ndarray  # int16
    taunt: jnp.ndarray  # int8, turns left
    trapped: jnp.ndarray  # int8, turns left
    trap_move: jnp.ndarray  # int16
    flinched: jnp.ndarray  # int8
    choice_lock: jnp.ndarray  # int16
    last_move: jnp.ndarray  # int16
    protect_count: jnp.ndarray  # int8
    rollout_count: jnp.ndarray  # int8
    charge_turn: jnp.ndarray  # int8
    recharge: jnp.ndarray  # int8
    pursuit_target: jnp.ndarray  # int8, reserved: no mechanic sets or reads it yet (only cleared)
    water_sport: jnp.ndarray  # int8
    fake_out_turn: jnp.ndarray  # int16, fakeOutTurnNumber
    move_hit: jnp.ndarray  # int16, moveHit: decomp move id, 0 = none
    foresight: jnp.ndarray  # int8, VOLATILE_CONDITION_FORESIGHT
    disabled_move: jnp.ndarray  # int16, MoveEffectsData.disabledMove: move id, 0 = none
    disable_turns: jnp.ndarray  # int8, MoveEffectsData.disabledTurns (hidden)
    locked_move: jnp.ndarray  # int16, moveLockedInto while VOLATILE_CONDITION_MOVE_LOCKED
    bide_turns: jnp.ndarray  # int8, VOLATILE_CONDITION_BIDE turns, 0-2
    bide_damage: jnp.ndarray  # int16, storedDamage while Biding (hidden), else 0
    curled: jnp.ndarray  # int8, VOLATILE_CONDITION_DEFENSE_CURL
    mud_sport: jnp.ndarray  # int8, MOVE_EFFECT_MUD_SPORT in moveEffectsMask
    # per side [2]
    spikes: jnp.ndarray  # int8, layers
    toxic_spikes: jnp.ndarray  # int8, layers
    stealth_rock: jnp.ndarray  # int8
    reflect: jnp.ndarray  # int8, turns left
    light_screen: jnp.ndarray  # int8
    safeguard: jnp.ndarray  # int8
    mist: jnp.ndarray  # int8
    lucky_chant: jnp.ndarray  # int8
    tailwind: jnp.ndarray  # int8
    wish_timer: jnp.ndarray  # int8
    wish_hp: jnp.ndarray  # int16
    future_timer: jnp.ndarray  # int8
    future_damage: jnp.ndarray  # int16
    trainer_items: jnp.ndarray  # int16[2,4], bag item ids (aiContext.trainerItems)
    trainer_item_count: jnp.ndarray  # int8, bag size at battle start (trainerItemCounts)
    # field (scalars)
    weather: jnp.ndarray  # int8, ids.Weather
    weather_timer: jnp.ndarray  # int8, 0 = permanent
    trick_room: jnp.ndarray  # int8
    gravity: jnp.ndarray  # int8
    turn: jnp.ndarray  # int16
    over: jnp.ndarray  # int8
    winner: jnp.ndarray  # int8
    player_faints: jnp.ndarray  # int8
    # AI knowledge of the player
    ai_seen_moves: jnp.ndarray  # int8[6,4], player slot x move slot
    ai_ability_known: jnp.ndarray  # int8[6]
    # faint budget (search data; the engine only records deaths and never reads k)
    deaths: jnp.ndarray  # int8[6], 1 where the player's party slot fainted this battle
    faint_budget: jnp.ndarray  # int8, k: the deaths the search allows
    protected: jnp.ndarray  # int8[6], 1 where the player's slot must not faint
    replace_mul: jnp.ndarray  # int8, the turn's powerMul while a player replacement is pending

    def replace_hp(self, side: int, slot: int, hp: int) -> BattleState:
        return self._replace(hp=self.hp.at[side, slot].set(hp))

    def with_stages(self, side: int, **stages: int) -> BattleState:
        """Set (not add) the active Pokemon's stages; keywords atk, def_, spa, spd,
        spe, acc, eva."""
        new = self.stages
        for key, value in stages.items():
            if key not in _STAGE_KEYWORDS:
                raise TypeError(f"unknown stage {key!r}; use one of {sorted(_STAGE_KEYWORDS)}")
            if not STAGE_MIN <= value <= STAGE_MAX:
                raise ValueError(f"stage {key}={value} outside [{STAGE_MIN}, {STAGE_MAX}]")
            new = new.at[side, _STAGE_KEYWORDS[key]].set(value)
        return self._replace(stages=new)

    def with_pp(self, side: int, slot: int, pp_list: Sequence[int]) -> BattleState:
        return self._replace(pp=self.pp.at[side, slot].set(jnp.asarray(pp_list, jnp.int8)))

    def with_status(self, side: int, slot: int, status: int, counter: int = 0) -> BattleState:
        return self._replace(
            status=self.status.at[side, slot].set(status),
            status_counter=self.status_counter.at[side, slot].set(counter),
        )

    def with_budget(self, k: int = 0, protected: Sequence[int] = ()) -> BattleState:
        """Set the faint budget ``k`` (0..``MAX_DEATH_BUDGET``) and the protected player
        slots: each must hold a living Pokemon (not empty, not fainted)."""
        if not 0 <= k <= MAX_DEATH_BUDGET:
            raise ValueError(f"faint budget {k} outside [0, {MAX_DEATH_BUDGET}]")
        mask = np.zeros(SLOTS, np.int8)
        for slot in protected:
            if not 0 <= slot < SLOTS:
                raise ValueError(f"protected slot {slot} outside [0, {SLOTS - 1}]")
            if int(self.species[0, slot]) == 0:
                raise ValueError(f"protected slot {slot} is empty")
            if int(self.hp[0, slot]) == 0:
                raise ValueError(f"protected slot {slot} has already fainted")
            mask[slot] = 1
        return self._replace(
            faint_budget=jnp.asarray(k, jnp.int8), protected=jnp.asarray(mask, jnp.int8)
        )


# Shape and storage dtype of every field, in BattleState order.
_SLOT, _SIDE, _FIELD = (SIDES, SLOTS), (SIDES,), ()
LAYOUT: dict[str, tuple[tuple[int, ...], type]] = {
    "species": (_SLOT, np.int16),
    "form": (_SLOT, np.int8),
    "level": (_SLOT, np.int8),
    "hp": (_SLOT, np.int16),
    "max_hp": (_SLOT, np.int16),
    "stats": ((*_SLOT, 5), np.int16),
    "types": ((*_SLOT, 2), np.int8),
    "ability": (_SLOT, np.int16),
    "ability_slot": (_SLOT, np.int8),
    "item": (_SLOT, np.int16),
    "item_consumed": (_SLOT, np.int16),
    "moves": ((*_SLOT, MOVES), np.int16),
    "pp": ((*_SLOT, MOVES), np.int8),
    "status": (_SLOT, np.int8),
    "status_counter": (_SLOT, np.int8),
    "gender": (_SLOT, np.int8),
    "friendship": (_SLOT, np.int16),
    "ivs": ((*_SLOT, 6), np.int8),
    "active": (_SIDE, np.int8),
    "stages": ((SIDES, 8), np.int8),
    "confusion": (_SIDE, np.int8),
    "leech_seed": (_SIDE, np.int8),
    "substitute_hp": (_SIDE, np.int16),
    "taunt": (_SIDE, np.int8),
    "trapped": (_SIDE, np.int8),
    "trap_move": (_SIDE, np.int16),
    "flinched": (_SIDE, np.int8),
    "choice_lock": (_SIDE, np.int16),
    "last_move": (_SIDE, np.int16),
    "protect_count": (_SIDE, np.int8),
    "rollout_count": (_SIDE, np.int8),
    "charge_turn": (_SIDE, np.int8),
    "recharge": (_SIDE, np.int8),
    "pursuit_target": (_SIDE, np.int8),
    "water_sport": (_SIDE, np.int8),
    "fake_out_turn": (_SIDE, np.int16),
    "move_hit": (_SIDE, np.int16),
    "foresight": (_SIDE, np.int8),
    "disabled_move": (_SIDE, np.int16),
    "disable_turns": (_SIDE, np.int8),
    "locked_move": (_SIDE, np.int16),
    "bide_turns": (_SIDE, np.int8),
    "bide_damage": (_SIDE, np.int16),
    "curled": (_SIDE, np.int8),
    "mud_sport": (_SIDE, np.int8),
    "spikes": (_SIDE, np.int8),
    "toxic_spikes": (_SIDE, np.int8),
    "stealth_rock": (_SIDE, np.int8),
    "reflect": (_SIDE, np.int8),
    "light_screen": (_SIDE, np.int8),
    "safeguard": (_SIDE, np.int8),
    "mist": (_SIDE, np.int8),
    "lucky_chant": (_SIDE, np.int8),
    "tailwind": (_SIDE, np.int8),
    "wish_timer": (_SIDE, np.int8),
    "wish_hp": (_SIDE, np.int16),
    "future_timer": (_SIDE, np.int8),
    "future_damage": (_SIDE, np.int16),
    "trainer_items": ((SIDES, BAG_SLOTS), np.int16),
    "trainer_item_count": (_SIDE, np.int8),
    "weather": (_FIELD, np.int8),
    "weather_timer": (_FIELD, np.int8),
    "trick_room": (_FIELD, np.int8),
    "gravity": (_FIELD, np.int8),
    "turn": (_FIELD, np.int16),
    "over": (_FIELD, np.int8),
    "winner": (_FIELD, np.int8),
    "player_faints": (_FIELD, np.int8),
    "ai_seen_moves": ((SLOTS, MOVES), np.int8),
    "ai_ability_known": ((SLOTS,), np.int8),
    "deaths": ((SLOTS,), np.int8),
    "faint_budget": (_FIELD, np.int8),
    "protected": ((SLOTS,), np.int8),
    "replace_mul": (_FIELD, np.int8),
}
if tuple(LAYOUT) != BattleState._fields:
    raise RuntimeError("LAYOUT must list every BattleState field, in order")


def _fill_slot(a: dict[str, np.ndarray], tables: Tables, side: int, slot: int, mon: PokemonSpec):
    species = mon.species
    if mon.stats is not None:
        if len(mon.stats) != 6:
            raise ValueError(f"species {species}: stats needs 6 values, got {len(mon.stats)}")
        stats = np.asarray(mon.stats)
    else:
        stats = calc_stats(
            np.asarray(tables.species_base_stats[species]),
            np.asarray(mon.ivs),
            np.asarray(mon.evs),
            mon.level,
            np.asarray(tables.nature_mods[mon.nature]),
            species=species,
        )
    max_hp = int(stats[Stat.HP])
    hp = max_hp if mon.current_hp is None else mon.current_hp
    if not 0 <= hp <= max_hp:
        raise ValueError(f"species {species}: current_hp {hp} outside [0, {max_hp}]")
    move_pp = np.asarray(tables.move_pp)
    a["species"][side, slot] = species
    a["form"][side, slot] = mon.form
    a["level"][side, slot] = mon.level
    a["hp"][side, slot] = hp
    a["max_hp"][side, slot] = max_hp
    a["stats"][side, slot] = stats[Stat.ATK : Stat.SPE + 1]
    a["types"][side, slot] = np.asarray(tables.species_types[species])
    a["ability"][side, slot] = int(tables.species_abilities[species, mon.ability_slot])
    a["ability_slot"][side, slot] = mon.ability_slot
    a["item"][side, slot] = mon.item
    a["moves"][side, slot] = mon.moves
    a["pp"][side, slot] = [move_pp[m] if m else 0 for m in mon.moves] if mon.pp is None else mon.pp
    a["status"][side, slot] = mon.status
    a["status_counter"][side, slot] = mon.status_counter
    a["gender"][side, slot] = mon.gender
    a["friendship"][side, slot] = mon.friendship
    a["ivs"][side, slot] = mon.ivs


def make_state(
    tables: Tables,
    player: list[PokemonSpec],
    opp: list[PokemonSpec],
    start_condition: int = Weather.NONE,
    opp_items: Sequence[int] = (),
    leads: tuple[int, int] = (0, 0),
) -> BattleState:
    """Build the turn-0 state: stats via ``calc_stats``, empty slots all zero
    (``species = 0, hp = 0``), the leads in party slots ``leads`` (player, opponent),
    no volatiles, hazards or screens, and
    ``weather = start_condition`` with a permanent timer (0). The leads'
    switch-in effects then run (``hooks.on_switch_in``, player first; the decomp's
    ``monSpeedOrder`` gives the same result in singles, see ``hooks``).

    ``opp_items`` is the trainer's bag (``Trainer.items``, decomp item ids). As
    ``BattleControllerPlayer_InitAI`` (battle_controller_player.c:4796) does, its
    non-empty items fill ``trainer_items[1]`` from slot 0 in order and their number is
    ``trainer_item_count[1]``; the player's bag stays empty. The leads'
    ``fake_out_turn`` is 0: the battle-start send-out does not run
    ``BattleSystem_UpdateAfterSwitch``, and ``BattleContext`` starts zeroed
    (battle_controller_player.c:180-184)."""
    from nuzline.engine.hooks import on_switch_in  # hooks imports this module

    a = {name: np.zeros(shape, dtype) for name, (shape, dtype) in LAYOUT.items()}
    for side, party in enumerate((player, opp)):
        if not 1 <= len(party) <= SLOTS:
            raise ValueError(f"side {side}: a party has 1-{SLOTS} Pokemon, got {len(party)}")
        for slot, mon in enumerate(party):
            _fill_slot(a, tables, side, slot, mon)
        lead = leads[side]
        if not 0 <= lead < len(party):
            raise ValueError(f"side {side}: lead slot {lead} is not in a party of {len(party)}")
        a["active"][side] = lead
    bag = [item for item in opp_items if item]
    if len(bag) > BAG_SLOTS:
        raise ValueError(f"a trainer's bag holds at most {BAG_SLOTS} items, got {len(bag)}")
    a["trainer_items"][1, : len(bag)] = bag
    a["trainer_item_count"][1] = len(bag)
    a["weather"][...] = Weather(start_condition)
    state = BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})
    for side in (0, 1):
        state = on_switch_in(tables, state, side)
    return state
