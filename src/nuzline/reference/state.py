"""The reference engine's battle state: ``BattleState``'s fields as Python ints and
lists, grouped the way the decomp groups them.

* ``RefPokemon``: one party slot (the decomp's party ``Pokemon`` plus the parts of
  ``BattleMon`` that belong to the Pokemon: HP, stats, status, PP, held item).
* ``RefSide``: a side's party, its active slot and the active battler's volatile
  state (``BattleMon.statBoosts``, ``statusVolatile``, ``moveEffectsData``), plus
  the side's ``SideConditions``.
* ``RefState``: two sides and the field (``FieldConditions``, ``totalTurns``, the
  battle result, the AI's knowledge).

Every encoding is ``BattleState``'s (``nuzline.engine.state``): stages 0 = neutral,
``stats`` is Atk, Def, SpA, SpD, Spe, ids from ``nuzline.engine.ids``. Side 0 is the
player.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class RefPokemon:
    species: int
    form: int
    level: int
    hp: int
    max_hp: int
    stats: list[int]  # Atk, Def, SpA, SpD, Spe
    types: list[int]
    ability: int
    ability_slot: int
    item: int
    item_consumed: int
    moves: list[int]
    pp: list[int]
    status: int
    status_counter: int
    gender: int
    friendship: int
    ivs: list[int]


@dataclass
class RefSide:
    party: list[RefPokemon]
    active: int
    stages: list[int]  # index ids.Stat, 0 = neutral
    confusion: int
    leech_seed: int
    substitute_hp: int
    taunt: int
    trapped: int
    trap_move: int
    flinched: int
    choice_lock: int
    last_move: int
    protect_count: int
    rollout_count: int
    charge_turn: int
    recharge: int
    pursuit_target: int
    water_sport: int
    fake_out_turn: int
    move_hit: int
    foresight: int
    disabled_move: int
    disable_turns: int
    locked_move: int
    bide_turns: int
    bide_damage: int
    curled: int
    mud_sport: int
    spikes: int
    toxic_spikes: int
    stealth_rock: int
    reflect: int
    light_screen: int
    safeguard: int
    mist: int
    lucky_chant: int
    tailwind: int
    wish_timer: int
    wish_hp: int
    future_timer: int
    future_damage: int
    trainer_items: list[int]
    trainer_item_count: int


@dataclass
class RefState:
    sides: list[RefSide]
    weather: int
    weather_timer: int
    trick_room: int
    gravity: int
    turn: int
    over: int
    winner: int
    player_faints: int
    ai_seen_moves: list[list[int]]
    ai_ability_known: list[int]
    deaths: list[int]
    faint_budget: int
    protected: list[int]
    replace_mul: int


# BattleState field names by level.
MON_FIELDS = tuple(f.name for f in fields(RefPokemon))
SIDE_FIELDS = tuple(f.name for f in fields(RefSide) if f.name != "party")
FIELD_FIELDS = tuple(f.name for f in fields(RefState) if f.name != "sides")
