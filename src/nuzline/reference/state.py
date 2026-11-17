"""The reference engine's battle state: ``BattleState``'s fields as Python ints and
lists, grouped the way the decomp groups them.

* ``RefPokemon``: one party slot (the decomp's party ``Pokemon`` plus the parts of
  ``BattleMon`` that belong to the Pokemon: HP, stats, status, PP, held item).
* ``RefSide``: a side's party, its active slot and the active battler's volatile
  state (``BattleMon.statBoosts``, ``statusVolatile``, ``moveEffectsData``), plus
  the side's ``SideConditions``.
* ``RefState``: two sides and the field (``FieldConditions``, ``totalTurns``, the
  battle result, the AI's knowledge), and the player party's experience fields and
  ``sideGetExpMask`` (``exp_mask``).

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
    ability_form: int  # the form whose personal row ability_slot indexes (engine.state)
    item: int
    item_consumed: int
    knocked_off: int  # the slot's knockedOffItemsMask bit (B10-6)
    moves: list[int]
    pp: list[int]
    status: int
    status_counter: int
    gender: int
    friendship: int
    ivs: list[int]
    evs: list[int]  # ids.Stat order
    nature: int


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
    encore_move: int  # B10-9
    encore_slot: int
    encore_turns: int
    locked_move: int
    bide_turns: int
    bide_damage: int
    curled: int
    mud_sport: int
    aqua_ring: int
    infatuated: int
    destiny_bond: int
    uproar: int  # VOLATILE_CONDITION_UPROAR turns left (and the battler's FIELD_CONDITION bit)
    curse: int
    perish_song: int
    perish_song_turns: int
    yawn: int
    lock_on: int
    lock_on_target: int
    ingrain: int  # MOVE_EFFECT_INGRAIN
    mean_look: int  # VOLATILE_CONDITION_MEAN_LOOK
    mean_look_target: int  # meanLookTarget as 1 + battler, 0 = none
    imprison: int  # MOVE_EFFECT_IMPRISON
    imprisoned: int  # MOVE_EFFECT_IMPRISONED
    roosting: int  # turnFlags.roosting while it can be read (engine.state docstring)
    can_unburden: int  # MoveEffectsData.canUnburden
    minimized: int  # MOVE_EFFECT_MINIMIZE
    semi_invulnerable: int  # ids.SemiKind: MOVE_EFFECT_SEMI_INVULNERABLE's bit (B10-12)
    focus_energy: int  # VOLATILE_CONDITION_FOCUS_ENERGY (B12-4)
    power_trick: int  # MOVE_EFFECT_POWER_TRICK (B12-10)
    power_trick_desync: int  # the Attack / Defense swap XOR power_trick (B12-10)
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
    future_move: int
    future_attacker: int  # 1 + battler, 0 = none
    future_sight: int  # SIDE_CONDITION_FUTURE_SIGHT
    trainer_items: list[int]
    trainer_item_count: int
    qc_roll: int


@dataclass
class RefState:
    sides: list[RefSide]
    weather: int
    weather_timer: int
    trick_room: int
    gravity: int
    war_swaps: int  # a weather war's open monSpeedOrder (engine.state docstring)
    entry_pending: list[int]  # a list newcomer's switch-in states wait (engine.state docstring)
    move_prev: int
    terrain: int
    turn: int
    over: int
    winner: int
    player_faints: int
    ai_seen_moves: list[list[int]]
    ai_item_known: list[int]  # battlerHeldItems of the player's slots (item ids)
    ai_ability_known: list[int]
    # Experience (K3d-exp): the player party's, per slot, and sideGetExpMask[0].
    exp: list[int]
    exp_mask: int
    exp_boost: list[int]
    pokerus: list[int]
    friend_bonus: list[int]
    learn_level: list[int]
    learn_cursor: list[int]
    learn_pin: int
    exp_refused: int
    deaths: list[int]
    faint_budget: int
    protected: list[int]
    replace_mul: int
    # E2 batch 6: the mid-turn switch decision (engine.state docstring): 0 between turns.
    pivot_battler: int
    ts_cursor: int
    ts_acted: list[int]
    ts_order: list[int]
    ts_actions: list[int]
    ts_move: list[int]
    ts_struggling: list[int]
    ts_locked: list[int]
    ts_item: list[int]
    ts_custap: list[int]
    ts_claw: list[int]
    ts_pursued: int
    ts_protect_ok: list[int]
    ts_taken: list[list[int]]
    ts_dragged: list[int]
    ts_move_failed: list[int]
    ts_used: int
    ts_damaged: int
    ts_dealt: int
    ts_alive: list[int]
    ts_power_mul: int
    ts_switched: list[int]
    ts_pick_mul: int


# BattleState field names by level.
MON_FIELDS = tuple(f.name for f in fields(RefPokemon))
SIDE_FIELDS = tuple(f.name for f in fields(RefSide) if f.name != "party")
FIELD_FIELDS = tuple(f.name for f in fields(RefState) if f.name != "sides")
