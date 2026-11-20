"""The reference engine's battle state: ``BattleState``'s fields as Python ints and
lists, grouped the way the decomp groups them.

* ``RefPokemon``: one party slot (the decomp's party ``Pokemon`` plus the parts of
  ``BattleMon`` that belong to the Pokemon: HP, stats, status, PP, held item).
* ``RefParty``: a party's ``RefPokemon`` list (``mons``), its trainer bag and the HP bitmask of
  the last faint check (``[P]`` fields).
* ``RefBattler``: a battler's active slot and its volatile state (``BattleMon.statBoosts``,
  ``statusVolatile``, ``moveEffectsData``) and per-battler turn state (``[B]`` fields).
* ``RefSideConds``: a side's ``SideConditions`` (``[2]`` fields).
* ``RefState``: ``parties[P]``, ``battlers[B]``, ``sides[2]`` and the field
  (``FieldConditions``, ``totalTurns``, the battle result, the AI's knowledge), and the player
  party's experience fields and ``sideGetExpMask`` (``exp_mask``).

Every encoding is ``BattleState``'s (``nuzline.engine.state``): stages 0 = neutral,
``stats`` is Atk, Def, SpA, SpD, Spe, ids from ``nuzline.engine.ids``. Party 0 is the
player's; battler ``b`` has side ``b & 1`` and party ``party(fmt, b)`` (``engine.format``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from nuzline.engine.format import FORMATS, Format


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
    pp_ups: list[int]  # BattleMon.ppUps (E1 Batch D)
    status: int
    status_counter: int
    gender: int
    friendship: int
    party_friendship: int  # the party Pokemon's u8 friendship (F-15)
    ivs: list[int]
    evs: list[int]  # ids.Stat order
    nature: int


@dataclass
class RefParty:
    mons: list[RefPokemon]
    chatot_cry: int  # BattleSystem.chatotCries as Chatter's parameter 0 / 1 / 2 (G-70)
    ts_alive: int


@dataclass
class RefBattler:
    active: int
    trainer_items: list[int]  # aiContext.trainerItems[battler >> 1] (ruling D1-5)
    trainer_item_count: int
    stages: list[int]  # index ids.Stat, 0 = neutral
    confusion: int
    leech_seed: int
    substitute_hp: int
    taunt: int
    torment: int  # VOLATILE_CONDITION_TORMENT (E3-2-1)
    me_first: int  # MoveEffectsData.meFirst (E3-2-3)
    me_first_turn: int  # MoveEffectsData.meFirstTurnNumber (E3-2-3)
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
    trace_announced: int  # BattleMon.traceAnnounced (E3-2-5)
    weather_announced: int  # BattleMon.weatherAbilityAnnounced
    intimidate_announced: int  # BattleMon.intimidateAnnounced
    download_announced: int  # BattleMon.downloadAnnounced
    recycle_item: int  # BattleContext.recycleItem[battler]: never cleared by a switch (B13-19)
    minimized: int  # MOVE_EFFECT_MINIMIZE
    semi_invulnerable: int  # ids.SemiKind: MOVE_EFFECT_SEMI_INVULNERABLE's bit (B10-12)
    micle: int  # MoveEffectsData.micleBerry (E1 Batch D)
    metronome_turns: int  # MoveEffectsData.metronomeTurns
    metronome_move: int  # BattleContext.metronomeMove[battler]
    focus_energy: int  # VOLATILE_CONDITION_FOCUS_ENERGY (B12-4)
    power_trick: int  # MOVE_EFFECT_POWER_TRICK (B12-10)
    power_trick_desync: int  # the Attack / Defense swap XOR power_trick (B12-10)
    magnet_rise: int  # MOVE_EFFECT_MAGNET_RISE (G-66)
    magnet_rise_turns: int  # MoveEffectsData.magnetRiseTurns (G-66)
    mimic_slots: int  # MoveEffectsData.mimickedMoveSlot bitmask (G-68)
    shadow_moves: list[int]  # the party's move of each flagged slot (G-68)
    shadow_pp: list[int]  # the party's PP of each flagged slot (G-68)
    transformed: int  # VOLATILE_CONDITION_TRANSFORM, kept until restored (G-69)
    shadow_species: int  # the party Pokemon's own while transformed (G-69)
    shadow_form: int
    shadow_stats: list[int]
    shadow_ivs: list[int]
    shadow_ability_slot: int
    shadow_ability_form: int
    truant: int  # MoveEffectsData.truant (G-69)
    embargo: int  # MoveEffectsData.embargoTurns (B13-29)
    flash_fire: int  # MoveEffectsData.flashFire (F-2)
    ability_suppressed: int  # MOVE_EFFECT_ABILITY_SUPPRESSED (E3-2-6a)
    claw_flag: int  # MoveEffectsData.quickClaw between turns (SF2 (k))
    custap_flag: int  # MoveEffectsData.custapBerry between turns (SF2 (k))
    wish_timer: int
    wish_hp: int
    future_timer: int
    future_damage: int
    future_move: int
    future_attacker: int  # 1 + battler, 0 = none
    qc_roll: int
    starf_roll: int  # a Starf Berry draw (Outcome.starf; engine.state docstring, F-10)
    starf_open: int  # a battle-start Starf Berry's stat waits for the root draw (F-10)
    entry_pending: int  # a list newcomer's switch-in states wait (engine.state docstring)
    ts_acted: int
    ts_order: int
    ts_actions: int
    ts_move: int
    ts_struggling: int
    ts_locked: int
    ts_item: int
    ts_custap: int
    ts_claw: int
    ts_protect_ok: int
    ts_endure_ok: int
    ts_taken: list[int]
    ts_dragged: int
    ts_move_failed: int
    ts_switched: int
    ts_assurance: int  # turnFlags.assuranceDamageMask != 0 (B13-17)
    ts_sub_hit: int  # the paused action's SELF_TURN_FLAG_SUBSTITUTE_HIT of the battler (F-4)
    ts_trace_fires: int  # Trace fires of the slot in this step (D2-16)
    chosen_target: int  # battlerActions[b][1] (D2-1)
    last_hit_by: int  # 1 + lastHitByBattler[b], 0 = none (D2-14)


@dataclass
class RefSideConds:
    spikes: int
    toxic_spikes: int
    stealth_rock: int
    reflect: int
    light_screen: int
    safeguard: int
    mist: int
    lucky_chant: int
    tailwind: int
    future_sight: int  # SIDE_CONDITION_FUTURE_SIGHT
    ts_follow_me: int  # 1 + followMeUser while followMe is set, else 0 (D2-11)


@dataclass
class RefState:
    parties: list[RefParty]
    battlers: list[RefBattler]
    sides: list[RefSideConds]
    weather: int
    weather_timer: int
    trick_room: int
    gravity: int
    acid_rain: int
    war_swaps: int  # a weather war's open monSpeedOrder (engine.state docstring)
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
    ts_pursued: int
    ts_used: int
    ts_damaged: int
    ts_dealt: int
    ts_se: int
    ts_power_mul: int
    ts_pick_mul: int


# BattleState field names by level.
MON_FIELDS = tuple(f.name for f in fields(RefPokemon))
PARTY_FIELDS = tuple(f.name for f in fields(RefParty) if f.name != "mons")
BATTLER_FIELDS = tuple(f.name for f in fields(RefBattler))
SIDE_FIELDS = tuple(f.name for f in fields(RefSideConds))
FIELD_FIELDS = tuple(
    f.name for f in fields(RefState) if f.name not in ("parties", "battlers", "sides")
)


def format_of_ref(ref: RefState) -> Format:
    """The ``Format`` of ``ref``: ``B`` is its battler count, ``P`` its party count."""
    b, p = len(ref.battlers), len(ref.parties)
    for fmt in FORMATS:
        if (fmt.B, fmt.P) == (b, p):
            return fmt
    raise ValueError(f"no battle format has {b} battlers and {p} parties")
