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
    knocked_item: int  # the party's item while its side's knocked_off bit is set (engine.state)
    moves: list[int]
    pp: list[int]
    pp_ups: list[int]  # BattleMon.ppUps (E1 Batch D)
    status: int
    status_counter: int
    gender: int
    friendship: int
    party_friendship: int  # the party Pokemon's u8 friendship (F-15)
    party_ability: int  # the party Pokemon's ability a script wrote, 0 = none (F-O2)
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
    leech_seed_recipient: int  # 1 + the seeder's battler, 0 = not seeded (D3-11b)
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
    move_copied: int  # BattleContext.moveCopied (E3-B2-2)
    move_copied_hit: list[int]  # moveCopiedHit[holder][attacker] (E3-B2-2)
    magic_coat: int  # turnFlags.magicCoat as 1 + the user's moveSlot (E3-B2-6)
    conv2_move: int  # BattleContext.conversion2Move (E3-B2-3)
    conv2_battler: int  # conversion2Battler, 1 + battler, 0 = none
    conv2_type: int  # conversion2Type
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
    grudge: int  # MOVE_EFFECT_GRUDGE (E3-B2-9)
    miracle_eye: int  # MOVE_EFFECT_MIRACLE_EYE (E3-B2-10)
    last_resort_count: int  # MoveEffectsData.lastResortCount (E3-B2-11)
    last_resort_moves: list[int]  # MoveEffectsData.lastResortMoves[4] (E3-B2-11)
    uproar: int  # VOLATILE_CONDITION_UPROAR turns left (and the battler's FIELD_CONDITION bit)
    curse: int
    nightmare: int  # VOLATILE_CONDITION_NIGHTMARE (G-31)
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
    charged_turns: int  # MoveEffectsData.chargedTurns with MOVE_EFFECT_CHARGE (E3-B2-8)
    camouflage: int  # MOVE_EFFECT_CAMOUFLAGE (G-20)
    mimic_slots: int  # MoveEffectsData.mimickedMoveSlot bitmask (G-68)
    shadow_moves: list[int]  # the party's move of each flagged slot (G-68)
    shadow_pp: list[int]  # the party's PP of each flagged slot (G-68)
    transformed: int  # VOLATILE_CONDITION_TRANSFORM, kept until restored (G-69)
    shadow_species: int  # the party Pokemon's own while transformed (G-69)
    shadow_form: int
    form_num: int  # BattleMon.formNum (battle_mon.h:79, H-5)
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
    speed_rec: int  # BattleContext.monSpeedValues (E3-B2-15)
    starf_roll: list[int]  # the turn's Starf Berry draws (engine.state docstring, F-O3)
    starf_taken: int  # the draws this turn's Starf Berry eats used (F-O3)
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
    ts_move_cur: int  # moveCur of the battler's action this turn (E3-B2-21)
    ts_eot: int  # monSpeedOrder of the turn end, sorted before a learn pause (E3-B2-21)
    ts_qc: int  # the turn's speedRand draw at a learn pause (E3-B2-21)
    ts_switched: int
    ts_assurance: int  # turnFlags.assuranceDamageMask != 0 (B13-17)
    ts_sub_hit: int  # the paused action's SELF_TURN_FLAG_SUBSTITUTE_HIT of the battler (F-4)
    # the paused action's SELF_TURN_FLAG_INFATUATED of the slot, after :70 (H-4): 1 flagged and the
    # sideEffectMon at the list, 2 flagged but superseded by a later pursuer's trigger (H-M1)
    ts_loved: int
    ts_speed_order: int  # the paused action's monSpeedOrder entry (D3-10; B = 4 only)
    ts_grudge: int  # fainted in the paused action holding Grudge (engine.state, review b2 #5)
    ts_trace_fires: int  # Trace fires of the slot in this step (D2-16)
    ts_helping_hand: int  # turnFlags.helpingHand, 0 between turns (D3 fix B)
    chosen_target: int  # battlerActions[b][1] (D2-1)
    last_hit_by: int  # 1 + lastHitByBattler[b], 0 = none (D2-14)
    move_slot: int  # BattleContext.moveSlot[b], written at a FIGHT's selection (E2-ML-5)


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
    # knockedOffItemsMask (battle_context.h:49) as one 0/1 per party slot: the side's bit, shared by
    # every party of the side (B10-6, D3 fix B)
    knocked_off: list[int]


@dataclass
class RefState:
    parties: list[RefParty]
    battlers: list[RefBattler]
    sides: list[RefSideConds]
    field_mask: int  # fieldConditionsMask (engine.state, engine.field)
    weather_turns: int  # FieldConditions.weatherTurns
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
    # Experience (K3d-exp): the player party's, per slot, and sideGetExpMask: an int at B = 2
    # (entry 0), a list of 2 at B = 4 (entry (b >> 1) & 1 of enemy battler b).
    exp: list[int]
    exp_mask: int | list[int]
    exp_boost: list[int]
    pokerus: list[int]
    friend_bonus: list[int]
    learn_level: list[int]
    learn_cursor: list[int]
    deaths: list[int]
    faint_budget: int
    protected: list[int]
    replace_mul: int
    # E2 batch 6: the mid-turn switch decision (engine.state docstring): 0 between turns.
    pivot_battler: int
    ts_cursor: int
    learn_site: int  # E3-B2-21: the learn pause's site (engine.state docstring), 0 none
    ts_ran: int
    ts_pursued: int
    ts_used: int
    ts_damaged: int
    ts_dealt: int
    ts_defender: int  # 1 + the paused action's battleCtx->defender (D3-10; B = 4 only)
    ts_se: int
    ts_power_mul: int
    ts_pick_mul: int
    ts_move_temp: int  # the paused action's moveTemp (E2-ML-4)


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
