"""The engine's vocabulary: effect, ability, item, status, stat, move-flag and type ids.

Tables store these ids (``tables.move_effect``, ``species_abilities``,
``item_effect``, ``trainer_item_effect``, ``move_flags``); the decomp's own ids are mapped onto them by
``tools/effect_map.py`` at table-build time. Anything outside the implemented set
is the ``UNSUPPORTED`` member, never an approximation.
"""

from enum import IntEnum, IntFlag


class EffectId(IntEnum):
    """Move effect. Params live in ``tables.move_param[m, 0..2]``:

    * stat effects (``STAT_SELF``, ``STAT_FOE``, ``SECONDARY_STAT_*``): ``(Stat, stages, 0)``;
      ``STAT_SELF`` with ``move_param[m, 2] == 1`` (Defense Curl) also sets ``curled``
    * status effects (``STATUS_FOE``, ``SECONDARY_STATUS``): ``(Status, type_checked,
      thaws_user)``; ``type_checked`` is 1 for a status move the decomp runs through
      the type chart (Thunder Wave: Ground is immune), else 0; ``thaws_user`` is 1 for
      ``BATTLE_EFFECT_THAW_AND_BURN_HIT`` (Flame Wheel, Sacred Fire), which a frozen
      user can use and which thaws it, else 0
    * ``DRAIN``, ``RECOIL``: ``(percent of damage dealt, 0, 0)``
    * everything else: zeros. ``LEVEL_DAMAGE`` (Night Shade, Seismic Toss) deals the user's
      level; ``HIDDEN_POWER`` and ``MAGNITUDE`` take their type and power from the
      battler and ``Outcome.magnitude`` (``engine/damage.py``).
    * ``SECONDARY_CONFUSE`` (Confusion, Psybeam, ...): zeros; confuses the target at the
      move's effect chance. ``MUD_SPORT``: zeros. ``EAT_BERRY`` (Bug Bite, Pluck): zeros;
      the user eats the target's Oran or Sitrus Berry (``hooks.pluck_berry``).
    ``AQUA_RING``: zeros; sets the user's ``aqua_ring`` (fails if set), healed at turn end
    (``step._aqua_ring_check``).
    ``DISABLE``: zeros; disables the target's last move for 3-6 turns (``effects._disable``).
    ``LIGHT_SCREEN``, ``REFLECT``: zeros; set the user's side screen for 5 turns
    (``effects._screen``). ``BRICK_BREAK``: zeros; ignores the screens in its damage and
    clears both of the defender's side after the hit (``effects._brick_break``).
    ``BIDE``, ``ROLLOUT`` (Rollout, Ice Ball): zeros; they lock the user into the move
    (``BattleState.locked_move``, ``engine/step.py``, ``engine/damage.py``).
    * ``MULTI_HIT``: ``(hits, 0, 0)``: ``hits == 0`` is the 2-5 law (``Outcome.n_hits``,
      ``BATTLE_EFFECT_MULTI_HIT``), else exactly that many hits (``BATTLE_EFFECT_HIT_TWICE``:
      Double Kick, Kaizo's DoubleSlap, ``SetMultiHit 2``).
    * Multi-stat effects: the stat list is the effect's own, so the params stay zeros.
      ``RAISE_ALL_STATS_HIT`` (AncientPower): Atk, Def, Spe, SpA, SpD +1 on the user at the
      move's effect chance; ``CURSE`` (a non-Ghost user; coverage refuses a Ghost-type one):
      Spe -1, Atk +1, Def +1 on the user.
    * ``RECOVER``: zeros; heals ``maxHP / 2``, fails at full HP. ``FAKE_OUT``: zeros; fails
      unless it is the user's first turn out, else flinches at the move's chance.
    * ``SELF_DESTRUCT`` (Selfdestruct, Explosion; Kaizo's Memento): zeros; the move script
      sets the user's HP to 0 before the accuracy check, and a physical one hits the halved
      Defense (``damage.py``). Damp would stop it; Damp is ``AbilityId.UNSUPPORTED``.
    * ``FLAIL`` (Flail, Reversal): zeros; the power comes from the user's HP pixels
      (``damage.flail_power``); the trainer AI scores it at the table power.
    * ``FANG_PARALYZE_FLINCH`` (Thunder Fang, ``BATTLE_EFFECT_FLINCH_PARALYZE_HIT``):
      ``(Status.PARALYSIS, 0, 0)``; two independent rolls at the move's chance, the
      paralysis (``Outcome.secondary``) then the flinch (``Outcome.fang_flinch``)
      (``effects._fang``).
    * ``PRESENT`` (Present, ``BATTLE_EFFECT_RANDOM_POWER_MAYBE_HEAL``): zeros;
      ``Outcome.present`` draws a power of 40/80/120 for the hit (``damage.PRESENT_POWER``)
      or a heal of the target by ``maxHP / 4`` that ignores immunities and reads no crit or
      roll (``effects._present``).
    * ``PROTECT`` (Protect, Detect, ``BATTLE_EFFECT_PROTECT``): zeros; ``BtlCmd_TryProtection``
      succeeds at 2^-n by ``protect_count`` unless the user acts last, and its success blocks
      the foe's ``MoveFlag.PROTECT`` moves for the rest of the turn (``step._use_move``).
    * ``ATTRACT`` (Attract, ``BATTLE_EFFECT_INFATUATE``): zeros; infatuates a living target of
      the opposite gender that is not in love yet (``effects._attract``); an infatuated
      battler is immobilized by ``Outcome.infatuation`` at its status check (``step``).
    * ``CAMOUFLAGE`` (Camouflage, ``BATTLE_EFFECT_CAMOUFLAGE``): zeros; ``BtlCmd_TryCamouflage``
      writes ``TERRAIN_TYPES[terrain]`` to both of the user's ``types`` unless it has Multitype
      or already has that type (``effects._camouflage``); ``step`` restores the party types
      when the battler leaves.
    * ``NATURE_POWER`` (Nature Power, ``BATTLE_EFFECT_NATURE_POWER``): zeros; the move runs the
      terrain's move ``TERRAIN_MOVES[terrain]`` from its accuracy check on
      (``order.called_move``, ``step._use_move``); it stays Nature Power for the status checks,
      the PP, Taunt, Disable, priority and ``last_move``.
    * ``TRI_ATTACK`` (Tri Attack, ``BATTLE_EFFECT_TRI_ATTACK``): zeros; at the move's chance
      (``Outcome.secondary``) a burn, freeze or paralysis (``TRI_ATTACK_STATUSES``, drawn by
      ``Outcome.tri_attack``) as a secondary status (``effects._tri_attack``).
    * ``ROAR`` (Roar, Whirlwind, ``BATTLE_EFFECT_FORCE_SWITCH``): zeros; in a trainer battle
      ``BtlCmd_TryWhirlwind`` drags a random eligible bench Pokemon of the target's side in
      (``Outcome.roar_target``), after the level check of a lower-levelled user
      (``Outcome.roar_level``); it fails with no eligible bench Pokemon (``step._use_move``).

    The proc chance of a ``SECONDARY_*``, ``FLINCH``, ``FAKE_OUT``, ``RAISE_ALL_STATS_HIT`` or
    ``FANG_PARALYZE_FLINCH``
    effect is the move's own ``move_effect_chance``; a ``SECONDARY_*`` move with chance 0 applies its
    effect on every hit (the decomp's ``MOVE_SIDE_EFFECT_ON_HIT``, e.g. Hammer Arm).
    """

    UNSUPPORTED = 0
    PLAIN = 1
    STAT_SELF = 2
    STAT_FOE = 3
    SECONDARY_STAT_SELF = 4
    SECONDARY_STAT_FOE = 5
    STATUS_FOE = 6
    SECONDARY_STATUS = 7
    FLINCH = 8
    HIGH_CRIT = 9
    DRAIN = 10
    STEALTH_ROCK = 11
    TRAP = 12
    PURSUIT = 13
    TAUNT = 14
    CONFUSE_FOE = 15
    LEECH_SEED = 16
    MULTI_HIT = 17
    WATER_SPORT = 18
    RECOIL = 19
    FORESIGHT = 20
    LEVEL_DAMAGE = 21
    HIDDEN_POWER = 22
    MAGNITUDE = 23
    SECONDARY_CONFUSE = 24
    MUD_SPORT = 25
    EAT_BERRY = 26
    DISABLE = 27
    BIDE = 28
    ROLLOUT = 29
    LIGHT_SCREEN = 30
    REFLECT = 31
    BRICK_BREAK = 32
    RECOVER = 33
    FAKE_OUT = 34
    RAISE_ALL_STATS_HIT = 35
    CURSE = 36
    SELF_DESTRUCT = 37
    FLAIL = 38
    AQUA_RING = 39
    FANG_PARALYZE_FLINCH = 40
    PRESENT = 41
    PROTECT = 42
    ATTRACT = 43
    CAMOUFLAGE = 44
    NATURE_POWER = 45
    TRI_ATTACK = 46
    ROAR = 47
    # Sandstorm, Rain Dance, Sunny Day, Hail (effect_script_0115.s, _0136.s, _0137.s,
    # _0164.s): ``(Weather, 0, 0)``. It fails when that weather is already up (either kind);
    # else every weather bit is cleared and the weather is set for 5 turns, plus the param of
    # the user's matching ``ItemId.WEATHER_ROCK`` (``effects._weather``).
    WEATHER = 48
    # E2 batch 4 (rulings B4-n, docs/kaizo/e2-mechanics-ledger.md).
    # SmellingSalt (effect_script_0171.s), Facade (_0169.s), Brine (_0221.s; Kaizo's Wring Out):
    # ``(condition, cures, 0)``: the script sets BTLVAR_POWER_MULTI 20 before CalcDamage when
    # ``condition`` holds (``damage.script_power_mul``): ``POWER_IF_TARGET_PARALYZED`` (and
    # ``cures``: an ON_HIT indirect subscript_heal_target_paralysis.s, ``effects._cure_paralysis``),
    # ``POWER_IF_USER_STATUSED`` (MON_CONDITION_FACADE_BOOST: poison, toxic, burn, paralysis),
    # ``POWER_IF_TARGET_HALF_HP`` (curHP <= maxHP / 2, a plain ``/``).
    DOUBLE_POWER = 49
    # Wring Out, Crush Grip (vanilla effect 237, BtlCmd_CalcCrushGripPower, battle_script.c:7265):
    # movePower = 1 + 120 * curHP / maxHP of the defender (``damage._pre_random``).
    HP_POWER = 50
    # Grass Knot, Low Kick (effect_script_0196.s, BtlCmd_CalcWeightBasedPower, battle_script.c:
    # 6808-6826): the power from the defender's weight (``tables.species_weight``,
    # include/data/battle/weight_to_power.h; ``damage.weight_power``).
    WEIGHT_POWER = 51
    # Super Fang (effect_script_0040.s): SYSCTL_IGNORE_TYPE_CHECKS, damage = Divide(curHP, 2)
    # of the defender, no crit or roll; a type immunity still deals 0 (``damage.fixed_damage``).
    HALVE_HP = 52
    # Fissure, Guillotine, Horn Drill (effect_script_0038.s, BtlCmd_TryOHKOMove, battle_script.c:
    # 4334-4386): its own hit test (``outcome._rate``), Sturdy, and the defender's max HP as
    # damage (``damage.fixed_damage``, bcp.c:3343-3345).
    OHKO = 53
    # False Swipe (BATTLE_EFFECT_LEAVE_WITH_1_HP, bcp.c:3381-3384): a hit that would KO leaves
    # the defender at 1 HP, before the Focus Sash (``effects.play_hit``).
    FALSE_SWIPE = 54
    # Swagger (effect_script_0118.s, subscript_swagger.s): Attack +2 then confusion
    # (``effects._swagger``).
    SWAGGER = 55
    # Dragon Dance (effect_script_0212.s, subscript_user_atk_and_speed_up_1_stage.s): Attack +1
    # and Speed +1 on the user; both at +6 is MOVE_STATUS_NO_MORE_WORK (``effects._dragon_dance``).
    DRAGON_DANCE = 56
    # Roost (effect_script_0214.s, subscript_roost.s): heals Divide(maxHP, 2) through
    # subscript_recover_hp.s and sets turnFlags.roosting, which skips the defender's Flying rows
    # of the type chart for the rest of the turn (battle_lib.c:2540-2543; ``damage.chart_types``).
    ROOST = 57
    # Tailwind (subscript_tailwind_start.s): the user's side gets ``TAILWIND_TURNS``, fails when up.
    TAILWIND = 58
    # Spikes / Toxic Spikes (``BtlCmd_TrySpikes`` / ``TryToxicSpikes``, battle_script.c:5388-5403,
    # :7592-7605): a layer on the defender's side up to ``SPIKES_MAX`` / ``TOXIC_SPIKES_MAX``; the
    # switch-in effects are ``effects.entry_hazards`` (subscript_hazards_check.s).
    SPIKES = 59
    TOXIC_SPIKES = 60
    # Destiny Bond (effect_script_0098.s, subscript_destiny_bond.s): ``BattleState.destiny_bond``;
    # ``step._use_move`` takes the attacker down (subscript_faint_check_destiny_bond.s).
    DESTINY_BOND = 61
    # Secret Power (effect_script_0197.s, BtlCmd_GetTerrainSecondaryEffect, battle_script.c:8063-8075):
    # a hit whose secondary effect is the terrain's (``SECRET_POWER_EFFECTS``,
    # ``order.move_effects``).
    SECRET_POWER = 62
    # Counter, Mirror Coat (effect_script_0089.s, _0144.s: BtlCmd_Counter / MirrorCoat,
    # battle_script.c:4615-4705) and Metal Burst (effect_script_0227.s, BtlCmd_TryMetalBurst,
    # :7153-7184; Kaizo's Bide): ``(source, multiplier numerator, 0)``: the damage the user took
    # this turn from the foe (``turnFlags``: ``COUNTER_PHYSICAL`` / ``_SPECIAL`` / ``_ANY``, its
    # last one, bcp.c:3413-3434) times ``numerator / 10`` (``step._use_move``,
    # ``effects.play_hit``), under SYSCTL_IGNORE_TYPE_CHECKS; 0 taken fails the move.
    COUNTER = 63
    # E2 batch 5 (rulings B5-n). The invokers: the move script draws or reads another move and
    # GoToMoveScript runs it as moveCur (battle_script.c:2111-2134; ``order.invoked_move``):
    # Sleep Talk (effect_script_0097.s: fails unless the user is asleep, BtlCmd_TrySleepTalk
    # :4907-4937 draws an eligible slot, ``Outcome.call``), Metronome (effect_script_0083.s,
    # BtlCmd_Metronome :4535-4562 draws a move id, ``Outcome.call``), Copycat
    # (effect_script_0242.s, BtlCmd_TryCopycat :7328-7341 reads ``BattleState.move_prev``).
    SLEEP_TALK = 64
    METRONOME = 65
    COPYCAT = 66
    # Uproar (effect_script_0159.s): a hit whose ON_HIT indirect effect, for a user not in an
    # uproar yet, runs subscript_uproar.s: the user is locked into the move (moveLockedInto) for
    # ``Random 3, 3`` (3-6) turns (``BattleState.uproar``, ``Outcome.duration``), during which
    # no battler sleeps (``effects._inflict``, ``step._use_move``, ``step._uproar_check``;
    # rulings B5-4..).
    UPROAR = 67
    # Future Sight, Doom Desire (effect_script_0148.s, BATTLE_EFFECT_HIT_IN_3_TURNS):
    # ``BtlCmd_TryFutureSight`` (battle_script.c:6065-6100) stores the damage on the defender's
    # battler for 3 turns (``BattleState.future_*``), which ``step._future_sight`` deals at the
    # turn's end (bcp.c:1758-1783, subscript_future_sight_damage.s); the use itself deals none and
    # never misses (rulings B5-8..).
    FUTURE_SIGHT = 68
    # E2 batch 6 (rulings B6-n). U-turn (effect_script_0228.s: an ON_HIT indirect
    # subscript_attack_then_switch_out.s) and Baton Pass (effect_script_0127.s: an ON_HIT direct
    # subscript_baton_pass.s): the user switches out mid-action through the party list, a decision
    # inside the turn (``BattleState.pivot_battler``; ``step``'s pause and resume).
    U_TURN = 69
    BATON_PASS = 70
    # E2 batch 7 (rulings B7-n), the state the trainer AI reads. Ingrain
    # (effect_script_0181.s): sets the user's MOVE_EFFECT_INGRAIN (``BattleState.ingrain``),
    # failing if set; it heals at the turn end (bcp.c:1311-1326, subscript_ingrain_heal.s), traps
    # (battle_lib.c:3256, :5523) and grounds (:2534-2536, :3100-3103) its user. Mean Look, Block,
    # Spider Web (effect_script_0106.s, a direct CHECK_HP | TO_DEFENDER subscript_mean_look.s): the
    # defender's VOLATILE_CONDITION_MEAN_LOOK and meanLookTarget (``mean_look``,
    # ``mean_look_target``). Imprison (effect_script_0192.s, BtlCmd_TryImprison
    # battle_script.c:6674-6727): the user's MOVE_EFFECT_IMPRISON, the foes' MOVE_EFFECT_IMPRISONED
    # (``imprison``, ``imprisoned``); its moves are then invalid for the foes (battle_lib.c:
    # 2287-2290, :3316-3338) and fail at their status check (bcp.c:2572-2583).
    INGRAIN = 71
    MEAN_LOOK = 72
    IMPRISON = 73
    # E2 batch 8 (rulings B8-n), the moves behind the AI's fields. Lock-On, Mind Reader
    # (effect_script_0094.s, a direct CHECK_HP | TO_DEFENDER subscript_lock_on.s): the defender's
    # MOVE_EFFECT_LOCK_ON counter (2) and lockOnTarget (``lock_on``, ``lock_on_target``); the
    # locker's next moves on it skip the accuracy roll (MON_IS_LOCKED_ONTO, bcp.c:3016-3021).
    LOCK_ON = 74
    # Perish Song (effect_script_0114.s, a direct ON_HIT subscript_perish_song_start.s:
    # BtlCmd_TryPerishSong, battle_script.c:5465-5487): every living battler without the effect
    # gets MOVE_EFFECT_PERISH_SONG and a count of 3 (``perish_song``, ``perish_song_turns``); the
    # side-condition pass counts it down and faints the battler at 0 (bcp.c:1791-1820).
    PERISH_SONG = 75
    # Yawn (effect_script_0187.s, a direct CHECK_HP | TO_DEFENDER subscript_yawn.s): the
    # defender's MOVE_EFFECT_YAWN counter (2, ``BtlCmd_TryYawn`` battle_script.c:6580-6590); the
    # turn end's MON_COND_CHECK_STATE_YAWN drops it and at 0 runs subscript_fall_asleep.s as a
    # move effect (bcp.c:1684-1697), its turns drawn as ``Outcome.yawn_sleep``.
    YAWN = 76


# ``EffectId.COUNTER``'s source (``move_param[m, 0]``).
COUNTER_PHYSICAL, COUNTER_SPECIAL, COUNTER_ANY = 1, 2, 3

# Snore (effect_script_0092.s) is ``FLINCH`` with ``move_param[m, 2] == SNORE_PARAM``: the
# script fails unless its user is asleep (:5, :15-17), and a sleeping user uses it
# (bcp.c:2455: ``moveCur == MOVE_SNORE``).
SNORE_PARAM = 1
# subscript_tailwind_start.s:11 sets both SIDE_CONDITION_TAILWIND bits: a 2-bit counter at 3.
TAILWIND_TURNS = 3
SPIKES_MAX = 3  # battle_script.c:5394
TOXIC_SPIKES_MAX = 2  # battle_script.c:7598

# ``EffectId.DOUBLE_POWER``'s condition (``move_param[m, 0]``).
POWER_IF_TARGET_PARALYZED = 1  # effect_script_0171.s:6 (SmellingSalt)
POWER_IF_USER_STATUSED = 2  # effect_script_0169.s:5 (Facade, MON_CONDITION_FACADE_BOOST)
POWER_IF_TARGET_HALF_HP = 3  # effect_script_0221.s:5-8 (Brine)


# Effects whose script runs no CalcCrit / CalcDamage: no crit and no damage roll is read
# (``damage.fixed_damage``): Night Shade's level (effect_script_0087.s), Bide's unleash,
# Super Fang (effect_script_0040.s), the OHKO moves (effect_script_0038.s); Future Sight's use
# deals nothing (effect_script_0148.s: its variance is TryFutureSight's, ``Outcome.roll[s, 0]``).
UNROLLED_EFFECTS = (
    EffectId.LEVEL_DAMAGE,
    EffectId.BIDE,
    EffectId.HALVE_HP,
    EffectId.OHKO,
    EffectId.COUNTER,
    EffectId.FUTURE_SIGHT,
)
# BtlCmd_TryFutureSight (battle_script.c:6074): futureSightTurns = 3.
FUTURE_SIGHT_TURNS = 3


class AbilityId(IntEnum):
    UNSUPPORTED = 0
    NONE = 1
    STURDY = 2
    ROCK_HEAD = 3
    MOLD_BREAKER = 4
    BLAZE = 5
    TORRENT = 6
    OVERGROW = 7
    INTIMIDATE = 8
    RIVALRY = 9
    KEEN_EYE = 10
    SIMPLE = 11
    UNAWARE = 12
    NATURAL_CURE = 13
    POISON_POINT = 14
    INNER_FOCUS = 15
    STATIC = 16
    GUTS = 17
    STEADFAST = 18
    INSOMNIA = 19
    CLOUD_NINE = 20
    COMPOUNDEYES = 21
    SHED_SKIN = 22
    SAND_VEIL = 23
    SNOW_CLOAK = 24
    SWIFT_SWIM = 25
    ROUGH_SKIN = 26
    FLAME_BODY = 27
    LIMBER = 28
    ADAPTABILITY = 29
    SOLID_ROCK = 30
    CUTE_CHARM = 31
    # BattleSystem_TriggerHeldItem / ..._OnStatus halve a pinch berry's param
    # (battle_lib.c:4774-4776 and its siblings), BattleSystem_CompareBattlerSpeed Custap's
    # (:1289-1291, :1355-1357); no other decomp code reads ABILITY_GLUTTONY.
    GLUTTONY = 32
    # E2 batch 3 (rulings B3-n, docs/kaizo/e2-mechanics-ledger.md). Damage formula
    # (damage.py): Technician, Thick Fat, Dry Skin's Fire boost, Solar Power, Reckless (its
    # recoil scripts' powerMul), Tinted Lens, Levitate, Scrappy; crit (outcome.crit_odds,
    # effects.play_hit): Super Luck, Shell Armor, Battle Armor; accuracy (outcome._rate): No
    # Guard, Tangled Feet; order (order.effective_speed): Quick Feet; secondary chance
    # (outcome.secondary_chance): Serene Grace; effects.py: Shield Dust, Dry Skin's Water
    # absorb (step._use_move); step.py: Truant, the weather abilities' turn-end HP (Dry Skin,
    # Solar Power); hooks.on_switch_in: Trace, Drought, Drizzle, Sand Stream, Snow Warning.
    # Lightningrod (BattleSystem_TriggerRedirectionAbilities, battle_lib.c:1756-1772) only
    # redirects to a battler other than the defender, which singles never has; Honey Gather
    # acts only after the battle (battle_script.c:7825).
    RECKLESS = 33
    TECHNICIAN = 34
    TRUANT = 35
    DRY_SKIN = 36
    LIGHTNINGROD = 37
    NO_GUARD = 38
    SERENE_GRACE = 39
    DROUGHT = 40
    HONEY_GATHER = 41
    LEVITATE = 42
    QUICK_FEET = 43
    SCRAPPY = 44
    SHELL_ARMOR = 45
    BATTLE_ARMOR = 46
    SHIELD_DUST = 47
    SOLAR_POWER = 48
    SUPER_LUCK = 49
    TANGLED_FEET = 50
    THICK_FAT = 51
    TINTED_LENS = 52
    TRACE = 53
    DRIZZLE = 54
    SAND_STREAM = 55
    SNOW_WARNING = 56
    # E2 batch 4. Filter (battle_lib.c:2658-2661, ``damage._finish``); Magic Guard (every reader
    # is Battler_Ability or CheckAbility: the residual damage, the recoil, full paralysis and
    # the entry hazards, ruling B4 list in the ledger); Sniper (battle_lib.c:7142-7144).
    FILTER = 57
    MAGIC_GUARD = 58
    SNIPER = 59
    # E2 batch 9 (rulings B9-n). Speed (order.effective_speed, BattleSystem_CompareBattlerSpeed):
    # Chlorophyll in sun (battle_lib.c:1228-1238), Unburden with no item after it held one
    # (:1268-1272; ``BattleState.can_unburden``); the damage formula (damage._pre_random,
    # BattleSystem_CalcMoveDamage): Huge Power / Pure Power (:6706-6708), Swarm (:6834-6838),
    # Flower Gift in sun (:6930-6941).
    CHLOROPHYLL = 60
    HUGE_POWER = 61
    PURE_POWER = 62
    SWARM = 63
    UNBURDEN = 64
    FLOWER_GIFT = 65


# The decomp ability id of every supported ``AbilityId`` (generated/abilities.txt line N is id
# N - 1; tests/engine/test_ids.py reads them back through effect_map.DECOMP_ABILITY_TO_ID and
# both games' abilities.json). ``damage.decomp_abilities`` reads it for a battler whose
# ``BattleMon.ability`` Trace rewrote (battle_lib.c:3720-3735, subscript_trace.s), which the
# species table no longer gives.
ABILITY_DECOMP_IDS = {
    AbilityId.NONE: 0,
    AbilityId.STURDY: 5,
    AbilityId.ROCK_HEAD: 69,
    AbilityId.MOLD_BREAKER: 104,
    AbilityId.BLAZE: 66,
    AbilityId.TORRENT: 67,
    AbilityId.OVERGROW: 65,
    AbilityId.INTIMIDATE: 22,
    AbilityId.RIVALRY: 79,
    AbilityId.KEEN_EYE: 51,
    AbilityId.SIMPLE: 86,
    AbilityId.UNAWARE: 109,
    AbilityId.NATURAL_CURE: 30,
    AbilityId.POISON_POINT: 38,
    AbilityId.INNER_FOCUS: 39,
    AbilityId.STATIC: 9,
    AbilityId.GUTS: 62,
    AbilityId.STEADFAST: 80,
    AbilityId.INSOMNIA: 15,
    AbilityId.CLOUD_NINE: 13,
    AbilityId.COMPOUNDEYES: 14,
    AbilityId.SHED_SKIN: 61,
    AbilityId.SAND_VEIL: 8,
    AbilityId.SNOW_CLOAK: 81,
    AbilityId.SWIFT_SWIM: 33,
    AbilityId.ROUGH_SKIN: 24,
    AbilityId.FLAME_BODY: 49,
    AbilityId.LIMBER: 7,
    AbilityId.ADAPTABILITY: 91,
    AbilityId.SOLID_ROCK: 116,
    AbilityId.CUTE_CHARM: 56,
    AbilityId.GLUTTONY: 82,
    AbilityId.RECKLESS: 120,
    AbilityId.TECHNICIAN: 101,
    AbilityId.TRUANT: 54,
    AbilityId.DRY_SKIN: 87,
    AbilityId.LIGHTNINGROD: 31,
    AbilityId.NO_GUARD: 99,
    AbilityId.SERENE_GRACE: 32,
    AbilityId.DROUGHT: 70,
    AbilityId.HONEY_GATHER: 118,
    AbilityId.LEVITATE: 26,
    AbilityId.QUICK_FEET: 95,
    AbilityId.SCRAPPY: 113,
    AbilityId.SHELL_ARMOR: 75,
    AbilityId.BATTLE_ARMOR: 4,
    AbilityId.SHIELD_DUST: 19,
    AbilityId.SOLAR_POWER: 94,
    AbilityId.SUPER_LUCK: 105,
    AbilityId.TANGLED_FEET: 77,
    AbilityId.THICK_FAT: 47,
    AbilityId.TINTED_LENS: 110,
    AbilityId.TRACE: 36,
    AbilityId.DRIZZLE: 2,
    AbilityId.SAND_STREAM: 45,
    AbilityId.SNOW_WARNING: 117,
    AbilityId.FILTER: 111,
    AbilityId.MAGIC_GUARD: 98,
    AbilityId.SNIPER: 97,
    AbilityId.CHLOROPHYLL: 34,
    AbilityId.HUGE_POWER: 37,
    AbilityId.PURE_POWER: 74,
    AbilityId.SWARM: 68,
    AbilityId.UNBURDEN: 84,
    AbilityId.FLOWER_GIFT: 122,
}

# ChooseTraceTarget (battle_lib.c:7750-7775) never picks a battler whose ``BattleMon.ability``
# is Forecast, Trace or Multitype (decomp ids, generated/abilities.txt).
ABILITY_FORECAST = 59
UNTRACEABLE_ABILITIES = (ABILITY_FORECAST, 36, 121)  # Forecast, Trace, Multitype
# ITEM_GRISEOUS_ORB (generated/items.txt): a Trace holder holding it never traces
# (battle_lib.c:3728).
ITEM_GRISEOUS_ORB = 112


class ItemId(IntEnum):
    """Held-item effect. ``tables.item_param[i, 0]`` is the decomp hold param
    (Oran Berry, Berry Juice: HP restored; Sitrus Berry: percent of max HP; Big Root: the
    percent added to a drain, 30; the power items below: the percent they add); column 1 is
    a resist berry's or a type-boosting item's type (below), else 0. Most ids are mapped by
    the decomp hold effect (``tools/effect_map.HOLD_EFFECT_TO_ID``), so every item sharing a
    hold effect behaves alike with its own param.

    * ``NONE``: no item, and every item with no hold effect (``HOLD_EFFECT_NONE``, 0) that is
      no berry (Nugget, Blue Scarf, a Poke Ball, ...): no implemented path reads such an item
      by its id (``Item_IsBerry`` for Pluck is the only id read, below).
    * ``NO_EFFECT_BERRY``: a berry with no hold effect (Razz, Nanab, Pomeg, ...: ids
      ``FIRST_BERRY_IDX..LAST_BERRY_IDX``, include/constants/items.h:56-58, the
      ``sBerryItemIDs`` list ``Item_IsBerry`` reads, src/item.c:357-365). Nothing triggers it;
      Pluck / Bug Bite eat it with no effect (``PLUCK_EFFECT_NONE``, battle_lib.c:5832-5835,
      ``hooks.pluck_berry``).
    * ``ORAN_BERRY``, ``BERRY_JUICE`` (both ``HOLD_EFFECT_HP_RESTORE``), ``SITRUS_BERRY``,
      ``CHESTO_BERRY`` (``SLP_RESTORE``), ``PECHA_BERRY`` (``PSN_RESTORE``, toxic too),
      ``LUM_BERRY`` (``STATUS_RESTORE``, confusion too), ``WHITE_HERB``
      (``STATDOWN_RESTORE``): ``hooks.held_item_check``. Berry Juice is its own id because
      its pluck effect is ``PLUCK_EFFECT_NONE`` and it is no berry, so Pluck / Bug Bite
      leave it (battle_lib.c:5833, ``hooks.pluck_berry``).
    * ``FOCUS_SASH`` (``HOLD_EFFECT_ENDURE``): ``effects.play_hit`` and the confusion
      self-hit (``step._use_move``).
    * ``BIG_ROOT`` (``HOLD_EFFECT_LEECH_BOOST``): drain (``effects._drain``) and Leech Seed
      (``step._leech_seed_check``).
    * ``LAGGING_TAIL`` (``HOLD_EFFECT_PRIORITY_DOWN``): ``order.player_second``.
    * ``QUICK_CLAW`` (``HOLD_EFFECT_SOMETIMES_PRIORITY``, param ``QUICK_CLAW_PARAM``):
      ``order.player_second`` with the holder's ``speedRand`` draw (``BattleState.qc_roll``
      for the turn order, ``Outcome.quick_claw`` for the end-of-turn order).
    * ``RESIST_BERRY`` (every ``HOLD_EFFECT_WEAKEN_SE_*`` berry, Occa to Babiri, and Chilan,
      ``HOLD_EFFECT_WEAKEN_NORMAL``): ``tables.item_param[i, 1]`` is the type the berry
      resists, ``RESIST_ANY_NORMAL`` for Chilan (any Normal-type hit, no super-effective
      test); ``effects.resist_berry`` (``subscript_type_resist_berry.s``).
    * ``TYPE_BOOST`` (every hold effect of ``sTypeBoostingItems``, battle_lib.c:6514-6549:
      the ``HOLD_EFFECT_STRENGTHEN_*`` items, incenses and plates): ``item_param[i, 1]`` is
      the boosted type; a move of that type gets ``power * (100 + param) / 100``
      (``damage._item_modifiers``, battle_lib.c:6716-6722).
    * ``CHOICE_BAND`` / ``CHOICE_SPECS`` (``HOLD_EFFECT_CHOICE_ATK`` / ``_SPATK``): Attack /
      Sp. Atk ``* 150 / 100`` in the damage formula (battle_lib.c:6724-6729);
      ``CHOICE_SCARF`` (``HOLD_EFFECT_CHOICE_SPEED``): Speed ``* 15 / 10``
      (``order.effective_speed``, battle_lib.c:1248, :1314). All three lock the holder into
      the move it used (``BattleState.choice_lock``: ``actions.usable_moves``,
      ``step._use_move``, ``step._pursuit_hit``).
    * The species items of ``BattleSystem_CalcMoveDamage`` (battle_lib.c:6730-6774,
      ``damage._item_modifiers``): ``LIGHT_BALL`` (Pikachu: power x2),
      ``THICK_CLUB`` (Cubone, Marowak: Attack x2), ``METAL_POWDER`` (a defending Ditto:
      Defense x2), ``DEEP_SEA_TOOTH`` / ``DEEP_SEA_SCALE`` (Clamperl: Sp. Atk / a defending
      Clamperl's Sp. Def x2), ``SOUL_DEW`` (Latias, Latios: Sp. Atk and a defending one's
      Sp. Def x1.5), ``ADAMANT_ORB`` / ``LUSTROUS_ORB`` / ``GRISEOUS_ORB`` (Dialga / Palkia /
      Giratina: Dragon and Steel / Water / Ghost power ``* (100 + param) / 100``).
    * ``MUSCLE_BAND`` / ``WISE_GLASSES`` (``HOLD_EFFECT_POWER_UP_PHYS`` / ``_SPEC``): a
      physical / special move's power ``* (100 + param) / 100`` (battle_lib.c:6776-6783).
    * ``EXPERT_BELT`` (``HOLD_EFFECT_POWER_UP_SE``): a super-effective hit
      ``* (100 + param) / 100`` after Solid Rock (``damage._finish``, battle_lib.c:2663).
    * ``LIFE_ORB`` (``HOLD_EFFECT_HP_DRAIN_ON_ATK``): the move script's damage
      ``* (100 + param) / 100`` after the crit (``damage.life_orb``, battle_script.c:1346),
      and ``maxHP / 10`` recoil after a move that dealt damage (``effects.life_orb_recoil``,
      battle_controller_player.c:4758-4772; Pursuit's hit: battle_lib.c:7357-7366).
    * ``CHERI_BERRY`` (``PRZ_RESTORE``), ``RAWST_BERRY`` (``BRN_RESTORE``), ``ASPEAR_BERRY``
      (``FRZ_RESTORE``), ``PERSIM_BERRY`` (``CONFUSE_RESTORE``): ``hooks.held_item_check``
      (battle_lib.c:4601-4634, :4655-4660), and Pluck's ``PLUCK_EFFECT_*_RESTORE``.
    * ``PINCH_BERRY`` (``HOLD_EFFECT_PINCH_ATK_UP`` .. ``_SPDEF_UP``: Liechi, Ganlon, Salac,
      Petaya, Apicot): ``item_param[i, 1]`` is the ``Stat`` it raises; +1 stage when ``curHP
      <= maxHP / param`` (param halved for Gluttony) and the stage is below +6
      (``hooks.held_item_check``, battle_lib.c:4773-4836); Pluck raises the eater's stat with
      no HP test (``PLUCK_EFFECT_*_UP``, :5753-5796).
    * ``CUSTAP_BERRY`` (``HOLD_EFFECT_PINCH_PRIORITY``): moves first within its priority when
      ``curHP <= maxHP / param`` (``order.custap_active``, battle_lib.c:1288-1300), and is eaten
      at the start of the holder's move when the turn order activated it
      (``subscript_check_quick_claw.s``, ``step._use_move``).
    * ``LEFTOVERS`` (``HOLD_EFFECT_HP_RESTORE_GRADUAL``), ``BLACK_SLUDGE``
      (``HOLD_EFFECT_HP_RESTORE_PSN_TYPE``): ``BattleSystem_TriggerLeftovers``
      (battle_lib.c:4922-4965, ``step._leftovers_check``).
    * ``TOXIC_ORB`` (``HOLD_EFFECT_PSN_USER``), ``FLAME_ORB`` (``HOLD_EFFECT_BRN_USER``),
      ``STICKY_BARB`` (``HOLD_EFFECT_DMG_USER_CONTACT_XFR``): ``BattleSystem_TriggerDetrimentalHeldItem``
      (battle_lib.c:5307-5350, ``step._detrimental_item_check``); the barb moves to an
      item-less contact attacker (``effects.sticky_barb_transfer``, :5387-5398, :7377-7384).
    * ``BRIGHT_POWDER`` (``HOLD_EFFECT_ACC_REDUCE``: BrightPowder, Lax Incense), ``WIDE_LENS``
      (``HOLD_EFFECT_ACCURACY_UP``), ``ZOOM_LENS`` (``HOLD_EFFECT_ACCURACY_UP_SLOWER``): the
      accuracy check (``outcome._rate``, battle_controller_player.c:2949-2965).
    * ``KINGS_ROCK`` (``HOLD_EFFECT_SOMETIMES_FLINCH``: King's Rock, Razor Fang): a flinch
      roll per hit (``Outcome.flinch_item``, ``effects._kings_rock``, bcp.c:4630-4653).
    * ``SHELL_BELL`` (``HOLD_EFFECT_HP_RESTORE_ON_DMG``): heals ``dealt / param`` after the
      move (``effects.shell_bell_heal``, bcp.c:4736-4752; Pursuit's hit: battle_lib.c:7345-7355).
    * ``LIGHT_CLAY`` (``HOLD_EFFECT_EXTEND_SCREENS``): Reflect / Light Screen last
      ``5 + param`` turns (``effects._screen``, battle_script.c:4229, :4272).
    """

    UNSUPPORTED = 0
    NONE = 1
    ORAN_BERRY = 2
    SITRUS_BERRY = 3
    FOCUS_SASH = 4
    BIG_ROOT = 5
    BERRY_JUICE = 6
    CHESTO_BERRY = 7
    PECHA_BERRY = 8
    LUM_BERRY = 9
    WHITE_HERB = 10
    LAGGING_TAIL = 11
    RESIST_BERRY = 12
    QUICK_CLAW = 13
    NO_EFFECT_BERRY = 14
    TYPE_BOOST = 15
    CHOICE_BAND = 16
    CHOICE_SPECS = 17
    CHOICE_SCARF = 18
    LIGHT_BALL = 19
    THICK_CLUB = 20
    METAL_POWDER = 21
    DEEP_SEA_TOOTH = 22
    DEEP_SEA_SCALE = 23
    SOUL_DEW = 24
    ADAMANT_ORB = 25
    LUSTROUS_ORB = 26
    GRISEOUS_ORB = 27
    MUSCLE_BAND = 28
    WISE_GLASSES = 29
    EXPERT_BELT = 30
    LIFE_ORB = 31
    CHERI_BERRY = 32
    RAWST_BERRY = 33
    ASPEAR_BERRY = 34
    PERSIM_BERRY = 35
    PINCH_BERRY = 36
    CUSTAP_BERRY = 37
    LEFTOVERS = 38
    BLACK_SLUDGE = 39
    TOXIC_ORB = 40
    FLAME_ORB = 41
    STICKY_BARB = 42
    BRIGHT_POWDER = 43
    WIDE_LENS = 44
    ZOOM_LENS = 45
    KINGS_ROCK = 46
    SHELL_BELL = 47
    LIGHT_CLAY = 48
    # Icy Rock, Smooth Rock, Heat Rock, Damp Rock (HOLD_EFFECT_EXTEND_HAIL / _SANDSTORM / _SUN
    # / _RAIN): ``item_param[i, 1]`` is the ``Weather`` it extends; the user's weather move of
    # that weather lasts ``5 + param`` turns (effect_script_0115.s, _0136.s, _0137.s, _0164.s:
    # CheckItemHoldEffect / GetItemEffectParam on the attacker; ``effects._weather``).
    WEATHER_ROCK = 49
    # Scope Lens, Razor Claw (HOLD_EFFECT_CRITRATE_UP), Lucky Punch (HOLD_EFFECT_CHANSEY_CRITRATE_UP)
    # and Stick (HOLD_EFFECT_FARFETCHD_CRITRATE_UP): ``item_param[i, 1]`` is the species the
    # item needs (0: any holder). ``BattleSystem_CalcCriticalMulti`` adds 1 stage for the first,
    # 2 for a species item on its species (battle_lib.c:7123-7128; ``outcome.crit_odds``).
    CRIT_ITEM = 50


# The three Choice items: ``BattleSystem_CheckInvalidMoves`` (battle_lib.c:2307-2315) and
# ``UpdateMoveBuffers`` (battle_controller_player.c:3923-3933) test exactly these hold
# effects.
CHOICE_ITEMS = (ItemId.CHOICE_BAND, ItemId.CHOICE_SPECS, ItemId.CHOICE_SCARF)

# Decomp species ids the item code of ``BattleSystem_CalcMoveDamage`` names
# (battle_lib.c:6730-6774; generated/species.txt line N is id N-1).
SPECIES_PIKACHU = 25
SPECIES_FARFETCHD = 83  # BattleSystem_CalcCriticalMulti (battle_lib.c:7128)
SPECIES_CHANSEY = 113  # :7127
SPECIES_CUBONE = 104
SPECIES_MAROWAK = 105
SPECIES_DITTO = 132
SPECIES_CLAMPERL = 366
SPECIES_LATIAS = 380
SPECIES_LATIOS = 381
SPECIES_DIALGA = 483
SPECIES_PALKIA = 484
SPECIES_GIRATINA = 487


# ``item_param[i, 1]`` of Chilan Berry (``HOLD_EFFECT_WEAKEN_NORMAL``): the berry halves any
# hit whose type is Normal, super-effective or not (``subscript_type_resist_berry.s:_128``);
# every other resist berry halves a super-effective hit of its type (``_135``-``_240``).
RESIST_ANY_NORMAL = -1


class TrainerItemId(IntEnum):
    """Effect of a trainer's bag item (``tables.trainer_item_effect``), used by the
    opponent's item actions (``actions.FIRST_ITEM_ACTION``). ``POTION`` restores
    ``POTION_HP_RESTORED`` HP to the active Pokemon, capped at its max HP."""

    UNSUPPORTED = 0
    NONE = 1
    POTION = 2


# Potion's ``itemUseParams.hpRestored`` (res/items/data/potion.json), the amount
# ``BattleSystem_UseBagItem`` (src/battle/battle_system.c:435) restores.
POTION_HP_RESTORED = 20

# Quick Claw's hold param (res/items/data/quick_claw.json ``effectParam`` 20, the only
# ``HOLD_EFFECT_SOMETIMES_PRIORITY`` item): ``BattleSystem_CompareBattlerSpeed`` activates it
# when ``speedRand % (100 / param) == 0`` (battle_lib.c:1278-1286, :1344-1352). The engine
# stores that test's result (``BattleState.qc_roll``, ``Outcome.quick_claw``), which is exact
# only while every holder has this param: ``game_constants`` asserts it for every game.
QUICK_CLAW_PARAM = 20
QUICK_CLAW_MODULUS = 100 // QUICK_CLAW_PARAM  # % 5


class Status(IntEnum):
    NONE = 0
    SLEEP = 1
    POISON = 2
    BURN = 3
    FREEZE = 4
    PARALYSIS = 5
    TOXIC = 6


class Stat(IntEnum):
    """Index into every stat vector (base stats, stats, nature mods) and stages."""

    HP = 0
    ATK = 1
    DEF = 2
    SPA = 3
    SPD = 4
    SPE = 5
    ACC = 6
    EVA = 7


class MoveFlag(IntFlag):
    """Bits of ``tables.move_flags``."""

    CONTACT = 1
    PROTECT = 2
    SOUND = 4
    PUNCH = 8
    KINGS_ROCK = 16  # MOVE_FLAG_TRIGGERS_KINGS_ROCK (BattleControllerPlayer_CheckExtraFlinch)
    # Engine bits from the decomp's code tables (tools/build_tables.py ``_move_flags``):
    # BtlCmd_TrySleepTalk's own exclusions (battle_script.c:4918-4925: Move_IsInvoker, Focus
    # Punch, Uproar, Chatter, Move_IsMultiTurn by the move's effect) and sCannotMetronomeMoves up
    # to its 0xFFFF delimiter (battle_lib.c:7152-7178, Move_CanBeMetronomed :7204-7218).
    NO_SLEEP_TALK = 32
    NO_METRONOME = 64
    # E2 batch 9 (B9-1): the move's decomp effect is BATTLE_EFFECT_THUNDER (hitRate 50 in sun,
    # bcp.c:2915-2917; MOVE_STATUS_MISSED cleared in rain, :3025-3027) or BATTLE_EFFECT_BLIZZARD
    # (cleared in hail, :3029-3031), all inside NO_CLOUD_NINE.
    THUNDER = 128
    BLIZZARD = 256


class MoveTarget(IntEnum):
    """``tables.move_target``: the decomp's ``RANGE_*`` (``generated/move_ranges.txt``
    order, without ``RANGE_UNUSED_*`` and ``RANGE_ALL``, which no move uses)."""

    RANGE_SINGLE_TARGET = 0
    RANGE_SINGLE_TARGET_SPECIAL = 1
    RANGE_RANDOM_OPPONENT = 2
    RANGE_ADJACENT_OPPONENTS = 3
    RANGE_ALL_ADJACENT = 4
    RANGE_USER = 5
    RANGE_USER_SIDE = 6
    RANGE_FIELD = 7
    RANGE_OPPONENT_SIDE = 8
    RANGE_ALLY = 9
    RANGE_USER_OR_ALLY = 10
    RANGE_SINGLE_TARGET_ME_FIRST = 11


class Weather(IntEnum):
    """``BattleState.weather``; also a battle's start condition (``make_state``)."""

    NONE = 0
    SUN = 1
    RAIN = 2
    SANDSTORM = 3
    HAIL = 4
    FOG = 5


# The switch-in weather abilities (SWITCH_IN_CHECK_STATE_WEATHER_ABILITIES, battle_lib.c:
# 3740-3790): each sets its weather permanently (subscript_drizzle.s, subscript_sand_stream.s,
# subscript_drought.s, subscript_snow_warning.s: FIELD_CONDITION_WEATHER off, then the
# ``*_PERM`` bit; the decomp skips the script when that permanent weather is already up, which
# leaves the same field), ``hooks.on_switch_in``.
WEATHER_ABILITIES = {
    AbilityId.DRIZZLE: Weather.RAIN,
    AbilityId.SAND_STREAM: Weather.SANDSTORM,
    AbilityId.DROUGHT: Weather.SUN,
    AbilityId.SNOW_WARNING: Weather.HAIL,
}
# Weather moves: NUM_WEATHER_TURNS (effect_script_0136.s etc. ``UpdateVar OPCODE_SET,
# BTLVAR_WEATHER_TURNS, 5``).
WEATHER_TURNS = 5


class Type(IntEnum):
    """Decomp type ids (``generated/pokemon_types.txt`` order), as stored in
    ``tables.move_type``, ``tables.species_types`` and the type chart axes."""

    NORMAL = 0
    FIGHTING = 1
    FLYING = 2
    POISON = 3
    GROUND = 4
    ROCK = 5
    BUG = 6
    GHOST = 7
    STEEL = 8
    MYSTERY = 9
    FIRE = 10
    WATER = 11
    GRASS = 12
    ELECTRIC = 13
    PSYCHIC = 14
    ICE = 15
    DRAGON = 16
    DARK = 17


# Decomp move id the engine names directly (``generated/moves.txt`` line 166, 0-based
# from MOVE_NONE): Struggle replaces the chosen move when it has no PP left.
MOVE_STRUGGLE = 165
# Snore and Sleep Talk (``generated/moves.txt`` lines 174, 215): the sleep check lets a sleeping
# battler use them (bcp.c:2455).
MOVE_SNORE = 173
MOVE_SLEEP_TALK = 214
# BtlCmd_Metronome (battle_script.c:4540): ``RandNext() % NUM_VALID_MOVES + 1`` with
# NUM_VALID_MOVES = MAX_MOVES - 1 (include/constants/moves.h:12; MAX_MOVES 468,
# generated/moves.txt), redrawn until the move is allowed: a move ``m`` weighs the count of u16
# values ``r`` with ``r % 467 == m - 1``: 141 for ``m <= 65536 % 467 = 156``, else 140.
METRONOME_DRAW = 467
METRONOME_HEAVY = 65536 % METRONOME_DRAW
# BtlCmd_TrySleepTalk (battle_script.c:4931-4933) redraws ``RandNext() % 4`` until the slot is
# allowed: uniform over the allowed slots (65536 % 4 == 0). ``Outcome.call`` holds a value
# ``u`` in 0..11 and the call is allowed slot ``u % count`` (12 is a multiple of every count
# 1-4), so the draw stays exact for whatever set the move sees when it runs.
SLEEP_TALK_VALUES = 12
# Fake Out (``generated/moves.txt`` line 253): ``subscript_flinch_mon.s:18`` shows Inner
# Focus's message, which names the ability, only for ``BTLVAR_CURRENT_MOVE == MOVE_FAKE_OUT``
# (or a direct flinch effect, which no supported move has).
MOVE_FAKE_OUT = 252
# Protect, Detect, Endure (``generated/moves.txt`` lines 183, 198, 204): ``BtlCmd_TryProtection``
# (battle_script.c:5154-5158) keeps the chain only when ``moveProtect`` is one of them, and
# ``AICmd_LoadProtectChain`` (trainer_ai.c:2546-2552) reads it the same way. The engine folds
# ``moveProtect`` into ``protect_count`` (``step._use_move``).
MOVE_PROTECT = 182
MOVE_DETECT = 197
MOVE_ENDURE = 203
# Baton Pass and U-turn (``generated/moves.txt`` lines 227, 370): UpdateMoveBuffers skips the
# Choice lock for them (bcp.c:3927-3928, by ``moveTemp``).
MOVE_BATON_PASS = 226
MOVE_U_TURN = 369
PROTECT_CHAIN_MOVES = (MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE)

# ABILITY_MULTITYPE (``generated/abilities.txt`` line 122): ``BtlCmd_TryCamouflage``
# (battle_script.c:8001-8004) fails for a Multitype user, read through
# ``tables.species_abilities_decomp`` (``damage.decomp_abilities``). Multitype itself is an
# unsupported ability, so coverage refuses a holder before this test can matter.
ABILITY_MULTITYPE = 121

# The battle terrain tables (include/data/terrain/to_move.h ``sTerrainMove``, to_type.h
# ``sTerrainCamouflageType``), indexed by the decomp's ``TERRAIN_*`` id
# (``nuzline.data.trainer.TERRAIN_NAMES``) up to ``TERRAIN_SPECIAL``: ``BtlCmd_GetTerrainMove``
# (battle_script.c:8036-8048) and ``BtlCmd_TryCamouflage`` (:8007-8012) read
# ``min(terrain, TERRAIN_SPECIAL)`` (include/constants/battle/terrain.h: ``TERRAIN_SPECIAL`` is
# ``TERRAIN_AARON``, 12). The moves are decomp move ids (``generated/moves.txt`` line N is id
# N-1): Earthquake 89, Seed Bomb 402, Rock Slide 157, Blizzard 59, Hydro Pump 56, Ice Beam 58,
# Tri Attack 161, Mud Bomb 426, Air Slash 403. tests/engine/test_terrain_moves.py reads both
# tables from the decomp and, through the ROM readers, from both ROMs.
TERRAIN_SPECIAL = 12
TERRAIN_MOVES = (89, 89, 402, 402, 157, 157, 59, 56, 58, 161, 426, 403, 161)

# Decomp ability ids the weather code names directly (``generated/abilities.txt`` line N is
# id N-1), read through ``tables.species_abilities_decomp`` (``Battler_Ability``) rather than
# an ``AbilityId``, for the abilities no table maps (Air Lock, Magic Guard, Ice Body) and the
# lock they share with Cloud Nine: Cloud Nine / Air Lock suppress weather (``NO_CLOUD_NINE``,
# include/battle/common.h:63-64); the end-of-turn chip (``BtlCmd_EndOfTurnWeatherEffect``,
# battle_script.c:5561-5647, and ``subscript_weather_continues.s``) skips Ice Body (which
# heals instead) and Magic Guard. Sand Veil and Snow Cloak are ``AbilityId`` members.
WEATHER_ABILITY_IDS = {
    "Cloud Nine": 13,
    "Air Lock": 76,
    "Magic Guard": 98,
    "Ice Body": 115,
}

# Decomp ability ids the held-item code names directly, read like ``WEATHER_ABILITY_IDS``
# (no table maps them): the status orbs' subscripts skip an Immunity holder (Toxic Orb,
# subscript_badly_poison.s:14), a Water Veil holder (Flame Orb, subscript_burn.s:6) and a Leaf
# Guard holder in sun (both, :15-17 / :7-9), each the holder's own ability (``CheckAbility``).
# Decomp ability ids Yawn's scripts name (subscript_yawn.s:5-9, :14; subscript_fall_asleep.s:7-14;
# line n + 1 of generated/abilities.txt is id n).
YAWN_ABILITY_IDS = {
    "Insomnia": 15,
    "Soundproof": 43,
    "Vital Spirit": 72,
    "Leaf Guard": 102,
}

ITEM_ABILITY_IDS = {
    "Immunity": 17,
    "Water Veil": 41,
    "Leaf Guard": 102,
}

# ``sTerrainCamouflageType`` (include/data/terrain/to_type.h), indexed like ``TERRAIN_MOVES``.
TERRAIN_TYPES = (
    Type.GROUND,  # TERRAIN_PLAIN
    Type.GROUND,  # TERRAIN_SAND
    Type.GRASS,  # TERRAIN_GRASS
    Type.GRASS,  # TERRAIN_PUDDLE
    Type.ROCK,  # TERRAIN_MOUNTAIN
    Type.ROCK,  # TERRAIN_CAVE
    Type.ICE,  # TERRAIN_SNOW
    Type.WATER,  # TERRAIN_WATER
    Type.ICE,  # TERRAIN_ICE
    Type.NORMAL,  # TERRAIN_BUILDING
    Type.GROUND,  # TERRAIN_GREAT_MARSH
    Type.FLYING,  # TERRAIN_BRIDGE
    Type.NORMAL,  # TERRAIN_SPECIAL
)
# ``sTerrainSideEffect`` (include/data/terrain/to_secondary_effect.h), indexed like
# ``TERRAIN_MOVES``: Secret Power's indirect TO_DEFENDER effect at the move's chance, as the
# engine's ``(EffectId, move_param)`` (``order.move_effects``).
SECRET_POWER_EFFECTS = (
    (EffectId.SECONDARY_STAT_FOE, (Stat.ACC, -1, 0)),  # TERRAIN_PLAIN: ACCURACY_DOWN_1_STAGE
    (EffectId.SECONDARY_STAT_FOE, (Stat.ACC, -1, 0)),  # TERRAIN_SAND
    (EffectId.SECONDARY_STATUS, (Status.SLEEP, 0, 0)),  # TERRAIN_GRASS: SLEEP
    (EffectId.SECONDARY_STATUS, (Status.SLEEP, 0, 0)),  # TERRAIN_PUDDLE
    (EffectId.FLINCH, (0, 0, 0)),  # TERRAIN_MOUNTAIN: FLINCH
    (EffectId.FLINCH, (0, 0, 0)),  # TERRAIN_CAVE
    (EffectId.SECONDARY_STATUS, (Status.FREEZE, 0, 0)),  # TERRAIN_SNOW: FREEZE
    (EffectId.SECONDARY_STAT_FOE, (Stat.ATK, -1, 0)),  # TERRAIN_WATER: ATTACK_DOWN_1_STAGE
    (EffectId.SECONDARY_STATUS, (Status.FREEZE, 0, 0)),  # TERRAIN_ICE
    (EffectId.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),  # TERRAIN_BUILDING: PARALYZE
    (EffectId.SECONDARY_STAT_FOE, (Stat.SPE, -1, 0)),  # TERRAIN_GREAT_MARSH: SPEED_DOWN
    (EffectId.SECONDARY_STAT_FOE, (Stat.EVA, -1, 0)),  # TERRAIN_BRIDGE: EVASION_DOWN
    (EffectId.SECONDARY_STATUS, (Status.PARALYSIS, 0, 0)),  # TERRAIN_SPECIAL: PARALYZE
)
# effect_script_0036.s: ``Random 2, 3`` (``BtlCmd_Random``, battle_script.c:3200-3210:
# ``RandNext() % 3 + 3``) picks MOVE_SUBSCRIPT_PTR_BURN, _FREEZE or _PARALYZE (3, 4, 5:
# generated/battle_move_subscript_ptrs.txt lines 4-6), ``Outcome.tri_attack`` 0, 1, 2.
TRI_ATTACK_STATUSES = (Status.BURN, Status.FREEZE, Status.PARALYSIS)
