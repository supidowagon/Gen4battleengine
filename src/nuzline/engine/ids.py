"""The engine's vocabulary: effect, ability, item, status, stat, move-flag and type ids.

Tables store these ids (``tables.move_effect``, ``species_abilities``,
``item_effect``, ``move_flags``); the decomp's own ids are mapped onto them by
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
    * ``DRAIN``: ``(percent of damage dealt, 0, 0)``; ``RECOIL``: ``(divisor of the damage
      dealt, 0, 0)`` (``DivideVarByValue``: 2, 3 or 4; E2 batch 9)
    * everything else: zeros. ``LEVEL_DAMAGE`` (Night Shade, Seismic Toss) deals the user's
      level; ``HIDDEN_POWER`` and ``MAGNITUDE`` take their type and power from the
      battler and ``Outcome.magnitude`` (``engine/damage.py``).
    * ``SECONDARY_CONFUSE`` (Confusion, Psybeam, ...): zeros; confuses the target at the
      move's effect chance. ``MUD_SPORT``: zeros.
    * ``LEVEL_DAMAGE``: ``(amount, 0, 0)``: the amount (Dragon Rage 40, SonicBoom 20), 0 for the
      user's level (Night Shade, Seismic Toss). ``EAT_BERRY`` (Bug Bite, Pluck): zeros;
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
      A move of another family with ``MoveFlag.HIT_TWICE`` also hits twice (Twineedle:
      ``SECONDARY_STATUS`` poison, a roll per hit; E2 batch 13, B13-1).
    * ``TRIPLE_KICK``: ``(step, 0, 0)``: three hits, hit k at power k x step (vanilla 10), or
      k x the move's power for step 0 (Kaizo); each later hit rolls its own accuracy check
      and the first miss ends the move. ``BEAT_UP``: zeros; one hit per eligible party member
      (``effects.beat_up_slots``), damage from its species and level (``effects.beat_up_damage``).
      E2 batch 13.
    * Multi-stat effects: the stat list is the effect's own, so the params stay zeros.
      ``RAISE_ALL_STATS_HIT`` (AncientPower): Atk, Def, Spe, SpA, SpD +1 on the user at the
      move's effect chance; ``CURSE`` (a non-Ghost user; a Ghost-type one is ``_curse``, B8-5):
      Spe -1, Atk +1, Def +1 on the user.
    * ``RECOVER``: zeros; heals ``maxHP / 2``, fails at full HP. ``FAKE_OUT``: zeros; fails
      unless it is the user's first turn out, else flinches at the move's chance.
    * ``SELF_DESTRUCT`` (Selfdestruct, Explosion; Kaizo's Memento): zeros; the move script
      sets the user's HP to 0 before the accuracy check, and a physical one hits the halved
      Defense (``damage.py``). Damp would stop it; Damp is ``AbilityId.UNSUPPORTED``.
    * ``FLAIL`` (Flail, Reversal): ``(kind, 0, 0)``; the power comes from the user's HP pixels
      (``damage.flail_power``); the trainer AI scores it at the table power. Kind
      ``FLAIL_FALLOFF`` (Water Spout, Eruption): power x curHP / maxHP (``damage.falloff_power``).
    * ``WEATHER_BALL``, ``GYRO_BALL``, ``ASSURANCE``, ``TRUMP_CARD``, ``PUNISHMENT``,
      ``JUDGMENT``: zeros; their power or type comes from the battle (``damage._pre_random``,
      ``damage.move_type_of``), E2 batch 13. ``NATURAL_GIFT``: zeros; the power and type of the
      held item (``damage.gift_item``), which the script removes. ``RECYCLE``, ``EMBARGO``:
      zeros.
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
    # Swagger (effect_script_0118.s, subscript_swagger.s) and Flatter (effect_script_0166.s,
    # subscript_flatter.s, E2 batch 12): ``(Stat, stages, 0)``: Attack +2 / Sp. Atk +1, then
    # confusion (``effects._swagger``).
    SWAGGER = 55
    # Dragon Dance (effect_script_0212.s, subscript_user_atk_and_speed_up_1_stage.s), Bulk Up,
    # Calm Mind, Cosmic Power / Defend Order (Kaizo's Stockpile): ``(Stat, Stat, 0)``, the two
    # stats +1 on the user in that order; both at +6 is MOVE_STATUS_NO_MORE_WORK
    # (``effects._dragon_dance``).
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
    # E2 batch 9 (B9-5). Follow Me (effect_script_0172.s): BtlCmd_FollowMe sets the user side's
    # followMe and followMeUser (battle_script.c:6261-6270), which BattleSystem_Defender
    # (battle_lib.c:1714-1717), the redirection abilities (:1743) and Counter (battle_script.c:
    # 4628) read for the foe's moves: in singles they name the side's one battler, the target
    # anyway, so the move only shows its message. Helping Hand (effect_script_0176.s):
    # BtlCmd_TryHelpingHand (:6291-6314) needs a double battle and a partner; in singles it jumps
    # to MOVE_STATUS_FAILED (:6309-6311, effect_script_0176.s:15-17).
    FOLLOW_ME = 77
    HELPING_HAND = 78
    # Belly Drum (subscript_belly_drum.s, B9-10): fails at Attack +6 or curHP <= Divide(maxHP, 2)
    # (:3-6, MOVE_STATUS_FAILED); else Attack +6 and Divide(maxHP, 2) lost through UPDATE_HP.
    BELLY_DRUM = 79
    # E2 batch 10, B10-1. Volt Tackle's family (262, BATTLE_EFFECT_RECOIL_PARALYZE_HIT) and Flare
    # Blitz's (253, _RECOIL_BURN_HIT): ``(Status, divisor, thaws_user)``. The script's indirect
    # MOVE_SIDE_EFFECT_PROBABILISTIC subscript (subscript_recoil_1_3_chance_to_{paralyze,burn}.s)
    # runs on every hit that hit: recoil Divide(hitDamage, divisor) (Rock Head, Magic Guard skip
    # it), then the status as a secondary one when the roll passed and the defender had HP
    # (battle_lib.c:1551-1567). Flare Blitz's family thaws its frozen user (bcp.c:2481, :2730).
    RECOIL_STATUS = 80
    # Jump Kick / Hi Jump Kick (45, BATTLE_EFFECT_CRASH_ON_MISS): zeros. A plain hit whose
    # SYSCTL_CRASH_DAMAGE makes subscript_missed.s run subscript_crash_on_miss.s: the user loses
    # min(Divide(damage, 2), Divide(defender maxHP, 2)) of the damage the script computed
    # (crit and roll included, then the type chart), Magic Guard excepted (B10-2).
    CRASH = 81
    # Substitute (79, BATTLE_EFFECT_SET_SUBSTITUTE): zeros. subscript_try_substitute.s: fails
    # behind one or at curHP <= Divide(maxHP, 4); else ``substitute_hp`` = Divide(maxHP, 4), the
    # user's bind cleared and the HP lost (``effects._substitute``, B10-4).
    SUBSTITUTE = 82
    # Endeavor (189, BATTLE_EFFECT_SET_HP_EQUAL_TO_USER): zeros. effect_script_0189.s: fails
    # (MOVE_STATUS_FAILED, :7, :13-14) unless the defender's curHP exceeds the attacker's; else
    # battleCtx->damage = their difference under SYSCTL_IGNORE_TYPE_CHECKS (no CalcCrit /
    # CalcDamage; an immunity still flags), B10-5.
    ENDEAVOR = 83
    # Pain Split (91, BATTLE_EFFECT_AVERAGE_HP): zeros. A direct ON_HIT subscript_pain_split.s:
    # fails behind a substitute (:5, :50-52); else both battlers' HP become (a + d) / 2 through
    # UPDATE_HP (:7-19), B10-5.
    PAIN_SPLIT = 84
    # Knock Off (188, BATTLE_EFFECT_REMOVE_HELD_ITEM): zeros. A table hit with an indirect ON_HIT
    # subscript_knock_off.s (``effects._knock_off``), B10-6.
    KNOCK_OFF = 85
    # Safeguard (124, BATTLE_EFFECT_PREVENT_STATUS): zeros. subscript_safeguard_start.s ->
    # BtlCmd_TrySafeguard (battle_script.c:5754-5772): fails with the user's side's Safeguard up,
    # else 5 turns on it (``effects._safeguard``), B10-7.
    SAFEGUARD = 86
    # Focus Punch (170, BATTLE_EFFECT_HIT_LAST_WHIFF_IF_HIT): zeros. effect_script_0170.s: a
    # user with its turnFlags physical / specialDamageAttackerMask set (a hit on its HP this turn,
    # bcp.c:3416-3424) gets MOVE_STATUS_LOST_FOCUS (:5-6, :12-16; a DID_NOT_HIT status:
    # subscript_missed.s:8-10, "lost its focus"); else a plain table hit (B10-8).
    FOCUS_PUNCH = 87
    # Encore (90, BATTLE_EFFECT_ENCORE): zeros. A direct ON_HIT subscript_encore_start.s ->
    # BtlCmd_TryEncore (battle_script.c:4725-4748), ``effects._encore``, B10-9.
    ENCORE = 88
    # Assist (180, BATTLE_EFFECT_USE_RANDOM_ALLY_MOVE): zeros. effect_script_0180.s: BtlCmd_TryAssist
    # (battle_script.c:6405-6444) then GoToMoveScript: an invoker (``order.CALL_ASSIST``), B10-11.
    ASSIST = 89
    # E2 batch 12 (rulings B12-n), the stat family. Tickle (205, BATTLE_EFFECT_ATK_DEF_DOWN):
    # ``(Stat, Stat, 0)``, a direct CHECK_HP | TO_DEFENDER subscript_target_atk_and_def_down_1_stage.s:
    # both stats -1 through UPDATE_STAT_STAGE in that order, MOVE_STATUS_NO_MORE_WORK with both at
    # -6 (:5-6), FAILED on a miss (:9) (``effects._tickle``).
    TICKLE = 90
    # Focus Energy (47, BATTLE_EFFECT_CRIT_UP_2): zeros. effect_script_0047.s fails with the user's
    # VOLATILE_CONDITION_FOCUS_ENERGY set (:5, :10-11), else a direct TO_ATTACKER
    # subscript_focus_energy.s sets it (:6): ``BattleState.focus_energy``, +2 crit stages
    # (BattleSystem_CalcCriticalMulti, battle_lib.c:7123).
    FOCUS_ENERGY = 91
    # Haze (25, BATTLE_EFFECT_RESET_STAT_CHANGES): zeros. An indirect ON_HIT
    # subscript_reset_all_stat_stages.s: BtlCmd_ResetAllStatChanges (battle_script.c:3534-3549)
    # zeroes every battler's stages and clears its Focus Energy.
    HAZE = 92
    # Psych Up (143, BATTLE_EFFECT_COPY_STAT_CHANGES): zeros. effect_script_0143.s:
    # BtlCmd_CopyStatStages (battle_script.c:6042-6053): the defender's stages, and its Focus
    # Energy ORed in, onto the attacker; no failure.
    PSYCH_UP = 93
    # Power Swap (243), Guard Swap (244): ``(Stat, Stat, 0)``. An indirect ON_HIT
    # subscript_exchange_{atk_and_spatk,def_and_spdef}_stages.s: the two stages swapped between
    # the attacker and the defender (:5-12); no failure.
    STAGE_SWAP = 94
    # Captivate (265, BATTLE_EFFECT_SP_ATK_DOWN_2_OPPOSITE_GENDER): ``(Stat, stages, 0)``.
    # effect_script_0265.s: an ignorable Oblivious (:5), a genderless attacker or defender
    # (:6-7) or the same gender (:8-9) ends it with MOVE_STATUS_NO_MORE_WORK; else a direct
    # TO_DEFENDER Sp. Atk -2 (:10) (``effects._captivate``).
    CAPTIVATE = 95
    # Acupressure (226, BATTLE_EFFECT_RANDOM_STAT_UP_2): zeros. effect_script_0226.s fails behind
    # the defender's substitute (:5) or with no stage below +6 (BtlCmd_BoostRandomStatBy2,
    # battle_script.c:8932-8956); else a direct TO_DEFENDER +2 on a stat drawn uniformly among
    # those below +6 (``Outcome.acupressure``). In singles its defender is the user (RANGE_USER_OR_ALLY).
    ACUPRESSURE = 96
    # Power Trick (238, BATTLE_EFFECT_SWAP_ATK_DEF): zeros. An indirect ON_HIT
    # subscript_user_swap_atk_and_def.s: the user's MOVE_EFFECT_POWER_TRICK toggled (:5) and its
    # BattleMon Attack and Defense swapped (:6-9) (``BattleState.power_trick``,
    # ``damage.battle_stats``).
    POWER_TRICK = 97
    # E2 batch 12, the healing family. Rest (37, BATTLE_EFFECT_REST): zeros. A direct ON_HIT
    # subscript_rest.s: the user's own Insomnia or Vital Spirit (CheckAbility, tagging it), sleep,
    # an uproar without Soundproof or full HP end it with MOVE_STATUS_NO_MORE_WORK; else its status
    # becomes sleep for ``REST_TURNS`` and it regains its max HP through UPDATE_HP (``effects._rest``).
    REST = 98
    # Morning Sun, Synthesis, Moonlight (132, BATTLE_EFFECT_HEAL_HALF_MORE_IN_SUN): zeros.
    # BtlCmd_WeatherHPRecovery (battle_script.c:5966-5979): maxHP / 2 with no weather (or a weather
    # lock), Divide(maxHP * 20, 30) in sun, else Divide(maxHP, 4); then subscript_present_heal.s ->
    # subscript_recover_hp.s ("HP is full!" at full HP) (``effects._weather_heal``).
    WEATHER_HEAL = 99
    # Wish (179, BATTLE_EFFECT_HEAL_IN_3_TURNS): zeros. BtlCmd_TryWish (battle_script.c:6373-6385)
    # fails with the user's battler's wishTurns set, else sets it to 2 (``BattleState.wish_timer``);
    # the field pass counts it down and heals that battler's Pokemon then (``step._wish``).
    WISH = 100
    # Heal Bell, Aromatherapy (102, BATTLE_EFFECT_CURE_PARTY_STATUS): zeros. An indirect ON_HIT
    # subscript_heal_bell.s: BtlCmd_TryPartyStatusRefresh (battle_script.c:5030-5078) and the party
    # (BtlIOCmd_RefreshPartyStatus, battle_io_command.c:634-656); Heal Bell spares a Soundproof
    # holder (``effects._heal_bell``).
    HEAL_BELL = 101
    # Refresh (193, BATTLE_EFFECT_HEAL_STATUS): zeros. A direct ON_HIT subscript_recover_psn_prz_brn.s:
    # the user's poison, toxic, burn or paralysis cured (:5-7), else MOVE_STATUS_FAILED (:15-16).
    REFRESH = 102
    # Psycho Shift (234, BATTLE_EFFECT_TRANSFER_STATUS): zeros. effect_script_0234.s:
    # BtlCmd_CheckCanShareStatus (battle_script.c:7542-7553) fails unless the defender has no
    # status nor substitute and the user has one; else a direct TO_DEFENDER
    # subscript_give_target_own_status.s gives the user's status through its status subscript, and
    # the user's is cleared when the defender then has one (``effects._psycho_shift``).
    PSYCHO_SHIFT = 103
    # Mist (46, BATTLE_EFFECT_PREVENT_STAT_REDUCTION): zeros. effect_script_0046.s: a direct
    # ON_HIT subscript_mist.s -> BtlCmd_TryMist (battle_script.c:4300-4317): fails
    # (MOVE_STATUS_FAILED) with SIDE_CONDITION_MIST on the user's side, else mistTurns =
    # NUM_SCREEN_TURNS (5) on it (``effects._mist``). BtlCmd_ChangeStatStage
    # (:2875-2882) blocks every drop by another battler while mistTurns is set (B12-23).
    MIST = 104
    # E2 batch 13, the multi-hit power family.
    # Triple Kick (vanilla 104, BATTLE_EFFECT_HIT_THREE_TIMES; Kaizo be_seq 104: Triple Kick, Rock
    # Wrecker; Kaizo be_seq 88, byte-identical to its 104: Triple Axel, Fury Cutter):
    # ``(step, 0, 0)``. ``SetMultiHit 3, SYSCTL_TRIPLE_KICK`` (effect_script_0104.s:5): three
    # hits; the script reloads per hit (bcp.c:3789) and adds to BTLVAR_MOVE_POWER, which
    # BattleContext_Init zeroes once per action (battle_lib.c:1979) and BattleSystem_SetupLoop
    # keeps (:3357-3362), before CalcCrit and CalcDamage: hit k (1-3) has power k x step, step
    # the vanilla script's constant 10 (``UpdateVar OPCODE_ADD, BTLVAR_MOVE_POWER, 10``, :7), or
    # 0 for the Kaizo scripts' ``GetCurrentMoveData MOVEATTRIBUTE_POWER`` (the move's own
    # power; rom_diff.toml be_seq 88, 104). SYSCTL_TRIPLE_KICK lacks SYSCTL_SKIP_ACCURACY_CHECK
    # but has _SKIP_ACCURACY_OVERRIDES (include/constants/battle/system_control.h:81-86): every
    # later hit rolls CheckMoveHitAccuracy again (bcp.c:3255-3259) without
    # CheckMoveHitOverrides (:3264-3268: no No Guard, Lock-On, Thunder / Blizzard weather), and
    # a later miss is MOVE_STATUS_MULTI_HIT_DISRUPTED (bcp.c:3320-3323), which ends the loop
    # (:3782). ``Outcome.hit[side, k]`` is hit k's roll (``outcome.hit_thresholds``,
    # ``overrides=False``), B13-5..B13-8.
    TRIPLE_KICK = 105
    # Beat Up (154, BATTLE_EFFECT_BEAT_UP): zeros. effect_script_0154.s: SYSCTL_IGNORE_IMMUNITIES
    # (:5), CalcCrit, then BtlCmd_BeatUp (battle_script.c:6173-6247) for its damage, no
    # CalcDamage: one hit per party slot, in party order, that is the user's or has HP, a
    # species, no egg and no status (BeatUpEligibleMon, :6157-6164); hit damage = the member's
    # species base Attack x power x (its level x 2 / 5 + 2) / the defender's species base
    # Defense / 50 + 2, x criticalMul, Helping Hand, then the variance (:6206-6218); no stage,
    # item, ability, status, screen, weather, STAB or type chart (CheckTypeChart skips it under
    # SYSCTL_IGNORE_IMMUNITIES, bcp.c:2355). Accuracy once (SYSCTL_MULTI_HIT_MOVE, :6187).
    # ``effects.beat_up_damage``, B13-9..B13-11.
    BEAT_UP = 106
    # E2 batch 13, single-hit variable power and type (B13-12..).
    # Weather Ball (203, effect_script_0203.s; BtlCmd_CalcWeatherBallParams, battle_script.c:
    # 6843-6872): zeros. Under NO_CLOUD_NINE and any FIELD_CONDITION_WEATHER bit (fog included,
    # condition.h:131-135) movePower = power x 2 and moveType Water / Rock / Fire / Ice for rain /
    # sand / sun / hail (fog keeps Normal); else nothing (the table power and type).
    # ``damage.weather_ball_params``; the type is battleCtx->moveType, which the chart, STAB,
    # the type items and the absorbing abilities read (``damage.move_type_of``).
    WEATHER_BALL = 107
    # Gyro Ball (219, BtlCmd_CalcGyroBallPower, battle_script.c:7123-7131): zeros. movePower =
    # 1 + 25 x monSpeedValues[defender] / monSpeedValues[attacker], at most 150; the speeds are
    # CompareBattlerSpeed's of BranchActions' SortMonSpeedOrder at the user's action start
    # (bcp.c:888; battle_lib.c:1372-1373). ``damage.gyro_ball_power``.
    GYRO_BALL = 108
    # Assurance (231, effect_script_0231.s): zeros. movePower = the move's power, x 2 when the
    # defender's turnFlags.assuranceDamageMask is set (:5-9): subscript_update_hp.s:18-19 writes
    # each update's hpCalcTemp <= 0 there (``BattleState.ts_assurance``).
    ASSURANCE = 109
    # Trump Card (235, BtlCmd_CalcTrumpCardPower, battle_script.c:7233-7245): zeros. movePower
    # = sCurrentPPScaledPower[min(ppCur, 4)] (200, 80, 60, 50, 40) of the user's moveSlot after
    # the PP decrement (bcp.c:3182). ``damage.trump_card_power``.
    TRUMP_CARD = 110
    # Punishment (245, BtlCmd_CalcPunishmentPower, battle_script.c:7354-7371): zeros. movePower
    # = 60 + 20 x the defender's stages above neutral (HP through evasion), at most 200.
    # ``damage.punishment_power``.
    PUNISHMENT = 111
    # Judgment (268, effect_script_0268.s): zeros. The attacker's held item's hold effect
    # (Battler_HeldItemEffect) HOLD_EFFECT_ARCEUS_* sets moveType to its type (:5-95);
    # ``damage.judgment_type``.
    JUDGMENT = 112
    # E2 batch 13, the held-item moves (B13-19..).
    # Recycle (184, effect_script_0184.s): zeros. Fails (MOVE_STATUS_FAILED) when the user holds
    # an item (:5) or its battler's recycleItem is empty (BtlCmd_TryRecycle, battle_script.c:
    # 9000-9013, which clears it); else the user holds that item again (:7, UpdateMonDataFromVar:
    # CopyToParty, no AI reveal). ``BattleState.recycle_item``; ``step._use_move``.
    RECYCLE = 113
    # Natural Gift (222, effect_script_0222.s): zeros. BtlCmd_CalcNaturalGiftParams (battle_script.c:
    # 8088-8102) sets movePower and moveType from the held item's naturalGiftPower / _Type
    # (Battler_HeldItem: Klutz and Embargo hide it, B13-29..31), failing at power 0; the script
    # then RemoveItem's the item (:8) before TryMove. ``damage.gift_item`` (B13-37).
    NATURAL_GIFT = 114
    # Embargo (232, effect_script_0232.s): zeros. Fails behind a substitute (:5, :7-8); else a
    # direct ON_HIT subscript_embargo_start.s on the defender: fails (MOVE_STATUS_FAILED, :57)
    # on a miss, a substitute, Multitype, an embargo already up, a held Griseous Orb or the
    # defender's quickClaw / custapBerry flag (:5-11); else embargoTurns = 5 and its
    # MOVE_EFFECT_EMBARGO bit (:13-14): ``BattleState.embargo``, read by ``state.held_item``
    # (B13-31).
    EMBARGO = 115
    # Endure (116, BATTLE_EFFECT_SURVIVE_WITH_1_HP): zeros. effect_script_0116.s:3 is the same direct
    # TO_ATTACKER ``MOVE_SUBSCRIPT_PTR_PROTECT`` as Protect (effect_script_0111.s:3), so
    # ``BtlCmd_TryProtection`` (battle_script.c:5149-5190) and its chain and roll are shared; a
    # success sets ``turnFlags.enduring`` (:5174-5177, not ``protecting``). ``enduring`` is read at
    # one site, ``BattleControllerPlayer_UpdateHP`` (battle_controller_player.c:3386-3404;
    # ``effects.play_hit``, G-65).
    ENDURE = 116
    # Magnet Rise (252, BATTLE_EFFECT_GIVE_GROUND_IMMUNITY): zeros. effect_script_0252.s fails
    # (MOVE_STATUS_FAILED) for a user with ``magnetRiseTurns``, Levitate (``Battler_Ability``) or
    # Ingrain, else sets ``MOVE_EFFECT_MAGNET_RISE`` and ``magnetRiseTurns = MAGNET_RISE_TURNS``
    # (``effects._magnet_rise``, G-66).
    MAGNET_RISE = 117
    # Gravity (215, BATTLE_EFFECT_GRAVITY): zeros. effect_script_0215.s fails (MOVE_STATUS_FAILED)
    # with ``FIELD_CONDITION_GRAVITY`` up, else the indirect subscript_gravity_start.s: Gravity for
    # ``GRAVITY_TURNS`` and its grounding loop (``effects._gravity``, G-67).
    GRAVITY = 118
    # Mimic (82, BATTLE_EFFECT_COPY_MOVE_FOR_BATTLE): zeros. effect_script_0082.s runs
    # subscript_mimic.s on a hit: ``BtlCmd_TryMimic`` (battle_script.c:4476-4523) copies the
    # defender's last move into the user's first Mimic slot (``effects._mimic``, G-68).
    MIMIC = 119
    # Transform (57, BATTLE_EFFECT_TRANSFORM): zeros. effect_script_0057.s runs
    # subscript_transform_into_target.s on a defender with HP: ``BtlCmd_Transform``
    # (battle_script.c:5330-5373) copies the defender's BattleMon (``effects._transform``, G-69).
    TRANSFORM = 120
    # Chatter (267, BATTLE_EFFECT_CHATTER): zeros. effect_script_0267.s is a plain hit with an ON_HIT
    # indirect ``subscript_chatter.s``: ``CheckChatterActivation`` (battle_script.c:8397-8436) then
    # subscript_confuse.s (``effects._chatter``, G-70).
    CHATTER = 121


# effect_script_0252.s:9: ``magnetRiseTurns`` at the move's success (G-66).
MAGNET_RISE_TURNS = 5
# FIELD_CONDITION_GRAVITY_INIT (condition.h:143): Gravity's turns, set by subscript_gravity_start.s:9
# (and, in Kaizo, again at 0 by subscript_gravity_end, sub_seq 238: ``Tables.gravity_rearm``).
GRAVITY_TURNS = 5
# The effects whose script sets only ``MOVE_SIDE_EFFECT_ON_HIT`` in its indirect flags and rolls
# nothing through ``effectChance``: BattleSystem_TriggerSecondaryEffect's ON_HIT branch
# (battle_lib.c:1524-1529) runs the subscript with no ``RandNext() % 100 < effectChance`` (only
# the PROBABILISTIC branch, :1549-1567, the bare-flags one, :1568-1585, and
# BtlCmd_CheckEffectActivation, battle_script.c:8366-8381, read it). Chatter
# (effect_script_0267.s:5) draws its own roll (CheckChatterActivation, :8397-8436), so its move
# row's effect_chance (Kaizo 50) is never read: ``outcome.read_chance`` is 0 for these (G-70).
ON_HIT_EFFECTS = (EffectId.CHATTER,)

# ``EffectId.COUNTER``'s source (``move_param[m, 0]``).
COUNTER_PHYSICAL, COUNTER_SPECIAL, COUNTER_ANY = 1, 2, 3

# Snore (effect_script_0092.s) is ``FLINCH`` with ``move_param[m, 2] == SNORE_PARAM``: the
# script fails unless its user is asleep (:5, :15-17), and a sleeping user uses it
# (bcp.c:2455: ``moveCur == MOVE_SNORE``).
SNORE_PARAM = 1
# Stomp (effect_script_0150.s) is ``FLINCH`` with ``move_param[m, 2] == STOMP_PARAM``: powerMul 20
# against a defender with MOVE_EFFECT_MINIMIZE (:6-8). Minimize (subscript_minimize.s) is
# ``STAT_SELF`` (Evasion +1) with ``move_param[m, 2] == MINIMIZE_PARAM``: it sets the user's
# ``minimized`` first (:5). E2 batch 9, B9-10.
STOMP_PARAM = 2
MINIMIZE_PARAM = 2
# subscript_tailwind_start.s:11 sets both SIDE_CONDITION_TAILWIND bits: a 2-bit counter at 3.
TAILWIND_TURNS = 3
SPIKES_MAX = 3  # battle_script.c:5394
TOXIC_SPIKES_MAX = 2  # battle_script.c:7598

# ``EffectId.DOUBLE_POWER``'s condition (``move_param[m, 0]``).
POWER_IF_TARGET_PARALYZED = 1  # effect_script_0171.s:6 (SmellingSalt)
POWER_IF_USER_STATUSED = 2  # effect_script_0169.s:5 (Facade, MON_CONDITION_FACADE_BOOST)
POWER_IF_TARGET_HALF_HP = 3  # effect_script_0221.s:5-8 (Brine)
# BtlCmd_CalcPaybackPower (battle_script.c:7198-7209, Payback): x2 once the defender has acted
# this turn (``target_moved`` in effects.play_hit), B9-12.
POWER_IF_TARGET_MOVED = 4
# Wake-Up Slap (effect_script_0217.s:5-8): SmellingSalt's script with MON_CONDITION_SLEEP and
# subscript_heal_target_sleep.s (``cures``), B13-16.
POWER_IF_TARGET_ASLEEP = 5
# ``EffectId.FLAIL``'s kind (``move_param[m, 0]``): 0 is Flail's HP-pixel bands
# (BtlCmd_CalcFlailPower); ``FLAIL_FALLOFF`` is Water Spout / Eruption (vanilla 190,
# BtlCmd_CalcHPFalloffPower, battle_script.c:6645-6657): power x curHP / maxHP, at least 1.
FLAIL_FALLOFF = 1


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
    EffectId.ENDEAVOR,  # effect_script_0189.s: the HP difference (B10-5)
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
    # B9-6: Volt Absorb / Water Absorb (BattleSystem_TriggerImmunityAbility, battle_lib.c:
    # 3500-3512; step._use_move beside Dry Skin), Rain Dish (BtlCmd_EndOfTurnWeatherEffect,
    # battle_script.c:5614-5618; step._weather_effect), Clear Body / Hyper Cutter
    # (BtlCmd_ChangeStatStage, :2884-2902; effects._foe_stage, hooks.intimidate).
    VOLT_ABSORB = 66
    WATER_ABSORB = 67
    RAIN_DISH = 68
    CLEAR_BODY = 69
    HYPER_CUTTER = 70
    # B9-7: Aftermath (BattleSystem_TriggerAbilityOnHit, battle_lib.c:4358-4371;
    # effects._contact_ability).
    AFTERMATH = 71
    # B9-8: Poison Heal (subscript_poison_damage.s:4-12; step._status_damage) and Bad Dreams
    # (MON_COND_CHECK_STATE_BAD_DREAMS, bcp.c:1484-1500; step._bad_dreams_check).
    POISON_HEAL = 72
    BAD_DREAMS = 73
    # B9-11: the trapping abilities (Battler_IsTrapped, battle_lib.c:5514-5536; actions.trapped).
    ARENA_TRAP = 74
    SHADOW_TAG = 75
    MAGNET_PULL = 76
    # E1 Gardenia batch, rulings G-8..G-11: Pressure's PP cost (BattleControllerPlayer_DecrementPP,
    # bcp.c:2235-2270; effects.pressure_pp_cost), the Pursuit and Counter-family extra PP
    # (battle_script.c:6919-6922, BattleSystem_DecPPForPressure battle_lib.c:7390-7397) and its
    # switch-in announcement (SWITCH_IN_CHECK_STATE_PRESSURE, battle_lib.c:4096-4110).
    PRESSURE = 77
    # E1 Gardenia batch, rulings G-12..G-17: Synchronize (BattleSystem_SynchronizeStatus,
    # battle_lib.c:4493-4534, from AFTER_MOVE_EFFECT_SYNCHRONIZE_STATUS, bcp.c:3665-3670;
    # effects.synchronize) and Leaf Guard in sun (the status subscripts' CheckIgnoreWeather /
    # FIELD_CONDITION_SUNNY / ABILITY_LEAF_GUARD tests; effects._inflict, _contact_ability).
    LEAF_GUARD = 78
    SYNCHRONIZE = 79
    # E1 Gardenia batch, rulings G-18..G-23: Wonder Guard (BattleSystem_ApplyTypeChart's
    # MOVE_STATUS_WONDER_GUARD, battle_lib.c:2649-2654; damage.wonder_guarded) and Download
    # (SWITCH_IN_CHECK_STATE_DOWNLOAD, battle_lib.c:3814-3851; hooks.download).
    WONDER_GUARD = 80
    DOWNLOAD = 81
    # E1 Batch C, rulings G-45..G-49: Ice Body (BtlCmd_EndOfTurnWeatherEffect, battle_script.c:
    # 5597-5602; step._weather_effect), White Smoke (BtlCmd_ChangeStatStage's Clear Body branch,
    # :2883-2884; effects._foe_stage, hooks.intimidate), Vital Spirit (every Insomnia site:
    # subscript_fall_asleep.s:8, :24, subscript_yawn.s:6, subscript_rest.s:8,
    # BattleSystem_RecoverStatusByAbility battle_lib.c:4404-4405, :4471-4472), Speed Boost
    # (BattleSystem_TriggerTurnEndAbility, battle_lib.c:3555-3564; step._turn_end_ability) and
    # Iron Fist (BattleSystem_CalcMoveDamage, battle_lib.c:6913-6918; damage._pre_random).
    ICE_BODY = 82
    WHITE_SMOKE = 83
    VITAL_SPIRIT = 84
    SPEED_BOOST = 85
    IRON_FIST = 86
    # E2 batch 13 (B13-29): Klutz. Battler_HeldItem returns ITEM_NONE for it (battle_lib.c:
    # 5354-5356): every gated held-item read (``state.held_item``). Its other reads are Fling's
    # (:5840, an unsupported move) and the AI scripts of Fling, Trick and Skill Swap
    # (trainer_ai/script.s:1245, :1273, :4270, :4293, :4322, :7204; unsupported moves).
    KLUTZ = 87
    # E1 trial batch, ruling G-63: Plus and Minus (BattleSystem_CalcMoveDamage,
    # battle_lib.c:6802-6809; damage.side_ability_count, damage._pre_random).
    PLUS = 88
    MINUS = 89
    # E1 batch F, ruling F-1: Heatproof (BattleSystem_CalcMoveDamage, battle_lib.c:6841-6844;
    # subscript_burn_damage.s:8-9; damage._pre_random, step._burn_check).
    HEATPROOF = 90
    # F-2: Flash Fire (BattleSystem_TriggerImmunityAbility, battle_lib.c:3515-3521; step._use_move
    # beside Dry Skin; the flag boost in damage._pre_random, :7074-7076).
    FLASH_FIRE = 91
    # F-3: Multitype (Battler_MonType, battle_lib.c:7629-7699): Arceus' types are its raw held
    # plate's (``damage.multitype_plate``); the failure tests of the moves it protects against
    # read the decomp id (``ABILITY_MULTITYPE``).
    MULTITYPE = 92
    # F-5: Hydration (BtlCmd_EndOfTurnWeatherEffect, battle_script.c:5626-5639; ``step._weather_effect``).
    HYDRATION = 93
    # F-6: Forecast (BattleSystem_TriggerFormChange, battle_lib.c:6313-6364; ``hooks.forecast``).
    FORECAST = 94


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
    AbilityId.VOLT_ABSORB: 10,
    AbilityId.WATER_ABSORB: 11,
    AbilityId.RAIN_DISH: 44,
    AbilityId.CLEAR_BODY: 29,
    AbilityId.HYPER_CUTTER: 52,
    AbilityId.AFTERMATH: 106,
    AbilityId.POISON_HEAL: 90,
    AbilityId.BAD_DREAMS: 123,
    AbilityId.ARENA_TRAP: 71,
    AbilityId.SHADOW_TAG: 23,
    AbilityId.MAGNET_PULL: 42,
    AbilityId.PRESSURE: 46,
    AbilityId.LEAF_GUARD: 102,
    AbilityId.SYNCHRONIZE: 28,
    AbilityId.WONDER_GUARD: 25,
    AbilityId.DOWNLOAD: 88,
    AbilityId.ICE_BODY: 115,
    AbilityId.WHITE_SMOKE: 73,
    AbilityId.VITAL_SPIRIT: 72,
    AbilityId.SPEED_BOOST: 3,
    AbilityId.IRON_FIST: 89,
    AbilityId.KLUTZ: 103,
    AbilityId.PLUS: 57,
    AbilityId.MINUS: 58,
    AbilityId.HEATPROOF: 85,
    AbilityId.FLASH_FIRE: 18,
    AbilityId.MULTITYPE: 121,
    AbilityId.HYDRATION: 93,
    AbilityId.FORECAST: 59,
}

# The abilities every sleep site tests together (ruling G-47): subscript_fall_asleep.s:7-8 and
# :23-24, subscript_yawn.s:5-6, subscript_rest.s:7-8, BattleSystem_RecoverStatusByAbility's
# shared case (battle_lib.c:4404-4405, :4471-4472) and the disobedience nap
# (battle_controller_player.c:2202-2203) name Insomnia and Vital Spirit side by side.
SLEEPLESS_ABILITIES = (AbilityId.INSOMNIA, AbilityId.VITAL_SPIRIT)


def sleepless(ability):
    """``ability`` (an ``AbilityId``, array or int) is Insomnia or Vital Spirit."""
    return (ability == AbilityId.INSOMNIA) | (ability == AbilityId.VITAL_SPIRIT)


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
    * ``FOCUS_BAND`` (``HOLD_EFFECT_MAYBE_ENDURE``, param ``FOCUS_BAND_PARAM``): a roll
      ``RandNext() % 100 < param`` at each damage that reaches ``UpdateHP`` (not while the
      defender is enduring, G-65) or ``CheckHoldOnWith1HP`` (bcp.c:3386-3389,
      battle_script.c:9109-9112;
      ``Outcome.focus_band`` / ``future_band``, ruling G-1).
    * ``FLAVOR_BERRY`` (``HOLD_EFFECT_HP_RESTORE_SPICY`` .. ``_SOUR``: Figy, Wiki, Mago, Aguav,
      Iapapa): ``item_param[i, 1]`` is its ``Flavor``; it heals ``Divide(maxHP, param)`` at
      ``curHP <= maxHP / 2`` and confuses a holder whose nature dislikes the flavor
      (battle_lib.c:4698-4771, :5121-5194; ``hooks.held_item_check``), Pluck at any HP below
      full (:5678-5751, ``hooks.pluck_berry``); ruling G-4.
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
    # Power Herb (HOLD_EFFECT_CHARGE_SKIP, generated/item_hold_effects.txt:100): a charge move's
    # first use hits at once and the herb is used up (subscript_item_skip_charge_turn.s,
    # effect_script_0039.s:7 and the other charge scripts, B10-12).
    POWER_HERB = 51
    # E1 Gardenia batch (G-1 .. G-6).
    FOCUS_BAND = 52
    FLAVOR_BERRY = 53
    # E1 Batch D (G-53 ..): the Fantina split's held items.
    # Leppa Berry (HOLD_EFFECT_PP_RESTORE): the first move slot at 0 PP regains ``param`` PP,
    # capped at its max PP (battle_lib.c:4636-4653, :5026-5043; Pluck :5609-5633), G-58.
    LEPPA_BERRY = 54
    # Jaboca / Rowap Berry (HOLD_EFFECT_RECOIL_PHYSICAL / _SPECIAL): a physical / special hit's
    # attacker loses ``Divide(maxHP, param)`` (TriggerHeldItemOnHit, battle_lib.c:5400-5419;
    # Jaboca also TriggerHeldItemOnPivotMove, :7368-7375), G-54.
    JABOCA_BERRY = 55
    ROWAP_BERRY = 56
    # Enigma Berry (HOLD_EFFECT_HP_RESTORE_SE): a super-effective hit's living holder heals
    # ``Divide(maxHP, param)`` (battle_lib.c:5421-5428), G-55.
    ENIGMA_BERRY = 57
    # Micle Berry (HOLD_EFFECT_PINCH_ACC_UP): at ``curHP <= maxHP / param`` sets the holder's
    # moveEffectsData.micleBerry, x120/100 on its next accuracy check (battle_lib.c:4897-4906,
    # :5110-5119; bcp.c:2967-2970; ``BattleState.micle``), G-59.
    MICLE_BERRY = 58
    # Grip Claw (HOLD_EFFECT_EXTEND_TRAPPING): its holder's Bind lasts 6 turns
    # (subscript_bind_start.s:10-12), G-57.
    GRIP_CLAW = 59
    # Metronome (HOLD_EFFECT_BOOST_REPEATED): damage x(10 + metronomeTurns)/10 for the same move
    # used again (battle_script.c:1350-1352; battle_lib.c:6160-6198), G-60.
    METRONOME_ITEM = 60
    # Iron Ball (HOLD_EFFECT_SPEED_DOWN_GROUNDED): Speed halved (battle_lib.c:1240-1246, read
    # by the raw held item), grounds its holder (:2534, :2602-2607, :2702, :2760, :3215, :5505;
    # subscript_hazards_check.s:7), G-56.
    IRON_BALL = 61
    # E1 Batch F, held items (F-7 ..). Hold effect 50 (HOLD_EFFECT_EVS_UP_SPEED_DOWN: Macho Brace;
    # Kaizo's Power items, vanilla's HOLD_EFFECT_LVLUP_*_EV_UP 117-122): no hook of its own, the
    # Speed halving and the EV doubling read the raw hold effect (``SPEED_HALVING_HOLD_EFFECTS``,
    # ``levelup``): an id so the item is no coverage refusal (F-7).
    MACHO_BRACE = 62
    # Kaizo's Berserk Gene (hold 36, a Liechi's effect on a non-berry): ``PINCH_BERRY``'s held
    # trigger and stat, but Pluck / Bug Bite leave it (PLUCK_EFFECT_NONE, no berry: F-8).
    BERSERK_GENE = 63
    # Lansat Berry (HOLD_EFFECT_PINCH_CRITRATE_UP): at ``curHP <= maxHP / param`` sets the
    # holder's Focus Energy (battle_lib.c:4838-4848; Pluck :5819-5825), F-9.
    LANSAT_BERRY = 64
    # Starf Berry (HOLD_EFFECT_PINCH_RANDOM_UP): at the same pinch raises a random stat below +6
    # of Attack, Defense, Speed, Sp. Atk, Sp. Def by 2 (battle_lib.c:4850-4872; Pluck :5798-5817,
    # ``Outcome.starf``), F-10.
    STARF_BERRY = 65
    # Kaizo's Lava Cookie / Old Gateau (hold 12, a Lum Berry's effect on a non-berry): Lum's cure
    # (``hooks._restore``), but Pluck / Bug Bite leave it (F-12).
    LAVA_COOKIE = 66


# subscript_bind_start.s:10-12: an attacker holding HOLD_EFFECT_EXTEND_TRAPPING (Grip Claw) sets
# the bind count to 6 instead of the Random 3, 3 draw (ruling G-57).
GRIP_CLAW_TURNS = 6

# sSpeedHalvingItemEffects (battle_lib.c:1160-1169): decomp hold effects (generated/
# item_hold_effects.txt line N+1 is id N) that halve Speed in BattleSystem_CompareBattlerSpeed
# (:1240-1246, :1302-1308), read through BattleSystem_GetItemData on the held item, so no Klutz
# or Embargo: Macho Brace (EVS_UP_SPEED_DOWN), Iron Ball (SPEED_DOWN_GROUNDED), the Power items
# (LVLUP_HP/ATK/DEF/SPEED/SPATK/SPDEF_EV_UP). Ruling G-56.
SPEED_HALVING_HOLD_EFFECTS = (50, 106, 122, 117, 118, 121, 119, 120)

# Starf Berry's draw (BattleSystem_TriggerHeldItem, battle_lib.c:4864-4866): ``RandNext() % 5``
# repeated until the stat (Attack, Defense, Speed, Sp. Atk, Sp. Def: ``BATTLE_STAT_ATTACK + i``,
# generated/battle_stats.txt:2-6) is below +6. By the project's ``% n`` convention uniform over
# the eligible stats: one value in [0, STARF_DRAWS) read modulo their count (1-5, lcm 60), F-10.
STARF_DRAWS = 60

# The three Choice items: ``BattleSystem_CheckInvalidMoves`` (battle_lib.c:2307-2315) and
# ``UpdateMoveBuffers`` (battle_controller_player.c:3923-3933) test exactly these hold
# effects.
CHOICE_ITEMS = (ItemId.CHOICE_BAND, ItemId.CHOICE_SPECS, ItemId.CHOICE_SCARF)

# Decomp species ids the item code of ``BattleSystem_CalcMoveDamage`` names
# (battle_lib.c:6730-6774; generated/species.txt line N is id N-1).
SPECIES_PIKACHU = 25
SPECIES_CASTFORM = 351  # BattleSystem_TriggerFormChange (battle_lib.c:6314), F-6
SPECIES_SHAYMIN = 492  # subscript_check_shaymin_form.s:6, F-16
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


class ItemParam(IntEnum):
    """Columns of ``tables.item_party_param[item]``: ``Item_Get`` (src/item.c:153-198) of each
    party param ``TrainerAI_ShouldUseItem`` (trainer_ai/trainer_ai.c:4056-4197) and
    ``BattleSystem_UseBagItem`` (battle/battle_system.c:435-758) read, in
    ``tools.rom.readers.items.PARTY_PARAM_FIELDS`` order: the ``ItemPartyParam`` field when
    the item's ``partyUse`` is TRUE, else its ``dummy`` byte for every one (E1 ruling G-43).
    ``HOLD_FRIENDSHIP_UP`` (engine column) is 1 when the item's own hold effect is
    ``HOLD_EFFECT_FRIENDSHIP_UP``, which ``UseBagItem``'s friendship gain reads (:746)."""

    HEAL_SLEEP = 0
    HEAL_POISON = 1
    HEAL_BURN = 2
    HEAL_FREEZE = 3
    HEAL_PARALYSIS = 4
    HEAL_CONFUSION = 5
    HEAL_ATTRACT = 6
    GUARD_SPEC = 7
    REVIVE = 8
    ATK_STAGES = 9
    DEF_STAGES = 10
    SPATK_STAGES = 11
    SPDEF_STAGES = 12
    SPEED_STAGES = 13
    ACC_STAGES = 14
    CRIT_STAGES = 15
    PP_RESTORE = 16
    PP_RESTORE_ALL = 17
    HP_RESTORE = 18
    GIVE_FRIENDSHIP_LOW = 19
    GIVE_FRIENDSHIP_MED = 20
    GIVE_FRIENDSHIP_HIGH = 21
    HP_RESTORED = 22
    PP_RESTORED = 23
    FRIENDSHIP_LOW = 24
    FRIENDSHIP_MED = 25
    FRIENDSHIP_HIGH = 26
    HOLD_FRIENDSHIP_UP = 27


class ItemAiCategory(IntEnum):
    """``AI_CONTEXT.usedItemType`` (``ITEM_AI_CATEGORY_*``), which picks the AI's item
    subscript (``BattleControllerPlayer_ItemCommand``, battle_controller_player.c:1905-1944);
    ``NONE``: ``TrainerAI_ShouldUseItem`` does not use the item."""

    NONE = 0
    FULL_RESTORE = 1
    RECOVER_HP = 2
    RECOVER_STATUS = 3
    STAT_BOOSTER = 4
    GUARD_SPEC = 5


# TrainerAI_ShouldUseItem tests this item by id before its params (trainer_ai.c:4099).
ITEM_FULL_RESTORE = 23
# ItemPartyParam.hpRestored codes UseBagItem reads as fractions of max HP
# (battle_system.c:676-699): all, half, a quarter (each at least 1).
HP_RESTORED_ALL, HP_RESTORED_HALF, HP_RESTORED_QUARTER = 0xFF, 0xFE, 0xFD

# Quick Claw's hold param (res/items/data/quick_claw.json ``effectParam`` 20, the only
# ``HOLD_EFFECT_SOMETIMES_PRIORITY`` item): ``BattleSystem_CompareBattlerSpeed`` activates it
# when ``speedRand % (100 / param) == 0`` (battle_lib.c:1278-1286, :1344-1352). The engine
# stores that test's result (``BattleState.qc_roll``, ``Outcome.quick_claw``), which is exact
# only while every holder has this param: ``game_constants`` asserts it for every game.
QUICK_CLAW_PARAM = 20
QUICK_CLAW_MODULUS = 100 // QUICK_CLAW_PARAM  # % 5

# Focus Band's hold param (res/items/data/focus_band.json ``effectParam`` 10, the only
# ``HOLD_EFFECT_MAYBE_ENDURE`` item): ``RandNext() % 100 < itemPower`` (bcp.c:3387,
# battle_script.c:9110). The engine stores that test's result (``Outcome.focus_band``,
# ``Outcome.future_band``), exact only while every holder has this param: ``game_constants``
# asserts it for every game (ruling G-1).
FOCUS_BAND_PARAM = 10


class Flavor(IntEnum):
    """``enum Flavor`` (include/constants/flavor.h): a flavor berry's ``msgTemp``
    (battle_lib.c:4701 etc.) and ``item_param[i, 1]`` of a ``FLAVOR_BERRY``."""

    SPICY = 0
    DRY = 1
    SWEET = 2
    BITTER = 3
    SOUR = 4


# The flavor each nature dislikes, -1 for none: the -1 entries of ``sNatureFlavorAffinities``
# (src/pokemon.c:61-237, read by ``Pokemon_GetFlavorAffinityOf``, :4123-4127, with the nature
# ``personality % 25``), indexed by the decomp's nature id (``BattleState.nature``).
# tests/engine/test_e1_gardenia.py reads the table back from the decomp.
NATURE_DISLIKED_FLAVOR = (
    -1,  # Hardy
    Flavor.SOUR,  # Lonely
    Flavor.SWEET,  # Brave
    Flavor.DRY,  # Adamant
    Flavor.BITTER,  # Naughty
    Flavor.SPICY,  # Bold
    -1,  # Docile
    Flavor.SWEET,  # Relaxed
    Flavor.DRY,  # Impish
    Flavor.BITTER,  # Lax
    Flavor.SPICY,  # Timid
    Flavor.SOUR,  # Hasty
    -1,  # Serious
    Flavor.DRY,  # Jolly
    Flavor.BITTER,  # Naive
    Flavor.SPICY,  # Modest
    Flavor.SOUR,  # Mild
    Flavor.SWEET,  # Quiet
    -1,  # Bashful
    Flavor.BITTER,  # Rash
    Flavor.SPICY,  # Calm
    Flavor.SOUR,  # Gentle
    Flavor.SWEET,  # Sassy
    Flavor.DRY,  # Careful
    -1,  # Quirky
)

# ``BattleState.confusion`` of a battler a flavor berry confused outside ``step`` (the
# battle start's and a faint replacement's switch-in item check, ``hooks.battle_start``,
# ``step.replace``): confused, its counter not drawn yet. The next ``step`` draws it first,
# ``Outcome.berry_confusion`` (``step.draw_pending_confusion``, ruling G-5). Above every
# drawn counter (``Random 3, 2``: 2-5), so every read of the bit sees a confused battler.
CONFUSION_PENDING = 6


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
    # E2 batch 10 family 7, the charge turn (B10-12..). CHARGE: Move_IsMultiTurn's charge-up
    # effects (battle_lib.c:2975-2992 without Bide: 39, 75, 145, 151, 155, 255, 256, 263, 272),
    # whose scripts charge on a first use (SYSCTL_FIRST_OF_MULTI_TURN and the direct
    # subscript_vanish_on_charge_turn.s, which locks the user) and hit while
    # VOLATILE_CONDITION_MOVE_LOCKED is set (effect_script_0039.s:5 and the others' line 5).
    CHARGE = 512
    # BATTLE_EFFECT_SKIP_CHARGE_TURN_IN_SUN (151, SolarBeam): effect_script_0151.s:5-6 hits at
    # once in sun unless CheckIgnoreWeather (Cloud Nine, Air Lock: battle_script.c:9195-9206).
    SUN_SKIP = 1024
    # BATTLE_EFFECT_CHARGE_TURN_DEF_UP (145, Skull Bash): effect_script_0145.s:8 the charge turn's
    # indirect TO_ATTACKER Defense +1 (with a Power Herb: subscript_power_herb_skull_bash.s:14-17).
    CHARGE_DEF_UP = 2048
    # The semi-invulnerable bit the charge sets on the user (moves.h:14-15, :26, :37):
    # MOVE_EFFECT_AIRBORNE (Fly 155, Bounce 263), _UNDERGROUND (Dig 256), _UNDERWATER (Dive 255),
    # _SHADOW_FORCE (272); effect_script_0155.s:7 and the others' line 7-8.
    SEMI_AIR = 4096
    SEMI_DIG = 8192
    SEMI_DIVE = 16384
    SEMI_SHADOW = 32768
    # SYSCTL_HIT_DURING_FLY / _DIG / _DIVE / _SHADOW_FORCE (system_control.h:6-8, :27) its script
    # sets: CheckMoveHitOverrides then lets the move hit a defender with that bit
    # (battle_controller_player.c:3034-3040). FLY: 146, 149, 152, 207; DIG: 126, 147; DIVE: 257,
    # 261; all four: 148, 168, 176.
    HIT_AIR = 65536
    HIT_DIG = 131072
    HIT_DIVE = 262144
    HIT_SHADOW = 524288
    # BTLVAR_POWER_MULTI 20 against a defender with the bit the move hits (effect_script_0146.s,
    # _0149.s: AIRBORNE; _0126.s, _0147.s: UNDERGROUND; _0257.s, _0261.s: UNDERWATER; line 8-9).
    DOUBLE_SEMI = 1048576
    # BTLVAR_CRITICAL_BOOSTS + 1 on the attack (effect_script_0039.s:17, _0075.s:16; E2 batch
    # 13, B13-2: Blaze Kick _0200.s:5, Poison Tail / Cross Poison _0209.s:5, whose effect is
    # their secondary status).
    CRIT_UP = 2097152
    # E2 batch 13 (B13-1): ``SetMultiHit 2, SYSCTL_MULTI_HIT_MOVE`` in a script whose effect is
    # another family's (Twineedle, effect_script_0077.s:6: effect 2's poison hit, twice):
    # ``effects.hit_count`` gives two hits, each running the family's effect with its own
    # secondary roll (bcp.c:3544-3554, the script reloaded per hit :3789). ``MULTI_HIT``
    # (Double Kick's effect_script_0044.s, a plain hit twice) keeps its count in its param.
    HIT_TWICE = 4194304


class SemiKind(IntEnum):
    """``BattleState.semi_invulnerable``: which MOVE_EFFECT_SEMI_INVULNERABLE bit
    (moves.h:14-15, :26, :37, :44) the battler's ``moveEffectsMask`` holds, 0 for none; a
    charge sets one at a time (B10-12)."""

    NONE = 0
    AIRBORNE = 1  # Fly, Bounce
    UNDERGROUND = 2  # Dig
    UNDERWATER = 3  # Dive
    SHADOW_FORCE = 4


# The ``MoveFlag`` a charge move sets on its user, and the one a move hits through, per kind.
SEMI_SET_FLAGS = {
    SemiKind.AIRBORNE: MoveFlag.SEMI_AIR,
    SemiKind.UNDERGROUND: MoveFlag.SEMI_DIG,
    SemiKind.UNDERWATER: MoveFlag.SEMI_DIVE,
    SemiKind.SHADOW_FORCE: MoveFlag.SEMI_SHADOW,
}
SEMI_HIT_FLAGS = {
    SemiKind.AIRBORNE: MoveFlag.HIT_AIR,
    SemiKind.UNDERGROUND: MoveFlag.HIT_DIG,
    SemiKind.UNDERWATER: MoveFlag.HIT_DIVE,
    SemiKind.SHADOW_FORCE: MoveFlag.HIT_SHADOW,
}


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
# An invoker call variant that assumes a faster foe's Gravity landed before the call (G-67):
# ``Outcome.call`` ``v + CALL_GRAVITY`` with ``v`` the value drawn from the law under Gravity
# (``order.gravity_state``). Above every Metronome rank (< METRONOME_DRAW) and every Sleep Talk
# value, and a multiple of SLEEP_TALK_VALUES: the call reads ``call % CALL_GRAVITY``
# (``order.invoked_move``), so ``step`` plays both encodings of ``v`` alike on the live set.
CALL_GRAVITY = SLEEP_TALK_VALUES * (METRONOME_DRAW // SLEEP_TALK_VALUES + 1)
# Fake Out (``generated/moves.txt`` line 253): ``subscript_flinch_mon.s:18`` shows Inner
# Focus's message, which names the ability, only for ``BTLVAR_CURRENT_MOVE == MOVE_FAKE_OUT``
# (or a direct flinch effect, which no supported move has).
MOVE_FAKE_OUT = 252
# MOVE_KNOCK_OFF (generated/moves.txt): the on-hit Sticky Barb transfer skips it
# (battle_lib.c:5391), B10-6.
MOVE_KNOCK_OFF = 282
# sCannotEncoreMoves (battle_lib.c:7212-7219): Move_CanBeEncored compares their battle effects
# (MOVE_DATA(move).effect, :7225) with the last move's: Transform, Mimic, Sketch, Mirror Move,
# Encore, Struggle (generated/moves.txt), B10-9.
CANNOT_ENCORE_MOVES = (144, 102, 166, 119, 227, 165)
# Move_IsInvoker (battle_lib.c:3032-3045): MOVE_NONE, Sleep Talk, Copycat, Assist, Me First,
# Mirror Move, Metronome (generated/moves.txt), B10-11.
INVOKER_MOVES = (0, 214, 383, 274, 382, 119, 118)
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
# Imprison (``generated/moves.txt`` line 287): BattleControllerPlayer_DecrementPP counts the
# other side's living Pressure holders for it by ``moveTemp``, whatever its range (bcp.c:2239-2240).
MOVE_IMPRISON = 286
# SolarBeam (generated/moves.txt line 77): BattleSystem_CalcMoveDamage halves it by move id under
# FIELD_CONDITION_SOLAR_DOWN (battle_lib.c:7058-7060), B10-13.
MOVE_SOLAR_BEAM = 76
PROTECT_CHAIN_MOVES = (MOVE_PROTECT, MOVE_DETECT, MOVE_ENDURE)
# ``Move_CanBeMimicked`` (battle_lib.c:7182-7192): the entries of ``sCannotMetronomeMoves`` before
# ``FORBIDDEN_BY_MIMIC_DELIM`` (:7152-7158): Metronome, Struggle, Sketch, Mimic, Chatter
# (generated/moves.txt lines 119, 166, 167, 103, 449), G-68.
MOVE_MIMIC = 102
# MOVE_WILL_O_WISP (generated/moves.txt line 262): Flash Fire absorbs it though it has no power
# (BattleSystem_TriggerImmunityAbility, battle_lib.c:3519), F-2.
MOVE_WILL_O_WISP = 261
MIMIC_FORBIDDEN = (118, 165, 166, MOVE_MIMIC, 448)
# Chatter's confusion (battle_script.c:8397-8436): the attacker must be a Chatot that is not
# Transformed and the defender have HP; ``effectChance`` is 0 / 10 / 30 for the Chatot cry parameter
# 0 / 1 / 2 (:8413-8426, sound_chatot.c:193-213) and the confusion comes on ``RandNext() % 100 <=
# chance`` (:8428-8430), G-70. Species ids: generated/species.txt line 442.
SPECIES_CHATOT = 441
# SPECIES_ARCEUS (generated/species.txt line 494): Battler_MonType's Multitype case
# (battle_lib.c:7629-7630), F-3.
SPECIES_ARCEUS = 493
MOVE_CHATTER = 448
CHATTER_CHANCES = (0, 10, 30)
# ``BtlCmd_TryMimic`` caps the copy's PP at 5 (battle_script.c:4508-4513), G-68.
MIMIC_PP_CAP = 5
# sMovesAffectedByGravity (battle_lib.c:3385-3392; generated/moves.txt lines 20, 341, 27, 137, 151,
# 394): Fly, Bounce, Jump Kick, Hi Jump Kick, Splash, Magnet Rise. ``Move_FailsInHighGravity``
# (:3394-3407) bans them at selection (CHECK_INVALID_GRAVITY, :2292), at the status check
# (CHECK_STATUS_STATE_GRAVITY, bcp.c:2586-2595) and for Metronome, Assist and Copycat
# (``Move_CanBeMetronomed``, :7198-7201), G-67.
GRAVITY_MOVES = (19, 340, 26, 136, 150, 393)

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
# an ``AbilityId``, for the abilities read by decomp id (Air Lock, Magic Guard) and the
# lock they share with Cloud Nine: Cloud Nine / Air Lock suppress weather (``NO_CLOUD_NINE``,
# include/battle/common.h:63-64); the end-of-turn chip (``BtlCmd_EndOfTurnWeatherEffect``,
# battle_script.c:5561-5647, and ``subscript_weather_continues.s``) skips Magic Guard. Sand
# Veil, Snow Cloak and Ice Body (since E1 Batch C, G-45) are ``AbilityId`` members.
WEATHER_ABILITY_IDS = {
    "Cloud Nine": 13,
    "Air Lock": 76,
    "Magic Guard": 98,
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

# E2 batch 12's decomp abilities: Captivate's Oblivious (effect_script_0265.s:5,
# CheckIgnorableAbility, B12-8); Rest's Insomnia, Vital Spirit and Soundproof
# (subscript_rest.s:7-10) and Heal Bell's Soundproof (battle_script.c:5038, battle_io_command.c:
# 651), B12-12.
STAT_MOVE_ABILITY_IDS = {
    "Oblivious": 12,
    "Insomnia": 15,
    "Soundproof": 43,
    "Vital Spirit": 72,
}
# MOVE_HEAL_BELL (BtlCmd_TryPartyStatusRefresh's and BtlIOCmd_RefreshPartyStatus's compare,
# battle_script.c:5035, battle_io_command.c:651), B12-12.
MOVE_HEAL_BELL = 215
# subscript_rest.s:32: MON_CONDITION_SLEEP_0 | MON_CONDITION_SLEEP_1, a sleep count of 3.
REST_TURNS = 3
# BtlCmd_TryWish (battle_script.c:6381): wishTurns = 2.
WISH_TURNS = 2
# subscript_recover_psn_prz_brn.s:5: MON_CONDITION_FACADE_BOOST, what Refresh cures (B12-16).
REFRESH_STATUSES = (Status.POISON, Status.TOXIC, Status.BURN, Status.PARALYSIS)
# BattleMon.gender's GENDER_NONE (include/constants/pokemon.h; ``BattleState.gender``).
GENDER_NONE = 2
# BtlCmd_BoostRandomStatBy2 (battle_script.c:8943-8947): BATTLE_STAT_ATTACK .. BATTLE_STAT_EVASION
# in the decomp's order (Speed before Sp. Atk and Sp. Def), as ``Stat`` (B12-9).
ACUPRESSURE_STATS = (
    Stat.ATK,
    Stat.DEF,
    Stat.SPE,
    Stat.SPA,
    Stat.SPD,
    Stat.ACC,
    Stat.EVA,
)
# Starf Berry's stats (battle_lib.c:4856-4866: BATTLE_STAT_ATTACK + i, i < 5), F-10.
STARF_STATS = ACUPRESSURE_STATS[:5]

ITEM_ABILITY_IDS = {
    "Immunity": 17,
    "Water Veil": 41,
    "Leaf Guard": 102,
    # subscript_knock_off.s / BtlCmd_TryKnockOff (battle_script.c:6614), B10-6.
    "Sticky Hold": 60,
    # subscript_confuse.s:6 (CheckAbility, the holder's own): a disliked flavor berry's
    # confusion (ruling G-5).
    "Own Tempo": 20,
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
