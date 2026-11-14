# Kaizo mechanics backlog

Generated from the opponent-side coverage check (`check_opponent`) after E1 batch 2a
(reads by path order, per-trainer AI flag sets), plus the AI's per-move refusals of the
trainers' moves (`policy.opponent_move_refusals`: AI move effects whose Basic/Expert
script paths are not transcribed, keyed by battle effect), which `check_opponent` skips.
It counts only what the trainers' teams, bags, AI and battle conditions need.
Your own Pokemon's moves, abilities and items are not counted.
Moves are grouped by the game's move-effect number and held items by hold-effect number,
since one implementation covers the whole group.

## By split

| Split | Fights | Runnable now | Singles | Distinct missing groups |
|---|---|---|---|---|
| Roark Split | 23 | 1 | 23 | 81 |
| Gardenia Split | 43 | 0 | 20 | 177 |
| Fantina Split | 41 | 0 | 37 | 218 |
| Maylene Split | 34 | 0 | 27 | 211 |
| Wake Split | 37 | 0 | 25 | 177 |
| Byron Split | 49 | 0 | 36 | 202 |
| Candice Split | 47 | 0 | 23 | 186 |
| Galactic Split | 102 | 0 | 40 | 301 |
| Volkner Split | 25 | 0 | 9 | 147 |
| Elite Four Split | 19 | 0 | 18 | 175 |

## Roark split

23 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 15 | AI move effect #103 PRIORITY_1 not transcribed (no Basic or Expert label): Aqua Jet, Bullet Punch, Ice Shard, Mach Punch, Quick Attack, Shadow Sneak, +1 more |
| 10 | AI move effect #17 BYPASS_ACCURACY not transcribed (Expert_BypassAccuracyMove): Aerial Ace, Aura Sphere, Aurora Beam, Dragon Pulse, Faint Attack, Magical Leaf, +5 more |
| 10 | AI move effect #97 USE_RANDOM_LEARNED_MOVE_SLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_SleepTalk): Sleep Talk |
| 10 | move effect #97: Sleep Talk |
| 9 | AI move effect #43 HIGH_CRITICAL not transcribed (Expert_HighCritical): Air Cutter, Aqua Cutter, Crabhammer, Cross Chop, Dragon Claw, Drill Peck, +11 more |
| 9 | AI move effect #67 STATUS_PARALYZE not transcribed (Basic_CheckCannotParalyze / Expert_StatusParalyze): Glare, Stun Spore, Thunder Wave |
| 7 | AI move effect #111 PROTECT not transcribed (Expert_Protect): Detect, Protect |
| 7 | AI move effect #224 EAT_BERRY not transcribed (Expert_Pluck): Bug Bite, Pluck |
| 7 | AI move effect #73 LOWER_ACCURACY_HIT not transcribed (no Basic or Expert label): Mirror Shot, Mud Bomb, Mud-Slap, Muddy Water, Octazooka |
| 6 | AI move effect #158 ALWAYS_FLINCH_FIRST_TURN_ONLY not transcribed (Basic_CheckFirstTurnInBattle / Expert_FakeOut): Fake Out |
| 6 | AI move effect #197 SECRET_POWER not transcribed (no Basic or Expert label): Secret Power |
| 6 | move effect #197: Secret Power |
| 5 | AI move effect #120 INFATUATE not transcribed (Basic_CheckCannotAttract): Attract |
| 4 | AI move effect #1 STATUS_SLEEP not transcribed (Basic_CheckCannotSleep / Expert_StatusSleep): Dark Void, GrassWhistle, Hypnosis, Lovely Kiss, Sing, Sleep Powder, +2 more |
| 4 | AI move effect #29 MULTI_HIT not transcribed (no Basic or Expert label): Bone Rush, Bullet Seed, Fury Swipes, Icicle Spear, Pin Missile, Rock Blast, +1 more |
| 4 | AI move effect #92 DAMAGE_WHILE_ASLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_Nightmare): Snore |
| 4 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it |
| 4 | move effect #92: Snore |
| 3 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 3 | AI move effect #171 DOUBLE_POWER_AND_CURE_PARALYSIS not transcribed (Expert_SmellingSalts): SmellingSalt |
| 3 | AI move effect #228 SWITCH_HIT not transcribed (Expert_UTurn): U-turn |
| 3 | AI move effect #49 STATUS_CONFUSE not transcribed (Basic_CheckCannotConfuse / Expert_StatusConfuse): Confuse Ray, Supersonic, Sweet Kiss, Teeter Dance |
| 3 | AI move effect #70 LOWER_SPEED_HIT not transcribed (Expert_SpeedDownOnHit): Bubble, BubbleBeam, Bulldoze, Icy Wind, Mud Shot, Rock Tomb |
| 3 | AI move-id case of TrainerAI_CalcDamage: Frustration, Return |
| 3 | move effect #171: SmellingSalt |
| 3 | move effect #228: U-turn |
| 2 | AI move effect #125 THAW_AND_BURN_HIT not transcribed (no Basic or Expert label): Flame Wheel, Sacred Fire, Scald |
| 2 | AI move effect #196 INCREASE_POWER_WITH_WEIGHT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Grass Knot, Low Kick |
| 2 | AI move effect #214 HEAL_HALF_REMOVE_FLYING_TYPE not transcribed (Basic_CheckCanRecoverHP / Expert_Recovery): Roost |
| 2 | AI move effect #227 METAL_BURST not transcribed (Basic_CheckMetalBurst / Expert_MetalBurst): Bide, Metal Burst |
| 2 | AI move effect #44 HIT_TWICE not transcribed (no Basic or Expert label): Bonemerang, Double Hit, Double Kick, DoubleSlap |
| 2 | AI move effect #58 ATK_DOWN_2 not transcribed (Basic_CheckLowStatStage_Attack / Expert_StatusAttackDown): Charm, FeatherDance |
| 2 | AI move effect #65 SET_REFLECT not transcribed (Basic_CheckAlreadyUnderReflect / Expert_Reflect): Reflect |
| 2 | AI move effect #83 CALL_RANDOM_MOVE not transcribed (no Basic or Expert label): Metronome |
| 2 | move effect #196: Grass Knot, Low Kick |
| 2 | move effect #214: Roost |
| 2 | move effect #227: Bide, Metal Burst |
| 2 | move effect #83: Metronome |
| 1 | AI: AI_FLAG_CHECK_HP |
| 1 | AI move effect #112 SET_SPIKES not transcribed (Basic_CheckSpikes / Expert_Spikes): Spikes |
| 1 | AI move effect #118 ATK_UP_2_STATUS_CONFUSION not transcribed (Basic_CheckCannotConfuse / Expert_Swagger): Swagger |
| 1 | AI move effect #122 RANDOM_POWER_MAYBE_HEAL not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Present |
| 1 | AI move effect #135 RANDOM_POWER_BASED_ON_IVS not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Hidden Power |
| 1 | AI move effect #136 WEATHER_RAIN not transcribed (Basic_CheckRainDance / Expert_RainDance): Rain Dance |
| 1 | AI move effect #139 RAISE_ATTACK_HIT not transcribed (no Basic or Expert label): Metal Claw, Meteor Mash |
| 1 | AI move effect #144 MIRROR_COAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_MirrorCoat): Mirror Coat |
| 1 | AI move effect #147 DOUBLE_DAMAGE_DIG not transcribed (no Basic or Expert label): Earthquake |
| 1 | AI move effect #148 HIT_IN_3_TURNS not transcribed (Basic_CheckFutureSight): Doom Desire, Future Sight |
| 1 | AI move effect #149 DOUBLE_DAMAGE_FLY_OR_BOUNCE not transcribed (no Basic or Expert label): Gust, Sky Uppercut |
| 1 | AI move effect #159 UPROAR not transcribed (no Basic or Expert label): Uproar |
| 1 | AI move effect #164 WEATHER_HAIL not transcribed (Basic_CheckHail / Expert_Hail): Hail |
| 1 | AI move effect #169 DOUBLE_POWER_WHEN_STATUSED not transcribed (Expert_Facade): Facade |
| 1 | AI move effect #20 SPEED_DOWN not transcribed (Basic_CheckLowStatStage_Speed / Expert_StatusSpeedDown): String Shot |
| 1 | AI move effect #202 BADLY_POISON_HIT not transcribed (no Basic or Expert label): Poison Fang |
| 1 | AI move effect #212 ATK_SPD_UP not transcribed (Basic_CheckDragonDance / Expert_DragonDance): Dragon Dance |
| 1 | AI move effect #221 DOUBLE_POWER_WHEN_BELOW_HALF not transcribed (Expert_Brine): Brine, Wring Out |
| 1 | AI move effect #225 DOUBLE_SPEED_3_TURNS not transcribed (Basic_CheckTailwind / Expert_Tailwind): Tailwind |
| 1 | AI move effect #23 ACC_DOWN not transcribed (Basic_CheckLowStatStage_Accuracy / Expert_StatusAccuracyDown): Flash, Sand-Attack, SmokeScreen |
| 1 | AI move effect #242 USE_LAST_USED_MOVE not transcribed (Basic_CheckCopycat / Expert_Copycat): Copycat |
| 1 | AI move effect #38 ONE_HIT_KO not transcribed (Basic_CheckOHKOWouldFail / Expert_OHKOMove): Fissure, Guillotine, Horn Drill |
| 1 | AI move effect #51 DEF_UP_2 not transcribed (Basic_CheckHighStatStage_Defense / Expert_StatusDefenseUp): Acid Armor, Barrier, Iron Defense |
| 1 | AI move effect #84 STATUS_LEECH_SEED not transcribed (Basic_CheckCannotLeechSeed / Expert_ToxicLeechSeed): Leech Seed |
| 1 | AI move effect #87 LEVEL_DAMAGE_FLAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Night Shade, Seismic Toss |
| 1 | AI move effect #98 KO_MON_THAT_DEFEATED_USER not transcribed (Expert_DestinyBond): Destiny Bond |
| 1 | AI move effect #99 INCREASE_POWER_WITH_LESS_HP not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Reversal): Flail, Reversal |
| 1 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) |
| 1 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) |
| 1 | move effect #112: Spikes |
| 1 | move effect #118: Swagger |
| 1 | move effect #144: Mirror Coat |
| 1 | move effect #147: Earthquake |
| 1 | move effect #148: Doom Desire, Future Sight |
| 1 | move effect #159: Uproar |
| 1 | move effect #169: Facade |
| 1 | move effect #202: Poison Fang |
| 1 | move effect #212: Dragon Dance |
| 1 | move effect #221: Brine, Wring Out |
| 1 | move effect #225: Tailwind |
| 1 | move effect #242: Copycat |
| 1 | move effect #38: Fissure, Guillotine, Horn Drill |
| 1 | move effect #98: Destiny Bond |

Groups by kind: AI 55, move effect 23, battle format 2, battle terrain 1

## Roark and Gardenia splits (up to the second gym)

66 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 37 | AI move effect #103 PRIORITY_1 not transcribed (no Basic or Expert label): Aqua Jet, Bullet Punch, Ice Shard, Mach Punch, Quick Attack, Shadow Sneak, +1 more |
| 30 | AI move effect #43 HIGH_CRITICAL not transcribed (Expert_HighCritical): Air Cutter, Aqua Cutter, Crabhammer, Cross Chop, Dragon Claw, Drill Peck, +11 more |
| 29 | AI move effect #1 STATUS_SLEEP not transcribed (Basic_CheckCannotSleep / Expert_StatusSleep): Dark Void, GrassWhistle, Hypnosis, Lovely Kiss, Sing, Sleep Powder, +2 more |
| 26 | AI move effect #17 BYPASS_ACCURACY not transcribed (Expert_BypassAccuracyMove): Aerial Ace, Aura Sphere, Aurora Beam, Dragon Pulse, Faint Attack, Magical Leaf, +5 more |
| 23 | AI move effect #97 USE_RANDOM_LEARNED_MOVE_SLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_SleepTalk): Sleep Talk |
| 23 | move effect #97: Sleep Talk |
| 18 | battle format: a multi battle |
| 15 | AI move effect #224 EAT_BERRY not transcribed (Expert_Pluck): Bug Bite, Pluck |
| 14 | AI: AI_FLAG_TAG_STRATEGY |
| 14 | AI move effect #67 STATUS_PARALYZE not transcribed (Basic_CheckCannotParalyze / Expert_StatusParalyze): Glare, Stun Spore, Thunder Wave |
| 13 | AI move effect #49 STATUS_CONFUSE not transcribed (Basic_CheckCannotConfuse / Expert_StatusConfuse): Confuse Ray, Supersonic, Sweet Kiss, Teeter Dance |
| 12 | AI move effect #111 PROTECT not transcribed (Expert_Protect): Detect, Protect |
| 12 | AI move effect #73 LOWER_ACCURACY_HIT not transcribed (no Basic or Expert label): Mirror Shot, Mud Bomb, Mud-Slap, Muddy Water, Octazooka |
| 11 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 11 | AI move effect #158 ALWAYS_FLINCH_FIRST_TURN_ONLY not transcribed (Basic_CheckFirstTurnInBattle / Expert_FakeOut): Fake Out |
| 11 | AI move effect #29 MULTI_HIT not transcribed (no Basic or Expert label): Bone Rush, Bullet Seed, Fury Swipes, Icicle Spear, Pin Missile, Rock Blast, +1 more |
| 10 | AI move effect #70 LOWER_SPEED_HIT not transcribed (Expert_SpeedDownOnHit): Bubble, BubbleBeam, Bulldoze, Icy Wind, Mud Shot, Rock Tomb |
| 9 | AI: AI_FLAG_HARRASSMENT |
| 9 | AI move effect #228 SWITCH_HIT not transcribed (Expert_UTurn): U-turn |
| 9 | ability: Magic Guard |
| 9 | move effect #228: U-turn |
| 8 | AI move effect #125 THAW_AND_BURN_HIT not transcribed (no Basic or Expert label): Flame Wheel, Sacred Fire, Scald |
| 8 | AI move effect #197 SECRET_POWER not transcribed (no Basic or Expert label): Secret Power |
| 8 | AI move effect #5 FREEZE_HIT not transcribed (no Basic or Expert label): Ice Beam, Ice Punch |
| 8 | ability: Chlorophyll |
| 8 | move effect #197: Secret Power |
| 7 | AI: AI_FLAG_CHECK_HP |
| 7 | AI move effect #198 RECOIL_THIRD not transcribed (Expert_RecoilMove): Double-Edge, Wood Hammer |
| 7 | AI move effect #221 DOUBLE_POWER_WHEN_BELOW_HALF not transcribed (Expert_Brine): Brine, Wring Out |
| 7 | AI move effect #65 SET_REFLECT not transcribed (Basic_CheckAlreadyUnderReflect / Expert_Reflect): Reflect |
| 7 | move effect #198: Double-Edge, Wood Hammer |
| 7 | move effect #221: Brine, Wring Out |
| 6 | AI move effect #120 INFATUATE not transcribed (Basic_CheckCannotAttract): Attract |
| 6 | AI move effect #147 DOUBLE_DAMAGE_DIG not transcribed (no Basic or Expert label): Earthquake |
| 6 | AI move effect #202 BADLY_POISON_HIT not transcribed (no Basic or Expert label): Poison Fang |
| 6 | AI move effect #44 HIT_TWICE not transcribed (no Basic or Expert label): Bonemerang, Double Hit, Double Kick, DoubleSlap |
| 6 | move effect #147: Earthquake |
| 6 | move effect #202: Poison Fang |
| 5 | AI move effect #169 DOUBLE_POWER_WHEN_STATUSED not transcribed (Expert_Facade): Facade |
| 5 | AI move effect #196 INCREASE_POWER_WITH_WEIGHT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Grass Knot, Low Kick |
| 5 | AI move effect #212 ATK_SPD_UP not transcribed (Basic_CheckDragonDance / Expert_DragonDance): Dragon Dance |
| 5 | AI move effect #249 TOXIC_SPIKES not transcribed (Basic_CheckToxicSpikes / Expert_ToxicSpikes): Toxic Spikes |
| 5 | AI move effect #87 LEVEL_DAMAGE_FLAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Night Shade, Seismic Toss |
| 5 | ability: Aftermath |
| 5 | battle format: a double battle |
| 5 | move effect #169: Facade |
| 5 | move effect #196: Grass Knot, Low Kick |
| 5 | move effect #212: Dragon Dance |
| 5 | move effect #249: Toxic Spikes |
| 4 | AI move effect #136 WEATHER_RAIN not transcribed (Basic_CheckRainDance / Expert_RainDance): Rain Dance |
| 4 | AI move effect #152 THUNDER not transcribed (no Basic or Expert label): Thunder |
| 4 | AI move effect #172 MAKE_GLOBAL_TARGET not transcribed (no Basic or Expert label): Follow Me |
| 4 | AI move effect #260 BLIZZARD not transcribed (Expert_Blizzard): Blizzard, Ice Ball, Sheer Cold |
| 4 | AI move effect #273 FLINCH_BURN_HIT not transcribed (no Basic or Expert label): Fire Fang |
| 4 | AI move effect #41 40_DAMAGE_FLAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Dragon Rage |
| 4 | AI move effect #84 STATUS_LEECH_SEED not transcribed (Basic_CheckCannotLeechSeed / Expert_ToxicLeechSeed): Leech Seed |
| 4 | AI move effect #92 DAMAGE_WHILE_ASLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_Nightmare): Snore |
| 4 | AI move effect #98 KO_MON_THAT_DEFEATED_USER not transcribed (Expert_DestinyBond): Destiny Bond |
| 4 | AI move-id case of TrainerAI_CalcDamage: Frustration, Return |
| 4 | ability: Filter |
| 4 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it |
| 4 | move effect #152: Thunder |
| 4 | move effect #172: Follow Me |
| 4 | move effect #260: Blizzard, Ice Ball, Sheer Cold |
| 4 | move effect #273: Fire Fang |
| 4 | move effect #41: Dragon Rage |
| 4 | move effect #92: Snore |
| 4 | move effect #98: Destiny Bond |
| 3 | AI: AI_FLAG_BATON_PASS |
| 3 | AI move effect #150 FLINCH_MINIMIZE_DOUBLE_HIT not transcribed (no Basic or Expert label): Stomp |
| 3 | AI move effect #171 DOUBLE_POWER_AND_CURE_PARALYSIS not transcribed (Expert_SmellingSalts): SmellingSalt |
| 3 | AI move effect #214 HEAL_HALF_REMOVE_FLYING_TYPE not transcribed (Basic_CheckCanRecoverHP / Expert_Recovery): Roost |
| 3 | AI move effect #225 DOUBLE_SPEED_3_TURNS not transcribed (Basic_CheckTailwind / Expert_Tailwind): Tailwind |
| 3 | AI move effect #276 RAISE_SP_ATK_HIT not transcribed (no Basic or Expert label): Charge Beam |
| 3 | AI move effect #50 ATK_UP_2 not transcribed (Basic_CheckHighStatStage_Attack / Expert_StatusAttackUp): Swords Dance |
| 3 | AI move effect #58 ATK_DOWN_2 not transcribed (Basic_CheckLowStatStage_Attack / Expert_StatusAttackDown): Charm, FeatherDance |
| 3 | AI move effect #83 CALL_RANDOM_MOVE not transcribed (no Basic or Expert label): Metronome |
| 3 | ability: Clear Body |
| 3 | form: trainer Pokemon in a data form: this game's party builder applies the form before computing its stats, which the engine's trainer party build does not model |
| 3 | held item effect #65: Focus Band |
| 3 | move effect #150: Stomp |
| 3 | move effect #171: SmellingSalt |
| 3 | move effect #214: Roost |
| 3 | move effect #225: Tailwind |
| 3 | move effect #83: Metronome |
| 2 | AI move effect #106 PREVENT_ESCAPE not transcribed (Basic_CheckMeanLook / Expert_BindingMove): Block, Mean Look, Spider Web |
| 2 | AI move effect #112 SET_SPIKES not transcribed (Basic_CheckSpikes / Expert_Spikes): Spikes |
| 2 | AI move effect #138 RAISE_DEF_HIT not transcribed (no Basic or Expert label): Steel Wing |
| 2 | AI move effect #142 MAX_ATK_LOSE_HALF_MAX_HP not transcribed (Basic_CheckBellyDrum / Expert_BellyDrum): Belly Drum |
| 2 | AI move effect #144 MIRROR_COAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_MirrorCoat): Mirror Coat |
| 2 | AI move effect #149 DOUBLE_DAMAGE_FLY_OR_BOUNCE not transcribed (no Basic or Expert label): Gust, Sky Uppercut |
| 2 | AI move effect #167 STATUS_BURN not transcribed (Basic_CheckCannotBurn): Will-O-Wisp |
| 2 | AI move effect #182 LOWER_OWN_ATK_AND_DEF not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Superpower): Brave Bird, Close Combat, Take Down |
| 2 | AI move effect #188 REMOVE_HELD_ITEM not transcribed (Basic_CheckCanRemoveItem / Expert_KnockOff): Knock Off |
| 2 | AI move effect #192 MAKE_SHARED_MOVES_UNUSEABLE not transcribed (Basic_CheckCanImprison / Expert_Imprison): Imprison |
| 2 | AI move effect #227 METAL_BURST not transcribed (Basic_CheckMetalBurst / Expert_MetalBurst): Bide, Metal Burst |
| 2 | AI move effect #257 DOUBLE_DAMAGE_DIVE not transcribed (no Basic or Expert label): Surf |
| 2 | AI move effect #274 FLINCH_FREEZE_HIT not transcribed (no Basic or Expert label): Avalanche, Ice Fang |
| 2 | AI move effect #36 TRI_ATTACK not transcribed (no Basic or Expert label): Aeroblast, Tri Attack |
| 2 | AI move effect #45 CRASH_ON_MISS not transcribed (no Basic or Expert label): Hi Jump Kick, Jump Kick |
| 2 | AI move effect #51 DEF_UP_2 not transcribed (Basic_CheckHighStatStage_Defense / Expert_StatusDefenseUp): Acid Armor, Barrier, Iron Defense |
| 2 | AI move effect #89 COUNTER not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Counter): Counter |
| 2 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 2 | ability: Arena Trap |
| 2 | ability: Bad Dreams |
| 2 | ability: Pure Power |
| 2 | ability: Rain Dish |
| 2 | ability: Sniper |
| 2 | ability: Swarm |
| 2 | ability: Unburden |
| 2 | ability: Volt Absorb |
| 2 | move effect #106: Block, Mean Look, Spider Web |
| 2 | move effect #112: Spikes |
| 2 | move effect #142: Belly Drum |
| 2 | move effect #144: Mirror Coat |
| 2 | move effect #182: Brave Bird, Close Combat, Take Down |
| 2 | move effect #188: Knock Off |
| 2 | move effect #192: Imprison |
| 2 | move effect #227: Bide, Metal Burst |
| 2 | move effect #257: Surf |
| 2 | move effect #274: Avalanche, Ice Fang |
| 2 | move effect #45: Hi Jump Kick, Jump Kick |
| 2 | move effect #89: Counter |
| 1 | AI move effect #10 ATK_UP not transcribed (Basic_CheckHighStatStage_Attack / Expert_StatusAttackUp): Howl, Meditate, Sharpen |
| 1 | AI move effect #108 EVA_UP_2_MINIMIZE not transcribed (Basic_CheckHighStatStage_Evasion / Expert_StatusEvasionUp): Minimize |
| 1 | AI move effect #118 ATK_UP_2_STATUS_CONFUSION not transcribed (Basic_CheckCannotConfuse / Expert_Swagger): Swagger |
| 1 | AI move effect #122 RANDOM_POWER_MAYBE_HEAL not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Present |
| 1 | AI move effect #124 PREVENT_STATUS not transcribed (Basic_CheckAlreadyUnderSafeguard): Safeguard |
| 1 | AI move effect #135 RANDOM_POWER_BASED_ON_IVS not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Hidden Power |
| 1 | AI move effect #137 WEATHER_SUN not transcribed (Basic_CheckSunnyDay / Expert_SunnyDay): Sunny Day |
| 1 | AI move effect #139 RAISE_ATTACK_HIT not transcribed (no Basic or Expert label): Metal Claw, Meteor Mash |
| 1 | AI move effect #148 HIT_IN_3_TURNS not transcribed (Basic_CheckFutureSight): Doom Desire, Future Sight |
| 1 | AI move effect #151 SKIP_CHARGE_TURN_IN_SUN not transcribed (Expert_ChargeTurnNoInvuln / Expert_Thunder): SolarBeam |
| 1 | AI move effect #159 UPROAR not transcribed (no Basic or Expert label): Uproar |
| 1 | AI move effect #164 WEATHER_HAIL not transcribed (Basic_CheckHail / Expert_Hail): Hail |
| 1 | AI move effect #170 HIT_LAST_WHIFF_IF_HIT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_FocusPunch): Focus Punch |
| 1 | AI move effect #176 BOOST_ALLY_POWER_BY_50_PERCENT not transcribed (Basic_CheckHelpingHand): Helping Hand |
| 1 | AI move effect #180 USE_RANDOM_ALLY_MOVE not transcribed (no Basic or Expert label): Assist |
| 1 | AI move effect #189 SET_HP_EQUAL_TO_USER not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Endeavor): Endeavor |
| 1 | AI move effect #20 SPEED_DOWN not transcribed (Basic_CheckLowStatStage_Speed / Expert_StatusSpeedDown): String Shot |
| 1 | AI move effect #208 ATK_DEF_UP not transcribed (Basic_CheckBulkUp / Expert_StatusDefenseUp): Bulk Up |
| 1 | AI move effect #23 ACC_DOWN not transcribed (Basic_CheckLowStatStage_Accuracy / Expert_StatusAccuracyDown): Flash, Sand-Attack, SmokeScreen |
| 1 | AI move effect #230 DOUBLE_POWER_IF_MOVING_SECOND not transcribed (Expert_Payback): Payback, Revenge |
| 1 | AI move effect #242 USE_LAST_USED_MOVE not transcribed (Basic_CheckCopycat / Expert_Copycat): Copycat |
| 1 | AI move effect #262 RECOIL_PARALYZE_HIT not transcribed (Expert_RecoilMove): Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 1 | AI move effect #271 LOWER_SP_DEF_2_HIT not transcribed (no Basic or Expert label): Seed Flare |
| 1 | AI move effect #28 FORCE_SWITCH not transcribed (Basic_CheckCanForceSwitch / Expert_ForceSwitch): Roar, Whirlwind |
| 1 | AI move effect #38 ONE_HIT_KO not transcribed (Basic_CheckOHKOWouldFail / Expert_OHKOMove): Fissure, Guillotine, Horn Drill |
| 1 | AI move effect #48 RECOIL_QUARTER not transcribed (Expert_RecoilMove): Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower |
| 1 | AI move effect #79 SET_SUBSTITUTE not transcribed (Basic_CheckCannotSubstitute / Expert_Substitute): Substitute |
| 1 | AI move effect #8 RECOVER_DAMAGE_SLEEP not transcribed (Basic_CheckDreamEater / Expert_DreamEater): Dream Eater |
| 1 | AI move effect #90 ENCORE not transcribed (Basic_CheckCannotEncore / Expert_Encore): Encore |
| 1 | AI move effect #91 AVERAGE_HP not transcribed (Expert_PainSplit): Pain Split |
| 1 | AI move effect #99 INCREASE_POWER_WITH_LESS_HP not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Reversal): Flail, Reversal |
| 1 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 1 | ability: Download |
| 1 | ability: Flower Gift |
| 1 | ability: Huge Power |
| 1 | ability: Hyper Cutter |
| 1 | ability: Leaf Guard |
| 1 | ability: Magnet Pull |
| 1 | ability: Poison Heal |
| 1 | ability: Pressure |
| 1 | ability: Shadow Tag |
| 1 | ability: Synchronize |
| 1 | ability: Water Absorb |
| 1 | ability: Wonder Guard |
| 1 | battle format: gauntlet Eterna Gym gym trainers (position 1; rules not recorded) |
| 1 | battle format: gauntlet Eterna Gym gym trainers (position 2; rules not recorded) |
| 1 | battle format: gauntlet Eterna Gym gym trainers (position 3; rules not recorded) |
| 1 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) |
| 1 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) |
| 1 | battle format: gauntlet Route 204 North (position 1; rules not recorded) |
| 1 | battle format: gauntlet Route 204 North (position 2; rules not recorded) |
| 1 | battle format: gauntlet Route 204 North (position 3; rules not recorded) |
| 1 | battle format: gauntlet Route 204 North B2B (position 1; rules not recorded) |
| 1 | battle format: gauntlet Route 204 North B2B (position 2; rules not recorded) |
| 1 | battle terrain: the battle terrain varies over its maps (TERRAIN_GRASS, TERRAIN_MOUNTAIN, TERRAIN_WATER), and Secret Power would read it |
| 1 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_BUILDING), and Secret Power would read it |
| 1 | held item effect #10: Leppa Berry |
| 1 | held item effect #15: Wiki Berry |
| 1 | held item effect #66: Lucky Egg |
| 1 | held item effect #89: Lucky Punch |
| 1 | move effect #108: Minimize |
| 1 | move effect #118: Swagger |
| 1 | move effect #124: Safeguard |
| 1 | move effect #148: Doom Desire, Future Sight |
| 1 | move effect #151: SolarBeam |
| 1 | move effect #159: Uproar |
| 1 | move effect #170: Focus Punch |
| 1 | move effect #176: Helping Hand |
| 1 | move effect #180: Assist |
| 1 | move effect #189: Endeavor |
| 1 | move effect #208: Bulk Up |
| 1 | move effect #230: Payback, Revenge |
| 1 | move effect #242: Copycat |
| 1 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 1 | move effect #38: Fissure, Guillotine, Horn Drill |
| 1 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower |
| 1 | move effect #79: Substitute |
| 1 | move effect #8: Dream Eater |
| 1 | move effect #90: Encore |
| 1 | move effect #91: Pain Split |

Groups by kind: AI 100, move effect 55, ability 25, battle format 12, held item effect 5, battle terrain 3, ROM change 2, form 1

## All ten splits (top 60)

420 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 163 | AI move effect #43 HIGH_CRITICAL not transcribed (Expert_HighCritical): Air Cutter, Aqua Cutter, Crabhammer, Cross Chop, Dragon Claw, Drill Peck, +11 more |
| 137 | AI move effect #147 DOUBLE_DAMAGE_DIG not transcribed (no Basic or Expert label): Earthquake |
| 137 | move effect #147: Earthquake |
| 130 | AI move effect #1 STATUS_SLEEP not transcribed (Basic_CheckCannotSleep / Expert_StatusSleep): Dark Void, GrassWhistle, Hypnosis, Lovely Kiss, Sing, Sleep Powder, +2 more |
| 121 | AI: AI_FLAG_TAG_STRATEGY |
| 121 | AI move effect #103 PRIORITY_1 not transcribed (no Basic or Expert label): Aqua Jet, Bullet Punch, Ice Shard, Mach Punch, Quick Attack, Shadow Sneak, +1 more |
| 115 | AI move effect #5 FREEZE_HIT not transcribed (no Basic or Expert label): Ice Beam, Ice Punch |
| 109 | AI move effect #48 RECOIL_QUARTER not transcribed (Expert_RecoilMove): Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower |
| 109 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 109 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower |
| 102 | AI move effect #17 BYPASS_ACCURACY not transcribed (Expert_BypassAccuracyMove): Aerial Ace, Aura Sphere, Aurora Beam, Dragon Pulse, Faint Attack, Magical Leaf, +5 more |
| 94 | AI: AI_FLAG_HARRASSMENT |
| 85 | AI: AI_FLAG_CHECK_HP |
| 78 | AI move effect #158 ALWAYS_FLINCH_FIRST_TURN_ONLY not transcribed (Basic_CheckFirstTurnInBattle / Expert_FakeOut): Fake Out |
| 71 | AI move effect #73 LOWER_ACCURACY_HIT not transcribed (no Basic or Expert label): Mirror Shot, Mud Bomb, Mud-Slap, Muddy Water, Octazooka |
| 71 | AI move effect #97 USE_RANDOM_LEARNED_MOVE_SLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_SleepTalk): Sleep Talk |
| 71 | move effect #97: Sleep Talk |
| 64 | battle format: a double_two_trainer battle |
| 64 | battle format: a multi battle |
| 63 | AI move effect #260 BLIZZARD not transcribed (Expert_Blizzard): Blizzard, Ice Ball, Sheer Cold |
| 63 | move effect #260: Blizzard, Ice Ball, Sheer Cold |
| 62 | AI move effect #274 FLINCH_FREEZE_HIT not transcribed (no Basic or Expert label): Avalanche, Ice Fang |
| 62 | ability: Magic Guard |
| 62 | move effect #274: Avalanche, Ice Fang |
| 61 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 60 | AI move effect #49 STATUS_CONFUSE not transcribed (Basic_CheckCannotConfuse / Expert_StatusConfuse): Confuse Ray, Supersonic, Sweet Kiss, Teeter Dance |
| 57 | ROM change: battle/skill/sub_seq.narc#251 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 52 | AI move effect #198 RECOIL_THIRD not transcribed (Expert_RecoilMove): Double-Edge, Wood Hammer |
| 52 | field condition: permanent Trick Room |
| 52 | move effect #198: Double-Edge, Wood Hammer |
| 49 | AI move effect #111 PROTECT not transcribed (Expert_Protect): Detect, Protect |
| 49 | AI move effect #182 LOWER_OWN_ATK_AND_DEF not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Superpower): Brave Bird, Close Combat, Take Down |
| 49 | AI move effect #228 SWITCH_HIT not transcribed (Expert_UTurn): U-turn |
| 49 | AI move effect #262 RECOIL_PARALYZE_HIT not transcribed (Expert_RecoilMove): Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 49 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 49 | move effect #182: Brave Bird, Close Combat, Take Down |
| 49 | move effect #228: U-turn |
| 49 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 46 | ROM change: arm9 MapHeader_GetMapType is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 46 | ability: Chlorophyll |
| 45 | AI move effect #29 MULTI_HIT not transcribed (no Basic or Expert label): Bone Rush, Bullet Seed, Fury Swipes, Icicle Spear, Pin Missile, Rock Blast, +1 more |
| 45 | AI move effect #67 STATUS_PARALYZE not transcribed (Basic_CheckCannotParalyze / Expert_StatusParalyze): Glare, Stun Spore, Thunder Wave |
| 45 | held item effect #65: Focus Band |
| 43 | AI move effect #152 THUNDER not transcribed (no Basic or Expert label): Thunder |
| 43 | move effect #152: Thunder |
| 41 | AI move effect #50 ATK_UP_2 not transcribed (Basic_CheckHighStatStage_Attack / Expert_StatusAttackUp): Swords Dance |
| 41 | AI move effect #79 SET_SUBSTITUTE not transcribed (Basic_CheckCannotSubstitute / Expert_Substitute): Substitute |
| 41 | move effect #79: Substitute |
| 40 | ability: Filter |
| 39 | ROM change: arm9 gFieldScriptCommands is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 37 | AI move effect #196 INCREASE_POWER_WITH_WEIGHT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Grass Knot, Low Kick |
| 37 | move effect #196: Grass Knot, Low Kick |
| 36 | AI move effect #112 SET_SPIKES not transcribed (Basic_CheckSpikes / Expert_Spikes): Spikes |
| 36 | AI move effect #53 SP_ATK_UP_2 not transcribed (Basic_CheckHighStatStage_SpAttack / Expert_StatusSpAttackUp): Nasty Plot, Tail Glow |
| 36 | move effect #112: Spikes |
| 35 | AI move effect #98 KO_MON_THAT_DEFEATED_USER not transcribed (Expert_DestinyBond): Destiny Bond |
| 35 | move effect #98: Destiny Bond |
| 34 | AI move effect #44 HIT_TWICE not transcribed (no Basic or Expert label): Bonemerang, Double Hit, Double Kick, DoubleSlap |
| 34 | battle format: a double battle |
| 33 | AI move effect #249 TOXIC_SPIKES not transcribed (Basic_CheckToxicSpikes / Expert_ToxicSpikes): Toxic Spikes |

Groups by kind: AI 169, move effect 111, ability 38, battle format 36, held item effect 26, ROM change 11, battle terrain 11, field condition 2, weather 1, form 1, form change 1

## Build order for the first two splits

Greedy: each step is the group that brings the most fights closest to runnable.

| Step | Group | Fights runnable after |
|---|---|---|
| 1 | AI move effect #103 PRIORITY_1 not transcribed (no Basic or Expert label): Aqua Jet, Bullet Punch, Ice Shard, Mach Punch, Quick Attack, Shadow Sneak, +1 more | 1 of 66 |
| 2 | AI move effect #1 STATUS_SLEEP not transcribed (Basic_CheckCannotSleep / Expert_StatusSleep): Dark Void, GrassWhistle, Hypnosis, Lovely Kiss, Sing, Sleep Powder, +2 more | 1 of 66 |
| 3 | AI move effect #43 HIGH_CRITICAL not transcribed (Expert_HighCritical): Air Cutter, Aqua Cutter, Crabhammer, Cross Chop, Dragon Claw, Drill Peck, +11 more | 1 of 66 |
| 4 | AI move effect #17 BYPASS_ACCURACY not transcribed (Expert_BypassAccuracyMove): Aerial Ace, Aura Sphere, Aurora Beam, Dragon Pulse, Faint Attack, Magical Leaf, +5 more | 1 of 66 |
| 5 | move effect #97: Sleep Talk | 1 of 66 |
| 6 | AI move effect #97 USE_RANDOM_LEARNED_MOVE_SLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_SleepTalk): Sleep Talk | 1 of 66 |
| 7 | battle format: a multi battle | 1 of 66 |
| 8 | AI move effect #73 LOWER_ACCURACY_HIT not transcribed (no Basic or Expert label): Mirror Shot, Mud Bomb, Mud-Slap, Muddy Water, Octazooka | 1 of 66 |
| 9 | AI move effect #67 STATUS_PARALYZE not transcribed (Basic_CheckCannotParalyze / Expert_StatusParalyze): Glare, Stun Spore, Thunder Wave | 1 of 66 |
| 10 | AI move effect #224 EAT_BERRY not transcribed (Expert_Pluck): Bug Bite, Pluck | 1 of 66 |
| 11 | AI move effect #49 STATUS_CONFUSE not transcribed (Basic_CheckCannotConfuse / Expert_StatusConfuse): Confuse Ray, Supersonic, Sweet Kiss, Teeter Dance | 1 of 66 |
| 12 | AI move effect #120 INFATUATE not transcribed (Basic_CheckCannotAttract): Attract | 2 of 66 |
| 13 | AI move effect #158 ALWAYS_FLINCH_FIRST_TURN_ONLY not transcribed (Basic_CheckFirstTurnInBattle / Expert_FakeOut): Fake Out | 2 of 66 |
| 14 | AI move effect #29 MULTI_HIT not transcribed (no Basic or Expert label): Bone Rush, Bullet Seed, Fury Swipes, Icicle Spear, Pin Missile, Rock Blast, +1 more | 3 of 66 |
| 15 | AI move effect #111 PROTECT not transcribed (Expert_Protect): Detect, Protect | 3 of 66 |
| 16 | AI: AI_FLAG_TAG_STRATEGY | 3 of 66 |
| 17 | AI: AI_FLAG_PRIORITIZE_EXTREMES | 3 of 66 |
| 18 | AI move effect #125 THAW_AND_BURN_HIT not transcribed (no Basic or Expert label): Flame Wheel, Sacred Fire, Scald | 3 of 66 |
| 19 | ability: Chlorophyll | 3 of 66 |
| 20 | ability: Aftermath | 4 of 66 |
| 21 | AI move effect #65 SET_REFLECT not transcribed (Basic_CheckAlreadyUnderReflect / Expert_Reflect): Reflect | 4 of 66 |
| 22 | AI move effect #70 LOWER_SPEED_HIT not transcribed (Expert_SpeedDownOnHit): Bubble, BubbleBeam, Bulldoze, Icy Wind, Mud Shot, Rock Tomb | 4 of 66 |
| 23 | ability: Magic Guard | 4 of 66 |
| 24 | AI: AI_FLAG_HARRASSMENT | 4 of 66 |
| 25 | AI: AI_FLAG_CHECK_HP | 4 of 66 |
| 26 | move effect #228: U-turn | 4 of 66 |
| 27 | AI move effect #228 SWITCH_HIT not transcribed (Expert_UTurn): U-turn | 4 of 66 |
| 28 | move effect #197: Secret Power | 4 of 66 |
| 29 | AI move effect #197 SECRET_POWER not transcribed (no Basic or Expert label): Secret Power | 4 of 66 |
| 30 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it | 4 of 66 |
| 31 | AI move-id case of TrainerAI_CalcDamage: Frustration, Return | 7 of 66 |
| 32 | AI move effect #5 FREEZE_HIT not transcribed (no Basic or Expert label): Ice Beam, Ice Punch | 7 of 66 |
| 33 | move effect #196: Grass Knot, Low Kick | 7 of 66 |
| 34 | AI move effect #196 INCREASE_POWER_WITH_WEIGHT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Grass Knot, Low Kick | 8 of 66 |
| 35 | move effect #198: Double-Edge, Wood Hammer | 8 of 66 |
| 36 | AI move effect #198 RECOIL_THIRD not transcribed (Expert_RecoilMove): Double-Edge, Wood Hammer | 8 of 66 |
| 37 | move effect #221: Brine, Wring Out | 8 of 66 |
| 38 | AI move effect #221 DOUBLE_POWER_WHEN_BELOW_HALF not transcribed (Expert_Brine): Brine, Wring Out | 8 of 66 |
| 39 | move effect #169: Facade | 8 of 66 |
| 40 | AI move effect #169 DOUBLE_POWER_WHEN_STATUSED not transcribed (Expert_Facade): Facade | 8 of 66 |
| 41 | AI move effect #87 LEVEL_DAMAGE_FLAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Night Shade, Seismic Toss | 8 of 66 |
| 42 | ability: Unburden | 9 of 66 |
| 43 | ability: Pressure | 10 of 66 |
| 44 | AI move effect #84 STATUS_LEECH_SEED not transcribed (Basic_CheckCannotLeechSeed / Expert_ToxicLeechSeed): Leech Seed | 10 of 66 |
| 45 | move effect #83: Metronome | 10 of 66 |
| 46 | AI move effect #83 CALL_RANDOM_MOVE not transcribed (no Basic or Expert label): Metronome | 11 of 66 |
| 47 | AI move effect #164 WEATHER_HAIL not transcribed (Basic_CheckHail / Expert_Hail): Hail | 12 of 66 |
| 48 | AI move effect #136 WEATHER_RAIN not transcribed (Basic_CheckRainDance / Expert_RainDance): Rain Dance | 12 of 66 |
| 49 | AI move effect #99 INCREASE_POWER_WITH_LESS_HP not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Reversal): Flail, Reversal | 13 of 66 |
| 50 | move effect #147: Earthquake | 13 of 66 |
| 51 | AI move effect #147 DOUBLE_DAMAGE_DIG not transcribed (no Basic or Expert label): Earthquake | 13 of 66 |
| 52 | move effect #260: Blizzard, Ice Ball, Sheer Cold | 13 of 66 |
| 53 | AI move effect #260 BLIZZARD not transcribed (Expert_Blizzard): Blizzard, Ice Ball, Sheer Cold | 13 of 66 |
| 54 | held item effect #65: Focus Band | 14 of 66 |
| 55 | move effect #249: Toxic Spikes | 14 of 66 |
| 56 | AI move effect #249 TOXIC_SPIKES not transcribed (Basic_CheckToxicSpikes / Expert_ToxicSpikes): Toxic Spikes | 14 of 66 |
| 57 | AI move effect #276 RAISE_SP_ATK_HIT not transcribed (no Basic or Expert label): Charge Beam | 14 of 66 |
| 58 | ability: Clear Body | 15 of 66 |
| 59 | form: trainer Pokemon in a data form: this game's party builder applies the form before computing its stats, which the engine's trainer party build does not model | 15 of 66 |
| 60 | AI: AI_FLAG_BATON_PASS | 16 of 66 |
| 61 | ability: Filter | 16 of 66 |
| 62 | move effect #152: Thunder | 16 of 66 |
| 63 | AI move effect #152 THUNDER not transcribed (no Basic or Expert label): Thunder | 16 of 66 |
| 64 | ability: Volt Absorb | 17 of 66 |
| 65 | move effect #41: Dragon Rage | 17 of 66 |
| 66 | AI move effect #41 40_DAMAGE_FLAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Dragon Rage | 18 of 66 |
| 67 | move effect #212: Dragon Dance | 18 of 66 |
| 68 | AI move effect #212 ATK_SPD_UP not transcribed (Basic_CheckDragonDance / Expert_DragonDance): Dragon Dance | 19 of 66 |
| 69 | ability: Magnet Pull | 20 of 66 |
| 70 | AI move effect #50 ATK_UP_2 not transcribed (Basic_CheckHighStatStage_Attack / Expert_StatusAttackUp): Swords Dance | 20 of 66 |
| 71 | AI move effect #28 FORCE_SWITCH not transcribed (Basic_CheckCanForceSwitch / Expert_ForceSwitch): Roar, Whirlwind | 21 of 66 |
| 72 | move effect #202: Poison Fang | 21 of 66 |
| 73 | AI move effect #202 BADLY_POISON_HIT not transcribed (no Basic or Expert label): Poison Fang | 21 of 66 |
| 74 | ability: Huge Power | 22 of 66 |
| 75 | AI move effect #44 HIT_TWICE not transcribed (no Basic or Expert label): Bonemerang, Double Hit, Double Kick, DoubleSlap | 22 of 66 |
| 76 | move effect #273: Fire Fang | 22 of 66 |
| 77 | AI move effect #273 FLINCH_BURN_HIT not transcribed (no Basic or Expert label): Fire Fang | 22 of 66 |
| 78 | ability: Swarm | 23 of 66 |
| 79 | battle format: a double battle | 23 of 66 |
| 80 | move effect #225: Tailwind | 23 of 66 |
| 81 | AI move effect #225 DOUBLE_SPEED_3_TURNS not transcribed (Basic_CheckTailwind / Expert_Tailwind): Tailwind | 24 of 66 |
| 82 | AI move effect #58 ATK_DOWN_2 not transcribed (Basic_CheckLowStatStage_Attack / Expert_StatusAttackDown): Charm, FeatherDance | 24 of 66 |
| 83 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) | 25 of 66 |
| 84 | move effect #214: Roost | 25 of 66 |
| 85 | AI move effect #214 HEAL_HALF_REMOVE_FLYING_TYPE not transcribed (Basic_CheckCanRecoverHP / Expert_Recovery): Roost | 26 of 66 |
| 86 | move effect #92: Snore | 26 of 66 |
| 87 | AI move effect #92 DAMAGE_WHILE_ASLEEP not transcribed (Basic_CheckAttackerAsleep / Expert_Nightmare): Snore | 26 of 66 |
| 88 | move effect #98: Destiny Bond | 26 of 66 |
| 89 | AI move effect #98 KO_MON_THAT_DEFEATED_USER not transcribed (Expert_DestinyBond): Destiny Bond | 27 of 66 |
| 90 | move effect #227: Bide, Metal Burst | 27 of 66 |
| 91 | AI move effect #227 METAL_BURST not transcribed (Basic_CheckMetalBurst / Expert_MetalBurst): Bide, Metal Burst | 28 of 66 |
| 92 | AI move effect #20 SPEED_DOWN not transcribed (Basic_CheckLowStatStage_Speed / Expert_StatusSpeedDown): String Shot | 29 of 66 |
| 93 | ability: Sniper | 29 of 66 |
| 94 | ability: Hyper Cutter | 30 of 66 |
| 95 | AI move effect #36 TRI_ATTACK not transcribed (no Basic or Expert label): Aeroblast, Tri Attack | 30 of 66 |
| 96 | ability: Download | 31 of 66 |
| 97 | AI move effect #149 DOUBLE_DAMAGE_FLY_OR_BOUNCE not transcribed (no Basic or Expert label): Gust, Sky Uppercut | 31 of 66 |
| 98 | AI move effect #23 ACC_DOWN not transcribed (Basic_CheckLowStatStage_Accuracy / Expert_StatusAccuracyDown): Flash, Sand-Attack, SmokeScreen | 32 of 66 |
| 99 | move effect #171: SmellingSalt | 32 of 66 |
| 100 | AI move effect #171 DOUBLE_POWER_AND_CURE_PARALYSIS not transcribed (Expert_SmellingSalts): SmellingSalt | 32 of 66 |
| 101 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) | 33 of 66 |
| 102 | move effect #150: Stomp | 33 of 66 |
| 103 | AI move effect #150 FLINCH_MINIMIZE_DOUBLE_HIT not transcribed (no Basic or Expert label): Stomp | 33 of 66 |
| 104 | ability: Arena Trap | 33 of 66 |
| 105 | AI move effect #138 RAISE_DEF_HIT not transcribed (no Basic or Expert label): Steel Wing | 34 of 66 |
| 106 | move effect #106: Block, Mean Look, Spider Web | 34 of 66 |
| 107 | AI move effect #106 PREVENT_ESCAPE not transcribed (Basic_CheckMeanLook / Expert_BindingMove): Block, Mean Look, Spider Web | 35 of 66 |
| 108 | move effect #172: Follow Me | 35 of 66 |
| 109 | AI move effect #172 MAKE_GLOBAL_TARGET not transcribed (no Basic or Expert label): Follow Me | 35 of 66 |
| 110 | move effect #257: Surf | 35 of 66 |
| 111 | AI move effect #257 DOUBLE_DAMAGE_DIVE not transcribed (no Basic or Expert label): Surf | 36 of 66 |
| 112 | AI move effect #51 DEF_UP_2 not transcribed (Basic_CheckHighStatStage_Defense / Expert_StatusDefenseUp): Acid Armor, Barrier, Iron Defense | 36 of 66 |
| 113 | AI move effect #135 RANDOM_POWER_BASED_ON_IVS not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Hidden Power | 37 of 66 |
| 114 | move effect #144: Mirror Coat | 37 of 66 |
| 115 | AI move effect #144 MIRROR_COAT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_MirrorCoat): Mirror Coat | 38 of 66 |
| 116 | move effect #192: Imprison | 38 of 66 |
| 117 | AI move effect #192 MAKE_SHARED_MOVES_UNUSEABLE not transcribed (Basic_CheckCanImprison / Expert_Imprison): Imprison | 38 of 66 |
| 118 | ability: Wonder Guard | 39 of 66 |
| 119 | move effect #188: Knock Off | 39 of 66 |
| 120 | AI move effect #188 REMOVE_HELD_ITEM not transcribed (Basic_CheckCanRemoveItem / Expert_KnockOff): Knock Off | 39 of 66 |
| 121 | battle format: gauntlet Route 204 North (position 2; rules not recorded) | 40 of 66 |
| 122 | ability: Bad Dreams | 40 of 66 |
| 123 | ability: Rain Dish | 40 of 66 |
| 124 | battle format: gauntlet Route 204 North B2B (position 2; rules not recorded) | 41 of 66 |
| 125 | move effect #274: Avalanche, Ice Fang | 41 of 66 |
| 126 | AI move effect #274 FLINCH_FREEZE_HIT not transcribed (no Basic or Expert label): Avalanche, Ice Fang | 41 of 66 |
| 127 | battle terrain: the battle terrain varies over its maps (TERRAIN_GRASS, TERRAIN_MOUNTAIN, TERRAIN_WATER), and Secret Power would read it | 42 of 66 |
| 128 | move effect #142: Belly Drum | 42 of 66 |
| 129 | AI move effect #142 MAX_ATK_LOSE_HALF_MAX_HP not transcribed (Basic_CheckBellyDrum / Expert_BellyDrum): Belly Drum | 42 of 66 |
| 130 | ability: Water Absorb | 43 of 66 |
| 131 | move effect #91: Pain Split | 43 of 66 |
| 132 | AI move effect #91 AVERAGE_HP not transcribed (Expert_PainSplit): Pain Split | 44 of 66 |
| 133 | move effect #8: Dream Eater | 44 of 66 |
| 134 | AI move effect #8 RECOVER_DAMAGE_SLEEP not transcribed (Basic_CheckDreamEater / Expert_DreamEater): Dream Eater | 45 of 66 |
| 135 | move effect #79: Substitute | 45 of 66 |
| 136 | AI move effect #79 SET_SUBSTITUTE not transcribed (Basic_CheckCannotSubstitute / Expert_Substitute): Substitute | 46 of 66 |
| 137 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge | 46 of 66 |
| 138 | AI move effect #262 RECOIL_PARALYZE_HIT not transcribed (Expert_RecoilMove): Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge | 47 of 66 |
| 139 | move effect #230: Payback, Revenge | 47 of 66 |
| 140 | AI move effect #230 DOUBLE_POWER_IF_MOVING_SECOND not transcribed (Expert_Payback): Payback, Revenge | 48 of 66 |
| 141 | move effect #208: Bulk Up | 48 of 66 |
| 142 | AI move effect #208 ATK_DEF_UP not transcribed (Basic_CheckBulkUp / Expert_StatusDefenseUp): Bulk Up | 49 of 66 |
| 143 | move effect #148: Doom Desire, Future Sight | 49 of 66 |
| 144 | AI move effect #148 HIT_IN_3_TURNS not transcribed (Basic_CheckFutureSight): Doom Desire, Future Sight | 50 of 66 |
| 145 | battle format: gauntlet Eterna Gym gym trainers (position 3; rules not recorded) | 50 of 66 |
| 146 | AI move effect #271 LOWER_SP_DEF_2_HIT not transcribed (no Basic or Expert label): Seed Flare | 51 of 66 |
| 147 | battle format: gauntlet Eterna Gym gym trainers (position 2; rules not recorded) | 51 of 66 |
| 148 | ability: Flower Gift | 52 of 66 |
| 149 | battle format: gauntlet Eterna Gym gym trainers (position 1; rules not recorded) | 52 of 66 |
| 150 | ability: Leaf Guard | 53 of 66 |
| 151 | move effect #45: Hi Jump Kick, Jump Kick | 53 of 66 |
| 152 | ability: Pure Power | 53 of 66 |
| 153 | AI move effect #45 CRASH_ON_MISS not transcribed (no Basic or Expert label): Hi Jump Kick, Jump Kick | 54 of 66 |
| 154 | move effect #112: Spikes | 54 of 66 |
| 155 | AI move effect #112 SET_SPIKES not transcribed (Basic_CheckSpikes / Expert_Spikes): Spikes | 54 of 66 |
| 156 | battle format: gauntlet Route 204 North (position 3; rules not recorded) | 54 of 66 |
| 157 | AI move effect #137 WEATHER_SUN not transcribed (Basic_CheckSunnyDay / Expert_SunnyDay): Sunny Day | 55 of 66 |
| 158 | AI move effect #167 STATUS_BURN not transcribed (Basic_CheckCannotBurn): Will-O-Wisp | 55 of 66 |
| 159 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower | 55 of 66 |
| 160 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 55 of 66 |
| 161 | AI move effect #48 RECOIL_QUARTER not transcribed (Expert_RecoilMove): Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower | 56 of 66 |
| 162 | move effect #242: Copycat | 56 of 66 |
| 163 | AI move effect #242 USE_LAST_USED_MOVE not transcribed (Basic_CheckCopycat / Expert_Copycat): Copycat | 56 of 66 |
| 164 | AI move effect #122 RANDOM_POWER_MAYBE_HEAL not transcribed (Basic_CheckNonStandardDamageOrChargeTurn): Present | 57 of 66 |
| 165 | move effect #176: Helping Hand | 57 of 66 |
| 166 | ability: Synchronize | 57 of 66 |
| 167 | AI move effect #176 BOOST_ALLY_POWER_BY_50_PERCENT not transcribed (Basic_CheckHelpingHand): Helping Hand | 58 of 66 |
| 168 | move effect #159: Uproar | 58 of 66 |
| 169 | AI move effect #159 UPROAR not transcribed (no Basic or Expert label): Uproar | 58 of 66 |
| 170 | AI move effect #139 RAISE_ATTACK_HIT not transcribed (no Basic or Expert label): Metal Claw, Meteor Mash | 59 of 66 |
| 171 | move effect #89: Counter | 59 of 66 |
| 172 | AI move effect #89 COUNTER not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Counter): Counter | 59 of 66 |
| 173 | held item effect #89: Lucky Punch | 59 of 66 |
| 174 | held item effect #66: Lucky Egg | 59 of 66 |
| 175 | held item effect #10: Leppa Berry | 60 of 66 |
| 176 | move effect #182: Brave Bird, Close Combat, Take Down | 60 of 66 |
| 177 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 60 of 66 |
| 178 | AI move effect #182 LOWER_OWN_ATK_AND_DEF not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Superpower): Brave Bird, Close Combat, Take Down | 60 of 66 |
| 179 | move effect #124: Safeguard | 60 of 66 |
| 180 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_BUILDING), and Secret Power would read it | 60 of 66 |
| 181 | AI move effect #124 PREVENT_STATUS not transcribed (Basic_CheckAlreadyUnderSafeguard): Safeguard | 61 of 66 |
| 182 | move effect #38: Fissure, Guillotine, Horn Drill | 61 of 66 |
| 183 | move effect #118: Swagger | 61 of 66 |
| 184 | AI move effect #38 ONE_HIT_KO not transcribed (Basic_CheckOHKOWouldFail / Expert_OHKOMove): Fissure, Guillotine, Horn Drill | 61 of 66 |
| 185 | AI move effect #118 ATK_UP_2_STATUS_CONFUSION not transcribed (Basic_CheckCannotConfuse / Expert_Swagger): Swagger | 62 of 66 |
| 186 | move effect #189: Endeavor | 62 of 66 |
| 187 | held item effect #15: Wiki Berry | 62 of 66 |
| 188 | AI move effect #189 SET_HP_EQUAL_TO_USER not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_Endeavor): Endeavor | 62 of 66 |
| 189 | AI move effect #10 ATK_UP not transcribed (Basic_CheckHighStatStage_Attack / Expert_StatusAttackUp): Howl, Meditate, Sharpen | 63 of 66 |
| 190 | move effect #180: Assist | 63 of 66 |
| 191 | move effect #151: SolarBeam | 63 of 66 |
| 192 | AI move effect #180 USE_RANDOM_ALLY_MOVE not transcribed (no Basic or Expert label): Assist | 63 of 66 |
| 193 | AI move effect #151 SKIP_CHARGE_TURN_IN_SUN not transcribed (Expert_ChargeTurnNoInvuln / Expert_Thunder): SolarBeam | 64 of 66 |
| 194 | move effect #170: Focus Punch | 64 of 66 |
| 195 | battle format: gauntlet Route 204 North (position 1; rules not recorded) | 64 of 66 |
| 196 | ability: Poison Heal | 64 of 66 |
| 197 | AI move effect #170 HIT_LAST_WHIFF_IF_HIT not transcribed (Basic_CheckNonStandardDamageOrChargeTurn / Expert_FocusPunch): Focus Punch | 65 of 66 |
| 198 | move effect #90: Encore | 65 of 66 |
| 199 | move effect #108: Minimize | 65 of 66 |
| 200 | battle format: gauntlet Route 204 North B2B (position 1; rules not recorded) | 65 of 66 |
| 201 | ability: Shadow Tag | 65 of 66 |
| 202 | AI move effect #90 ENCORE not transcribed (Basic_CheckCannotEncore / Expert_Encore): Encore | 65 of 66 |
| 203 | AI move effect #108 EVA_UP_2_MINIMIZE not transcribed (Basic_CheckHighStatStage_Evasion / Expert_StatusEvasionUp): Minimize | 66 of 66 |
