# Kaizo mechanics backlog

Generated from the opponent-side coverage check (`check_opponent`) on main after the K3
merge (exp and level-ups). It counts only what the trainers' teams, bags, AI and battle conditions need.
Your own Pokemon's moves, abilities and items are not counted.
Moves are grouped by the game's move-effect number and held items by hold-effect number,
since one implementation covers the whole group.

## By split

| Split | Fights | Runnable now | Singles | Distinct missing groups |
|---|---|---|---|---|
| Roark Split | 23 | 1 | 23 | 62 |
| Gardenia Split | 43 | 0 | 20 | 146 |
| Fantina Split | 41 | 0 | 37 | 160 |
| Maylene Split | 34 | 0 | 27 | 149 |
| Wake Split | 37 | 0 | 25 | 131 |
| Byron Split | 49 | 0 | 36 | 156 |
| Candice Split | 47 | 0 | 23 | 145 |
| Galactic Split | 102 | 0 | 40 | 245 |
| Volkner Split | 25 | 0 | 9 | 115 |
| Elite Four Split | 19 | 0 | 18 | 118 |

## Roark split

23 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 22 | AI: ability-guess check for this trainer |
| 11 | AI: flag set 0x7 |
| 10 | move effect #97: Sleep Talk |
| 6 | AI: AI_FLAG_SETUP_FIRST_TURN |
| 6 | move effect #197: Secret Power |
| 4 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it |
| 4 | move effect #92: Snore |
| 3 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 3 | AI: flag set 0x27 |
| 3 | AI: flag set 0xf |
| 3 | ability: Reckless |
| 3 | ability: Technician |
| 3 | ability: Truant |
| 3 | move effect #171: SmellingSalt |
| 3 | move effect #228: U-turn |
| 2 | ability: Dry Skin |
| 2 | ability: Lightningrod |
| 2 | ability: No Guard |
| 2 | ability: Serene Grace |
| 2 | move effect #196: Grass Knot, Low Kick |
| 2 | move effect #214: Roost |
| 2 | move effect #227: Bide, Metal Burst |
| 2 | move effect #83: Metronome |
| 1 | AI: AI_FLAG_CHECK_HP |
| 1 | AI: AI_FLAG_WEATHER |
| 1 | AI: flag set 0x1 |
| 1 | AI: flag set 0x10f |
| 1 | AI: flag set 0x1f |
| 1 | AI: flag set 0x20f |
| 1 | AI: flag set 0x5 |
| 1 | ability: Drought |
| 1 | ability: Honey Gather |
| 1 | ability: Levitate |
| 1 | ability: Quick Feet |
| 1 | ability: Scrappy |
| 1 | ability: Shell Armor |
| 1 | ability: Shield Dust |
| 1 | ability: Solar Power |
| 1 | ability: Super Luck |
| 1 | ability: Tangled Feet |
| 1 | ability: Thick Fat |
| 1 | ability: Tinted Lens |
| 1 | ability: Trace |
| 1 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) |
| 1 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) |
| 1 | held item effect #113: Damp Rock |
| 1 | move effect #112: Spikes |
| 1 | move effect #118: Swagger |
| 1 | move effect #136: Rain Dance |
| 1 | move effect #144: Mirror Coat |
| 1 | move effect #147: Earthquake |
| 1 | move effect #148: Doom Desire, Future Sight |
| 1 | move effect #159: Uproar |
| 1 | move effect #164: Hail |
| 1 | move effect #169: Facade |
| 1 | move effect #202: Poison Fang |
| 1 | move effect #212: Dragon Dance |
| 1 | move effect #221: Brine, Wring Out |
| 1 | move effect #225: Tailwind |
| 1 | move effect #242: Copycat |
| 1 | move effect #38: Fissure, Guillotine, Horn Drill |
| 1 | move effect #98: Destiny Bond |

Groups by kind: move effect 25, ability 20, AI 13, battle format 2, held item effect 1, battle terrain 1

## Roark and Gardenia splits (up to the second gym)

66 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 65 | AI: ability-guess check for this trainer |
| 24 | AI: AI_FLAG_SETUP_FIRST_TURN |
| 23 | move effect #97: Sleep Talk |
| 19 | AI: flag set 0x7 |
| 18 | battle format: a multi battle |
| 14 | AI: AI_FLAG_TAG_STRATEGY |
| 12 | ability: Levitate |
| 11 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 9 | AI: AI_FLAG_HARRASSMENT |
| 9 | ability: Magic Guard |
| 9 | move effect #228: U-turn |
| 8 | AI: flag set 0xf |
| 8 | ability: Chlorophyll |
| 8 | move effect #197: Secret Power |
| 7 | AI: AI_FLAG_CHECK_HP |
| 7 | move effect #198: Double-Edge, Wood Hammer |
| 7 | move effect #221: Brine, Wring Out |
| 6 | AI: flag set 0x27 |
| 6 | move effect #147: Earthquake |
| 6 | move effect #202: Poison Fang |
| 5 | ability: Aftermath |
| 5 | ability: Dry Skin |
| 5 | ability: Technician |
| 5 | ability: Thick Fat |
| 5 | battle format: a double battle |
| 5 | move effect #169: Facade |
| 5 | move effect #196: Grass Knot, Low Kick |
| 5 | move effect #212: Dragon Dance |
| 5 | move effect #249: Toxic Spikes |
| 4 | AI: AI_FLAG_WEATHER |
| 4 | ability: Drought |
| 4 | ability: Filter |
| 4 | ability: Lightningrod |
| 4 | ability: Reckless |
| 4 | ability: Trace |
| 4 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it |
| 4 | held item effect #113: Damp Rock |
| 4 | move effect #136: Rain Dance |
| 4 | move effect #152: Thunder |
| 4 | move effect #172: Follow Me |
| 4 | move effect #260: Blizzard, Ice Ball, Sheer Cold |
| 4 | move effect #273: Fire Fang |
| 4 | move effect #41: Dragon Rage |
| 4 | move effect #92: Snore |
| 4 | move effect #98: Destiny Bond |
| 4 | weather: Rain |
| 3 | AI: AI_FLAG_BATON_PASS |
| 3 | AI: flag set 0x8f |
| 3 | ability: Clear Body |
| 3 | ability: Scrappy |
| 3 | ability: Solar Power |
| 3 | ability: Tinted Lens |
| 3 | ability: Truant |
| 3 | form: trainer Pokemon in a data form: this game's party builder applies the form before computing its stats, which the engine's trainer party build does not model |
| 3 | held item effect #65: Focus Band |
| 3 | move effect #150: Stomp |
| 3 | move effect #171: SmellingSalt |
| 3 | move effect #214: Roost |
| 3 | move effect #225: Tailwind |
| 3 | move effect #83: Metronome |
| 2 | AI: flag set 0x20f |
| 2 | AI: flag set 0x417 |
| 2 | AI: flag set 0x41f |
| 2 | AI: flag set 0x87 |
| 2 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 2 | ability: Arena Trap |
| 2 | ability: Bad Dreams |
| 2 | ability: No Guard |
| 2 | ability: Pure Power |
| 2 | ability: Rain Dish |
| 2 | ability: Serene Grace |
| 2 | ability: Sniper |
| 2 | ability: Snow Warning |
| 2 | ability: Super Luck |
| 2 | ability: Swarm |
| 2 | ability: Tangled Feet |
| 2 | ability: Unburden |
| 2 | ability: Volt Absorb |
| 2 | held item effect #112: Heat Rock |
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
| 1 | AI: flag set 0x1 |
| 1 | AI: flag set 0x10f |
| 1 | AI: flag set 0x117 |
| 1 | AI: flag set 0x11f |
| 1 | AI: flag set 0x197 |
| 1 | AI: flag set 0x1f |
| 1 | AI: flag set 0x207 |
| 1 | AI: flag set 0x21f |
| 1 | AI: flag set 0x437 |
| 1 | AI: flag set 0x43f |
| 1 | AI: flag set 0x5 |
| 1 | AI: flag set 0x587 |
| 1 | AI: flag set 0x58b |
| 1 | AI: flag set 0x597 |
| 1 | AI: flag set 0x97 |
| 1 | AI: flag set 0xaf |
| 1 | AI: flag set 0xc7 |
| 1 | AI: flag set 0xd |
| 1 | AI: flag set 0xe7 |
| 1 | AI: flag set 0xef |
| 1 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 1 | ability: Battle Armor |
| 1 | ability: Download |
| 1 | ability: Drizzle |
| 1 | ability: Flower Gift |
| 1 | ability: Honey Gather |
| 1 | ability: Huge Power |
| 1 | ability: Hyper Cutter |
| 1 | ability: Leaf Guard |
| 1 | ability: Magnet Pull |
| 1 | ability: Poison Heal |
| 1 | ability: Pressure |
| 1 | ability: Quick Feet |
| 1 | ability: Sand Stream |
| 1 | ability: Shadow Tag |
| 1 | ability: Shell Armor |
| 1 | ability: Shield Dust |
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
| 1 | move effect #137: Sunny Day |
| 1 | move effect #148: Doom Desire, Future Sight |
| 1 | move effect #151: SolarBeam |
| 1 | move effect #159: Uproar |
| 1 | move effect #164: Hail |
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

Groups by kind: move effect 58, ability 49, AI 36, battle format 12, held item effect 7, battle terrain 3, ROM change 2, form 1, weather 1

## All ten splits (top 60)

420 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 419 | AI: ability-guess check for this trainer |
| 154 | AI: AI_FLAG_SETUP_FIRST_TURN |
| 137 | move effect #147: Earthquake |
| 121 | AI: AI_FLAG_TAG_STRATEGY |
| 109 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 109 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower |
| 107 | AI: flag set 0x7 |
| 94 | AI: AI_FLAG_HARRASSMENT |
| 85 | AI: AI_FLAG_CHECK_HP |
| 81 | ability: Levitate |
| 71 | move effect #97: Sleep Talk |
| 64 | battle format: a double_two_trainer battle |
| 64 | battle format: a multi battle |
| 63 | move effect #260: Blizzard, Ice Ball, Sheer Cold |
| 62 | ability: Magic Guard |
| 62 | move effect #274: Avalanche, Ice Fang |
| 61 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 60 | weather: Rain |
| 57 | ROM change: battle/skill/sub_seq.narc#251 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 52 | field condition: permanent Trick Room |
| 52 | move effect #198: Double-Edge, Wood Hammer |
| 49 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 49 | ability: Thick Fat |
| 49 | move effect #182: Brave Bird, Close Combat, Take Down |
| 49 | move effect #228: U-turn |
| 49 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 46 | ROM change: arm9 MapHeader_GetMapType is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 46 | ability: Chlorophyll |
| 45 | ability: Technician |
| 45 | held item effect #65: Focus Band |
| 43 | move effect #152: Thunder |
| 41 | move effect #79: Substitute |
| 40 | ability: Filter |
| 39 | AI: flag set 0xf |
| 39 | ROM change: arm9 gFieldScriptCommands is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 38 | ability: Reckless |
| 38 | ability: Serene Grace |
| 38 | weather: Hail |
| 37 | move effect #196: Grass Knot, Low Kick |
| 36 | move effect #112: Spikes |
| 35 | move effect #98: Destiny Bond |
| 34 | battle format: a double battle |
| 33 | ability: Scrappy |
| 33 | move effect #249: Toxic Spikes |
| 32 | move effect #197: Secret Power |
| 31 | ability: Aftermath |
| 31 | ability: Tinted Lens |
| 31 | move effect #253: Blast Burn, Eruption, Flare Blitz, Overheat |
| 30 | move effect #211: Calm Mind |
| 30 | weather: Fog |
| 28 | ability: Shadow Tag |
| 27 | ability: Pressure |
| 26 | AI: AI_FLAG_BATON_PASS |
| 26 | ability: Clear Body |
| 26 | ability: Super Luck |
| 25 | ability: Dry Skin |
| 25 | ability: Lightningrod |
| 25 | move effect #225: Tailwind |
| 24 | ROM change: battle/skill/be_seq.narc#141 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 24 | ROM change: overlay16 BattleControllerPlayer_CheckMoveHitAccuracy is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |

Groups by kind: move effect 114, AI 86, ability 62, battle format 36, held item effect 28, ROM change 11, battle terrain 11, weather 4, field condition 2, form 1, form change 1

## Build order for the first two splits

Greedy: each step is the group that brings the most fights closest to runnable.

| Step | Group | Fights runnable after |
|---|---|---|
| 1 | AI: ability-guess check for this trainer | 1 of 66 |
| 2 | AI: flag set 0x7 | 1 of 66 |
| 3 | move effect #97: Sleep Talk | 1 of 66 |
| 4 | AI: AI_FLAG_SETUP_FIRST_TURN | 1 of 66 |
| 5 | battle format: a multi battle | 1 of 66 |
| 6 | AI: AI_FLAG_TAG_STRATEGY | 1 of 66 |
| 7 | ability: Levitate | 1 of 66 |
| 8 | ability: Dry Skin | 2 of 66 |
| 9 | AI: AI_FLAG_PRIORITIZE_EXTREMES | 2 of 66 |
| 10 | move effect #196: Grass Knot, Low Kick | 3 of 66 |
| 11 | move effect #169: Facade | 3 of 66 |
| 12 | AI: AI_FLAG_CHECK_HP | 3 of 66 |
| 13 | AI: AI_FLAG_HARRASSMENT | 3 of 66 |
| 14 | ability: Magic Guard | 3 of 66 |
| 15 | move effect #228: U-turn | 3 of 66 |
| 16 | ability: Chlorophyll | 3 of 66 |
| 17 | AI: flag set 0xf | 3 of 66 |
| 18 | move effect #197: Secret Power | 3 of 66 |
| 19 | AI: flag set 0x27 | 3 of 66 |
| 20 | move effect #198: Double-Edge, Wood Hammer | 3 of 66 |
| 21 | ability: Thick Fat | 3 of 66 |
| 22 | move effect #202: Poison Fang | 3 of 66 |
| 23 | move effect #92: Snore | 3 of 66 |
| 24 | move effect #227: Bide, Metal Burst | 4 of 66 |
| 25 | move effect #221: Brine, Wring Out | 4 of 66 |
| 26 | ability: Aftermath | 4 of 66 |
| 27 | move effect #249: Toxic Spikes | 4 of 66 |
| 28 | move effect #147: Earthquake | 4 of 66 |
| 29 | move effect #212: Dragon Dance | 4 of 66 |
| 30 | ability: Super Luck | 5 of 66 |
| 31 | ability: Tangled Feet | 6 of 66 |
| 32 | ability: Lightningrod | 6 of 66 |
| 33 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it | 6 of 66 |
| 34 | ability: Reckless | 6 of 66 |
| 35 | ability: Technician | 6 of 66 |
| 36 | ability: Truant | 9 of 66 |
| 37 | move effect #260: Blizzard, Ice Ball, Sheer Cold | 9 of 66 |
| 38 | ability: Trace | 9 of 66 |
| 39 | move effect #98: Destiny Bond | 10 of 66 |
| 40 | move effect #171: SmellingSalt | 10 of 66 |
| 41 | move effect #273: Fire Fang | 10 of 66 |
| 42 | move effect #242: Copycat | 11 of 66 |
| 43 | move effect #230: Payback, Revenge | 12 of 66 |
| 44 | move effect #214: Roost | 12 of 66 |
| 45 | move effect #148: Doom Desire, Future Sight | 13 of 66 |
| 46 | ability: Huge Power | 14 of 66 |
| 47 | AI: flag set 0x10f | 15 of 66 |
| 48 | weather: Rain | 15 of 66 |
| 49 | move effect #150: Stomp | 15 of 66 |
| 50 | move effect #257: Surf | 16 of 66 |
| 51 | ability: Solar Power | 16 of 66 |
| 52 | ability: Drought | 16 of 66 |
| 53 | AI: flag set 0x8f | 16 of 66 |
| 54 | held item effect #65: Focus Band | 17 of 66 |
| 55 | battle format: a double battle | 17 of 66 |
| 56 | move effect #106: Block, Mean Look, Spider Web | 18 of 66 |
| 57 | move effect #152: Thunder | 18 of 66 |
| 58 | ability: Filter | 18 of 66 |
| 59 | ability: Scrappy | 18 of 66 |
| 60 | form: trainer Pokemon in a data form: this game's party builder applies the form before computing its stats, which the engine's trainer party build does not model | 19 of 66 |
| 61 | move effect #159: Uproar | 20 of 66 |
| 62 | battle format: gauntlet Eterna Gym gym trainers (position 3; rules not recorded) | 21 of 66 |
| 63 | move effect #83: Metronome | 21 of 66 |
| 64 | ability: Serene Grace | 22 of 66 |
| 65 | move effect #41: Dragon Rage | 22 of 66 |
| 66 | AI: AI_FLAG_WEATHER | 22 of 66 |
| 67 | move effect #136: Rain Dance | 22 of 66 |
| 68 | held item effect #113: Damp Rock | 22 of 66 |
| 69 | AI: flag set 0x20f | 23 of 66 |
| 70 | AI: flag set 0x207 | 24 of 66 |
| 71 | ability: Clear Body | 24 of 66 |
| 72 | AI: flag set 0x11f | 25 of 66 |
| 73 | move effect #225: Tailwind | 25 of 66 |
| 74 | ability: Volt Absorb | 25 of 66 |
| 75 | ability: Sniper | 25 of 66 |
| 76 | ability: Arena Trap | 26 of 66 |
| 77 | ability: Hyper Cutter | 27 of 66 |
| 78 | AI: flag set 0x197 | 28 of 66 |
| 79 | ability: Unburden | 28 of 66 |
| 80 | AI: flag set 0x87 | 29 of 66 |
| 81 | move effect #172: Follow Me | 29 of 66 |
| 82 | ability: Tinted Lens | 29 of 66 |
| 83 | ability: Swarm | 29 of 66 |
| 84 | AI: flag set 0x43f | 30 of 66 |
| 85 | ability: Snow Warning | 30 of 66 |
| 86 | ability: Download | 31 of 66 |
| 87 | AI: AI_FLAG_BATON_PASS | 31 of 66 |
| 88 | AI: flag set 0xe7 | 32 of 66 |
| 89 | AI: flag set 0x417 | 32 of 66 |
| 90 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge | 33 of 66 |
| 91 | move effect #274: Avalanche, Ice Fang | 33 of 66 |
| 92 | battle terrain: the battle terrain varies over its maps (TERRAIN_GRASS, TERRAIN_MOUNTAIN, TERRAIN_WATER), and Secret Power would read it | 34 of 66 |
| 93 | move effect #142: Belly Drum | 34 of 66 |
| 94 | move effect #208: Bulk Up | 35 of 66 |
| 95 | move effect #188: Knock Off | 35 of 66 |
| 96 | battle format: gauntlet Route 204 North (position 2; rules not recorded) | 36 of 66 |
| 97 | ability: Rain Dish | 36 of 66 |
| 98 | ability: Water Absorb | 37 of 66 |
| 99 | ability: Bad Dreams | 37 of 66 |
| 100 | battle format: gauntlet Route 204 North B2B (position 2; rules not recorded) | 38 of 66 |
| 101 | AI: flag set 0x41f | 38 of 66 |
| 102 | move effect #8: Dream Eater | 39 of 66 |
| 103 | held item effect #112: Heat Rock | 39 of 66 |
| 104 | AI: flag set 0x21f | 40 of 66 |
| 105 | ability: No Guard | 40 of 66 |
| 106 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) | 41 of 66 |
| 107 | move effect #45: Hi Jump Kick, Jump Kick | 41 of 66 |
| 108 | ability: Pure Power | 42 of 66 |
| 109 | move effect #192: Imprison | 42 of 66 |
| 110 | move effect #144: Mirror Coat | 42 of 66 |
| 111 | ability: Shell Armor | 43 of 66 |
| 112 | move effect #112: Spikes | 43 of 66 |
| 113 | move effect #91: Pain Split | 43 of 66 |
| 114 | AI: flag set 0x437 | 44 of 66 |
| 115 | move effect #79: Substitute | 44 of 66 |
| 116 | AI: flag set 0xd | 45 of 66 |
| 117 | move effect #38: Fissure, Guillotine, Horn Drill | 45 of 66 |
| 118 | move effect #118: Swagger | 46 of 66 |
| 119 | move effect #180: Assist | 46 of 66 |
| 120 | move effect #151: SolarBeam | 47 of 66 |
| 121 | move effect #176: Helping Hand | 47 of 66 |
| 122 | ability: Synchronize | 48 of 66 |
| 123 | move effect #164: Hail | 48 of 66 |
| 124 | AI: flag set 0x1f | 49 of 66 |
| 125 | move effect #137: Sunny Day | 49 of 66 |
| 126 | battle format: gauntlet Route 204 North (position 3; rules not recorded) | 50 of 66 |
| 127 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) | 50 of 66 |
| 128 | ability: Honey Gather | 51 of 66 |
| 129 | battle format: gauntlet Eterna Gym gym trainers (position 2; rules not recorded) | 51 of 66 |
| 130 | ability: Flower Gift | 52 of 66 |
| 131 | battle format: gauntlet Eterna Gym gym trainers (position 1; rules not recorded) | 52 of 66 |
| 132 | ability: Leaf Guard | 53 of 66 |
| 133 | ability: Wonder Guard | 53 of 66 |
| 134 | AI: flag set 0xaf | 54 of 66 |
| 135 | ability: Shield Dust | 54 of 66 |
| 136 | AI: flag set 0x1 | 55 of 66 |
| 137 | ability: Sand Stream | 55 of 66 |
| 138 | AI: flag set 0xef | 56 of 66 |
| 139 | ability: Quick Feet | 56 of 66 |
| 140 | AI: flag set 0x5 | 57 of 66 |
| 141 | ability: Pressure | 57 of 66 |
| 142 | AI: flag set 0x597 | 58 of 66 |
| 143 | ability: Magnet Pull | 58 of 66 |
| 144 | AI: flag set 0x117 | 59 of 66 |
| 145 | ability: Drizzle | 59 of 66 |
| 146 | AI: flag set 0x587 | 60 of 66 |
| 147 | move effect #182: Brave Bird, Close Combat, Take Down | 60 of 66 |
| 148 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 60 of 66 |
| 149 | move effect #189: Endeavor | 60 of 66 |
| 150 | held item effect #15: Wiki Berry | 61 of 66 |
| 151 | move effect #89: Counter | 61 of 66 |
| 152 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower | 61 of 66 |
| 153 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 61 of 66 |
| 154 | AI: flag set 0x97 | 62 of 66 |
| 155 | move effect #170: Focus Punch | 62 of 66 |
| 156 | battle format: gauntlet Route 204 North (position 1; rules not recorded) | 62 of 66 |
| 157 | ability: Poison Heal | 63 of 66 |
| 158 | move effect #124: Safeguard | 63 of 66 |
| 159 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_BUILDING), and Secret Power would read it | 63 of 66 |
| 160 | AI: flag set 0xc7 | 64 of 66 |
| 161 | move effect #90: Encore | 64 of 66 |
| 162 | move effect #108: Minimize | 64 of 66 |
| 163 | battle format: gauntlet Route 204 North B2B (position 1; rules not recorded) | 64 of 66 |
| 164 | ability: Shadow Tag | 65 of 66 |
| 165 | held item effect #89: Lucky Punch | 65 of 66 |
| 166 | held item effect #66: Lucky Egg | 65 of 66 |
| 167 | held item effect #10: Leppa Berry | 65 of 66 |
| 168 | ability: Battle Armor | 65 of 66 |
| 169 | AI: flag set 0x58b | 66 of 66 |
