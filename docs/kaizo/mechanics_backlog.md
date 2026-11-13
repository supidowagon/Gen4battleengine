# Kaizo mechanics backlog

Generated from the opponent-side coverage check (`check_opponent`) on main after the K3
merge (exp and level-ups). It counts only what the trainers' teams, bags, AI and battle conditions need.
Your own Pokemon's moves, abilities and items are not counted.
Moves are grouped by the game's move-effect number and held items by hold-effect number,
since one implementation covers the whole group.

## By split

| Split | Fights | Runnable now | Singles | Distinct missing groups |
|---|---|---|---|---|
| Roark Split | 23 | 1 | 23 | 94 |
| Gardenia Split | 43 | 0 | 20 | 181 |
| Fantina Split | 41 | 0 | 37 | 201 |
| Maylene Split | 34 | 0 | 27 | 189 |
| Wake Split | 37 | 0 | 25 | 161 |
| Byron Split | 49 | 0 | 36 | 197 |
| Candice Split | 47 | 0 | 23 | 179 |
| Galactic Split | 102 | 0 | 40 | 282 |
| Volkner Split | 25 | 0 | 9 | 135 |
| Elite Four Split | 19 | 0 | 18 | 136 |

## Roark split

23 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 22 | AI: ability-guess check for this trainer |
| 11 | AI: flag set 0x7 |
| 10 | move effect #97: Sleep Talk |
| 8 | held item effect #86: Silk Scarf |
| 6 | AI: AI_FLAG_SETUP_FIRST_TURN |
| 6 | move effect #197: Secret Power |
| 4 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it |
| 4 | held item effect #101: Flame Orb |
| 4 | held item effect #96: Expert Belt |
| 4 | move effect #92: Snore |
| 3 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 3 | AI: flag set 0x27 |
| 3 | AI: flag set 0xf |
| 3 | ability: Reckless |
| 3 | ability: Technician |
| 3 | ability: Truant |
| 3 | held item effect #55: Choice Band |
| 3 | held item effect #78: Mystic Water |
| 3 | held item effect #84: Charcoal |
| 3 | move effect #171: SmellingSalt |
| 3 | move effect #228: U-turn |
| 2 | ability: Dry Skin |
| 2 | ability: Gluttony |
| 2 | ability: Lightningrod |
| 2 | ability: No Guard |
| 2 | ability: Serene Grace |
| 2 | held item effect #100: Toxic Orb |
| 2 | held item effect #20: Passho Berry |
| 2 | held item effect #69: Leftovers |
| 2 | held item effect #74: Miracle Seed, Rose Incense |
| 2 | held item effect #79: Sharp Beak |
| 2 | held item effect #98: Life Orb |
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
| 1 | held item effect #0: Bicycle, Big Nugget, Big Pearl, Blue Scarf, Cherish Ball, Nanab Berry, +3 more |
| 1 | held item effect #104: Zoom Lens |
| 1 | held item effect #113: Damp Rock |
| 1 | held item effect #116: Sticky Barb |
| 1 | held item effect #19: Occa Berry |
| 1 | held item effect #21: Wacan Berry |
| 1 | held item effect #36: Berserk Gene, Liechi Berry |
| 1 | held item effect #38: Salac Berry |
| 1 | held item effect #45: Custap Berry |
| 1 | held item effect #48: BrightPowder, Lax Incense |
| 1 | held item effect #5: Cheri Berry |
| 1 | held item effect #56: King's Rock, Razor Fang |
| 1 | held item effect #57: SilverPowder |
| 1 | held item effect #72: Soft Sand |
| 1 | held item effect #77: Magnet |
| 1 | held item effect #80: Poison Barb |
| 1 | held item effect #83: Odd Incense, TwistedSpoon |
| 1 | held item effect #88: Shell Bell |
| 1 | held item effect #94: Muscle Band |
| 1 | held item effect #97: Light Clay |
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

Groups by kind: held item effect 32, move effect 25, ability 21, AI 13, battle format 2, battle terrain 1

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
| 14 | held item effect #98: Life Orb |
| 13 | held item effect #86: Silk Scarf |
| 12 | ability: Levitate |
| 11 | AI: AI_FLAG_PRIORITIZE_EXTREMES |
| 9 | AI: AI_FLAG_HARRASSMENT |
| 9 | ability: Magic Guard |
| 9 | held item effect #69: Leftovers |
| 9 | move effect #228: U-turn |
| 8 | AI: flag set 0xf |
| 8 | ability: Chlorophyll |
| 8 | held item effect #101: Flame Orb |
| 8 | held item effect #56: King's Rock, Razor Fang |
| 8 | move effect #197: Secret Power |
| 7 | AI: AI_FLAG_CHECK_HP |
| 7 | held item effect #78: Mystic Water |
| 7 | held item effect #94: Muscle Band |
| 7 | move effect #198: Double-Edge, Wood Hammer |
| 7 | move effect #221: Brine, Wring Out |
| 6 | AI: flag set 0x27 |
| 6 | ability: Gluttony |
| 6 | held item effect #96: Expert Belt |
| 6 | move effect #147: Earthquake |
| 6 | move effect #202: Poison Fang |
| 5 | ability: Aftermath |
| 5 | ability: Dry Skin |
| 5 | ability: Technician |
| 5 | ability: Thick Fat |
| 5 | battle format: a double battle |
| 5 | held item effect #48: BrightPowder, Lax Incense |
| 5 | held item effect #97: Light Clay |
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
| 4 | held item effect #74: Miracle Seed, Rose Incense |
| 4 | held item effect #95: Wise Glasses |
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
| 3 | held item effect #19: Occa Berry |
| 3 | held item effect #45: Custap Berry |
| 3 | held item effect #55: Choice Band |
| 3 | held item effect #65: Focus Band |
| 3 | held item effect #79: Sharp Beak |
| 3 | held item effect #80: Poison Barb |
| 3 | held item effect #84: Charcoal |
| 3 | held item effect #93: Wide Lens |
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
| 2 | held item effect #0: Bicycle, Big Nugget, Big Pearl, Blue Scarf, Cherish Ball, Nanab Berry, +3 more |
| 2 | held item effect #100: Toxic Orb |
| 2 | held item effect #112: Heat Rock |
| 2 | held item effect #20: Passho Berry |
| 2 | held item effect #36: Berserk Gene, Liechi Berry |
| 2 | held item effect #77: Magnet |
| 2 | held item effect #81: NeverMeltIce |
| 2 | held item effect #82: Spell Tag |
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
| 1 | held item effect #104: Zoom Lens |
| 1 | held item effect #109: Black Sludge |
| 1 | held item effect #115: Choice Scarf |
| 1 | held item effect #116: Sticky Barb |
| 1 | held item effect #141: Iron Plate |
| 1 | held item effect #15: Wiki Berry |
| 1 | held item effect #21: Wacan Berry |
| 1 | held item effect #23: Yache Berry |
| 1 | held item effect #26: Shuca Berry |
| 1 | held item effect #28: Payapa Berry |
| 1 | held item effect #29: Tanga Berry |
| 1 | held item effect #30: Charti Berry |
| 1 | held item effect #38: Salac Berry |
| 1 | held item effect #39: Petaya Berry |
| 1 | held item effect #5: Cheri Berry |
| 1 | held item effect #57: SilverPowder |
| 1 | held item effect #66: Lucky Egg |
| 1 | held item effect #72: Soft Sand |
| 1 | held item effect #73: Hard Stone |
| 1 | held item effect #75: BlackGlasses |
| 1 | held item effect #83: Odd Incense, TwistedSpoon |
| 1 | held item effect #88: Shell Bell |
| 1 | held item effect #89: Lucky Punch |
| 1 | held item effect #91: Thick Club |
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

Groups by kind: move effect 58, held item effect 54, ability 50, AI 36, battle format 12, battle terrain 3, ROM change 2, form 1, weather 1

## All ten splits (top 60)

420 fights. A group's count is the number of fights it blocks.

| Fights blocked | Group |
|---|---|
| 419 | AI: ability-guess check for this trainer |
| 154 | AI: AI_FLAG_SETUP_FIRST_TURN |
| 137 | move effect #147: Earthquake |
| 128 | held item effect #69: Leftovers |
| 121 | AI: AI_FLAG_TAG_STRATEGY |
| 116 | held item effect #98: Life Orb |
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
| 50 | held item effect #48: BrightPowder, Lax Incense |
| 49 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) |
| 49 | ability: Thick Fat |
| 49 | move effect #182: Brave Bird, Close Combat, Take Down |
| 49 | move effect #228: U-turn |
| 49 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge |
| 47 | held item effect #56: King's Rock, Razor Fang |
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
| 37 | held item effect #86: Silk Scarf |
| 37 | move effect #196: Grass Knot, Low Kick |
| 36 | move effect #112: Spikes |
| 35 | move effect #98: Destiny Bond |
| 34 | battle format: a double battle |
| 33 | ability: Scrappy |
| 33 | move effect #249: Toxic Spikes |
| 32 | move effect #197: Secret Power |
| 31 | ability: Aftermath |
| 31 | ability: Tinted Lens |
| 31 | held item effect #93: Wide Lens |
| 31 | move effect #253: Blast Burn, Eruption, Flare Blitz, Overheat |
| 30 | move effect #211: Calm Mind |
| 30 | weather: Fog |
| 28 | ability: Shadow Tag |
| 27 | ability: Pressure |
| 27 | held item effect #45: Custap Berry |
| 26 | AI: AI_FLAG_BATON_PASS |

Groups by kind: move effect 114, held item effect 97, AI 86, ability 63, battle format 36, ROM change 11, battle terrain 11, weather 4, field condition 2, form 1, form change 1

## Build order for the first two splits

Greedy: each step is the group that brings the most fights closest to runnable.

| Step | Group | Fights runnable after |
|---|---|---|
| 1 | AI: ability-guess check for this trainer | 1 of 66 |
| 2 | AI: AI_FLAG_SETUP_FIRST_TURN | 1 of 66 |
| 3 | AI: flag set 0x7 | 1 of 66 |
| 4 | move effect #97: Sleep Talk | 1 of 66 |
| 5 | held item effect #86: Silk Scarf | 1 of 66 |
| 6 | battle format: a multi battle | 1 of 66 |
| 7 | held item effect #98: Life Orb | 1 of 66 |
| 8 | AI: AI_FLAG_TAG_STRATEGY | 1 of 66 |
| 9 | ability: Levitate | 1 of 66 |
| 10 | AI: AI_FLAG_PRIORITIZE_EXTREMES | 1 of 66 |
| 11 | held item effect #69: Leftovers | 1 of 66 |
| 12 | ability: Super Luck | 2 of 66 |
| 13 | AI: AI_FLAG_CHECK_HP | 2 of 66 |
| 14 | AI: AI_FLAG_HARRASSMENT | 2 of 66 |
| 15 | ability: Magic Guard | 2 of 66 |
| 16 | held item effect #101: Flame Orb | 2 of 66 |
| 17 | move effect #228: U-turn | 2 of 66 |
| 18 | ability: Chlorophyll | 2 of 66 |
| 19 | held item effect #56: King's Rock, Razor Fang | 2 of 66 |
| 20 | ability: Gluttony | 2 of 66 |
| 21 | move effect #221: Brine, Wring Out | 2 of 66 |
| 22 | move effect #169: Facade | 2 of 66 |
| 23 | ability: Aftermath | 2 of 66 |
| 24 | held item effect #94: Muscle Band | 2 of 66 |
| 25 | AI: flag set 0xf | 2 of 66 |
| 26 | move effect #249: Toxic Spikes | 2 of 66 |
| 27 | move effect #198: Double-Edge, Wood Hammer | 2 of 66 |
| 28 | held item effect #78: Mystic Water | 2 of 66 |
| 29 | move effect #197: Secret Power | 2 of 66 |
| 30 | move effect #92: Snore | 2 of 66 |
| 31 | AI: flag set 0x10f | 3 of 66 |
| 32 | held item effect #96: Expert Belt | 3 of 66 |
| 33 | AI: flag set 0x27 | 3 of 66 |
| 34 | move effect #147: Earthquake | 3 of 66 |
| 35 | move effect #212: Dragon Dance | 3 of 66 |
| 36 | move effect #202: Poison Fang | 3 of 66 |
| 37 | ability: Thick Fat | 3 of 66 |
| 38 | ability: Reckless | 3 of 66 |
| 39 | ability: Technician | 3 of 66 |
| 40 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_WATER), and Secret Power would read it | 3 of 66 |
| 41 | held item effect #55: Choice Band | 3 of 66 |
| 42 | ability: Truant | 6 of 66 |
| 43 | move effect #152: Thunder | 6 of 66 |
| 44 | move effect #260: Blizzard, Ice Ball, Sheer Cold | 6 of 66 |
| 45 | move effect #150: Stomp | 6 of 66 |
| 46 | move effect #257: Surf | 7 of 66 |
| 47 | ability: Dry Skin | 7 of 66 |
| 48 | held item effect #19: Occa Berry | 7 of 66 |
| 49 | held item effect #20: Passho Berry | 8 of 66 |
| 50 | move effect #196: Grass Knot, Low Kick | 8 of 66 |
| 51 | held item effect #74: Miracle Seed, Rose Incense | 8 of 66 |
| 52 | held item effect #84: Charcoal | 8 of 66 |
| 53 | held item effect #77: Magnet | 9 of 66 |
| 54 | held item effect #97: Light Clay | 9 of 66 |
| 55 | weather: Rain | 9 of 66 |
| 56 | ability: Drought | 9 of 66 |
| 57 | ability: Scrappy | 9 of 66 |
| 58 | move effect #159: Uproar | 10 of 66 |
| 59 | held item effect #79: Sharp Beak | 10 of 66 |
| 60 | ability: Tangled Feet | 11 of 66 |
| 61 | ability: Lightningrod | 11 of 66 |
| 62 | held item effect #65: Focus Band | 11 of 66 |
| 63 | AI: flag set 0x8f | 12 of 66 |
| 64 | move effect #273: Fire Fang | 12 of 66 |
| 65 | held item effect #48: BrightPowder, Lax Incense | 12 of 66 |
| 66 | ability: Swarm | 12 of 66 |
| 67 | AI: flag set 0x43f | 13 of 66 |
| 68 | battle format: a double battle | 13 of 66 |
| 69 | held item effect #95: Wise Glasses | 13 of 66 |
| 70 | ability: Filter | 13 of 66 |
| 71 | form: trainer Pokemon in a data form: this game's party builder applies the form before computing its stats, which the engine's trainer party build does not model | 14 of 66 |
| 72 | ability: Trace | 14 of 66 |
| 73 | ability: Solar Power | 14 of 66 |
| 74 | move effect #214: Roost | 14 of 66 |
| 75 | battle format: gauntlet Eterna Gym gym trainers (position 3; rules not recorded) | 15 of 66 |
| 76 | move effect #41: Dragon Rage | 15 of 66 |
| 77 | AI: AI_FLAG_WEATHER | 15 of 66 |
| 78 | move effect #136: Rain Dance | 15 of 66 |
| 79 | held item effect #113: Damp Rock | 15 of 66 |
| 80 | AI: flag set 0x20f | 16 of 66 |
| 81 | held item effect #93: Wide Lens | 16 of 66 |
| 82 | move effect #83: Metronome | 16 of 66 |
| 83 | held item effect #45: Custap Berry | 16 of 66 |
| 84 | move effect #98: Destiny Bond | 16 of 66 |
| 85 | held item effect #36: Berserk Gene, Liechi Berry | 16 of 66 |
| 86 | AI: flag set 0x207 | 17 of 66 |
| 87 | ability: Volt Absorb | 17 of 66 |
| 88 | AI: flag set 0x197 | 18 of 66 |
| 89 | ability: Sniper | 18 of 66 |
| 90 | ability: Hyper Cutter | 19 of 66 |
| 91 | ability: Clear Body | 19 of 66 |
| 92 | AI: flag set 0x11f | 20 of 66 |
| 93 | move effect #172: Follow Me | 20 of 66 |
| 94 | ability: Tinted Lens | 20 of 66 |
| 95 | move effect #227: Bide, Metal Burst | 20 of 66 |
| 96 | held item effect #100: Toxic Orb | 21 of 66 |
| 97 | move effect #106: Block, Mean Look, Spider Web | 21 of 66 |
| 98 | held item effect #115: Choice Scarf | 22 of 66 |
| 99 | held item effect #112: Heat Rock | 22 of 66 |
| 100 | AI: flag set 0x21f | 23 of 66 |
| 101 | ability: Unburden | 23 of 66 |
| 102 | AI: flag set 0x87 | 24 of 66 |
| 103 | move effect #225: Tailwind | 24 of 66 |
| 104 | ability: Snow Warning | 24 of 66 |
| 105 | ability: Download | 25 of 66 |
| 106 | AI: AI_FLAG_BATON_PASS | 25 of 66 |
| 107 | AI: flag set 0xe7 | 26 of 66 |
| 108 | AI: flag set 0x417 | 26 of 66 |
| 109 | move effect #262: Frenzy Plant, Sky Attack, Thrash, Volt Tackle, Wild Charge | 27 of 66 |
| 110 | move effect #171: SmellingSalt | 27 of 66 |
| 111 | held item effect #0: Bicycle, Big Nugget, Big Pearl, Blue Scarf, Cherish Ball, Nanab Berry, +3 more | 27 of 66 |
| 112 | move effect #242: Copycat | 28 of 66 |
| 113 | held item effect #80: Poison Barb | 28 of 66 |
| 114 | held item effect #81: NeverMeltIce | 28 of 66 |
| 115 | ability: Huge Power | 29 of 66 |
| 116 | ability: Serene Grace | 29 of 66 |
| 117 | held item effect #38: Salac Berry | 30 of 66 |
| 118 | move effect #188: Knock Off | 30 of 66 |
| 119 | battle format: gauntlet Route 204 North (position 2; rules not recorded) | 31 of 66 |
| 120 | ability: Arena Trap | 31 of 66 |
| 121 | held item effect #73: Hard Stone | 32 of 66 |
| 122 | move effect #274: Avalanche, Ice Fang | 32 of 66 |
| 123 | battle terrain: the battle terrain varies over its maps (TERRAIN_GRASS, TERRAIN_MOUNTAIN, TERRAIN_WATER), and Secret Power would read it | 33 of 66 |
| 124 | move effect #142: Belly Drum | 33 of 66 |
| 125 | move effect #208: Bulk Up | 34 of 66 |
| 126 | move effect #144: Mirror Coat | 34 of 66 |
| 127 | ability: Shell Armor | 35 of 66 |
| 128 | ability: Bad Dreams | 35 of 66 |
| 129 | AI: flag set 0x41f | 35 of 66 |
| 130 | move effect #8: Dream Eater | 36 of 66 |
| 131 | ability: Rain Dish | 36 of 66 |
| 132 | battle format: gauntlet Route 204 North B2B (position 2; rules not recorded) | 37 of 66 |
| 133 | move effect #45: Hi Jump Kick, Jump Kick | 37 of 66 |
| 134 | ability: Pure Power | 38 of 66 |
| 135 | move effect #192: Imprison | 38 of 66 |
| 136 | held item effect #82: Spell Tag | 38 of 66 |
| 137 | move effect #91: Pain Split | 38 of 66 |
| 138 | AI: flag set 0x437 | 39 of 66 |
| 139 | move effect #164: Hail | 39 of 66 |
| 140 | AI: flag set 0x1f | 40 of 66 |
| 141 | move effect #112: Spikes | 40 of 66 |
| 142 | ability: No Guard | 40 of 66 |
| 143 | move effect #137: Sunny Day | 40 of 66 |
| 144 | battle format: gauntlet Route 204 North (position 3; rules not recorded) | 41 of 66 |
| 145 | held item effect #83: Odd Incense, TwistedSpoon | 41 of 66 |
| 146 | held item effect #104: Zoom Lens | 42 of 66 |
| 147 | held item effect #5: Cheri Berry | 42 of 66 |
| 148 | battle format: gauntlet Oreburgh Mine (position 2; rules not recorded) | 43 of 66 |
| 149 | battle format: gauntlet Eterna Gym gym trainers (position 2; rules not recorded) | 43 of 66 |
| 150 | ability: Flower Gift | 44 of 66 |
| 151 | battle format: gauntlet Eterna Gym gym trainers (position 1; rules not recorded) | 44 of 66 |
| 152 | ability: Leaf Guard | 45 of 66 |
| 153 | ability: Wonder Guard | 45 of 66 |
| 154 | AI: flag set 0xaf | 46 of 66 |
| 155 | ability: Shield Dust | 46 of 66 |
| 156 | AI: flag set 0x1 | 47 of 66 |
| 157 | ability: Quick Feet | 47 of 66 |
| 158 | AI: flag set 0x5 | 48 of 66 |
| 159 | ability: Pressure | 48 of 66 |
| 160 | AI: flag set 0x597 | 49 of 66 |
| 161 | ability: Drizzle | 49 of 66 |
| 162 | AI: flag set 0x587 | 50 of 66 |
| 163 | move effect #182: Brave Bird, Close Combat, Take Down | 50 of 66 |
| 164 | ROM change: battle/skill/be_seq.narc#182 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 50 of 66 |
| 165 | move effect #189: Endeavor | 50 of 66 |
| 166 | held item effect #15: Wiki Berry | 51 of 66 |
| 167 | move effect #89: Counter | 51 of 66 |
| 168 | move effect #79: Substitute | 51 of 66 |
| 169 | held item effect #30: Charti Berry | 51 of 66 |
| 170 | AI: flag set 0xd | 52 of 66 |
| 171 | move effect #48: Draco Meteor, Giga Impact, Hydro Cannon, Hyper Beam, Outrage, Superpower | 52 of 66 |
| 172 | ROM change: battle/skill/be_seq.narc#48 is not modelled and this fight reaches it (docs/kaizo/rom_diff.md) | 52 of 66 |
| 173 | AI: flag set 0x97 | 53 of 66 |
| 174 | move effect #230: Payback, Revenge | 53 of 66 |
| 175 | held item effect #91: Thick Club | 53 of 66 |
| 176 | held item effect #109: Black Sludge | 54 of 66 |
| 177 | move effect #180: Assist | 54 of 66 |
| 178 | move effect #151: SolarBeam | 54 of 66 |
| 179 | held item effect #29: Tanga Berry | 55 of 66 |
| 180 | move effect #176: Helping Hand | 55 of 66 |
| 181 | held item effect #39: Petaya Berry | 55 of 66 |
| 182 | ability: Synchronize | 56 of 66 |
| 183 | move effect #148: Doom Desire, Future Sight | 56 of 66 |
| 184 | held item effect #72: Soft Sand | 56 of 66 |
| 185 | held item effect #21: Wacan Berry | 57 of 66 |
| 186 | move effect #124: Safeguard | 57 of 66 |
| 187 | battle terrain: the battle terrain varies over its maps (TERRAIN_PLAIN, TERRAIN_GRASS, TERRAIN_BUILDING), and Secret Power would read it | 57 of 66 |
| 188 | AI: flag set 0xc7 | 58 of 66 |
| 189 | held item effect #75: BlackGlasses | 58 of 66 |
| 190 | held item effect #26: Shuca Berry | 58 of 66 |
| 191 | ability: Water Absorb | 59 of 66 |
| 192 | held item effect #57: SilverPowder | 59 of 66 |
| 193 | battle format: gauntlet Oreburgh Mine (position 1; rules not recorded) | 59 of 66 |
| 194 | ability: Honey Gather | 60 of 66 |
| 195 | held item effect #23: Yache Berry | 60 of 66 |
| 196 | ability: Sand Stream | 60 of 66 |
| 197 | AI: flag set 0xef | 61 of 66 |
| 198 | held item effect #141: Iron Plate | 61 of 66 |
| 199 | ability: Magnet Pull | 61 of 66 |
| 200 | AI: flag set 0x117 | 62 of 66 |
| 201 | move effect #90: Encore | 62 of 66 |
| 202 | move effect #108: Minimize | 62 of 66 |
| 203 | battle format: gauntlet Route 204 North B2B (position 1; rules not recorded) | 62 of 66 |
| 204 | ability: Shadow Tag | 63 of 66 |
| 205 | move effect #38: Fissure, Guillotine, Horn Drill | 63 of 66 |
| 206 | move effect #118: Swagger | 63 of 66 |
| 207 | held item effect #88: Shell Bell | 63 of 66 |
| 208 | held item effect #116: Sticky Barb | 64 of 66 |
| 209 | move effect #170: Focus Punch | 64 of 66 |
| 210 | held item effect #28: Payapa Berry | 64 of 66 |
| 211 | battle format: gauntlet Route 204 North (position 1; rules not recorded) | 64 of 66 |
| 212 | ability: Poison Heal | 65 of 66 |
| 213 | held item effect #89: Lucky Punch | 65 of 66 |
| 214 | held item effect #66: Lucky Egg | 65 of 66 |
| 215 | held item effect #10: Leppa Berry | 65 of 66 |
| 216 | ability: Battle Armor | 65 of 66 |
| 217 | AI: flag set 0x58b | 66 of 66 |
