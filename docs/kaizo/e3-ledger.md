# E3 ledger: measured remaining engine work

Branch `claude/engine-e3-remeasure-632cbe`, worktree `strange-feistel-a35c9f`. E3 owns
`tools/kaizo/backlog.py`; touches only `tools/`, `docs/kaizo/` and the backlog tool's tests
(`tests/tools/test_backlog.py`). Reports go to the managing session only.

## RESUME

- Batch 1 (whole-game opponent numbers): done, committed; report sent.
- Batch 2 (player side): option A done (go from the managing session, 2026-11-14): the
  "Your side (upper bound)" and "AI partners" sections of the backlog. Option B is PARKED
  (see below); do not start it without a new go.
- Batch 4 (every sheet battle in a split): done 2026-11-14, merged into main (5612764).
- Batch 5 (merge-check follow-ups): done 2026-11-14, merged (37711e1).
- Batch 6 (join safety, tests, wording): done 2026-11-14, see below. Stop after the report.
- Batch 3 (prep starting HP and status): report sent 2026-11-14 (read-only, nothing built). Engine and line --party
  support any HP and status, sleep turns included; prep picks HP regions itself, prestatus is
  poison, burn or paralysis only, drops the save's HP and status (leveling.py), and cannot
  pin a chosen state; line [[team]] has no hp or status keys.
- Regenerate the backlog: `uv run python tools/kaizo/backlog.py docs/kaizo/mechanics_backlog.md`
  (about 4 s). Test: `uv run pytest tests/tools/test_backlog.py -q`.
- This worktree's `data/` is current (trainers.json with terrain, 2026-11-13). The main
  checkout's `data/intermediate_kaizo/`, `data/gen4_platinum_kaizo/` and
  `data/intermediate/trainers.json` are stale (2026-11-07/08, no terrain fields).

## Batch 1 facts

- Battles: union-find over split rows and their `opponent_2`. 420 split rows, 359 battles
  (258 singles, 34 doubles, 32 two-trainer, 35 multi; 7 multi with no known partner team).
- Two second opponents have no split of their own (446 Halimede, 505 Galatea, Byron multis);
  they are now checked with their battle (Byron needs 266 groups, was 263).
- Four Gardenia multi rows have no `opponent_2`: one battle each.
- Whole game: 509 distinct missing groups (AI flags 5, AI move effects 164, move effects 114,
  held items 97, abilities 63, battle conditions 53, other 13). Of the 53 battle conditions,
  33 are unrecorded gauntlet positions (data calls, not engine code).
- New groups per split: 136, 145, 87, 32, 19, 18, 15, 53, 0, 4.
- AI move-effect groups are counted for every row of every split (needed per split:
  53, 80, 92, 98, 78, 91, 85, 128, 65, 79).
- Not counted: the player's side, and AI partner teams in multi battles.

## Batch 2 facts (option A)

- Player side, upper bound: 333 groups, 268 shared with the opponent union, 65 player-only
  (AI reads of your moves/abilities 13, move effects 22, held items 25, abilities 5); 63 of
  them from Roark on, 2 (Judgment) from Galactic (cap 100).
- Hold effects with no battle effect (not counted): Amulet Coin/Luck Incense, Cleanse Tag/
  Pure Incense, Everstone and the seven trade-evolution items; `NO_BATTLE_EFFECT` in
  backlog.py, re-checked against the decomp on every run. Kept as battle-relevant: Exp.
  Share, Lucky Egg (experience is modelled), Soothe Bell (battle_system.c:746), Smoke Ball
  (Battler_IsTrappedMsg, read by Teleport, battle_script.c:6150).
- AI partners: after the partner fix, 19 rows, 122 missing groups, none outside the opponent
  union.

## Batch 2 add-on: partner teams (2026-11-14)

- Partner teams were never missing: every partner is a ROM trainer row (Cheryl 608, Riley
  609, Marley 610, Mira 612/864, Lucas/Dawn 613-626 and 800/801, Barry 607/619/620, the
  Coronet Highlands three 861/863/865). What was missing is the link: the ROM script reader
  leaves `partner` null when the partner variable is not set right before the battle
  (`battle_structures`' unknown_partner), and the cross-check compared only
  `partner_present`, so PKCalc's partner (flags.js `battleType.tag`, already in
  `pkcalc.json` `formats.tag`) was never used.
- Fix: `tools/kaizo/resolve.py` `fill_pk_partners` fills a null partner from PKCalc's tag
  triple, expanding `{RIVAL2_NAME}` / starter placeholders into variants (gender from the
  ROM-evaluated rival-2 variants, starter from each partner row). 23 rows filled (14 split
  rows = the 7 battles, plus 9 rows with no split). No ROM-evaluated partner disagrees with
  PKCalc. Test: tests/kaizo/test_pkcalc_partner.py. Rebuild:
  `uv run --group rom python -m tools.build_tables --game kaizo` (about 2 s).
- Found, not fixed (a recorded non-blocking ruling): 60 sheet headers map to no trainer id,
  so battles such as Spear Pillar Mars #3 + Jupiter #3 (528/407, beside Barry) and the
  Coronet Highlands multis (562/572, 581/606, 582/583) and Iron Island 835/836 have no split
  and are not in the 359 battles.

## Parked: option B, real per-split player availability

Not started; the managing session will raise it with the user (matters for prep and
training). Scope: which species, TMs, held items and tutor moves the player can really have
by each split, replacing option A's upper bound.
- Wild encounters from the patched ROM's encounter archive (tools/rom), cross-checked
  against the spreadsheet's Encounters / RAW Field Enc / RAW Water Enc sheets.
- Gifts, trades, statics, honey trees, Great Marsh: the sheet's Non-Wild Encounters,
  Universal Encounters and Great Marsh Encounters (free text: hand-curated, cited).
- Field items, hidden items, marts, TM locations and shard tutors (tutor_moves.json has
  location stems) from ROM field scripts and mart data.
- Area -> split from the split sheets' section names (first split an area appears in).
- Encounter limits and Nuzlocke rules are the user's run rules: not in any data; ask.

## Batch 4: the 60 unmapped sheet headers (2026-11-14)

- 49 mapped by `[trainer_ids]` in data/kaizo/overrides.toml, each with the ROM row, map
  placement and PKCalc label in its comment (name + same species order; sheet levels are
  older). 11 settled in the new `[sheet_unmapped]` table: 7 `ally` (Dawn/Lucas tag-partner
  headers), 3 `duplicate` (Route 212 Rangers 96/95/94, listed in Byron and Galactic; counted
  in Galactic), 1 `user_call` (Maid Yoki, Gardenia Split:111). `check_sheet_unmapped` now
  refuses any unsettled unmapped header (resolve and `overrides check`).
- Backlog: tag partners (every partner variant id; each checked against its sheet header's
  `tag_partner` tag) are no longer counted as battles; 11 had been counted as singles.
- Battles 359 -> 387 (singles 258 -> 275, doubles 34 -> 37, two-trainer 32, multi 35 -> 43);
  39 new battles, 11 ally rows out. Union 509 -> 521: 15 groups new to the union, 3 left
  (needed only by allies: Marvel Scale, hold effects #17, #66; still on the partners line).
- New user calls: Maid Yoki; Route 212 Rangers' split; Iron Island Triton #2 / Neptune #2
  weather (sheet and PKCalc rain, ROM none; report "area weather 835").

## Batch 5: merge-check follow-ups (2026-11-14)

- Ally reasons now name the rival's Pokemon and the right player starter; the test
  `test_ally_reasons_name_the_partner_rows_starter` derives both from the ROM partner rows.
- Shared script files: `trainer_maps` (tools/rom/readers/maps.py) places a battle in a file
  several headers share only on headers whose bg/object/coord events or init scripts run
  the entry that reaches it (`map_script_ids`, `init_script_ids`; script_manager.c:243-246,
  :412-416, :738-806). Kaizo's Coronet Highlands tag battles live in scr_seq 401
  (scripts_empty, 136 headers): entry 9 -> MT_CORONET_OUTSIDE_SOUTH (Cheryl, 582/583),
  entry 10 -> MT_CORONET_4F_ROOMS_1_AND_2 + header 250 (Riley, 562/572), entry 11 ->
  MT_CORONET_5F (Marley, 581/606). They also sit on STARK_MOUNTAIN_ROOM_2 (a second,
  postgame placement the noise hid). NOISY_MAP_JOIN removed. Side effects: 318 Fantina's
  start condition is now none (header 89 drops out), 676 Corbin (rematch of 583) hail; both
  removed from data/kaizo/start_conditions.json.
- The four Gardenia Eterna Forest rows (202, 203, 411, 413): no source pairs them (ROM sight
  pairing only runs on a calc/sheet claim; calc singles; sheet solo spots). They stay four
  battles. Open rule question for the managing session: encounter.c:730-740 gives a lone
  enemy beside a walking partner BATTLE_TYPE_TRAINER (not a tag battle), while K2a Ruling 3
  makes every enemy on a partner map `multi`.
- Totals: 387 battles (275 singles, 37 doubles, 32 two-trainer, 43 multi); union 520.

## Batch 6: join safety and wording (2026-11-14)

- `join_trainer_maps` / `header_battles` (tools/rom/readers/maps.py): a shared-file header
  also runs the local ids game code starts directly (`C_STARTED_SCRIPTS` 2-8, cites
  verified: encounter.c:637, start_menu.c:1457, overlay005/field_control.c:450,
  overlay009/ov9_02249960.c:99,106,5411,7732-7756,8765 with rows :11050-12709), and falls
  back to the whole file for a clone object (script 0xFFFF), an id past the table, or a run
  entry that reaches ScrCmd_Unused_013 (scrcmd.c:960-971). Kaizo placements unchanged.
  The docstring lists what is and is not read. Synthetic tests in
  tests/tools/rom/test_shared_scripts.py cover every branch.
- Unannounced in batch 5, same rule: 342 Harrison and 345 Christine lost Kaizo header 609,
  which shares the Trainers' School script file (31) but runs none of its entries; they
  stay on TRAINERS_SCHOOL (345 also ETERNA_CITY_POKECENTER_1F).
- 676 Corbin: inherited through the vanilla rematch table from 583; no object carries 583's
  trainer script (checked: no object script 3582/5582), and the Vs. Seeker rematches only
  trainer objects (vs_seeker.c:500-540, :692-694), so probably unreachable. 583 Titania's
  start_conditions entry left (one placement now). Stale 676/318 text fixed in
  overrides.toml, overrides.py and start_conditions.json.

## Batch 7: CallCommonScript (2026-11-14)

- `starts_local_script`: a run entry reaching ScrCmd_CallCommonScript with a local id
  (1-1999; scrcmd.c:973-985 has no range check) now falls back to the whole file, as
  Unused_013 does. Tests pin it, the common-id case, and Unused_013 reached only through a
  C-started entry. The ov9_02249960.c:5411 comment now says it assigns a script.
- Survey: no Kaizo (605 calls) or vanilla (573) script calls CallCommonScript with a local
  id, and none uses Unused_013. Real extraction: rom_trainer_maps.json byte-identical.
- E3 is finished; the worktree stays as it is.

---

# E3 mechanics ledger (engine build rounds)

Three parts: the current state, the orders, the rulings. Everything above this line is the
earlier E3 stream's (backlog measurement). Rule sheets: `_scratch/e3/rules/` of the managing
worktree. Conventions: section 5 of docs/doubles-state-design.md. Process: docs/process.md.

## 1. Current state

- Branch `claude/e3-batch1` (from E2's battler re-index 0eceaf4). Round 1 of the first batch:
  #71 Mist Ball, #85 Thief (Kaizo Splash script), #86 Disable AI, #153 Teleport, #218 Hammer
  Arm. Rulings E3-1-1 .. E3-1-6 below. Tables rebuilt (vanilla and Kaizo) after the effect map
  change; `src/nuzline/ai/read_bounds.py` regenerated (`uv run python -m tools.ai_read_bounds`).
- Not run locally: the four guards and the targeted merge run (the test pod's job).

## 2. Orders (Owed)

Each is open; close it with its ruling.

1. Kaizo Magic Coat on Disable (E3-1-3): Disable carries CAN_MAGIC_COAT in Kaizo
   (data/intermediate_kaizo/moves.json). Magic Coat is unmodelled engine-wide (batch 2), so a
   Disable aimed at a Magic Coat user is not reflected today. Model it with Magic Coat. Torment
   (E3-2-1) and Worry Seed (E3-2-2) carry CAN_MAGIC_COAT as well (Torment in Kaizo only, Worry
   Seed in both games).
2. Mirror Move's AI table reads the player's last move by id (script.s:1959, MOVE_THIEF) and
   Me First's `sCannotMeFirstMoves` compares by effect (lib.c:7233-7251), so in Kaizo every
   effect-85 move is uncopyable (sheet e241): both belong to their own groups (Mirror Move
   label, #241 Me First).
3. Tag Strategy in singles (E3-2-4): the engine's AI has no Tag Strategy flag (bit 7,
   script.s:26), but a singles trainer whose header sets it runs TagStrategy_Main
   (ai.c:249-271, :297-309). Its Trick Room entry (script.s:6979-7025) reads the empty
   partner slots (curHP 0, maxHP 0), and the unsigned 0/0 returns the dividend (0), so
   script.s:6981 always jumps to ScoreMinus30: Trick Room gets -30 every turn, on top of Basic
   and Expert (sheet e259, "Tag Strategy in singles"). The same 0/0 read shapes the other Tag
   Strategy entries (:6663, :6675, :6836, :6847, :7057-7071, :7156). Model the flag with
   empty slots as zeroed battlers when the Tag Strategy group is built; not modelled here.
4. Tag Strategy's Skill Swap entry (E3-2-5): `IfMoveEqualTo MOVE_SKILL_SWAP` (script.s:6734) ->
   TagStrategy_SkillSwap (:7195-7213: the AI's own Truant, Slow Start, Stall or Klutz +5; else
   the player's Shadow Tag, Pure Power, Huge Power, Mold Breaker, Solid Rock, Filter or Flower
   Gift +2) runs in singles for a trainer whose header sets the flag (ai.c:249-271), on top of
   Expert_ChangeUserAbility. Not modelled, with the rest of the Tag Strategy flag (Owed 3).

## 3. Rulings

- E3-1-1 Mist Ball (#71) is fully modelled; no mechanic change. `tools/effect_map.py` maps
  effect 71 to SECONDARY_STAT_FOE (Sp. Atk, -1) rolled at the move's chance (50; Kaizo
  Mystical Fire 100): effect_script_0071.s:5-7, battle_lib.c:1570-1586, bs.c:2804-2976 (Mist
  :2876-2882, Shield Dust :2918-2920, floor :2969-2973). The AI has no Basic or Expert entry
  (script.s), so 71 joins `KAIZO_TRANSCRIBED` as a label-less effect (the generic damage path
  is already transcribed); the backlog row closes. Tests: tests/engine/test_e3_batch1.py
  (mapping, proc, Shield Dust, Mist, both games), tests/ai/test_kaizo_policy.py (refusal lifted,
  Expert adds 0).
- E3-1-2 Splash / Kaizo Thief (#85): new `EffectId.DO_NOTHING` (effect 85, `effect_map`). The
  script only sets MOVE_STATUS_SPLASH (effect_script_0085.s:5): no CalcCrit, no CalcDamage, so
  `battleCtx->damage` stays 0 (battle_lib.c:1974-1976) and UpdateHP does nothing (bcp.c:3347:
  no HP change, no Substitute hit, no Focus Sash, no contact ability, no Life Orb or Shell Bell).
  Accuracy, Protect and the type chart run as for any move with power (bcp.c:2350-2370); the
  move still counts as used (bcp.c:3212, :3945-3949: PP, `last_move`). Engine: the effect is in
  `ids.UNROLLED_EFFECTS` (no crit message) and `damage.fixed_damage` returns 0 (reference:
  `_fixed_damage`). Classified INPUT_NEUTRAL and NON_TABLE_DAMAGE; `_NO_ACC_EVA_WRITE` in the
  accuracy-class test. AI: nothing to transcribe (no Basic or Expert entry, effect 85 already in
  `ENCORE_ENCOURAGED_EFFECTS`, script.s:3220; Copycat's table has 168). The AI keeps the 40-power
  damage estimate: effect 85 is not in sNoDamageCalcMoveEffects (trainer_ai.c:31-45), so
  `commands.is_damaging` is true and `damage.ai_damage_for_roll`, which does not go through
  `fixed_damage`, gives a positive estimate while the engine deals 0. 85 joins
  `KAIZO_TRANSCRIBED`. Vanilla Splash (power 0) takes the same effect.
- E3-1-3 Disable (#86): the mechanic was modelled (effects._disable, step._disable_check; the
  decomp cites are in the sheet and tests/engine/test_disable.py). New AI blocks.
  Basic_CheckCannotDisable (script.s:650-653, entry :184): the player under Disable -8, where
  CHECK_DISABLE reads `disabledTurns != 0` (trainer_ai.c:1833-1836), so on the last disabled
  turn (counter 0, move still set) the AI does not see it. Expert_Disable (script.s:3075-3093,
  entry :1687): slower than the player (the SLOWER compare, which reads no tie) nothing; else
  the power of the player's `movePrevByBattler` (ai.c:1068-1076, :2526-2530; none reads 0): above
  0 +1, else -1 unless the 100 roll at :3089 jumps. One random read on the power-0 path
  (`outcome.SITE_N[3089] = 100`); read bounds regenerated. Kaizo's Thief has power 40, so a
  player whose last move was Thief reads as damaging (+1). Kaizo Magic Coat: Owed 1.
- E3-1-4 Teleport (#153): new `EffectId.FLEE` (effect 153). effect_script_0153.s:5, :9-10 sets
  MOVE_STATUS_FAILED in a trainer battle (every Nuzline battle): `step._use_move` adds it to the
  failing scripts (reference `_script_fails`), PP is spent and `last_move` set (bcp.c:3178-3183,
  :3212, :3945-3949). The wild-only flee (:6, subscript_teleport.s) is not modelled. Accuracy 0 and
  RANGE_USER: no roll (bcp.c:2901-2903). Kaizo priority +1 comes from the move row (moves.json);
  the test checks the order in both games. AI: Basic_ScoreMoveEffect :221 jumps straight to
  ScoreMinus10 (`policy._basic_score_minus_10`); no Expert entry, no roll.
- E3-1-5 Hammer Arm (#218): the mechanic was modelled (SECONDARY_STAT_SELF (Spe, -1), certain:
  effect_script_0218.s:5-7, battle_lib.c:1524-1529: no HP, Substitute or chance test; bs.c:2876
  skips the user's guards; floor -6 :2924-2937, :2969-2973). New test of the substitute, faint and
  floor cases. AI: Expert_HammerArm (script.s:4849-4867, entry :1762): immune, half or quarter
  damage -1; else slower than the player (SLOWER compare, no tie) +1; else 0. No roll.
- E3-1-6 AI plumbing: `commands.BattleEffect` gains LOWER_SP_ATK_HIT (71), DO_NOTHING (85),
  DISABLE (86), FLEE_FROM_WILD_BATTLE (153), SPEED_DOWN_HIT (218); `outcome.KAIZO_TRANSCRIBED`
  gains all five; `KAIZO_EXPERT_LABELS` / `KAIZO_EXPERT_TRANSCRIBED` gain 86 and 218. Coverage:
  `coverage.IMPLEMENTED_EFFECTS` covers every EffectId, so the two new ids are supported with no
  list edit; `tests/ai/test_kaizo_policy.py` `test_batch13_exact_equals_loose` sweeps
  Disable, Hammer Arm, Thief and Mist Ball against the loose oracle.
- E3-2-1 Torment (#165): new `EffectId.TORMENT` (effect 165) and the per-battler `[B]` field
  `torment` (int8; statusVolatile bit VOLATILE_CONDITION_TORMENT, condition.h:63; STATE.md;
  `state.VOLATILE_FIELDS`, so a switch or faint clears it and Baton Pass drops it:
  battle_lib.c:2071, :2075, condition.h:72-76). `effects._torment`: effect_script_0165.s:5 runs
  subscript_torment_start.s, which fails on a target that has the bit (:5) and on MISSED |
  SEMI_INVULNERABLE (:6, `h.effective`); it has no Substitute and no ability test (goes through a
  Substitute) and no count (nothing ends it while the target stays). The block is selection-time
  only: CHECK_INVALID_TORMENTED (battle_lib.c:2275-2279) flags the slot holding
  `movePrevByBattler` (`last_move`: 0 after a move that did not succeed, Struggle's id in no
  slot, so nothing is blocked then). It is read by the player's menu (:2331-2335), the AI's
  `CHECK_INVALID_ALL` (ai.c:234-237), the Struggle test (bcp.c:433) and Sleep Talk's slot list
  (:4927): `actions.usable_moves` and `order.sleep_talk_slots` (reference `_usable`,
  `_invoked_move`). There is no execution-time check (bcp.c:2543-2560), so a move chosen before
  the Torment still runs. A lone blocked move leaves no usable slot (Struggle). Classified
  INPUT_NEUTRAL and NON_TABLE_DAMAGE; `_NO_ACC_EVA_WRITE`; the observation adds `torment` to the
  non-zero side fields. AI: `commands.BattleEffect.TORMENT`, `VolatileCondition.TORMENT` (1 << 31,
  read from `torment`), `policy._basic_check_torment` (Basic :229 -> script.s:874-877: the player
  already tormented, -10); no Expert entry; 165 joins `KAIZO_TRANSCRIBED` (its SetupFirstTurn
  :6480 and Harrassment :8055 table entries, and the Encore table :3247, were already in
  `commands`). Kaizo's CAN_MAGIC_COAT flag on Torment: Owed 1. Tests: tests/engine/test_e3_batch1.py,
  tests/ai/test_kaizo_policy.py `test_torment_basic`.
- E3-2-2 Worry Seed (#247): new `EffectId.WORRY_SEED` (effect 247). `effects._worry_seed`
  (effect_script_0247.s:5; subscript_give_target_insomnia.s:5-11): fails on a miss or
  semi-invulnerable target, a Substitute, a raw Truant or Multitype ability, a held Griseous Orb;
  else `ability[party, mon]` = Insomnia (no new state: the existing per-battler written ability,
  restored by `step._restore_ability`, battle_lib.c:124) and the AI tags it (`know_ability`,
  battle_script.c:3073-3075). `damage.decomp_abilities` (and the reference's) now read the
  written ability whenever `ability` differs from the personal row's, not only for Trace. The
  sleeper wakes at the move's end by the existing MoveEnd recovery (bcp.c:3965-3967,
  battle_lib.c:4404-4410). Hit chance: `outcome._threshold_grid` field variant (the side's own
  ability becomes Insomnia, Klutz item live); crit: `CRIT_ODDS_EFFECTS` write (Super Luck, Klutz
  item; Battle/Shell Armor is another battler's hit, none follows in singles); the AI-roll
  `loose` flag; enumerate: INPUT_CLAUSE effect (the worry-seed clause: foe_move, end-speed tie).
  AI: Basic_CheckWorrySeed (script.s:1428-1443), Expert_WorrySeed (:5973-5993; rolls :5984, :5988);
  read bounds regenerated. Kaizo and vanilla CAN_MAGIC_COAT: Owed 1. Tests:
  tests/engine/test_e3_batch1.py, tests/ai/test_kaizo_policy.py.
- E3-2-3 Me First (#241): new `EffectId.ME_FIRST` (effect 241) and `order.CALL_ME_FIRST`, an
  invoker ordered like Copycat (no draw: call value 0 = the target has not acted, 1 = it has, the
  call fails). `order.me_first_copy` is `BtlCmd_TryMeFirst` (battle_script.c:7286-7312): the
  target's encored move when it still holds it in its slot, else its selected move
  (`Battler_SelectedMove`, battle_lib.c:2858-2868: none when it chose no FIGHT); it fails for a
  struggling target, a move whose battle effect (`tables.move_ai_effect`) equals that of Counter,
  Mirror Coat, Thief, Covet, Focus Punch or Chatter (`ids.ME_FIRST_BANNED_MOVES`, by move id per
  game: Move_MeFirstCanCopy, battle_lib.c:7242-7252; Kaizo's Covet slot has effect 0, so every
  plain hit and Kaizo Thief are refused), and for a move without power (:7300). `step._use_move`
  reads the target's action from the `pursuit` triple `_act` already passes, and the call fails
  once `target_moved`. New per-battler `[B]` state `me_first` (int8) and `me_first_turn` (int16):
  `MoveEffectsData.meFirst` / `meFirstTurnNumber` (zeroed with the volatiles, lib.c:2096-2100; not
  Baton-passed); `meFirstTurnOrder` is `turn` + 1 (lib.c:2028, bcp.c:1853). The x1.5 is
  `damage.me_first_boost` after the crit, Life Orb and Metronome item and before the variance
  (battle_script.c:1346-1364) on every CalcDamage of that battler while `me_first_active`; the
  counter writes (number - 1 when equal to the order, clear at 2 or more above, :1356-1363) run in
  `effects.play_hit` where a hit calculated damage (reference `_me_first_after_calc`). Enumeration:
  a Me First hit is never a damage class (the boost is not in the turn-start table); its calls
  are Copycat's two orders. Me First's own defender is its target (RANGE_SINGLE_TARGET_ME_FIRST),
  so it needs a living foe, unlike the self-targeting invokers. AI: Expert_MeFirst
  (script.s:5519-5550; rolls :5532, :5538, :5542), no Basic entry, Risky's table already lists
  241; read bounds regenerated. Tests: tests/engine/test_e3_batch1.py,
  tests/ai/test_kaizo_policy.py `test_me_first_expert`.
- E3-2-4 Trick Room (#259): new `EffectId.TRICK_ROOM` (effect 259) and `ids.TRICK_ROOM_TURNS` = 5
  (condition.h:145). `effects._trick_room` toggles the existing `trick_room` field scalar:
  `FIELD_CONDITION_TRICK_ROOM_INIT` is set when no Trick Room bit is up and cleared when one is
  (effect_script_0259.s:5-6); the move reads like Gravity's (a failed move leaves it alone). The
  countdown is its own block after Perish Song's (`step._trick_room_check`, reference
  `_trick_room_check`): SIDE_COND_CHECK_STATE_TRICK_ROOM (battle_controller_player.c:1822-1830)
  drops the counter by 1 and, at 0, runs `subscript_trick_room_end`: vanilla's only prints (up
  through turn T + 4); Kaizo's (sub_seq 251, byte for byte the move's own script) finds the bits
  clear and sets them again: it never ends. New table scalar `Tables.trick_room_rearm` (0 vanilla,
  1 Kaizo, `tools/build_tables.py`, modelled on `gravity_rearm`), manifest version 34; rom_diff
  member 251 is now `modelled`. Area start (map weather 18,
  subscript_overworld_trick_room.s:14): `coverage.start_trick_room` gives 5, threaded as
  `make_state(trick_room=...)` through `prep` and `cli`; `coverage` no longer refuses it. The
  mid-turn re-sort (`trickRoomActivated`, bcp.c:3990-3994) changes no action in singles (the
  foe has acted or acts next; the user is marked done), so the effect is an INPUT_NEUTRAL, no-
  damage enumeration class; a doubles partner's re-sort is the doubles design's (section 5,
  "has acted" per battler). AI: Basic_CheckTrickRoom (script.s:1501-1506) and Expert_TrickRoom
  (:6190-6214; roll :6210); read bounds regenerated. Tests: tests/engine/test_e3_batch1.py,
  tests/ai/test_kaizo_policy.py, tests/engine/test_coverage.py `test_area_trick_room_is_modelled`.
- E3-2-5 Skill Swap (#191): new `EffectId.SKILL_SWAP` (effect 191; no manifest bump: no array or
  key changed). `effects._skill_swap` (effect_script_0191.s:5; subscript_exchange_abilities.s):
  fails on a miss or semi-invulnerable defender (:5), a raw Wonder Guard (:6-7) or Multitype
  (:8-9) ability of either (`decomp_abilities`, so a written ability counts), a held Griseous Orb
  of either (:10-11, raw `state.item`), both abilities ABILITY_NONE (:12-13; the user's alone is
  fine); no Substitute test (it goes through one). Else `truant` of both = `turn & 1` (:17-20),
  `ability` swapped (:21-24: no new field, the per-battler written ability restored by
  `step._restore_ability`, battle_lib.c:124; the Gastro Acid suppression stays with its
  battler), both tagged for the AI (`know_ability`, battle_script.c:3402-3404). Slow Start
  (:25-38): unsupported engine-wide (no ability id; coverage refuses a holder), and a battler
  only receives it from one that has it, so no `slow_start_turn` / announced / finished field is
  added. New per-battler `[B]` fields `trace_announced`, `weather_announced`,
  `intimidate_announced`, `download_announced` (BattleMon, battle_lib.c:103-106: in
  `VOLATILE_FIELDS`, cleared by Transform too, battle_script.c:5350-5362; set by the hook that
  acts on a living holder: lib.c:3726-3733, :3748-3786, :3799-3803, :3822-3836); `hooks.
  move_end_switch_in` is MoveEnd's TriggerEffectOnSwitch (bcp.c:3973) over every battler, called
  from `step._move_end` after the two RecoverStatusByAbility calls (bcp.c:3965-3967) while the
  battle goes on: trace, weather, Intimidate, Download for a battler whose flag is clear, a
  no-op otherwise. The entry hooks (`switch_in_effects`, `show_battle_mon`) keep their explicit
  conditions and read `trace_announced` for a Trace holder (a Trace that traced and got Trace
  back does not copy twice, :3726). Rulings: the pass runs battlers in the engine's order
  (player first), as `show_battle_mon`: two battlers gaining an ability of the same state in one
  swap cannot happen (a swap gives each the other's ability, whose flag is set unless the
  battler never held it); Anticipation, Forewarn, Frisk and Slow Start have no ability id (no
  flag; their holders are refused), Pressure and Mold Breaker only print (their AI reveal is
  the swap's own tag). A test that writes an entry ability into an ACTIVE battler after
  `make_state` must set its flag (`tests/engine/test_e1_trial.py` does). Hit chance:
  `outcome._threshold_grid`'s field variant (foe holds the side's ability, side the foe's; a weather
  ability one gained acts at the move end: the one whose flag is clear; Klutz item live for a
  side that loses it), `future_thresholds`' `loose` flag; crit: `CRIT_ODDS_EFFECTS` write (every
  battler gets its foe's ability: Super Luck, Klutz item) and `armored` counts a foe's Skill Swap
  like Transform; enumerate: INPUT_CLAUSE effect (skill-swap clause), NON_TABLE_DAMAGE, end-speed
  tie and Shed Skin handover. AI: Expert_ChangeUserAbility (script.s:4415-4433, entry :1743)
  with `commands.CHANGE_USER_ABILITY_DESIRABLE` (:4435-4461) and the 50 roll at :4429; a
  defender ability in neither table falls through to -1 (the comment's "+0" is wrong, sheet
  e191); no Basic entry; the Encore table entries (:3229, :3256) were already in `commands`;
  Tag Strategy's entry: Owed 4; read bounds regenerated; backlog regenerated (the two #191 rows
  close). Tests: tests/engine/test_e3_batch1.py `test_skill_swap_*`,
  `test_the_announced_flags_*`, tests/ai/test_kaizo_policy.py `test_change_user_ability_*`.
