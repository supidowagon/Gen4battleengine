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
   (E3-2-1), Worry Seed (E3-2-2) and Gastro Acid (E3-2-6) carry CAN_MAGIC_COAT as well (Torment
   in Kaizo only; Worry Seed and Gastro Acid in both games: res/moves/gastro_acid/data.json,
   data/intermediate_kaizo/moves.json).
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
5. Tag Strategy's Gastro Acid entry (E3-2-6): `IfMoveEqualTo MOVE_GASTRO_ACID` (script.s:7454) ->
   TagStrategy_PartnerGastroAcid (:7624-7645: the AI's partner already suppressed -30; its
   partner's Truant or Slow Start +5) is the doubles partner-targeting path: it reads
   AI_BATTLER_ATTACKER_PARTNER, an empty slot in singles, and is not modelled with the rest of
   the Tag Strategy flag (Owed 3).
6. Doubles: `outcome._threshold_grid`'s Gastro Acid variant and `outcome.armored` assume the
   singles pairing (the foe's Gastro Acid reaches the side's own ability only; no ally Gastro Acid
   reaches a defender's Battle Armor / Shell Armor / Sand Veil before a partner's hit). The
   doubles milestone adds those pairs (docs/doubles-state-design.md section 5).
7. `effects.MOVE_EFFECT_MASK_FIELDS` lacks the Embargo bit (`embargo`, MOVE_EFFECT_EMBARGO,
   moves.h:34), which `ingrain_anchors` (subscript_force_target_to_switch_or_flee.s:7 compares
   the whole mask) should count like Gastro Acid's; found while adding the Gastro Acid bit, not
   changed here (the same one-line change in the tuple, the reference's `_ingrain_anchors` and
   a Roar test).
8. Skill Swap's Slow Start lines (E3-2-5): subscript_exchange_abilities.s:25-38 reset
   `slowStartTurnNumber` to totalTurns + 1 and clear `slowStartAnnounced` / `slowStartFinished`
   for a battler that receives Slow Start (ability 112). Not modelled: Slow Start is an
   unsupported ability, so no battler holds it. Model these lines with the Slow Start group.

Checked, no change (batch 1 gate round): Trace retries at every move end (battle_lib.c:3718-3737,
`hooks.move_end_switch_in`), so with two Trace holders a Worry Seed or Skill Swap can make one
Trace succeed mid-turn. The enumeration covers it: the side's later hit is unclassified by the
foe's Worry Seed / Skill Swap / Gastro Acid as a whole (`enumerate` worry-seed, skill-swap and
gastro-acid clauses: `foe_effect`, not the ability it writes), and the turn end keeps the speed tie
and Shed Skin for any Trace holder on the field (`tracer`) and for any Skill Swap (`swap_used`).

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
  player whose last move was Thief reads as damaging (+1). Kaizo Magic Coat: closed in E3-B2-6.
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
  `commands`). Kaizo's CAN_MAGIC_COAT flag on Torment: closed in E3-B2-6. Tests: tests/engine/test_e3_batch1.py,
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
  read bounds regenerated. Kaizo and vanilla CAN_MAGIC_COAT: closed in E3-B2-6. Tests:
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
  `effects.play_hit` where a hit calculated damage (reference `_me_first_after_calc`): hit 0 once
  the script ran (`script_calc`: acted, a target, not failed, not charging; BeforeMove loads the
  script before TryMove, bcp.c:3214-3216, so CalcDamage runs before the accuracy roll, Protect and
  the semi-invulnerable test, :3254-3270), a later hit when it lands. A cleared `me_first` leaves
  `me_first_turn` 0 (gate round). Enumeration:
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
  fails on MOVE_STATUS_MISSED | SEMI_INVULNERABLE (:5, the move status: Lock-On / Mind Reader
  or either side's No Guard return before the semi test, bcp.c:3016-3021, so the swap goes
  through; `h.landed` alone, gate round), a raw Wonder Guard (:6-7) or Multitype
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
- E3-2-6a Gastro Acid 1/2 (#239), refactor, no mechanic: one suppressed-ability reader. New
  per-battler `[B]` int8 `ability_suppressed` = MOVE_EFFECT_ABILITY_SUPPRESSED (moves.h:29),
  STATE.md `mapped`, in `VOLATILE_FIELDS` (switch-out clears the mask, battle_lib.c:2071) and
  `BATON_PASSED_FIELDS` (moves.h:46-48, :2076); reference `RefBattler.ability_suppressed`,
  `BATON_PASSED`, `_reset_battler`. Nothing sets it yet. Accessors: `state.battler_ability(state,
  b, slot=None)` (AbilityId), `damage.battler_decomp_ability(tables, state, b, slot=None)`
  (decomp ids), both through `state.ability_is_suppressed` (the bit applies to the battler's
  active slot only); NONE when the bit is set and the raw ability is not Multitype (lib.c:3088-
  3093). Multitype has no AbilityId (`UNSUPPORTED`), so the AbilityId accessor keeps a raw
  `UNSUPPORTED`; the decomp-id one tests `ABILITY_MULTITYPE` exactly. Reference twins
  `_ability(s, mon)`, `_battler_decomp_ability(tables, s, mon)` via `_suppressed(s, mon)` (the
  battler whose active Pokemon `mon` is). The Gravity / Ingrain Levitate lines (lib.c:3095-3103)
  stay where they were (`damage.ignorable`, `damage.grounded`, `effects.magnet_rise_fails`,
  `actions` BattlerIsGrounded's Gravity term; reference `_ignorable`, `_entry_hazards`,
  `_script_fails`), applied to the routed value. AI: `commands.MoveEffectMask.ABILITY_SUPPRESSED
  = 1 << 21` (IfMoveEffect); `attacker_ability` and `_defender_abilities` (LoadBattlerAbility /
  CheckBattlerAbility) give ABILITY_NONE, known, no coin, for a set bit (trainer_ai.c:1177-1178,
  :1221-1222: the bit alone, no Multitype test). With every bit 0 each accessor is the raw read,
  so singles outputs are unchanged by construction.
  Routed (Battler_Ability / IgnorableAbility / CountAbility / CheckAbility): `state.held_item`
  Klutz (lib.c:5352-5356); damage: `ignorable`, `identified` Scrappy, `trace_copy`'s holder,
  `grounded` Levitate, `hazards_loss` Magic Guard, `entry_ability` own, `script_power_mul`
  Reckless, `weather_suppressed` (CountAbility), `beat_up_damage` Sniper (:7144),
  `side_ability_count`, `_pre_random` attacker / defender / Technician / Flower Gift (:6653-
  6654), `_finish` attacker; actions `trapped` (BattlerIsGrounded, Shadow Tag, Magnet Pull, Arena
  Trap, :5500-5536); order `effective_speed` (:1213-1214), `pinch_divisor` Gluttony, `truant_loafs`
  (:3308); effects: Camouflage (script.c:8001), recoil Rock Head / Magic Guard, `pressure_holder`,
  `captivate_fails`, `rest_fails`, Heal Bell user (script.c:5038), `magnet_rise_fails`,
  `yawn_blocks`, `yawn_sleep`, Knock Off Mold Breaker and Sticky Hold, `_contact_ability`
  (TriggerAbilityOnHit and the attacker's Limber / Leaf Guard / Magic Guard), `synchronize`, the
  on-hit berry, entry Toxic Spikes, Life Orb; hooks: Download, `reveal_ability`'s trigger,
  `weather_ability`, `weather_war`, `swap_war`, `arm_unburden`, `recover_status_by_ability`,
  berry Own Tempo, Pluck Klutz (lib.c:5840), Intimidate and its blockers (script.c:2883);
  step: Steadfast, full paralysis Magic Guard, No Guard (bcp.c:3018-3019, the Future Sight hit),
  crash Magic Guard, leech seed / poison / Poison Heal / curse / bind / Bad Dreams (bcp.c:1485),
  Leftovers, the status orbs, Speed Boost / Shed Skin, `_weather_effect` (int32[B] decomp ids);
  enumerate: `_holds`, `_end_speeds` (Quick Feet, Unburden, Chlorophyll; the newcomer view clears
  the bit), the outcome-law reads of the active battlers (Guts, Cute Charm, Mold Breaker / Limber
  / Insomnia, Synchronize / Static, incoming setters); outcome: `_rate`, `stage_total`, Klutz
  holder, `armored` Mold Breaker and target, `secondary_chance` Serene Grace (lib.c:1552);
  search.playout the newcomer's Intimidate / Download; ai.switching `wonder_guard_bit`
  (lib.c:2649). Reference: every matching site of `engine.py` (103 lines).
  Left raw, each with a comment: Super Luck (`outcome.side_crit_odds`, ref `_crits`; lib.c:7122);
  Trace target choice and copy (`damage.trace_copy` foe, `entry_ability` traced value,
  `hooks.trace`, `outcome._armed_trace`, coverage `trace_copy_refusals`, ref `_trace`;
  lib.c:7754-7776, subscript_trace.s:6-7); Natural Cure on switch-out (`step._natural_cure`,
  ref `_u_turn`, Baton Pass, `_roar`, `_switch`; script.c:9151); the party ability restore
  (`step._restore_ability`, ref `_restore_ability`; lib.c:124); Worry Seed's checks and write
  (subscript_give_target_insomnia.s:7-8, :11), Skill Swap's checks and swap
  (subscript_exchange_abilities.s:6-13, :21-24), Knock Off's Multitype (subscript_knock_off.s:6),
  Embargo's Multitype (subscript_embargo_start.s:7), Transform's copy (script.c:5342-5348;
  also outcome's field variant, `armored`'s copied armor, enumerate's Transform Cute Charm /
  Wonder Guard); the AI record of a tagged ability (`effects.know_ability`, `hooks._know`,
  `reveal_ability`'s record, the Transform record settle, ref `_know_ability`; script.c:11805);
  Heal Bell's Mold Breaker in the IO message (battle_controller.c:1539, battle_io_command.c:645)
  and the bench's party Soundproof (:648); the AI's own raw reads (trainer_ai.c:1185-1188,
  :1208, :1256; AI_CannotDamageWonderGuard :3298; the bench's MON_DATA_ABILITY :2162); the
  whole-party scans (enumerate's lock / setter / Unburden / Trace / Shed Skin / Wonder Guard
  over-approximations, outcome's tracer, the AI refusal scans): raw is a superset of every
  Battler_Ability they stand for. Not done here: observation of the bit (`observe`) and its
  test strategy, with the mechanic in 2/2. Local check: ruff clean; the byte-identity run did
  not fit the 4 GB local cap (a jitted or eager `step` passes 3.8 GB): pod.
- E3-2-6 Gastro Acid 2/2 (#239): new `EffectId.GASTRO_ACID` (effect 239, BATTLE_EFFECT_SUPRESS_
  ABILITY; move 380; no manifest bump: no array or key changed, only the move row's effect
  value). `effects._gastro_acid` (in the dispatch tuple, after Worry Seed) runs
  subscript_suppress_target_ability.s: fails (:5-8) behind a Substitute (``h.shielded``), on a
  defender that has the bit (:6), on a miss or a semi-invulnerable defender (``h.effective``), on
  a raw Multitype ability (``decomp_abilities``, not the accessor); else :10 sets
  `ability_suppressed[target]`. `BattleMon.ability` is not written, so no `know_ability` tag (the
  AI reads the bit, trainer_ai.c:1177-1178). Reference clause beside Worry Seed's. The bit is in
  `effects.MOVE_EFFECT_MASK_FIELDS` and the reference's `_ingrain_anchors`: Roar's Ingrain test
  compares the whole mask to MOVE_EFFECT_INGRAIN (subscript_force_target_to_switch_or_flee.s:7),
  so a suppressed Ingrain user is dragged out. Not modelled, message only:
  subscript_gravity_start.s:17 (a suppressed Levitate holder prints no "couldn't stay airborne").
  Hit chance: `outcome._threshold_grid` gains the foe's Gastro Acid as a field variant (the
  side's own bit set before its check: Compoundeyes, Hustle, Simple, Unaware, No Guard read NONE,
  bcp.c:2876-2947), the Klutz-lost item live as for Worry Seed, and `future_thresholds`' loose
  flag. Crit: `CRIT_ODDS_EFFECTS[GASTRO_ACID]` = `_suppress_all` (every battler's bit): Super Luck
  stays (raw, lib.c:7122, :7126), a Klutz holder's crit item goes live (:7117, :5352-5356); the
  defender's Battle Armor / Shell Armor read (:7135-7136) is `armored`'s and in singles no Gastro
  Acid precedes the side's hit against its target (Owed 6); test_batch12's writer scan reads
  `battler_ability` as well as the raw ability. Enumerate: INPUT_CLAUSE effect (gastro-acid
  clause), NON_TABLE_DAMAGE, `damage_inputs_can_change` (foe Gastro Acid), end-speed tie
  (a speed ability or Shed Skin suppressed mid-turn). Klutz: `state.held_item` already reads
  `battler_ability` (E3-2-6a). Observation: `observe._NONZERO_SIDE` gains the bit (the message
  at :12), `plan_tree._COND_TEXT` its text; the Hypothesis state strategy
  (`tests/strategies._draw_klutz`) draws the bit and teaches Gastro Acid. AI:
  Basic_CheckGastroAcid (script.s:1363-1376): the player's bit -10, then LoadBattlerAbility in
  {Multitype, Truant, Slow Start, Stench, Run Away, Pickup, Honey Gather} -10
  (`commands.ABILITY_IDS` gains Slow Start 112, Stench 1, Run Away 50, Pickup 53, Honey Gather
  118, and `data/game_constants.py` lists them); Expert_GastroAcid (:5471-5494, the code, not
  the comment): 64 roll (:5481) jumping ends with 0, else +1, HP above 70 % ends, the 128 roll
  (:5484) jumping skips the first -1, else -1, above 50 % ends, else -1, above 30 % ends, else -1
  (`SITE_N` 5481: 64, 5484: 128). `KAIZO_TRANSCRIBED`, `KAIZO_EXPERT_LABELS` (:1782) and
  `KAIZO_EXPERT_TRANSCRIBED` carry 239; the Encore table already had it (:3276, once). Tag
  Strategy's partner path (:7454, :7624-7645): Owed 5, doubles only. Read bounds and the backlog
  regenerated (85 missing groups from 87: the two #239 rows, "move effect #239: Gastro Acid" and
  "AI move effect #239", used by 9 opposing rows, close; Metronome's callable list is 11 of 442
  unmodelled).
  Tests: tests/engine/test_e3_batch1.py `test_gastro_acid_*`, `test_the_suppressed_bit_is_observed`,
  tests/ai/test_kaizo_policy.py `test_gastro_acid_*`. Local check: ruff clean, the id-table scan
  and constants (16), test_state (24) passed; the step-running tests (a jitted `step` passes the
  4 GB local cap) and the AI scoring tests: pod.

## Batch 2 (rulings E3-B2-n)

- E3-B2-1 Fling (#233): new `EffectId.FLING` (effect 233, BATTLE_EFFECT_FLING; no manifest bump: no
  array or key changed, only the move row's effect value; `item_fling_power`, `item_fling_effect`
  and `item_hold_param` already carry every item) and `ids.FlingEffect` (FLING_EFFECT_*,
  items.h:86-116). Order, from effect_script_0233.s: `step._use_move` fails the move on the raw
  Multitype ability or a raw Griseous Orb of the user (:5-6, `decomp_abilities`, `state.item`), then
  TryFling (bs.c:8144-8154, `BattleSystem_FlingItem`, lib.c:5857-6158): the power is
  `Battler_ItemFlingPower` (0 under the user's Embargo, no Klutz, :5491-5498) and 0 fails
  (:5863-5865); the power feeds `in_power`. The TryFling-time writes land even on a miss:
  `hooks.fling_try` restores the foe's most depleted PP slot (Leppa, :5908-5933) and resets the
  foe's lowered stages (White Herb, :6029-6036). CalcCrit and CalcDamage (:9-10) read the item
  still held (`damage.flung_view`: the Dark boosters, Choice Band, Life Orb, the Metronome item,
  Scope Lens and the crit odds), `consume_item` is RemoveItem (:18), before TryMove, so a miss or
  Protect still loses it (Unburden arms); ApplyTypeChart (Expert Belt) and the accuracy check (Wide
  Lens, Zoom Lens) see it gone (`outcome._rate`). The hit-time half, `effects._fling` (second
  dispatch tuple), is subscript_fling.s: skipped for a defender at 0 HP or behind a Substitute
  (:5-7), and for a defender with MOVE_EFFECT_EMBARGO (lib.c:6145-6146); then `hooks.fling_effect`
  (HP and percent heals, the flavor berries' heal and dislike confusion, status and confusion cures,
  infatuation, Atk to Sp. Def +1, Starf +2, Lansat's Focus Energy, Micle's flag) and the INDIRECT
  flinch (King's Rock, Razor Fang: not before the foe's move, Inner Focus, Shield Dust) and
  statuses (`_inflict`: Substitute, Safeguard, immunities, Shield Dust, Limber, Water Veil...). The
  reference engine has the same blocks (`_fling_try`, `_fling_hit`, `_Turn.fling`).
  Rulings. (1) Starf's draw is the `acupressure` rank among the foe's five stats below +6 (value %
  count, the "uniform by the % n convention"): the decomp draws `RandNext() % 5` until the stat is
  not 12 (lib.c:6110-6129), whose per-stat weights are not exactly uniform (the redraw of a
  maxed stat); the engine uses 1/count (the convention of Acupressure, B12-9), no new outcome
  field. (2) The PLUCK flag Fling sets on the user (lib.c:6150-6152) is not modelled: it dies with
  selfTurnFlags at the next BattleContext_Init (bcp.c:3998) and is read only by a PLUCK_CHECK,
  which only a restore subscript of this same hit runs; the user's item is gone and the foe's items
  cannot newly trigger when the script is skipped, so no read of it is reachable. (3) The flung
  item is the raw item (BATTLEMON_HELD_ITEM), whatever Klutz hides, and the effect param is the raw
  hold param.
  Hit chance: `outcome._threshold_grid` adds the foe's Fling as a variant: a flung White Herb
  resets the side's accuracy stage before its check (`foe_herb`), a flung Micle sets its micle flag
  (`foe_micle`, bcp.c:2967-2970); the user's own Wide Lens is not applied. Crit:
  `CRIT_ODDS_EFFECTS[FLING]` = `_flung_focus_energy` (Focus Energy only on a battler another
  active battler's FLING_EFFECT_CRIT_UP item can be flung at, battle_lib.c:6131-6134: with no
  such item the write is a no-op, so a fight without one keeps its crit law and brute-force
  width; the writes take `(tables, state)`; test_batch12's writer scan
  plays Fling with a Lansat user and a foe without Focus Energy). Enumerate: INPUT_CLAUSE effect
  (fling clause), NON_TABLE_DAMAGE (power and item read at the hit), `damage_inputs_can_change`
  (foe Fling), Starf joins the acupressure relevance (the draw needs two or more stats), the end
  speed tie, and the flung paralysis and confusion join `para_inflict` / `confuse_inflict`.
  `outcome.kings_rock_chance` is over-approximate for a flung King's Rock.
  AI: Basic_CheckFling (script.s:1199-1286, dispatch :263): the chart immune -10 (:1201), the AI's
  `LoadFlingPower` (new `commands.load_fling_power`, trainer_ai.c:2014) below 10 -10, its own
  Multitype -10, then by its held item's hold effect (`commands.FLING_ITEMS_POISON` = PSN_USER,
  STRENGTHEN_POISON; `_BURN` = BRN_USER; `_PARALYZE` = PIKA_SPATK_UP, :1288-1299): poison and burn
  go to their AttackerChecks (-5 per test, else +3) when the status cannot land on the player
  (Safeguard, a status, a type or ability immunity; the poison block also on the AI's own Poison
  Heal, :1221-1222) and end otherwise; Light Ball -5 for a player that cannot be paralyzed
  (:1280-1286). `commands.ABILITY_IDS` gains Klutz 103 and Guts 62 (listed in game_constants).
  Expert_Fling (:5231-5286, dispatch :1776; the code, not the comment): a resisted or immune
  chart class goes to the item check (-1 unless the AI's hold effect is SOMETIMES_FLINCH,
  STRENGTHEN_POISON, PSN_USER, BRN_USER or PIKA_SPATK_UP, `FLING_DESIRABLE_EFFECTS`, :5288-5294);
  else by the fling power: below 30 -2 and end; above 90 the weakness check (a double or
  quadruple hit +4 then the 64 roll; else the 128 roll (:5268) jumping to the 64 roll, else +1
  first); above 60 the 64 roll (:5276, +1 unless it jumps); else the 128 roll (:5257) jumping
  ends, else -1 (`SITE_N` 5257: 128, 5268: 128, 5276: 64). `KAIZO_TRANSCRIBED`,
  `KAIZO_EXPERT_LABELS` (:1776) and `KAIZO_EXPERT_TRANSCRIBED` carry 233; the Encore table already
  had it. Read bounds and the backlog regenerated (the two #233 rows, "move effect #233: Fling" and
  "AI move effect #233", close). STATE.md: flingTemp and flingScript are "not needed" (one hit).
  Not built: Tag Strategy's partner paths (:7454, :7624-7645) for Fling: doubles only (Owed 5);
  `coverage.ITEM_MOVING_EFFECTS` keeps 233 (the exp-item party read, a separate reason).
  Tests: tests/engine/test_e3_batch2.py (`test_fling_*`, `test_flung_*`), tests/ai/test_kaizo_policy.py
  (`test_fling_*`); test_batch12's crit-writer scan and test_accuracy_classes' `_FOLLOWED` list
  carry Fling.

- E3-B2-2 Mirror Move (#9, BATTLE_EFFECT_COPY_MOVE; the only move with effect 9 in either game):
  new `EffectId.MIRROR_MOVE` (131), `order.CALL_MIRROR_MOVE` (an invoker with no draw in singles,
  ordered like Copycat and Me First), `MoveFlag.MIRROR_MOVE` (bit 1 << 23 from
  MOVE_FLAG_CAN_MIRROR_MOVE; `tools/build_tables._move_flags`; manifest 35: the packed `move_flags`
  values changed), state `move_copied` `[B]` and `move_copied_hit` `[B, B]` (int16, in
  `VOLATILE_FIELDS` for the holder's own entry and row; STATE.md rows mapped).
  Script: effect_script_0009.s:5-13 attack message, SetMirrorMove, else "Mirror Move failed!" and
  MOVE_STATUS_NO_MORE_WORK (the engine's failure path is Dream Eater's, B10-4: `failed`, no further
  difference). `BtlCmd_SetMirrorMove` (bs.c:3483-3525): `order.mirror_copied` (moveCopied, else in a
  doubles battle, `fmt.B > 2`, the `call % count`-th nonzero `move_copied_hit` entry: the decomp's
  `RandNext() % 4` until nonzero is uniform over the nonzero entries; `outcome._draw_calls` draws it,
  fold 44; singles never reads moveCopiedHit: it is moveCopied's twin there), then `order.mirror_call`:
  0 for MOVE_NONE or a move whose battle effect (`tables.move_ai_effect`) is that of a
  `CANNOT_ENCORE_MOVES` move (`Move_CanBeEncored`, lib.c:7221-7231: Transform, Mimic, Sketch, Mirror
  Move, Encore, Struggle); no Gravity or Metronome ban (no invoker has the flag, so none is ever
  recorded). A success is GoToMoveScript on the called move as the user's (the existing invoker
  path: `invoked_move`, `called_move`, a defender re-picked by the called move's own range,
  `subscript_no_target` with none); PP is Mirror Move's alone (DecrementPP ran at BeforeMove,
  bcp.c:3181-3185). Choice lock and `last_move` are moveTemp's (Mirror Move).
  Writer (`UpdateFlagsWhenHit`, bcp.c:4480-4486): `step._use_move` computes it from moveTemp
  (`move`, not the called move), `MoveFlag.MIRROR_MOVE`, the attack message shown (`shown_t`) and the
  defender: the foe (`record`) or the user for a flagged self-defended move (Trick Room, RANGE_FIELD;
  a failed Metal Burst, RANGE_SINGLE_TARGET_SPECIAL: the Counter family's defender is the user until
  its command succeeds), and `step._move_end` applies it after the faint check like `move_hit`
  (LOOP_FAINTED precedes it, bcp.c:3820-3837), so a fainted defender keeps it until its replacement
  switches in; a miss or Protect still records (the message showed), a confusion self-hit does not
  (no message). A pivot's resumed tail writes for the foe only. Reference engine: `_copied_writes`
  and `_finish_action`. Clears: `state.clear_copied` (every foe's `move_copied`, every holder's
  column of the battler) beside the own entry and row (`VOLATILE_FIELDS`) at a switch-in or faint
  (`step._reset_battler`, lib.c:2125-2149, :2203-2220), and in `outcome.pivot_view`; the reference
  `_clear_copied`. Nothing counts down.
  Hit chance, damage and crit: those of the called move (no draw of its own; the accuracy roll is the
  called move's, Mirror Move has accuracy 0). Enumeration: `INPUT_NEUTRAL_EFFECTS` and
  `NON_TABLE_DAMAGE_EFFECTS` (an invoker), `outcome_laws`' `copycat` branch (two orders), call law
  `_side_call_law`: value 0 the turn-start moveCopied, value 1 the foe's chosen move when it has
  the flag and is not self-ranged (`order.MIRROR_SELF_RANGES`; a Counter-family move counts as aimed,
  over-approximate) else the turn-start entry, `order.used_moves`, `can_call_effect`.
  AI: Expert_MirrorMove (script.s:1907-1926, dispatch :1632; rolls :1915 (128, +2 when it does not
  jump) and :1922 (80, -1 when it does not jump): `SITE_N`, `KAIZO_TRANSCRIBED` 9,
  `KAIZO_EXPERT_LABELS`, `KAIZO_EXPERT_TRANSCRIBED`, `BattleEffect.COPY_MOVE`); the table
  (:1928-1976) is `commands.MIRROR_MOVE_TABLE`, the same 47 moves in the same order as
  `COPYCAT_ENCOURAGED_MOVES` (compared name by name against the decomp by
  `test_mirror_move_tables_are_the_decomps`; no new ids, so no game_constants entry). It reads
  `LoadBattlerPreviousMove` (movePrevByBattler, `last_move`), not moveCopied. Risky_RiskyEffects
  already lists 9 (:6539). No Basic entry. Read bounds and the backlog regenerated.
  Rulings. (1) skipPressureCheck (bs.c:3521, set on a failed call) is not modelled: it is read only
  by DecrementPP (bcp.c:2238), which already ran in this action, and dies with selfTurnFlags at the
  next BattleContext_Init (bcp.c:3998). (2) SYSCTL_REUSE_LAST_MOVE (Magic Coat / Snatch, bcp.c:3077,
  :3099) keeps their reuse of a move from recording; Magic Coat now skips the write (E3-B2-6,
  `step` `record`, reference `_actions`); Snatch is owed (Owed 1) and must skip it too. (3) The write's defender for a self-defended flagged
  move records for the user, so a user's own Trick Room can be Mirror Moved by itself later.
  Not built: Tag Strategy and the doubles path of the enumeration (the doubles draw is in
  `mirror_copied` / `_draw_calls`, no doubles format runs `call_laws` yet). Tests:
  tests/engine/test_e3_batch2.py (`test_mirror_move_*`, `test_flagged_*`,
  `test_a_self_defended_*`, `test_a_switch_clears_*`), tests/ai/test_kaizo_policy.py
  (`test_mirror_move_tables_*`, `test_expert_mirror_move`); test_accuracy_classes' `_FOLLOWED` list
  and the state strategy (`move_copied`, `_draw_mirror_move`) carry it.

- E3-B2-3 Conversion 2 (#93, BATTLE_EFFECT_CONVERSION2; move 176, the only move with effect 93 in
  either game): new `EffectId.CONVERSION_2` (132), state `conv2_move` / `conv2_battler` (`1 +
  battler`) / `conv2_type` `[B]` (`VOLATILE_FIELDS`; STATE.md rows mapped), `Outcome.conv2`
  `int8[B, 18]`. No manifest bump: no array or key changed. The 112 rows of
  `sTypeMatchupMultipliers` come from `type_chart_order` / `type_chart_foresight` (decomp row =
  order, +1 for the Foresight rows after the 0xFE separator at 108); `damage.conv2_pairs`, checked
  against the decomp source by `test_the_row_table_is_the_decomps`.
  Script: effect_script_0093.s:5 direct ON_HIT (lib.c:1485-1491), subscript_conversion_2.s:5
  TryConversion2, failure -> :12-14 MOVE_STATUS_FAILED with the attack message skipped.
  `BtlCmd_TryConversion2` (bs.c:4765-4820): `effects.conv2_fails` = raw-ability Multitype (:4770,
  `battler_decomp_ability`), no record (:4775-4776), a recorded `Move_IsMultiTurn` move (lib.c:2975-2992:
  Bide and `MoveFlag.CHARGE`) whose attacker has `locked_move` (:4778-4782), or no eligible row.
  Eligible row: pair listed, multiplier 0 or 5, defending type not the user's type 1 or 2
  (`damage.conv2_eligible`). Draw law (lib.c:2994-3007; `RandNext` is a u16): up to 1000 tries of
  `% 112`, rows 0-15 weigh 586 and the rest 585 of 65536; first eligible try wins; after 1000 misses the
  linear scan takes the first eligible row. So `P(d) = (1 - p0) w_d / W + p0 [d is the first row]`,
  `p0 = (1 - W/65536)^1000`. `enumerate._conv2` computes it with fractions and writes it on a 2^-52
  grid (the largest weight takes the rounding residue): the exact denominator `65536^1000 * W` does
  not fit int64 weights; the error is below float64's resolution. `outcome._conv2_draw` samples
  `damage.conv2_probs` (float32). The value is the picked defending type, one element per (battler,
  recorded type) because the law depends on both; the handler reads `conv2[user, conv2_type]`; a
  draw not eligible in the live state (an outcome built for another state) takes the first row.
  Relevance (`outcome_laws`, `_branch_masks`): with the user first, the turn-start record's type; with
  the foe first, the type of the foe's move when it is aimed at the user (`not self_defender`: it
  overwrites or clears the record), else the turn-start one; loose: every type. Over-approximate
  (the foe's move may miss: Conversion 2 then draws nothing).
  Writer (UpdateFlagsWhenHit, bcp.c:4498-4536, not SYSCTL_REUSE_LAST_MOVE): for the foe defender
  (`record`), where `succeeded` and no NO_EFFECTS (`~no_effects`: a charge turn counts as working) and
  the range of moveCur (the called move) is not user / user side / field / opponent side / ally /
  user-or-ally, `(moveCur, attacker + 1, move_type_of)`; every other case clears all three.
  `step._use_move` computes it (`ctl["conv2"]`, masked by `runs_move`), `step._move_end` applies it
  after the faint check like `move_hit`, then clears the actor's own `conv2_move` (ClearFlags
  :4390). A switch-in or faint clears all three (lib.c:2131-2133, :2209-2211). A pivot's resumed tail
  writes for its landed hit. Reference engine: `_conv2_writes`, `_conv2_fails`, `_conv2_type`,
  `_finish_action`. A failed Conversion 2 joins `items_full` (no AfterMoveEffects, as
  `weather_failed`) and `shown_t` (no attack message: no movePrev, no Choice lock).
  Hit chance: none (accuracy 0, RANGE_USER). Damage and crit: none of its own; the new types are
  the foe's later chart (`INPUT_CLAUSE_EFFECTS`, the foe-effect camouflage clause, and
  `NON_TABLE_DAMAGE_EFFECTS`). Held items: none.
  AI: no Basic or Expert label; 93 in `KAIZO_TRANSCRIBED` (label-less list); the tables naming it
  (script.s:3223, :7798, :7847, :7958) were already transcribed. Read bounds and backlog regenerated.
  Rulings. (1) Normalize (CalcCurrentMoveType, bcp.c:3612-3621) is an unsupported ability, refused by
  coverage, so the recorded type is `move_type_of`. (2) A confusion self-hit makes the user its own
  defender (bcp.c:2629): nothing is written for the foe. (3) SYSCTL_REUSE_LAST_MOVE (Magic Coat /
  Snatch) skips the write: Magic Coat does (E3-B2-6, the Magic Coat pass writes no conversion2), Snatch is owed (Owed 1). (4) Only a battler's own action
  changes its types, so the law reads the turn-start types.
  Not built: doubles (the partner as `conv2_battler` is stored as a battler, the MOVE_LOCKED test
  reads it, but no doubles format runs the enumeration yet). Tests: tests/engine/test_e3_batch2.py
  (`test_conversion_2_*`, `test_the_*`, `test_a_*` of the Conversion 2 section), tests/ai/
  test_kaizo_policy.py (`test_conversion_2_has_no_basic_or_expert_label`), tests/engine/test_state.py
  (`test_conversion_2_record_maps_to_conv2_fields`); `fullset`-free; pod: test_enumerate's marginals
  need a `conv2` oracle clause (not added: the law is checked by the retry-loop test).

- E3-B2-4 Spite (#100, BATTLE_EFFECT_DECREASE_LAST_MOVE_PP; move 180, the only move with effect 100 in
  either game): new `EffectId.SPITE` (133). No new state, no `Outcome` field and no draw (BtlCmd_
  TrySpite reads no random). Tables: only `move_effect` of Spite (effect-map entry); no manifest bump:
  no array or key changed (as Conversion 2).
  Script: effect_script_0100.s:5 direct ON_HIT TO_DEFENDER (lib.c:1485-1491: not on NO_EFFECTS, so not
  on a miss, Protect or a semi-invulnerable foe), subscript_reduce_target_pp.s:5 skips MISSED |
  SEMI_INVULNERABLE, :6 TrySpite, failure -> :12-14 `MOVE_STATUS_FAILED` with the attack message (:7)
  skipped. `BtlCmd_TrySpite` (bs.c:4990-5016): `effects.spite_slot`: the first slot holding the
  defender's `movePrevByBattler` (`last_move`; Battler_SlotForMove, lib.c:2376-2387) fails when
  `last_move` is 0 (:4995, :5011-5013), is in no slot (Struggle, a move no longer held: :4996) or its
  slot is at 0 PP (:4996-4998); else `dec = min(4, ppCur)` (:5000-5003), `ppCur -= dec`, CopyToParty.
  `effects._spite` writes `pp[party, active, slot]` only: a Mimic slot's `shadow_pp` stays (CopyToParty
  skips flagged slots). Through a Substitute, no ability check.
  Failure joins `primary_effect_fails` (no AfterMoveEffects) and `shown_t` (no attack message: no
  movePrev, no Choice lock, no moveHit) beside Conversion 2's. Reference engine: `_spite_slot`, the
  `_move_effect` branch, the `full` flag, and `_actions` clearing the foe's moveHit for a failed Spite.
  Hit chance: acc 100, standard. Damage and crit: none (status move). The enumeration classes it
  `INPUT_NEUTRAL_EFFECTS` / `NON_TABLE_DAMAGE_EFFECTS` (no damage input changes; a foe move it empties
  fails with NO_PP, which only removes later draws); `test_accuracy_classes` lists it as writing no stage.
  Held items: the defender's Leppa Berry (after-move TriggerHeldItem, bcp.c:3694; lib.c:4636-4653)
  restores the slot Spite emptied in the same action (`hooks._leppa`, tested).
  Same-turn effect (bcp.c:2275-2286, :3209): a faster Spite leaves the foe's chosen move at 0 PP:
  DecrementPP sets MOVE_STATUS_NO_PP (a NO_EFFECTS status) and the move fails. Engine fix found by the
  test: `step._use_move` wrote `last_move` for such a move (SYSCTL_MOVE_SUCCEEDED is set only on the
  non-NO_EFFECTS branch, :3211-3212); it now writes 0 (the Metronome count still follows, :3221).
  AI: no Basic or Expert label; 100 in `KAIZO_TRANSCRIBED` (label-less list); its tables, CheckHP_Target_
  DiscourageAtLowHP (script.s:7960) and Harrassment_Effects (:8051), were already transcribed
  (commands.py). Read bounds and backlog regenerated.
  Rulings. (1) Kaizo's Magic Coat on Spite (a reflected Spite cuts the user's own Spite PP): Magic
  Coat is built (E3-B2-6, tested with Kaizo's Spite there); Snatch is still owed (Owed 1 of E3-B2-3). (2) The
  target's slot is read at the move's hit, after any faster move of this turn (`last_move` is
  written when the foe acts, bcp.c:3945-3949).
  Not built: doubles (the defender is the chosen foe, `last_move[target_side]`
  is per battler, but no doubles format runs the enumeration yet). Tests: tests/engine/test_e3_batch2.py
  (`test_spite_*`, `test_a_*` of the Spite section), tests/ai/test_kaizo_policy.py
  (`test_spite_has_no_basic_or_expert_label`); `fullset`-free.

- E3-B2-5 Role Play (#178, BATTLE_EFFECT_COPY_ABILITY; move 272, the only move with effect 178 in either
  game, acc 0, no PROTECT / MIRROR_MOVE flag): new `EffectId.ROLE_PLAY` (134). No new state, no
  `Outcome` field and no draw. Tables: only `move_effect` of Role Play (effect-map entry); no manifest bump:
  no array or key changed. Machinery is Skill Swap's (E3-2-5), nothing new.
  Script: effect_script_0178.s:5-6 fails (:10-12 `MOVE_STATUS_FAILED`) on the user's `Battler_Ability`
  Multitype (`CheckAbility CHECK_HAVE`, battle_script.c:3145-3183; Multitype survives Gastro Acid,
  battle_lib.c:3090-3091) or raw held Griseous Orb; :7 runs `subscript_copy_ability.s` as a direct
  `CHECK_HP | TO_DEFENDER` (living defender, battle_lib.c:1492-1507). The subscript fails (:5-9, :31-32) on
  MISSED | SEMI_INVULNERABLE (`h.landed`), the defender's raw Wonder Guard, Multitype or ABILITY_NONE, the
  user's raw Multitype; no Protect flag and no Substitute test (it goes through both). Else `truant` of the
  USER = `totalTurns & 1` (:11-13) and its `BattleMon.ability` = the defender's raw one (:14-15; a Gastro
  Acid suppressed defender still gives it, the user's own suppression bit stays); only the user changes.
  `effects.role_play_fails` (shared by `_role_play`, `primary_effect_fails`, `step`'s `role_failed`) and
  `effects._role_play`; the AI is tagged with the user's new ability (`know_ability`,
  battle_script.c:3401-3404; the GET at :14 reveals nothing). A failure skips the attack message (the
  subscript's `Call`, :10 / effect script's early End): `step`'s `shown_t` and `items_full`, the
  reference `_fight` (`attack_message = False`) and `_actions` (moveHit record cleared, as Spite).
  Accuracy 0: no roll (bcp.c:2901-2903), so the hit draw is irrelevant.
  Entry abilities: no announced flag is touched, so an entry ability the user now holds acts at the
  move's end through `hooks.move_end_switch_in` (bcp.c:3973) when its flag is clear (Intimidate,
  weather, Trace, Download; tested with Intimidate). Persistence: the BattleMon only, `_restore_ability`.
  Hit chance: a faster foe's Role Play gives the foe the side's ability (`outcome._threshold_grid`'s
  field variant, `rp`: the foe holds the side's ability, a weather one acting at the move's end;
  `future_thresholds`' `loose`); crit: no `CRIT_ODDS_EFFECTS` entry (it writes only the user's ability,
  never another battler's crit inputs: test_batch12's writer scan stays), `outcome.armored` counts a
  foe's Role Play like Transform / Skill Swap; enumerate: `INPUT_CLAUSE_EFFECTS` (role-play clause),
  `NON_TABLE_DAMAGE_EFFECTS`, `damage_inputs_can_change`, end-speed tie and Shed Skin copy.
  AI: Expert_Main :1733 -> Expert_ChangeUserAbility (script.s:4415-4433, shared with Skill Swap's :1743;
  roll site 4429); `BattleEffect.COPY_ABILITY` (the Encore table entry :3252 was already in `commands`);
  no Basic entry; read bounds and backlog regenerated (the #178 AI row closes).
  Rulings. (1) A copied Slow Start (:16-21) cannot arise: unsupported ability, coverage refuses its
  holders (as Skill Swap). (2) Skill Swap's own failure paths do not yet clear the attack message /
  movePrev record (its subscript fails before its `Call` too, subscript_exchange_abilities.s:5-14);
  found while building this, not changed here: owed to Skill Swap. (3) Doubles: the target is the chosen
  `h.target`; `role_play_fails` reads `state.active[target_side]`.
  Tests: tests/engine/test_e3_batch2.py (`test_role_play_*`, `test_the_copy_ends_with_the_battler`),
  tests/ai/test_kaizo_policy.py (`test_role_play_shares_the_change_user_ability_block`, the exact-equals-loose
  list), tests/engine/test_accuracy_classes.py (`_FOLLOWED`); `fullset`-free.

- E3-B2-6 Magic Coat (#183, BATTLE_EFFECT_APPLY_MAGIC_COAT; the move of that name, priority +4): new
  `EffectId.MAGIC_COAT` (135), `MoveFlag.MAGIC_COAT` (bit 16777216, MOVE_FLAG_CAN_MAGIC_COAT, a
  `DECOMP_FLAG_BITS` entry; vanilla and Kaizo flags, Gastro Acid and Kaizo's Torment / Spite among
  them) and the state `magic_coat` (per battler, int8, 1 + the user's move slot; STATE.md `turnFlags`
  row, `magicCoatMon` reworded). Manifest bumped to 36 (the flag array changed; both tables rebuilt).
  Script: `BtlCmd_TrySetMagicCoat` (battle_script.c:6459-6478) sets the flag and fails when
  `waitingBattlers == 1` (the user acts last: `h.target_moved`); `effects._magic_coat`. A move aimed at
  the holder with MOVE_FLAG_CAN_MAGIC_COAT is stolen in BEFORE_MOVE_STATE_CHECK_STOLEN
  (bcp.c:3196-3200, `MoveStolen` :3064-3085): the flag clears, `movePrev` / `movePrevByBattler` take
  the move, SYSCTL_REUSE_LAST_MOVE is set and `subscript_magic_coat` swaps attacker and defender
  (bs.c:6487-6513); the move then runs as the holder's own on the mover, accuracy, damage and
  effects included, and LoopSpreadMoves swaps back (bcp.c:3831-3835). Built as a loop phase
  `_REFLECT` of `step._execute_actions` (`MODE_REFLECT`; the reference recurses with
  `_fight(reflect_pass=True)`), so `_use_move` is traced once.
  Rulings. (1) REUSE_LAST_MOVE skips the moveHit, moveCopied and conversion2 writes (bcp.c:4481, :4498,
  :4509) and UpdateMoveBuffers' movePrev / moveProtect writes; the Choice-lock clear (:4489-4496) is
  outside the gate and still runs (`lock_record`). Mirror Move's exemption (Owed 1 of E3-B2-2,
  bcp.c:3077) is closed by the same flag. (2) `BattleSystem_DecPPForPressure(holder, mover)` takes
  1 PP from the holder's slot when the mover has Pressure (bcp.c:3083, lib.c:7390-7397). (3) The
  Metronome count updates in the holder's second pass only (bcp.c:3221) and verifies for the mover
  after the swap back. (4) Draws: the holder's pass is the reflected move's, so
  `order.used_moves` substitutes the mover's move for the holder's Magic Coat when it is stolen
  (the hit classes come from the threshold grid, so the substitution keeps the draws exact).
  (5) Kaizo's Spite reflected cuts the user's own Spite PP (Owed 1 of E3-B2-4, closed).
  AI: Expert_MagicCoat (script.s:4492-4515, entry :1736), rolls at :4499 (100), :4505 (150), :4511
  (30; the 50 at :4508 follows a GoTo, unreachable); `outcome.SITE_N`, KAIZO_TRANSCRIBED,
  KAIZO_EXPERT_LABELS, `commands`, `policy._expert_magic_coat`; read bounds and backlog regenerated.
  Enumeration: `INPUT_NEUTRAL_EFFECTS`, `NON_TABLE_DAMAGE_EFFECTS`; `test_accuracy_classes`
  `_NO_ACC_EVA_WRITE`.
  Owed 1 lines closed: CAN_MAGIC_COAT of Torment / Worry Seed / Thief-AI (E3-1-3, E3-2-2, E3-1-1),
  Mirror Move's REUSE_LAST_MOVE exemption (E3-B2-2), Conversion 2's (E3-B2-3), Spite's Kaizo
  reflection (E3-B2-4). Still owed: Snatch (REUSE_LAST_MOVE, bcp.c:3099).
  Not built: doubles (the reflected move's target is the mover's side; the pass is per side).
  Tests: tests/engine/test_e3_batch2.py (`test_magic_coat_*`, `test_a_reflect*`, `test_the_metronome_count_follows_the_passes`),
  tests/ai/test_kaizo_policy.py (`test_magic_coat_*`, `test_expert_magic_coat`, the exact-equals-loose list),
  tests/engine/test_state.py (`test_magic_coat_flag_maps_to_magic_coat`),
  tests/engine/test_accuracy_classes.py; `fullset`-free.

- E3-B2-7 Lucky Chant (#240, BATTLE_EFFECT_PREVENT_CRITS; move 381, the only move with effect 240 in
  either game, acc 0; priority 0 vanilla, +2 Kaizo): new `EffectId.LUCKY_CHANT` (136). No new state
  (`lucky_chant [2]` and its countdown, `step.FIELD_SIDE_TIMERS`, existed), no `Outcome` field and no draw.
  Tables: only `move_effect` of Lucky Chant (effect-map entry); no manifest bump: no array or key changed.
  Script: effect_script_0240.s:5, :9-11 fails (`MOVE_STATUS_FAILED`) with SIDE_CONDITION_LUCKY_CHANT on the
  user's side, else :6 an indirect ON_HIT `subscript_lucky_chant_start.s:5-6` ORs 0x5000 into the side
  conditions: the 3-bit counter (bits 12-14, condition.h:100-108) = 5 (`ids.LUCKY_CHANT_TURNS`,
  `effects._lucky_chant`, failure in `primary_effect_fails`; reference clause and failure beside Mist's).
  The turn-end pass (bcp.c:1062-1083, after Tailwind, before Wish) counts it down: active on the use turn
  and the 4 after. Side state: survives switching and Baton Pass; no Snatch / Magic Coat / Protect flag.
  Crit: `BattleSystem_CalcCriticalMulti` (battle_lib.c:7132-7140) draws `RandNext() % rate` first, then
  no crit when the defender's side conditions (battle_script.c:2158-2163) have the condition: the draw is
  spent either way. New `effects.crit_blocked` (= `armored` or the defending side's `lucky_chant` as of the
  hit) replaces `armored` at the three step crit sites (the hit, Jump Kick's crash, the Pursuit hit) and
  the move-damage call in `effects.play_hit`; the reference's `_crit_blocked` at its three. The turn-start
  `no_crit` of `outcome.sample_outcome` / `enumerate._crit` stays (a condition up at the turn start is up
  for the whole turn: the counter only falls at the turn's end). Same-turn case (Kaizo's +2, or a faster
  Lucky Chant): the draw is the usual one and `step` blocks the crit at the hit; no `CRIT_ODDS_EFFECTS`
  entry (it writes no crit input of the attacker, it blocks the crit outright). Enumeration:
  `INPUT_CLAUSE_EFFECTS` (the lucky-chant clause in `damage_inputs_can_change`: a foe's Lucky Chant while
  its side has none changes the side's later hit, so the turn-start crit classes are not used for it),
  `NON_TABLE_DAMAGE_EFFECTS`; `test_accuracy_classes._NO_ACC_EVA_WRITE`.
  AI: Basic_ScoreMoveEffect :270 -> Basic_CheckLuckyChant (:1378-1381, the AI's side under it -10;
  `commands.SideCondition.LUCKY_CHANT` = 0x7000 reading `lucky_chant`); Expert_Main :1783 ->
  Expert_LuckyChant (:5496-5517): AI HP below 70 % -1; else the player knowing HIGH_CRITICAL (43),
  HIGH_CRITICAL_BURN_HIT (200) or HIGH_CRITICAL_POISON_HIT (209) +1; else the 64 roll (:5506) jumping +1,
  else 0 (`SITE_N` 5506: 64; `policy._basic_check_lucky_chant`, `_expert_lucky_chant`;
  `BattleEffect.PREVENT_CRITS`, KAIZO_TRANSCRIBED, KAIZO_EXPERT_LABELS, KAIZO_EXPERT_TRANSCRIBED). The
  tables at :3277 (Encore), :6493 (SetupFirstTurn) and :7806 (CheckHP medium) were transcribed already.
  Read bounds and backlog regenerated (the two #240 rows, "move effect #240: Lucky Chant" and "AI move
  effect #240", close: 71 distinct missing groups from 73).
  Rulings. (1) The sheet's `no_crit` variant for a faster Lucky Chant is `step`'s read at the hit, not an
  enumeration variant: the crit draw's law is the same with or without it. (2) Doubles: the side field is
  indexed by `side_of`; `crit_blocked` reads the defender's side, so it protects the partner too.
  Not built: Owed lines naming Lucky Chant: none. Not modelled, message only: "The Lucky Chant shielded
  your team from critical hits!" and the wear-off message.
  Tests: tests/engine/test_e3_batch2.py (`test_lucky_chant_*`, `test_a_faster_lucky_chant_*`,
  `test_kaizos_priority_puts_lucky_chant_*`, `test_a_lucky_chant_in_the_turn_*`),
  tests/ai/test_kaizo_policy.py (`test_lucky_chant_*`, `test_expert_lucky_chant`, the exact-equals-loose
  list), tests/engine/test_accuracy_classes.py, tests/engine/test_damage_classes.py (two `slow` EXACT_CASES:
  a faster and a slower foe Lucky Chant, brute force); `fullset`-free.

- E3-B2-8 Charge (#174, BATTLE_EFFECT_SP_DEF_UP_DOUBLE_ELECTRIC_POWER; move 268, the only move with effect
  174 in either game, acc 0, RANGE_USER, CAN_SNATCH; PP 20 vanilla, 1 Kaizo): no new `EffectId`: Charge is
  `STAT_SELF` (Sp. Def +1) with `move_param[m, 2] = ids.CHARGE_PARAM` (3), beside Defense Curl (1) and Minimize
  (2); `tools/effect_map.py` entry 174. New per-battler `charged_turns [B]` int8 (0-2: `chargedTurns`, 2 bits,
  battle_mon.h:11, with its `MOVE_EFFECT_CHARGE` bit, moves.h:17, set and cleared together) in
  `VOLATILE_FIELDS`, not in `BATON_PASSED_FIELDS`; no `Outcome` field and no draw. Tables: `move_effect` and
  `move_param` values of Charge only; no manifest bump (no array or key changed).
  Script: effect_script_0174.s:5 an indirect ON_HIT `subscript_charge.s`: :5-6 MOVE_EFFECT_CHARGE and
  chargedTurns = 2 on the attacker (`ids.CHARGE_TURNS`, `effects._stat_self`), then :11-13 Sp. Def +1 through
  UPDATE_STAT_STAGE (a +6 stage fails with a message only; the counter was set first and the move does not
  fail: no `primary_effect_fails` entry). No fail condition, no Substitute or Shield Dust relevance (the
  target is the user).
  Turn end: `MON_COND_CHECK_STATE_CHARGE` (bcp.c:1626-1632; enum :1276-1278: after Lock-On, before Taunt):
  chargedTurns above 0 drops by 1 and the bit goes at 0 (`step._charge_check`, in `_uproar_then_disable`
  after `_lock_on_check`; reference `_charge_check` at the same place in its list). Set on turn T: 2 -> 1 at
  T's end, up for T + 1, 0 at T + 1's end.
  Power: `BattleSystem_CalcMoveDamage` (battle_lib.c:6690-6692) doubles `movePower` when the attacker has the
  bit and `moveType` is Electric, after powerMul (:6688), before Helping Hand (:6694) and Technician (:6698):
  `damage._pre_random` (move type now resolved before powerMul; the AI's estimate, `ai=True`, reads the same
  `charged_turns` with `ai_case_type`) and the reference's `_calc_move_damage`. The hit does not consume it.
  Cleared at switch and faint (lib.c:2071, :2101-2104, :2166, :2185-2188); not Baton Passed (moves.h:46-48).
  `effects.MOVE_EFFECT_MASK_FIELDS` (Roar's whole-mask Ingrain test, subscript_force_target_to_switch_or_flee.s:7)
  and the reference's `_ingrain_anchors` include it. Observation: `observe._NONZERO_SIDE` "charged_turns"
  ("{0} began charging power!", subscript_charge.s:7), `plan_tree._COND_TEXT`.
  Enumeration: no new clause. A user's own Electric move that turn is its one action, so the counter is a
  turn-start damage input (the turn-start tables read it), and the foe's Charge changes the side's Sp. Def
  stage through the existing `STAT_SELF` clauses. No `INPUT_CLAUSE_EFFECTS`, `NON_TABLE_DAMAGE_EFFECTS` or
  `CRIT_ODDS_EFFECTS` entry; no accuracy / evasion write (`STAT_SELF` with Sp. Def).
  AI: no Basic or Expert label for 174 (script.s names it once, in Expert_Encore_EncouragedMoveEffects, :3250,
  already transcribed: `commands.ENCORE_ENCOURAGED_EFFECTS`); no roll; no read-bounds row (regenerated: no
  change). Backlog regenerated: the row "move effect #174: Charge" (Roark Split) closes.
  Rulings. (1) STAT_SELF with a param, not a new `EffectId`: the stat step, the Metronome call
  and every STAT_SELF clause are shared, and the two decomp commands (flag set, then stage) are the
  same order as Minimize's. (2) The doubling reads the engine's `move_type_of` (Normalize is unsupported, as
  everywhere in the engine and reference). (3) Doubles: the field is per battler, the doubling reads the
  attacking battler's own `charged_turns`, and the countdown runs in the per-battler end-of-turn loop.
  Not built: Owed lines naming Charge: none. Snatch (CAN_SNATCH) is not modelled in the engine (nothing
  steals a stat-raising move yet). Not modelled, message only: "{0} began charging power!".
  Tests: tests/engine/test_e3_batch2.py (`test_charge_*`, `test_a_charged_*`, `test_a_faster_foe_hits_before_
  the_charge_lands`, `test_the_ai_estimate_reads_the_charge`, `test_a_switch_clears_the_charge`),
  tests/ai/test_kaizo_policy.py (`test_charge_has_no_ai_label_and_one_table_entry`,
  `test_expert_encore_encourages_a_players_last_charge`); `fullset`-free.

- E3-B2-9 Grudge (#194, BATTLE_EFFECT_REMOVE_ALL_PP_ON_DEFEAT; move 288, the only move with effect 194 in either
  game, acc 0, PP 5, RANGE_USER; player-only in Kaizo, Metronome-callable): new `EffectId.GRUDGE` (137,
  `tools/effect_map.py` entry 194); new per-battler `grudge [B]` int8 (`MOVE_EFFECT_GRUDGE`, moves.h:22, a bit with
  no count) in `VOLATILE_FIELDS` (cleared at switch and faint, lib.c:2071, :2166), not in `BATON_PASSED_FIELDS`
  (moves.h:46-48), in `effects.MOVE_EFFECT_MASK_FIELDS` (Roar's whole-mask Ingrain test,
  subscript_force_target_to_switch_or_flee.s:7; the reference's `_ingrain_anchors` list has it too). No `Outcome` field, no draw. Tables: `move_effect` of Grudge only; no
  manifest bump (no array or key changed).
  Script: effect_script_0194.s:6 sets the bit (`effects._grudge`, an indirect ON_HIT message, :7-9). Its :5 fail
  branch (the bit already set) never runs: CheckStatusDisruption clears the bit at the start of every move
  attempt of the user (`CHECK_STATUS_START`, bcp.c:2424-2427, before the script, whether the checks then
  stop the move or not), so there is no `primary_effect_fails` entry. `step._use_move` clears it beside
  Destiny Bond's clear (`checks`), the reference's `_fight` the same.
  Trigger: `BtlCmd_TryGrudge` (battle_script.c:6743-6763), reached from subscript_faint_check_destiny_bond.s:28
  (after its Destiny Bond block, :5-25) and subscript_attack_then_switch_out.s:40 (after its own copy of the
  block, :14-34). The first runs from LOOP_FAINTED after a move (`AnyFainted`, onlyFaint FALSE, bcp.c:3820-3823,
  :4423-4427) and from subscript_pursuit.s:85; faints outside a move (residual, recoil, Future Sight, bcp.c:940,
  :1295, :1750, :4004, :4722) take the onlyFaint TRUE path: no Grudge. If the fainted battler held the bit, the
  attacker is on the other side, the attacker's HP is above 0 (a Destiny Bond kill, which runs first, leaves
  nothing) and `moveTemp` is not Struggle: `ppCur[ATTACKER_MOVE_SLOT]` = 0 and `BattleMon_CopyToParty`
  (a Mimic-flagged slot is not copied: the party's `shadow_pp` stays). `effects.try_grudge`, called at the
  engine's three Destiny Bond sites (`step._use_move`'s after-the-hits site and U-turn site, `step._pursuit_hit`)
  and the reference's (`_fight`, `_u_turn`, `_pursuit`) through `_try_grudge`.
  Slot: `ATTACKER_MOVE_SLOT` (common.h:8, `moveSlot[attacker]`) is written at command selection (bcp.c:445,
  :526) only, so an invoker's call (Sleep Talk, Metronome, Mirror Move ...) zeroes the invoker's own slot, an
  Encore's forced slot is the encored one, and Pursuit's is the pursuer's chosen slot (the one Pursuit spent,
  battle_script.c:6919); the engine's `slot` of `_use_move` is that slot already (the PP decrement's). The
  reference reads `turn.move_slot` (set from the action at `_fight`'s start).
  Leppa Berry: the attacker's after-move held-item check (`AfterMoveEffects`, bcp.c:3687) runs before LOOP_FAINTED,
  so it misses the zero; the end-of-turn check (bcp.c:1363; lib.c:4636-4653) restores it. Both engines already run
  the item check inside the hit loop, before the Grudge write.
  Enumeration: `INPUT_NEUTRAL_EFFECTS` and `NON_TABLE_DAMAGE_EFFECTS` (a status move that reads and writes PP only
  after a faint; a PP the same turn cannot change a hit: each side has one action). No `INPUT_CLAUSE_EFFECTS`,
  `CRIT_ODDS_EFFECTS` entry, no accuracy / evasion write (`test_accuracy_classes._NO_ACC_EVA_WRITE`).
  Observation: `observe._NONZERO_SIDE` "grudge" ("{0} wants the foe to bear a grudge!", effect_script_0194.s:7),
  `plan_tree._COND_TEXT`.
  AI: no Basic or Expert label for 194 (script.s names it twice, in Expert_Encore_EncouragedMoveEffects, :3259, and
  CheckHP_DiscourageAtHighHP, :7757, both already transcribed: `commands.ENCORE_ENCOURAGED_EFFECTS`,
  `commands.CHECK_HP_TABLES`); no roll; no AI command reads the bit. 194 joins `outcome.KAIZO_TRANSCRIBED`
  (a label-less transcribed effect, as Spite: its move takes the generic paths, `opponent_move_refusals` is
  empty). Read bounds regenerated; backlog regenerated: the row "move effect #194: Grudge" (Roark Split) closes.
  Rulings. (1) U-turn is a site the sheet did not list: subscript_attack_then_switch_out.s:40 runs TryGrudge
  right after its Destiny Bond block (closes the E2 batch 6 open point "Grudge at :40", e2-mechanics-ledger.md
  "Around the pivots"). (2) After a paused U-turn's list the action goes on to LOOP_FAINTED with the defender
  still flagged (SYSCTL_MON_FAINTED, battle_script.c:1770-1771, cleared only by PlayFaintAnimation, :1799),
  so TryGrudge would run again with the attacker battler's newcomer (moveSlot is per battler). NOT built: the
  paused state has already reset the fainted defender's volatiles, so the bit is gone at the resume; owed
  (Owed 1 below). (3) A reflected Magic Coat pass and Pursuit's Struggle cannot meet it (a
  reflected move is a status move; Pursuit is not Struggle); `try_grudge` takes `struggling` for the
  fight command's own Struggle. (4) The reference's `_ingrain_anchors` `others` has `sd.grudge`; no both-engine test: Roar's -6 priority
  makes the foe move after the holder's own attempt, which clears the bit (engine-only test above).
  Not built: Owed 1 (added): Grudge's second TryGrudge on a U-turn newcomer (needs the fainted defender's bit
  kept in the pause scratch, `ts_*`). Owed lines naming Grudge before: none. Doubles: `try_grudge` tests the side of the attacker against the
  fainted battler's (bs.c:6750, so a fainting partner's attacker on the same side is excluded) and the slot is
  per battler; the reference's `_try_grudge` is singles. Not modelled, message only: "{0}'s {1} lost all its PP
  due to the grudge!", "{0} wants the foe to bear a grudge!".
  Tests: tests/engine/test_e3_batch2.py (`test_grudge_*`, `test_a_ko_zeroes_the_ko_moves_slot_only`,
  `test_the_users_next_move_attempt_clears_grudge`, `test_a_user_that_cannot_move_still_loses_the_bit`,
  `test_a_faint_outside_a_move_takes_no_pp`, `test_a_destiny_bond_ko_leaves_no_attacker_for_grudge`,
  `test_an_invokers_own_slot_is_zeroed_not_the_called_moves`, `test_struggle_is_never_avenged`,
  `test_a_u_turn_ko_zeroes_the_users_slot_before_it_switches`, `test_a_pursuit_ko_of_a_switching_grudge_
  user_zeroes_the_pursuers_slot`, `test_the_attackers_leppa_misses_the_zero_until_the_turn_end`,
  `test_a_mimicked_slot_loses_only_the_battlemons_pp_to_grudge`, `test_a_switch_clears_grudge_and_the_
  ingrain_test_reads_it`), tests/ai/test_kaizo_policy.py (`test_grudge_has_no_ai_label_and_two_table_entries`,
  `test_expert_encore_encourages_a_players_last_grudge`); `fullset`-free.

- E3-B2-10 Miracle Eye (#216, BATTLE_EFFECT_IGNORE_EVATION_REMOVE_DARK_IMMUNE; move 357, the only move with
  effect 216 in either game (the ROM's effect, discrepancies.md: the sheet says 217), acc 0, PP 40,
  RANGE_SINGLE_TARGET, PROTECT and MIRROR_MOVE flags; player-only in Kaizo, Metronome-callable): new
  `EffectId.MIRACLE_EYE` (138, `tools/effect_map.py` entry 216); new per-battler `miracle_eye [B]` int8
  (`MOVE_EFFECT_MIRACLE_EYE`, moves.h:30, a bit with no count) in `VOLATILE_FIELDS` (cleared at switch and faint,
  lib.c:2071, :2166), not in `BATON_PASSED_FIELDS` (moves.h:46-48), in `effects.MOVE_EFFECT_MASK_FIELDS` (Roar's
  whole-mask Ingrain test; the reference's `_ingrain_anchors` list has it too). No `Outcome` field, no draw.
  Tables: `move_effect` of Miracle Eye only; no manifest bump (no array or key changed); vanilla and Kaizo tables
  rebuilt.
  Script: effect_script_0216.s:5 an indirect ON_HIT `subscript_miracle_eye.s`, :6 SYSCTL_IGNORE_IMMUNITIES.
  The subscript (:5) sets the bit on BTLSCR_DEFENDER with no jump: no failure when it is already set, no
  Substitute test (the move goes through a Substitute). The ON_HIT subscript runs unless
  MOVE_STATUS_NO_EFFECTS (`BattleSystem_TriggerSecondaryEffect`, lib.c:1524-1529): a Protect or a
  semi-invulnerable target stops it, a miss with acc 0 cannot happen. `effects._miracle_eye` is the
  `_foresight` shape (`h.landed`, the target's side). SYSCTL_IGNORE_IMMUNITIES is inert: CheckTypeChart skips
  power-0 moves (bcp.c:2352-2357); the effect joins `effects.IGNORES_TYPE_CHECKS` and the reference's lists
  for the record (`RESIST_BERRY_SKIPS`, the super-effective clear).
  Reads of the bit, all through `MON_IS_IDENTIFIED` (common.h:48-49: Foresight volatile or this bit) or
  `BasicTypeMulApplies` (lib.c:2551-2555):
  (1) Accuracy: `CheckMoveHitAccuracy` (bcp.c:2888-2890): a positive evasion stage counts as 0 for every
  attacker of the holder. `outcome._rate` (the live and grid threshold), `outcome.stage_total` (Future Sight's
  hit) and the reference's `_hit_rate` read `foresight | miracle_eye`; `hit_thresholds`' "free" clause (the stage
  sum takes every value when a move that can change it is in the turn) names `MIRACLE_EYE` like Foresight.
  (2) Type chart: `BasicTypeMulApplies` skips a row whose defending type is Dark with multiplier 0, i.e. Psychic
  hits a Dark defender at the other type's multiplier (pure Dark x1, Dark/Poison x2). It is a flag of its own:
  `damage.CHART_EYED` (4, beside `CHART_IDENTIFIED` 1 for the Ghost rows after the 0xFE separator,
  lib.c:2611-2621, which only Foresight and Scrappy lift, and `CHART_GROUNDED` 2), set by `damage.identified`,
  read by `chart_entry`; the reference's `_identified` / `_chart_rows` flag 4; `switching.chart_multipliers`'
  `eyed` (a fifth value of `defender_chart_flags`, passed through `type_chart_flags`). Only `ApplyTypeChart`
  reads it, not `CalcEffectiveness` (lib.c:2681+): `calc_effectiveness_flags` (the bench's candidate scoring)
  is unchanged. The AI's `ApplyTypeChart` call sites (ai.c:1281, :1320, :3096, :3306, :3394, :3403, :3582,
  :3609) all go through `damage.identified` or `defender_chart_flags`.
  (3) `damage.ai_damage`'s fixed-damage immunity test (:3106, Psywave is Psychic) used `foresight` alone: it
  now reads the full `identified` flags (Foresight, Scrappy, Ingrain / Iron Ball / Gravity, Miracle Eye), which
  changes nothing for its types (Normal, Fighting, Ghost, Dragon, Psychic) but the Miracle Eye case.
  Enumeration: `INPUT_NEUTRAL_EFFECTS` and `NON_TABLE_DAMAGE_EFFECTS` (a status move on the foe's flag: the only
  attacker of the target this turn is the user, who has acted). No `INPUT_CLAUSE_EFFECTS` or `CRIT_ODDS_EFFECTS`
  entry. No accuracy / evasion write (`test_accuracy_classes._NO_ACC_EVA_WRITE`); the identified evasion is the
  `free` clause above. `tests/strategies.py` draws the move and the bit (`_draw_miracle_eye`, `RARE_EFFECTS`).
  Observation: `observe._NONZERO_SIDE` "miracle_eye" ("{0} identified {1}!", subscript_miracle_eye.s:6),
  `plan_tree._COND_TEXT`.
  AI: Basic_CheckMiracleEye (script.s:1031-1034, label entry :255): `IfMoveEffect AI_BATTLER_DEFENDER,
  MOVE_EFFECT_MIRACLE_EYE` -10 (`commands.MoveEffectMask.MIRACLE_EYE` = 1 << 22, read from `miracle_eye`;
  `policy._basic_check_miracle_eye`). Expert_MiracleEye (:4805-4827, label entry :1760;
  `policy._expert_miracle_eye`): a Dark defender type 1 or 2 goes to the extra gate (the 80 roll :4820, to the end
  when it jumps) and then _ScorePlus2; else an evasion stage above 8 (decomp units: +3 or more) goes to
  _ScorePlus2; else -2 and end. _ScorePlus2 is a second 80 roll (:4823), to the end when it jumps, else +2: a
  Dark defender gets +2 at (176/256)^2. `SITE_N` 4820 and 4823 (80 each); 216 joins `KAIZO_TRANSCRIBED`,
  `KAIZO_EXPERT_LABELS` (`("Expert_MiracleEye", 1760)`) and `KAIZO_EXPERT_TRANSCRIBED`;
  `commands.BattleEffect.IGNORE_EVATION_REMOVE_DARK_IMMUNE`. The only table naming 216 is
  Expert_Encore_EncouragedMoveEffects (:3266, already transcribed). Refusals removed: Miracle Eye left
  `switching.MOVE_IDS`, `_MOVES`, `PLAYER_MOVE_REFUSALS` and `policy._STATE_MOVE_IDS` (the three tables are empty
  now, and out of `game_constants` and its test's `LISTED`; the machinery stays for a later refusal);
  `player_move_refusal` is None for it. Read bounds regenerated (216: 2 reads, one per roll; two rows each);
  backlog regenerated: the rows "move effect #216: Miracle Eye" and "AI reads your move: AI switching: move Miracle
  Eye ..." (Roark Split) close, and the Metronome row's unmodelled count drops from 3 to 2.
  Rulings. (1) The sheet's "Not modelled: 216 absent from _EFFECTS" is closed by the entry; the other readers it
  did not list (`damage.ai_damage`'s immunity test, `line_search`'s Thunder Wave check) now read the full
  `identified` flags. (2) AI switching reads the bit on the player's battler only, set by the player's own Miracle
  Eye in a fight: it never changes the AI side's own reads. (3) A Substitute does not block; a Protect or
  semi-invulnerable target does (tests). (4) Doubles: the bit is on the target battler (`h.target_side`, the chosen
  defender); every attacker of it reads it, the partner included, through `identified(atk_side, def_side)` and
  `outcome._rate(target)`: nothing computes `1 - side`; the AI's `defender_chart_flags` reads the player's
  battler it is passed.
  Not built: Owed lines naming Miracle Eye: none. Not modelled, message only: "{0} identified {1}!".
  Tests: tests/engine/test_e3_batch2.py (`test_miracle_eye_*`, `test_the_decomp_scripts_are_as_the_sheet_says`,
  `test_protect_and_a_semi_invulnerable_target_stop_it`, `test_an_identified_target_ignores_positive_evasion_in_
  every_accuracy_read`, `test_the_dark_immunity_is_skipped_for_an_identified_defender`, `test_a_psychic_hit_on_an_
  identified_dark_foe_deals_damage_in_a_step`, `test_a_switch_clears_miracle_eye_and_the_ingrain_test_reads_it`),
  tests/ai/test_kaizo_policy.py (`test_miracle_eye_tables_are_the_decomps`, `test_miracle_eye_basic`,
  `test_expert_miracle_eye`, `test_the_ai_reads_the_players_miracle_eye_bit_in_the_chart`,
  `test_miracle_eye_is_scored_not_refused_and_exact_equals_loose`, and the batch13 exact-equals-loose list);
  `fullset` edits only: tests/ai/test_switching.py (the Miracle Eye refusal case dropped),
  tests/prep/test_pool.py (`test_player_move_refusal`), tests/data/test_game_constants.py (the three emptied
  tables out of `LISTED`).
- E3-B2-11 Last Resort (#246, BATTLE_EFFECT_FAIL_IF_NOT_USED_ALL_OTHER_MOVES; move 387, the only move with effect
  246 in either game, 130 BP Normal physical, acc 100, PP 5, flags CONTACT, PROTECT, MIRROR_MOVE, KINGS_ROCK; no
  Kaizo trainer knows it; player-only, Metronome-callable): new `EffectId.LAST_RESORT` (139, `tools/effect_map.py`
  entry 246), `ids.MOVE_LAST_RESORT` (387, also in `data/game_constants.py`); new per-battler `last_resort_count [B]`
  int8 (`lastResortCount`, 3 bits, battle_mon.h:27) and `last_resort_moves [B, 4]` int16 (`lastResortMoves`,
  battle_mon.h:50), both in `VOLATILE_FIELDS` (the whole `moveEffectsData` is zeroed at a switch and a faint,
  lib.c:2100-2104, :2166), not in `BATON_PASSED_FIELDS` (the list at :2107-2115 has neither), not in
  `effects.MOVE_EFFECT_MASK_FIELDS` (they are not `moveEffectsMask` bits: no Roar-anchor read). No `Outcome` field,
  no draw. Tables: `move_effect` of Last Resort only; no manifest bump (no array or key changed); vanilla and Kaizo
  tables rebuilt.
  Script: effect_script_0246.s:5 `TryLastResort` (fail to :10-11 MOVE_STATUS_FAILED), then :6-7 CalcCrit,
  CalcDamage: a plain hit behind one test. `BtlCmd_TryLastResort` (bs.c:7567-7579) fails when
  `lastResortCount < numMoves - 1 || numMoves < 2`, `numMoves = Battler_CountMoves` (lib.c:3461-3471: the index of
  the first empty slot of the BattleMon's own moves, so a Mimic copy counts as its slot and Last Resort itself need
  not be in a slot when Metronome calls it). `effects.last_resort_fails` (step's `failed`, the Focus Punch place;
  reference `_script_fails`), read after the recording below, so the user's own earlier moves count.
  Recording: `BattleSystem_UpdateLastResort` (lib.c:3443-3459), called at BeforeMove's tail only on the
  not-`MOVE_STATUS_NO_EFFECTS` branch (bcp.c:3209-3218). `effects.last_resort_update` at the Metronome count's site in
  `step._use_move` (`head_on & succeeded & ~reflect`) with `& ~no_pp` (DecrementPP's MOVE_STATUS_NO_PP is a NO_EFFECTS
  status, bcp.c:2285, so a no-PP move records nothing while the Metronome count still follows it, G-60); the
  reference at its `_metronome_update` site (`_update_last_resort`, guarded by `struggling or locked or pp[action]`).
  It records `moveTemp` (`move`: the chosen or encored move, an invoker itself, never the move it calls; Struggle's
  id, bcp.c:1867, and a locked move's own id) unless it is Last Resort, the count is 4 (:3445-3448) or the id is among
  the first `count` entries (:3451-3455); a miss or a Protect later in the action still counts (the tail comes before
  the accuracy check). Disrupted moves (sleep, freeze, flinch, paralysis, confusion self-hit, Truant, Taunt, Imprison,
  Disable, Gravity, infatuation, a Bide that fails to unleash), a no-target move and a Magic Coat first pass (bcp.c:
  3064-3085) return before the tail and record nothing. The holder's reflected pass reaches the tail with the mover's
  move as moveTemp (the same site and moveTemp the Metronome count uses), so the holder lists the reflected move.
  Zeroing the count: `effects._transform` (bs.c:5340, whenever Transform lands) and `effects._mimic` when the copied
  move is Last Resort (bs.c:4514-4516; the Mimic was recorded before its script, so a Mimic of Last Resort ends at
  0, any other copy keeps the +1). The ids of `last_resort_moves` stay after a zeroing, as in the decomp: only the
  first `count` entries are read.
  Enumeration: `enumerate.INPUT_NEUTRAL_EFFECTS` and `TABLE_DAMAGE_EFFECTS` (a plain hit that can fail, like Focus
  Punch: its failure reads only the user's own count and moves, which nothing the foe does within a turn changes: a
  foe's Mimic, Transform and Sketch change the foe's moves only); `test_accuracy_classes._NO_ACC_EVA_WRITE`;
  `tests/strategies.py` draws the move and a history (`_draw_last_resort`, `RARE_EFFECTS`). No observation entry (no
  message of its own: the failure is the ordinary "But it failed!").
  AI: Basic_CheckLastResort (script.s:1420-1426, entry :275): `IfCanUseLastResort ATTACKER` (new
  `commands.if_can_use_last_resort`, trainer_ai.c:2030-2042: `lastResortCount >= numKnownMoves - 1 &&
  numKnownMoves > 1` on the battler's true moves, the mechanic's test negated) to the end when it can, else -10
  (`policy._basic_check_last_resort`). Expert_LastResort (:5953-5971, entry :1789; `policy._expert_last_resort`):
  immune, half or quarter damage -1 (the move's chart class, `_not_resisted`), else +1 when the AI can use it, else
  0; no roll, so no `SITE_N` entry. 246 joins `KAIZO_TRANSCRIBED`, `KAIZO_EXPERT_LABELS`
  (`("Expert_LastResort", 1789)`), `KAIZO_EXPERT_TRANSCRIBED`; `commands.BattleEffect.
  FAIL_IF_NOT_USED_ALL_OTHER_MOVES`. No table names 246 (script.s has the effect at :275 and :1789 only). Read bounds
  regenerated (246: 1 read; four rows and four drawn rows); backlog regenerated: the row "move effect #246: Last
  Resort" (Roark Split) closes and the Metronome row's unmodelled count drops from 2 to 1.
  Rulings. (1) The sheet's section 7 "missing entirely" is closed by the entry, the state and both labels. (2) The
  no-PP exclusion is the one difference from the Metronome count's condition (above); the encored move with no PP
  (B10-9) and a Spite-emptied slot (E3-B2-4) are the ways to reach it. (3) Struggle records MOVE_STRUGGLE (165), which
  counts toward `numMoves - 1` like any id, so a user that struggled enough times (a history of 4 distinct ids) can use
  Last Resort without having used every move it knows. (4) Doubles (docs/doubles-state-design.md section 5): both
  fields are per battler (`[B]`, leading axis the battler); `last_resort_update`, `last_resort_fails` and
  `if_can_use_last_resort` take the battler (`side` here, the battler's index) as an argument and no code computes
  `1 - side`; nothing is side-wide; the target is chosen data and plays no part.
  Not built: Owed 1 (added): Sketch's zeroing (bs.c:4876-4878, `msgMoveTemp == MOVE_LAST_RESORT` after the learned
  move is written) needs the Sketch move (backlog row "move effect #95: Sketch"), which is not modelled: when it is
  built it must zero `last_resort_count` when the learned move is Last Resort. Owed 2 (added): Snatch (still owed,
  Owed 1 of E3-B2-3): a stolen move returns from BeforeMove at the Snatch pass (bcp.c:3086-3112) before the tail, so
  `last_resort_update` must be skipped there like the Metronome count. Owed lines naming Last Resort before: none in
  this ledger; the G-69 line of e1-mechanics-ledger.md ("Transform ... clears Last Resort's count") is built here.
  Not modelled, message only: the failure text.
  Tests: tests/engine/test_e3_batch2.py (`test_last_resort_*`, `test_a_miss_still_counts`, `test_struggle_is_recorded_
  by_its_own_id`, `test_a_full_list_records_nothing_more`, `test_no_pp_and_a_sleepers_move_record_nothing`,
  `test_an_invoker_is_recorded_not_the_move_it_calls`, `test_mimic_copying_last_resort_and_transform_zero_the_count`,
  `test_a_reflected_move_is_recorded_for_the_magic_coat_holder_only`, `test_a_switch_clears_the_history_and_baton_pass_
  does_not_keep_it`, `test_the_history_is_per_battler_and_indexed_by_the_battler`, each stepped through both engines),
  tests/ai/test_kaizo_policy.py (`test_last_resort_tables_are_the_decomps`, `test_if_can_use_last_resort_is_the_
  mechanics_test`, `test_last_resort_basic`, `test_expert_last_resort`, `test_last_resort_is_scored_not_refused_and_
  exact_equals_loose`, and the batch13 exact-equals-loose list, a slow test run on the pod).
- E3-B2-13 Feint (#223, BATTLE_EFFECT_REMOVE_PROTECT; move 364, the only move with effect 223 in either game, Normal
  physical, acc 100, PP 10, priority +2, vanilla 50 BP and Kaizo 80 BP, flags none: no CAN_PROTECT, CONTACT or
  KINGS_ROCK; no Kaizo trainer knows it; player-only; Metronome cannot call it, lib.c:7175): new `EffectId.FEINT` (140,
  `tools/effect_map.py` entry 223). No new `BattleState` field, no `Outcome` field, no draw beyond the plain hit's crit
  and roll. Tables: `move_effect` of Feint only; no manifest bump (no array or key changed); vanilla and Kaizo tables
  rebuilt.
  Script (effect_script_0223.s): :5 `TryFeint _008` (fail to :11-13, MOVE_STATUS_FAILED), :6 an indirect
  `ON_HIT | SUBSCRIPT_PTR_FEINT`, :7-8 CalcCrit, CalcDamage. `BtlCmd_TryFeint` (bs.c:7518-7528) fails the move unless
  the defender's `turnFlags.protecting` is set. `step._use_move` fails it with the other script failures
  (`failed |= FEINT & ~foe_protecting`, the Fake Out place: the accuracy roll is still drawn, no crit or roll;
  reference `_fight`'s `feint_fails`); `foe_protecting` is the argument the Protect block already had
  (`read["protect_ok"][foe]`). Feint has no CAN_PROTECT, so `CheckMoveHitOverrides` (bcp.c:3007-3014) never blocks
  it, and a semi-invulnerable foe is missed as usual (bcp.c:3034-3040).
  The clear: subscript_feint.s:20 `SetTurnFlag DEFENDER PROTECTING 0` runs as the secondary ON_HIT effect, i.e. unless
  `MOVE_STATUS_NO_EFFECTS` (`BattleSystem_TriggerSecondaryEffect`, lib.c:1524-1529): a hit that was not a miss, not
  ineffective (a Ghost foe, Wonder Guard) and not failed. Shadow Force's attack turn runs the same subscript
  (effect_script_0272.s:21, before its CalcCrit and CalcDamage, also on the Power Herb path), so it clears the flag
  too (`step._feint_clears`: effect FEINT or the `MoveFlag.SEMI_SHADOW` charge flag; reference `_move_effect`). In `step`
  the hit loop's new carry `unprotect` is `landed & info.effective & _feint_clears` (`effective` is
  `MoveHit.effective`: damage > 0, a Substitute hit counts), returned as `ctl["unprotect"]`; `_execute_actions` zeroes the target's
  `protect_ok` read (`read["protect_ok"][foe]`, a merge cannot write a 0) while the shown event stays; the reference
  does `turn.protecting[target] = False` in `_move_effect`. The flag is what the end of the turn's Future Sight check
  (`_future_sight`'s `protecting`) and a battler acting later read; in singles the Feint user is always the later
  actor (+2 under Protect's +3), so no later reader exists but the end of the turn, and both Future Sight and Doom
  Desire have no CAN_PROTECT flag (res/moves/*/data.json): the clear is unobservable in singles play. It is
  modelled whole, because doubles has later readers (the partner of the Feint user hitting the same target).
  `test_a_feint_hit_clears_protecting_for_the_rest_of_the_turn` makes it observable with a stored Tackle.
  Success needs `turnFlags.protecting`, which only a Protect or Detect that passed its chain roll and acted first
  sets (bs.c:5174-5177): a failed Protect, Endure (`turnFlags.enduring`, a different flag) and no Protect all fail
  Feint. Sleep Talk (priority 0) calling Feint reads the same flag (`used` is the called move).
  Enumeration: `enumerate.INPUT_NEUTRAL_EFFECTS` and `TABLE_DAMAGE_EFFECTS` (a plain hit that can fail, like Last
  Resort; the foe's Protect roll is already an enumerated read, `protect[s]`, and a Feint's own reads stay
  over-approximated, which the relevance masks allow: Feint's crit, roll and hit are read in rows where the Protect
  failed and Feint fails, merged after `step`); `test_accuracy_classes._NO_ACC_EVA_WRITE`; `tests/strategies.py` draws
  the move beside a Protect (`_draw_feint`, `RARE_EFFECTS`). No observation entry (the failure is the ordinary "But it
  failed!").
  AI: no Basic entry. Expert_Feint (script.s:4892-4950, entry :1766; `policy._expert_feint`): a player knowing a
  Protect-effect move (`IfMoveEffectKnown`) goes on, else the 64 roll (:4912) jumping goes on, else it ends; the AI
  badly poisoned, cursed, perishing, infatuated, seeded or drowsy goes to the +1 block (+1 unless the 128 roll :4927
  jumps); else a player at full HP or holding an item (as the AI knows it, `ai_item_known`) whose hold effect is not
  Leftovers' or Black Sludge's goes to the chain, and a wounded holder of either falls into the +1 block; the chain of
  the player (`LoadProtectChain DEFENDER`): 0 and, by the fall-through, exactly 2: +1 unless the 128 roll (:4937)
  jumps; 1: +1 unless the 192 roll (:4942) jumps; above 2: -2. New `commands.BattleEffect.REMOVE_PROTECT`,
  `commands.FEINT_GRADUAL_RECOVERY_EFFECTS` (69, 109: hold effects, line - 1 of generated/item_hold_effects.txt,
  checked by name in `test_feint_tables_are_the_decomps`; the other reads of 223, the Encore table :3269 and
  Expert_Protect's `IfMoveKnown MOVE_FEINT` :3536, were transcribed before); four `SITE_N` entries; 223 joins `KAIZO_TRANSCRIBED`,
  `KAIZO_EXPERT_LABELS` (`("Expert_Feint", 1766)`), `KAIZO_EXPERT_TRANSCRIBED`. Read bounds regenerated (223: 3
  reads on the Expert path; four rows and four drawn rows); backlog regenerated: the row "move effect #223: Feint"
  (Roark Split, player-only) closes, 22 to 21 groups.
  Prep: Expert_Feint reads the player's HP percent with `IfHPPercentEqualTo 100` (:4922), the first equal check the
  prep regions meet (tests/prep/test_variants.py's pins ask for it): `variants.AI_EFFECT_DEFENDER_HP_CHECKS[223] =
  (("equal", 100),)` and `variants.hp_cuts` cuts an equal or not-equal check at both edges of its band (the highest HP
  below it, as a "less" check, and its own top, as a "greater" check): for 100 % the region top max_hp - 1
  (`test_feint_adds_the_full_hp_band_to_the_prep_cuts`).
  Rulings. (1) The sheet's section 7 "missing entirely" is closed by the effect id, the gate input that already
  existed (`foe_protecting`) and the new clear. (2) The comment of Expert_Feint says "otherwise 50 % chance"; the code
  is transcribed, not the comment: a chain of exactly 2 falls into `_NoProtectChain` (the first `IfLoaded` tests are
  0, 1 and "greater than 2"), and the "+1 for a wounded Leftovers holder" is the fall-through of
  `IfLoadedNotInTable` into `_TryScorePlus1`. (3) The ON_HIT clear is tied to `effective` (damage > 0), not to
  `landed`: a Ghost target (Normal 0x) is MOVE_STATUS_INEFFECTIVE, a NO_EFFECTS status, and keeps its flag. (4)
  Doubles (docs/doubles-state-design.md section 5): the flag is a fact about the named target, read and cleared per
  battler (`read["protect_ok"][foe]`, the argument `foe_protecting`); nothing computes "the other side" inside the
  mechanic beyond `_execute_actions`' existing `_foe(actor)`, which the doubles milestone replaces with the chosen
  defender, and the clear then reaches the partner that moves later.
  Not built: Owed lines naming Feint: none. Nothing owed: Shadow Force's clear, which the E2 batch 10 entry left out
  as unobservable in singles, is built here with Feint's (above). Not modelled, message only: "{0} fell for the
  feint!" / "It broke through {0}'s protection!" (subscript_feint.s:5-17).
  Tests: tests/engine/test_e3_batch2.py (`test_feint_*`, `test_a_feint_*`, `test_a_called_feint_reads_the_same_flag`,
  each stepped through both engines; `test_shadow_force_clears_protecting_with_its_attack_turn`;
  `test_feint_enumeration_matches_the_brute_force` is slow), tests/ai/test_kaizo_policy.py
  (`test_feint_tables_are_the_decomps`, `test_expert_feint`, `test_feint_is_scored_not_refused`, the slow
  `test_feint_exact_equals_loose`, and the batch13 exact-equals-loose list, a slow test run on the pod).
- E3-B2-14 Defog (#258, BATTLE_EFFECT_REMOVE_HAZARDS_SCREENS_EVA_DOWN; move 432, the only move with effect 258 in either
  game, Flying status, acc 0, PP 15 (Kaizo 1), flags PROTECT and MIRROR_MOVE, no Kaizo trainer knows it; player-only,
  Metronome-callable) and the Metronome close-out (#83, E3-B2-14): new `EffectId.DEFOG` (141, `tools/effect_map.py`
  entry 258 with params `(Stat.EVA, -1, 0)`). No new `BattleState` field, no `Outcome` field, no draw. Tables: `move_effect`
  and `move_param` of Defog only; no manifest bump (no array or key changed); vanilla and Kaizo tables rebuilt.
  The sheet e258_defog.md was built here by the same group because Metronome's close-out needs it: it was the last
  Kaizo-callable move still refused (the probe over `coverage.move_effect_line` listed exactly Defog, 432).
  Script: effect_script_0258.s:5 is a direct `ON_HIT | TO_DEFENDER | SUBSCRIPT_PTR_DEFOG`, a primary effect
  (`BattleSystem_TriggerPrimaryEffect`, lib.c:1481-1491: no HP test, not `MOVE_STATUS_NO_EFFECTS`). subscript_defog.s:
  :5-13 only picks whether the attack message plays first (a screen, Mist, Safeguard, hazard on `side_of(t)` or deep
  fog); :19-20 evasion -1 on `t` through subscript_update_stat_stage.s (`effects._foe_stage`: Mist :2877, Clear Body /
  White Smoke :2883-2884 (Mold Breaker pierces), -6 :2904, Substitute :2921-2923; Keen Eye / Hyper Cutter name other stats,
  Shield Dust is indirect-only); then, whatever the drop did, Reflect, Light Screen, Mist, Safeguard, Spikes, Toxic
  Spikes and Stealth Rock of `side_of(t)` cleared (:21-61; Mist after the drop met it) and the field's deep fog
  (:64-69, `weather` FOG -> none, timer 0). `effects._defog` (second dispatch tuple); the user's side is never touched.
  Failure: a drop that reaches the Substitute test sets MOVE_STATUS_FAILED (subscript_update_stat_stage.s:60), which is in
  `MOVE_STATUS_NO_EFFECTS` and skips AfterMoveEffects: `effects.defog_stage_fails` joins `primary_effect_fails` (Mist,
  Clear Body / White Smoke and -6 are `result == 1` or a return: no failure). That failure branch prints no message, so
  with nothing to clear the attack message is never shown and moveHit / movePrev / Choice lock are not updated
  (`effects.defog_silent_fail`, `step._use_move`'s `shown_t`; reference `_defog_fails`, `turn.defog_failed`).
  Hit chance: acc 0, no roll; a faster foe's Defog clears the fog before the side's check, so
  `outcome._threshold_grid` has the foe-Defog fog variant (its own evasion drop is the side's, read only by the foe's
  hit). Damage inputs: Defog never changes a hit's inputs but through the fog (Weather Ball power, SolarBeam halving):
  `enumerate.INPUT_CLAUSE_EFFECTS` (clause `foe Defog & fog at the turn start`) and `NON_TABLE_DAMAGE_EFFECTS`.
  AI: Basic_CheckDefog (:1476-1499, `policy._basic_check_defog`: evasion not -6, Light Screen / Reflect or deep fog end
  it; no living party member for the player -10; no hazard -10) and Expert_Defog (:6125-6188, `policy._expert_defog`, the
  code: screens -> _ScreenScrubbing, hazards -2, then the HP / evasion block; rolls :6171 (128) and :6180 (50)); new
  `commands.BattleEffect.REMOVE_HAZARDS_SCREENS_EVA_DOWN`, `commands.if_stat_stage_not_equal_to`, two `SITE_N` entries,
  258 in `KAIZO_TRANSCRIBED`, `KAIZO_EXPERT_LABELS`, `KAIZO_EXPERT_TRANSCRIBED`; the SetupFirstTurn and Harrassment
  tables already held it. Prep: Expert_Defog's `IfHPPercentGreaterThan DEFENDER 70` (:6184) is
  `variants.AI_EFFECT_DEFENDER_HP_CHECKS[258]`. Read bounds regenerated; backlog regenerated.
  Metronome close-out: no code of Metronome's own changes (`order.metronome_candidates`, `coverage.metronome_calls`).
  With Defog the Kaizo callable set is fully modelled: `move_effect_line` passes every Kaizo Metronome holder
  (`test_every_kaizo_metronome_call_is_modelled`, `test_a_kaizo_metronome_holder_passes_the_party_check`, both
  `fullset`). The batch-1 and group 1-13 Owed lines naming Metronome's callable list close. Rulings: (1) the called
  move's range is `RANGE_SINGLE_TARGET`; the call through `GoToMoveScript` is the existing invoker path. (2) Doubles
  (docs/doubles-state-design.md section 5): clears index `side_of(target_side)`, the failure helpers take
  `target_side`; the single-target call's random opponent draw (lib.c:1685ff) and the partner's loss of the side's
  screens belong to the doubles milestone's `Outcome.call` form and its chosen-defender resolver, which replace
  `_foe` where they exist; nothing in the mechanic computes `1 - side`.
  Not built: vanilla's 55 callable unmodelled moves (the second family: Heal Block's `Move_CanBeMetronomed` test comes
  with it, order.py note); not a Kaizo case. Message only: the "blew away" texts (subscript_defog.s).
  Tests: tests/engine/test_e3_batch2.py (`test_defog_*`, `test_mist_stops_the_drop_but_is_cleared_after_it`,
  `test_clear_body_and_stage_minimum_stop_the_drop_not_the_clears`, `test_metronome_calling_defog_*`,
  `test_every_kaizo_metronome_call_is_modelled`, `test_a_kaizo_metronome_holder_passes_the_party_check`),
  tests/engine/test_accuracy_classes.py (Defog in `_FOLLOWED`), tests/ai/test_kaizo_policy.py (`test_defog_*`,
  `test_if_stat_stage_not_equal_to_negates_the_equal_command`, `test_expert_defog`, `test_defog_basic`, the slow
  `test_defog_exact_equals_loose`, and the batch13 exact-equals-loose list, a slow test run on the pod).
