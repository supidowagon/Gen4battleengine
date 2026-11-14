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
