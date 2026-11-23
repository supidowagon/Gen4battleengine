# E4 ledger: rulings of the Tag Strategy build (E4-A0)

Rules sheets: `_scratch/e4/rules/A-8.md`, `A-10.md`. Decomp paths are relative to
`external/pokeplatinum`; `ai` is `src/battle/trainer_ai/trainer_ai.c`.

## T-1 to T-5 (commands)

- **E4-1. Battler arguments.** `commands.Battler` has the four decomp values
  (`include/constants/battle/trainer_ai.h:28-31`). `commands._side` is the one resolver
  (`AIScript_Battler`, ai:2765-2782): attacker 1, defender 0, the partner arguments `^ 2`.
- **E4-2. A battler the format lacks (`b >= B`) is the zeroed `BattleMon`** (`commands.CLEARED`,
  `_cleared`): `BattleContext_New` clears the context (`battle_controller_player.c:180-182`) and
  `InitBattleMons` fills `i < maxBattlers` only (:223-227). hp% 0 (G-53), ability 0, species 0
  (abilities 0, 0 in both ROMs), types Normal twice, status, volatile and move-effect masks 0, stat
  stages stored 0 (decomp units: not the neutral 6), no moves, `heldItem` 0, flash fire 0. The AI
  record of such a battler is 0 (`state.ai_record_*`). This is the game's B = 2 state.
- **E4-3. `GetPartner`** (`battle_system.c:394-411`): `commands._get_partner`, the battler itself
  when not doubles. `LoadTypeFrom` `ATTACKER_PARTNER_TYPE_1` / `_2` (values 6 and 8,
  `generated/ai_load_type_targets.txt`) read it (ai:965-978).
- **E4-4. Partner cases.** `CheckBattlerAbility` branches on the argument (ai:1223): both
  DEFENDER arguments take the knowledge branch, both ATTACKER arguments the raw ability.
  `IfMoveKnown` / `IfMoveNotKnown`: ATTACKER_PARTNER does not jump with `curHP == 0`
  (ai:1650-1653, :1707-1709), DEFENDER_PARTNER has no case (:1678-1679). `IfHeldItemEqualTo`:
  the attacker's side reads the true `heldItem` (ai:1925-1926), the other side the AI record.
  `LoadBattlerAbility`: ATTACKER_PARTNER the raw ability, DEFENDER_PARTNER the knowledge branch
  without a coin when species 0 has abilities 0, 0.
- **E4-5. `LoadBattlerSpeedRank`, B = 2.** One compare of battlers 0 and 1 (`CompareBattlerSpeed(0,
  1, TRUE)`); a partner argument leaves `calcTemp`. The compare still writes `monSpeedValues`
  (battle_lib.c:1372-1373) unless one battler is dead (:1204-1211). A tie is one read of
  probability 1/2 (loose `SPEED_TIE_DRAWN`; a partner argument's draw is no exact read, loose
  `SPEED_TIE`).
- **E4-6. `CheckIfHighestDamageWithPartner`, B = 2** (ai:2369-2436). Two sweeps over the attacker's
  own moves with independent Psywave and Magnitude draws; one `_calc_read` over eight estimates;
  no new `ReadKind` (the digit reads of `CALC_PSYWAVE` / `CALC_MAGNITUDE` and the exact layout's
  one yes/no read cover it). With no drawing id it equals `FlagMoveDamageScore` and makes no read.
- B = 4 reads of these commands (the partner's moveset in the second sweep, the six-compare
  speed rank, a present ally's `LoadBattlerAbility` coin) are E4-A's; the format gate refuses
  B = 4 until then.

## T-6 to T-8 (Tag Strategy, bit 7)

- **E4-7. `ALLIN_READS` 18 -> 21 (c944513e; 22 before the review fix that decides the cleared
  partners' tests and the effectiveness class in `tools/ai_read_bounds.py`).** `layout.ALLIN_READS` is the largest row sum of
  `read_bounds.READ_BOUNDS` and `DRAWN_READ_BOUNDS` over every flag, so adding bit 7 to
  `KAIZO_FLAG_BITS` raised it from 18 to 21 (the row
  `DRAWN_READ_BOUNDS[(23, True, True, True)]`, Tag column 3: IfCurrentMoveKills or the
  CheckIfHighestDamageWithPartner compare, with R@6694); `ALLIN.slot_reads` follows. What changes:
  - Shapes only. `outcome.sample_ai_outcome` draws read column i from `fold_in(k[1], i)` and the
    damage draws of column i from `fold_in(fold_in(key, 0x43414C43), i)`, each of shape `(MOVES,)` /
    `(MOVES, CALC_DRAWS)`, then stacks the columns: `script_rolls` is `uint8[MOVES, 21]` and
    `calc_draws` `uint16[MOVES, 21, CALC_DRAWS]` under `ALLIN` (were 18). Columns 0-17 are drawn
    from the same keys as before, so a fixed-key sample of a fight without bit 7 has the same
    draws in those columns; columns 18-20 are new and are read only by a path that reaches an
    18th to 21st read, which no flag but bit 7 has. The other fields (`switch_stop`, `tiebreak`,
    `sim_roll`) come from `split(key, 4)` and `slot_reads` does not enter them. A fight without
    bit 7 therefore scores and samples as before (a read past `slot_reads` is refused,
    `AiContext.missing_read`, so a larger bound refuses less, never differently).
  - What a snapshot compare sees: arrays of `ALLIN` outcomes (`script_rolls`, `calc_draws`) are
    wider (21 columns), so a stored 18-column array differs in shape although its first 18 columns
    are equal; compare those columns, or regenerate the snapshot. `policy._check_outcome`
    (`shapes["script_rolls"] = (MOVES, layout.slot_reads)`) refuses an outcome of the
    old width passed under `ALLIN`.
  - The exact entry points refuse `ALLIN` (`_exact_layout`), so the exact distributions are
    unchanged; a fight's own layout takes its own bound (`fight_slot_reads`).
- **E4-8. `Sandstorm` (effect 115) is not in `outcome.KAIZO_TRANSCRIBED`.** A bit-7 fight whose AI
  knows Sandstorm stays refused by the effect check (`policy._refused` `move`: the layout does not
  transcribe the effect), not by bit 7; `_tag_strategy`'s Sandstorm block (s:6886-6916) is
  written and tested with the effect added to the layout by hand (tests/ai/test_tag_strategy.py
  `SANDSTORM_EFFECT`). Transcribing 115 is a separate job (its Basic / Expert paths).
- **E4-9. Tag Strategy at B = 4 is refused by the policy.** `policy._refused`: a state with
  `format_of(state).B == 4` under any layout (the flat VANILLA one, which has no Tag transcription, and every Kaizo one) whose `thinking_mask` has bit 7 (the
  trainer's flag or the doubles bit, trainer_ai.c:252-255) sets `policy`, hence `scoring` and
  `any`. Before c944513e `ALLIN` lacked bit 7 and `(mask & ~layout.flags) != 0` refused it;
  with bit 7 in `ALLIN.flags` that test passes, and `_tag_strategy` models B = 2 only (A-8
  sections 1, 5; A-10 sections 2, 3 are E4-A's). Every entry point calls `_refused`, so the one
  refusal covers `ALLIN`, `VANILLA`, `kaizo_layout` / `layout_of` fight layouts (with or without bit 7 in
  their flags), the exact and the loose paths. `coverage.IMPLEMENTED_FORMATS` still refuses
  `double`, `double_two_trainer`, `multi`. Test: tests/ai/test_tag_strategy.py
  `test_tag_strategy_at_b4_is_refused` (`fullset`). To be deleted by E4-A.
- **E4-10. Snapshot compare of 39168d89 against a de494d32 capture: no part sees `ALLIN`'s width
  or a bit-7 trainer; no `UPGRADES` row.** Per part (`tools/snap_compare.py`):
  - `step`, `jitstep`, `sample`, `rest`, `gen`: vanilla tables (`Ctx.tables_for`, :944-951) and
    the engine only (`group_outputs` :960-1003: `step_events`, `ref_step`, `step`,
    `engine.outcome.sample_outcome`, `enumerate_turn`, `decision_actions`, `root_belief`;
    `draw_inputs` :1017-1075 through `tests/strategies.py` and `gated_states`). None calls the AI;
    the diff touches no engine file but `engine/coverage.py` (docstring and the Kaizo row of
    `IMPLEMENTED_AI_FLAGS`, :251), which these parts do not read.
  - `ai`: `ai_outputs` (:1006-1014) stores `ai_action_distribution(tables, s)`
    (`policy.py:6052-6066`), a `float32[NUM_OPP_ACTIONS]`, not an `AiOutcome`, on vanilla
    `gated_states`. Its layout is `layout_of(tables)` (`policy._layout` :5285-5292,
    `layout.py:360-377`) = `VANILLA` (`layout.py:185-191`: flat, `slot_reads` 0, flags `0b111`);
    `_exact_layout` (`policy.py:5322-5331`) would refuse `ALLIN`. No bit 7, no `slot_reads`.
    It does run the shared command code E4-A0 rewrote (`commands._side`, `_defender_abilities(b)`,
    `_row`), at B = 2 with the default battlers, which keeps every vanilla read.
  - `roark.rank`: `rank_actions` on vanilla tables (`roark_steps` :1118-1128,
    `stage_a.py:214` `layout_of` = `VANILLA`); its playouts draw `sample_ai_outcome(..., VANILLA)`
    (`playout.py:571`), the flat branch (`outcome.py:1371-1374`, `calc_draws` width 0); the stored
    ranking holds no `AiOutcome`.
  - `roark.trace`: Kaizo tables, `kaizo_roark_02.json` through `tools/trace_to_state.py`
    (`trace_to_states`, `replay_turn`, `ref_apply_turn`), which imports no `nuzline.ai` module;
    Roark's flags (`KAIZO.flags` `0b10111`, `layout.py:210`) lack bit 7 in any case.
  - The 16 bit-7 singles rows (6 opponents: 49, 50, 187, 646, 725, 898; 10 tag partners: 607, 608,
    609, 610, 612, 619, 620, 861, 863, 865, whose Tag Strategy runs only at B = 4): no part runs
    `coverage` or draws a Kaizo trainer other than Roark.
  So the compare should show `compare: ok` for every part. A divergence in `ai` would be a value
  change of the rewritten shared commands under `VANILLA` (a bug to fix, not a layout upgrade).

- **E4-11. Registry pins of e40a579b (`data/game_constants.py`).** Ability 31 is pinned as
  `Lightningrod`, the name both games' data use. Move 222 is its own constant
  `ai/commands.py MOVE_MAGNITUDE` (out of `MOVE_IDS`), name-checked against vanilla only
  (`vanilla_policy`); `tests/ai/test_tag_strategy.py` pins Kaizo's name: Kaizo names id 222 Bulldoze, and the script compares ids
  (`AICmd_IfMoveEqualTo`, trainer_ai.c:834-844), so Bulldoze takes Magnitude's block. Cause:
  39168d89 pinned names that are not in the data, so `load_game` raised `GameDataError` for both
  games; the review read the code and did not run it. Rule from now on: a tip goes to the pod only
  after it has passed `load_game` of both games and the four guards' fast parts locally.

- **E4-12. B = 2 change: the trainer AI uses no item under Embargo** (E4-A W1). `policy._use_item`
  (`maybe_use_item`) tests the AI battler's `embargo` (ai:4076-4078; Embargo is modelled, E2
  B13-29). Before W1 an Embargoed AI with a usable bag item took the item action; now it takes
  `NO_ITEM` and scores its moves: `TrainerAI_ShouldUseItem` returns FALSE (trainer_ai.c:4075-4078) and
  `TrainerAI_PickCommand` returns `PLAYER_INPUT_FIGHT` (:4034-4038). So the AI's whole action changes on
  such a state, from the one-hot item action to its move distribution (`policy._policy_and_context`,
  `stay = jnp.where(item != NO_ITEM, ...)`), and every output that reads the AI's action law there may
  differ with it: `ai_action_distribution`, `enumerate_turn`, `decision_actions`, `root_belief`,
  `rank_actions` (`tools/snap_compare.py`) and the differential's AI lines. Expected in any compare of
  B = 2 states with the AI under Embargo and a usable item; outside `tests/ai/b2_snapshot.py` (commands
  only). Test: tests/ai/test_policy.py `test_no_item_under_embargo`.

- **E4-13. Snapshot compare of E4-A0.** Against the dd68cd8f capture (snapcmp-e4a0-d1ab9b7): 0 diverged,
  17 generator drift. Cause: Hypothesis's local-constants pool. E4-A0 adds 24 int literals of 100 or more:
  356, 435, 436 and the 21 `SITE_N` script lines. Seeds 114 and 143 draw 6972 and 7012, two of them. The manager
  ruled the pool argument is no proof, because drawn inputs pass through `legal_actions` / `battle_start`. Main
  fd3c1fcc switches the pool off in `draw_inputs` (E1 `ops/hyp-constants-off`). The gate compare of the merge with
  `train/w4` 3445e13c (main c9dffa37 + e2-m4 3af9a9fc) runs against the capture of that main; expected 0 diverged and
  0 generator drift. The coordinator records the result at the gate.
  `ai/read_bounds.py` in this merge is regenerated by `tools/ai_read_bounds.py` on the merged tree, not hand-merged:
  it is the union of E4-A0's bit-7 column and main's Sketch rows (effect 95, 16fe2777); no other row changed on
  either side. Sketch's bit-7 entries equal effect 0's: `TagStrategy_Main` (script.s:6637) has no Sketch case
  (`BATTLE_EFFECT_SKETCH` appears nowhere in script.s), so Sketch takes the generic path.

- **E4-14. Snapshot compare of W1 (train with E4-A0 on `train/w4` 3445e13c).** W1 has had no snapshot compare
  before. Its one B = 2 value change is E4-12: an AI battler under Embargo (`MOVE_EFFECT_EMBARGO`,
  trainer_ai.c:4076) with a usable bag item now takes `NO_ITEM` (`policy.py` `state.embargo[b] == 0` in the item
  mask). Its other changes are commands at B = 4 and helpers; `tests/ai/b2_snapshot.py` pins their B = 2 values.
  Expected against the capture of that main: 0 generator drift; 0 diverged, except two divergence classes, both
  in the AI's action law and every output downstream of it (`ai_action_distribution`, `enumerate_turn`,
  `decision_actions`, `root_belief`, `rank_actions`): (a) E4-12, a captured state whose AI battler is under
  Embargo and has a usable bag item (item action -> move distribution); (b) E4-15, a captured state whose AI
  party has a living bench member with its side's `knocked_off` bit set, scored by `IfPartyMemberDealsMoreDamage`
  (Expert scripts) with a Natural Gift or Judgment move or Klutz. Any other divergence is a defect. `ai/read_bounds.py` regenerated on this merge equals E4-A0's (79d26b70).

- **E4-15. B = 2 change: `IfPartyMemberDealsMoreDamage` passes the party's held item** (W1 review fix,
  review_w1_92eb8c44 MUST 2). The command reads each member's `MON_DATA_HELD_ITEM` (trainer_ai.c:2160); the party
  keeps a Knock Off victim's item while its side's `knockedOffItemsMask` bit is set (battle_io_command.c:590-592),
  which `state.party_held_item` models. Before the fix the engine passed `state.item` (the BattleMon's 0). The item
  reaches only Natural Gift's and Judgment's cases and the Klutz test (`TrainerAI_CalcAllDamage`). At B = 4 the
  same fix reads the DOUBLES partner's active as its party Pokemon (`commands._party_copy`: moves, PP, IVs, ability
  of the `shadow_*` under Mimic / Transform, battle_io_command.c:579-586), for this command and
  `IfAnyPartyMemberUsedPP` (:1999). Tests: tests/ai/test_commands_four.py `test_used_pp_reads_the_party_copy`,
  `test_party_copy_values`, `test_party_member_deals_more_reads_the_party_copy`.

- **E4-16. B = 4 read bounds and the width of the real split fights** (E4-A W2, A-10 sections 2.7-2.9,
  A-15 section 5). `tools/ai_read_bounds.py` walks script.s at B = 4 with the target kind in the key
  (`READ_BOUNDS_B4`, `DRAWN_READ_BOUNDS_B4` in `ai/read_bounds.py`): `IfTargetIsPartner` is decided by the
  kind (trainer_ai.c:2630-2639), an ally path ends at the nine Terminate lines and enters
  `TagStrategy_Partner` at script.s:6638 and :7697 (bit 8 counts the part again), every partner read is
  open (the B = 2 constants of the zeroed partners do not apply), and each `LoadBattlerSpeedRank` call counts
  2 reads (four ranks, two pattern bits). The walk is an upper bound: partner-dependent branches are not
  decided by the actual partner. Measurement (`_scratch/e4/width/report.md`, walker `measure_b4.py`):
  the most reads per slot over the split B = 4 fights (DOUBLES, TWO_TRAINERS, MULTI per partner variant,
  every AI battler with its own flags | Tag Strategy, its party's moves, the whole fight's moves deciding
  the drawing rows) is **14** = 65597 scorings, half `WIDTH_CAP`, set by non-rank paths (trainers 79, 435,
  583). A battler whose partner is the player's (MULTI battler 2; partner battler 0, battle_system.c:394-411)
  sweeps unknown moves that may draw in `CheckIfHighestDamageWithPartner` (trainer_ai.c:2414, :3014-3048), so it
  takes the drawing rows always (`move_read_rows(.., partner_unknown=True)`): 43 (fight, partner) bounds rise by
  1-2, partner 612's to 14 in six fights, the maximum stays 14 (`_scratch/e4/width/partner_drawn_b4.py`,
  engine-free, from the committed tables). One read per tied compare (6 per call) would put 22 nodes over the cap. Pinned by
  `tests/ai/test_layout.py::test_b4_split_width` (`fullset`). Copied moves (A-10 section 2.8,
  `fight_tables(.., fellow_moves=..)`): the pool at B = 4 is every battler the copier can target; a copied
  Future Sight reads 14 (exact); under 0x7ff (trainer 455) a Mimicked Sand Attack, Smoke Screen, Flash or Double
  Team passes the cap and the layout is sampled and marked (`AiLayout.sampled`, `sample_n`, `sample_seed`;
  `BEYOND_CAP_REFUSE` raises instead; `policy._exact_layout` refuses a sampled layout). `WIDTH_CAP` 18 measured
  (run e4-w2width-4723845, `_scratch/e4/width/cap18_cost.md`; `policy._slot_histograms`, every Kaizo bit, B = 2):
  peak RSS 30.4 / 44.5 / 72.6 / 126.8 GB at 14-17 reads, XLA temp doubling per read (14.06e9 to 112.5e9 bytes),
  18 reads killed at 238-254 GB twice (about 225e9 temp per call); a B = 4 battler costs about 3x (one program per
  target). Not cheap: the cap constant stays, and the fights past it (trainer 455 under 0x7ff) stay sampled and
  marked (A-10 section 2.9).
- **E4-17. Doubles damage refusal kept (E4-A W2, W2-16, plan decision 12).** E2's D45-8 doubles factors are not in
  `main`: `damage._pre_random` halves a screen (singles) and has no spread factor. `CheckIfHighestDamageWithPartner`
  reads the right battlers at B = 4, but the factor depends only on `MOVE_DATA(move).range` and the alive counts
  (battle_lib.c:7035-7044), so every estimate that can reach a spread move counts: a state where a move with range
  RANGE_ADJACENT_OPPONENTS (a side of two living battlers) or RANGE_ALL_ADJACENT (three or more living) is a damaging
  active move of an AI battler or its partner (trainer_ai.c:1041, :1560, :1611, :2398-2414), a damaging move of any
  member of an AI battler's party (`IfPartyMemberDealsMoreDamage`, :2142-2165), or any battler's last move with power
  (`IfBattlerDealsMoreDamage`, :2228-2237, no damaging test), or a side of two living battlers has Reflect /
  Light Screen, is a named refusal `policy._doubles_damage`
  ("doubles damage factor (E2 D45-8)", `_Refused.doubles_damage`; battle_lib.c:6982-6990, :7023-7044), with a
  `fullset` test. The step that rebases on D45-8 deletes both and uses the factors.
