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
