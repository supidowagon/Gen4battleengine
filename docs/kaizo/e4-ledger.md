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
