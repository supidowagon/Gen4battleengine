# E2 mechanics ledger (Kaizo Roark split)

Session E2 builds the battle mechanics the Roark split's trainers need (held items,
abilities, move effects, battle terrain), batch by family. The AI side belongs to E1.
Goal: no Roark-split fight blocked by a "move effect", "held item effect", "ability" or
"battle terrain" entry of `docs/kaizo/mechanics_backlog.md`.

Baseline (main e139cd1): 79 such entries (32 held item, 25 move effect, 21 ability,
1 terrain). Cold compile of `jit(step)` on vanilla Roark: 18.7 s (`jit(vmap(step))`, batch
64: 14.6 s), measured with `_scratch/e2/compile_time.py` on G:.

## Batches

1. Attacker damage items and resist berries: the type-boosting table, the species items
   of `BattleSystem_CalcMoveDamage`, Muscle Band / Wise Glasses, the Choice items, Expert
   Belt, Life Orb, every resist berry, the hold-effect-0 items.
2. Status orbs and the other berries and held items.
3. Abilities.
4. Move effects and Secret Power's terrain.

## Rulings

(One line each, with the decomp citation.)

- B1-1 Item order: the held-item block runs before every ability modifier and in the decomp's
  order (type item, Choice Band/Specs, Soul Dew, DeepSeaTooth/Scale, Light Ball, Metal Powder,
  Thick Club, orbs, Muscle Band/Wise Glasses), so e.g. Thick Club x2 truncates before Guts x1.5;
  the u16 stores never truncate in reach (battle_lib.c:6716-6783 before :6793).
- B1-2 Light Ball doubles `movePower` for any move class, physical and special alike
  (battle_lib.c:6748-6751).
- B1-3 The species items read `BATTLEMON_SPECIES`; Soul Dew's Battle Frontier test never holds in
  a trainer battle and the Griseous Orb's Transform test never holds while Transform is
  unsupported (battle_lib.c:6730-6739, :6768-6772).
- B1-4 The confusion self-hit gets the holder's own `BattleSystem_CalcMoveDamage` items (Choice
  Band, a Normal-type item for Struggle's type, Muscle Band, Light Ball, its own Metal Powder /
  DeepSeaScale / Soul Dew as defender), never Life Orb or Expert Belt (`CALC_SELF_HIT`,
  include/battle/common.h:35; battle_controller_player.c:2633).
- B1-5 Life Orb's x1.3 is the move script's (`BattleScript_CalcMoveDamage`, battle_script.c:1346,
  after `criticalMul`): not in the AI's `TrainerAI_CalcDamage` estimate (trainer_ai.c:3081) and not
  on a fixed-damage move (Night Shade / Seismic Toss: effect_script_0087.s has no CalcDamage; Bide);
  Pursuit's switch hit gets it (subscript_pursuit.s CalcDamage).
- B1-6 Life Orb recoil is `Divide(maxHP, 10)` (not the param), once per move after all hits,
  when a hit's `UpdateHP` ran with damage != 0 (`SYSCTL_MOVE_HIT`, battle_controller_player.c:3340,
  :3441), `moveCur` is not a status move and the holder lives: fixed-damage hits recoil too, a hit
  capped to 0 by a 1-HP Focus Sash still counts, the confusion self-hit never does
  (battle_controller_player.c:4758-4772).
- B1-7 Life Orb recoil also follows Pursuit's pre-switch hit unless it did not hit
  (subscript_pursuit.s:17, :48-49, :94-95 -> `BattleSystem_TriggerHeldItemOnPivotMove`,
  battle_lib.c:7357-7366).
- B1-8 Life Orb recoil runs inside the action, before its faint check: a holder that KOs the
  last foe and faints to its recoil ends the battle with both sides out, a loss (ruling 2;
  `TRIGGER_AFTER_HIT_EFFECTS` before `MOVE_END`, battle_controller_player.c:3897-3915).
- B1-9 Choice lock: set at `UpdateMoveBuffers` only when the attack message showed, to `moveTemp`
  (Nature Power itself, not its call), never to Struggle; a non-holder's lock is cleared there; a
  disrupted move (sleep, freeze, flinch, Disable, Taunt, confusion self-hit, paralysis, love)
  changes nothing (battle_controller_player.c:3921-3934).
- B1-10 The defender's lock is cleared in `UpdateFlagsWhenHit` unless it holds a Choice item, when
  the attack message showed (battle_controller_player.c:4489-4496); `BtlCmd_TryPursuit` locks a
  Choice holder into Pursuit (battle_script.c:6943-6950).
- B1-11 A Choice-locked battler whose locked move has no PP (or is disabled) has no usable move and
  Struggles; a lock the battler does not know is cleared, read as no lock
  (battle_lib.c:2307-2315; `STRUGGLING_ALL`, battle_controller_player.c:433).
- B1-12 A lock set on a battler that fainted during its move is never read and is cleared at the
  next switch-in (battle_lib.c:2100-2103); the engine clears it at the faint check.
- B1-13 Choice Scarf's x1.5 Speed comes after Swift Swim and before paralysis (battle_lib.c:1228,
  :1248, :1260).
- B1-14 Expert Belt multiplies a `MOVE_STATUS_SUPER_EFFECTIVE` hit after Solid Rock / Filter, inside
  `BattleSystem_ApplyTypeChart`, so the AI's estimate has it too (battle_lib.c:2657-2665;
  trainer_ai.c:3096).
- B1-15 Items with no hold effect do nothing, except that a berry among them (`Item_IsBerry`,
  src/item.c:357-365) is eaten by Pluck / Bug Bite with no effect, leaving the pluck flag set
  (battle_lib.c:5832-5835, :5848); the prep never offers such a berry (it can only help the foe).
