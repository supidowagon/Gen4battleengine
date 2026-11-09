# K3b rule audit against the Platinum decomp (read-only)

Date 2026-11-08. Worktree `bold-mendeleev-477a50`, branch `claude/kaizo-k3b`, HEAD `577db70` (Task 11 landed while this audit ran).
Decomp paths are relative to `external/pokeplatinum`. Abbreviations: bcp = `src/battle/battle_controller_player.c`, lib = `src/battle/battle_lib.c`, script = `src/battle/battle_script.c`, ai = `src/battle/trainer_ai/trainer_ai.c`, sub = `res/battle/scripts/subscripts/`, eff = `res/battle/scripts/effects/`.

## Summary: WRONG / UNCERTAIN

| # | Where | Rule as written | Verdict | Decomp behaviour |
|---|---|---|---|---|
| W1 | plan Task 11 (:320-322), Review Focus 3 (:59), test name `test_focus_sash_multi_hit_first_only`, survey §6 (:613) | "only a full-HP hit of a multi-hit triggers it" / "survives a multi-hit's first hit only" | **WRONG** (code at 577db70 is right) | `focusItemActivated` is set in `UpdateHP` whenever a damaging hit lands on a full-HP Sash holder, **even if that hit is not lethal** (bcp.c:3386-3393). It is a SelfTurnFlag that is cleared only by `BattleContext_Init` at the end of the action (bcp.c:3998 -> lib.c:2013-2016). So every later hit of the same move that would KO leaves 1 HP (bcp.c:3396-3405). The holder survives the whole multi-hit, including a non-lethal first hit followed by a lethal second hit. The sash is removed once, in the followup message (sub/subscript_move_followup_message.s:11-19). `effects.play_hit`/`focus_armed` and the test body implement this. Only the plan/survey text and the test's name are stale. |
| W2 | survey §6 Flail (:573), §7 (:651-652), plan Task 9 (:281) | "TrainerAI_CalcDamage has no Flail case, so the AI computes Flail at table power 1" (§7 says the same for Present) | **WRONG rationale** (engine result OK) | The AI never runs a damage calc for a move whose data power is ≤ 1 (Flail and Present are power 1). Each damage query has a `power > 1` gate: ai:1035-1036 (FlagMoveDamageScore -> `AI_NO_COMPARISON_MADE`), :1553-1554 and :1604-1605 (IfCurrentMoveKills / DoesNotKill never jump), :2827-2828 (CalcAllDamage -> 0). The engine matches only because `ai/commands.py:ai_damage` multiplies by `is_damaging` (power > 1). `ai_damage_for_roll(..., Flail)` on its own returns a power-1 number the AI never computes. Do not call it without that gate. |
| W3 | Task 8 Self-Destruct x AI (latent; K3c) | (not stated) | **WRONG if left as is** | `BATTLE_EFFECT_HALVE_DEFENSE` (Selfdestruct/Explosion) is in `sNoDamageCalcMoveEffects` (ai:31-46). The AI never computes its damage: no comparison, 0, never "kills". `ai/commands.py:is_damaging` (:207-211) tests only power > 1, so Bonsly's Selfdestruct (power 200) would be scored as damaging. The other noCalc effects are listed at ai:32-44. Note that Flail is not among them: its effect is `INCREASE_POWER_WITH_LESS_HP`. The listed `DECREASE_POWER_WITH_LESS_USER_HP` is Eruption. |
| W4 | plan Task 6 (:215) | "33 -> STATUS_FOE(Status.TOXIC, type_checked=1)" | **WRONG text** (code right) | `CheckTypeChart` applies the chart only to moves with power, or to Thunder Wave (bcp.c:2350-2357). Toxic's Poison/Steel immunity comes from sub/subscript_badly_poison.s (`_128` onward). `tools/effect_map.py:110` correctly maps it with param[1] = 0. |
| W5 | survey §3.3 (:352-353), §5.2 (:488) (adjacent to the audited sections) | "almost every move has accuracy 100 (threshold 100 -> one value)"; "Toxic/Present/Thunder Fang never miss" | **WRONG** | Accuracy 100 still rolls `RandNext()%100+1 > hitRate` (bcp.c:2976). Only accuracy 0 skips the roll (bcp.c:2900-2903). In Roark's permanent sand, Gible's Sand Veil makes it 80 (bcp.c:2926-2928), and stages change it too. `outcome.hit_threshold` already models this. The width estimates and the "never miss" wording are wrong. |
| U1 | plan Task 13 (resist berries) | "halves the final HP change ... Chilan on any Normal hit" | **INCOMPLETE / UNCERTAIN** | (a) The berry check runs inside sub/subscript_update_hp.s. It is skipped if `SKIP_SPRITE_BLINK` is set or `waitingBattlers == 0`, so it applies to every in-action UpdateHP. That includes the **confusion self-hit** (sub/subscript_hurt_self_in_confusion.s calls UPDATE_HP with no skip flag). The type it reads is the current move's data type (`moveType` is reset to Normal, so `GetCurrentMoveData`; sub/subscript_type_resist_berry.s:_020-_031). So a confused Chilan holder whose *selected* move is Normal-type halves its self-hit and loses the berry. Rindo cannot trigger there (no SE flag). This is uncertain only in that `moveCur` should be confirmed set at the status check, which is likely. Water Pulse confusion and a player Chilan make this reachable. (b) Skipped for OHKO and for `IGNORE_TYPE_CHECKS`/`IGNORE_IMMUNITIES` moves (:5-6). (c) `storedDamage` (Bide), Counter/Mirror Coat damage and `lastDamageTaken` keep the **unhalved** damage (set in bcp.c:3407-3433 before the subscript). Drain and recoil use the halved `hitDamage` (script.c:1699-1704). (d) "resist after Focus Sash" (`test_resist_after_focus_sash`) cannot happen: one Pokemon holds one item. The ordering is right (sash cap at bcp.c:3396-3405, then halving) but untestable in a real state. (e) Pluck/Bug Bite on a Rindo/Chilan holder eats it with no effect (pluckEffect NONE + `Item_IsBerry`, lib.c:5833-5848). |
| U2 | plan Task 3 (:134) | "Hail chip: the same, except Ice types are immune and Snow Cloak replaces Sand Veil" | **AMBIGUOUS** (code right) | In hail, Rock/Ground/Steel are **not** immune. Only Ice type and Snow Cloak are, and Ice Body heals 1/16 instead (script.c:5598-5610). `step._weather_effect` and the reference do this correctly. |
| U3 | plan Task 5 (:197-198) | "clears both screens ... after the hit" | **WORDING** (code right) | The break is a direct primary effect. It runs after TryMove but **before** `UpdateHP` (bcp.c:3295-3310 then 3334-3338), gated on the defender having HP at that moment (lib.c:1495-1499). A KO hit still breaks the screens; a miss or Protect does not; an immune target does. The engine uses the pre-damage `landed`, so it matches. |
| U4 | survey §6 Roar (:587-591), K3b2 | fail conditions listed: Suction Cups, Ingrain, empty bench | **INCOMPLETE** | Roar is in `sSoundMoves` (lib.c:3473-3475). An (ignorable) Soundproof target blocks it in `TriggerImmunityAbilities` (lib.c:3523-3527). Add it to the K3b2 rules or a coverage refusal. |
| U5 | survey §6 Obedience | "Kaizo always obeys" | **NOT DECOMP-CHECKABLE** | This is a Kaizo ROM-patch claim. Vanilla has `BattleControllerPlayer_CheckObedience` (bcp.c:2109). Verify against the ROM diff, not the decomp. |

Minor notes, not errors:
- Curse "fails only when every stage is capped": Curse never sets FAILED. Each stat change is direct, so a capped stage takes the `jumpNoChange` message path (script.c:2841-2851; sub/subscript_update_stat_stage.s `_041`). All three capped is a no-op, so the state is identical. `effects._curse` is correct.
- Lagging Tail: the plan omits the two-holder case. If both hold it, they compare speed **reversed** (faster goes second, Trick Room ignored, lib.c:1412-1417). `order.player_second` implements it.

## 1. Survey §3.2 (draw descriptions)

| Rule | Verdict |
|---|---|
| Thunder Fang: ON_HIT, no outer roll; two `CheckEffectActivation` (`%100 < effectChance` and side-effect mon alive), paralysis then flinch | VERIFIED eff/effect_script_0275.s; lib.c:1524-1529; sub/subscript_paralyze_or_flinch.s; script.c:8363-8383 |
| AncientPower one 10% roll, TO_ATTACKER, indirect | VERIFIED eff/effect_script_0140.s; lib.c:1570-1586 (recipient = user must have HP); res/moves/ancient_power chance 10 |
| Tri Attack `Random 2,3` -> 3/4/5 burn/freeze/paralyze, before accuracy, then the chance roll | VERIFIED script.c:3200-3210 (`%3+3`); include/data/move_side_effect_subscripts.h; eff/effect_script_0036.s; lib.c:1570-1586 |
| Present `rnd=RandNext()&0xFF`; <102 40, <178 80, <204 120, else heal maxHP/4; law (102,76,26,52)/256; heal reads no crit/roll | VERIFIED script.c:5789-5806 (255*40/100=102, 178, 204); eff/effect_script_0122.s (heal path skips CalcCrit/CalcDamage); heal ignores immunities, fails on miss (sub/subscript_present_heal.s) |
| Roar: level check only if user level < target level, `(((r&0xFF)*(La+Ld))>>8)+1 > Ld/4`; uniform over eligible bench by rejection loop; applies in trainer battles | VERIFIED script.c:5241-5320, lib.c:3072-3086 |
| Protect `rate[n] >= RandNext()`, 1, 1/2, 1/4, 1/8 (count capped at 3), fails if last to move | VERIFIED script.c:5128-5190 (counter caps at index 3, :5182) |
| Attract: `RandNext()&1` true -> acts, false -> immobilized | VERIFIED bcp.c:2667-2690 |
| Quick Claw `speedRand % (100/param) == 0`, drawn per turn (and at battle start) | VERIFIED lib.c:1278-1286, :1344-1352; bcp.c:853-857; lib.c:2024. Quick Claw param 20 (res/items/data/quick_claw.json) |
| Static, Flame Body, Cute Charm `RandNext()%10<3`; Rough Skin no roll, maxHP/8 Divide | VERIFIED lib.c:4214-4229, :4321-4337, :4339-4355, :4257-4271 |
| Sand Veil x80/100 in sand, ignorable | VERIFIED bcp.c:2926-2928 |
| Sand Tomb `Random 3,3` | VERIFIED sub/subscript_bind_start.s (`%4+3`) |
| Focus Band would need a draw | VERIFIED bcp.c:3387 (note: it also sets the same per-action flag) |

## 2. Survey §4 (turn structure, sandstorm, screens)

| Rule | Verdict |
|---|---|
| `BranchActions` re-sorts `monSpeedOrder` then goes to field conditions | VERIFIED bcp.c:873-895 |
| Field order REFLECT, LIGHT_SCREEN, MIST, SAFEGUARD, TAILWIND, LUCKY_CHANT, WISH, RAIN, SAND, SUN, HAIL, FOG, GRAVITY; AnyFainted/CheckBattleOver before every step | VERIFIED bcp.c:897-944 |
| Screens side 0 then 1, `--turns == 0` ends | VERIFIED bcp.c:947-991 |
| Permanent weather never counts down; temporary decrements, ends at 0 with no effect that turn | VERIFIED bcp.c:1138-1161 (end subscript instead of weather_continues) |
| Trick Room counts down in the side-condition pass | VERIFIED bcp.c:1821-1829 |
| Mon conditions INGRAIN, AQUA_RING, ... per battler in monSpeedOrder, faint check on every re-entry | VERIFIED bcp.c:1256-1330 |
| Sand chip: `NO_CLOUD_NINE` (alive Cloud Nine/Air Lock), not Rock/Steel/Ground, curHP>0, `Battler_Ability != Sand Veil` (not ignorable), no Dig/Dive flag, `Divide(-maxHP,16)`; Magic Guard skips in the script | VERIFIED script.c:5573-5583; include/battle/common.h:63-64; lib.c:3599-3619; sub/subscript_weather_continues.s |
| Faint timing: all chips inside one subscript, battle end checked when the loop resumes -> simultaneous last faints = draw | VERIFIED sub/subscript_weather_continues.s (FAINT_MON inline); bcp.c:939-944; CheckBattleOver ORs WIN and LOSE (bcp.c:4242-4261) = draw |
| Rock SpD x1.5, rain/sun power gated on NO_CLOUD_NINE | VERIFIED lib.c:6921-6929, :7046-7072 |
| Sand Veil after Compoundeyes, before fog; Snow Cloak twin; Swift Swim x2 gated | VERIFIED bcp.c:2922-2937; lib.c:1228-1238 |
| Start sand from map weather is permanent | VERIFIED lib.c:3679-3682; sub/subscript_overworld_sand.s sets `SANDSTORM_PERM` |
| Light Screen fails if up, else `NUM_SCREEN_TURNS`(5) + Light Clay | VERIFIED script.c:4214-4244; include/constants/battle.h:142 |
| Screen halves when `criticalMul == 1` and effect != REMOVE_SCREENS; singles 1/2 | VERIFIED lib.c:6982-6991, :7023-7032 |
| Brick Break breaks on immune, not on miss | VERIFIED lib.c:1492-1508 (see U3 on timing); script.c:6550-6567 |
| Script (crit, damage, Present, Tri Attack, Self-Destruct) runs before TryMove; primary effects after it | VERIFIED bcp.c:3209-3216, 3237-3310 |

## 3. Survey §6 mechanic notes

| Rule | Verdict |
|---|---|
| Obedience | U5 |
| Self-Destruct: Damp check; HP=0 in script before accuracy, so it faints on a miss or into a Ghost; no target -> fails earlier, user survives; Defense halved; both-last -> draw | VERIFIED eff/effect_script_0007.s; bcp.c:3190 (HasNoTarget before LoadScript :3214); lib.c:6943-6945; bcp.c:4242-4261. The AI side is W3 |
| Recover Divide(maxHP,2); fails at full | VERIFIED sub/subscript_recover_half_max_hp.s, subscript_recover_hp.s |
| Curse non-Ghost: Spe-1, Atk+1, Def+1; Ghost branch | VERIFIED eff/effect_script_0109.s; sub/subscript_curse_normal.s (see minor note) |
| AncientPower Atk, Def, Spe, SpA, SpD +1 | VERIFIED sub/subscript_boost_all_stats.s |
| Toxic immunities | VERIFIED sub/subscript_badly_poison.s (see W4 on the plan wording) |
| Sand Tomb = TRAP (CHECK_HP_AND_SUBSTITUTE, bind); Dive x2 unreachable | VERIFIED eff/effect_script_0261.s vs effect_script_0042.s |
| Swift family bare hit, never-miss from accuracy 0; Pay Day plain; Gust x2 vs airborne only | VERIFIED eff/effect_script_0017.s, _0034.s, _0149.s; bcp.c:2900-2903 |
| DoubleSlap SetMultiHit 2, one accuracy check | VERIFIED eff/effect_script_0044.s; bcp.c:3777-3792 |
| Fake Out fails unless `fakeOutTurnNumber == totalTurns`; flinch via normal subscript | VERIFIED eff/effect_script_0158.s; lib.c:2117, :2195; sub/subscript_flinch_mon.s (Inner Focus, not moved, plus Substitute and Shield Dust) |
| Flail table and `App_PixelCount` | VERIFIED script.c:4941-4972; src/unk_0208C098.c:41-49. AI claim: W2 |
| Present heal maxHP/4, ignores immunities, fails on miss | VERIFIED |
| Protect needs moveProtect reset, per-turn flag | VERIFIED script.c:5154-5158 |
| Attract fail cases; cleared when the source leaves; status order CONFUSION, PARALYSIS, ATTRACT, BIDE; Cute Charm needs both alive and attacker not infatuated | VERIFIED script.c:5726-5741; lib.c:2087-2091, :2174-2176; bcp.c:2375-2392; lib.c:4339-4355 |
| Roar | VERIFIED + U4 |
| Camouflage types; fails if Multitype or already has the type; building -> Normal | VERIFIED script.c:7997-8022; include/data/terrain/to_type.h |
| Nature Power building -> Tri Attack | VERIFIED script.c:8036-8048; include/data/terrain/to_move.h |
| Aqua Ring: fails if set; Divide(maxHP,16) when not full and alive; second in mon order; Big Root after | VERIFIED eff/effect_script_0251.s; bcp.c:1332-1353; sub/subscript_aqua_ring_heal.s |
| Rough Skin 1/8, attacker alive, contact, damage taken, not Magic Guard, `Battler_Ability` of the holder | VERIFIED lib.c:4257-4271 |
| Flame Body 30%; Limber blocks paralysis (ignorable) and cures at switch-in and move end | VERIFIED lib.c:4321-4337; sub/subscript_paralyze.s:5; lib.c:4397-4402; bcp.c:3965-3967 |
| Adaptability x2; Solid Rock Divide(dmg*3,4) on net SE (flags cancel), ignorable; the AI shares it | VERIFIED lib.c:2592-2598, :2655-2666, :7542-7579; ai:3081-3103 |
| Focus Sash sites: UpdateHP, confusion self-hit, Pursuit, Future Sight; consumed in followup message | VERIFIED script.c:9098-9125; sub/subscript_hurt_self_in_confusion.s, subscript_pursuit.s, subscript_future_sight_damage.s. **Multi-hit sentence: W1** |
| Quick Claw > Lagging Tail > Stall > TR/speed, equal priority only; also in SortMonSpeedOrder | VERIFIED lib.c:1401-1447, :3364-3381 |
| Status berries (Chesto, Pecha incl. toxic, Lum incl. confusion); White Herb restores every lowered stage (incl. acc/eva) | VERIFIED lib.c:4601-4696, :4875-4887, :4991-5100. Call sites: bcp.c:1363, 1701, 3687, 3694, 3745; lib.c:4151-4155 (switch-in, so White Herb after Intimidate) |
| Resist berries | VERIFIED core; U1 additions |
| Big Root x(100+30)/100 after base, no min; drain half via Divide | VERIFIED sub/subscript_drain_half_damage_dealt.s, subscript_leech_seed_effect.s, subscript_aqua_ring_heal.s; res/items/data/big_root.json param 30 |
| Berry Juice: HP_RESTORE param 20; Pluck does not eat it | VERIFIED lib.c:4585-4591; res/items/data/berry_juice.json (pluckEffect NONE, not a berry); lib.c:5833-5836 |

## 4. Plan Tasks 3-13 rule bullets

- Task 3: all VERIFIED (§2 above); hail wording is U2. Chip values 46->2, 17->1, 15->1 match Divide.
- Task 4: VERIFIED (Rock SpD gating, Sand Veil 80 after Compoundeyes, Snow Cloak, Swift Swim gating).
- Task 5: VERIFIED; wording U3.
- Task 6: Toxic wording W4; the rest VERIFIED.
- Task 7: VERIFIED (Curse minor note).
- Task 8: VERIFIED; AI side W3.
- Task 9: power bands VERIFIED; AI rationale W2.
- Task 10: VERIFIED.
- Task 11: Focus Sash multi-hit **W1**; the rest VERIFIED (Lagging Tail two-holder nuance noted).
- Task 12: VERIFIED.
- Task 13: core VERIFIED; U1.

## 5. Code audit (Tasks 3-10 at 17cc3ab, plus Task 11 at 577db70)

| Area | File | Matches decomp? |
|---|---|---|
| Field pass order, screens/side timers, weather countdown, sand/hail chip, Magic Guard, Ice Body, one faint check after the weather, Gravity; monSpeedOrder sorted once before the field pass | step.py `field_conditions`/`_weather_effect`; reference `_field_conditions` | Yes |
| Cloud Nine gating via alive decomp ability (incl. a Self-Destruct user at 0 HP and the switched-in target) | damage.py `weather_suppressed`; outcome.py `_hit_threshold` | Yes |
| Sand Veil/Snow Cloak (Mold Breaker pierces), fog, gravity order | outcome.py:108-152 | Yes |
| Swift Swim before paralysis /4 and Tailwind | order.py `effective_speed` | Yes (lib.c:1225-1276) |
| Rock SpD x1.5 (gated), Self-Destruct Def//2 before stages, screens skip Brick Break and crits | damage.py `_pre_random` | Yes |
| Adaptability, Solid Rock on net SE after the chart; AI variance last | damage.py `_finish` | Yes |
| Flail bands; AI keeps power 1 | damage.py `flail_power`, `ai=True` | Yes, provided callers keep the `is_damaging` gate (W2) |
| Light Screen/Reflect set 5, fail if up; Brick Break on pre-damage `landed` | effects.py `_screen`, `_brick_break` | Yes |
| Recover Divide(max,2), fails full; AncientPower on roll, user alive; Curse three independent stages, user is defender | effects.py | Yes |
| Fake Out failure before accuracy; flinch with Inner Focus message only for Fake Out | step.py `_use_move`; effects.py `_flinch` | Yes |
| DoubleSlap fixed count; n_hits relevance only for the 2-5 law | effects.py `hit_count`; enumerate.py | Yes |
| Self-Destruct HP 0 before accuracy, only with a target; multi-hit stops when the attacker faints | step.py `_use_move` | Yes |
| Rough Skin/Static/Poison Point/Flame Body: holder need not be alive; attacker alive, contact, effective, dealt > 0; attacker's Limber stops Static silently | effects.py `_contact_ability` | Yes (a capped 0-damage later hit gives no proc, which also matches: DamageTaken 0, lib.c:4220) |
| Limber block (ignorable) and cure at move end/switch-in | effects.py `_inflict`; hooks.py `recover_status_by_ability` | Yes |
| Toxic mapping param[1]=0 | tools/effect_map.py:110 | Yes |
| Focus Sash per-action flag, non-lethal arming, consumed once; confusion self-hit via CheckHoldOnWith1HP | effects.py `play_hit`/`focus_sash_damage`; step.py | Yes (577db70) |
| Lagging Tail incl. two holders, also end-of-turn order | order.py `player_second` | Yes |
| Big Root on drain and Leech Seed | effects.py `big_root`; step.py `_leech_seed_check` | Yes |
| Engine AI `is_damaging` | ai/commands.py:207-211 | **No** for Self-Destruct/Explosion (W3) |

Gen-4 pitfall checklist (the brief's list): Sturdy OHKO-only (bcp.c:3343; hooks docstring) OK; Rough Skin 1/8 OK; Flame Body/Static 30% OK; paralysis /4 OK (lib.c:1259-1261); sand/hail 1/16 OK; Light Screen 1/2 singles OK; crit x2 OK (lib.c:7139-7144); Adaptability OK; Curse OK; Fake Out OK; Protect odds OK; Self-Destruct Def halving OK; Recover rounding OK; Big Root 30% OK; status berry timing OK; White Herb OK; Lagging Tail vs Quick Claw OK; Swift Swim OK; Sand Veil 20% OK; Cute Charm/Attract OK; Roar level check in trainer battles OK (+U4); Present OK; Flail OK (+W2); Thunder Fang OK; Nature Power -> Tri Attack OK; Camouflage OK; Lum/Chesto OK; Berry Juice OK; Rindo/Chilan after the Sash cap OK (+U1); Pluck vs Berry Juice OK; toxic counter reset on switch-out OK (battle_controller.c:1482; step.py:215-222); Focus Sash per-action flag **W1** (plan text only).
