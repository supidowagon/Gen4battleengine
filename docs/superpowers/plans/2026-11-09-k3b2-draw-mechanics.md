# K3b2: draw mechanics, terrain and exact enumeration, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the remaining mechanics Kaizo Roark and the user's pre-Roark pool need, exactly
and in both engines, and keep every Roark-vs-pool action pair exactly enumerable. The mechanics are
Thunder Fang, Protect, Present, Attract and Cute Charm, Quick Claw, Roar, Camouflage, and Nature
Power with a per-trainer terrain. The plan also fixes the mid-turn accuracy bug.

**Architecture:** The same layer path as K3b1 (ids → effect map → JAX → reference → coverage →
tests). New random draws become `Outcome` fields, drawn with `fold_in` so existing seeded draws stay
put. New state follows `docs/kaizo/k3b2-rules.md` §12. Exactness under the width cap comes from
tighter, still sound relevance, plus three enumeration techniques (rules §10.3):
- order-conditioned masks;
- joint factors;
- damage-class compression.

**Tech Stack:** Python 3.12 (uv), JAX, pytest, Hypothesis, ndspy/Capstone (the terrain ROM reader).

**Spec:** `docs/superpowers/specs/2026-11-07-kaizo-k3-design.md` §4, §8.

**Rule source (binding):** `docs/kaizo/k3b2-rules.md`. Every rule in it was read from the decomp
with a file:line citation, and every task implements the rules of the section it names. A task that
finds the code saying otherwise follows the code and reports it. Nothing comes from general Pokémon
knowledge. Also in force: `docs/kaizo/user-reported-quirks.md` and `docs/kaizo/k3b-rule-audit.md`.

## Global Constraints

- The K3b1 Global Constraints still apply (`docs/superpowers/plans/2026-11-08-k3b1-field-and-mechanics.md`):
  - both engines in the same task, with fixed-state reference tests;
  - an enum member lands in the same commit as its code;
  - every vanilla table change bumps `MANIFEST_VERSION` and rebuilds both games, and the tracked
    manifest is committed;
  - hard-coded decomp ids go in `game_constants`;
  - STATE.md stays current;
  - no ROM bytes or saves in commits; never touch a `backup` folder; commit trailer
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never push or merge;
  - the controller runs the full suite.
- **Vanilla exact results.** They are unchanged except where this plan fixes a real engine bug
  (Task 1). Every such change gets a `docs/discrepancies.md` entry and a test.
- **Outcome fields.** New fields take `fold_in(key, 16 + k)` in order of landing.
  `tests/engine/test_outcome.py::test_existing_draws_unchanged` stays green.
- **Relevance must be sound.** A field marked irrelevant must not change the stepped state. Every
  task that adds or tightens relevance adds a test enumerating the affected pairs: the exact
  distribution equals a brute-force enumeration with all fields relevant, then merged by
  identical states.
- **`RandNext() % n` stays treated as uniform** (project convention, enumerate.py:322-325).
  rules §0.3 gives the exact u16 weights; the convention is recorded as a user decision for
  later.

## Review Focus

1. A Sand-Attack (or Double Team) earlier in the turn changes a later move's hit chance in that
   same turn. Task 1.
2. Relevance tightening must never drop an outcome that changes the state. The brute-force
   equality tests in Tasks 2 and 3 cover this.
3. Quick Claw's draw from the turn before decides this turn's order. The battle-start draw is a
   chance node, and Stage A and Stage B both handle it. Task 8.
4. Roar's dragged-in Pokémon takes Stealth Rock and triggers switch-in effects, and the AI's
   knowledge and replacement logic stay consistent. Task 11.
5. Every Roark-vs-pool action pair from both saves is exactly enumerable (width ≤ 131072), or is
   listed by name as sampled. Task 12.

---

### Task 1: Mid-turn accuracy (interval-class `hit`)

Rules §0.2b-c, §10.3 item 7. **Files:** `engine/outcome.py`, `engine/enumerate.py`,
`engine/step.py`, `reference/engine.py`, `tools/trace_to_state.py` (if it infers `hit`),
`docs/discrepancies.md`. **Test:** `tests/engine/test_accuracy_classes.py`.

`hit[side, 0]` becomes the interval class of `u = RandNext()%100+1`:
- The class boundaries are every threshold the move can have at its own TryMove this turn: the
  turn-start threshold, plus each one an earlier action can produce.
- The law is the interval widths over 100.
- `step` compares the class against the live threshold.
- Sampling draws `u` and maps it to its class.

- [ ] **Step 1: Write the tests.**
  - `test_sand_attack_lowers_a_later_move_this_turn`: the faster Sand-Attack drops the slower
    100-accuracy move's exact hit probability to 75/100, matching bcp.c:2865-2979.
  - `test_accuracy_class_law_sums_to_one`.
  - `test_no_change_keeps_two_classes`.
  - `test_accuracy_classes_match_reference`.
  - `test_sampled_hit_frequencies_match_law`.
- [ ] **Steps 2-4:** fail, implement, run the focused tests plus `tests/engine`,
  `tests/test_differential.py`, `tests/search`, and `tests/test_trace_to_state.py`.
- [ ] **Step 5:** Add the discrepancies entry. Note which vanilla results can change: a Sand-Attack
  or evasion move used earlier in a turn. Commit `fix(engine): accuracy is read at the move's own
  check (interval-class hit)`.

### Task 2: Order-conditioned relevance and joint factors

Rules §10.3 items 1-4. **Files:** `engine/enumerate.py`, and `engine/outcome.py` if laws move.
**Test:** `tests/engine/test_enumerate.py` (extend).

The changes:
- Enumerate `speed_tie` first and compute the masks per order branch.
- Gate reads by order: a flinch or status-inflicted roll is read only if the reader acts after the
  inflicter.
- Gate on status eligibility.
- Add a joint-factor mechanism for dependent fields: (secondary, duration) and (secondary, the foe's
  thaw/full_para). Each joint factor lists only the combinations that change the state.
- Later tasks register their own joint factors here: Task 5 (present, crit, roll), Task 10 (tri,
  secondary).

- [ ] **Step 1: Write the tests.**
  - `test_order_branches_equal_bruteforce`: on random states from the strategies, the exact
    distribution with the new masks equals brute-force all-relevant enumeration, merged by state.
    It covers at least 200 states, including speed ties.
  - `test_width_drops_for_goldeen_psybeam_vs_cranidos`: the rules §10.2 pair goes from 262,144 to
    ≤ 30,000 on Kaizo tables.
- [ ] **Steps 2-5:** fail, implement, then run `tests/engine/test_enumerate.py`, `tests/search` and
  `tests/test_differential.py`. Commit `feat(enumerate): order-conditioned masks and joint factors`.

### Task 3: Damage-class compression

Rules §10.3 item 5. **Files:** `engine/enumerate.py`, `engine/damage.py` (an exposed damage table
per hit). **Test:** `tests/engine/test_damage_classes.py`.

How it works:
- Per hit, (crit, roll) pairs that give the same final damage merge into one class, with summed
  probability.
- Compression is applied only when no earlier action this turn can change that hit's damage
  inputs: stages, screens, types, HP for Flail, weather, abilities, items consumed.
- Otherwise the classes are refined over every reachable input variant, or compression is skipped
  for that hit.
- Each class carries a representative (crit, roll) that `step` consumes.

- [ ] **Step 1: Write the tests.**
  - `test_damage_classes_equal_bruteforce` covers multi-hit, Focus Sash, resist berries and crit
    stage changes.
  - `test_doubleslap_vs_corsola_water_pulse_fits`: Kaizo tables with the probe or base save
    Skitty; the width is ≤ 131072 and equals the brute-force merged distribution.
  - `test_compression_disabled_when_inputs_can_change`.
- [ ] **Steps 2-5:** fail, implement, run (enumerate, search, differential), then commit
  `feat(enumerate): exact damage-class compression`.

### Task 4: Thunder Fang

Rules §1. Map effect 275 to a new `EffectId.FANG_PARALYZE_FLINCH`. Add the `fang_flinch[2]` field;
the paralysis roll reuses `secondary`. Relevance is order-gated (Task 2). Do not map the Fire Fang
or Ice Fang effects (274, 276) until their scripts are read (rules §11 item 7).

**Test:** `tests/engine/test_fangs.py`. It covers:
- both rolls independent;
- flinch only if the target has not moved and lacks Inner Focus;
- paralysis gated by `can_have_status`/Limber;
- Static and Rough Skin contact;
- a reference test;
- a sampled-frequency test.

**Commit:** `feat(engine): Thunder Fang`.

### Task 5: Present

Rules §3. Map effect 122 to `EffectId.PRESENT` and add the `present[2]` field. The law is
(102, 76, 26, 52)/256 for power 40/80/120 and the heal. The heal branch reads no crit or roll; it
heals the target by maxHP/4 (Divide) and ignores immunities. Present is drawn before the accuracy
check. Register the (present, crit, roll) joint factor. The AI path skips damage (power ≤ 1).

**Test:** `tests/engine/test_present.py`.

**Commit:** `feat(engine): Present`.

### Task 6: Protect

Rules §2. Map effects 111 and Detect's to `EffectId.PROTECT` and add the `protect[2]` field. The
law is 2^-n by `protect_count`. The chain reset folds into `protect_count` (rules §12). Settle
UNCERTAIN item 2 with a failed-Protect-then-Protect test against the decomp's writers.

The rules:
- Protect fails when the user moves last.
- It blocks moves with `MoveFlag.PROTECT`.
- Fake Out against Protect is a speed tie at +3.
- Observation and plan tree get the protect message.

**Test:** `tests/engine/test_protect.py`.

**Commit:** `feat(engine): Protect and Detect`.

### Task 7: Attract and Cute Charm

Rules §4. Map effect 120 to `EffectId.ATTRACT`, add `AbilityId.CUTE_CHARM`, the `infatuated[2]`
state, and the `infatuation[2]` field (1/2).

The rules:
- Attract fails on the same gender, a genderless Pokémon, or one already infatuated.
- Infatuation is cleared when its source leaves (switch or faint).
- The status-check order is confusion, then paralysis, then attract.
- Cute Charm needs both Pokémon alive, a non-infatuated attacker and opposite known genders, and
  uses the `contact_proc` roll.
- Oblivious is refused (unsupported).
- Observation and plan tree are updated.

**Test:** `tests/engine/test_attract.py`.

**Commit:** `feat(engine): Attract and Cute Charm`.

### Task 8: Quick Claw

Rules §5 and §0.2a. Add `ItemId.QUICK_CLAW` and the hidden state `qc_roll[2]`, carried into the
next turn and set from the reserved `quick_claw[2]` Outcome field with law 1/5 for param 20.

The rules:
- Turn N's order reads the draw made during turn N-1.
- Turn N's own draw sets the end-of-turn order, the replacement order, the AI's speed compares and
  turn N+1's order.
- Quick Claw beats Lagging Tail (the K3b1 seam in `order.player_second`).
- The tight storage rule (rules §5b): keep the draw only while the holder is active or a
  replacement is pending.
- The battle-start draw is a chance node. Stage A samples it. Stage B gets a root chance node, and
  the information set is grouped by equal observation; settle rules §11 item 1 in this task. Read
  the Stage B design doc (`docs/superpowers/specs/2026-10-30-stage-b-solver-design.md`) and extend
  it in place with a short section.

**Test:** `tests/engine/test_quick_claw.py`, plus a `tests/search` test of the root chance node.

**Commit:** `feat(engine): Quick Claw`.

### Task 9: Terrain pipeline

Rules §9. **Files:**
- `tools/rom/readers/maps.py` (terrain per header and per trainer: battleBG, matrix cells,
  land-data tile behaviours, the decomp's `CalcTerrain`);
- `tools/extract_rom.py` and `tools/extract_decomp.py` (both games);
- `tools/build_tables.py` (`Trainer.terrain`, decomp `TERRAIN_*`, -1 for mixed);
- `src/nuzline/data/trainer.py`;
- `engine/state.py` (`terrain` field scalar; `make_state(..., terrain)`);
- `engine/coverage.py`: refuse effects 173 and 213 only for a mixed or unknown terrain, naming the
  set;
- `data/kaizo/rom_diff.toml`: flip `MapHeader_GetBattleBG` and `MapHeader_GetMapMatrixID` to
  `modelled`, drop their `when`, cite the reader test;
- `check_kaizo_map_headers` extended to those two fields.

**Test:** `tests/tools/rom/test_terrain.py`. Roark (246) gives BUILDING in both games. Also list
how many trainers are mixed, per game.

**Commit:** `feat(rom): terrain per trainer from the ROM`.

### Task 10: Camouflage and Nature Power

Rules §7, §8. Add `EffectId.CAMOUFLAGE` (213) and `EffectId.NATURE_POWER` (173). Add the terrain
tables to/from move and type from `include/data/terrain/` (in game_constants; clamp at
TERRAIN_SPECIAL). Add the `tri_attack[2]` field (1/3 each) and register the (tri, secondary)
joint factor.

The rules:
- Camouflage writes `types` for the active battler and restores them on leave. It fails on
  Multitype or when the user already has that type.
- Nature Power calls Tri Attack for accuracy, damage, Protect and `move_hit`. It stays Nature Power
  for PP, Taunt, Disable, last move and the AI's seen-moves list.

**Test:** `tests/engine/test_terrain_moves.py`. Kaizo Tri Attack data (95 BP, chance 30) comes
from the tables.

**Commit:** `feat(engine): Camouflage and Nature Power`.

### Task 11: Roar

Rules §6. Map effect 28 to `EffectId.ROAR`. Add the `roar_level[2]` and `roar_target[2]` fields.

The rules:
- **Level check:** the draw happens only when the user's level is lower than the target's.
- **Target:** uniform over the eligible bench (species set, not an egg, HP > 0, not active),
  rejection-sampled in the decomp.
- **Failure cases:** Suction Cups, Ingrain, Soundproof (lib.c:3473-3475, :3523-3527), and no
  eligible bench.
- **The dragged-in Pokémon:** it takes the hazards check, then its switch-in effects at move end.
  Natural Cure applies to the outgoing Pokémon. Infatuation, Camouflage types and protect state are
  cleared for the outgoing Pokémon.
- **Downstream:** the AI's knowledge of the dragged-in Pokémon and the replacement logic stay
  consistent. Settle rules §11 item 4.
- **Observation:** the dragged-in species.

**Test:** `tests/engine/test_roar.py`.

**Commit:** `feat(engine): Roar`.

### Task 12: Width audit on Roark versus the pool

**File:** `tests/kaizo/test_roark_widths.py`. It skips without the saves and the Kaizo tables.

For both saves (`kaizo_save_pre_roark`, and the probe save path in the K3a handover), take every
pool candidate and every legal action pair against every Roark party member at turn 0. For each,
enumerate the width and assert it is ≤ 131072, or that the pair is in an explicit
`SAMPLED_PAIRS` list with a reason. Record the widths table in the report.

**Commit:** `test(kaizo): Roark action-pair widths`.

### Task 13: Close K3b2

Rebuild both games and run the full suite (the controller). Run coverage on both saves
(`--trainer roark --cap 16`).

**Expected result:**
- only the K3c lines remain: the AI policy, `AI_FLAG_RISKY`, the untranscribed move scripts, and
  `Kaizo ROM: 1 code changes unclassified`;
- the pool is empty.

Update `docs/kaizo/pipeline.md` and spec §4. Write the K3c handover in the shared notes folder:
- the AI items from the K3b1 handover;
- the AI reads listed in rules §1e-§8e;
- Quick Claw's AI speed compare.

**Commit:** `docs(kaizo): K3b2 done`.

---

**Plan-level rulings (controller):**
- The `% n` uniformity convention is kept for K3b2. The exact u16 weights (rules §0.3, §11 item 8)
  are a project-wide decision listed for the user.
- Damage-class compression (Task 3) is an exact merge of outcomes that give identical states, as
  design.md §4.2 describes. It replaces approximation, so it is in scope.
- Task order: correctness fixes and enumeration infrastructure (1-3) come before the mechanics that
  widen the enumeration (4-11).
