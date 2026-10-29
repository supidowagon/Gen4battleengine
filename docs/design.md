# Platinum Nuzlocke Line Finder — Design Spec

Status: draft v3 (second review pass folded in) · Date: 2026-10-28

## 1. Purpose

A tool that finds **high-EV lines for single-player nuzlocke battles in vanilla Pokémon Platinum (US)**. Given the player's save file and a chosen trainer fight, it recommends:

1. **Pre-battle setup**: which Pokémon to bring from the box, the lead, held items, prestatus, predamage, and optionally movesets.
2. **In-battle play**: a ranked list of actions each turn, plus an exportable plan tree ("if X happens, do Y").

"High EV" means **maximizing the probability of winning with zero faints**. Every line also reports P(win at all) and expected faints, and the user can choose which of these orders the list (see §3, *Objective*). Ties are broken by fewest resources spent, then most HP/PP remaining.

Speed comes from a battle engine written in **pure JAX** so that huge batches of battles can be `jit`ted and `vmap`ped on a GPU. The engine runs every game phase on every step and masks out the results that don't apply, so each action has the same computational shape. The compiled step is **specialized per fight** to the mechanics actually present (§4.2).

## 2. Scope

### In scope (v1)
- Vanilla Platinum US mechanics, single battles only.
- Nuzlocke rules: **Set mode** (no free switch after a KO), **level caps**, and the player uses **held items only** (no bag items in battle).
- Full knowledge of the opponent: team, moves, items and AI behavior. (The *AI's* knowledge of the player is partial and is itself modeled; see §4.2, *Trainer AI*.)
- Platinum trainer AI, including its switching, item use, internal random draws and move/ability knowledge, modeled from the decomp.
- Battle start conditions set by the overworld location (weather, fog).
- Party, PC boxes and bag read from emulator saves (`.sav` / `.dsv`).
- Levels and EVs are taken from the save as-is. "Level up to the cap" and **evolution** (by level or by a stone in the bag) are optional prep variables (§4.4); EV training is not searched.
- **Level caps** come from a per-trainer config table shipped with defaults from the common nuzlocke rulesets; the user can override any entry.
- **HP tracking from observations**: during play the tool carries the set of exact HP values consistent with what the player can see (HP bar, messages) rather than asking for numbers it cannot know.
- Local web UI from an early milestone.

### Out of scope for v1 (but don't design it out)
- Double battles. The state layout keeps a slot dimension for two active slots per side, but targeting logic is not pre-built — that is the real cost of doubles and is deferred.
- Chaining battles (gauntlets with carry-over). The engine already emits the end-of-battle party state.
- Other generations. See §9.
- Exact cartridge RNG reproduction or RNG manipulation.
- General hidden-information or belief tracking beyond the HP-set tracking above.
- Wild battles.
- EV training and vitamin use as prep variables.

## 3. Decisions

| Area | Decision |
|---|---|
| Objective | Primary: maximize P(win with zero faints). Also reported per line: P(win), E[faints], P(≤1 faint). User selects the ordering; default is lexicographic on the primary, then P(win). |
| Ties | Two lines tie when their estimates are statistically indistinguishable (overlapping confidence intervals, or within a configurable tolerance, default 0.5 pp). Never exact equality on noisy estimates. |
| Tie-breaker | Fewest resources spent (TMs, Heart Scales, shards, held items, prep effort), with **user-weighted costs** so "prep effort" is a number, then most HP/PP remaining |
| Terminal states | A player faint is **terminal** for the primary objective (counts as a loss). Playouts and search stop there. |
| Randomness | Exact probability distributions, not the cartridge's RNG sequence |
| Format | Singles first; state ready for doubles |
| Rules | Set mode, level caps, held items only |
| Opponent | Full player knowledge of the opponent; decomp-accurate trainer AI including its random draws and partial knowledge of the player |
| Pre-battle search | Team + lead from box, held item assignment, prestatus, predamage (HP brackets derived from the damage calc), optional level-to-cap |
| Moveset search | Optional toggle; sources: bag TMs, tutors, Move Relearner; costs reported |
| Feasibility | User declares what's available (prestatus methods, tutor access, etc.) |
| Unsupported mechanics | Refuse loudly and name what's missing; never approximate silently. In prep search, exclude the affected *candidate* with a warning instead of refusing the whole run. |
| Output | Recommended setup + per-turn ranking + plan tree branching on observable events |
| Latency | Seconds per turn; minutes for pre-battle setup; optional overnight runs with checkpoints |
| Sources | Decomp (`pret/pokeplatinum`) is authoritative, including its `res/` data tree where it is parseable; Showdown gen4 is a cross-check and a test oracle; Smogon damage calculator for exact damage arrays; PKHeX as the save-layout cross-check |
| Fidelity reporting | Every mechanic carries a confidence class (decomp-verified, golden-traced, Showdown-derived, research-only). The UI shows the weakest class in play for each fight next to the odds. |
| Reproducibility | Every run records the save hash, trainer, settings, seeds, table versions and result, so any recommendation can be regenerated later. |
| Interface | Local web UI (Python backend) plus a Python API |
| Hardware | CPU JAX on native Windows for development; GPU via WSL2 or cloud later |

## 4. Architecture

Four layers, each usable and testable on its own:

```
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1. Data     │──▶│ 2. Engine    │──▶│ 3. Battle    │──▶│ 4. Prep      │
 │ tables, save│   │ JAX step, AI │   │ search       │   │ search       │
 └─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                              ▲                   ▲
                                              └──── Web UI / API ─┘
```

### 4.1 Data layer

**Static game data** is generated offline into versioned `.npz` tables (plus a JSON manifest mapping names ↔ integer ids):

- Species: base stats, types, abilities, gender ratio, base friendship, weight (for Low Kick / Grass Knot), **forms** (Rotom, Giratina, Wormadam, Shaymin — each form has its own stats and types).
- Moves: power, type, category (physical/special/status), accuracy, PP, priority, target, **effect id**, effect parameters, effect chance, and flags (contact, sound, punch, protectable, etc.).
- Type chart, natures, items (held effects), abilities.
- Learnsets, TM/HM compatibility, tutor compatibility, relearner moves.
- **Trainers**: party (species, level, moves, held items, and the per-Pokémon difficulty value that sets IVs; the generated PID sets nature, gender and ability), AI flags, trainer bag items, and **battle start conditions** (weather or fog inherited from the overworld). Examples: hail on Routes 216/217 and at Lake Acuity, rain on Routes 212/215, sandstorm on Route 228, fog on parts of Route 210 and Mt. Coronet. Fog has no move equivalent and reduces accuracy, so it must be a field state the engine understands.

Pipeline: `tools/extract_decomp.py` (species, moves, learnsets, trainers, AI scripts and start conditions from the decomp's `res/` tree) and `tools/extract_showdown.py` (cross-check, and the fallback for anything the decomp does not hold in parseable form) → `tools/build_tables.py` → `data/gen4_platinum/*.npz`. Conflicts between the sources are written to a report, and the decomp wins. Making the decomp the sole bulk source where possible removes a whole class of source conflicts; milestone 1 checks how much of the `res/` tree is usable.

Exact decomp file paths are to be located during milestone 1. The completeness of the battle and AI code in the decomp is checked first (see §10).

**Save parser** (`saves/`): reads Gen 4 save blocks, validates checksums, **selects the newer of the two save slots by save counter**, and decrypts and unshuffles Pokémon data (PID-based block order plus PRNG XOR). It outputs party, PC boxes, bag and badges as plain Python objects. Supports raw `.sav` and DeSmuME `.dsv` (footer stripped).

Boxed Pokémon store no stats, so the parser **computes level from experience and stats from IVs, EVs, nature and level itself**. The party's stored stats are used as a test of that computation, not as the source of truth. PKHeX is the de facto reference for Gen 4 save layouts; a few saves exported through it are diffed against the parser's party, box and bag output (§5).

### 4.2 Engine layer (JAX)

**State** is a fixed-shape pytree of small integer arrays (`int8`/`int16` for storage; `int32` wherever arithmetic needs it, which includes every damage-calc intermediate). The authoritative field list is **transcribed from the decomp's battle structs** (the per-Pokémon battle struct, the per-side struct and the field/context struct) and ticked off against them in a checklist in `engine/STATE.md`; the list below is the minimum, not the inventory.

- Per side × 6 slots: species, form, level, stats, current HP, status (+ sleep/toxic counters), moves, PP, ability, held item, **last consumed item** (not just a flag — Recycle, Unburden, Knock Off, Trick and Fling all need the id), **current types** (mutable: Conversion, Conversion 2, Roost, form changes), gender, friendship, IVs or precomputed Hidden Power type/power.
- Per side, active slot: stat stages, volatile statuses, choice lock, last move used, multi-turn move state, protect counter. The volatile set includes at least: confusion, Leech Seed (with source), Substitute HP, Taunt, Encore, Torment, Disable, Yawn, Ingrain, Aqua Ring, Magnet Rise, Embargo, Heal Block, Nightmare, Curse, Perish Song counter, Destiny Bond, Grudge, Endure, Focus Energy, Lock-On/Mind Reader, Foresight/Miracle Eye, Charge, Stockpile count, Defense Curl flag, Minimize flag, Bide/Rage/Uproar/Rollout/Fury Cutter counters, semi-invulnerable and recharge turns, binding (move, turns) and trapping, Counter/Mirror Coat/Metal Burst damage tracking, Truant and Slow Start counters, Transform and Mimic state, Power Trick, Baton Pass carry-over.
- Per side: hazards (Spikes layers, Toxic Spikes, Stealth Rock), screens and timers (Reflect, Light Screen, Safeguard, Mist, Lucky Chant, Tailwind), Wish (timer, healer HP), Future Sight / Doom Desire (timer, damage, target), trainer item inventory.
- Field: weather + timer (including **Fog**), Trick Room + timer, Gravity, Mud Sport / Water Sport, turn count, battle-over flag, winner, a **faint flag per player Pokémon** (for the zero-death objective).
- **AI knowledge** (see *Trainer AI* below): which player moves the AI has seen, whether the player's ability has been revealed, and whatever else the decomp's AI context remembers across turns.

**Actions** (singles): player = 4 moves + 5 switches = 9 discrete actions with a legality mask. Opponent = moves, switches and trainer item use. Struggle and forced moves are resolved inside the engine. Forced replacement after a faint is a separate decision point with its own mask (Set mode means only the fainted side picks). **Simultaneous faints** need an ordering rule for who chooses first and what the second chooser can see; this is taken from the decomp, not assumed. Under the primary objective a player faint is terminal, so the player's replacement decision only arises for the secondary objectives.

**Explicit chance outcomes.** The core function takes chance outcomes as an input instead of drawing them internally:

```python
step(state, player_action, opp_action, outcome) -> state
sample_outcome(state, actions, key) -> outcome          # Monte Carlo
enumerate_outcomes(state, actions) -> (outcomes, probs)  # exact chance nodes
```

`outcome` is a fixed-size struct with a **documented worst-case layout**, per actor:

- per hit, up to 5 (multi-hit moves): hit/miss, crit, damage roll (0–15 = 85–100%), secondary effect proc;
- per action: speed-tie winner, multi-hit count, full paralysis, confusion self-hit, sleep/freeze/confusion duration rolls, Quick Claw, King's Rock / Razor Fang flinch, Focus Band, freeze thaw, contact-ability procs (Static, Flame Body, Effect Spore, Poison Point, Cute Charm, Rough Skin), Shed Skin, Magnitude level, Present outcome, Psywave roll;
- per decision: the AI's internal draws (see *Trainer AI*).

Sampling and exact enumeration both use the same `step`, so there is only one source of truth for the rules. Enumeration **merges outcomes that produce identical states** (e.g. damage rolls that all KO) to keep chance nodes small.

**Sampling-only mechanics.** Moves whose chance node is too wide to enumerate — Metronome, Assist, Sleep Talk, and anything else the coverage registry marks as such — are supported in Monte Carlo (Stage A) but refused by exact enumeration (Stage B). The coverage report says which class each mechanic is in, so a fight that needs them still gets Stage A odds and an honest "no exact plan tree" message.

**Turn structure** (every phase always runs, masked when not applicable):
1. Opponent AI chooses (as a probability distribution; see below).
2. Determine order: switches/items first, then priority bracket, speed (Trick Room, paralysis, Choice Scarf, Quick Claw, etc.), and the speed-tie outcome.
3. Execute the first actor, check faints, execute the second actor, check faints.
4. End-of-turn effects in Platinum's exact order (taken from the decomp).
5. Faint replacement decision points.
6. Check whether the battle is over (including the player-faint terminal rule).

**Move effects are data-driven.** Each move maps to an effect id and parameters (e.g. "stat change: stat, stages, target, chance"). The engine implements each *effect* once rather than each move, which keeps the computation graph and compile time manageable. Abilities and held items are handled the same way: hook points at fixed phases (damage modifiers, on-switch-in, end of turn, on-hit), each evaluated as a masked select. Code may use `lax.cond`/`lax.switch` for readability; under `vmap` they lower to the same masked selects.

**Per-fight specialization.** The set of effect ids, abilities and held items present in a matchup — the opponent's team plus the union of the player's candidate pool — is passed to `jit` as a static argument. Only those branches are compiled. For prep search the pool is the whole eligible box, which is still far smaller than every mechanic in the game. Recompiling once per fight (seconds to a minute) fits the latency budget.

**Damage** follows the Gen 4 formula with the decomp's exact integer truncation order: physical/special split, 2× crits (3× with Sniper), 16 damage rolls, STAB, type effectiveness, and item/ability/weather modifiers in their exact positions. The engine exposes `damage_table(state, attacker, move) -> int32[16]` (and the crit variant) as a reusable primitive; prep search and the HP tracker both consume it.

**Trainer AI** (`ai/`): a pure JAX function `ai_policy(state, ai_outcome) -> opponent action`, together with `enumerate_ai_outcomes(state) -> (ai_outcomes, probs)` and a sampler. It reproduces the decomp's AI script scoring for the trainer's AI flags, its switching logic and its item-use rules. Two things the first draft missed are now explicit:

- **The AI has its own random draws beyond tie-breaks**: the simulated damage roll in its KO check, the ability guess when the player's ability has not been revealed, and the "if random less than N" branches in the scripts. These are enumerated as chance variables (`ai_outcome`) exactly like battle outcomes, so the AI's action distribution is exact. If enumeration of a script turns out too wide, the registry marks that trainer's AI as sampling-only and Stage B says so.
- **The AI has partial knowledge of the player**: it scores only the player moves it has seen, and it guesses unrevealed abilities (and items, if the decomp does so). This knowledge is state (see above) and evolves during the battle. "Full knowledge" in §2 refers to the *player's* knowledge of the opponent.

The output is a distribution, so search treats the opponent's choice as a chance node rather than an adversary.

**Coverage registry** (`engine/coverage.py`): a list of every implemented effect id, ability, held item and AI script feature, each tagged *exact* or *sampling-only*. Before any search, `check_coverage(battle_setup)` either passes or raises an error listing exactly what's missing (e.g. *"Gardenia: Grass Knot (move effect), Natural Cure (ability)"*). In prep search the check runs **per candidate Pokémon**: an unsupported candidate is excluded with a warning rather than failing the whole run. The same report drives a progress view per trainer.

**HP tracker** (`engine/observe.py`): during play the tool knows the opponent's exact starting HP and the engine's damage tables, so after each turn it carries the **set of exact HP values consistent with the visible HP bar and messages** instead of a single assumed number. Search runs over that set (weighted by the outcome probabilities that produced each value). The per-turn UI asks for observations (HP bar, status, messages), not for internal outcomes, and infers which enumerated outcome(s) occurred.

**Reference engine** (`reference/`): a slow, plain-Python, branch-heavy implementation of the same rules and API, written for readability and checked line by line against the decomp. It is the oracle for differential tests (§5).

### 4.3 Battle search layer

Input: a battle state (or HP-tracked set of states). Output: a ranking of player actions by P(zero-death win), each with a confidence interval, plus P(win), E[faints], and an optional plan tree.

**Stage A: Monte Carlo.** For each legal action, run N batched playouts to the end of the battle or to the first player faint. The **playout policy** is a **one-ply expectimax** over the engine's exact damage tables: it picks the action with the best immediate zero-faint-safe expected outcome (safe KOs first, switch out of matchups that threaten a faint). This is cheap in batch and risk-aware, unlike greedy-by-damage, which is risk-neutral. Playout odds are a lower bound on optimal play. **Common random numbers**: the actions compared at a node share the same outcome seeds, which sharply reduces ranking variance. **Racing**: rather than a fixed N per action, sampling continues only for actions whose confidence intervals still overlap the leader's, so the budget goes where the ranking is actually uncertain. Stage A is also the fast evaluator for prep search.

**Stage B: depth-limited expectimax.** Search player decisions to depth *d*, with exact chance nodes for both battle outcomes and the opponent's AI distribution (including the AI's own draws), and Stage A rollouts at the leaves. It is batched level by level: all frontier states at a level are evaluated in one `vmap`. The explored tree is exported as the plan tree, whose branches are labeled by **observable events** ("Geodude survives above ~40%", "Stealth Rock goes up") rather than outcome ids. Depth adapts to the time budget.

**Opponent-action pruning with an error bound.** The AI distribution multiplies branching by every scored action. Opponent actions below a small probability epsilon are dropped from the tree, and the dropped mass is carried as an explicit error term on the reported probability (reported as "P ≥ x, at most y unexplored"), so pruning never makes a line look safer than it is.

Ties are ranked by the tie rule and tie-breaker in §3.

### 4.4 Prep search layer

**Configuration space:** team (≤6 from the box, respecting level caps) × lead × held item assignment (from the bag) × per-mon prestatus (from the user's declared methods) × per-mon **HP bracket** × (optional) level-to-cap × (optional) movesets.

**HP brackets are derived per matchup from the damage calc**, not from fixed fractions. The breakpoints are: every damage value the opponent's moves can deal to the candidate (from `damage_table`, including crits) — these decide survival; plus the behavior thresholds: ability pinch at ≤1/3, pinch stat berries and Lansat/Starf/Custap/Micle at ≤1/4, Sitrus/Oran **and the Figy family at ≤1/2** (Gen 4), Reversal/Flail power bands, Focus Sash at full HP, and the AI's HP checks found in the decomp. Each bracket is represented by its most favorable exact HP, and the search can split a bracket when two exact values in it behave differently.

**Level-to-cap toggle:** when enabled, each candidate may also be evaluated at any level up to the cap, with stats and newly learned moves updated; the cost reported is the level gap.

**Evolution toggle:** a candidate that can evolve by level (within the cap) or by a stone in the bag is also evaluated in its evolved form, with the new stats, types, abilities and learnset. Evolution is free in nuzlocke terms, so its only reported cost is the stone or the level gap.

**Moveset toggle:** when enabled, candidate moves come from bag TMs (single-use in Gen 4), tutors and the Move Relearner as declared available. Each line lists what it costs ("uses TM26 Earthquake, 1 Heart Scale"). **Locked moves**: the user marks moves that must stay (HMs needed for traversal, for example), and the search never replaces them.

**Resource reservation:** TMs, Heart Scales, shards and stones are single-use or scarce, so the user can mark any of them as reserved for a later fight. Reserved resources are excluded from the search rather than merely penalized in the tie-breaker.

**Funnel:**
1. **Prune** with fast heuristics: type matchups against the trainer's team, speed comparisons, and OHKO checks both ways. Pruning is **conservative**: it only drops candidates that are dominated in every role (lead, sweeper, pivot), never merely "gets OHKO'd", since a fragile mon can still be the right safe switch-in.
2. **Screen** survivors with a small Stage A budget each, all in one batch.
3. **Successive halving:** repeatedly double the simulations per candidate and drop the bottom half.
4. **Finalists** get a full Stage B search.

The whole funnel, including step 1, is **validated against exhaustive search on Roark** (milestone 5) before it is trusted on larger fights.

**Anytime and checkpointing:** a time budget (default: minutes; overnight optional). A current best answer is always available and improves over time. Long runs checkpoint their state to disk and can resume.

### 4.5 Web UI and API

- **Backend:** FastAPI wrapping the Python API: load a save, list trainers, show coverage, run prep search (with progress streaming), and per-turn advice where you enter **what you observed** (HP bars, statuses, messages) and the tool infers the battle state.
- **Frontend:** a simple single-page app for the recommended setup, ranked actions with odds, a plan tree viewer keyed by observable events, and a **fidelity label per fight** ("decomp-verified + golden trace" versus "Showdown-derived" and so on) shown next to the odds, so the user knows when a line rests on unverified behavior.
- It runs locally on `localhost`.

## 5. Correctness and testing

Fidelity is the product: a wrong mechanic silently corrupts every recommendation.

- **Unit tests from the decomp:** damage formula cases (including truncation edge cases), stat calculation, the type chart, speed order and priority.
- **Damage-array tests:** `damage_table` compared exactly against the Smogon damage calculator's 16-roll output across a generated matrix of matchups, items, abilities and weather.
- **Differential tests:** random valid states plus random actions plus fixed outcomes, run through the JAX engine and the reference engine; states must match exactly. Property-based generation with Hypothesis.
- **Showdown oracle, made deterministic:** scripted battles run in Showdown's gen4 mod; the battle log is parsed for the *realized* hit/miss, crit and damage values, which are fed to `step` as the `outcome`. The resulting states must match turn by turn. Known discrepancies are documented in an allowlist that names the decomp behavior that wins.
- **Golden traces, generated by the emulator:** DeSmuME and melonDS run Lua scripts. A script (`tools/trace_battle.lua`) plays scripted inputs against the real ROM and dumps the battle RAM and RNG state after every turn, which yields the exact realized outcomes for free. Those traces are replayed through `step` with the recorded outcomes and the states must match. This is the closest available thing to running the game's own code as an oracle, and it turns golden traces from a hand-recorded chore into a generated suite covering every supported trainer.
- **AI tests:** score tables for specific states checked against hand-traced decomp AI scripts, including the AI's knowledge state (seen moves, unrevealed ability) and its simulated-roll KO checks.
- **Save parser tests:** computed stats for party Pokémon must equal the stats stored in the save, and party, box and bag output is diffed against PKHeX exports of the same saves.
- **JAX hygiene tests:** `step` jits, vmaps over batches, has no dtype promotion to 64-bit, and keeps shapes stable.
- **Chance-enumeration tests:** enumerated probabilities sum to 1 and match sampled frequencies, for both battle outcomes and AI outcomes.
- **State checklist:** `engine/STATE.md` maps every field of the decomp's battle structs to a pytree field or an explicit "not needed because …" note; a test fails if a struct field is unaccounted for.

## 6. Performance

- Benchmark: battles per second (full playouts) and `step`s per second at batch sizes 1k–1M, tracked over time. Compile time is a first-class metric.
- **Milestone 0 spike** (before the engine exists): a throwaway masked step with ~100 fake effects at batch 10k on CPU, measuring steps/s and compile time, to confirm the one-big-step design before committing to it.
- Keep the graph small with data-driven effects and per-fight specialization.
- Small integer dtypes save memory bandwidth on GPU; they are not expected to help on CPU, and damage intermediates are always `int32`.
- No Python control flow inside `jit`, fixed loop bounds with masking, and a hard turn cap. Playouts terminate on the first player faint, which ends most losing lines early.
- **Chunked playouts with compaction.** Under `vmap`, a `while_loop` runs until the slowest battle in the batch finishes, so one long stall wastes the whole batch. Playouts instead run in fixed `scan` chunks of a few turns; between chunks, finished battles are compacted out and the batch is refilled, which keeps utilization high.
- **Persistent compilation cache.** JAX's persistent cache is enabled so per-fight specializations survive restarts; cache keys include the table version.
- Develop on CPU (native Windows); GPU runs via WSL2 or cloud, since JAX CUDA wheels are Linux-only.

## 7. Repository layout (proposed)

```
pyproject.toml            # uv-managed, Python 3.12
docs/design.md
tools/                    # offline extraction, table building, trace_battle.lua
config/level_caps.toml    # per-trainer level caps (user-overridable)
runs/                     # run records: save hash, trainer, settings, seeds, result
data/gen4_platinum/       # generated tables (+ manifest)
src/nuzline/
  data/                   # table loading, id maps
  saves/                  # Gen 4 save parser
  engine/                 # JAX state, step, outcomes, effects, coverage, observe
  engine/STATE.md         # decomp struct -> pytree field checklist
  ai/                     # trainer AI policy + AI outcome enumeration
  reference/              # slow Python reference engine
  search/                 # battle search (MC, expectimax, plan tree)
  prep/                   # prep search (funnel, brackets, checkpoints)
  web/                    # FastAPI backend + static frontend
tests/
benchmarks/
```

Tooling: `uv`, `pytest`, `hypothesis`, `ruff`, and `jax[cpu]` for development.

## 8. Milestones

0. **Performance spike:** the masked-step benchmark from §6. Go/no-go on the single-step design and a first compile-time number.
1. **Roark end to end:** data tables for everything in Roark's fight and a sample player team, extracted from the decomp's `res/` tree where possible; `engine/STATE.md` transcribed from the decomp structs; JAX engine + reference engine + Roark's AI (with AI outcomes and knowledge state); coverage registry; the emulator Lua trace script and a first Roark golden trace; differential, damage-array and decomp-derived tests passing; simple Stage A per-turn ranking from a hand-entered party.
2. **Web UI v0:** load a hand-entered party, pick Roark, show coverage, per-turn ranking driven by observations (HP tracker).
3. **Save parser:** party, boxes and bag from `.sav`/`.dsv`, with computed stats; wired into the UI.
4. **Battle search Stage B:** exact chance nodes (battle + AI), expectimax, plan tree export and viewer with observable-event labels.
5. **Prep search:** team/lead/items/prestatus/derived HP brackets, validated against exhaustive search on Roark; then level-to-cap and the moveset toggle; anytime and checkpointing.
6. **Coverage expansion:** gym by gym, then rivals, Galactic bosses and the Elite Four + Champion, tracked by the coverage report. Start-condition fights (hail, rain, sandstorm, fog) are explicit entries on this list.
7. **GPU deployment:** WSL2 or cloud setup and benchmarks.

## 9. Extensibility to other generations

- The **data schema** is generation-neutral; each generation gets its own table directory.
- **Mechanics** are generation-specific modules selected by a static config passed to `jit` (`static_argnums`), so each generation compiles only its own rules.
- No generic multi-generation abstractions until a second generation is actually being added.

## 10. Risks and open items

- **Decomp completeness:** how much of `pokeplatinum`'s battle and AI code is decompiled and named. *Check first in milestone 1.* If parts are missing, fall back to Showdown plus documented research (e.g. Smogon and Bulbapedia mechanics write-ups) and mark those mechanics as lower confidence.
- **AI fidelity:** the trainer AI is the largest correctness risk and the hardest to verify. Its random draws and knowledge model are now explicit, which makes the tests precise but also enlarges the surface. Golden traces from real play are the best check.
- **AI outcome width:** some AI scripts may roll often enough that exact enumeration is impractical. Mitigation: merge draws that lead to the same scored action; otherwise mark sampling-only.
- **Compile time** of a large unified `step`. Mitigated by data-driven effects and per-fight specialization; measured from milestone 0.
- **Chance-node blow-up** in Stage B. Mitigated by merging outcomes, the sampling-only class and adapting depth.
- **Prep search size.** Mitigated by pruning, successive halving and anytime budgets; the funnel and screening policy are validated against exhaustive search on Roark.
- **Simultaneous-faint ordering and AI item knowledge** are open questions to settle from the decomp in milestone 1.
- **Windows GPU:** requires WSL2 or cloud.
