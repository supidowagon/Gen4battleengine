# Stage B Solver — Design

Date: 2026-10-30 · Status: approved by the user 2026-10-30 · Parent spec: `docs/design.md` (§4.2, §4.3)

## Why

The product is a fast engine that **solves** single-player nuzlocke battles: it tells
the player the line that maximizes P(win with zero faints). Today the project has the
engine, Roark's AI and Stage A (Monte Carlo ranking of the first move, `nuzline rank`).
Stage A's odds are a lower bound under a heuristic playout policy; it cannot say what
to do on turn 2 or later, and it cannot show a line. Stage B is the solver: an exact
expectimax over the battle's chance events and the AI's action distribution, with
Stage A as the leaf evaluator, that outputs a ranked first move **and a plan tree**
("do this; if you see X, do Y").

## Decisions taken with the user (2026-10-30)

- **The web UI is dropped until the engine, solver and prep search are complete.** The
  UI is only a viewer for the user; nothing in building or testing the engine needs it.
  `docs/design.md` §8 is reordered accordingly (see *Milestone order* below). Until then
  the CLI is the interface.
- Milestone 2 is now this solver.

## Goals

1. **Exact chance nodes**: `enumerate_outcomes(tables, state, actions) -> (outcomes,
   probs)` (spec §4.2) — every battle outcome of a turn for a fixed action pair, with
   exact probabilities, merged where the resulting states are identical.
2. **Honest search**: depth-limited expectimax in which the player chooses only on what
   the player can **observe**, not on hidden state (see *Information sets*).
3. **Plan tree** export (JSON) with branches labeled by observable events, plus a
   readable text rendering in the CLI.
4. **`nuzline solve`** CLI command for Roark: root ranking with values, confidence
   intervals and the unexplored-probability bound, and the plan tree.
5. **Anytime**: a time or playout budget; a result is always available and improves with
   budget.

## Non-goals (this milestone)

- Web UI, live turn-by-turn advice from typed-in observations (the plan tree already
  branches on observations; an interactive "I saw X, now what" CLI walk is a later,
  thin addition).
- Prep search, save parsing, fights other than Roark.
- Optimizing the secondary objectives (P(win), E[faints]); they are reported, not
  optimized (see *Objective*).
- GPU work.

## Anti-goals (would make it a failure even if it runs)

- **Peeking**: a plan whose turn-2 choice depends on something the player cannot see
  (the opponent's exact HP inside one HP-bar pixel, a hidden sleep counter, the AI's
  internal draws). Its odds would be unachievable.
- **Silent optimism**: pruned probability mass, or Monte Carlo leaf noise, that makes a
  line look safer than it is without saying so.
- **Silent approximation**: a mechanic that cannot be enumerated exactly being sampled
  inside Stage B without the report saying so (spec §3: refuse loudly).

## How it works

### 1. Exact battle chance nodes (`engine/enumerate.py`)

For a state and a fixed action pair, list the joint outcomes of every `Outcome` field
that can matter, with exact probabilities from the same formulas `sample_outcome` uses
(accuracy, crit odds, 1/16 rolls, effect chances, paralysis, confusion, durations,
speed ties, multi-hit counts, contact procs, thaw):

- **Relevance masks first.** A field is enumerated only when this turn can read it
  (a damaging move reads hit/crit/roll; secondary only when the move has an effect
  chance; `n_hits` only for multi-hit moves; `full_para` only when paralyzed; and so
  on). Irrelevant fields take the `empty_outcome` value with probability 1.
- **Pre-merge damage rolls**: the 16 rolls of a hit collapse to the distinct damage
  values of `damage_table` (and all rolls that KO collapse to one), weighting each by
  its count.
- **Step and merge**: the remaining joint combinations are run through `step` in one
  batch and identical next states are merged (probabilities summed). Merging is exact by
  construction: two outcomes are merged only when `step` produced bit-identical states.
- **Width cap**: if the pre-merge combination count exceeds a cap (builder's choice,
  order 10^5), or the turn uses a mechanic graded sampling-only, the action pair is
  **sampling-only**. At the root that player action is refused by name (the other root
  actions are still solved). Below the root the node is valued as a Stage A leaf instead
  of being expanded, and the report says so: how many such nodes and their total reach
  probability. *(Controller's call.)* Note: `engine/coverage.py` has no sampling-only
  grade yet (nothing is graded `SAMPLING_ONLY`); this milestone adds that path. Measured for
  Roark turn 1 with the sample party: 1 to ~111 distinct next states per action pair
  (mean ~14) after merging.

The opponent's action is its own chance layer before the battle outcomes: the exact
`ai_action_distribution` (already factored and exact). The opponent's forced
replacement after a faint is deterministic given the state (`send_replacements`).

### 2. Information sets (the observation function, `engine/observe.py`)

`observe(state_before, actions, state_after) -> Observation`: everything the player
sees during the turn and at its end. Two children of a node with equal observations are
indistinguishable to the player and form one **information set**; the player's next
decision is one action for the whole set.

Observable (Platinum, singles): the opponent's action **only when it actually runs**
(move used, switch target, item used; a choice that never runs because the opponent
fainted first, was fully paralyzed, asleep, frozen or flinched is not seen); turn order; per hit: hit or miss, critical hit, effectiveness message,
number of hits; every status, volatile, stat-stage, weather, hazard and screen message;
ability and item activation messages; flinch, full paralysis, confusion self-hit, sleep
and thaw messages; faints and replacements; the **player's exact HP** (shown as
numbers); the **opponent's HP as HP-bar pixels** (the decomp's bar formula, 48 pixels,
confirmed against the decomp by the builder). Not observable: the opponent's exact HP
within a pixel, damage rolls as such, hidden counters (sleep, confusion, Bind turns
left), the AI's internal draws.

Practical form: the observation is computed from the two states and the outcome
(fields such as "missed" and "crit" come from the outcome; HP from states), hashed to a
key. Because hidden counters stay hidden, an information set can hold states that
differ in those counters or in the opponent's exact HP; that is exactly the HP-set
tracking of spec §2 / §4.2 (*HP tracker*), and it replaces a separate tracker module.

### 3. Search (`search/stage_b.py`)

A tree over **belief nodes**. A belief node is a weighted set of states (initially one
state) at a player decision point.

- **Player node**: for each legal action `a` (the player's legality is observable and
  identical across the set), value(a) = Σ over states s in the set, weight(s) ×
  Σ over opponent actions o ~ AI(s) × Σ over merged outcomes → child state, grouped by
  observation into child belief nodes. The node's value is max over `a`. One action per
  belief node is what makes the plan executable.
- **Terminal states**: a player faint is a loss for the primary objective (value 0,
  stop); an opponent wipe is a win (value 1).
- **Leaves**: at the depth limit, each state is valued by Stage A playouts
  (`playout`, existing code). Leaf budgets are allocated by **reach probability**
  (a state reached with probability 10^-4 does not get the playouts a 30 % state gets),
  with a floor so every leaf gets some playouts; identical leaf states (transpositions)
  are evaluated once.
- **Pruning with a bound**: chance children (opponent actions and outcomes) whose reach
  probability is below ε are not expanded. Two numbers are reported for every action:
  the **estimate**, in which pruned children are valued by a small Stage A batch (this
  is what ranks and chooses), and the **lower bound** "P ≥ x, at most y unexplored", in
  which pruned children count as 0 (spec §4.3). The pruning test checks y against the
  summed reach probability of pruned children. Opponent actions below ε are handled the
  same way. *(Controller's call.)*
- **Batched level by level** (spec §4.3): all frontier states of a depth are enumerated
  and stepped in large `vmap` batches; Python orchestrates the tree between levels.
  Merging and grouping happen on the host.
- **Anytime**: iterative deepening (depth 1, 2, …) under a time or playout budget;
  each completed depth replaces the previous answer. Depth 0 equals Stage A.

**Leaves see the full state.** The leaf evaluator is Stage A's `playout_policy`, which
reads concrete state (including the opponent's exact HP). That is peeking, but only
inside a heuristic estimate of what comes after the plan; the plan itself never peeks.
It is accepted for this milestone and stated in the report and docs; an
observation-only playout policy is a later improvement. This keeps "depth 0 = Stage A".
The exhaustive test solver (§6) has no leaves, so it is exact over information sets.
*(Controller's call.)*

**Confidence**: the reported value of an action is an expectation over leaf estimates;
its confidence interval propagates the leaf estimates' binomial variances through the
tree's weights. Max-over-noisy-children bias is real; the root ranking uses
`tied_with_best` exactly as Stage A does (overlapping intervals are reported, not
reordered), and the report states the leaf budget.

#### Amendment 2026-10-31: precision target

Amends the implementation plan's rulings 8 and 9
(`docs/superpowers/plans/2026-10-30-stage-b-solver.md`). A fixed leaf budget per pass is
either too small (wide intervals) or wasteful: a depth-2 solve measured needed about 135 000
leaf playouts for the leader's half-width to reach 1.5 percentage points, while the
benchmark's starting budget (`2 * K_2 * 4300` = 696 600) overshot about 7 times. Passes now
stop at a precision target (`SolveSettings.precision`, a fraction; `nuzline solve
--precision PP`, default 1.5 pp; `--playouts N` keeps the fixed budget and turns the target
off).

- **Ruling 8 (allocation), amended.** The floor is `leaf_floor = 1` per half (was 2;
  measured: the same inner-decision quality, 13 % cheaper), and it applies to **round 0
  only**: round 0 is the old allocation (`n_j = 2 * max(f, round(B_d * c_j / sum c / 2))`,
  pruned and sampled evaluations exactly `2f`). A **top-up round** `r >= 1` adds a batch
  as large as everything the pass has run so far (the cumulative total doubles), split in
  proportion to each evaluation's summed reach, **forced evaluations (pruned opponent
  actions, sampled pairs) and pruned groups included**, with **no floor**: an evaluation
  whose rounded share is 0 gets nothing, and a refused one gets none. Each round draws
  fresh key streams (`fold_in(fold_in(fold_in(pass key, 2), r), 0 | 1)` for leaves and
  forced pairs; round 0 keeps `fold_in(pass key, 0 | 1)`); wins and playouts add into each
  evaluation's halves and `refused` is or-ed.
- **Ruling 9 (budgets), amended.** `B_d` (`budget_playouts`, else `n_depth0` per legal root
  action) is the **round-0** budget. After each round the pass picks the leader by half-A
  root values (the inner nodes' argmax, ties to the lowest action) and computes its half-A
  half-width `1.96 * sqrt(sum_j W_j^2 * var_a_j)`, `var_a = p~(1-p~)/n_a`, `p~ = (k_a+1) /
  (n_a+2)`. It tops up while that exceeds `precision`, for at most 8 top-up rounds and not
  after `budget_seconds` (when set) has elapsed since the solve started. The pass always
  completes; the backup, ranking and plan run once on the merged values. Half B never
  decides a stop or a share (ruling 5); it still gives every reported value and interval.
  Without a target a pass is round 0 alone, exactly as before. The result records, for the
  final pass, the top-up rounds, the half-A half-width and whether the target was met.

### 4. Objective

Stage B optimizes the primary objective, P(win with zero player faints). The engine
currently ends the battle on the first player faint (`_faint_check`), so P(win) equals
the primary objective and E[faints] adds nothing; `nuzline solve` therefore reports the
primary objective only, and the spec's tie-break on P(win) is inert for now. Real P(win)
and E[faints] need battles that continue past a player faint (a non-terminal mode in
`step`, a player replacement policy); that is out of scope here and belongs with the
secondary orderings, later. *(Controller's call.)*

### 5. Output

- `nuzline solve --party P --trainer roark [--lead N] [--opp-active N] [--budget SECONDS
  | --playouts N] [--max-depth D] [--epsilon E] [--tree FILE.json] [--show-depth K]`.
- Prints the root ranking (action, P(win, 0 faints) estimate with CI, lower bound,
  depth reached, unexplored mass, sampled-node mass) and the plan tree to `--show-depth` levels, ASCII
  only (like `nuzline rank`), e.g.:

  ```
  1. Ember   P(win, 0 faints) 74.1 % [72.0, 76.2]   unexplored <= 0.3 %
     if Geodude uses Rock Throw, misses        -> Ember
     if Geodude uses Rock Throw, hits, you 31/58 HP -> Ember
     if Geodude uses Rock Throw, crit, you 12/58 HP -> Switch -> Starly
  ```
- `--tree` writes the full plan tree as JSON: per node the observation that leads to it
  (structured fields plus a rendered label), its probability, the recommended action
  and every action's value, the depth, and leaf/pruned markers. A run record (save or
  party hash, trainer, settings, seeds, table and code versions, result) is written
  under `runs/` (spec §3, *Reproducibility*).
- Refusals (unsupported or sampling-only mechanics, refused AI) print `refused: ...`
  and exit with the existing refusal code.

### 6. Correctness tests

- `enumerate_outcomes`: probabilities sum to 1; for random states and action pairs the
  enumerated distribution over next states matches the frequencies of
  `sample_outcome` + `step` (statistical test, fixed seeds); a merged child equals
  `step` on each of its member outcomes; the reference engine agrees on the child set
  for generated cases.
- `observe`: unit tests per message class; an equal-observation pair from hidden
  counters and an HP-within-a-pixel pair are grouped; the player's exact HP difference
  splits.
- Stage B: on small constructed endgames (e.g. one Pokemon each, low HP, few moves)
  where the game tree is finite and tiny, an **exhaustive exact solver** (no leaves,
  no pruning, written plainly in Python over `step` and `enumerate_outcomes`) gives the
  true value; Stage B at sufficient depth must match it exactly. Depth 0 must equal
  Stage A on the same keys. An information-set test: a constructed position where
  peeking would change the decision must produce the non-peeking choice.
- Pruning bound: the reported unexplored mass equals the summed reach probability of
  pruned children.

### 7. Performance

Measured today on the development CPU (i5-13600K, 20 host devices): about 2,800 full
playouts per second (Roark, mean 8.6 turns) and ~20 s cold compile for the playout
program. Stage B's cost is dominated by leaf playouts; the budget allocation by reach
probability is what makes depth 2 affordable. Target for this milestone, on that CPU:
`nuzline solve` for Roark's opening with the sample party completes depth 2 within
**10 minutes** and depth 1 within **2 minutes**, cold, **at a precision where the
leading root action's 95 % interval is at most ±1.5 percentage points** (so the time
target cannot be met by starving leaves). *(Controller's call.)* The enumeration program is
compiled once per fight and goes through the existing compile and trace caches.
Levers if the target is missed (builder's choice, in this order): faster playout policy
per turn (it dominates turn cost), chunked playouts with compaction (spec §6), smaller
leaf floors.

## Milestone order (replaces `docs/design.md` §8 items 2–7)

1. ~~Roark end to end~~ (done, plus the ranking speed work).
2. **Battle search Stage B** (this document).
3. **Save parser** (spec §4.1), wired into the CLI.
4. **Prep search** (spec §4.4), validated against exhaustive search on Roark.
5. **Coverage expansion** gym by gym.
6. **GPU deployment.**
7. **Web UI** (spec §4.5), designed once around the finished engine's output.

The milestone-1 golden trace (needs the user's ROM) stays open and can be recorded at
any point.

## Left to the builder

- Module boundaries beyond the names above, data structures for the tree, the width
  cap and ε defaults (ε default suggestion: 10^-4), the leaf budget floor and
  allocation formula, JSON schema details, how merging is hashed.
- Whether the exhaustive test solver lives in `tests/` or `reference/`.
- Exact wording of rendered observation labels (ASCII only).
