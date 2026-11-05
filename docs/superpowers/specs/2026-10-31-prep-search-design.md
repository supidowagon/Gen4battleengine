# Milestone 4: prep search — design

Status: draft for approval, 2026-10-31. Parent: `docs/design.md` (§4.4 prep search, §8 milestones).
Builds on Stage B (`docs/superpowers/specs/2026-10-30-stage-b-solver-design.md`) and the save
parser (`docs/superpowers/specs/2026-10-31-save-parser-design.md`).

## 1. Intent

The user's words: the tool should "do everything" and find "the absolute safest line, losing as
little Pokémon as possible, preferably zero", using every strategy a strong nuzlocker would. The
long-term goal is a tool that beats every nuzlocke — vanilla and especially difficulty hacks —
deathless. Milestone 4 is the step from "advise me turn by turn with this team" (milestones 1–2)
to "tell me what to bring and how to play it" for one fight.

Before a fight the user loads their save and names the trainer. The tool returns:

1. **A setup:** which Pokémon (up to 6, from party and boxes), who leads, what each holds, which
   moves each knows, and the preparation steps that gets them there (Rare Candies to the cap, a
   Heart Scale at the Move Relearner, a TM, entering at a chosen HP), with what each step costs.
2. **The odds ladder:** the best line with 0 deaths allowed, then with up to 1, 2 and 3, each with
   its chance and interval (e.g. "deathless 85% ±1.5, at most one death 99.2% ±0.6").
3. **A branching turn-by-turn plan** for the chosen rung: the Stage B plan tree, keyed on what the
   player sees on screen, including who to send in after a planned sacrifice.

The user trusts the tool because it is correct; the design keeps two kinds of correctness apart:
search correctness (proved here, against exhaustive search on Roark) and engine fidelity (proved by
golden traces from the real game — the user has agreed to record the first one, §8).

## 2. Goals, non-goals, anti-goals

**Goals**
- Per-fight prep search over team, lead, held items, moves (current, Move Relearner, TMs),
  level-to-cap and entry HP/status, from a real save.
- Objective: highest chance of winning with at most `k` deaths, `k` chosen by the user
  (default 0); the 0–3 ladder is always reported.
- Deaths as a tool: with `k > 0` the plan may deliberately sacrifice a Pokémon, because in Set
  mode its faint is a free switch into the right counter (user's rule of thumb).
- Toggles that reserve scarce resources (TMs, Heart Scales, shards, stones) and specific Pokémon
  (protected: may come, must not die; §10).
- Anytime: a current best answer is always available; long runs checkpoint and resume.
- Run on the user's real save before Roark (6 in party, 5 boxed, level cap 16).

**Non-goals for milestone 4** (each is a later milestone or a later toggle)
- Reading data from a ROM, and difficulty hacks (needs a ROM data reader and possibly the Fairy
  type: its own milestone, after coverage expansion).
- Planning across fights (keeping a Pokémon safe for Gardenia, saving a TM for later). Each fight
  is planned alone; reservations are how the user protects the future.
- Bag items in battle, Shift mode, other rule sets (later toggles).
- Evolution toggle and move tutors (none affects the Roark proving ground; design.md §4.4 keeps
  them for later).
- Gyms other than Roark (milestone 5).

**Anti-goals**
- Mis-simulating. A candidate that touches a mechanic the engine does not implement is excluded
  and named; it is never silently approximated (as `coverage` does today).
- A line that risks a whiteout. Losing the fight ends the run, so it is worse than any number of
  deaths: every rung maximises P(win and deaths ≤ k), and a loss is never traded for fewer deaths.
- Hiding cost. Every resource a setup spends is listed.

## 3. Rules the tool plays by (user-decided)

Set mode; level caps; held items only (no bag items in battle); any faint is a death. These are the
defaults; other rule sets become settings later. The level cap is an input (`--cap 16`); a
`config/level_caps.toml` of default caps per trainer is my call, overridable on the command line.

The user's 929 Rare Candies make levelling to the cap free in practice: level-to-cap is on by
default and its cost is reported as candies, not as risk.

## 4. Objective (user-decided)

- Primary: maximise P(win with deaths ≤ k), default `k = 0`. The user picked "99% deathless but
  sometimes two deaths" over "97% deathless, never more than one".
- Every result shows the distribution P(0), P(1), P(2), P(≥3), P(whiteout). The ladder is how the
  user decides between "safest deathless" and "allow deaths" (the user's "maybe have the user
  decide"). My call: other objective functions ("fewest expected deaths") are deferred; they need
  new interval maths, and the ladder plus the distribution already expose the trade-off.
- Ranking (user decision 2026-11-03, recorded 2026-11-04; supersedes the cost-first tie rule of
  plan ruling 7). The user's words: "This should depend, on whether the user has toggled the option
  to allow items. If they have show the safest line." Among the setups the user's toggles allow,
  the higher P(win, deaths <= k) leads. When win chances are statistically tied (overlapping
  intervals, design.md section 4.1; user decision 2026-11-04), fewer expected deaths lead (a lost
  playout counts `LOSS_DEATHS`), then the higher win chance. Only then does the lower resource cost
  lead (user-weighted). A seeded tie rank decides last. A resource the user allows (its toggle on,
  not reserved) never outranks win chance or deaths; a reserved resource is never spent (it is
  removed from the bag before the search). This is the battle tie rule (plan 4b ruling 5,
  `stage_a.tie_ordered`) applied at every position. Cost is always listed on the card; when a
  cheaper finalist is tied within noise with the chosen setup, the card names it and what it saves.
  Expected HP and PP left are reported, not ranked: measured over the whole party they favour
  bigger teams. Within rung `k`, a line that loses fewer Pokemon is preferred at equal P(win,
  deaths <= k), so "deaths as a tool" never means dying for nothing.

**Ladder semantics (my call).** Each rung has its own best setup and plan: allowing a death may
change who you bring. One playout yields the death count, so the Stage A screen and halving score
every candidate on all rungs at once; Stage B (the expensive part) runs on the finalists of the
rung the user asked for (`--deaths K`, default 0). The other rungs' numbers come from the screen,
are labelled as such, and the user can re-run for any rung.

**What `k` does in a simulated battle (my call).** A battle always continues until the player wins
or whites out; deaths are counted, not terminal. `k` only scores the result: a playout counts as a
success for rung `k` when the player wins with deaths ≤ k. After a line goes over budget, the
plan keeps playing for P(win) with as few further deaths as possible, and says so. For `k = 0` the
engine may still stop a playout at the first faint for speed when the distribution is not needed;
the regression requirement is that P(win, 0 deaths) is statistically unchanged, not byte-identical.

## 5. Engine work this milestone needs

Today the engine ends the battle at the first player faint (`engine/step.py` `_faint_check`), only
the opponent gets replacements, and Stage A/B values are hard-wired to P(win, 0 faints). Two pieces
of engine work come first.

### 5.1 Faint budget and Set-mode player replacements
- A faint budget `k` (a traced value, not a recompile): the battle is lost when deaths exceed `k`
  or the player has no Pokémon left; below that, a fainted player Pokémon is replaced.
- The player's replacement is a decision: a decision node in Stage B (expand, observe, plan tree)
  and a heuristic in playouts. Set mode: the player gets no switch when the opponent's Pokémon
  faints. The simultaneous-faint ordering (design.md §10, open) is settled from the decomp.
- The playout policy stops valuing "no player faint this turn" absolutely when `k > 0`: a death
  costs budget, a whiteout costs everything.
- Stage A/B report the death distribution, not only P(win, 0 faints); `k = 0` results are
  unchanged (regression-tested against today's numbers).

### 5.2 Coverage for the user's Roark pool
`nuzline coverage --save <user save> --trainer roark` refuses the fight today. Missing, with the
mons that need them:
- Moves: Foresight (Furret, Venonat), Night Shade (Spinarak), Disable (Psyduck, Venonat), Bide
  (Kricketot), Mud Sport (Geodude). Foresight also needs the AI-side support Roark's scripts read.
- Abilities: Insomnia (Spinarak), Cloud Nine (Psyduck), Compoundeyes (Venonat), Shed Skin
  (Kricketot).
- Plus every move the candidates could gain this milestone (level-up moves up to the cap via the
  Move Relearner, and TM10 Hidden Power from the bag) that is not yet implemented.

Each is implemented exactly from the decomp, with the same tests as milestone 1 (decomp-derived
unit tests, differential against the reference engine). Anything still missing excludes the
candidate or the move with a named warning (§2 anti-goal); the run never fails because one boxed
Pokémon is unsupported.

### 5.3 Data
- TM/HM compatibility per species (not extracted today; `tools/extract_decomp.py` reads only
  level-up learnsets). Move Relearner = level-up moves at or below the current level.
- Level-to-cap recomputes stats (`stats=None`, full HP, table PP) rather than keeping a save's
  stored stats.

## 6. The search

design.md §4.4 already fixes the funnel; this milestone implements it:

1. **Candidate pool:** every party and box Pokémon (eggs excluded) at the cap, each with its move
   options (current set, relearner and TM variants within the user's reservations), its held-item
   options (implemented holdable items in the bag, or none), and its entry states.
2. **Entry HP and status (pre-damage, prestatus):** HP brackets derived from the damage tables and
   behaviour thresholds (design.md §4.4); each bracket uses its most favourable HP. My call: any
   HP is treated as reachable (the user can arrange it with wild battles) and is a toggle; a
   prestatus is only offered from methods the user declares (off by default).
3. **Conservative prune**, then a **Stage A screen** of every surviving (team, lead, variant)
   in one batch, then **successive halving**, then **Stage B** on the finalists with the faint
   budget, at the precision target from the milestone-2 amendment.
   Halving compares setups at equal playout counts. In a variant expansion the new neighbours
   halve alone. The surviving neighbours and the parents are then topped up together to the
   first halving count at or above the parents' count, and cut to the survivor floor there.
   At a cut that reaches a survivor floor, setups tied exactly (score and deaths) with the last
   survivor are kept and screened further, doubling each round, until the tie breaks or
   `tie_playouts` (1016) is reached. Only then do cost and the seeded tie rank decide. Cost, HP
   and registration order never decide which setups get more playouts while the playouts can
   still tell them apart. (Added 2026-11-04 after the exhaustive rung-0 miss.)
   The finalists are chosen by a **fresh race**: before Stage B, every final survivor of the
   requested rung is screened again with `race_playouts` (2040) new playouts, and the finalists
   are the best of that fresh screen alone (win chance, then expected deaths, then cost). The
   accumulated screen stats are biased: a setup survived every cut partly by luck, so its
   accumulated estimate overstates it (the winner's curse). In the rung-0 seed-4 re-check, 6 of
   the 8 final survivors were within the stored best's interval, yet the pick by accumulated
   stats was #1719, which screened 0.998 over 1016 playouts against a true 0.989. Simulated over
   those 8, a fresh race picks wrong 1.1 % of the time at 1016 playouts each, 0.1 % at 2040. The
   race's playouts are counted but never merged into the halving stats, and its picks are saved
   in the checkpoint, so a resumed run never races again. `race_playouts = 0` turns it off.
   (Added 2026-11-04, user approved.)
4. **Anytime and checkpointing:** a time budget (`--budget`), a current best answer always
   printable, state saved to disk so a run can stop and resume.

Performance facts the builder starts from (benchmarks/README.md): ~2,800 playouts/s in Stage A on
the dev machine, a depth-2 Stage B solve ~3.5 min cold; tables and states are runtime data, so new
teams do not recompile, but `make_state` costs ~1 s per state in Python and needs a batched or
cached builder for screening hundreds of candidates.

**Targets (my call, from the user's "90% of fights under an hour by a human; vanilla should be
trivial"):** Roark from the user's save, end to end, within 15 minutes cold on the dev machine;
any vanilla gym's prep search should finish within 60 minutes. Overnight runs are acceptable for
hard fights.

### 6.1 Long-horizon strategies (honest limit)
The user asked for pre-damage baiting, pivoting, PP stalling and every other nuzlocke strategy.
None is coded as a named strategy: they emerge because the trainer AI is simulated exactly and the
search keeps what works. Baiting by entry HP/status (§6 step 2), pivoting and sacrifices (§5.1)
fit the current search. Long strategies such as a 30-turn PP stall do not: Stage B looks two
decisions deep and the playout player looks one turn ahead. This milestone adds test fights in
which a PP stall is the only deathless line; how far the search must get is in §10 item 2 (a
smarter playout player, deeper search on the line in question, or both, is left to the builder).

## 7. Output

- A setup card: team, lead, held items, moves, levels, and a numbered prep list with costs.
- The odds ladder (k = 0..3) with intervals and the death distribution.
- The plan tree of the chosen rung (`plan_tree.render` / `to_json`), with replacement decisions
  after planned deaths labelled ("Monferno faints here by design; send in Staravia").
- A run record (save hash, trainer, settings, seeds, result), as every run writes today.
- CLI: `nuzline prep --save S --trainer roark [--cap 16] [--deaths K] [--reserve ITEM...]
  [--budget SECONDS] [--resume FILE]`.

## 8. Validation

- **Exhaustive check on Roark (my call on its size):** the pool is the 6 party Pokémon plus 2
  boxed ones, current moves plus one relearner variant each, no items, full HP, every team of 1–6
  from that pool with every lead; k = 0 and k = 1; values by Stage B depth 1 at the precision
  target. It runs as a one-off overnight job whose result is committed under `benchmarks/`; a CI
  version shrinks the pool to 4 Pokémon. Pass: the funnel's pick has an interval overlapping the
  exhaustive best's (a tie counts), on both rungs, and no setup the prune dropped beats it.
  A funnel change after the exhaustive run is re-checked against the stored values
  (`benchmarks/exhaustive_roark.py --recheck`): the new pick is looked up in phase 2, else
  phase 1, else solved at the precision target, and judged by the same overlap rule over
  seeds 0-4.
- **The user's save** is the fixture for the real-fight runs. It is not committed: tests and
  benchmarks find it through `NUZLINE_USER_SAVE` and skip without it, as the PKHeX oracle does.
  Its bag: TM10, one Heart Scale, no holdable battle items.
- **Golden trace from the user's game:** the user plays Roark in an emulator from their save,
  following the tool's line, recording with `tools/trace_battle.lua` (docs/tracing.md). The trace
  becomes `tests/golden/roark_01.json` and the engine must reproduce it turn by turn. This is the
  first golden trace; `tests/test_golden.py` is skipped until it exists.
- `k = 0` regression: Stage A/B numbers unchanged by the faint-budget work.
- PP-stall test fight (§6.1).

## 9. Order of work (suggested)

1. Coverage for the user's pool (§5.2) and TM data (§5.3) — unblocks running the real fight.
2. Faint budget and Set-mode replacements (§5.1).
3. Prep search pipeline and CLI (§6, §7).
4. Exhaustive Roark check, PP-stall test, timing benchmark.
5. Golden trace with the user (§8).

Steps 1 and 2 are independent and can run in parallel.

## 10. Decided after review (2026-10-31)

1. **Protected Pokémon (user-decided).** Reserving a Pokémon means it may be brought but must not
   die: any line in which it faints counts as a failure on every rung, whatever `k` is. This needs
   deaths tracked per Pokémon (who fainted, not only how many), in the engine state, the playout
   result and the Stage B leaves. A plain "leave it out of the team" exclusion is also offered,
   since it is free once candidates are filtered.
2. **PP stall (my recommendation, user asked for it).** Two tiers:
   - *Required:* a short stall, where the opponent's only damaging move has a few PP left (about
     5) and outlasting it is the only deathless line. The search must find it ranked first. This
     forces the playout player and the search to value PP, which every longer stall builds on.
   - *Best effort, reported:* a long stall (20+ turns). The milestone records whether it is found
     and how far the search falls short; it does not block the milestone. It becomes a required
     item before difficulty hacks, where stalling matters most.

## 11. Left to the builder

Data structures for candidates and checkpoints; the screening budget per stage; the playout
heuristic for player replacements; how HP brackets are split; file layout under `src/nuzline/prep/`.
