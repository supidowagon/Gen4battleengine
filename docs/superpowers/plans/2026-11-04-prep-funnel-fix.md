# Prep Funnel Fix (rung-0 gate miss) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to carry out this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the prep funnel return the safest setup the user's toggles allow. That means ranking by win chance first, then expected deaths, with cost used only as the last tie-break and always shown. Comparisons must be made at equal playout counts, and saturated ties must be broken by more playouts, not by whole-party HP. Success means the exhaustive Roark gate passes on rungs 0 and 1 for seeds PRNGKey(0..4), checked against the stored exhaustive values, while the funnel stays inside the spec's time targets.

**Architecture.** The base is the "minimal" design, which scored highest with all three judges (correctness 8/8/8, cost 9/9/8, simplicity 9/8/8, and the only design replayed end to end on the real save: seeds 0 and 1 both pass). It has three parts:
1. A merged, equal-n stage inside each variant expansion, made up of ordinary halving units.
2. Retention of exact ties at survivor-floor cuts, bounded by `tie_playouts`.
3. Win chance first in `prep_order`.

On top of that come the ideas all three judges agreed to graft:
- (a) `prep_order` follows the battle tie rule (`stage_a.tie_ordered`, ruling 5), applied at every position: among the remaining rows whose interval overlaps the top-scoring remaining row's, the next is the one with the fewest `tie_deaths`, then the higher score, the lower cost, the tie rank, the index. The `DEATHS_TOL` band is removed: cost never enters before deaths, and a clear (non-overlapping) win-chance gap always decides first. (User decision 2026-11-04.)
- (b) Whole-party `hp_left`/`pp_left` and the raw index leave every ranking key. They are replaced by a seeded blake2b tie rank of the turn-0 row, which is rebuilt on restore and never checkpointed.
- (c) The finalist picks are stored in the finalists position (`pos["picks"]`).
- (d) Each merged and tie-retention round is its own anytime unit.
- (e) A display-only "Cheaper, tied within noise" line on the card and in the record.
- (f) An explicit test that a reserved resource is never spent.
- (g) A `--recheck` mode in `benchmarks/exhaustive_roark.py` that judges fresh funnel picks against the stored exhaustive values.

The merged stage tops up only rung k's set, not the ladder rungs (which are screen-only answers). This keeps the time cost below minimal's.

**Rejected or deferred, with reasons:**
- Per-rung `DEATHS_TOL` (0.02 at k=0). Any deaths difference inside a win-chance tie now decides (user decision 2026-11-04), so a tolerance band has no job left; it only let cost decide among near-equal deaths.
- The statistical design's empirical-Bayes posterior, KL bounds and finalist race. That is the most new code, it renumbers the phase ids (which reseeds every run), and it was never replayed on the real funnel. The race stays the next lever if the gate must hit the exact argmax.
- `tie_survive`. It is a further knob to rescue a noise loss (#890 at n=56).

**Tech stack:** Python 3.12, JAX (CPU devices), pytest, ruff (line length 100), uv.

**Worktree:** `G:/Farry/Claude Workspace/Pokemon Platinum Game Engine/.claude/worktrees/m4-prep-search`, branch `claude/m4-prep-search`. All paths below are relative to it.

---

## Global Constraints

- **User decision (binding, verbatim):** "This should depend, on whether the user has toggled the option to allow items. If they have show the safest line."
  - Ruling: a resource the user allows (toggle on, not reserved) never outranks win chance.
  - Ranking = win chance (rung k), then expected deaths (`deaths.tie_deaths`, so a lost playout counts `LOSS_DEATHS` = 7), then resource cost as the last tie-break. Cost is always listed on the card.
  - **User decision 2026-11-04 (binding):** when two setups' win chances are statistically tied (95 % intervals overlap, `stage_a.tied_with`), fewer expected deaths decides; cost comes only after deaths. Asked "When two setups' win chances are statistically tied, should fewer expected deaths decide?", the user answered "Yes thats fine". A clear win-chance gap (no overlap) always decides first.
  - Reserved resources are never spent.
  - The user's priority is the safest line and the fewest deaths, with safety over speed.
- The ruling supersedes plan ruling 7 (cost-first leader, `DEATHS_TOL`) and spec section 4's "Ties" bullet. Record the supersession dated **2026-11-04** (Task 1).
- Do not change the value function: `src/nuzline/search/playout.py`, `stage_b.py`, `leaves.py` and the engine stay as they are. The stored exhaustive values are the ground truth for the re-check, and they stay valid only if the value function is untouched. `_resources_left` (whole-party HP/PP) is still computed and reported. It is simply no longer ranked.
- Time targets (spec section 6):
  - Roark from the user's save, cold: 15 min (900 s). The baseline of 964.2 s already misses it by 64 s (Task 13 ruling).
  - Any vanilla gym: 60 min.
  - The exhaustive gate's funnel (20 CPU devices) is 165 s today and must stay well under 900 s per run.
  - The extra cost of this plan is bounded by construction. Tie retention fires only at a survivor-floor cut (at most `2*keep` setups) and only while `T_r < tie_playouts` (1016). The merged stage tops up only rung k's at most `keep + keep` setups to the parents' count.
- Keep anytime, checkpoint and resume exact. Every new round is a unit: `before_unit()`, `unit_done()`, and a checkpoint after it. Keys stay derived from the seed, phase and round only. Bump `CHECKPOINT_VERSION` to 3 (the position shape changes). Old checkpoints are refused with the existing one-line message.
- Never create, edit, move or delete anything inside a folder named `backup` (any case, any depth). Read-only access is fine.
- The user's save is never committed. Benchmarks find it through `NUZLINE_USER_SAVE`.
- Other constraints:
  - ruff `line-length = 100`.
  - ASCII-only CLI output.
  - No new runtime dependencies (`hashlib` is stdlib).
  - Default-suite test files stay under about 60 s on one device.
  - Match the surrounding docstring density.
- The commit attribution trailer on every commit is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Stored exhaustive values: `C:/Users/Farry/AppData/Local/Temp/claude/G--Farry-Claude-Workspace-Pokemon-Platinum-Game-Engine/85796cb6-f7cf-40cf-b587-639d7f9398fa/scratchpad/exhaustive_values.ckpt.json` (below: `$VALUES`).
  - Shape: `funnel.rungs["0"|"1"].{phase1, phase2, pick}`, each value `[estimate, ci_low, ci_high, e_deaths]`, keyed by the str index into `restricted_setups + pruned_setups` (3555 setups, 0 pruned).
  - Rung 0: phase 2 has 2335 setups, best #891 `[0.99900, 0.99624, 1.0, 0.0010]`, old pick #3357 `[0.98903, 0.98231, 0.99575, 0.0199]`.
  - Rung 1: phase 2 has 2930 setups, best #23 `[1.0, 0.99283, 1.0, 0.0180]`, old pick #2961 `[1.0, 0.99803, 1.0, 0.0050]`.
  - This file is a version-2 checkpoint and is read as raw JSON, never through `read_checkpoint`.

---

## Task 0: Preflight (no commit)

**Files:** none.

- [ ] **Step 1: Confirm the workspace.** Run `git -C "<worktree>" status --short` and `git -C "<worktree>" log -1 --oneline`.
  Expected: the tree is clean and HEAD is `e5a70e3 bench(prep): exhaustive Roark result (rung 0 fails, rung 1 passes)` or a descendant.
- [ ] **Step 2: Confirm that no run depends on version-2 checkpoints.** Run PowerShell `Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId, CommandLine`.
  Expected: no `benchmarks.exhaustive_roark` and no `nuzline prep` process. If one is running, **stop and ask the user**: Task 6 makes its checkpoint unresumable.
- [ ] **Step 3: Get a baseline.** Run `uv run pytest tests/prep -q -x`.
  Expected: all pass. Record the count and the wall time in the ledger later.
- [ ] **Step 4: Check the stored values.** Run `uv run python -c "import json;d=json.load(open(r'$VALUES'));r=d['funnel']['rungs'];print(d['funnel']['setups'],len(r['0']['phase2']),len(r['1']['phase2']),r['0']['pick'],r['1']['pick'])"`.
  Expected: `3555 2335 2930 3357 2961`.

---

## Task 1: Spec, plan rulings and ledger (ruling 7 superseded)

**Files:**
- Modify: `docs/superpowers/specs/2026-10-31-prep-search-design.md` (section 4 "Ties" bullet, lines 77-80; section 6 step 3, line ~144; section 8 first bullet, line ~181)
- Modify: `docs/superpowers/plans/2026-10-31-m4c-prep-search.md` (rulings 2, 3, 7, 12; lines 34-39 and the ruling 12 paragraph)
- Modify: `.superpowers/sdd/2026-10-31-m4c-prep-search/progress.md` (append under "## 2026-11-03 rung 0 gate FAILED")

- [ ] **Step 1: Replace spec section 4's "Ties" bullet (lines 77-80) with:**

```markdown
- Ranking (user decision 2026-11-03, recorded 2026-11-04; supersedes the cost-first tie rule of
  plan ruling 7). The user's words: "This should depend, on whether the user has toggled the
  option to allow items. If they have show the safest line." Among the setups the user's toggles
  allow, the higher P(win, deaths <= k) leads. When win chances are statistically tied
  (overlapping intervals, design.md section 4.1; user decision 2026-11-04), fewer expected
  deaths lead (a lost playout counts `LOSS_DEATHS`), then the higher win chance. Only then
  does the lower resource cost lead (user-weighted). A seeded tie rank decides last. A
  resource the user allows (its toggle on, not reserved) never outranks win chance or deaths;
  a reserved resource is never spent (it is removed from the bag before the search). This is
  the battle tie rule (plan ruling 5) applied at every position. Cost is always listed on the card; when a
  cheaper finalist is tied within noise with the chosen setup, the card names it and what it
  saves. Expected HP and PP left are reported, not ranked: measured over the whole party they
  favour bigger teams. Within rung `k`, a line that loses fewer Pokemon is preferred at equal
  P(win, deaths <= k), so "deaths as a tool" never means dying for nothing.
```

- [ ] **Step 2: Append to spec section 6 step 3 (after "...from the milestone-2 amendment."):**

```markdown
   Halving compares setups at equal playout counts. In a variant expansion the new neighbours
   halve alone. The surviving neighbours and the parents are then topped up together to the
   first halving count at or above the parents' count, and cut to the survivor floor there.
   At a cut that reaches a survivor floor, setups tied exactly (score and deaths) with the last
   survivor are kept and screened further, doubling each round, until the tie breaks or
   `tie_playouts` (1016) is reached. Only then do cost and the seeded tie rank decide. Cost, HP
   and registration order never decide which setups get more playouts while the playouts can
   still tell them apart. (Added 2026-11-04 after the exhaustive rung-0 miss.)
```

- [ ] **Step 3: Append to spec section 8's exhaustive bullet:**

```markdown
  A funnel change after the exhaustive run is re-checked against the stored values
  (`benchmarks/exhaustive_roark.py --recheck`): the new pick is looked up in phase 2, else
  phase 1, else solved at the precision target, and judged by the same overlap rule over
  seeds 0-4.
```

- [ ] **Step 4: Update the plan rulings.**
  - **Ruling 7:** rewrite it to `**Tie rule** (prep_order; superseded 2026-11-04 by the user decision of 2026-11-03, see spec section 4). Rows follow the battle tie rule (ruling 5) at every position: among the remaining rows overlapping the top-scoring remaining row, the fewest tie_deaths, then higher score, lower weighted cost, tie rank, index (user decision 2026-11-04). DEATHS_TOL is removed. HP and PP are reported, not ranked. Default weights unchanged (candy 0, heart_scale 3, tm 10, hm 5, held 1), now the last tie-break only.` Keep the old text struck through below it as `~~...~~` so the history stays readable.
  - **Ruling 2:** change the halving order to `(-score, tie_deaths, cost, tie rank)`. Add the merged equal-n stage (rung k only, keys salted by `MERGED = 1 << 20`, starting round `ladder_round(screen_n, max parent n)`), tie retention at floor cuts while `T_r < tie_playouts`, and `tie_playouts = 1016`.
  - **Ruling 3:** the finalists are the top `finalists` of the final `S_k` by the new `prep_order`, and they are saved in the position.
  - **Ruling 12:** `CHECKPOINT_VERSION = 3`. A variants position may carry `"stage": "merged"` (and then no `"parents"`). A finalists position carries `"picks"`.
- [ ] **Step 5: Add a ledger entry** under the failed-gate section, headed `### 2026-11-04 fix plan (docs/superpowers/plans/2026-11-04-prep-funnel-fix.md)`. It names:
  - the causes fixed: (b) by the merged equal-n stage plus tie retention plus tie rank; (d) by the new `prep_order` (fewer deaths inside a win-chance tie, cost after deaths);
  - the causes accepted as noise: (a) #890 at n=56; (c) #3357's lucky 248/248 (it is now re-screened to 1016 before any floor decision);
  - the settings added (`tie_playouts` 1016);
  - `CHECKPOINT_VERSION` 3;
  - the deferred levers (a finalist race, more finalists);
  - the user decision of 2026-11-04 (answer "Yes thats fine" to "When two setups' win chances are statistically tied, should fewer expected deaths decide?"): inside an overlap fewer deaths decides, cost only after deaths; `DEATHS_TOL` is removed because any deaths difference now decides.
- [ ] **Step 6: Commit.**
  ```
  git add docs/superpowers/specs/2026-10-31-prep-search-design.md docs/superpowers/plans/2026-10-31-m4c-prep-search.md .superpowers/sdd/2026-10-31-m4c-prep-search/progress.md
  git commit -m "docs(prep): rank allowed setups by win chance first; ruling 7 superseded (user decision)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

---

## Task 2: `prep_order` is win chance, deaths, cost, tie rank

**Files:**
- Modify: `src/nuzline/prep/screen.py` (module docstring lines 1-11; `__all__` line 24; delete `DEATHS_TOL` line 26; `Scored` lines 108-126; `scored` 129-135; `prep_order` 138-160)
- Test: `tests/prep/test_screen.py`

- [ ] **Step 1: Write the failing tests** in `tests/prep/test_screen.py`.

```python
def test_prep_order():
    a = Scored(0.90, 0.85, 0.95, 0.30, 0, 0.5, 0.5)
    b = Scored(0.88, 0.83, 0.93, 0.10, 3, 0.5, 0.5)
    c = Scored(0.87, 0.82, 0.92, 0.12, 0, 0.4, 0.5)
    d = Scored(0.60, 0.55, 0.65, 0.0, 0, 1.0, 1.0)
    # a, b, c overlap: fewest deaths first at every position (b, then c, then a); d is clear
    assert prep_order([a, b, c, d]) == [1, 2, 0, 3]
    c3 = Scored(0.87, 0.82, 0.92, 0.12, 3, 0.4, 0.5)
    assert prep_order([a, b, c3, d]) == [1, 2, 0, 3]  # cost never before deaths
    assert prep_order([]) == []


def test_prep_order_fewer_deaths_decides_inside_a_tie():
    """User decision 2026-11-04 on the rung-1 exhaustive values: #23 100 % (0.018 deaths)
    and #2961 99.9 % (0.005 deaths) overlap, so the one losing fewer Pokemon leads."""
    s23 = Scored(1.0, 0.99283, 1.0, 0.0180, 0.0, 0.5, 0.5)
    s2961 = Scored(0.998, 0.99803, 1.0, 0.0050, 3.0, 0.5, 0.5)
    assert prep_order([s23, s2961]) == [1, 0]
    clear = Scored(0.999, 0.995, 1.0, 0.5, 0.0, 0.5, 0.5)
    safe_but_worse = Scored(0.95, 0.93, 0.97, 0.0, 0.0, 0.5, 0.5)
    assert prep_order([safe_but_worse, clear]) == [1, 0]  # a clear win-chance gap decides first


def test_prep_order_cost_never_outranks_win_chance():
    """The rung-0 gate miss (2026-11-03): Stage B rows of #3357 (cost 0), #2130 and #1721
    (cost 3, one Heart Scale). The old rule led with the cheap #3357."""
    s3357 = Scored(0.98704, 0.97896, 0.99511, 0.0287, 0.0, 0.9665, 0.9936, 0.0010)
    s2130 = Scored(0.99213, 0.98491, 0.99935, 0.0079, 3.0, 0.9657, 0.9922, 0.0)
    s1721 = Scored(0.99202, 0.98619, 0.99785, 0.0080, 3.0, 0.9635, 0.9929, 0.0)
    rows = [s3357, s2130, s1721]
    assert prep_order(rows) == [1, 2, 0]
    for costs in itertools.permutations((0.0, 3.0, 10.0)):
        for hp in ((0.1, 0.5, 0.9), (0.9, 0.1, 0.5)):
            varied = [dataclasses.replace(r, cost=c, hp_left=h, pp_left=h)
                      for r, c, h in zip(rows, costs, hp, strict=True)]
            assert prep_order(varied) == [1, 2, 0]


def test_prep_order_tie_breaks():
    base = Scored(0.99, 0.98, 1.0, 0.01, 3.0, 0.5, 0.5, 0.0, tie=5)
    cheaper = dataclasses.replace(base, cost=0.0)
    assert prep_order([base, cheaper]) == [1, 0]           # cost only at equal score and deaths
    fewer = dataclasses.replace(base, e_deaths=0.0)
    assert prep_order([cheaper, fewer]) == [1, 0]          # deaths before cost
    low_tie = dataclasses.replace(base, tie=1)
    assert prep_order([base, low_tie]) == [1, 0]           # then the tie rank
    more_hp = dataclasses.replace(base, hp_left=0.99, pp_left=0.99)
    assert prep_order([base, more_hp]) == [0, 1]           # HP/PP never rank; index last
    higher = dataclasses.replace(base, score=0.991, e_deaths=0.5)
    assert prep_order([base, higher]) == [0, 1]            # inside an overlap, deaths decide
    same_deaths = dataclasses.replace(base, score=0.991, cost=10.0)
    assert prep_order([base, same_deaths]) == [1, 0]       # equal deaths: score before cost


def test_scored_carries_the_tie_rank():
    stats = ScreenStats((10, 6, 0, 0, 0, 0, 4, 0), 7, 5.0, 10.0, False)
    assert scored(stats, 1, 2.0).tie == 0
    assert scored(stats, 1, 2.0, tie=42).tie == 42
```

Add `import dataclasses` and `import itertools` at the top. Keep `test_prep_order_never_trades_a_whiteout_for_fewer_deaths` **unchanged**: its rows tie exactly on score, so `tie_deaths` (a loss counts 7) still decides.

- [ ] **Step 2: Run** `uv run pytest tests/prep/test_screen.py -q`.
  Expected: FAIL. `test_prep_order` gives `[2, 0, 1, 3] != [1, 2, 0, 3]`, `test_prep_order_cost_never_outranks_win_chance` gives `[0, ...]`, and `Scored` does not accept `tie`.
- [ ] **Step 3: Implement.**
  - In `Scored`, add the field `tie: int = 0` after `loss`. Docstring: "``tie``: the setup's seeded tie rank (``search.tie_rank``), the last real key".
  - `scored(stats: ScreenStats, k: int, cost: float, tie: int = 0) -> Scored` passes `tie`.
  - Replace `prep_order`:

```python
def prep_order(rows: Sequence[Scored]) -> list[int]:
    """Row indices best first (spec section 4, user decisions 2026-11-03 and 2026-11-04): the
    battle tie rule (``stage_a.tie_ordered``, ruling 5) at every position. Among the remaining
    rows whose interval overlaps the top-scoring remaining row's (``stage_a.tied_with``), the
    next has the lower deaths key (``Scored.tie_deaths``: a lost playout counts
    ``deaths.LOSS_DEATHS``), then the higher score, the lower cost, the lower tie rank, the
    lower index. An allowed resource never outranks win chance or deaths. HP and PP left are
    not ranked."""
    left = sorted(range(len(rows)), key=lambda i: (-rows[i].score, rows[i].tie_deaths, i))
    order = []
    while left:
        top = rows[left[0]]
        nxt = min(
            (i for i in left if tied_with(rows[i], top)),
            key=lambda i: (rows[i].tie_deaths, -rows[i].score, rows[i].cost, rows[i].tie, i),
        )
        order.append(nxt)
        left.remove(nxt)
    return order
```

  - Delete `DEATHS_TOL` and remove it from `__all__`.
  - Rewrite the module docstring's tie-rule sentence to match.
  - The `tied_with` import stays (here and in Task 8). The loop is quadratic; `prep_order` only sees `S_k` (at most a few hundred rows) and the finalists.
- [ ] **Step 4: Run** `uv run pytest tests/prep/test_screen.py tests/prep/test_search.py -q`.
  Expected: PASS. If `test_search.py` fails, the cause can only be a test that pinned the old leader; there is none today (`test_budget_stops_after_first_round` uses distinct `falling` values).
- [ ] **Step 5: Commit.**
  `git commit -am "fix(prep): prep_order ranks win chance, deaths, then cost; DEATHS_TOL removed (user decision)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 3: Seeded tie rank replaces HP, PP and index in the halving key

**Files:**
- Modify: `src/nuzline/prep/search.py`
  - imports: add `import hashlib`
  - `halving_key` lines 103-107
  - `_Funnel.__init__` ~line 283: add `self.seed`, `self.tie`
  - `register` lines 316-327
  - `row` line 332-333
  - `top` line 342
  - `restore` lines 404-409
  - `finalist_row` lines 593-602
  - `_seed_of` line 709
  - module docstring lines 13-16
  - `__all__`: add `"tie_rank"`
- Test: `tests/prep/test_search.py` (replace `test_halving_does_not_cut_by_registration_order`, lines 423-447; update the `Expansion` helper), `tests/prep/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests.** In `tests/prep/test_search.py`, replace `test_halving_does_not_cut_by_registration_order` with:

```python
def test_halving_key_ignores_hp_pp_and_index():
    """Whole-party HP left favours big teams (bench members never lose HP); in the rung-0 gate
    it dropped the two best setups (#933, #92 at 248/248). Only score, deaths, cost and the
    seeded tie rank order a cut."""
    hi, lo = _saturated(6, 6, 0.9), _saturated(6, 6, 0.3)
    assert search.halving_key(hi, 0, 0.0, 5) == search.halving_key(lo, 0, 0.0, 5)
    assert search.halving_key(_saturated(761, 762, 0.9), 0, 0.0, 0) > search.halving_key(
        lo, 0, 0.0, 1
    )  # the point estimate first, whatever the counts
    assert search.halving_key(lo, 0, 0.0, 9) < search.halving_key(lo, 0, 3.0, 0)  # cost, then tie
    assert search.halving_key(lo, 0, 0.0, 1) < search.halving_key(lo, 0, 0.0, 2)


def test_tie_rank_is_seeded_and_order_free():
    assert search.tie_rank(b"seed", b"row") == search.tie_rank(b"seed", b"row")
    assert search.tie_rank(b"seed", b"row") != search.tie_rank(b"seed2", b"row")
    assert 0 <= search.tie_rank(b"", b"") < 2**64
```

  In `Expansion.__init__`, add the arguments `cost=None, tie=None, tie_playouts=1016` and set:
  - `funnel.cost = list(cost) if cost is not None else [0.0] * total`
  - `funnel.tie = list(tie) if tie is not None else list(range(total))`
  - pass `tie_playouts` into `PrepSettings` (that field lands in Task 4: add the kwarg there; for now leave `tie_playouts` out of the helper).

  In `tests/prep/test_checkpoint.py` add:

```python
def test_restore_rebuilds_the_same_tie_ranks(tables, names):
    prob = make_problem(tables, names)
    teams = team_setups(prob, 2)

    def funnel(key=KEY):
        return search._Funnel(tables, names, prob, SETTINGS, key, None, time.perf_counter)

    f = funnel()
    f.register(teams)
    g = funnel()
    g.restore(f.snapshot()["funnel"])
    assert g.tie == f.tie and len(set(f.tie)) == len(f.tie)
    h = funnel()
    h.register(list(reversed(teams)))
    assert sorted(h.tie) == sorted(f.tie)  # registration order does not change any rank
    other = funnel(jax.random.PRNGKey(8))
    other.register(teams)
    assert other.tie != f.tie
```

  (Import `time`, `team_setups`, `search`, `make_problem`, `SETTINGS` and `KEY` the way the file already imports its helpers.)
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_search.py::test_halving_key_ignores_hp_pp_and_index tests/prep/test_search.py::test_tie_rank_is_seeded_and_order_free tests/prep/test_checkpoint.py::test_restore_rebuilds_the_same_tie_ranks -q`.
  Expected: FAIL (`AttributeError: tie_rank`, and the keys differ by HP).
- [ ] **Step 3: Implement.**

```python
def tie_rank(seed: bytes, row: bytes) -> int:
    """A setup's last tie-break: 8 bytes of blake2b over the run seed and its turn-0 state
    bytes. Unbiased by enumeration order, cost or HP, and the same after a resume (rebuilt from
    the setups, never stored)."""
    return int.from_bytes(hashlib.blake2b(seed + row, digest_size=8).digest(), "little")


def halving_key(stats: ScreenStats, j: int, cost: float, tie: int) -> tuple:
    """A halving round's sort key on rung ``j``, best first (module docstring): the point
    estimate, ``deaths.tie_deaths``, the cost, the seeded ``tie`` rank. HP and PP left are not
    keys: measured over the whole party they favour bigger teams."""
    row = scored(stats, j, cost)
    return (-row.score, row.tie_deaths, cost, tie)
```

  - `_seed_bytes(key) -> bytes` returns `np.asarray(data).tobytes()`, with the same `try/except` as `_seed_of`. `_seed_of` becomes `int.from_bytes(_seed_bytes(key), "little")`.
  - In `__init__`: `self.seed = _seed_bytes(key)` and `self.tie: list[int] = []`.
  - In `register()`, after `self.cost.append(costs[i])`: `self.tie.append(tie_rank(self.seed, blob))`.
  - In `restore()`, inside the loop: `self.tie.append(tie_rank(self.seed, keys[i].tobytes()))`.
  - `row(i, j)` returns `scored(self.stats[i], j, self.cost[i], self.tie[i])`.
  - `top` uses `halving_key(self.stats[i], j, self.cost[i], self.tie[i])`.
  - `finalist_row` passes `tie=self.tie[i]` to `Scored`.
  - Rewrite the module docstring sentence on "The order (halving_key)" to: point estimate, `tie_deaths`, cost, seeded tie rank; HP/PP reported, not ranked.
- [ ] **Step 4: Update the two N1 tests to the HP-free key** (`tests/prep/test_search.py` lines 488-517). HP no longer rescues a neighbour through saturated rounds, so give the truly 1.000 neighbour a playout signal:
  - `test_expansion_keeps_a_better_neighbour_against_tight_parents`: `losses(i, m)` becomes `0 if i == best else (0 if m == 0 else 1 + m // 100)`, so every other neighbour loses one of its first 8. Assertion unchanged: `best in out`. Docstring: "...the 1.000 neighbour leads from round 0 and is compared with the parents at their 4088 playouts (Task 5)".
  - `test_expansion_keeps_a_parent_against_lucky_neighbours`: unchanged.
  - Both tests run under Task 5's merged stage. In this task the first one may still pass or fail depending on the old one-step cut. Run it, and if it fails, mark it `@pytest.mark.xfail(strict=True, reason="needs the merged equal-n stage, Task 5")` and remove the mark in Task 5.
- [ ] **Step 5: Run** `uv run pytest tests/prep -q`.
  Expected: PASS (plus at most the one strict xfail noted above). `test_team_phase_halving` keeps its `(9, 4), (5, 8), (3, 16)` shapes because `falling` has no exact ties. `test_reproducible` still passes because its result is compared with itself.
- [ ] **Step 6: Commit.**
  `git commit -am "fix(prep): seeded tie rank replaces whole-party HP/PP and index in halving cuts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 4: `tie_playouts` and exact-tie retention at survivor-floor cuts

**Files:**
- Modify: `src/nuzline/prep/search.py` (`PrepSettings` lines 110-153; `_Funnel.top` lines 340-343; the `halving` round body lines 482-500)
- Test: `tests/prep/test_search.py`

- [ ] **Step 1: Write the failing tests.**

```python
def _bare(stats, tie=None):
    funnel = search._Funnel(None, None, SimpleNamespace(deaths=0), PrepSettings(), KEY, None, float)
    funnel.stats = dict(enumerate(stats))
    funnel.cost = [0.0] * len(stats)
    funnel.tie = list(tie) if tie is not None else list(range(len(stats)))
    return funnel


def test_top_keeps_the_tied_block():
    stats = [_saturated(10, 10, 0.5)] * 6 + [_saturated(9, 10, 0.5)] * 4 + [_saturated(8, 10, 0.5)] * 2
    f = _bare(stats)
    assert f.top(range(12), 0, 8, ties=True) == list(range(10))  # ranks 7-10 tie at the cut
    assert f.top(range(12), 0, 8) == list(range(8))
    assert f.top(range(12), 0, 6, ties=True) == list(range(6))   # no tie across the boundary
    assert f.top(range(12), 0, 12, ties=True) == list(range(12))


def test_floor_cut_screens_an_exact_tie_further(monkeypatch):
    """12 setups, keep 8: the best (last by tie rank) never loses; the other 11 lose exactly one
    playout after 300. A cut at n/n used to go by HP, then index, and dropped it."""
    best = 11

    def losses(i, m):
        return 0 if i == best else int(m > 300)

    hp = {i: (0.1 if i == best else 0.9) for i in range(12)}
    e = Expansion(monkeypatch, [], 12, losses, hp, keep=8)
    out = e.funnel.halving("teams", "teams", phase_id("teams"), [list(e.new)] * search.RUNGS, 8)[0]
    assert best in out and len(out) == 8
    assert max(e.funnel.stats[i].n for i in out) == 1016  # bounded by tie_playouts
    assert e.funnel.playouts == 12 * 1016
    off = Expansion(monkeypatch, [], 12, losses, hp, keep=8, tie_playouts=0)
    out = off.funnel.halving("teams", "teams", phase_id("teams"), [list(off.new)] * search.RUNGS, 8)[0]
    assert best not in out and len(out) == 8  # retention off: the old saturated cut
```

  Also add `{"tie_playouts": -1}` to the bad-settings list in `test_phase_keys_are_disjoint`.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_search.py -q -k "tied_block or exact_tie or disjoint"`.
  Expected: FAIL (`top() got an unexpected keyword 'ties'`; `PrepSettings` has no `tie_playouts`).
- [ ] **Step 3: Implement.**
  - `PrepSettings` gets the field `tie_playouts: int = 1016` (after `weights`). Docstring: "``tie_playouts``: at a cut reaching a survivor floor, setups tied exactly with the last survivor are kept and screened further until the tie breaks or they reach this many playouts (0: off; 1016 = ``screen_n`` 8's round 6)".
  - `__post_init__`: `if self.tie_playouts < 0: raise ValueError(f"tie_playouts must not be negative, not {self.tie_playouts}")`.
  - The fingerprint picks the new field up through `_plain(settings)`, so `checkpoint.py` needs no code change.

```python
    def top(self, indices: Sequence[int], j: int, count: int, ties: bool = False) -> list[int]:
        """The ``count`` best on rung ``j`` by ``halving_key``, ascending. With ``ties``, every
        setup tied with the ``count``-th on (score, deaths) is kept as well."""
        def key(i):
            return halving_key(self.stats[i], j, self.cost[i], self.tie[i])

        best = sorted(indices, key=key)
        end = count
        if ties and 0 < count < len(best):
            edge = key(best[count - 1])[:2]
            while end < len(best) and key(best[end])[:2] == edge:
                end += 1
        return sorted(best[:end])
```

  In the `halving` round, replace line 491 with:

```python
            t_r = st.screen_n * (2 ** (r + 1) - 1)
            counts = [max(floors[j], ceil(len(p) / 2)) for j, p in enumerate(pools)]
            pools = [
                self.top(p, j, counts[j], ties=(
                    j == k and parents is None and counts[j] == floors[j]
                    and t_r < st.tie_playouts
                ))
                for j, p in enumerate(pools)
            ]
```

  Reuse `t_r` for the `top_up` target on line 486. Retention runs only on rung k, only in the team phase (and, in Task 5, in the merged stage), and never in neighbour-only rounds. The existing loop guard `while not (r and len(pools[k]) <= keep)` re-doubles a kept block. It terminates because the first round with `t_r >= tie_playouts` cuts to the floor exactly. Before a floor cut `len <= 2*keep`, so a kept block holds at most `2*keep` setups.
  - In the `Expansion` helper, pass `tie_playouts` through to `PrepSettings`.
- [ ] **Step 4: Run** `uv run pytest tests/prep -q`.
  Expected: PASS. `test_team_phase_halving` is unchanged (no exact ties). The resume tests still pass because every retention round is an ordinary unit.
- [ ] **Step 5: Commit.**
  `git commit -am "fix(prep): keep exact ties at survivor-floor cuts and screen them further (tie_playouts 1016)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 5: Equal-n merged stage at the end of a variant expansion

**Files:**
- Modify: `src/nuzline/prep/search.py`
  - add a module constant `MERGED = 1 << 20` and `ladder_round()`
  - `halving()` signature and body, lines 459-503
  - module docstring lines 17-24
- Test: `tests/prep/test_search.py`

- [ ] **Step 1: Write the failing regression test** (the rung-0 expansion-1 cut, built without the save):

```python
def test_expansion_compares_parents_and_neighbours_at_equal_n(monkeypatch):
    """The rung-0 gate miss, expansion 1's final cut (trace cut 9): 32 parents at 248, two of
    them truly ~0.999 with little HP left (#933, #92), one lucky (#3357, truly ~0.989, the most
    HP); 105 cost-3 neighbours stop halving at 120/120 with more HP. The one-step cut at
    unequal n kept the neighbours and the lucky parent and dropped both safe parents."""
    safe, lucky = (0, 1), 2
    parents = (
        [_saturated(248, 248, 0.944), _saturated(248, 248, 0.886), _saturated(248, 248, 0.9665)]
        + [_saturated(246, 248, 0.5)] * 29
    )
    n_new = 105
    total = 32 + n_new
    hp = {i: (0.963 + 0.007 * (i % 10) / 10 if i >= 32 else 0.5) for i in range(total)}
    cost = [0.0] * 32 + [3.0] * n_new

    def losses(i, m):
        if i in safe:
            return 0
        if i == lucky:
            return max(0, m - 248) // 90
        if i < 32:
            return max(0, m - 248) // 120
        return max(0, m - 120) // 100

    e = Expansion(monkeypatch, parents, n_new, losses, hp, keep=8, cost=cost)
    out = e.run()
    assert set(safe) <= set(out) and len(out) == 8
    assert {e.funnel.stats[i].n for i in out} == {1016}  # one count for the whole cut
    first = min(out, key=lambda i: search.halving_key(e.funnel.stats[i], 0, cost[i], e.funnel.tie[i]))
    assert first in safe


def test_ladder_round():
    assert [search.ladder_round(8, n) for n in (1, 24, 25, 248, 762, 1016, 4088)] == [1, 1, 2, 4, 6, 6, 8]
```

  Expected trace with the fix:
  - Neighbours halve 105 → 53 → 27 → 14 → 8 (T = 8 to 120, all saturated, by tie rank).
  - The merged stage starts at `ladder_round(8, 248) = 4`. It tops the 8 neighbours up to 248 (247/248 each), cuts 40 → 8 with the tied neighbour block kept (11 left), goes on to 504 (A and B 504/504, L 502/504, neighbours 501/504, still 11), then to 1016.
  - At 1016, L and the neighbours tie at 1008/1016. Cost keeps L and tie rank keeps 5 neighbours.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_search.py -q -k "equal_n or ladder_round or expansion"`.
  Expected: FAIL. `safe` is not a subset of `out` (the old one-step cut at unequal n); `ladder_round` does not exist.
- [ ] **Step 3: Implement.**

```python
MERGED = 1 << 20  # key salt of a variant expansion's merged stage (disjoint from round keys)


def ladder_round(screen_n: int, n: int) -> int:
    """The first halving round ``r >= 1`` whose ``T_r = screen_n * (2**(r + 1) - 1)`` reaches
    ``n`` playouts."""
    r = 1
    while screen_n * (2 ** (r + 1) - 1) < n:
        r += 1
    return r
```

  `halving(..., parents=None, merged: bool = False)`. Inside the round:
  - `base = jax.random.fold_in(self.key, pid)`, then `if merged: base = jax.random.fold_in(base, MERGED)`, then `rkey = jax.random.fold_in(base, r)`.
  - `members = sorted(pools[k]) if merged else sorted(set().union(*pools))`. The merged stage tops up rung k only; ladder rungs are screen-only answers, cut once at their counts.
  - `counts = [floors[j] if merged else max(floors[j], ceil(len(p) / 2)) for j, p in enumerate(pools)]`.
  - The retention condition from Task 4 stays (`parents is None` is true in the merged stage).
  - `pos`: add `if merged: pos["stage"] = "merged"`. Never set `"parents"` together with `merged` (`assert not (merged and parents is not None)` at the top).

  Replace the tail (line 501-503):

```python
        if parents is None:
            return pools
        union = whole(pools)
        if len(union[k]) <= keep:
            return [self.top(p, j, floors[j]) for j, p in enumerate(union)]
        start = ladder_round(st.screen_n, max(self.stats[i].n for i in union[k]))
        return self.halving(phase, bucket, pid, union, keep, v, start, merged=True)
```

  Each merged round is a normal unit: `before_unit()`, `unit_done()` and a checkpoint, so `--budget` stops between rounds (ruling 11).
  - Rewrite the module docstring lines 17-24: neighbours halve alone; then rung k's parents and surviving neighbours are topped up to the first halving count at or above the parents' largest count (round keys salted by `MERGED`) and cut to the floor there, with exact ties kept until `tie_playouts`; ladder rungs are cut to their floors in the first merged round at their counts.
  - Remove the Task 3 xfail mark, if one was added.
- [ ] **Step 4: Run** `uv run pytest tests/prep/test_search.py -q`.
  Expected: PASS, including both N1 tests.
  - The tight-parents test: the best neighbour reaches 4088/4088 at merged round 8, against the parents' 4076/4088.
  - The lucky-neighbours test: merged round 6 at 1016 gives the parents 7 failures against the neighbours' 11, so `set(out) == set(range(8))`.
- [ ] **Step 5: Run** `uv run pytest tests/prep -q`.
  Expected: PASS. If `test_resume_equals_uninterrupted` fails, it is Task 6's resume branch: continue to Task 6 before committing, or commit with that test marked `xfail(strict=True)` and remove the mark in Task 6.
- [ ] **Step 6: Commit.**
  `git commit -am "fix(prep): variant expansions cut parents and neighbours at equal playout counts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 6: Resume inside the merged stage, saved finalist picks, `CHECKPOINT_VERSION` 3

**Files:**
- Modify: `src/nuzline/prep/search.py` (`expansion` resume branch lines 527-539; `finalists_phase` lines 604-633; module docstring checkpoint paragraph lines 32-40)
- Modify: `src/nuzline/prep/checkpoint.py:37`
- Test: `tests/prep/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests.**

```python
def test_resume_inside_the_merged_stage(tables, names, monkeypatch, tmp_path):
    prob, _ = setup_run(monkeypatch, tables, names)
    probe = tmp_path / "probe.ckpt.json"
    seen = []
    a = run_prep(tables, names, prob, SETTINGS, KEY, checkpoint=probe,
                 progress=lambda p: seen.append(json.loads(probe.read_text())["funnel"]["pos"]),
                 **VERSIONS)
    merged = [n for n, pos in enumerate(seen, 1) if pos.get("stage") == "merged"]
    assert merged, "the run never reached a merged round"
    assert "parents" not in seen[merged[0] - 1]
    path = tmp_path / "merged.ckpt.json"
    progress, _ = interrupting(merged[0])
    b = run_prep(tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS)
    assert b.stats.stopped == "interrupted"
    assert run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS) == a


def test_resume_reuses_the_saved_finalist_picks(tables, names, monkeypatch, tmp_path):
    prob, fakes = setup_run(monkeypatch, tables, names)
    units = []
    run_prep(tables, names, prob, SETTINGS, KEY, progress=units.append, **VERSIONS)
    first_final = next(n for n, u in enumerate(units, 1) if u.phase == "finalists")
    path = tmp_path / "picks.ckpt.json"
    progress, _ = interrupting(first_final)
    run_prep(tables, names, prob, SETTINGS, KEY, checkpoint=path, progress=progress, **VERSIONS)
    doc = json.loads(path.read_text())
    pos = doc["funnel"]["pos"]
    assert pos["phase"] == "finalists" and len(pos["picks"]) == SETTINGS.finalists
    other = next(i for i in pos["pool"] if i not in pos["picks"])
    pos["picks"][1] = other  # a resume must solve what the position says, not re-rank the pool
    path.write_text(json.dumps(doc))
    record = []
    monkeypatch.setattr(search, "solve", fake_solve(fakes, record, lambda s, i: 0.5))
    run_prep(tables, names, prob, SETTINGS, KEY, resume=path, **VERSIONS)
    setups = [setup_from_json(s) for s in doc["funnel"]["setups"]]
    assert record[0][0] == setups[other]


def test_version_2_checkpoint_is_refused(tables, names, saved):
    path, print_ = saved  # adapt to the existing `saved` fixture's shape
    doc = json.loads(path.read_text())
    before = json.dumps({**doc, "version": 2})
    path.write_text(before)
    with pytest.raises(ValueError, match="version 2"):
        read_checkpoint(path, print_)
    assert path.read_text() == before  # nothing written
```

  - If `pos["pool"]` holds no index outside the picks (pool size equals `finalists`), use `SETTINGS` with `variant_keep=3`.
  - Use the existing `saved` fixture and `read_checkpoint` import exactly as `test_resume_refuses_a_damaged_or_future_file` does; adjust the unpacking to that fixture.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_checkpoint.py -q`.
  Expected: FAIL. The merged resume calls `resume["parents"]` and raises `KeyError`; `pos` has no `"picks"`; the version is 2.
- [ ] **Step 3: Implement.**
  - In `expansion()`, the resume branch becomes:

```python
        if resume is not None:
            out = self.halving(
                f"variants {v + 1}", "variants", phase_id("variants", v), resume["pools"],
                st.variant_keep, v, resume["r"],
                None if resume.get("stage") == "merged" else resume["parents"],
                merged=resume.get("stage") == "merged",
            )
            self.variant_rounds += 1
            return out
```

    A merged position resumed after its last round exits the loop at once (`r > 0` and `|S_k| <= keep`) and returns `pools`, which are already cut to the floors.
  - `finalists_phase`: `picks = list(resume["picks"]) if resume is not None and "picks" in resume else self.ranked(pool, self.k)[: st.finalists]`. Put `"picks": picks` in both `pos` dicts (lines 619 and 631).
  - `checkpoint.py:37`: `CHECKPOINT_VERSION = 3  # 3: merged variant stage, finalist picks, tie ranks (2026-11-04)`.
  - Update the search.py module docstring's position description: `"stage": "merged"` (no `"parents"`) and finalists `"picks"`.
- [ ] **Step 4: Run** `uv run pytest tests/prep/test_checkpoint.py tests/prep/test_search.py -q`.
  Expected: PASS. `test_resume_refuses_a_damaged_or_future_file` uses `CHECKPOINT_VERSION + 1 = 4` automatically, and `test_fingerprint_ignores_the_budget_only` still holds with `tie_playouts` included.
- [ ] **Step 5: Commit.**
  `git commit -am "fix(prep): resume inside the merged stage and reuse saved finalist picks; checkpoint v3" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 7: A reserved resource is never spent

**Files:**
- Test: `tests/prep/test_search.py` (a source change is needed only if the test fails)

- [ ] **Step 1: Write the test.**

```python
@pytest.mark.parametrize("scales", [0, 1])
def test_a_reserved_heart_scale_is_never_spent(tables, names, monkeypatch, scales):
    """Reserving the Heart Scale leaves 0 in the problem's bag (problem.resources_without);
    no setup the funnel registers may then use the Move Relearner."""
    party = sample_party(names)
    bite = names.move("Bite")
    cands = [candidate(names, party[i], i, relearner=(bite,)) for i in range(3)]
    prob = problem(tables, names, cands, resources=Resources(scales, (), (), ()),
                   levelled=[()] * 3)
    fakes = Fakes(monkeypatch, falling)
    settings = PrepSettings(screen_n=4, team_keep=2, variant_keep=2, variant_rounds=1,
                            max_depth=0, max_team=2)
    run_prep(tables, names, prob, settings, KEY)
    spent = [setup_cost(prob, s).heart_scales for s in fakes.order]
    assert fakes.order
    if scales == 0:
        assert not any(spent)
    else:
        assert any(spent)  # the neighbour space does offer the relearner when allowed
```

  (Import `setup_cost` from `nuzline.prep.setups`.)
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_search.py -q -k reserved`.
  Expected: PASS for both. `Cost.fits` already filters (setups.py:208). If `scales=0` fails, stop and report it as a bug. Do not paper over it.
- [ ] **Step 3: Commit.**
  `git commit -am "test(prep): a reserved Heart Scale is never spent" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 8: "Cheaper, tied within noise" (display only)

**Files:**
- Modify: `src/nuzline/prep/screen.py` (new `cheaper_alternative`, added to `__all__`)
- Modify: `src/nuzline/prep/search.py` (`PrepResult` gets `cheaper: Finalist | None = None` as its last field; `_Funnel.result`)
- Modify: `src/nuzline/prep/report.py` (`render` after `cost_line`, line ~315; `record` adds `"cheaper_alternative"`)
- Test: `tests/prep/test_screen.py`, `tests/prep/test_report.py`

- [ ] **Step 1: Write the failing tests.**

```python
def test_cheaper_alternative():
    lead = Scored(0.99213, 0.98491, 0.99935, 0.0079, 3.0, 0.5, 0.5)
    tied_cheap = Scored(0.98704, 0.97896, 0.99511, 0.0287, 0.0, 0.5, 0.5)
    tied_dear = Scored(0.99202, 0.98619, 0.99785, 0.0080, 3.0, 0.5, 0.5)
    far_cheap = Scored(0.95, 0.94, 0.96, 0.05, 0.0, 0.5, 0.5)
    assert cheaper_alternative([lead, tied_dear, tied_cheap]) == 2
    assert cheaper_alternative([lead, tied_dear, far_cheap]) is None
    assert cheaper_alternative([tied_cheap, lead]) is None  # the leader is already the cheapest
    assert cheaper_alternative([]) is None
```

  In `tests/prep/test_report.py`, add `test_cheaper_alternative_line`. Use the same `PrepResult` construction as `test_render_with_plan`, setting `cheaper=` to a second finalist whose setup costs nothing while the chosen one spends a Heart Scale. Assert that `render(...)` contains exactly one line starting with `"Cheaper, tied within noise: "` and that this line contains `"Cost: none"`. Assert that `record(...)["cheaper_alternative"]["label"]` equals that setup's label. With `cheaper=None`, assert there is no such line and that `record(...)["cheaper_alternative"] is None`.
- [ ] **Step 2: Run** `uv run pytest tests/prep/test_screen.py tests/prep/test_report.py -q`.
  Expected: FAIL (an `ImportError` and a missing line).
- [ ] **Step 3: Implement.**

```python
def cheaper_alternative(rows: Sequence[Scored]) -> int | None:
    """``rows`` in ``prep_order`` (row 0 the answer): the cheapest later row whose interval
    overlaps the answer's and that costs less (ties: the earlier row), else ``None``. Shown on
    the card only; it never reorders (spec section 4)."""
    if not rows:
        return None
    cheaper = [i for i in range(1, len(rows))
               if rows[i].cost < rows[0].cost and tied_with(rows[i], rows[0])]
    return min(cheaper, key=lambda i: (rows[i].cost, i)) if cheaper else None
```

  - In `_Funnel.result`, when `self.finalists` is set: `rows = [self.finalist_row(f) for f in self.finalists]` and `c = cheaper_alternative(rows)`. Pass `cheaper=self.finalists[c] if c is not None else None`.
  - In `render`, after `cost_line(...)`: if `result.cheaper`, append `f"Cheaper, tied within noise: {setup_label(names, problem, f.setup)} {top.estimate:.1%} ({top.ci_low:.1%}-{top.ci_high:.1%}), {cost_line(problem, names, f.setup)}"`, with `f = result.cheaper` and `top = f.result.ranking[0]`.
  - In `record`: `"cheaper_alternative": None`, or `{"label", "value", "ci_low", "ci_high", "cost": cost_line(...)}`.
- [ ] **Step 4: Run** `uv run pytest tests/prep -q`.
  Expected: PASS. `PrepResult` equality in the resume tests still holds, because `cheaper` is derived from state that is restored.
- [ ] **Step 5: Commit.**
  `git commit -am "feat(prep): name a cheaper finalist tied within noise on the card (display only)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 9: Wording sweep, full suite, lint

**Files:**
- Modify: `src/nuzline/cli.py:279`. The `--weight` help becomes `"cost weight of candy, heart_scale, tm, hm or held (last tie-break, after win chance and deaths); repeatable"`.
- Modify: `src/nuzline/prep/variants.py:50`. Add the comment `# resource cost weights: the last tie-break only, after win chance and expected deaths (spec section 4)`. `DEFAULT_WEIGHTS` values are unchanged.
- Modify: `src/nuzline/search/playout.py:112` docstring only (no code): "hp_left/pp_left are reported by the prep search, not ranked". If this risks the jax code-version hash used by the compile cache (`jaxcache.code_version`), skip it: a recompile is acceptable, but the kernel source must not change otherwise.

- [ ] **Step 1: Make the three edits.**
- [ ] **Step 2: Run** `uv run ruff check src tests benchmarks` and `uv run ruff format --check src tests benchmarks`.
  Expected: clean.
- [ ] **Step 3: Run** `uv run pytest -q`.
  Expected: the whole default suite passes. Pay particular attention to `tests/prep/test_exhaustive.py::test_funnel_matches_exhaustive_small[0|1]` and `test_funnel_matches_exhaustive_ci`, which judge the funnel against a small exhaustive space and must still pass, and to `tests/prep/test_pp_stall.py` (the PP-stall finding relies on score, not on `pp_left` ordering). Check that no test file exceeds about 60 s (`--durations=15`).
- [ ] **Step 4: Commit.**
  `git commit -am "docs(prep): weights and HP/PP are tie-break/report only" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 10: `--recheck` mode for the exhaustive benchmark

**Files:**
- Modify: `src/nuzline/prep/exhaustive.py` (new `stored_value`, `recheck_verdict` next to `verdict`, line ~201)
- Modify: `benchmarks/exhaustive_roark.py` (argparse lines 45-50; a branch right after `settings = PrepSettings(...)`, line ~152; module docstring)
- Test: `tests/prep/test_exhaustive.py`

- [ ] **Step 1: Write the failing tests** (pure, no jax work):

```python
RUNG0 = {
    "phase1": {"891": [0.9985, 0.996, 1.0, 0.0015], "3357": [0.99, 0.985, 0.995, 0.02],
               "510": [0.996, 0.993, 0.999, 0.004], "77": [1.0, 0.998, 1.0, 0.0]},
    "phase2": {"891": [0.99900, 0.99624, 1.0, 0.0010], "3357": [0.98903, 0.98231, 0.99575, 0.0199],
               "510": [0.99602, 0.98970, 1.0, 0.0040]},
}


def test_stored_value_prefers_phase2_then_phase1():
    assert stored_value(RUNG0, 510) == ("phase2", Value(0.99602, 0.98970, 1.0, 0.0040))
    assert stored_value(RUNG0, 77) == ("phase1", Value(1.0, 0.998, 1.0, 0.0))
    assert stored_value(RUNG0, 12345) is None


def test_recheck_verdict_on_the_stored_rung0_values():
    old = recheck_verdict(RUNG0, 3357, Value(0.98903, 0.98231, 0.99575, 0.0199))
    assert old.best == 891 and not old.passed  # the gate's original failure
    new = recheck_verdict(RUNG0, 510, Value(0.99602, 0.98970, 1.0, 0.0040))
    assert new.best == 891 and new.passed
    coarse = recheck_verdict(RUNG0, 77, Value(1.0, 0.998, 1.0, 0.0))
    assert coarse.best == 891  # a value put in from phase 1 never becomes the best
```

- [ ] **Step 2: Run** `uv run pytest tests/prep/test_exhaustive.py -q -k "stored_value or recheck"`.
  Expected: FAIL (`ImportError`).
- [ ] **Step 3: Implement** in `exhaustive.py`:

```python
def stored_value(rung: Mapping, pick: int) -> tuple[str, Value] | None:
    """The pick's value in a stored exhaustive rung (``{"phase1": .., "phase2": ..}``, values
    ``[estimate, ci_low, ci_high, e_deaths]`` keyed by str index): phase 2, else phase 1."""
    for phase in ("phase2", "phase1"):
        if str(pick) in rung.get(phase, {}):
            return phase, Value(*rung[phase][str(pick)])
    return None


def recheck_verdict(rung: Mapping, pick: int, value: Value, pruned: Collection[int] = ()) -> Verdict:
    """``verdict`` for a new funnel pick against a stored rung: the best is the stored phase-2
    best (the pick's own ``value`` never becomes it); the pick ties when its interval overlaps
    the best's."""
    stored = {int(i): Value(*v) for i, v in rung["phase2"].items()}
    pruned_set = set(pruned)
    best = min((i for i in stored if i not in pruned_set), key=lambda i: (-stored[i].estimate, i))
    ties = tied_with(value, stored[best])
    beaten = tuple(sorted(i for i in pruned_set if i in stored and stored[i].ci_low > value.ci_high))
    return Verdict(best, pick, ties, beaten, ties and not beaten)
```

  In `benchmarks/exhaustive_roark.py`:
  - Add `--recheck VALUES` (`type=Path`, default None: "judge fresh funnel picks against this exhaustive checkpoint's stored values; values nothing else") and `--seeds` (default `"0,1,2,3,4"`).
  - Add `RECHECK = HERE / "exhaustive_roark_recheck.json"`. When `--recheck` is given and `--out` is left at its default, write to `RECHECK` and never overwrite `exhaustive_roark.json`.
  - Right after `settings = PrepSettings(...)`: `if args.recheck is not None: return _recheck(args, tables, names, base, setups, pruned_ids, settings)`. Use the name the file already gives the pruned index set (it is used at line 244).
  - `_recheck`:
    - Load `doc = json.loads(args.recheck.read_text())` (raw JSON: the stored file is version 2 with the old fingerprint, so do not use `read_checkpoint`) and `raise SystemExit` unless `doc["funnel"]["setups"] == len(setups)`.
    - For `k in (0, 1)` and each seed:
      - time `run_prep(tables, names, replace(base, deaths=k), settings, jax.random.PRNGKey(seed))`;
      - set `pick = setups.index(result.chosen)` (`SystemExit` if absent);
      - set `src, value = stored_value(rung, pick)`, or else `("solved", exhaustive_values(tables, names, prob, [setups[pick]], jax.random.fold_in(jax.random.PRNGKey(3), k), depth=1, precision=PHASE2_PRECISION)[0])`;
      - set `v = recheck_verdict(rung, pick, value, pruned_ids)`;
      - print `rung {k} seed {seed}: {'PASS' if v.passed else 'FAIL'} pick #{pick} ({src}) {value} vs best #{v.best} {stored best}, funnel {s:.1f} s, {result.stats.playouts} playouts`;
      - record `{"k", "seed", "pick", "setup": setup_label(...), "source", "value", "best", "best_value", "passed", "funnel_seconds", "playouts", "finalists": [{"index", "estimate", "ci_low", "ci_high", "stored": stored_value(...)}]}`.
    - Write `{"values_file": str(args.recheck), "runs": [...], "passed": all(...), "machine": machine(), "devices": jax.device_count(), "settings": _plain(settings)}` and return 0 (a failed verdict is a finding, as in the full run).
  - Document the mode in the module docstring.
- [ ] **Step 4: Run** `uv run pytest tests/prep/test_exhaustive.py -q` and `uv run python -m benchmarks.exhaustive_roark --help`.
  Expected: PASS, and the help lists `--recheck` and `--seeds`.
- [ ] **Step 5: Commit.**
  `git commit -am "bench(prep): --recheck judges fresh funnel picks against stored exhaustive values" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 11: Re-check rungs 0 and 1 against the stored exhaustive values, seeds PRNGKey(0..4)

**Files:**
- Create (by the run): `benchmarks/exhaustive_roark_recheck.json`
- Modify: `benchmarks/README.md` (the exhaustive section, after line ~210's "the fixes are checked against these stored values" sentence)
- Modify: the ledger

- [ ] **Step 1: Run the recheck** (about 10 funnels × 200-300 s, so roughly 35-50 min on 20 devices; run it in the background):
  ```
  uv run python -m benchmarks.exhaustive_roark --save "$NUZLINE_USER_SAVE" --devices 20 --recheck "C:/Users/Farry/AppData/Local/Temp/claude/G--Farry-Claude-Workspace-Pokemon-Platinum-Game-Engine/85796cb6-f7cf-40cf-b587-639d7f9398fa/scratchpad/exhaustive_values.ckpt.json" --seeds 0,1,2,3,4
  ```
  If the scratchpad file is gone, use the committed original `benchmarks/results/exhaustive_roark_20261102_184141.ckpt.json` (read-only).
- [ ] **Step 2: Check the results against the acceptance criteria.**
  - All 10 runs print `PASS`. The rung-0 best is #891 `[0.99900, 0.99624, 1.0]`, so a pick passes when its stored `ci_high >= 0.99624`. The rung-1 best is #23 `[1.0, 0.99283, 1.0]`.
  - The minimal-fix scratch replay (6 devices, different tie-breaks) gave rung-0 picks #510 `[0.99602, 0.98970, 1.0]` for seed 0 and #231 `[0.9960, 0.99163, 1.0]` for seed 1. The finalist pools held #933 and #92 (0.998). Different picks under this plan's hash tie rank are expected. What has to hold is the pass.
  - #3357 must not be the pick for any seed.
  - Any pick whose `source` is `phase1` is provisional: re-solve it at 0.015 (the `solved` path, by deleting it from a copy of the values file) and record both values.
  - Each funnel finishes in under 300 s on 20 devices (165 s before). If any funnel exceeds 900 s, flag it.
- [ ] **Step 3: If any run FAILS:** do not tune parameters to make it pass. Record the seed, the pick, its value, the finalists and their stored values in the ledger as a finding. Report to the user with the next levers:
  - a capped finalist race (successive elimination against the leader, at most 1024-2048 playouts per arm, over the final `S_k` before Stage B);
  - `finalists` 5;
  - a tighter Stage B precision for the finalists.
  Stop there. Each lever costs time against a Roark budget that is already over.
- [ ] **Step 4: Add one line to the README:** `2026-11-04: after the funnel fix (equal-n variant cuts, exact-tie retention, win chance first), re-checked against these stored values over seeds 0-4: rung 0 X/5, rung 1 Y/5 pass (benchmarks/exhaustive_roark_recheck.json).` Add a ledger entry with the per-seed picks.
- [ ] **Step 5: Commit.**
  `git add benchmarks/exhaustive_roark_recheck.json benchmarks/README.md .superpowers/sdd/2026-10-31-m4c-prep-search/progress.md` then `git commit -m "bench(prep): funnel fix re-checked against stored exhaustive values (seeds 0-4)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`

---

## Task 12: Re-measure `nuzline prep` wall time against 900 s

**Files:**
- Create (by the run): `benchmarks/results/prep_<timestamp>.json`
- Modify: `benchmarks/README.md` (the "Milestone 4: nuzline prep wall time" table, lines 220-230, and the paragraph at lines 237-247)
- Modify: the ledger

- [ ] **Step 1: Run** (one cold and one warm `nuzline prep --save S --trainer roark`, about 35 min; run it in the background):
  `uv run python -m benchmarks.prep_timing --save "$NUZLINE_USER_SAVE"`
- [ ] **Step 2: Compare.**
  - Baseline: cold 964.2 s, warm 878.2 s, 1,144,478 playouts, rung 0 1.000.
  - Expected: cold about 990-1030 s. The real save's team floor cut is at T=2040 >= `tie_playouts`, so retention does not fire there. The cost comes from the merged stage topping up about 8 rung-k neighbours from about 250 to about 2040 in each of 2 expansions, roughly 30k-60k playouts at about 1,310 playouts/s.
  - Rung 0 must stay >= 0.997, and the record must show the chosen setup's cost line.
  - If cold exceeds 1100 s (more than 136 s over the baseline), stop and report before committing.
- [ ] **Step 3: Record the result.** Add a README table row `**after funnel fix 2026-11-04** (defaults, tie_playouts 1016)` with cold, warm, phase seconds, setups screened, playouts and rung 0. Add to the paragraph: the cold run now misses 900 s by N s; this is accepted under the Task 13 ruling and the user's "safety over speed"; the remedy is the queued faster round-0 screening, not a weaker funnel. Vanilla gyms are not measured here. The added cost is bounded per expansion (at most `2*keep` setups × the parents' count), so the 60 min target is not at risk.
- [ ] **Step 4: Commit.**
  `git add benchmarks/README.md benchmarks/results/prep_*.json .superpowers/sdd/2026-10-31-m4c-prep-search/progress.md` then `git commit -m "bench(prep): wall time after the funnel fix" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  (If `benchmarks/results/` is gitignored, commit only the README and the ledger.)

---

## Report to the user at the end

- The recheck table (rung, seed, pick, stored value, pass).
- The cold and warm wall time against 900 s, and the change from 964.2 s.
- The behaviour change the user confirmed on 2026-11-04: inside a win-chance tie fewer deaths decides at every position, cost only after deaths; `DEATHS_TOL` is gone.
- The deferred levers (finalist race, more finalists) if the gate is ever required to hit the exact argmax.
