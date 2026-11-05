# Task 10 report: plan tree labels, --deaths, --protect, docs

BASE = a9790eae1b383e2941dacbb7aacb40a3c740c121

## Implemented
- `plan_tree.label(..., prev_mine=None, budget=None)`: `<species> fainted` when the player's active shows 0 HP, plus
  `(death d of k allowed)` / `(death d, over the k allowed)` with `budget = (k, n)`; `you sent in <species>` when the
  active changed without a switch action. `n` = fainted slots at the root that are not deaths of this battle
  (`fainted slots - root.player_faints`), so a root that is already a replacement decision numbers its deaths correctly.
- `render`: header `P(win, 0 faints)` at k = 0, else `P(win, <=K deaths)`, `over budget` column; replacement decision
  outcome `send in <species>`; failed node outcome `(over the death budget: play on for the win, not planned)`;
  `deaths under the leader: ...` line (`distribution_line`, public, reused by `rank`); `protected: <names> (a line where one faints fails)`.
- `to_json`: nodes `replacement`; ranking rows `distribution` (`as_dict()` or None), `e_deaths`, `failed`; top level `deaths_allowed`, `protected`.
- CLI: `--deaths K` (0..3, else ValueError exit 1) and `--protect NAME` (append) on rank and solve; `protected_slots`;
  `all_zero_footer(k)`; rank prints `P(win, <=K deaths)`, `P(win)`, `E[deaths]`, ladder `<=0 <=1 <=2 <=3 (screen)` (`-` without a ladder),
  the leader's distribution line and the protected line; solve run record has `deaths` and `protected`.
- README usage; cli module docstring.

## Rulings applied
Ruling 10 (Task 10): `--protect` digits are 0-based party indices (like `--lead`); tests adapted (`["1"] == (1,)`, `["6"]`/`["9"]` raise, bad-flag test uses 6 and 9).
`test_cli_user_save_rank` has no `Missing (` skip.

## TDD
RED: `uv run pytest tests/search/test_plan_tree.py -q -x` -> `TypeError: label() got an unexpected keyword argument 'budget'`;
`uv run pytest tests/test_cli.py -q -k "protected_slots or all_zero or deaths or bad_budget or default_budget"` -> `18 failed, 1 passed`.
GREEN: full suite `uv run pytest -q` -> 877 passed, 7 skipped in 2153 s; ruff check and format clean.
Manual: `uv run nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth 1 --deaths 1 --devices 1` prints the `P(win, <=1 deaths)` header, the distribution line and a plan.

## Files
src/nuzline/search/plan_tree.py, src/nuzline/cli.py, README.md, tests/search/test_plan_tree.py, tests/test_cli.py.

## Concerns
- The names table is upper case (`MONFERNO`); output keeps that (consistent with the rest of the CLI), so the brief's `Monferno fainted` / `protected: Prinplup` strings are asserted case-insensitively.
- The brief's `budget = (k, fainted slots at the root)`: implemented as fainted slots at the root minus the battle's own deaths at the root (same for the brief's tests, correct for a root that already has a death).
- `kids()` keeps its 3-tuple return and takes `prev_mine` as a keyword, for existing callers.
- The party limit is 6, so the ambiguous-species test replaces Budew with a second Starly.
- A first focused pytest run stalled (CPU idle) and was killed; rerun was clean and fast (cause not identified).

## Fix round 1

Finding: README.md:20 claimed the `<=0 .. <=3` ladder for both rank and solve. Solve prints only the requested rung column, the over-budget column and the distribution line. Reworded so the ladder is claimed for `rank` only ("`rank` also shows every rung `<=0 .. <=3` from the screen"). Doc-only change; no code or tests touched, so no tests were needed.
Command: `git diff --stat` (README.md only). Ruff not applicable to a Markdown change.
