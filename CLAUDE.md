# Nuzline: standing rules

Design: `docs/design.md`. Pipeline: `docs/process.md`.

- **Battle rules come from the decomp only** (`external/pokeplatinum`). Never state or build a rule from memory; cite file and line. For every rule read both the C (`src/battle`) and the battle scripts (`res/battle/scripts`): checks such as Shield Dust sit in scripts.
- **Game data comes from the ROM, then the calc repo, then the docs sheet** (`docs/kaizo/pipeline.md`). Fill a blank from the next source and keep the citation; nothing is "unknown" until all three are silent. Every real battle belongs to a split.
- **Doubles-ready code:** new engine and AI code follows section 5 of `docs/doubles-state-design.md`.
- **Build for the complete engine:** no tests, guards or special cases that only matter while coverage is partial. Model the missing mechanic instead.
- **Safety over speed:** a line's win chance outranks saving allowed resources.
- **Never write to any `backup` folder.** Never commit the user's ROMs or save files; copy them only into the git-ignored `.cache/`.
- **Pushing:** only the managing session pushes, only `main`, after each merge that passed a fresh review agent. Work streams never push. Every merge into local `main` goes through a fresh review agent.
- **Batches:** a batch is 15-20 mechanic groups of one family (fewer only when the family is smaller). Send the family list to the managing session before starting a batch. One review per batch. Per batch run the focused tests plus the four guards: `tests/prep/test_variants.py`, `tests/data/test_game_constants.py`, `tests/tools/test_backlog.py`, `tests/test_cli.py`. No stream starts a third batch while two of its batches are unmerged.
- **Merges:** a merge into `main` needs the review agent plus a targeted run on the merged tip: the four guards, `tests/test_differential.py`, and the tests of every area the change touches; under an hour. Several ready batches may go into one merge.
- **After merging `main` into a branch:** rebuild the vanilla and Kaizo tables before any test, scan for names, enum members and table rows defined twice, and run the tests that assert something is unsupported.
- **Test tiers:** the fast tier (`-m "not slow"`) runs once a night on `main`; a red nightly is fixed before new batches start. The full suite runs only when a split closes. A fast-tier test stays under 60 s and 4 GB, or it is marked `slow`.
- **Usage:** no idle turns: run tests inside the subagent that needs the result. Clear a stream at about 100k tokens, at a commit point. Sonnet for mechanical subagent jobs checked on the spot (test runs, table rebuilds, ledger and backlog edits); Opus for merges, review fixes and anything written from the decomp.
- Project files live inside the project folder: scratch, worktrees, scripts and logs in the git-ignored `_scratch/`; nothing on C: or beside the project.
