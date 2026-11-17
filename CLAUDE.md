# Nuzline: standing rules

Keep this file short. Design and roadmap live in `docs/design.md`.

- **Battle rules come from the decomp only** (`external/pokeplatinum`). Never state or build a rule from memory; cite file and line.
- **Game data comes from the ROM, then the calc repo, then the docs sheet** (`docs/kaizo/pipeline.md`). Where one is blank, fill from the next and keep the citation; nothing is "unknown" until all three are silent. Every real battle belongs to a split.
- **Doubles-ready code:** new engine and AI code follows section 5 of `docs/doubles-state-design.md`.
- **Pushing:** only the managing session pushes, and only `main`, after each merge that passed a fresh review agent (standing yes from the user, 2026-11-15). Work streams never push. Merges into local `main` go through a fresh review agent.
- **Safety over speed:** a line's win chance outranks saving allowed resources.
- **Never write to any `backup` folder.** The user's save files and ROMs are never committed, and never copied anywhere except the project's git-ignored `.cache` folder.
- **Build for the complete engine:** no tests, guards or special cases that only matter while coverage is partial. Model the missing mechanic instead.
- **Lean process:** mechanics in batches by family, one review per batch, focused tests plus the cross-area guards (`tests/prep/test_variants.py`, `tests/data/test_game_constants.py`, `tests/tools/test_backlog.py`, `tests/test_cli.py`) per batch. A merge into `main` needs the review agent plus a targeted run on the merged tip (the four guards, `tests/test_differential.py`, and the tests of every area the change touches; under an hour). The fast tier (`-m "not slow"`) runs once a night on `main`, not per merge; a red nightly is fixed before new batches start. Several ready batches may be stacked into one merge. Full suite only when a split closes. A test in the fast tier stays under 60 s and 4 GB, or it is marked slow. No stream starts a third batch while two of its batches are unmerged.
- **After merging `main` into a branch:** rebuild the vanilla and Kaizo tables before any test, scan for names, enum members and table rows defined twice (both sides adding the same thing merges without a conflict), and run the tests that assert something is unsupported.
- **Usage:** no idle turns (tests run inside the subagent that needs the result, so a stream wakes once with the result); streams are cleared at about 200k tokens, at commit points; Sonnet only for mechanical subagent jobs whose answer is checked on the spot, Opus for merges, review fixes and anything written from the decomp; a stream sends its family list to the managing session before it starts a batch.
- **Scratch files and caches go on G:**, never on C:.
