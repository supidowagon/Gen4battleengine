# Nuzline: standing rules

Keep this file short. Design and roadmap live in `docs/design.md`.

- **Battle rules come from the decomp only** (`external/pokeplatinum`). Never state or build a rule from memory; cite file and line.
- **Game data comes from the ROM, then the calc repo, then the docs sheet** (`docs/kaizo/pipeline.md`). Where one is blank, fill from the next and keep the citation; nothing is "unknown" until all three are silent. Every real battle belongs to a split.
- **Doubles-ready code:** new engine and AI code follows section 5 of `docs/doubles-state-design.md`.
- **Never push** without the user's explicit go for that push. Merges into local `main` go through a fresh review agent.
- **Safety over speed:** a line's win chance outranks saving allowed resources.
- **Never write to any `backup` folder.** The user's save files and ROMs are never copied or committed.
- **Build for the complete engine:** no tests, guards or special cases that only matter while coverage is partial. Model the missing mechanic instead.
- **Lean process:** mechanics in batches by family, one review per batch, focused tests plus the cross-area guards (`tests/prep/test_variants.py`, `tests/data/test_game_constants.py`, `tests/tools/test_backlog.py`, `tests/test_cli.py`) per batch, fast tier (`-m "not slow"`) once per merge into `main`, full suite only when a split closes. No stream starts a third batch while two of its batches are unmerged.
- **Scratch files and caches go on G:**, never on C:.
