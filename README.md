# nuzline

A tool that finds high-EV lines for single-player nuzlocke battles in Pokemon Platinum (US) and Platinum Kaizo (`--game kaizo`). Given the player's save file and a chosen trainer fight, it recommends pre-battle setup (box picks, lead, held items, prestatus, predamage) and ranked in-battle actions each turn, maximizing the probability of winning with zero faints. See `docs/design.md`.

## Status: frozen (23 November 2026)

This repository is the JAX implementation. It is frozen: no new mechanics are built here. Work continues as a Rust port in a separate repository.

- Why: the product is a line finder that searches one battle deeply and cuts a line the moment it fails. JAX is built for running many battles in lockstep on a GPU; one loaded engine step needs about 6.3 GB, so tests and search could not run on an ordinary PC.
- State at the freeze: singles mechanics are complete; 240 of the 387 Platinum Kaizo battles are playable. The missing ones are almost all doubles and multi battles. Unmerged doubles work is on branches of the private development clone, not here.
- What this repository remains good for: the rules as read from the decompilation with file and line citations, the tests that state what the game does, the trainer AI model, and the ROM data pipeline.
- The process rules in `CLAUDE.md` and `docs/process.md` (rented test machines, batches, merge gates) describe how this repository was run until the freeze.

## Taking over this project

- Standing rules: `CLAUDE.md`.
- How the project is run, and how to take it over: `docs/process.md`, section 12 "Taking over".
- Standing decisions and approvals: `docs/decisions.md`.
- Live state (tips, runs in flight, next steps): `_scratch/handoff/` in the main checkout (git-ignored).

## Commands

```
uv sync
uv run pytest
uv run ruff check .
```

Rank the opening actions, or solve the fight start with the Stage B plan search (exact chance nodes, observations, plan tree):

```
uv run nuzline rank --party tests/fixtures/sample_party.toml --trainer roark
uv run nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth 2 --tree plan.json
```

Both take `--deaths K` (0 to 3, default 0: maximise the chance of winning with at most K player faints, with the death distribution shown; `rank` also shows every rung `<=0 .. <=3` from the screen) and `--protect NAME` (repeatable; a species or a 0-based party index like `--lead`: a line where that Pokemon faints counts as a failure):

```
uv run nuzline rank --party tests/fixtures/sample_party.toml --trainer roark --deaths 1 --protect Prinplup
uv run nuzline solve --party tests/fixtures/sample_party.toml --trainer roark --max-depth 1 --deaths 1
```

Find the setup to bring (team, lead, moves, held items, entry HP) from your save's party and boxes, with its odds:

```
uv run nuzline prep --save path/to/platinum.sav --trainer roark --deaths 1 --protect Prinplup
uv run nuzline prep --save path/to/platinum.sav --trainer roark --budget 600 --checkpoint roark.ckpt.json
uv run nuzline prep --save path/to/platinum.sav --trainer roark --resume roark.ckpt.json
```

`--save` is required. `--cap N` is the level cap (default: `config/level_caps.toml`, which lists only Roark; any other trainer needs `--cap`). `--protect` and `--exclude` (repeatable) name a Pokemon by species, nickname, `party:N` or `box:B:S` (slots 0-based, boxes 1-based, as in `save show`); a species both in the party and a box is ambiguous and must be named that way, a bare number is refused. `--reserve ITEM ...` keeps bag items out of the search, `--prestatus poison|burn|paralysis` (repeatable) allows a status inflicted beforehand, `--no-entry-hp` always enters at full HP, `--weight KIND=W` (candy, heart_scale, tm, hm, held) sets the resource cost weights of the tie-break. Search size: `--max-depth` (0 = screen only), `--finalists`, `--screen-n`, `--max-team N` (the largest team brought, 1-6), `--seed`, `--devices`; the search screens every (team, lead) of the pool at once, so a pool whose team space is over 25,000 setups (about 13 Pokemon at teams of 6) is refused with a message: leave Pokemon out with `--exclude`, lower `--max-team`, or raise the limit with `--max-setups N` (hours and many GB of memory); `--tree FILE.json` writes the plan tree and `--show-depth` sets the plan levels printed.

The output has the `Setup:` card, the `Prep steps:` (Rare Candies, Heart Scale relearns, TMs, items, deposits and withdrawals, in order), the `Cost:` line (every resource spent), the `Odds ladder` for 0-3 deaths (the requested rung is planned by Stage B, the others are labelled "screen"), the Stage B `Plan:` and a `Search:` summary. Progress lines (`[prep] ...`) go to stderr. A run record is written to `runs/prep_<time>.json` (with the save's `save_sha256`) and a checkpoint to `runs/prep_<time>.ckpt.json` after every unit of work. `--budget SECONDS` stops starting new work after that time and prints the best answer so far ("screen only, no plan" when no solve finished), as does Ctrl-C; the exit code is 0. `--resume FILE` continues from a checkpoint and gives the same answer as an uninterrupted run; it needs the same save, trainer, settings, seed and program version (otherwise one line says what differs, exit 1) and cannot be combined with `--checkpoint`. If a run crashes, continue it with `--resume runs/prep_<time>.ckpt.json` (the path is the `[prep] checkpoint:` line printed to stderr at the start); do not start over. Exit codes: 0 ok, 1 error, 2 the opponent is unsupported or every setup is refused.

Score a line (one answer per opposing Pokemon, written in a line file), or search for one (spec: `docs/superpowers/specs/2026-11-12-l1-line-player-design.md` §3, §6):

```
uv run nuzline line odds --game kaizo --save path/to/kaizo.sav --trainer-id 246 --line roark.toml --deaths 3
uv run nuzline line find --game kaizo --save path/to/kaizo.sav --trainer-id 246 --line team.toml --deaths 1 --out found.toml
```

The line file has the team (`[[team]]`, with `--save`) and the rules (`[[rule]]`: `against`, `use`, `move`, optional `first`, `via`, `from`, `wait`); `--party P.toml` gives the team instead. `odds` prints the line and, from `--n` playouts (default 2000), the ladder P(win with at most k deaths) for k = 0..3 and P(win) with 95 % intervals, the death distribution, mean turns and the default player's share. `find` starts from the file's rules (if any), prints `[line]` progress to stderr, then the found line, the same block from `--final-n` hold-out playouts and its `[[rule]]` TOML; `--out` writes the team and the found line (it must not be the `--line` file). Search size: `--budget SECONDS`, `--restarts`, `--sweeps`, `--pick-n`; Ctrl-C stops the search and prints the best line kept so far. A heuristic: a good line, not a proven best. Playouts that meet a mechanic the engine does not model are left out and named. Exit codes: 0 ok, 1 a bad file, name or flag, 2 the fight is refused or every playout is.

Read a Platinum save (see `docs/saves.md`), or use it in place of `--party` with `--save`:

```
uv run nuzline save show path/to/platinum.sav
uv run nuzline rank --save path/to/platinum.sav --trainer roark
```

Every command takes `--game vanilla|kaizo` (default vanilla; Kaizo needs `python -m tools.build_tables --game kaizo` first, and the game is never guessed from a save): `uv run nuzline trainers --game kaizo --save path/to/kaizo.sav` lists the Kaizo trainers, optional and gauntlet fights marked, with only the rival variants of the save's starter.

Build the Platinum Kaizo data set from your own ROM, patch, PKCalc zip and Kaizo docs spreadsheet (paths in `data/local.toml`; inputs, commands, corrections and the open user calls: `docs/kaizo/pipeline.md`):

```
uv run --group rom python -m tools.extract_rom --game kaizo --out data/intermediate_kaizo/rom
uv run --group rom python -m tools.extract_pkcalc --out data/intermediate_kaizo/pkcalc
uv run --group rom python -m tools.extract_kaizo_docs --out data/intermediate_kaizo/docs
uv run --group rom python -m tools.kaizo.overrides check
uv run --group rom python -m tools.build_tables --game kaizo
```

## Files

* `config/level_caps.toml`: the level cap per trainer id (Roark, 246: 16) that `prep` uses; a trainer without an entry needs `--cap`.

## Tests against your own save

Two tests need your own Platinum save and are skipped unless the variable is set. Never commit a
save; keep it outside the repository.

* `NUZLINE_ORACLE_DIR`: a folder for the PKHeX cross-check (`tests/saves/test_pkhex_oracle.py`, see `docs/saves.md`).
* `NUZLINE_USER_SAVE`: the path of the save file. `tests/test_user_save.py` checks that the candidate pool built from it is fully covered against Roark.

```
NUZLINE_USER_SAVE=path/to/platinum.dsv uv run pytest tests/test_user_save.py -v
```

## Test tiers

```
uv run pytest -m "not slow"
```

runs the fast tier. Tests marked `slow` (brute-force, loose, save- or ROM-bound) are left out of it. `test_funnel_matches_exhaustive_ci_full` in `tests/prep/test_exhaustive.py` also needs `NUZLINE_SLOW_TESTS=1`. Tier limits and cadence: `CLAUDE.md`.

## Licence and credits

The code in this repository is under the MIT licence (`LICENSE`). The repository contains no ROM, no save file and no ROM tables; build the tables from your own copy. It does hold small excerpts the tests and citations need: one recorded battle trace (`tests/golden/`) and ROM hashes, trainer names and patch notes (`data/kaizo/`).

This project rests on the work of others:

- [pret/pokeplatinum](https://github.com/pret/pokeplatinum), the decompilation of Pokemon Platinum. Every battle rule here was read from it and is cited by file and line.
- PKCalc, the damage calculator for Platinum Kaizo, used as the second data source.
- [@smogon/calc](https://github.com/smogon/damage-calc), used as the independent damage check (`tests/test_damage_oracle.py`).
- Pokemon Platinum Kaizo (Nightmare Update) and its documentation spreadsheet, by the hack's author.

Pokemon is a trademark of Nintendo, Creatures and Game Freak. This project is not affiliated with or endorsed by them.
