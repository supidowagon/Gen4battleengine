# nuzline

A tool that finds high-EV lines for single-player nuzlocke battles in vanilla Pokemon Platinum (US). Given the player's save file and a chosen trainer fight, it recommends pre-battle setup (box picks, lead, held items, prestatus, predamage) and ranked in-battle actions each turn, maximizing the probability of winning with zero faints. See `docs/design.md`.

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

`--save` is required. `--cap N` is the level cap (default: `config/level_caps.toml`, which lists only Roark; any other trainer needs `--cap`). `--protect` and `--exclude` (repeatable) name a Pokemon by species, nickname, `party:N` or `box:B:S` (slots 0-based, boxes 1-based, as in `save show`); a species both in the party and a box is ambiguous and must be named that way, a bare number is refused. `--reserve ITEM ...` keeps bag items out of the search, `--prestatus poison|burn|paralysis` (repeatable) allows a status inflicted beforehand, `--no-entry-hp` always enters at full HP, `--weight KIND=W` (candy, heart_scale, tm, hm, held) sets the resource cost weights of the tie-break. Search size: `--max-depth` (0 = screen only), `--finalists`, `--screen-n`, `--seed`, `--devices`; `--tree FILE.json` writes the plan tree and `--show-depth` sets the plan levels printed.

The output has the `Setup:` card, the `Prep steps:` (Rare Candies, Heart Scale relearns, TMs, items, deposits and withdrawals, in order), the `Cost:` line (every resource spent), the `Odds ladder` for 0-3 deaths (the requested rung is planned by Stage B, the others are labelled "screen"), the Stage B `Plan:` and a `Search:` summary. Progress lines (`[prep] ...`) go to stderr. A run record is written to `runs/prep_<time>.json` (with the save's `save_sha256`) and a checkpoint to `runs/prep_<time>.ckpt.json` after every unit of work. `--budget SECONDS` stops starting new work after that time and prints the best answer so far ("screen only, no plan" when no solve finished), as does Ctrl-C; the exit code is 0. `--resume FILE` continues from a checkpoint and gives the same answer as an uninterrupted run; it needs the same save, trainer, settings, seed and program version (otherwise one line says what differs, exit 1) and cannot be combined with `--checkpoint`. Exit codes: 0 ok, 1 error, 2 the opponent is unsupported or every setup is refused.

Read a Platinum save (see `docs/saves.md`), or use it in place of `--party` with `--save`:

```
uv run nuzline save show path/to/platinum.sav
uv run nuzline rank --save path/to/platinum.sav --trainer roark
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

Slow tests (minutes each: the exhaustive check of the prep funnel on Roark, the long PP-stall fight) are skipped unless `NUZLINE_SLOW_TESTS=1` is set; the default suite stays under about a minute per file on one device, except `tests/test_cli.py` (one end-to-end `nuzline prep` run at depth 1, about 2 minutes for the file).
