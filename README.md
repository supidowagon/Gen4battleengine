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
