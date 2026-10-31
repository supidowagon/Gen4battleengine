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

Read a Platinum save (see `docs/saves.md`), or use it in place of `--party` with `--save`:

```
uv run nuzline save show path/to/platinum.sav
uv run nuzline rank --save path/to/platinum.sav --trainer roark
```
