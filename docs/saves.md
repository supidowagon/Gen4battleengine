# Platinum saves

nuzline reads the player's party, PC boxes and bag straight from a Pokemon Platinum (US)
save (`.sav` or `.dsv`). Design: `docs/superpowers/specs/2026-10-31-save-parser-design.md`.

## Commands

```
uv run nuzline save show SAVE [--json] [--allow-corrupt]
uv run nuzline save export SAVE -o party.toml [--allow-corrupt]
```

* `save show` prints the trainer, which save copy was loaded, the party and every box
  (one line per Pokemon); `--json` prints the whole save instead.
* `save export` writes the party (eggs dropped) as a `--party` TOML file.
* `coverage`, `rank` and `solve` take `--save SAVE [--allow-corrupt]` instead of
  `--party`: the party is the save's, eggs left out (`--lead` indexes the egg-free party).
* A corrupt save (the game would fall back to an older block) is refused unless
  `--allow-corrupt` is given. A bad save or record is one line on stderr and exit code 1.

## PKHeX oracle test

`tests/saves/test_pkhex_oracle.py` compares the parser with PKHeX's reading of the same
save. It is skipped unless `NUZLINE_ORACLE_DIR` is set. Never commit saves; keep the
folder outside the repository.

Build the folder from your own save:

1. Copy the save (`.sav` or `.dsv`) into a new folder, e.g. `C:\oracle\platinum`.
2. Open the save in PKHeX and export its Box Data Report as CSV
   (PKHeX's Box Data Report, under Tools > Data in recent versions). Put the `.csv` in the
   folder.
3. Export each party Pokemon as a `.pk4` file into the same folder (PKHeX can save a
   Pokemon as a `.pk4` file, e.g. via its right-click menu in the party view, or by
   dragging the Pokemon out to the folder). The test accepts 136- or 236-byte files.

The folder must hold exactly one `.sav`/`.dsv` and one `.csv`, plus the `.pk4` files.

Run it:

```
$env:NUZLINE_ORACLE_DIR = "C:\oracle\platinum"   # PowerShell
uv run pytest tests/saves/test_pkhex_oracle.py -q
```

(In a POSIX shell: `NUZLINE_ORACLE_DIR=/path/to/oracle uv run pytest tests/saves/test_pkhex_oracle.py -q`.)

What it checks: the save loads with result `ok`; each party slot's decrypted record equals
the `.pk4` with the same personality over their common length; each CSV row (species,
level, nature, ability, held item, four moves, IVs, EVs) equals the parser's Pokemon at
that box and slot (1-based in PKHeX). CSV columns are found by header name,
case-insensitively; if a required column has no match the test fails and lists the headers
it saw, so the alias table in the test can be extended. The box column must be numeric: if
your boxes have custom names, the report needs another numeric box column (e.g. `Box Number`).
The `.pk4` files must have distinct personalities. Accents are ignored when comparing names. Names are compared ignoring case,
spaces and punctuation.
