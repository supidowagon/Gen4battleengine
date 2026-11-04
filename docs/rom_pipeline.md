# ROM pipeline

Reads the vanilla and Kaizo Platinum ROMs into the extraction tables (spec
`docs/superpowers/specs/2026-11-03-kaizo-data-design.md`). Code lives in `tools/rom/`;
install its dependencies with the `rom` group (`uv run --group rom ...`).

## Inputs

Inputs are local files that are never committed: ROM bytes, NARC members, code, the
patched ROM and anything that only makes sense with them stay out of git.

Each path is read from an environment variable, else from `data/local.toml`
(git-ignored; copy `data/local.example.toml`). Relative TOML paths resolve against the
repository root. Use single-quoted literal strings for Windows paths.

| TOML key | Environment variable | Required | Hash checked |
| --- | --- | --- | --- |
| `vanilla_rom` | `NUZLINE_VANILLA_ROM` | yes | SHA-1 `0862ec35b24de5c7e2dcb88c9eea0873110d755c` |
| `kaizo_patch` | `NUZLINE_KAIZO_PATCH` | no | SHA-256 `b787e743b5fa78e1b31bae9277e656482b7248b82e490a6e95236606ae92f433` |
| `pkcalc_zip` | `NUZLINE_PKCALC_ZIP` | no | none |
| `kaizo_docs` | `NUZLINE_KAIZO_DOCS` | no | none |
| `kaizo_save` | `NUZLINE_KAIZO_SAVE` | no | none |

A hash mismatch stops the build with the path, the expected hash and the actual hash.
Tests that need a local input skip when it is absent.

**Never commit `data/local.toml`, a ROM, a patch or a save.**

## Decomp build

The ROM diff names the function or data symbol behind every changed byte from the
Metrowerks linker map of the vanilla decomp build (`build/main.nef.xMAP`, written by
`-map closure,unused` at `meson.build:99`). Build it once, in WSL
(`external/pokeplatinum/INSTALL.md`, WSL2 section and §2-3):

0. Precheck, in PowerShell: `wsl -l -v` must list `Ubuntu` with `VERSION 2`. On WSL 1,
   `INSTALL.md` advises building on the Windows drive instead of the Linux filesystem;
   stop and ask rather than adapting the steps.
1. In `wsl -d Ubuntu` (needs your sudo password; `INSTALL.md:72-75`):
   `sudo dpkg --add-architecture i386 && sudo apt update`
   `sudo apt install bison flex g++ gcc-arm-none-eabi git make ninja-build pkg-config wget python3 xz-utils nasm libc6:i386 libpng-dev`
2. Then, in the Linux filesystem (`INSTALL.md:66-69, 315-318, 324-334`):
   `cd ~ && git clone https://github.com/pret/pokeplatinum && cd pokeplatinum && git -c advice.detachedHead=false checkout 579d2d5c65e736a1a4268fb7acbfd70712177c6c && make`
   `make` is `release check` (`Makefile:96-106`): it downloads metroskrew and meson 1.12.0,
   builds `build/pokeplatinum.us.nds`, and its "ROM Checksum" test must pass (SHA-1
   `0862ec35…755c`, `INSTALL.md:334`).
3. `ls build/*.xMAP`. Expected `build/main.nef.xMAP`.
4. Copy the map and `platinum.us/main.lsf` into the worktree. In WSL, with `WT` set to the
   worktree's Windows path exactly as `git rev-parse --show-toplevel` prints it there (for
   example `G:/Farry/Claude Workspace/Pokemon Platinum Game Engine/.claude/worktrees/k1-rom-pipeline`):
   ```bash
   DEST="$(wslpath -a "$WT")/.cache/decomp/579d2d5c65e736a1a4268fb7acbfd70712177c6c"
   mkdir -p "$DEST" && cp build/main.nef.xMAP platinum.us/main.lsf "$DEST/" && ls -l "$DEST"
   ```

Expected: `sha1sum build/pokeplatinum.us.nds` prints `0862ec35b24de5c7e2dcb88c9eea0873110d755c`.

`tools/rom/symbols.py` reads the map from `.cache/decomp/<commit>/main.nef.xMAP`
(git-ignored); the symbol tests skip when it is absent. **Never commit the map or
`main.lsf`**; only the hand-cut excerpt `tests/tools/rom/fixtures/xmap_excerpt.txt`
(names and addresses) is committed.

## ROM diff

`python -m tools.rom.diff` compares the vanilla ROM with the patched Kaizo ROM (header,
files, NARC members, code by linker-map symbol) and writes `docs/kaizo/rom_diff.md`: names,
addresses, sizes and counts only, never ROM bytes. Defaults find `data/local.toml`, the map
at `.cache/decomp/<commit>/main.nef.xMAP` and the decomp at `external/pokeplatinum`; the flags
`--symbols`, `--decomp`, `--classification` and `--out` override them.

`data/kaizo/rom_diff.toml` classifies every changed symbol (`[[code]]` tables with `module`,
`symbol`, `status` and `note`; `symbol` is `"0xSTART-0xEND"` for a range outside every symbol;
`status` is `modelled`, `irrelevant` or `unknown`). `--init-classification` writes an
all-`unknown` skeleton and refuses to overwrite an existing file. An entry that matches no
current change is an error naming it, so a renamed or moved change cannot hide.

## Extraction

```bash
uv run --group rom python -m tools.extract_rom --game vanilla [--out DIR] [--decomp external/pokeplatinum] \
    [--symbols .cache/decomp/<commit>/main.nef.xMAP] [--config data/local.toml]
```

`--out` defaults to `data/intermediate` (vanilla) or `data/intermediate_kaizo` (kaizo).
`species, moves, items, learnsets, exp_tables` and `trainers` (with start conditions derived
from the ROM's map headers, events and scripts) are read from the ROM; `types, natures,
charmap, abilities` are copied from a fresh `extract_decomp.extract_static` run (the charmap is
the font encoding and has no table in the ROM). For vanilla, the output equals the decomp
extractor's, which `tests/tools/rom/test_extract_rom.py` checks file by file.

`rom_sources.json` beside the outputs maps each file to `"rom"` or a `"decomp: ..."` reason.

Writes go to a temporary directory next to `--out` and are moved in file by file with
`os.replace`; other files in `--out` are kept, and on failure `--out` is not created or changed.

`--game kaizo` runs the ROM diff, checks that the sources of the reused tables
(`sTypeMatchupMultipliers`, `sNatureStatAffinities`, the ability, type and nature name banks)
are unchanged, runs the readers that do not depend on the trainer layout as layout checks, and
then refuses: `trainers: the Kaizo trainer layout is not confirmed (spec §5.4); Kaizo
extraction lands in K2b`. Nothing is written.
