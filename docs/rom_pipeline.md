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
