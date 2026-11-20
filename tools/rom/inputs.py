"""Locate and verify the local ROM pipeline inputs.

Paths come from environment variables (``NUZLINE_*``) first, then ``data/local.toml``
(git-ignored; ``data/local.example.toml`` lists every key). ``vanilla_rom`` is
required, the rest may be absent. Existence is not checked here; ``read_checked``
checks a file's hash when it is read, and any mismatch stops the build (spec §6.1).
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Platinum (U) rev 1, game code CPUE.
VANILLA_SHA1 = "0862ec35b24de5c7e2dcb88c9eea0873110d755c"
# Platinum Kaizo Nightmare Update Final.xdelta
KAIZO_PATCH_SHA256 = "b787e743b5fa78e1b31bae9277e656482b7248b82e490a6e95236606ae92f433"
# PKCalc-main.zip audited 2026-11-03. Recorded only, not checked by read_checked (plan
# Ruling 19): extract_pkcalc records the zip's hash; audited-count tests skip on another.
PKCALC_SHA256 = "849733e658fc0116aaf0fc2f27db71d416165d79107cf3a0e29effce7062eb0f"
# Platinum Kaizo Docs.xlsx audited 2026-11-03. Recorded only, not checked by read_checked
# (plan Ruling 19).
KAIZO_DOCS_SHA256 = "41c0c7fe8bb2b5d2179670f6e92655431ff86752a45370a11605715be8b1d74f"

# TOML key -> environment variable.
_ENV = {
    "vanilla_rom": "NUZLINE_VANILLA_ROM",
    "kaizo_patch": "NUZLINE_KAIZO_PATCH",
    "pkcalc_zip": "NUZLINE_PKCALC_ZIP",
    "kaizo_docs": "NUZLINE_KAIZO_DOCS",
    "kaizo_save": "NUZLINE_KAIZO_SAVE",
    "kaizo_save_pre_roark": "NUZLINE_KAIZO_SAVE_PRE_ROARK",
    "kaizo_save_probe": "NUZLINE_KAIZO_SAVE_PROBE",
}


class RomInputError(Exception):
    """A ROM pipeline input is missing, malformed or fails its hash check."""


class RomInputNotSet(RomInputError):
    """``vanilla_rom`` is configured nowhere (neither the environment nor the TOML)."""


@dataclass(frozen=True)
class RomInputs:
    """Paths to the local inputs; only ``vanilla_rom`` is required."""

    vanilla_rom: Path
    kaizo_patch: Path | None
    pkcalc_zip: Path | None
    kaizo_docs: Path | None
    kaizo_save: Path | None
    kaizo_save_pre_roark: Path | None = None  # a second, earlier save (pre-Roark)
    kaizo_save_probe: Path | None = None  # the trace probe's save (the party taken to Roark)


def _read_toml(config: Path) -> dict[str, object]:
    """Parse ``config`` (empty when absent), rejecting unknown keys. The top-level ``run``
    key, whatever its type, is the Nuzlocke rules table of ``nuzline prep``
    (``nuzline.data.gauntlet.load_local`` reads and validates it): skipped here, and
    ``load_inputs`` never reads it."""
    if not config.exists():
        return {}
    try:
        with config.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise RomInputError(
            f"{config}: {e}; use single-quoted literal strings for Windows paths "
            "(vanilla_rom = 'G:\\dir\\file.nds')"
        ) from e
    unknown = sorted(set(data) - set(_ENV) - {"run"})
    if unknown:
        raise RomInputError(f"{config}: unknown key(s) {', '.join(unknown)}; known: {list(_ENV)}")
    return data


def load_inputs(
    config: Path = REPO_ROOT / "data" / "local.toml",
    environ: Mapping[str, str] = os.environ,
) -> RomInputs:
    """Resolve the inputs: environment variable, then ``config``, then absent.

    Relative TOML paths resolve against ``REPO_ROOT``; environment paths are used
    as given.
    """
    toml = _read_toml(config)
    found: dict[str, Path | None] = {}
    for key, env in _ENV.items():
        if environ.get(env):
            found[key] = Path(environ[env])
        elif key in toml:
            found[key] = REPO_ROOT / str(toml[key])
        else:
            found[key] = None
    if found["vanilla_rom"] is None:
        raise RomInputNotSet(
            f"vanilla_rom is not set: add it to {config} or set NUZLINE_VANILLA_ROM"
        )
    return RomInputs(**found)  # type: ignore[arg-type]


def read_checked(path: Path, expected_sha: str, algo: str) -> bytes:
    """Read ``path`` and verify its ``algo`` (a ``hashlib`` name) digest."""
    try:
        data = path.read_bytes()
    except OSError as e:
        raise RomInputError(f"{path}: cannot read ({e.strerror or e})") from e
    actual = hashlib.new(algo, data).hexdigest()
    if actual != expected_sha:
        raise RomInputError(f"{path}: {algo} mismatch: expected {expected_sha}, got {actual}")
    return data
