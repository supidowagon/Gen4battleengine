"""Apply the Kaizo xdelta patch to the vanilla ROM, with a verified cache.

``data/kaizo/rom.toml`` records the patched ROM's hash and header (no ROM bytes).
The cache holds ``kaizo-<patch sha256[:16]>.nds`` and a ``.sha1`` sidecar; a hit is
re-hashed against the sidecar, anything else is rebuilt (plan K1 Ruling 6).
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tools.rom.inputs import (
    KAIZO_PATCH_SHA256,
    REPO_ROOT,
    VANILLA_SHA1,
    RomInputError,
    RomInputs,
    read_checked,
)
from tools.rom.vcdiff import apply_vcdiff

ROM_TOML = REPO_ROOT / "data" / "kaizo" / "rom.toml"


@dataclass(frozen=True)
class RomHeader:
    """The NDS header fields compared between vanilla and patched ROMs."""

    title: str
    game_code: str
    maker: str
    revision: int


def rom_header(data: bytes) -> RomHeader:
    """Read title (0x00-0x0B), game code (0x0C-0x0F), maker (0x10-0x11), revision (0x1E)."""
    if len(data) < 0x20:
        raise RomInputError(f"ROM too short for an NDS header ({len(data)} bytes)")
    return RomHeader(
        title=data[0x00:0x0C].rstrip(b"\0").decode("ascii"),
        game_code=data[0x0C:0x10].decode("ascii"),
        maker=data[0x10:0x12].decode("ascii"),
        revision=data[0x1E],
    )


def expected_patched(path: Path = ROM_TOML) -> dict | None:
    """The recorded patched-ROM values, or None when ``path`` does not exist."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        return tomllib.load(f)


def _cached(rom: Path, sidecar: Path) -> str | None:
    """The ROM's SHA-1 when it is intact against its sidecar, else None."""
    try:
        recorded = sidecar.read_text().strip()
        actual = hashlib.sha1(rom.read_bytes()).hexdigest()
    except OSError:
        return None
    return actual if actual == recorded else None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def kaizo_rom(
    inputs: RomInputs, cache_dir: Path = REPO_ROOT / ".cache" / "rom"
) -> tuple[Path, str]:
    """Return the patched ROM's path and SHA-1, building it into ``cache_dir`` if needed.

    The vanilla SHA-1 and patch SHA-256 are checked before anything is patched; the
    patched size, header and SHA-1 are checked against ``data/kaizo/rom.toml``.
    """
    if inputs.kaizo_patch is None:
        raise RomInputError(
            "kaizo_patch is not set: add it to data/local.toml or NUZLINE_KAIZO_PATCH"
        )
    vanilla = read_checked(inputs.vanilla_rom, VANILLA_SHA1, "sha1")
    patch = read_checked(inputs.kaizo_patch, KAIZO_PATCH_SHA256, "sha256")
    exp = expected_patched()

    rom = cache_dir / f"kaizo-{KAIZO_PATCH_SHA256[:16]}.nds"
    sidecar = rom.with_name(rom.name + ".sha1")
    sha = _cached(rom, sidecar)
    if sha is None:
        patched = apply_vcdiff(vanilla, patch)
        sha = hashlib.sha1(patched).hexdigest()
        _check(patched, sha, vanilla, exp)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(rom, patched)
        _write_atomic(sidecar, sha.encode("ascii"))
    elif exp is not None and sha != exp["sha1"]:
        raise RomInputError(f"{rom}: cached SHA-1 {sha} != {exp['sha1']} (data/kaizo/rom.toml)")
    return rom, sha


def _check(patched: bytes, sha: str, vanilla: bytes, exp: dict | None) -> None:
    """Verify the freshly patched ROM against vanilla's header and ``rom.toml``."""
    h, v = rom_header(patched), rom_header(vanilla)
    if (h.title, h.maker, h.revision) != (v.title, v.maker, v.revision):
        raise RomInputError(f"patched header {h} differs from vanilla {v} in title/maker/revision")
    if exp is None:
        return
    if len(patched) != exp["size"]:
        raise RomInputError(f"patched ROM size {len(patched)} != {exp['size']} (rom.toml)")
    if h.game_code != exp["game_code"]:
        raise RomInputError(f"patched game code {h.game_code} != {exp['game_code']} (rom.toml)")
    if sha != exp["sha1"]:
        raise RomInputError(f"patched SHA-1 {sha} != {exp['sha1']} (rom.toml)")
