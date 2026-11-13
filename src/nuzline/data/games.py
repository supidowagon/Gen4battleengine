"""The game registry: which data set (vanilla Platinum, Kaizo) lives where.

``load_game`` checks the files on disk before it imports ``nuzline.data.tables`` (which
imports jax), so a missing or mismatched data set is refused with the rebuild command
instead of a jax start-up or a ``FileNotFoundError``. It then checks every hard-coded game
id against the game's data (``nuzline.data.game_constants``, spec §7.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuzline.data.tables import Names, Tables

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data"

VANILLA = "platinum_us"
KAIZO = "platinum_kaizo_nightmare_final"


# enum Language (include/constants/versions.h:23-26): LANGUAGE_ENGLISH = 2.
LANGUAGE_ENGLISH = 2


@dataclass(frozen=True)
class Game:
    """``rom_language`` is the ROM's ``gGameLanguage`` (src/game_version.c:4), a build
    constant: ``-DGAME_LANGUAGE=LANGUAGE_ENGLISH`` (meson.build:74) for the US ROM, game
    code ``CPUE`` (platinum.us/rom_rev1.ini:4). Kaizo (game code ``JAK7``,
    docs/kaizo/rom_diff.md:19) stores ``LANGUAGE_ENGLISH`` in its ``gGameLanguage`` byte
    (tests/tools/rom/test_game_language.py); its ROM diff has no entry for it.
    The exp boost compares a Pokemon's language with it (battle_script.c:9758), never
    with the save's trainer-language byte."""

    id: str
    tables_dir: Path
    intermediate_dir: Path
    rom_language: int = LANGUAGE_ENGLISH


GAMES: dict[str, Game] = {
    VANILLA: Game(VANILLA, _DATA / "gen4_platinum", _DATA / "intermediate"),
    KAIZO: Game(KAIZO, _DATA / "gen4_platinum_kaizo", _DATA / "intermediate_kaizo"),
}
ALIASES = {"vanilla": VANILLA, "kaizo": KAIZO}
# The command that writes each game's intermediate_dir (Kaizo's is the resolved directory
# build_tables writes; extract_rom's raw rows go to its rom/ subdirectory).
EXTRACT = {
    VANILLA: "python -m tools.extract_decomp --out data/intermediate",
    KAIZO: "python -m tools.build_tables --game kaizo",
}


class GameDataError(ValueError):
    """A game's tables are missing or belong to another game."""


def get_game(name: str) -> Game:
    """The game with id or alias ``name`` (case-insensitive)."""
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in GAMES:
        raise ValueError(f"unknown game {name!r}; choose one of: {', '.join(ALIASES)}")
    return GAMES[key]


def manifest_game(manifest: dict) -> str:
    """The game id a manifest was built for (a manifest without ``game`` is vanilla)."""
    return manifest.get("game", VANILLA)


def _alias(game_id: str) -> str:
    return next((a for a, i in ALIASES.items() if i == game_id), game_id)


def load_game(game: Game) -> tuple[Tables, Names]:
    """Load ``game``'s tables and names, refusing absent tables or another game's manifest."""
    npz, manifest_path = game.tables_dir / "tables.npz", game.tables_dir / "manifest.json"
    build = f"python -m tools.build_tables --game {_alias(game.id)}"
    if not npz.exists() or not manifest_path.exists():
        raise GameDataError(f"{game.id}: no tables in {game.tables_dir}; build them with: {build}")
    found = manifest_game(json.loads(manifest_path.read_text(encoding="utf-8")))
    if found != game.id:
        raise GameDataError(
            f"{manifest_path} is for game {found!r}, not {game.id!r}; rebuild with: {build}"
        )
    items_path = game.intermediate_dir / "items.json"
    if not items_path.exists():
        extract = EXTRACT.get(game.id, "the game's extractor")
        raise GameDataError(f"{game.id}: no {items_path}; extract it with: {extract}")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    from nuzline.data.game_constants import check_constants
    from nuzline.data.tables import load_names, load_tables

    tables, names = load_tables(game.tables_dir), load_names(game.tables_dir)
    lines = check_constants(game.id, names, items)
    if lines:
        raise GameDataError(
            f"{game.id}: {len(lines)} hard-coded id(s) contradicted by the game's data "
            "(nuzline.data.game_constants):\n  " + "\n  ".join(lines)
        )
    return tables, names
