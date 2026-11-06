"""The game registry and ``load_game``'s checks, which run before jax is imported."""

import json
import shutil
import subprocess
import sys

import pytest

from nuzline.data.games import (
    GAMES,
    Game,
    GameDataError,
    get_game,
    load_game,
    manifest_game,
)
from tests.conftest import TABLES_DIR, _require_tables

KAIZO = "platinum_kaizo_nightmare_final"


def test_aliases_and_ids():
    assert get_game("vanilla") is GAMES["platinum_us"] is get_game("PLATINUM_US")
    assert get_game("kaizo").tables_dir.name == "gen4_platinum_kaizo"
    assert get_game("Kaizo") is GAMES[KAIZO]
    assert GAMES["platinum_us"].tables_dir.name == "gen4_platinum"
    assert GAMES["platinum_us"].intermediate_dir.name == "intermediate"
    assert GAMES[KAIZO].intermediate_dir.name == "intermediate_kaizo"


def test_unknown_game():
    with pytest.raises(ValueError, match="vanilla, kaizo"):
        get_game("emerald")


def test_missing_tables_names_the_build_command(tmp_path):
    g = Game(KAIZO, tmp_path, tmp_path)
    with pytest.raises(GameDataError, match=r"tools\.build_tables --game kaizo"):
        load_game(g)


def test_manifest_of_other_game_refused(tmp_path):
    _require_tables()
    for f in ("tables.npz", "manifest.json"):
        shutil.copy(TABLES_DIR / f, tmp_path / f)
    with pytest.raises(GameDataError, match=f"platinum_us.*{KAIZO}"):
        load_game(Game(KAIZO, tmp_path, tmp_path))


def test_manifest_without_game_is_vanilla():
    assert manifest_game({"version": 8}) == "platinum_us"
    assert manifest_game({"version": 8, "game": KAIZO}) == KAIZO


def test_vanilla_manifest_game():
    manifest = json.loads((TABLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["game"] == "platinum_us"


def test_vanilla_loads():
    _require_tables()
    tables, names = load_game(GAMES["platinum_us"])
    assert tables.move_power.shape[0] == len(names.move_names)


def test_refusals_do_not_import_jax(tmp_path):
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from nuzline.data.games import Game, GameDataError, load_game\n"
        f"g = Game({KAIZO!r}, Path({str(tmp_path)!r}), Path({str(tmp_path)!r}))\n"
        "try:\n    load_game(g)\nexcept GameDataError:\n    pass\n"
        "assert 'jax' not in sys.modules and 'nuzline.data.tables' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
