"""The ROM's ``gGameLanguage`` (src/game_version.c:4) is ``Game.rom_language`` in both
games: the exp boost's language test reads it (battle_script.c:9758; K3d-exp design §1.3,
§8 U2). Skips without the ROMs or the linker map."""

from nuzline.data.games import GAMES, KAIZO, LANGUAGE_ENGLISH, VANILLA
from tools.rom.readers import symbol_bytes


def test_both_roms_are_english(vanilla_files, kaizo_files, symbols):
    # Kaizo's header game code is JAK7, not CPUE (docs/kaizo/rom_diff.md:19), so the game
    # code proves nothing here; the byte itself does.
    assert vanilla_files.game_code == "CPUE"
    assert symbol_bytes(vanilla_files, symbols, "gGameLanguage") == bytes([LANGUAGE_ENGLISH])
    assert symbol_bytes(kaizo_files, symbols, "gGameLanguage") == bytes([LANGUAGE_ENGLISH])
    assert GAMES[VANILLA].rom_language == GAMES[KAIZO].rom_language == LANGUAGE_ENGLISH
