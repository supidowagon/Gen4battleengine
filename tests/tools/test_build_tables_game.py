"""``build_tables`` writes the manifest's ``game`` (and only there)."""

import json
from pathlib import Path

import pytest

from tools import build_tables

INTER = Path(__file__).resolve().parents[2] / "data" / "intermediate"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (INTER / "species.json").exists():
        pytest.skip("intermediates not built")
    out = tmp_path_factory.mktemp("built")
    build_tables.build(INTER, out)
    return out


def test_default_game_is_vanilla(built):
    assert (
        json.loads((built / "manifest.json").read_text(encoding="utf-8"))["game"] == "platinum_us"
    )


def test_game_is_not_in_the_npz(built):
    import numpy as np

    with np.load(built / "tables.npz") as z:
        assert "game" not in z.files


def test_unknown_game_refused(tmp_path):
    with pytest.raises(ValueError, match="vanilla, kaizo"):
        build_tables.build(INTER, tmp_path, game="emerald")
