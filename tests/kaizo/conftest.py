"""Input fixtures for the source-level Kaizo tests.

Each fixture skips when its local input is absent: ``load_inputs()`` raising
``RomInputError`` or the input's path being ``None`` or missing, and node missing for
the PKCalc data files (``shutil.which("node")``).
"""

import shutil

import pytest

from nuzline.data.games import GAMES, KAIZO
from tools.rom.inputs import RomInputError, load_inputs


@pytest.fixture(scope="session")
def pkcalc_zip():
    """The configured PKCalc zip; skips when it is not configured or absent."""
    try:
        path = load_inputs().pkcalc_zip
    except RomInputError as e:
        pytest.skip(str(e))
    if path is None:
        pytest.skip("pkcalc_zip is not configured (data/local.toml or NUZLINE_PKCALC_ZIP)")
    if not path.exists():
        pytest.skip(f"configured PKCalc zip {path} is absent")
    return path


@pytest.fixture(scope="session")
def pkcalc(pkcalc_zip, tmp_path_factory):
    """``pkcalc.json`` extracted from the real zip once per session; skips without node."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    from tools.extract_pkcalc import extract_pkcalc

    return extract_pkcalc(pkcalc_zip, tmp_path_factory.mktemp("pkcalc"))


@pytest.fixture(scope="session")
def kaizo_intermediate():
    """The resolved Kaizo intermediates directory; skips until it is built."""
    inter = GAMES[KAIZO].intermediate_dir
    if not (inter / "renamed.json").exists():
        pytest.skip("Kaizo intermediates not resolved (python -m tools.build_tables --game kaizo)")
    return inter
