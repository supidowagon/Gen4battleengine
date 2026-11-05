"""Level caps per trainer (plan 4c Task 3)."""

import pytest

from nuzline.data.trainer import load_trainers
from nuzline.prep.caps import default_cap, load_level_caps
from tests.engine.helpers import INTERMEDIATE, roark_trainer


def test_level_caps_file():
    assert load_level_caps() == {246: ("Roark", 16)}
    assert default_cap(roark_trainer()) == 16
    other = next(t for t in load_trainers(INTERMEDIATE) if t.id == 858)
    with pytest.raises(ValueError, match="--cap") as e:
        default_cap(other)
    assert other.name in str(e.value)


def test_cap_file_naming_another_trainer_is_refused(tmp_path):
    path = tmp_path / "caps.toml"
    path.write_text('[caps]\n246 = { name = "Gardenia", cap = 16 }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Gardenia"):
        default_cap(roark_trainer(), path)
    path.write_text('[caps]\n246 = { name = "ROARK", cap = 17 }\n', encoding="utf-8")
    assert default_cap(roark_trainer(), path) == 17
