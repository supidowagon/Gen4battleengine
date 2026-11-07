"""Level caps per trainer (plan 4c Task 3)."""

import pytest

from nuzline.data.trainer import Trainer, load_trainers
from nuzline.prep.caps import (
    CAPS_FILES,
    default_cap,
    load_kaizo_caps,
    load_level_caps,
    next_fight_cap,
)
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


# --- Kaizo (plan K2b Task 16) ---

KAIZO = "platinum_kaizo_nightmare_final"


def kt(label, split, order, optional=False, starter=None):
    return Trainer(
        id=1,
        name=label,
        trainer_class="x",
        class_id=0,
        class_gender=0,
        ai_flags=0,
        items=(),
        party=(),
        start_condition="none",
        format="single",
        label=label,
        split=split,
        split_order=order,
        optional=optional,
        player_starter=starter,
    )


def write(tmp_path, body):
    p = tmp_path / "caps.toml"
    p.write_text('source = "test"\n' + body, encoding="utf-8")
    return p


def test_vanilla_unchanged():
    assert load_level_caps() == {246: ("Roark", 16)}
    assert default_cap(roark_trainer()) == 16
    assert default_cap(roark_trainer(), game="platinum_us") == 16


def test_kaizo_split_cap():
    assert default_cap(kt("Ace Trainer X", "Fantina", (2, 4)), game=KAIZO) == 38
    assert default_cap(kt("Grunt", "Galactic", (5, 0)), game=KAIZO) == 100
    assert default_cap(kt("Grunt", "Elite Four Split", (9, 0)), game=KAIZO) == 100  # sheet name


def test_kaizo_barry_fight_cap():
    barry = kt("Pokémon Trainer Barry #3 [Empoleon]", "Maylene", (3, 0))
    assert default_cap(barry, game=KAIZO) == 42
    assert default_cap(kt("Pokémon Trainer Barry #5", "Byron", (6, 0)), game=KAIZO) == 63
    # a key ends at a word boundary: Barry #30 is not Barry #3
    assert default_cap(kt("Pokémon Trainer Barry #30", "Maylene", (3, 1)), game=KAIZO) == 47


def test_kaizo_no_split_errors():
    with pytest.raises(ValueError, match="data/kaizo/level_caps.toml.*--cap"):
        default_cap(kt("Lady Emi", None, None), game=KAIZO)
    with pytest.raises(ValueError, match="--cap"):
        default_cap(kt("Lady Emi", "Nowhere", (1, 1)), game=KAIZO)


def test_kaizo_file_needs_a_source(tmp_path):
    p = tmp_path / "caps.toml"
    p.write_text("[splits]\nRoark = 16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        default_cap(kt("Leader Roark", "Roark", (0, 9)), p, KAIZO)
    assert load_kaizo_caps(CAPS_FILES[KAIZO])[0]["Galactic"] == 100


def test_next_fight_cap_crosses_split(tmp_path):
    caps = write(
        tmp_path, '[splits]\nRoark = 16\nGardenia = 28\n[fights]\n"Rival [Torterra]" = 20\n'
    )
    a = kt("Leader Roark", "Roark", (0, 9))
    b = kt("Lass B", "Gardenia", (1, 0), optional=True)
    c = kt("Rival [Torterra]", "Gardenia", (1, 1), starter=393)
    d = kt("Rival [Infernape]", "Gardenia", (1, 2), starter=387)
    ts = [a, b, c, d]
    assert next_fight_cap(a, ts, KAIZO, 393, caps) == 20  # skips optional b; c matches Piplup
    assert next_fight_cap(a, ts, KAIZO, 387, caps) == 28  # skips c; d by split
    assert next_fight_cap(d, ts, KAIZO, 387, caps) is None


def test_every_leader_cap_is_his_max_level(kaizo_trainers):  # spec 5.12
    # Spec 5.12: a leader's cap is his highest level. "Leader Fantina?" (id 343, levels 31-34) is
    # the Hearthome gym decoy, not the leader (id 318), so labels ending in "?" are skipped.
    leaders = (
        t
        for t in kaizo_trainers
        if t.label and t.label.startswith("Leader ") and not t.label.endswith("?")
    )
    for t in leaders:
        assert default_cap(t, game=KAIZO) == max(m.level for m in t.party), t.label
