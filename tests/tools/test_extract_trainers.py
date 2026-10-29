import json
from pathlib import Path

import pytest

from tools.extract_decomp import extract_trainers

DECOMP = Path("external/pokeplatinum")
pytestmark = pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")


@pytest.fixture(scope="module")
def trainers(tmp_path_factory):
    d = tmp_path_factory.mktemp("inter")
    extract_trainers(DECOMP, d)
    return json.loads((d / "trainers.json").read_text(encoding="utf-8"))


def roark(trainers):
    return next(
        t
        for t in trainers
        if t["name"].lower() == "roark" and t["class"].lower().startswith("leader")
    )


def test_roark_party_shape(trainers):
    r = roark(trainers)
    assert [p["level"] for p in r["party"]] == [12, 12, 14]
    assert r["ai_flags"] != 0


def test_roark_start_condition_is_none(trainers):
    assert roark(trainers)["start_condition"] == "none"


def test_some_trainer_has_hail(trainers):
    assert any(t["start_condition"] == "hail" for t in trainers)


def test_roark_is_the_gym_fight_with_padded_moves(trainers):
    # res/trainers/data/leader_roark.json: TRAINER_LEADER_ROARK (id 246), iv_scale 50,
    # 2x Potion, AI flags BASIC | EVAL_ATTACK | EXPERT (bits 0-2). A listed moveset
    # fills all four slots; the rest stay MOVE_NONE (tools/dataproc/src/trainerproc.c).
    r = roark(trainers)
    assert r["id"] == 246
    assert r["ai_flags"] == 0b111
    assert len(r["items"]) == 2
    assert [p["difficulty"] for p in r["party"]] == [50, 50, 50]
    assert all(len(p["moves"]) == 4 for p in r["party"])
    assert r["party"][0]["moves"][2:] == [0, 0]
    assert all(p["item"] == 0 for p in r["party"])


def test_every_spec_weather_appears(trainers):
    # Spec 4.1: hail (Routes 216/217), rain (212/215), sandstorm (228), fog (210, Coronet).
    conditions = {t["start_condition"] for t in trainers}
    assert {"none", "hail", "rain", "sandstorm", "fog"} <= conditions


def test_start_condition_overrides_are_keyed_by_trainer_id(trainers):
    # The brief's interface: tools/start_conditions.json is {trainer_id: condition}.
    # Each entry also names its TRAINER_* constant, which the extractor checks
    # against generated/trainers.txt so an id drift cannot go unnoticed.
    raw = json.loads(Path("tools/start_conditions.json").read_text(encoding="utf-8"))
    entries = {k: v for k, v in raw.items() if k != "comment"}
    assert entries
    consts = (DECOMP / "generated" / "trainers.txt").read_text(encoding="utf-8").split()
    for key, entry in entries.items():
        assert key.isdecimal(), key
        tid = int(key)
        assert consts[tid] == entry["trainer"]
        assert trainers[tid]["start_condition"] == entry["condition"]
