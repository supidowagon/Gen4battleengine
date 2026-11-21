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


def test_trainer_class_id_and_gender(trainers):
    # TrainerData_BuildParty seeds the PID from header.trainerType (the class id) and
    # picks genderMod from TrainerClass_Gender (include/data/trainer_class_genders.h).
    r = roark(trainers)
    assert (r["class_id"], r["class_gender"]) == (62, 0)  # TRAINER_CLASS_LEADER_ROARK, male
    classes = (DECOMP / "generated" / "trainer_classes.txt").read_text(encoding="utf-8").split()
    assert classes[r["class_id"]] == "TRAINER_CLASS_LEADER_ROARK"
    assert {t["class_gender"] for t in trainers} == {0, 1, 2}
    gardenia = next(t for t in trainers if t["name"] == "Gardenia")
    assert gardenia["class_gender"] == 1


def test_battle_format_follows_double_battle(trainers):
    # res/trainers/data/twins_emma_and_lil.json sets "double_battle": true, which
    # trainerproc turns into TrainerHeader.battleType = BATTLE_TYPE_DOUBLES.
    assert roark(trainers)["format"] == "single"
    consts = (DECOMP / "generated" / "trainers.txt").read_text(encoding="utf-8").split()
    emma = trainers[consts.index("TRAINER_TWINS_EMMA_AND_LIL")]
    assert (emma["name"], emma["format"]) == ("Emma & Lil", "double")
    doubles = {
        p.stem
        for p in (DECOMP / "res" / "trainers" / "data").glob("*.json")
        if json.loads(p.read_text(encoding="utf-8"))["double_battle"]
    }
    assert sum(t["format"] == "double" for t in trainers) == len(doubles) == 28
    assert {t["format"] for t in trainers} == {"single", "double", "double_two_trainer", "multi"}


def test_battle_structures(trainers):
    consts = (DECOMP / "generated" / "trainers.txt").read_text(encoding="utf-8").split()
    by = dict(zip(consts[: len(trainers)], trainers, strict=True))  # drop the count sentinel
    assert sum(t["format"] == "double_two_trainer" for t in trainers) == 20  # 10 pairs (Ruling 2)
    g1, g2 = by["TRAINER_GALACTIC_GRUNT_IRON_ISLAND_1"], by["TRAINER_GALACTIC_GRUNT_IRON_ISLAND_2"]
    riley = by["TRAINER_RILEY_IRON_ISLAND"]
    assert (g1["format"], g1["opponent_2"], g2["opponent_2"]) == ("multi", g2["id"], g1["id"])
    assert g1["partner"] == [{"trainer": riley["id"], "starter": None, "player_gender": None}]
    mars = by["TRAINER_COMMANDER_MARS_SPEAR_PILLAR"]
    jupiter = by["TRAINER_COMMANDER_JUPITER_SPEAR_PILLAR"]
    assert (mars["format"], mars["opponent_2"]) == ("multi", jupiter["id"])
    assert {(p["starter"], p["player_gender"]) for p in mars["partner"]} == {
        (387, None),
        (390, None),
        (393, None),
    }
    assert len(by["TRAINER_GALACTIC_GRUNT_JUBILIFE_CITY_1"]["partner"]) == 6
    jack, cheryl = (
        by["TRAINER_BUG_CATCHER_JACK"],
        by["TRAINER_CHERYL_ETERNA_FOREST"],
    )  # Eterna Forest
    # A lone placed enemy on a walking-partner map is BATTLE_TYPE_TRAINER ORed with the
    # partner's header type (encounter.c:738-739, trainer_data.c:31-50): Cheryl's is single.
    assert jack["format"] == "single"
    assert jack["partner"] == []
    assert jack["walking_partner"] == [cheryl["id"]]
    # The scripted pair on Riley's map stays multi, and carries the walking partner.
    assert g1["walking_partner"] == g2["walking_partner"] == [riley["id"]]
    for partner in (cheryl, riley):  # Ruling 3: partners are not placed enemies
        assert (partner["format"], partner["partner"], partner["opponent_2"]) == (
            "single",
            [],
            None,
        )
    assert roark(trainers)["partner"] == [] and roark(trainers)["opponent_2"] is None


def test_partner_variants_name_the_rival_team(trainers):
    # scripts_spear_pillar.s:295-303: Barry's team follows the player's starter only.
    consts = (DECOMP / "generated" / "trainers.txt").read_text(encoding="utf-8").split()
    by = dict(zip(consts[: len(trainers)], trainers, strict=True))  # drop the count sentinel
    mars = by["TRAINER_COMMANDER_MARS_SPEAR_PILLAR"]
    team = {p["starter"]: p["trainer"] for p in mars["partner"]}
    assert team == {
        390: by["TRAINER_RIVAL_SPEAR_PILLAR_CHIMCHAR"]["id"],
        387: by["TRAINER_RIVAL_SPEAR_PILLAR_TURTWIG"]["id"],
        393: by["TRAINER_RIVAL_SPEAR_PILLAR_PIPLUP"]["id"],
    }
    # scripts_jubilife_city.s:901-923: Dawn partners a male player, Lucas a female one.
    grunt = by["TRAINER_GALACTIC_GRUNT_JUBILIFE_CITY_2"]
    dawn = by["TRAINER_DAWN_JUBILIFE_CITY_CHIMCHAR"]["id"]
    lucas = by["TRAINER_LUCAS_JUBILIFE_CITY_PIPLUP"]["id"]
    assert {"trainer": dawn, "starter": 390, "player_gender": 0} in grunt["partner"]
    assert {"trainer": lucas, "starter": 393, "player_gender": 1} in grunt["partner"]
    for p in grunt["partner"]:  # a partner routine's teams are not placed enemies
        assert trainers[p["trainer"]]["format"] == "single"
