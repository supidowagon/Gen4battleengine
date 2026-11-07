"""Kaizo saves: the same layout read with the Kaizo tables (spec §5.11), the player's
starter, and the impossible-content warnings (spec §8). The real-save tests skip unless
the save is configured (``data/local.toml`` or ``NUZLINE_KAIZO_SAVE`` /
``NUZLINE_KAIZO_SAVE_PRE_ROARK``) and the Kaizo tables are built."""

from __future__ import annotations

import copy

import pytest

from nuzline.data.games import KAIZO
from nuzline.saves import load_save
from nuzline.saves.check import impossible_contents
from tests.saves import writer

_STARTERS = (387, 390, 393)  # Turtwig, Chimchar, Piplup


def _party_mon(species=1, level=5, moves=(33, 0, 0, 0), pp=(35, 0, 0, 0), item=0):
    return writer.mon_record(
        species=species,
        level=level,
        moves=moves,
        pp=pp,
        item=item,
        party={"status": 0, "level": level, "hp": 20, "stats": (20,) * 6},
    )


def _write(tmp_path, image: bytes):
    path = tmp_path / "game.sav"
    path.write_bytes(image)
    return path


def bulbasaur(moves=(33,)):
    """A party Bulbasaur (species 1) record with ``moves`` (PP 5 each)."""
    moves = tuple(moves) + (0,) * (4 - len(moves))
    return _party_mon(moves=moves, pp=tuple(5 if m else 0 for m in moves))


def save_with_party(*records, bag=None, tmp_path, tables, names):
    return load_save(
        _write(tmp_path, writer.build_image(party=list(records), bag=bag)),
        tables=tables,
        names=names,
    )


def test_growth_rate_differs_between_games(tmp_path, tables, kaizo_tables):
    # Onix 95: Medium Fast in vanilla, Slow in Kaizo.
    image = writer.build_image(
        party=[_party_mon()], boxes={(0, 0): writer.mon_record(species=95, exp=100_000)}
    )
    path = _write(tmp_path, image)
    assert load_save(path).boxes[0][0].level == 46  # 46^3 <= 100000 < 47^3
    assert load_save(path, game=KAIZO).boxes[0][0].level == 43  # 5*43^3/4 <= 100000
    assert load_save(path, game="kaizo").boxes[0][0].level == 43  # aliases resolve


def test_unknown_game_refused(tmp_path, tables):
    path = _write(tmp_path, writer.build_image(party=[_party_mon()]))
    with pytest.raises(ValueError, match="unknown game"):
        load_save(path, game="emerald")


def test_other_games_tables_refused(tables):
    from nuzline.data.games import GAMES, VANILLA, GameDataError
    from nuzline.saves.save import default_tables_and_names

    with pytest.raises(GameDataError, match="is for game 'platinum_us', not 'platinum_kaizo"):
        default_tables_and_names(None, None, GAMES[VANILLA].tables_dir, game="kaizo")


def test_impossible_move_warned(tmp_path, tables, names, learnsets):  # Bulbasaur with Surf
    save = save_with_party(bulbasaur(moves=(57,)), tmp_path=tmp_path, tables=tables, names=names)
    lines = impossible_contents(save, tables, names, learnsets)
    # The game's manifest name: upper case in vanilla, title case in Kaizo (controller ruling).
    assert lines == ["party slot 0 BULBASAUR: Surf is not learnable"]


def test_impossible_move_warned_kaizo_title_case(
    tmp_path, kaizo_tables, kaizo_names, kaizo_learnsets
):
    save = save_with_party(
        bulbasaur(moves=(57,)), tmp_path=tmp_path, tables=kaizo_tables, names=kaizo_names
    )
    lines = impossible_contents(save, kaizo_tables, kaizo_names, kaizo_learnsets)
    assert lines == ["party slot 0 Bulbasaur: Surf is not learnable"]


def _lines_for(tmp_path, tables, names, learnsets, species, move, form=0):
    rec = _party_mon(species=species, level=50, moves=(move, 0, 0, 0))
    if form:
        rec = writer.mon_record(
            species=species,
            level=50,
            form=form,
            moves=(move, 0, 0, 0),
            pp=(5, 0, 0, 0),
            party={"status": 0, "level": 50, "hp": 20, "stats": (20,) * 6},
        )
    save = save_with_party(rec, tmp_path=tmp_path, tables=tables, names=names)
    return impossible_contents(save, tables, names, learnsets)


@pytest.mark.parametrize(
    ("form", "move"),
    [(1, "Overheat"), (2, "Hydro Pump"), (3, "Blizzard"), (4, "Air Slash"), (5, "Leaf Storm")],
)
def test_rotom_form_moves(tmp_path, tables, names, learnsets, form, move):
    # Pokemon_SetRotomForm, src/pokemon.c:4520-4537: only the matching form knows it.
    ok = _lines_for(tmp_path, tables, names, learnsets, 479, names.move(move), form)
    assert ok == []
    other = 1 if form != 1 else 2
    bad = _lines_for(tmp_path, tables, names, learnsets, 479, names.move(move), other)
    assert bad == [f"party slot 0 ROTOM: {move} is not learnable"]


def test_draco_meteor_for_dragon_types(tmp_path, tables, names, learnsets):
    # scripts_route_210_grandma_wilma_house.s:105,129-138: any Dragon-type.
    draco = names.move("Draco Meteor")
    for species in (names.species("Dratini"), names.species("Altaria")):
        assert _lines_for(tmp_path, tables, names, learnsets, species, draco) == []
    lines = _lines_for(tmp_path, tables, names, learnsets, names.species("Swablu"), draco)
    assert lines == ["party slot 0 SWABLU: Draco Meteor is not learnable"]  # Normal/Flying


@pytest.mark.parametrize(
    ("species", "move"),
    [
        ("Charizard", "Blast Burn"),
        ("Infernape", "Blast Burn"),
        ("Blastoise", "Hydro Cannon"),
        ("Empoleon", "Hydro Cannon"),
        ("Venusaur", "Frenzy Plant"),
        ("Torterra", "Frenzy Plant"),
    ],
)
def test_ultimate_moves_for_final_starters(tmp_path, tables, names, learnsets, species, move):
    # scripts_route_228_north_house.s:55,69 (CheckFinalStageStarter :85-110).
    lines = _lines_for(tmp_path, tables, names, learnsets, names.species(species), names.move(move))
    assert lines == []


def test_ultimate_move_needs_the_final_stage(tmp_path, tables, names, learnsets):
    blast_burn = names.move("Blast Burn")
    lines = _lines_for(tmp_path, tables, names, learnsets, names.species("Monferno"), blast_burn)
    assert lines == ["party slot 0 MONFERNO: Blast Burn is not learnable"]


def test_volt_tackle_for_the_pichu_line(tmp_path, tables, names, learnsets):
    # Egg_TryGiveVoltTackle, src/overlay005/daycare.c:602-617 (Pichu eggs, :772).
    volt = names.move("Volt Tackle")
    for species in ("Pichu", "Pikachu", "Raichu"):
        sid = names.species(species)
        assert _lines_for(tmp_path, tables, names, learnsets, sid, volt) == []
    lines = _lines_for(tmp_path, tables, names, learnsets, names.species("Shinx"), volt)
    assert lines == ["party slot 0 SHINX: Volt Tackle is not learnable"]


def test_learnable_moves_pass(tmp_path, tables, names, learnsets):
    # Tackle (level-up), Solar Beam (TM22), Petal Dance (egg move), Seed Bomb (tutor).
    moves = (names.move("Tackle"), names.move("SolarBeam"), names.move("Petal Dance"))
    tutor = names.move("Seed Bomb")
    assert any(t.move == tutor for t in learnsets.tutor_moves())
    save = save_with_party(
        bulbasaur(moves=(*moves, tutor)), tmp_path=tmp_path, tables=tables, names=names
    )
    assert impossible_contents(save, tables, names, learnsets) == []


def test_pre_evolution_moves_count(tmp_path, tables, names, learnsets):
    # Venusaur (3) with Petal Dance, an egg move of Bulbasaur only.
    rec = _party_mon(species=3, level=50, moves=(names.move("Petal Dance"), 0, 0, 0))
    save = save_with_party(rec, tmp_path=tmp_path, tables=tables, names=names)
    assert impossible_contents(save, tables, names, learnsets) == []


def test_smeargle_skipped(tmp_path, tables, names, learnsets):
    rec = _party_mon(species=235, level=20, moves=(57, 0, 0, 0))
    save = save_with_party(rec, tmp_path=tmp_path, tables=tables, names=names)
    assert impossible_contents(save, tables, names, learnsets) == []


def test_nameless_items_warned(tmp_path, tables, names, learnsets):
    # No item is nameless in either game's tables today: blank one in a copy of ``names``.
    blank = 100
    blanked = copy.copy(names)
    blanked.item_names = tuple("" if i == blank else n for i, n in enumerate(names.item_names))
    out_of_range = len(names.item_names) + 5
    image = writer.build_image(
        party=[_party_mon()],
        boxes={(2, 4): writer.mon_record(species=1, level=5, moves=(33, 0, 0, 0), item=blank)},
        bag={"items": [(17, 1), (blank, 2), (out_of_range, 1)]},
    )
    save = load_save(_write(tmp_path, image), tables=tables, names=names)
    assert impossible_contents(save, tables, blanked, learnsets) == [
        f"box 3 slot 4 BULBASAUR: held item {blank} has no name",
        f"bag items slot 1: item {blank} has no name",
        f"bag items slot 2: item {out_of_range} has no name",
    ]


def _check_real_save(path, kaizo_tables, kaizo_names, kaizo_learnsets):
    save = load_save(path, game=KAIZO)
    assert save.load_result == "ok"
    for mon in save.party:  # every party mon converts
        if not mon.is_egg:
            mon.to_spec(kaizo_tables)
    for box in save.boxes:
        for mon in box:
            if mon is not None and not mon.is_egg:
                mon.to_spec(kaizo_tables)
    return save, impossible_contents(save, kaizo_tables, kaizo_names, kaizo_learnsets)


def test_maylene_save_kaizo(kaizo_tables, kaizo_names, kaizo_learnsets, kaizo_save_path):
    save, lines = _check_real_save(kaizo_save_path, kaizo_tables, kaizo_names, kaizo_learnsets)
    assert save.player_starter in _STARTERS
    assert lines == []


def test_pre_roark_save_kaizo(
    kaizo_tables, kaizo_names, kaizo_learnsets, kaizo_save_pre_roark_path
):
    save, lines = _check_real_save(
        kaizo_save_pre_roark_path, kaizo_tables, kaizo_names, kaizo_learnsets
    )
    assert save.player_starter in (0, *_STARTERS)
    assert save.party
    assert any(save.bag.values())  # "with all bag items available"
    assert lines == []
