import json
import tomllib

from nuzline.data.party import load_party
from nuzline.saves import load_save
from nuzline.saves.export import party_toml, save_json, show_lines
from tests.saves.synthetic import SAMPLE, party_record, save_image
from tests.saves.writer import build_image


def _save(tmp_path, tables, names, **kw):
    p = tmp_path / "g.sav"
    p.write_bytes(save_image(tables, names, **kw))
    return load_save(p, tables=tables, names=names)


def test_party_toml_loads_back_to_the_specs(tmp_path, tables, names):
    save = _save(tmp_path, tables, names)
    out = tmp_path / "party.toml"
    out.write_text(party_toml(save, names, tables), encoding="utf-8")
    assert load_party(out, names) == [m.to_spec(tables, save.trainer) for m in save.party]


def test_party_toml_status_pp_and_eggs(tmp_path, tables, names):
    party = [
        party_record(tables, names, "Starly", 5, ("Tackle",), egg=True),
        party_record(tables, names, "Monferno", 14, SAMPLE[0][2], status=1 << 4, hp=7),
        party_record(tables, names, "Grotle", 14, ("Tackle", "Absorb"), status=3),
    ]
    p = tmp_path / "g.sav"
    p.write_bytes(build_image(party=party))
    save = load_save(p, tables=tables, names=names)
    text = party_toml(save, names, tables)
    entries = tomllib.loads(text)["pokemon"]
    assert [e["species"] for e in entries] == ["MONFERNO", "GROTLE"]
    assert entries[0]["status"] == "burn" and entries[0]["hp"] == 7
    assert entries[1]["status"] == "sleep" and entries[1]["sleep_turns"] == 3
    assert entries[1]["moves"] == ["Tackle", "Absorb"] and len(entries[1]["pp"]) == 2
    out = tmp_path / "party.toml"
    out.write_text(text, encoding="utf-8")
    assert load_party(out, names) == [
        m.to_spec(tables, save.trainer) for m in save.party if not m.is_egg
    ]


def test_show_lines_ascii_and_json(tmp_path, tables, names):
    save = _save(tmp_path, tables, names)
    lines = show_lines(save, names)
    assert all(line.isascii() for line in lines)
    assert any("MONFERNO Lv14" in line for line in lines)
    data = save_json(save, names)
    text = json.dumps(data)
    assert json.loads(text)["party"][0]["species_name"] == "MONFERNO"
    assert data["boxes"][0][0] is None and data["bag"]["items"] == []


def test_non_ascii_nickname_is_escaped_in_json_and_lines(tmp_path, tables, names):
    code = next(c for c, ch in names.charmap.items() if len(ch) == 1 and not ch.isascii())
    sp = names.species("Monferno")
    from tests.saves.writer import mon_record

    box = mon_record(species=sp, level=20, nickname_codes=(code,))
    p = tmp_path / "n.sav"
    party = [party_record(tables, names, *SAMPLE[0])]
    p.write_bytes(build_image(party=party, boxes={(0, 0): box}))
    save = load_save(p, tables=tables, names=names)
    assert save.boxes[0][0].nickname == names.charmap[code]
    assert all(line.isascii() for line in show_lines(save, names))
    assert json.dumps(save_json(save, names)).isascii()


def test_empty_boxes_are_not_listed(tmp_path, tables, names):
    lines = show_lines(_save(tmp_path, tables, names), names)
    assert not any(line.startswith("Box") for line in lines)


def test_show_lines_form_and_nickname(tmp_path, tables, names):
    inv = {v: k for k, v in names.charmap.items() if len(v) == 1}
    party = [
        party_record(
            tables, names, *SAMPLE[0], form=1, nickname_codes=tuple(inv[c] for c in "Blaze")
        ),
        party_record(tables, names, *SAMPLE[1], nickname_codes=tuple(inv[c] for c in "PRINPLUP")),
    ]
    p = tmp_path / "f.sav"
    p.write_bytes(build_image(party=party))
    lines = show_lines(load_save(p, tables=tables, names=names), names)
    assert 'MONFERNO Lv14 | form 1 | "Blaze" |' in lines[3]
    assert "form" not in lines[4] and '"' not in lines[4]  # same name as the species


def test_export_round_trips_a_form(tmp_path, tables, names):
    # The brief's ``export_party([spec], names)`` is ``party_toml(save, names, tables)``,
    # and its ``_write`` is the ``out.write_text`` of the tests above.
    import numpy as np

    from nuzline.engine.stats import calc_stats
    from tests.saves.synthetic import IVS
    from tests.saves.writer import mon_record, personality_for

    sp, form, row, level = 479, 2, 504, 20
    stats = tuple(
        int(v)
        for v in calc_stats(
            np.asarray(tables.species_base_stats)[row],
            IVS,
            (0,) * 6,
            level,
            np.asarray(tables.nature_mods)[3],
            species=sp,
        )
    )
    rec = mon_record(
        species=sp,
        form=form,
        personality=personality_for(3, 5),
        level=level,
        ivs=IVS,
        moves=(names.move("Astonish"), 0, 0, 0),
        pp=(int(np.asarray(tables.move_pp)[names.move("Astonish")]), 0, 0, 0),
        ability=names.ability("Levitate"),
        party={"status": 0, "level": level, "hp": stats[0], "stats": stats},
    )
    p = tmp_path / "g.sav"
    p.write_bytes(build_image(party=[rec]))
    save = load_save(p, tables=tables, names=names)
    text = party_toml(save, names, tables)
    assert 'form = "Wash"' in text
    out = tmp_path / "party.toml"
    out.write_text(text, encoding="utf-8")
    assert load_party(out, names) == [save.party[0].to_spec(tables, save.trainer)]
    # A base form writes no form key.
    assert "form" not in party_toml(_save(tmp_path, tables, names), names, tables)


def test_export_writes_a_cosmetic_form_as_its_number(tmp_path, tables, names):
    rec = party_record(tables, names, "Unown", 20, ("Hidden Power",), form=5)
    p = tmp_path / "u.sav"
    p.write_bytes(build_image(party=[rec]))
    save = load_save(p, tables=tables, names=names)
    text = party_toml(save, names, tables)
    assert "\nform = 5\n" in text
    out = tmp_path / "party.toml"
    out.write_text(text, encoding="utf-8")
    assert load_party(out, names) == [save.party[0].to_spec(tables, save.trainer)]


def test_party_toml_keeps_the_exp_fields(tmp_path, tables, names):
    # K3d-exp T3: exp (off the level's base), a resolved non-OT boost, Pokerus and the
    # Luxury Ball survive party_toml -> load_party.
    from nuzline.saves.pkm import ITEM_LUXURY_BALL

    sp = names.species("Monferno")
    exp = int(tables.exp_table[int(tables.species_exp_rate[sp]), 14]) + 7
    rec = party_record(
        tables,
        names,
        "Monferno",
        14,
        SAMPLE[0][2],
        exp=exp,
        ot_id=99,
        language=1,
        pokerus=0x10,
        pokeball=ITEM_LUXURY_BALL,
    )
    p = tmp_path / "g.sav"
    p.write_bytes(build_image(party=[rec]))
    save = load_save(p, tables=tables, names=names)
    out = tmp_path / "party.toml"
    out.write_text(party_toml(save, names, tables), encoding="utf-8")
    (spec,) = load_party(out, names)
    assert (spec.exp, spec.exp_boost, spec.pokerus, spec.friend_bonus) == (exp, 170, 1, 1)
    assert [spec] == [save.party[0].to_spec(tables, save.trainer)]


def test_party_toml_keeps_egg_location_and_pp_ups(tmp_path, tables, names):
    """T7 review M5: the coverage inputs ``egg_location`` and ``pp_ups`` round-trip."""
    party = [
        party_record(
            tables,
            names,
            "Grotle",
            14,
            ("Tackle", "Absorb"),
            pp_ups=(1, 0, 2, 3),
            egg_location_dp=7,
        )
    ]
    p = tmp_path / "g.sav"
    p.write_bytes(build_image(party=party))
    save = load_save(p, tables=tables, names=names)
    text = party_toml(save, names, tables)
    entry = tomllib.loads(text)["pokemon"][0]
    assert entry["egg_location"] == 7 and entry["pp_ups"] == [1, 0, 2, 3]
    out = tmp_path / "party.toml"
    out.write_text(text, encoding="utf-8")
    (spec,) = load_party(out, names)
    assert (spec.egg_location, spec.pp_ups) == (7, (1, 0, 2, 3))
    assert [spec] == [m.to_spec(tables, save.trainer) for m in save.party]
