import hashlib
import subprocess
import sys

import pytest

from nuzline.saves import RecordError, SaveError, SaveFile, load_save
from nuzline.saves.layout import BOXES_SIZE, NORMAL_SIZE, PARTY_COUNT
from nuzline.saves.save import DSV_MARKER, read_image, select_copies
from tests.saves.writer import build_image as _build_image
from tests.saves.writer import (
    corrupt,
    mon_record,
    personality_for,
    set_save_counter,
)


def _party_mon(species, level, stats):
    return mon_record(
        species=species,
        level=level,
        personality=personality_for(3, 5),
        party={"status": 0, "level": level, "hp": stats[0], "stats": stats},
    )


_STATS = (30, 10, 10, 30, 20, 25)  # Abra-ish stats; not checked by load_save


def build_image(**kw):
    """The writer's image with one party Pokemon by default (count 0 is refused)."""
    kw.setdefault("party", [_party_mon(63, 12, _STATS)])
    return _build_image(**kw)


def _write(tmp_path, image: bytes, name="game.sav"):
    p = tmp_path / name
    p.write_bytes(image)
    return p


def _load(tmp_path, image, tables, names, **kw) -> SaveFile:
    return load_save(_write(tmp_path, image), tables=tables, names=names, **kw)


def test_raw_and_dsv(tmp_path, tables, names):
    image = build_image(tid=4321)
    raw = _load(tmp_path, image, tables, names)
    dsv_bytes = image + DSV_MARKER + b"x" * 122
    dsv = load_save(_write(tmp_path, dsv_bytes, "game.dsv"), tables=tables, names=names)
    assert raw.tid == dsv.tid == 4321
    assert read_image(dsv_bytes) == image
    assert dsv.sha256 == hashlib.sha256(dsv_bytes).hexdigest()
    assert raw.sha256 == hashlib.sha256(image).hexdigest()


def test_wrong_size_raises(tmp_path, tables, names):
    with pytest.raises(SaveError, match="not a Platinum save: 262144 bytes"):
        load_save(_write(tmp_path, b"\0" * 0x40000), tables=tables, names=names)
    with pytest.raises(SaveError, match="not a Platinum save"):
        read_image(b"\0" * 0x40000 + DSV_MARKER + b"x" * 10)
    # A .dsv reports the size of its image, not of the whole file with the footer.
    with pytest.raises(SaveError, match="not a Platinum save: 4096 bytes"):
        read_image(b"\0" * 4096 + DSV_MARKER + b"x" * 122)


def test_party_boxes_bag_trainer(tmp_path, tables, names):
    inv = {v: k for k, v in names.charmap.items() if len(v) == 1}
    party = [
        _party_mon(names.species("Abra"), 12, (30, 10, 10, 30, 20, 25)),
        _party_mon(names.species("Starly"), 7, (22, 14, 11, 10, 10, 16)),
    ]
    boxes = {
        (0, 0): mon_record(species=names.species("Bidoof"), level=20),
        (17, 29): mon_record(species=names.species("Shinx"), level=33),
    }
    bag = {"items": [(17, 3), (26, 1)], "key_items": [], "tms_hms": [(328, 1)]}
    image = build_image(
        party=party,
        boxes=boxes,
        bag=bag,
        trainer_name_codes=tuple(inv[c] for c in "Lucas"),
        tid=12345,
        sid=54321,
        badges=0b101,
    )
    save = _load(tmp_path, image, tables, names)
    assert save.trainer_name == "Lucas"
    assert (save.tid, save.sid, save.badges) == (12345, 54321, 0b101)
    assert [m.species for m in save.party] == [names.species("Abra"), names.species("Starly")]
    assert [m.location for m in save.party] == [("party", 0, 0), ("party", 0, 1)]
    assert save.party[0].level == 12 and save.party[1].current_hp == 22
    assert len(save.boxes) == 18 and all(len(b) == 30 for b in save.boxes)
    assert save.boxes[0][0].species == names.species("Bidoof")
    assert save.boxes[0][0].level == 20 and save.boxes[0][0].location == ("box", 0, 0)
    assert save.boxes[17][29].species == names.species("Shinx")
    assert sum(m is not None for b in save.boxes for m in b) == 2
    assert save.bag["items"] == [(17, 3), (26, 1)]
    assert save.bag["tms_hms"] == [(328, 1)]
    assert list(save.bag) == [
        "items",
        "key_items",
        "tms_hms",
        "mail",
        "medicine",
        "berries",
        "poke_balls",
        "battle_items",
    ]
    assert all(save.bag[k] == [] for k in list(save.bag)[3:]) and save.bag["key_items"] == []
    assert save.load_result == "ok"
    assert (save.normal_copy, save.boxes_copy, save.save_counter) == ("primary", "primary", 1)


def test_bag_pocket_stops_at_first_empty_slot(tmp_path, tables, names):
    image = build_image(bag={"medicine": [(17, 2), (0, 5), (18, 1)]})
    assert _load(tmp_path, image, tables, names).bag["medicine"] == [(17, 2)]


def test_newer_primary_wins(tmp_path, tables, names):
    save = _load(tmp_path, build_image(counters=((6, 6, 6), (5, 5, 5))), tables, names)
    assert (save.normal_copy, save.boxes_copy, save.save_counter) == ("primary", "primary", 6)


def test_newer_backup_wins(tmp_path, tables, names):
    save = _load(tmp_path, build_image(counters=((5, 5, 5), (6, 6, 6))), tables, names)
    assert save.load_result == "ok"
    assert (save.normal_copy, save.boxes_copy, save.save_counter) == ("backup", "backup", 6)


def test_counter_wrap(tmp_path, tables, names):
    image = build_image(counters=((0xFFFFFFFF,) * 3, (0, 0, 0)))
    save = _load(tmp_path, image, tables, names)
    assert (save.normal_copy, save.boxes_copy, save.save_counter) == ("backup", "backup", 0)


def test_equal_global_counter_block_counter_decides():
    # boxes saved more recently in the backup copy within the same global save
    image = build_image(counters=((5, 5, 3), (5, 4, 4)))
    assert select_copies(image) == ("ok", 0, 1)


def test_one_copy_corrupt_is_ok(tmp_path, tables, names):
    image = corrupt(build_image(counters=((5, 5, 5), (6, 6, 6))), 1, "normal")
    assert select_copies(image) == ("corrupt", 0, 0)
    with pytest.raises(SaveError, match="corrupt"):
        _load(tmp_path, image, tables, names)
    save = _load(tmp_path, image, tables, names, allow_corrupt=True)
    assert save.load_result == "corrupt"
    assert (save.normal_copy, save.boxes_copy, save.save_counter) == ("primary", "primary", 5)


def test_older_copy_corrupt_is_still_corrupt():
    # SaveData_LoadCheck: a partially valid normal block is CORRUPT even when the valid
    # copy is the newest one.
    image = corrupt(build_image(counters=((5, 5, 5), (6, 6, 6))), 0, "normal")
    assert select_copies(image) == ("corrupt", 1, 1)


def test_older_boxes_copy_corrupt_is_ok():
    # ...but a partially valid boxes block whose newest copy matches the normal one is OK.
    image = corrupt(build_image(counters=((5, 5, 5), (6, 6, 6))), 0, "boxes")
    assert select_copies(image) == ("ok", 1, 1)


def test_mismatched_counters_corrupt(tmp_path, tables, names):
    # The backup's normal block was saved at global 6 but its boxes block says 5 with an
    # older block counter: the newest boxes block is the primary's (global 5), which
    # disagrees with the newest normal block (global 6).
    image = build_image(counters=((5, 5, 5), (6, 6, 4)))
    image = set_save_counter(image, 1, "boxes", 5)
    assert select_copies(image) == ("corrupt", 0, 0)
    save = _load(tmp_path, image, tables, names, allow_corrupt=True)
    assert save.load_result == "corrupt" and save.save_counter == 5


def test_partial_normal_matches_neither_box_copy_is_error():
    image = build_image(counters=((5, 5, 5), (6, 6, 6)))
    image = set_save_counter(image, 0, "boxes", 9)
    image = corrupt(image, 1, "normal")
    assert select_copies(image)[0] == "error"


def test_both_partial_on_the_same_copy_is_ok():
    # Saved once since the new game: the other copy is still erased (0xFF).
    image = build_image(counters=((1, 1, 1), None))
    assert select_copies(image) == ("ok", 0, 0)


def test_both_partial_on_different_copies_is_corrupt():
    image = build_image(counters=((5, 5, 5), (5, 5, 5)))
    image = corrupt(corrupt(image, 0, "boxes"), 1, "normal")
    assert select_copies(image) == ("corrupt", 0, 1)


def test_empty_and_error(tmp_path, tables, names):
    image = build_image(counters=(None, None))
    assert select_copies(image)[0] == "empty"
    with pytest.raises(SaveError, match="empty"):
        _load(tmp_path, image, tables, names)
    image = build_image(counters=((1, 1, 1), (2, 2, 2)))
    image = corrupt(corrupt(image, 0, "boxes"), 1, "boxes")
    assert select_copies(image)[0] == "error"
    with pytest.raises(SaveError, match="error"):
        _load(tmp_path, image, tables, names, allow_corrupt=True)


@pytest.mark.parametrize("count", [0, 7, -1])
def test_party_count_out_of_range(tmp_path, tables, names, count):
    image = build_image(party=[], party_count=count & 0xFFFFFFFF)
    with pytest.raises(SaveError, match="party count"):
        _load(tmp_path, image, tables, names)


def test_party_records_beyond_count_ignored(tmp_path, tables, names):
    stats = (30, 10, 10, 30, 20, 25)
    bad = bytearray(_party_mon(63, 12, stats))
    bad[0x20] ^= 1  # would raise RecordError if decoded
    image = build_image(party=[_party_mon(63, 12, stats), bytes(bad)], party_count=1)
    assert len(_load(tmp_path, image, tables, names).party) == 1
    assert PARTY_COUNT == 0x9C


def test_empty_slot_within_party_count_raises(tmp_path, tables, names):
    image = build_image(party=[_party_mon(63, 12, (30, 10, 10, 30, 20, 25))], party_count=2)
    with pytest.raises(SaveError, match="party slot 1"):
        _load(tmp_path, image, tables, names)


def test_bad_record_names_location(tmp_path, tables, names):
    rec = bytearray(mon_record(species=63, level=10))
    rec[0x20] ^= 1
    image = build_image(boxes={(4, 9): bytes(rec)})
    with pytest.raises(RecordError, match="box 5 slot 9"):
        _load(tmp_path, image, tables, names)


def test_default_tables_and_names(tmp_path, tables):
    # ``tables`` fixture only so the test skips when tables are not built
    save = load_save(_write(tmp_path, build_image(boxes={(0, 0): mon_record(species=63)})))
    assert save.boxes[0][0].species == 63


def test_import_saves_does_not_import_jax():
    code = "import sys, nuzline.saves; assert 'jax' not in sys.modules, 'jax imported'"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_layout_sizes_match_decomp_arithmetic():
    fields = 4 + 18 * 30 * 0x88 + 18 * 20 * 2 + 18 + 1  # PCBoxes members, 0x121C7
    sizeof = (fields + 3) // 4 * 4  # struct padded to its u32 alignment
    body = sizeof + (4 - sizeof % 4) + 4  # SaveTableEntry_BodySize
    assert BOXES_SIZE == body + 20
    assert NORMAL_SIZE == 0xCF2C


def test_default_tables_and_names_refuse_stale_tables(tmp_path):
    from nuzline.saves.save import default_tables_and_names

    (tmp_path / "manifest.json").write_text('{"version": 7}')
    with pytest.raises(ValueError, match="re-run tools/extract_decomp.py"):
        default_tables_and_names(None, None, tmp_path)
    with pytest.raises(ValueError, match="re-run tools/extract_decomp.py"):
        default_tables_and_names(object(), object(), tmp_path)


@pytest.mark.parametrize("version", [None, 7])
def test_default_tables_and_names_refuse_a_stale_npz(tmp_path, version):
    # The tracked manifest is current, the git-ignored npz is not (final review).
    import json

    import numpy as np

    from nuzline.data.tables import MANIFEST_VERSION
    from nuzline.saves.save import default_tables_and_names

    (tmp_path / "manifest.json").write_text(json.dumps({"version": MANIFEST_VERSION}))
    arrays = {"species_base_stats": np.zeros((496, 6), np.int16)}
    if version is not None:
        arrays["version"] = np.int16(version)
    np.savez_compressed(tmp_path / "tables.npz", **arrays)
    stale = r"tables\.npz is version \w+; .*re-run tools/extract_decomp\.py and tools/build_tables"
    with pytest.raises(ValueError, match=stale):
        default_tables_and_names(None, None, tmp_path)
