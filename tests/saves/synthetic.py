"""A synthetic save file for the CLI and export tests: the sample party's species and
moves at levels whose stored stats are exactly ``calc_stats`` (the band accepts them)."""

from __future__ import annotations

import numpy as np

from nuzline.engine.stats import calc_stats
from tests.saves.writer import build_image, mon_record, personality_for

SAMPLE = (  # species, level, moves
    ("Monferno", 14, ("Scratch", "Leer", "Ember", "Mach Punch")),
    ("Prinplup", 14, ("Pound", "Growl", "Bubble", "Water Sport")),
    ("Grotle", 14, ("Tackle", "Withdraw", "Absorb", "Razor Leaf")),
)
IVS = (20,) * 6


def party_record(
    tables, names, species, level, moves, *, hp=None, status=0, egg=False, ability_slot=0, **kw
):
    sp = names.species(species)
    personality = personality_for(3, 5)
    stats = tuple(
        int(v)
        for v in calc_stats(
            np.asarray(tables.species_base_stats)[sp],
            IVS,
            (0,) * 6,
            level,
            np.asarray(tables.nature_mods)[3],
        )
    )
    ids = [names.move(m) for m in moves]
    ids += [0] * (4 - len(ids))
    pp = [int(np.asarray(tables.move_pp)[m]) if m else 0 for m in ids]
    return mon_record(
        species=sp,
        personality=personality,
        level=level,
        ivs=IVS,
        moves=tuple(ids),
        pp=tuple(pp),
        ability=int(tables.species_abilities_decomp[sp][ability_slot]),
        is_egg=egg,
        party={
            "status": status,
            "level": level,
            "hp": stats[0] if hp is None else hp,
            "stats": stats,
        },
        **kw,
    )


def box_record(tables, names, species, level, moves, *, ability_slot=0, **kw):
    """The same record as :func:`party_record` without the party block (a box slot)."""
    sp = names.species(species)
    ids = [names.move(m) for m in moves]
    ids += [0] * (4 - len(ids))
    pp = [int(np.asarray(tables.move_pp)[m]) if m else 0 for m in ids]
    return mon_record(
        species=sp,
        personality=personality_for(3, 5),
        level=level,
        ivs=IVS,
        moves=tuple(ids),
        pp=tuple(pp),
        ability=int(tables.species_abilities_decomp[sp][ability_slot]),
        **kw,
    )


def save_image(tables, names, *, egg_first=False, **kw) -> bytes:
    party = [party_record(tables, names, *row) for row in SAMPLE]
    if egg_first:
        party.insert(0, party_record(tables, names, "Starly", 5, ("Tackle",), egg=True))
    return build_image(party=party, **kw)


def write_save(tmp_path, tables, names, **kw):
    path = tmp_path / "game.sav"
    path.write_bytes(save_image(tables, names, **kw))
    return path
