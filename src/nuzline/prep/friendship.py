"""Battle-start friendship of a Gym Leader, Elite Four or Champion fight (design v5.3 sections
5 and 7, ledger E3-B3-5.a).

When the battle is a trainer battle and enemy 1's or enemy 2's trainer class is one of
``GYM_E4_CHAMPION_CLASSES``, ``BattleSys_New`` raises the friendship of every member of the
player's party before the battlers are built (src/battle/battle_main.c:1360-1372; classes
``TrainerIsGymLeaderE4OrChampion``, :1376-1397), so Return and Frustration in that fight read
the raised value. Per member that is ``Pokemon_UpdateFriendship(mon,
FRIENDSHIP_EVENT_BEAT_GYM_LEADER_E4_OR_CHAMPION, mapHeader)`` (src/pokemon.c:2635-2689):
:func:`battle_start_spec`. The partner party (``BATTLER_PLAYER_2``, battle_main.c:1368-1371)
gets the same raise; no tag or multi fight in the game is against these classes, so prep
applies it to party 0 only."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.levelup import FRIENDSHIP_TIERS, HOLD_EFFECT_FRIENDSHIP_UP, MAX_FRIENDSHIP

__all__ = [
    "BEAT_GYM_FRIENDSHIP",
    "GYM_E4_CHAMPION_CLASSES",
    "battle_start_friendship",
    "battle_start_party",
    "battle_start_spec",
    "starts_with_friendship",
]

# sFriendshipChangeTable[FRIENDSHIP_EVENT_BEAT_GYM_LEADER_E4_OR_CHAMPION] (src/pokemon.c:2626):
# the change by tier, tier 0 below 100, tier 1 from 100, tier 2 from 200 (:2652-2660;
# LOW_FRIENDSHIP_LIMIT and MED_FRIENDSHIP_LIMIT, include/constants/pokemon.h:29-30).
BEAT_GYM_FRIENDSHIP = (3, 2, 1)

# The 13 classes of TrainerIsGymLeaderE4OrChampion (src/battle/battle_main.c:1376-1397) by class
# id: line N of generated/trainer_classes.txt is id N-1 (LEADER_ROARK 62, LEADER_BYRON 64,
# ELITE_FOUR_AARON..LUCIAN 65-68, CHAMPION_CYNTHIA 69, LEADER_GARDENIA..VOLKNER 74-79). Kaizo
# keeps the class ids (Roark 246/858 is 62 in data/intermediate_kaizo/trainers.json).
GYM_E4_CHAMPION_CLASSES = frozenset({62, 64, 65, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79})


def starts_with_friendship(class_ids: Iterable[int | None]) -> bool:
    """True when one of the enemy trainers' class ids (``None``: no such enemy) is a Gym
    Leader, Elite Four member or Champion class (battle_main.c:1360-1362)."""
    return any(c in GYM_E4_CHAMPION_CLASSES for c in class_ids if c is not None)


def battle_start_friendship(friendship: int, *, luxury: bool, egg_match: bool, soothe: bool) -> int:
    """The friendship after ``Pokemon_UpdateFriendship`` for the Gym event (src/pokemon.c:
    2635-2689): the tier change of the value before it (:2652-2660), +1 for a Luxury Ball
    (:2664-2666), +1 when the egg location equals the map (:2668-2670), then ``* 150 / 100`` in
    integers for a holder of HOLD_EFFECT_FRIENDSHIP_UP (:2672-2676), clamped to 0..255
    (:2678-2686). Every change is positive, so each bonus applies."""
    low, mid = FRIENDSHIP_TIERS
    change = BEAT_GYM_FRIENDSHIP[(friendship >= low) + (friendship >= mid)]
    change += bool(luxury) + bool(egg_match)
    if soothe:
        change = change * 150 // 100
    return max(0, min(MAX_FRIENDSHIP, friendship + change))


def battle_start_spec(tables: Tables, spec: PokemonSpec, map_label: int | None) -> PokemonSpec:
    """``spec`` with its friendship after the battle-start raise. ``map_label`` is the map
    label text id of the fight's header, which is what ``Pokemon_UpdateFriendship`` receives as
    ``mapID`` (``battleSys->mapHeader = dto->mapLabelTextID``, battle_main.c:1054;
    ``dto->mapLabelTextID = MapHeader_GetMapLabelTextID(mapHeaderID)``,
    src/field_battle_data_transfer.c:263; not the header id). ``None``: not read. Then a member
    whose egg location is 0 gets no match (no Gym, Elite Four or Champion fight placement in the
    Kaizo headers has label 0: ledger E3-B3-5.a) and a member with a nonzero egg location is
    refused with ``ValueError``: whether it matches is unknown."""
    if map_label is None:
        if spec.egg_location:
            raise ValueError(
                f"egg location {spec.egg_location} may add 1 friendship at the start of a Gym, "
                "Elite Four or Champion fight (src/pokemon.c:2668-2670), and the fight's map "
                "label text id is not known: pass it"
            )
        egg_match = False
    else:
        egg_match = spec.egg_location == map_label
    soothe = int(np.asarray(tables.item_hold_effect)[spec.item]) == HOLD_EFFECT_FRIENDSHIP_UP
    friendship = battle_start_friendship(
        spec.friendship, luxury=bool(spec.friend_bonus), egg_match=egg_match, soothe=soothe
    )
    return dataclasses.replace(spec, friendship=friendship)


def battle_start_party(
    tables: Tables,
    specs: Sequence[PokemonSpec],
    class_ids: Iterable[int | None],
    map_label: int | None = None,
) -> list[PokemonSpec]:
    """``specs`` (party 0 in party order) as they stand when the battle's party copy is made
    against enemies of ``class_ids``: every member raised by :func:`battle_start_spec` when one
    class is a Gym Leader, Elite Four or Champion class, else unchanged."""
    if not starts_with_friendship(class_ids):
        return list(specs)
    return [battle_start_spec(tables, s, map_label) for s in specs]
