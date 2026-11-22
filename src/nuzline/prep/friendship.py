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
applies it to party 0 only.

The egg location bonus compares with the fight's map label. Where the trainer's headers fix it
(:func:`fight_label`) the raise has one value. Where they do not, prep carries both bounds
(:func:`battle_start_bounds`) and each reader takes the one that hurts it
(:func:`fight_friendship`): Return and friendship evolution the low bound, Frustration the
high one. Kaizo's Return and Frustration ignore friendship (docs/kaizo/rom_diff.md:269)."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Sequence

import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Tables
from nuzline.engine.coverage import FRIENDSHIP_POWER_EFFECTS, learnable_moves
from nuzline.engine.ids import EffectId
from nuzline.engine.levelup import FRIENDSHIP_TIERS, HOLD_EFFECT_FRIENDSHIP_UP, MAX_FRIENDSHIP

__all__ = [
    "BEAT_GYM_FRIENDSHIP",
    "GYM_E4_CHAMPION_CLASSES",
    "battle_start_bounds",
    "battle_start_friendship",
    "battle_start_party",
    "battle_start_spec",
    "egg_match_bounds",
    "fight_friendship",
    "fight_label",
    "friendship_readers",
    "starts_with_friendship",
]

# The decomp battle effects of Return (BATTLE_EFFECT_POWER_BASED_ON_FRIENDSHIP, 121) and
# Frustration (..._LOW_FRIENDSHIP, 123) as ``tables.move_ai_effect`` holds them (``coverage``).
RETURN_EFFECT, FRUSTRATION_EFFECT = FRIENDSHIP_POWER_EFFECTS
FRIENDSHIP_POWER = frozenset(FRIENDSHIP_POWER_EFFECTS)
# The moves that run a move the fight holds: Mimic and Transform copy one into the user's
# moves, Mirror Move, Copycat and Me First run one just used, Assist one of the party's.
_COPIERS = frozenset(
    {
        EffectId.MIMIC,
        EffectId.TRANSFORM,
        EffectId.MIRROR_MOVE,
        EffectId.COPYCAT,
        EffectId.ME_FIRST,
        EffectId.ASSIST,
    }
)

# sFriendshipChangeTable[FRIENDSHIP_EVENT_BEAT_GYM_LEADER_E4_OR_CHAMPION] (src/pokemon.c:2626):
# the change by tier, tier 0 below 100, tier 1 from 100, tier 2 from 200 (:2652-2660;
# LOW_FRIENDSHIP_LIMIT and MED_FRIENDSHIP_LIMIT, include/constants/pokemon.h:29-30).
BEAT_GYM_FRIENDSHIP = (3, 2, 1)

# The 13 classes of TrainerIsGymLeaderE4OrChampion (src/battle/battle_main.c:1376-1397) by class
# id: line N of generated/trainer_classes.txt is id N-1 (LEADER_ROARK 62, LEADER_BYRON 64,
# ELITE_FOUR_AARON..LUCIAN 65-68, CHAMPION_CYNTHIA 69, LEADER_GARDENIA..VOLKNER 74-79). Both
# ROMs' compiled switch returns TRUE for exactly these (``code_tables.gym_e4_champion_classes``,
# tests/tools/rom/test_code_tables.py).
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


def fight_label(map_labels: Sequence[int]) -> int | None:
    """The map label of the fight when its header fixes it, else None. ``mapID`` in the egg
    location test is the battle header's label (``battleSys->mapHeader = dto->mapLabelTextID``,
    battle_main.c:1054; ``MapHeader_GetMapLabelTextID``, src/field_battle_data_transfer.c:263),
    so the location is fixed when every header that places the trainer has the same label
    (``Trainer.map_labels``, the sorted labels of the trainer -> map join, holds one value).
    Several labels (a trainer on several headers, Kaizo Byron 250: 7 and 119) or none (no map
    places the trainer, so any header may hold the fight) leave it open."""
    labels = set(map_labels)
    return next(iter(labels)) if len(labels) == 1 else None


def egg_match_bounds(egg_location: int, map_labels: Sequence[int]) -> tuple[bool, bool]:
    """``(low, high)``: whether ``Pokemon_UpdateFriendship``'s egg location test
    (``MON_DATA_EGG_LOCATION == mapID``, src/pokemon.c:2668-2670) passes, at its least and at
    its most over the fight's possible labels. A fixed label (:func:`fight_label`) gives one
    answer twice. An open one never passes at least, and may pass when the egg location is one
    of the labels or no label is known. A Pokemon not hatched has egg location 0, which matches
    a header with label 0 (Mystery Zone)."""
    label = fight_label(map_labels)
    if label is not None:
        return egg_location == label, egg_location == label
    return False, not map_labels or egg_location in map_labels


def battle_start_bounds(
    tables: Tables, spec: PokemonSpec, map_labels: Sequence[int]
) -> tuple[int, int]:
    """``(low, high)``: ``spec``'s friendship after the battle-start raise without and with the
    egg location bonus where the fight's label is open (:func:`egg_match_bounds`); equal where
    it is fixed. The bonus adds 1 to a change that is always positive for this event (:2626)
    before Soothe Bell's x150/100 (:2672-2676), so ``low <= high``."""
    soothe = int(np.asarray(tables.item_hold_effect)[spec.item]) == HOLD_EFFECT_FRIENDSHIP_UP
    luxury = bool(spec.friend_bonus)
    low, high = (
        battle_start_friendship(spec.friendship, luxury=luxury, egg_match=m, soothe=soothe)
        for m in egg_match_bounds(spec.egg_location, map_labels)
    )
    return low, high


def friendship_readers(
    tables: Tables, moves: Iterable[int], fight_moves: Iterable[int]
) -> frozenset[int]:
    """The friendship-power effects (``RETURN_EFFECT``, ``FRUSTRATION_EFFECT``) a Pokemon with
    ``moves`` (its own and those it can learn in the fight) can use in a fight whose moves, both
    sides, are ``fight_moves``. They are the fight's only readers of its friendship: Return's
    power is ``friendship * 10 / 25`` (res/battle/scripts/effects/effect_script_0121.s:5-7),
    Frustration's ``(255 - friendship) * 10 / 25`` (effect_script_0123.s:5-8), both from
    ``BATTLEMON_FRIENDSHIP`` (src/battle/battle_lib.c:398-399, copied from the party at :136
    and :179); the trainer AI's damage estimate reads the same value
    (trainer_ai.c:3018-3026). Metronome can call either; a move that copies or runs a move of
    the fight (``_COPIERS``) reaches those of ``fight_moves``. Kaizo's Return and Frustration
    have effect 0 (docs/kaizo/rom_diff.md:269), so a Kaizo fight has no reader."""
    ai_effect = np.asarray(tables.move_ai_effect)
    effect = np.asarray(tables.move_effect)
    own = [int(m) for m in moves if m]
    kinds = {int(effect[m]) for m in own}
    if EffectId.METRONOME in kinds:
        return FRIENDSHIP_POWER
    reached = {int(ai_effect[m]) for m in own}
    if kinds & _COPIERS:
        reached |= {int(ai_effect[m]) for m in fight_moves if m}
    return frozenset(reached) & FRIENDSHIP_POWER


def _power(friendship: int, effect: int) -> int:
    """Return's or Frustration's power at ``friendship`` (effect_script_0121.s, 0123.s)."""
    if effect == RETURN_EFFECT:
        return friendship * 10 // 25
    return (MAX_FRIENDSHIP - friendship) * 10 // 25


def fight_friendship(low: int, high: int, readers: Collection[int]) -> int:
    """The friendship the fight plays with, of the two bounds: the one that hurts its
    ``readers`` (:func:`friendship_readers`). Return's power rises with friendship, so a Return
    user takes ``low``; Frustration's falls, so a Frustration user takes ``high``. A Pokemon that
    can use both takes the bound where the stronger of the two is weaker. Ties and no reader:
    ``low``, the bound friendship evolution reads (``EVO_LEVEL_HAPPINESS*`` need
    ``EVOLVE_FRIENDSHIP_THRESHOLD <= friendship``, src/pokemon.c:3565-3583,
    include/constants/pokemon.h:25; an evolution can always be cancelled, so less friendship
    offers fewer options)."""
    if not readers:
        return low

    def strongest(friendship: int) -> int:
        return max(_power(friendship, e) for e in readers)

    return high if strongest(high) < strongest(low) else low


def battle_start_spec(
    tables: Tables,
    spec: PokemonSpec,
    map_labels: Sequence[int],
    readers: Collection[int] = (),
) -> PokemonSpec:
    """``spec`` with its friendship after the battle-start raise in a fight on a header with one
    of ``map_labels``: the one value where the label is fixed (:func:`fight_label`), else the
    bound of :func:`battle_start_bounds` its fight ``readers`` take (:func:`fight_friendship`)."""
    low, high = battle_start_bounds(tables, spec, map_labels)
    return dataclasses.replace(spec, friendship=fight_friendship(low, high, readers))


def battle_start_party(
    tables: Tables,
    specs: Sequence[PokemonSpec],
    class_ids: Iterable[int | None],
    map_labels: Sequence[int],
    opp: Sequence[PokemonSpec],
) -> list[PokemonSpec]:
    """``specs`` (party 0 in party order) as they stand when the battle's party copy is made
    against ``opp``, enemies of ``class_ids``, on a header with one of ``map_labels`` (enemy 1's
    ``Trainer.map_labels``): every member raised by :func:`battle_start_spec` when one class is
    a Gym Leader, Elite Four or Champion class, else unchanged. A member's readers come from its
    moves and the moves it can learn in the fight (``coverage.learnable_moves``), against every
    move of the fight (both sides, learnable moves included)."""
    if not starts_with_friendship(class_ids):
        return list(specs)
    learn = [learnable_moves(tables, s, opp) for s in specs]
    fight = [m for s in (*specs, *opp) for m in s.moves] + [m for ms in learn for m in ms]
    return [
        battle_start_spec(tables, s, map_labels, friendship_readers(tables, (*s.moves, *ms), fight))
        for s, ms in zip(specs, learn, strict=True)
    ]
