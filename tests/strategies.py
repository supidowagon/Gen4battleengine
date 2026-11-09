"""Hypothesis strategies for engine tests: random M1 Pokemon, battle states, legal
actions and chance outcomes.

Everything stays inside the M1 implemented set: species from ``M1_SPECIES``, moves
from their level-up learnsets with a supported effect (``move_effect !=
UNSUPPORTED``), their own abilities (all implemented), an implemented held item
(``M1_ITEMS``) or none;
the opponent's bag holds Potions or nothing in each slot.
State fields that no M1 mechanic can set are left at 0, except those the engines
read (screens, Tailwind, Trick Room, Mud Sport, weather), which are drawn so the
code reading them is exercised. Active Pokemon have HP >= 1; benched ones may have
fainted (plan ruling).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume
from hypothesis import strategies as st

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import IntermediateData
from nuzline.engine.actions import legal_actions, legal_opponent_actions
from nuzline.engine.coverage import halved_defense_line, low_defense
from nuzline.engine.ids import MOVE_STRUGGLE, AbilityId, EffectId, Stat, Status, Type, Weather
from nuzline.engine.outcome import MAX_HITS, Outcome
from nuzline.engine.state import (
    BAG_SLOTS,
    MAX_DEATH_BUDGET,
    SLOTS,
    STAGE_MAX,
    STAGE_MIN,
    BattleState,
    make_state,
)
from nuzline.engine.stats import calc_stats

INTERMEDIATE = Path(__file__).resolve().parents[1] / "data" / "intermediate"

# The M1 species set: the sample team plus Roark's and the route-encounter species, a
# holder of each weather ability (Swift Swim, Sand Veil, Snow Cloak), and of Rough Skin
# (Carvanha), Flame Body (Magby), Limber (Glameow), Adaptability (Eevee) and Solid Rock
# (Camerupt): their other slot is empty or unimplemented, so these are always drawn.
M1_SPECIES = (
    "Monferno",
    "Prinplup",
    "Grotle",
    "Starly",
    "Shinx",
    "Budew",
    "Geodude",
    "Onix",
    "Cranidos",
    "Zubat",
    "Bidoof",
    "Furret",
    "Staravia",
    "Spinarak",
    "Psyduck",
    "Venonat",
    "Kricketot",
    "Goldeen",
    "Sandshrew",
    "Swinub",
    "Carvanha",
    "Magby",
    "Glameow",
    "Eevee",
    "Camerupt",
)
M1_ITEMS = (
    "",
    "Oran Berry",
    "Sitrus Berry",
    "Focus Sash",
    "Big Root",
    "Berry Juice",
    "Chesto Berry",
    "Pecha Berry",
    "Lum Berry",
    "White Herb",
    "Lagging Tail",
)
M1_BAG_ITEMS = ("", "Potion")  # the opponent's bag (ids.TrainerItemId)

# decomp gender ratios (``tables.species_gender``) with a single possible gender
GENDERLESS, MALE_ONLY, FEMALE_ONLY = 255, 0, 254
_SLEEP_TURNS = (1, 5)  # BtlCmd_Random 3, 2 stores 2-5; a counter of 1 wakes next turn
_TOXIC_MAX = 15  # step.TOXIC_COUNTER_MAX
_CONFUSION_MAX, _TAUNT_MAX, _BIND_MAX = 5, 5, 6  # outcome.sample_outcome's duration ranges
_DURATION = (2, 6)  # outcome.duration over every move: 2-5 sleep/confusion, 3-6 Bind, 3-5 Taunt
_DISABLE_MAX = 6  # Disable stores RandNext() % 4 + 3
_TIMER_MAX = 5  # screens, Tailwind, Trick Room: turns left


@lru_cache(maxsize=1)
def _learnsets() -> dict[int, list[tuple[int, int]]]:
    if not (INTERMEDIATE / "learnsets.json").exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")
    return IntermediateData(INTERMEDIATE).learnsets


# The fixture and effect-test moves, given to every species so the infliction,
# trapping, flinch, multi-hit (2-5 and fixed two), stat-stage, Recover, Fake Out and Curse
# branches run in the differential test (no M1 species is Ghost-type: Curse is the
# non-Ghost one).
SHARED_MOVES = (
    "Stealth Rock",
    "Taunt",
    "Leech Seed",
    "Bind",
    "Pursuit",
    "Supersonic",
    "Headbutt",
    "Rock Polish",
    "Fury Swipes",
    "Thunder Wave",
    "Stun Spore",
    "Foresight",
    "Night Shade",
    "Seismic Toss",
    "Hidden Power",
    "Magnitude",
    "Confusion",
    "Mud Sport",
    "Bug Bite",
    "Disable",
    "Bide",
    "Rollout",
    "Ice Ball",
    "Defense Curl",
    "Recover",
    "Fake Out",
    "AncientPower",
    "Curse",
    "Double Kick",
    "Selfdestruct",
    "Flail",
    "Aqua Ring",
)


def m1_move_pools(tables: Tables, names: Names) -> dict[int, tuple[int, ...]]:
    """Species id -> its level-up learnset moves with a supported effect plus
    ``SHARED_MOVES``, in id order."""
    effect = np.asarray(tables.move_effect)
    learnsets = _learnsets()
    shared = {names.move(m) for m in SHARED_MOVES}
    unsupported = sorted(m for m in shared if effect[m] == EffectId.UNSUPPORTED)
    if unsupported:
        raise ValueError(f"SHARED_MOVES has unsupported moves: {unsupported}")
    pools = {}
    for name in M1_SPECIES:
        species = names.species(name)
        if Type.GHOST in np.asarray(tables.species_types[species]).tolist():
            # Curse is shared: a Ghost user takes the branch coverage refuses.
            raise ValueError(f"M1 species {name} is Ghost-type")
        moves = {m for _, m in learnsets[species] if effect[m] != EffectId.UNSUPPORTED}
        moves |= shared
        pools[species] = tuple(sorted(moves))
    return pools


def _max_hp(tables: Tables, species, ivs, evs, level, nature) -> int:
    stats = calc_stats(
        np.asarray(tables.species_base_stats[species]),
        np.asarray(ivs),
        np.asarray(evs),
        level,
        np.asarray(tables.nature_mods[nature]),
    )
    return int(stats[Stat.HP])


def possible_genders(ratio: int) -> tuple[int, ...]:
    """Genders (0 male, 1 female, 2 genderless) a species with this decomp ratio can have."""
    if ratio == GENDERLESS:
        return (2,)
    if ratio == MALE_ONLY:
        return (0,)
    if ratio == FEMALE_ONLY:
        return (1,)
    return (0, 1)


@st.composite
def supported_pokemon(draw, tables: Tables, names: Names) -> PokemonSpec:
    """A random M1 Pokemon: level 5-20, random nature, IVs, EVs (at most 510 in all),
    1-4 supported learnset moves, its own ability, an M1 item or none, any current HP
    including 0 (full HP drawn often: Focus Sash needs it), and a random status (none when
    fainted)."""
    pools = m1_move_pools(tables, names)
    species = draw(st.sampled_from(sorted(pools)))
    level = draw(st.integers(5, 20))
    nature = draw(st.integers(0, len(names.nature_names) - 1))
    ivs = tuple(draw(st.lists(st.integers(0, 31), min_size=6, max_size=6)))
    evs = tuple(draw(st.lists(st.integers(0, 85), min_size=6, max_size=6)))
    moves = draw(st.lists(st.sampled_from(pools[species]), min_size=1, max_size=4, unique=True))
    abilities = names.species_abilities(species)
    # a slot whose ability is not implemented (Run Away, Damp, ...) is never drawn
    supported = np.asarray(tables.species_abilities[species]) != AbilityId.UNSUPPORTED
    ability_slot = draw(st.sampled_from([i for i, a in enumerate(abilities) if a and supported[i]]))
    item = names.item(draw(st.sampled_from(M1_ITEMS)))
    gender = draw(st.sampled_from(possible_genders(int(tables.species_gender[species]))))
    max_hp = _max_hp(tables, species, ivs, evs, level, nature)
    hp = draw(st.one_of(st.just(max_hp), st.integers(0, max_hp)))
    status = draw(st.sampled_from(list(Status))) if hp > 0 else Status.NONE
    return PokemonSpec(
        species=species,
        level=level,
        nature=nature,
        ivs=ivs,
        evs=evs,
        moves=tuple(moves + [0] * (4 - len(moves))),
        item=item,
        ability_slot=ability_slot,
        gender=gender,
        friendship=draw(st.integers(0, 255)),
        current_hp=hp,
        status=int(status),
    )


def _status_counter(draw, status: int) -> int:
    if status == Status.SLEEP:
        return draw(st.integers(*_SLEEP_TURNS))
    if status == Status.TOXIC:
        return draw(st.integers(0, _TOXIC_MAX))
    return 0


@st.composite
def _party(draw, tables: Tables, names: Names) -> tuple[list[PokemonSpec], int]:
    """1-6 Pokemon and the active slot, whose Pokemon has HP >= 1."""
    party = draw(st.lists(supported_pokemon(tables, names), min_size=1, max_size=SLOTS))
    active = draw(st.integers(0, len(party) - 1))
    lead = party[active]
    if lead.current_hp == 0:
        max_hp = _max_hp(tables, lead.species, lead.ivs, lead.evs, lead.level, lead.nature)
        hp = draw(st.integers(1, max_hp))
        party[active] = replace(
            lead, current_hp=hp, status=int(draw(st.sampled_from(list(Status))))
        )
    return party, active


def _draw_lock(draw, a: dict, side: int, active: int, names: Names) -> None:
    """Maybe lock the active Pokemon into a Bide or Rollout / Ice Ball it knows (not while
    it sleeps: ``step`` unlocks a sleeper), with the matching hidden counters: Bide's
    turns left (1 or 2: the counter starts at 2 and the unleash turn finds 1; a locked Bide
    never rests at 0) and stored damage, Rollout's count (1-4)."""
    if a["status"][side, active] == Status.SLEEP or not draw(st.booleans()):
        return
    bide, rollout, ice_ball = (names.move(m) for m in ("Bide", "Rollout", "Ice Ball"))
    known = [int(m) for m in a["moves"][side, active] if int(m) in (bide, rollout, ice_ball)]
    if not known:
        return
    move = draw(st.sampled_from(known))
    a["locked_move"][side] = move
    if move == bide:
        a["bide_turns"][side] = draw(st.integers(1, 2))
        a["bide_damage"][side] = draw(st.integers(0, 2 * int(a["max_hp"][side, active])))
    else:
        a["rollout_count"][side] = draw(st.integers(1, 4))


@st.composite
def supported_state(draw, tables: Tables, names: Names) -> BattleState:
    """A mid-battle state: 1-6 Pokemon per side (``supported_pokemon``), any active
    slot, random PP, status counters, stat stages, confusion, Taunt, Bind, Leech Seed, Disable (a move of the active and its turns), Defense Curl, a Bide / Rollout lock (``_draw_lock``),
    Water Sport, Foresight, Stealth Rock, screens, Tailwind, Trick Room, Mud Sport, weather,
    the turn, each active's ``fake_out_turn`` (0 to ``turn + 1``), ``move_hit`` (none or
    one of the foe's active's moves) and ``last_move`` (none, one of its own moves or
    Struggle), and the opponent's bag (a size of 0-4 in
    ``trainer_item_count``, ``M1_BAG_ITEMS`` in each slot below it, gaps allowed as
    after a use) and the AI's knowledge of the player (any ``ai_seen_moves`` and
    ``ai_ability_known`` flags), the death record (``deaths`` = the fainted player slots), a
    faint budget 0-3 and a random set of living player slots as ``protected``, each within
    its legal range; the battle is not over."""
    parties = [draw(_party(tables, names)) for _ in (0, 1)]
    for side in (0, 1):  # coverage refuses a Selfdestruct facing a Defense below 8
        foe_defense = low_defense(tables, parties[1 - side][0])
        moves = [m for mon in parties[side][0] for m in mon.moves if m]
        assume(not any(halved_defense_line(tables, m, foe_defense) for m in moves))
    s = make_state(tables, parties[0][0], parties[1][0])
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    trap_moves = np.flatnonzero(np.asarray(tables.move_effect) == EffectId.TRAP)
    move_pp = np.asarray(tables.move_pp)
    turn = draw(st.integers(0, 50))
    for side, (party, active) in enumerate(parties):
        a["active"][side] = active
        for slot in range(len(party)):
            for k, move in enumerate(a["moves"][side, slot]):
                if move:
                    a["pp"][side, slot, k] = draw(st.integers(0, int(move_pp[move])))
            a["status_counter"][side, slot] = _status_counter(draw, int(a["status"][side, slot]))
        a["stages"][side, Stat.ATK :] = draw(
            st.lists(st.integers(STAGE_MIN, STAGE_MAX), min_size=7, max_size=7)
        )
        a["confusion"][side] = draw(st.integers(0, _CONFUSION_MAX))
        a["taunt"][side] = draw(st.integers(0, _TAUNT_MAX))
        trapped = draw(st.integers(0, _BIND_MAX))
        a["trapped"][side] = trapped
        a["trap_move"][side] = draw(st.sampled_from(trap_moves.tolist())) if trapped else 0
        a["leech_seed"][side] = draw(st.integers(0, 1))
        a["water_sport"][side] = draw(st.integers(0, 1))
        a["mud_sport"][side] = draw(st.integers(0, 1))
        a["aqua_ring"][side] = draw(st.integers(0, 1))
        a["curled"][side] = draw(st.integers(0, 1))
        a["foresight"][side] = draw(st.integers(0, 1))
        a["stealth_rock"][side] = draw(st.integers(0, 1))
        a["reflect"][side] = draw(st.integers(0, _TIMER_MAX))
        a["light_screen"][side] = draw(st.integers(0, _TIMER_MAX))
        a["tailwind"][side] = draw(st.integers(0, _TIMER_MAX))
        a["fake_out_turn"][side] = draw(st.integers(0, turn + 1))
        foe_moves = [int(m) for m in a["moves"][1 - side, parties[1 - side][1]] if m]
        a["move_hit"][side] = draw(st.sampled_from([0, *foe_moves]))
        own_moves = [int(m) for m in a["moves"][side, active] if m]
        a["last_move"][side] = draw(st.sampled_from([0, *own_moves, MOVE_STRUGGLE]))
        disabled = draw(st.sampled_from([0, *own_moves]))
        a["disabled_move"][side] = disabled
        turns = draw(st.integers(0, _DISABLE_MAX))
        a["disable_turns"][side] = turns if disabled else 0
        _draw_lock(draw, a, side, active, names)
    count = draw(st.integers(0, BAG_SLOTS))
    bag = draw(st.lists(st.sampled_from(M1_BAG_ITEMS), min_size=count, max_size=count))
    a["trainer_items"][1] = [names.item(i) for i in bag] + [0] * (BAG_SLOTS - count)
    a["trainer_item_count"][1] = count
    a["weather"][...] = draw(st.sampled_from(list(Weather)))  # permanent: timer 0
    a["trick_room"][...] = draw(st.integers(0, _TIMER_MAX))
    a["turn"][...] = turn
    for name in ("ai_seen_moves", "ai_ability_known"):
        flags = draw(st.lists(st.integers(0, 1), min_size=a[name].size, max_size=a[name].size))
        a[name][...] = np.reshape(flags, a[name].shape)
    fainted = (a["species"][0] > 0) & (a["hp"][0] == 0)
    a["player_faints"][...] = int(np.sum(fainted))
    a["deaths"][...] = fainted
    a["faint_budget"][...] = draw(st.integers(0, MAX_DEATH_BUDGET))
    living = (a["species"][0] > 0) & (a["hp"][0] > 0)
    a["protected"][...] = [int(draw(st.booleans())) if alive else 0 for alive in living]
    a["replace_mul"][...] = 0
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})


@st.composite
def legal_action_pair(draw, tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32[2]``: a legal action for each side: the player's from
    ``engine.actions.legal_actions``, the opponent's (moves, switches and bag items) from
    ``legal_opponent_actions``."""
    acts = []
    for mask in (legal_actions(tables, state, 0), legal_opponent_actions(tables, state)):
        legal = np.flatnonzero(np.asarray(mask))
        acts.append(draw(st.sampled_from(legal.tolist())))
    return jnp.asarray(acts, jnp.int32)


def _ints(lo: int, hi: int, shape: tuple[int, ...]) -> st.SearchStrategy[jnp.ndarray]:
    size = int(np.prod(shape, dtype=int))
    return st.lists(st.integers(lo, hi), min_size=size, max_size=size).map(
        lambda v: jnp.asarray(np.asarray(v, np.int8).reshape(shape))
    )


def _flag(shape: tuple[int, ...]) -> st.SearchStrategy[jnp.ndarray]:
    return _ints(0, 1, shape)


def any_outcome() -> st.SearchStrategy[Outcome]:
    """Every ``Outcome`` field uniform over its range."""
    hits, sides = (2, MAX_HITS), (2,)
    return st.builds(
        Outcome,
        speed_tie=_flag(()),
        hit=_flag(hits),
        crit=_flag(hits),
        roll=_ints(0, 15, hits),
        secondary=_flag(hits),
        contact_proc=_flag(hits),
        n_hits=_ints(2, 5, sides),
        full_para=_flag(sides),
        confusion_hit=_flag(sides),
        duration=_ints(*_DURATION, sides),
        thaw=_flag(sides),
        flinch_item=_flag(sides),
        quick_claw=_flag(sides),
        end_speed_tie=_flag(()),
        magnitude=_ints(4, 10, sides),
        shed_skin=_flag(sides),
    )


# Moves whose outcome laws the enumeration tests need (tests/engine/test_enumerate.py):
# sleep (Spore, Hypnosis), Taunt, Bind (TRAP), multi-hit (Fury Swipes), confusion
# (Supersonic, Confuse Ray), secondary paralysis / freeze / burn with a thawing move
# (Spark, Ice Beam, Flame Wheel), flinch (Headbutt), a Speed drop (Bubble), a Speed
# boost (Rock Polish), a high-crit move (Razor Leaf) and a 95 % plain move (Tackle).
LAW_MOVES = (
    "Spore",
    "Hypnosis",
    "Taunt",
    "Bind",
    "Fury Swipes",
    "Supersonic",
    "Confuse Ray",
    "Spark",
    "Ice Beam",
    "Flame Wheel",
    "Headbutt",
    "Thunder Wave",
    "Bubble",
    "Rock Polish",
    "Razor Leaf",
    "Tackle",
    "Magnitude",
    "Confusion",
    "Disable",
)
# Abilities the outcome relevance reads (contact procs, Steadfast, Simple's and Swift
# Swim's Speed) or the hit odds read (Compoundeyes, Sand Veil, Snow Cloak), and Rough Skin
# and Limber, which sit next to the contact procs and the paralysis they relate to.
LAW_ABILITIES = (
    AbilityId.STATIC,
    AbilityId.POISON_POINT,
    AbilityId.FLAME_BODY,
    AbilityId.ROUGH_SKIN,
    AbilityId.LIMBER,
    AbilityId.STEADFAST,
    AbilityId.SIMPLE,
    AbilityId.SHED_SKIN,
    AbilityId.COMPOUNDEYES,
    AbilityId.SAND_VEIL,
    AbilityId.SNOW_CLOAK,
    AbilityId.SWIFT_SWIM,
)


@st.composite
def outcome_law_state(draw, tables: Tables, names: Names) -> BattleState:
    """``supported_state`` widened for the outcome-law tests: Lucky Chant on either
    side (0-5 turns), and on each side, independently, maybe one of ``LAW_MOVES`` in
    the active Pokemon's slot 0 (full PP) and maybe one of ``LAW_ABILITIES`` on the
    active Pokemon. Only ``sample_outcome`` reads Lucky Chant."""
    s = draw(supported_state(tables, names))
    a = {name: np.asarray(getattr(s, name)).copy() for name in BattleState._fields}
    move_pp = np.asarray(tables.move_pp)
    for side in (0, 1):
        mon = int(a["active"][side])
        a["lucky_chant"][side] = draw(st.sampled_from([0, 0, 1, 5]))
        if draw(st.booleans()):
            move = names.move(draw(st.sampled_from(LAW_MOVES)))
            a["moves"][side, mon, 0] = move
            a["pp"][side, mon, 0] = move_pp[move]
        if draw(st.integers(0, 3)) == 0:
            a["ability"][side, mon] = draw(st.sampled_from(LAW_ABILITIES))
    return BattleState(**{name: jnp.asarray(a[name]) for name in BattleState._fields})
