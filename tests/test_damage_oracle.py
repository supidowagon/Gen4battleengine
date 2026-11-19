"""Damage-array oracle: ``damage_table`` against @smogon/calc's Gen 4 calculator
(``tools/smogon_calc/damage.js``) on generated matchups of M1 species and supported
damaging moves, requiring all 16 rolls to be equal.

No items. Abilities are the species' own implemented ones (as ``tests.strategies``); the
damage-relevant ones (Blaze, Torrent, Overgrow, Rivalry, Simple, Unaware, Mold
Breaker) are exercised through low HP, genders and stat stages, Adaptability and Solid
Rock by the extra rows of ``_ability_rows``; the others have no damage effect in either
calculator. Matchups where @smogon/calc departs from the
decomp are not generated; each such case is logged in docs/discrepancies.md
("Damage oracle") with the decomp lines that settle it. A Cloud Nine battler in weather
is generated: both calculators drop the weather modifiers (``NO_CLOUD_NINE``,
battle_lib.c:6921, :7046; ``damage.weather_suppressed``).

Hidden Power, Magnitude, Trump Card, Wring Out and Low Kick take their power (and Hidden
Power its type) from the engine (``damage.hidden_power``; ``damage_table`` takes Magnitude at
level 4; ``damage.trump_card_power``, ``damage.hp_power``, ``damage.weight_power``), passed to
@smogon/calc as move overrides. A crit drawn
against an armored defender is dropped before either calculator sees it: the game's crit
roll never crits it (battle_lib.c:7134-7136, ``effects.armored``). Fixed-damage moves (Night Shade, Seismic Toss, Bide)
have no base power to compare: the tables carry a placeholder 1, @smogon/calc 0.

Skips when ``node`` or the npm package is missing (``npm ci --prefix tools/smogon_calc``);
fails instead when ``NUZLINE_REQUIRE_NODE=1``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.engine.damage import (
    MAGNITUDE_POWER,
    STATUS,
    _pre_random,
    chart_entry,
    damage_table,
    effectiveness,
    fixed_damage,
    hidden_power,
    hp_power,
    ignorable,
    trump_card_power,
    weather_ball_params,
    weight_power,
)
from nuzline.engine.effects import armored
from nuzline.engine.ids import (
    POWER_IF_TARGET_MOVED,
    AbilityId,
    EffectId,
    Stat,
    Status,
    Type,
    Weather,
)
from nuzline.engine.order import effective_speed
from nuzline.engine.state import make_state
from tests.strategies import m1_move_pools, possible_genders

ROOT = Path(__file__).resolve().parents[1]
SMOGON_DIR = ROOT / "tools" / "smogon_calc"
MATCHUPS = 300
ABILITY_ROWS = 60
SELF_DESTRUCT_ROWS = 4
SEED = 20261029

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")  # ids.Stat order
_BOOSTED = (Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD)
_WEATHER = {
    Weather.NONE: "",
    Weather.SUN: "Sun",
    Weather.RAIN: "Rain",
    Weather.SANDSTORM: "Sand",
    Weather.HAIL: "Hail",
}
_GENDER = {0: "M", 1: "F", 2: "N"}
_CATEGORY = {"Physical": 0, "Special": 1}
# No table power to compare: fixed damage, and Flail (the table's 1 against @smogon/calc's 0;
# both then take the power from the attacker's HP in 64ths, the same bands).
_NO_POWER = (EffectId.LEVEL_DAMAGE, EffectId.BIDE, EffectId.FLAIL)
# Effects whose damage @smogon/calc 0.12.0 computes against the decomp: never generated
# (docs/discrepancies.md, "Damage oracle").
_DEPARTED_EFFECTS = (
    # The game deals the defender's max HP (BtlCmd_TryOHKOMove, battle_script.c:4334-4386;
    # battle_controller_player.c:3343-3345); @smogon/calc gives power 0.
    EffectId.OHKO,
    # The use deals nothing: TryFutureSight only stores the damage (effect_script_0148.s;
    # battle_script.c:6065-6094); @smogon/calc hits at once.
    EffectId.FUTURE_SIGHT,
    # Power 1 + 25 x defender speed / attacker speed, at most 150 (BtlCmd_CalcGyroBallPower,
    # battle_script.c:7127-7129); @smogon/calc leaves out the 1 (gen4.js:326-328).
    EffectId.GYRO_BALL,
    # Super Fang deals Divide(defender curHP, 2) (effect_script_0040.s:5-8), Endeavor the
    # defender's curHP minus the attacker's (effect_script_0189.s:5-9), Counter twice the
    # physical damage taken this turn (effect_script_0089.s; BtlCmd_Counter,
    # battle_script.c:4615-4627); @smogon/calc has no fixed-damage case for them
    # (util.js:683-694) and deals 0 at power 0 (gen4.js:171-174).
    EffectId.HALVE_HP,
    EffectId.ENDEAVOR,
    EffectId.COUNTER,
)
# sWeightToPower (include/data/battle/weight_to_power.h): the first row whose weight
# (hectograms) is at least the defender's (BtlCmd_CalcWeightBasedPower,
# battle_script.c:6808-6826); @smogon/calc's Low Kick takes the next row at a bound
# (``w >= 10`` kg, gen4.js:320-325).
_WEIGHT_BOUNDS = (100, 250, 500, 1000, 2000)
# Every other damaging effect of the M1 move pools: compared (``_matchups``).
_COMPARED_EFFECTS = (
    EffectId.PLAIN, EffectId.SECONDARY_STAT_SELF, EffectId.SECONDARY_STAT_FOE,
    EffectId.SECONDARY_STATUS, EffectId.FLINCH, EffectId.HIGH_CRIT, EffectId.DRAIN,
    EffectId.TRAP, EffectId.PURSUIT, EffectId.MULTI_HIT, EffectId.RECOIL,
    EffectId.LEVEL_DAMAGE, EffectId.HIDDEN_POWER, EffectId.MAGNITUDE,
    EffectId.SECONDARY_CONFUSE, EffectId.EAT_BERRY, EffectId.BIDE, EffectId.ROLLOUT,
    EffectId.BRICK_BREAK, EffectId.FAKE_OUT, EffectId.RAISE_ALL_STATS_HIT,
    EffectId.SELF_DESTRUCT, EffectId.FLAIL, EffectId.FANG_PARALYZE_FLINCH,
    EffectId.DOUBLE_POWER, EffectId.HP_POWER, EffectId.WEIGHT_POWER, EffectId.HALVE_HP,
    EffectId.COUNTER, EffectId.UPROAR, EffectId.RECOIL_STATUS, EffectId.CRASH,
    EffectId.ENDEAVOR, EffectId.ASSURANCE, EffectId.TRUMP_CARD, EffectId.CHATTER,
)  # fmt: skip

jit_damage_table = jax.jit(damage_table)


@pytest.fixture(scope="module")
def node() -> str:
    missing = pytest.fail if os.environ.get("NUZLINE_REQUIRE_NODE") == "1" else pytest.skip
    exe = shutil.which("node")
    if exe is None:
        missing("node is not installed")
    if not (SMOGON_DIR / "node_modules" / "@smogon" / "calc").is_dir():
        missing("@smogon/calc is not installed (npm ci --prefix tools/smogon_calc)")
    return exe


def _mon(rng: np.random.Generator, tables: Tables, names: Names, species: int) -> PokemonSpec:
    abilities = names.species_abilities(species)
    # Only implemented abilities (as tests.strategies draws them): Venonat's Tinted Lens, say,
    # would double a resisted hit in @smogon/calc and nowhere in the engine.
    supported = np.asarray(tables.species_abilities[species]) != AbilityId.UNSUPPORTED
    return PokemonSpec(
        species=species,
        level=int(rng.integers(5, 21)),
        nature=int(rng.integers(0, len(names.nature_names))),
        ivs=tuple(int(x) for x in rng.integers(0, 32, 6)),
        evs=tuple(int(x) for x in rng.integers(0, 86, 6)),
        moves=(0, 0, 0, 0),
        item=0,
        ability_slot=int(rng.choice([i for i, a in enumerate(abilities) if a and supported[i]])),
        gender=int(rng.choice(possible_genders(int(tables.species_gender[species])))),
        friendship=70,
    )


def _smogon_departs(
    m: dict, tables: Tables, names: Names, ability: dict[int, int], move: int, s
) -> bool:
    """Matchups where @smogon/calc 0.12.0 departs from the decomp (docs/discrepancies.md,
    "Damage oracle"); the engine follows the decomp in all. ``s`` is the matchup's state."""
    physical = int(tables.move_category[move]) == 0
    if int(tables.move_effect[move]) in _DEPARTED_EFFECTS:
        return True
    a_stage = m["atk_stages"][0 if physical else 2]  # [atk, def, spa, spd]
    d_stage = m["def_stages"][1 if physical else 3]
    # 1. Sandstorm raises a Rock-type defender's SpD before its stage applies
    #    (battle_lib.c:6925, before the stage at :7008); @smogon/calc raises it after.
    rock = Type.ROCK in [int(t) for t in tables.species_types[m["def_species"]]]
    if m["weather"] == Weather.SANDSTORM and not physical and rock and d_stage != 0:
        return True
    # 2. A crit ignores the attacker's stage <= 0 and the defender's >= 0 after Simple
    #    doubled them (battle_lib.c:6850, :6868, then :6947-6973); @smogon/calc applies
    #    a Simple stage whatever the crit.
    a_simple = ability[0] == AbilityId.SIMPLE and ability[1] != AbilityId.UNAWARE
    d_simple = ability[1] == AbilityId.SIMPLE and ability[0] not in (
        AbilityId.MOLD_BREAKER,
        AbilityId.UNAWARE,
    )
    if m["crit"] and ((a_simple and a_stage < 0) or (d_simple and d_stage > 0)):
        return True
    # 3. Selfdestruct / Explosion halve the raw Defense, then its stage applies
    #    (battle_lib.c:6943-6945, then :6963-6975); @smogon/calc halves the staged Defense.
    halved = int(tables.move_effect[move]) == EffectId.SELF_DESTRUCT and physical
    if halved and d_stage != 0:
        return True
    # 4. Payback doubles when the defender has acted this turn (BtlCmd_CalcPaybackPower,
    #    battle_script.c:7198-7209), which ``damage_table`` never sees; @smogon/calc doubles
    #    whenever the attacker is not the faster (gen4.js:290, :330-335).
    payback = int(tables.move_effect[move]) == EffectId.DOUBLE_POWER and (
        int(tables.move_param[move, 0]) == POWER_IF_TARGET_MOVED
    )
    if payback and int(effective_speed(tables, s, 0)) <= int(effective_speed(tables, s, 1)):
        return True
    # 5. Low Kick against a defender whose weight is a row bound (_WEIGHT_BOUNDS).
    weight = int(tables.species_weight[m["def_species"]])
    if int(tables.move_effect[move]) == EffectId.WEIGHT_POWER and weight in _WEIGHT_BOUNDS:
        return True
    # 6. A defender's Dry Skin or Water Absorb takes a Water move with power, its Flash Fire a Fire
    #    move with power unless the holder is frozen, its Volt Absorb an Electric move (with or
    #    without power), unless the attacker's Mold Breaker hides it (BattleSystem_TriggerImmunity-
    #    Ability, battle_lib.c:3500-3521, :3538-3544): TRY_MOVE_STATE_TRIGGER_IMMUNITY_ABILITIES
    #    (battle_controller_player.c:3281-3286, :2789-2798) ends the move with
    #    MOVE_STATUS_NO_MORE_WORK before BattleSystem_CalcMoveDamage, which has no Water or
    #    Electric case for them (its one Dry Skin factor is Fire's, :6845-6848). ``damage_table``
    #    is that calculation; the absorb is ``step._use_move``'s. @smogon/calc returns 0.
    #    (Motor Drive and Soundproof are not modelled abilities: no ``AbilityId``.)
    seen = int(ignorable(s, 0, 1))
    powered = int(tables.move_power[move]) != 0
    # The game's moveType, which Weather Ball's script sets from the weather
    # (BtlCmd_CalcWeatherBallParams, battle_script.c:6843-6872) before TryMove reads it
    # (battle_lib.c:3492-3498), not the table's (Normalize is not modelled).
    kind = m["type"]
    if int(tables.move_effect[move]) == EffectId.WEATHER_BALL:
        kind = int(weather_ball_params(tables, s, move)[2])
    water = seen in (AbilityId.DRY_SKIN, AbilityId.WATER_ABSORB) and kind == Type.WATER
    flash = (
        seen == AbilityId.FLASH_FIRE
        and kind == Type.FIRE
        and int(s.status[1, int(s.active[1])]) != Status.FREEZE
    )
    volt = seen == AbilityId.VOLT_ABSORB and kind == Type.ELECTRIC
    if ((water or flash) and powered) or volt:
        return True
    # 7. The type chart rounding at low damage (_chart_rounding_departs).
    return _chart_rounding_departs(tables, names, s, move, m["crit"])


# The engine's damage before the variance roll (``damage._pre_random``) and the move type.
_pre_chart = jax.jit(lambda t, s, move, crit: _pre_random(t, s, 0, move, crit, def_side=1)[:2])


def _divide(x: int, d: int) -> int:
    """``BattleSystem_Divide`` (battle_lib.c:3599-3618) for non-negative ``x``."""
    return 0 if x == 0 else max(x // d, 1)


# @smogon/calc 0.12.0's order of a defender's two types (gen4.js:104-130).
_SMOGON_TYPE_ORDER = (
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON", "GROUND",
    "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL",
)  # fmt: skip


def _chart_rounding_departs(tables: Tables, names: Names, s, move: int, crit: bool) -> bool:
    """The decomp applies each defender type's chart row as ``BattleSystem_Divide(damage *
    mul, 10)`` (``ApplyTypeMultiplier``, battle_lib.c:7547), in chart order
    (``tables.type_chart_order``), and ``BattleSystem_Divide`` lifts a non-zero dividend
    that truncates to 0 up to 1 (battle_lib.c:3599-3618). @smogon/calc floors each factor
    (in its own type order) and takes ``max(1, damage)`` once at the end (gen4.js:228-237).
    So 1 x0.5 x2 is 2 in the decomp (1 -> 1 -> 2) and 1 in @smogon/calc (1 -> 0 -> 0 -> 1).
    Both are computed here from the engine's own pre-chart damage (``damage._pre_random``,
    the variance and STAB as ``damage._finish``), and the matchup is skipped only when they
    differ at some roll."""
    if bool(fixed_damage(tables, s, 0, move)[0]):
        return False
    pre, move_type = _pre_chart(tables, s, move, crit)
    pre, move_type = int(pre), int(move_type)
    a_types = {int(t) for t in s.types[0, int(s.active[0])]}
    t1, t2 = (int(t) for t in s.types[1, int(s.active[1])])
    defender = [t1] if t1 == t2 else [t1, t2]
    mul = {t: int(chart_entry(tables, move_type, t)) for t in defender}  # tenths
    decomp_order = sorted(defender, key=lambda t: int(tables.type_chart_order[move_type, t]))
    smogon_order = sorted(
        defender,
        key=lambda t: (
            _SMOGON_TYPE_ORDER.index(names.type_names[t])
            if names.type_names[t] in _SMOGON_TYPE_ORDER
            else -1
        ),
    )
    product = math.prod(mul.values())
    a_ability, d_ability = int(s.ability[0, 0]), int(s.ability[1, 0])
    solid = (
        d_ability == AbilityId.SOLID_ROCK
        and a_ability != AbilityId.MOLD_BREAKER
        and product > 10 ** len(defender)
    )
    for roll in range(16):
        base = max(pre * (85 + roll) // 100, 1)  # battle_lib.c:7084-7089
        if move_type in a_types:
            base = base * 2 if a_ability == AbilityId.ADAPTABILITY else base * 15 // 10
        ours = base
        for t in decomp_order:
            ours = _divide(ours * mul[t], 10)
        theirs = base
        for t in smogon_order:
            theirs = theirs * mul[t] // 10
        if solid:
            ours, theirs = _divide(ours * 3, 4), theirs * 3 // 4
        theirs = max(1, theirs) if product else 0
        if ours != theirs:
            return True
    return False


def _matchups(tables: Tables, names: Names) -> list[dict]:
    """``MATCHUPS`` seeded random matchups, each with its engine state and its
    @smogon/calc input."""
    rng = np.random.default_rng(SEED)
    pools = m1_move_pools(tables, names)
    damaging = {
        species: [m for m in moves if int(tables.move_category[m]) != STATUS]
        for species, moves in pools.items()
    }
    species_ids = sorted(pools)
    out: list[dict] = []
    while len(out) < MATCHUPS:
        atk_species, def_species = (int(x) for x in rng.choice(species_ids, 2))
        move = int(rng.choice(damaging[atk_species]))
        m = _matchup(rng, tables, names, atk_species, def_species, move)
        if m is not None:
            out.append(m)
    return out + _self_destruct_rows(tables, names)


def _self_destruct_rows(tables: Tables, names: Names) -> list[dict]:
    """``SELF_DESTRUCT_ROWS`` seeded Selfdestruct / Explosion matchups (their halved Defense),
    which the draw above can lose: a defender Defense stage other than 0 departs
    (``_smogon_departs`` 3), 12 draws in 13."""
    rng = np.random.default_rng(SEED + 2)
    pools = m1_move_pools(tables, names)
    users = [
        (species, m)
        for species, moves in sorted(pools.items())
        for m in moves
        if int(tables.move_effect[m]) == EffectId.SELF_DESTRUCT
    ]
    species_ids = sorted(pools)
    out: list[dict] = []
    while len(out) < SELF_DESTRUCT_ROWS:
        atk, move = (int(x) for x in users[int(rng.integers(len(users)))])
        m = _matchup(rng, tables, names, atk, int(rng.choice(species_ids)), move)
        if m is not None:
            out.append(m)
    return out


def _ability_rows(tables: Tables, names: Names) -> list[dict]:
    """``ABILITY_ROWS`` seeded matchups for Adaptability (Eevee's Normal-type moves) and
    Solid Rock (Camerupt hit by a super-effective move, some by a Mold Breaker Cranidos),
    which the random draw above rarely reaches. @smogon/calc floors ``damage * 0.75``
    (0.75 is exact in binary, so it is ``damage * 3 // 4``) where the decomp takes
    ``BattleSystem_Divide(damage * 3, 4)``; they differ only at damage 1 (0 against 1), and
    @smogon/calc's final ``max(1, damage)`` closes that gap, so no row is skipped."""
    rng = np.random.default_rng(SEED + 1)
    pools = m1_move_pools(tables, names)
    eevee, camerupt = names.species("Eevee"), names.species("Camerupt")
    stab = [m for m in pools[eevee] if _damaging(tables, m) and tables.move_type[m] == Type.NORMAL]
    cam_types = tables.species_types[camerupt]
    hitters = [
        (species, m)
        for species, moves in sorted(pools.items())
        for m in moves
        if _damaging(tables, m) and int(effectiveness(tables, tables.move_type[m], cam_types)) > 100
    ]
    cranidos = names.species("Cranidos")
    breakers = [h for h in hitters if h[0] == cranidos]
    out: list[dict] = []
    while len(out) < ABILITY_ROWS:
        if len(out) % 2 == 0:
            atk, move = eevee, int(rng.choice(stab))
            d = int(rng.choice(sorted(pools)))
        else:
            pick = breakers if len(out) % 4 == 3 else hitters
            atk, move = (int(x) for x in pick[int(rng.integers(len(pick)))])
            d = camerupt
        m = _matchup(rng, tables, names, atk, d, move)
        if m is not None:
            out.append(m)
    return out


def _damaging(tables: Tables, move: int) -> bool:
    """A damaging move with a table type and power (not Hidden Power's IV type, nor the
    placeholder power 1 of ``_NO_POWER``)."""
    effect = int(tables.move_effect[move])
    return (
        int(tables.move_category[move]) != STATUS
        and effect not in _NO_POWER
        and effect != EffectId.HIDDEN_POWER
    )


def _matchup(
    rng: np.random.Generator,
    tables: Tables,
    names: Names,
    atk_species: int,
    def_species: int,
    move: int,
) -> dict | None:
    """One seeded matchup with its engine state and @smogon/calc input, or None where
    @smogon/calc departs from the decomp (``_smogon_departs``)."""
    a, d = _mon(rng, tables, names, atk_species), _mon(rng, tables, names, def_species)
    m = {
        "def_species": def_species,
        "move": move,
        "crit": bool(rng.random() < 0.25),
        "weather": Weather(int(rng.choice(list(_WEATHER)))),
        "reflect": bool(rng.random() < 0.2),
        "light_screen": bool(rng.random() < 0.2),
        "burn": bool(rng.random() < 0.2),
        "atk_stages": [int(x) for x in rng.integers(-6, 7, 4)],
        "def_stages": [int(x) for x in rng.integers(-6, 7, 4)],
    }
    s = make_state(tables, [a], [d])
    # A third of attackers are at or below 1/3 HP (Blaze, Torrent, Overgrow).
    max_hp = int(s.max_hp[0, 0])
    hp = int(rng.integers(1, max_hp // 3 + 1 if rng.random() < 1 / 3 else max_hp + 1))
    m["hp"] = hp
    ability = {0: int(s.ability[0, 0]), 1: int(s.ability[1, 0])}
    m["power"], m["type"] = int(tables.move_power[move]), int(tables.move_type[move])
    effect = int(tables.move_effect[move])
    if effect == EffectId.HIDDEN_POWER:
        m["type"], m["power"] = (int(x) for x in hidden_power(np.asarray(a.ivs)))
    elif effect == EffectId.MAGNITUDE:
        m["power"] = int(MAGNITUDE_POWER[0])  # damage_table's default level 4
    elif effect == EffectId.TRUMP_CARD:
        m["power"] = _trump_card_power(s, move)
    elif effect == EffectId.HP_POWER:
        # The defender's HP is not changed below. BtlCmd_CalcWringOutPower
        # (battle_script.c:7261-7268) and @smogon/calc (gen4.js:359-362) agree on the formula.
        m["power"] = int(hp_power(s, 1))
    elif effect == EffectId.WEIGHT_POWER:
        m["power"] = int(weight_power(tables, s, 1))
    m["no_power"] = effect in _NO_POWER
    overrides = {}
    overridden = (
        EffectId.HIDDEN_POWER,
        EffectId.MAGNITUDE,
        EffectId.TRUMP_CARD,
        EffectId.HP_POWER,
        EffectId.WEIGHT_POWER,
    )
    if effect in overridden:
        overrides = {
            "basePower": m["power"],
            "type": names.type_names[m["type"]].capitalize(),
        }
    stages = np.zeros((2, 8), np.int8)  # overwrites a lead's Intimidate
    for side, key in ((0, "atk_stages"), (1, "def_stages")):
        stages[side, list(_BOOSTED)] = m[key]
    s = s.replace_hp(0, 0, hp)._replace(stages=s.stages.at[...].set(stages))
    if m["burn"]:
        s = s.with_status(0, 0, Status.BURN)
    s = s._replace(
        weather=s.weather.at[...].set(m["weather"]),
        reflect=s.reflect.at[1].set(int(m["reflect"])),
        light_screen=s.light_screen.at[1].set(int(m["light_screen"])),
    )
    m["state"] = s
    # The crit roll never crits an armored defender (battle_lib.c:7134-7136); the draw above
    # stays, so the sample does not shift.
    m["crit"] = m["crit"] and not bool(armored(s, 0, 1))
    if _smogon_departs(m, tables, names, ability, move, s):
        return None
    m["smogon"] = {
        "attacker": _smogon_mon(names, a, m["atk_stages"], hp, m["burn"]),
        "defender": _smogon_mon(names, d, m["def_stages"], None, False),
        "move": names.move_names[move],
        "overrides": overrides,
        "crit": m["crit"],
        "weather": _WEATHER[m["weather"]],
        "reflect": m["reflect"],
        "lightScreen": m["light_screen"],
    }
    return m


def _trump_card_power(s, move: int) -> int:
    """The engine's Trump Card power for side 0's lead (``damage._pre_random``): the PP of
    the first slot holding ``move`` after the use's decrement (BtlCmd_CalcTrumpCardPower,
    battle_script.c:7233-7245)."""
    moves = np.asarray(s.moves[0, 0])
    first = int(np.argmax(moves == move))
    return int(trump_card_power(max(int(s.pp[0, 0, first]) - 1, 0)))


def _smogon_mon(names: Names, mon: PokemonSpec, stages, hp, burn) -> dict:
    return {
        "species": names.species_names[mon.species],
        "level": mon.level,
        "nature": names.nature_names[mon.nature],
        "ivs": dict(zip(_STAT_KEYS, mon.ivs)),
        "evs": dict(zip(_STAT_KEYS, mon.evs)),
        "gender": _GENDER[mon.gender],
        "ability": names.ability_names[names.species_abilities(mon.species)[mon.ability_slot]],
        "boosts": dict(zip(("atk", "def", "spa", "spd"), stages)),
        "status": "brn" if burn else "",
        "curHP": hp,
    }


def _run_smogon(node: str, matchups: list[dict]) -> list[dict]:
    proc = subprocess.run(
        [node, str(SMOGON_DIR / "damage.js")],
        input=json.dumps([m["smogon"] for m in matchups]),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def oracle(tables, names, node):
    matchups = _matchups(tables, names)
    return matchups, _run_smogon(node, matchups)


def _describe(m: dict) -> str:
    return json.dumps(m["smogon"], sort_keys=True)


def test_every_drawn_effect_is_compared_or_departs(tables, names):
    """A new damaging effect in the M1 pools fails here, not as a reshuffled sample."""
    pools = m1_move_pools(tables, names)
    drawn = {
        int(tables.move_effect[m])
        for moves in pools.values()
        for m in moves
        if int(tables.move_category[m]) != STATUS
    }
    unlisted = drawn - set(_COMPARED_EFFECTS) - set(_DEPARTED_EFFECTS)
    assert not unlisted, (
        f"effects in the M1 pools neither compared nor departed: "
        f"{sorted(EffectId(e).name for e in unlisted)}; add each to _COMPARED_EFFECTS, or to "
        f"_DEPARTED_EFFECTS with its decomp cite and a docs/discrepancies.md entry"
    )


@pytest.mark.slow
def test_oracle_inputs_match_the_tables(tables, names, oracle):
    """Stats and move data agree, so a damage difference is a formula difference."""
    for m, r in zip(*oracle):
        s, move = m["state"], m["move"]
        for side, key in ((0, "attacker"), (1, "defender")):
            ours = [int(s.max_hp[side, 0])] + [int(x) for x in s.stats[side, 0]]
            assert ours == [r[key][k] for k in _STAT_KEYS], _describe(m)
        if not m["no_power"]:
            assert r["move"]["bp"] == m["power"], _describe(m)
        assert names.type(r["move"]["type"]) == m["type"], _describe(m)
        assert _CATEGORY[r["move"]["category"]] == int(tables.move_category[move])


@pytest.mark.slow
def test_damage_table_matches_smogon_calc(tables, names, oracle):
    matchups, results = oracle
    assert len(matchups) >= 200
    effects = {int(tables.move_effect[m["move"]]) for m in matchups}
    assert EffectId.SELF_DESTRUCT in effects  # Selfdestruct / Explosion's halved Defense
    assert EffectId.FLAIL in effects  # Flail / Reversal: @smogon/calc has the same 64ths bands
    mismatches = []
    for m, r in zip(matchups, results):
        ours = np.asarray(jit_damage_table(tables, m["state"], 0, m["move"], m["crit"])).tolist()
        if ours != r["damage"]:
            mismatches.append(f"{_describe(m)}\n  ours   {ours}\n  smogon {r['damage']}")
    assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])


@pytest.fixture(scope="module")
def ability_oracle(tables, names, node):
    rows = _ability_rows(tables, names)
    return rows, _run_smogon(node, rows)


@pytest.mark.slow
def test_adaptability_and_solid_rock_rows_match_smogon_calc(tables, names, ability_oracle):
    rows, results = ability_oracle
    abilities = {(int(m["state"].ability[0, 0]), int(m["state"].ability[1, 0])) for m in rows}
    assert any(a == AbilityId.ADAPTABILITY for a, _ in abilities)
    assert any(d == AbilityId.SOLID_ROCK for _, d in abilities)
    assert (AbilityId.MOLD_BREAKER, AbilityId.SOLID_ROCK) in abilities  # Cranidos: ignored
    mismatches = []
    for m, r in zip(rows, results):
        if not m["no_power"]:
            assert r["move"]["bp"] == m["power"], _describe(m)
        ours = np.asarray(jit_damage_table(tables, m["state"], 0, m["move"], m["crit"])).tolist()
        if ours != r["damage"]:
            mismatches.append(f"{_describe(m)}\n  ours   {ours}\n  smogon {r['damage']}")
    assert not mismatches, f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:10])
