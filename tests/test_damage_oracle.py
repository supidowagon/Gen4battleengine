"""Damage-array oracle: ``damage_table`` against @smogon/calc's Gen 4 calculator
(``tools/smogon_calc/damage.js``) on generated matchups of M1 species and supported
damaging moves, requiring all 16 rolls to be equal.

No items. Abilities are the species' own implemented ones (as ``tests.strategies``); the
damage-relevant ones (Blaze, Torrent, Overgrow, Rivalry, Simple, Unaware, Mold
Breaker) are exercised through low HP, genders and stat stages, the others have no
damage effect in either calculator. Matchups where @smogon/calc departs from the
decomp are not generated; each such case is logged in docs/discrepancies.md
("Damage oracle") with the decomp lines that settle it. A Cloud Nine battler in weather
is generated: both calculators drop the weather modifiers (``NO_CLOUD_NINE``,
battle_lib.c:6921, :7046; ``damage.weather_suppressed``).

Hidden Power and Magnitude take their power (and Hidden Power its type) from the
engine (``damage.hidden_power``; ``damage_table`` takes Magnitude at level 4), passed
to @smogon/calc as move overrides. Fixed-damage moves (Night Shade, Seismic Toss, Bide)
have no base power to compare: the tables carry a placeholder 1, @smogon/calc 0.

Skips when ``node`` or the npm package is missing
(``npm install --prefix tools/smogon_calc``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.engine.damage import MAGNITUDE_POWER, STATUS, damage_table, hidden_power
from nuzline.engine.ids import AbilityId, EffectId, Stat, Status, Type, Weather
from nuzline.engine.state import make_state
from tests.strategies import m1_move_pools, possible_genders

ROOT = Path(__file__).resolve().parents[1]
SMOGON_DIR = ROOT / "tools" / "smogon_calc"
MATCHUPS = 300
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

jit_damage_table = jax.jit(damage_table)


@pytest.fixture(scope="module")
def node() -> str:
    exe = shutil.which("node")
    if exe is None:
        pytest.skip("node is not installed")
    if not (SMOGON_DIR / "node_modules" / "@smogon" / "calc").is_dir():
        pytest.skip("@smogon/calc is not installed (npm install --prefix tools/smogon_calc)")
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


def _smogon_departs(m: dict, tables: Tables, ability: dict[int, int], move: int) -> bool:
    """Matchups where @smogon/calc 0.12.0 departs from the decomp (docs/discrepancies.md,
    "Damage oracle"); the engine follows the decomp in both."""
    physical = int(tables.move_category[move]) == 0
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
    return halved and d_stage != 0


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
        if _smogon_departs(m, tables, ability, move):
            continue
        m["power"], m["type"] = int(tables.move_power[move]), int(tables.move_type[move])
        effect = int(tables.move_effect[move])
        if effect == EffectId.HIDDEN_POWER:
            m["type"], m["power"] = (int(x) for x in hidden_power(np.asarray(a.ivs)))
        elif effect == EffectId.MAGNITUDE:
            m["power"] = int(MAGNITUDE_POWER[0])  # damage_table's default level 4
        m["no_power"] = effect in _NO_POWER
        overrides = {}
        if effect in (EffectId.HIDDEN_POWER, EffectId.MAGNITUDE):
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
        out.append(m)
    return out


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
