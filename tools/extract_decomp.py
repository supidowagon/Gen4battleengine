"""Extract static Gen 4 Platinum game data from the pret/pokeplatinum decomp.

Writes intermediate JSON (species, moves, types, natures, items, abilities,
learnsets, trainers) that a later step packs into ``data/gen4_platinum/tables.npz``.
Every source path and format is documented in ``docs/decomp_paths.md``.

Conventions:

* A ``generated/<name>.txt`` constant's numeric id is its 0-based line number.
  A trailing count sentinel (``MAX_MOVES``, ``NUM_POKEMON_TYPES``,
  ``NATURE_COUNT``) is not an id and is dropped. Every output list is indexed by the decomp's own id.
* Stat vectors use the order HP, Atk, Def, SpA, SpD, Spe (index = ``ids.Stat``).
  The decomp orders stats HP, Atk, Def, Spe, SpA, SpD
  (``generated/pokemon_stats.txt``), so this module reorders them.
* Enumerated fields (types, abilities, move class, battle effect, hold effect)
  are stored as the decomp's numeric ids. A move's ``target`` is stored as its
  ``RANGE_*`` constant name, because ``move_ranges`` is a metang bit mask whose
  numeric values are not line numbers.
* Anything that cannot be parsed raises ``ExtractError`` naming the offending
  file (and line, for C sources); nothing is skipped silently.

CLI: ``uv run python -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from tools.battle_scripts import (
    Op,
    battle_structures,
    has_partner,
    is_var,
    placed_enemies,
    structure_fields,
    var_ids,
)

# Our stat order (index = ids.Stat) as keys of the decomp's species `base_stats`.
STAT_KEYS = ("hp", "attack", "defense", "special_attack", "special_defense", "speed")

# Column of src/pokemon.c sNatureStatAffinities (indexed `[STAT_X - 1]` in the
# decomp's Atk, Def, Spe, SpA, SpD order) -> our stat index.
NATURE_STAT_INDEX = {
    "STAT_ATTACK": 1,
    "STAT_DEFENSE": 2,
    "STAT_SPECIAL_ATTACK": 3,
    "STAT_SPECIAL_DEFENSE": 4,
    "STAT_SPEED": 5,
}

# Type-chart multipliers in tenths (include/constants/battle.h:144). A pair not
# listed in sTypeMatchupMultipliers is neutral, i.e. x1 = 10 tenths.
TYPE_MULTI = {
    "TYPE_MULTI_IMMUNE": 0,
    "TYPE_MULTI_NOT_VERY_EFF": 5,
    "TYPE_MULTI_SUPER_EFF": 20,
}
TYPE_MULTI_NEUTRAL = 10

# Species whose US Pokedex weight in pounds is 0.1 lb too low (a Gen 3
# conversion error); the decomp's build adds 0.1 lb back before converting to
# hectograms (tools/dataproc/src/speciesproc.c, WeightRoundingOverride returns
# -0.1 and ReadWeight subtracts it).
WEIGHT_OVERRIDE_SPECIES = {
    "SPECIES_MR_MIME",
    "SPECIES_FURRET",
    "SPECIES_PORYGON2",
    "SPECIES_LINOONE",
    "SPECIES_LOMBRE",
    "SPECIES_TORKOAL",
    "SPECIES_RAYQUAZA",
}
LB_TO_KG = 0.45359237  # speciesproc.c LB_TO_KG


class ExtractError(RuntimeError):
    """Raised when a decomp source cannot be parsed exactly."""


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


_SENTINEL = re.compile(r"(MAX|NUM)_[A-Z0-9_]+|[A-Z0-9_]+_COUNT")


def _constants(decomp: Path, name: str, prefix: str) -> list[str]:
    """Return the constant names of ``generated/<name>.txt``; list index = id."""
    path = decomp / "generated" / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", line):
            raise ExtractError(f"{path}:{lineno}: unexpected constant line {raw!r}")
        if lineno == len(lines) and _SENTINEL.fullmatch(line):
            break  # count sentinel such as MAX_MOVES, NUM_POKEMON_TYPES, NATURE_COUNT
        if not line.startswith(prefix):
            raise ExtractError(f"{path}:{lineno}: constant {line!r} lacks prefix {prefix}")
        out.append(line)
    return out


def _index(names: list[str], what: str):
    lookup = {n: i for i, n in enumerate(names)}

    def get(name: str) -> int:
        if name not in lookup:
            raise ExtractError(f"unknown {what} constant {name!r}")
        return lookup[name]

    return get


def _stem(constant: str, prefix: str) -> str:
    return constant[len(prefix) :].lower()


def _round_half_away(x: float) -> int:
    """C ``round`` for non-negative values (Python's ``round`` rounds half to even)."""
    return math.floor(x + 0.5)


def _text_bank(decomp: Path, name: str) -> list[str]:
    return [m["en_US"] for m in _load_json(decomp / "res" / "text" / f"{name}.json")["messages"]]


def _gender_ratios(decomp: Path) -> dict[str, int]:
    path = decomp / "generated" / "gender_ratios.txt"
    out = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = re.fullmatch(r"\s*(GENDER_RATIO_[A-Z0-9_]+)\s*=\s*(\d+)\s*", raw)
        if not m:
            raise ExtractError(f"{path}:{lineno}: unparsed gender ratio line {raw!r}")
        out[m.group(1)] = int(m.group(2))
    return out


def _weight_hg(dex: dict, species: str, path: Path) -> int:
    """Weight in hectograms, computed as speciesproc.c ReadDexData does."""
    if "weight_kilograms" in dex:
        kg = float(dex["weight_kilograms"])
    elif "weight_pounds" in dex:
        lb = float(dex["weight_pounds"])
        if species in WEIGHT_OVERRIDE_SPECIES:
            lb += 0.1
        kg = lb * LB_TO_KG
    else:
        raise ExtractError(f"{path}: no weight_kilograms or weight_pounds")
    return _round_half_away(kg * 10)


def _species_json(decomp: Path, const: str) -> tuple[Path, dict]:
    path = decomp / "res" / "pokemon" / _stem(const, "SPECIES_") / "data.json"
    return path, _load_json(path)


def extract_species(decomp: Path) -> list[dict]:
    species = _constants(decomp, "species", "SPECIES_")
    type_id = _index(_constants(decomp, "pokemon_types", "TYPE_"), "type")
    ability_id = _index(_constants(decomp, "abilities", "ABILITY_"), "ability")
    exp_rate_id = _index(_constants(decomp, "exp_rates", "EXP_RATE_"), "exp rate")
    ratios = _gender_ratios(decomp)
    out = []
    for const in species:
        path, d = _species_json(decomp, const)
        dex = d["pokedex_data"]
        if d["gender_ratio"] not in ratios:
            raise ExtractError(f"{path}: unknown gender ratio {d['gender_ratio']!r}")
        out.append(
            {
                "name": dex["en"]["name"],
                "base_stats": [int(d["base_stats"][k]) for k in STAT_KEYS],
                "types": [type_id(t) for t in d["types"]],
                "abilities": [ability_id(a) for a in d["abilities"]],
                "gender_ratio": ratios[d["gender_ratio"]],
                "weight_hg": _weight_hg(dex, const, path),
                "base_friendship": int(d["base_friendship"]),
                "exp_rate": exp_rate_id(d["exp_rate"]),
            }
        )
    return out


def extract_exp_tables(decomp: Path) -> list[list[int]]:
    """Experience needed per level: ``[rate][level]`` for levels 0..100, rate order =
    ``generated/exp_rates.txt`` (``res/pokemon/.shared/exp_tables.csv`` columns)."""
    path = decomp / "res" / "pokemon" / ".shared" / "exp_tables.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if header[0] != "level" or len(header) != 9:
        raise ExtractError(f"{path}: unexpected header {header!r}")
    if [int(r[0]) for r in body] != list(range(101)):
        raise ExtractError(f"{path}: levels are not 0..100")
    return [[int(r[1 + rate]) for r in body] for rate in range(8)]


def extract_charmap(decomp: Path) -> dict[int, str]:
    """Text code -> character from ``tools/msgenc/charmap.txt``; ``{command}`` entries and
    comments are skipped. Trailing whitespace is part of the value (code 0x01DE is a space)."""
    path = decomp / "tools" / "msgenc" / "charmap.txt"
    out: dict[int, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.lstrip(" 	")
        if not line or line.startswith("//"):
            continue
        code, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[0-9A-F]{4}", code):
            raise ExtractError(f"{path}:{lineno}: unparsed charmap line {raw!r}")
        if len(value) > 2 and value.startswith("{") and value.endswith("}"):
            continue
        out[int(code, 16)] = value
    return out


def extract_moves(decomp: Path) -> list[dict]:
    moves = _constants(decomp, "moves", "MOVE_")
    type_id = _index(_constants(decomp, "pokemon_types", "TYPE_"), "type")
    class_id = _index(_constants(decomp, "move_classes", "CLASS_"), "move class")
    effect_id = _index(_constants(decomp, "move_battle_effects", "BATTLE_EFFECT_"), "effect")
    ranges = set(_constants(decomp, "move_ranges", "RANGE_"))
    flags = set(_constants(decomp, "move_flags", "MOVE_FLAG_"))
    out = []
    for const in moves:
        path = decomp / "res" / "moves" / _stem(const, "MOVE_") / "data.json"
        d = _load_json(path)
        if d["range"] not in ranges:
            raise ExtractError(f"{path}: unknown range {d['range']!r}")
        bad = [f for f in d["flags"] if f not in flags]
        if bad:
            raise ExtractError(f"{path}: unknown move flags {bad!r}")
        out.append(
            {
                "name": d["name"],
                "power": int(d["power"]),
                "type": type_id(d["type"]),
                "category": class_id(d["class"]),
                "accuracy": int(d["accuracy"]),
                "pp": int(d["pp"]),
                "priority": int(d["priority"]),
                "target": d["range"],
                "effect": effect_id(d["effect"]["type"]),
                "effect_chance": int(d["effect"]["chance"]),
                "flags": list(d["flags"]),
            }
        )
    return out


def _c_array_body(path: Path, decl_regex: str) -> list[tuple[int, str]]:
    """Return (lineno, line) pairs between a C array declaration and its ``};``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if re.search(decl_regex, line)), None)
    if start is None:
        raise ExtractError(f"{path}: declaration matching {decl_regex!r} not found")
    body = []
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == "};":
            return body
        body.append((i + 1, lines[i]))
    raise ExtractError(f"{path}:{start + 1}: array never closed")


def extract_types(decomp: Path) -> dict:
    types = _constants(decomp, "pokemon_types", "TYPE_")
    type_id = _index(types, "type")
    names = _text_bank(decomp, "pokemon_type_names")
    if len(names) != len(types):
        raise ExtractError(f"{len(names)} type names for {len(types)} type ids")
    n = len(types)
    chart = [[TYPE_MULTI_NEUTRAL] * n for _ in range(n)]
    # Row index of each listed pair in sTypeMatchupMultipliers (-1: unlisted).
    # BattleSystem_ApplyTypeChart (battle_lib.c:2611-2646) walks the rows in
    # order, so a dual type's two multipliers apply in this order, not type 1
    # then type 2; with integer division the order can change the result.
    order = [[-1] * n for _ in range(n)]
    rows = 0
    foresight = []
    path = decomp / "src" / "battle" / "battle_lib.c"
    row = re.compile(r"\s*\{\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\},\s*")
    after_separator = ended = False
    for lineno, line in _c_array_body(path, r"sTypeMatchupMultipliers\[\]\[3\]\s*=\s*\{"):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        m = row.fullmatch(line)
        if not m or ended:
            raise ExtractError(f"{path}:{lineno}: unparsed type chart line {line!r}")
        atk, dfn, mul = m.groups()
        if (atk, dfn) == ("0xFE", "0xFE"):
            after_separator = True
            continue
        if (atk, dfn) == ("0xFF", "0xFF"):
            ended = True
            continue
        if mul not in TYPE_MULTI:
            raise ExtractError(f"{path}:{lineno}: unknown multiplier {mul!r}")
        a, d = type_id(atk), type_id(dfn)
        if order[a][d] >= 0:
            raise ExtractError(f"{path}:{lineno}: duplicate type chart pair {atk}, {dfn}")
        chart[a][d] = TYPE_MULTI[mul]
        order[a][d] = rows
        rows += 1
        if after_separator:
            foresight.append([a, d])
    if not ended:
        raise ExtractError(f"{path}: type chart has no 0xFF terminator")
    return {
        "names": names,
        "chart": chart,
        "chart_order": order,
        "encoding": {
            "unit": "tenths: damage multiplier = value / 10",
            "immune": TYPE_MULTI["TYPE_MULTI_IMMUNE"],
            "not_very_effective": TYPE_MULTI["TYPE_MULTI_NOT_VERY_EFF"],
            "neutral": TYPE_MULTI_NEUTRAL,
            "super_effective": TYPE_MULTI["TYPE_MULTI_SUPER_EFF"],
            "source": "src/battle/battle_lib.c sTypeMatchupMultipliers; "
            "include/constants/battle.h TYPE_MULTI_*; unlisted pairs are neutral",
        },
        # [attacker, defender] rows after the 0xFE separator: ignored under
        # Foresight / Scrappy (battle_lib.c BasicTypeMulApplies).
        "foresight_immunities": foresight,
    }


def extract_natures(decomp: Path) -> list[dict]:
    natures = _constants(decomp, "natures", "NATURE_")
    names = _text_bank(decomp, "nature_names")
    if len(names) != len(natures):
        raise ExtractError(f"{len(names)} nature names for {len(natures)} nature ids")
    path = decomp / "src" / "pokemon.c"
    head = re.compile(r"\s*\[(NATURE_\w+)\]\s*=\s*\{\s*")
    cell = re.compile(r"\s*\[(STAT_\w+)\s*-\s*1\]\s*=\s*(-?\d+),\s*")
    tail = re.compile(r"\s*\},?\s*")
    affinities: dict[str, dict[str, int]] = {}
    current = None
    for lineno, line in _c_array_body(path, r"sNatureStatAffinities\[\]\[5\]\s*=\s*\{"):
        if not line.strip():
            continue
        if m := head.fullmatch(line):
            current = affinities.setdefault(m.group(1), {})
        elif (m := cell.fullmatch(line)) and current is not None:
            if m.group(1) not in NATURE_STAT_INDEX:
                raise ExtractError(f"{path}:{lineno}: unknown stat {m.group(1)!r}")
            current[m.group(1)] = int(m.group(2))
        elif tail.fullmatch(line):
            current = None
        else:
            raise ExtractError(f"{path}:{lineno}: unparsed nature line {line!r}")
    out = []
    for const, name in zip(natures, names, strict=True):
        row = affinities.get(const)
        if row is None or set(row) != set(NATURE_STAT_INDEX):
            raise ExtractError(f"{path}: incomplete nature row for {const}")
        ups = [NATURE_STAT_INDEX[s] for s, v in row.items() if v == 1]
        downs = [NATURE_STAT_INDEX[s] for s, v in row.items() if v == -1]
        if any(v not in (-1, 0, 1) for v in row.values()) or len(ups) != len(downs) or len(ups) > 1:
            raise ExtractError(f"{path}: unexpected nature row for {const}: {row}")
        out.append(
            {"name": name, "up": ups[0] if ups else None, "down": downs[0] if downs else None}
        )
    return out


def extract_items(decomp: Path) -> list[dict]:
    items = _constants(decomp, "items", "ITEM_")
    hold_id = _index(_constants(decomp, "item_hold_effects", "HOLD_EFFECT_"), "hold effect")
    move_id = _index(_constants(decomp, "moves", "MOVE_"), "move")
    out = []
    for const in items:
        if const.startswith("ITEM_UNUSED_"):
            # tools/dataproc/src/itemproc.c HasData: unused ids have no data file.
            out.append(
                {
                    "name": const,
                    "hold_effect": hold_id("HOLD_EFFECT_NONE"),
                    "hold_param": 0,
                    "teaches_move": None,
                }
            )
            continue
        d = _load_json(decomp / "res" / "items" / "data" / f"{_stem(const, 'ITEM_')}.json")
        out.append(
            {
                "name": d["name"],
                "hold_effect": hold_id(d["holdEffect"]),
                "hold_param": int(d["effectParam"]),
                # res/items/data/tm10.json `teachesMove`; null for every other item.
                "teaches_move": move_id(d["teachesMove"]) if d.get("teachesMove") else None,
            }
        )
    return out


def extract_abilities(decomp: Path) -> list[dict]:
    abilities = _constants(decomp, "abilities", "ABILITY_")
    names = _text_bank(decomp, "ability_names")
    if len(names) != len(abilities):
        raise ExtractError(f"{len(names)} ability names for {len(abilities)} ability ids")
    return [{"name": n.strip()} for n in names]


def extract_learnsets(decomp: Path) -> dict[str, list[list[int]]]:
    species = _constants(decomp, "species", "SPECIES_")
    move_id = _index(_constants(decomp, "moves", "MOVE_"), "move")
    out = {}
    for sid, const in enumerate(species):
        _, d = _species_json(decomp, const)
        by_level = d.get("learnset", {}).get("by_level", [])
        out[str(sid)] = [[int(level), move_id(move)] for level, move in by_level]
    return out


_NUM_TMS = 92  # NUM_TMS
_NUM_HMS = 8  # NUM_HMS


def extract_tm_learnsets(decomp: Path) -> dict[str, list[int]]:
    """Species id -> sorted TM/HM indices from ``learnset.by_tm``, numbered as
    tools/dataproc/src/speciesproc.c ProcTMLearnset sets the bits: TM01-TM92 are
    0-91 and HM01-HM08 are 92-99."""
    species = _constants(decomp, "species", "SPECIES_")
    out = {}
    for sid, const in enumerate(species):
        _, d = _species_json(decomp, const)
        bits = set()
        for marker in d.get("learnset", {}).get("by_tm", []):
            m = re.fullmatch(r"(TM|HM)(\d+)", marker)
            is_hm = bool(m) and m.group(1) == "HM"
            if not m or not 1 <= int(m.group(2)) <= (_NUM_HMS if is_hm else _NUM_TMS):
                raise ExtractError(f"species {const}: bad TM/HM marker {marker!r}")
            bits.add(int(m.group(2)) - 1 + (_NUM_TMS if is_hm else 0))
        out[str(sid)] = sorted(bits)
    return out


# Battle start condition for each overworld weather that the battle inherits
# (src/battle/battle_lib.c BattleSystem_TriggerEffectOnSwitch, state
# SWITCH_IN_CHECK_STATE_FIELD_WEATHER). Every other weather, including
# OVERWORLD_WEATHER_HAILING, starts the battle with no condition.
WEATHER_CONDITION = {
    "OVERWORLD_WEATHER_RAINING": "rain",
    "OVERWORLD_WEATHER_HEAVY_RAIN": "rain",
    "OVERWORLD_WEATHER_THUNDERSTORM": "rain",
    "OVERWORLD_WEATHER_SNOWING": "hail",
    "OVERWORLD_WEATHER_HEAVY_SNOW": "hail",
    "OVERWORLD_WEATHER_BLIZZARD": "hail",
    "OVERWORLD_WEATHER_SANDSTORM": "sandstorm",
    "OVERWORLD_WEATHER_FOG": "fog",
    "OVERWORLD_WEATHER_DEEP_FOG": "fog",
    "OVERWORLD_WEATHER_HARSH_SUN": "sun",
    # Battle Frontier only; not a start condition this schema can express, so a
    # trainer fought under it fails extraction (_start_condition).
    "OVERWORLD_WEATHER_TRICK_ROOM": "trick_room",
}
START_CONDITIONS = ("none", "rain", "hail", "sandstorm", "sun", "fog")

# Map-script lines whose TRAINER_* operands start a battle on the map that owns the
# script. SetVar feeds the `StartTrainerBattle VAR_...` form.
_BATTLE_SCRIPT_LINE = re.compile(
    r"\s*(StartTrainerBattle|StartTagBattle|StartFirstBattle|SetVar)\b.*"
)
_PARTY_SIZE_MAX = 6  # MAX_PARTY_SIZE
_TRAINER_ITEMS_MAX = 4  # MAX_TRAINER_ITEMS
_MOVES_MAX = 4  # LEARNED_MOVES_MAX
START_CONDITIONS_FILE = Path(__file__).with_name("start_conditions.json")


def _ai_flag_bits(decomp: Path) -> dict[str, int]:
    """``generated/ai_flags.txt`` is a metang mask: line 1 is AI_FLAG_NONE (0) and
    line n + 2 is bit n. AI_FLAG_ALL is left out (no trainer uses it)."""
    flags = _constants(decomp, "ai_flags", "AI_FLAG_")
    return {f: (0 if i == 0 else 1 << (i - 1)) for i, f in enumerate(flags) if f != "AI_FLAG_ALL"}


def _map_headers(decomp: Path) -> dict[str, dict[str, str]]:
    """``include/data/map_headers.h``: MAP_HEADER_* -> its scripts, events and weather."""
    path = decomp / "include" / "data" / "map_headers.h"
    src = path.read_text(encoding="utf-8")
    out = {}
    for name, body in re.findall(
        r"\n    \[(MAP_HEADER_\w+)\] = \{\n(.*?)\n    \},", src, re.DOTALL
    ):
        fields = dict(re.findall(r"\.(\w+) = ([^,\n]+),", body))
        try:
            out[name] = {k: fields[k] for k in ("scriptsArchiveID", "eventsArchiveID", "weather")}
        except KeyError as e:
            raise ExtractError(f"{path}: {name} has no {e.args[0]}") from None
    if len(out) != src.count("\n    [MAP_HEADER_"):
        raise ExtractError(f"{path}: parsed {len(out)} map headers, file has more")
    return out


def _weather_conditions(decomp: Path) -> dict[str, set[str]]:
    """Overworld weather constant -> the set of battle start conditions it can give.

    A "yearly" weather (``OVERWORLD_WEATHER_YEARLY_START + n``) is looked up by date
    in ``sYearlyWeather`` (``src/field_overworld_weather.c``, FieldSystem_GetWeather),
    so it can give a different condition on different days.
    """
    hdr = (decomp / "include" / "constants" / "overworld_weather.h").read_text(encoding="utf-8")
    yearly = {
        name: int(col)
        for name, col in re.findall(
            r"#define (OVERWORLD_WEATHER_\w+)\s+\(OVERWORLD_WEATHER_YEARLY_START \+ (\d+)\)", hdr
        )
    }
    yearly.pop("OVERWORLD_WEATHER_YEARLY_END", None)  # one past the last column
    path = decomp / "src" / "field_overworld_weather.c"
    days = []
    for lineno, line in _c_array_body(path, r"sYearlyWeather\[.*\]\s*=\s*\{"):
        if not line.strip() or line.strip().startswith("//"):
            continue
        m = re.fullmatch(r"\s*\[DAY_OF_YEAR_\w+ - 1\] = \{([^}]*)\},?\s*", line)
        if not m:
            raise ExtractError(f"{path}:{lineno}: unparsed yearly weather line {line!r}")
        days.append([w.strip() for w in m.group(1).split(",")])
    if len(days) != 366 or any(len(d) != len(yearly) for d in days):
        raise ExtractError(f"{path}: expected 366 rows of {len(yearly)} yearly weathers")

    out = {
        name: {WEATHER_CONDITION.get(d[col], "none") for d in days} for name, col in yearly.items()
    }
    for weather in re.findall(r"^#define (OVERWORLD_WEATHER_\w+)\s+\d+\s*$", hdr, re.MULTILINE):
        out[weather] = {WEATHER_CONDITION.get(weather, "none")}
    return out


def _object_trainers(decomp: Path, header: dict[str, str]) -> list[str]:
    """The TRAINER_* scripts of a map's object events (``res/field/events/<id>.json``)."""
    events = decomp / "res" / "field" / "events" / f"{header['eventsArchiveID']}.json"
    return [
        obj["script"]
        for obj in _load_json(events)["object_events"]
        if str(obj["script"]).startswith("TRAINER_")
    ]


def _rematches(decomp: Path) -> dict[str, list[str]]:
    """Base TRAINER_* -> its Vs. Seeker rematch TRAINER_* ids (``gVsSeekerRematchData``,
    ``src/overlay005/vs_seeker.c``)."""
    path = decomp / "src" / "overlay005" / "vs_seeker.c"
    out: dict[str, list[str]] = {}
    for lineno, line in _c_array_body(path, r"gVsSeekerRematchData\[\]\s*=\s*\{"):
        s = line.strip()
        if not s or s.startswith("//") or re.fullmatch(r"NoUniqueRematches\(\w+\),", s):
            continue
        m = re.fullmatch(r"\{([^}]*)\},?", s)
        if not m:
            raise ExtractError(f"{path}:{lineno}: unparsed rematch line {line!r}")
        base, *rematches = [t.strip() for t in m.group(1).split(",")]
        out.setdefault(base, []).extend(t for t in rematches if t.startswith("TRAINER_"))
    return out


def _trainer_maps(decomp: Path, headers: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    """TRAINER_* -> the map headers where a battle with that trainer starts.

    A trainer is placed on a map when the map's events file has an object whose
    script is that trainer, or when a battle-starting line of the map's script file
    names it. A Vs. Seeker rematch (``gVsSeekerRematchData``,
    ``src/overlay005/vs_seeker.c``) is fought on its base trainer's maps.
    """
    where: dict[str, set[str]] = {}
    for header, h in headers.items():
        for trainer in _object_trainers(decomp, h):
            where.setdefault(trainer, set()).add(header)
        script = decomp / "res" / "field" / "scripts" / f"{h['scriptsArchiveID']}.s"
        for line in script.read_text(encoding="utf-8").splitlines():
            if _BATTLE_SCRIPT_LINE.fullmatch(line):
                for trainer in re.findall(r"\bTRAINER_\w+", line):
                    where.setdefault(trainer, set()).add(header)
    for base, rematches in _rematches(decomp).items():
        for t in rematches:
            where.setdefault(t, set()).update(where.get(base, set()))
    return where


# Field script commands kept as ops (tools/battle_scripts.py): name -> the kinds of its
# operands. "trainer" and "var" operands must resolve. A "value" that is a variable, or a
# constant outside _script_symbols (OBJ_EVENT_GFX_*, DIR_*, ITEM_*, ...), makes the op
# `other`: the ROM assembles the first as a different command (SetVarFromVar,
# CompareVarToVar; asm/macros/scrcmd.inc:178-184,309-315), and the evaluator raises on an
# `other` it meets, so neither is ever read as a value.
_SCRIPT_OPS = {
    "StartTrainerBattle": ("trainer", "trainer"),
    "StartTagBattle": ("trainer", "trainer", "trainer"),
    "SetVar": ("var", "value"),
    "SetVarFromValue": ("var", "value"),
    "GoToIfEq": ("var", "value", "label"),
    "GoTo": ("label",),
    "Call": ("label",),
    "Return": (),
    "End": (),
    "GetPlayerGender": ("var",),
    "GetPlayerStarterSpecies": ("var",),
    "SetHasPartner": (),
}
# A constant with one of these prefixes names an id that must resolve.
_STRICT_PREFIXES = ("TRAINER_", "SPECIES_", "GENDER_", "VAR_")


def _script_symbols(decomp: Path) -> dict[str, int]:
    """Constants a kept op may name: trainers, species and genders (their
    ``generated/*.txt`` line), variables (``battle_scripts.var_ids``), TRUE and FALSE."""
    out = {"TRUE": 1, "FALSE": 0}
    for name, prefix in (("trainers", "TRAINER_"), ("species", "SPECIES_"), ("genders", "GENDER_")):
        out.update({c: i for i, c in enumerate(_constants(decomp, name, prefix))})
    out.update({k: v for k, v in var_ids(decomp).items() if is_var(v)})
    return out


def _script_file_ops(path: Path, symbols: dict[str, int]) -> list[Op]:
    """One ``.s`` file's ops in file order: a label per ``Name:`` line, an op per command
    line (``_SCRIPT_OPS``, else ``other``). The file's own ``#define``s are expanded."""
    defines: dict[str, str] = {}
    ops: list[Op] = []

    def value(token: str, where: str, depth: int = 0) -> int | None:
        token = token.strip()
        if re.fullmatch(r"\d+|0[xX][0-9a-fA-F]+", token):
            return int(token, 0)
        if token in defines and depth < 8:
            return value(defines[token], where, depth + 1)
        if token in symbols:
            return symbols[token]
        if token.startswith(_STRICT_PREFIXES):
            raise ExtractError(f"{where}: unknown constant {token!r}")
        return None

    text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"//|@", raw, maxsplit=1)[0].strip()  # `@` comments (GNU as, ARM)
        where = f"{path}:{lineno}"
        if not line:
            continue
        if line.startswith("#"):
            m = re.fullmatch(r"#define\s+(\w+)\s+(.+)", line)
            if m:
                defines[m.group(1)] = m.group(2).strip()
            elif not line.startswith("#include"):
                raise ExtractError(f"{where}: unhandled directive {line!r}")
            continue
        m = re.fullmatch(r"(\w+):", line)
        if m:
            ops.append(Op("label", (m.group(1),)))
            continue
        name, _, rest = line.partition(" ")
        kinds = _SCRIPT_OPS.get(name)
        if kinds is None:
            ops.append(Op("other", (name,)))
            continue
        args = [a.strip() for a in rest.split(",")] if rest.strip() else []
        if name == "StartTrainerBattle" and len(args) == 1:
            args.append("0")  # enemyTrainer2=0 (scrcmd.inc:1427)
        if len(args) != len(kinds):
            raise ExtractError(f"{where}: {name} takes {len(kinds)} operands: {line!r}")
        out: list[int | str] | None = []
        for kind, a in zip(kinds, args, strict=True):
            if kind == "label":
                if not re.fullmatch(r"\w+", a):
                    raise ExtractError(f"{where}: {name} target {a!r} is not a label")
                out.append(a)
                continue
            v = value(a, where)
            if kind == "value":
                if v is None or is_var(v):
                    out = None
                    break
            elif v is None:
                raise ExtractError(f"{where}: {name} operand {a!r} is not a {kind}")
            elif kind == "var" and not is_var(v):
                raise ExtractError(f"{where}: {name} operand {a!r} is not a variable")
            out.append(v)
        if out is None:
            ops.append(Op("other", (name,)))
        else:
            ops.append(Op("SetVar" if name == "SetVarFromValue" else name, tuple(out)))
    return ops


def _script_ops(decomp: Path, files) -> dict[str, list[Op]]:
    """``res/field/scripts/<file>.s`` -> its ops, for each script archive id in ``files``."""
    symbols = _script_symbols(decomp)
    return {
        f: _script_file_ops(decomp / "res" / "field" / "scripts" / f"{f}.s", symbols)
        for f in sorted(set(files))
    }


def _placed_enemies(
    decomp: Path, headers: dict[str, dict[str, str]], scripts: dict[str, list[Op]]
) -> dict[str, set[int]]:
    """MAP_HEADER_* -> its placed enemy trainer ids (Ruling 3), for each header whose script
    runs ``SetHasPartner``: trainer object events, the script's battle enemies, their
    Vs. Seeker rematches, less the partner ids (``battle_scripts.placed_enemies``)."""
    trainer = _index(_constants(decomp, "trainers", "TRAINER_"), "trainer")
    rematches = {trainer(b): [trainer(t) for t in ts] for b, ts in _rematches(decomp).items()}
    out = {}
    for header, h in headers.items():
        ops = scripts[h["scriptsArchiveID"]]
        if has_partner(ops):
            objects = {trainer(t) for t in _object_trainers(decomp, h)}
            out[header] = placed_enemies(ops, objects, rematches, header)
    return out


def _start_condition_overrides(path: Path, trainers: list[str]) -> dict[str, str]:
    """Read ``{trainer_id: {"trainer": TRAINER_*, "condition": ...}}`` and return
    TRAINER_* -> condition. Each entry's constant must be the one at its id in
    ``generated/trainers.txt``, so an id drift fails instead of moving an override."""
    out = {}
    for key, entry in _load_json(path).items():
        if key == "comment":
            continue
        if not key.isdecimal() or int(key) >= len(trainers):
            raise ExtractError(f"{path}: {key!r} is not a trainer id")
        if not isinstance(entry, dict) or set(entry) != {"trainer", "condition"}:
            raise ExtractError(f"{path}: entry {key} must be {{trainer, condition}}")
        const, cond = entry["trainer"], entry["condition"]
        if trainers[int(key)] != const:
            raise ExtractError(f"{path}: id {key} is {trainers[int(key)]}, not {const}")
        if cond not in START_CONDITIONS:
            raise ExtractError(f"{path}: {key} ({const}) has unknown start condition {cond!r}")
        out[const] = cond
    return out


def _start_condition(const: str, derived: set[str], overrides: dict[str, str]) -> str:
    if not derived <= set(START_CONDITIONS):
        raise ExtractError(f"{const} starts under {sorted(derived)}; not a supported condition")
    if const in overrides:
        if len(derived) == 1 and derived != {overrides[const]}:
            raise ExtractError(
                f"{START_CONDITIONS_FILE}: {const} is {overrides[const]!r} but the decomp "
                f"gives {next(iter(derived))!r}"
            )
        return overrides[const]
    if len(derived) > 1:
        raise ExtractError(
            f"{const} can start with any of {sorted(derived)}; list it in {START_CONDITIONS_FILE}"
        )
    return next(iter(derived)) if derived else "none"


def _trainer_class_genders(decomp: Path, classes: list[str]) -> list[int]:
    """Gender id (``generated/genders.txt`` line) of each trainer class, from
    ``sTrainerClassGender`` in ``include/data/trainer_class_genders.h``."""
    path = decomp / "include" / "data" / "trainer_class_genders.h"
    gender_id = _index(_constants(decomp, "genders", "GENDER_"), "gender")
    found: dict[str, int] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        m = re.fullmatch(r"\s*\[(TRAINER_CLASS_[A-Z0-9_]+)\]\s*=\s*(GENDER_[A-Z]+),\s*", raw)
        if m:
            if m.group(1) in found:
                raise ExtractError(f"{path}:{lineno}: duplicate class {m.group(1)}")
            found[m.group(1)] = gender_id(m.group(2))
    missing = sorted(set(classes) - set(found))
    unknown = sorted(set(found) - set(classes))
    if missing or unknown:
        raise ExtractError(f"{path}: classes without a gender {missing}, unknown {unknown}")
    return [found[c] for c in classes]


def extract_trainers(decomp_root: Path, out_dir: Path) -> None:
    """Write ``trainers.json``: one entry per ``generated/trainers.txt`` id.

    * ``class`` is the class's display name (``res/text/trainer_class_names.json``),
      ``class_id`` its ``generated/trainer_classes.txt`` id (``header.trainerType``) and
      ``class_gender`` its ``generated/genders.txt`` id (0 male, 1 female, 2 none) from
      ``include/data/trainer_class_genders.h``. TrainerData_BuildParty
      (``src/trainer_data.c``) derives each party member's PID from both.
    * ``ai_flags`` is the bit mask trainerproc builds (bit n = ai_flags.txt line n+2).
    * ``items`` are the bag items the AI may use, as item ids.
    * Each party member's ``difficulty`` is the decomp's ``iv_scale``. As in
      ``tools/dataproc/src/trainerproc.c``, the first party member decides whether
      every member has moves and a held item. With moves, each list is padded with
      MOVE_NONE (0) to four slots, because TrainerData_BuildParty
      (``src/trainer_data.c``) writes all four. Without, ``moves`` is null: the mon
      keeps the species' default moveset at its level. No held item is item 0.
    * ``format`` is "double" when the file sets ``double_battle`` and "single"
      otherwise: trainerproc (``tools/dataproc/src/trainerproc.c``) turns it into
      ``TrainerHeader.battleType`` (``BATTLE_TYPE_DOUBLES`` or 0). The map scripts
      override it (``tools/battle_scripts.py``, plan Rulings 2-5): "multi" for an enemy
      of a tag battle or one placed on a walking-partner map, else
      "double_two_trainer" for an operand of a ``StartTrainerBattle`` with two trainer
      constants. ``opponent_2`` is the other enemy of that pair or tag battle (else
      null), and ``partner`` the teams the player's partner can field
      ([{``trainer``, ``starter``, ``player_gender``}], a field the team does not
      depend on being null; empty unless "multi").
    * ``start_condition`` comes from the weather of the maps where the trainer is
      fought (``_trainer_maps``, ``_weather_conditions``), assuming the player has
      not cleared fog with Defog (``src/field_map_change.c``). A trainer the decomp
      places on no map (unused and dummy entries, or trainers set up by a shared
      script) gets "none". A trainer whose maps or dates give more than one
      condition must be listed in ``tools/start_conditions.json``
      ({trainer_id: {"trainer": TRAINER_*, "condition": ...}}), which may not
      contradict an unambiguous derivation.
    """
    decomp, out_dir = Path(decomp_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainers = _constants(decomp, "trainers", "TRAINER_")
    classes = _constants(decomp, "trainer_classes", "TRAINER_CLASS_")
    class_id = _index(classes, "trainer class")
    class_names = _text_bank(decomp, "trainer_class_names")
    if len(class_names) != len(classes):
        raise ExtractError(f"{len(class_names)} class names for {len(classes)} trainer classes")
    class_genders = _trainer_class_genders(decomp, classes)
    species_id = _index(_constants(decomp, "species", "SPECIES_"), "species")
    move_id = _index(_constants(decomp, "moves", "MOVE_"), "move")
    item_id = _index(_constants(decomp, "items", "ITEM_"), "item")
    ai_bits = _ai_flag_bits(decomp)

    headers = _map_headers(decomp)
    weather = _weather_conditions(decomp)
    where = _trainer_maps(decomp, headers)
    scripts = _script_ops(decomp, [h["scriptsArchiveID"] for h in headers.values()])
    try:
        structures = battle_structures(
            {name: scripts[h["scriptsArchiveID"]] for name, h in headers.items()},
            _placed_enemies(decomp, headers, scripts),
        )
    except ValueError as e:
        raise ExtractError(f"battle structures: {e}") from e
    for header, h in headers.items():
        if h["weather"] not in weather:
            raise ExtractError(f"{header}: unknown weather {h['weather']!r}")
    overrides = _start_condition_overrides(START_CONDITIONS_FILE, trainers)
    unknown = sorted(set(where) - set(trainers))
    if unknown:
        raise ExtractError(f"maps name unknown trainers {unknown}")

    out = []
    for tid, const in enumerate(trainers):
        path = decomp / "res" / "trainers" / "data" / f"{_stem(const, 'TRAINER_')}.json"
        d = _load_json(path)
        party = d["party"]
        if len(party) > _PARTY_SIZE_MAX or len(d["items"]) > _TRAINER_ITEMS_MAX:
            raise ExtractError(f"{path}: party or bag larger than the game allows")
        has_moves = bool(party) and isinstance(party[0]["moves"], list)
        has_item = bool(party) and isinstance(party[0]["item"], str)
        members = []
        for mon in party:
            if isinstance(mon["moves"], list) != has_moves:
                raise ExtractError(f"{path}: party members disagree on having moves")
            if isinstance(mon["item"], str) != has_item:
                raise ExtractError(f"{path}: party members disagree on having items")
            moves = None
            if has_moves:
                if len(mon["moves"]) > _MOVES_MAX:
                    raise ExtractError(f"{path}: more than {_MOVES_MAX} moves")
                moves = [move_id(m) for m in mon["moves"]]
                moves += [0] * (_MOVES_MAX - len(moves))
            members.append(
                {
                    "species": species_id(mon["species"]),
                    "level": int(mon["level"]),
                    "difficulty": int(mon["iv_scale"]),
                    "moves": moves,
                    "item": item_id(mon["item"]) if has_item else 0,
                    "form": int(mon["form"]),
                }
            )
        ai_flags = 0
        for flag in d["ai_flags"]:
            if flag not in ai_bits:
                raise ExtractError(f"{path}: unknown AI flag {flag!r}")
            ai_flags |= ai_bits[flag]
        if not isinstance(d["double_battle"], bool):
            raise ExtractError(f"{path}: double_battle is not a boolean")
        derived = set()
        for header in where.get(const, ()):
            derived |= weather[headers[header]["weather"]]
        out.append(
            {
                "id": tid,
                "name": d["name"],
                "class": class_names[class_id(d["class"])],
                "class_id": class_id(d["class"]),
                "class_gender": class_genders[class_id(d["class"])],
                "ai_flags": ai_flags,
                "items": [item_id(i) for i in d["items"]],
                "party": members,
                **structure_fields(
                    structures.get(tid), "double" if d["double_battle"] else "single"
                ),
                "start_condition": _start_condition(const, derived, overrides),
            }
        )
    _write(out_dir / "trainers.json", out)


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_static(decomp_root: Path, out_dir: Path) -> None:
    decomp_root, out_dir = Path(decomp_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "species.json", extract_species(decomp_root))
    _write(out_dir / "moves.json", extract_moves(decomp_root))
    _write(out_dir / "types.json", extract_types(decomp_root))
    _write(out_dir / "natures.json", extract_natures(decomp_root))
    _write(out_dir / "items.json", extract_items(decomp_root))
    _write(out_dir / "abilities.json", extract_abilities(decomp_root))
    _write(out_dir / "learnsets.json", extract_learnsets(decomp_root))
    _write(out_dir / "tm_learnsets.json", extract_tm_learnsets(decomp_root))
    _write(out_dir / "exp_tables.json", extract_exp_tables(decomp_root))
    _write(
        out_dir / "charmap.json",
        {f"{k:04X}": v for k, v in extract_charmap(decomp_root).items()},
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract game data and trainers from the decomp.")
    p.add_argument("--decomp", type=Path, default=Path("external/pokeplatinum"))
    p.add_argument("--out", type=Path, default=Path("data/intermediate"))
    args = p.parse_args(argv)
    extract_static(args.decomp, args.out)
    extract_trainers(args.decomp, args.out)


if __name__ == "__main__":
    main()
