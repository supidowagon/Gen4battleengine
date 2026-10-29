"""Extract static Gen 4 Platinum game data from the pret/pokeplatinum decomp.

Writes intermediate JSON (species, moves, types, natures, items, abilities,
learnsets) that a later step packs into ``data/gen4_platinum/tables.npz``.
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
import json
import math
import re
from pathlib import Path

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
            }
        )
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
        chart[a][d] = TYPE_MULTI[mul]
        if after_separator:
            foresight.append([a, d])
    if not ended:
        raise ExtractError(f"{path}: type chart has no 0xFF terminator")
    return {
        "names": names,
        "chart": chart,
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
    out = []
    for const in items:
        if const.startswith("ITEM_UNUSED_"):
            # tools/dataproc/src/itemproc.c HasData: unused ids have no data file.
            out.append({"name": const, "hold_effect": hold_id("HOLD_EFFECT_NONE"), "hold_param": 0})
            continue
        d = _load_json(decomp / "res" / "items" / "data" / f"{_stem(const, 'ITEM_')}.json")
        out.append(
            {
                "name": d["name"],
                "hold_effect": hold_id(d["holdEffect"]),
                "hold_param": int(d["effectParam"]),
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract static game data from the decomp.")
    p.add_argument("--decomp", type=Path, default=Path("external/pokeplatinum"))
    p.add_argument("--out", type=Path, default=Path("data/intermediate"))
    args = p.parse_args(argv)
    extract_static(args.decomp, args.out)


if __name__ == "__main__":
    main()
