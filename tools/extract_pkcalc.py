"""Extract PKCalc's Kaizo data files into ``pkcalc.json`` (spec 6.3, plan Task 6).

PKCalc code is never run. Only the data files in ``PKCALC_FILES`` are copied out of the zip
(to ``.cache/pkcalc/<sha256>/``, git-ignored) and each is evaluated by
``tools/kaizo/pkcalc_eval.js`` in a hardened node ``vm`` context. Node's ``vm`` is not a
security boundary: the protection rests on the fixed file list, the zip's SHA-256 being
recorded (``pkcalc.json`` ``sha256``) and reviewed, and the hardened context (null-prototype
global, no string code generation, no ``require``/``process``/timers, a time limit), in that
order. Node runs from the repository with paths as arguments, never from the extraction
folder.

Writes ``<out>/raw/<GLOBAL>.json`` (each file's global as evaluated) and the normalised
``<out>/pkcalc.json``; see ``normalise`` for its schema. PKCalc citations are
``file:line`` in the audited zip (``PKCALC_SHA256``), relative to its top folder.

Usage: python -m tools.extract_pkcalc [--zip PATH] [--out data/intermediate_kaizo/pkcalc]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

from tools.extract_decomp import ExtractError
from tools.rom.inputs import REPO_ROOT, RomInputError, load_inputs

EVAL_JS = Path(__file__).resolve().parent / "kaizo" / "pkcalc_eval.js"
CACHE = REPO_ROOT / ".cache" / "pkcalc"
DEFAULT_OUT = REPO_ROOT / "data" / "intermediate_kaizo" / "pkcalc"
EVAL_TIMEOUT_MS = 10_000
MAX_MEMBER_BYTES = 16 * 1024 * 1024  # the largest listed file, species.js, is 1.5 MB

# Zip member (under the zip's single top folder) -> the global it declares. Nothing else in
# the zip is ever extracted or evaluated.
PKCALC_FILES: dict[str, str] = {
    "src/js/data/dex/species.js": "SPECIES",
    "src/js/data/dex/moves.js": "MOVES",
    "src/js/data/dex/abilities.js": "ABILITIES",
    "src/js/data/dex/items.js": "ITEMS",
    "src/js/data/dex/natures.js": "NATURES",
    "src/js/data/dex/types.js": "TYPES",
    "src/js/data/sets.js": "SETDEX_PK",
    "src/js/data/party_order.js": "PARTY_ORDER_PK",
    "src/js/data/trainer_flags.js": "TRAINER_AI_FLAGS_PK",
    "src/js/data/flags.js": "FLAGS_PK",
}

# PKCalc type ids -> TYPE_* (decomp generated/pokemon_types.txt order; "mystery" is ???).
TYPE_IDS = {
    name: i
    for i, name in enumerate(
        (
            "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
            "steel", "mystery", "fire", "water", "grass", "electric", "psychic", "ice",
            "dragon", "dark",
        )
    )
}  # fmt: skip
# PKCalc move categories -> CLASS_* (decomp generated/move_classes.txt).
CATEGORY_IDS = {"physical": 0, "special": 1, "status": 2}
# PKCalc growthRate -> EXP_RATE_* (decomp generated/exp_rates.txt).
GROWTH_RATES = {
    "Medium Fast": 0,
    "Erratic": 1,
    "Fluctuating": 2,
    "Medium Slow": 3,
    "Fast": 4,
    "Slow": 5,
}
# PKCalc genderRatio strings without a female share -> the personal byte (generated/
# gender_ratios.txt); the others are "<m>% Male, <f>% Female" with byte = f * 256 - 1
# (GENDER_RATIO_FEMALE_12_5 = 31 ... _87_5 = 223), see _gender_ratio.
GENDER_RATIO_FIXED = {"100% Male": 0, "100% Female": 254, "Genderless": 255}
# Plan Ruling 4: PKCalc form key -> (form, personal row). Rows follow
# Pokemon_GetFormNarcIndex (src/pokemon.c:4933-4955) and data/intermediate/forms.json.
FORMS: dict[str, tuple[int, dict[str, tuple[int, int]]]] = {
    "deoxys": (386, {"attack": (1, 496), "defense": (2, 497), "speed": (3, 498)}),
    "wormadam": (413, {"sandy": (1, 499), "trash": (2, 500)}),
    "giratina": (487, {"origin": (1, 501)}),
    "shaymin": (492, {"sky": (1, 502)}),
    "rotom": (
        479,
        {"heat": (1, 503), "wash": (2, 504), "frost": (3, 505), "fan": (4, 506), "mow": (5, 507)},
    ),
}
BASE_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")  # species.js baseStats, ids.Stat order
IV_KEYS = ("hp", "at", "df", "sa", "sd", "sp")  # sets.js ivs, ids.Stat order
# trainer_flags.js keys in bit order (coverage.AI_FLAG_NAMES order).
AI_FLAG_KEYS = (
    "Basic",
    "EvaluateAttack",
    "Expert",
    "Setup",
    "Risky",
    "DamagePriority",
    "BatonPass",
    "TagStrategy",
    "CheckHP",
    "Weather",
    "Harassment",
)
# dex-controls.js:331: a tutor move's price lists the four shard counts in this order.
SHARDS = ("red", "blue", "yellow", "green")
TM_COUNT = 92  # machine > 0 is TM<machine>; machine < 0 is HM<-machine> (dex-controls.js:215)
HM_COUNT = 8
GENDERS = {"Male": "Male", "Female": "Female", "": None}  # sets.js gender; "" = genderless


# --- evaluation -------------------------------------------------------------------------


def run_eval(
    js: Path, global_name: str, timeout_ms: int = EVAL_TIMEOUT_MS, out: Path | None = None
) -> object:
    """Evaluate ``js`` with ``pkcalc_eval.js`` and return its global ``global_name``.

    Node runs from the repository root with absolute paths as arguments; ``NODE_*``
    environment variables (``NODE_OPTIONS`` could preload code) are not passed on.
    Raises ``ExtractError`` when node is missing, the code throws (e.g. ``EvalError`` on
    string code generation) or the time limit is exceeded.
    """
    node = shutil.which("node")
    if node is None:
        raise ExtractError(
            "node is not on PATH; it is needed to evaluate the PKCalc data files "
            f"({EVAL_JS.relative_to(REPO_ROOT).as_posix()})"
        )
    if out is None:
        with tempfile.TemporaryDirectory() as tmp:
            return run_eval(js, global_name, timeout_ms, Path(tmp) / f"{global_name}.json")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("NODE_")}
    cmd = [
        node,
        "--disallow-code-generation-from-strings",
        str(EVAL_JS),
        str(Path(js).resolve()),
        global_name,
        str(Path(out).resolve()),
        str(timeout_ms),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_ms / 1000 + 60,
        )
    except subprocess.TimeoutExpired as e:
        raise ExtractError(f"{js}: node timed out evaluating {global_name}") from e
    if proc.returncode != 0:
        raise ExtractError(f"{js}: {global_name}: {proc.stderr.strip() or proc.returncode}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


# --- the zip ----------------------------------------------------------------------------


def _check_member_names(names: list[str]) -> str:
    """Refuse ``..`` and absolute member paths; return the zip's single top folder."""
    for name in names:
        posix = name.replace("\\", "/")
        if posix.startswith("/") or re.match(r"[A-Za-z]:", posix):
            raise ExtractError(f"PKCalc zip member {name!r} is an absolute path")
        if ".." in posix:
            raise ExtractError(f"PKCalc zip member {name!r} contains '..'")
    tops = {name.replace("\\", "/").split("/", 1)[0] for name in names}
    if len(tops) != 1 or not all("/" in name.replace("\\", "/") for name in names):
        raise ExtractError(f"PKCalc zip has no single top folder: {sorted(tops)}")
    return tops.pop()


def _copy_members(data: bytes, dest: Path) -> dict[str, Path]:
    """Copy only the ``PKCALC_FILES`` members to ``dest``; return member -> path."""
    copied = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = _check_member_names(z.namelist())
        for member in PKCALC_FILES:
            try:
                info = z.getinfo(f"{top}/{member}")
            except KeyError:
                raise ExtractError(f"PKCalc zip has no {top}/{member}") from None
            if info.file_size > MAX_MEMBER_BYTES:
                raise ExtractError(f"PKCalc zip member {member} is {info.file_size} bytes")
            path = dest / member  # built from the fixed list, never from a zip name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(info))
            copied[member] = path
    return copied


def extract_pkcalc(zip_path: Path, out: Path, cache: Path = CACHE) -> dict:
    """Extract, evaluate and normalise the PKCalc data files of ``zip_path`` into ``out``.

    Returns the normalised data, also written to ``out/pkcalc.json``; each evaluated
    global is written to ``out/raw/<GLOBAL>.json``.
    """
    data = Path(zip_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    copied = _copy_members(data, Path(cache) / sha)
    raw_dir = Path(out) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        glob: run_eval(copied[member], glob, out=raw_dir / f"{glob}.json")
        for member, glob in PKCALC_FILES.items()
    }
    result = normalise(raw, sha)
    (Path(out) / "pkcalc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return result


# --- normalisation ----------------------------------------------------------------------


class _Names:
    """Display-name lookup for one kind: every listed field of every entry -> its num.
    A name two entries share is ambiguous and raises when looked up."""

    def __init__(self, kind: str, entries: dict, fields: tuple[str, ...], value=None):
        self.kind = kind
        self._map: dict[str, object] = {}
        self._ambiguous: set[str] = set()
        for key, entry in entries.items():
            target = entry["num"] if value is None else value(key, entry)
            for field in fields:
                name = entry.get(field)
                if not name:
                    continue
                if name in self._map and self._map[name] != target:
                    self._ambiguous.add(name)
                self._map.setdefault(name, target)

    def __call__(self, name: str, where: str):
        if name in self._ambiguous:
            raise ExtractError(f"{where}: {self.kind} {name!r} names more than one entry")
        if name not in self._map:
            raise ExtractError(f"{where}: unknown {self.kind} {name!r}")
        return self._map[name]


def _species_key(key: str, s: dict) -> tuple[str, int, int]:
    """species.js entry -> (personal key, num, form) by plan Ruling 4."""
    base = s.get("baseSpecies") or ""
    if not base:
        return str(s["num"]), s["num"], 0
    num, forms = FORMS.get(base, (None, {}))
    form_key = key[len(base) :] if key.startswith(base) else None
    if form_key not in forms or s["num"] != num:
        raise ExtractError(
            f"species.js {key!r}: unknown form of {base!r} (num {s['num']}); "
            "plan Ruling 4 lists the form keys"
        )
    form, row = forms[form_key]
    return str(row), num, form


def _gender_ratio(text: str, where: str) -> int:
    if text in GENDER_RATIO_FIXED:
        return GENDER_RATIO_FIXED[text]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)% Male, (\d+(?:\.\d+)?)% Female", text or "")
    if m:
        male, female = Fraction(m[1]), Fraction(m[2])
        byte = female * 256 / 100 - 1
        if male + female == 100 and byte.denominator == 1 and 0 < byte < 254:
            return int(byte)
    raise ExtractError(f"{where}: genderRatio {text!r} is not a personal gender byte")


def _lookup(table: dict, key: str, where: str, kind: str):
    if key not in table:
        raise ExtractError(f"{where}: unknown {kind} {key!r}")
    return table[key]


def _two(values: list, pad, where: str, kind: str) -> list:
    if not 1 <= len(values) <= 2:
        raise ExtractError(f"{where}: {len(values)} {kind}")
    return values + [pad] * (2 - len(values))


def _species(raw: dict) -> tuple[dict, dict]:
    """``species[personal key]`` and species display name -> (num, form)."""
    moves = {key: m["num"] for key, m in raw["MOVES"].items()}
    abilities = {key: a["num"] for key, a in raw["ABILITIES"].items()}
    nums = {key: s["num"] for key, s in raw["SPECIES"].items()}
    out: dict[str, dict] = {}
    for key, s in raw["SPECIES"].items():
        if s["num"] == 0:  # species.js:2, the "-----" placeholder: no data
            continue
        where = f"species.js {key}"
        pkey, num, form = _species_key(key, s)
        if pkey in out:
            raise ExtractError(f"{where}: personal key {pkey} is also {out[pkey]['name']!r}")
        level_up, tm, tutor = [], [], []
        for entry in s["learnset"]:
            method = entry["method"]
            if method == "prevo":  # a pre-evolution's level-up move, not this species' own
                continue
            move = _lookup(moves, entry["move"], where, "move")
            if method == "level":
                level_up.append([entry["level"], move])
            elif method == "tm":
                tm.append(move)
            elif method == "tutor":
                tutor.append(move)
            else:
                raise ExtractError(f"{where}: unknown learnset method {method!r}")
        out[pkey] = {
            "num": num,
            "form": form,
            "name": s["name"],
            "base_stats": [s["baseStats"][k] for k in BASE_STAT_KEYS],
            "types": _two(
                [_lookup(TYPE_IDS, t, where, "type") for t in s["types"]],
                None,
                where,
                "types",
            ),
            "abilities": _two(
                [_lookup(abilities, a, where, "ability") for a in s["abilities"]],
                0,
                where,
                "abilities",
            ),
            "catch_rate": s["catchRate"],
            "exp_yield": s["expYield"],
            "growth_rate": _lookup(GROWTH_RATES, s["growthRate"], where, "growthRate"),
            "gender_ratio": _gender_ratio(s["genderRatio"], where),
            "evolutions": [
                {
                    "species": _lookup(nums, e["species"], where, "species"),
                    "method": e["method"],
                    "extra": e.get("extra", ""),
                }
                for e in s["evos"]
            ],
            "level_up": level_up,
            "tm": tm,
            "tutor": tutor,
        }
        if out[pkey]["types"][1] is None:  # one type: the personal record repeats it
            out[pkey]["types"][1] = out[pkey]["types"][0]
    by_name = {
        raw["SPECIES"][key]["name"]: _species_key(key, raw["SPECIES"][key])[1:]
        for key in raw["SPECIES"]
        if raw["SPECIES"][key]["num"]
    }
    return out, by_name


def _moves(raw: dict) -> tuple[dict, dict, list]:
    """``moves[num]``, ``tm_table`` and ``tutor_table`` (move-number order, no tutor slot)."""
    out: dict[str, dict] = {}
    tm_table: dict[int, int] = {}
    tutor_table = []
    for key, m in raw["MOVES"].items():
        where = f"moves.js {key}"
        num = m["num"]
        if str(num) in out:
            raise ExtractError(f"{where}: num {num} is also {out[str(num)]['name']!r}")
        tutor = m["tutor"] or None
        price = m["price"]
        if (tutor is None) != (not price) or (price and len(price) != len(SHARDS)):
            raise ExtractError(f"{where}: tutor {m['tutor']!r} with price {price!r}")
        out[str(num)] = {
            "name": m["name"],
            "aliases": [m["shortName"], m["calcName"]],
            "type": _lookup(TYPE_IDS, m["type"], where, "type"),
            "category": _lookup(CATEGORY_IDS, m["category"], where, "category"),
            "power": m["basePower"],
            "accuracy": m["accuracy"],
            "pp": m["pp"],
            "chance": m["secondaryChance"],
            "priority": m["priority"],
            "target": m["target"],
            "effect_text": m["effect"],
            "machine": m["machine"],
            "tutor": tutor,
            "price": price,
        }
        machine = m["machine"]
        if machine:
            index = machine if machine > 0 else TM_COUNT - machine  # HM01 = 93 ... HM08 = 100
            if not 1 <= index <= TM_COUNT + HM_COUNT or index in tm_table:
                raise ExtractError(f"{where}: machine {machine} is out of range or repeated")
            tm_table[index] = num
        if tutor is not None:
            tutor_table.append(
                {"move": num, "cost": dict(zip(SHARDS, price, strict=True)), "location": tutor}
            )
    tutor_table.sort(key=lambda t: t["move"])
    return out, {str(k): tm_table[k] for k in sorted(tm_table)}, tutor_table


def _trainers(raw: dict, species_by_name: dict) -> dict:
    """``trainers[label]`` in party_order.js order; the sets of sets.js joined by label."""
    moves = _Names("move", raw["MOVES"], ("name", "calcName", "shortName"))
    items = _Names("item", raw["ITEMS"], ("name", "calcName"))
    abilities = _Names("ability", raw["ABILITIES"], ("name", "calcName"))
    natures = {n["name"] for n in raw["NATURES"].values()}
    sets = raw["SETDEX_PK"]
    used = set()
    out: dict[str, dict] = {}
    for label, order in raw["PARTY_ORDER_PK"].items():
        if not order:  # the rival templates ({RIVAL_STARTER_n}, {RIVAL2_*}): no sets
            out[label] = {"party": [], "template": True}
            continue
        if len(set(order)) != len(order):
            raise ExtractError(f"party_order.js {label!r}: a species twice, sets cannot tell")
        party = []
        for name in order:
            where = f"sets.js {name!r} {label!r}"
            if label not in sets.get(name, {}):
                raise ExtractError(f"party_order.js {label!r}: no set for {name!r}")
            used.add((name, label))
            s = sets[name][label]
            species, form = _lookup(species_by_name, name, where, "species")
            if s["nature"] not in natures:
                raise ExtractError(f"{where}: unknown nature {s['nature']!r}")
            party.append(
                {
                    "species": species,
                    "form": form,
                    "level": s["level"],
                    "item": items(s["item"], where) if s.get("item") else None,
                    "ability": abilities(s["ability"], where),
                    "nature": s["nature"],
                    "gender": _lookup(GENDERS, s["gender"], where, "gender"),
                    "ivs": [s["ivs"][k] for k in IV_KEYS],
                    "moves": [moves(m, where) for m in s["moves"]],
                }
            )
        out[label] = {"party": party}
    unused = sorted((n, lab) for n, by_label in sets.items() for lab in by_label)
    unused = [x for x in unused if x not in used]
    if unused:
        raise ExtractError(
            f"sets.js: {len(unused)} set(s) in no party_order.js party: {unused[:5]}"
        )
    return out


def _ai_flags(raw: dict, trainers: dict) -> dict:
    out = {}
    for label, flags in raw["TRAINER_AI_FLAGS_PK"].items():
        where = f"trainer_flags.js {label!r}"
        if label not in trainers:
            raise ExtractError(f"{where}: not a party_order.js label")
        if set(flags) != set(AI_FLAG_KEYS) or set(flags.values()) - {0, 1}:
            raise ExtractError(f"{where}: flags {flags!r} are not the {len(AI_FLAG_KEYS)} bits")
        out[label] = sum(1 << bit for bit, k in enumerate(AI_FLAG_KEYS) if flags[k])
    return out


def _flags(raw: dict) -> dict:
    """flags.js: battle formats (line 3), hidden trainers (405) and weather (515)."""
    flags = raw["FLAGS_PK"]
    if set(flags) != {"battleType", "hidden", "weather"}:
        raise ExtractError(f"flags.js: FLAGS_PK keys {sorted(flags)}")
    battle = flags["battleType"]
    if set(battle) != {"double", "trueDouble", "tag"}:
        raise ExtractError(f"flags.js: battleType keys {sorted(battle)}")
    if any(v != {} for v in battle["trueDouble"].values()):
        raise ExtractError("flags.js: a battleType.trueDouble entry is not {}")

    def pairs(kind: str, keys: tuple[str, ...]) -> list[list[str]]:
        rows = []
        for entry in battle[kind]:
            if set(entry) != set(keys):
                raise ExtractError(f"flags.js: battleType.{kind} entry {entry!r}")
            rows.append([entry[k] for k in keys])
        return rows

    return {
        "formats": {
            "true_double": list(battle["trueDouble"]),
            "double": pairs("double", ("enemy1", "enemy2")),
            "tag": pairs("tag", ("enemy1", "enemy2", "partner")),
        },
        "hidden": list(flags["hidden"]),
        "weather": {kind: list(labels) for kind, labels in flags["weather"].items()},
    }


def normalise(raw: dict, sha256: str) -> dict:
    """The ``pkcalc.json`` schema from the evaluated globals.

    ``moves[num]``: name, aliases (shortName, calcName), type (TYPE_*), category (CLASS_*),
    power, accuracy, pp, chance, priority, target (PKCalc's string), effect_text, machine
    (TM n > 0, HM n < 0), tutor (location or null), price (shard counts, red/blue/yellow/
    green). ``species[personal key]`` (the personal row as a string, forms by Ruling 4):
    num, form, name, base_stats, types (two, repeated when one), abilities (two, 0 pads),
    catch_rate, exp_yield, growth_rate (EXP_RATE_*), gender_ratio (personal byte),
    evolutions, level_up [[level, move]], tm, tutor. ``items``/``abilities[num]``: name,
    aliases, desc. ``trainers[label]``: party [{species, form, level, item, ability, nature,
    gender, ivs (ids.Stat order), moves}], empty rival templates with ``template: true``.
    ``ai_flags[label]``: bit mask, ``AI_FLAG_KEYS`` order. ``formats``, ``hidden``,
    ``weather``: flags.js lists. ``flag_labels_without_party``: flags.js labels that name
    no party_order.js trainer. ``tm_table[machine index]`` (TM01-TM92 = 1-92, HM01-HM08 =
    93-100) = move. ``tutor_table``: [{move, cost, location}] in move-number order, with
    no tutor slot (PKCalc has none).
    """
    species, species_by_name = _species(raw)
    moves, tm_table, tutor_table = _moves(raw)
    trainers = _trainers(raw, species_by_name)
    flags = _flags(raw)
    named = set(flags["hidden"]) | {x for v in flags["weather"].values() for x in v}
    named |= set(flags["formats"]["true_double"])
    named |= {x for kind in ("double", "tag") for row in flags["formats"][kind] for x in row}
    return {
        "sha256": sha256,
        "moves": moves,
        "species": species,
        "items": {
            str(i["num"]): {"name": i["name"], "aliases": [i["calcName"]], "desc": i["desc"]}
            for i in raw["ITEMS"].values()
        },
        "abilities": {
            str(a["num"]): {"name": a["name"], "aliases": [a["calcName"]], "desc": a["desc"]}
            for a in raw["ABILITIES"].values()
        },
        "trainers": trainers,
        "ai_flags": _ai_flags(raw, trainers),
        **flags,
        "flag_labels_without_party": sorted(named - set(trainers)),
        "tm_table": tm_table,
        "tutor_table": tutor_table,
    }


# --- CLI --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--zip", type=Path, help="PKCalc zip (default: pkcalc_zip of local.toml)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        zip_path = args.zip or load_inputs().pkcalc_zip
        if zip_path is None:
            raise RomInputError("pkcalc_zip is not set (data/local.toml or NUZLINE_PKCALC_ZIP)")
        data = extract_pkcalc(zip_path, args.out)
    except (RomInputError, ExtractError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    real = [t for t in data["trainers"].values() if not t.get("template")]
    print(f"sha256: {data['sha256']}")
    print(
        f"trainers: {len(real)} ({sum(len(t['party']) for t in real)} sets), "
        f"templates: {len(data['trainers']) - len(real)}; moves: {len(data['moves'])}; "
        f"species: {len(data['species'])}"
    )
    if data["flag_labels_without_party"]:
        print(f"flags.js labels without a party: {data['flag_labels_without_party']}")
    print(f"wrote {args.out / 'pkcalc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
