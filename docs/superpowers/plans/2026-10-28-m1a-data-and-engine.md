# Milestone 1a: Data Tables, JAX Engine and Reference Engine (Roark slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data pipeline, the JAX battle engine and the plain-Python reference engine for every mechanic that appears in Roark's fight against a fixed sample team, with differential and decomp-derived tests passing.

**Architecture:** Offline tools extract the decomp's data into intermediate JSON, then build versioned `.npz` tables. The engine is a fixed-shape `BattleState` pytree, a `step` that runs every phase masked, an explicit `Outcome` input for chance, and a data-driven `EffectId` dispatch. A branchy reference engine with the same API is the oracle for Hypothesis-driven differential tests. Trainer AI, coverage registry and search are the next plan (`2026-10-28-m1b-roark-ai-and-ranking.md`); this plan ends with an engine that can be stepped by hand with any actions.

**Tech Stack:** Python 3.12, uv, `jax[cpu]`, numpy, pytest, hypothesis, ruff. External: a clone of `pret/pokeplatinum` at `external/pokeplatinum` (gitignored).

**Spec:** `docs/design.md` §3 (Decisions), §4.1 (Data layer), §4.2 (Engine layer), §5 (Correctness and testing), §8 milestone 1. Requires the milestone 0 plan to be done (scaffold exists).

## Global Constraints

- The decomp (`pret/pokeplatinum`) is authoritative; where this plan quotes a formula and the decomp disagrees, the decomp wins and the plan's test values are corrected with a comment citing the decomp file (spec §3 Sources).
- Never approximate silently: any move, ability or item outside the implemented set maps to `EffectId.UNSUPPORTED` / `AbilityId.UNSUPPORTED` / `ItemId.UNSUPPORTED` (spec §3 Unsupported mechanics).
- State storage dtypes `int8`/`int16`; all damage-calc intermediates `int32`; no 64-bit anywhere (spec §4.2).
- `step` is pure, jit-able and vmap-able, with fixed shapes (spec §5 JAX hygiene).
- Sampling and (later) enumeration share the one `step`; `step` draws no randomness itself (spec §4.2).
- A player faint is terminal for the primary objective: `step` sets `over = 1, winner = 1` when any player Pokémon faints (spec §3 Terminal states). Secondary objectives are out of this plan's scope.
- Package layout per spec §7: `src/nuzline/{data,engine,reference}`, `tools/`, `data/gen4_platinum/`, `tests/`.
- **Action encoding (this plan fixes it; spec §4.2 says 9 actions):** actions are `int32` scalars `0..9`: `0–3` = use move slot, `4–9` = switch to party slot `action - 4`. The active slot's own switch is always illegal, so 9 legal actions remain. Update spec §4.2 to say so in Task 9.
- Sample player team (fixed for all tests in this plan), in party order, with explicit movesets (a nuzlocker may keep older moves, so these are not the "last four learned"): Monferno L14 Scratch/Leer/Ember/Mach Punch (Blaze); Prinplup L14 Pound/Growl/Bubble/Water Sport (Torrent); Grotle L14 Tackle/Withdraw/Absorb/Razor Leaf (Overgrow, holds Sitrus Berry); Starly L10 Tackle/Growl/Quick Attack/Wing Attack (Keen Eye, holds Oran Berry); Shinx L13 Tackle/Leer/Spark (Rivalry); Budew L10 Absorb/Growth/Water Sport/Stun Spore (Natural Cure). Neutral natures, IVs 20, EVs 0, all male. Stored as `tests/fixtures/sample_party.toml` in Task 5. Every move here must map to a non-`UNSUPPORTED` effect in Task 4 (Growth is `STAT_SELF (SPA, +1)` in Gen 4; Quick Attack and Mach Punch are `PLAIN` with priority 1).

## Review Focus

1. A Pokémon with current HP 0 in the party file must load as fainted and be an illegal switch target, not crash `make_state`. Test in Task 6.
2. A move with 0 PP must be illegal, and Struggle must be chosen when all four are 0 PP, rather than indexing move slot 0. Test in Task 9.
3. Speed ties must use the `speed_tie` outcome and never silently favor side 0. Test in Task 8a.
4. Stealth Rock on switch-in must be able to faint the incoming Pokémon and set the terminal flag before it acts. Test in Task 8c.
5. A trainer party JSON with a difficulty value of 255 must produce IVs of exactly 31 (integer `255*31//255`), and 0 must produce 0, with no float rounding. Test in Task 5.

---

### Task 1: Locate the decomp sources

**Files:**
- Create: `docs/decomp_paths.md`
- Create: `tools/clone_decomp.sh`

**Interfaces:**
- Produces: `docs/decomp_paths.md`, a table every later extraction task reads. Columns: *What*, *Path in external/pokeplatinum*, *Format*, *Notes*. Required rows: species data; move data; type chart; natures; items; abilities; level-up learnsets; TM/HM compatibility; trainer parties; trainer AI flags and scripts; the function that builds a trainer's party (IVs from difficulty, PID); the battle-state structs (per-Pokémon, per-side, field/context); the damage calculation function; the stat calculation function; the end-of-turn effect order; the move effect id list; any overworld-to-battle weather/fog hand-off.

- [ ] **Step 1: Write `tools/clone_decomp.sh`**

Clones `https://github.com/pret/pokeplatinum` into `external/pokeplatinum` at a pinned commit (record the hash in the script) if the directory is absent; prints the commit otherwise.

- [ ] **Step 2: Run it**

Run: `bash tools/clone_decomp.sh`
Expected: `external/pokeplatinum` exists and the pinned hash is printed.

- [ ] **Step 3: Fill `docs/decomp_paths.md`**

Use `rg` over `external/pokeplatinum` to find each row. Add a final section **Completeness** stating, with file references, whether the battle engine and the AI script interpreter are fully decompiled and named, and listing any `sub_XXXXXXXX`-style unnamed functions on the damage, end-of-turn or AI paths. This section discharges spec §10 "Decomp completeness".

- [ ] **Step 4: Verify every required row is filled**

Run: `grep -c '^|' docs/decomp_paths.md`
Expected: at least 18 (header rows plus 16 required rows).

- [ ] **Step 5: Commit**

```bash
git add docs/decomp_paths.md tools/clone_decomp.sh
git commit -m "docs: map decomp source locations for data, battle and AI"
```

---

### Task 2: Extract static game data to intermediate JSON

**Files:**
- Create: `tools/extract_decomp.py`
- Create: `tools/__init__.py` (empty)
- Test: `tests/tools/test_extract_decomp.py`

**Interfaces:**
- Produces: `extract_static(decomp_root: Path, out_dir: Path) -> None` writing to `out_dir`:
  - `species.json`: list indexed by the decomp's species id (index 0 = none) of `{"name": str, "base_stats": [hp, atk, def, spa, spd, spe], "types": [t1, t2], "abilities": [a1, a2], "gender_ratio": int, "weight_hg": int, "base_friendship": int}`.
  - `moves.json`: list indexed by move id of `{"name", "power", "type", "category" (0 physical, 1 special, 2 status), "accuracy", "pp", "priority", "target", "effect" (decomp effect number), "effect_chance", "flags": [str, ...]}`.
  - `types.json`: `{"names": [...], "chart": [[...]]}` with the decomp's own multiplier encoding (expected `0, 5, 10, 20`; confirm from the decomp and record in the file).
  - `natures.json`: list of 25 `{"name", "up": stat_index|null, "down": stat_index|null}`.
  - `items.json`: list indexed by item id of `{"name", "hold_effect", "hold_param"}`.
  - `abilities.json`: list indexed by ability id of `{"name"}`.
  - `learnsets.json`: `{species_id: [[level, move_id], ...]}`.
  - CLI: `uv run python -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_extract_decomp.py
import json, pytest
from pathlib import Path
from tools.extract_decomp import extract_static

DECOMP = Path("external/pokeplatinum")
pytestmark = pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")

@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("inter")
    extract_static(DECOMP, d)
    return d

def test_bulbasaur(out):
    sp = json.loads((out / "species.json").read_text())
    assert sp[1]["name"].lower() == "bulbasaur"
    assert sp[1]["base_stats"] == [45, 49, 49, 65, 65, 45]

def test_tackle_is_gen4_tackle(out):
    mv = json.loads((out / "moves.json").read_text())
    tackle = next(m for m in mv if m["name"].lower() == "tackle")
    assert (tackle["power"], tackle["accuracy"], tackle["pp"], tackle["category"]) == (35, 95, 35, 0)

def test_type_chart_fire_grass_and_ghost_normal(out):
    t = json.loads((out / "types.json").read_text())
    n = [x.lower() for x in t["names"]]
    assert t["chart"][n.index("fire")][n.index("grass")] == 20
    assert t["chart"][n.index("ghost")][n.index("normal")] == 0

def test_natures_count_and_adamant(out):
    nat = json.loads((out / "natures.json").read_text())
    assert len(nat) == 25
    adamant = next(x for x in nat if x["name"].lower() == "adamant")
    assert (adamant["up"], adamant["down"]) == (1, 3)

def test_learnset_chimchar_has_ember_at_7(out):
    ls = json.loads((out / "learnsets.json").read_text())
    sp = json.loads((out / "species.json").read_text())
    mv = json.loads((out / "moves.json").read_text())
    chimchar = next(i for i, s in enumerate(sp) if s["name"].lower() == "chimchar")
    ember = next(i for i, m in enumerate(mv) if m["name"].lower() == "ember")
    assert [7, ember] in ls[str(chimchar)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_extract_decomp.py -v`
Expected: FAIL with `ModuleNotFoundError: tools.extract_decomp`.

- [ ] **Step 3: Implement `tools/extract_decomp.py`**

Parse the formats named in `docs/decomp_paths.md`. One function per output file; `extract_static` calls them all. Preserve the decomp's numeric ids as list indices. If a file is JSON in the decomp, load it; if it is C or assembly, parse with regexes anchored on the decomp's macro names, and fail loudly (raise with the line) on anything unparsed rather than skipping.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_extract_decomp.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Run the CLI and commit the tool (intermediate output is gitignored)**

```bash
uv run python -m tools.extract_decomp --decomp external/pokeplatinum --out data/intermediate
git add tools/__init__.py tools/extract_decomp.py tests/tools/test_extract_decomp.py
git commit -m "feat(data): extract species, moves, types, natures, items, abilities, learnsets from decomp"
```

---

### Task 3: Extract trainers and battle start conditions

**Files:**
- Modify: `tools/extract_decomp.py`
- Test: `tests/tools/test_extract_trainers.py`

**Interfaces:**
- Produces: `extract_trainers(decomp_root: Path, out_dir: Path) -> None` writing `trainers.json`: list of `{"id": int, "name": str, "class": str, "ai_flags": int, "items": [item_id, ...], "party": [{"species": int, "level": int, "difficulty": int, "moves": [m0, m1, m2, m3] | null, "item": int, "form": int}], "start_condition": "none" | "rain" | "hail" | "sandstorm" | "sun" | "fog"}`. `moves: null` means the decomp gives the species' default moveset at that level (the party builder in Task 5 fills it from `learnsets.json`). The CLI from Task 2 also writes this file.
- `start_condition` is derived from the map the trainer stands on, via the hand-off found in Task 1. If no automatic derivation is possible, a hand-maintained `tools/start_conditions.json` (`{trainer_id: condition}`) is the source and the test below still passes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_extract_trainers.py
import json, pytest
from pathlib import Path
from tools.extract_decomp import extract_trainers

DECOMP = Path("external/pokeplatinum")
pytestmark = pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")

@pytest.fixture(scope="module")
def trainers(tmp_path_factory):
    d = tmp_path_factory.mktemp("inter")
    extract_trainers(DECOMP, d)
    return json.loads((d / "trainers.json").read_text())

def roark(trainers):
    return [t for t in trainers if t["name"].lower() == "roark" and t["class"].lower().startswith("leader")][0]

def test_roark_party_shape(trainers):
    r = roark(trainers)
    assert [p["level"] for p in r["party"]] == [12, 12, 14]
    assert r["ai_flags"] != 0

def test_roark_start_condition_is_none(trainers):
    assert roark(trainers)["start_condition"] == "none"

def test_some_trainer_has_hail(trainers):
    assert any(t["start_condition"] == "hail" for t in trainers)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_extract_trainers.py -v`
Expected: FAIL with `ImportError: extract_trainers`.

- [ ] **Step 3: Implement `extract_trainers`**

Read the trainer and trainer-party files named in `docs/decomp_paths.md`. Roark's gym fight is the first `leader`-class Roark entry (there is a later Battleground rematch). If Roark's party levels in the decomp differ from `[12, 12, 14]`, the decomp wins: fix the test and cite the file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tools/test_extract_trainers.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/extract_decomp.py tools/start_conditions.json tests/tools/test_extract_trainers.py
git commit -m "feat(data): extract trainers, AI flags and battle start conditions"
```

---

### Task 4: Effect map, table build and table loading

**Files:**
- Create: `src/nuzline/engine/ids.py`
- Create: `tools/effect_map.py`
- Create: `tools/build_tables.py`
- Create: `src/nuzline/data/__init__.py` (empty)
- Create: `src/nuzline/data/tables.py`
- Test: `tests/data/test_tables.py`

**Interfaces:**
- `src/nuzline/engine/ids.py` produces three `IntEnum`s. These are the engine's vocabulary; every later task uses these names.
  - `EffectId`: `UNSUPPORTED=0, PLAIN=1, STAT_SELF=2, STAT_FOE=3, SECONDARY_STAT_SELF=4, SECONDARY_STAT_FOE=5, STATUS_FOE=6, SECONDARY_STATUS=7, FLINCH=8, HIGH_CRIT=9, DRAIN=10, STEALTH_ROCK=11, TRAP=12, PURSUIT=13, TAUNT=14, CONFUSE_FOE=15, LEECH_SEED=16, MULTI_HIT=17, WATER_SPORT=18, RECOIL=19`. Params (`move_param[m, 0..2]`): stat effects `(stat_index, stages, 0)`; status effects `(status_id, 0, 0)`; `DRAIN (percent,0,0)`; `RECOIL (percent,0,0)`; others zero. `effect_chance` is the move's own column.
  - `AbilityId`: `UNSUPPORTED=0, NONE=1, STURDY, ROCK_HEAD, MOLD_BREAKER, BLAZE, TORRENT, OVERGROW, INTIMIDATE, RIVALRY, KEEN_EYE, SIMPLE, UNAWARE, NATURAL_CURE, POISON_POINT, INNER_FOCUS, STATIC, GUTS, STEADFAST`.
  - `ItemId`: `UNSUPPORTED=0, NONE=1, ORAN_BERRY, SITRUS_BERRY`.
  - `Status`: `NONE=0, SLEEP=1, POISON=2, BURN=3, FREEZE=4, PARALYSIS=5, TOXIC=6`.
  - `Stat`: `HP=0, ATK=1, DEF=2, SPA=3, SPD=4, SPE=5, ACC=6, EVA=7`.
- `tools/effect_map.py` produces `DECOMP_EFFECT_TO_ID: dict[int, tuple[EffectId, tuple[int,int,int]]]`, `DECOMP_ABILITY_TO_ID: dict[str, AbilityId]`, `DECOMP_ITEM_TO_ID: dict[str, ItemId]`. Hand-maintained; anything absent maps to the `UNSUPPORTED` member.
- `tools/build_tables.py` produces `build(intermediate_dir: Path, out_dir: Path) -> None` writing `out_dir/tables.npz` and `out_dir/manifest.json` (`{"version": 1, "decomp_commit": str, "species_names": [...], "move_names": [...], "item_names": [...], "ability_names": [...], "type_names": [...], "nature_names": [...]}`). CLI `uv run python -m tools.build_tables --inter data/intermediate --out data/gen4_platinum`.
- `src/nuzline/data/tables.py` produces:
  - `class Tables(NamedTuple)`: `species_base_stats int16[S,6]`, `species_types int8[S,2]`, `species_abilities int16[S,2]` (already mapped to `AbilityId`), `species_weight int32[S]`, `species_gender int16[S]`, `species_friendship int8[S]`, `move_power int16[M]`, `move_type int8[M]`, `move_category int8[M]`, `move_accuracy int8[M]`, `move_pp int8[M]`, `move_priority int8[M]`, `move_target int8[M]`, `move_effect int16[M]` (`EffectId`), `move_effect_chance int8[M]`, `move_param int16[M,3]`, `move_flags int16[M]` (bitmask: `CONTACT=1, PROTECT=2, SOUND=4, PUNCH=8`), `type_chart int8[T,T]`, `nature_mods int8[25,6]` (−1/0/+1 per stat), `item_effect int16[I]` (`ItemId`), `item_param int16[I,2]`.
  - `load_tables(dir: Path = DEFAULT) -> Tables` (jnp arrays) and `class Names` with `species(name) -> int`, `move(name) -> int`, `item(name) -> int`, `ability(name) -> int` (case-insensitive; `item("")` and `move("")` return 0 meaning none), `load_names(dir) -> Names`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_tables.py
import jax.numpy as jnp, pytest
from pathlib import Path
from nuzline.data.tables import load_tables, load_names
from nuzline.engine.ids import EffectId, AbilityId, ItemId

TABLES = Path("data/gen4_platinum")
pytestmark = pytest.mark.skipif(not (TABLES / "tables.npz").exists(), reason="tables not built")

def test_dtypes_and_shapes():
    t = load_tables(TABLES)
    assert t.species_base_stats.dtype == jnp.int16 and t.species_base_stats.shape[1] == 6
    assert t.move_param.shape[1] == 3 and t.move_param.dtype == jnp.int16
    assert t.type_chart.shape[0] == t.type_chart.shape[1] == 17
    assert all(x.dtype.itemsize <= 4 for x in t)

def test_effect_mapping():
    t, n = load_tables(TABLES), load_names(TABLES)
    assert int(t.move_effect[n.move("Tackle")]) == EffectId.PLAIN
    assert int(t.move_effect[n.move("Stealth Rock")]) == EffectId.STEALTH_ROCK
    assert int(t.move_effect[n.move("Leer")]) == EffectId.STAT_FOE
    assert tuple(int(x) for x in t.move_param[n.move("Leer")]) == (2, -1, 0)
    assert int(t.move_effect[n.move("Metronome")]) == EffectId.UNSUPPORTED

def test_ability_and_item_mapping():
    t, n = load_tables(TABLES), load_names(TABLES)
    assert int(t.species_abilities[n.species("Geodude"), 0]) == AbilityId.ROCK_HEAD
    assert int(t.item_effect[n.item("Oran Berry")]) == ItemId.ORAN_BERRY
    assert int(t.item_effect[n.item("Leftovers")]) == ItemId.UNSUPPORTED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_tables.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ids.py`, `effect_map.py`, `build_tables.py`, `tables.py`**

`build_tables` stacks the JSON lists into numpy arrays with the dtypes above and maps effects/abilities/items through `effect_map`. `load_tables` wraps each array in `jnp.asarray`. Geodude's first ability in Gen 4 is Rock Head and second is Sturdy; the mapping dict keys are the decomp's ability names.

- [ ] **Step 4: Build the tables, run tests to verify they pass**

```bash
uv run python -m tools.build_tables --inter data/intermediate --out data/gen4_platinum
uv run pytest tests/data/test_tables.py -v
```
Expected: 3 PASS. Note `data/gen4_platinum/*.npz` is gitignored; `manifest.json` is committed.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/ids.py tools/effect_map.py tools/build_tables.py src/nuzline/data tests/data/test_tables.py data/gen4_platinum/manifest.json
git commit -m "feat(data): effect/ability/item id maps, table build and loading"
```

---

### Task 5: Stat calculation, Pokémon specs, trainer party build and party file

**Files:**
- Create: `src/nuzline/engine/stats.py`
- Create: `src/nuzline/data/spec.py`
- Create: `src/nuzline/data/trainer.py`
- Create: `src/nuzline/data/party.py`
- Create: `tests/fixtures/sample_party.toml`
- Test: `tests/engine/test_stats.py`, `tests/data/test_trainer.py`, `tests/data/test_party.py`

**Interfaces:**
- `stats.py`: `calc_stats(base: int[6], ivs: int[6], evs: int[6], level: int, nature_mods: int[6]) -> np.int32[6]` (plain numpy, used at state construction). Formulas (confirm against the decomp's stat function): `HP = (2*base + iv + ev//4) * level // 100 + level + 10`; other stats `((2*base + iv + ev//4) * level // 100 + 5)`, then `* 110 // 100` if nature up, `* 90 // 100` if nature down (integer truncation).
- `spec.py`: `@dataclass(frozen=True) class PokemonSpec`: `species: int, level: int, nature: int, ivs: tuple[int,6], evs: tuple[int,6], moves: tuple[int,4]` (0 = empty), `item: int, ability_slot: int (0|1), gender: int (0 male, 1 female, 2 none), friendship: int, form: int = 0, current_hp: int | None = None, status: int = 0`.
- `trainer.py`: `load_trainers(dir: Path) -> list[Trainer]` (`Trainer` dataclass mirroring the JSON), `difficulty_to_ivs(difficulty: int) -> int` = `difficulty * 31 // 255`, `build_trainer_party(trainer: Trainer, inter: IntermediateData) -> list[PokemonSpec]`: IVs all `difficulty_to_ivs(difficulty)`, EVs 0, PID from the decomp's party-build function (Task 1 row) giving nature `pid % 25`, gender from `pid & 0xFF` versus gender ratio, ability slot `pid & 1`; default moves from the learnset when `moves` is null (last four learned at or below the level, in learn order). `IntermediateData` is a thin loader of the JSON files from Task 2.
- `party.py`: `load_party(path: Path, names: Names) -> list[PokemonSpec]` reading TOML: `[[pokemon]] species = "Monferno" level = 14 nature = "Hardy" ivs = [20,20,20,20,20,20] evs = [0,0,0,0,0,0] moves = ["Scratch","Leer","Ember","Mach Punch"] item = "" ability = "Blaze" gender = "male" hp = 0` (`hp` optional, omitted means full).

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_stats.py
import numpy as np
from nuzline.engine.stats import calc_stats

def test_level_100_neutral():
    s = calc_stats(np.full(6, 100), np.full(6, 31), np.full(6, 252), 100, np.zeros(6, int))
    assert s.tolist() == [404, 299, 299, 299, 299, 299]

def test_adamant_level_100():
    s = calc_stats(np.full(6, 100), np.full(6, 31), np.full(6, 252), 100, np.array([0, 1, 0, -1, 0, 0]))
    assert s[1] == 328 and s[3] == 269

def test_low_level_truncation():
    s = calc_stats(np.array([44, 58, 44, 58, 44, 61]), np.full(6, 20), np.zeros(6, int), 14, np.zeros(6, int))
    assert s[0] == (2 * 44 + 20) * 14 // 100 + 14 + 10
```

```python
# tests/data/test_trainer.py
import pytest
from pathlib import Path
from nuzline.data.trainer import load_trainers, build_trainer_party, IntermediateData

INTER = Path("data/intermediate")
pytestmark = pytest.mark.skipif(not INTER.exists(), reason="intermediate data not extracted")

def test_difficulty_to_ivs_exact_integer():
    from nuzline.data.trainer import difficulty_to_ivs
    assert (difficulty_to_ivs(255), difficulty_to_ivs(0), difficulty_to_ivs(128)) == (31, 0, 15)

def test_party_ivs_follow_difficulty():
    inter = IntermediateData(INTER)
    roark = next(t for t in load_trainers(INTER) if t.name.lower() == "roark")
    party = build_trainer_party(roark, inter)
    for mon, raw in zip(party, roark.party):
        assert mon.ivs == (raw.difficulty * 31 // 255,) * 6
        assert mon.evs == (0,) * 6

def test_roark_party_has_four_moves_each():
    inter = IntermediateData(INTER)
    roark = next(t for t in load_trainers(INTER) if t.name.lower() == "roark")
    for mon in build_trainer_party(roark, inter):
        assert all(m != 0 for m in mon.moves) or mon.level < 5
```

```python
# tests/data/test_party.py
from pathlib import Path
from nuzline.data.tables import load_names
from nuzline.data.party import load_party

def test_sample_party_loads():
    n = load_names(Path("data/gen4_platinum"))
    p = load_party(Path("tests/fixtures/sample_party.toml"), n)
    assert len(p) == 6 and p[0].species == n.species("Monferno") and p[0].level == 14
    assert p[3].item == n.item("Oran Berry") and p[2].item == n.item("Sitrus Berry")

def test_fainted_party_member_keeps_hp_zero():
    n = load_names(Path("data/gen4_platinum"))
    p = load_party(Path("tests/fixtures/sample_party_fainted.toml"), n)
    assert p[5].current_hp == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_stats.py tests/data/test_trainer.py tests/data/test_party.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the four modules and write the fixtures**

`tests/fixtures/sample_party.toml` is the sample team from Global Constraints with movesets taken from `learnsets.json` (last four moves at the given level). `tests/fixtures/sample_party_fainted.toml` is the same with `hp = 0` on Budew. Use `tomllib` from the standard library.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_stats.py tests/data/test_trainer.py tests/data/test_party.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/stats.py src/nuzline/data/spec.py src/nuzline/data/trainer.py src/nuzline/data/party.py tests/fixtures tests/engine/test_stats.py tests/data/test_trainer.py tests/data/test_party.py
git commit -m "feat(data): stat calc, Pokemon specs, trainer party build, party file loader"
```

---

### Task 6: Battle state and state construction

**Files:**
- Create: `src/nuzline/engine/STATE.md`
- Create: `src/nuzline/engine/__init__.py` (empty)
- Create: `src/nuzline/engine/state.py`
- Test: `tests/engine/test_state.py`

**Interfaces:**
- `STATE.md`: a table with one row per field of each decomp battle struct found in Task 1 (`Struct.field | pytree field | status`), where status is `mapped`, `later` (with the milestone) or `not needed because …`. Every field of the three structs appears.
- `state.py` produces `class BattleState(NamedTuple)` (unbatched; side axis 0 = player, 1 = opponent):
  - per side × slot `[2,6]`: `species int16, form int8, level int8, hp int16, max_hp int16, stats int16[2,6,5]` (atk, def, spa, spd, spe), `types int8[2,6,2], ability int16, item int16, item_consumed int16, moves int16[2,6,4], pp int8[2,6,4], status int8, status_counter int8, gender int8, friendship int8, hp_iv int8` (for later Hidden Power: `ivs int8[2,6,6]`);
  - per side active `[2]`: `active int8, stages int8[2,8], confusion int8, leech_seed int8, substitute_hp int16, taunt int8, trapped int8, trap_move int16, flinched int8, choice_lock int16, last_move int16, protect_count int8, rollout_count int8, charge_turn int8, recharge int8, pursuit_target int8, water_sport int8`;
  - per side `[2]`: `spikes int8, toxic_spikes int8, stealth_rock int8, reflect int8, light_screen int8, safeguard int8, mist int8, lucky_chant int8, tailwind int8, wish_timer int8, wish_hp int16, future_timer int8, future_damage int16, trainer_items int16[2,4]`;
  - field (scalars): `weather int8, weather_timer int8, trick_room int8, gravity int8, mud_sport int8, turn int16, over int8, winner int8` (0 none, 1 player lost or any player faint, 2 player won), `player_faints int8`;
  - AI knowledge: `ai_seen_moves int8[6,4]` (player slot × move slot), `ai_ability_known int8[6]`.
  - `Weather` IntEnum in `ids.py`: `NONE=0, SUN=1, RAIN=2, SANDSTORM=3, HAIL=4, FOG=5`.
  - `make_state(tables: Tables, player: list[PokemonSpec], opp: list[PokemonSpec], start_condition: int = Weather.NONE) -> BattleState`: computes stats via `calc_stats`, fills empty slots with `species = 0, hp = 0`, sets `active = 0` for both sides, `hp = current_hp` when given, `weather = start_condition` with `weather_timer = 0` meaning permanent.
  - `empty_outcome()` lives in Task 8a; this task does not need it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_state.py
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from nuzline.data.tables import load_tables, load_names
from nuzline.data.party import load_party
from nuzline.engine.state import make_state, BattleState
from nuzline.engine.ids import Weather

T = load_tables(Path("data/gen4_platinum")); N = load_names(Path("data/gen4_platinum"))

def _teams():
    p = load_party(Path("tests/fixtures/sample_party.toml"), N)
    return p, p[:3]

def test_shapes_dtypes_and_no_64bit():
    s = make_state(T, *_teams())
    assert s.hp.shape == (2, 6) and s.hp.dtype == jnp.int16
    assert s.stats.shape == (2, 6, 5) and s.moves.shape == (2, 6, 4)
    assert all(x.dtype.itemsize <= 4 for x in s)

def test_empty_slots_are_zero_and_full_hp_default():
    p, o = _teams()
    s = make_state(T, p, o)
    assert int(s.species[1, 5]) == 0 and int(s.hp[1, 5]) == 0
    assert int(s.hp[0, 0]) == int(s.max_hp[0, 0]) > 0

def test_fainted_member_and_start_weather():
    p = load_party(Path("tests/fixtures/sample_party_fainted.toml"), N)
    s = make_state(T, p, p[:2], start_condition=Weather.HAIL)
    assert int(s.hp[0, 5]) == 0 and int(s.max_hp[0, 5]) > 0
    assert int(s.weather) == Weather.HAIL and int(s.weather_timer) == 0

def test_state_is_a_valid_pytree_for_vmap():
    s = make_state(T, *_teams())
    batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), s)
    assert batched.hp.shape == (3, 2, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `STATE.md` from the decomp structs, then implement `state.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_state.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/STATE.md src/nuzline/engine/__init__.py src/nuzline/engine/state.py tests/engine/test_state.py
git commit -m "feat(engine): BattleState pytree, decomp struct checklist, make_state"
```

---

### Task 7: Damage calculation

**Files:**
- Create: `src/nuzline/engine/damage.py`
- Test: `tests/engine/test_damage.py`

**Interfaces:**
- `damage_table(tables: Tables, state: BattleState, atk_side: int, move: int, crit: bool) -> int32[16]` — damage for each of the 16 rolls against the defending side's active Pokémon; 0 for status moves, 0 when type effectiveness is 0.
- `damage_for_roll(tables, state, atk_side, move, crit, roll: int) -> int32` — the single value the engine uses.
- `effectiveness(tables, move_type: int, def_types: int8[2]) -> int32` — product of the two chart entries in the decomp's encoding (so neutral is `100` when the chart uses `10`).
- Gen 4 order, from the damage-formula research and to be confirmed line by line against the decomp's damage function (Task 1 row); the decomp wins:
  1. `level_factor = 2*level//5 + 2`; `A`, `D` = stat with stage multiplier (`max(2, 2+s) / max(2, 2-s)` truncated), Unaware ignores the other side's stages; burn halves physical `A` unless Guts; Guts ×1.5 on `A` when statused; ability base-power mods (none in M1).
  2. `base = (level_factor * power * A // D) // 50`.
  3. Reflect / Light Screen halve (not on crit); weather ×1.5 / ×0.5 for Fire/Water in sun/rain (Fog: none); Overgrow/Blaze/Torrent ×1.5 when `hp*3 <= max_hp` (applied to power at step 1 in the decomp; check).
  4. `+ 2`.
  5. Crit ×2 (×3 Sniper, not in M1).
  6. Item mods (none in M1), Me First (none).
  7. Random: `* (85 + roll) // 100`.
  8. STAB ×1.5.
  9. Type effectiveness, applied per type in sequence with truncation after each (×2, ×½ or 0).
  10. Rivalry ×1.25 / ×0.75 by gender, Mold Breaker ignores Sturdy (OHKO moves only in Gen 4, so no effect in M1).
  Every multiply is integer with truncation after each step.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_damage.py
import numpy as np, jax.numpy as jnp
from nuzline.engine.damage import damage_table, effectiveness
from tests.engine.helpers import synthetic_state  # built in this task: one active mon per side with given level/stats/types/moves

NEUTRAL_128 = [108,110,111,112,113,115,116,117,119,120,121,122,124,125,126,128]

def test_neutral_table_matches_hand_computation(tables):
    # L100 attacker, Atk 300 vs Def 200, 100 BP physical, no STAB, neutral type
    s, move = synthetic_state(tables, atk_level=100, atk_stats=[300,100,100,100,100], def_stats=[100,200,100,100,100],
                              move_power=100, move_physical=True, atk_types=("Normal","Normal"), def_types=("Normal","Normal"), move_type="Fighting")
    assert damage_table(tables, s, 0, move, crit=False).tolist() == NEUTRAL_128

def test_stab_applies_after_random(tables):
    s, move = synthetic_state(tables, atk_level=100, atk_stats=[300,100,100,100,100], def_stats=[100,200,100,100,100],
                              move_power=100, move_physical=True, atk_types=("Fighting","Normal"), def_types=("Normal","Normal"), move_type="Fighting")
    assert damage_table(tables, s, 0, move, crit=False).tolist() == [v * 3 // 2 for v in NEUTRAL_128]

def test_crit_doubles_before_random(tables):
    s, move = synthetic_state(tables, atk_level=100, atk_stats=[300,100,100,100,100], def_stats=[100,200,100,100,100],
                              move_power=100, move_physical=True, atk_types=("Normal","Normal"), def_types=("Normal","Normal"), move_type="Fighting")
    assert damage_table(tables, s, 0, move, crit=True).tolist() == [256 * (85 + r) // 100 for r in range(16)]

def test_immunity_and_effectiveness(tables, names):
    assert int(effectiveness(tables, names.type("Ghost"), jnp.array([names.type("Normal"), names.type("Normal")]))) == 0
    assert int(effectiveness(tables, names.type("Fire"), jnp.array([names.type("Grass"), names.type("Ice")]))) == 400

def test_status_move_deals_zero(tables, names):
    s, _ = synthetic_state(tables, atk_level=14, atk_stats=[40]*5, def_stats=[40]*5, move_power=0, move_physical=True,
                           atk_types=("Normal","Normal"), def_types=("Normal","Normal"), move_type="Normal")
    assert damage_table(tables, s, 0, names.move("Leer"), crit=False).tolist() == [0] * 16
```

`tests/engine/helpers.py` provides `synthetic_state(...)` (builds a `BattleState` via `make_state` from two one-member `PokemonSpec` lists, overriding stats directly) and a `tests/conftest.py` with `tables` and `names` fixtures loading `data/gen4_platinum`. `Names.type(name) -> int` is added to `tables.py` in this task.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_damage.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `damage.py`**

Pure `jnp` with `int32` throughout; stage multiplier via two lookup tables indexed by `stage + 6`. Read the decomp's damage function and annotate each line of `damage.py` with the decomp line it mirrors; where the order above is wrong, follow the decomp and correct the test's expected values with a comment.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_damage.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/damage.py tests/engine/test_damage.py tests/engine/helpers.py tests/conftest.py src/nuzline/data/tables.py
git commit -m "feat(engine): Gen 4 damage calculation with 16-roll table"
```

---

### Task 8a: Outcome struct, turn order, plain damage, faints and battle end

**Files:**
- Create: `src/nuzline/engine/outcome.py`
- Create: `src/nuzline/engine/order.py`
- Create: `src/nuzline/engine/step.py`
- Test: `tests/engine/test_step_basic.py`

**Interfaces:**
- `outcome.py` produces `class Outcome(NamedTuple)`: `speed_tie int8[]` (0 = player first), `hit int8[2,5]`, `crit int8[2,5]`, `roll int8[2,5]` (0–15), `secondary int8[2,5]` (the attacker's own secondary effect, indexed by attacking side), `contact_proc int8[2,5]` (the defender's contact ability, indexed by defending side), `n_hits int8[2]` (2–5), `full_para int8[2]`, `confusion_hit int8[2]`, `duration int8[2]` (sleep/confusion turns rolled at infliction), `thaw int8[2]`, `flinch_item int8[2]`, `quick_claw int8[2]`. `empty_outcome() -> Outcome` = all hits land, no crits, roll 15, no procs. `sample_outcome(key, tables, state, actions: int32[2]) -> Outcome` draws every field from its Gen 4 probability given the state (accuracy × stages, crit stage table `1/16, 1/8, 1/4, 1/3, 1/2`, 25 % full paralysis, 50 % confusion self-hit, 20 % thaw, sleep 2–5 turns, confusion 2–5 turns, multi-hit `2,2,3,3,4,5` weighting `3/8,3/8,1/8,1/8`).
- `order.py` produces `action_order(tables, state, actions, outcome) -> int8[2]` (side indices in execution order): switches before moves; higher priority first; then effective speed (stages, paralysis ×¼, Trick Room reversal); ties by `outcome.speed_tie`.
- `step.py` produces `step(tables: Tables, state: BattleState, actions: int32[2], outcome: Outcome) -> BattleState`, structured as the six phases in spec §4.2, each a function in `step.py` taking and returning `BattleState`. This task implements: switching (reset stages/volatiles, keep hazards; Stealth Rock damage comes in 8c), `PLAIN` moves with accuracy check against `outcome.hit[side,0]`, PP decrement, damage via `damage_for_roll`, faint check, `over`/`winner` with the terminal rule, and `turn + 1`. Struggle when the chosen slot has 0 PP (50 BP typeless recoil ¼ max HP, Gen 4).
- `needs_replacement(state) -> int8[2]` and `replace(state, side, slot) -> BattleState` for the faint-replacement decision point (opponent only, by the terminal rule).

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_step_basic.py
import jax, jax.numpy as jnp, numpy as np
from nuzline.engine.step import step, needs_replacement, replace
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.order import action_order
from tests.engine.helpers import roark_state  # sample party vs built Roark party, both active slot 0

def test_faster_side_moves_first_and_deals_damage(tables, names):
    s = roark_state(tables, names)       # Monferno (spe 14-level) vs Geodude
    o = empty_outcome()
    out = step(tables, s, jnp.array([names.move_slot(s, 0, "Scratch"), names.move_slot(s, 1, "Tackle")]), o)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])
    assert int(out.turn) == 1 and int(out.pp[0, 0, names.move_slot(s, 0, "Scratch")]) == int(s.pp[0, 0, names.move_slot(s, 0, "Scratch")]) - 1

def test_speed_tie_uses_outcome(tables, names):
    s = roark_state(tables, names, equal_speed=True)
    o = empty_outcome()
    first = action_order(tables, s, jnp.array([0, 0]), o)
    second = action_order(tables, s, jnp.array([0, 0]), o._replace(speed_tie=jnp.int8(1)))
    assert int(first[0]) == 0 and int(second[0]) == 1

def test_miss_deals_no_damage(tables, names):
    s = roark_state(tables, names)
    o = empty_outcome()._replace(hit=jnp.zeros((2, 5), jnp.int8))
    out = step(tables, s, jnp.array([0, 0]), o)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])

def test_player_faint_is_terminal(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1)   # helper: set player's active HP to 1
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.over) == 1 and int(out.winner) == 1 and int(out.player_faints) == 1

def test_opponent_faint_needs_replacement_then_replace(tables, names):
    s = roark_state(tables, names).replace_hp(1, 0, 1)
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.over) == 0
    assert needs_replacement(out).tolist() == [0, 1]
    out2 = replace(out, 1, 1)
    assert int(out2.active[1]) == 1 and needs_replacement(out2).tolist() == [0, 0]

def test_all_opponents_fainted_is_win(tables, names):
    s = roark_state(tables, names, opp_count=1).replace_hp(1, 0, 1)
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.over) == 1 and int(out.winner) == 2

def test_switch_goes_before_moves_and_resets_stages(tables, names):
    s = roark_state(tables, names).with_stages(0, atk=2)
    out = step(tables, s, jnp.array([4 + 1, 0]), empty_outcome())
    assert int(out.active[0]) == 1 and int(out.stages[0, 1]) == 0 and int(out.hp[0, 1]) < int(s.hp[0, 1])

def test_struggle_when_no_pp(tables, names):
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    out = step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) < int(s.hp[1, 0]) and int(out.hp[0, 0]) < int(s.hp[0, 0])
```

`helpers.py` gains `roark_state(tables, names, *, equal_speed=False, opp_count=3)` plus the small `BattleState` helper methods `replace_hp`, `with_stages`, `with_pp` (implemented as free functions in helpers and attached only in tests), and `Names.move_slot(state, side, name) -> int`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_step_basic.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `outcome.py`, `order.py`, `step.py` for the scope above**

All phases are functions of `(tables, state, actions, outcome, order)`; move execution runs for both actor positions in sequence with `jnp.where` masks for "actor fainted / flinched / battle over". Unimplemented `EffectId`s behave as `PLAIN` in this task; Task 8b–8e replace that.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_step_basic.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/outcome.py src/nuzline/engine/order.py src/nuzline/engine/step.py tests/engine/test_step_basic.py tests/engine/helpers.py
git commit -m "feat(engine): outcome struct, turn order, plain damage, faints, battle end, switching"
```

---

### Task 8b: Stat changes, status moves, secondary effects, flinch, high crit, drain, recoil, multi-hit

**Files:**
- Create: `src/nuzline/engine/effects.py`
- Modify: `src/nuzline/engine/step.py`
- Test: `tests/engine/test_effects.py`

**Interfaces:**
- `effects.py` produces `apply_move_effect(tables, state, side, move, outcome, hit_index) -> BattleState`, dispatching on `tables.move_effect[move]` with one function per `EffectId` listed in Task 4, each a masked update. Stage changes clamp to ±6; Simple doubles stage deltas; Keen Eye blocks accuracy drops; Inner Focus blocks flinch; Steadfast +1 Spe on flinch; status moves respect type immunities (Electric vs Ground for paralysis from Thunder Wave is not in M1; poison vs Steel/Poison, burn vs Fire, paralysis from Stun Spore none) and never overwrite an existing status. `MULTI_HIT` loops over `outcome.n_hits[side]` hits with per-hit `hit/crit/roll`, stopping on faint. `DRAIN` heals `damage * percent // 100` (min 1). `RECOIL` costs `damage * percent // 100` (Rock Head prevents). `HIGH_CRIT` raises the crit stage used by `sample_outcome` (so the outcome sampler reads it; the engine itself just reads `outcome.crit`).
- `step.py` calls `apply_move_effect` from the execution phase.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_effects.py
import jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.ids import Status
from tests.engine.helpers import roark_state, A  # A(player_move_name, opp_move_name, state, names) -> int32[2]

def test_leer_lowers_defense_and_clamps(tables, names):
    s = roark_state(tables, names).with_stages(1, def_=-6)
    out = step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(out.stages[1, 2]) == -6

def test_rock_polish_raises_speed_two(tables, names):
    s = roark_state(tables, names, opp_moves=["Rock Polish"])
    out = step(tables, s, A("Scratch", "Rock Polish", s, names), empty_outcome())
    assert int(out.stages[1, 5]) == 2

def test_ember_burn_only_on_secondary_proc(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos")
    no = step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    yes = step(tables, s, A("Ember", "Leer", s, names), empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8)))
    assert int(no.status[1, 0]) == Status.NONE and int(yes.status[1, 0]) == Status.BURN

def test_burn_halves_physical_damage(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos")
    burned = s.with_status(1, 0, Status.BURN)
    a, b = step(tables, s, A("Scratch", "Headbutt", s, names), empty_outcome()), step(tables, burned, A("Scratch", "Headbutt", s, names), empty_outcome())
    assert (int(s.hp[0, 0]) - int(b.hp[0, 0])) < (int(s.hp[0, 0]) - int(a.hp[0, 0]))

def test_headbutt_flinch_blocks_slower_target(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_faster=True)
    out = step(tables, s, A("Scratch", "Headbutt", s, names), empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8)))
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])          # Monferno flinched
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0])     # and spent no PP

def test_inner_focus_ignores_flinch(tables, names):
    s = roark_state(tables, names, player_species="Zubat", opp_species="Cranidos", opp_faster=True)
    out = step(tables, s, A("Bite", "Headbutt", s, names), empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8)))
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])

def test_absorb_heals_half_damage_dealt(tables, names):
    s = roark_state(tables, names, player_species="Budew").replace_hp(0, 0, 5)
    out = step(tables, s, A("Absorb", "Leer", s, names), empty_outcome())
    dealt = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert int(out.hp[0, 0]) == min(int(s.max_hp[0, 0]), 5 + max(1, dealt // 2))

def test_fury_swipes_hits_n_times(tables, names):
    s = roark_state(tables, names, player_moves=["Fury Swipes"])
    two = step(tables, s, A("Fury Swipes", "Leer", s, names), empty_outcome()._replace(n_hits=jnp.array([2, 2], jnp.int8)))
    five = step(tables, s, A("Fury Swipes", "Leer", s, names), empty_outcome()._replace(n_hits=jnp.array([5, 5], jnp.int8)))
    assert (int(s.hp[1, 0]) - int(five.hp[1, 0])) > (int(s.hp[1, 0]) - int(two.hp[1, 0]))

def test_simple_doubles_stage_change(tables, names):
    s = roark_state(tables, names, player_species="Bidoof", player_ability="Simple", player_moves=["Defense Curl"])
    out = step(tables, s, A("Defense Curl", "Leer", s, names), empty_outcome())
    assert int(out.stages[0, 2]) == 2 - 1   # +2 from Simple Defense Curl, -1 from Leer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_effects.py -v`
Expected: FAIL with `ImportError` / assertion failures on the plain-damage fallback.

- [ ] **Step 3: Implement `effects.py` and wire it into `step.py`**

`roark_state` gains keyword overrides `player_species, player_moves, player_ability, opp_species, opp_moves, opp_faster` (sets the opponent's Spe stat above the player's), and `with_status`. Defense Curl maps to `STAT_SELF (DEF, +1)` in `effect_map` (its Rollout flag is a later milestone).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_effects.py tests/engine/test_step_basic.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/effects.py src/nuzline/engine/step.py tests/engine/test_effects.py tests/engine/helpers.py tools/effect_map.py
git commit -m "feat(engine): stat, status, secondary, flinch, drain, recoil and multi-hit effects"
```

---

### Task 8c: Stealth Rock, Bind, Pursuit, Taunt, Leech Seed, Water Sport, confusion

**Files:**
- Modify: `src/nuzline/engine/effects.py`, `src/nuzline/engine/step.py`
- Test: `tests/engine/test_field_effects.py`

**Interfaces:**
- Stealth Rock: set `stealth_rock[target_side] = 1`; on switch-in, damage `max_hp * eff // 800` where `eff` is Rock effectiveness in the ×100 encoding (so neutral = 1/8), applied before the incoming Pokémon acts, with the faint/terminal check.
- Bind (`TRAP`): sets `trapped = 2–5 turns` (from `outcome.duration[side]`), `trap_move`; end of turn deals `max_hp // 16`; switching is illegal while trapped (Task 9).
- Pursuit: 40 BP; if the target chose a switch this turn, Pursuit executes before the switch at 80 BP against the outgoing Pokémon.
- Taunt: 3–5 turns from `outcome.duration` (Gen 4: counter set to a random 3–5 then decremented; the decomp fixes the exact rule); status moves are illegal while taunted (Task 9) and fail if forced.
- Leech Seed: sets `leech_seed[target] = 1` (Grass types immune); end of turn drains `max_hp // 8` to the other side's active.
- Water Sport: `water_sport[side] = 1` while that Pokémon is active; Fire damage halved (checked in `damage.py`).
- Confusion (`CONFUSE_FOE`): sets `confusion = 2–5` from `outcome.duration`; at move time, if `outcome.confusion_hit[side]`, the actor hits itself with a 40 BP typeless physical attack instead of moving.
- End-of-turn order for these is the decomp's (Task 1 row); encode it as an ordered list `END_OF_TURN_ORDER` in `step.py` with a comment per entry citing the decomp.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_field_effects.py
import jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import roark_state, A

def test_stealth_rock_sets_and_damages_on_switch(tables, names):
    s = roark_state(tables, names)                               # Geodude leads with Stealth Rock
    out = step(tables, s, A("Scratch", "Stealth Rock", s, names), empty_outcome())
    assert int(out.stealth_rock[0]) == 1
    out2 = step(tables, out, A(4 + 3, "Tackle", out, names), empty_outcome())   # switch to Starly (Flying: 2x)
    assert int(out2.hp[0, 3]) <= int(out.max_hp[0, 3]) - int(out.max_hp[0, 3]) // 4

def test_stealth_rock_can_faint_incoming_and_end_battle(tables, names):
    s = roark_state(tables, names).replace_hp(0, 3, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    out = step(tables, s, A(4 + 3, "Tackle", s, names), empty_outcome())
    assert int(out.hp[0, 3]) == 0 and int(out.over) == 1 and int(out.winner) == 1

def test_bind_traps_and_ticks(tables, names):
    s = roark_state(tables, names, opp_species="Onix", opp_moves=["Bind"])
    out = step(tables, s, A("Scratch", "Bind", s, names), empty_outcome()._replace(duration=jnp.array([0, 4], jnp.int8)))
    assert int(out.trapped[0]) > 0
    tick = step(tables, out, A("Leer", "Leer", out, names), empty_outcome())   # no damage this turn except the Bind tick
    assert int(out.hp[0, 0]) - int(tick.hp[0, 0]) == int(s.max_hp[0, 0]) // 16
    assert int(tick.trapped[0]) == int(out.trapped[0]) - 1

def test_pursuit_hits_switching_target_double(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=["Pursuit"])
    stay = step(tables, s, A("Leer", "Pursuit", s, names), empty_outcome())
    go = step(tables, s, A(4 + 1, "Pursuit", s, names), empty_outcome())
    assert (int(s.hp[0, 0]) - int(go.hp[0, 0])) > (int(s.hp[0, 0]) - int(stay.hp[0, 0]))
    assert int(go.active[0]) == 1

def test_taunt_sets_counter(tables, names):
    s = roark_state(tables, names, player_moves=["Taunt"])
    out = step(tables, s, A("Taunt", "Tackle", s, names), empty_outcome()._replace(duration=jnp.array([3, 0], jnp.int8)))
    assert int(out.taunt[1]) > 0

def test_leech_seed_drains_and_grass_immune(tables, names):
    s = roark_state(tables, names, player_species="Budew", player_moves=["Leech Seed"]).replace_hp(0, 0, 5)
    out = step(tables, s, A("Leech Seed", "Leer", s, names), empty_outcome())
    assert int(out.leech_seed[1]) == 1 and int(out.hp[0, 0]) > 5
    g = roark_state(tables, names, player_species="Budew", player_moves=["Leech Seed"], opp_species="Budew")
    assert int(step(tables, g, A("Leech Seed", "Absorb", g, names), empty_outcome()).leech_seed[1]) == 0

def test_confusion_self_hit_replaces_move(tables, names):
    s = roark_state(tables, names, player_species="Zubat", player_moves=["Supersonic"])
    out = step(tables, s, A("Supersonic", "Leer", s, names), empty_outcome()._replace(duration=jnp.array([0, 3], jnp.int8)))
    assert int(out.confusion[1]) > 0
    out2 = step(tables, out, A("Supersonic", "Tackle", out, names), empty_outcome()._replace(confusion_hit=jnp.array([0, 1], jnp.int8)))
    assert int(out2.hp[1, 0]) < int(out.hp[1, 0]) and int(out2.hp[0, 0]) == int(out.hp[0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_field_effects.py -v`
Expected: assertion failures.

- [ ] **Step 3: Implement**

`A` accepts an `int` for a raw action (switch) or a move name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine tests/engine
git commit -m "feat(engine): Stealth Rock, Bind, Pursuit, Taunt, Leech Seed, Water Sport, confusion"
```

---

### Task 8d: Major status mechanics and end-of-turn

**Files:**
- Modify: `src/nuzline/engine/step.py`
- Test: `tests/engine/test_status.py`

**Interfaces:**
- Sleep: `status_counter` from `outcome.duration` at infliction (2–5), decremented at the start of the sleeper's action; cannot move while > 0; wakes when it reaches 0 (Gen 4: the wake-up turn allows a move; confirm from decomp).
- Paralysis: speed ×¼ in `order.py`; `outcome.full_para[side]` skips the move.
- Burn: `max_hp // 8` at end of turn; physical halving already in damage.
- Poison: `max_hp // 8`; Toxic: `max_hp * counter // 16` with counter incrementing (not inflicted in M1 but the tick is implemented for Toxic Spikes later).
- Freeze: cannot move unless `outcome.thaw[side]`; Fire moves used by the frozen Pokémon thaw (none in M1 except Flame Wheel; implement the flag).
- Natural Cure: clears status on switch-out. Poison Point: 30 % contact proc via `outcome.contact_proc[def_side, hit_index]`, poisons the attacker. Static: same, paralysis.
- End-of-turn order list from 8c is completed here with status damage entries and Oran/Sitrus checks (8e).

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_status.py
import jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.ids import Status
from tests.engine.helpers import roark_state, A

def test_paralysis_quarters_speed_and_full_para_skips(tables, names):
    s = roark_state(tables, names, player_species="Shinx", player_moves=["Spark"], opp_species="Cranidos")
    hit = step(tables, s, A("Spark", "Leer", s, names), empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8)))
    assert int(hit.status[1, 0]) == Status.PARALYSIS
    skipped = step(tables, hit, A("Spark", "Headbutt", hit, names), empty_outcome()._replace(full_para=jnp.array([0, 1], jnp.int8)))
    assert int(skipped.hp[0, 0]) == int(hit.hp[0, 0])

def test_burn_ticks_one_eighth(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos").with_status(1, 0, Status.BURN)
    out = step(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == int(s.max_hp[1, 0]) // 8

def test_sleep_counts_down_and_wakes(tables, names):
    s = roark_state(tables, names).with_status(1, 0, Status.SLEEP, counter=2)
    t1 = step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(t1.hp[0, 0]) == int(s.hp[0, 0])
    t2 = step(tables, t1, A("Leer", "Tackle", t1, names), empty_outcome())
    assert int(t2.status[1, 0]) == Status.NONE

def test_freeze_thaw_outcome(tables, names):
    s = roark_state(tables, names).with_status(1, 0, Status.FREEZE)
    frozen = step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    thawed = step(tables, s, A("Leer", "Tackle", s, names), empty_outcome()._replace(thaw=jnp.array([0, 1], jnp.int8)))
    assert int(frozen.hp[0, 0]) == int(s.hp[0, 0]) and int(thawed.hp[0, 0]) < int(s.hp[0, 0])

def test_poison_point_procs_on_contact(tables, names):
    s = roark_state(tables, names, player_species="Budew", player_ability="Poison Point", opp_species="Cranidos", opp_moves=["Headbutt"])
    o = empty_outcome()._replace(contact_proc=jnp.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], jnp.int8))
    out = step(tables, s, A("Absorb", "Headbutt", s, names), o)
    assert int(out.status[1, 0]) == Status.POISON
    no = step(tables, s, A("Absorb", "Headbutt", s, names), empty_outcome())
    assert int(no.status[1, 0]) == Status.NONE

def test_natural_cure_on_switch(tables, names):
    s = roark_state(tables, names, player_species="Budew", player_ability="Natural Cure").with_status(0, 0, Status.BURN)
    out = step(tables, s, A(4 + 1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_status.py -v`
Expected: assertion failures.

- [ ] **Step 3: Implement in `step.py` and `order.py`**

`with_status(side, slot, status, counter=0)` is added to helpers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine tests/engine
git commit -m "feat(engine): sleep, paralysis, burn, poison, freeze, contact abilities, end-of-turn"
```

---

### Task 8e: Remaining M1 abilities and items

**Files:**
- Create: `src/nuzline/engine/hooks.py`
- Modify: `src/nuzline/engine/step.py`, `src/nuzline/engine/damage.py`
- Test: `tests/engine/test_abilities_items.py`

**Interfaces:**
- `hooks.py` produces `on_switch_in(tables, state, side) -> BattleState` (Intimidate: −1 Atk to the opposing active; Stealth Rock damage is here too). `make_state` calls `on_switch_in` for both leads (player first, then opponent, per the decomp's lead-ability order; confirm), `on_end_of_turn_items(tables, state) -> BattleState` and `after_damage_items(tables, state, side) -> BattleState` (Oran: heal 10 at `hp*2 <= max_hp`; Sitrus: heal `max_hp // 4` at `hp*2 <= max_hp`; both consume the item and set `item_consumed`; checked right after damage, per the decomp). Pinch abilities (Blaze/Torrent/Overgrow) and Rivalry, Guts, Unaware, Keen Eye, Simple, Steadfast, Rock Head, Mold Breaker, Inner Focus already live in `damage.py`/`effects.py`; Sturdy in Gen 4 only blocks OHKO moves, which are not in M1, so it is a no-op registered as implemented.
- `engine/ids.py` gains nothing; the registry of implemented ids is Task 1 of the next plan.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_abilities_items.py
import jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.ids import ItemId
from tests.engine.helpers import roark_state, A

def test_intimidate_on_lead_and_on_switch_in(tables, names):
    s = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert int(s.stages[1, 1]) == -1                                  # make_state runs on_switch_in for leads
    out = step(tables, s, A(4 + 1, "Tackle", s, names), empty_outcome())
    assert int(out.stages[1, 1]) == -1                                # Prinplup (slot 1) has no Intimidate: unchanged

def test_blaze_boosts_fire_at_third_hp(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos")
    low = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 3)
    normal = step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    boosted = step(tables, low, A("Ember", "Leer", low, names), empty_outcome())
    assert (int(s.hp[1, 0]) - int(boosted.hp[1, 0])) > (int(s.hp[1, 0]) - int(normal.hp[1, 0]))

def test_oran_berry_triggers_and_is_consumed(tables, names):
    s = roark_state(tables, names, player_species="Starly")                  # holds Oran Berry
    half = int(s.max_hp[0, 0]) // 2 + 1
    s = s.replace_hp(0, 0, half)
    out = step(tables, s, A("Growl", "Tackle", s, names), empty_outcome())
    dealt = half - (int(out.hp[0, 0]) - 10)
    assert dealt > 0 and int(out.item[0, 0]) == names.item("") and int(out.item_consumed[0, 0]) == names.item("Oran Berry")

def test_sitrus_heals_quarter(tables, names):
    s = roark_state(tables, names, player_species="Grotle")                  # holds Sitrus Berry
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2 + 1)
    out = step(tables, s, A("Withdraw", "Tackle", s, names), empty_outcome())
    assert int(out.item[0, 0]) == names.item("") and int(out.hp[0, 0]) > int(s.hp[0, 0]) - (int(s.max_hp[0, 0]) // 4)

def test_rivalry_same_gender_boost(tables, names):
    male = roark_state(tables, names, player_species="Shinx", player_ability="Rivalry", player_gender="male", opp_species="Cranidos", opp_gender="male")
    female = roark_state(tables, names, player_species="Shinx", player_ability="Rivalry", player_gender="male", opp_species="Cranidos", opp_gender="female")
    a = step(tables, male, A("Tackle", "Leer", male, names), empty_outcome())
    b = step(tables, female, A("Tackle", "Leer", female, names), empty_outcome())
    assert (int(male.hp[1, 0]) - int(a.hp[1, 0])) > (int(female.hp[1, 0]) - int(b.hp[1, 0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_abilities_items.py -v`
Expected: failures.

- [ ] **Step 3: Implement `hooks.py`, wire into `step.py` and `make_state`**

`roark_state` gains `player_gender`, `opp_gender`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine tests/engine
git commit -m "feat(engine): Intimidate, pinch abilities, Rivalry, Oran and Sitrus berries"
```

---

### Task 9: Legal actions and decision points

**Files:**
- Create: `src/nuzline/engine/actions.py`
- Modify: `docs/design.md` (§4.2 action encoding sentence)
- Test: `tests/engine/test_actions.py`

**Interfaces:**
- `legal_actions(tables, state, side) -> bool[10]`: moves legal when slot non-empty, PP > 0, not taunted-status, not disabled (later); if no move is legal, action 0 is legal and means Struggle; switches legal to non-active, non-empty, non-fainted slots, and never while `trapped > 0`.
- `legal_replacements(state, side) -> bool[6]`.
- `NUM_ACTIONS = 10`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/engine/test_actions.py
from nuzline.engine.actions import legal_actions, legal_replacements, NUM_ACTIONS
from nuzline.engine.ids import Status
from tests.engine.helpers import roark_state

def test_full_party_lead(tables, names):
    s = roark_state(tables, names)
    m = legal_actions(tables, s, 0).tolist()
    assert len(m) == NUM_ACTIONS and m[:4] == [True] * 4 and m[4] is False and all(m[5:10])

def test_fainted_and_empty_slots_illegal(tables, names):
    s = roark_state(tables, names, opp_count=2).replace_hp(1, 1, 0)
    m = legal_actions(tables, s, 1).tolist()
    assert m[4:10] == [False] * 6
    assert legal_replacements(s, 1).tolist() == [False] * 6

def test_zero_pp_and_struggle(tables, names):
    s = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0])
    m = legal_actions(tables, s, 0).tolist()
    assert m[0] is True and m[1:4] == [False] * 3

def test_trapped_cannot_switch_and_taunt_blocks_status(tables, names):
    s = roark_state(tables, names)
    s = s._replace(trapped=s.trapped.at[0].set(2), taunt=s.taunt.at[0].set(2))
    m = legal_actions(tables, s, 0).tolist()
    assert not any(m[4:10])
    leer = names.move_slot(s, 0, "Leer")
    assert m[leer] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_actions.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `actions.py`; update spec §4.2 to the 10-wide encoding with the self-slot masked**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/engine/test_actions.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/engine/actions.py tests/engine/test_actions.py docs/design.md
git commit -m "feat(engine): legal action and replacement masks"
```

---

### Task 10: Reference engine

**Files:**
- Create: `src/nuzline/reference/__init__.py` (empty)
- Create: `src/nuzline/reference/state.py`
- Create: `src/nuzline/reference/engine.py`
- Create: `src/nuzline/reference/convert.py`
- Test: `tests/reference/test_reference_smoke.py`

**Interfaces:**
- `reference/state.py`: `@dataclass class RefPokemon`, `RefSide`, `RefState` with the same fields as `BattleState`, as Python ints and lists.
- `reference/convert.py`: `to_ref(state: BattleState) -> RefState`, `from_ref(ref: RefState) -> BattleState`, round-trip exact.
- `reference/engine.py`: `ref_step(tables: TablesPy, state: RefState, actions: tuple[int,int], outcome: OutcomePy) -> RefState`. `TablesPy` is the `Tables` tuple with numpy arrays, produced by `to_numpy(tables)` added to `data/tables.py`; `OutcomePy` is the `Outcome` tuple as nested Python lists, produced by `outcome_to_py(outcome)` added to `engine/outcome.py`. Written as straight-line, branchy Python that mirrors the decomp's control flow (one `if` per rule), with a comment per rule citing the decomp file and function. It must not import from `nuzline.engine` except `ids` and `stats`.
- `ref_damage_table(tables_py, state, atk_side, move, crit) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/reference/test_reference_smoke.py
from nuzline.reference.convert import to_ref, from_ref
from nuzline.reference.engine import ref_step, ref_damage_table
from nuzline.engine.damage import damage_table
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.data.tables import to_numpy
from tests.engine.helpers import roark_state, A

def test_round_trip_exact(tables, names):
    s = roark_state(tables, names)
    back = from_ref(to_ref(s))
    for a, b in zip(s, back):
        assert (a == b).all()

def test_one_turn_matches_jax(tables, names):
    s = roark_state(tables, names)
    acts = A("Scratch", "Tackle", s, names)
    jax_out = step(tables, s, acts, empty_outcome())
    ref_out = ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(empty_outcome()))
    for a, b in zip(jax_out, from_ref(ref_out)):
        assert (a == b).all()

def test_damage_tables_match(tables, names):
    s = roark_state(tables, names)
    mv = int(s.moves[0, 0, 0])
    assert damage_table(tables, s, 0, mv, False).tolist() == ref_damage_table(to_numpy(tables), to_ref(s), 0, mv, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/reference -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the reference package**

Mirror every rule from Tasks 7–8e. Where the JAX engine and the reference disagree, read the decomp: the engine that matches the decomp is right, and the other is fixed. Record each such discrepancy in `docs/discrepancies.md` with the decomp citation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/reference -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/nuzline/reference tests/reference docs/discrepancies.md src/nuzline/data/tables.py src/nuzline/engine/outcome.py
git commit -m "feat(reference): plain-Python reference engine with state conversion"
```

---

### Task 11: Differential tests and JAX hygiene tests

**Files:**
- Create: `tests/strategies.py`
- Create: `tests/test_differential.py`
- Create: `tests/test_jax_hygiene.py`

**Interfaces:**
- `tests/strategies.py` produces Hypothesis strategies: `supported_pokemon(names) -> PokemonSpec` drawing only from the M1 species set (sample team species plus Geodude, Onix, Cranidos, Zubat, Bidoof) with moves drawn from their learnsets filtered to `move_effect != UNSUPPORTED`, random levels 5–20, random IVs/EVs, random current HP including 0, random status; `supported_state(tables, names) -> BattleState` building 1–6 per side with random stages, hazards, volatile counters within their legal ranges; `legal_action_pair(tables, state)`; `any_outcome()` drawing every `Outcome` field uniformly over its range.

- [ ] **Step 1: Write the tests**

```python
# tests/test_differential.py
from hypothesis import given, settings, HealthCheck, strategies as st
import numpy as np
from tests.strategies import supported_state, legal_action_pair, any_outcome
from nuzline.engine.step import step
from nuzline.reference.engine import ref_step
from nuzline.reference.convert import to_ref, from_ref
from nuzline.data.tables import to_numpy
from nuzline.engine.outcome import outcome_to_py

@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_jax_matches_reference(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    jax_out = step(tables, s, acts, o)
    ref_out = from_ref(ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(o)))
    for name, a, b in zip(jax_out._fields, jax_out, ref_out):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
```

```python
# tests/test_jax_hygiene.py
import jax, jax.numpy as jnp
from nuzline.engine.step import step
from nuzline.engine.outcome import empty_outcome, sample_outcome
from tests.engine.helpers import roark_state

def test_step_jits_and_vmaps_with_stable_shapes(tables, names):
    s = roark_state(tables, names)
    B = 16
    bs = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), s)
    bo = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), empty_outcome())
    f = jax.jit(jax.vmap(step, in_axes=(None, 0, 0, 0)))
    out = f(tables, bs, jnp.zeros((B, 2), jnp.int32), bo)
    for a, b in zip(bs, out):
        assert a.shape == b.shape and a.dtype == b.dtype

def test_no_64bit_in_jaxpr(tables, names):
    s = roark_state(tables, names)
    jaxpr = jax.make_jaxpr(step)(tables, s, jnp.zeros(2, jnp.int32), empty_outcome())
    assert "int64" not in str(jaxpr) and "float64" not in str(jaxpr)

def test_sample_outcome_is_jittable(tables, names):
    s = roark_state(tables, names)
    o = jax.jit(sample_outcome, static_argnums=())(jax.random.PRNGKey(0), tables, s, jnp.zeros(2, jnp.int32))
    assert o.roll.shape == (2, 5) and int(o.roll.max()) <= 15
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_differential.py tests/test_jax_hygiene.py -v`
Expected: failures that are real engine/reference disagreements (fix the wrong side per the decomp, log in `docs/discrepancies.md`) or strategy bugs. Iterate until 500 examples pass.

- [ ] **Step 3: Run the whole suite and ruff**

```bash
uv run pytest -q
uv run ruff check .
```
Expected: all PASS, no findings.

- [ ] **Step 4: Commit**

```bash
git add tests/strategies.py tests/test_differential.py tests/test_jax_hygiene.py docs/discrepancies.md src/nuzline
git commit -m "test: Hypothesis differential tests against reference engine and JAX hygiene tests"
```
