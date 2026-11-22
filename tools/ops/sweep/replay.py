"""Replay the nightly sweep's test_mask_sound / test_mask_sound_with_a_trainer_item falsifiers on a
repo, with a per-Outcome-field split and the facts needed to group them into classes.

Usage (on the pod; loads the engine):
    python replay.py <repo root> <example dir | folder of example dirs> [...]
An example dir is what extract.py writes (node.txt names the test). Tables: <repo root>/data/
gen4_platinum (built). Per example it prints the setup facts, PASS/FAIL of the test body (the
test's own tests.engine.test_enumerate._assert_mask_sound on the drawn inputs), on FAIL every
Outcome field whose swap alone changes the stepped state (mask, one bad row, the changed state
fields; this split re-runs the body's row choice locally), and a one-line CLASS KEY; a CLASS KEY
summary with counts closes the run. When the local rows disagree with the test's verdict (the test
body changed and this split did not follow) the key's field part is DRIFT-<verdict>, so the
mismatch shows instead of a wrong split.
"""

import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / "src"), str(root)]

import jax
import jax.numpy as jnp
import numpy as np

import tests.engine.test_enumerate as T
from nuzline.data.tables import load_names, load_tables
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    MOVE_ACTIONS,
    legal_actions,
    legal_opponent_actions,
)
from nuzline.engine.enumerate import SamplingOnly, enumerate_drawn
from nuzline.engine.ids import EffectId, Status
from nuzline.engine.outcome import Outcome
from nuzline.engine.state import BattleState

TABLES = root / "data" / "gen4_platinum"
tables = load_tables(TABLES)
names = load_names(TABLES)


def Array(x, dtype=None, shape=None):
    a = np.array(x, dtype=dtype)
    return jnp.asarray(a.reshape(shape) if shape is not None else a)


def keep(cls):
    def make(**kw):
        extra = sorted(set(kw) - set(cls._fields))
        missing = sorted(set(cls._fields) - set(kw))
        if extra:
            print("  dropped fields not in this repo:", extra)
        if missing:
            raise ValueError(f"{cls.__name__} fields missing from the log: {missing}")
        return cls(**{k: v for k, v in kw.items() if k in cls._fields})

    return make


ENV = {
    "Array": Array,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "bool": bool,
    "BattleState": keep(BattleState),
    "Outcome": keep(Outcome),
}


def name_of(seq, i):
    i = int(i)
    return f"{seq[i]}({i})" if 0 <= i < len(seq) else f"?({i})"


def enum_name(cls, v):
    try:
        return cls(int(v)).name
    except ValueError:
        return str(int(v))


def effect(move):
    return enum_name(EffectId, np.asarray(tables.move_effect)[int(move)])


def read(here: Path, part: str):
    return eval((here / f"{part}.txt").read_text(), ENV)


def setup(here: Path):
    """The test body's (state, actions, arbitrary, seed, bag item id or None)."""
    node = (here / "node.txt").read_text().strip()
    s = read(here, "state")
    arbitrary = read(here, "outcome")
    seed = int(read(here, "seed"))
    if node == "test_mask_sound":
        return node, s, np.asarray(read(here, "acts"), np.int32), arbitrary, seed, None
    assert node == "test_mask_sound_with_a_trainer_item", node
    item = names.item(read(here, "item"))
    s = s._replace(
        trainer_items=s.trainer_items.at[1, 0].set(item),
        trainer_item_count=s.trainer_item_count.at[1].set(max(int(s.trainer_item_count[1]), 1)),
    )
    if not bool(np.asarray(legal_opponent_actions(tables, s))[FIRST_ITEM_ACTION]):
        print("  WARNING: the bag item is not legal here (the test's assume() would reject it)")
    player = np.flatnonzero(np.asarray(legal_actions(tables, s, 0))).tolist()
    choice = int(read(here, "choice"))
    if choice not in player:
        print("  WARNING: player action", choice, "not in the legal list", player)
    return node, s, np.array([choice, FIRST_ITEM_ACTION], np.int32), arbitrary, seed, item


def facts(s, acts, item):
    """Prints the dedup facts; returns (item, abilities, move effects) for the class key."""
    g = lambda f: np.asarray(getattr(s, f))
    act = g("active")
    abilities, effects = [], []
    print(
        "  battle_type",
        int(np.asarray(tables.battle_type)),
        "over",
        int(g("over")),
        "acts",
        acts.tolist(),
    )
    for side in (0, 1):
        m = int(act[side])
        ab = name_of(names.ability_names, g("ability")[side, m])
        supp = int(g("ability_suppressed")[side])
        abilities.append(ab + ("[supp]" if supp else ""))
        moves = g("moves")[side, m]
        a = int(acts[side])
        if a < MOVE_ACTIONS:
            used = f"move {name_of(names.move_names, moves[a])} {effect(moves[a])}"
            effects.append(effect(moves[a]))
        elif a < FIRST_ITEM_ACTION:
            used = f"switch to slot {a - MOVE_ACTIONS}"
            effects.append("SWITCH")
        else:
            bag = int(g("trainer_items")[side, a - FIRST_ITEM_ACTION])
            used = f"bag item {name_of(names.item_names, bag)}"
            effects.append("BAG")
        print(
            f"  side {side} slot {m} {name_of(names.species_names, g('species')[side, m])}"
            f" lv {int(g('level')[side, m])} hp {int(g('hp')[side, m])}/{int(g('max_hp')[side, m])}"
            f" ability {ab} suppressed {supp}"
            f" held {name_of(names.item_names, g('item')[side, m])}"
            f" consumed {int(g('item_consumed')[side, m])}"
            f" status {enum_name(Status, g('status')[side, m])}"
            f" counter {int(g('status_counter')[side, m])}"
            f" confusion {int(g('confusion')[side])} stages {g('stages')[side].tolist()}"
        )
        print(f"    uses: {used}")
        print(
            "    moveset:",
            [f"{name_of(names.move_names, x)} {effect(x)}" for x in moves],
            "pp",
            g("pp")[side, m].tolist(),
        )
    bag_item = name_of(names.item_names, item) if item is not None else "-"
    print(
        "  trainer bag",
        g("trainer_items").tolist(),
        "count",
        g("trainer_item_count").tolist(),
        "item used",
        bag_item,
    )
    return bag_item, tuple(abilities), tuple(effects)


def run(here: Path):
    print("==", here.name, flush=True)
    node, s, acts, arbitrary, seed, item = setup(here)
    print("  node", node, "seed", seed)
    key_facts = facts(s, acts, item)
    try:
        T._assert_mask_sound(tables, s, acts, arbitrary, seed)
        verdict = "PASS"
    except AssertionError:
        verdict = "FAIL"
    print("  TEST BODY", verdict, flush=True)
    laws = T.laws_of(tables, s, acts)
    try:
        outcomes, _, drawn = enumerate_drawn(laws)
        k = len(outcomes.hit)
        idx = (
            np.arange(k)
            if k <= T.CHUNK
            else np.random.default_rng(seed).choice(k, T.CHUNK // 2, replace=False)
        )
        picked, kept = T.rows(outcomes, idx), T.rows(drawn, idx)
        print("  enumerated rows", k)
    except SamplingOnly:
        print("  sampling only")
        keys = jax.random.split(jax.random.key(seed), 5000)
        sampled_ = jax.tree.map(
            lambda f: np.asarray(f)[: T.CHUNK // 2], T._sample(keys, tables, s, acts)
        )
        picked = Outcome(*(np.where(m, f, e) for f, m, e in zip(sampled_, laws.relevant, T.EMPTY)))
        kept = Outcome(*(np.broadcast_to(m, np.shape(f)) for f, m in zip(picked, laws.relevant)))
    changed = Outcome(*(np.where(m, f, np.asarray(a)) for f, m, a in zip(picked, kept, arbitrary)))
    n = len(picked.hit)
    F = Outcome._fields
    parts = [picked, changed] + [
        Outcome(*(c if g == f else p for g, p, c in zip(F, picked, changed))) for f in F
    ]
    allrows = Outcome(*(np.concatenate([np.asarray(getattr(q, g)) for q in parts]) for g in F))
    N = len(allrows.hit)
    out = []
    for start in range(0, N, T.CHUNK):
        idx = np.arange(start, min(start + T.CHUNK, N))
        pad = np.concatenate([idx, np.full(T.CHUNK - len(idx), idx[0])])
        st_ = T._step(tables, s, acts, T.rows(allrows, pad))
        m = len(idx)
        out.append(jax.tree.map(lambda f, m=m: np.asarray(f)[:m], st_))
    full = jax.tree.map(lambda *xs: np.concatenate(xs), *out)

    def bad_rows(lo):
        bad = np.zeros(n, bool)
        for x in full:
            bad |= np.any((x[:n] != x[lo : lo + n]).reshape(n, -1), axis=1)
        return bad

    whole = bad_rows(n)
    print(
        "  RESULT",
        "FAIL" if whole.any() else "PASS",
        "bad rows",
        int(whole.sum()),
        "of",
        n,
        flush=True,
    )
    fields, state_fields = [], set()
    for i, f in enumerate(F):
        lo = (i + 2) * n
        bad = bad_rows(lo)
        if not bad.any():
            continue
        fields.append(f)
        r = int(np.argmax(bad))
        p, c = np.asarray(getattr(picked, f))[r], np.asarray(getattr(changed, f))[r]
        print(
            "  FIELD",
            f,
            "bad rows",
            int(bad.sum()),
            "of",
            n,
            "first",
            r,
            "picked",
            p.tolist(),
            "arb",
            c.tolist(),
            "drawn",
            np.asarray(getattr(kept, f))[r].astype(int).tolist(),
            "mask",
            np.asarray(getattr(laws.relevant, f)).astype(int).tolist(),
            "swapped elements",
            np.argwhere(np.atleast_1d(p != c)).tolist(),
        )
        for name, x in zip(BattleState._fields, full):
            if np.any(x[r] != x[lo + r]):
                state_fields.add(name)
                print("     ", name, x[r].tolist(), "->", x[lo + r].tolist())
        print(
            "    picked row",
            r,
            {g: np.asarray(getattr(picked, g))[r].tolist() for g in F},
            flush=True,
        )
    if whole.any() and not fields:
        fields = ["JOINT-ONLY"]  # no single field's swap breaks it; only several together
    field_key = "+".join(fields) if fields else "PASS"
    if ("FAIL" if whole.any() else "PASS") != verdict:
        print(
            "  DRIFT: the local rows say",
            "FAIL" if whole.any() else "PASS",
            "the test body",
            verdict,
        )
        field_key = f"DRIFT-{verdict}"
    bag_item, abilities, effects = key_facts
    key = f"({field_key}, {bag_item}, {'/'.join(abilities)}, {'/'.join(effects)})"
    print("  CHANGED STATE FIELDS", sorted(state_fields))
    print("  CLASS KEY", key, flush=True)
    return key


def examples(args):
    for a in map(Path, args):
        if (a / "node.txt").exists():
            yield a
        else:
            yield from sorted(d for d in a.iterdir() if (d / "node.txt").exists())


keys = Counter()
for d in examples(sys.argv[2:]):
    try:
        keys[run(d)] += 1
    except Exception as e:  # noqa: BLE001 (one bad example must not stop the rest)
        print("  ERROR", type(e).__name__, e, flush=True)
        keys[f"ERROR {type(e).__name__}"] += 1
print("== CLASS KEYS")
for k, v in keys.most_common():
    print(f"  {v:3d}  {k}")
print("done")
