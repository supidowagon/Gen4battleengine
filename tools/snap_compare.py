"""Snapshot compare: capture the engine's outputs on drawn singles turns at one commit, then
check a later commit gives the same bytes.

Usage:

    uv run python tools/snap_compare.py capture DIR A-B [--chunk 10] [--repo PATH] [--no-roark]
        [--parts G,...] [-v]
    uv run python tools/snap_compare.py compare DIR A-B [--chunk 10] [--repo PATH] [--no-roark]
        [--parts G,...] [--allow-new NAME[,NAME...]] [--same-ok] [-v]

    DIR      the snapshot folder (capture writes it, compare reads it).
    A-B      the seed range, inclusive (one seed: A).
    --chunk  seeds per part (default 10); compare must use the capture's value.
    --repo   the checkout whose code runs (default: the one holding this script). Its ``src``
             and root go first on sys.path, so this script captures an older commit with
             ``--repo <that checkout>`` from any environment with the same dependencies.
    --no-roark  leave out the two Roark parts.
    --parts  run only these groups/parts (step, jitstep, sample, rest, gen, roark.trace,
             roark.rank); a left-out part is neither run nor counted.
    --allow-new  field names (any segment of an entry path, e.g. ``new_field``) whose presence
             on one side only does not fail; name both names of a renamed field.
    --same-ok  allow compare at the capture's own commit (otherwise a failure).
    -v       print each part's timings per output.

Needs a checkout and its built tables only (tools/ops/build_tables.sh: ``data/gen4_platinum``,
``data/gen4_platinum_kaizo``, ``data/intermediate``, ``data/intermediate_kaizo``); never the
ROM, a save, the calc archive or the docs sheet. A part whose built file is missing is
SKIPPED with the file named; a skip is a failure (exit 1).

capture: per seed of tools/differential.py's generator (tests.strategies, Hypothesis @seed, no
shrink), 5 drawn turns: the drawn input state, actions and outcome, and each segment of a turn
that pauses at a party list (the pick and a fresh outcome drawn as tests/test_differential.py's
_step_both draws them). Per segment: jit(step_events), jit(step), ref_step via from_ref,
jit(sample_outcome)(PRNGKey(seed)), enumerate_turn (or its refusal), ai_action_distribution,
jit(decision_actions), root_belief. Roark: the golden replay of tests/golden/kaizo_roark_02.json
(its converted states, replay_turn and the reference replay of each turn) and rank_actions at
tests.engine.helpers.roark_real_state, N = 256, PRNGKey(0).

capture also writes DIR/capture.json: the checkout's ``git rev-parse HEAD`` and whether its
tracked files are dirty. compare prints that commit and its own, and fails when the capture
records none or both are the same commit (unless --same-ok).

compare: reruns every output from the saved inputs (no redraw); its "gen" part redraws the
generator and checks it still draws the saved inputs (on capture, "gen" records them).

Parts: each seed chunk runs as five processes ("step": the draw on capture, jit(step_events),
ref_step; "jitstep": jit(step); "sample": jit(sample_outcome); "rest": enumerate_turn,
ai_action_distribution, decision_actions, root_belief; "gen": the generator check), then
"roark.trace" and "roark.rank", one process each. The parent prints each part's wall time and
peak memory and flags a part over 60 s or 4 GB (OVER LIMIT).

Version independence: inputs are saved as plain {field: array} dicts, not pickled NamedTuples.
A later commit rebuilds them by field name: a field it has that the capture lacks starts at the
singles zero (BattleState: ``state.LAYOUT``'s shape and dtype) or at ``empty_outcome()``'s value
(Outcome); a captured field it lacks is dropped. Each such input field is reported as one-sided
(``in_state.NAME`` / ``in_outcome.NAME``). Outputs (states, Outcomes, events, reference outputs)
are flattened to "path" -> (kind, dtype, shape, bytes) entries keyed by name. compare checks
the entries present on both sides byte for byte (dtype, shape and bytes) and lists, per part,
the entries present on one side only (indices folded to []).

An output that raised is stored as its exception (a refusal is an output) and compared like
one: a raise against a result is a divergence (a changed entry, or entries on one side only).
Each part prints its raise count (raised/calls) and, per function, count/calls and the first
error.

Pass: compare prints "compare: ok" and exits 0 only when every listed part ran (no skip, no
error), no entry diverged, every one-sided input field or output entry is named by
--allow-new, no part had every output raise, and the capture's commit is recorded and differs
from this checkout's (or --same-ok). Anything else exits 1. capture exits 1 on a skip, an
error or a part where every output raised.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TURNS = 5
RANK_N = 256
LIMIT_S = 60.0
LIMIT_BYTES = 4 * 2**30
SEED_GROUPS = ("step", "jitstep", "sample", "rest", "gen")
LIST_SHOWN = 8  # one-sided field names printed per part

# Built files each part reads (relative to the checkout).
VANILLA_TABLES = ("data/gen4_platinum/tables.npz", "data/gen4_platinum/manifest.json")
NEEDS = {
    "seed": (*VANILLA_TABLES, "data/intermediate/learnsets.json"),
    "roark.trace": (
        *VANILLA_TABLES,
        "data/gen4_platinum_kaizo/tables.npz",
        "data/gen4_platinum_kaizo/manifest.json",
        "data/intermediate_kaizo/trainers.json",
        "tests/golden/kaizo_roark_02.json",
    ),
    "roark.rank": (*VANILLA_TABLES, "data/intermediate/trainers.json"),
}


def set_root(path: Path) -> None:
    """Run the code of checkout ``path``: its ``src`` and root first on sys.path."""
    global ROOT
    ROOT = path.resolve()
    for p in (ROOT, ROOT / "src"):
        sys.path.insert(0, str(p))


def missing(part: str) -> list[str]:
    key = part if part.startswith("roark.") else "seed"
    return [f for f in NEEDS[key] if not (ROOT / f).exists()]


# --- serialization --------------------------------------------------------------------


def canon(x, path: str = "", out: dict | None = None) -> dict:
    """Flatten ``x`` to ``{path: (kind, dtype, shape, bytes)}``, keyed by name."""
    import numpy as np

    out = {} if out is None else out
    if isinstance(x, BaseException):
        out[path] = ("raise", type(x).__name__, "", str(x).encode())
    elif isinstance(x, tuple) and hasattr(x, "_fields"):
        for name in x._fields:
            canon(getattr(x, name), f"{path}.{name}", out)
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            canon(x[k], f"{path}.{k}", out)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            canon(getattr(x, f.name), f"{path}.{f.name}", out)
    elif isinstance(x, (list, tuple)):
        out[f"{path}#len"] = ("py", "len", "", str(len(x)).encode())
        for i, v in enumerate(x):
            canon(v, f"{path}[{i}]", out)
        for k in sorted(getattr(x, "__dict__", {})):  # Ranking.all_zero
            canon(vars(x)[k], f"{path}.{k}", out)
    elif isinstance(x, (str, bool, int, float)) or x is None:
        out[path] = ("py", type(x).__name__, "", repr(x).encode())
    else:
        a = np.ascontiguousarray(np.asarray(x))
        out[path] = ("arr", a.dtype.str, str(a.shape), a.tobytes())
    return out


def digest(entries: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(entries):
        kind, dt, shape, data = entries[k]
        h.update(f"{k}\0{kind}\0{dt}\0{shape}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def first_diff(want: dict, got: dict) -> str | None:
    """The first entry present in both whose bytes differ, or None."""
    import numpy as np

    for k in sorted(set(want) & set(got)):
        a, b = want[k], got[k]
        if a == b:
            continue
        if a[0] == b[0] == "arr" and a[1:3] == b[1:3]:
            shape = tuple(json.loads(a[2].replace("(", "[").replace(",)", "]").replace(")", "]")))
            x = np.frombuffer(a[3], np.dtype(a[1])).reshape(shape)
            y = np.frombuffer(b[3], np.dtype(b[1])).reshape(shape)
            idx = tuple(int(i) for i in np.argwhere(x != y)[0]) if x.ndim else ()
            return f"{k}{list(idx)}: {x[idx]!r} -> {y[idx]!r}"
        return f"{k}: {a[:3]} {a[3][:80]!r} -> {b[:3]} {b[3][:80]!r}"
    return None


def fold(key: str) -> str:
    """A field path with its indices folded (``.x[3].y`` -> ``.x[].y``)."""
    return re.sub(r"\[\d+\]", "[]", key)


def plain(x):
    """``x`` with every NamedTuple turned into a ``{field: value}`` dict (saved inputs)."""
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return {n: plain(getattr(x, n)) for n in x._fields}
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    return x


# Saved input fields one side lacks: {"base": {...}, "new": {...}} (compare fails on them).
INPUT_ONE_SIDED: dict[str, set[str]] = {"base": set(), "new": set()}


def _note_input_fields(kind: str, saved: dict, fields: tuple[str, ...]) -> None:
    INPUT_ONE_SIDED["base"] |= {f"{kind}.{n}" for n in set(saved) - set(fields)}
    INPUT_ONE_SIDED["new"] |= {f"{kind}.{n}" for n in set(fields) - set(saved)}


def rebuild_state(saved: dict):
    """A ``BattleState`` of this checkout from a saved field dict (see the docstring)."""
    import numpy as np

    from nuzline.engine.state import LAYOUT, BattleState

    _note_input_fields("in_state", saved, BattleState._fields)
    return BattleState(
        **{
            n: saved[n] if n in saved else np.zeros(LAYOUT[n][0], LAYOUT[n][1])
            for n in BattleState._fields
        }
    )


def rebuild_outcome(saved: dict):
    """An ``Outcome`` of this checkout from a saved field dict (see the docstring)."""
    import jax

    from nuzline.engine.outcome import Outcome, empty_outcome

    empty = jax.device_get(empty_outcome())
    _note_input_fields("in_outcome", saved, Outcome._fields)
    return Outcome(**{n: saved.get(n, getattr(empty, n)) for n in Outcome._fields})


# --- the runs ---------------------------------------------------------------------------

TIMES: dict[str, float] = {}
CALLS: dict[str, int] = {}
RAISED: dict[str, list] = {}  # name -> [count, first "Type: message"]


def timed(name, f):
    """``device_get(f())``, or the exception it raised (a refusal is an output; counted)."""
    import jax

    start = time.monotonic()
    CALLS[name] = CALLS.get(name, 0) + 1
    try:
        return jax.device_get(f())
    except Exception as e:  # noqa: BLE001
        r = RAISED.setdefault(name, [0, f"{type(e).__name__}: {str(e)[:160]}"])
        r[0] += 1
        return e
    finally:
        TIMES[name] = TIMES.get(name, 0.0) + time.monotonic() - start


class Ctx:
    def __init__(self):
        from nuzline.data.games import GAMES, KAIZO, VANILLA
        from nuzline.data.tables import load_names, load_tables, to_numpy
        from nuzline.jaxcache import cache_root, jax_version, set_compile_cache_dir

        set_compile_cache_dir(cache_root() / "jax-tests" / jax_version())
        self.vdir, self.kdir = GAMES[VANILLA].tables_dir, GAMES[KAIZO].tables_dir
        self.tables, self.names = load_tables(self.vdir), load_names(self.vdir)
        self.np_tables = to_numpy(self.tables)
        self._jit = {}

    def jit(self, f):
        import jax

        if f not in self._jit:
            self._jit[f] = jax.jit(f)
        return self._jit[f]


def group_outputs(ctx: Ctx, group: str, seed: int, s, acts, o) -> dict:
    """The outputs of one segment that ``group``'s program produces."""
    import jax
    import jax.numpy as jnp

    t = ctx.tables
    acts = jnp.asarray(acts, jnp.int32)
    if group == "step":
        from nuzline.engine.outcome import outcome_to_py
        from nuzline.engine.step import step_events
        from nuzline.reference.convert import from_ref, to_ref
        from nuzline.reference.engine import ref_step

        py_acts = tuple(int(x) for x in jax.device_get(acts))
        return {
            "step_events": timed("step_events", lambda: ctx.jit(step_events)(t, s, acts, o)),
            "ref_step": timed(
                "ref_step",
                lambda: from_ref(ref_step(ctx.np_tables, to_ref(s), py_acts, outcome_to_py(o))),
            ),
        }
    if group == "jitstep":
        from nuzline.engine.step import step

        return {"jit_step": timed("jit_step", lambda: ctx.jit(step)(t, s, acts, o))}
    if group == "sample":
        from nuzline.engine.outcome import sample_outcome

        key = jax.random.PRNGKey(seed)
        return {
            "sample_outcome": timed(
                "sample_outcome", lambda: ctx.jit(sample_outcome)(key, t, s, acts)
            )
        }
    from nuzline.ai.policy import ai_action_distribution
    from nuzline.engine.actions import decision_actions
    from nuzline.engine.enumerate import enumerate_turn
    from nuzline.engine.step import root_belief

    return {
        "enumerate_turn": timed("enumerate_turn", lambda: enumerate_turn(t, s, acts)),
        "ai_action_distribution": timed(
            "ai_action_distribution", lambda: ai_action_distribution(t, s)
        ),
        "decision_actions": timed("decision_actions", lambda: ctx.jit(decision_actions)(t, s)),
        "root_belief": timed("root_belief", lambda: root_belief(t, s)),
    }


def draw_inputs(ctx: Ctx, seed_value: int, saved: dict | None = None) -> list[tuple[str, dict]]:
    """``[(key, {in_state, in_actions, in_outcome})]`` (plain dicts) drawn as
    tools/differential.py draws a seed (TURNS examples, no shrink), each pause resumed as
    _step_both resumes it. With ``saved`` (compare's generator check) a pause and its state are
    read from the saved inputs instead of stepping, so the check loads no step program."""
    import jax
    from hypothesis import HealthCheck, Phase, given, seed, settings
    from hypothesis import strategies as st

    from nuzline.engine.step import step_events
    from tests.strategies import any_outcome, legal_action_pair, pivot_actions, supported_state

    out: list[tuple[str, dict]] = []
    turn = [0]
    events = None if saved else ctx.jit(step_events)  # [0] is step's state

    @seed(seed_value)
    @settings(
        max_examples=TURNS,
        deadline=None,
        database=None,
        phases=[Phase.explicit, Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(data=st.data())
    def gen(data):
        s = data.draw(supported_state(ctx.tables, ctx.names))
        acts = data.draw(legal_action_pair(ctx.tables, s))
        o = data.draw(any_outcome())
        for seg in range(3):
            key = f"seed {seed_value} turn {turn[0]} segment {seg}"
            inputs = {"in_state": s, "in_actions": acts, "in_outcome": o}
            out.append((key, plain(jax.device_get(inputs))))
            if saved is not None:
                nkey = f"seed {seed_value} turn {turn[0]} segment {seg + 1}"
                if nkey not in saved:
                    break
                s = rebuild_state(saved[nkey]["in_state"])
            else:
                nxt = events(ctx.tables, s, acts, o)[0]
                if int(nxt.pivot_battler) == 0:
                    break
                s = nxt
            acts = data.draw(pivot_actions(s, acts))
            o = data.draw(any_outcome())
        turn[0] += 1

    start = time.monotonic()
    gen()
    TIMES["draw"] = TIMES.get("draw", 0.0) + time.monotonic() - start
    return out


def seed_steps(ctx: Ctx, directory: Path, mode: str, lo: int, hi: int, group: str):
    inputs_path = directory / f"{lo}-{hi}.inputs.pkl.gz"
    if mode == "capture" and group == "gen":
        with gzip.open(inputs_path, "rb") as f:
            return pickle.load(f)
    if mode == "capture" and group == "step":
        inputs = [kv for v in range(lo, hi + 1) for kv in draw_inputs(ctx, v)]
        with gzip.open(inputs_path, "wb") as f:
            pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(inputs_path, "rb") as f:
            inputs = pickle.load(f)
    if group == "gen":  # compare only: the generator still draws the saved inputs
        saved = dict(inputs)
        return [kv for v in range(lo, hi + 1) for kv in draw_inputs(ctx, v, saved)]
    steps = []
    for key, inp in inputs:
        seed_value = int(key.split()[1])
        s, o = rebuild_state(inp["in_state"]), rebuild_outcome(inp["in_outcome"])
        outs = group_outputs(ctx, group, seed_value, s, inp["in_actions"], o)
        if group == "step":
            outs = {**inp, **outs}
        steps.append((key, outs))
    return steps


def roark_steps(ctx: Ctx, which: str):
    import jax
    import numpy as np

    steps = []
    if which == "rank":
        from nuzline.search.stage_a import rank_actions
        from tests.engine.helpers import roark_real_state

        state = roark_real_state(ctx.tables, ctx.names)
        ranking = timed(
            "rank_actions",
            lambda: rank_actions(ctx.tables, ctx.names, state, RANK_N, jax.random.PRNGKey(0)),
        )
        return [("rank_actions roark_real_state", {"state": state, "ranking": ranking})]
    from nuzline.data.tables import load_names, load_tables, to_numpy
    from nuzline.engine.outcome import outcome_to_py
    from nuzline.engine.step import needs_replacement
    from nuzline.reference.convert import from_ref, to_ref
    from nuzline.reference.engine import ref_advance_learns, ref_learn, ref_replace, ref_step
    from tools.trace_to_state import learn_array, replay_turn, trace_to_states

    kt, kn = load_tables(ctx.kdir), load_names(ctx.kdir)
    knp = to_numpy(kt)
    turns = timed(
        "trace_to_states",
        lambda: trace_to_states(ROOT / "tests" / "golden" / "kaizo_roark_02.json", kt, kn),
    )
    for i, before in enumerate(turns):
        rec = {
            "state": before.state,
            "actions": np.asarray(before.actions),
            "outcome": before.outcome,
            "player_slot": before.player_slot,
            "learn_choices": tuple(before.learn_choices),
        }
        if i + 1 < len(turns):
            opp_slot = int(turns[i + 1].state.active[1])
            rec["replay_turn"] = timed(
                "replay_turn",
                lambda b=before, o=opp_slot: replay_turn(
                    kt,
                    b.state,
                    b.actions,
                    b.outcome,
                    b.player_slot,
                    o,
                    learn_array(b.learn_choices),
                ),
            )

            def ref_replay(b=before, opp=opp_slot):
                ref = ref_step(
                    knp, to_ref(b.state), tuple(int(a) for a in b.actions), outcome_to_py(b.outcome)
                )
                need = np.asarray(needs_replacement(from_ref(ref))) != 0
                ref = ref_advance_learns(knp, ref)
                for choice in b.learn_choices:
                    ref = ref_learn(knp, ref, choice)
                if need[0] and b.player_slot >= 0:
                    ref = ref_replace(knp, ref, 0, b.player_slot)
                if need[1] and opp >= 0:
                    ref = ref_replace(knp, ref, 1, opp)
                return from_ref(ref)

            rec["ref_replay"] = timed("ref_replay", ref_replay)
        steps.append((f"kaizo_roark_02 turn {i}", rec))
    return steps


# --- child / parent ---------------------------------------------------------------------


def _peak_bytes() -> int:
    """This process's peak working set (Windows) or max RSS (Linux)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize)
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def child(mode: str, directory: Path, part: str) -> int:
    warnings.filterwarnings("ignore", "Generating overly large repr")
    head, group = part.split(".")
    ctx = Ctx()
    directory.mkdir(parents=True, exist_ok=True)
    if head == "roark":
        steps = roark_steps(ctx, group)
    else:
        lo, hi = map(int, head.split("-"))
        steps = seed_steps(ctx, directory, mode, lo, hi, group)
    dump = [(key, canon(outs)) for key, outs in steps]
    path = directory / f"{part}.pkl.gz"
    result = {
        "steps": len(dump),
        "diverged": None,
        "only_base": [],
        "only_new": [],
        "calls": sum(CALLS.values()),
        "raised": {k: [v[0], CALLS[k], v[1]] for k, v in sorted(RAISED.items())},
    }
    if mode == "capture":
        record = [(key, digest(d), d) for key, d in dump]
        with gzip.open(path, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        (directory / f"{part}.sha256").write_text(
            "".join(f"{h}  {key}\n" for key, h, _ in record), encoding="utf-8"
        )
    else:
        with gzip.open(path, "rb") as f:
            record = pickle.load(f)
        only_base: set[str] = set()
        only_new: set[str] = set()
        if [k for k, _, _ in record] != [k for k, _ in dump]:
            result["diverged"] = f"step list differs: {len(record)} captured, {len(dump)} rerun"
        else:
            for (key, _, want), (_, got) in zip(record, dump, strict=True):
                only_base |= {fold(k) for k in set(want) - set(got)}
                only_new |= {fold(k) for k in set(got) - set(want)}
                diff = first_diff(want, got)
                if diff and not result["diverged"]:
                    result["diverged"] = f"{key}: {diff}"
        only_base |= INPUT_ONE_SIDED["base"]
        only_new |= INPUT_ONE_SIDED["new"]
        result["only_base"], result["only_new"] = sorted(only_base), sorted(only_new)
    result["peak_bytes"] = _peak_bytes()
    print("TIMES " + json.dumps({k: round(v, 1) for k, v in TIMES.items()}), file=sys.stderr)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


def _names(fields: list[str]) -> str:
    shown = ", ".join(fields[:LIST_SHOWN])
    return shown + (f" (+{len(fields) - LIST_SHOWN} more)" if len(fields) > LIST_SHOWN else "")


def allowed(path: str, allow: set[str]) -> bool:
    """``path`` (a folded entry path) names an ``--allow-new`` field as one of its segments."""
    return any(seg in allow for seg in re.split(r"[.\[\]#]+", path) if seg)


def commit_of(repo: Path) -> dict:
    """``{"commit", "dirty"}`` of checkout ``repo`` (commit None outside git)."""

    def git(*a):
        r = subprocess.run(
            ["git", "-C", str(repo), *a], capture_output=True, text=True, check=False
        )
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _show(c: dict) -> str:
    return f"{c.get('commit')}{' (dirty)' if c.get('dirty') else ''}"


def parent(args) -> int:
    lo, _, hi = args.seeds.partition("-")
    lo, hi = int(lo), int(hi or lo)
    directory = Path(args.dir).resolve()
    allow = {n.strip() for n in (args.allow_new or "").split(",") if n.strip()}
    groups = [g.strip() for g in args.parts.split(",")] if args.parts else None
    parts = [
        f"{a}-{min(a + args.chunk - 1, hi)}.{g}"
        for a in range(lo, hi + 1, args.chunk)
        for g in SEED_GROUPS
        if groups is None or g in groups
    ]
    if not args.no_roark:
        parts += [p for p in ("roark.trace", "roark.rank") if groups is None or p in groups]
    print(f"{args.mode}: checkout {ROOT}, snapshot {directory}, {len(parts)} parts", flush=True)
    bad, overs, skipped, total = 0, [], [], time.monotonic()
    here = commit_of(ROOT)
    if args.mode == "capture":
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "capture.json").write_text(json.dumps(here, indent=1), encoding="utf-8")
        print(f"capture commit {_show(here)}", flush=True)
    else:
        try:
            base = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            base = {}
        print(f"capture commit {_show(base)}; this checkout {_show(here)}", flush=True)
        if not base.get("commit"):
            print("FAIL: the capture records no commit (capture.json)", flush=True)
            bad += 1
        elif base["commit"] == here["commit"] and not args.same_ok:
            print("FAIL: same commit on both sides (pass --same-ok to allow)", flush=True)
            bad += 1
    for part in parts:
        lack = missing(part)
        if lack:
            print(
                f"{part}: SKIPPED: needs {', '.join(lack)} (built tables; not a pass)", flush=True
            )
            skipped.append(part)
            continue
        start = time.monotonic()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "_child", args.mode, str(directory),
             part, str(ROOT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )  # fmt: skip
        took = time.monotonic() - start
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if proc.returncode or line is None:
            print(
                f"{part}: ERROR (exit {proc.returncode}, {took:.0f} s)\n{proc.stdout[-3000:]}"
                f"\n{proc.stderr[-6000:]}",
                flush=True,
            )
            bad += 1
            continue
        res = json.loads(line[7:])
        bad += report(args, part, res, took, allow, proc.stderr)
        overs += [part] if took > LIMIT_S or res["peak_bytes"] > LIMIT_BYTES else []
    if skipped:
        bad += len(skipped)
    verdict = "ok" if not bad else f"FAILED ({bad} failure(s), skips included)"
    print(
        f"{args.mode}: {verdict} ({time.monotonic() - total:.0f} s); "
        f"over 60 s / 4 GB: {', '.join(overs) or 'none'}; "
        f"skipped (each a failure): {', '.join(skipped) or 'none'}"
    )
    return 1 if bad else 0


def report(args, part: str, res: dict, took: float, allow: set[str], stderr: str) -> int:
    """Print one part's result; 1 if it failed: a divergence, a one-sided field not in
    ``--allow-new``, or every output raised."""
    gb = res["peak_bytes"] / 2**30
    over = took > LIMIT_S or res["peak_bytes"] > LIMIT_BYTES
    if args.mode == "capture":
        verdict = "captured"
    else:
        verdict = f"DIVERGED: {res['diverged']}" if res["diverged"] else "identical"
    print(
        f"{part}: {verdict}; {res['steps']} steps, {took:.1f} s, peak {gb:.2f} GB"
        f"{'  OVER LIMIT' if over else ''}",
        flush=True,
    )
    failed = bool(res["diverged"])
    for label, key in (("only in the capture", "only_base"), ("only in this checkout", "only_new")):
        hard = [f for f in res[key] if not allowed(f, allow)]
        ok = [f for f in res[key] if allowed(f, allow)]
        if hard:
            print(f"  FAIL {label}: {_names(hard)}", flush=True)
            failed = True
        if ok:
            print(f"  {label} (--allow-new): {_names(ok)}", flush=True)
    n_raised = sum(r[0] for r in res["raised"].values())
    print(f"  raised: {n_raised}/{res['calls']} outputs", flush=True)
    for name, (count, calls, first) in res["raised"].items():
        print(f"  raised: {name} {count}/{calls} ({first})", flush=True)
    if res["calls"] and n_raised == res["calls"]:
        print(f"  FAIL: every output raised ({n_raised}/{res['calls']})", flush=True)
        failed = True
    if args.verbose:
        print("  " + next((ln for ln in stderr.splitlines() if ln.startswith("TIMES")), ""))
    return int(failed)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        set_root(Path(argv[4]))
        try:
            return child(argv[1], Path(argv[2]), argv[3])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 2
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=("capture", "compare"))
    p.add_argument("dir")
    p.add_argument("seeds", help="A-B (inclusive)")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--no-roark", action="store_true")
    p.add_argument("--parts", help="only these groups/parts, e.g. step,rest,roark.trace")
    p.add_argument("--allow-new", help="NAME[,NAME...]: one-sided fields that do not fail")
    p.add_argument("--same-ok", action="store_true", help="allow the capture's own commit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    set_root(args.repo)
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
