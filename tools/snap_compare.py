"""Snapshot compare: capture the engine's outputs on drawn singles turns at one commit, then
check a later commit gives the same bytes.

Usage:

    uv run python tools/snap_compare.py capture DIR A-B [--chunk 10] [--repo PATH]
        [--commit SHA] [--no-roark] [--parts G,...] [--allow-all-raise F,...] [--dirty-ok]
        [--raise-warn R] [--raise-fail R] [-v]
    uv run python tools/snap_compare.py compare DIR A-B [--chunk 10] [--repo PATH]
        [--commit SHA] [--no-roark] [--parts G,...] [--allow-new NAME,...]
        [--allow-all-raise F,...] [--subset-ok] [--same-ok] [--dirty-ok]
        [--raise-warn R] [--raise-fail R] [-v]

    DIR      the snapshot folder (capture writes it, compare reads it).
    A-B      the seed range, inclusive (one seed: A); B >= A.
    --chunk  seeds per part (default 10, at least 1); compare must use the capture's value.
    --repo   the checkout whose code runs (default: the one holding this script). Its ``src``
             and root go first on sys.path, so this script captures an older commit with
             ``--repo <that checkout>`` from any environment with the same dependencies. The
             child process checks that ``nuzline`` is imported from ``<repo>/src``.
    --commit the full 40-hex commit of --repo. Required when --repo is not the top of a git
             checkout (a ``git archive`` checkout on the pod); when git is there it must agree.
    --no-roark  leave out the two Roark parts.
    --parts  run only these groups/parts: step, jitstep, sample, rest, ai, gen, roark.trace,
             roark.rank (any other name is a usage error); a left-out part is neither run nor
             counted.
    --allow-new  field names (any segment of an entry path, e.g. ``new_field``) NEW in this
             checkout whose absence from the capture does not fail. A field only in the capture
             always fails ("dropped field without an upgrade row"): it needs an ``UPGRADES`` row.
    --allow-all-raise  function names (e.g. ``enumerate_turn``) allowed to raise on every call
             of a part; default none.
    --subset-ok  compare may run a strict subset of the captured parts.
    --same-ok  allow compare at the capture's own commit, or against a checkout whose content
             hash (see capture.json below) equals the capture's.
    --dirty-ok  allow a dirty checkout (tracked changes, or untracked files under src/, tools/,
             tests/strategies*, tests/engine/helpers.py or tests/ai/) on either side.
    --raise-warn, --raise-fail  a part whose outputs raised in more than this share of its
             calls (functions named by --allow-all-raise left out) prints a warning (default
             0.5) or fails (default 0.95); 0 <= warn <= fail < 1.
    -v       print each part's timings per output.

A usage error (unknown --parts name, B < A, --chunk < 1, an empty part list, raise shares
outside 0 <= --raise-warn <= --raise-fail < 1) exits 2.

Needs a checkout and its built tables only (tools/ops/build_tables.sh: ``data/gen4_platinum``,
``data/gen4_platinum_kaizo``, ``data/intermediate``, ``data/intermediate_kaizo``); never the
ROM, a save, the calc archive or the docs sheet. A part whose built file is missing is
SKIPPED with the file named; a skip is a failure (exit 1).

capture: per seed of tools/differential.py's generator (tests.strategies, Hypothesis @seed, no
shrink), 5 drawn turns: the drawn input state, actions and outcome, and each segment of a turn
that pauses at a party list (the pick and a fresh outcome drawn as tests/test_differential.py's
_step_both draws them). Per segment: jit(step_events), jit(step), ref_step via from_ref,
jit(sample_outcome)(PRNGKey(seed)), enumerate_turn (or its refusal), jit(decision_actions),
root_belief. Per drawn turn also one state the vanilla AI accepts (tests.ai.test_ai_legality's
gated_states: Roark's real party against the sample party, with random HP, stages, trap,
0..BAG_SLOTS Potions in the bag and AI knowledge; the random segments' opponents mostly hold
moves whose AI script paths are not transcribed, so the AI refuses them):
ai_action_distribution. Roark: the golden replay of
tests/golden/kaizo_roark_02.json (its converted states, replay_turn and the reference replay of
each turn) and rank_actions at tests.engine.helpers.roark_real_state, N = 256, PRNGKey(0).

DIR/capture.json records its ``format`` (CAPTURE_FORMAT), the commit (git HEAD or --commit),
whether the checkout was dirty (null without git: "not checkable"), ``content_hash``, each
captured part with its status (ok / failed / pending) and each seed run [A, B, chunk].
``content_hash`` is a sha256 over everything the child imports or reads, by content: the
running snap_compare.py (``__file__``, not --repo's copy), src/nuzline/**/*.py, tools/*.py,
tests/__init__.py, tests/strategies*.py, tests/ai_legality.py, tests/engine/__init__.py,
tests/engine/helpers.py, tests/ai/*.py, tests/fixtures/sample_party.toml, the golden trace and
every built file a part reads (both tables.npz and manifest.json, and learnsets, trainers,
species, forms and form_rows .json of data/intermediate and data/intermediate_kaizo). A .npz
counts by its arrays (names, dtypes, shapes, bytes; not the zip's timestamps), a .json by its
parsed value, any other file by its bytes with CRLF read as LF; so two builds of the same tables
hash equal. capture refuses a non-empty DIR unless its capture.json names the same commit (a
partial re-capture then adds or replaces parts); it fails before running on a dirty checkout
(unless --dirty-ok), when no commit is known, when the folder's earlier parts were captured from
other code (another content hash), or by an older snap_compare (see compare).

compare: reruns every output from the saved inputs (no redraw); its "gen" part redraws the
generator and checks it still draws the saved inputs (on capture, "gen" records them). The
engine parts compare on the captured inputs only, and an input field the capture lacks is pinned
at its empty / zero value (see Version independence); so a commit that adds Outcome draws needs
its own tests for their non-default values, and an explicit acceptance of its "gen" divergence
(its changed draws shift the generator's stream). Before
running it fails when: capture.json records no commit or no part list; a part to run was not
captured or is not "ok"; it runs a strict subset of the captured parts (unless --subset-ok);
both commits, or both content hashes, are the same (unless --same-ok; both hashes are
printed, so a checkout without git is tied to its code, not only to --commit); capture.json
was written by an older snap_compare (no ``content_hash`` or ``dirty`` key, or another
``format``: it says to re-capture); either side is dirty (unless --dirty-ok); or this
checkout's commit is unknown.

Parts: each seed chunk runs as six processes ("step": the draw on capture, jit(step_events),
ref_step; "jitstep": jit(step); "sample": jit(sample_outcome); "rest": enumerate_turn,
decision_actions, root_belief; "ai": ai_action_distribution on the AI states; "gen": the
generator check), then "roark.trace" and "roark.rank", one process each. The parent prints each
part's wall time and peak memory and flags a part over 60 s or 4 GB (OVER LIMIT).

Version independence: inputs are saved as plain {field: array} dicts, not pickled NamedTuples.
A later commit rebuilds them by field name, through ``UPGRADES`` first (below): a field it has
that the capture lacks and no row writes starts at the singles zero (BattleState:
``state.LAYOUT``'s shape and dtype) or at ``empty_outcome()``'s value (Outcome) and is reported
(``in_state.NAME`` / ``in_outcome.NAME``: a failure unless --allow-new names it); a captured
field it lacks and no row reads fails the part ("dropped field without an upgrade row"); a saved
BattleState field whose shape or dtype differs from ``LAYOUT``, a saved Outcome field whose shape
or dtype differs from ``empty_outcome()``'s, or saved actions that are not ``int32[B]`` is an
error (the part fails).

Layout upgrades: ``UPGRADES`` maps an older commit's fields to this checkout's, one row per
line: kind (``state`` / ``outcome``), the saved fields it reads (dtype, old shape after any
leading axes; "B" / "P" a battler / party axis of any format), the fields it writes, a fill, the
ledger ids whose rule decides the mapping and a one-line reason. Row builders: ``reshape`` (a
field's shape changed), ``derive`` (several reads, several writes), ``zero`` (a new field that is
0 wherever an older commit saved a state), ``dropped`` (a field gone with no successor: its
value leaves). A row runs where every read is there in its old dtype and shape (any leading
axes, the same for all reads: a batch, e.g. ``enumerate_turn``'s outcomes) and its new writes
are not (a newer capture); the reads leave unless kept. Rows run wherever the capture is read
back: the saved inputs (``in_state``, ``in_ai_state``: state; ``in_outcome``: outcome) as they
are loaded and in ``rebuild_state`` / ``rebuild_outcome``, and in the captured output entries of
every part, on each group of entries under one path prefix that holds the kind's ``MARKS`` (a
BattleState or Outcome at any depth: ``.root_belief[][]``, ``.step_events[]``,
``.ref_replay``, ``.enumerate_turn[]``). The compare header lists the rows with their ledger ids
and reasons; each part prints the rows it ran with their counts, and a line before the final one
sums them. Today's rows: ``starf_roll`` (B,) -> (B, 3) and ``Outcome.starf`` (B, 2) -> (B, 4)
(F-O3, G-27..G-30); ``weather``, ``weather_timer``, ``gravity``, ``trick_room`` (with
``uproar``) -> ``field_mask`` (``ids.clean_field_mask``) and ``weather_turns`` (the timer)
(G-4, G-11); ``party_ability``, ``nightmare``, ``camouflage``, ``starf_taken`` = 0 (F-O2,
G-31, G-20, G-27); ``acid_rain`` dropped, no successor (G-12, G-13);
``learn_pin``, ``exp_refused`` dropped, no successor (E3-B2-21).

Outputs (states, Outcomes, events, reference outputs) are flattened to "path" -> (kind, dtype,
shape, bytes) entries keyed by name. compare checks the entries present on both sides byte for
byte (dtype, shape and bytes) and lists, per part, the entries present on one side only
(indices folded to []; the entries under a path that raised on either side are left out: the
raise is the divergence). Columns a row invents (``new_columns``: ``Outcome.starf``'s 2-3, for
which the capture has no draw) are taken from the rerun before the compare and counted per part
as "sampler drift (new columns)", not a divergence; columns 0-1 compare strictly. The "gen"
part's divergence is "generator drift: first at seed S turn T", counted, not a failure.

An output that raised is stored as its exception (a refusal is an output) and compared like
one: a raise against a result (a value or a structure) is a divergence, never a one-sided entry
that --allow-new could pass. Floats (Python, np.float64) compare by their IEEE-754 bytes, as
arrays do: a float and an np.float64 of one value are equal; -0.0 against 0.0, or two NaN
payloads, differ. Each part prints its raise count and, per
function, count/calls and the first error.

Pass: compare prints "compare: ok" and exits 0 only when every precondition above holds, every
listed part ran (no skip, no error), no entry diverged (but "gen": generator drift), no field
was dropped without an upgrade row, every field new in this checkout is named by --allow-new,
and no function raised on every one of its calls in a part (unless named by --allow-all-raise),
no part ran zero steps or (but "gen") zero calls, and no part's raise share is over
--raise-fail. The final line counts the parts per class: ok / diverged / error (any other
failure: a skip, a crash, a dropped field, ...) / sampler drift / generator drift (exit 0 only
when diverged = error = 0), names the groups compared (a seed group by its name, a Roark part by
its full name) and the seed range with its count, or "seeds: none" when no seed part ran. Anything else exits 1 (2 for a usage error). capture exits
0 ("capture: ok") only when every part ran with no skip, no error and no all-raise function.

Pod (``git archive`` checkouts, no .git; one snapshot folder SNAP):

    uv run python NEW/tools/snap_compare.py capture SNAP 0-199 --repo BASE --commit <base sha>
    uv run python NEW/tools/snap_compare.py compare SNAP 0-199 --repo NEW --commit <new sha>
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import re
import struct
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
SEED_GROUPS = ("step", "jitstep", "sample", "rest", "ai", "gen")
ROARK_PARTS = ("roark.trace", "roark.rank")
KNOWN_PARTS = (*SEED_GROUPS, *ROARK_PARTS)
SHA = re.compile(r"[0-9a-f]{40}")
LIST_SHOWN = 8  # one-sided field names printed per part
RAISE_WARN = 0.5  # a part's raise share above which it warns
RAISE_FAIL = 0.95  # ... and fails
NO_CALLS_OK = ("gen",)  # groups with no timed call (the generator check)
# Beyond src/nuzline/**/*.py, the running script and the built files in NEEDS, what the child
# imports or reads (traced by its imports and opens): the content hash covers these. The
# checkout's own tools/snap_compare.py is left out: the script that runs is hashed instead.
HASHED = (
    "tests/__init__.py",
    "tests/strategies*.py",
    "tests/ai_legality.py",
    "tests/engine/__init__.py",
    "tests/engine/helpers.py",
    "tests/ai/*.py",
    "tests/fixtures/sample_party.toml",
    "tools/*.py",
)
# Paths whose untracked files make a git checkout dirty (tracked changes count anywhere).
WATCHED = (
    "src",
    "tools",
    "tests/__init__.py",
    "tests/strategies*",
    "tests/ai_legality.py",
    "tests/engine/__init__.py",
    "tests/engine/helpers.py",
    "tests/ai",
    "tests/fixtures/sample_party.toml",
)
CAPTURE_FORMAT = 2  # capture.json / the saved entries; older captures must be re-captured

# Built files each part reads (relative to the checkout).
VANILLA_TABLES = ("data/gen4_platinum/tables.npz", "data/gen4_platinum/manifest.json")
INTERMEDIATE = ("learnsets", "trainers", "species", "forms", "form_rows")  # IntermediateData
VANILLA_INTERMEDIATE = tuple(f"data/intermediate/{n}.json" for n in INTERMEDIATE)
NEEDS = {
    "seed": (*VANILLA_TABLES, *VANILLA_INTERMEDIATE),
    "roark.trace": (
        *VANILLA_TABLES,
        *VANILLA_INTERMEDIATE,
        "data/gen4_platinum_kaizo/tables.npz",
        "data/gen4_platinum_kaizo/manifest.json",
        *(f"data/intermediate_kaizo/{n}.json" for n in INTERMEDIATE),
        "tests/golden/kaizo_roark_02.json",
    ),
    "roark.rank": (*VANILLA_TABLES, *VANILLA_INTERMEDIATE),
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
    elif isinstance(x, float):  # Python and np.float64: by bytes, as arrays (NaN payloads too)
        out[path] = ("py", "float", "", struct.pack("<d", float(x)))
    elif isinstance(x, (str, bool, int)) or x is None:
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
        dt = np.dtype(a[1]) if a[0] == "arr" else None
        if a[0] == b[0] == "arr" and a[1:3] == b[1:3] and len(a[3]) == len(b[3]) and dt.itemsize:
            # Bytes, not values: -0.0 vs 0.0 and NaN payloads differ here though x == y.
            shape = tuple(ast.literal_eval(a[2]))
            rows = (np.frombuffer(a[3], np.uint8) != np.frombuffer(b[3], np.uint8)).reshape(
                -1, dt.itemsize
            )
            at = int(np.flatnonzero(rows.any(axis=1))[0])
            idx = tuple(int(i) for i in np.unravel_index(at, shape)) if shape else ()
            x = np.frombuffer(a[3], dt).reshape(shape)
            y = np.frombuffer(b[3], dt).reshape(shape)
            return f"{k}{list(idx)}: {x[idx]!r} -> {y[idx]!r}"
        if a[:2] == b[:2] == ("py", "float"):
            x, y = (struct.unpack("<d", e[3])[0] for e in (a, b))
            return f"{k}: {x!r} ({a[3].hex()}) -> {y!r} ({b[3].hex()})"
        return f"{k}: {a[:3]} {a[3][:80]!r} -> {b[:3]} {b[3][:80]!r}"
    return None


def raise_mismatch(want: dict, got: dict) -> str | None:
    """A path where one side raised and the other returned something (a value, or a structure
    whose entries lie under the path), or None. Never a one-sided entry: --allow-new cannot
    pass it."""
    for side, (a, b) in (("capture", (want, got)), ("this checkout", (got, want))):
        for k in sorted(a):
            if a[k][0] != "raise" or (k in b and b[k][0] == "raise"):
                continue
            under = any(o.startswith((f"{k}.", f"{k}[", f"{k}#")) for o in b)
            if k in b or under:
                other = "this checkout" if side == "capture" else "the capture"
                return f"{k}: raised on the {side} ({a[k][1]}), {other} returned a result"
    return None


def compare_dumps(record: list, dump: list) -> tuple[str | None, set[str], set[str]]:
    """``(first divergence or None, only_base, only_new)`` of the captured ``record``
    ``[(key, digest, entries)]`` against the rerun ``dump`` ``[(key, entries)]``."""
    if [k for k, _, _ in record] != [k for k, _ in dump]:
        return f"step list differs: {len(record)} captured, {len(dump)} rerun", set(), set()
    diverged, only_base, only_new = None, set(), set()
    for (key, _, want), (_, got) in zip(record, dump, strict=True):
        raised = [k for k, e in (*want.items(), *got.items()) if e[0] == "raise"]
        only_base |= {fold(k) for k in set(want) - set(got) if not under(k, raised)}
        only_new |= {fold(k) for k in set(got) - set(want) if not under(k, raised)}
        diff = raise_mismatch(want, got) or first_diff(want, got)
        if diff and not diverged:
            diverged = f"{key}: {diff}"
    return diverged, only_base, only_new


def under(key: str, paths: list[str]) -> bool:
    """``key`` is one of ``paths`` or lies under one (a raise against a result is a divergence,
    raise_mismatch; its entries are not one-sided fields)."""
    return any(key == p or key.startswith((f"{p}.", f"{p}[", f"{p}#")) for p in paths)


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


def check_layout(
    saved: dict, layout: dict, what: str = "state fields", ref: str = "LAYOUT"
) -> None:
    """Raise unless every saved field ``layout`` names has its shape and dtype."""
    import numpy as np

    bad = []
    for name in sorted(set(saved) & set(layout)):
        shape, dtype = layout[name]
        a = np.asarray(saved[name])
        if a.shape != tuple(shape) or a.dtype != np.dtype(dtype):
            bad.append(f"{name} {a.dtype}{a.shape} != {np.dtype(dtype)}{tuple(shape)}")
    if bad:
        raise ValueError(f"saved {what} differ from {ref}: " + "; ".join(bad))


def check_actions(saved, battlers: int) -> None:
    """Raise unless the saved actions are ``int32[battlers]`` (the legal_action_pair form)."""
    import numpy as np

    a = np.asarray(saved)
    if a.shape != (battlers,) or a.dtype != np.dtype(np.int32):
        raise ValueError(f"saved actions {a.dtype}{a.shape} != int32({battlers},)")


# --- layout upgrades (see the docstring) -----------------------------------------------

# Named axes of an old shape: a battler and a party axis take any count a format has
# (engine.format: B 2 or 4, P 2, 3 or 4), so a row matches singles and doubles alike.
DIMS = {"B": (2, 4), "P": (2, 3, 4)}
# Fields that make a captured output group (the entries under one path prefix) a BattleState or
# an Outcome, so that a row of that kind runs on it (a state saved by any commit has these).
MARKS = {"state": ("species", "active", "hp"), "outcome": ("hit", "roll", "crit")}


@dataclasses.dataclass(frozen=True)
class Upgrade:
    """One row of ``UPGRADES``: the saved fields ``reads`` of a ``kind`` ("state" / "outcome")
    BattleState or Outcome (``(field, dtype, old shape)``, the shape after any leading axes:
    "B" / "P" a battler / party axis, ``DIMS``) read as this checkout's ``writes`` through
    ``fill({read: array}, lead)``, ``lead`` the leading axes (a batch) the reads share. The reads
    leave the state unless named in ``keeps`` or ``writes``. ``fill`` None: the read is a field
    dropped with no successor (``writes`` empty). A row runs only when every read is there in its
    old dtype and shape and no write that is not also a read is there yet (a newer capture).
    ``new_columns``: ``(field, columns)`` the fill invents along the last axis (no old data:
    compared as sampler drift, see the docstring). ``ledger``: the ledger ids whose rule decides
    the mapping; ``reason``: why the mapping is lossless."""

    kind: str
    reads: tuple
    writes: tuple
    fill: object
    label: str
    ledger: str
    reason: str
    keeps: tuple = ()
    new_columns: tuple = ()

    @property
    def name(self) -> str:
        return f"{self.kind}.{self.label}"

    def lead(self, saved: dict, batched: bool = True) -> tuple | None:
        """The leading axes this row runs with on ``saved`` (a field dict), or None; none
        unless ``batched`` (a saved input is one state)."""
        import numpy as np

        leads = set()
        for field, dtype, old in self.reads:
            if field not in saved:
                return None
            a = np.asarray(saved[field])
            n = a.ndim - len(old)
            if a.dtype.str != dtype or n < 0:
                return None
            if not all(d == m or m in DIMS.get(d, ()) for d, m in zip(old, a.shape[n:])):
                return None
            leads.add(a.shape[:n])
        if len(leads) != 1 or any(w in saved for w in self.writes if w not in self.fields):
            return None
        lead = leads.pop()
        return lead if batched or not lead else None

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(r[0] for r in self.reads)


def _shape(t) -> str:
    return "(" + ", ".join(map(str, t)) + ("," if len(t) == 1 else "") + ")"


def reshape(kind, field, dtype, old, new, fill, ledger, reason, new_columns=()):
    """A field whose shape changed: ``old`` -> ``new`` (shapes after any leading axes)."""
    label = f"{field} {_shape(old)} -> {_shape(new)}"
    reads = ((field, dtype, old),)
    cols = ((field, new_columns),) if new_columns else ()
    return Upgrade(kind, reads, (field,), fill, label, ledger, reason, new_columns=cols)


def derive(kind, reads, writes, fill, ledger, reason, keeps=()):
    """New fields computed from old ones (several reads, several writes)."""
    label = f"{'+'.join(r[0] for r in reads)} -> {'+'.join(writes)}"
    return Upgrade(kind, reads, writes, fill, label, ledger, reason, keeps=keeps)


def zero(kind, field, like, ledger, reason):
    """A new field whose value at every point the older commit saved a state is 0, shaped as
    the old field ``like`` ``(name, dtype, old shape)`` with the new field's dtype: ``like`` gives
    the shape and the leading axes."""
    name, dtype, old, new_dtype = like

    def fill(a, lead):
        import numpy as np

        return {field: np.zeros(np.shape(a[name]), new_dtype)}

    label = f"{field} = 0 (shape of {name})"
    return Upgrade(kind, ((name, dtype, old),), (field,), fill, label, ledger, reason, (name,))


def dropped(kind, field, dtype, old, ledger, reason):
    """A field gone from the engine with no successor: the capture's value is dropped."""
    label = f"{field} {_shape(old)}: dropped, no successor"
    return Upgrade(kind, ((field, dtype, old),), (), None, label, ledger, reason)


def _starf_queue(a, lead):
    """``starf_roll`` int8[B] -> the turn's queue int8[B, 3]: the draw in entry 0, entries 1-2
    at 0 (what step_events / replace store where no draw is loaded), F-O3 (G-27)."""
    import numpy as np

    old = a["starf_roll"]
    new = np.zeros((*old.shape, 3), old.dtype)
    new[..., 0] = old
    return {"starf_roll": new}


def _starf_columns(a, lead):
    """``Outcome.starf`` int8[B, 2] -> int8[B, 4]: columns 0-1 kept; the old engine's second and
    third eats read the one draw of column 0, so columns 2-3 copy it (F-O3, G-27)."""
    import numpy as np

    old = a["starf"]
    return {"starf": np.concatenate([old, old[..., :1], old[..., :1]], axis=-1)}


def _field_mask(a, lead):
    """``field_mask`` = ``ids.clean_field_mask`` of the old fields (the engine's own encoding
    of them, as tests.strategies draws a state): the weather's TEMP bit while ``weather_timer``
    runs, else its PERM bit (deep fog: bit 15), ``gravity << 12``, ``trick_room << 16``, battler
    b's Uproar bit 8 + b while ``uproar[b]``; ``weather_turns`` = ``weather_timer`` (0 for a
    permanent weather: make_state's weatherTurns), FCM 4.4 (G-4, G-11)."""
    import numpy as np

    from nuzline.engine.ids import clean_field_mask

    mask = np.zeros(lead, np.int32)
    for i in np.ndindex(*lead):
        mask[i] = clean_field_mask(
            int(a["weather"][i]),
            int(a["weather_timer"][i]),
            int(a["gravity"][i]),
            int(a["trick_room"][i]),
            a["uproar"][i].tolist(),
        )
    return {"field_mask": mask, "weather_turns": a["weather_timer"].astype(np.int32)}


TABLES = None  # the child's vanilla tables (Ctx), for a row that reads them (_camouflage)


def _personal_types(species, form, ability_form, ability_slot, item) -> tuple[int, int]:
    """``damage.personal_types`` on ``TABLES`` (the old engine's ``effects.camouflaged``
    compared a battler's types with it), memoized by its arguments."""
    key = (species, form, ability_form, ability_slot, item)
    if key not in _PERSONAL:
        from nuzline.engine.damage import personal_types

        if TABLES is None:
            raise RuntimeError("the camouflage upgrade reads the tables: none are loaded")
        _PERSONAL[key] = tuple(int(x) for x in personal_types(TABLES, *key))
    return _PERSONAL[key]


_PERSONAL: dict[tuple, tuple[int, int]] = {}


def _camouflage(a, lead):
    """``camouflage`` (G-20) of an older commit's state: its MOVE_EFFECT_CAMOUFLAGE was
    ``effects.camouflaged``, the active battler's types differing from its personal row's
    (``damage.personal_types``; only a Camouflage that ran writes ``types``), which also read a
    Transformed copy of a camouflaged foe as camouflaged. The bit is that inference on a battler
    not Transformed, 0 on a Transformed one (BtlCmd_Transform does not copy the mask,
    battle_script.c:5346-5348): the one reading G-20 changed."""
    from types import SimpleNamespace

    import numpy as np

    from nuzline.engine.format import format_of

    unbatched = {f: np.empty(a[f].shape[len(lead) :]) for f in ("active", "species")}
    fmt = format_of(SimpleNamespace(**unbatched))  # B and P: the state's format
    out = np.zeros(a["active"].shape, np.int8)
    for i in np.ndindex(*lead):
        for b in range(fmt.B):
            pm = (*i, fmt.party_of[b], int(a["active"][(*i, b)]))
            mon = (a[f][pm] for f in ("species", "form", "ability_form", "ability_slot", "item"))
            own = _personal_types(*(int(x) for x in mon))
            moved = tuple(int(x) for x in a["types"][pm]) != own
            out[(*i, b)] = moved and not a["transformed"][(*i, b)]
    return {"camouflage": out}


_I1, _I2 = "|i1", "<i2"
_MON_READS = tuple(
    (f, d, ("P", 6))
    for f, d in (("species", _I2), ("form", _I1), ("ability_form", _I1), ("ability_slot", _I1),
                 ("item", _I2))
) + (("types", _I1, ("P", 6, 2)), ("active", _I1, ("B",)), ("transformed", _I1, ("B",)))  # fmt: skip
_FIELD_READS = (
    ("weather", _I1, ()),
    ("weather_timer", _I1, ()),
    ("gravity", _I1, ()),
    ("trick_room", _I1, ()),
    ("uproar", _I1, ("B",)),
)
_ACTIVE = ("active", _I1, ("B",), "|i1")  # a battler field: zero rows of int8[B]
UPGRADES = (
    reshape("state", "starf_roll", _I1, ("B",), ("B", 3), _starf_queue, "F-O3, G-27", "the old draw is entry 0; entries 1-2 at 0, what step_events / replace store where no draw is loaded"),
    reshape("outcome", "starf", _I1, ("B", 2), ("B", 4), _starf_columns, "F-O3, G-27..G-30", "columns 0-1 kept; the old engine's 2nd and 3rd eats read column 0, so 2-3 copy it", (2, 3)),
    derive("state", _FIELD_READS, ("field_mask", "weather_turns"), _field_mask, "G-4, G-11", "ids.clean_field_mask of the old fields (TEMP bit while the timer runs, else PERM); weather_turns = the timer", ("uproar",)),
    zero("state", "party_ability", ("species", _I2, ("P", 6), "<i2"), "F-O2, G-1..G-3", "0 = no written ability, the personal row's (step._restore_ability); initial_arrays leaves it 0 and the old engine wrote none"),
    zero("state", "nightmare", _ACTIVE, "G-31", "the old engine refused Nightmare (effect 107): no battler had the bit"),
    derive("state", _MON_READS, ("camouflage",), _camouflage, "G-20", "the old engine's own bit (effects.camouflaged: types != the personal row's), but 0 on a Transformed battler", tuple(r[0] for r in _MON_READS)),
    zero("state", "starf_taken", _ACTIVE, "F-O3, G-27", "0 between turns and after step / replace: every point a state is saved"),
    dropped("state", "acid_rain", _I1, (), "G-12, G-13", "E2's interim Acid Rain guard flag (main 08a57000, SF2 (f)); batch G models the write in field_mask and removed the flag"),
    dropped("state", "learn_pin", _I1, (), "E3-B2-21", "1 + the selected slot a deferred learn masked (main 08a57000); the prompt now pauses at the payout, before the slot is re-read (review b2 #6): learn_site is a pause site, not a slot, and 0 at every saved point"),
    dropped("state", "exp_refused", _I1, (), "E3-B2-21", "main's sticky window (c) refusal code; the payout's prompt now pauses the turn, so learn work never refuses (review b2 #6)"),
)  # fmt: skip
UPGRADED: dict[str, int] = {}  # Upgrade.name -> fields upgraded in this process
INPUT_KINDS = {"in_state": "state", "in_ai_state": "state", "in_outcome": "outcome"}


def upgrade_fields(
    kind: str, saved: dict, invented: dict | None = None, batched: bool = False
) -> dict:
    """``saved`` (a field dict of ``kind``; with ``batched``, leading axes allowed) through
    every ``UPGRADES`` row of ``kind`` that runs on it (``Upgrade.lead``), in table order; every
    other field as it was. ``invented``, when given, gets ``{field: columns}`` of each row's
    ``new_columns`` that ran."""
    import numpy as np

    out = dict(saved)
    for u in UPGRADES:
        lead = u.lead(out, batched) if u.kind == kind else None
        if lead is None:
            continue
        UPGRADED[u.name] = UPGRADED.get(u.name, 0) + 1
        made = u.fill({f: np.asarray(out[f]) for f in u.fields}, lead) if u.fill else {}
        for f in u.fields:
            if f not in u.keeps and f not in made:
                del out[f]
        out.update(made)
        if invented is not None:
            invented.update(u.new_columns)
    return out


def upgrade_inputs(inputs: list) -> list:
    """Saved ``[(key, {in_state, in_actions, in_outcome} | {in_ai_state})]``, upgraded."""
    return [
        (key, {n: upgrade_fields(INPUT_KINDS[n], v) if n in INPUT_KINDS else v
               for n, v in inp.items()})
        for key, inp in inputs
    ]  # fmt: skip


def upgrade_entries(entries: dict, invented: dict | None = None) -> dict:
    """Captured canon ``entries`` with each BattleState / Outcome among them (the array entries
    under one path prefix that hold the kind's ``MARKS``; a batch of them too) upgraded by
    ``upgrade_fields``: dropped reads leave, writes join. ``invented`` gets ``{path: columns}``
    of each upgraded entry whose last-axis ``columns`` the upgrade invented."""
    import numpy as np

    groups: dict[str, dict[str, str]] = {}
    for path, e in entries.items():
        if e[0] == "arr" and "." in path:
            prefix, _, name = path.rpartition(".")
            groups.setdefault(prefix, {})[name] = path
    out = dict(entries)
    for prefix, names in groups.items():
        kind = next((k for k, m in MARKS.items() if all(f in names for f in m)), None)
        if kind is None or not any(f in names for u in UPGRADES for f in u.fields):
            continue
        saved = {}
        for name, path in names.items():
            e = entries[path]
            saved[name] = np.frombuffer(e[3], np.dtype(e[1])).reshape(ast.literal_eval(e[2]))
        cols: dict = {}
        new = upgrade_fields(kind, saved, cols, batched=True)
        for name in set(saved) - set(new):
            del out[names[name]]
        for name, a in new.items():
            if name in saved and a is saved[name]:
                continue
            a = np.ascontiguousarray(a)
            out[f"{prefix}.{name}"] = ("arr", a.dtype.str, str(a.shape), a.tobytes())
        if invented is not None:
            invented.update({f"{prefix}.{f}": c for f, c in cols.items()})
    return out


def settle_new_columns(want: dict, got: dict, invented: dict) -> int:
    """Take each ``invented`` column (``{path: columns}`` from ``upgrade_entries``) of ``want``
    from ``got``, so the compare reads only the columns the capture holds; the number of entries
    whose invented columns differed (sampler drift, new columns: counted, not a divergence)."""
    import numpy as np

    drift = 0
    for path, cols in invented.items():
        a, b = want.get(path), got.get(path)
        if a is None or b is None or a[0] != "arr" or a[:3] != b[:3]:
            continue
        shape = ast.literal_eval(a[2])
        x = np.frombuffer(a[3], np.dtype(a[1])).reshape(shape).copy()
        y = np.frombuffer(b[3], np.dtype(b[1])).reshape(shape)
        idx = list(cols)
        if x[..., idx].tobytes() != y[..., idx].tobytes():
            drift += 1
            x[..., idx] = y[..., idx]
            want[path] = ("arr", a[1], a[2], np.ascontiguousarray(x).tobytes())
    return drift


def rebuild_state(saved: dict):
    """A ``BattleState`` of this checkout from a saved field dict (see the docstring)."""
    import numpy as np

    from nuzline.engine.state import LAYOUT, BattleState

    saved = upgrade_fields("state", saved)
    check_layout(saved, LAYOUT)
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
    import numpy as np

    from nuzline.engine.outcome import Outcome, empty_outcome

    empty = jax.device_get(empty_outcome())
    saved = upgrade_fields("outcome", saved)
    check_layout(
        saved,
        {
            n: (np.shape(getattr(empty, n)), np.asarray(getattr(empty, n)).dtype)
            for n in Outcome._fields
        },
        "outcome fields",
        "empty_outcome()",
    )
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
        global TABLES
        TABLES = self.tables  # the camouflage upgrade reads them (_camouflage)
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
    from nuzline.engine.actions import decision_actions
    from nuzline.engine.enumerate import enumerate_turn
    from nuzline.engine.step import root_belief

    return {
        "enumerate_turn": timed("enumerate_turn", lambda: enumerate_turn(t, s, acts)),
        "decision_actions": timed("decision_actions", lambda: ctx.jit(decision_actions)(t, s)),
        "root_belief": timed("root_belief", lambda: root_belief(t, s)),
    }


def ai_outputs(ctx: Ctx, s) -> dict:
    """The "ai" group's output on one AI state."""
    from nuzline.ai.policy import ai_action_distribution

    return {
        "ai_action_distribution": timed(
            "ai_action_distribution", lambda: ai_action_distribution(ctx.tables, s)
        )
    }


def draw_inputs(ctx: Ctx, seed_value: int, saved: dict | None = None) -> list[tuple[str, dict]]:
    """``[(key, {in_state, in_actions, in_outcome})]`` (plain dicts) drawn as
    tools/differential.py draws a seed (TURNS examples, no shrink), each pause resumed as
    _step_both resumes it, then per example ``(key, {in_ai_state})``: a state the vanilla AI
    accepts (``gated_states``), drawn after the segments: the segments of a seed's first example
    draw as without it, later examples do not (Hypothesis shares one random across the examples
    of a seed, so the extra draw shifts them). With ``saved``
    (compare's generator check) a pause and its state are read from the saved inputs instead of
    stepping, so the check loads no step program."""
    import jax
    from hypothesis import HealthCheck, Phase, given, seed, settings
    from hypothesis import strategies as st

    from nuzline.engine.step import step_events
    from tests.ai.test_ai_legality import gated_states
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
        ai_s = data.draw(gated_states(ctx.tables, ctx.names))
        ai_inputs = {"in_ai_state": ai_s}
        out.append((f"seed {seed_value} turn {turn[0]} ai", plain(jax.device_get(ai_inputs))))
        turn[0] += 1

    start = time.monotonic()
    gen()
    TIMES["draw"] = TIMES.get("draw", 0.0) + time.monotonic() - start
    return out


def seed_steps(ctx: Ctx, directory: Path, mode: str, lo: int, hi: int, group: str):
    import numpy as np

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
            inputs = upgrade_inputs(pickle.load(f))
    if group == "gen":  # compare only: the generator still draws the saved inputs
        saved = dict(inputs)
        return [kv for v in range(lo, hi + 1) for kv in draw_inputs(ctx, v, saved)]
    steps = []
    for key, inp in inputs:
        if "in_ai_state" in inp:  # an AI state: the "ai" group's only input
            if group == "ai":
                steps.append((key, ai_outputs(ctx, rebuild_state(inp["in_ai_state"]))))
            continue
        if group == "ai":
            continue
        seed_value = int(key.split()[1])
        s, o = rebuild_state(inp["in_state"]), rebuild_outcome(inp["in_outcome"])
        check_actions(inp["in_actions"], np.asarray(s.active).shape[0])
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
            saved = pickle.load(f)
        record, drift = [], 0
        for (key, h, d), (rerun, got) in zip(saved, dump, strict=False):
            invented: dict = {}
            want = upgrade_entries(d, invented)
            drift += key == rerun and bool(settle_new_columns(want, got, invented))
            record.append((key, h, want))
        record += [(key, h, upgrade_entries(d)) for key, h, d in saved[len(record) :]]
        result["sampler_drift"] = drift
        result["diverged"], only_base, only_new = compare_dumps(record, dump)
        only_base |= INPUT_ONE_SIDED["base"]
        only_new |= INPUT_ONE_SIDED["new"]
        result["only_base"], result["only_new"] = sorted(only_base), sorted(only_new)
    result["upgraded"] = dict(sorted(UPGRADED.items()))
    result["peak_bytes"] = _peak_bytes()
    print("TIMES " + json.dumps({k: round(v, 1) for k, v in TIMES.items()}), file=sys.stderr)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


def check_imported_from(module_file: str, root: Path) -> None:
    """Raise unless ``module_file`` lies under ``root / "src"`` (the --repo checkout's code)."""
    src = os.path.normcase(str((root / "src").resolve()))
    got = os.path.normcase(str(Path(module_file).resolve()))
    try:
        inside = os.path.commonpath([src, got]) == src
    except ValueError:  # another drive
        inside = False
    if not inside:
        raise RuntimeError(f"nuzline imported from {module_file}, not from {root / 'src'}")


def _names(fields: list[str]) -> str:
    shown = ", ".join(fields[:LIST_SHOWN])
    return shown + (f" (+{len(fields) - LIST_SHOWN} more)" if len(fields) > LIST_SHOWN else "")


def _counts(upgraded: dict[str, int]) -> str:
    return ", ".join(f"{name} x {n}" for name, n in sorted(upgraded.items())) or "none"


def _csv(text: str | None) -> set[str]:
    return {n.strip() for n in (text or "").split(",") if n.strip()}


def allowed(path: str, allow: set[str]) -> bool:
    """``path`` (a folded entry path) names an ``--allow-new`` field as one of its segments."""
    return any(seg in allow for seg in re.split(r"[.\[\]#]+", path) if seg)


# --- commits and the part list ----------------------------------------------------------


class UsageError(ValueError):
    pass


def plan_parts(seeds: str, chunk: int, parts_arg: str | None, no_roark: bool):
    """``(lo, hi, parts)``; a UsageError for a bad range, chunk or part name, or no part."""
    try:
        a, _, b = seeds.partition("-")
        lo, hi = int(a), int(b or a)
    except ValueError:
        raise UsageError(f"seed range {seeds!r} is not A or A-B") from None
    if hi < lo:
        raise UsageError(f"seed range {seeds!r}: B < A")
    if chunk < 1:
        raise UsageError(f"--chunk {chunk}: must be at least 1")
    groups = None
    if parts_arg is not None:
        groups = [g.strip() for g in parts_arg.split(",") if g.strip()]
        unknown = [g for g in groups if g not in KNOWN_PARTS]
        if unknown:
            raise UsageError(
                f"--parts: unknown {', '.join(unknown)} (known: {', '.join(KNOWN_PARTS)})"
            )
    parts = [
        f"{x}-{min(x + chunk - 1, hi)}.{g}"
        for x in range(lo, hi + 1, chunk)
        for g in SEED_GROUPS
        if groups is None or g in groups
    ]
    if not no_roark:
        parts += [p for p in ROARK_PARTS if groups is None or p in groups]
    if not parts:
        raise UsageError("no part to run (--parts / --no-roark leave nothing)")
    return lo, hi, parts


def content_hash(repo: Path) -> str:
    """sha256 over the code the child runs and the files it reads (see the docstring): each
    file's relative path and content, in path order, then the running script (``__file__``,
    which is not ``repo``'s own tools/snap_compare.py when it captures another checkout);
    missing built files are skipped (the part is SKIPPED anyway). Content, so that two builds
    of the same tables hash equal: a .npz file is its arrays (sorted names, dtype, shape,
    bytes; np.savez_compressed stamps the zip with the build time), a .json file its parsed
    value re-dumped in its own order (whitespace and line endings left out), any other file its
    bytes with CRLF read as LF."""
    files = set((repo / "src" / "nuzline").rglob("*.py"))
    for pattern in HASHED:
        files |= set(repo.glob(pattern))
    files.discard(repo / "tools" / "snap_compare.py")
    files |= {repo / f for fs in NEEDS.values() for f in fs}
    named = [(p.relative_to(repo).as_posix(), p) for p in files if p.is_file()]
    h = hashlib.sha256()
    for name, p in [*sorted(named), ("<running>/snap_compare.py", Path(__file__).resolve())]:
        data = file_content(p)
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def file_content(p: Path) -> bytes:
    """``p``'s content as content_hash reads it."""
    if p.suffix == ".npz":
        import numpy as np

        h = hashlib.sha256()
        with np.load(p, allow_pickle=False) as z:
            for name in sorted(z.files):
                a = np.ascontiguousarray(z[name])
                h.update(f"{name}\0{a.dtype.str}\0{a.shape}\0".encode())
                h.update(a.tobytes())
        return h.digest()
    if p.suffix == ".json":
        value = json.loads(p.read_text(encoding="utf-8"))
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return p.read_bytes().replace(b"\r\n", b"\n")


def git_info(repo: Path) -> dict | None:
    """``{"commit", "dirty", "changes"}`` when ``repo`` is the top of a git checkout, else None
    (no git, or a folder inside some other repository). Dirty: a tracked change anywhere or an
    untracked file under src/, tools/, tests/strategies*, tests/engine/helpers.py or tests/ai/."""

    def git(*a):
        try:
            r = subprocess.run(
                ["git", "-C", str(repo), *a], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if top is None or os.path.normcase(str(Path(top).resolve())) != os.path.normcase(
        str(repo.resolve())
    ):
        return None
    commit = git("rev-parse", "HEAD")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("status", "--porcelain", "--untracked-files=all", "--", *WATCHED)
    if commit is None or tracked is None or untracked is None:
        return None
    changes = sorted(set(tracked.splitlines()) | set(untracked.splitlines()))
    return {"commit": commit, "dirty": bool(changes), "changes": changes}


def identify(repo: Path, given: str | None) -> tuple[dict, list[str]]:
    """This checkout's ``{"commit", "dirty", "source", "content_hash"}`` and its problems: an
    invalid --commit, a --commit git contradicts, no commit at all, or ``dirty: ...``.
    ``dirty`` is None (not checkable) without git."""
    problems = []
    if given is not None and not SHA.fullmatch(given):
        problems.append(f"--commit {given!r} is not a full 40-hex commit")
    g = git_info(repo)
    if g is not None:
        if given is not None and given != g["commit"]:
            problems.append(f"--commit {given} but git HEAD of {repo} is {g['commit']}")
        if g["dirty"]:
            problems.append(f"dirty: {repo}: {_names(g['changes'])} (--dirty-ok to allow)")
        here = {"commit": g["commit"], "dirty": g["dirty"], "source": "git"}
        return {**here, "content_hash": content_hash(repo)}, problems
    if given is None:
        problems.append(f"{repo} has no git and no --commit was given: the commit is unknown")
    here = {"commit": given, "dirty": None, "source": "--commit"}
    return {**here, "content_hash": content_hash(repo)}, problems


def _show(c: dict) -> str:
    marks = {True: " (dirty)", None: " (dirty: not checkable)"}
    dirty = marks.get(c["dirty"], "") if "dirty" in c else ""  # no key: an older capture
    return f"{c.get('commit')}{dirty}"


def read_capture(directory: Path) -> dict | None:
    try:
        return json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_capture(directory: Path, record: dict) -> None:
    (directory / "capture.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def _keep(problems: list[str], dirty_ok: bool) -> list[str]:
    return [p for p in problems if not (dirty_ok and p.startswith("dirty:"))]


def capture_problems(args, directory: Path, here: dict, problems: list[str]) -> list[str]:
    """Why capture must not start: no commit, a dirty checkout, or a folder of another commit."""
    out = _keep(problems, args.dirty_ok)
    if directory.exists() and any(directory.iterdir()):
        old = read_capture(directory)
        if old is None or not old.get("commit"):
            out.append(f"{directory} is not empty and records no commit: use an empty folder")
        elif old["commit"] != here["commit"]:
            out.append(
                f"{directory} holds a capture of {old['commit']}, not {here['commit']}: "
                "use an empty folder"
            )
        elif why := old_capture(old):
            out.append(f"{directory}: {why}")
        elif old["content_hash"] != here["content_hash"]:
            out.append(
                f"{directory} holds parts captured from other code (content hash "
                f"{old['content_hash']}, this checkout {here['content_hash']}): "
                "use an empty folder"
            )
    return out


def old_capture(c: dict) -> str | None:
    """Why capture.json ``c`` was written by an older snap_compare (no ``content_hash`` or
    ``dirty`` key, or another ``format``), or None. Its hash, dirty state or saved floats
    cannot be checked: it is refused, never read as "not checkable"."""
    lacks = [k for k in ("content_hash", "dirty") if k not in c]
    if not lacks and c.get("format") == CAPTURE_FORMAT:
        return None
    what = f"no {' / '.join(lacks)}" if lacks else f"format {c.get('format')}"
    return (
        f"the capture was made by an older snap_compare (capture.json: {what}; this script "
        f"writes format {CAPTURE_FORMAT}): re-capture it with this script into an empty folder"
    )


def compare_problems(args, base: dict | None, here: dict, problems, parts) -> list[str]:
    """Why compare must not start (the docstring's list)."""
    out = _keep(problems, args.dirty_ok)
    if base is None or not base.get("commit"):
        return [*out, "the capture records no commit (capture.json)"]
    why = old_capture(base)
    if why:
        return [*out, why]
    if base["dirty"] and not args.dirty_ok:
        out.append(f"the capture's checkout was dirty ({base['commit']}; --dirty-ok to allow)")
    if base["dirty"] is None:  # the key is there (old_capture): recorded without git
        print("the capture's checkout: dirty state not checkable (no git)", flush=True)
    if base["commit"] == here["commit"] and not args.same_ok:
        out.append("same commit on both sides (--same-ok to allow)")
    if not base["content_hash"]:
        out.append("the capture records no content hash (capture.json)")
    elif base["content_hash"] == here["content_hash"] and not args.same_ok:
        out.append(f"same content hash on both sides ({here['content_hash']}; --same-ok to allow)")
    captured = base.get("parts")
    if not isinstance(captured, dict) or not captured:
        return [*out, "the capture records no part list (capture.json)"]
    absent = [p for p in parts if p not in captured]
    if absent:
        out.append(f"not captured: {_names(absent)}")
    not_ok = [f"{p} ({captured[p]})" for p in parts if p in captured and captured[p] != "ok"]
    if not_ok:
        out.append(f"captured but not ok: {_names(not_ok)}")
    left = sorted(set(captured) - set(parts))
    if left and not args.subset_ok:
        out.append(f"runs a strict subset; left out: {_names(left)} (--subset-ok to allow)")
    return out


# --- parent -----------------------------------------------------------------------------


def run_part(mode: str, directory: Path, part: str) -> tuple[int, str, str]:
    """Run one part in a child process: ``(returncode, stdout, stderr)``."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_child", mode, str(directory), part,
         str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )  # fmt: skip
    return proc.returncode, proc.stdout, proc.stderr


def check_shares(warn: float, fail: float) -> None:
    """A UsageError unless 0 <= warn <= fail < 1 (NaN, or a share of 1 or more, would turn the
    raise-share check off)."""
    if not 0 <= warn <= fail < 1:
        raise UsageError(f"--raise-warn {warn} / --raise-fail {fail}: need 0 <= warn <= fail < 1")


def parent(args) -> int:
    try:
        lo, hi, parts = plan_parts(args.seeds, args.chunk, args.parts, args.no_roark)
        check_shares(args.raise_warn, args.raise_fail)
    except UsageError as e:
        print(f"{args.mode}: usage error: {e}", flush=True)
        return 2
    directory = Path(args.dir).resolve()
    allow, allow_raise = _csv(args.allow_new), _csv(args.allow_all_raise)
    print(f"{args.mode}: checkout {ROOT}, snapshot {directory}, {len(parts)} parts", flush=True)
    here, problems = identify(ROOT, args.commit)
    if args.mode == "capture":
        print(f"capture commit {_show(here)}; content hash {here['content_hash']}", flush=True)
        stop = capture_problems(args, directory, here, problems)
    else:
        base = read_capture(directory)
        print(f"capture commit {_show(base or {})}; this checkout {_show(here)}", flush=True)
        print(
            f"content hash: capture {(base or {}).get('content_hash')}; "
            f"this checkout {here['content_hash']}",
            flush=True,
        )
        stop = compare_problems(args, base, here, problems, parts)
        for u in UPGRADES:
            print(f"layout upgrade: {u.name} ({u.ledger}): {u.reason}", flush=True)
    if stop:
        for p in stop:
            print(f"FAIL: {p}", flush=True)
        print(f"{args.mode}: FAILED ({len(stop)} precondition(s); nothing run)", flush=True)
        return 1
    record = None
    if args.mode == "capture":
        directory.mkdir(parents=True, exist_ok=True)
        record = read_capture(directory) or {"commit": here["commit"], "parts": {}, "seeds": []}
        dirties = [record.get("dirty", False), here["dirty"]]
        record.update(
            format=CAPTURE_FORMAT,
            source=here["source"],
            dirty=True if True in dirties else (None if None in dirties else False),
            content_hash=here["content_hash"],
        )
        record["parts"].update({p: "pending" for p in parts})
        record["seeds"].append([lo, hi, args.chunk])
        write_capture(directory, record)
    bad, overs, skipped, total = 0, [], [], time.monotonic()
    upgraded: dict[str, int] = {}
    classes = dict.fromkeys(CLASSES, 0)
    for part in parts:
        failed, kind = 1, "error"
        lack = missing(part)
        if lack:
            print(f"{part}: SKIPPED: needs {', '.join(lack)} (built tables; not a pass)",
                  flush=True)  # fmt: skip
            skipped.append(part)
        else:
            start = time.monotonic()
            code, stdout, stderr = run_part(args.mode, directory, part)
            took = time.monotonic() - start
            line = next((ln for ln in stdout.splitlines() if ln.startswith("RESULT ")), None)
            if code or line is None:
                print(f"{part}: ERROR (exit {code}, {took:.0f} s)\n{stdout[-3000:]}"
                      f"\n{stderr[-6000:]}", flush=True)  # fmt: skip
            else:
                res = json.loads(line[7:])
                failed = report(args, part, res, took, allow, allow_raise, stderr)
                kind = part_class(part, res, failed)
                for name, n in res.get("upgraded", {}).items():
                    upgraded[name] = upgraded.get(name, 0) + n
                overs += [part] if took > LIMIT_S or res["peak_bytes"] > LIMIT_BYTES else []
        bad += failed
        classes[kind] += 1
        if record is not None:
            record["parts"][part] = "failed" if failed else "ok"
            write_capture(directory, record)
    counts = ""
    if args.mode == "compare":
        print(f"layout upgrades run: {_counts(upgraded)}", flush=True)
        counts = "parts: " + ", ".join(f"{n} {c}" for c, n in classes.items()) + "; "
    verdict = "ok" if not bad else f"FAILED ({bad} failure(s), skips included)"
    seeded = [p for p in parts if not p.startswith("roark.")]
    groups = ", ".join(dict.fromkeys(p.split(".", 1)[1] if p in seeded else p for p in parts))
    seeds = f"seeds: {lo}-{hi} ({hi - lo + 1})" if seeded else "seeds: none"
    print(
        f"{args.mode}: {verdict} ({time.monotonic() - total:.0f} s); {counts}"
        f"groups: {groups}; {seeds}; "
        f"over 60 s / 4 GB: {', '.join(overs) or 'none'}; "
        f"skipped (each a failure): {', '.join(skipped) or 'none'}"
    )
    return 1 if bad else 0


CLASSES = ("ok", "diverged", "error", "sampler drift", "generator drift")


def part_class(part: str, res: dict, failed: int) -> str:
    """One compared part's class (``CLASSES``): diverged, else error (any other failure), else
    generator drift (the "gen" part's divergence), sampler drift, or ok."""
    gen = part.rsplit(".", 1)[-1] == "gen"
    if res["diverged"] and not gen:
        return "diverged"
    if failed:
        return "error"
    if res["diverged"]:
        return "generator drift"
    return "sampler drift" if res.get("sampler_drift") else "ok"


def generator_drift(diverged: str) -> str:
    """The "gen" part's divergence read as generator drift: its first seed and turn."""
    at = re.match(r"seed (\d+) turn (\d+)", diverged)
    where = f"seed {at[1]} turn {at[2]}" if at else diverged.split(":", 1)[0]
    return f"generator drift: first at {where} ({diverged})"


def report(args, part, res, took, allow: set[str], allow_raise: set[str], stderr: str) -> int:
    """Print one part's result; 1 if it failed: a divergence (but the "gen" part's: generator
    drift), a field only in the capture (a dropped field without an upgrade row), a field only
    in this checkout not in ``--allow-new``, or a function (not in ``--allow-all-raise``) that
    raised on every call. Sampler drift (new columns) is counted, never a failure."""
    gb = res["peak_bytes"] / 2**30
    over = took > LIMIT_S or res["peak_bytes"] > LIMIT_BYTES
    group = part.rsplit(".", 1)[-1]
    gen_drift = group == "gen" and bool(res["diverged"])
    if args.mode == "capture":
        verdict = "captured"
    elif gen_drift:
        verdict = generator_drift(res["diverged"])
    else:
        verdict = f"DIVERGED: {res['diverged']}" if res["diverged"] else "identical"
    print(
        f"{part}: {verdict}; {res['steps']} steps, {took:.1f} s, peak {gb:.2f} GB"
        f"{'  OVER LIMIT' if over else ''}",
        flush=True,
    )
    failed = bool(res["diverged"]) and not gen_drift
    if res.get("sampler_drift"):
        print(f"  sampler drift (new columns): {res['sampler_drift']} steps", flush=True)
    if res["only_base"]:
        print(
            f"  FAIL dropped field without an upgrade row: {_names(res['only_base'])}", flush=True
        )
        failed = True
    hard = [f for f in res["only_new"] if not allowed(f, allow)]
    ok = [f for f in res["only_new"] if allowed(f, allow)]
    if hard:
        print(f"  FAIL only in this checkout: {_names(hard)}", flush=True)
        failed = True
    if ok:
        print(f"  only in this checkout (--allow-new): {_names(ok)}", flush=True)
    if res["steps"] == 0:
        print("  FAIL zero steps", flush=True)
        failed = True
    if res["calls"] == 0 and group not in NO_CALLS_OK:
        print("  FAIL zero calls (nothing was run)", flush=True)
        failed = True
    n_raised = sum(r[0] for r in res["raised"].values())
    print(f"  raised: {n_raised}/{res['calls']} outputs", flush=True)
    counted = sum(r[0] for k, r in res["raised"].items() if k not in allow_raise)
    calls = sum(r[1] for k, r in res["raised"].items() if k not in allow_raise)
    calls += res["calls"] - sum(r[1] for r in res["raised"].values())
    rate = counted / calls if calls else 0.0
    if rate > args.raise_fail:
        print(f"  FAIL raise share {rate:.2f} > --raise-fail {args.raise_fail}", flush=True)
        failed = True
    elif rate > args.raise_warn:
        print(f"  WARN raise share {rate:.2f} > --raise-warn {args.raise_warn}", flush=True)
    for name, (count, calls, first) in res["raised"].items():
        every = calls > 0 and count == calls
        if every and name not in allow_raise:
            print(f"  FAIL raised on every call: {name} {count}/{calls} ({first})", flush=True)
            failed = True
        else:
            note = " (--allow-all-raise)" if every else ""
            print(f"  raised: {name} {count}/{calls}{note} ({first})", flush=True)
    if res.get("upgraded"):
        print(f"  layout upgrades: {_counts(res['upgraded'])}", flush=True)
    if args.verbose:
        print("  " + next((ln for ln in stderr.splitlines() if ln.startswith("TIMES")), ""))
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=("capture", "compare"))
    p.add_argument("dir")
    p.add_argument("seeds", help="A-B (inclusive)")
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--repo", type=Path, default=ROOT)
    p.add_argument("--commit", help="the --repo checkout's full commit (required without git)")
    p.add_argument("--no-roark", action="store_true")
    p.add_argument("--parts", help="only these groups/parts, e.g. step,rest,roark.trace")
    p.add_argument("--allow-new", help="NAME[,NAME...]: one-sided fields that do not fail")
    p.add_argument("--allow-all-raise", help="F[,F...]: functions allowed to raise on every call")
    p.add_argument("--subset-ok", action="store_true", help="compare a subset of the capture")
    p.add_argument("--same-ok", action="store_true", help="allow the capture's own commit")
    p.add_argument("--dirty-ok", action="store_true", help="allow a dirty checkout")
    p.add_argument("--raise-warn", type=float, default=RAISE_WARN, help="warn above this share")
    p.add_argument("--raise-fail", type=float, default=RAISE_FAIL, help="fail above this share")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        set_root(Path(argv[4]))
        try:
            import nuzline

            check_imported_from(nuzline.__file__, ROOT)
            return child(argv[1], Path(argv[2]), argv[3])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 2
    args = build_parser().parse_args(argv)
    set_root(args.repo)
    return parent(args)


if __name__ == "__main__":
    sys.exit(main())
