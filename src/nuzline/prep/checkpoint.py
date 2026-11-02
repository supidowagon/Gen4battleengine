"""The prep checkpoint: a fingerprint, an atomic write and a refusing read (plan 4c Task 8;
ruling 12; spec section 6).

A checkpoint is one JSON document ``{"version", "fingerprint", "funnel"}``. ``funnel`` is the
search's own snapshot (``prep.search``); this module owns the file format, the fingerprint that
ties a checkpoint to one problem, one set of settings, one seed, one table version and one code
version, and the JSON forms of a ``Setup`` and ``ScreenStats``. ``read_checkpoint`` never writes:
a mismatch is a ``ValueError`` naming the first thing that differs, so ``--resume`` refuses
instead of continuing another run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path

from nuzline.prep.problem import PrepProblem
from nuzline.prep.screen import ScreenStats
from nuzline.prep.setups import Setup
from nuzline.prep.variants import Member, MoveChange

__all__ = [
    "CHECKPOINT_VERSION",
    "fingerprint",
    "read_checkpoint",
    "setup_from_json",
    "setup_to_json",
    "stats_from_json",
    "stats_to_json",
    "write_checkpoint",
]

CHECKPOINT_VERSION = 1

# The fingerprint keys in the order they are compared, with what a mismatch says.
_MISMATCH = (
    ("problem", "checkpoint is for another save or pool, or other fight settings (deaths, cap)"),
    ("settings", "checkpoint is for other settings"),
    ("seed", "checkpoint is for another seed"),
    ("tables", "checkpoint is for other tables"),
    ("code", "checkpoint is for another code version"),
)


def _plain(value):
    """``value`` as canonical-JSON data: dataclasses by field, sets sorted, tuples as lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def _digest(value) -> str:
    text = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(
    problem: PrepProblem, settings, seed: int, table_version: str, code_version: str
) -> dict[str, str]:
    """What a checkpoint is tied to. ``problem``: the sha256 of the canonical JSON of the
    candidates (specs, locations, relearner and TM options, candies), the ``levelled`` moves,
    the opponent specs, the trainer id, start condition, resources, deaths budget, protected
    set, prestatus, cap and source. ``settings``: the same for the settings without
    ``budget_seconds`` (a resumed run may have another budget)."""
    body = {
        "candidates": problem.candidates,
        "levelled": problem.levelled,
        "opp": problem.opp,
        "trainer": problem.trainer.id,
        "start_condition": problem.start_condition,
        "resources": problem.resources,
        "deaths": problem.deaths,
        "protected": problem.protected,
        "prestatus": problem.prestatus,
        "cap": problem.cap,
        "source": problem.source,
    }
    knobs = {k: v for k, v in _plain(settings).items() if k != "budget_seconds"}
    return {
        "problem": _digest(body),
        "settings": _digest(knobs),
        "seed": str(seed),
        "tables": table_version,
        "code": code_version,
    }


def write_checkpoint(path: Path, doc: dict) -> None:
    """Write ``doc`` to ``path`` atomically: a temp file in the same directory, flushed, then
    ``os.replace``. On any failure the temp file is removed and ``path`` keeps its old bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_checkpoint(path: Path, expected: dict[str, str]) -> dict:
    """The checkpoint at ``path`` if it is for ``expected`` (a :func:`fingerprint`). Otherwise
    ``ValueError``: unreadable JSON or a document of the wrong shape ("checkpoint FILE is
    damaged: ..."), a wrong ``version``, or the first differing fingerprint key. Never writes."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        version, found, funnel = doc["version"], doc["fingerprint"], doc["funnel"]
        if not isinstance(found, dict) or not isinstance(funnel, dict):
            raise TypeError("fingerprint and funnel must be objects")
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"checkpoint {path} is damaged: {e}") from e
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint {path} has version {version}, this program reads {CHECKPOINT_VERSION}"
        )
    for key, message in _MISMATCH:
        if found.get(key) != expected.get(key):
            raise ValueError(message)
    return doc


def setup_to_json(setup: Setup) -> list:
    """``setup`` as nested lists: per member ``[candidate, [[slot, move, source], ...], item,
    hp, status]``."""
    return [
        [m.candidate, [[c.slot, c.move, c.source] for c in m.moves], m.item, m.hp, m.status]
        for m in setup.members
    ]


def setup_from_json(data: list) -> Setup:
    """The inverse of :func:`setup_to_json`."""
    return Setup(
        tuple(
            Member(
                candidate=cand,
                moves=tuple(MoveChange(*c) for c in moves),
                item=item,
                hp=hp,
                status=status,
            )
            for cand, moves, item, hp, status in data
        )
    )


def stats_to_json(stats: ScreenStats) -> dict:
    return {
        "buckets": [int(b) for b in stats.buckets],
        "deaths": int(stats.deaths),
        "hp_left": float(stats.hp_left),
        "pp_left": float(stats.pp_left),
        "refused": bool(stats.refused),
    }


def stats_from_json(data: dict) -> ScreenStats:
    return ScreenStats(
        tuple(data["buckets"]),
        data["deaths"],
        data["hp_left"],
        data["pp_left"],
        data["refused"],
    )
