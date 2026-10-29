"""Golden emulator traces replayed through ``step`` (docs/tracing.md).

Skips until ``tests/golden/roark_01.json`` has been recorded: that needs the user's ROM
and save. One change from the plan's test: a turn whose opponent faints is followed by
its replacement into the slot the trace shows (``replace``). The game runs the
replacement before the next command selection, where the next turn is recorded, and
``step`` stops before it (step.py phase 5).
"""

import itertools
from pathlib import Path

import jax
import pytest

from nuzline.engine.step import needs_replacement, replace, step
from tools.trace_to_state import COMPARED_FIELDS, trace_to_states

TRACE = Path(__file__).resolve().parent / "golden" / "roark_01.json"
pytestmark = pytest.mark.skipif(not TRACE.exists(), reason="no golden trace")

_step = jax.jit(step)


def test_roark_trace_replays(tables, names):
    turns = trace_to_states(TRACE, tables, names)
    assert len(turns) >= 3
    for (before, actions, outcome), (after, _, _) in itertools.pairwise(turns):
        out = _step(tables, before, actions, outcome)
        if needs_replacement(out)[1]:
            out = replace(tables, out, 1, int(after.active[1]))
        for name in COMPARED_FIELDS:
            assert getattr(out, name).tolist() == getattr(after, name).tolist(), (
                f"turn {int(before.turn)} field {name}"
            )
