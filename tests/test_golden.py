"""Golden emulator traces replayed through ``apply_turn`` (docs/tracing.md).

Each bundle skips until it has been recorded under ``tests/golden/``: that needs the
user's ROM and save. A turn is ``step`` followed by the replacement of a fainted active
Pokemon into the slot the trace shows (``apply_turn``): the game runs the replacement
before the next command selection, where the next turn is recorded, and ``step`` stops
before it (step.py phase 5). The player's replacement slot is the converter's
(``ReplayTurn.player_slot``); the opponent's is the next turn's active slot.
"""

import itertools
from pathlib import Path

import jax
import pytest

from tools.trace_to_state import COMPARED_FIELDS, ReplayTurn, apply_turn, trace_to_states

GOLDEN = Path(__file__).resolve().parent / "golden"

_apply = jax.jit(apply_turn)


def replay_mismatches(tables, turns: list[ReplayTurn]) -> list[str]:
    """``"turn N field F"`` for each ``COMPARED_FIELDS`` entry on which ``apply_turn`` of a
    turn differs from the next turn's state; empty when the trace replays."""
    wrong = []
    for before, after in itertools.pairwise(turns):
        assert before.learn_choices == ()
        out = _apply(
            tables,
            before.state,
            before.actions,
            before.outcome,
            before.player_slot,
            int(after.state.active[1]),
        )
        wrong += [
            f"turn {int(before.state.turn)} field {name}"
            for name in COMPARED_FIELDS
            if getattr(out, name).tolist() != getattr(after.state, name).tolist()
        ]
    return wrong


@pytest.mark.parametrize(
    ("bundle", "tables_fixture", "names_fixture"),
    [
        ("roark_01.json", "tables", "names"),
        ("kaizo_roark_02.json", "kaizo_tables", "kaizo_names"),
    ],
)
def test_golden_trace_replays(bundle, tables_fixture, names_fixture, request):
    path = GOLDEN / bundle
    if not path.exists():
        pytest.skip(f"no golden trace {bundle}")
    tables = request.getfixturevalue(tables_fixture)
    names = request.getfixturevalue(names_fixture)
    turns = trace_to_states(path, tables, names)
    assert len(turns) >= 3
    assert replay_mismatches(tables, turns) == []
