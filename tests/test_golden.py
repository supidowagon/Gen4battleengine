"""Golden emulator traces replayed through ``apply_turn`` (docs/tracing.md).

Each bundle skips until it has been recorded under ``tests/golden/``: that needs the
user's ROM and save. A turn is ``step`` followed by the replacement of a fainted active
Pokemon into the slot the trace shows (``apply_turn``): the game runs the replacement
before the next command selection, where the next turn is recorded, and ``step`` stops
before it (step.py phase 5). The player's replacement slot is the converter's
(``ReplayTurn.player_slot``); the opponent's is the next turn's active slot. The player's
answers to learn-move prompts are the converter's too (``ReplayTurn.learn_choices``).

``kaizo_roark_02.json`` is the K3d-exp acceptance trace (design §5.6): the user's Kaizo
Roark fight from the probe save, whose party is 1 exp short of level 17, with level-ups
mid-turn and from the bench, a declined move, and a player faint and replacement. It
converts with no mismatched field on any turn. Turn numbers below are the converter's
(record ``k + 1`` is the result of turn ``k``). Its replay also runs through the reference
engine (``reference_mismatches``), which must agree with ``apply_turn`` on every
``BattleState`` field of every turn.
"""

import itertools
from pathlib import Path

import jax
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.levelup import GIVE_UP
from nuzline.engine.outcome import outcome_to_py
from nuzline.engine.state import BattleState
from nuzline.engine.step import needs_replacement
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_advance_learns, ref_learn, ref_replace, ref_step
from tools.trace_to_state import (
    COMPARED_FIELDS,
    ReplayTurn,
    apply_turn,
    convert,
    learn_array,
    trace_to_states,
)

GOLDEN = Path(__file__).resolve().parent / "golden"

_apply = jax.jit(apply_turn)


def replay_mismatches(tables, turns: list[ReplayTurn]) -> list[str]:
    """``"turn N field F"`` for each ``COMPARED_FIELDS`` entry on which ``apply_turn`` of a
    turn differs from the next turn's state; empty when the trace replays."""
    wrong = []
    for before, after in itertools.pairwise(turns):
        out = _apply(
            tables,
            before.state,
            before.actions,
            before.outcome,
            before.player_slot,
            int(after.state.active[1]),
            learn_array(before.learn_choices),
        )
        wrong += [
            f"turn {int(before.state.turn)} field {name}"
            for name in COMPARED_FIELDS
            if getattr(out, name).tolist() != getattr(after.state, name).tolist()
        ]
    return wrong


def reference_mismatches(tables, turns: list[ReplayTurn]) -> list[str]:
    """``"turn N field F"`` for each ``BattleState`` field on which the reference engine's
    replay of a turn differs from ``apply_turn``'s: ``ref_step``, the learnset scans and
    the learn answers (``ref_advance_learns``, ``ref_learn``), then the replacements, the
    player's before the opponent's (``apply_turn``'s order). Empty when both agree."""
    np_tables = to_numpy(tables)
    wrong = []
    for before, after in itertools.pairwise(turns):
        opp_slot = int(after.state.active[1])
        out = _apply(
            tables,
            before.state,
            before.actions,
            before.outcome,
            before.player_slot,
            opp_slot,
            learn_array(before.learn_choices),
        )
        actions = tuple(int(a) for a in before.actions)
        ref = ref_step(np_tables, to_ref(before.state), actions, outcome_to_py(before.outcome))
        need = np.asarray(needs_replacement(from_ref(ref))) != 0
        ref = ref_advance_learns(np_tables, ref)
        for choice in before.learn_choices:
            ref = ref_learn(np_tables, ref, choice)
        if need[0] and before.player_slot >= 0:
            ref = ref_replace(np_tables, ref, 0, before.player_slot)
        if need[1] and opp_slot >= 0:
            ref = ref_replace(np_tables, ref, 1, opp_slot)
        ref_state = from_ref(ref)
        wrong += [
            f"turn {int(before.state.turn)} field {name}"
            for name in BattleState._fields
            if not np.array_equal(np.asarray(getattr(out, name)), getattr(ref_state, name))
        ]
    return wrong


@pytest.mark.parametrize(
    ("bundle", "tables_fixture", "names_fixture"),
    [
        ("roark_01.json", "tables", "names"),
        pytest.param("kaizo_roark_02.json", "kaizo_tables", "kaizo_names", marks=pytest.mark.slow),
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
    if bundle.startswith("kaizo"):  # slow: the reference engine runs in plain Python
        assert reference_mismatches(tables, turns) == []


@pytest.mark.slow
def test_kaizo_roark_02_acceptance(kaizo_tables, kaizo_names):
    path = GOLDEN / "kaizo_roark_02.json"
    if not path.exists():
        pytest.skip("no golden trace kaizo_roark_02.json")
    turns = convert(path, kaizo_tables, kaizo_names)
    assert len(turns) == 31
    assert [i for i, t in enumerate(turns) if t.mismatched] == []
    state = [t.state for t in turns]  # state[k + 1] is turn k's result

    def mon(k, slot):
        s = state[k]
        return int(s.level[0, slot]), int(s.hp[0, slot]), int(s.max_hp[0, slot])

    # Turn 1: Prinplup KOs Bonsly and is paid at MoveEnd; the sandstorm chip reads the new
    # max HP: 44 + 3 = 47, minus 49 // 16 = 3.
    assert mon(1, 0) == (16, 44, 46) and mon(2, 0) == (17, 44, 49)
    # Turn 4: Wormadam's Leech Life heals 38 -> 44, then the level-up adds 3.
    assert mon(5, 1) == (17, 47, 52)
    # Turn 8: the player's Delibird faints; Rhyhorn (slot 2) takes Stealth Rock 55 // 16.
    assert turns[8].player_slot == 2
    assert mon(8, 2)[1] == 52 and mon(9, 2)[1] == 49
    # Turn 12: Rhyhorn, a participant on the bench, levels: 23/55 -> 26/58.
    assert mon(12, 2) == (16, 23, 55) and mon(13, 2) == (17, 26, 58)
    # Turn 15: Sandshrew reaches 17 and gives up Slash (163); no learn work is left.
    assert [i for i, t in enumerate(turns) if t.learn_choices] == [15]
    assert turns[15].learn_choices == (GIVE_UP,)
    assert mon(16, 4)[0] == 17
    assert state[16].moves[0, 4].tolist() == state[15].moves[0, 4].tolist()
    assert not np.asarray(state[16].learn_level).any()
    # Turn 18: Water Pulse confuses Wormadam for 3 turns (the refit); turns 19-20 self-hits.
    assert int(turns[18].outcome.secondary[1, 0]) == 1 and int(turns[18].outcome.duration[1]) == 3
    assert [int(turns[k].outcome.confusion_hit[0]) for k in (19, 20)] == [1, 1]
    # Turn 27: Sand Tomb lasts 4 turns; on turn 29, 20 + 2 - 8 - 3 (the bind, 52 // 16) = 11.
    assert int(turns[27].outcome.duration[1]) == 4
    assert mon(30, 1)[1] == 11
    # The final record: every fainted opponent has status 0 (the converter's zeroing).
    fainted = np.asarray(state[30].hp[1]) == 0
    assert np.asarray(state[30].status[1])[fainted].tolist() == [0] * int(fainted.sum())
