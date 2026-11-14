"""Kaizo Roark's AI against both saves' pools (K3c Task 8; docs/ai/kaizo_roark.md 14).

For both Kaizo saves (``kaizo_save_pre_roark`` and the trace probe's ``kaizo_save_probe``),
every pool candidate at the level cap (``prep.pool``, its current moves and held item) leads a
team (the rest: the save's party, at the cap) against every Roark party member as the active
Pokemon, at turn 0 (Roark's terrain and permanent sandstorm, ``prep.states.make_states``).
For each of these decisions:

- the AI refuses nothing (``policy.ai_refusals`` is empty);
- ``policy.ai_width``, the combinations the factored distribution merges (plan K3c Ruling
  12), is at most ``WIDTH_CAP``, and it is the pinned value of the active (``SITE_SUMS``)
  with the guess columns of the decision (E1 Ruling R1);
- every decision has at most ``LOOSE_JOINT_BITS`` (18) loose reads on the paths of the four
  slots, so that the loose oracle enumerates the slots jointly over at most 2^18
  combinations, and there the exact distribution and each slot's score distribution equal
  the loose ones (``tests.ai.test_kaizo_outcome._assert_exact_equals_loose``: every
  ``IfRandomLessThan`` and every ``LoadBattlerAbility DEFENDER`` its own draw, no site
  table, no guess columns). All ``DECISIONS`` decisions of a save are compared: a later read
  that pushes one past 18 fails here instead of silently shrinking the check.

The loose oracle builds its switch part from the exact path's own ``switch_sites``,
``switch_stop_probs`` and ``switch_action`` (``policy.ai_action_distribution_loose``), so
exact == loose here checks the scoring and the move pick, not the switch categorical. The
switch has its own independent oracle (``tests/ai/test_kaizo_switching.py``:
``_loose_switch`` and ``test_six_bench_layout_fits``; kaizo_roark.md 8.6, 11.1).

The flat products (``policy.ai_flat_width``, reported, not used) are recorded with the widths
in kaizo_roark.md 14. Skips without the saves or the Kaizo tables.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from nuzline.ai import policy
from nuzline.ai.commands import ability_guessed
from nuzline.ai.layout import KAIZO
from nuzline.data.trainer import IntermediateData, build_trainer_party
from nuzline.engine.coverage import start_weather
from nuzline.engine.enumerate import WIDTH_CAP
from nuzline.prep.pool import pool_candidates
from nuzline.prep.states import make_states
from tests.ai.test_kaizo_outcome import _assert_exact_equals_loose
from tests.conftest import KAIZO_INTERMEDIATE

CAP = 16  # the Roark level cap (docs/kaizo/k3b-survey.md 1.1)

SAVES = {"pre_roark": "kaizo_save_pre_roark_path", "probe": "kaizo_save_probe_path"}

# Decisions per save: 12 pool candidates (kaizo_roark.md 14) x Roark's 6 actives.
DECISIONS = 72

# sum_slots 2^(sites of the slot) per Roark active, the same against every candidate of both
# pools (measured 2026-11-11; kaizo_roark.md 14). ai_width is 45 switch stops plus
# sum_slots 2^(sites + guess columns): 2^3 times the sum when the AI guesses the
# candidate's ability (KAIZO.slot_guesses, E1 Ruling R1), the sum itself when not.
SITE_SUMS = {
    "Bonsly": 264,
    "Lileep": 26,
    "Gible": 8,
    "Corsola": 28,
    "Shuckle": 52,
    "Cranidos": 10,
}


def audit_rows(save_path, tables, names, learnsets, roark) -> list[dict]:
    """One row per (pool candidate, Roark active) of the save: the candidate's label, the
    active's name, the AI refusals, ``ai_width``, ``ai_flat_width``, the number of loose
    reads and the state."""
    from nuzline.data.games import KAIZO as KAIZO_GAME
    from nuzline.saves.save import load_save

    opp = list(build_trainer_party(roark, IntermediateData(KAIZO_INTERMEDIATE)))
    save = load_save(save_path, game=KAIZO_GAME, tables=tables, names=names)
    cands, left_out = pool_candidates(save, tables, names, learnsets, CAP)
    assert cands and not left_out, left_out
    party = [c for c in cands if c.location[0] == "party"]
    teams = [[c.spec, *(p.spec for p in party if p is not c)][:6] for c in cands]
    rows = []
    for j in range(len(opp)):  # Roark's active in slot 0, the rest behind it in order
        order = [opp[j], *(m for k, m in enumerate(opp) if k != j)]
        states = make_states(
            tables, teams, order, start_weather(roark), roark.items, terrain=roark.terrain
        )
        for i, cand in enumerate(cands):
            s = jax.tree.map(lambda a, i=i: a[i], states)
            batch = jax.tree.map(lambda a: np.asarray(a)[None], s)
            reads, _, _ = policy._loose_reads(tables, batch, KAIZO)
            rows.append(
                {
                    "candidate": cand.label,
                    "active": names.species_names[opp[j].species],
                    "refusals": policy.ai_refusals(tables, s, names),
                    "width": policy.ai_width(tables, s),
                    "guessed": bool(ability_guessed(tables, s)),
                    "flat": policy.ai_flat_width(tables, s),
                    "reads": sum(len(r) for r in reads),
                    "state": s,
                }
            )
    return rows


@pytest.mark.slow
@pytest.mark.parametrize("save", sorted(SAVES))
def test_roark_ai_against_the_pool(
    request, save, kaizo_tables, kaizo_names, kaizo_learnsets, kaizo_roark
):
    t = kaizo_tables
    path = request.getfixturevalue(SAVES[save])
    rows = audit_rows(path, t, kaizo_names, kaizo_learnsets, kaizo_roark)
    assert len(rows) == len({r["candidate"] for r in rows}) * len(SITE_SUMS)
    refused = [(r["candidate"], r["active"], r["refusals"]) for r in rows if r["refusals"]]
    assert not refused, refused[:5]
    assert max(r["width"] for r in rows) <= WIDTH_CAP
    stops = KAIZO.switch_sites + 1
    for r in rows:
        columns = 2**KAIZO.slot_guesses if r["guessed"] else 1
        assert r["width"] == stops + columns * SITE_SUMS[r["active"]], (r["candidate"], r["active"])
    assert len(rows) == DECISIONS, len(rows)
    joint = [r for r in rows if r["reads"] <= policy.LOOSE_JOINT_BITS]
    too_big = [
        (r["candidate"], r["active"], r["reads"])
        for r in rows
        if r["reads"] > policy.LOOSE_JOINT_BITS
    ]
    assert len(joint) == DECISIONS, too_big[:5]
    for r in joint:
        _assert_exact_equals_loose(t, r["state"], KAIZO)
