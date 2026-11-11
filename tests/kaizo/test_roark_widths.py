"""Roark's action-pair widths against the user's pool (K3b2 Task 12; plan Review Focus 5).

For both Kaizo saves (``kaizo_save_pre_roark`` and the trace probe's ``kaizo_save_probe``),
every pool candidate at the level cap leads a team (the rest: the save's party, at the cap),
holding each item it could hold, against every Roark party member as the active Pokemon at
turn 0 (Roark's terrain and permanent sandstorm, ``prep.states.make_states``). Each legal
action pair (``legal_actions`` / ``legal_opponent_actions``) is enumerated exactly
(``enumerate_outcomes``), which raises ``SamplingOnly`` above ``WIDTH_CAP``. A pair over the
cap must be listed in ``SAMPLED_PAIRS`` with its reason; a listed pair that fits fails too,
so the list never goes stale.

What a candidate could bring (``prep.pool``): its current moves, its Move Relearner moves and
the bag's TMs, four to a moveset; held items: none, its own, each item a pool Pokemon holds,
and each held-pocket bag item with a hold effect. Over-approximate: an item a pool Pokemon
holds stays on that Pokemon too (two Quick Claws, say), which only adds relevance.

Pairs with identical ``outcome_laws`` have identical widths, so each distinct law row is
enumerated once. Skips without the saves or the Kaizo tables.
"""

from __future__ import annotations

import dataclasses
import hashlib

import jax
import numpy as np
import pytest

from nuzline.data.trainer import IntermediateData, build_trainer_party
from nuzline.engine.actions import (
    FIRST_ITEM_ACTION,
    MOVE_ACTIONS,
    legal_actions,
    legal_opponent_actions,
)
from nuzline.engine.coverage import start_weather
from nuzline.engine.enumerate import WIDTH_CAP, SamplingOnly, enumerate_outcomes
from nuzline.prep.pool import pool_candidates
from nuzline.prep.problem import HELD_POCKETS
from nuzline.prep.states import make_states
from nuzline.search.expand import laws_rows
from nuzline.search.rows import concat, take
from tests.conftest import KAIZO_INTERMEDIATE

# The Roark level cap (docs/kaizo/k3b-survey.md §1.1; plan Task 13 runs coverage at --cap 16).
CAP = 16

# (save, candidate label, held item, player move or "switch N", Roark active, Roark action)
# -> why the pair is sampled. Empty: every pair fits under WIDTH_CAP (Task 12 report).
SAMPLED_PAIRS: dict[tuple[str, str, str, str, str, str], str] = {}

SAVES = {"pre_roark": "kaizo_save_pre_roark_path", "probe": "kaizo_save_probe_path"}


def _movesets(cand) -> list[tuple[int, int, int, int]]:
    """The candidate's current moves, then its other learnable moves four at a time.

    The extra moves are tested in fixed groups of four (learnset order), not every set of
    four: every move is used, and a pair's width depends only on the move the row uses,
    never on its moveset neighbours. A mechanic reading the other moves (Encore, Sleep
    Talk, an AI read of the moveset) would need every combination here."""
    known = [m for m in cand.spec.moves if m]
    extra = [m for m in (*cand.relearner, *(m for _, m in cand.tms)) if m not in known]
    out = [tuple(cand.spec.moves)]
    for i in range(0, len(extra), 4):
        chunk = list(extra[i : i + 4])
        out.append(tuple(chunk + [0] * (4 - len(chunk))))
    return out


def _law_keys(laws) -> list[bytes]:
    """One key per row: the bytes of every ``OutcomeLaws`` leaf of that row."""
    leaves = jax.tree.leaves(jax.tree.map(np.asarray, laws))
    n = len(leaves[0])
    flat = np.concatenate(
        [np.ascontiguousarray(leaf).reshape(n, -1).view(np.uint8) for leaf in leaves], axis=1
    )
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in flat]


def roark_widths(save_path, tables, names, learnsets, roark):
    """``[(key, width or None)]``: every pair of the save (``key`` as ``SAMPLED_PAIRS``;
    ``None`` above the cap) and the number of distinct law rows enumerated."""
    from nuzline.data.games import KAIZO
    from nuzline.saves.save import load_save

    opp = list(build_trainer_party(roark, IntermediateData(KAIZO_INTERMEDIATE)))
    save = load_save(save_path, game=KAIZO, tables=tables, names=names)
    cands, left_out = pool_candidates(save, tables, names, learnsets, CAP)
    assert cands and not left_out, left_out
    party = [c for c in cands if c.location[0] == "party"]
    hold = np.asarray(tables.item_hold_effect)
    bag = {i for p in HELD_POCKETS for i, _ in save.bag.get(p, ()) if hold[i]}
    items = sorted({0, *bag, *(c.spec.item for c in cands if c.spec.item)})

    teams, meta = [], []  # meta: (label, item, moveset index, moveset)
    for cand in cands:
        bench = [c.spec for c in party if c is not cand][:5]
        for item in items:
            for mi, ms in enumerate(_movesets(cand)):
                pp = cand.spec.pp if mi == 0 else None
                lead = dataclasses.replace(cand.spec, item=item, moves=ms, pp=pp)
                teams.append([lead, *bench])
                meta.append((cand.label, item, mi, ms))
    parts, rows_meta = [], []
    for j in range(len(opp)):  # Roark's active in slot 0, the rest behind it in order
        order = [opp[j], *(m for k, m in enumerate(opp) if k != j)]
        parts.append(
            make_states(
                tables, teams, order, start_weather(roark), roark.items, terrain=roark.terrain
            )
        )
        rows_meta += [(*m, order) for m in meta]
    states = concat(parts)
    player = np.asarray(jax.vmap(lambda s: legal_actions(tables, s, 0))(states))
    foe = np.asarray(jax.vmap(lambda s: legal_opponent_actions(tables, s))(states))
    index, acts = [], []
    for i, (_, _, mi, _, _) in enumerate(rows_meta):
        for a in np.flatnonzero(player[i]):
            if mi > 0 and a >= MOVE_ACTIONS:
                continue  # a switch does not depend on the lead's moveset: once per item
            for b in np.flatnonzero(foe[i]):
                index.append(i)
                acts.append((a, b))
    index = np.asarray(index)
    acts = np.asarray(acts, np.int32)
    laws = laws_rows(tables, take(states, index), acts)
    keys = _law_keys(laws)
    laws = jax.tree.map(np.asarray, laws)
    widths: dict[bytes, int | None] = {}
    for r, key in enumerate(keys):
        if key in widths:
            continue
        try:
            widths[key] = len(enumerate_outcomes(jax.tree.map(lambda a, r=r: a[r], laws))[1])
        except SamplingOnly:
            widths[key] = None

    def action_name(moves, party_, a) -> str:
        if a >= FIRST_ITEM_ACTION:
            return f"bag slot {a - FIRST_ITEM_ACTION}"
        if a >= MOVE_ACTIONS:
            return f"switch {names.species_names[party_[a - MOVE_ACTIONS].species]}"
        return names.move_names[moves[a]]

    out = []
    for r, (i, (a, b)) in enumerate(zip(index, acts)):
        label, item, _, ms, order = rows_meta[i]
        team = teams[i % len(teams)]
        key = (
            label,
            names.item_names[item],
            action_name(ms, team, int(a)),
            names.species_names[order[0].species],
            action_name(order[0].moves, order, int(b)),
        )
        out.append((key, widths[keys[r]]))
    return out, len(widths)


@pytest.mark.parametrize("save", sorted(SAVES))
def test_roark_pairs_fit_the_width_cap(
    request, save, kaizo_tables, kaizo_names, kaizo_learnsets, kaizo_roark
):
    path = request.getfixturevalue(SAVES[save])
    pairs, distinct = roark_widths(path, kaizo_tables, kaizo_names, kaizo_learnsets, kaizo_roark)
    assert distinct > 0 and len(pairs) > 1000
    over = {(save, *key) for key, w in pairs if w is None}
    listed = {k for k in SAMPLED_PAIRS if k[0] == save}
    assert over <= listed, sorted(over - listed)[:20]
    assert listed <= over, f"listed as sampled but within the cap: {sorted(listed - over)}"
    widths = [w for _, w in pairs if w is not None]
    assert max(widths) <= WIDTH_CAP
