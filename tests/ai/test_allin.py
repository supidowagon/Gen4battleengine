"""``layout.ALLIN``: the sampled AI of every Kaizo fight in one program (``ai.layout`` module
docstring, "Two shapes of one transcription"). It must pick what the fight's own layout picks
for the same key, its masked flag scripts must consume no random read (E1 Ruling R4), and the
exact entry points stay per fight."""

from __future__ import annotations

import dataclasses
import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import policy
from nuzline.ai.commands import AiContext
from nuzline.ai.layout import (
    ALLIN,
    ALLIN_READS,
    KAIZO_FLAG_BITS,
    fight_slot_reads,
    fight_tables,
    layout_of,
    move_read_rows,
)
from nuzline.ai.outcome import sample_ai_outcome
from nuzline.ai.read_bounds import READ_BOUND_BITS
from nuzline.data.games import GAMES, KAIZO
from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
from nuzline.engine.state import make_state

MASK = sum(1 << bit for bit in KAIZO_FLAG_BITS)
KEYS = 24  # sampled decisions per state
# Kaizo Roark's flag set (trainer 246, ai_flags 23): its fights read on every path, under a
# read bound of their own below ALLIN_READS, so the two shapes' draws differ in shape.
ROARK_FLAGS = 0b10111


@pytest.fixture(scope="module")
def fights(kaizo_tables):
    """``{flags: (trainer, party)}``: for each flag set the Kaizo fights run (their AI flags
    masked to the transcribed bits, as coverage does), its first singles trainer in a split
    (by id), else its first trainer played as a singles fight (the flags are what is tested)."""
    d = GAMES[KAIZO]
    inter = IntermediateData(d.intermediate_dir)
    out: dict[int, tuple] = {}
    for t in sorted(load_trainers(d.intermediate_dir), key=lambda t: (t.format != "single", t.id)):
        flags = int(t.ai_flags) & MASK
        if t.split and flags not in out:
            out[flags] = (dataclasses.replace(t, format="single"), build_trainer_party(t, inter))
    return out


def _cover(flag_sets) -> list[int]:
    """A few flag sets in which every transcribed bit is both on and off, greedily."""
    want = {(bit, on) for bit in KAIZO_FLAG_BITS for on in (0, 1)}
    chosen: list[int] = []
    while want:
        best = max(
            sorted(flag_sets),
            key=lambda f: len({(b, f >> b & 1) for b in KAIZO_FLAG_BITS} & want),
        )
        chosen.append(best)
        want -= {(b, best >> b & 1) for b in KAIZO_FLAG_BITS}
    return chosen


def _fight(kaizo_tables, trainer, party, seed: int):
    """``(tables, states, keys)``: the fight's tables (``fight_tables``, flags masked) and a
    batch over its leads and ``KEYS`` keys: each opponent Pokemon leading against the next one
    as the player's, both actives at random HP."""
    t = fight_tables(kaizo_tables, trainer, party, int(trainer.ai_flags) & MASK)
    rng = np.random.default_rng(seed)
    states = []
    for j in range(len(party)):
        s = make_state(t, [party[(j + 1) % len(party)]], list(party), leads=(0, j))
        for side, slot in ((0, 0), (1, j)):
            s = s.replace_hp(side, slot, int(rng.integers(1, int(s.max_hp[side, slot]) + 1)))
        states.append(s)
    states = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *states)
    states = jax.tree.map(lambda x: jnp.repeat(x, KEYS, axis=0), states)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(party) * KEYS)
    return t, states, keys


# The AiContext fields compared (every field a flag script can write; not the inputs, nor the
# loose oracle's draws, None outside the oracle).
CONTEXT_FIELDS = tuple(
    f for f in AiContext._fields if f not in {"tables", "state", "outcome", "loose"}
)


def _program(layout):
    """A jitted program over a batch of rows, each with its own fight flags and moves
    (``Tables.ai_flags``, ``ai_moves``) on otherwise shared tables (``lax.map``, not
    ``vmap``: the unbatched program compiles in a fraction of the memory): per row, the
    sampled opponent action and the scoring context (``CONTEXT_FIELDS``), from one sampled
    outcome and one scoring (``policy._policy_and_context``, ``ai_policy``'s program)."""

    def one(t, s, k, flags, moves):
        t = t._replace(ai_flags=flags, ai_moves=moves)
        o = sample_ai_outcome(k, t, s, layout)
        action, ctx = policy._policy_and_context(t, s, o, layout)
        return action, {f: getattr(ctx, f) for f in CONTEXT_FIELDS}

    return jax.jit(lambda t, rows: jax.lax.map(lambda x: one(t, *x), rows))


def _rows(t, states, keys):
    n = len(keys)
    flags = jnp.full((n,), t.ai_flags, jnp.int32)
    moves = jnp.broadcast_to(jnp.asarray(t.ai_moves), (n, *np.shape(t.ai_moves)))
    return states, keys, flags, moves


def _runs(kaizo_tables, fights, flag_sets, seed0: int) -> dict:
    """``{flags: (own, allin)}``, each ``(actions, context)`` on the fight's batch
    (``_fight``). The fight's own layouts run fight by fight, each program dropped after its
    use. ALLIN runs once over every fight's rows together, each row with its own fight's
    flags and moves: one cross-fight batch, and one compiled program (asserted)."""
    batches = {f: _fight(kaizo_tables, *fights[f], seed0 + i) for i, f in enumerate(flag_sets)}
    own = {}
    for f, (t, states, keys) in batches.items():
        program = _program(layout_of(t))
        own[f] = jax.device_get(program(t, _rows(t, states, keys)))
        del program
        gc.collect()
    parts = [_rows(t, states, keys) for t, states, keys in batches.values()]
    rows = jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)
    program = _program(ALLIN)
    t0 = next(iter(batches.values()))[0]  # every fight's tables but ai_flags and ai_moves
    acts, ctx = jax.device_get(program(t0, rows))
    assert program._cache_size() == 1
    out, start = {}, 0
    for f, part in zip(batches, parts, strict=True):
        n = len(part[1])
        out[f] = (
            own[f],
            (acts[start : start + n], {k: v[start : start + n] for k, v in ctx.items()}),
        )
        start += n
    del program
    gc.collect()
    return out


def _fast_sets(fights) -> list[int]:
    """The fast tier's flag sets: the cover and Kaizo Roark's."""
    return sorted({*_cover(fights), ROARK_FLAGS})


@pytest.fixture(scope="module")
def cover_runs(kaizo_tables, fights):
    return _runs(kaizo_tables, fights, _fast_sets(fights), 0)


def _same_actions(fights, runs) -> None:
    accepted = 0
    for flags, ((want, _), (got, _)) in runs.items():
        want, got = np.asarray(want), np.asarray(got)
        bad = np.flatnonzero(got != want)
        name = fights[flags][0].name
        assert not bad.size, (name, f"{flags:#x}", bad[:5], want[bad[:5]], got[bad[:5]])
        accepted += int((want >= 0).sum())
    assert accepted > 0, "every decision was refused"


def test_the_cover_holds_every_bit_on_and_off(fights, kaizo_tables):
    cover = _cover(fights)
    for bit in KAIZO_FLAG_BITS:
        assert {f >> bit & 1 for f in cover} == {0, 1}, bit
    # A fast-tier fight whose own read bound is below ALLIN_READS: the draws of its own
    # layout are a prefix of ALLIN's (sample_ai_outcome), not the same shape.
    trainer, party = fights[ROARK_FLAGS]
    t = fight_tables(kaizo_tables, trainer, party, ROARK_FLAGS)
    assert 1 < layout_of(t).slot_reads < ALLIN_READS


def test_allin_picks_the_fights_own_action(fights, cover_runs):
    # A cover of the real flag sets (every transcribed bit on in one, off in another): the
    # all-in program, the fight's flags read at run time, picks what the fight's own layout
    # picks for every key.
    _same_actions(fights, cover_runs)


@pytest.mark.slow
def test_allin_picks_the_fights_own_action_for_every_flag_set(kaizo_tables, fights):
    _same_actions(fights, _runs(kaizo_tables, fights, sorted(fights), 100))


def test_a_masked_flag_script_consumes_no_read(kaizo_tables, fights, cover_runs):
    # The all-in scoring leaves exactly the fight's own scoring context: read counts, path
    # weights, scores, missing-read code, calcTemp and every other field a script writes. The masked scripts have reads to consume: some of the fight's moves
    # read under a flag it lacks (read_bounds.READ_BOUNDS, the most reads each flag's script
    # makes on a move's path).
    masked_reads = 0
    for flags, ((_, want), (_, got)) in cover_runs.items():
        trainer, party = fights[flags]
        for name in CONTEXT_FIELDS:
            assert np.array_equal(want[name], got[name]), (trainer.name, f"{flags:#x}", name)
        rows = move_read_rows(kaizo_tables, [m for spec in party for m in spec.moves])
        off = [col for col, bit in enumerate(READ_BOUND_BITS) if not flags >> bit & 1]
        masked_reads += int(rows[:, off].sum())
    assert masked_reads > 0, "no masked flag script reads on these fights' moves"


def test_draws_do_not_depend_on_the_read_bound(kaizo_tables, fights):
    trainer, party = fights[min(fights)]
    t, states, keys = _fight(kaizo_tables, trainer, party, seed=3)
    s = jax.tree.map(lambda x: x[0], states)
    lay = layout_of(t)
    own = np.asarray(sample_ai_outcome(keys[0], t, s, lay).script_rolls)
    allin = np.asarray(sample_ai_outcome(keys[0], t, s, ALLIN).script_rolls)
    assert allin.shape == (4, ALLIN_READS) and np.array_equal(allin[:, : lay.slot_reads], own)


def test_every_fight_reads_within_the_allin_bound(kaizo_tables):
    # ALLIN never refuses a path (missing_read) that a fight's own layout accepts.
    d = GAMES[KAIZO]
    inter = IntermediateData(d.intermediate_dir)
    for t in load_trainers(d.intermediate_dir):
        if not t.split:
            continue
        moves = [m for spec in build_trainer_party(t, inter) for m in spec.moves]
        rows = move_read_rows(kaizo_tables, moves)
        assert fight_slot_reads(rows, int(t.ai_flags) & MASK) <= ALLIN_READS, t.name
    assert ALLIN.key.endswith("-run") and ALLIN.flags == MASK


def test_the_exact_distribution_refuses_allin(kaizo_tables, fights):
    trainer, party = fights[min(fights)]
    t, states, _ = _fight(kaizo_tables, trainer, party, seed=4)
    s = jax.tree.map(lambda x: x[0], states)
    for exact in (
        policy.ai_action_distribution,
        policy.ai_action_distribution_loose,
        policy.ai_action_distribution_enumerated,
    ):
        with pytest.raises(ValueError, match="per fight"):
            exact(t, s, ALLIN)
    # The path refusals replay the exact distribution's 2**slot_reads scorings: per fight too.
    with pytest.raises(ValueError, match="per fight"):
        policy.ai_path_refusals(t, s, layout=ALLIN)


def test_allin_refuses_unset_flags(kaizo_tables, fights):
    # ALLIN reads the fight's flags at run time: tables whose flags were never set
    # (AI_FLAGS_UNSET, -1; ai.layout.fight_tables sets them) have no fight to run.
    trainer, party = fights[min(fights)]
    t, states, _ = _fight(kaizo_tables, trainer, party, seed=5)
    t = t._replace(ai_flags=np.int32(-1))
    s = jax.tree.map(lambda x: x[0], states)
    assert any("AI flags" in r for r in policy.ai_refusals(t, s, layout=ALLIN))
    # Traced, _policy gives REFUSED_ACTION on _Refused.any, which the policy refusal sets.
    assert bool(policy._refused(t, s, ALLIN).policy) and bool(policy._refused(t, s, ALLIN).any)
