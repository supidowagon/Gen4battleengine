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


def _program(layout):
    """A jitted program over a fight's batch (``lax.map``, not ``vmap``: the unbatched
    program compiles in a fraction of the memory): per (state, key), the sampled opponent
    action and the scoring context fields a flag script writes (scores, read counts, path
    weights, missing-read code), from the same sampled outcome."""

    def one(t, s, k):
        # ai_policy's own program (_policy), with the context it scored (one scoring).
        o = sample_ai_outcome(k, t, s, layout)
        action, ctx = policy._policy_and_context(t, s, o, layout)
        return action, (ctx.scores, ctx.reads, ctx.weight, ctx.missing_read)

    return jax.jit(lambda t, states, keys: jax.lax.map(lambda x: one(t, *x), (states, keys)))


def _runs(kaizo_tables, fights, flag_sets, seed0: int) -> dict:
    """``{flags: (own, allin)}``, each ``(actions, context fields)`` on the fight's batch
    (``_fight``). ALLIN's one program serves every fight (its tables are traced); each program
    is dropped after its use, so one compiled program is resident at a time."""
    batches = {f: _fight(kaizo_tables, *fights[f], seed0 + i) for i, f in enumerate(flag_sets)}

    def run(layout_of_fight):
        out, programs = {}, {}
        for f, (t, states, keys) in batches.items():
            lay = layout_of_fight(t)
            if lay.key not in programs:
                programs = {lay.key: _program(lay)}  # drops the last one
                gc.collect()
            out[f] = jax.device_get(programs[lay.key](t, states, keys))
        return out

    allin = run(lambda t: ALLIN)
    own = run(layout_of)
    return {f: (own[f], allin[f]) for f in flag_sets}


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
    # The all-in scoring leaves exactly the fight's own read counts, path weights, scores and
    # missing-read code. The masked scripts have reads to consume: some of the fight's moves
    # read under a flag it lacks (read_bounds.READ_BOUNDS, the most reads each flag's script
    # makes on a move's path).
    masked_reads = 0
    for flags, ((_, want), (_, got)) in cover_runs.items():
        trainer, party = fights[flags]
        names = ("scores", "reads", "weight", "missing_read")
        for name, a, b in zip(names, want, got, strict=True):
            assert np.array_equal(a, b), (trainer.name, f"{flags:#x}", name)
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


@pytest.mark.parametrize("flags", [-1, 0x40 | 0x7])
def test_allin_refuses_unset_or_untranscribed_flags(kaizo_tables, fights, flags):
    trainer, party = fights[min(fights)]
    t, states, _ = _fight(kaizo_tables, trainer, party, seed=5)
    t = t._replace(ai_flags=np.int32(flags))
    s = jax.tree.map(lambda x: x[0], states)
    assert any("AI flags" in r for r in policy.ai_refusals(t, s, layout=ALLIN))
    # Traced, _policy gives REFUSED_ACTION on _Refused.any, which the policy refusal sets.
    assert bool(policy._refused(t, s, ALLIN).policy) and bool(policy._refused(t, s, ALLIN).any)
