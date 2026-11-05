"""Builders for prep tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.learnsets import load_learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.engine.damage import damage_table
from nuzline.engine.outcome import hit_threshold
from nuzline.prep import search
from nuzline.prep.pool import Candidate
from nuzline.prep.problem import PrepProblem, Resources, levelled_moves
from nuzline.prep.screen import ScreenStats
from nuzline.prep.setups import Setup, setup_states, team_setups
from nuzline.search.rows import row_bytes
from tests.engine.helpers import roark_party, roark_trainer


def candidate(
    names: Names,
    spec: PokemonSpec,
    slot: int,
    relearner: Sequence[int] = (),
    tms: Sequence[tuple[int, int]] = (),
    candies: int = 0,
) -> Candidate:
    """A party candidate in ``slot``."""
    label = f"party slot {slot} {names.species_names[spec.species]}"
    return Candidate(("party", 0, slot), label, spec, candies, tuple(relearner), tuple(tms))


def problem(
    tables: Tables,
    names: Names,
    candidates: Sequence[Candidate],
    *,
    opp: Sequence[PokemonSpec] | None = None,
    resources: Resources | None = None,
    deaths: int = 0,
    protected: Sequence[int] = (),
    prestatus: Sequence[int] = (),
    cap: int = 16,
    levelled=None,
) -> PrepProblem:
    """A problem against Roark (his real party unless ``opp``), an empty bag unless
    ``resources``; ``levelled=None`` computes it as ``build_problem`` does."""
    if levelled is None:
        levelled = levelled_moves(tables, candidates, load_learnsets(), cap)
    return PrepProblem(
        candidates=tuple(candidates),
        trainer=roark_trainer(),
        opp=tuple(roark_party(names) if opp is None else opp),
        start_condition=0,
        resources=resources or Resources(0, (), (), ()),
        deaths=deaths,
        protected=frozenset(protected),
        prestatus=tuple(prestatus),
        cap=cap,
        source="test",
        levelled=tuple(levelled),
    )


# ---------------------------------------------------------------------------- PP stall
# Plan 4c ruling 15 and Task 11: a fight only a stalled-out opponent loses. Roark's Cranidos
# alone throws Rock Throw (the only damaging move, ``stall_pp`` PP left) and Leer (30 PP). The
# deathless lines let a tank that cannot be hurt eat every Rock Throw (Defense Curl), then
# bring in an attacker that Rock Throw would KO and that 2HKOs Cranidos.

STALL_LEER_PP = 30
STALL_LEVEL = 14  # Roark's Cranidos; the candidates' level and the cap
# (max HP, Atk, Def, SpA, SpD, Spe), the stored stats ``damage_table`` reads (spec.stats).
TANK_STATS = (400, 10, 300, 10, 300, 5)
STARLY_STATS = (20, 100, 10, 10, 10, 5)
SHINX_STATS = (20, 50, 10, 10, 10, 5)


def pp_stall_max_turns(stall_pp: int) -> int:
    """The playout horizon of a PP-stall fight: Cranidos's total PP, so it never reaches
    Struggle (ruling: ruling 15's "never Struggles" holds only within this horizon)."""
    return stall_pp + STALL_LEER_PP


def _stall_spec(names: Names, species: str, moves: Sequence[str], stats: tuple[int, ...]):
    base = roark_party(names)[0]
    ids = [names.move(m) for m in moves]
    return replace(
        base,
        species=names.species(species),
        level=STALL_LEVEL,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=0,
        gender=2,
        current_hp=None,
        status=0,
        pp=None,
        stats=stats,
    )


def pp_stall_problem(tables: Tables, names: Names, stall_pp: int) -> PrepProblem:
    """Cranidos (Rock Throw with ``stall_pp`` PP, Leer) alone against a tank (Geodude,
    Defense Curl), an attacker (Starly) and a decoy (Shinx); no bag, no resources, cap =
    their level. Asserts the premise from the damage tables (crits included)."""
    cranidos = roark_party(names)[2]
    rock_throw, leer = names.move("Rock Throw"), names.move("Leer")
    opp = replace(
        cranidos,
        moves=(rock_throw, leer, 0, 0),
        pp=(stall_pp, STALL_LEER_PP, 0, 0),
        current_hp=None,
        status=0,
    )
    specs = (
        _stall_spec(names, "Geodude", ("Defense Curl",), TANK_STATS),
        _stall_spec(names, "Starly", ("Wing Attack",), STARLY_STATS),
        _stall_spec(names, "Shinx", ("Spark",), SHINX_STATS),
    )
    cands = [candidate(names, spec, i) for i, spec in enumerate(specs)]
    prob = problem(
        tables,
        names,
        cands,
        opp=[opp],
        cap=STALL_LEVEL,
        levelled=[()] * len(cands),
    )
    prob = replace(prob, trainer=replace(roark_trainer(), items=()))
    _check_stall_premise(tables, names, prob, stall_pp)
    return prob


def _lone_state(tables: Tables, prob: PrepProblem, candidate_index: int):
    """The turn-0 state of the team of one candidate against the lone opponent."""
    setup = team_setups(prob, 1)[candidate_index]
    states = setup_states(tables, prob, [setup])
    return jax.tree.map(lambda f: jnp.asarray(f)[0], states)


def _check_stall_premise(tables: Tables, names: Names, prob: PrepProblem, stall_pp: int) -> None:
    """The fight's design, checked from the engine's own damage tables."""
    rock_throw = names.move("Rock Throw")
    max_hp = [int(_lone_state(tables, prob, i).max_hp[0, 0]) for i in range(3)]
    cran = _lone_state(tables, prob, 0)
    cran_hp = int(cran.max_hp[1, 0])

    def table(state, atk_side, move, crit):
        return np.asarray(damage_table(tables, state, atk_side, move, crit))

    # The tank at Defense -6 (worse than anything it can be left at) takes the worst hit
    # stall_pp times and survives.
    tank = cran.with_stages(0, def_=-6)
    worst = int(table(tank, 1, rock_throw, True).max())
    assert worst * stall_pp < max_hp[0], "the tank must outlast every Rock Throw"
    for i in (1, 2):
        state = _lone_state(tables, prob, i)
        assert int(table(state, 1, rock_throw, False).min()) >= max_hp[i], (
            "Rock Throw must KO each attacker"
        )
        best = [int(table(state, 0, m, False).min()) for m in prob.candidates[i].spec.moves if m]
        most = [int(table(state, 0, m, False).max()) for m in prob.candidates[i].spec.moves if m]
        assert max(best) * 2 >= cran_hp, "each attacker must 2HKO Cranidos"
        assert max(most) < cran_hp, "no attacker may OHKO Cranidos"
    state = cran
    threshold = int(np.asarray(hit_threshold(tables, state, [0, 0]))[1])
    assert threshold == 90, "Rock Throw hits at 90 %"


@contextmanager
def recorded_screens() -> Iterator[dict[Setup, ScreenStats]]:
    """Patch the funnel's ``setup_states`` and ``screen`` to record every setup's accumulated
    screen stats (the funnel keeps them private); the dict fills while the block runs."""
    of_row: dict[bytes, Setup] = {}
    total: dict[Setup, ScreenStats] = {}
    real_states, real_screen = search.setup_states, search.screen

    def states_of(tables, prob, setups):
        states = real_states(tables, prob, setups)
        for blob, setup in zip(row_bytes(states, len(setups)), setups, strict=True):
            of_row.setdefault(blob.tobytes(), setup)
        return states

    def screen_of(tables, states, n, key, max_turns=100):
        out = real_screen(tables, states, n, key, max_turns)
        for blob, stats in zip(row_bytes(states, len(out)), out, strict=True):
            setup = of_row[blob.tobytes()]
            total[setup] = total[setup].merge(stats) if setup in total else stats
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(search, "setup_states", states_of)
        patch.setattr(search, "screen", screen_of)
        yield total
