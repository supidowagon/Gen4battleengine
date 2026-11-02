"""Builders for prep tests."""

from __future__ import annotations

from collections.abc import Sequence

from nuzline.data.learnsets import load_learnsets
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.prep.pool import Candidate
from nuzline.prep.problem import PrepProblem, Resources, levelled_moves
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
