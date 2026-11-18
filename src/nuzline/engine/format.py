"""The battle format: how many battlers and parties a fight has (docs/doubles-state-design.md
section 4; ledger "D1: change plan", rule R1).

A ``Format`` is static: it is read off the shapes of a ``BattleState`` (``format_of``), which are
static under ``jit``, so no jitted function takes it as an argument. Battler ``b`` has side
``b & 1`` and partner ``b ^ 2``; ``party_of[b]`` is the party (the ``[P, 6]`` axis) it sends from.
Singles is the only format the engine runs today; the others are the shapes D2 onwards fill in.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Format:
    B: int  # battlers on the field
    P: int  # parties (the leading axis of every per-party array)
    party_of: tuple[int, ...]  # the party each battler sends from


SINGLES = Format(2, 2, (0, 1))
DOUBLES = Format(4, 2, (0, 1, 0, 1))
TWO_TRAINERS = Format(4, 3, (0, 1, 0, 2))
MULTI = Format(4, 4, (0, 1, 2, 3))
FORMATS = (SINGLES, DOUBLES, TWO_TRAINERS, MULTI)


def format_of(state) -> Format:
    """The ``Format`` whose ``B`` is ``state.active.shape[0]`` and ``P`` is
    ``state.species.shape[0]`` (ValueError when none has that pair)."""
    b, p = state.active.shape[0], state.species.shape[0]
    for fmt in FORMATS:
        if (fmt.B, fmt.P) == (b, p):
            return fmt
    raise ValueError(f"no battle format has {b} battlers and {p} parties")


def side_of(b):
    """The side (0 the player's, 1 the opponent's) of battler ``b``."""
    return b & 1


def partner(b):
    """The battler on the same side as ``b``."""
    return b ^ 2


def foe(b):
    """The default defender of battler ``b``: the battler across from it, until D2's resolver
    chooses the target as data."""
    return b ^ 1


def party(fmt: Format, b):
    """The party index of battler ``b``: ``b`` when ``party_of`` is the identity, ``b & 1`` when
    there are two parties, else a gather of ``party_of``."""
    if fmt.party_of == tuple(range(fmt.B)):
        return b
    if fmt.P == 2:
        return b & 1
    return jnp.asarray(fmt.party_of)[b]
