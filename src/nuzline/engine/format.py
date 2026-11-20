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
import numpy as np


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


def foes(x):
    """``x`` with its leading (battler) axis read through ``foe``: row ``b`` of the result is row
    ``foe(b)`` of ``x`` (singles: the reversed pair)."""
    return x[foe(np.arange(x.shape[0]))]


def party(fmt: Format, b):
    """The party index of battler ``b``: ``b`` when ``party_of`` is the identity, ``b & 1`` when
    there are two parties, else ``party_of[b]``: a Python int for a Python int ``b`` (the
    reference indexes lists with it), a gather for an array."""
    if fmt.party_of == tuple(range(fmt.B)):
        return b
    if fmt.P == 2:
        return b & 1
    if isinstance(b, int):
        return fmt.party_of[b]
    return jnp.asarray(fmt.party_of)[b]


def party_sides(fmt: Format) -> tuple[int, ...]:
    """The side of each party: the side of the battlers it sends to (``party_of``). SINGLES and
    DOUBLES ``(0, 1)``; TWO_TRAINERS ``(0, 1, 1)``; MULTI ``(0, 1, 0, 1)``. Pure Python on the
    static format."""
    return tuple(side_of(fmt.party_of.index(p)) for p in range(fmt.P))


def side_out_parties(fmt: Format, side: int) -> tuple[int, ...]:
    """The parties whose total emptiness ends the battle against ``side`` (CheckBattleOver,
    battle_controller_player.c:4181-4262, branches a / b / c): the player's party 0; the
    opponent's party 1 in singles, else the sorted distinct parties of battlers 1 and 3 (DOUBLES
    party 1; TWO_TRAINERS parties 1 and 2; MULTI parties 1 and 3). Pure Python on the static
    format."""
    if side == 0:
        return (0,)
    if fmt.B == 2:
        return (1,)
    return tuple(sorted({fmt.party_of[1], fmt.party_of[3]}))


def blackout_parties(fmt: Format, b: int) -> tuple[int, ...]:
    """The parties ``BtlCmd_CheckBlackOut`` (battle_script.c:8846-8915) sums for battler ``b``:
    its own party in singles and DOUBLES; TWO_TRAINERS player battlers party 0, enemy battlers
    parties 1 and 2; MULTI battler 0 party 0, battler 2 parties 2 and 0, enemy battlers parties
    1 and 3."""
    if fmt.B == 2 or fmt == DOUBLES:
        return (party(fmt, b),)
    if fmt == TWO_TRAINERS:
        return (0,) if b % 2 == 0 else (1, 2)
    if b % 2 == 1:
        return (1, 3)
    return (0,) if b == 0 else (2, 0)
