"""Gauntlet transitions of the prep side (design v5.3 sections 3 and 5).

Built so far: the level-cap sit-out (ledger E3-B3-5.b). The other transitions of the option
registry (``data.gauntlet.OPTIONS``, each option's ``transition`` name) land with the groups
that need the Carry."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import NamedTuple

from nuzline.data.gauntlet import OPTIONS

__all__ = ["CapSitOut", "cap_sit_out"]


class CapSitOut(NamedTuple):
    """The level-cap sit-out at one gap. ``over``: the party slots newly over the next cap;
    ``deposited``: the slots that leave the party at this gap (``cap_sit_out_in_party =
    forbidden``); ``barred``: the slots that stay in the party and may not fight, the earlier
    sit-outs included (``allowed``); ``lost``: a ``forbidden`` sit-out that no deposit at this
    gap can take, so the playout row counts as lost."""

    over: tuple[int, ...]
    deposited: tuple[int, ...]
    barred: tuple[int, ...]
    lost: bool


def _checked(name: str, value: str) -> str:
    allowed = OPTIONS[name].values
    if value not in allowed:
        raise ValueError(f"{name} {value!r} is not one of {', '.join(allowed)}")
    return value


def cap_sit_out(
    levels: Sequence[int | None],
    next_cap: int | None,
    *,
    cap_overflow: str,
    cap_sit_out_in_party: str,
    deposit_room: int | None,
    sat_out: Collection[int] = (),
) -> CapSitOut:
    """The sit-out after a won fight, from the party's levels after it (``levels`` by party
    slot, ``None`` for an empty slot) and the next fight's level cap (``next_cap``, from
    ``prep.caps.next_fight_cap``, or ``default_cap`` of the next fight along the resolved
    order; ``None`` after the last fight).

    Under ``cap_overflow = sit_out_after_fight`` a member whose level exceeds ``next_cap`` takes
    no part in any later fight of the chain; under ``fight`` nothing is removed (design section
    3). ``cap_sit_out_in_party = forbidden``: every such member must be deposited at this gap, in
    the PC (a ``pc_gaps`` or walk-out gap) or by a swap; ``deposit_room`` is how many members
    the gap can take (``None``: any number, a PC or a walk-out; ``0``: none, a gap without a PC
    or a swap), and more over-cap members than that is ``lost``. ``allowed``: they stay in the
    party, barred. ``sat_out`` holds the slots that already sit out: they stay barred. These
    are user settings (house rules), not game rules; the game lets the member fight."""
    _checked("cap_overflow", cap_overflow)
    _checked("cap_sit_out_in_party", cap_sit_out_in_party)
    if deposit_room is not None and deposit_room < 0:
        raise ValueError(f"deposit_room {deposit_room} is negative")
    before = tuple(sorted(set(sat_out)))
    if cap_overflow == "fight" or next_cap is None:
        return CapSitOut((), (), before, False)
    over = tuple(
        slot
        for slot, level in enumerate(levels)
        if level is not None and level > next_cap and slot not in before
    )
    if cap_sit_out_in_party == "allowed":
        return CapSitOut(over, (), tuple(sorted({*before, *over})), False)
    if deposit_room is not None and len(over) > deposit_room:
        return CapSitOut(over, (), before, True)
    return CapSitOut(over, over, before, False)
