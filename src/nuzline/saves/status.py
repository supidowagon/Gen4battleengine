"""Decode a ``MON_CONDITION_*`` word into an engine status."""

from __future__ import annotations

from nuzline.engine.ids import Status

__all__ = ["decode_status", "status_conditions"]

# MON_CONDITION_* (include/constants/battle/condition.h)
_SLEEP_MASK, _TOXIC_COUNTER_SHIFT = 0x7, 8
_STATUS_BITS = (
    (1 << 3, Status.POISON),
    (1 << 4, Status.BURN),
    (1 << 5, Status.FREEZE),
    (1 << 6, Status.PARALYSIS),
    (1 << 7, Status.TOXIC),
)


def status_conditions(raw: int) -> list[str]:
    """The conditions a ``MON_CONDITION_*`` word holds (sleep turns, then each status bit),
    e.g. ``["sleep 3", "poison", "burn", "freeze", "paralysis"]`` for 0x7B. A battle tests
    each bit on its own (``status & MON_CONDITION_BURN``, battle_lib.c:4479; wake-up clears
    only the sleep bits, battle_controller_player.c:2449; thaw only the freeze bit,
    subscript_thaw_out.s), so a word with several keeps them all."""
    out = [f"sleep {raw & _SLEEP_MASK}"] if raw & _SLEEP_MASK else []
    return out + [s.name.lower() for bit, s in _STATUS_BITS if raw & bit]


def decode_status(raw: int) -> tuple[Status, int]:
    """``(ids.Status, status_counter)`` from a ``MON_CONDITION_*`` word: sleep turns in
    bits 0-2, the toxic counter in bits 8-11."""
    if raw & _SLEEP_MASK:
        return Status.SLEEP, raw & _SLEEP_MASK
    for bit, status in reversed(_STATUS_BITS):
        if raw & bit:
            counter = (raw >> _TOXIC_COUNTER_SHIFT) & 0xF if status == Status.TOXIC else 0
            return status, counter
    return Status.NONE, 0
