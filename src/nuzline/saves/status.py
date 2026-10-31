"""Decode a ``MON_CONDITION_*`` word into an engine status."""

from __future__ import annotations

from nuzline.engine.ids import Status

__all__ = ["decode_status"]

# MON_CONDITION_* (include/constants/battle/condition.h)
_SLEEP_MASK, _TOXIC_COUNTER_SHIFT = 0x7, 8
_STATUS_BITS = (
    (1 << 3, Status.POISON),
    (1 << 4, Status.BURN),
    (1 << 5, Status.FREEZE),
    (1 << 6, Status.PARALYSIS),
    (1 << 7, Status.TOXIC),
)


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
