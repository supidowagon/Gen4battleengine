"""Rebuild a NamedTuple of this repo from the keyword fields of a Hypothesis log. Plain Python: no
nuzline or jax import (tools/ops/sweep/replay.py gives it the engine's zero maker).

A log written by another commit can hold fields this repo does not have (dropped, with a note) and
lack fields this repo has (a coming batch adds BattleState.sketch_move). A missing field is
zero-filled by `zero(name)` when the caller gives one (as tools/snap_compare.py rebuild_state fills
a field an old saved state lacks: 0 = no move, no status, nothing recorded), else it is an error.
"""


def keep(cls, zero=None, note=print):
    """A maker `make(**fields)` of `cls`. `zero(name)` returns the zero value of a field, or raises
    KeyError for a field it cannot fill (then that field is an error, named)."""

    def make(**kw):
        extra = sorted(set(kw) - set(cls._fields))
        missing = sorted(set(cls._fields) - set(kw))
        if extra:
            note("  dropped fields not in this repo:", extra)
        if missing and zero is not None:
            filled = {}
            for name in missing:
                try:
                    filled[name] = zero(name)
                except KeyError:
                    pass
            if filled:
                note("  zero-filled fields missing from the log:", sorted(filled))
            kw = {**kw, **filled}
            missing = [n for n in missing if n not in filled]
        if missing:
            raise ValueError(f"{cls.__name__} fields missing from the log: {missing}")
        return cls(**{k: v for k, v in kw.items() if k in cls._fields})

    return make
