"""Per-trainer level caps (``config/level_caps.toml``, spec §7; plan 4c ruling 13).

Kaizo caps (plan K2b Task 16, spec §5.12) live in ``data/kaizo/level_caps.toml``."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path

from nuzline.data.trainer import Trainer

_ROOT = Path(__file__).resolve().parents[3]
CAPS = _ROOT / "config" / "level_caps.toml"
KAIZO = "platinum_kaizo_nightmare_final"
CAPS_FILES = {
    "platinum_us": CAPS,
    KAIZO: _ROOT / "data" / "kaizo" / "level_caps.toml",
}
_SHOWN = {"platinum_us": "config/level_caps.toml", KAIZO: "data/kaizo/level_caps.toml"}


def load_level_caps(path: Path = CAPS) -> dict[int, tuple[str, int]]:
    """Trainer id -> ``(name, cap)``."""
    with open(path, "rb") as f:
        caps = tomllib.load(f).get("caps", {})
    return {int(k): (str(v["name"]), int(v["cap"])) for k, v in caps.items()}


def load_kaizo_caps(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """``([splits], [fights])`` of a Kaizo caps file; its top-level ``source`` must be set."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not str(data.get("source", "")).strip():
        raise ValueError(f"{path}: no top-level `source` citing where the caps come from")
    splits = {str(k): int(v) for k, v in data.get("splits", {}).items()}
    fights = {str(k): int(v) for k, v in data.get("fights", {}).items()}
    return splits, fights


def _kaizo_cap(trainer: Trainer, path: Path, shown: str) -> int:
    splits, fights = load_kaizo_caps(path)
    label = trainer.label
    if label is not None:
        for key, cap in fights.items():
            if label == key or label.startswith(key + " "):
                return cap
    if trainer.split is not None:
        # the sheet names are "Roark Split"; the table is keyed by the leader ("Roark")
        for key in (trainer.split, trainer.split.removesuffix(" Split")):
            if key in splits:
                return splits[key]
    why = "has no split" if trainer.split is None else f"split {trainer.split!r} is not listed"
    raise ValueError(
        f"no level cap for {trainer.label or trainer.name} (id {trainer.id}) in {shown}: "
        f"it {why}; pass --cap"
    )


def default_cap(trainer: Trainer, path: Path | None = None, game: str = "platinum_us") -> int:
    """The configured cap for ``trainer``. Vanilla: ``ValueError`` when the file has none for
    its id, or names a different trainer under that id. Kaizo: the ``[fights]`` entry whose
    key starts the trainer's label, else its split's ``[splits]`` cap, else ``ValueError``.
    The sheets name splits "Roark Split"; a trailing " Split" is stripped to find the table key.
    ``path`` overrides the game's file in ``CAPS_FILES``."""
    if game not in CAPS_FILES:
        raise ValueError(f"no level caps file for game {game!r}")
    shown = _SHOWN[game]
    path = path or CAPS_FILES[game]
    if game == KAIZO:
        return _kaizo_cap(trainer, path, shown)
    entry = load_level_caps(path).get(trainer.id)
    if entry is None:
        raise ValueError(
            f"no level cap for {trainer.name} (id {trainer.id}) in {shown}; pass --cap"
        )
    name, cap = entry
    if name.casefold() != trainer.name.casefold():
        raise ValueError(
            f"no level cap for {trainer.name} (id {trainer.id}) in {shown}: "
            f"that id is listed as {name}; pass --cap"
        )
    return cap


def next_fight_cap(
    trainer: Trainer,
    trainers: Sequence[Trainer],
    game: str,
    player_starter: int | None,
    path: Path | None = None,
) -> int | None:
    """The cap of the first later non-optional fight in ``split_order`` that applies to a
    player with ``player_starter`` (its ``player_starter`` is None or equal); None after the
    last fight. ``ValueError`` when ``trainer`` has no ``split_order``."""
    if trainer.split_order is None:
        raise ValueError(f"{trainer.label or trainer.name} has no split order; pass --cap")
    later = sorted(
        (t for t in trainers if t.split_order is not None and t.split_order > trainer.split_order),
        key=lambda t: t.split_order,
    )
    for t in later:
        if t.optional or (t.player_starter is not None and t.player_starter != player_starter):
            continue
        return default_cap(t, path, game)
    return None
