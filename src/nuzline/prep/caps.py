"""Per-trainer level caps (``config/level_caps.toml``, spec §7; plan 4c ruling 13)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from nuzline.data.trainer import Trainer

CAPS = Path(__file__).resolve().parents[3] / "config" / "level_caps.toml"


def load_level_caps(path: Path = CAPS) -> dict[int, tuple[str, int]]:
    """Trainer id -> ``(name, cap)``."""
    with open(path, "rb") as f:
        caps = tomllib.load(f).get("caps", {})
    return {int(k): (str(v["name"]), int(v["cap"])) for k, v in caps.items()}


def default_cap(trainer: Trainer, path: Path = CAPS) -> int:
    """The configured cap for ``trainer``. ``ValueError`` when the file has none for its id,
    or names a different trainer under that id."""
    entry = load_level_caps(path).get(trainer.id)
    if entry is None:
        raise ValueError(
            f"no level cap for {trainer.name} (id {trainer.id}) in config/level_caps.toml; "
            "pass --cap"
        )
    name, cap = entry
    if name.casefold() != trainer.name.casefold():
        raise ValueError(
            f"no level cap for {trainer.name} (id {trainer.id}) in config/level_caps.toml: "
            f"that id is listed as {name}; pass --cap"
        )
    return cap
