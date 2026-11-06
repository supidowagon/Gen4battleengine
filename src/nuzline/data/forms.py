"""Personal-row lookup for a species' form (host-only, no jax import).

Personal rows are the species ids 0-495, then the data forms in
``res/pokemon/form_registry.json`` order (496-507), as ``Pokemon_GetFormNarcIndex``
(``src/pokemon.c:4929``) maps them. The grid (``manifest["species_form_row"]``,
``Tables.species_form_row``, intermediate ``form_rows.json``) holds, per species and form,
the data row of a data form, the species itself for a form below the species'
``*_FORM_COUNT`` (``include/constants/forms.h``) without its own data, and -1 beyond it.
"""

from __future__ import annotations

FORM_SLOTS = 28  # UNOWN_FORM_COUNT, the widest form count (include/constants/forms.h)


def form_row(grid, species: int, form: int) -> int:
    """``grid[species][form]``; ``grid`` is the manifest's nested lists or an int array.
    Raises ``ValueError`` when the species has no such form."""
    species, form = int(species), int(form)
    if not 0 <= species < len(grid):
        raise ValueError(f"species {species} out of range")
    if not 0 <= form < FORM_SLOTS:
        raise ValueError(f"species {species} has no form {form}")
    row = int(grid[species][form])
    if row < 0:
        raise ValueError(f"species {species} has no form {form}")
    return row


_grids: dict[int, tuple[object, object]] = {}


def personal_row(tables, species: int, form: int = 0) -> int:
    """``form_row`` on ``tables.species_form_row`` (a jax or numpy ``Tables`` array), read
    through a host copy made once per array (jax scalar indexing is slow)."""
    import numpy as np

    grid = tables.species_form_row
    hit = _grids.get(id(grid))
    if hit is None or hit[0] is not grid:
        hit = (grid, np.asarray(grid))
        _grids[id(grid)] = hit
    return form_row(hit[1], species, form)
