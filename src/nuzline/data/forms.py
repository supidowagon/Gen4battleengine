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
