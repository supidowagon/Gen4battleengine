"""A party member as the battle engine needs it, before stats are computed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PokemonSpec:
    """Ids are the decomp's own (species, nature, move, item).

    ``ivs``/``evs`` are in ``ids.Stat`` order; ``moves`` holds four move ids with
    0 for an empty slot; ``ability_slot`` (0 or 1) indexes the species' two
    abilities; ``gender`` is 0 male, 1 female, 2 none. ``current_hp=None`` means
    full HP; ``status`` is an ``ids.Status`` and ``status_counter`` its turn counter
    (sleep turns left). ``pp`` (current PP per move slot) and ``stats`` (the stored
    in-game stats, ``ids.Stat`` order with max HP first) override the table PP and
    the ``calc_stats`` result when not ``None``.

    Experience (K3d-exp): ``exp`` is ``MON_DATA_EXPERIENCE`` (``None``: the base of
    ``level``). ``exp_boost`` is the exp multiplier in percent the battle applies to this
    Pokemon (``BattleScript_GetExpTask``, battle_script.c:9757-9763): 100 when the player
    is its original trainer, else 150, or 170 when its language differs from the ROM's;
    0 means unresolved (``SaveMon.to_spec`` without the trainer). A hand-typed party is
    the user's own, so 100 is the default. ``pokerus`` is 1 when the Pokerus byte is
    nonzero (``Pokemon_HasPokerus``, pokemon.c:4184-4199); ``friend_bonus`` is 1 for a
    Pokemon caught in a Luxury Ball (``Pokemon_UpdateFriendship``, pokemon.c:2664-2666).

    Coverage inputs (K3d-exp T7; ``engine.coverage.exp_refusals``): ``egg_location`` is
    ``MON_DATA_EGG_LOCATION`` (pokemon.c:1159-1166), whose match with the battle map adds 1
    to a level-up's friendship (pokemon.c:2668-2670); ``pp_ups`` are the PP Ups per move
    slot (``MON_DATA_MOVE1_PP_UPS``), which a move learned into that slot inherits
    (``BoxPokemon_SetMoveSlot``, pokemon.c:3925-3933). A hand-typed party has 0 for both.

    Data forms (E1 Batch A, G-41): ``ability_form`` is the form whose personal row
    ``ability_slot`` indexes and ``stats_form`` the form whose base stats ``calc_stats``
    reads; ``None`` means ``form``. A trainer Pokemon gets its ability from the base species
    (``BoxPokemon_InitWith`` reads ``SpeciesData_GetSpeciesValue(species, ...)`` before the
    form is set, pokemon.c:473-485; setting ``MON_DATA_FORM`` only stores it,
    pokemon.c:1608-1609), so ``build_trainer_party`` sets ``ability_form=0``; vanilla's
    builder never recomputes the stats after the form (src/trainer_data.c:216-218), so there
    ``stats_form=0`` too. Types always follow ``form`` (``MON_DATA_TYPE_1/2`` read the form
    row, pokemon.c:1219-1225).
    """

    species: int
    level: int
    nature: int
    ivs: tuple[int, int, int, int, int, int]
    evs: tuple[int, int, int, int, int, int]
    moves: tuple[int, int, int, int]
    item: int
    ability_slot: int
    gender: int
    friendship: int
    form: int = 0
    current_hp: int | None = None
    status: int = 0
    status_counter: int = 0
    pp: tuple[int, int, int, int] | None = None
    stats: tuple[int, ...] | None = None
    exp: int | None = None
    exp_boost: int = 100
    pokerus: int = 0
    friend_bonus: int = 0
    egg_location: int = 0
    pp_ups: tuple[int, int, int, int] = (0, 0, 0, 0)
    ability_form: int | None = None
    stats_form: int | None = None

    @property
    def ability_from(self) -> int:
        """The form whose personal row the ability slot indexes."""
        return self.form if self.ability_form is None else self.ability_form

    @property
    def stats_from(self) -> int:
        """The form whose personal row's base stats the stats are computed from."""
        return self.form if self.stats_form is None else self.stats_form
