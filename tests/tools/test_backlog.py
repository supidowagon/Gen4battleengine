"""The backlog tool's counting helpers (tools/kaizo/backlog.py): battle grouping, battle
formats, group kinds and per-split new groups. Pure; no game data needed."""

from types import SimpleNamespace

import pytest

from tools.kaizo.backlog import (
    KINDS,
    ally_ids,
    ancestors,
    battle_format,
    battle_split,
    battles,
    evolution_parents,
    group_kind,
    kind_counts,
    levelup_moves,
    metronome_unmodelled,
    new_by_split,
    partner_unknown,
    reachable_species,
    runtime_section,
    sheet_cites,
)


def row(id, split="A", format="single", opponent_2=None, partner=()):
    return SimpleNamespace(
        id=id, split=split, format=format, opponent_2=opponent_2, partner=partner
    )


def index(*rows):
    return {r.id: r for r in rows}


def test_singles_are_one_battle_each():
    rs = [row(1), row(2), row(3, split="B")]
    assert battles(rs, index(*rs)) == [(1,), (2,), (3,)]


def test_two_trainer_pair_is_one_battle_either_direction():
    a = row(1, format="double_two_trainer", opponent_2=2)
    b = row(2, format="double_two_trainer", opponent_2=1)
    c = row(3, format="double_two_trainer", opponent_2=4)  # one-way link
    d = row(4, format="double_two_trainer", opponent_2=None)
    rs = [a, b, c, d]
    bs = battles(rs, index(*rs))
    assert bs == [(1, 2), (3, 4)]
    assert [battle_format(b, index(*rs)) for b in bs] == ["double_two_trainer"] * 2


def test_second_opponent_outside_the_splits_joins_its_battle():
    a = row(1, format="multi", opponent_2=9, partner=None)
    lone = row(9, split=None, format="multi", opponent_2=1, partner=None)
    by_id = index(a, lone)
    assert battles([a], by_id) == [(1, 9)]
    assert battle_split((1, 9), by_id) == "A"


def test_row_without_second_opponent_is_one_battle_whatever_its_format():
    a = row(1, format="multi", opponent_2=None, partner=[object()])
    assert battles([a], index(a)) == [(1,)]
    assert battle_format((1,), index(a)) == "multi"


def test_battle_spanning_two_splits_is_refused():
    a = row(1, split="A", format="double_two_trainer", opponent_2=2)
    b = row(2, split="B", format="double_two_trainer", opponent_2=1)
    with pytest.raises(ValueError, match="spans splits"):
        battles([a, b], index(a, b))


def test_battle_with_mixed_formats_is_refused():
    a = row(1, format="double_two_trainer", opponent_2=2)
    b = row(2, format="single", opponent_2=None)
    with pytest.raises(ValueError, match="formats"):
        battle_format((1, 2), index(a, b))


def test_partner_unknown_only_for_multi_with_no_partner_rows():
    m = row(1, format="multi", opponent_2=2, partner=None)
    n = row(2, format="multi", opponent_2=1, partner=None)
    k = row(3, format="multi", opponent_2=4, partner=None)
    p = row(4, format="multi", opponent_2=3, partner=[object()])
    t = row(5, format="double_two_trainer", opponent_2=None, partner=None)
    by_id = index(m, n, k, p, t)
    assert partner_unknown((1, 2), by_id)
    assert not partner_unknown((3, 4), by_id)
    assert not partner_unknown((5,), by_id)


@pytest.mark.parametrize(
    "group, kind",
    [
        (("AI", "AI_FLAG_PRIORITIZE_EXTREMES"), "AI flags"),
        (
            ("AI", "move effect #103 PRIORITY_1 not transcribed (no Basic or Expert label)"),
            "AI move effects",
        ),
        (("AI", "move-id case of TrainerAI_CalcDamage"), "AI move effects"),
        (("trainer AI", "x"), "AI move effects"),
        (("move effect", "#97"), "move effects"),
        (("level-up in battle move effect", "x"), "move effects"),
        (("held item effect", "#86"), "held items"),
        (("held item", "Odd Item"), "held items"),
        (("ability", "Technician"), "abilities"),
        (("battle format", "a double battle"), "battle conditions"),
        (("battle terrain", "x"), "battle conditions"),
        (("weather", "Rain"), "battle conditions"),
        (("field condition", "permanent Gravity"), "battle conditions"),
        (("start condition", "x"), "battle conditions"),
        (("ROM change", "x"), "other"),
        (("form", "x"), "other"),
        (("trainer item", "x"), "other"),
        (("setup", "x"), "other"),
    ],
)
def test_group_kind(group, kind):
    assert group_kind(group) == kind


def test_new_by_split_counts_each_group_in_its_first_split_only():
    needs = {"A": {1, 2}, "B": {2, 3}, "C": {1, 2, 3}, "D": {4}}
    new = new_by_split(["A", "B", "C", "D"], needs)
    assert new == {"A": {1, 2}, "B": {3}, "C": set(), "D": {4}}
    assert set().union(*new.values()) == set().union(*needs.values())


def test_kind_counts_cover_every_kind():
    c = kind_counts({("AI", "AI_FLAG_X"), ("AI", "move effect #1"), ("ability", "Truant")})
    assert list(c) == list(KINDS)
    assert c["AI flags"] == 1 and c["AI move effects"] == 1 and c["abilities"] == 1
    assert sum(c.values()) == 3


METHODS = ["none", "level_happiness", "level", "trade", "use_item"]
# 1 -> 2 at level 16 -> 3 at level 36; 4 -> 5 by friendship; 6 -> 7 by trade, 7 -> 8 at 20;
# 9 -> 10 at 30 or by item (either path).
EVOS = {
    "1": [[2, 16, 2]],
    "2": [[2, 36, 3]],
    "3": [],
    "4": [[1, 0, 5]],
    "6": [[3, 0, 7]],
    "7": [[2, 20, 8]],
    "9": [[2, 30, 10], [4, 99, 10]],
}


def test_evolution_parents():
    parents = evolution_parents(EVOS, METHODS)
    assert parents[2] == [(1, "level", 16)]
    assert parents[5] == [(4, "level_happiness", 0)]
    assert sorted(parents[10]) == [(9, "level", 30), (9, "use_item", 99)]
    assert 1 not in parents


def test_reachable_species_by_cap():
    parents = evolution_parents(EVOS, METHODS)
    all_ = list(range(1, 11))
    at15 = reachable_species(all_, parents, 15)
    assert 2 not in at15 and 3 not in at15
    assert at15[1] is False  # no pre-evolution: plain
    assert at15[5] is True  # friendship: assumed at any level
    assert at15[7] is True and 8 not in at15  # trade assumed; then a level step above cap
    assert at15[10] is True  # only the item path fits
    at36 = reachable_species(all_, parents, 36)
    assert at36[2] is False and at36[3] is False
    assert at36[8] is True  # trade step anywhere on the path keeps it assumed
    assert at36[10] is False  # the level path fits: preferred over the assumed one


def test_ancestors_and_levelup_moves():
    parents = evolution_parents(EVOS, METHODS)
    assert sorted(ancestors(3, parents)) == [1, 2, 3]
    assert ancestors(1, parents) == [1]
    assert levelup_moves([[1, 33], [16, 55], [17, 66], [0, 77]], 16) == {33, 55, 77}


def test_ally_ids_are_every_partner_variant():
    P = SimpleNamespace
    a = row(1, format="multi", opponent_2=2, partner=[P(trainer=608), P(trainer=609)])
    b = row(2, format="multi", opponent_2=1, partner=None)
    c = row(3, partner=())
    assert ally_ids([a, b, c]) == {608, 609}


def test_sheet_cites():
    t = SimpleNamespace(sources={"split": "sheet Galactic Split:1165, Galactic Split:1175"})
    assert sheet_cites(t) == ["Galactic Split:1165", "Galactic Split:1175"]
    assert sheet_cites(SimpleNamespace(sources={})) == []
    assert sheet_cites(SimpleNamespace(sources={"split": "overrides fields.5"})) == []


def test_metronome_unmodelled_counts_the_refused_calls(tables, names):
    """Ruling B5-2: of the moves Metronome can call (ids 1-467 but itself and the
    NO_METRONOME moves), the count whose own line coverage refuses. One call is made
    unmodelled (``unmodelled``) so the count is not 0 whatever the engine models."""
    import numpy as np

    from nuzline.engine.coverage import Support, move_effect_line
    from nuzline.engine.ids import MoveFlag
    from tests.engine.helpers import unmodelled

    metronome = names.move_names.index("Metronome")
    flags = np.asarray(tables.move_flags)
    calls = [m for m in range(1, 468) if m != metronome and not flags[m] & MoveFlag.NO_METRONOME]
    tables = unmodelled(tables, calls[0])
    refused = [m for m in calls if move_effect_line(tables, names, m, 1)[0] == Support.NONE]
    n, m = metronome_unmodelled(tables, names, 1)
    assert (n, m) == (len(refused), len(calls))
    assert 0 < n < m
    # Metronome is refused while any of them is (coverage.move_effect_line).
    assert move_effect_line(tables, names, metronome, 1)[0] == Support.NONE


def test_runtime_kind_is_a_battle_condition_and_lists_the_fight_as_playable():
    # SF2 (f): a trainer with a Pursuit user reports "field condition: Acid Rain (runtime
    # refusal)" in CoverageReport.runtime; the group kind is a battle condition, and the fight
    # is listed playable (it is not a missing group), with the exposure note.
    line = "field condition: Acid Rain (runtime refusal)"
    assert group_kind(("field condition", "Acid Rain")) == "battle conditions"
    text = "\n".join(runtime_section([("7 Pursuer", [line, line])]))
    assert "## Playable, lines hitting the glitch refused" in text
    assert "| 7 Pursuer | " + line + " |" in text
    assert "any fight is exposed when the player carries Pursuit" in text
    assert "1 battles." in text
