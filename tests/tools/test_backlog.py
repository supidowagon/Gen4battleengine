"""The backlog tool's counting helpers (tools/kaizo/backlog.py): battle grouping, battle
formats, group kinds and per-split new groups. Pure; no game data needed."""

from types import SimpleNamespace

import pytest

from tools.kaizo.backlog import (
    KINDS,
    battle_format,
    battle_split,
    battles,
    group_kind,
    kind_counts,
    new_by_split,
    partner_unknown,
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
