"""The line file (spec §6): reading, resolving, rendering and writing ``[[team]]`` and
``[[rule]]`` tables."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from nuzline.data.party import load_party
from nuzline.search.line_text import (
    LineFile,
    Rule,
    line_to_rules,
    read_line_file,
    render_rules,
    resolve_rules,
    rules_to_line,
    team_from_save,
    write_line_file,
)
from tests.engine.helpers import SAMPLE_PARTY, roark_party

# Sample party: Monferno, Prinplup, Grotle, Starly, Shinx, Budew.
# Roark: Geodude, Onix, Cranidos.


@pytest.fixture
def player(names):
    return load_party(SAMPLE_PARTY, names)


@pytest.fixture
def opp(names):
    return roark_party(names)


def _write(tmp_path, text: str):
    path = tmp_path / "line.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _resolve(tmp_path, names, player, opp, text: str) -> list[Rule]:
    return resolve_rules(read_line_file(_write(tmp_path, text)), names, player, opp)


def test_round_trip(tmp_path, names, player, opp):
    rules = [
        Rule(against=0, use=2, move=3),
        Rule(against=1, use=1, move=2, first=1, via=2, via_from=0),
        Rule(against=2, use=0, move=3, wait=True),
    ]
    line = rules_to_line(rules)
    assert np.asarray(line.use).tolist() == [2, 1, 0, -1, -1, -1]
    back = line_to_rules(line)
    assert back == rules
    team = ({"pokemon": 'Grotle "Leafy"', "moves": ["Tackle", "Absorb"], "item": ""},)
    path = tmp_path / "out.toml"
    write_line_file(path, LineFile(team=team, rules=()), render_rules(back, names, player, opp))
    file = read_line_file(path)
    assert file.team == team
    assert resolve_rules(file, names, player, opp) == rules


def test_header_is_a_comment_that_reads_back(tmp_path):
    team = ({"pokemon": "Prinplup", "item": "Quick Claw"},)
    rule = '[[rule]]\nagainst = "Bonsly"\nuse = "Prinplup"\nmove = "Aqua Cutter"'
    path = tmp_path / "found.toml"
    header = "python -m benchmarks.line_roark --n 4000; seed 0, deaths 1, commit abc\nsecond"
    write_line_file(path, LineFile(team=team, rules=()), [rule], header=header)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# python -m benchmarks.line_roark --n 4000; seed 0, deaths 1")
    assert "\n# second\n" in text
    file = read_line_file(path)
    assert file.team == team
    assert file.rules == ({"against": "Bonsly", "use": "Prinplup", "move": "Aqua Cutter"},)


def test_render_uses_slot_for_a_shared_species(names, player, opp):
    twins = [opp[0], opp[1], opp[0]]
    text = render_rules([Rule(against=2, use=0, move=0)], names, player, twins)
    assert 'against = "slot:2"' in text[0]
    assert 'use = "monferno"' in text[0].lower()  # the table's species spelling


def test_user_shape_parses(tmp_path, names, player, opp):
    rules = _resolve(
        tmp_path,
        names,
        player,
        opp,
        """
[[rule]]
against = "onix"
use = "Prinplup"
move = "Bubble"
first = "Growl"
via = "Grotle"
from = "slot:0"
wait = false

[[rule]]
against = "slot:2"
use = "monferno"
move = "Mach Punch"
""",
    )
    assert rules == [
        Rule(against=1, use=1, move=2, first=1, via=2, via_from=0, wait=False),
        Rule(against=2, use=0, move=3),
    ]


def _bad(tmp_path, names, player, opp, body: str, match: str):
    with pytest.raises(ValueError, match=match):
        _resolve(tmp_path, names, player, opp, body)


def test_unknown_foe_species(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Pikachu"\nuse = "Monferno"\nmove = "Ember"\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Pikachu\): .*Pikachu")


def test_foe_species_twice_needs_slot(tmp_path, names, player, opp):
    twins = [opp[0], opp[1], opp[0]]
    body = '[[rule]]\nagainst = "Geodude"\nuse = "Monferno"\nmove = "Ember"\n'
    _bad(tmp_path, names, player, twins, body, r"rule 1 \(against Geodude\): .*slot:N")


def test_unknown_team_member(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Onix"\nuse = "Piplup"\nmove = "Bubble"\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): .*Piplup")


def test_move_not_known(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Onix"\nuse = "Monferno"\nmove = "Bubble"\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): .*does not know")


def test_from_without_via(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\nfrom = "Grotle"\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): .*from.*via")


def test_wait_with_via(tmp_path, names, player, opp):
    body = (
        '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\n'
        'via = "Grotle"\nwait = true\n'
    )
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): .*wait.*via")


def test_two_rules_for_one_foe(tmp_path, names, player, opp):
    body = (
        '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\n\n'
        '[[rule]]\nagainst = "slot:1"\nuse = "Grotle"\nmove = "Absorb"\n'
    )
    _bad(tmp_path, names, player, opp, body, r"rule 2 \(against slot:1\): .*rule 1")


def test_via_equal_to_use(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\nvia = "slot:1"\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): .*via.*use")


def test_team_from_save(tmp_path, tables, names):
    from nuzline.saves import load_save
    from tests.saves.synthetic import write_save

    save = load_save(write_save(tmp_path, tables, names), tables=tables, names=names)
    file = LineFile(
        team=(
            {"pokemon": "party:0", "item": "Oran Berry"},
            {"pokemon": "Grotle", "moves": ["Razor Leaf", "Tackle"]},
        ),
        rules=(),
    )
    team = team_from_save(file, save, tables, names)
    assert team[0] == dataclasses.replace(
        save.party[0].to_spec(tables), item=names.item("Oran Berry")
    )
    assert team[1].moves == (names.move("Razor Leaf"), names.move("Tackle"), 0, 0)
    assert team[1].pp is None
    bad = LineFile(team=({"pokemon": "party:1", "moves": ["Not A Move"]},), rules=())
    with pytest.raises(ValueError, match=r"team 1 \(party:1\): .*Not A Move"):
        team_from_save(bad, save, tables, names)
    twice = LineFile(team=({"pokemon": "party:2"}, {"pokemon": "grotle"}), rules=())
    with pytest.raises(ValueError, match=r"team 2 \(grotle\): .*twice"):
        team_from_save(twice, save, tables, names)


def test_module_does_not_import_jax():
    code = "import nuzline.search.line_text, sys; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# --- review follow-ups ---------------------------------------------------------------------


def test_member_by_team_listing(tmp_path, names, player, opp):
    team = "".join(f'[[team]]\npokemon = "{p}"\n\n' for p in ("Fuego", "party:1", "Grotle"))
    rules = _resolve(
        tmp_path,
        names,
        player,
        opp,
        team + '[[rule]]\nagainst = "Onix"\nuse = "PARTY:1"\nmove = "Bubble"\nvia = "fuego"\n',
    )
    assert rules == [Rule(against=1, use=1, move=2, via=0)]


def test_ambiguous_team_member(tmp_path, names, player, opp):
    twins = [player[0], player[1], player[0]]
    body = '[[rule]]\nagainst = "Onix"\nuse = "Monferno"\nmove = "Ember"\n'
    _bad(tmp_path, names, twins, opp, body, r"rule 1 \(against Onix\): .*slots 0, 2.*slot:N")


def test_from_equal_to_via(tmp_path, names, player, opp):
    body = (
        '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\n'
        'via = "Grotle"\nfrom = "slot:2"\n'
    )
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): from is the same .*via")


def test_from_equal_to_use(tmp_path, names, player, opp):
    body = (
        '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\n'
        'via = "Grotle"\nfrom = "Prinplup"\n'
    )
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): from is the same .*use")


def test_wait_wrong_type(tmp_path, names, player, opp):
    body = '[[rule]]\nagainst = "Onix"\nuse = "Prinplup"\nmove = "Bubble"\nwait = 1\n'
    _bad(tmp_path, names, player, opp, body, r"rule 1 \(against Onix\): wait must be true")


def test_rules_to_line_slot_range():
    with pytest.raises(ValueError, match="opposing slot 6 outside 0-5"):
        rules_to_line([Rule(against=6, use=0, move=0)])
    with pytest.raises(ValueError, match="opposing slot -1"):
        rules_to_line([Rule(against=-1, use=0, move=0)])


def test_write_refuses_unknown_team_key(tmp_path):
    path = tmp_path / "out.toml"
    file = LineFile(team=({"pokemon": "Grotle", "level": 20},), rules=())
    with pytest.raises(ValueError, match=r"team 1 \(Grotle\): unknown key\(s\) level"):
        write_line_file(path, file, [])
    assert not path.exists()


def test_team_from_save_locations(tmp_path, tables, names):
    from nuzline.saves import load_save
    from tests.saves.synthetic import box_record, write_save

    oran = names.item("Oran Berry")
    boxes = {(1, 3): box_record(tables, names, "Bidoof", 4, ("Tackle",), item=oran)}
    path = write_save(tmp_path, tables, names, egg_first=True, boxes=boxes)
    save = load_save(path, tables=tables, names=names)
    team = team_from_save(
        LineFile(team=({"pokemon": "box:2:3", "item": ""}, {"pokemon": "party:3"}), rules=()),
        save,
        tables,
        names,
    )
    assert save.boxes[1][3].item == oran
    assert team[0].species == names.species("Bidoof") and team[0].item == 0
    assert team[1].species == names.species("Grotle")
    egg = LineFile(team=({"pokemon": "party:0"},), rules=())
    with pytest.raises(ValueError, match=r"team 1 \(party:0\): .*egg"):
        team_from_save(egg, save, tables, names)
    seven = LineFile(team=tuple({"pokemon": f"party:{i}"} for i in range(7)), rules=())
    with pytest.raises(ValueError, match="1-6 entries"):
        team_from_save(seven, save, tables, names)
