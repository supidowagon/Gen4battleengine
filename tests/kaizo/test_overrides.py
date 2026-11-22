"""Committed Kaizo overrides: schema, citation checks and the reviewed draft (Task 10)."""

import json
import tomllib

import pytest

from nuzline.data.gauntlet import (
    GauntletFacts,
    GauntletSettingsError,
    Partner,
    check_swap_gap,
)
from tools.kaizo.overrides import (
    DEFAULT_PATH,
    AreaException,
    AreaOverride,
    Gauntlet,
    Mappings,
    Overrides,
    OverridesError,
    build_mappings,
    check_citations,
    check_gauntlet_members,
    check_rom_trick_room,
    check_trainer_ids,
    check_two_placement,
    draft_overrides,
    load_overrides,
)
from tools.kaizo.trainer_map import Mapping, MappingError, map_trainers


def write(tmp_path, text):
    p = tmp_path / "overrides.toml"
    p.write_text(text, encoding="utf-8")
    return p


def overrides_with(**kw):
    return Overrides(**kw)


# --- loader ---------------------------------------------------------------------------


def test_entry_without_source_refused(tmp_path):
    with pytest.raises(OverridesError, match="area.246.*source"):
        load_overrides(write(tmp_path, "[area.246]\ntrick_room = true\n"))


def test_rules_key_refused(tmp_path):
    with pytest.raises(OverridesError, match="'rules'"):
        load_overrides(
            write(
                tmp_path,
                '[[gauntlet]]\nname = "g"\ntrainers = [1, 2]\nrules = {heal = true}\n'
                'source = "user, 2026-11-03"\n',
            )
        )


@pytest.mark.parametrize(
    "source", ["somewhere", "pkcalc:sets.js", "user 2026-11-03", "Roark Split:x", "rom:"]
)
def test_malformed_source_refused(tmp_path, source):
    with pytest.raises(OverridesError, match="area.7: source"):
        load_overrides(write(tmp_path, f'[area.7]\ntrick_room = true\nsource = "{source}"\n'))


def test_unknown_key_refused(tmp_path):
    with pytest.raises(OverridesError, match="area.7.*'trickroom'"):
        load_overrides(write(tmp_path, '[area.7]\ntrickroom = true\nsource = "rom:x"\n'))


def test_unknown_table_refused(tmp_path):
    with pytest.raises(OverridesError, match="'level_caps'"):
        load_overrides(write(tmp_path, "[level_caps]\nGalactic = 100\n"))


def test_area_and_exception_on_one_id_refused(tmp_path):
    text = (
        '[area.7]\ntrick_room = true\nsource = "rom:x"\n'
        '[area_exception.7]\nreason = "r"\nsource = "rom:y"\n'
    )
    with pytest.raises(OverridesError, match="7"):
        load_overrides(write(tmp_path, text))


def test_loads_every_table(tmp_path):
    text = """
[area.403]
trick_room = true
source = "rom:map header 308 weather 18 (Trick Room)"

[area.870]
gravity = true
source = "pkcalc:src/js/shared_controls.js:611"

[area_exception.509]
reason = "two maps"
source = "rom:trainers.json start_condition unknown"

[[gauntlet]]
name = "Oreburgh Mine"
trainers = [196, 195]
source = "Roark Split:229"

[trainer_ids."Gardenia Split:91"]
id = 296
source = "pkcalc:src/js/data/sets.js:988"

[fields.107]
optional = true
source = "Wake Split:195"

[move_lists.punch]
ids = [5, 7]
source = "user, 2026-11-03"
"""
    o = load_overrides(write(tmp_path, text))
    assert o.area == {
        403: AreaOverride(True, False, None, "rom:map header 308 weather 18 (Trick Room)"),
        870: AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611"),
    }
    assert o.area_exceptions == {
        509: AreaException("two maps", "rom:trainers.json start_condition unknown")
    }
    assert o.gauntlets == (Gauntlet("Oreburgh Mine", (196, 195), "Roark Split:229"),)
    assert o.trainer_ids == {"Gardenia Split:91": 296}
    assert o.fields == {107: {"optional": True}}
    assert o.move_lists == {"punch": [5, 7]}
    assert o.sources["trainer_ids.Gardenia Split:91"] == "pkcalc:src/js/data/sets.js:988"


FACTS = (
    "orders = [[[1], [2]]]\nleave_gaps = [1]\nescape = false\npc_gaps = []\nheal_gaps = []\n"
    'locked_gaps = []\norders_source = "rom:header 1 events 2"\n'
)


def gauntlet_text(trainers="[1, 2]", facts=FACTS):
    return f'[[gauntlet]]\nname = "g"\ntrainers = {trainers}\nsource = "user, 2026-11-03"\n{facts}'


def test_gauntlet_facts_load(tmp_path):
    f = load_overrides(write(tmp_path, gauntlet_text())).gauntlets[0].facts
    assert f.name == "g" and f.source == "user, 2026-11-03"
    assert f.orders == (((1,), (2,)),) and f.leave_gaps == ((1,),) and f.escape is False
    assert f.pc_gaps == f.heal_gaps == f.locked_gaps == () and f.partner is None
    assert f.orders_source == f.leave_gaps_source == f.locked_gaps_source == "rom:header 1 events 2"


def test_gauntlet_without_facts_has_none(tmp_path):
    text = '[[gauntlet]]\nname = "g"\ntrainers = [1, 2]\nsource = "user, 2026-11-03"\n'
    assert load_overrides(write(tmp_path, text)).gauntlets[0].facts is None


def test_gauntlet_permutations_and_all(tmp_path):
    facts = (
        'orders = { permutations = [[1], [2], [3]] }\nleave_gaps = "all"\nescape = false\n'
        'pc_gaps = []\nheal_gaps = "all"\nlocked_gaps = []\norders_source = "rom:x"\n'
        'heal_gaps_source = "rom:y z"\n'
    )
    f = load_overrides(write(tmp_path, gauntlet_text("[1, 2, 3]", facts))).gauntlets[0].facts
    assert len(f.orders) == 6 and ((1,), (2,), (3,)) in f.orders
    assert f.leave_gaps == ((1, 2),) * 6 and f.heal_gaps == (1, 2)
    assert f.heal_gaps_source == "rom:y z" and f.pc_gaps_source == "rom:x"


def test_gauntlet_leave_gaps_per_order_and_partner(tmp_path):
    facts = (
        "orders = [[[1], [2]], [[2], [1]]]\nleave_gaps = [[], [1]]\nescape = true\npc_gaps = []\n"
        "heal_gaps = []\nlocked_gaps = []\n"
        'partner = { trainer = 9, join_before = [2], source = "rom:header 1 events 3" }\n'
        'orders_source = "rom:header 1 events 2"\n'
    )
    f = load_overrides(write(tmp_path, gauntlet_text(facts=facts))).gauntlets[0].facts
    assert f.leave_gaps == ((), (1,)) and f.leave_gaps_of(((2,), (1,))) == (1,)
    assert f.partner == Partner(9, (2,), "rom:header 1 events 3")


@pytest.mark.parametrize(
    "old, new, match",
    [
        ("locked_gaps = []\n", "", "all or none"),
        ("orders = [[[1], [2]]]", "orders = [[[1], [3]]]", "exactly once"),
        ("orders = [[[1], [2]]]", "orders = [[[1, 2, 3]]]", "one or two"),
        ("orders = [[[1], [2]]]", "orders = [[[1], [2]], [[1], [2]]]", "twice"),
        ("leave_gaps = [1]", "leave_gaps = [2]", "within 1..1"),
        ("leave_gaps = [1]", "leave_gaps = [[1], [1]]", "one list per order"),
        ("pc_gaps = []", "pc_gaps = [1, 1]", "ascending"),
        ("escape = false", "escape = 0", "true or false"),
        (
            "locked_gaps = []",
            "locked_gaps = []\npartner = { trainer = 5, join_before = [3] }",
            "1..2",
        ),
        (
            "locked_gaps = []",
            "locked_gaps = []\npartner = { trainer = 5, join_before = [1, 2] }",
            "one fight",
        ),
        ('"rom:header 1 events 2"', '"somewhere"', "orders_source"),
    ],
)
def test_gauntlet_facts_refused(tmp_path, old, new, match):
    with pytest.raises(OverridesError, match=match):
        load_overrides(write(tmp_path, gauntlet_text(facts=FACTS.replace(old, new))))


def test_gauntlet_fight_headers_one_header_or_one_list_per_order(tmp_path):
    f = load_overrides(write(tmp_path, gauntlet_text(facts=FACTS))).gauntlets[0].facts
    assert f.fight_headers == ()  # not recorded
    one = FACTS + "fight_headers = 284\n"
    f = load_overrides(write(tmp_path, gauntlet_text(facts=one))).gauntlets[0].facts
    assert f.fight_headers == ((284, 284),)
    per = FACTS + "fight_headers = [[284, 380]]\n"
    f = load_overrides(write(tmp_path, gauntlet_text(facts=per))).gauntlets[0].facts
    assert f.fight_headers == ((284, 380),) and f.to_dict()["fight_headers"] == [[284, 380]]
    for bad in ("[[284]]", "[284, 380]", '"284"'):
        text = gauntlet_text(facts=FACTS + f"fight_headers = {bad}\n")
        with pytest.raises(OverridesError, match="fight_headers"):
            load_overrides(write(tmp_path, text))


# Route 204 North's shape: 1 is a double-battle trainer who stops the walk at two live and is
# not fought at one; the two-live order is 1, 2, 3.
LIVE_ROWS = [
    {"fought": [], "live": 1, "next": [[2]], "headers": [7]},
    {"fought": [], "live": 2, "next": [[1]], "headers": [7]},
    {"fought": [[1]], "live": 1, "next": [[2]], "headers": [7], "gap": ["leave"]},
    {"fought": [[1]], "live": 1, "to": 2, "at": "bag", "next": [[2]], "headers": [7]},
    {"fought": [[1]], "live": 2, "next": [[2]], "headers": [7], "gap": ["leave"]},
    {"fought": [[2]], "live": 1, "next": [[3]], "headers": [7], "gap": []},
    {"fought": [[2]], "live": 1, "to": 2, "at": "bag", "next": [[1], [3]], "headers": [7, 7]},
    {"fought": [[2]], "live": 2, "next": [[1], [3]], "headers": [7, 7], "gap": []},
    {"fought": [[1], [2]], "live": 1, "next": [[3]], "headers": [7], "gap": []},
    {"fought": [[1], [2]], "live": 1, "to": 2, "at": "bag", "next": [[3]], "headers": [7]},
    {"fought": [[1], [2]], "live": 2, "next": [[3]], "headers": [7], "gap": []},
    {"fought": [[2], [1]], "live": 1, "next": [[3]], "headers": [7], "gap": []},
    {"fought": [[2], [1]], "live": 1, "to": 2, "at": "bag", "next": [[3]], "headers": [7]},
    {"fought": [[2], [1]], "live": 2, "next": [[3]], "headers": [7], "gap": []},
    {"fought": [[2], [3]], "live": 1, "next": [], "headers": [], "gap": []},
    {"fought": [[2], [3]], "live": 1, "to": 2, "at": "bag", "next": [[1]], "headers": [7]},
    {"fought": [[2], [3]], "live": 2, "next": [[1]], "headers": [7], "gap": []},
]


def _toml_row(r: dict) -> str:
    def value(v):
        return f'"{v}"' if isinstance(v, str) else str(v).replace("'", '"')

    return "{ " + ", ".join(f"{k} = {value(v)}" for k, v in r.items()) + " }"


def live_text(rows=LIVE_ROWS, leave=None) -> str:
    # LIVE_ROWS leave at gap 1 after 1 first only: leaving depends on the order, so the group
    # has no flat leave_gaps (``leave``: a line to put one back).
    three = FACTS.replace("[[[1], [2]]]", "[[[1], [2], [3]]]").replace(
        "leave_gaps = [1]\n", "" if leave is None else f"leave_gaps = {leave}\n"
    )
    table = "live_next = [\n" + "".join(f"  {_toml_row(r)},\n" for r in rows) + "]\n"
    return gauntlet_text("[1, 2, 3]", three + "fight_headers = 7\n" + table)


def test_gauntlet_live_next_rows_load_and_round_trip(tmp_path):
    f = load_overrides(write(tmp_path, live_text())).gauntlets[0].facts
    assert f.orders == (((1,), (2,), (3,)),) and len(f.live_next) == len(LIVE_ROWS)
    rows = {r.key: r for r in f.live_rows()}
    assert rows[((), 1, 1, "")].next == ((2,),) and rows[((), 2, 2, "")].next == ((1,),)
    assert rows[(((2,), (3,)), 1, 2, "bag")].next == ((1,),)  # the revive reopens 1
    assert rows[(((1,),), 2, 2, "")].gap == ("leave",)
    row = f.to_dict()
    assert row["live_next"] == LIVE_ROWS
    assert GauntletFacts.from_dict(row) == f
    plain = load_overrides(write(tmp_path, gauntlet_text())).gauntlets[0].facts
    assert plain.live_next == () and "live_next" not in plain.to_dict()
    assert f.leave_gaps is None and "leave_gaps" not in row and f.heal_gaps == f.pc_gaps == ()


@pytest.mark.parametrize(
    "text, match",
    [
        # leave depends on the order in LIVE_ROWS: a flat list is refused, whatever it holds
        (lambda: live_text(leave="[1]"), "leave_gaps must be absent"),
        (lambda: live_text(leave="[]"), "leave_gaps must be absent"),
        # heal is [] under every order and count: it keeps its flat list
        (lambda: live_text().replace("heal_gaps = []\n", ""), "heal_gaps None is not the list"),
        (lambda: live_text().replace("pc_gaps = []\n", "pc_gaps = [1]\n"), "pc_gaps"),
        # without live_next every gap field is required
        (lambda: gauntlet_text().replace("heal_gaps = []\n", ""), "missing heal_gaps"),
    ],
)
def test_gauntlet_gap_fields_follow_the_live_table(tmp_path, text, match):
    with pytest.raises(OverridesError, match=match):
        load_overrides(write(tmp_path, text()))


def test_a_gap_that_depends_on_the_live_count_is_refused_not_read_as_empty(tmp_path):
    # After 1 the walk leaves at one live only: a swap at gap 1 (it needs a PC or a walk-out;
    # pc_gaps is []) is refused with the live-count reason, not as "no walk-out".
    rows = [
        {**r, "gap": []} if r["fought"] == [[1]] and r["live"] == 2 and "at" not in r else r
        for r in LIVE_ROWS
    ]
    f = load_overrides(write(tmp_path, live_text(rows))).gauntlets[0].facts
    order = f.orders[0]
    assert f.leave_gaps is None and f.gap_holds(order, 1, "leave") is None
    assert f.walkout_at(order, 1) is None and f.gap_holds(order, 2, "leave") is False
    with pytest.raises(ValueError, match="depends on the live count"):
        f.leave_gaps_of(order)
    with pytest.raises(GauntletSettingsError, match="depends on the live count"):
        check_swap_gap(f, order, 1)
    with pytest.raises(GauntletSettingsError, match="under no open order"):
        check_swap_gap(f, order, 2)


def test_gauntlet_without_live_next_is_the_orders_prefixes(tmp_path):
    facts = (
        "orders = [[[1], [2]], [[2], [1]]]\nleave_gaps = [[], [1]]\nescape = true\n"
        'pc_gaps = [1]\nheal_gaps = []\nlocked_gaps = []\norders_source = "rom:x"\n'
        "fight_headers = 5\n"
    )
    f = load_overrides(write(tmp_path, gauntlet_text(facts=facts))).gauntlets[0].facts
    rows = {r.key: r for r in f.live_rows()}
    for live in (1, 2):
        assert rows[((), live, live, "")].next == ((1,), (2,))
        assert (
            rows[((), live, live, "")].headers == (5, 5) and rows[((), live, live, "")].gap is None
        )
        assert rows[(((1,),), live, live, "")].gap == ("pc",)
        assert rows[(((2,),), live, live, "")].gap == ("leave", "pc")
        assert rows[(((2,),), live, live, "")].next == ((1,),)
    assert rows[(((1,),), 1, 2, "bag")].next == ((2,),)
    assert rows[(((1,),), 2, 1, "pc")].next == rows[(((1,),), 1, 2, "pc")].next == ((2,),)
    assert len(rows) == 2 + 2 * (2 + 3)


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda rows: rows[1:], "no row before the first fight"),
        (lambda rows: [*rows, rows[0]], "two rows for one key"),
        (lambda rows: [r for r in rows if r["fought"] != [[1], [2]]], "no row after"),
        (lambda rows: [{**rows[0], "next": [[4]]}, *rows[1:]], "unfought fights"),
        (lambda rows: [{**rows[0], "gap": []}, *rows[1:]], "gap facts"),
        (
            lambda rows: [
                {**rows[3], "to": 1, "at": "bag"} if i == 3 else r for i, r in enumerate(rows)
            ],
            "change",
        ),
        (
            lambda rows: [
                {**r, "next": [[2], [3]], "headers": [7, 7]}
                if r["fought"] == [[1]] and r["live"] == 2 and "at" not in r
                else r
                for r in rows
            ],
            "not the orders|no row after",
        ),
        (lambda rows: [{**rows[0], "live": 3}, *rows[1:]], "1 or 2"),
        (lambda rows: [{**rows[0], "nxt": [[2]]}, *rows[1:]], "unknown|nxt"),
    ],
)
def test_gauntlet_live_next_refused(tmp_path, change, match):
    with pytest.raises(OverridesError, match=match):
        load_overrides(write(tmp_path, live_text(change(LIVE_ROWS))))


def test_route_204_north_a_faint_switches_the_continuation():
    # The committed table (tools/kaizo/gauntlet_map.py live_table on the Kaizo ROM). Twins 15
    # are a double-battle trainer: at two live they stop the walk first, at one they are passed
    # (trainer_encounter.c:108-116). A faint in the twins' fight leaves 13 next, which no order
    # of one live Pokemon starts with; a faint in 13's fight before the twins drops them.
    f = next(g for g in load_overrides().gauntlets if g.name == "Route 204 North").facts
    rows = {r.key: r for r in f.live_rows()}

    def nxt(fought, live, to=None, at=""):
        return rows[(tuple((t,) for t in fought), live, to or live, at)].next

    assert nxt([], 2) == ((15,),) and nxt([], 1) == ((13,),)
    assert nxt([15], 2) == nxt([15], 1) == ((13,),)
    assert nxt([13], 2) == ((14,), (15,)) and nxt([13], 1) == ((14,),)
    assert nxt([13, 14], 2) == ((15,),) and nxt([13, 14], 1) == ()  # one live: the chain ends
    # twins planned after a revive back to two live (a bag item on the player's tile)
    assert nxt([13, 14], 1, 2, "bag") == ((15,),)
    assert all(r.headers == (346,) * len(r.next) for r in f.live_rows())
    # leaving depends on the count (after 13: at one live, not at two): no flat leave_gaps;
    # the order's gaps are read from live_next
    assert f.leave_gaps is None and f.heal_gaps == f.pc_gaps == ()
    assert f.leave_gaps_of(f.orders[0]) == (1, 2)


def test_route_213_heal_is_read_from_live_next_by_order():
    # The lobby healer is reached at every gap after 106 first, at none after 110 first: no
    # flat heal_gaps, and no reader takes it as empty.
    f = next(g for g in load_overrides().gauntlets if g.name == "Route 213").facts
    assert f.heal_gaps is None and "heal_gaps" not in f.to_dict()
    assert f.pc_gaps == () and f.leave_gaps == ((1, 2, 3, 4),) * 2
    east, west = sorted(f.orders, key=lambda o: o[0] != (106,))
    assert [f.gap_holds(east, k, "heal") for k in range(1, 5)] == [True] * 4
    assert [f.gap_holds(west, k, "heal") for k in range(1, 5)] == [False] * 4


def test_missing_file_is_empty(tmp_path):
    assert load_overrides(tmp_path / "absent.toml") == Overrides()


# --- citation checks (synthetic docs) ---------------------------------------------------


def header(cite, text, tags=None, party=(("Onix", 12),)):
    return {
        "cite": cite,
        "text": text,
        "name": text.split(" (")[0],
        "tags": tags or {},
        "party": [{"species": s, "level": lv} for s, lv in party],
    }


def section(cite, name, trainers, tags=None):
    return {"cite": cite, "name": name, "text": name, "tags": tags or {}, "trainers": trainers}


def docs_of(*sections):
    return {"splits": {"S": {"sheet": "S Split", "sections": list(sections)}}}


SYN = docs_of(
    section(
        "S Split:1",
        "Warehouse",
        [header("S Split:2", "Grunt A"), header("S Split:3", "Grunt B")],
        {"permanent": "trick_room"},
    ),
    section(
        "S Split:10",
        "Mine",
        [
            header("S Split:11", "Worker A (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:12", "Worker B (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:13", "Hiker C"),
        ],
    ),
    section(
        "S Split:20",
        "League",
        [
            header("S Split:21", "Champion X (Permanent Gravity)", {"permanent": "gravity"}),
            header("S Split:22", "Hooded (PERMANENT TRICK ROOM)", {"permanent": "trick_room"}),
        ],
    ),
)


def test_section_tag_supports_header_citation():
    o = overrides_with(area={5: AreaOverride(True, False, None, "S Split:3")})
    assert check_citations(o, SYN) == []


def test_header_without_claim_is_stale():
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "S Split:13"),
            6: AreaOverride(False, True, None, "S Split:22"),
        }
    )
    assert check_citations(o, SYN) == [
        "area.5: S Split:13 does not say Trick Room",
        "area.6: S Split:22 does not say Gravity",
    ]


def test_citation_to_missing_row():
    o = overrides_with(area={5: AreaOverride(True, False, None, "S Split:99")})
    assert check_citations(o, SYN) == ["area.5: S Split:99 is no split sheet row"]


def test_gauntlet_citation_needs_gauntlet_tag():
    o = overrides_with(
        gauntlets=(
            Gauntlet("ok", (1, 2), "S Split:11"),
            Gauntlet("bad", (3, 4), "S Split:13"),
        )
    )
    assert check_citations(o, SYN) == ["gauntlet 'bad': S Split:13 does not say gauntlet"]


def test_non_sheet_sources_are_not_checked_against_docs():
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "rom:map header 1"),
            6: AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611"),
        }
    )
    assert check_citations(o, SYN) == []


def test_rom_trick_room_needs_override_or_exception():
    effects = [
        {"id": 5, "name": "A", "effect": "trick_room", "map_headers": [305]},
        {"id": 6, "name": "B", "effect": "trick_room", "map_headers": [305]},
        {"id": 7, "name": "C", "effect": "trick_room", "map_headers": [307]},
        {"id": 8, "name": "D", "effect": "trick_room", "map_headers": [307]},
    ]
    o = overrides_with(
        area={
            5: AreaOverride(True, False, None, "rom:map header 305 weather 18 (Trick Room)"),
            8: AreaOverride(False, True, None, "rom:x"),
        },
        area_exceptions={6: AreaException("two maps", "rom:y")},
    )
    assert check_rom_trick_room(o, effects) == [
        (
            "area.7: ROM Trick Room (C, map headers [307]) has no trick_room override "
            "and no area_exception"
        ),
        (
            "area.8: ROM Trick Room (D, map headers [307]) has no trick_room override "
            "and no area_exception"
        ),
    ]


TR_ROWS = [
    {"id": 5, "name": "Saturn", "effect": "trick_room", "map_headers": [305]},
    {"id": 6, "name": "Io", "effect": "trick_room", "map_headers": [305, 306]},
]


@pytest.mark.parametrize(
    ("tid", "source", "problem"),
    [
        (5, "rom:map header 305 weather 18 (Trick Room)", None),
        (6, "rom:map headers 305, 306 weather 18 (Trick Room)", None),
        (246, "rom:map header 999 weather 18 (Trick Room)", "has no Trick Room row"),
        (5, "rom:map header 306 weather 18 (Trick Room)", "has map headers [305]"),
        (5, "rom:somewhere with Trick Room", "is no ROM Trick Room citation"),
    ],
)
def test_rom_trick_room_citations_checked_back(tid, source, problem):
    rows = [*TR_ROWS, {"id": 7, "name": "X", "effect": "trick_room", "map_headers": [1]}]
    o = overrides_with(
        area={tid: AreaOverride(True, False, None, source)},
        area_exceptions={i: AreaException("r", "rom:y") for i in (5, 6, 7) if i != tid},
    )
    got = check_rom_trick_room(o, rows)
    if problem is None:
        assert got == []
    else:
        assert len(got) == 1 and got[0].startswith(f"area.{tid}: ") and problem in got[0]


@pytest.mark.parametrize(
    ("tid", "source", "problem"),
    [
        (5, "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 305)", None),
        (
            9,
            "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 305)",
            "not 9",
        ),
        (
            5,
            "rom:only Saturn on a Trick Room map (rom_field_effects.json, map header 999)",
            "[305]",
        ),
        (5, "rom:only Mars on a Trick Room map (rom_field_effects.json, map header 305)", "not 5"),
        (5, "rom:only Saturn somewhere", "is no ROM Trick Room citation"),
    ],
)
def test_rom_only_trainer_id_citations_checked(tid, source, problem):
    o = overrides_with(
        trainer_ids={"S Split:2": tid},
        sources={"trainer_ids.S Split:2": source},
        area_exceptions={5: AreaException("r", "rom:y"), 6: AreaException("r", "rom:y")},
    )
    got = check_rom_trick_room(o, TR_ROWS)
    if problem is None:
        assert got == []
    else:
        assert len(got) == 1 and got[0].startswith("trainer_ids.S Split:2: ") and problem in got[0]


def test_gauntlet_members_tied_to_cited_group():
    sheet = [
        Mapping("S Split:11", 21, "exact", (21,)),
        Mapping("S Split:12", 22, "exact", (22,)),
        Mapping("S Split:13", 23, "exact", (23,)),
        Mapping("S Split:2", 2, "exact", (2,)),
    ]
    o = overrides_with(
        gauntlets=(
            Gauntlet("ok", (21, 22), "S Split:11"),
            Gauntlet("wrong", (2, 23), "S Split:11"),
        )
    )
    got = check_gauntlet_members(o, SYN, sheet)
    assert [p.split(":")[0] for p in got] == ["gauntlet 'wrong'", "gauntlet 'wrong'"]
    assert "trainer 2 " in got[0] and "trainer 23 " in got[1]  # 23 is outside the chain


def test_trainer_in_two_gauntlets_refused(tmp_path):
    g = '[[gauntlet]]\nname = "{}"\ntrainers = {}\nsource = "user, 2026-11-03"\n'
    text = g.format("a", "[1, 2]") + g.format("b", "[2, 3]")
    with pytest.raises(OverridesError, match="gauntlet 'b': trainer 2 .*'a'"):
        load_overrides(write(tmp_path, text))


def test_ids_must_be_rom_trainers():
    o = overrides_with(
        area={5: AreaOverride(True, False, None, "rom:x")},
        gauntlets=(Gauntlet("g", (5, 9), "S Split:11"),),
    )
    assert check_trainer_ids(o, {5, 6}) == ["gauntlet 'g': 9 is no ROM trainer id"]


# --- draft (synthetic) ----------------------------------------------------------------


def syn_mappings(sheet, field_effects=(), starts=None, pkcalc=()):
    rom = {
        i: {"id": i, "name": f"T{i}", "start_condition": (starts or {}).get(i, "none")}
        for i in range(1, 40)
    }
    return Mappings(
        pkcalc=tuple(pkcalc),
        sheet=tuple(
            Mapping(c, i, "exact" if i else "unmapped", (i,) if i else ()) for c, i in sheet
        ),
        rom_trainers=rom,
        field_effects=tuple(field_effects),
    )


DRAFT_DOCS = docs_of(
    section(
        "S Split:1",
        "Warehouse",
        [
            header("S Split:2", "Grunt A"),
            header("S Split:3", "Grunt B"),
            header("S Split:4", "Grunt C"),
            header("S Split:5", "Grunt D"),
        ],
        {"permanent": "trick_room"},
    ),
    section(
        "S Split:10",
        "Highlands",
        [header("S Split:11", "Grunt B"), header("S Split:12", "Hiker")],
        {"weather": "hail", "assumed_permanent": True},
    ),
    section(
        "S Split:20",
        "Mine",
        [
            header("S Split:21", "W1 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:22", "W2 (Gauntlet)", {"gauntlet": "middle"}),
            header("S Split:23", "W3 (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:24", "W4 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:25", "W5 (Gauntlet End)", {"gauntlet": "end"}),
            header("S Split:26", "Galactic M (B2B With Galactic T)", {"b2b_with": "Galactic T"}),
            header("S Split:27", "GalacticfT (B2B With Galactic M)", {"b2b_with": "Galactic M"}),
            header("S Split:28", "U1 (Gauntlet Start)", {"gauntlet": "start"}),
            header("S Split:29", "U2 (Gauntlet End)", {"gauntlet": "end"}),
        ],
    ),
    section(
        "S Split:30",
        "Gym",
        [
            header("S Split:31", "Lass"),
            header("S Split:32", "Artist"),
            header("S Split:33", "Leader G"),
        ],
        {"gym_gauntlet": True},
    ),
    section(
        "S Split:40",
        "Cave",
        [
            header("S Split:41", "Mira (Tag Partner)", {"tag_partner": True}),
            header("S Split:42", "C1 (MULTI, GAUNTLET)", {"gauntlet": "middle"}),
            header("S Split:43", "C2 (MULTI, GAUNTLET)", {"gauntlet": "middle"}),
            header("S Split:44", "C3 (MULTI, GAUNTLET END)", {"gauntlet": "end"}),
            header("S Split:45", "C4 (MULTI, GAUNTLET END)", {"gauntlet": "end"}),
        ],
        {"optional": True, "gauntlet_with": "Mira"},
    ),
    section(
        "S Split:50",
        "League",
        [
            header("S Split:51", "Hooded (PERMANENT TRICK ROOM)", {"permanent": "trick_room"}),
            header("S Split:52", "Champion Cynthia (Permanent Gravity)", {"permanent": "gravity"}),
        ],
    ),
)
DRAFT_SHEET = [
    ("S Split:2", 2), ("S Split:3", 3), ("S Split:4", 4), ("S Split:5", None),
    ("S Split:11", 3), ("S Split:12", 12),
    ("S Split:21", 21), ("S Split:22", 22), ("S Split:23", 23),
    ("S Split:24", 24), ("S Split:25", 25), ("S Split:26", 26), ("S Split:27", 27),
    ("S Split:28", 28), ("S Split:29", None),
    ("S Split:31", 31), ("S Split:32", 32), ("S Split:33", 33),
    ("S Split:41", 34), ("S Split:42", 35), ("S Split:43", 36), ("S Split:44", 37),
    ("S Split:45", 38),
    ("S Split:51", 39), ("S Split:52", 1),
]  # fmt: skip
DRAFT_EFFECTS = [
    {"id": 2, "name": "T2", "effect": "trick_room", "map_headers": [305]},
    {"id": 3, "name": "T3", "effect": "trick_room", "map_headers": [307]},
    {"id": 4, "name": "T4", "effect": "trick_room", "map_headers": [305, 306]},
    {"id": 6, "name": "T6", "effect": "trick_room", "map_headers": [306]},
    {"id": 7, "name": "T7", "effect": "trick_room", "map_headers": [306]},
]


@pytest.fixture
def syn_draft(tmp_path):
    m = syn_mappings(
        DRAFT_SHEET,
        DRAFT_EFFECTS,
        starts={3: "unknown", 7: "unknown"},
        pkcalc=[Mapping("Champion Cynthia", 1, "exact", (1,))],
    )
    text = draft_overrides(DRAFT_DOCS, m)
    return text, load_overrides(write(tmp_path, text))


def test_draft_is_loadable_and_cites_clean(syn_draft):
    _, o = syn_draft
    assert check_citations(o, DRAFT_DOCS) == []


def test_draft_rom_trick_room_from_the_rom(syn_draft):
    _, o = syn_draft
    assert o.area[2] == AreaOverride(
        True, False, None, "rom:map header 305 weather 18 (Trick Room)"
    )
    assert o.area[4].source == "rom:map headers 305, 306 weather 18 (Trick Room)"
    assert o.area[6].trick_room  # ROM only, no sheet row
    assert check_rom_trick_room(o, DRAFT_EFFECTS) == []


def test_draft_sheet_only_trick_room_from_the_sheet(syn_draft):
    text, o = syn_draft
    assert "no map of this trainer has weather 18 and start_condition is 'none'" in text
    assert o.area[39] == AreaOverride(True, False, None, "S Split:51")


def test_draft_conflicts_commented_with_both_citations(syn_draft):
    text, o = syn_draft
    assert 3 not in o.area and 7 not in o.area
    assert set(o.area_exceptions) == {3, 7}
    block = text[text.index("# CONFLICT 3") :].split("\n\n")[0]
    assert "S Split:3" in block and "S Split:11" in block and "unknown" in block
    assert "# [area.3]" in block and "# trick_room = true" in block


def test_draft_unmapped_trick_room_header_listed(syn_draft):
    text, _ = syn_draft
    assert "# UNMAPPED S Split:5" in text


def test_draft_gravity_from_pkcalc(syn_draft):
    _, o = syn_draft
    assert o.area[1] == AreaOverride(False, True, None, "pkcalc:src/js/shared_controls.js:611")


def test_draft_gauntlet_groups(syn_draft):
    text, o = syn_draft
    got = {g.trainers: (g.source, g.facts) for g in o.gauntlets}
    assert got == {
        (21, 22, 23): ("S Split:21", None),
        (24, 25): ("S Split:24", None),
        (26, 27): ("S Split:26", None),
        (31, 32): ("S Split:30", None),  # leader excluded
        (35, 36, 37, 38): ("S Split:40", None),  # tag partner excluded
    }
    assert "rules = {}" not in text and "\norders = " not in text  # the draft has no facts
    # a group with an unmapped member is drafted commented out, citing the row
    block = text[text.index("# UNSETTLED gauntlet") :].split("\n\n")[0]
    assert "S Split:28" in block and "S Split:29" in block and "# [[gauntlet]]" in block


def test_draft_gauntlet_without_end_is_unsettled():
    d = docs_of(
        section(
            "S Split:1",
            "Mine",
            [
                header("S Split:2", "A (Gauntlet Start)", {"gauntlet": "start"}),
                header("S Split:3", "B (Gauntlet)", {"gauntlet": "middle"}),
            ],
        )
    )
    text = draft_overrides(d, syn_mappings([("S Split:2", 2), ("S Split:3", 3)]))
    assert "# UNSETTLED gauntlet" in text and "\n[[gauntlet]]" not in text


def test_build_mappings_checks_override_keys():
    pk = {"Leader X": {"party": []}}
    with pytest.raises(MappingError, match="Typo Split:1"):
        build_mappings(pk, SYN, [], [], [], {"Typo Split:1": 5})


# --- real inputs (each skips without them) ----------------------------------------------


@pytest.fixture(scope="module")
def field_effects(kaizo_rom_dir):
    return json.loads((kaizo_rom_dir / "rom_field_effects.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mappings(pkcalc, docs, kaizo_rom_trainers, species, field_effects):
    return build_mappings(
        pkcalc["trainers"],
        docs,
        kaizo_rom_trainers,
        species,
        field_effects,
        load_overrides().trainer_ids,
    )


def test_stale_citation_detected(docs):  # Roark Split:7's header has no TRICK ROOM tag
    o = overrides_with(area={7: AreaOverride(True, False, None, "Roark Split:7")})
    assert check_citations(o, docs) == ["area.7: Roark Split:7 does not say Trick Room"]


@pytest.mark.slow
def test_no_facts_in_draft(docs, mappings):
    assert "\norders = " not in draft_overrides(docs, mappings)


def test_real_draft_loads_and_cites_clean(docs, mappings, field_effects, tmp_path):
    o = load_overrides(write(tmp_path, draft_overrides(docs, mappings)))
    assert check_citations(o, docs) == []
    assert check_rom_trick_room(o, field_effects) == []
    assert {509, 559, 649} <= set(o.area_exceptions)  # start condition unknown (Task 8)
    assert not {509, 559, 649} & set(o.area)


def test_committed_file_checks_clean(docs):  # skips without the workbook
    assert check_citations(load_overrides(), docs) == []


def test_committed_file_covers_rom_trick_room(field_effects, kaizo_rom_trainers):
    o = load_overrides()
    assert check_rom_trick_room(o, field_effects) == []
    assert check_trainer_ids(o, {t["id"] for t in kaizo_rom_trainers}) == []


def test_committed_override_keys_known(mappings):
    by_label = {m.label: m for m in (*mappings.pkcalc, *mappings.sheet)}
    for label, tid in load_overrides().trainer_ids.items():
        assert (by_label[label].how, by_label[label].id) == ("override", tid)


def test_committed_gauntlet_members_tied_to_their_cite(docs, mappings):
    assert check_gauntlet_members(load_overrides(), docs, mappings.sheet) == []
    probe = overrides_with(gauntlets=(Gauntlet("wrong", (246, 870), "Roark Split:229"),))
    assert len(check_gauntlet_members(probe, docs, mappings.sheet)) == 2


def test_committed_rom_citations_hold(field_effects):
    o = load_overrides()
    assert check_rom_trick_room(o, field_effects) == []
    probe = overrides_with(
        area={246: AreaOverride(True, False, None, "rom:map header 999 weather 18 (Trick Room)")}
    )
    assert any(p.startswith("area.246: ") for p in check_rom_trick_room(probe, field_effects))


def test_trick_room_and_gravity_present():
    o = load_overrides()
    # K2b Task 15 fix round 1: 35 ROM Trick Room trainers with a second, non-Trick-Room
    # placement became area_exceptions (start condition refused); together >= 51 as before.
    tr = {i for i, a in o.area.items() if a.trick_room}
    assert len(tr) >= 19 and len(tr | set(o.area_exceptions)) >= 51
    assert any(a.gravity for a in o.area.values())


# The user's answers (2026-11-07) to the K2b start-condition calls; 676 Corbin stays unknown.
USER_WEATHERS = {
    318: "none", 325: "fog", 343: "none", 610: "rain", 746: "none", 747: "none",
    103: "rain", 105: "rain", 106: "rain", 107: "rain", 108: "rain", 110: "rain",
    111: "rain", 491: "rain", 291: "fog",
    **{i: "none" for i in (*range(874, 892), 894, 895, 896)},
}  # fmt: skip


def test_committed_file_records_the_users_start_answers():
    o = load_overrides()
    assert 676 not in o.area and 676 not in o.area_exceptions
    for tid, weather in USER_WEATHERS.items():
        a = o.area[tid]
        assert (a.weather, a.source, a.trick_room) == (weather, "user, 2026-11-07", False), tid
    # Only a user's answer sets a weather; the two-placement Trick Room answers say "none".
    assert all(a.user for a in o.area.values() if a.weather is not None)
    for tid in (509, 559, 649):
        a = o.area[tid]
        assert (a.trick_room, a.weather, a.user) == (True, "none", True), tid
    assert o.area[870].gravity and o.area[870].source == "Elite Four Split:298"


def test_committed_facts_recorded():
    raw = tomllib.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert raw["gauntlet"] and all("rules" not in g for g in raw["gauntlet"])
    o = load_overrides()
    assert all(g.facts is not None for g in o.gauntlets)
    wayward = next(g for g in o.gauntlets if g.name == "Wayward Cave")
    assert len(wayward.facts.orders) == 2 and wayward.facts.partner.trainer == 612
    route = next(g for g in o.gauntlets if g.name == "Route 213")
    assert len(route.facts.orders) == 2  # no Surf before the Fen Badge (walk-fix round)
    assert all(len(g.facts.fight_headers) == len(g.facts.orders) for g in o.gauntlets)


def test_elite_four_and_champion_run_through_coverage(kaizo_names, kaizo_trainers):
    # Elite Four + Champion (866-870) run as single fights until batch 4 adds their group with the
    # chain (B4-19; manager ruling 21 Nov, docs/gauntlet-design.md:18) and enter it with no new
    # blocker (:331): coverage refuses none of them, whether or not the group is recorded.
    from nuzline.data.tables import load_tables
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import Support, check_opponent, start_weather
    from tests.conftest import KAIZO_INTERMEDIATE, KAIZO_TABLES_DIR

    tables, inter = load_tables(KAIZO_TABLES_DIR), IntermediateData(KAIZO_INTERMEDIATE)
    for tid in range(866, 871):
        trainer = kaizo_trainers[tid]
        opp = build_trainer_party(trainer, inter)
        report = check_opponent(tables, kaizo_names, trainer, opp, start_weather(trainer))
        assert (report.missing, report.level) == ([], Support.EXACT), (tid, report.missing)


def test_real_mapping_with_overrides(pkcalc, kaizo_rom_trainers, species):
    m = map_trainers(pkcalc["trainers"], kaizo_rom_trainers, species, load_overrides().trainer_ids)
    assert sum(x.id is not None for x in m) >= 473


def test_fields_format_unknown_or_the_users_answer(tmp_path):  # Task 15; final review M17
    ok = '[fields.556]\nformat = "unknown"\nsource = "rom:zone events 257"\n'
    assert load_overrides(write(tmp_path, ok)).fields == {556: {"format": "unknown"}}
    bad = '[fields.556]\nformat = "double"\nsource = "rom:zone events 257"\n'
    with pytest.raises(OverridesError, match="with a 'user, <date>' source"):
        load_overrides(write(tmp_path, bad))
    user = '[fields.556]\nformat = "single"\nsource = "user, 2026-11-07"\n'
    assert load_overrides(write(tmp_path, user)).fields == {556: {"format": "single"}}
    pair = (
        '[fields.556]\nformat = "double_two_trainer"\nopponent_2 = 557\n'
        'source = "user, 2026-11-07"\n'
    )
    o = load_overrides(write(tmp_path, pair))
    assert o.fields == {556: {"format": "double_two_trainer", "opponent_2": 557}}
    assert check_trainer_ids(o, {556}) == ["fields.556: opponent_2 557 is no ROM trainer id"]
    for text, message in (
        (pair.replace("opponent_2 = 557\n", ""), "needs opponent_2"),
        (user.replace('"single"', '"triple"'), "one of"),
        (user.replace('"single"\n', '"single"\nopponent_2 = 557\n'), "opponent_2 needs format"),
        (user.replace('"single"\n', '"single"\npartner = 864\n'), "partner needs format 'multi'"),
    ):
        with pytest.raises(OverridesError, match=message):
            load_overrides(write(tmp_path, text))
    multi = (
        '[fields.521]\nformat = "multi"\nopponent_2 = 527\npartner = 864\n'
        'source = "user, 2026-11-07"\n'
    )
    o = load_overrides(write(tmp_path, multi))
    assert o.fields == {521: {"format": "multi", "opponent_2": 527, "partner": 864}}
    assert check_trainer_ids(o, {521, 527}) == ["fields.521: partner 864 is no ROM trainer id"]
    assert check_trainer_ids(o, {521, 527, 864}) == []


# --- the user's answer to a two-placement Trick Room trainer (final review I5) -------------

TWO_EFFECTS = [{"id": 9, "name": "Grunt", "effect": "trick_room", "map_headers": [310]}]
TWO_MAPS = {9: [310, 337]}


def test_area_map_header_needs_a_user_source(tmp_path):
    user = '[area.9]\nweather = "none"\nmap_header = 337\nsource = "user, 2026-11-07"\n'
    o = load_overrides(write(tmp_path, user))
    assert o.area[9] == AreaOverride(False, False, "none", "user, 2026-11-07", 337)
    assert o.area[9].user
    with pytest.raises(OverridesError, match="map_header names a user's choice"):
        load_overrides(write(tmp_path, user.replace("user, 2026-11-07", "rom:x")))
    with pytest.raises(OverridesError, match="map_header must be an integer"):
        load_overrides(write(tmp_path, user.replace("337", '"337"')))


@pytest.mark.parametrize(
    ("area", "exceptions", "problem"),
    [
        ({}, {9: AreaException("two maps", "rom:y")}, None),  # refused until answered
        ({9: AreaOverride(False, False, "none", "user, 2026-11-07")}, {}, None),
        ({9: AreaOverride(True, False, None, "user, 2026-11-07", 310)}, {}, None),
        ({9: AreaOverride(False, False, "none", "user, 2026-11-07", 337)}, {}, None),
        (
            {9: AreaOverride(True, False, None, "rom:map header 310 weather 18 (Trick Room)")},
            {},
            "needs an area_exception",
        ),
        ({}, {}, "needs an area_exception"),
        (
            {9: AreaOverride(False, False, "none", "user, 2026-11-07", 310)},
            {},
            "map_header 310 has Trick Room, but trick_room is false",
        ),
        (
            {9: AreaOverride(True, False, None, "user, 2026-11-07", 337)},
            {},
            "map_header 337 has no Trick Room",
        ),
        (
            {9: AreaOverride(True, False, None, "user, 2026-11-07", 1)},
            {},
            "map_header 1 is not one of [310, 337]",
        ),
    ],
)
def test_two_placement_rule_takes_the_users_answer(area, exceptions, problem):
    o = overrides_with(area=area, area_exceptions=exceptions)
    got = check_two_placement(o, TWO_EFFECTS, TWO_MAPS)
    if problem is None:
        assert got == [] and check_rom_trick_room(o, TWO_EFFECTS) == []
    else:
        assert len(got) == 1 and problem in got[0]


def _real_inputs():
    from tools.kaizo.overrides import INTERMEDIATE

    if not (INTERMEDIATE / "rom" / "rom_trainer_maps.json").exists():
        pytest.skip("Kaizo intermediates not extracted")
    if not (INTERMEDIATE / "docs" / "docs.json").exists():
        pytest.skip("Kaizo docs not extracted")
    return INTERMEDIATE


def test_overrides_check_runs_the_two_placement_rule(tmp_path, capsys):
    # The documented route (pipeline.md): the user's [area.539] replaces the area_exception
    # (the committed file holds the answer, the Galactic HQ fight). The other placement is
    # an answer too; a rom: Trick Room entry instead fails `overrides check`, as the build does.
    from tools.kaizo.overrides import main

    inter = _real_inputs()
    assert main(["check", "--overrides", str(DEFAULT_PATH), "--intermediate", str(inter)]) == 0
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    start = text.index("[area.539]")
    end = text.index("\n\n", start) + 2
    user = '[area.539]\nweather = "none"\nmap_header = 337\nsource = "user, 2026-11-07"\n\n'
    path = write(tmp_path, text[:start] + user + text[end:])
    assert main(["check", "--overrides", str(path), "--intermediate", str(inter)]) == 0
    rom = '[area.539]\ntrick_room = true\nsource = "rom:map header 310 weather 18 (Trick Room)"\n\n'
    path = write(tmp_path, text[:start] + rom + text[end:])
    assert main(["check", "--overrides", str(path), "--intermediate", str(inter)]) == 1
    assert "trainer 539: Trick Room headers [310]" in capsys.readouterr().out
