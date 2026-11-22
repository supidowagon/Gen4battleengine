"""The gauntlet option registry, precedence and tracked facts (GT-1; synthetic groups and
settings only, no user choice) and the settings input: flags, a rules file, ``[run]`` of
``data/local.toml``, the per-run re-run inputs and every input error (GT-3)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nuzline.data.gauntlet import (
    COUNTED,
    GAP_CAP_KEYS,
    GAP_LEVEL_KEYS,
    KINDS,
    LEVEL,
    OPTIONS,
    RUN,
    SET,
    GauntletFacts,
    GauntletSettingsError,
    Layer,
    Origin,
    RunInput,
    SettingsError,
    Swap,
    check_swap_gap,
    load_local,
    load_rules_file,
    missing_transitions,
    parse_flags,
    parse_ids,
    parse_members,
    resolve,
    resolve_settings,
)
from nuzline.data.trainer import Trainer


def facts_dict(**over) -> dict:
    """A synthetic three-fight group with a partner fact; ``over`` replaces keys."""
    d = {
        "name": "Test Cave",
        "source": "rom:test",
        "orders": [[[10], [11, 12], [13]], [[11, 12], [10], [13]]],
        "orders_source": "rom:test orders",
        "leave_gaps": [[1], []],
        "leave_gaps_source": "rom:test leave",
        "escape": True,
        "escape_source": "rom:test escape",
        "pc_gaps": [2],
        "pc_gaps_source": "rom:test pc",
        "heal_gaps": [],
        "heal_gaps_source": "rom:test heal",
        "locked_gaps": [2],
        "locked_gaps_source": "rom:test locked",
        "partner": {"trainer": 99, "join_before": [2], "source": "rom:test partner"},
    }
    return {**d, **over}


@pytest.fixture
def facts():
    return GauntletFacts.from_dict(facts_dict())


def _layer(src):
    return Layer(
        src, gauntlet={"heal": "full", "team": "moves"}, gaps={2: {"heal": "full", "swap_cap": 3}}
    )


def test_every_option_is_complete():
    for name, o in OPTIONS.items():
        assert o.name == name and o.kind in KINDS and o.transition and callable(o.validate)
        if o.has_default:
            assert o.validate(o.default) == o.default
        assert o.values is not None or o.kind == COUNTED
    assert not missing_transitions({o.transition for o in OPTIONS.values()})
    assert missing_transitions(set()) == tuple(OPTIONS)


def test_defaults_are_the_first_value():
    for o in OPTIONS.values():
        if o.kind in (LEVEL, SET) or o.scope == RUN:
            assert o.default == o.values[0], o.name
    assert OPTIONS["order"].default == "worst" and "partner_from" not in OPTIONS
    assert OPTIONS["time"].default is None and not OPTIONS["time"].has_default
    assert OPTIONS["swap"].default == Swap(0, 0) and OPTIONS["bag_uses"].default == 0
    assert OPTIONS["move_sources"].default == ("bag",)


def test_strictest_first_order():
    assert OPTIONS["heal"].values == ("none", "bag", "full")
    assert OPTIONS["pc_heal"].values == ("off", "on")
    assert OPTIONS["team"].values == ("none", "reorder", "moves")
    assert OPTIONS["growth_items"].values == ("forbidden", "allowed")
    assert OPTIONS["cap_sit_out_in_party"].values == ("forbidden", "allowed")
    assert OPTIONS["cap_overflow"].values == ("sit_out_after_fight", "fight")
    assert OPTIONS["field_poison"].values == ("worst_case", "ignore")
    assert OPTIONS["bag_uses"].default == 0
    assert "partner_heal" not in OPTIONS


def test_scopes_and_gap_keys():
    assert {n for n, o in OPTIONS.items() if o.scope == RUN} == {
        "deaths_scope",
        "cap_overflow",
        "cap_sit_out_in_party",
        "revived_dead",
        "shedinja",
        "field_poison",
    }
    assert GAP_LEVEL_KEYS == ("heal", "pc_heal", "team", "growth_items")
    assert GAP_CAP_KEYS == ("swap_cap", "bag_cap")
    assert {n for n, o in OPTIONS.items() if o.per_gap} == set(GAP_LEVEL_KEYS)


def test_validators():
    swap = OPTIONS["swap"].validate
    assert swap({"bench": 1, "budget": -1}) == Swap(1, -1)
    for bad in ({"bench": -2, "budget": 0}, {"bench": 1}, 3):
        with pytest.raises(SettingsError):
            swap(bad)
    assert OPTIONS["bag_uses"].validate(-1) == -1
    with pytest.raises(SettingsError):
        OPTIONS["bag_uses"].validate(-2)
    assert OPTIONS["order"].validate([12, 10]) == (12, 10)
    with pytest.raises(SettingsError):
        OPTIONS["order"].validate("fastest")
    assert OPTIONS["move_sources"].validate(["bag", "tutor", "relearner"])
    with pytest.raises(SettingsError):
        OPTIONS["move_sources"].validate(["tutor"])
    with pytest.raises(SettingsError):
        OPTIONS["heal"].validate("most")


def test_resolve_defaults(facts):
    s = resolve(facts, facts.orders[0])
    assert all(s[n].origin == Origin.DEFAULT for n in OPTIONS)
    assert s["heal"].value == "none" and s["time"].value is None
    assert s["time"].source == "unset" and s["heal"].source == "default"


def test_precedence_per_value(facts):
    order = facts.orders[0]
    flag = Layer("--rule", gauntlet={"heal": "full"})
    rules = Layer("rules.toml", gauntlet={"heal": "bag"})
    local = Layer("local.toml", gauntlet={"heal": "bag"}, run={"shedinja": "allowed"})
    s = resolve(facts, order, flag, rules, local)
    assert s["heal"] == ("full", Origin.FLAG, "--rule")
    assert s["shedinja"] == ("allowed", Origin.LOCAL, "local.toml")
    assert resolve(facts, order, None, rules, local)["heal"] == (
        "bag",
        Origin.RULES_FILE,
        "rules.toml",
    )
    local2 = Layer("local.toml", gauntlet={"heal": "full"})
    assert resolve(facts, order, None, None, local2)["heal"].origin == Origin.LOCAL
    assert resolve(facts, order)["heal"].origin == Origin.DEFAULT


def test_locked_gap_beats_every_origin(facts):
    order = facts.orders[0]
    s = resolve(facts, order, _layer("--rule"), _layer("rules"), _layer("local"))
    gap = s.at_gap(2)
    assert gap["heal"] == ("none", Origin.LOCKED, "rom:test locked")
    assert gap["team"].value == "none" and gap["pc_heal"].value == "off"
    assert gap["growth_items"].value == "forbidden"
    assert gap["swap_cap"] == (0, Origin.LOCKED, "rom:test locked") and gap["bag_cap"].value == 0
    # drop the lock: the flag value answers, then rules file, local, default
    open_facts = GauntletFacts.from_dict(facts_dict(locked_gaps=[]))
    free = resolve(open_facts, order, _layer("--rule"), _layer("rules"), _layer("local"))
    assert free.at_gap(2)["heal"].origin == Origin.FLAG and free.at_gap(2)["swap_cap"].value == 3
    no_flag = resolve(open_facts, order, None, _layer("rules"), _layer("local"))
    assert no_flag.at_gap(2)["heal"].origin == Origin.RULES_FILE
    only_local = resolve(open_facts, order, None, None, _layer("local"))
    assert only_local.at_gap(2)["heal"].origin == Origin.LOCAL
    assert resolve(open_facts, order).at_gap(2)["heal"].origin == Origin.DEFAULT


def test_per_gap_override_and_caps(facts):
    order = facts.orders[0]
    open_facts = GauntletFacts.from_dict(facts_dict(locked_gaps=[]))
    layer = Layer(
        "rules",
        gauntlet={"heal": "bag", "swap": {"bench": 1, "budget": 2}},
        gaps={1: {"heal": "full", "swap_cap": 0}},
    )
    s = resolve(open_facts, order, None, layer)
    assert s.at_gap(1)["heal"].value == "full" and s.at_gap(1)["swap_cap"].value == 0
    assert s.at_gap(2)["heal"].value == "bag" and s.at_gap(2)["swap_cap"].value is None
    assert s["swap"].value == Swap(1, 2)


def test_input_errors(facts):
    order = facts.orders[0]
    bad = [
        Layer("x", gauntlet={"nope": 1}),
        Layer("x", gauntlet={"heal": "most"}),
        Layer("x", gauntlet={"shedinja": "allowed"}),  # run key in a gauntlet table
        Layer("x", run={"heal": "bag"}),  # gauntlet key in run
        Layer("x", gauntlet={"partner_from": 1}),  # the game fixes the join point
        Layer("x", gaps={3: {"heal": "bag"}}),  # n = 3: gaps 1..2
        Layer("x", gaps={1: {"order": "worst"}}),  # not per gap
        Layer("x", gaps={1: {"swap_cap": -1}}),
    ]
    for layer in bad:
        with pytest.raises(SettingsError):
            resolve(facts, order, layer)
    with pytest.raises(ValueError):
        resolve(facts, ((1,), (2,)))


def test_facts_round_trip_and_helpers(facts):
    assert GauntletFacts.from_dict(facts.to_dict()) == facts
    assert facts.to_dict() == facts_dict()
    assert facts.partner.trainer == 99 and facts.partner.join_before == (2,)
    assert facts.partner.joins == 2
    o0, o1 = facts.orders
    assert facts.n_fights(o0) == 3 and facts.fight_ids(o0) == (10, 11, 13)
    assert facts.fight_ids(o1) == (11, 10, 13)
    assert facts.leave_gaps_of(o1) == ()
    with pytest.raises(ValueError):
        GauntletFacts.from_dict(facts_dict(leave_gaps=[[1]]))
    # the game fixes one join point: a list of several is not a fact
    two = {"trainer": 99, "join_before": [1, 2], "source": "rom:test partner"}
    with pytest.raises(ValueError, match="not one"):
        GauntletFacts.from_dict(facts_dict(partner=two))


def test_walkout_gaps(facts):
    o0, o1 = facts.orders
    # the partner joins before fight 2: gap 1 has no partner yet, so escape works there
    assert facts.walkout_gaps(o0) == (1,)
    assert facts.walkout_gaps(o1) == (1,)
    # joins before fight 1 (Wayward Cave): set at every gap, only the leave gap remains
    at_1 = {"trainer": 99, "join_before": [1], "source": "rom:test partner"}
    first = GauntletFacts.from_dict(facts_dict(partner=at_1))
    assert first.walkout_gaps(o0) == (1,) and first.walkout_gaps(o1) == ()
    # escape holds and no partner: every gap is a walk-out gap
    alone = GauntletFacts.from_dict(facts_dict(partner=None))
    assert alone.walkout_gaps(o0) == (1, 2) and alone.walkout_gaps(o1) == (1, 2)
    no_escape = GauntletFacts.from_dict(facts_dict(escape=False, partner=None))
    assert no_escape.walkout_gaps(o0) == (1,) and no_escape.walkout_gaps(o1) == ()


def test_trainer_gauntlet_is_group_and_fight_id():
    t = Trainer(1, "A", "C", 0, 0, 7, (), (), "none", "single", gauntlet=("Test Cave", 11))
    assert t.gauntlet == ("Test Cave", 11) and t.gauntlet_facts is None


# --- GT-3: the settings input -----------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
GROUP = "Test Cave"


@pytest.fixture
def plain():
    """No partner, no escape, nothing locked: the walk-out gaps are the leave gaps (gap 1
    of the first order, none of the second), the PC is at gap 2, no free heal."""
    return GauntletFacts.from_dict(facts_dict(escape=False, partner=None, locked_gaps=[]))


def setup(facts, rule=(), gap_rule=(), **kw):
    return resolve_settings(facts, flags=parse_flags(list(rule), list(gap_rule)), **kw)


def refused(facts, rule=(), gap_rule=(), **kw) -> GauntletSettingsError:
    with pytest.raises(GauntletSettingsError) as e:
        setup(facts, rule, gap_rule, **kw)
    return e.value


def test_error_names_key_group_and_origin(facts):
    e = refused(facts, ["order=10,13,11"])
    assert (e.key, e.group, e.origin) == ("order", GROUP, "--rule/--gap-rule")
    assert str(e).startswith(f"--rule/--gap-rule: group '{GROUP}': order: ")
    assert isinstance(e, SettingsError)


def test_parse_flags():
    raw = parse_flags(
        [
            "heal=bag",
            "swap=bench:-1,budget:2",
            "order=11,10,13",
            "move_sources=bag,tutor,relearner",
            "bag_uses=3",
            "time=night",
            "shedinja=allowed",
        ],
        ["2:heal=full", "1:swap_cap=2", "1:pc_heal=on"],
    )
    assert raw.run == {"shedinja": "allowed"}
    assert raw.gauntlet == {
        "heal": "bag",
        "swap": {"bench": -1, "budget": 2},
        "order": [11, 10, 13],
        "move_sources": ["bag", "tutor", "relearner"],
        "bag_uses": 3,
        "time": "night",
    }
    assert raw.gaps == {2: {"heal": "full"}, 1: {"swap_cap": 2, "pc_heal": "on"}}
    assert parse_flags(["order=sheet"]).gauntlet == {"order": "sheet"}
    assert parse_flags([], []).gauntlet == {}


def test_flag_values_resolve(plain, facts):
    s = setup(plain, ["swap=bench:-1,budget:2", "order=11,10,13", "bag_uses=3"])
    assert s.settings["swap"] == (Swap(-1, 2), Origin.FLAG, "--rule/--gap-rule")
    assert s.settings["order"].value == (11, 10, 13) and s.orders == (plain.orders[1],)
    assert s.settings["bag_uses"].value == 3
    ms = setup(plain, ["move_sources=bag,tutor,relearner"]).settings["move_sources"]
    assert ms.value == ("bag", "tutor", "relearner")


def test_flag_errors():
    for item in ("heal", "nope=1", "order=1,x", "swap=bench:1,bench:2", "swap=1"):
        with pytest.raises(GauntletSettingsError):
            parse_flags([item])
    with pytest.raises(GauntletSettingsError, match="given twice"):
        parse_flags(["heal=bag", "heal=full"])
    for item in ("heal=full", "x:heal=full", "2:order=worst", "2:swap_cap=a", "2:nope=1"):
        with pytest.raises(GauntletSettingsError):
            parse_flags([], [item])
    with pytest.raises(GauntletSettingsError, match="given twice"):
        parse_flags([], ["2:heal=full", "2:heal=bag"])


def test_value_errors_resolve(plain):
    for rule in ("heal=most", "swap=bench:1", "bag_uses=x", "time=noon", "shedinja=maybe"):
        e = refused(plain, [rule])
        assert e.group == GROUP and rule.split("=")[0] in str(e)


def test_scope_and_unknown_key_errors(plain, tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text(f'[gauntlet."{GROUP}"]\nshedinja = "allowed"\n')
    with pytest.raises(GauntletSettingsError, match="shedinja.*run"):
        resolve_settings(plain, rules_file=load_rules_file(rules))
    rules.write_text('[run]\nheal = "bag"\n')
    with pytest.raises(GauntletSettingsError, match="heal.*gauntlet"):
        resolve_settings(plain, rules_file=load_rules_file(rules))
    rules.write_text(f'[gauntlet."{GROUP}"]\nnope = 1\n')
    e = refused(plain, rules_file=load_rules_file(rules))
    assert "nope" in str(e) and e.group == GROUP


def test_rules_file_shape(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[run]\nshedinja = "allowed"\n[gauntlet."A".gap.2]\nheal = "bag"\n')
    raw = load_rules_file(rules)
    assert raw.run == {"shedinja": "allowed"} and raw.source == str(rules)
    assert raw.groups["A"].gaps == {2: {"heal": "bag"}}
    assert raw.layer("A").gaps == {2: {"heal": "bag"}} and raw.layer("B").gaps == {}
    for text in ("nope = 1\n", '[gauntlet."A".gap.x]\nheal = "bag"\n', "gauntlet = 3\n", "[[run"):
        rules.write_text(text)
        with pytest.raises(GauntletSettingsError):
            load_rules_file(rules)
    with pytest.raises(GauntletSettingsError, match="cannot read"):
        load_rules_file(tmp_path / "absent.toml")


def test_unknown_group_in_a_rules_file(plain, tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[gauntlet."Nope"]\nheal = "bag"\n')
    raw = load_rules_file(rules)
    resolve_settings(plain, rules_file=raw)  # no group list given: nothing to check against
    with pytest.raises(GauntletSettingsError, match="Nope"):
        resolve_settings(plain, rules_file=raw, known_groups={GROUP, "Other"})


def test_precedence_across_four_inputs(facts, tmp_path):
    local_file = tmp_path / "local.toml"
    local_file.write_text(
        "vanilla_rom = 'v.nds'\n"
        '[run]\ndeaths_scope = "fight"\ncap_overflow = "fight"\nshedinja = "allowed"\n'
    )
    rules_file = tmp_path / "rules.toml"
    rules_file.write_text(
        '[run]\ncap_overflow = "sit_out_after_fight"\nshedinja = "forbidden"\n'
        f'[gauntlet."{GROUP}"]\nheal = "bag"\nteam = "reorder"\n'
        f'[gauntlet."{GROUP}".gap.1]\nheal = "bag"\n'
        '[gauntlet."Other"]\nheal = "full"\n'
    )
    s = resolve_settings(
        facts,
        flags=parse_flags(["shedinja=allowed", "team=moves"]),
        rules_file=load_rules_file(rules_file),
        local=load_local(local_file),
        known_groups={GROUP, "Other"},
    ).settings
    src_local, src_rules, src_flag = f"{local_file} [run]", str(rules_file), "--rule/--gap-rule"
    assert s["deaths_scope"] == ("fight", Origin.LOCAL, src_local)
    assert s["cap_overflow"] == ("sit_out_after_fight", Origin.RULES_FILE, src_rules)
    assert s["shedinja"] == ("allowed", Origin.FLAG, src_flag)
    assert s["heal"] == ("bag", Origin.RULES_FILE, src_rules)
    assert s["team"] == ("moves", Origin.FLAG, src_flag)
    assert s["field_poison"] == ("worst_case", Origin.DEFAULT, "default")
    assert s.at_gap(1)["heal"] == ("bag", Origin.RULES_FILE, src_rules)


def test_local_without_run_or_absent_gives_defaults(plain, tmp_path):
    local_file = tmp_path / "local.toml"
    local_file.write_text("vanilla_rom = 'v.nds'\nkaizo_save = 's.sav'\n")
    for path in (local_file, tmp_path / "absent.toml"):
        raw = load_local(path)
        assert raw.run == {}
        s = resolve_settings(plain, local=raw).settings
        assert all(s[n].origin == Origin.DEFAULT for n in OPTIONS)
    local_file.write_text("run = 3\n")
    with pytest.raises(GauntletSettingsError, match="not a table"):
        load_local(local_file)
    local_file.write_text('[run]\nheal = "bag"\n')  # a gauntlet key in [run]
    with pytest.raises(GauntletSettingsError, match="heal"):
        resolve_settings(plain, local=load_local(local_file))


def test_example_run_block_is_the_code_defaults():
    raw = load_local(REPO / "data" / "local.example.toml")
    defaults = {n: o.default for n, o in OPTIONS.items() if o.scope == RUN}
    assert dict(raw.run) == defaults
    facts = GauntletFacts.from_dict(facts_dict(partner=None))
    s = resolve_settings(facts, local=raw).settings
    assert all(s[n] == (v, Origin.LOCAL, raw.source) for n, v in defaults.items())


def test_order_defaults_to_worst(facts):
    s = resolve_settings(facts)
    assert s.settings["order"] == ("worst", Origin.DEFAULT, "default")
    assert s.orders == facts.orders and not s.sheet_fallback
    best = setup(facts, ["order=best"])
    assert best.settings["order"].value == "best" and best.orders == facts.orders
    sheet = setup(facts, ["order=sheet"])
    assert sheet.sheet_fallback and sheet.orders == facts.orders


def test_order_errors_and_no_join_point_setting(facts, tmp_path):
    assert refused(facts, ["order=10,13,11"]).key == "order"
    # the game fixes when the partner joins: there is no setting for it
    assert refused(facts, ["partner_from=1"]).key == "partner_from"
    rules = tmp_path / "rules.toml"
    rules.write_text(f'[gauntlet."{GROUP}"]\npartner_from = 1\n')
    with pytest.raises(GauntletSettingsError, match="partner_from"):
        resolve_settings(facts, rules_file=load_rules_file(rules))


def test_rerun_with_and_without_a_partner(facts, plain):
    assert resolve_settings(facts, run=RunInput(beaten=(10,))).orders == (facts.orders[0],)
    assert resolve_settings(plain, run=RunInput(beaten=(10,))).orders == (plain.orders[0],)


def test_beaten_and_next_narrow_the_orders(plain):
    o0, o1 = plain.orders
    assert resolve_settings(plain, run=RunInput(beaten=(10, 11))).orders == (o0,)
    assert resolve_settings(plain, run=RunInput(beaten=(12,))).orders == (o1,)  # either id
    assert resolve_settings(plain, run=RunInput(next_fight=12)).orders == (o1,)
    assert resolve_settings(plain, run=RunInput(beaten=(10,), next_fight=11)).orders == (o0,)
    for run, key in (
        (RunInput(beaten=(13,)), "--beaten"),  # no order starts with 13
        (RunInput(next_fight=13), "--next"),  # no order opens with 13
        (RunInput(beaten=(10,), next_fight=13), "--next"),  # 13 is the third fight, not the second
    ):
        with pytest.raises(GauntletSettingsError) as e:
            resolve_settings(plain, run=run)
        assert e.value.key == key
    with pytest.raises(GauntletSettingsError) as e:  # the order given does not start with 10
        setup(plain, ["order=11,10,13"], run=RunInput(beaten=(10,)))
    assert e.value.key == "--beaten"


def test_gap_outside_the_order(plain):
    for gap in (0, 3):
        e = refused(plain, gap_rule=[f"{gap}:heal=bag"])
        assert e.key == f"gap {gap}" and "outside 1..2" in str(e)
    assert setup(plain, gap_rule=["2:heal=bag"]).settings.at_gap(2)["heal"].value == "bag"


def test_locked_gap_names_the_source(facts, tmp_path):
    for gap_rule in ("2:heal=bag", "2:pc_heal=on", "2:team=moves", "2:swap_cap=1", "2:bag_cap=1"):
        e = refused(facts, gap_rule=[gap_rule])
        assert "rom:test locked" in str(e) and e.key.startswith("gap 2") and e.group == GROUP
    ok = setup(facts, gap_rule=["2:heal=none", "2:swap_cap=0", "2:pc_heal=off"])
    assert ok.settings.at_gap(2)["heal"].origin == Origin.LOCKED
    rules = tmp_path / "rules.toml"
    rules.write_text(f'[gauntlet."{GROUP}".gap.2]\nheal = "full"\n')
    with pytest.raises(GauntletSettingsError, match="rom:test locked") as err:
        resolve_settings(facts, rules_file=load_rules_file(rules))
    assert err.value.origin == str(rules)
    # a group, rules-file or default value reaching a locked gap is not an error
    inherited = setup(facts, ["heal=full", "team=moves"])
    assert inherited.settings.at_gap(2)["heal"] == ("none", Origin.LOCKED, "rom:test locked")


def test_map_settled_heal(plain):
    # gap 1 is a walk-out gap of the first order only; no free heal anywhere
    assert setup(plain, gap_rule=["1:heal=full"]).settings.at_gap(1)["heal"].value == "full"
    e = refused(plain, ["order=11,10,13"], ["1:heal=full"])
    assert e.key == "gap 1 heal" and "free full heal" in str(e) and "rom:test" in str(e)
    assert refused(plain, gap_rule=["2:heal=full"]).key == "gap 2 heal"
    setup(plain, gap_rule=["2:heal=bag", "1:heal=bag"])  # only `full` needs the map
    with_heal = GauntletFacts.from_dict(
        facts_dict(escape=False, partner=None, locked_gaps=[], heal_gaps=[2])
    )
    setup(with_heal, gap_rule=["2:heal=full"])


def test_map_settled_pc_and_swap(plain):
    setup(plain, gap_rule=["2:pc_heal=on", "2:swap_cap=1"])  # the PC gap
    setup(plain, gap_rule=["1:pc_heal=on", "1:swap_cap=1"])  # walk-out gap of one open order
    e = refused(plain, ["order=11,10,13"], ["1:pc_heal=on"])
    assert e.key == "gap 1 pc_heal" and "a PC" in str(e)
    assert refused(plain, ["order=11,10,13"], ["1:swap_cap=1"]).key == "gap 1 swap_cap"
    setup(plain, ["order=11,10,13"], ["1:swap_cap=0"])  # a cap of 0 asks for nothing


def test_map_settled_with_a_partner():
    # escape holds at the gaps before the partner joins; no PC and no leave gap in the group
    later = GauntletFacts.from_dict(facts_dict(locked_gaps=[], pc_gaps=[], leave_gaps=[[], []]))
    setup(later, gap_rule=["1:pc_heal=on"])  # joins before fight 2: gap 1 has escape
    assert refused(later, gap_rule=["2:pc_heal=on"]).key == "gap 2 pc_heal"
    at_1 = {"trainer": 99, "join_before": [1], "source": "rom:test partner"}
    first = GauntletFacts.from_dict(
        facts_dict(locked_gaps=[], pc_gaps=[], leave_gaps=[[], []], partner=at_1)
    )
    e = refused(first, gap_rule=["1:pc_heal=on"])  # set from fight 1: no escape at any gap
    assert e.key == "gap 1 pc_heal"


def test_planned_swap(plain):
    o0, o1 = plain.orders
    check_swap_gap(plain, o0, 2)  # the PC gap
    check_swap_gap(plain, o0, 1)  # a walk-out gap
    check_swap_gap(plain, o1, 2)
    with pytest.raises(GauntletSettingsError, match="a swap at gap 1") as e:
        check_swap_gap(plain, o1, 1)  # no PC, no walk-out
    assert e.value.key == "swap" and e.value.group == GROUP


def _mon(hp):
    return SimpleNamespace(current_hp=hp)


def _save():
    return SimpleNamespace(
        party=(_mon(40), _mon(0), _mon(7)),
        boxes=((_mon(None), None), (_mon(None),)),
    )


def test_parse_members_and_ids():
    got = parse_members(["party:2,box:1:3", "box:2:0", "party:2"], "--dead")
    assert got == (("box", 0, 3), ("box", 1, 0), ("party", 0, 2))
    assert [m.label for m in got] == ["box:1:3", "box:2:0", "party:2"]
    assert parse_members([], "--dead") == ()
    for bad in ("2", "party", "box:0:1", "box:1", "slot:1"):
        with pytest.raises(GauntletSettingsError) as e:
            parse_members([bad], "--bench")
        assert e.value.key == "--bench"
    assert parse_ids(["10,11", "13", "10"], "--beaten") == (10, 11, 13, 10)
    with pytest.raises(GauntletSettingsError, match="--beaten"):
        parse_ids(["10,x"], "--beaten")


def test_dead_and_bench_against_the_save(plain):
    save = _save()

    def run(dead=(), bench=()):
        return RunInput(dead=parse_members(dead, "--dead"), bench=parse_members(bench, "--bench"))

    ok = resolve_settings(plain, run=run(["party:0", "box:1:0"], ["box:2:0"]), save=save)
    assert ok.run.dead == (("box", 0, 0), ("party", 0, 0)) and ok.run.bench == (("box", 1, 0),)
    resolve_settings(plain, run=run(["party:2"]), save=save)  # hurt, not dead
    cases = [
        (run(["party:3"]), "--dead", "party has 3"),
        (run(["party:1"]), "--dead", "0 HP"),
        (run(["box:1:1"]), "--dead", "no Pokemon"),  # an empty slot
        (run(["box:1:5"]), "--dead", "no Pokemon"),  # a slot outside the box
        (run(["box:3:0"]), "--dead", "no Pokemon"),  # a box outside the save
        (run(bench=["party:0"]), "--bench", "box members only"),
        (run(bench=["box:1:1"]), "--bench", "no Pokemon"),
        (run(["box:1:0"], ["box:1:0"]), "--bench", "both"),
    ]
    for inputs, key, text in cases:
        with pytest.raises(GauntletSettingsError, match=text) as e:
            resolve_settings(plain, run=inputs, save=save)
        assert e.value.key == key and e.value.group == GROUP
    with pytest.raises(GauntletSettingsError, match="--save") as e:  # no save to check against
        resolve_settings(plain, run=run(["party:0"]))
    assert e.value.key == "--dead/--bench"


def test_run_input_counts(plain):
    for run, key in (
        (RunInput(swaps_left=-2), "--swaps-left"),
        (RunInput(horizon=-1), "--horizon"),
        (RunInput(growth_cap=-2), "--growth-cap"),
    ):
        with pytest.raises(GauntletSettingsError) as e:
            resolve_settings(plain, run=run)
        assert e.value.key == key
    ok = resolve_settings(plain, run=RunInput(swaps_left=-1, horizon=0, growth_cap=-1))
    assert ok.run.swaps_left == -1 and ok.run.growth_cap == -1


# --- field-move cases (B3 fix 3 C6) ---------------------------------------------------------


def _case(moves, orders=(((10,), (11, 12), (13,)),), **over) -> dict:
    d = {
        "moves": list(moves),
        "orders": [[list(f) for f in o] for o in orders],
        "leave_gaps": [[1]] * len(orders),
        "pc_gaps": [2],
        "heal_gaps": [],
        "locked_gaps": [2],
    }
    return {**d, **over}


def case_facts(**over) -> dict:
    """``facts_dict`` with Rock Smash relevant: without it only the first order."""
    d = facts_dict(
        field_moves=["rock_smash"], field_moves_source="rom:test rocks", cases=[_case([])]
    )
    return {**d, **over}


def test_field_move_cases_round_trip_and_follow_the_party():
    f = GauntletFacts.from_dict(case_facts())
    assert GauntletFacts.from_dict(f.to_dict()) == f
    # a party without Rock Smash takes the () case; a move outside field_moves changes nothing
    assert f.case(["cut"]).orders == (((10,), (11, 12), (13,)),)
    top = f.case(["rock_smash", "surf"])
    assert (top.moves, top.orders, top.leave_gaps) == (("rock_smash",), f.orders, f.leave_gaps)
    with pytest.raises(ValueError, match="unknown field moves"):
        f.case(["fly"])
    # a group without field moves: every party takes the top level
    plain = GauntletFacts.from_dict(facts_dict())
    assert plain.case(["cut"]).orders == plain.orders and "cases" not in plain.to_dict()


def test_a_case_with_no_full_sequence_is_refused_by_name():
    empty = {"moves": [], "orders": [], "pc_gaps": [], "heal_gaps": [], "locked_gaps": []}
    f = GauntletFacts.from_dict(case_facts(cases=[{**empty, "leave_gaps": []}]))
    with pytest.raises(ValueError, match="needs Rock Smash"):
        f.case(())


@pytest.mark.parametrize(
    "over, match",
    [
        ({"cases": []}, "every other subset"),
        ({"field_moves": ["rock_smash", "cut"]}, "in that order"),
        ({"field_moves": ["cut", "rock_smash"]}, "every other subset"),
        ({"field_moves_source": ""}, "needs field_moves_source"),
        ({"cases": [_case([], orders=(((10,), (13,)),))]}, "not the group"),
        ({"cases": [_case([], leave_gaps=[[1], [1]])]}, "one entry per order"),
        ({"cases": [_case([], heal_gaps=None)]}, "absent without a live_next"),
        (
            {"cases": [_case([], live_next=[{"fought": [], "live": 1, "next": [[10]]}])]},
            "no row before the first fight at 2",
        ),
    ],
)
def test_field_move_cases_refused(over, match):
    with pytest.raises(ValueError, match=match):
        GauntletFacts.from_dict(case_facts(**over))


# --- skips behind a pushed boulder (B3 fix 3 C5c) ---------------------------------------------


def _skip(**over) -> dict:
    d = {
        "order": [[10], [13]],
        "skipped": [11, 12],
        "header": 284,
        "boulder": 26,
        "boulder_from": [28, 16],
        "boulder_to": [41, 42],
        "hold": [[40, 42]],
    }
    return {**d, **over}


def test_a_skip_round_trips_and_names_the_push():
    f = GauntletFacts.from_dict(facts_dict(skips=[_skip()], skips_source="rom:test skip"))
    assert GauntletFacts.from_dict(f.to_dict()) == f
    (s,) = f.skips
    assert s.order == ((10,), (13,)) and s.skipped == (11, 12)
    text = s.describe()
    assert "(28, 16)" in text and "(41, 42)" in text and "(40, 42)" in text
    assert "Without the push you fight 11, 12" in text
    assert "skips" not in GauntletFacts.from_dict(facts_dict()).to_dict()


@pytest.mark.parametrize(
    "skips, match",
    [
        ([_skip(order=[[13], [10]])], "not an order of the group"),
        ([_skip(order=[[10], [11, 12], [13]])], "not an order of the group"),
        ([_skip(skipped=[11], order=[[10], [12], [13]])], None),
        ([_skip(skipped=[])], "skips no one"),
        ([_skip(hold=[])], "holds no tile"),
        ([_skip(), _skip()], "listed twice"),
        ([_skip(hold=[[40]])], "is not a tile"),
    ],
)
def test_skips_refused(skips, match):
    d = facts_dict(skips=skips, skips_source="rom:test skip")
    if match is None:  # a fight of two that loses one is a fight of one
        assert GauntletFacts.from_dict(d).skips[0].order == ((10,), (12,), (13,))
        return
    with pytest.raises(ValueError, match=match):
        GauntletFacts.from_dict(d)


def test_skips_need_their_source():
    with pytest.raises(ValueError, match="need skips_source"):
        GauntletFacts.from_dict(facts_dict(skips=[_skip()]))
