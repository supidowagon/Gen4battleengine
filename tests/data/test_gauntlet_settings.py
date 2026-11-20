"""The gauntlet option registry, precedence and tracked facts (GT-1; synthetic groups and
settings only, no user choice). The loader half is GT-3's."""

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
    Layer,
    Origin,
    SettingsError,
    Swap,
    missing_transitions,
    resolve,
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
        "partner": {"trainer": 99, "join_before": [1, 2, 3], "source": "rom:test partner"},
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
        if o.kind in (LEVEL, SET) or o.scope == RUN or o.name == "partner_from":
            assert o.default == o.values[0], o.name
    assert OPTIONS["order"].default == "worst" and OPTIONS["partner_from"].default == "worst"
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
        Layer("x", gauntlet={"partner_from": 4}),
        Layer("x", gaps={3: {"heal": "bag"}}),  # n = 3: gaps 1..2
        Layer("x", gaps={1: {"order": "worst"}}),  # not per gap
        Layer("x", gaps={1: {"swap_cap": -1}}),
    ]
    for layer in bad:
        with pytest.raises(SettingsError):
            resolve(facts, order, layer)
    ok = resolve(facts, order, Layer("x", gauntlet={"partner_from": 2}))
    assert ok["partner_from"].value == 2
    no_partner = GauntletFacts.from_dict(facts_dict(partner=None))
    with pytest.raises(SettingsError):
        resolve(no_partner, order, Layer("x", gauntlet={"partner_from": "worst"}))
    with pytest.raises(ValueError):
        resolve(facts, ((1,), (2,)))


def test_facts_round_trip_and_helpers(facts):
    assert GauntletFacts.from_dict(facts.to_dict()) == facts
    assert facts.to_dict() == facts_dict()
    assert facts.partner.trainer == 99 and facts.partner.join_before == (1, 2, 3)
    o0, o1 = facts.orders
    assert facts.n_fights(o0) == 3 and facts.fight_ids(o0) == (10, 11, 13)
    assert facts.fight_ids(o1) == (11, 10, 13)
    assert facts.leave_gaps_of(o1) == ()
    with pytest.raises(ValueError):
        GauntletFacts.from_dict(facts_dict(leave_gaps=[[1]]))


def test_walkout_gaps(facts):
    o0, o1 = facts.orders
    # escape holds and no partner set: every gap is a walk-out gap
    assert facts.walkout_gaps(o0, None) == (1, 2)
    # partner joins before fight 1: set at every gap, only the leave gap remains
    assert facts.walkout_gaps(o0, 1) == (1,)
    assert facts.walkout_gaps(o1, 1) == ()
    # joins before fight 2: gap 1 has no partner yet
    assert facts.walkout_gaps(o1, 2) == (1,)
    no_escape = GauntletFacts.from_dict(facts_dict(escape=False, partner=None))
    assert no_escape.walkout_gaps(o0, None) == (1,)
    with pytest.raises(ValueError):
        no_escape.walkout_gaps(o0, 1)


def test_trainer_gauntlet_is_group_and_fight_id():
    t = Trainer(1, "A", "C", 0, 0, 7, (), (), "none", "single", gauntlet=("Test Cave", 11))
    assert t.gauntlet == ("Test Cave", 11) and t.gauntlet_facts is None
