"""tools/ops/ramgate.py: weight lookup, id selection, the decision, the pytest gate. No real RAM
read, no real waiting: free RAM, the clock and the sleep are injected."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ramgate_under_test", ROOT / "tools/ops/ramgate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["ramgate_under_test"] = rg
_spec.loader.exec_module(rg)

TABLE = {
    "tests/a": 3.0,
    "tests/a/test_x.py": 5.0,
    "tests/a/test_x.py::test_one": 9.0,
    "tests/b/test_y.py": 4.0,
    "tests/heavy.py": 30.0,
}


class Clock:
    """A fake clock whose sleep advances it."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def item(nodeid, *fixtures, direct=(), indirect=()):
    """A fake item. `direct`/`indirect`: parametrize argnames (pytest gives a direct one a pseudo
    fixture def named get_direct_param_fixture_func, an indirect one its real fixture def)."""

    def get_direct_param_fixture_func():
        pass

    def real_fixture():
        pass

    names = [*fixtures, *direct, *indirect]
    defs = {n: [SimpleNamespace(func=get_direct_param_fixture_func)] for n in direct}
    defs.update({n: [SimpleNamespace(func=real_fixture)] for n in indirect})
    params = {n: 1 for n in (*direct, *indirect)}
    return SimpleNamespace(
        nodeid=nodeid,
        fixturenames=["request", *names],
        callspec=SimpleNamespace(params=params) if params else None,
        _fixtureinfo=SimpleNamespace(name2fixturedefs=defs),
    )


ENGINE = ["os", "jax", "nuzline.engine.step"]
NO_ENGINE = ["os", "jax", "nuzline.data.tables"]
ROM = item("tests/tools/rom/test_nds.py::test_a", "vanilla_bytes")
PLAIN = item("tests/a/test_x.py::test_b", "tmp_path")  # weight 5
ENG = item("tests/a/test_e.py::test_c")  # weight 3


def gate(items, free=100.0, modules=NO_ENGINE, env=None, rom_table=None, **kw):
    clock = kw.pop("clock", None) or Clock()
    lines = kw.pop("lines", [])
    return rg.pytest_gate(
        items,
        modules=modules,
        env={} if env is None else env,
        table=TABLE,
        rom_table={} if rom_table is None else rom_table,
        free=free if callable(free) else (lambda: free),
        sleep=clock.sleep,
        clock=clock.now,
        say=lines.append,
        **kw,
    )


def test_weight_exact_and_prefixes():
    assert rg.weight("tests/a/test_x.py", TABLE) == 5.0
    assert rg.weight("tests/a/test_x.py::test_two", TABLE) == 5.0
    assert rg.weight("tests/a/test_x.py::test_one[p]", TABLE) == 9.0
    assert rg.weight("tests/a/other.py", TABLE) == 3.0
    assert rg.weight("tests/b/test_y.py[x]", TABLE) == 4.0


def test_weight_longest_wins_and_no_match():
    assert rg.weight("tests/a/test_x.py::test_one", TABLE) == 9.0
    assert rg.weight("tests/ab/test_z.py", TABLE) == 20.0  # "tests/a" must end at a separator
    assert rg.weight("tests/zzz.py", TABLE) == 20.0


def test_weight_backslashes():
    assert rg.weight("tests\\a\\test_x.py", TABLE) == 5.0


def test_load_weights(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("# c\ntests/a.py 3 run1\nbad x\n\ntests/b.py 4.5\n", encoding="utf-8")
    assert rg.load_weights(f) == {"tests/a.py": 3.0, "tests/b.py": 4.5}
    assert rg.load_weights() and all(v > 0 for v in rg.load_weights().values())


def test_selected_ids_drop_options_and_values():
    args = [
        "-m",
        "not slow and not fullset",
        "-q",
        "-k",
        "foo",
        "-p",
        "no:cacheprovider",
        "--durations",
        "40",
        "-o",
        "addopts=",
        "--maxfail=2",
        "tests\\a\\test_x.py::test_one",
        "./tests/b/",
    ]
    assert rg.selected_ids(args, ROOT) == ["tests/a/test_x.py::test_one", "tests/b"]


def test_selected_ids_absolute_and_invocation_dir():
    assert rg.selected_ids([str(ROOT / "tests" / "a.py")], ROOT) == ["tests/a.py"]
    assert rg.selected_ids(["a.py"], ROOT, ROOT / "tests") == ["tests/a.py"]
    assert rg.selected_ids(["-x", "-q"], ROOT) == []


def test_run_weight_heaviest_and_whole_tree():
    assert rg.run_weight(["tests/a/test_x.py", "tests/b/test_y.py"], TABLE) == 5.0
    assert rg.run_weight(["tests/a/test_x.py", "tests/new.py"], TABLE) == 20.0
    assert rg.run_weight([], TABLE) == 30.0


def test_decide_boundaries():
    assert rg.decide(7.99, 100) == "GO"
    assert rg.decide(8, 100) == "POD"
    assert rg.decide(20, 0) == "POD"
    assert rg.decide(4, 10) == "WAIT"  # free == 6 + w waits
    assert rg.decide(4, 10.01) == "GO"
    assert rg.decide(7.99, 13.99) == "WAIT"


def test_decide_rom_gate_allows_heavy_but_still_waits():
    assert rg.decide(30, 100, rom_gate=True) == "GO"
    assert rg.decide(30, 36, rom_gate=True) == "WAIT"


def test_is_rom_only_by_fixture_and_by_path():
    assert rg.is_rom_only(ROM)
    assert rg.is_rom_only(item("tests/x.py::t", "tmp_path", "pkcalc"))
    assert not rg.is_rom_only(PLAIN)
    assert not rg.is_rom_only(ENG)
    path_test = (
        "tests/kaizo/test_sheet_unmapped.py::test_ally_reasons_name_the_partner_rows_starter"
    )
    assert rg.is_rom_only(item(path_test + "[a]"))
    for f in ("rom_inputs", "species", "kaizo_save_probe_path", "docs", "symbols"):
        assert f in rg.ROM_FIXTURES


def test_engine_loaded_prefix_only():
    assert rg.engine_loaded(["nuzline.engine"]) and rg.engine_loaded(ENGINE)
    assert not rg.engine_loaded(NO_ENGINE + ["nuzline.engineering", "jax"])


def test_gate_engine_free_run_goes():
    assert gate([PLAIN, ENG], modules=NO_ENGINE)[0] == "GO"  # nothing imported the engine


def test_gate_all_rom_only_goes_even_with_engine_loaded():
    assert gate([ROM, ROM], modules=ENGINE)[0] == "GO"


def test_gate_mixed_run_is_refused_naming_the_quick_lane():
    items = [ROM] + [item(f"tests/e/test_{n}.py::t") for n in range(7)]
    verdict, msg = gate(items, modules=ENGINE)
    assert verdict == "REFUSE"
    assert "quick lane" in msg and "q-<stream>-<sha7>" in msg and "section 5" in msg
    assert "tests/e/test_0.py::t" in msg and "tests/e/test_4.py::t" in msg
    assert "tests/e/test_5.py::t" not in msg and "and 2 more" in msg
    assert "test_nds" not in msg  # the ROM-only test is not listed


def test_gate_override_allows_but_still_waits():
    env = {rg.ENV_ROM_GATE: "1"}
    assert gate([ENG], modules=ENGINE, env=env)[0] == "GO"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_ROM_GATE: "0"})[0] == "REFUSE"
    verdict, _ = gate([ENG], free=7.0, modules=ENGINE, env={**env, rg.ENV_TIMEOUT: "1"})
    assert verdict == "TIMEOUT"  # weight 5 needs > 11 free


def test_gate_collect_only_allowed_without_waiting():
    clock = Clock()
    assert gate([ENG], free=0.0, modules=ENGINE, collect_only=True, clock=clock)[0] == "GO"
    assert clock.sleeps == []


def test_gate_skipped_on_pod_and_in_nested_runs():
    assert gate([ENG], modules=ENGINE, env={rg.ENV_ON_POD: "1"})[0] == "SKIP"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_DONE: "1"})[0] == "SKIP"
    assert gate([ENG], modules=ENGINE, env={rg.ENV_NESTED: "x::y (call)"})[0] == "SKIP"


def test_gate_weight_is_the_heaviest_collected_item_not_the_args():
    unknown = item("tests/zzz.py::t", "tmp_path")  # no key = 20
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([PLAIN, unknown], free=25.9, env=env)[0] == "TIMEOUT"
    assert gate([PLAIN, unknown], free=26.1)[0] == "GO"
    assert gate([PLAIN], free=11.5)[0] == "GO"  # weight 5 -> needs > 11
    assert gate([PLAIN], free=10.9, env=env)[0] == "TIMEOUT"


def test_rom_only_items_weigh_fourteen_until_measured():
    light = item("tests/b/test_y.py::t[x]", "kaizo_inputs")  # pod table says 4
    assert rg.ROM_UNMEASURED_GB == 14.0
    assert rg.item_weight(light, TABLE, {}) == 14.0
    assert rg.item_weight(item("tests/zzz.py::t", "docs"), TABLE, {}) == 14.0  # no pod key
    assert rg.item_weight(item("tests/heavy.py::t", "docs"), TABLE, {}) == 30.0
    assert rg.item_weight(item("tests/b/test_y.py::t", "tmp_path"), TABLE, {}) == 4.0
    measured = {"tests/b/test_y.py": 7.0, "tests/b/test_y.py::t": 9.0}
    assert rg.item_weight(light, TABLE, measured) == 9.0  # longest prefix wins
    assert rg.item_weight(item("tests/b/test_y.py::u", "docs"), TABLE, measured) == 7.0
    env = {rg.ENV_TIMEOUT: "0"}
    assert gate([light], free=20.1, rom_table={}, env=env)[0] == "GO"  # needs > 20
    assert gate([light], free=19.9, rom_table={}, env=env)[0] == "TIMEOUT"
    assert gate([light], free=13.1, rom_table={"tests/b/test_y.py": 7.0}, env=env)[0] == "GO"


def test_load_rom_weights(tmp_path):
    f = tmp_path / "rom_weights.txt"
    f.write_text("# c\ntests/a.py 7 local-measure-abc1234-20261122\n", encoding="utf-8")
    assert rg.load_rom_weights(f) == {"tests/a.py": 7.0}
    assert rg.load_rom_weights(tmp_path / "missing.txt") == {}
    assert rg.ROM_WEIGHTS.parent.name == "day" and rg.ROM_WEIGHTS.name == "rom_weights.txt"


def test_species_as_a_parametrize_name_is_not_the_fixture():
    direct = item("tests/data/test_learnsets.py::t[1]", "species", direct=("species",))
    indirect = item("tests/saves/test_to_spec.py::t[1]", "species", indirect=("species",))
    plain_fixture = item("tests/x.py::t", "species")
    assert not rg.is_rom_only(direct)
    assert rg.is_rom_only(indirect)
    assert rg.is_rom_only(plain_fixture)
    assert rg.is_rom_only(item("tests/x.py::t[a]", "tmp_path", "species", direct=("which",)))


def test_getfixturevalue_save_export_ids_are_rom_only_for_every_param():
    base = "tests/test_cli.py::test_kaizo_save_export_checks_kaizo_tables"
    for which in ("kaizo_save_pre_roark_path", "kaizo_save_path"):
        assert rg.is_rom_only(item(f"{base}[{which}]", "tmp_path", "request"))


def test_gate_waits_then_goes():
    clock = Clock()
    seq = iter([5.0, 9.0, 11.5])
    lines = []
    verdict, _ = gate([PLAIN], free=lambda: next(seq), clock=clock, lines=lines)
    assert verdict == "GO"  # weight 5: needs > 11
    assert clock.sleeps == [60.0, 60.0]
    assert len(lines) == 2 and "need > 11" in lines[0]


def test_gate_wait_timeout_exits():
    clock = Clock()
    verdict, msg = gate([PLAIN], free=7.0, env={rg.ENV_TIMEOUT: "3"}, clock=clock)
    assert verdict == "TIMEOUT"
    assert clock.sleeps == [60.0] * 3
    assert "3 min" in msg


def test_cli_pod_rom_gate_and_build(monkeypatch):
    monkeypatch.delenv(rg.ENV_ROM_GATE, raising=False)
    monkeypatch.setattr(rg, "load_weights", lambda: TABLE)
    out = []

    def run(argv, free=100.0):
        return rg.main(argv, free=lambda: free, sleep=lambda s: None, say=out.append)

    assert run(["tests/a/test_x.py"]) == 0 and out[-1] == "GO 5"
    assert run(["tests/heavy.py"]) == 3 and out[-1] == "POD 30"
    assert run(["tests/heavy.py", "--rom-gate"]) == 0 and out[-1] == "GO 30"
    assert run(["tools.kaizo.x", "--weight", "9"]) == 3 and out[-1] == "POD 9"
    # a build never goes to the pod: weight 20 by default, only the RAM wait applies
    assert run(["--build"]) == 0 and out[-1] == "GO 20"
    assert run(["--build", "--weight", "12"]) == 0 and out[-1] == "GO 12"


def test_cli_weight_without_a_value_is_a_usage_error():
    out = []
    assert rg.main(["tests/a.py", "--weight"], say=out.append) == 2
    assert rg.main(["--weight", "x"], say=out.append) == 2
    assert "usage" in out[0]


def test_cli_honours_the_timeout_env(monkeypatch):
    monkeypatch.setenv(rg.ENV_TIMEOUT, "2")
    clock = Clock()
    out = []
    code = rg.main(
        ["--build"], free=lambda: 1.0, sleep=clock.sleep, clock=clock.now, say=out.append
    )
    assert code == 4 and clock.sleeps == [60.0, 60.0] and out[-1] == "TIMEOUT 20"


def test_cli_build_waits_for_ram():
    seq = iter([10.0, 25.0, 26.5])
    sleeps = []
    out = []
    code = rg.main(["--build"], free=lambda: next(seq), sleep=sleeps.append, say=out.append)
    assert code == 0 and sleeps == [60.0, 60.0] and out[-1] == "GO 20"
    assert not any(m.startswith("POD") for m in out)


@pytest.mark.parametrize("name", ["unit.sh", "setup.sh"])
def test_pod_entry_points_export_the_pod_marker(name):
    text = (ROOT / "tools/ops/day/pod" / name).read_text(encoding="utf-8")
    assert rg.ENV_ON_POD + "=1" in text


def test_blocker_blocks_engine_imports_only():
    b = rg.EngineBlocker()
    for name in ("nuzline.engine", "nuzline.engine.x", "nuzline.engine.step.y"):
        with pytest.raises(ImportError, match="quick lane"):
            b.find_spec(name)
    for name in ("nuzline.engineering", "nuzline.data.tables", "jax", "os"):
        assert b.find_spec(name) is None


def test_blocker_in_a_fake_run_stops_the_import_and_is_removed(monkeypatch):
    monkeypatch.delenv(rg.ENV_ENGINE_BLOCKED, raising=False)  # restored at teardown
    path = []
    b = rg.install_blocker(path)
    assert path == [b]
    # the real import machinery, with the finder first on a copy of the path
    saved = sys.meta_path
    sys.meta_path = [b, *saved]
    try:
        with pytest.raises(ImportError, match="quick lane"):
            importlib.import_module("nuzline.engine.fake_ramgate_probe")
        assert "nuzline.engine.fake_ramgate_probe" not in sys.modules
        assert importlib.import_module("json")
    finally:
        sys.meta_path = saved
    rg.remove_blocker(b, path)
    assert path == []
    rg.remove_blocker(b, path)  # twice is harmless


def test_blocker_wanted_only_for_engine_free_runs_without_override():
    assert rg.blocker_wanted([PLAIN, ROM], {})
    assert rg.blocker_wanted([PLAIN], {rg.ENV_ROM_GATE: "0"})
    assert not rg.blocker_wanted([ROM, ROM], {})
    assert not rg.blocker_wanted([PLAIN], {rg.ENV_ROM_GATE: "1"})


def test_blocker_sets_the_child_env_and_removal_unsets_it(monkeypatch):
    monkeypatch.delenv(rg.ENV_ENGINE_BLOCKED, raising=False)
    path = []
    b = rg.install_blocker(path)
    assert rg.os.environ[rg.ENV_ENGINE_BLOCKED] == "1"
    rg.remove_blocker(b, path)
    assert rg.ENV_ENGINE_BLOCKED not in rg.os.environ


def test_engine_package_refuses_a_child_process_under_the_env():
    import os
    import subprocess

    env = {**os.environ, rg.ENV_ENGINE_BLOCKED: "1", "PYTHONPATH": str(ROOT / "src")}
    bad = subprocess.run(
        [sys.executable, "-c", "import nuzline.engine"],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert bad.returncode != 0 and "quick lane" in bad.stderr and "ImportError" in bad.stderr
    env.pop(rg.ENV_ENGINE_BLOCKED)
    ok = subprocess.run(
        [sys.executable, "-c", "import nuzline.engine"],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert ok.returncode == 0, ok.stderr


def _body_fetching_a_rom_fixture(request):
    return request.getfixturevalue("kaizo_save_path")


def _body_fetching_a_plain_fixture(request):
    return request.getfixturevalue("tmp_path"), request.getfixturevalue(request.param)


def test_a_literal_getfixturevalue_of_a_rom_fixture_is_rom_only():
    fetching = item("tests/x.py::t", "request")
    fetching.function = _body_fetching_a_rom_fixture
    plain = item("tests/x.py::t", "request")
    plain.function = _body_fetching_a_plain_fixture
    assert rg.is_rom_only(fetching)
    assert not rg.is_rom_only(plain)  # a name held in a variable needs ROM_ONLY_IDS
    plain.function = print  # no source for a builtin: not ROM-only, no crash
    assert not rg.is_rom_only(plain)


def test_the_dynamic_getfixturevalue_tests_are_listed():
    for name in (
        "tests/test_cli.py::test_kaizo_save_export_checks_kaizo_tables",
        "tests/test_cli.py::test_kaizo_roark_coverage_on_the_saves_is_exact",
        "tests/kaizo/test_roark_ai_audit.py::test_roark_ai_against_the_pool",
        "tests/kaizo/test_roark_widths.py::test_roark_pairs_fit_the_width_cap",
    ):
        assert name in rg.ROM_ONLY_IDS
        file, _, test = name.partition("::")
        assert f"def {test}(" in (ROOT / file).read_text(encoding="utf-8")


def _body_with_a_conditional_fetch(request, g):
    return request.getfixturevalue("tables" if g else "kaizo_save_path")


def test_a_conditional_getfixturevalue_counts_every_literal():
    it = item("tests/x.py::t", "request")
    it.function = _body_with_a_conditional_fetch
    assert rg.is_rom_only(it)


def test_timeout_message_names_the_env_and_the_weights_files():
    verdict, msg = gate([PLAIN], free=1.0, env={rg.ENV_TIMEOUT: "0"})
    assert verdict == "TIMEOUT"
    assert rg.ENV_TIMEOUT in msg and "weights.txt" in msg and "rom_weights.txt" in msg


def test_refusal_says_ids_never_a_file_path_and_rom_gate_is_a_last_resort():
    _, msg = gate([ENG], modules=ENGINE)
    assert "never a file path" in msg and "--rom-ids" in msg and "last resort" in msg


def test_rom_ids_group_by_file_and_print_ready_commands():
    a1 = item("tests/test_cli.py::test_a[x y]", "kaizo_save_path")
    a2 = item("tests/test_cli.py::test_engine", "tmp_path")
    a3 = item("tests/test_cli.py::test_b", "docs")
    b1 = item("tests/kaizo/test_z.py::test_c", "pkcalc")
    by_file = rg.rom_ids_by_file([a1, a2, a3, b1])
    assert by_file == {
        "tests/test_cli.py": ["tests/test_cli.py::test_a[x y]", "tests/test_cli.py::test_b"],
        "tests/kaizo/test_z.py": ["tests/kaizo/test_z.py::test_c"],
    }
    lines = rg.rom_commands(by_file)
    assert lines[0] == (
        "python -m pytest -p no:cacheprovider -q 'tests/test_cli.py::test_a[x y]' tests/test_cli.py::test_b"
    )
    assert len(lines) == 2 and "test_engine" not in lines[0]


def test_no_wait_env_skips_the_ram_wait_but_not_the_refusal():
    clock = Clock()
    env = {rg.ENV_NO_WAIT: "1"}
    assert gate([PLAIN], free=0.0, env=env, clock=clock)[0] == "GO"
    assert clock.sleeps == []
    assert gate([ENG], free=0.0, modules=ENGINE, env=env)[0] == "REFUSE"
    assert gate([PLAIN], free=0.0, env={rg.ENV_NO_WAIT: "0", rg.ENV_TIMEOUT: "0"})[0] == "TIMEOUT"
