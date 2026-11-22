"""Root conftest: the local test gate (docs/process.md section 5, tools/ops/ramgate.py).

No engine tests on the PC: after collection a run is allowed only when every test is ROM-only or
`nuzline.engine` was never imported; it then waits for free RAM > 6 GB + weight. The pod sets
NUZLINE_ON_POD=1 (tools/ops/day/pod/unit.sh, setup.sh), which skips the gate.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent / "tools" / "ops" / "ramgate.py"
_spec = importlib.util.spec_from_file_location("nuzline_ramgate", _PATH)
ramgate = importlib.util.module_from_spec(_spec)
sys.modules["nuzline_ramgate"] = ramgate
_spec.loader.exec_module(ramgate)


_BLOCKER = pytest.StashKey()


def pytest_unconfigure(config):
    blocker = config.stash.get(_BLOCKER, None)
    if blocker is not None:
        ramgate.remove_blocker(blocker)


# trylast: -m and -k have deselected by then, so the gate sees exactly what will run.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if hasattr(config, "workerinput"):  # an xdist worker: the controller gated the run
        return
    verdict, msg = ramgate.pytest_gate(
        items,
        collect_only=bool(getattr(config.option, "collectonly", False)),
        say=lambda m: print(m, file=sys.stderr, flush=True),
    )
    if verdict == "REFUSE":
        pytest.exit(msg, returncode=3)
    if verdict == "TIMEOUT":
        pytest.exit(msg, returncode=4)
    if verdict == "GO":
        # Engine-free run: an import of nuzline.engine at run time fails loudly (until unconfigure).
        if not config.option.collectonly and ramgate.blocker_wanted(items, os.environ):
            config.stash[_BLOCKER] = ramgate.install_blocker()
        # Child pytest processes of a test (pytester subprocesses) inherit this and skip the gate.
        os.environ[ramgate.ENV_DONE] = "1"
