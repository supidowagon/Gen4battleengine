"""Root conftest: the local test gate (docs/process.md section 5, tools/ops/ramgate.py).

No engine tests on the PC: after collection a run is allowed only when every test is ROM-only or
`nuzline.engine` was never imported; it then waits for free RAM > 6 GB + weight and takes the
machine lock (one local run at a time), held until pytest_unconfigure. The pod sets
NUZLINE_ON_POD=1 (tools/ops/day/pod/unit.sh, setup.sh, quick.sh), which skips the gate on Linux.
Exit codes 13 (refused), 14 (wait timed out), 16 (free RAM unreadable): apart from pytest's 0-5.
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
_LOCK = pytest.StashKey()


def pytest_configure(config):
    msg = ramgate.xdist_refusal(config.option, os.environ)
    if msg:
        pytest.exit(msg, returncode=ramgate.EXIT_REFUSE)


def pytest_unconfigure(config):
    blocker = config.stash.get(_BLOCKER, None)
    if blocker is not None:
        ramgate.remove_blocker(blocker)
    lock = config.stash.get(_LOCK, None)
    if lock is not None:
        lock.release()
    if os.environ.get(ramgate.ENV_DONE) == str(os.getpid()):
        os.environ.pop(ramgate.ENV_DONE)


# trylast: -m and -k have deselected by then, so the gate sees exactly what will run.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    lock = ramgate.MachineLock()
    config.stash[_LOCK] = lock  # released at unconfigure, whatever the verdict
    verdict, msg = ramgate.pytest_gate(
        items,
        collect_only=bool(getattr(config.option, "collectonly", False)),
        lock=lock,
        say=lambda m: print(m, file=sys.stderr, flush=True),
    )
    if verdict == "REFUSE":
        pytest.exit(msg, returncode=ramgate.EXIT_REFUSE)
    if verdict == "TIMEOUT":
        pytest.exit(msg, returncode=ramgate.EXIT_TIMEOUT)
    if verdict == "ERROR":
        pytest.exit(msg, returncode=ramgate.EXIT_NO_RAM_READ)
    if verdict == "GO":
        # Engine-free run: an import of nuzline.engine at run time fails loudly (until unconfigure).
        if not config.option.collectonly and ramgate.blocker_wanted(items, os.environ):
            config.stash[_BLOCKER] = ramgate.install_blocker()
        # Child pytest processes of a test (pytester subprocesses) inherit this and skip the gate.
        os.environ[ramgate.ENV_DONE] = str(os.getpid())
