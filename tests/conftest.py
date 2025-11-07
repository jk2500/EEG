import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ensure_mne_stub():
    if "mne" in sys.modules:
        return

    stub = types.ModuleType("mne")
    stub.set_log_level = lambda *args, **kwargs: None

    io_namespace = types.SimpleNamespace()
    io_namespace.read_raw_brainvision = lambda *args, **kwargs: types.SimpleNamespace()
    stub.io = io_namespace

    sys.modules["mne"] = stub


_ensure_mne_stub()
