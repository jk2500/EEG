import os
import sys
import types


def _install_stub():
    stub = types.ModuleType("mne")
    stub.set_log_level = lambda *args, **kwargs: None
    io_namespace = types.SimpleNamespace()
    io_namespace.read_raw_brainvision = lambda *args, **kwargs: types.SimpleNamespace()
    stub.io = io_namespace
    sys.modules["mne"] = stub


_FORCE_STUB = os.environ.get("EEG_FORCE_MNE_STUB")

if _FORCE_STUB == "1" and "mne" not in sys.modules:
    _install_stub()
