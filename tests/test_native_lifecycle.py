import sys
from types import SimpleNamespace

from app import Lab


def test_native_initialization_failure_signals_shutdown(monkeypatch):
    class BrokenScene:
        def __init__(self):
            raise RuntimeError("test initialization failure")

    monkeypatch.setitem(sys.modules, "scene", SimpleNamespace(DesktopScene=BrokenScene))
    lab = Lab("http://127.0.0.1:8911", native=True)
    lab.run()
    assert lab.snapshot()["phase"] == "error"
    assert not lab.snapshot()["busy"]
    assert lab.closed.is_set()
