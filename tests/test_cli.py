"""CLI behavior when a requested/only backend isn't usable on this host.

Forcing an unavailable backend used to let a raw driver-library traceback
(NVMLError, etc.) escape all the way to the terminal. These pin down the
friendly-message-and-clean-exit-code behavior instead.
"""

from __future__ import annotations

import pytest

from gpu_top import cli
from gpu_top.backends.base import GpuBackend
from gpu_top.models import HostSnapshot


class _NeverAvailable(GpuBackend):
    name = "nevergonnahappen"

    @classmethod
    def is_available(cls) -> bool:
        return False

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def poll(self) -> HostSnapshot:
        return HostSnapshot(gpus=(), backend=self.name)


@pytest.fixture
def register_never_available(monkeypatch: pytest.MonkeyPatch):
    import gpu_top.backends as backends_module

    monkeypatch.setitem(backends_module._BY_NAME, _NeverAvailable.name, _NeverAvailable)


def test_explicit_unavailable_backend_exits_cleanly(register_never_available, capsys):
    code = cli.main(["--backend", "nevergonnahappen"])
    assert code == 1
    err = capsys.readouterr().err
    assert "nevergonnahappen" in err or "demo" in err.lower()
    assert "--backend demo" in err


def test_unknown_backend_name_exits_with_usage_error(capsys):
    code = cli.main(["--backend", "totally-not-a-backend"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Unknown backend" in err


def test_unavailable_hint_mentions_demo_for_known_vendors():
    for name in ("nvidia", "amd", "apple"):
        msg = cli._unavailable_message(name)
        assert "--backend demo" in msg


def test_negative_interval_rejected(capsys):
    code = cli.main(["--backend", "demo", "--interval", "-1"])
    assert code == 2
    assert "--interval" in capsys.readouterr().err
