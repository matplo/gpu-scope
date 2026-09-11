"""AppleBackend tests, driven off a real ``ioreg -a`` plist captured on an
Apple Silicon Mac (see conftest fixture below) so parsing is exercised
against actual field shapes rather than a hand-typed guess.
"""

from __future__ import annotations

import plistlib
import subprocess

import pytest

from gpu_scope.backends import apple as apple_module
from gpu_scope.backends.apple import AppleBackend

_SAMPLE_ENTRY = {
    "IORegistryEntryID": 4294968510,
    "model": "Apple M4 Max",
    "gpu-core-count": 40,
    "PerformanceStatistics": {
        "Device Utilization %": 37,
        "In use system memory": 2316926976,
        "Alloc system memory": 25694470144,
    },
}


def _fake_completed(stdout: bytes) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=(), returncode=0, stdout=stdout, stderr=b"")


@pytest.fixture
def stub_ioreg(monkeypatch: pytest.MonkeyPatch):
    plist_bytes = plistlib.dumps([_SAMPLE_ENTRY])

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ioreg":
            return _fake_completed(plist_bytes)
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="51539607552\n", stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(apple_module.subprocess, "run", fake_run)


def test_is_available_true_when_accelerator_present(stub_ioreg, monkeypatch):
    monkeypatch.setattr(apple_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_module.platform, "machine", lambda: "arm64")
    assert AppleBackend.is_available() is True


def test_is_available_false_on_intel_mac(stub_ioreg, monkeypatch):
    monkeypatch.setattr(apple_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(apple_module.platform, "machine", lambda: "x86_64")
    assert AppleBackend.is_available() is False


def test_is_available_false_on_non_darwin(monkeypatch):
    monkeypatch.setattr(apple_module.platform, "system", lambda: "Linux")
    assert AppleBackend.is_available() is False


def test_poll_translates_ioreg_fields(stub_ioreg):
    backend = AppleBackend()
    backend.open()
    snapshot = backend.poll()

    assert snapshot.backend == "apple"
    assert snapshot.error is None
    assert len(snapshot.gpus) == 1

    gpu = snapshot.gpus[0]
    assert gpu.name == "Apple M4 Max"
    assert gpu.vendor == "Apple (40-core)"
    assert gpu.utilization_pct == 37.0
    assert gpu.memory_used_bytes == 2316926976
    assert gpu.memory_total_bytes == 51539607552
    # Fields ioreg simply doesn't expose degrade to None, not a crash.
    assert gpu.temperature_c is None
    assert gpu.power_draw_w is None
    assert gpu.processes == ()


def test_poll_reports_error_when_no_accelerator(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "ioreg":
            return _fake_completed(plistlib.dumps([]))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="0\n", stderr="")

    monkeypatch.setattr(apple_module.subprocess, "run", fake_run)

    backend = AppleBackend()
    backend.open()
    snapshot = backend.poll()
    assert snapshot.gpus == ()
    assert snapshot.error is not None
