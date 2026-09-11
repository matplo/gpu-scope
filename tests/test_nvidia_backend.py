"""Exercise NvidiaBackend against a stubbed-out ``pynvml`` module.

These tests don't need a real NVIDIA GPU: they monkeypatch the handful of
NVML calls the backend uses so we can verify the translation into
``GpuSnapshot``/``GpuProcess``, and that a field NVML reports as
"not supported" degrades to ``None`` instead of raising.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pynvml = pytest.importorskip("pynvml")

from gpu_top.backends import nvidia as nvidia_module  # noqa: E402
from gpu_top.backends.nvidia import NvidiaBackend  # noqa: E402


class _Mem(SimpleNamespace):
    used: int
    total: int


class _Util(SimpleNamespace):
    gpu: int


class _Proc(SimpleNamespace):
    pid: int
    usedGpuMemory: int


def _not_supported(*_args, **_kwargs):
    raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)


@pytest.fixture
def stub_nvml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetHandleByIndex", lambda i: f"handle-{i}"
    )
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetName", lambda h: "Stub GPU 9000")
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetUUID", lambda h: "GPU-stub-uuid")
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetUtilizationRates", lambda h: _Util(gpu=42)
    )
    monkeypatch.setattr(
        nvidia_module.pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda h: _Mem(used=2 * 1024**3, total=8 * 1024**3),
    )
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetTemperature", lambda h, s: 65)
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetPowerUsage", lambda h: 120_000)
    # Exercise the "unsupported field" path explicitly.
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetEnforcedPowerLimit", _not_supported
    )
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetFanSpeed", lambda h: 55)
    monkeypatch.setattr(nvidia_module.pynvml, "nvmlDeviceGetClockInfo", lambda h, c: 1500)
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetEncoderUtilization", lambda h: (0, 0)
    )
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetDecoderUtilization", lambda h: (0, 0)
    )
    monkeypatch.setattr(
        nvidia_module.pynvml,
        "nvmlDeviceGetComputeRunningProcesses",
        lambda h: [_Proc(pid=4242, usedGpuMemory=512 * 1024**2)],
    )
    monkeypatch.setattr(
        nvidia_module.pynvml, "nvmlDeviceGetGraphicsRunningProcesses", lambda h: []
    )
    monkeypatch.setattr(nvidia_module, "_process_name", lambda pid: "stub-process")
    return nvidia_module.pynvml


def test_poll_translates_nvml_fields(stub_nvml):
    backend = NvidiaBackend()
    backend.open()
    snapshot = backend.poll()
    backend.close()

    assert snapshot.backend == "nvidia"
    assert snapshot.error is None
    assert len(snapshot.gpus) == 1

    gpu = snapshot.gpus[0]
    assert gpu.name == "Stub GPU 9000"
    assert gpu.utilization_pct == 42.0
    assert gpu.memory_used_bytes == 2 * 1024**3
    assert gpu.memory_used_pct == 25.0
    assert gpu.temperature_c == 65.0
    assert gpu.power_draw_w == 120.0
    # Unsupported on this "GPU" -> None, not an exception.
    assert gpu.power_limit_w is None
    assert gpu.fan_speed_pct == 55.0

    assert len(gpu.processes) == 1
    proc = gpu.processes[0]
    assert proc.pid == 4242
    assert proc.name == "stub-process"
    assert proc.memory_bytes == 512 * 1024**2


def test_is_available_false_without_driver(monkeypatch: pytest.MonkeyPatch):
    def _raise():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)

    monkeypatch.setattr(nvidia_module.pynvml, "nvmlInit", _raise)
    assert NvidiaBackend.is_available() is False
