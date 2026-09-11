"""AmdBackend tests against hand-built rocm-smi JSON.

No AMD/ROCm hardware was available to capture real output, so these use
JSON shaped after rocm-smi's documented ``--json`` schema, in two "eras" of
key naming, to exercise the case-insensitive :func:`~gpu_top.backends.amd._find`
matching this backend relies on to survive that drift.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from gpu_top.backends import amd as amd_module
from gpu_top.backends.amd import AmdBackend, _find

_ROCM_SMI_OUTPUT = {
    "card0": {
        "Card series": "Instinct MI210",
        "Unique ID": "0x1234567890abcdef",
        "Temperature (Sensor edge) (C)": "45.0",
        "Temperature (Sensor junction) (C)": "50.0",
        "Average Graphics Package Power (W)": "120.0",
        "GPU use (%)": "62",
        "GPU memory use (%)": "40",
        "VRAM Total Memory (B)": "68702699520",
        "VRAM Total Used Memory (B)": "27481079808",
        "fan speed (%)": "35",
        "sclk clock speed:": "(1500Mhz)",
        "mclk clock speed:": "(1600Mhz)",
    }
}


def _fake_run(cmd, **kwargs):
    assert cmd[0] == "rocm-smi"
    return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout=json.dumps(_ROCM_SMI_OUTPUT), stderr=""
    )


@pytest.fixture
def stub_rocm_smi(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(amd_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(amd_module.shutil, "which", lambda name: "/usr/bin/rocm-smi")


def test_find_is_case_and_order_insensitive():
    card = {"GPU use (%)": "62"}
    assert _find(card, "gpu", "use") == "62"
    assert _find(card, "GPU", "USE") == "62"
    assert _find(card, "nonexistent") is None


def test_find_exclude_disambiguates_total_from_used():
    # "VRAM Total Used Memory (B)" is itself a substring match of
    # ("vram", "total", "memory") -- exclude must rule it out regardless
    # of which key rocm-smi happens to list first.
    card = {
        "VRAM Total Used Memory (B)": "used-value",
        "VRAM Total Memory (B)": "total-value",
    }
    assert _find(card, "vram", "total", exclude=("used",)) == "total-value"

    reordered = {
        "VRAM Total Memory (B)": "total-value",
        "VRAM Total Used Memory (B)": "used-value",
    }
    assert _find(reordered, "vram", "total", exclude=("used",)) == "total-value"


def test_is_available(stub_rocm_smi):
    assert AmdBackend.is_available() is True


def test_is_available_false_without_binary(monkeypatch):
    monkeypatch.setattr(amd_module.shutil, "which", lambda name: None)
    assert AmdBackend.is_available() is False


def test_poll_translates_rocm_smi_fields(stub_rocm_smi):
    backend = AmdBackend()
    backend.open()
    snapshot = backend.poll()

    assert snapshot.backend == "amd"
    assert snapshot.error is None
    assert len(snapshot.gpus) == 1

    gpu = snapshot.gpus[0]
    assert gpu.index == 0
    assert gpu.name == "Instinct MI210"
    assert gpu.vendor == "AMD"
    assert gpu.utilization_pct == 62.0
    assert gpu.memory_used_bytes == 27481079808
    assert gpu.memory_total_bytes == 68702699520
    assert gpu.temperature_c == 45.0
    assert gpu.power_draw_w == 120.0
    assert gpu.fan_speed_pct == 35.0
    assert gpu.clock_sm_mhz == 1500
    assert gpu.clock_mem_mhz == 1600
    assert gpu.processes == ()


def test_poll_reports_error_when_no_cards(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(amd_module.subprocess, "run", fake_run)

    backend = AmdBackend()
    snapshot = backend.poll()
    assert snapshot.gpus == ()
    assert snapshot.error is not None
