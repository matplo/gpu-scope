"""Apple Silicon backend, built on ``ioreg``'s ``IOAccelerator`` registry.

There's no public NVML-equivalent SDK for Apple GPUs. What every community
tool in this space (asitop, mactop, macmon, ...) actually uses is the same
place Activity Monitor's GPU tab gets its numbers: the ``PerformanceStatistics``
dictionary under each ``IOAccelerator``/``AGXAccelerator`` entry in the IORegistry,
readable via ``ioreg -a`` (a plist) with no elevated privileges.

That gives us live utilization and a GPU-attributed memory figure. It does
*not* give us temperature, fan speed, clock frequency, or per-process
attribution -- those require ``powermetrics``, which macOS restricts to root.
Rather than prompt for sudo from inside a TUI, this backend reports those
fields as unavailable (``None``) and leaves room to add an opt-in
``powermetrics`` path later for anyone already running as root.

Since the GPU shares unified memory with the CPU, "GPU memory total" is
reported as total system RAM (``hw.memsize``) -- there's no separate VRAM
pool to size against.
"""

from __future__ import annotations

import platform
import plistlib
import subprocess

from gpu_top.backends.base import GpuBackend
from gpu_top.models import GpuSnapshot, HostSnapshot

_IOREG_CMD = ("ioreg", "-r", "-d", "1", "-c", "IOAccelerator", "-a")
_TIMEOUT = 3.0


def _run_ioreg() -> list[dict]:
    result = subprocess.run(
        _IOREG_CMD, capture_output=True, timeout=_TIMEOUT, check=True
    )
    data = plistlib.loads(result.stdout)
    return list(data) if isinstance(data, list) else []


def _total_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            ("sysctl", "-n", "hw.memsize"),
            capture_output=True,
            timeout=_TIMEOUT,
            check=True,
            text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


class AppleBackend(GpuBackend):
    name = "apple"

    def __init__(self) -> None:
        self._total_memory: int | None = None

    @classmethod
    def is_available(cls) -> bool:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return False
        try:
            entries = _run_ioreg()
        except Exception:
            return False
        return any("PerformanceStatistics" in e for e in entries)

    def open(self) -> None:
        self._total_memory = _total_memory_bytes()

    def close(self) -> None:
        pass

    def poll(self) -> HostSnapshot:
        if self._total_memory is None:
            self._total_memory = _total_memory_bytes()

        gpus: list[GpuSnapshot] = []
        error: str | None = None
        try:
            entries = _run_ioreg()
        except Exception as exc:
            entries = []
            error = f"ioreg failed: {exc}"

        index = 0
        for entry in entries:
            perf = entry.get("PerformanceStatistics")
            if not isinstance(perf, dict):
                continue
            gpus.append(self._read_device(index, entry, perf))
            index += 1

        if not gpus and error is None:
            error = "no Apple GPU accelerator found in IORegistry"

        return HostSnapshot(gpus=tuple(gpus), backend=self.name, error=error)

    def _read_device(self, index: int, entry: dict, perf: dict) -> GpuSnapshot:
        name = entry.get("model") or "Apple GPU"
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")

        registry_id = entry.get("IORegistryEntryID")
        uuid = f"apple-gpu-{registry_id}" if registry_id is not None else f"apple-gpu-{index}"

        utilization_pct = perf.get("Device Utilization %")

        mem_used = perf.get("In use system memory")
        if not isinstance(mem_used, int):
            mem_used = None

        core_count = entry.get("gpu-core-count")
        vendor = f"Apple ({core_count}-core)" if core_count else "Apple"

        return GpuSnapshot(
            index=index,
            uuid=str(uuid),
            name=str(name),
            vendor=vendor,
            utilization_pct=float(utilization_pct) if isinstance(utilization_pct, (int, float)) else None,
            memory_used_bytes=mem_used,
            memory_total_bytes=self._total_memory,
            # Unavailable without root (powermetrics) or private SMC access.
            temperature_c=None,
            power_draw_w=None,
            power_limit_w=None,
            fan_speed_pct=None,
            clock_sm_mhz=None,
            clock_mem_mhz=None,
            encoder_util_pct=None,
            decoder_util_pct=None,
            # Per-process GPU attribution needs `powermetrics --show-process-gpu`
            # (root-only); not implemented here.
            processes=(),
        )
