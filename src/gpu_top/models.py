"""Data model shared by every backend.

Backends translate whatever a vendor's tooling reports into these plain
dataclasses so the UI never has to know whether a value came from NVML,
``nvidia-smi``, ``rocm-smi``, or Apple's ``powermetrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GpuProcess:
    """A single process holding a handle on a GPU."""

    pid: int
    name: str
    memory_bytes: int | None = None
    """Device memory attributed to this process, if the backend can report it."""


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    """A single point-in-time reading for one physical GPU."""

    index: int
    uuid: str
    name: str
    vendor: str
    """Short vendor tag, e.g. ``"NVIDIA"``, ``"AMD"``, ``"Apple"``."""

    utilization_pct: float | None
    """Overall (SM/compute) engagement, 0-100. ``None`` if unavailable."""

    memory_used_bytes: int | None
    memory_total_bytes: int | None

    temperature_c: float | None = None
    power_draw_w: float | None = None
    power_limit_w: float | None = None
    fan_speed_pct: float | None = None
    clock_sm_mhz: int | None = None
    clock_mem_mhz: int | None = None

    encoder_util_pct: float | None = None
    decoder_util_pct: float | None = None

    processes: tuple[GpuProcess, ...] = field(default_factory=tuple)

    @property
    def memory_used_pct(self) -> float | None:
        if not self.memory_used_bytes or not self.memory_total_bytes:
            return None
        return 100.0 * self.memory_used_bytes / self.memory_total_bytes


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """A single poll across every GPU visible on the node."""

    gpus: tuple[GpuSnapshot, ...]
    backend: str
    error: str | None = None
    """Set when the backend could not be queried this poll (device lost, etc.)."""
