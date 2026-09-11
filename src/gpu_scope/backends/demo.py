"""Synthetic backend for developing/demoing the UI without a real GPU.

Not auto-discovered by :func:`gpu_scope.backends.available_backends` — select
it explicitly with ``gpu-scope --backend demo`` (or ``gpu-top --backend demo``).
"""

from __future__ import annotations

import math
import random
import time

from gpu_scope.backends.base import GpuBackend
from gpu_scope.models import GpuProcess, GpuSnapshot, HostSnapshot

_FAKE_PROC_NAMES = ("python", "pytorch_worker", "ffmpeg", "blender", "llama.cpp")


class DemoBackend(GpuBackend):
    name = "demo"

    def __init__(self, gpu_count: int = 2) -> None:
        self._gpu_count = gpu_count
        self._t0 = time.monotonic()
        self._rng = random.Random(0)

    @classmethod
    def is_available(cls) -> bool:
        return True

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def poll(self) -> HostSnapshot:
        t = time.monotonic() - self._t0
        gpus = tuple(self._read_device(i, t) for i in range(self._gpu_count))
        return HostSnapshot(gpus=gpus, backend=self.name)

    def _read_device(self, index: int, t: float) -> GpuSnapshot:
        phase = index * 2.1
        util = 50 + 45 * math.sin(t / 6 + phase) + self._rng.uniform(-4, 4)
        util = max(0.0, min(100.0, util))

        total = 24 * 1024**3
        used = int(total * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(t / 14 + phase))))

        temp = 40 + util * 0.4 + self._rng.uniform(-1, 1)
        power_limit = 350.0
        power = power_limit * (0.15 + 0.008 * util)

        processes = ()
        if util > 25:
            n = 1 + int(util // 40)
            processes = tuple(
                GpuProcess(
                    pid=1000 + index * 10 + i,
                    name=self._rng.choice(_FAKE_PROC_NAMES),
                    memory_bytes=used // max(n, 1),
                )
                for i in range(n)
            )

        return GpuSnapshot(
            index=index,
            uuid=f"demo-gpu-{index}",
            name=f"Demo GPU {index} (synthetic)",
            vendor="Demo",
            utilization_pct=util,
            memory_used_bytes=used,
            memory_total_bytes=total,
            temperature_c=temp,
            power_draw_w=power,
            power_limit_w=power_limit,
            fan_speed_pct=max(0.0, min(100.0, util * 0.9)),
            clock_sm_mhz=int(1200 + util * 6),
            clock_mem_mhz=9500,
            encoder_util_pct=max(0.0, util - 60) if index == 0 else 0.0,
            decoder_util_pct=0.0,
            processes=processes,
        )
