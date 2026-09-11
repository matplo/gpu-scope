"""NVIDIA backend, built on NVML via the ``nvidia-ml-py`` bindings.

NVML (not ``nvidia-smi`` text scraping) is used because it's the same
library ``nvidia-smi`` itself calls into, is much cheaper to poll at a
sub-second interval, and returns typed values instead of formatted strings.
"""

from __future__ import annotations

from gpu_top.backends.base import GpuBackend
from gpu_top.models import GpuProcess, GpuSnapshot, HostSnapshot

try:
    import pynvml

    _NVML_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised when extra not installed
    pynvml = None  # type: ignore[assignment]
    _NVML_IMPORT_ERROR = exc


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:
        return f"pid {pid}"


def _try(fn, *args):
    """Call an NVML getter, returning ``None`` on any NVML error.

    A lot of fields (power limit, fan speed, encoder util, ...) are simply
    unsupported on some GPUs/drivers/virtualization setups; NVML signals
    that with ``NVML_ERROR_NOT_SUPPORTED`` rather than omitting the field.
    """
    try:
        return fn(*args)
    except pynvml.NVMLError:
        return None


class NvidiaBackend(GpuBackend):
    name = "nvidia"

    def __init__(self) -> None:
        self._open = False

    @classmethod
    def is_available(cls) -> bool:
        if pynvml is None:
            return False
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def open(self) -> None:
        if self._open:
            return
        if pynvml is None:
            raise RuntimeError(
                "nvidia-ml-py is not installed (it's a core gpu-top dependency; "
                "try `pip install --force-reinstall gpu-top`)"
            ) from _NVML_IMPORT_ERROR
        pynvml.nvmlInit()
        self._open = True

    def close(self) -> None:
        if not self._open:
            return
        try:
            pynvml.nvmlShutdown()
        finally:
            self._open = False

    def poll(self) -> HostSnapshot:
        if not self._open:
            self.open()

        gpus: list[GpuSnapshot] = []
        error: str | None = None
        try:
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                gpus.append(self._read_device(index))
        except pynvml.NVMLError as exc:
            error = str(exc)

        return HostSnapshot(gpus=tuple(gpus), backend=self.name, error=error)

    def _read_device(self, index: int) -> GpuSnapshot:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        name = _try(pynvml.nvmlDeviceGetName, handle) or "Unknown NVIDIA GPU"
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")

        uuid = _try(pynvml.nvmlDeviceGetUUID, handle) or f"gpu-{index}"
        if isinstance(uuid, bytes):
            uuid = uuid.decode("utf-8", "replace")

        util = _try(pynvml.nvmlDeviceGetUtilizationRates, handle)
        utilization_pct = float(util.gpu) if util is not None else None

        mem = _try(pynvml.nvmlDeviceGetMemoryInfo, handle)
        mem_used = int(mem.used) if mem is not None else None
        mem_total = int(mem.total) if mem is not None else None

        temperature = _try(
            pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
        )

        power_draw = _try(pynvml.nvmlDeviceGetPowerUsage, handle)
        power_draw_w = power_draw / 1000.0 if power_draw is not None else None

        power_limit = _try(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
        power_limit_w = power_limit / 1000.0 if power_limit is not None else None

        fan = _try(pynvml.nvmlDeviceGetFanSpeed, handle)

        clock_sm = _try(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM)
        clock_mem = _try(pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM)

        encoder = _try(pynvml.nvmlDeviceGetEncoderUtilization, handle)
        encoder_pct = float(encoder[0]) if encoder is not None else None
        decoder = _try(pynvml.nvmlDeviceGetDecoderUtilization, handle)
        decoder_pct = float(decoder[0]) if decoder is not None else None

        processes = tuple(self._read_processes(handle))

        return GpuSnapshot(
            index=index,
            uuid=uuid,
            name=name,
            vendor="NVIDIA",
            utilization_pct=utilization_pct,
            memory_used_bytes=mem_used,
            memory_total_bytes=mem_total,
            temperature_c=float(temperature) if temperature is not None else None,
            power_draw_w=power_draw_w,
            power_limit_w=power_limit_w,
            fan_speed_pct=float(fan) if fan is not None else None,
            clock_sm_mhz=int(clock_sm) if clock_sm is not None else None,
            clock_mem_mhz=int(clock_mem) if clock_mem is not None else None,
            encoder_util_pct=encoder_pct,
            decoder_util_pct=decoder_pct,
            processes=processes,
        )

    def _read_processes(self, handle) -> list[GpuProcess]:
        by_pid: dict[int, int | None] = {}

        for getter in (
            pynvml.nvmlDeviceGetComputeRunningProcesses,
            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            for proc in _try(getter, handle) or []:
                used = getattr(proc, "usedGpuMemory", None)
                if isinstance(used, int) and used < 0:  # NVML "N/A" sentinel
                    used = None
                # Prefer a known memory value over an earlier None from the
                # other list (a process can show up in both).
                if by_pid.get(proc.pid) is None:
                    by_pid[proc.pid] = used

        return [
            GpuProcess(pid=pid, name=_process_name(pid), memory_bytes=used)
            for pid, used in sorted(by_pid.items())
        ]
