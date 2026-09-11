"""Backend protocol: anything that can enumerate and poll GPUs."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gpu_scope.models import HostSnapshot


class GpuBackend(ABC):
    """One data source for GPU telemetry (a vendor SDK, or a CLI wrapper)."""

    #: Short, stable identifier shown in the UI and used on the CLI (``--backend``).
    name: str = "base"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Cheaply check whether this backend can run on the current host.

        Must not raise; return ``False`` on any failure (missing library,
        missing driver, no devices, wrong platform, ...).
        """

    @abstractmethod
    def open(self) -> None:
        """Acquire any handles/connections needed before polling."""

    @abstractmethod
    def close(self) -> None:
        """Release handles/connections. Must be safe to call multiple times."""

    @abstractmethod
    def poll(self) -> HostSnapshot:
        """Return a fresh snapshot of every GPU this backend manages."""

    def __enter__(self) -> "GpuBackend":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
