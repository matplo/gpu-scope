"""Backend discovery.

Add a new vendor by writing a :class:`~gpu_top.backends.base.GpuBackend`
subclass and listing it in ``_ALL_BACKENDS`` below — the app, CLI, and tests
never need to change.
"""

from __future__ import annotations

from gpu_top.backends.amd import AmdBackend
from gpu_top.backends.apple import AppleBackend
from gpu_top.backends.base import GpuBackend
from gpu_top.backends.demo import DemoBackend
from gpu_top.backends.nvidia import NvidiaBackend

#: Auto-discovered, in priority order. Backends that only make sense when
#: explicitly requested (e.g. DemoBackend) are left out of this list and
#: looked up by name instead.
_ALL_BACKENDS: tuple[type[GpuBackend], ...] = (NvidiaBackend, AmdBackend, AppleBackend)

_BY_NAME: dict[str, type[GpuBackend]] = {b.name: b for b in _ALL_BACKENDS}
_BY_NAME[DemoBackend.name] = DemoBackend


def available_backends() -> list[type[GpuBackend]]:
    """Backends whose ``is_available()`` reports true on this host."""
    return [b for b in _ALL_BACKENDS if b.is_available()]


def get_backend(name: str) -> type[GpuBackend]:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_NAME))
        raise ValueError(f"Unknown backend {name!r}. Known backends: {known}") from exc


__all__ = [
    "GpuBackend",
    "NvidiaBackend",
    "AmdBackend",
    "AppleBackend",
    "DemoBackend",
    "available_backends",
    "get_backend",
]
