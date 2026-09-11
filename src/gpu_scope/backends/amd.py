"""AMD backend, built on ``rocm-smi`` (ROCm System Management Interface).

There's no pip-installable AMD equivalent of NVML with the same reach as
``nvidia-ml-py`` -- ROCm ships ``rocm-smi`` as a CLI, so this backend shells
out to ``rocm-smi -a --json`` and parses its output instead.

``rocm-smi``'s JSON key names have drifted across ROCm releases (e.g.
``"GPU use (%)"`` vs ``"GFX Activity"``, temperature sensor naming, ...).
Rather than hard-code one version's exact keys and silently misreport on
another, :func:`_find` does a case-insensitive substring match over the
card's keys for each metric, so minor renames degrade gracefully instead of
breaking. This backend is implemented against the *documented* rocm-smi
JSON shape and has not been exercised against real ROCm hardware -- if a
field looks off on your GPU, the key-matching in ``_find`` is the place to
extend.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

from gpu_scope.backends.base import GpuBackend
from gpu_scope.models import GpuSnapshot, HostSnapshot

_TIMEOUT = 5.0
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _find(card: dict, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
    """First value whose key contains all ``needles`` (case-insensitive).

    ``exclude`` rules out keys containing any of those substrings too --
    needed because e.g. "vram total memory" is itself a substring match of
    "VRAM Total *Used* Memory (B)", not just the total-capacity key.
    """
    wanted = [n.lower() for n in needles]
    unwanted = [n.lower() for n in exclude]
    for key, value in card.items():
        key_lower = key.lower()
        if all(n in key_lower for n in wanted) and not any(n in key_lower for n in unwanted):
            return value
    return None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    match = _NUMBER_RE.search(str(value))
    return float(match.group()) if match else None


def _as_int(value: object) -> int | None:
    f = _as_float(value)
    return int(f) if f is not None else None


def _run_rocm_smi(*args: str) -> dict:
    result = subprocess.run(
        ("rocm-smi", *args, "--json"),
        capture_output=True,
        timeout=_TIMEOUT,
        check=True,
        text=True,
    )
    return json.loads(result.stdout)


class AmdBackend(GpuBackend):
    name = "amd"

    @classmethod
    def is_available(cls) -> bool:
        if shutil.which("rocm-smi") is None:
            return False
        try:
            data = _run_rocm_smi("-a")
        except Exception:
            return False
        return any(k.lower().startswith("card") for k in data)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def poll(self) -> HostSnapshot:
        gpus: list[GpuSnapshot] = []
        error: str | None = None
        try:
            data = _run_rocm_smi("-a")
        except Exception as exc:
            data = {}
            error = f"rocm-smi failed: {exc}"

        for key in sorted(k for k in data if k.lower().startswith("card")):
            card = data[key]
            if not isinstance(card, dict):
                continue
            match = re.search(r"\d+", key)
            index = int(match.group()) if match else len(gpus)
            gpus.append(self._read_device(index, key, card))

        if not gpus and error is None:
            error = "no AMD GPU reported by rocm-smi"

        return HostSnapshot(gpus=tuple(gpus), backend=self.name, error=error)

    def _read_device(self, index: int, card_key: str, card: dict) -> GpuSnapshot:
        name = (
            _find(card, "card", "series")
            or _find(card, "card", "model")
            or _find(card, "device", "name")
            or "AMD GPU"
        )
        uuid = _find(card, "unique", "id") or f"amd-{card_key}"

        utilization_pct = _as_float(_find(card, "gpu", "use", "%") or _find(card, "gfx", "activity"))

        mem_used = _as_int(_find(card, "vram", "used"))
        mem_total = _as_int(_find(card, "vram", "total", exclude=("used",)))

        temperature = _as_float(
            _find(card, "temperature", "edge")
            or _find(card, "temperature", "junction")
            or _find(card, "temperature")
        )

        power = _as_float(_find(card, "average", "power") or _find(card, "power", "w"))

        fan = _as_float(_find(card, "fan", "speed", "%") or _find(card, "fan", "%"))

        clock_sm = _as_int(_find(card, "sclk"))
        clock_mem = _as_int(_find(card, "mclk"))

        return GpuSnapshot(
            index=index,
            uuid=str(uuid),
            name=str(name),
            vendor="AMD",
            utilization_pct=utilization_pct,
            memory_used_bytes=mem_used,
            memory_total_bytes=mem_total,
            temperature_c=temperature,
            power_draw_w=power,
            power_limit_w=None,
            fan_speed_pct=fan,
            clock_sm_mhz=clock_sm,
            clock_mem_mhz=clock_mem,
            encoder_util_pct=None,
            decoder_util_pct=None,
            # rocm-smi's per-process view (--showpidgpus) has no stable JSON
            # shape across versions; left unimplemented rather than guessed.
            processes=(),
        )
