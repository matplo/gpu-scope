"""Small Rich-renderable helpers: colored meter bars and sparklines.

Kept dependency-free (no extra libs beyond Rich) so they're easy to reuse
from anywhere that wants an nvtop-style bar or trend line.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text

_BLOCKS = " ▏▎▍▌▋▊▉█"
_SPARK_LEVELS = "▁▂▃▄▅▆▇█"

# Thresholds mirror nvtop's low/medium/high coloring.
_LOW = 60.0
_MEDIUM = 85.0


def level_color(pct: float | None) -> str:
    """Green/yellow/red style for a 0-100 percentage."""
    if pct is None:
        return "grey50"
    if pct < _LOW:
        return "green3"
    if pct < _MEDIUM:
        return "gold3"
    return "red3"


def meter_bar(pct: float | None, width: int, *, color: str | None = None) -> Text:
    """A single-line ``[███████░░░]`` bar, filled proportionally to ``pct``."""
    color = color or level_color(pct)
    inner_width = max(width - 2, 1)
    if pct is None:
        text = Text("[", style="grey50")
        text.append("·" * inner_width, style="grey50")
        text.append("]", style="grey50")
        return text

    clamped = max(0.0, min(100.0, pct))
    filled_cells = clamped / 100.0 * inner_width
    full_cells = int(filled_cells)
    remainder = filled_cells - full_cells
    partial_index = int(remainder * (len(_BLOCKS) - 1))

    bar_str = _BLOCKS[-1] * full_cells
    if full_cells < inner_width:
        bar_str += _BLOCKS[partial_index]
        bar_str += " " * (inner_width - full_cells - 1)

    text = Text("[", style="grey50")
    text.append(bar_str, style=color)
    text.append("]", style="grey50")
    return text


def sparkline(values: Sequence[float | None], width: int) -> Text:
    """A compact unicode trend line for the most recent ``width`` samples."""
    sample = [v for v in values[-width:]]
    if not sample:
        return Text(_SPARK_LEVELS[0] * width, style="grey50")

    numeric = [v for v in sample if v is not None]
    lo, hi = (0.0, 100.0) if not numeric else (0.0, max(100.0, max(numeric)))
    span = hi - lo or 1.0

    text = Text()
    pad = width - len(sample)
    if pad > 0:
        text.append(_SPARK_LEVELS[0] * pad, style="grey35")

    for v in sample:
        if v is None:
            text.append(_SPARK_LEVELS[0], style="grey35")
            continue
        idx = int((v - lo) / span * (len(_SPARK_LEVELS) - 1))
        idx = max(0, min(len(_SPARK_LEVELS) - 1, idx))
        text.append(_SPARK_LEVELS[idx], style=level_color(v))
    return text


def human_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"
