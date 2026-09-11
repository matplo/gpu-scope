"""The per-GPU panel: an nvtop-style card with meters, clocks, and processes."""

from __future__ import annotations

from collections import deque

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from gpu_scope.models import GpuSnapshot
from gpu_scope.widgets.meters import human_bytes, level_color, meter_bar, sparkline

HISTORY_LEN = 90


class GpuPanel(Static):
    """A single card showing one GPU's live stats. Update via ``set_snapshot``."""

    DEFAULT_CSS = """
    GpuPanel {
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        padding: 0 1;
        height: auto;
        min-width: 46;
    }
    GpuPanel.high-load {
        border: round $error;
    }
    """

    snapshot: reactive[GpuSnapshot | None] = reactive(None)

    def __init__(self, index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index = index
        self._util_history: deque[float | None] = deque([None] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self._mem_history: deque[float | None] = deque([None] * HISTORY_LEN, maxlen=HISTORY_LEN)

    def set_snapshot(self, snap: GpuSnapshot) -> None:
        self._util_history.append(snap.utilization_pct)
        self._mem_history.append(snap.memory_used_pct)
        self.snapshot = snap  # triggers watch_snapshot -> re-render

    def watch_snapshot(self, snap: GpuSnapshot | None) -> None:
        if snap is None:
            self.update("waiting for data…")
            return

        self.border_title = f"GPU {snap.index} · {snap.name}"
        self.set_class((snap.utilization_pct or 0) >= 85, "high-load")
        self.update(self._build_renderable(snap))

    def _build_renderable(self, snap: GpuSnapshot) -> Group:
        # self.size is already the *content* box (border/padding excluded).
        width = self.size.width or 40
        value_col_width = 17  # fits "999.9GiB/999.9GiB"
        # label(5) + gutter + value column + gutter, with a little slack.
        bar_width = max(min(width - 5 - value_col_width - 4, 30), 10)

        grid = Table.grid(padding=(0, 1))
        grid.add_column(justify="left", width=5, no_wrap=True)
        grid.add_column(no_wrap=True, overflow="crop")
        grid.add_column(justify="right", width=value_col_width, no_wrap=True, overflow="crop")

        grid.add_row(
            Text("Util", style="bold"),
            meter_bar(snap.utilization_pct, bar_width),
            self._pct_text(snap.utilization_pct),
        )
        grid.add_row(
            Text("Mem", style="bold"),
            meter_bar(snap.memory_used_pct, bar_width),
            Text(
                f"{human_bytes(snap.memory_used_bytes)}/{human_bytes(snap.memory_total_bytes)}",
                style="grey78",
            ),
        )

        info = Text(overflow="fold")
        info.append(self._field("Temp", f"{snap.temperature_c:.0f}°C" if snap.temperature_c is not None else "—"))
        info.append("  ")
        info.append(self._field("Fan", f"{snap.fan_speed_pct:.0f}%" if snap.fan_speed_pct is not None else "—"))
        info.append("  ")
        info.append(self._field("Pwr", self._power_text(snap)))
        info.append("\n")
        info.append(self._field("SM", f"{snap.clock_sm_mhz}MHz" if snap.clock_sm_mhz else "—"))
        info.append("  ")
        info.append(self._field("Mem", f"{snap.clock_mem_mhz}MHz" if snap.clock_mem_mhz else "—"))
        info.append("  ")
        info.append(
            self._field(
                "Enc/Dec",
                f"{snap.encoder_util_pct:.0f}%/{snap.decoder_util_pct:.0f}%"
                if snap.encoder_util_pct is not None and snap.decoder_util_pct is not None
                else "—",
            )
        )

        spark_width = max(width - 5 - 2, 10)
        spark = Table.grid(padding=(0, 1))
        spark.add_column(width=5, no_wrap=True)
        spark.add_column(no_wrap=True, overflow="crop")
        spark.add_row(Text("Hist", style="bold"), sparkline(list(self._util_history), spark_width))

        parts = [grid, info, spark]
        proc_table = self._process_table(snap)
        if proc_table is not None:
            parts.append(proc_table)

        return Group(*parts)

    @staticmethod
    def _field(label: str, value: str) -> Text:
        t = Text()
        t.append(f"{label} ", style="grey58")
        t.append(value, style="white")
        return t

    @staticmethod
    def _power_text(snap: GpuSnapshot) -> str:
        if snap.power_draw_w is None:
            return "—"
        if snap.power_limit_w:
            return f"{snap.power_draw_w:.0f}/{snap.power_limit_w:.0f}W"
        return f"{snap.power_draw_w:.0f}W"

    @staticmethod
    def _pct_text(pct: float | None) -> Text:
        if pct is None:
            return Text("—", style="grey50")
        return Text(f"{pct:.0f}%", style=level_color(pct))

    def _process_table(self, snap: GpuSnapshot) -> Table | None:
        if not snap.processes:
            return None
        table = Table(
            box=None,
            show_header=True,
            header_style="bold grey70",
            pad_edge=False,
            padding=(0, 1, 0, 0),
        )
        table.add_column("PID", justify="right", style="grey78", width=7)
        table.add_column("PROCESS", overflow="ellipsis", no_wrap=True)
        table.add_column("MEM", justify="right", style="grey78", width=9)
        for proc in snap.processes[:6]:
            table.add_row(str(proc.pid), proc.name, human_bytes(proc.memory_bytes))
        if len(snap.processes) > 6:
            table.add_row("…", f"+{len(snap.processes) - 6} more", "")
        return table

    def on_resize(self) -> None:
        if self.snapshot is not None:
            self.update(self._build_renderable(self.snapshot))
