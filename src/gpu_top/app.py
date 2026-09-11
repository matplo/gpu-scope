"""The Textual application: layout, polling loop, and key bindings."""

from __future__ import annotations

import asyncio
import math

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.widgets import Footer, Header, Static

from gpu_top.backends.base import GpuBackend
from gpu_top.models import HostSnapshot
from gpu_top.widgets.gpu_panel import GpuPanel

MIN_INTERVAL = 0.2
MAX_INTERVAL = 10.0


class GpuTopApp(App[None]):
    """nvtop-style live GPU dashboard."""

    TITLE = "gpu-top"

    CSS = """
    Screen {
        background: $surface;
    }
    #status {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    #gpu-grid {
        grid-gutter: 1 2;
        grid-rows: auto;
        padding: 0 2 1 2;
        height: auto;
        max-height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("+", "faster", "Refresh faster"),
        ("-", "slower", "Refresh slower"),
        ("d", "toggle_dark", "Toggle theme"),
    ]

    def __init__(self, backend: GpuBackend, interval: float = 1.0) -> None:
        super().__init__()
        self.backend = backend
        self.interval = interval
        self._panels: dict[int, GpuPanel] = {}
        self._poll_timer = None
        self._grid_columns = 1

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="status")
        yield Grid(id="gpu-grid")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            await asyncio.to_thread(self.backend.open)
        except Exception as exc:
            # is_available() is checked before the app is even launched, but
            # that's a point-in-time check (a driver can vanish, a permission
            # can be missing) -- fail here the same friendly way rather than
            # letting a raw traceback escape the TUI.
            self.exit(
                return_code=1,
                message=(
                    f"Could not start the {self.backend.name!r} backend: {exc}\n"
                    "Pass `--backend demo` to preview the UI without a GPU."
                ),
            )
            return
        self._poll_timer = self.set_interval(self.interval, self.poll_once)
        await self.poll_once()

    async def on_unmount(self) -> None:
        await asyncio.to_thread(self.backend.close)

    async def poll_once(self) -> None:
        status = self.query_one("#status", Static)
        try:
            snapshot: HostSnapshot = await asyncio.to_thread(self.backend.poll)
        except Exception as exc:  # backend hiccup shouldn't kill the UI
            status.update(Text(f"poll error: {exc}", style="bold red"))
            return

        self._ensure_panels(snapshot)

        if snapshot.error:
            status.update(Text(f"[{snapshot.backend}] {snapshot.error}", style="bold red"))
        else:
            status.update(
                Text(
                    f"backend: {snapshot.backend}   gpus: {len(snapshot.gpus)}   "
                    f"refresh: {self.interval:.1f}s   (q quit, +/- refresh rate)",
                    style="grey62",
                )
            )

        for gpu in snapshot.gpus:
            self._panels[gpu.index].set_snapshot(gpu)

    def _ensure_panels(self, snapshot: HostSnapshot) -> None:
        grid = self.query_one("#gpu-grid", Grid)

        wanted_columns = max(1, min(3, math.ceil(math.sqrt(max(len(snapshot.gpus), 1)))))
        self._grid_columns = wanted_columns
        grid.styles.grid_size_columns = wanted_columns

        for gpu in snapshot.gpus:
            if gpu.index not in self._panels:
                panel = GpuPanel(gpu.index, id=f"gpu-panel-{gpu.index}")
                self._panels[gpu.index] = panel
                grid.mount(panel)

    def action_faster(self) -> None:
        self._set_interval(self.interval / 1.5)

    def action_slower(self) -> None:
        self._set_interval(self.interval * 1.5)

    def _set_interval(self, new_interval: float) -> None:
        self.interval = max(MIN_INTERVAL, min(MAX_INTERVAL, new_interval))
        if self._poll_timer is not None:
            self._poll_timer.stop()
        self._poll_timer = self.set_interval(self.interval, self.poll_once)
