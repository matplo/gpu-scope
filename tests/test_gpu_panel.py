"""Regression tests for GpuPanel's rendering at various panel widths.

These exist because Rich's Table.grid will happily *wrap* a cell instead of
respecting the width we computed for it, if the other cells in the same
column leave less room than expected (e.g. a long "12.4GiB/24.0GiB" value
squeezing the meter-bar column) -- which used to split a bar's closing
bracket onto its own line. We render at a handful of realistic widths and
assert no line is narrower-than-content orphan text.
"""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from gpu_top.app import GpuTopApp
from gpu_top.backends.demo import DemoBackend


@pytest.mark.parametrize("gpu_count,size", [(1, (100, 30)), (2, (80, 30)), (8, (160, 60))])
def test_panel_renders_without_orphaned_wrap(gpu_count: int, size: tuple[int, int]):
    async def run() -> list[str]:
        app = GpuTopApp(backend=DemoBackend(gpu_count=gpu_count), interval=0.1)
        async with app.run_test(size=size) as pilot:
            for _ in range(3):
                await asyncio.sleep(0.05)
                await pilot.pause()

            lines: list[str] = []
            for panel in app._panels.values():
                assert panel.snapshot is not None
                renderable = panel._build_renderable(panel.snapshot)
                console = Console(width=panel.size.width or 40)
                with console.capture() as cap:
                    console.print(renderable)
                lines.extend(cap.get().splitlines())
            return lines

    lines = asyncio.run(run())
    assert lines, "expected at least one rendered line"
    for line in lines:
        stripped = line.strip()
        # A wrapped meter bar leaves a line that's just the closing
        # bracket (optionally with trailing spaces) on its own.
        assert stripped != "]", f"meter bar wrapped onto its own line: {line!r}"
