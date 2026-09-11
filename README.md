# gpu-top

A modern, [Textual](https://textual.textualize.io/)-based TUI for watching GPU
engagement on a node — in the spirit of `nvtop`, built on Rich/Textual, and
designed to grow beyond NVIDIA.

![status](https://img.shields.io/badge/status-early-orange)

## Features

- Live per-GPU panels: utilization, memory, temperature, power, fan, clocks,
  encoder/decoder engagement, and a rolling utilization sparkline.
- Per-GPU process table (PID, name, device memory).
- Adjustable refresh rate (`+` / `-` at runtime).
- Backend-pluggable: NVIDIA today (via NVML/`nvidia-ml-py`), designed so AMD
  (ROCm SMI) and Apple Silicon (`powermetrics`) backends can be added without
  touching the UI.
- A `demo` backend with synthetic data, so you can preview the UI on a
  machine with no GPU at all.

## Install

```bash
pip install gpu-top          # core + demo backend
pip install "gpu-top[nvidia]"  # + NVIDIA support (nvidia-ml-py)
```

## Usage

```bash
gpu-top                  # auto-detects the first available backend
gpu-top --backend demo   # synthetic data, no GPU required
gpu-top --list-backends  # show what's available on this host
gpu-top --interval 0.5   # poll twice a second
```

Keys: `q` quit · `+` / `-` refresh rate · `d` toggle light/dark theme.

## Architecture

```
gpu_top/
├── models.py          # GpuSnapshot / GpuProcess / HostSnapshot dataclasses
├── backends/
│   ├── base.py         # GpuBackend ABC — open() / poll() / close()
│   ├── nvidia.py        # NVML-backed implementation
│   └── demo.py           # synthetic backend for development/preview
├── widgets/
│   ├── meters.py         # Rich-renderable bars/sparklines (no extra deps)
│   └── gpu_panel.py       # the per-GPU card widget
├── app.py               # Textual App: layout + polling loop
└── cli.py                # argparse entry point (`gpu-top`)
```

Adding a vendor means writing one `GpuBackend` subclass that returns
`HostSnapshot`/`GpuSnapshot` objects — the app and widgets are backend-agnostic.

## Development

```bash
pip install -e ".[dev]"
pytest
gpu-top --backend demo   # preview the UI with synthetic data, no GPU required
```

For live Textual devtools (a separate log console while the TUI runs), see
[the Textual docs](https://textual.textualize.io/guide/devtools/) — in short,
run `textual console` in one terminal and `textual run --dev "gpu_top.cli:main"`
in another.

## License

MIT
