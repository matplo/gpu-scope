# gpu-top

A modern, [Textual](https://textual.textualize.io/)-based TUI for watching GPU
engagement on a node — in the spirit of `nvtop`, built on Rich/Textual, and
covering NVIDIA, AMD, and Apple Silicon.

![status](https://img.shields.io/badge/status-early-orange)

## Features

- Live per-GPU panels: utilization, memory, temperature, power, fan, clocks,
  encoder/decoder engagement, and a rolling utilization sparkline.
- Per-GPU process table (PID, name, device memory) — NVIDIA only for now.
- Adjustable refresh rate (`+` / `-` at runtime).
- Backend-pluggable, three vendors today:
  - **NVIDIA** — via NVML (`nvidia-ml-py`). Full field coverage, including
    per-process memory.
  - **AMD** — via the `rocm-smi` CLI (ROCm must be installed). Utilization,
    memory, temperature, power, fan, clocks; no per-process view yet.
    Parses `rocm-smi`'s JSON defensively (case-insensitive key matching) to
    tolerate key-name drift across ROCm versions, but hasn't been run
    against real ROCm hardware — if a field looks off, please open an issue
    with your `rocm-smi -a --json` output.
  - **Apple Silicon** — via `ioreg`'s `IOAccelerator` registry (no sudo
    required, macOS only). Utilization and a GPU-attributed memory figure
    (against total unified memory, there's no separate VRAM pool). Apple
    doesn't expose temperature/power/fan/clocks/per-process GPU stats
    without `powermetrics`, which requires root — those fields show as `—`.
- A `demo` backend with synthetic data, so you can preview the UI on a
  machine with no GPU at all.

Adding a vendor means writing one `GpuBackend` subclass that returns
`HostSnapshot`/`GpuSnapshot` objects — the app and widgets are backend-agnostic.

## Install

Not yet published to PyPI — install straight from GitHub:

```bash
pip install "git+https://github.com/matplo/gpu-top.git"                  # core + demo backend + AMD/Apple
pip install "gpu-top[nvidia] @ git+https://github.com/matplo/gpu-top.git"  # + NVIDIA support (nvidia-ml-py)
```

(Once published to PyPI, this becomes `pip install gpu-top` / `pip install "gpu-top[nvidia]"`.)

The AMD backend needs ROCm's `rocm-smi` on your `PATH` (a system package,
not something `pip` installs); the Apple backend needs nothing beyond macOS
itself on Apple Silicon.

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
│   ├── amd.py             # rocm-smi (CLI + JSON) implementation
│   ├── apple.py            # ioreg IOAccelerator implementation
│   └── demo.py               # synthetic backend for development/preview
├── widgets/
│   ├── meters.py         # Rich-renderable bars/sparklines (no extra deps)
│   └── gpu_panel.py       # the per-GPU card widget
├── app.py               # Textual App: layout + polling loop
└── cli.py                # argparse entry point (`gpu-top`)
```

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
