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

```bash
pip install gpu-scope
```

The PyPI *distribution* name is `gpu-scope` — PyPI's typosquat-similarity
check blocked `gpu-top` itself as too close to an unrelated existing
package (`gputop`). Nothing else changes because of that: the command you
run is still `gpu-top`, `import gpu_top` still works, and this repo is
still `matplo/gpu-top`. Only the string after `pip install` is different.

One install gets every backend — there's nothing to opt into. NVIDIA support
(`nvidia-ml-py`) ships as a core dependency: it's a small, pure-Python
ctypes wrapper with no platform-specific build, so it installs cleanly
everywhere and simply reports "unavailable" at runtime on a machine with no
NVIDIA driver. AMD and Apple need no extra pip packages at all — they shell
out to system tools (`rocm-smi`, `ioreg`) instead. The one thing `pip`
can't do for you: AMD support only *activates* if ROCm's `rocm-smi` is on
your `PATH` (a system package, install it via your distro/ROCm docs).

`pip install gpu-scope[nvidia]` is still accepted (as a no-op) if you're
used to typing an extra — it doesn't install anything beyond the plain
command above.

Installing from source instead: `pip install "git+https://github.com/matplo/gpu-top.git"`.

## Usage

```bash
gpu-top                  # auto-detects the first available backend
gpu-top --backend demo   # synthetic data, no GPU required
gpu-top --list-backends  # show what's available on this host
gpu-top --interval 0.5   # poll twice a second
```

Running with no GPU present, or forcing `--backend nvidia`/`--backend amd` on a
host without that vendor's tooling, exits with a short explanation and a
`--backend demo` suggestion instead of a driver-library stack trace.

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
