"""``gpu-top`` command-line entry point."""

from __future__ import annotations

import argparse
import sys

from gpu_top import __version__
from gpu_top.backends import available_backends, get_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-top",
        description="An nvtop-style TUI for GPU engagement, built on Textual.",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default=None,
        help="Backend to use (default: first available). See --list-backends.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print backends available on this host and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"gpu-top {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_backends:
        found = available_backends()
        if not found:
            print("No GPU backends available on this host.")
            print("(Try `pip install gpu-top[nvidia]` on a machine with an NVIDIA GPU,")
            print(" or run with `--backend demo` to preview the UI.)")
        for backend_cls in found:
            print(backend_cls.name)
        return 0

    if args.backend:
        try:
            backend_cls = get_backend(args.backend)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        found = available_backends()
        if not found:
            print(
                "No GPU backends available on this host.\n"
                "Install support for your vendor (e.g. `pip install gpu-top[nvidia]`),\n"
                "or pass `--backend demo` to preview the UI without a GPU.",
                file=sys.stderr,
            )
            return 1
        backend_cls = found[0]

    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2

    from gpu_top.app import GpuTopApp  # deferred: keep --list-backends/--version fast

    backend = backend_cls()
    app = GpuTopApp(backend=backend, interval=args.interval)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
