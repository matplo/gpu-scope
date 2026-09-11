"""``gpu-scope`` command-line entry point (also installed as ``gpu-top``)."""

from __future__ import annotations

import argparse
import sys

from gpu_scope import __version__
from gpu_scope.backends import available_backends, get_backend

_NO_BACKEND_HINT = (
    "NVIDIA and Apple Silicon support are built in; AMD needs ROCm's "
    "`rocm-smi` on your PATH. Pass `--backend demo` to preview the UI "
    "without a GPU."
)

_DEMO_SUGGESTION = "Pass `--backend demo` to preview the UI without a GPU."

_UNAVAILABLE_HINTS = {
    "nvidia": "No NVIDIA driver/GPU was detected (or the NVIDIA Management Library couldn't be loaded).",
    "amd": "No AMD GPU was detected via `rocm-smi` -- is ROCm installed and `rocm-smi` on your PATH?",
    "apple": "This isn't an Apple Silicon Mac (or its IOAccelerator registry has nothing to report).",
}


def _unavailable_message(backend_name: str) -> str:
    hint = _UNAVAILABLE_HINTS.get(
        backend_name, f"The {backend_name!r} backend reported itself unavailable on this host."
    )
    return f"{hint}\n{_DEMO_SUGGESTION}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # No explicit prog= -- argparse infers it from argv[0], so --help
        # shows "gpu-scope" or "gpu-top" depending on which alias was run.
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
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_backends:
        found = available_backends()
        if not found:
            print("No GPU backends available on this host.")
            print(_NO_BACKEND_HINT)
        for backend_cls in found:
            print(backend_cls.name)
        return 0

    if args.backend:
        try:
            backend_cls = get_backend(args.backend)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not backend_cls.is_available():
            print(_unavailable_message(backend_cls.name), file=sys.stderr)
            return 1
    else:
        found = available_backends()
        if not found:
            print(
                f"No GPU backends available on this host.\n{_NO_BACKEND_HINT}",
                file=sys.stderr,
            )
            return 1
        backend_cls = found[0]

    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2

    from gpu_scope.app import GpuScopeApp  # deferred: keep --list-backends/--version fast

    backend = backend_cls()
    app = GpuScopeApp(backend=backend, interval=args.interval)
    app.run()
    return app.return_code or 0


if __name__ == "__main__":
    raise SystemExit(main())
