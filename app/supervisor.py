from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _port(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    public_port = _port("PORT", 9000)
    internal_port = _port("SAM3_INTERNAL_PORT", 9001)
    if public_port == internal_port:
        raise ValueError("PORT and SAM3_INTERNAL_PORT must differ")

    unified_python = Path(sys.executable)
    if not unified_python.is_file():
        raise RuntimeError(f"Unified Python runtime not found: {unified_python}")

    base_environment = os.environ.copy()
    segmenter_environment = dict(base_environment)
    segmenter_environment["PORT"] = str(internal_port)
    gateway_environment = dict(base_environment)
    gateway_environment["PORT"] = str(public_port)

    processes: list[subprocess.Popen[bytes]] = []

    def handle_signal(_signum: int, _frame: object) -> None:
        _terminate(processes)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        processes.append(
            subprocess.Popen(
                [str(unified_python), "-m", "segmenter.serve"],
                env=segmenter_environment,
            )
        )
        processes.append(
            subprocess.Popen(
                [sys.executable, "-m", "app.serve"],
                env=gateway_environment,
            )
        )
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return return_code or 1
            time.sleep(0.2)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    raise SystemExit(main())
