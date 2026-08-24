from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class InterProcessGpuLock:
    """Serialize GPU work across the isolated SAM3 and SAM3D processes."""

    def __init__(self, path: Path) -> None:
        if not path.is_absolute() or path == Path("/"):
            raise ValueError("GPU lock path must be a non-root absolute path")
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
