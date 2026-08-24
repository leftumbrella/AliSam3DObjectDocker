"""Concurrency contract for the cross-process GPU lock."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from shared.gpu_lock import InterProcessGpuLock


class InterProcessGpuLockTests(unittest.TestCase):
    def test_separate_lock_instances_serialize_gpu_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            barrier = threading.Barrier(2)
            counter_lock = threading.Lock()
            active = 0
            peak = 0

            def worker() -> None:
                nonlocal active, peak
                barrier.wait()
                with InterProcessGpuLock(path).acquire():
                    with counter_lock:
                        active += 1
                        peak = max(peak, active)
                    time.sleep(0.03)
                    with counter_lock:
                        active -= 1

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(peak, 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_path_must_be_absolute_and_not_root(self) -> None:
        for path in (Path("gpu.lock"), Path("/")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                InterProcessGpuLock(path)


if __name__ == "__main__":
    unittest.main()
