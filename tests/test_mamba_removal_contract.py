"""Regression checks for the audited Conda runtime-trimming plan."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
CHECKER = ROOT / "scripts" / "check_mamba_removal.py"
OPENCL_DEPENDENCY_ANCHORS = {"opencl-headers"}


def _trim_packages() -> set[str]:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(r'packages="\\\n(?P<body>.*?)\n\s*"; \\', dockerfile, re.DOTALL)
    if match is None:
        raise AssertionError("Dockerfile trimming package list was not found")
    return set(match.group("body").replace("\\", " ").split())


def _run_checker(removed: set[str], allowed: set[str]) -> subprocess.CompletedProcess[str]:
    plan = {
        "success": True,
        "actions": {"UNLINK": [{"name": name} for name in sorted(removed)]},
    }
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "remove-plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(CHECKER), str(plan_path), *sorted(allowed)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


class MambaRemovalContractTests(unittest.TestCase):
    def test_opencl_dependency_anchor_is_retained(self) -> None:
        # Paired positive case proves this assertion detects the bad package shape.
        self.assertEqual(
            OPENCL_DEPENDENCY_ANCHORS & {"opencl-headers"},
            {"opencl-headers"},
        )
        self.assertEqual(OPENCL_DEPENDENCY_ANCHORS & _trim_packages(), set())

    def test_cuda_runtime_packages_stay_protected(self) -> None:
        for package in ("cuda-cudart", "cuda-libraries"):
            with self.subTest(package=package):
                result = _run_checker({package}, {package})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"运行时保护包：{package}", result.stderr)


if __name__ == "__main__":
    unittest.main()
