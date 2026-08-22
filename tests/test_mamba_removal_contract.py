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
AUDITED_OPENCL_CASCADE = {"cuda-libraries", "cuda-opencl", "ocl-icd"}


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
    def test_dockerfile_explicitly_lists_audited_opencl_cascade(self) -> None:
        self.assertTrue(AUDITED_OPENCL_CASCADE <= _trim_packages())

    def test_audited_opencl_cascade_is_allowed_by_guard(self) -> None:
        result = _run_checker(AUDITED_OPENCL_CASCADE, AUDITED_OPENCL_CASCADE)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_concrete_cuda_runtime_stays_protected(self) -> None:
        result = _run_checker({"cuda-cudart"}, {"cuda-cudart"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("运行时保护包：cuda-cudart", result.stderr)


if __name__ == "__main__":
    unittest.main()
