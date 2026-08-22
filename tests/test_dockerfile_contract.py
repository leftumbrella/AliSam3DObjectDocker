"""Regression checks for the FC image build contract."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
OFFLINE_PATCHER = ROOT / "scripts" / "patch_offline_runtime.py"


def _run_blocks(dockerfile: str) -> list[str]:
    """Return shell-form RUN instructions with continuations joined."""
    blocks: list[str] = []
    lines = iter(dockerfile.splitlines())
    for line in lines:
        if not line.startswith("RUN "):
            continue
        block = [line]
        while block[-1].rstrip().endswith("\\"):
            block.append(next(lines))
        blocks.append("\n".join(block))
    return blocks


def _micromamba_run_options(block: str) -> list[str]:
    """Return the option text before each continued micromamba command."""
    return re.findall(r"micromamba run\s+([^\\\n]+?)\s*\\", block)


class DockerfileCudaBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.run_blocks = _run_blocks(cls.dockerfile)

    def _extension_block(self, package: str) -> str:
        matches = [block for block in self.run_blocks if f'"{package} @' in block]
        self.assertEqual(len(matches), 1, f"expected one {package} build block")
        return matches[0]

    def test_cuda_extensions_use_supported_micromamba_run_options(self) -> None:
        for package in ("pytorch3d", "gsplat"):
            with self.subTest(package=package):
                block = self._extension_block(package)
                options = _micromamba_run_options(block)
                self.assertGreaterEqual(len(options), 1)
                self.assertEqual(
                    options,
                    ["-n sam3d-objects"] * len(options),
                )

    def test_cuda_preflight_precedes_pytorch3d_compilation(self) -> None:
        block = self._extension_block("pytorch3d")
        preflight = block.find("python /tmp/check_cuda_build_env.py")
        install = block.find("python -m pip install")
        self.assertGreaterEqual(preflight, 0)
        self.assertGreater(install, preflight)

    def test_runtime_defaults_to_zero_egress_model_loading(self) -> None:
        for setting in (
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, self.dockerfile)

        patcher = OFFLINE_PATCHER.read_text(encoding="utf-8")
        self.assertIn('source = "local"', patcher)
        self.assertIn('"pretrained": False', patcher)
        self.assertIn("SAM3D_DINOV2_REPO", patcher)
        self.assertIn("python /tmp/patch_offline_runtime.py", self.dockerfile)


if __name__ == "__main__":
    unittest.main()
