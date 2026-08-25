"""Regression checks for the FC image build contract."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
OFFLINE_PATCHER = ROOT / "scripts" / "patch_offline_runtime.py"
UNIFIED_CONSTRAINTS = ROOT / "constraints-unified.txt"


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


class DockerfileCudaBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.run_blocks = _run_blocks(cls.dockerfile)

    def _extension_block(self, package: str) -> str:
        matches = [block for block in self.run_blocks if f'"{package} @' in block]
        self.assertEqual(len(matches), 1, f"expected one {package} build block")
        return matches[0]

    def test_cuda_extensions_use_one_torch_cuda_abi(self) -> None:
        self.assertEqual(self.dockerfile.count("torch==2.7.1+cu126"), 1)
        self.assertEqual(self.dockerfile.count("torchvision==0.22.1+cu126"), 1)
        self.assertNotIn("torch==2.5.1", self.dockerfile)
        self.assertNotIn("cu121", self.dockerfile)
        torch_install = self.dockerfile.index("torch==2.7.1+cu126")

        for package in ("pytorch3d", "gsplat"):
            with self.subTest(package=package):
                block = self._extension_block(package)
                self.assertIn("--no-deps", block)
                self.assertIn("--no-build-isolation", block)
                self.assertNotIn("micromamba run", block)
                self.assertGreater(self.dockerfile.index(f'"{package} @'), torch_install)

    def test_ordinary_resolver_cannot_replace_the_preinstalled_torch_abi(self) -> None:
        constraints = UNIFIED_CONSTRAINTS.read_text(encoding="utf-8")
        self.assertIn("torch==2.7.1+cu126", constraints)
        self.assertIn("torchvision==0.22.1+cu126", constraints)
        self.assertIn("triton==3.3.1", constraints)
        self.assertIn("COPY constraints-unified.txt", self.dockerfile)
        self.assertIn("--constraint /tmp/constraints-unified.txt", self.dockerfile)

    def test_pytorch3d_pin_contains_the_cuda126_header_fix(self) -> None:
        self.assertIn(
            "ARG PYTORCH3D_REF=33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba",
            self.dockerfile,
        )
        self.assertIn("pytorch3d.git@${PYTORCH3D_REF}", self.dockerfile)
        self.assertNotIn(
            "pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47",
            self.dockerfile,
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
        runtime_patch = (ROOT / "patches" / "fc-runtime.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('source = "local"', patcher)
        self.assertIn('"pretrained": False', patcher)
        self.assertIn("SAM3D_DINOV2_REPO", patcher)
        self.assertIn("python /tmp/patch_offline_runtime.py", self.dockerfile)
        self.assertIn(
            '-os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]',
            runtime_patch,
        )

    def test_offline_source_patch_keeps_expensive_cuda_layers_cacheable(self) -> None:
        gsplat_build = self.dockerfile.index('"gsplat @ git+https://github.com/')
        patch_copy = self.dockerfile.index(
            "COPY scripts/patch_offline_runtime.py /tmp/patch_offline_runtime.py"
        )
        patch_run = self.dockerfile.index(
            "RUN python /tmp/patch_offline_runtime.py /opt/sam-3d-objects"
        )

        self.assertGreater(patch_copy, gsplat_build)
        self.assertGreater(patch_run, patch_copy)

    def test_final_runtime_repeats_the_import_and_abi_gate(self) -> None:
        runtime_stage = self.dockerfile.index("FROM ubuntu:22.04 AS runtime")
        final_copy = self.dockerfile.index(
            "COPY scripts/check_runtime_imports.py /tmp/check_runtime_imports.py",
            runtime_stage,
        )
        final_gate = self.dockerfile.index(
            "RUN python /tmp/check_runtime_imports.py", final_copy
        )
        self.assertGreater(final_gate, final_copy)


if __name__ == "__main__":
    unittest.main()
