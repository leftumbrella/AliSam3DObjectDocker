"""Build-contract tests for the combined SAM 3 and SAM 3D image."""

from __future__ import annotations

from pathlib import Path
import re
import stat
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "requirements-segmenter.txt"
FC_REQUIREMENTS = ROOT / "requirements-fc.txt"
INITIALIZER = ROOT / "scripts" / "fc_initializer.sh"
ASSETS = ROOT / "scripts" / "prepare_offline_assets.py"
SUPERVISOR = ROOT / "app" / "supervisor.py"


class SegmenterDockerfileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.requirements = REQUIREMENTS.read_text(encoding="utf-8")
        cls.fc_requirements = FC_REQUIREMENTS.read_text(encoding="utf-8")
        cls.initializer = INITIALIZER.read_text(encoding="utf-8")
        cls.assets = ASSETS.read_text(encoding="utf-8")
        cls.supervisor = SUPERVISOR.read_text(encoding="utf-8")

    def test_unified_runtime_is_python312_torch27_cuda126(self) -> None:
        self.assertIn(
            "FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS unified-builder",
            self.dockerfile,
        )
        self.assertIn("FROM ubuntu:22.04 AS runtime", self.dockerfile)
        self.assertEqual(self.dockerfile.count("\nFROM "), 1)
        self.assertIn("micromamba create -y -p /opt/venv", self.dockerfile)
        self.assertIn("python=3.12.11", self.dockerfile)
        self.assertIn("torch==2.7.1+cu126", self.dockerfile)
        self.assertIn("torchvision==0.22.1+cu126", self.dockerfile)
        self.assertIn("/whl/cu126", self.dockerfile)
        self.assertNotIn("torch==2.5.1", self.dockerfile)
        self.assertNotIn("cu121", self.dockerfile)
        self.assertNotIn("segmenter-builder", self.dockerfile)
        self.assertNotIn("sam3d-builder", self.dockerfile)
        self.assertRegex(self.dockerfile, r"ARG SAM3_REF=[0-9a-f]{40}")

    def test_actual_model_builder_import_dependencies_are_explicit_and_gated(self) -> None:
        for dependency in (
            "einops==0.8.1",
            "setuptools==80.9.0",
            "psutil==7.0.0",
            "pycocotools==2.0.10",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, self.requirements)
        self.assertIn("from sam3.model_builder import build_tracker", self.dockerfile)
        self.assertIn(
            "from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor",
            self.dockerfile,
        )

    def test_both_services_share_one_runtime_inside_one_image(self) -> None:
        self.assertIn("COPY segmenter /srv/segmenter", self.dockerfile)
        self.assertIn("COPY app /srv/app", self.dockerfile)
        self.assertIn("COPY shared /srv/shared", self.dockerfile)
        self.assertIn("COPY --from=unified-builder /opt/venv /opt/venv", self.dockerfile)
        self.assertIn(
            "COPY --from=unified-builder /opt/sam-3d-objects /opt/sam-3d-objects",
            self.dockerfile,
        )
        self.assertIn("COPY --from=unified-builder /opt/sam3 /opt/sam3", self.dockerfile)
        self.assertNotIn("SAM3_PYTHON", self.dockerfile)
        self.assertNotIn("SAM3_PYTHON", self.supervisor)
        self.assertIn('[str(unified_python), "-m", "segmenter.serve"]', self.supervisor)
        self.assertIn('[sys.executable, "-m", "app.serve"]', self.supervisor)
        self.assertIn(
            "SAM3_CHECKPOINT_PATH=/mnt/nas/sam3d/sam3/sam3.pt",
            self.dockerfile,
        )
        self.assertIn('CMD ["python", "-m", "app.supervisor"]', self.dockerfile)
        self.assertFalse((ROOT / "Dockerfile.segmenter").exists())

    def test_sparse_runtime_uses_the_same_cuda126_family(self) -> None:
        self.assertIn("spconv-cu126==2.3.8", self.fc_requirements)
        self.assertNotIn("spconv-cu121", self.fc_requirements)

    def test_checkpoint_pin_is_traceable_and_separate_from_source_pin(self) -> None:
        self.assertIn("huggingface.co/api/models/facebook/sam3/revision/", self.assets)
        self.assertIn("SAM3_WEIGHT_SIZE = 3_450_062_241", self.assets)
        self.assertIn(
            "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
            self.assets,
        )
        source_ref = re.search(r"ARG SAM3_REF=([0-9a-f]{40})", self.dockerfile)
        weight_ref = re.search(r'SAM3_REVISION = "([0-9a-f]{40})"', self.assets)
        self.assertIsNotNone(source_ref)
        self.assertIsNotNone(weight_ref)
        self.assertNotEqual(source_ref.group(1), weight_ref.group(1))
        self.assertIn("independently versioned", self.dockerfile)

    def test_initializer_is_sync_executable_and_copied_into_combined_image(self) -> None:
        self.assertTrue(INITIALIZER.stat().st_mode & stat.S_IXUSR)
        self.assertEqual(self.initializer.splitlines()[0], "#!/bin/sh")
        self.assertIn("exec curl", self.initializer)
        self.assertIn("--fail", self.initializer)
        self.assertIn('--max-time "$INITIALIZER_TIMEOUT"', self.initializer)
        self.assertIn("http://127.0.0.1:9000/initialize", self.initializer)
        self.assertNotRegex(self.initializer, r"(?m)(?:^|[ \t])&[ \t]*(?:$|#)")
        self.assertIn(
            "COPY scripts/fc_initializer.sh /srv/scripts/fc_initializer.sh",
            self.dockerfile,
        )
        self.assertIn("curl", self.dockerfile)


if __name__ == "__main__":
    unittest.main()
