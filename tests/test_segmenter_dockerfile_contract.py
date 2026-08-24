"""Build-contract tests for the standalone SAM 3 segmenter image."""

from __future__ import annotations

from pathlib import Path
import re
import stat
import unittest


ROOT = Path(__file__).resolve().parents[1]
SEGMENTER_DOCKERFILE = ROOT / "Dockerfile.segmenter"
GENERATOR_DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "requirements-segmenter.txt"
INITIALIZER = ROOT / "scripts" / "fc_initializer.sh"
ASSETS = ROOT / "scripts" / "prepare_offline_assets.py"


class SegmenterDockerfileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.segmenter = SEGMENTER_DOCKERFILE.read_text(encoding="utf-8")
        cls.generator = GENERATOR_DOCKERFILE.read_text(encoding="utf-8")
        cls.requirements = REQUIREMENTS.read_text(encoding="utf-8")
        cls.initializer = INITIALIZER.read_text(encoding="utf-8")
        cls.assets = ASSETS.read_text(encoding="utf-8")

    def test_segmenter_runtime_is_python312_torch27_cuda126(self) -> None:
        self.assertGreaterEqual(self.segmenter.count("FROM python:3.12-slim-bookworm"), 2)
        self.assertIn("torch==2.7.1+cu126", self.segmenter)
        self.assertIn("torchvision==0.22.1+cu126", self.segmenter)
        self.assertIn("/whl/cu126", self.segmenter)
        self.assertRegex(self.segmenter, r"ARG SAM3_REF=[0-9a-f]{40}")

    def test_actual_model_builder_import_dependencies_are_explicit_and_gated(self) -> None:
        for dependency in (
            "einops==0.8.1",
            "setuptools==80.9.0",
            "psutil==7.0.0",
            "pycocotools==2.0.10",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, self.requirements)
        self.assertIn("from sam3.model_builder import build_tracker", self.segmenter)
        self.assertIn(
            "from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor",
            self.segmenter,
        )

    def test_segmenter_entrypoint_and_checkpoint_are_isolated(self) -> None:
        self.assertIn("COPY segmenter /srv/segmenter", self.segmenter)
        self.assertIn(
            "SAM3_CHECKPOINT_PATH=/mnt/nas/sam3d/sam3/sam3.pt",
            self.segmenter,
        )
        self.assertIn('CMD ["python", "-m", "segmenter.serve"]', self.segmenter)

    def test_checkpoint_pin_is_traceable_and_separate_from_source_pin(self) -> None:
        self.assertIn("huggingface.co/api/models/facebook/sam3/revision/", self.assets)
        self.assertIn("SAM3_WEIGHT_SIZE = 3_450_062_241", self.assets)
        self.assertIn(
            "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
            self.assets,
        )
        source_ref = re.search(r"ARG SAM3_REF=([0-9a-f]{40})", self.segmenter)
        weight_ref = re.search(r'SAM3_REVISION = "([0-9a-f]{40})"', self.assets)
        self.assertIsNotNone(source_ref)
        self.assertIsNotNone(weight_ref)
        self.assertNotEqual(source_ref.group(1), weight_ref.group(1))
        self.assertIn("independently versioned", self.segmenter)

    def test_initializer_is_sync_executable_and_copied_into_both_images(self) -> None:
        self.assertTrue(INITIALIZER.stat().st_mode & stat.S_IXUSR)
        self.assertEqual(self.initializer.splitlines()[0], "#!/bin/sh")
        self.assertIn("exec curl", self.initializer)
        self.assertIn("--fail", self.initializer)
        self.assertIn('--max-time "$INITIALIZER_TIMEOUT"', self.initializer)
        self.assertIn("http://127.0.0.1:9000/initialize", self.initializer)
        self.assertNotRegex(self.initializer, r"(?m)(?:^|[ \t])&[ \t]*(?:$|#)")
        for dockerfile in (self.segmenter, self.generator):
            self.assertIn(
                "COPY scripts/fc_initializer.sh /srv/scripts/fc_initializer.sh",
                dockerfile,
            )
            self.assertIn("curl", dockerfile)


if __name__ == "__main__":
    unittest.main()
