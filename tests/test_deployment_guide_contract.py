"""Regression checks for the zero-to-FC deployment guide."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "DEPLOYMENT.md"


def _blocks(markdown: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)\n```", markdown, re.DOTALL)


class DeploymentGuideContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README.read_text(encoding="utf-8")
        cls.guide = GUIDE.read_text(encoding="utf-8")

    def test_readme_links_the_complete_guide(self) -> None:
        self.assertIn("[阿里云 FC GPU 从零部署手册](DEPLOYMENT.md)", self.readme)

    def test_build_and_manifest_gates_are_fc_compatible(self) -> None:
        build_blocks = [
            block
            for block in _blocks(self.guide, "bash")
            if "docker buildx build" in block
        ]
        self.assertEqual(len(build_blocks), 1)
        self.assertIn("--platform linux/amd64", build_blocks[0])
        self.assertIn("--provenance=false", build_blocks[0])
        self.assertIn("--sbom=false", build_blocks[0])
        self.assertIn("grep -q 'unknown/unknown'", self.guide)
        self.assertIn("远程镜像不包含 linux/amd64 manifest", self.guide)

    def test_offline_assets_are_uploaded_and_mounted_at_runtime_paths(self) -> None:
        for required in (
            "scripts/prepare_offline_assets.py",
            "--verify-only",
            '"$TRANSFER_ROOT/storage/"',
            '"oss://${OSS_BUCKET}/${OSS_PREFIX}/"',
            "/mnt/nas/sam3d/hf/pipeline.yaml",
            "/mnt/nas/sam3d/hf/moge/model.pt",
            "/mnt/nas/sam3d/cache/torch/hub/facebookresearch_dinov2_main",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.guide)

    def test_fc_runtime_is_explicitly_offline(self) -> None:
        for setting in (
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "HF_DATASETS_OFFLINE=1",
            "HF_HUB_DISABLE_TELEMETRY=1",
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, self.guide)

        self.assertIn("source: local", self.guide)
        self.assertIn("pretrained: False", self.guide)

    def test_region_specific_network_boundaries_are_documented(self) -> None:
        self.assertIn("https://oss-cn-shenzhen.aliyuncs.com", self.guide)
        self.assertIn("深圳内网 Endpoint", self.guide)
        self.assertIn("香港 ECS 必须使用公网地址", self.guide)
        self.assertIn("不要使用包含 `-vpc`", self.guide)


if __name__ == "__main__":
    unittest.main()
