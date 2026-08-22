"""Regression checks for the documented FC image publishing contract."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _bash_blocks(markdown: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", markdown, re.DOTALL)


class ReadmeImagePublishingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README.read_text(encoding="utf-8")
        cls.buildx_build_blocks = [
            block for block in _bash_blocks(cls.readme) if "docker buildx build" in block
        ]
        cls.manifest_check_blocks = [
            block
            for block in _bash_blocks(cls.readme)
            if "docker buildx imagetools inspect" in block
        ]

    def test_all_buildx_examples_disable_fc_incompatible_attestations(self) -> None:
        self.assertGreaterEqual(len(self.buildx_build_blocks), 1)
        for block in self.buildx_build_blocks:
            with self.subTest(block=block):
                self.assertIn("--platform linux/amd64", block)
                self.assertIn("--provenance=false", block)
                self.assertIn("--sbom=false", block)

    def test_acr_workflow_rejects_unknown_platform_manifests(self) -> None:
        self.assertEqual(len(self.manifest_check_blocks), 1)
        block = self.manifest_check_blocks[0]
        self.assertIn("grep -q 'unknown/unknown'", block)
        self.assertIn("exit 1", block)
        self.assertIn("docker manifest inspect --verbose", block)


if __name__ == "__main__":
    unittest.main()
