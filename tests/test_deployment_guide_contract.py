"""Regression checks for the focused ACR publishing guide."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "DEPLOYMENT.md"


class DeploymentGuideContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README.read_text(encoding="utf-8")
        cls.guide = GUIDE.read_text(encoding="utf-8")

    def test_readme_links_the_focused_guide(self) -> None:
        self.assertIn("[香港 ECS 构建并推送 ACR 手册](DEPLOYMENT.md)", self.readme)

    def test_guide_requires_only_the_acr_inputs(self) -> None:
        for required in (
            "ACR 完整公网仓库地址",
            "ACR 登录用户名",
            "ACR Registry 密码",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.guide)

        self.assertIn("脚本没有任何可选参数", self.guide)
        self.assertNotIn("./scripts/deploy_from_hk.sh --", self.guide)

    def test_guide_scopes_the_script_to_build_and_push(self) -> None:
        self.assertIn("构建一张 `linux/amd64` 统一镜像", self.guide)
        self.assertIn("登录 ACR", self.guide)
        self.assertIn("推送镜像", self.guide)
        self.assertIn("不会下载模型权重", self.guide)
        self.assertIn("不会访问 OSS", self.guide)
        self.assertIn("不会创建或修改函数计算", self.guide)
        self.assertIn("不会启动 GPU", self.guide)

    def test_guide_documents_automatic_safety_gates(self) -> None:
        for required in (
            "linux/amd64",
            "provenance=false",
            "sbom=false",
            "unknown/unknown",
            "自动生成不可变 tag",
            "Docker 构建缓存",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.guide)


if __name__ == "__main__":
    unittest.main()
