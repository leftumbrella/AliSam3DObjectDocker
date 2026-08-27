"""Regression checks for the focused OSS and ACR publishing guide."""

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
        self.assertIn("[香港 ECS 上传 OSS 并推送 ACR 手册](DEPLOYMENT.md)", self.readme)

    def test_guide_documents_required_and_conditional_inputs(self) -> None:
        for required in (
            "深圳 OSS Bucket 名",
            "ACR 完整公网仓库地址",
            "ACR 登录用户名",
            "ACR Registry 密码",
            "Hugging Face Access Token",
            "OSS AccessKey Secret",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.guide)

        self.assertIn("脚本没有任何可选参数", self.guide)
        self.assertNotIn("./scripts/deploy_from_hk.sh --", self.guide)

    def test_guide_scopes_the_script_to_assets_oss_and_acr(self) -> None:
        self.assertIn("准备并校验完整离线模型资源", self.guide)
        self.assertIn("上传深圳 OSS", self.guide)
        self.assertIn("sam3/sam3.pt", self.guide)
        self.assertIn("offline-assets.sha256", self.guide)
        self.assertIn("构建一张 `linux/amd64` 统一镜像", self.guide)
        self.assertIn("登录 ACR", self.guide)
        self.assertIn("推送镜像", self.guide)
        self.assertIn("不会创建或修改函数计算", self.guide)
        self.assertIn("不会启动 GPU", self.guide)
        self.assertIn("https://modelscope.cn/models/facebook/sam3", self.guide)
        self.assertIn("SAM3 从 ModelScope", self.guide)

    def test_guide_documents_automatic_safety_gates(self) -> None:
        for required in (
            "linux/amd64",
            "provenance=false",
            "sbom=false",
            "unknown/unknown",
            "自动生成不可变 tag",
            "Docker 构建缓存",
            "精确上传清单",
            "CRC64",
            "/root/sam3d-transfer/ossutil-output",
            "构建完成后才登录 ACR",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.guide)


if __name__ == "__main__":
    unittest.main()
