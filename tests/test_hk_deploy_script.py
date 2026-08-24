"""Contract and smoke tests for the Hong Kong ECS deployment entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_from_hk.sh"
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT = ROOT / "DEPLOYMENT.md"


def _safe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HF_TOKEN",
        "OSS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET",
        "OSS_SESSION_TOKEN",
        "ACR_PASSWORD",
        "ACR_IMAGE",
        "ACR_HOST",
        "ACR_REPOSITORY",
    ):
        environment.pop(name, None)
    environment["NO_COLOR"] = "1"
    return environment


class HongKongDeployScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.deployment = DEPLOYMENT.read_text(encoding="utf-8")

    def test_script_is_an_executable_bash_entrypoint(self) -> None:
        self.assertTrue(SCRIPT.stat().st_mode & stat.S_IXUSR)
        self.assertEqual(self.script.splitlines()[0], "#!/usr/bin/env bash")
        result = subprocess.run(
            [str(SCRIPT), "--help"],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--non-interactive", result.stdout)
        self.assertIn("--acr-image", result.stdout)
        self.assertIn("敏感信息不接受命令行参数", result.stdout)

    def test_dry_run_renders_the_full_plan_without_root_or_linux(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--dry-run",
                "--oss-bucket",
                "example-bucket",
                "--acr-image",
                "crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/namespace/sam3dobject",
                "--acr-username",
                "example-user",
            ],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("执行计划", result.stdout)
        self.assertIn("example-bucket", result.stdout)
        self.assertIn(
            "crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/namespace/"
            "sam3dobject:cu121-",
            result.stdout,
        )
        self.assertIn("linux/amd64", result.stdout)
        self.assertIn("dry-run：未执行任何系统或云端修改", result.stdout)

    def test_acr_image_rejects_historical_address_pitfalls(self) -> None:
        cases = (
            (
                "https://crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/ns/repo",
                "ACR_IMAGE 必须是域名/namespace/repository",
            ),
            (
                "crpi-example-vpc.cn-shenzhen.personal.cr.aliyuncs.com/ns/repo",
                "不能使用 -vpc 或 -internal",
            ),
            (
                "crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/ns/repo:latest",
                "不能包含协议、tag、命令或多余路径",
            ),
            (
                "crpi-example.cn-hongkong.personal.cr.aliyuncs.com/ns/repo",
                "必须是深圳地域地址",
            ),
        )
        for acr_image, expected_error in cases:
            with self.subTest(acr_image=acr_image):
                result = subprocess.run(
                    [
                        str(SCRIPT),
                        "--dry-run",
                        "--oss-bucket",
                        "example-bucket",
                        "--acr-image",
                        acr_image,
                        "--acr-username",
                        "example-user",
                    ],
                    cwd=ROOT,
                    env=_safe_environment(),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_non_interactive_mode_requires_the_complete_acr_image(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--dry-run",
                "--non-interactive",
                "--yes",
                "--oss-bucket",
                "example-bucket",
                "--acr-username",
                "example-user",
            ],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("无交互模式缺少： ACR_IMAGE", result.stderr)

    def test_unknown_and_secret_shaped_options_are_rejected_before_actions(self) -> None:
        for option in ("--acr-password", "--hf-token"):
            with self.subTest(option=option):
                result = subprocess.run(
                    [str(SCRIPT), option, "not-a-real-secret", "--dry-run"],
                    cwd=ROOT,
                    env=_safe_environment(),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"未知选项：{option}", result.stderr)
                self.assertNotIn("not-a-real-secret", result.stderr)

    def test_credentials_are_not_passed_as_cli_arguments_or_traced(self) -> None:
        self.assertIn("set +x", self.script)
        self.assertIn("--password-stdin", self.script)
        self.assertIn('export DOCKER_CONFIG="$TEMP_DIR/docker-config"', self.script)
        self.assertIn("read -r -s -p 'ACR Registry password", self.script)
        self.assertIn("read -r -s -p 'OSS AccessKey Secret", self.script)
        self.assertIn("read -r -s -p 'Hugging Face Access Token", self.script)
        self.assertIn('token=os.environ["HF_TOKEN"]', self.script)
        self.assertIn("深圳 ACR 完整公网仓库地址", self.script)
        self.assertNotIn("--acr-password)", self.script)
        self.assertNotIn("--acr-host HOST", self.script)
        self.assertNotIn("--acr-repository PATH", self.script)
        self.assertNotIn("ACR_REPOSITORY=", self.script)
        self.assertNotIn('"$HF_BIN" auth login', self.script)
        self.assertNotIn('auth login --token "$HF_TOKEN"', self.script)
        self.assertNotIn("HF_LOGIN_CREATED", self.script)
        self.assertIn(
            "unset ACR_PASSWORD HF_TOKEN OSS_ACCESS_KEY_ID",
            self.script,
        )

    def test_build_upload_and_manifest_safety_gates_are_present(self) -> None:
        for required in (
            "--platform linux/amd64",
            "--provenance=false",
            "--sbom=false",
            '"$OSSUTIL_BIN" cp -u -r',
            "--checkpoint-dir",
            "--verify-main-only",
            "unknown/unknown",
            "远程镜像必须且只能包含 linux/amd64",
            "docker push 连续失败 3 次",
            "git -C \"$PROJECT_ROOT\" status --porcelain",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)

    def test_build_pins_match_the_dockerfile(self) -> None:
        sam3d_ref = re.search(r"^ARG SAM3D_REF=(\S+)$", self.dockerfile, re.MULTILINE)
        cuda_arch = re.search(
            r"^ARG TORCH_CUDA_ARCH_LIST=(\S+)$",
            self.dockerfile,
            re.MULTILINE,
        )
        self.assertIsNotNone(sam3d_ref)
        self.assertIsNotNone(cuda_arch)
        self.assertIn(f"readonly SAM3D_REF='{sam3d_ref.group(1)}'", self.script)
        self.assertIn(
            f"readonly TORCH_CUDA_ARCH_LIST_VALUE='{cuda_arch.group(1)}'",
            self.script,
        )

    def test_assets_use_content_addressed_prefix_and_nonsecret_result(self) -> None:
        self.assertIn("storage/offline-assets.sha256", self.script)
        self.assertIn('ASSET_RELEASE="bundle-${digest}"', self.script)
        self.assertIn('OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"', self.script)
        self.assertIn("deployment-result.env", self.script)
        result_block = self.script.split("write_deployment_result()", 1)[1].split(
            "print_completion()",
            1,
        )[0]
        for secret in (
            "HF_TOKEN",
            "OSS_ACCESS_KEY_ID",
            "OSS_ACCESS_KEY_SECRET",
            "OSS_SESSION_TOKEN",
            "ACR_PASSWORD",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, result_block)

    def test_deployment_guide_uses_the_one_command_path(self) -> None:
        self.assertIn("## 推荐：一键完成香港 ECS 动作", self.deployment)
        self.assertIn("./scripts/deploy_from_hk.sh", self.deployment)
        self.assertIn("deployment-result.env", self.deployment)
        self.assertIn("--dry-run", self.deployment)
        self.assertIn("--acr-image", self.deployment)
        self.assertIn("read -rp '深圳 ACR 完整公网仓库地址", self.deployment)
        self.assertNotIn("--acr-host 'crpi-xxxx", self.deployment)


if __name__ == "__main__":
    unittest.main()
