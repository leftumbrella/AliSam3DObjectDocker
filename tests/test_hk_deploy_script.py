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
        "FC_ROLE_ARN",
        "FC_ACR_IMAGE",
        "FC_ACR_INSTANCE_ID",
        "ALIBABA_CLOUD_ACCESS_KEY_ID",
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
        "ALIBABA_CLOUD_SECURITY_TOKEN",
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
        self.assertIn("--skip-configure-fc", result.stdout)
        self.assertIn("--sam3-source", result.stdout)
        self.assertIn("--fc-function-name", result.stdout)
        self.assertIn("--provisioned-instances", result.stdout)
        self.assertIn("--reserved-concurrency", result.stdout)
        self.assertIn("--gpu-type", result.stdout)
        self.assertIn("--gpu-memory-size", result.stdout)
        self.assertIn("敏感信息不接受命令行参数", result.stdout)

    def test_script_runs_in_foreground_without_tmux_relaunch(self) -> None:
        result = subprocess.run(
            [str(SCRIPT), "--help"],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("当前终端前台连续执行", result.stdout)
        self.assertIn("install_base_tools\n  ensure_docker", self.script)
        for removed in (
            "tmux",
            "--no-tmux",
            "SAM3D_DEPLOY_BOOTSTRAPPED",
            "SAM3D_DEPLOY_INSIDE_TMUX",
            "shell_quote_command",
            "maybe_relaunch_in_tmux",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.script)

    def test_dry_run_renders_the_full_plan_without_root_or_linux(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--dry-run",
                "--non-interactive",
                "--yes",
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
            "sam3dobject:sam3-sam3d-",
            result.stdout,
        )
        self.assertIn("linux/amd64", result.stdout)
        self.assertNotIn("sam3-cu126-", result.stdout)
        self.assertIn("FC 配置：        启用", result.stdout)
        self.assertIn("最小实例数：     0", result.stdout)
        self.assertIn("函数总并发上限： 1", result.stdout)
        self.assertIn("<正式执行时将交互询问>", result.stdout)
        self.assertIn("dry-run：未执行任何系统或云端修改", result.stdout)

    def test_dry_run_can_explicitly_skip_fc_configuration(self) -> None:
        result = subprocess.run(
            [
                str(SCRIPT),
                "--dry-run",
                "--skip-configure-fc",
                "--oss-bucket",
                "example-bucket",
                "--acr-image",
                "crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/ns/repo",
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
        self.assertIn("已通过 --skip-configure-fc 跳过", result.stdout)

    def test_dry_run_accepts_the_supported_hopper_pair_only(self) -> None:
        base_command = [
            str(SCRIPT),
            "--dry-run",
            "--oss-bucket",
            "example-bucket",
            "--acr-image",
            "crpi-example.cn-shenzhen.personal.cr.aliyuncs.com/ns/repo",
            "--acr-username",
            "example-user",
            "--gpu-type",
            "fc.gpu.hopper.1",
        ]
        accepted = subprocess.run(
            [*base_command, "--gpu-memory-size", "98304"],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("fc.gpu.hopper.1:98304MB", accepted.stdout)

        rejected = subprocess.run(
            [*base_command, "--gpu-memory-size", "49152"],
            cwd=ROOT,
            env=_safe_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("GPU 组合仅支持", rejected.stderr)

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

    def test_ossutil_contract_uses_cobra_cp_help(self) -> None:
        self.assertIn('"$candidate" cp --help', self.script)
        self.assertNotIn('"$candidate" help cp', self.script)
        self.assertIn("ossutil cp --help | grep -q -- '--checkpoint-dir'", self.deployment)
        self.assertNotIn("ossutil help cp", self.deployment)

    def test_build_upload_and_manifest_safety_gates_are_present(self) -> None:
        for required in (
            "--platform linux/amd64",
            "--provenance=false",
            "--sbom=false",
            '"$OSSUTIL_BIN" cp -u -r',
            "--checkpoint-dir",
            "--verify-main-only",
            "--verify-sam3-only",
            "sam3/sam3.pt",
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

        sam3_ref = re.search(r"^ARG SAM3_REF=(\S+)$", self.dockerfile, re.MULTILINE)
        self.assertIsNotNone(sam3_ref)
        self.assertIn(f"readonly SAM3_REF='{sam3_ref.group(1)}'", self.script)

    def test_one_combined_image_and_fc_helper_are_in_the_one_command_path(self) -> None:
        for required in (
            'push_one_image_with_retry "$LOCAL_IMAGE" "$REMOTE_IMAGE"',
            'verify_remote_manifest "$REMOTE_IMAGE"',
            '"$PROJECT_ROOT/scripts/configure_fc.py"',
            '--build-arg "SAM3D_REF=${SAM3D_REF}"',
            '--build-arg "SAM3_REF=${SAM3_REF}"',
            "--function-name",
            "--provisioned-instances",
            "--reserved-concurrency",
            "--gpu-type",
            "--gpu-memory-size",
            "FC_DEPLOYMENT_RESULT_FILE",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)
        self.assertNotIn("Dockerfile.segmenter", self.script)
        self.assertNotIn("SEGMENTER_LOCAL_IMAGE", self.script)

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
            "ALIBABA_CLOUD_ACCESS_KEY_ID",
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
            "ALIBABA_CLOUD_SECURITY_TOKEN",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, result_block)
        self.assertIn("FC_FUNCTION_NAME", result_block)
        self.assertIn("FC_HTTP_URL", result_block)

    def test_deployment_guide_uses_the_one_command_path(self) -> None:
        self.assertIn("## 推荐：一键完成香港 ECS 动作", self.deployment)
        self.assertIn("./scripts/deploy_from_hk.sh", self.deployment)
        self.assertIn("deployment-result.env", self.deployment)
        self.assertIn("--dry-run", self.deployment)
        self.assertIn("--acr-image", self.deployment)
        self.assertIn("read -rp '深圳 ACR 完整公网仓库地址", self.deployment)
        self.assertNotIn("--acr-host 'crpi-xxxx", self.deployment)
        self.assertNotIn("tmux", self.deployment)


if __name__ == "__main__":
    unittest.main()
