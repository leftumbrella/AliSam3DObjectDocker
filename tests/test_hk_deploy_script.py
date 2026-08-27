"""Contract tests for the zero-argument ACR image publishing entrypoint."""

from __future__ import annotations

from pathlib import Path
import re
import stat
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_from_hk.sh"
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT = ROOT / "DEPLOYMENT.md"


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
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("脚本不接受任何参数", result.stderr)

    def test_script_has_no_optional_parameter_surface(self) -> None:
        self.assertIn("require_no_arguments \"$@\"", self.script)
        self.assertIn("[[ $# -eq 0 ]]", self.script)
        for removed in (
            "parse_args",
            "usage()",
            "--dry-run",
            "--non-interactive",
            "--yes",
            "--skip-configure-fc",
            "--oss-bucket",
            "--acr-image",
            "--acr-username",
            "--gpu-type",
            "--gpu-memory-size",
            "--provisioned-instances",
            "--reserved-concurrency",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.script)

    def test_only_required_acr_information_is_prompted(self) -> None:
        reads = re.findall(r"^\s*read -r .*?$", self.script, re.MULTILINE)
        self.assertEqual(len(reads), 3)
        self.assertIn("ACR 完整公网仓库地址", self.script)
        self.assertIn("ACR 登录用户名", self.script)
        self.assertIn("ACR Registry 密码", self.script)
        self.assertIn("read -r -s", self.script)
        self.assertNotIn("确认执行", self.script)
        self.assertIn("ACR_IMAGE=''", self.script)
        self.assertIn("ACR_USERNAME=''", self.script)
        self.assertNotIn('${ACR_IMAGE:-', self.script)
        self.assertNotIn('${ACR_USERNAME:-', self.script)

    def test_script_only_builds_and_pushes_the_image(self) -> None:
        for required in (
            "install_base_tools",
            "ensure_docker",
            "build_image",
            "login_acr",
            "push_image",
            "verify_remote_manifest",
            "docker push \"$REMOTE_IMAGE\"",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)

        for removed in (
            "OSS_BUCKET",
            "OSS_ACCESS_KEY",
            "ossutil",
            "HF_TOKEN",
            "huggingface",
            "prepare_offline_assets.py",
            "configure_fc.py",
            "FC_ROLE_ARN",
            "FC_PROVISIONED_INSTANCES",
            "ALIBABA_CLOUD_ACCESS_KEY",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.script)

        main_block = self.script.split("main() {", 1)[1]
        self.assertLess(main_block.index("login_acr"), main_block.index("build_image"))

    def test_build_and_manifest_safety_gates_are_automatic(self) -> None:
        for required in (
            "--platform linux/amd64",
            "--provenance=false",
            "--sbom=false",
            "unknown/unknown",
            "远程镜像必须且只能包含 linux/amd64",
            "docker push 连续失败 3 次",
            'git -C "$PROJECT_ROOT" status --porcelain',
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)
        self.assertIn('IMAGE_TAG="sam3-sam3d-${GIT_COMMIT_SHORT}-${digest_hex:0:12}"', self.script)
        self.assertIn("remote_digest_matches_local_with_retry", self.script)

    def test_config_digest_comes_from_buildx_metadata(self) -> None:
        self.assertIn('--metadata-file "$BUILD_METADATA_FILE"', self.script)
        self.assertIn('."containerimage.config.digest"', self.script)
        self.assertIn(
            '."containerimage.descriptor".annotations["config.digest"]',
            self.script,
        )
        self.assertNotIn("--format '{{.Id}}'", self.script)

    def test_build_pins_match_the_dockerfile(self) -> None:
        for docker_arg, script_constant in (
            ("SAM3D_REF", "SAM3D_REF"),
            ("SAM3_REF", "SAM3_REF"),
            ("TORCH_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST_VALUE"),
        ):
            match = re.search(
                rf"^ARG {docker_arg}=(\S+)$",
                self.dockerfile,
                re.MULTILINE,
            )
            self.assertIsNotNone(match)
            self.assertIn(
                f"readonly {script_constant}='{match.group(1)}'",
                self.script,
            )

        self.assertIn("readonly MAX_JOBS_VALUE='2'", self.script)
        self.assertIn("readonly NVCC_THREADS_VALUE='2'", self.script)

    def test_registry_password_is_hidden_and_ephemeral(self) -> None:
        self.assertIn("set +x", self.script)
        self.assertIn("read -r -s", self.script)
        self.assertIn("--password-stdin", self.script)
        self.assertIn('export DOCKER_CONFIG="$TEMP_DIR/docker-config"', self.script)
        self.assertIn("docker logout \"$ACR_HOST\"", self.script)
        self.assertIn("safe_remove_temp_dir", self.script)

    def test_documentation_exposes_one_zero_argument_command(self) -> None:
        self.assertIn("## 运行", self.deployment)
        self.assertIn("脚本没有任何可选参数", self.deployment)
        self.assertIn("./scripts/deploy_from_hk.sh", self.deployment)
        self.assertNotIn("./scripts/deploy_from_hk.sh --", self.deployment)
        self.assertIn("不会访问 OSS", self.deployment)
        self.assertIn("不会创建或修改函数计算", self.deployment)


if __name__ == "__main__":
    unittest.main()
