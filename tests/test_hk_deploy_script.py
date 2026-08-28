"""Contract tests for the zero-argument OSS and ACR publishing entrypoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import re
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deploy_from_hk.sh"
DOCKERFILE = ROOT / "Dockerfile"
DEPLOYMENT = ROOT / "DEPLOYMENT.md"


class HongKongDeployScriptTests(unittest.TestCase):
    RECIPE_ID = "a" * 64

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

    def test_required_oss_and_acr_information_is_prompted(self) -> None:
        self.assertIn("深圳 OSS Bucket 名", self.script)
        self.assertIn("ACR 完整公网仓库地址", self.script)
        self.assertIn("ACR 登录用户名", self.script)
        self.assertIn("ACR Registry 密码", self.script)
        self.assertNotIn("Hugging Face Access Token", self.script)
        self.assertNotIn("HF_TOKEN", self.script)
        self.assertIn("OSS AccessKey Secret", self.script)
        self.assertIn("read -r -s", self.script)
        self.assertNotIn("确认执行", self.script)
        self.assertIn("OSS_BUCKET=''", self.script)
        self.assertIn("ACR_IMAGE=''", self.script)
        self.assertIn("ACR_USERNAME=''", self.script)
        self.assertNotIn('${ACR_IMAGE:-', self.script)
        self.assertNotIn('${ACR_USERNAME:-', self.script)

    def test_script_prepares_uploads_complete_assets_and_pushes_image(self) -> None:
        for required in (
            "install_base_tools",
            "ensure_docker",
            "ensure_tool_venv",
            "ensure_ossutil",
            "prepare_offline_assets",
            "ensure_oss_access",
            "upload_offline_assets",
            "verify_remote_oss_inventory",
            "reuse_remote_offline_assets",
            "publish_asset_receipt",
            "complete.json",
            "scripts/prepare_offline_assets.py",
            "--download-sam3d",
            "--download-sam3",
            "--verify-only",
            '"$OSSUTIL_BIN" cp -u -r',
            '"$OSSUTIL_BIN" hash crc64',
            '--files-from-raw "$OSS_UPLOAD_LIST"',
            '--output-dir "$output_dir"',
            "build_image",
            "login_acr",
            "push_image",
            "verify_remote_manifest",
            "run_docker push \"$REMOTE_IMAGE\"",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.script)

        for removed in (
            "configure_fc.py",
            "FC_ROLE_ARN",
            "FC_PROVISIONED_INSTANCES",
            "ALIBABA_CLOUD_ACCESS_KEY",
            "tmux",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.script)

        main_block = self.script.split("main() {", 1)[1]
        ordered_steps = (
            "ensure_tool_venv",
            "ensure_ossutil",
            "resolve_asset_recipe",
            "ensure_oss_access",
            "reuse_remote_offline_assets",
            "prepare_offline_assets",
            "upload_offline_assets",
            "build_image",
            "login_acr",
            "push_image",
            "verify_remote_manifest",
        )
        positions = [main_block.index(step) for step in ordered_steps]
        self.assertEqual(positions, sorted(positions))

    def test_non_root_execution_keeps_host_tooling_isolated(self) -> None:
        self.assertIn('TRANSFER_ROOT="$user_home/sam3d-transfer"', self.script)
        self.assertIn('sudo -- "$@"', self.script)
        self.assertIn("当前用户可直接访问 Docker daemon", self.script)
        self.assertIn("仅对 Docker 命令使用 sudo", self.script)
        self.assertIn("run_docker buildx build", self.script)
        self.assertIn('export DOCKER_CONFIG="$TEMP_DIR/docker-config"', self.script)
        self.assertIn('tools_root="$TEMP_DIR/python-venv"', self.script)
        self.assertIn(
            '"$SYSTEM_PYTHON" -I -m venv --without-pip "$tools_root"',
            self.script,
        )
        self.assertIn("readonly SYSTEM_PYTHON='/usr/bin/python3'", self.script)
        self.assertIn('"$TOOLS_PYTHON" -I "$@"', self.script)
        self.assertIn('candidate="$TEMP_DIR/native-tools/ossutil"', self.script)
        self.assertIn("-u PYTHONPATH", self.script)
        self.assertIn("-u VIRTUAL_ENV", self.script)
        for forbidden in (
            "请先执行 sudo -i",
            "/opt/sam3d-tools",
            "/root/sam3d-transfer",
            "python3-pip",
            "pip install",
            'export PATH="$tools_root/bin:$PATH"',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.script)

    def test_transfer_root_defaults_to_the_invoking_users_home(self) -> None:
        env = os.environ.copy()
        env["HOME"] = "/tmp/sam3d-test-home"
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; initialize_user_paths; printf "%s" "$TRANSFER_ROOT"',
                "bash",
                str(SCRIPT),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "/tmp/sam3d-test-home/sam3d-transfer")

    def test_both_main_checkpoints_use_modelscope_without_hf_token(self) -> None:
        self.assertIn(
            "SAM3D 主 checkpoint 缺失，将从 ModelScope 下载公开权重",
            self.script,
        )
        self.assertIn(
            "SAM3 checkpoint 缺失，将从 ModelScope 下载公开权重",
            self.script,
        )
        self.assertNotIn("ensure_huggingface_access", self.script)
        self.assertNotIn("huggingface_hub", self.script)
        self.assertNotIn("needs_huggingface", self.script)

    def test_complete_remote_recipe_skips_model_download_and_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "complete.json"
            calls = root / "ossutil-calls.log"
            fake_ossutil = root / "ossutil"
            manifest_sha256 = "b" * 64
            asset_release = "bundle-" + manifest_sha256[:12]
            receipt.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "recipe_id": self.RECIPE_ID,
                        "asset_release": asset_release,
                        "manifest_sha256": manifest_sha256,
                        "object_count": 2,
                        "objects": [
                            {"crc64": "111", "key": "offline-assets.sha256"},
                            {"crc64": "222", "key": "sam3/sam3.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fake_ossutil.write_text(
                """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_OSS_LOG"
if [ "$1" = "cp" ] && [ "$2" = "-f" ]; then
  cp "$FAKE_RECEIPT" "$4"
elif [ "$1" = "hash" ]; then
  case "$3" in
    */offline-assets.sha256) printf '111  %s\\n' "$3" ;;
    */sam3/sam3.pt) printf '222  %s\\n' "$3" ;;
    *) exit 3 ;;
  esac
fi
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            env = os.environ.copy()
            env["FAKE_OSS_LOG"] = str(calls)
            env["FAKE_RECEIPT"] = str(receipt)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
TRANSFER_ROOT="$2"
OSS_BUCKET='example-bucket'
ASSET_RECIPE_ID="$4"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$3"
ensure_transfer_root
reuse_remote_offline_assets
printf 'REUSED_PREFIX=%s\\n' "$OSS_PREFIX"
""",
                    "bash",
                    str(SCRIPT),
                    str(root / "transfer"),
                    str(fake_ossutil),
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"REUSED_PREFIX=sam3d/releases/{asset_release}",
                result.stdout,
            )
            calls_text = calls.read_text(encoding="utf-8")
            self.assertIn(
                f"oss://example-bucket/sam3d/recipes/{self.RECIPE_ID}/complete.json",
                calls_text,
            )
            self.assertIn("offline-assets.sha256", calls_text)
            self.assertIn("sam3/sam3.pt", calls_text)
            self.assertNotIn("cp -u -r", calls_text)

    def test_remote_recipe_crc_mismatch_falls_back_to_local_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "complete.json"
            fake_ossutil = root / "ossutil"
            manifest_sha256 = "c" * 64
            receipt.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "recipe_id": self.RECIPE_ID,
                        "asset_release": "bundle-" + manifest_sha256[:12],
                        "manifest_sha256": manifest_sha256,
                        "object_count": 1,
                        "objects": [
                            {"crc64": "111", "key": "offline-assets.sha256"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            fake_ossutil.write_text(
                """#!/bin/sh
if [ "$1" = "cp" ] && [ "$2" = "-f" ]; then
  cp "$FAKE_RECEIPT" "$4"
elif [ "$1" = "hash" ]; then
  printf '999  %s\\n' "$3"
fi
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            env = os.environ.copy()
            env["FAKE_RECEIPT"] = str(receipt)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
OSS_BUCKET='example-bucket'
ASSET_RECIPE_ID="$3"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$2"
if reuse_remote_offline_assets; then
  exit 9
fi
printf 'FALLBACK\\n'
""",
                    "bash",
                    str(SCRIPT),
                    str(fake_ossutil),
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FALLBACK", result.stdout)
            self.assertIn("CRC64 不匹配", result.stderr)

    def test_missing_remote_recipe_falls_back_to_local_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_ossutil = root / "ossutil"
            fake_ossutil.write_text(
                """#!/bin/sh
printf 'Error: NoSuchKey: receipt does not exist\n' >&2
exit 1
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
OSS_BUCKET='example-bucket'
ASSET_RECIPE_ID="$3"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$2"
if reuse_remote_offline_assets; then
  exit 9
fi
printf 'FALLBACK\n'
""",
                    "bash",
                    str(SCRIPT),
                    str(fake_ossutil),
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FALLBACK", result.stdout)
            self.assertIn("尚无当前资源配方的完成凭据", result.stdout)

    def test_upload_verifies_every_checksum_manifest_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer_root = root / "transfer"
            storage = transfer_root / "storage"
            storage.mkdir(parents=True)
            keys = (
                "hf/pipeline.yaml",
                "sam3/sam3.pt",
                "cache/torch/hub/checkpoints/dinov2.pth",
            )
            manifest = storage / "offline-assets.sha256"
            manifest.write_text(
                "".join(f"{'0' * 64}  {key}\n" for key in keys),
                encoding="utf-8",
            )
            asset_release = "bundle-" + hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()[:12]
            for key in keys:
                path = storage / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(key, encoding="utf-8")
            git_metadata = storage / "cache" / "torch" / "hub" / "dinov2" / ".git" / "HEAD"
            git_metadata.parent.mkdir(parents=True)
            git_metadata.write_text("ref: refs/heads/main\n", encoding="utf-8")

            calls = root / "ossutil-calls.log"
            receipt_capture = root / "complete.json"
            fake_ossutil = root / "ossutil"
            fake_ossutil.write_text(
                """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_OSS_LOG"
if [ "$1" = "hash" ]; then
  printf '123  %s\\n' "$3"
elif [ "$1" = "cp" ] && [ "$2" = "-f" ]; then
  case "$4" in
    oss://*/complete.json) cp "$3" "$FAKE_RECEIPT_CAPTURE" ;;
  esac
fi
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            env = os.environ.copy()
            env["FAKE_OSS_LOG"] = str(calls)
            env["FAKE_RECEIPT_CAPTURE"] = str(receipt_capture)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
TRANSFER_ROOT="$2"
OSS_BUCKET='example-bucket'
ASSET_RELEASE="$4"
OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"
ASSET_RECIPE_ID="$5"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$3"
upload_offline_assets
""",
                    "bash",
                    str(SCRIPT),
                    str(transfer_root),
                    str(fake_ossutil),
                    asset_release,
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls_text = calls.read_text(encoding="utf-8")
            self.assertIn(
                "cp -u -r " + str(storage) + "/ oss://example-bucket/",
                calls_text,
            )
            self.assertIn("--files-from-raw", calls_text)
            self.assertIn(
                "--output-dir " + str(transfer_root / "ossutil-output"),
                calls_text,
            )
            self.assertIn("offline-assets.sha256", calls_text)
            for key in keys:
                with self.subTest(key=key):
                    self.assertIn(key, calls_text)
            upload_list = transfer_root / f"oss-upload-files-{asset_release}.txt"
            self.assertEqual(
                upload_list.read_text(encoding="utf-8").splitlines(),
                ["offline-assets.sha256", *keys],
            )
            self.assertNotIn(".git", upload_list.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_capture.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], 1)
            self.assertEqual(receipt["recipe_id"], self.RECIPE_ID)
            self.assertEqual(receipt["asset_release"], asset_release)
            self.assertEqual(receipt["object_count"], len(keys) + 1)
            self.assertEqual(
                {entry["key"] for entry in receipt["objects"]},
                {"offline-assets.sha256", *keys},
            )

    def test_crc_mismatch_is_repaired_with_an_exact_object_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer_root = root / "transfer"
            storage = transfer_root / "storage"
            checkpoint = storage / "sam3" / "sam3.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")
            manifest = storage / "offline-assets.sha256"
            manifest.write_text(
                f"{'0' * 64}  sam3/sam3.pt\n",
                encoding="utf-8",
            )
            asset_release = "bundle-" + hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()[:12]

            calls = root / "ossutil-calls.log"
            repaired = root / "repaired"
            fake_ossutil = root / "ossutil"
            fake_ossutil.write_text(
                """#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_OSS_LOG"
if [ "$1" = "hash" ]; then
  case "$3" in
    oss://*/sam3/sam3.pt)
      if [ -f "$FAKE_REPAIRED" ]; then
        printf '111  %s\\n' "$3"
      else
        printf '222  %s.bak\\n' "$3"
      fi
      ;;
    *) printf '111  %s\\n' "$3" ;;
  esac
elif [ "$1" = "cp" ] && [ "$2" = "-f" ]; then
  : > "$FAKE_REPAIRED"
fi
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            env = os.environ.copy()
            env["FAKE_OSS_LOG"] = str(calls)
            env["FAKE_REPAIRED"] = str(repaired)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
TRANSFER_ROOT="$2"
OSS_BUCKET='example-bucket'
ASSET_RELEASE="$4"
OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"
ASSET_RECIPE_ID="$5"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$3"
upload_offline_assets
""",
                    "bash",
                    str(SCRIPT),
                    str(transfer_root),
                    str(fake_ossutil),
                    asset_release,
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(repaired.is_file())
            calls_text = calls.read_text(encoding="utf-8")
            self.assertIn(
                "cp -f "
                + str(checkpoint)
                + f" oss://example-bucket/sam3d/releases/{asset_release}/sam3/sam3.pt",
                calls_text,
            )
            self.assertNotIn("ls oss://example-bucket/sam3d/releases", calls_text)

    def test_failed_upload_writes_report_outside_the_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer_root = root / "transfer"
            storage = transfer_root / "storage"
            checkpoint = storage / "sam3" / "sam3.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")
            manifest = storage / "offline-assets.sha256"
            manifest.write_text(
                f"{'0' * 64}  sam3/sam3.pt\n",
                encoding="utf-8",
            )
            asset_release = "bundle-" + hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()[:12]

            fake_ossutil = root / "ossutil"
            fake_ossutil.write_text(
                """#!/bin/sh
if [ "$1" = "cp" ]; then
  previous=''
  for argument in "$@"; do
    if [ "$previous" = "--output-dir" ]; then
      mkdir -p "$argument"
      : > "$argument/ossutil_report_test.report"
      exit 4
    fi
    previous=$argument
  done
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_ossutil.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
source "$1"
TRANSFER_ROOT="$2"
OSS_BUCKET='example-bucket'
ASSET_RELEASE="$4"
OSS_PREFIX="sam3d/releases/${ASSET_RELEASE}"
ASSET_RECIPE_ID="$5"
ASSET_RECEIPT_KEY="sam3d/recipes/${ASSET_RECIPE_ID}/complete.json"
OSSUTIL_BIN="$3"
upload_offline_assets
""",
                    "bash",
                    str(SCRIPT),
                    str(transfer_root),
                    str(fake_ossutil),
                    asset_release,
                    self.RECIPE_ID,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 4, result.stderr)
            self.assertTrue(
                (transfer_root / "ossutil-output" / "ossutil_report_test.report").is_file()
            )
            self.assertFalse((ROOT / "ossutil_output").exists())

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
        self.assertIn("run_docker logout \"$ACR_HOST\"", self.script)
        self.assertIn("safe_remove_temp_dir", self.script)

    def test_documentation_exposes_one_zero_argument_command(self) -> None:
        self.assertIn("## 运行", self.deployment)
        self.assertIn("脚本没有任何可选参数", self.deployment)
        self.assertIn("./scripts/deploy_from_hk.sh", self.deployment)
        self.assertNotIn("./scripts/deploy_from_hk.sh --", self.deployment)
        self.assertIn("上传深圳 OSS", self.deployment)
        self.assertIn("sam3/sam3.pt", self.deployment)
        self.assertIn("不会创建或修改函数计算", self.deployment)
        self.assertIn("使用普通用户直接运行脚本", self.deployment)
        self.assertIn("不调用 `pip`", self.deployment)


if __name__ == "__main__":
    unittest.main()
