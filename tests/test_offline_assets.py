"""Regression tests for the offline FC model asset bundle."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_offline_assets.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("prepare_offline_assets", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


offline_assets = _load_script_module()


def _pipeline_text(*, include_all_paths: bool = True, moge_entries: int = 1) -> str:
    paths = [
        ("ss_generator_config_path", "ss_generator.yaml"),
        ("ss_generator_ckpt_path", "ss_generator.ckpt"),
        ("slat_generator_config_path", "slat_generator.yaml"),
        ("slat_generator_ckpt_path", "slat_generator.ckpt"),
        ("ss_decoder_config_path", "ss_decoder.yaml"),
        ("ss_decoder_ckpt_path", "ss_decoder.ckpt"),
        ("slat_decoder_gs_config_path", "slat_decoder_gs.yaml"),
        ("slat_decoder_gs_ckpt_path", "slat_decoder_gs.ckpt"),
        ("slat_decoder_gs_4_config_path", "slat_decoder_gs_4.yaml"),
        ("slat_decoder_gs_4_ckpt_path", "slat_decoder_gs_4.ckpt"),
        ("slat_decoder_mesh_config_path", "slat_decoder_mesh.yaml"),
        ("slat_decoder_mesh_ckpt_path", "slat_decoder_mesh.ckpt"),
    ]
    if not include_all_paths:
        paths.pop()
    lines = [f"{key}: {value}" for key, value in paths]
    lines.extend(
        "    pretrained_model_name_or_path: Ruicheng/moge-vitl"
        for _ in range(moge_entries)
    )
    return "\n".join(lines) + "\n"


class OfflineAssetPreparationTests(unittest.TestCase):
    def test_transfer_root_rejects_filesystem_root_before_locking(self) -> None:
        with self.assertRaisesRegex(offline_assets.AssetError, "must not be /"):
            offline_assets.validated_transfer_root(Path("/"))

    def test_preparation_lock_rejects_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transfer_root = Path(directory)
            with offline_assets.preparation_lock(transfer_root):
                with self.assertRaisesRegex(
                    offline_assets.AssetError,
                    "already running",
                ):
                    with offline_assets.preparation_lock(transfer_root):
                        self.fail("second writer acquired the preparation lock")

    def test_patch_pipeline_is_idempotent_and_keeps_backup_outside_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transfer_root = Path(directory)
            pipeline = transfer_root / "storage" / "hf" / "pipeline.yaml"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text(_pipeline_text(), encoding="utf-8")

            changed = offline_assets.patch_pipeline_for_local_moge(
                pipeline,
                transfer_root,
            )

            self.assertTrue(changed)
            self.assertIn(
                offline_assets.MOGE_RUNTIME_PATH,
                pipeline.read_text(encoding="utf-8"),
            )
            backup = transfer_root / "backups" / "pipeline.yaml.before-offline"
            self.assertEqual(backup.read_text(encoding="utf-8"), _pipeline_text())
            self.assertFalse(
                offline_assets.patch_pipeline_for_local_moge(
                    pipeline,
                    transfer_root,
                )
            )

    def test_patch_pipeline_does_not_consume_adjacent_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transfer_root = Path(directory)
            pipeline = transfer_root / "storage" / "hf" / "pipeline.yaml"
            pipeline.parent.mkdir(parents=True)
            original = "before: true\n\n" + _pipeline_text() + "\nafter: true\n"
            pipeline.write_text(original, encoding="utf-8")

            offline_assets.patch_pipeline_for_local_moge(pipeline, transfer_root)

            expected = original.replace(
                "Ruicheng/moge-vitl",
                offline_assets.MOGE_RUNTIME_PATH,
            )
            self.assertEqual(pipeline.read_text(encoding="utf-8"), expected)

    def test_checkpoint_copy_accepts_only_moge_localization_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            original = _pipeline_text()
            source.joinpath("pipeline.yaml").write_text(original, encoding="utf-8")
            for key in offline_assets.MAIN_CHECKPOINT_KEYS:
                filename, _ = offline_assets.KNOWN_CHECKPOINT_FILES[key]
                source.joinpath(filename).write_bytes(b"fixture")
            destination.joinpath("pipeline.yaml").write_text(
                original.replace(
                    "Ruicheng/moge-vitl",
                    offline_assets.MOGE_RUNTIME_PATH,
                ),
                encoding="utf-8",
            )

            offline_assets.copy_checkpoint_tree(source, destination)

            self.assertIn(
                offline_assets.MOGE_RUNTIME_PATH,
                destination.joinpath("pipeline.yaml").read_text(encoding="utf-8"),
            )

    def test_patch_pipeline_rejects_ambiguous_moge_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transfer_root = Path(directory)
            pipeline = transfer_root / "storage" / "hf" / "pipeline.yaml"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text(_pipeline_text(moge_entries=2), encoding="utf-8")

            with self.assertRaisesRegex(
                offline_assets.AssetError,
                "exactly one",
            ):
                offline_assets.patch_pipeline_for_local_moge(
                    pipeline,
                    transfer_root,
                )

    def test_patch_pipeline_tolerates_other_pretrained_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            transfer_root = Path(directory)
            pipeline = transfer_root / "storage" / "hf" / "pipeline.yaml"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text(
                "    pretrained_model_name_or_path: another/model\n"
                + _pipeline_text(),
                encoding="utf-8",
            )

            offline_assets.patch_pipeline_for_local_moge(pipeline, transfer_root)

            text = pipeline.read_text(encoding="utf-8")
            self.assertIn("another/model", text)
            self.assertIn(offline_assets.MOGE_RUNTIME_PATH, text)

    def test_main_checkpoint_inventory_rejects_missing_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Path(directory) / "pipeline.yaml"
            pipeline.write_text(
                _pipeline_text(include_all_paths=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                offline_assets.AssetError,
                "slat_decoder_mesh_ckpt_path",
            ):
                offline_assets.main_checkpoint_files(pipeline)

    def test_main_checkpoint_inventory_rejects_wrong_official_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.yaml"
            pipeline.write_text(_pipeline_text(), encoding="utf-8")
            for filename, _ in offline_assets.KNOWN_CHECKPOINT_FILES.values():
                root.joinpath(filename).write_bytes(b"git-lfs pointer")

            with self.assertRaisesRegex(
                offline_assets.AssetError,
                "size mismatch",
            ):
                offline_assets.main_checkpoint_files(pipeline)

    def test_checkpoint_inventory_includes_configured_optional_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = root / "pipeline.yaml"
            pipeline.write_text(
                _pipeline_text()
                + "ss_encoder_config_path: ss_encoder.yaml\n"
                + "ss_encoder_ckpt_path: null\n",
                encoding="utf-8",
            )

            inventory = offline_assets.checkpoint_inventory(pipeline)

            self.assertEqual(
                inventory["ss_encoder_config_path"],
                root.resolve() / "ss_encoder.yaml",
            )
            self.assertNotIn("ss_encoder_ckpt_path", inventory)

    def test_known_file_check_rejects_truncated_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "model.pt"
            weight.write_bytes(b"not-a-model")

            with self.assertRaisesRegex(
                offline_assets.AssetError,
                "size mismatch",
            ):
                offline_assets.verify_known_file(
                    weight,
                    expected_size=123,
                    expected_sha256="0" * 64,
                    label="test model",
                )

    def test_sam3_checkpoint_copy_uses_pinned_size_and_sha256(self) -> None:
        payload = b"small SAM 3 checkpoint fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            destination = root / "storage" / "sam3" / "sam3.pt"
            source.write_bytes(payload)
            with (
                mock.patch.object(offline_assets, "SAM3_WEIGHT_SIZE", len(payload)),
                mock.patch.object(
                    offline_assets,
                    "SAM3_WEIGHT_SHA256",
                    hashlib.sha256(payload).hexdigest(),
                ),
            ):
                offline_assets.copy_sam3_checkpoint(source, destination)
                offline_assets.verify_sam3_checkpoint(destination)
            self.assertEqual(destination.read_bytes(), payload)

    def test_sam3_download_uses_pinned_modelscope_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            download_root = Path(directory)
            destination = download_root / offline_assets.SAM3_FILENAME
            with mock.patch.object(offline_assets, "download_known_file") as download:
                result = offline_assets.download_sam3_checkpoint(download_root)

            self.assertEqual(result, destination)
            download.assert_called_once_with(
                destination,
                url=offline_assets.SAM3_MODELSCOPE_URL,
                expected_size=offline_assets.SAM3_WEIGHT_SIZE,
                expected_sha256=offline_assets.SAM3_WEIGHT_SHA256,
                label="SAM 3 checkpoint from ModelScope",
            )
            self.assertIn(
                offline_assets.SAM3_MODELSCOPE_REVISION,
                offline_assets.SAM3_MODELSCOPE_URL,
            )
            self.assertIn(
                "modelscope.cn/models/facebook/sam3/resolve/",
                offline_assets.SAM3_MODELSCOPE_URL,
            )
            self.assertNotIn("hf-mirror.com", offline_assets.SAM3_MODELSCOPE_URL)

    def test_auxiliary_weights_use_hf_mirror_urls(self) -> None:
        self.assertEqual(
            offline_assets.DINOV2_WEIGHT_URL,
            "https://hf-mirror.com/facebook/ShapeR/resolve/main/"
            "dinov2_vitl14_reg4_pretrain.pth",
        )
        self.assertEqual(
            offline_assets.MOGE_WEIGHT_URL,
            f"https://hf-mirror.com/Ruicheng/moge-vitl/resolve/"
            f"{offline_assets.MOGE_REF}/model.pt",
        )

    def test_bundle_recipe_id_tracks_content_instead_of_mirror_urls(self) -> None:
        recipe_id = offline_assets.bundle_recipe_id()
        self.assertRegex(recipe_id, r"^[0-9a-f]{64}$")

        with (
            mock.patch.object(
                offline_assets,
                "DINOV2_REPOSITORY",
                "https://example.invalid/dinov2.git",
            ),
            mock.patch.object(
                offline_assets,
                "MOGE_WEIGHT_URL",
                "https://example.invalid/moge/model.pt",
            ),
        ):
            self.assertEqual(offline_assets.bundle_recipe_id(), recipe_id)

        with mock.patch.object(
            offline_assets,
            "DINOV2_WEIGHT_SHA256",
            "0" * 64,
        ):
            self.assertNotEqual(offline_assets.bundle_recipe_id(), recipe_id)

    def test_sam3d_download_uses_pinned_modelscope_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            download_root = Path(directory)
            checkpoint_root = download_root / "checkpoints"
            with mock.patch.object(offline_assets, "download_known_file") as download:
                result = offline_assets.download_sam3d_checkpoint(download_root)

            self.assertEqual(result, checkpoint_root)
            expected_calls = [
                mock.call(
                    checkpoint_root / filename,
                    url=f"{offline_assets.SAM3D_MODELSCOPE_BASE_URL}/{filename}",
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                    label=f"SAM 3D Objects {filename} from ModelScope",
                )
                for filename, (expected_size, expected_sha256) in (
                    offline_assets.SAM3D_MODELSCOPE_FILES.items()
                )
            ]
            self.assertEqual(download.call_args_list, expected_calls)
            self.assertIn(
                offline_assets.SAM3D_MODELSCOPE_REVISION,
                offline_assets.SAM3D_MODELSCOPE_BASE_URL,
            )
            self.assertIn(
                "modelscope.cn/models/facebook/sam-3d-objects/resolve/",
                offline_assets.SAM3D_MODELSCOPE_BASE_URL,
            )
            self.assertNotIn(
                "hf-mirror.com",
                offline_assets.SAM3D_MODELSCOPE_BASE_URL,
            )

    def test_sam3_checkpoint_rejects_truncation_and_destination_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truncated = root / "sam3.pt"
            truncated.write_bytes(b"git-lfs pointer")
            with self.assertRaisesRegex(offline_assets.AssetError, "size mismatch"):
                offline_assets.verify_sam3_checkpoint(truncated)

            destination = root / "destination.pt"
            destination.symlink_to(root / "missing-target.pt")
            payload = truncated.read_bytes()
            with (
                mock.patch.object(offline_assets, "SAM3_WEIGHT_SIZE", len(payload)),
                mock.patch.object(
                    offline_assets,
                    "SAM3_WEIGHT_SHA256",
                    hashlib.sha256(payload).hexdigest(),
                ),
                self.assertRaisesRegex(offline_assets.AssetError, "symlink"),
            ):
                offline_assets.copy_sam3_checkpoint(truncated, destination)

    def test_checksum_manifest_detects_transfer_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            asset = storage / "hf" / "checkpoint.ckpt"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"complete checkpoint")
            offline_assets.write_checksum_manifest(storage, [asset])
            offline_assets.verify_checksum_manifest(storage, [asset])

            asset.write_bytes(b"corrupted checkpoint")

            with self.assertRaisesRegex(offline_assets.AssetError, "checksum mismatch"):
                offline_assets.verify_checksum_manifest(storage, [asset])

    def test_cli_help_and_incomplete_verify_contract(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--verify-only", help_result.stdout)
        self.assertIn("--verify-main-only", help_result.stdout)
        self.assertIn("--verify-sam3-only", help_result.stdout)
        self.assertIn("--print-recipe-id", help_result.stdout)
        self.assertIn("--download-sam3", help_result.stdout)
        self.assertIn("--sam3-source", help_result.stdout)
        self.assertIn("ModelScope", help_result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            verify_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--verify-only",
                    "--transfer-root",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(verify_result.returncode, 1)
        self.assertIn("pipeline.yaml", verify_result.stderr)
        self.assertNotEqual(verify_result.stderr.strip(), "")

        with tempfile.TemporaryDirectory() as directory:
            main_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--verify-main-only",
                    "--transfer-root",
                    directory,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(main_result.returncode, 1)
        self.assertIn("pipeline.yaml", main_result.stderr)

    def test_documented_fc_offline_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        preparer = SCRIPT.read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("scripts/prepare_offline_assets.py", readme)
        self.assertIn("/mnt/nas/sam3d/hf/moge/model.pt", readme)
        self.assertIn("facebookresearch_dinov2_main", readme)
        self.assertIn("HF_HUB_OFFLINE=1", readme)
        self.assertIn("HF_HUB_OFFLINE=1", env_example)
        self.assertNotIn("hf auth login", preparer)
        self.assertNotIn("HF_TOKEN", preparer)
        self.assertNotIn("use scripts/deploy_from_hk.sh", preparer)
        self.assertIn(
            "modelscope.cn/models/facebook/sam-3d-objects/resolve/",
            preparer,
        )
        self.assertIn(
            "https://modelscope.cn/models/facebook/sam-3d-objects",
            readme,
        )
        self.assertIn(
            "ARG PYTORCH_WHEEL_URL=https://mirrors.aliyun.com/pytorch-wheels/cu126",
            dockerfile,
        )
        self.assertIn('--find-links "${PYTORCH_WHEEL_URL}"', dockerfile)


if __name__ == "__main__":
    unittest.main()
