#!/usr/bin/env python3
"""Prepare and verify the complete offline model bundle used by Alibaba FC."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


SAM3D_REPOSITORY = "facebook/sam-3d-objects"
# Pinned to the public ModelScope mirror:
# https://modelscope.cn/models/facebook/sam-3d-objects/resolve/64ec50f24a12eb631aceff750993d162d751dc25/checkpoints
SAM3D_MODELSCOPE_REVISION = "64ec50f24a12eb631aceff750993d162d751dc25"
SAM3D_MODELSCOPE_BASE_URL = (
    f"https://modelscope.cn/models/{SAM3D_REPOSITORY}/resolve/"
    f"{SAM3D_MODELSCOPE_REVISION}/checkpoints"
)
SAM3D_MODELSCOPE_FILES = {
    "pipeline.yaml": (
        3_548,
        "53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c",
    ),
    "ss_generator.yaml": (
        5_076,
        "3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd",
    ),
    "ss_generator.ckpt": (
        6_690_136_964,
        "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146",
    ),
    "slat_generator.yaml": (
        1_986,
        "53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e",
    ),
    "slat_generator.ckpt": (
        4_906_537_684,
        "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb",
    ),
    "ss_decoder.yaml": (
        244,
        "baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae",
    ),
    "ss_decoder.ckpt": (
        147_609_242,
        "6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340",
    ),
    "slat_decoder_gs.yaml": (
        576,
        "53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670",
    ),
    "slat_decoder_gs.ckpt": (
        171_476_155,
        "f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111",
    ),
    "slat_decoder_gs_4.yaml": (
        575,
        "3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48",
    ),
    "slat_decoder_gs_4.ckpt": (
        170_269_801,
        "731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186",
    ),
    "slat_decoder_mesh.yaml": (
        300,
        "8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e",
    ),
    "slat_decoder_mesh.ckpt": (
        363_726_862,
        "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33",
    ),
}
SAM3_REPOSITORY = "facebook/sam3"
# Pinned to the public ModelScope mirror:
# https://modelscope.cn/models/facebook/sam3/resolve/96f3e1b404ba14f2cfac60ee6ae87c269a7b7923/sam3.pt
# The SAM 3 Git source commit used by Dockerfile's unified builder belongs to a
# separate repository/history and is therefore pinned independently.
SAM3_MODELSCOPE_REVISION = "96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
SAM3_FILENAME = "sam3.pt"
SAM3_MODELSCOPE_URL = (
    f"https://modelscope.cn/models/{SAM3_REPOSITORY}/resolve/"
    f"{SAM3_MODELSCOPE_REVISION}/{SAM3_FILENAME}"
)
SAM3_WEIGHT_SIZE = 3_450_062_241
SAM3_WEIGHT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
DINOV2_REPOSITORY = (
    "https://v4.gh-proxy.org/https://github.com/facebookresearch/dinov2.git"
)
DINOV2_REF = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_WEIGHT_URL = (
    "https://hf-mirror.com/facebook/ShapeR/resolve/main/"
    "dinov2_vitl14_reg4_pretrain.pth"
)
DINOV2_WEIGHT_SIZE = 1_217_607_321
DINOV2_WEIGHT_SHA256 = (
    "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"
)
MOGE_REF = "ad326bfb61facd6c52b5a825bc1e34d7c97d9672"
MOGE_WEIGHT_URL = (
    f"https://hf-mirror.com/Ruicheng/moge-vitl/resolve/{MOGE_REF}/model.pt"
)
MOGE_WEIGHT_SIZE = 1_256_823_446
MOGE_WEIGHT_SHA256 = (
    "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f"
)
MOGE_REMOTE_ID = "Ruicheng/moge-vitl"
MOGE_RUNTIME_PATH = "/mnt/nas/sam3d/hf/moge/model.pt"
DINOV2_DIRECTORY = "facebookresearch_dinov2_main"
DINOV2_WEIGHT = "dinov2_vitl14_reg4_pretrain.pth"
SOURCE_REF_MARKER = ".sam3d-offline-ref"
CHECKSUM_MANIFEST = "offline-assets.sha256"
# Increment this whenever bundle paths or deterministic transformations change.
BUNDLE_RECIPE_SCHEMA = 1
MOGE_ENTRY_PATTERN = re.compile(
    r"^(?P<prefix>[ \t]*pretrained_model_name_or_path[ \t]*:[ \t]*)"
    r"(?P<quote>['\"]?)(?P<value>[^'\"\s#]+)(?P=quote)"
    r"(?P<suffix>[ \t]*(?:#.*)?)$",
    re.MULTILINE,
)

KNOWN_CHECKPOINT_FILES = {
    "ss_generator_config_path": ("ss_generator.yaml", 5_076),
    "ss_generator_ckpt_path": ("ss_generator.ckpt", 6_690_136_964),
    "slat_generator_config_path": ("slat_generator.yaml", 1_986),
    "slat_generator_ckpt_path": ("slat_generator.ckpt", 4_906_537_684),
    "ss_decoder_config_path": ("ss_decoder.yaml", 244),
    "ss_decoder_ckpt_path": ("ss_decoder.ckpt", 147_609_242),
    "slat_decoder_gs_config_path": ("slat_decoder_gs.yaml", 576),
    "slat_decoder_gs_ckpt_path": ("slat_decoder_gs.ckpt", 171_476_155),
    "slat_decoder_gs_4_config_path": ("slat_decoder_gs_4.yaml", 575),
    "slat_decoder_gs_4_ckpt_path": ("slat_decoder_gs_4.ckpt", 170_269_801),
    "slat_decoder_mesh_config_path": ("slat_decoder_mesh.yaml", 300),
    "slat_decoder_mesh_ckpt_path": ("slat_decoder_mesh.ckpt", 363_726_862),
    "ss_encoder_config_path": ("ss_encoder.yaml", 231),
    "ss_encoder_ckpt_path": ("ss_encoder.ckpt", 119_085_402),
}


def bundle_recipe() -> dict[str, object]:
    """Describe content inputs without coupling identity to download mirrors."""
    return {
        "schema": BUNDLE_RECIPE_SCHEMA,
        "sam3d": {
            "repository": SAM3D_REPOSITORY,
            "revision": SAM3D_MODELSCOPE_REVISION,
            "files": [
                {
                    "path": filename,
                    "size": expected_size,
                    "sha256": expected_sha256,
                }
                for filename, (expected_size, expected_sha256) in sorted(
                    SAM3D_MODELSCOPE_FILES.items()
                )
            ],
        },
        "sam3": {
            "repository": SAM3_REPOSITORY,
            "revision": SAM3_MODELSCOPE_REVISION,
            "path": f"sam3/{SAM3_FILENAME}",
            "size": SAM3_WEIGHT_SIZE,
            "sha256": SAM3_WEIGHT_SHA256,
        },
        "dinov2": {
            "source_ref": DINOV2_REF,
            "source_directory": DINOV2_DIRECTORY,
            "source_ref_marker": SOURCE_REF_MARKER,
            "weight_path": f"cache/torch/hub/checkpoints/{DINOV2_WEIGHT}",
            "weight_size": DINOV2_WEIGHT_SIZE,
            "weight_sha256": DINOV2_WEIGHT_SHA256,
        },
        "moge": {
            "source_ref": MOGE_REF,
            "runtime_path": MOGE_RUNTIME_PATH,
            "weight_size": MOGE_WEIGHT_SIZE,
            "weight_sha256": MOGE_WEIGHT_SHA256,
        },
    }


def bundle_recipe_id() -> str:
    encoded = json.dumps(
        bundle_recipe(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
MAIN_CHECKPOINT_KEYS = (
    "ss_generator_config_path",
    "ss_generator_ckpt_path",
    "slat_generator_config_path",
    "slat_generator_ckpt_path",
    "ss_decoder_config_path",
    "ss_decoder_ckpt_path",
    "slat_decoder_gs_config_path",
    "slat_decoder_gs_ckpt_path",
    "slat_decoder_gs_4_config_path",
    "slat_decoder_gs_4_ckpt_path",
    "slat_decoder_mesh_config_path",
    "slat_decoder_mesh_ckpt_path",
)


class AssetError(RuntimeError):
    """Raised when the offline bundle cannot be prepared or verified safely."""


def validated_transfer_root(path: Path) -> Path:
    transfer_root = path.expanduser().resolve()
    if transfer_root == Path("/"):
        raise AssetError("--transfer-root must not be /")
    return transfer_root


@contextmanager
def preparation_lock(transfer_root: Path):
    transfer_root.mkdir(parents=True, exist_ok=True)
    lock_path = transfer_root / ".prepare_offline_assets.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise AssetError(f"cannot open preparation lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AssetError(
                f"another offline asset preparation is already running: {transfer_root}"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def run(command: list[str], *, label: str) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise AssetError(f"{label}: command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise AssetError(f"{label} failed with exit code {exc.returncode}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_known_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file() or path.is_symlink():
        raise AssetError(f"{label} is missing or is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise AssetError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}: {path}"
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise AssetError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}: {path}"
        )


def download_known_file(
    destination: Path,
    *,
    url: str,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    if destination.is_symlink():
        raise AssetError(f"refusing to replace symlink for {label}: {destination}")
    if destination.exists():
        verify_known_file(
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=label,
        )
        print(f"[reuse] {label}: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.is_symlink():
        raise AssetError(f"refusing to resume through symlink for {label}: {partial}")
    if partial.exists() and partial.stat().st_size == expected_size:
        verify_known_file(
            partial,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"completed partial {label}",
        )
        os.replace(partial, destination)
        print(f"[recovered] {label}: {destination}")
        return
    if partial.exists() and partial.stat().st_size > expected_size:
        raise AssetError(
            f"partial {label} is larger than expected; move it aside and retry: {partial}"
        )
    run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        label=f"download {label}",
    )
    verify_known_file(
        partial,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=f"downloaded {label}",
    )
    os.replace(partial, destination)
    print(f"[downloaded] {label}: {destination}")


def _unquote_yaml_scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def checkpoint_inventory(pipeline: Path) -> dict[str, Path]:
    if not pipeline.is_file():
        raise AssetError(f"pipeline.yaml is missing: {pipeline}")

    values: dict[str, str] = {}
    key_pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*(.*?)\s*$")
    for line in pipeline.read_text(encoding="utf-8").splitlines():
        match = key_pattern.match(line)
        if not match or not match.group(1).endswith(("_config_path", "_ckpt_path")):
            continue
        key = match.group(1)
        if key in values:
            raise AssetError(f"pipeline.yaml contains duplicate key: {key}")
        value = _unquote_yaml_scalar(match.group(2))
        if value and value.lower() not in {"none", "null", "~"}:
            values[key] = value

    missing = [key for key in MAIN_CHECKPOINT_KEYS if not values.get(key)]
    if missing:
        raise AssetError(
            "pipeline.yaml is missing required checkpoint path(s): " + ", ".join(missing)
        )

    root = pipeline.parent.resolve()
    inventory: dict[str, Path] = {}
    for key, value in values.items():
        configured = Path(value)
        if configured.is_absolute():
            raise AssetError(f"{key} must be relative to pipeline.yaml: {configured}")
        resolved = (root / configured).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise AssetError(f"{key} escapes the checkpoint directory: {configured}") from exc
        expected = KNOWN_CHECKPOINT_FILES.get(key)
        if expected is not None and configured.name != expected[0]:
            raise AssetError(
                f"checkpoint filename for {key} does not match pinned revision "
                f"{SAM3D_MODELSCOPE_REVISION}: expected {expected[0]}, "
                f"got {configured.name}"
            )
        inventory[key] = resolved
    return inventory


def main_checkpoint_files(pipeline: Path) -> list[Path]:
    inventory = checkpoint_inventory(pipeline)
    files: list[Path] = []
    for key, resolved in inventory.items():
        if not resolved.is_file() or resolved.is_symlink():
            raise AssetError(f"checkpoint file for {key} is missing: {resolved}")
        expected = KNOWN_CHECKPOINT_FILES.get(key)
        if expected is not None and resolved.stat().st_size != expected[1]:
            raise AssetError(
                f"checkpoint size mismatch for {key}: expected {expected[1]}, "
                f"got {resolved.stat().st_size}: {resolved}"
            )
        files.append(resolved)
    return files


def _moge_entry(text: str) -> re.Match[str]:
    entries = [
        entry
        for entry in MOGE_ENTRY_PATTERN.finditer(text)
        if entry.group("value") in {MOGE_REMOTE_ID, MOGE_RUNTIME_PATH}
    ]
    if len(entries) != 1:
        raise AssetError(
            "pipeline.yaml must contain exactly one "
            "pretrained_model_name_or_path entry for MoGe"
        )
    return entries[0]


def _pipeline_with_local_moge(text: str) -> str:
    entry = _moge_entry(text)
    value = entry.group("value")
    if value == MOGE_RUNTIME_PATH:
        return text
    if value != MOGE_REMOTE_ID:
        raise AssetError(
            "unexpected MoGe pretrained_model_name_or_path: "
            f"{value!r}; expected {MOGE_REMOTE_ID!r} or {MOGE_RUNTIME_PATH!r}"
        )
    replacement = (
        entry.group("prefix")
        + entry.group("quote")
        + MOGE_RUNTIME_PATH
        + entry.group("quote")
        + entry.group("suffix")
    )
    return text[: entry.start()] + replacement + text[entry.end() :]


def patch_pipeline_for_local_moge(pipeline: Path, transfer_root: Path) -> bool:
    if not pipeline.is_file():
        raise AssetError(f"pipeline.yaml is missing: {pipeline}")

    text = pipeline.read_text(encoding="utf-8")
    entry = _moge_entry(text)
    value = entry.group("value")
    if value == MOGE_RUNTIME_PATH:
        return False
    updated = _pipeline_with_local_moge(text)

    backup = transfer_root / "backups" / "pipeline.yaml.before-offline"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.is_symlink():
        raise AssetError(f"refusing to use symlink as pipeline backup: {backup}")
    if not backup.exists():
        shutil.copy2(pipeline, backup)

    mode = pipeline.stat().st_mode & 0o7777
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=pipeline.parent,
            prefix=".pipeline.yaml.",
            delete=False,
        ) as stream:
            stream.write(updated)
            temporary = Path(stream.name)
        os.chmod(temporary, mode)
        os.replace(temporary, pipeline)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return True


def _same_checkpoint_file(left: Path, right: Path, relative: Path) -> bool:
    if relative == Path("pipeline.yaml"):
        try:
            left_text = _pipeline_with_local_moge(left.read_text(encoding="utf-8"))
            right_text = _pipeline_with_local_moge(right.read_text(encoding="utf-8"))
        except (AssetError, UnicodeDecodeError):
            return False
        return left_text == right_text
    return left.stat().st_size == right.stat().st_size and sha256(left) == sha256(right)


def copy_checkpoint_tree(source: Path, destination: Path) -> None:
    source = source.resolve()
    source_pipeline = source / "pipeline.yaml"
    if not source_pipeline.is_file() or source_pipeline.is_symlink():
        raise AssetError(f"SAM 3D checkpoint source has no pipeline.yaml: {source}")
    source_files = [source_pipeline, *checkpoint_inventory(source_pipeline).values()]
    for source_path in source_files:
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise AssetError(f"unsupported checkpoint entry: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.is_symlink():
            raise AssetError(f"refusing to copy checkpoint through symlink: {destination_path}")
        if destination_path.exists():
            if not destination_path.is_file() or not _same_checkpoint_file(
                source_path, destination_path, relative
            ):
                raise AssetError(
                    "refusing to overwrite a different checkpoint file: "
                    f"{destination_path}"
                )
            continue
        shutil.copy2(source_path, destination_path, follow_symlinks=True)


def download_sam3d_checkpoint(download_root: Path) -> Path:
    checkpoint_root = download_root / "checkpoints"
    for filename, (expected_size, expected_sha256) in SAM3D_MODELSCOPE_FILES.items():
        download_known_file(
            checkpoint_root / filename,
            url=f"{SAM3D_MODELSCOPE_BASE_URL}/{filename}",
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            label=f"SAM 3D Objects {filename} from ModelScope",
        )
    return checkpoint_root


def verify_sam3_checkpoint(path: Path) -> None:
    verify_known_file(
        path,
        expected_size=SAM3_WEIGHT_SIZE,
        expected_sha256=SAM3_WEIGHT_SHA256,
        label="SAM 3 checkpoint",
    )


def copy_sam3_checkpoint(source: Path, destination: Path) -> None:
    source_file = source / SAM3_FILENAME if source.is_dir() else source
    verify_sam3_checkpoint(source_file)
    if destination.is_symlink():
        raise AssetError(f"refusing to copy SAM 3 checkpoint through symlink: {destination}")
    if destination.exists():
        verify_sam3_checkpoint(destination)
        print(f"[reuse] SAM 3 checkpoint: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise AssetError(f"SAM 3 checkpoint partial path already exists: {temporary}")
    try:
        shutil.copyfile(source_file, temporary)
        verify_sam3_checkpoint(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[copied] SAM 3 checkpoint: {destination}")


def download_sam3_checkpoint(download_root: Path) -> Path:
    destination = download_root / SAM3_FILENAME
    download_known_file(
        destination,
        url=SAM3_MODELSCOPE_URL,
        expected_size=SAM3_WEIGHT_SIZE,
        expected_sha256=SAM3_WEIGHT_SHA256,
        label="SAM 3 checkpoint from ModelScope",
    )
    return destination


def verify_dinov2_source(source: Path) -> list[Path]:
    required = source / "dinov2" / "hub" / "backbones.py"
    if not required.is_file():
        raise AssetError(f"DINOv2 source is incomplete; missing: {required}")
    marker = source / SOURCE_REF_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != DINOV2_REF:
        raise AssetError(f"DINOv2 source marker does not match pinned ref {DINOV2_REF}: {marker}")
    if (source / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != DINOV2_REF:
            raise AssetError(f"DINOv2 checkout is not pinned to {DINOV2_REF}: {source}")
        run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], label="verify DINOv2 checkout")
        run(["git", "-C", str(source), "fsck", "--connectivity-only", "--no-dangling"], label="verify DINOv2 git objects")
    elif not marker.is_file():
        raise AssetError(f"DINOv2 source has no verifiable ref marker: {source}")
    runtime_files: list[Path] = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if ".git" in relative.parts or path.is_dir():
            continue
        if not path.is_file() or path.is_symlink():
            raise AssetError(f"unsupported DINOv2 source entry: {path}")
        runtime_files.append(path)
    if not runtime_files:
        raise AssetError(f"DINOv2 source contains no runtime files: {source}")
    return runtime_files


def prepare_dinov2_source(destination: Path) -> None:
    if destination.exists():
        verify_dinov2_source(destination)
        print(f"[reuse] DINOv2 source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dinov2-", dir=destination.parent))
    try:
        run(["git", "-C", str(temporary), "init", "--quiet"], label="initialize DINOv2 checkout")
        run(
            ["git", "-C", str(temporary), "fetch", "--depth", "1", DINOV2_REPOSITORY, DINOV2_REF],
            label="download pinned DINOv2 source",
        )
        run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"], label="checkout pinned DINOv2 source")
        (temporary / SOURCE_REF_MARKER).write_text(DINOV2_REF + "\n", encoding="utf-8")
        verify_dinov2_source(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"[downloaded] DINOv2 source: {destination}")


def _manifest_relative_path(storage: Path, path: Path) -> str:
    resolved_storage = storage.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_storage)
    except ValueError as exc:
        raise AssetError(f"manifest file escapes storage directory: {path}") from exc
    if not relative.parts or "\n" in relative.as_posix():
        raise AssetError(f"invalid manifest path: {relative}")
    return relative.as_posix()


def write_checksum_manifest(storage: Path, files: list[Path]) -> Path:
    entries = {
        _manifest_relative_path(storage, path): sha256(path)
        for path in files
    }
    manifest = storage / CHECKSUM_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=manifest.parent,
            prefix=f".{CHECKSUM_MANIFEST}.",
            delete=False,
        ) as stream:
            for relative, digest in sorted(entries.items()):
                stream.write(f"{digest}  {relative}\n")
            temporary = Path(stream.name)
        os.replace(temporary, manifest)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return manifest


def verify_checksum_manifest(storage: Path, files: list[Path]) -> None:
    manifest = storage / CHECKSUM_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise AssetError(f"offline checksum manifest is missing: {manifest}")
    expected = {_manifest_relative_path(storage, path) for path in files}
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise AssetError(f"invalid checksum manifest line {line_number}: {manifest}")
        digest, relative = parts
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in entries:
            raise AssetError(f"unsafe or duplicate checksum path on line {line_number}: {relative}")
        entries[relative] = digest
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        raise AssetError(
            "checksum manifest inventory mismatch; "
            f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
        )
    for relative, expected_digest in sorted(entries.items()):
        path = storage / relative
        if not path.is_file() or path.is_symlink():
            raise AssetError(f"manifest file is missing or not regular: {path}")
        actual_digest = sha256(path)
        if actual_digest != expected_digest:
            raise AssetError(
                f"checksum mismatch for {relative}: expected {expected_digest}, "
                f"got {actual_digest}"
            )


def verify_bundle(transfer_root: Path) -> None:
    storage = transfer_root / "storage"
    pipeline = storage / "hf" / "pipeline.yaml"
    main_files = main_checkpoint_files(pipeline)
    moge_value = _moge_entry(pipeline.read_text(encoding="utf-8")).group("value")
    if moge_value != MOGE_RUNTIME_PATH:
        raise AssetError(
            "pipeline.yaml does not point MoGe to " + MOGE_RUNTIME_PATH
        )
    moge_model = storage / "hf" / "moge" / "model.pt"
    verify_known_file(
        moge_model,
        expected_size=MOGE_WEIGHT_SIZE,
        expected_sha256=MOGE_WEIGHT_SHA256,
        label="MoGe model",
    )
    dinov2_source_files = verify_dinov2_source(
        storage / "cache" / "torch" / "hub" / DINOV2_DIRECTORY
    )
    dinov2_model = (
        storage / "cache" / "torch" / "hub" / "checkpoints" / DINOV2_WEIGHT
    )
    verify_known_file(
        dinov2_model,
        expected_size=DINOV2_WEIGHT_SIZE,
        expected_sha256=DINOV2_WEIGHT_SHA256,
        label="DINOv2 model",
    )
    sam3_model = storage / "sam3" / SAM3_FILENAME
    verify_sam3_checkpoint(sam3_model)
    hf_cache = storage / "cache" / "huggingface"
    if not hf_cache.is_dir():
        raise AssetError(f"Hugging Face cache directory is missing: {hf_cache}")
    verify_checksum_manifest(
        storage,
        [
            pipeline,
            *main_files,
            moge_model,
            dinov2_model,
            *dinov2_source_files,
            sam3_model,
        ],
    )
    print(f"[verified] main checkpoint: {len(main_files)} referenced files and SHA-256")
    print("[verified] MoGe model: exact size and SHA-256")
    print("[verified] DINOv2 source ref and model SHA-256")
    print("[verified] SAM 3 checkpoint exact size and SHA-256")


def prepare(args: argparse.Namespace) -> None:
    transfer_root = validated_transfer_root(args.transfer_root)
    storage = transfer_root / "storage"
    checkpoint_destination = storage / "hf"
    download_root = transfer_root / "download"

    source = args.sam3d_source.expanduser().resolve() if args.sam3d_source else download_root / "checkpoints"
    if args.download_sam3d:
        source = download_sam3d_checkpoint(download_root)
    if source.exists():
        copy_checkpoint_tree(source, checkpoint_destination)
    elif not (checkpoint_destination / "pipeline.yaml").is_file():
        raise AssetError(
            "pipeline.yaml is unavailable; use --download-sam3d to download "
            "it from ModelScope, "
            "or pass --sam3d-source /path/to/checkpoints"
        )

    pipeline = checkpoint_destination / "pipeline.yaml"
    main_checkpoint_files(pipeline)
    changed = patch_pipeline_for_local_moge(pipeline, transfer_root)
    print(
        "[patched] pipeline.yaml now uses local MoGe"
        if changed
        else "[reuse] pipeline.yaml already uses local MoGe"
    )
    (storage / "cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    prepare_dinov2_source(storage / "cache" / "torch" / "hub" / DINOV2_DIRECTORY)
    download_known_file(
        storage / "cache" / "torch" / "hub" / "checkpoints" / DINOV2_WEIGHT,
        url=DINOV2_WEIGHT_URL,
        expected_size=DINOV2_WEIGHT_SIZE,
        expected_sha256=DINOV2_WEIGHT_SHA256,
        label="DINOv2 model",
    )
    download_known_file(
        storage / "hf" / "moge" / "model.pt",
        url=MOGE_WEIGHT_URL,
        expected_size=MOGE_WEIGHT_SIZE,
        expected_sha256=MOGE_WEIGHT_SHA256,
        label="MoGe model",
    )
    sam3_destination = storage / "sam3" / SAM3_FILENAME
    sam3_source = (
        args.sam3_source.expanduser().resolve()
        if args.sam3_source
        else download_root / "sam3" / SAM3_FILENAME
    )
    if args.download_sam3:
        sam3_source = download_sam3_checkpoint(download_root / "sam3")
    if sam3_source.exists():
        copy_sam3_checkpoint(sam3_source, sam3_destination)
    elif not sam3_destination.is_file():
        raise AssetError(
            "SAM 3 checkpoint is unavailable; use --download-sam3 to download "
            "it from ModelScope, "
            "or pass --sam3-source /path/to/sam3.pt"
        )
    verify_sam3_checkpoint(sam3_destination)
    main_files = main_checkpoint_files(pipeline)
    dinov2_source_files = verify_dinov2_source(
        storage / "cache" / "torch" / "hub" / DINOV2_DIRECTORY
    )
    legacy_manifest_files = [
        pipeline,
        *main_files,
        storage / "hf" / "moge" / "model.pt",
        storage / "cache" / "torch" / "hub" / "checkpoints" / DINOV2_WEIGHT,
        *dinov2_source_files,
    ]
    manifest_files = [*legacy_manifest_files, sam3_destination]
    manifest = storage / CHECKSUM_MANIFEST
    if manifest.exists():
        try:
            verify_checksum_manifest(storage, manifest_files)
            print(f"[reuse] verified checksum manifest: {manifest}")
        except AssetError as exc:
            if "checksum manifest inventory mismatch" not in str(exc):
                raise
            # Safely migrate a bundle created before SAM 3 was added. The old
            # manifest must verify exactly before the new checkpoint is added.
            verify_checksum_manifest(storage, legacy_manifest_files)
            write_checksum_manifest(storage, manifest_files)
            print(f"[extended] checksum manifest with SAM 3 checkpoint: {manifest}")
    else:
        write_checksum_manifest(storage, manifest_files)
        print(f"[created] checksum manifest: {manifest}")
    verify_bundle(transfer_root)
    print("\nOffline bundle is ready. Upload it with:")
    quoted_root = shlex.quote(str(transfer_root))
    print(f"  TRANSFER_ROOT={quoted_root}")
    print("  OSS_BUCKET='your-real-shenzhen-bucket-name'")
    print("  : \"${OSS_BUCKET:?set OSS_BUCKET to the real OSS bucket name}\"")
    print("  ossutil cp -u -r \"$TRANSFER_ROOT/storage/\" \"oss://${OSS_BUCKET}/sam3d/\" \\")
    print("    --checkpoint-dir /root/oss-upload-checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the SAM 3D Objects offline bundle on an internet-connected Linux host."
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=Path("/root/sam3d-transfer"),
        help="working directory containing storage/, download/, and backups/",
    )
    parser.add_argument(
        "--sam3d-source",
        type=Path,
        help="existing facebook/sam-3d-objects checkpoints directory",
    )
    parser.add_argument(
        "--download-sam3d",
        action="store_true",
        help="download the pinned SAM 3D Objects checkpoint from ModelScope",
    )
    parser.add_argument(
        "--sam3-source",
        type=Path,
        help="existing facebook/sam3 sam3.pt file or directory containing it",
    )
    parser.add_argument(
        "--download-sam3",
        action="store_true",
        help="download the pinned SAM 3 checkpoint from ModelScope",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform no downloads or edits; verify an already prepared bundle",
    )
    parser.add_argument(
        "--verify-main-only",
        action="store_true",
        help="verify only the main checkpoint already copied under storage/hf",
    )
    parser.add_argument(
        "--verify-sam3-only",
        action="store_true",
        help="verify only storage/sam3/sam3.pt",
    )
    parser.add_argument(
        "--print-recipe-id",
        action="store_true",
        help="print the deterministic offline bundle recipe ID and exit",
    )
    args = parser.parse_args()
    exclusive_modes = sum(
        int(value)
        for value in (
            args.verify_only,
            args.verify_main_only,
            args.verify_sam3_only,
            args.print_recipe_id,
        )
    )
    if exclusive_modes > 1:
        parser.error("verification and recipe modes are mutually exclusive")
    if exclusive_modes and (
        args.download_sam3d
        or args.sam3d_source
        or args.download_sam3
        or args.sam3_source
    ):
        parser.error(
            "verification and recipe modes cannot be combined with download/source options"
        )
    if args.download_sam3d and args.sam3d_source:
        parser.error("--download-sam3d and --sam3d-source are mutually exclusive")
    if args.download_sam3 and args.sam3_source:
        parser.error("--download-sam3 and --sam3-source are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.print_recipe_id:
            print(bundle_recipe_id())
            return 0
        transfer_root = validated_transfer_root(args.transfer_root)
        if args.verify_only:
            verify_bundle(transfer_root)
        elif args.verify_main_only:
            files = main_checkpoint_files(
                transfer_root / "storage" / "hf" / "pipeline.yaml"
            )
            print(f"[verified] main checkpoint: {len(files)} referenced files")
        elif args.verify_sam3_only:
            verify_sam3_checkpoint(
                transfer_root / "storage" / "sam3" / SAM3_FILENAME
            )
            print("[verified] SAM 3 checkpoint exact size and SHA-256")
        else:
            with preparation_lock(transfer_root):
                prepare(args)
    except (AssetError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
