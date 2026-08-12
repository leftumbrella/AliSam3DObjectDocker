from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是 true 或 false")


def _read_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须大于 0")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    sam3d_root: Path
    config_path: Path
    compile_model: bool
    max_upload_bytes: int
    max_request_bytes: int
    max_image_pixels: int
    tmp_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        sam3d_root = Path(os.getenv("SAM3D_ROOT", "/opt/sam-3d-objects"))
        config_path = Path(
            os.getenv(
                "SAM3D_CONFIG_PATH",
                "/mnt/nas/sam3d/hf/pipeline.yaml",
            )
        )
        max_upload_mb = _read_positive_int("SAM3D_MAX_UPLOAD_MB", 20)
        max_request_mb = _read_positive_int("SAM3D_MAX_REQUEST_MB", 30)
        if max_upload_mb > max_request_mb:
            raise ValueError("SAM3D_MAX_UPLOAD_MB 不能大于 SAM3D_MAX_REQUEST_MB")
        return cls(
            sam3d_root=sam3d_root,
            config_path=config_path,
            compile_model=_read_bool("SAM3D_COMPILE", False),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_request_bytes=max_request_mb * 1024 * 1024,
            max_image_pixels=_read_positive_int("SAM3D_MAX_IMAGE_PIXELS", 40_000_000),
            tmp_dir=Path(os.getenv("SAM3D_TMP_DIR", "/tmp/sam3d")),
        )
