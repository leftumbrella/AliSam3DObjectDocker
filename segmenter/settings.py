from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_positive_int(name: str, default: int, *, fallback: str | None = None) -> int:
    raw = os.getenv(name)
    if raw is None and fallback is not None:
        raw = os.getenv(fallback)
    value = int(raw if raw is not None else default)
    if value <= 0:
        raise ValueError(f"环境变量 {name} 必须大于 0")
    return value


def _read_origins(name: str, default: str = "*") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    origins = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not origins:
        raise ValueError(f"环境变量 {name} 至少需要一个 Origin")
    return origins


@dataclass(frozen=True, slots=True)
class Settings:
    sam3_root: Path
    checkpoint_path: Path
    max_upload_bytes: int
    max_image_pixels: int
    max_points: int
    cors_allow_origins: tuple[str, ...]
    gpu_lock_path: Path = Path("/tmp/sam3d-gpu.lock")

    @classmethod
    def from_env(cls) -> Settings:
        max_upload_mb = _read_positive_int(
            "SAM3_MAX_UPLOAD_MB",
            20,
            fallback="SAM3D_MAX_UPLOAD_MB",
        )
        return cls(
            sam3_root=Path(os.getenv("SAM3_ROOT", "/opt/sam3")),
            checkpoint_path=Path(
                os.getenv(
                    "SAM3_CHECKPOINT_PATH",
                    "/mnt/nas/sam3d/sam3/sam3.pt",
                )
            ),
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_image_pixels=_read_positive_int(
                "SAM3_MAX_IMAGE_PIXELS",
                40_000_000,
                fallback="SAM3D_MAX_IMAGE_PIXELS",
            ),
            max_points=_read_positive_int("SAM3_MAX_POINTS", 64),
            cors_allow_origins=_read_origins("CORS_ALLOW_ORIGINS"),
            gpu_lock_path=Path(
                os.getenv("GPU_LOCK_PATH", "/tmp/sam3d-gpu.lock")
            ),
        )
