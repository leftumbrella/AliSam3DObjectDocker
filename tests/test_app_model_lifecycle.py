"""Unit tests for the SAM3D FC model lifecycle contract."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image

from app.model import ModelManager, ModelNotReadyError, decode_image
from app.settings import Settings


def _settings(root: Path) -> Settings:
    return Settings(
        sam3d_root=root / "source",
        config_path=root / "pipeline.yaml",
        compile_model=False,
        max_upload_bytes=1024 * 1024,
        max_request_bytes=2 * 1024 * 1024,
        max_image_pixels=1_000_000,
        tmp_dir=root / "tmp",
        cors_allow_origins=("*",),
    )


class Sam3DModelLifecycleTests(unittest.TestCase):
    def test_initializer_holds_the_shared_gpu_lock_while_loading(self) -> None:
        events: list[str] = []

        class RecordingLock:
            def acquire(self) -> RecordingLock:
                events.append("acquire")
                return self

            def __enter__(self) -> None:
                events.append("enter")

            def __exit__(self, *_args: object) -> None:
                events.append("exit")

        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(  # type: ignore[arg-type]
                _settings(Path(directory)),
                gpu_lock=RecordingLock(),
            )
            with patch.object(
                manager,
                "_load_model_locked",
                side_effect=lambda: events.append("load"),
            ):
                manager._load_model()

        self.assertEqual(events, ["acquire", "enter", "load", "exit"])

    def test_generate_never_initializes_model_from_a_business_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(_settings(Path(directory)))
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            mask = np.ones((4, 5), dtype=np.bool_)
            with (
                patch.object(manager, "initialize", new=AsyncMock()) as initialize,
                self.assertRaisesRegex(ModelNotReadyError, "FC Initializer"),
            ):
                asyncio.run(
                    manager.generate(
                        image=image,
                        mask=mask,
                        seed=42,
                        output_format="ply",
                        output_path=Path(directory) / "result.ply",
                    )
                )
            initialize.assert_not_awaited()

    def test_decode_image_applies_exif_orientation_before_inference(self) -> None:
        source = Image.new("RGB", (2, 3), color=(30, 60, 90))
        exif = Image.Exif()
        exif[274] = 6
        payload = BytesIO()
        source.save(payload, format="JPEG", exif=exif)

        decoded = decode_image(payload.getvalue(), max_pixels=100)

        self.assertEqual(decoded.shape, (2, 3, 3))
        self.assertEqual(decoded.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
