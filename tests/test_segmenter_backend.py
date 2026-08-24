"""Unit and HTTP contract tests for the FC SAM3 point segmenter."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from segmenter.main import create_app
from segmenter.model import (
    PointPrompt,
    SegmenterManager,
    SegmenterNotReadyError,
    _load_sam3_predictor,
    _tracker_state_dict,
)
from segmenter.settings import Settings


def _settings(root: Path) -> Settings:
    checkpoint = root / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    return Settings(
        sam3_root=root / "sam3-source",
        checkpoint_path=checkpoint,
        max_upload_bytes=1024 * 1024,
        max_image_pixels=1_000_000,
        max_points=8,
        cors_allow_origins=("*",),
    )


class FakePredictor:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.set_image_calls = 0
        self.predict_calls = 0
        self.last_coords: np.ndarray | None = None
        self.last_labels: np.ndarray | None = None
        self.last_multimask: bool | None = None
        self._shape = (0, 0)
        self._active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1
        self._shape = image.shape[:2]

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._counter_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.predict_calls += 1
            self.last_coords = np.asarray(kwargs["point_coords"])
            self.last_labels = np.asarray(kwargs["point_labels"])
            self.last_multimask = bool(kwargs["multimask_output"])
            height, width = self._shape
            count = 3 if self.last_multimask else 1
            masks = np.zeros((count, height, width), dtype=np.bool_)
            masks[-1, : max(1, height // 2), : max(1, width // 2)] = True
            scores = (
                np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
                if count == 3
                else np.asarray([0.8], dtype=np.float32)
            )
            logits = np.zeros((count, 256, 256), dtype=np.float32)
            return masks, scores, logits
        finally:
            with self._counter_lock:
                self._active -= 1


class SegmenterManagerTests(unittest.TestCase):
    def test_tracker_checkpoint_mapping_is_prefix_strict(self) -> None:
        mapped = _tracker_state_dict(
            {
                "model": {
                    "tracker.encoder.weight": "encoder",
                    "tracker.decoder.bias": "decoder",
                    "detector.backbone.vision_backbone.trunk.weight": "visual",
                    "detector.encoder.weight": "excluded-detector",
                    "detector.backbone.language_backbone.weight": "excluded-text",
                }
            }
        )
        self.assertEqual(
            mapped,
            {
                "encoder.weight": "encoder",
                "decoder.bias": "decoder",
                "backbone.vision_backbone.trunk.weight": "visual",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "没有 tracker"):
            _tracker_state_dict(
                {
                    "model": {
                        "detector.backbone.vision_backbone.weight": object(),
                    }
                }
            )
        with self.assertRaisesRegex(RuntimeError, "没有 detector 视觉 backbone"):
            _tracker_state_dict({"model": {"tracker.weight": object()}})

    def test_production_loader_builds_only_tracker_with_strict_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            settings.sam3_root.mkdir()

            class FakeTracker:
                def __init__(self) -> None:
                    self.loaded_state: dict[str, object] | None = None
                    self.strict: bool | None = None
                    self.device: str | None = None
                    self.evaluated = False

                def load_state_dict(
                    self, state: dict[str, object], *, strict: bool
                ) -> None:
                    self.loaded_state = state
                    self.strict = strict

                def to(self, *, device: str) -> FakeTracker:
                    self.device = device
                    return self

                def eval(self) -> FakeTracker:
                    self.evaluated = True
                    return self

            tracker = FakeTracker()
            build_arguments: list[tuple[bool, bool]] = []

            builder_module = ModuleType("sam3.model_builder")

            def build_tracker(
                *, apply_temporal_disambiguation: bool, with_backbone: bool
            ) -> FakeTracker:
                build_arguments.append(
                    (apply_temporal_disambiguation, with_backbone)
                )
                return tracker

            builder_module.build_tracker = build_tracker  # type: ignore[attr-defined]
            predictor_module = ModuleType("sam3.model.sam1_task_predictor")

            class FakeInteractivePredictor:
                def __init__(self, loaded_tracker: FakeTracker) -> None:
                    self.tracker = loaded_tracker

            predictor_module.SAM3InteractiveImagePredictor = (  # type: ignore[attr-defined]
                FakeInteractivePredictor
            )
            sam3_module = ModuleType("sam3")
            sam3_module.__path__ = []  # type: ignore[attr-defined]
            model_module = ModuleType("sam3.model")
            model_module.__path__ = []  # type: ignore[attr-defined]

            load_calls: list[tuple[str, str, bool]] = []

            def load(path: str, *, map_location: str, weights_only: bool) -> object:
                load_calls.append((path, map_location, weights_only))
                return {
                    "model": {
                        "tracker.encoder.weight": "tracker-value",
                        "detector.backbone.vision_backbone.trunk.weight": (
                            "backbone-value"
                        ),
                        "detector.encoder.weight": "excluded-detector-value",
                    }
                }

            fake_torch = SimpleNamespace(
                load=load,
                bfloat16=object(),
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    get_device_properties=lambda _index: SimpleNamespace(major=8),
                ),
                backends=SimpleNamespace(
                    cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                    cudnn=SimpleNamespace(allow_tf32=False),
                ),
            )
            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "sam3": sam3_module,
                    "sam3.model": model_module,
                    "sam3.model_builder": builder_module,
                    "sam3.model.sam1_task_predictor": predictor_module,
                },
            ):
                loaded = _load_sam3_predictor(settings)

            self.assertTrue(callable(loaded.set_image))
            self.assertEqual(build_arguments, [(False, True)])
            self.assertEqual(
                tracker.loaded_state,
                {
                    "encoder.weight": "tracker-value",
                    "backbone.vision_backbone.trunk.weight": "backbone-value",
                },
            )
            self.assertTrue(tracker.strict)
            self.assertEqual(tracker.device, "cuda")
            self.assertTrue(tracker.evaluated)
            self.assertEqual(
                load_calls,
                [(str(settings.checkpoint_path), "cpu", True)],
            )
            self.assertTrue(fake_torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(fake_torch.backends.cudnn.allow_tf32)

    def test_initializer_is_idempotent_and_import_can_be_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = FakePredictor()
            loads = 0

            def loader(_settings: Settings) -> FakePredictor:
                nonlocal loads
                loads += 1
                return predictor

            manager = SegmenterManager(
                _settings(Path(directory)),
                predictor_loader=loader,
            )

            async def initialize_twice() -> None:
                await manager.initialize()
                await manager.initialize()

            asyncio.run(initialize_twice())
            self.assertTrue(manager.loaded)
            self.assertEqual(loads, 1)
            self.assertIsNone(manager.load_error)

    def test_segment_rejects_unwarmed_instances_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loads = 0

            def loader(_settings: Settings) -> FakePredictor:
                nonlocal loads
                loads += 1
                return FakePredictor()

            manager = SegmenterManager(
                _settings(Path(directory)),
                predictor_loader=loader,
            )
            with self.assertRaisesRegex(SegmenterNotReadyError, "FC Initializer"):
                asyncio.run(
                    manager.segment(
                        np.zeros((4, 5, 3), dtype=np.uint8),
                        (PointPrompt(1, 1, 1),),
                    )
                )
            self.assertEqual(loads, 0)

    def test_last_image_embedding_is_reused_and_points_keep_both_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = FakePredictor()
            manager = SegmenterManager(
                _settings(Path(directory)),
                predictor_loader=lambda _settings: predictor,
            )
            first = np.zeros((6, 8, 3), dtype=np.uint8)
            second = first.copy()
            second[0, 0] = 255
            prompts = (PointPrompt(2, 3, 1), PointPrompt(4, 5, 0))

            async def exercise() -> None:
                await manager.initialize()
                mask, score = await manager.segment(first, prompts)
                self.assertEqual(mask.shape, (6, 8))
                self.assertAlmostEqual(score, 0.8, places=5)
                await manager.segment(first.copy(), prompts)
                await manager.segment(second, prompts)

            asyncio.run(exercise())
            self.assertEqual(predictor.set_image_calls, 2)
            self.assertEqual(predictor.predict_calls, 3)
            np.testing.assert_array_equal(predictor.last_labels, [1, 0])
            np.testing.assert_array_equal(predictor.last_coords, [[2, 3], [4, 5]])
            self.assertFalse(predictor.last_multimask)

    def test_gpu_inference_is_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = FakePredictor(delay=0.03)
            manager = SegmenterManager(
                _settings(Path(directory)),
                predictor_loader=lambda _settings: predictor,
            )
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            prompts = (PointPrompt(1, 1, 1),)

            async def exercise() -> None:
                await manager.initialize()
                await asyncio.gather(
                    manager.segment(image, prompts),
                    manager.segment(image, prompts),
                )

            asyncio.run(exercise())
            self.assertEqual(predictor.max_active, 1)
            self.assertEqual(predictor.set_image_calls, 1)


class FakeRuntime:
    def __init__(self, *, loaded: bool = True) -> None:
        self.loaded = loaded
        self.load_error: str | None = None
        self.initialize_calls = 0
        self.last_image: np.ndarray | None = None
        self.last_points: tuple[PointPrompt, ...] = ()

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.loaded = True

    async def segment(
        self,
        image: np.ndarray,
        points: tuple[PointPrompt, ...],
    ) -> tuple[np.ndarray, float]:
        if not self.loaded:
            raise SegmenterNotReadyError("not warmed")
        self.last_image = image
        self.last_points = points
        mask = np.zeros(image.shape[:2], dtype=np.bool_)
        mask[0, 0] = True
        return mask, 0.875


def _oriented_jpeg() -> bytes:
    source = Image.new("RGB", (2, 3), color=(10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    source.save(output, format="JPEG", exif=exif)
    return output.getvalue()


class SegmenterHttpContractTests(unittest.TestCase):
    def test_segment_accepts_positive_and_negative_points_and_returns_binary_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime()
            app = create_app(
                settings=_settings(Path(directory)),
                segmenter=runtime,  # type: ignore[arg-type]
            )
            with TestClient(app) as client:
                response = client.post(
                    "/segment",
                    files={"image": ("oriented.jpg", _oriented_jpeg(), "image/jpeg")},
                    data={
                        "points": json.dumps(
                            [
                                {"x": 2, "y": 1, "label": 1},
                                {"x": 0, "y": 0, "label": 0},
                            ]
                        )
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.headers["content-type"], "image/png")
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(response.headers["x-segment-score"], "0.875000")
            with Image.open(BytesIO(response.content)) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.size, (3, 2))
                self.assertLessEqual(set(mask.getdata()), {0, 255})
            self.assertEqual(runtime.last_image.shape, (2, 3, 3))
            self.assertEqual([point.label for point in runtime.last_points], [1, 0])

    def test_segment_returns_503_before_initializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                settings=_settings(Path(directory)),
                segmenter=FakeRuntime(loaded=False),  # type: ignore[arg-type]
            )
            with TestClient(app) as client:
                response = client.post(
                    "/segment",
                    files={"image": ("image.jpg", _oriented_jpeg(), "image/jpeg")},
                    data={"points": '[{"x":0,"y":0,"label":1}]'},
                )
            self.assertEqual(response.status_code, 503)

    def test_points_require_a_positive_label_and_corrected_image_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                settings=_settings(Path(directory)),
                segmenter=FakeRuntime(),  # type: ignore[arg-type]
            )
            with TestClient(app) as client:
                negative_only = client.post(
                    "/segment",
                    files={"image": ("image.jpg", _oriented_jpeg(), "image/jpeg")},
                    data={"points": '[{"x":0,"y":0,"label":0}]'},
                )
                outside = client.post(
                    "/segment",
                    files={"image": ("image.jpg", _oriented_jpeg(), "image/jpeg")},
                    data={"points": '[{"x":3,"y":1,"label":1}]'},
                )
            self.assertEqual(negative_only.status_code, 422)
            self.assertIn("label=1", negative_only.json()["detail"])
            self.assertEqual(outside.status_code, 422)
            self.assertIn("3x2", outside.json()["detail"])

    def test_initializer_is_the_only_load_path_and_cors_preflight_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = FakeRuntime(loaded=False)
            app = create_app(
                settings=_settings(Path(directory)),
                segmenter=runtime,  # type: ignore[arg-type]
            )
            with TestClient(app) as client:
                initialized = client.post("/initialize")
                preflight = client.options(
                    "/segment",
                    headers={
                        "Origin": "https://demo.example",
                        "Access-Control-Request-Method": "POST",
                    },
                )
            self.assertEqual(initialized.status_code, 200)
            self.assertEqual(runtime.initialize_calls, 1)
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(preflight.headers["access-control-allow-origin"], "*")


if __name__ == "__main__":
    unittest.main()
