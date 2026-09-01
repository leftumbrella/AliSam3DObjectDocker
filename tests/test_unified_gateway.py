"""HTTP aggregation contract for the single public FC endpoint."""

from __future__ import annotations

import asyncio
import shutil
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

import app.main as gateway
from app.segmenter_client import SegmentResult
from shared.internal_http import FC_WARMUP_PATH


class _FakeModelManager:
    def __init__(self) -> None:
        self.loaded = False
        self.load_error = None
        self.initialize_calls = 0
        self.generate_calls: list[dict[str, object]] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.loaded = True

    async def generate(self, **kwargs: object) -> None:
        self.generate_calls.append(kwargs)
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        output_path.write_bytes(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")


class _FakeSegmenterClient:
    def __init__(self) -> None:
        self.loaded = False
        self.load_error = None
        self.ready_calls = 0
        self.segment_calls = 0

    async def ready_status(self) -> dict[str, object]:
        self.ready_calls += 1
        self.loaded = True
        return {
            "ready": self.loaded,
            "model_loaded": self.loaded,
            "checkpoint_present": True,
            "checkpoint_path": "/mnt/nas/sam3d/sam3/sam3.pt",
        }

    async def gpu_status(self) -> dict[str, object]:
        return {"cuda_available": True}

    async def segment(self, **_kwargs: object) -> SegmentResult:
        self.segment_calls += 1
        return SegmentResult(content=b"png", score="0.900000")


class UnifiedGatewayTests(unittest.TestCase):
    def test_private_warmup_syncs_models_and_business_routes_remain_public(self) -> None:
        model = _FakeModelManager()
        segmenter = _FakeSegmenterClient()

        async def exercise() -> None:
            ready = await gateway.readyz()
            self.assertTrue(ready["ready"])
            self.assertTrue(ready["models"]["sam3"]["model_loaded"])
            self.assertTrue(ready["models"]["sam3d"]["model_loaded"])

            upload = UploadFile(
                file=BytesIO(b"image"),
                filename="input.png",
                headers={"content-type": "image/png"},
            )
            response = await gateway.segment(
                image=upload,
                points='[{"x":1,"y":2,"label":1}]',
            )
            self.assertEqual(response.body, b"png")
            self.assertEqual(response.headers["x-segment-score"], "0.900000")

            image_payload = BytesIO()
            Image.new("RGB", (2, 2), "green").save(image_payload, format="PNG")
            mask_payload = BytesIO()
            Image.new("L", (2, 2), 255).save(mask_payload, format="PNG")
            model_response = await gateway.generate(
                image=UploadFile(file=BytesIO(image_payload.getvalue()), filename="input.png"),
                mask=UploadFile(file=BytesIO(mask_payload.getvalue()), filename="mask.png"),
                seed=42,
            )
            try:
                self.assertEqual(model_response.media_type, "model/gltf-binary")
                self.assertEqual(model_response.filename, "sam3d-result.glb")
                self.assertEqual(Path(model_response.path).suffix, ".glb")
            finally:
                shutil.rmtree(model_response._cleanup_dir, ignore_errors=True)

        with (
            patch.object(gateway, "model_manager", model),
            patch.object(gateway, "segmenter_client", segmenter),
        ):
            with TestClient(gateway.app, client=("127.0.0.1", 50000)) as client:
                removed = client.post("/initialize")
                warmed = client.post(FC_WARMUP_PATH)
                openapi_paths = client.get("/openapi.json").json()["paths"]
            with TestClient(gateway.app) as external_client:
                blocked = external_client.post(FC_WARMUP_PATH)
            asyncio.run(exercise())

        self.assertEqual(removed.status_code, 404)
        self.assertEqual(blocked.status_code, 404)
        self.assertEqual(warmed.status_code, 200, warmed.text)
        self.assertEqual(warmed.json()["model"], "sam3d")
        self.assertNotIn("/initialize", openapi_paths)
        self.assertNotIn(FC_WARMUP_PATH, openapi_paths)
        self.assertEqual(model.initialize_calls, 1)
        self.assertGreaterEqual(segmenter.ready_calls, 2)
        self.assertEqual(segmenter.segment_calls, 1)
        self.assertEqual(len(model.generate_calls), 1)
        self.assertNotIn("output_format", model.generate_calls[0])


if __name__ == "__main__":
    unittest.main()
