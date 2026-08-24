"""HTTP aggregation contract for the single public FC endpoint."""

from __future__ import annotations

import asyncio
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi import UploadFile

import app.main as gateway
from app.segmenter_client import SegmentResult


class _InitializationProbe:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def enter(self) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1


class _FakeModelManager:
    def __init__(self, probe: _InitializationProbe) -> None:
        self.loaded = False
        self.load_error = None
        self._probe = probe

    async def initialize(self) -> None:
        await self._probe.enter()
        self.loaded = True


class _FakeSegmenterClient:
    def __init__(self, probe: _InitializationProbe) -> None:
        self.loaded = False
        self.load_error = None
        self._probe = probe
        self.segment_calls = 0

    async def initialize(self) -> dict[str, bool]:
        await self._probe.enter()
        self.loaded = True
        return {"initialized": True}

    async def ready_status(self) -> dict[str, object]:
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
    def test_initializer_loads_both_models_and_one_endpoint_proxies_segment(self) -> None:
        probe = _InitializationProbe()
        model = _FakeModelManager(probe)
        segmenter = _FakeSegmenterClient(probe)

        async def exercise() -> None:
            initialized = await gateway.initialize()
            self.assertEqual(
                initialized["models"],
                {"sam3": True, "sam3d": True},
            )
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

        with (
            patch.object(gateway, "model_manager", model),
            patch.object(gateway, "segmenter_client", segmenter),
        ):
            asyncio.run(exercise())

        self.assertEqual(probe.peak, 2)
        self.assertEqual(segmenter.segment_calls, 1)


if __name__ == "__main__":
    unittest.main()
