"""Loopback proxy contract for the SAM3 runtime inside the combined image."""

from __future__ import annotations

import unittest

import httpx

from app.segmenter_client import SegmenterBackendError, SegmenterClient


class SegmenterClientTests(unittest.IsolatedAsyncioTestCase):
    def test_only_plain_loopback_origins_are_accepted(self) -> None:
        for url in (
            "https://127.0.0.1:9001",
            "http://example.com:9001",
            "http://127.0.0.1:9001/path",
            "http://user@127.0.0.1:9001",
            "http://127.0.0.1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SegmenterClient(url, startup_timeout=1, request_timeout=1)

    async def test_initialize_status_and_segment_share_the_internal_runtime(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/initialize":
                return httpx.Response(200, json={"initialized": True})
            if request.url.path == "/readyz":
                return httpx.Response(
                    200,
                    json={"ready": True, "model_loaded": True},
                )
            if request.url.path == "/gpu":
                return httpx.Response(200, json={"cuda_available": True})
            if request.url.path == "/segment":
                return httpx.Response(
                    200,
                    content=b"png",
                    headers={"X-Segment-Score": "0.875"},
                )
            return httpx.Response(404)

        client = SegmenterClient(
            "http://127.0.0.1:9001",
            startup_timeout=1,
            request_timeout=1,
            transport=httpx.MockTransport(handler),
        )

        await client.initialize()
        ready = await client.ready_status()
        gpu = await client.gpu_status()
        result = await client.segment(
            image=b"image",
            filename="input.png",
            content_type="image/png",
            points='[{"x":1,"y":2,"label":1}]',
        )

        self.assertTrue(client.loaded)
        self.assertTrue(ready["ready"])
        self.assertTrue(gpu["reachable"])
        self.assertEqual(result.content, b"png")
        self.assertEqual(result.score, "0.875")
        self.assertEqual(
            requests,
            [
                ("POST", "/initialize"),
                ("GET", "/readyz"),
                ("GET", "/gpu"),
                ("POST", "/segment"),
            ],
        )

    async def test_initialize_fails_closed_on_a_malformed_success_body(self) -> None:
        client = SegmenterClient(
            "http://127.0.0.1:9001",
            startup_timeout=1,
            request_timeout=1,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={"status": "ok"})
            ),
        )

        with self.assertRaisesRegex(SegmenterBackendError, "无效的初始化结果"):
            await client.initialize()
        self.assertFalse(client.loaded)


if __name__ == "__main__":
    unittest.main()
