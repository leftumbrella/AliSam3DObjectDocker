from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx


class SegmenterBackendError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class SegmentResult:
    content: bytes
    score: str | None


async def _wait_for_task(task: asyncio.Task[httpx.Response]) -> None:
    """Do not release the outer GPU lock while a cancelled proxy call still runs."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - only completion matters after cancellation.
            break


class SegmenterClient:
    """Loopback-only client for the SAM3 runtime inside the same container."""

    def __init__(
        self,
        base_url: str,
        *,
        startup_timeout: float,
        request_timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = _validate_loopback_url(base_url)
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._transport = transport
        self._load_lock = asyncio.Lock()
        self._loaded = False
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def initialize(self) -> dict[str, Any]:
        if self._loaded:
            return {"initialized": True}

        async with self._load_lock:
            if self._loaded:
                return {"initialized": True}
            deadline = asyncio.get_running_loop().time() + self._startup_timeout
            while True:
                try:
                    response = await self._request("POST", "/initialize")
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    if asyncio.get_running_loop().time() >= deadline:
                        self._load_error = f"{type(exc).__name__}: {exc}"
                        raise SegmenterBackendError(
                            "SAM3 内部服务未能在启动窗口内监听"
                        ) from exc
                    await asyncio.sleep(0.1)

            payload = _response_payload(response)
            if response.is_error:
                detail = _response_detail(payload, response)
                self._load_error = detail
                raise SegmenterBackendError(detail, status_code=response.status_code)
            if payload.get("initialized") is not True:
                self._load_error = "SAM3 内部服务返回了无效的初始化结果"
                raise SegmenterBackendError(self._load_error)
            self._loaded = True
            self._load_error = None
            return payload

    async def ready_status(self) -> dict[str, Any]:
        try:
            response = await self._request("GET", "/readyz", timeout=5.0)
        except httpx.HTTPError as exc:
            self._loaded = False
            return {
                "ready": False,
                "model_loaded": False,
                "last_load_error": f"{type(exc).__name__}: {exc}",
            }
        payload = _response_payload(response)
        if response.is_error:
            self._loaded = False
            return {
                "ready": False,
                "model_loaded": False,
                "last_load_error": _response_detail(payload, response),
            }
        self._loaded = bool(payload.get("model_loaded", payload.get("ready", False)))
        return payload

    async def gpu_status(self) -> dict[str, Any]:
        try:
            response = await self._request("GET", "/gpu", timeout=5.0)
        except httpx.HTTPError as exc:
            return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
        payload = _response_payload(response)
        payload["reachable"] = not response.is_error
        return payload

    async def segment(
        self,
        *,
        image: bytes,
        filename: str,
        content_type: str,
        points: str,
    ) -> SegmentResult:
        response = await self._request(
            "POST",
            "/segment",
            files={"image": (filename, image, content_type)},
            data={"points": points},
        )
        if response.is_error:
            payload = _response_payload(response)
            raise SegmenterBackendError(
                _response_detail(payload, response),
                status_code=response.status_code,
            )
        return SegmentResult(
            content=response.content,
            score=response.headers.get("X-Segment-Score"),
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout or self._request_timeout,
            transport=self._transport,
            trust_env=False,
        ) as client:
            task = asyncio.create_task(client.request(method, path, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await _wait_for_task(task)
                raise


def _validate_loopback_url(raw: str) -> str:
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("SAM3 internal URL must be a plain HTTP loopback origin")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("SAM3 internal URL contains an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("SAM3 internal URL must include a valid port")
    return raw.rstrip("/")


def _response_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_detail(payload: dict[str, Any], response: httpx.Response) -> str:
    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    return f"SAM3 内部服务返回 HTTP {response.status_code}"
