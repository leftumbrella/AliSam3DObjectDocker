from __future__ import annotations

import json
import logging
import math
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from shared.internal_http import FC_WARMUP_PATH, require_loopback
from segmenter.model import (
    PointPrompt,
    SegmenterManager,
    SegmenterNotReadyError,
    decode_image,
    encode_binary_mask,
)
from segmenter.settings import Settings

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    segmenter: SegmenterManager | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    runtime_segmenter = segmenter or SegmenterManager(runtime_settings)

    app = FastAPI(
        title="SAM 3 Point Segmenter on Alibaba Cloud FC",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(runtime_settings.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Segment-Score"],
    )

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "sam3-segmenter-fc",
            "status": "ok",
            "model_loaded": runtime_segmenter.loaded,
            "endpoints": ["/healthz", "/readyz", "/gpu", "/segment"],
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, object]:
        checkpoint_present = runtime_settings.checkpoint_path.is_file()
        return {
            "ready": runtime_segmenter.loaded,
            "model_loaded": runtime_segmenter.loaded,
            "checkpoint_present": checkpoint_present,
            "checkpoint_path": str(runtime_settings.checkpoint_path),
            "last_load_error": runtime_segmenter.load_error,
        }

    @app.get("/gpu")
    async def gpu() -> dict[str, object]:
        try:
            import torch
        except ImportError:
            return {"torch_installed": False, "cuda_available": False}

        cuda_available = torch.cuda.is_available()
        result: dict[str, object] = {
            "torch_installed": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        }
        if cuda_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            result.update(
                {
                    "memory_free_bytes": free_bytes,
                    "memory_total_bytes": total_bytes,
                    "memory_allocated_bytes": torch.cuda.memory_allocated(0),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(0),
                }
            )
        return result

    @app.post(FC_WARMUP_PATH, include_in_schema=False)
    async def warmup(request: Request) -> dict[str, object]:
        require_loopback(request)
        try:
            await runtime_segmenter.initialize()
        except SegmenterNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("SAM3 点选分割模型初始化失败")
            raise HTTPException(status_code=500, detail="SAM3 点选分割模型初始化失败") from exc
        return {
            "warmed": True,
            "model": "sam3",
            "checkpoint_path": str(runtime_settings.checkpoint_path),
        }

    @app.post(
        "/segment",
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}},
    )
    async def segment(
        image: Annotated[UploadFile, File(description="RGB/RGBA 输入图片")],
        points: Annotated[
            str,
            Form(description='点选 JSON，例如 [{"x":10,"y":20,"label":1}]'),
        ],
    ) -> Response:
        image_data = await _read_upload(
            image,
            max_bytes=runtime_settings.max_upload_bytes,
        )
        try:
            image_array = await run_in_threadpool(
                decode_image,
                image_data,
                runtime_settings.max_image_pixels,
            )
            prompts = _parse_points(
                points,
                width=image_array.shape[1],
                height=image_array.shape[0],
                max_points=runtime_settings.max_points,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            mask, score = await runtime_segmenter.segment(image_array, prompts)
            png = await run_in_threadpool(encode_binary_mask, mask)
        except SegmenterNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("SAM3 点选分割推理失败")
            raise HTTPException(status_code=500, detail="SAM3 点选分割推理失败") from exc

        return Response(
            content=png,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Segment-Score": f"{score:.6f}",
            },
        )

    return app


def _parse_points(
    raw: str,
    *,
    width: int,
    height: int,
    max_points: int,
) -> tuple[PointPrompt, ...]:
    if len(raw.encode("utf-8")) > 64 * 1024:
        raise ValueError("points JSON 不能超过 64 KB")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("points 必须是有效的 JSON 数组") from exc
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("points 必须是非空 JSON 数组")
    if len(decoded) > max_points:
        raise ValueError(f"点选数量不能超过 {max_points}")

    prompts: list[PointPrompt] = []
    has_positive = False
    for index, item in enumerate(decoded):
        if not isinstance(item, dict):
            # This is request validation and must become an HTTP 422 via ValueError.
            raise ValueError(f"points[{index}] 必须是对象")  # noqa: TRY004
        x = item.get("x")
        y = item.get("y")
        label = item.get("label")
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
        ):
            raise ValueError(f"points[{index}].x 必须是有限数字")
        if (
            isinstance(y, bool)
            or not isinstance(y, (int, float))
            or not math.isfinite(y)
        ):
            raise ValueError(f"points[{index}].y 必须是有限数字")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"points[{index}].label 必须是 0 或 1")
        if not 0 <= float(x) < width or not 0 <= float(y) < height:
            raise ValueError(
                f"points[{index}] 超出图片范围 {width}x{height}"
            )
        has_positive = has_positive or label == 1
        prompts.append(PointPrompt(x=float(x), y=float(y), label=label))

    if not has_positive:
        raise ValueError("至少需要一个加选点（label=1）")
    return tuple(prompts)


async def _read_upload(upload: UploadFile, *, max_bytes: int) -> bytes:
    try:
        data = await upload.read(max_bytes + 1)
    finally:
        await upload.close()
    if not data:
        raise HTTPException(status_code=422, detail="image 文件不能为空")
    if len(data) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"image 文件不能超过 {max_mb} MB")
    return data


app = create_app()
