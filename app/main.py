from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send

from app.model import ModelManager, ModelNotReadyError, decode_image, decode_mask
from app.segmenter_client import SegmenterBackendError, SegmenterClient
from app.settings import Settings
from shared.internal_http import FC_WARMUP_PATH, require_loopback

LOGGER = logging.getLogger(__name__)

settings = Settings.from_env()
gpu_inference_lock = asyncio.Lock()
model_manager = ModelManager(settings, inference_lock=gpu_inference_lock)
segmenter_client = SegmenterClient(
    settings.sam3_internal_url,
    request_timeout=settings.sam3_internal_request_timeout,
)

app = FastAPI(
    title="SAM 3 + SAM 3D Objects on Alibaba Cloud FC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Segment-Score"],
)


class TemporaryFileResponse(FileResponse):
    def __init__(
        self,
        *,
        path: Path,
        cleanup_dir: Path,
        media_type: str,
        filename: str,
    ) -> None:
        super().__init__(path=path, media_type=media_type, filename=filename)
        self._cleanup_dir = cleanup_dir

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            shutil.rmtree(self._cleanup_dir, ignore_errors=True)


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "service": "sam3-sam3d-fc",
        "status": "ok",
        "model_loaded": model_manager.loaded and segmenter_client.loaded,
        "endpoints": [
            "/healthz",
            "/readyz",
            "/gpu",
            "/segment",
            "/generate",
            "/invoke",
        ],
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    sam3_status = await segmenter_client.ready_status()
    config_present = settings.config_path.is_file()
    sam3d_status = {
        "ready": model_manager.loaded,
        "model_loaded": model_manager.loaded,
        "config_present": config_present,
        "config_path": str(settings.config_path),
        "last_load_error": model_manager.load_error,
    }
    sam3_ready = bool(
        sam3_status.get("ready", sam3_status.get("model_loaded", False))
    )
    ready = model_manager.loaded and sam3_ready
    return {
        "ready": ready,
        "model_loaded": ready,
        "config_present": config_present,
        "config_path": str(settings.config_path),
        "checkpoint_present": bool(sam3_status.get("checkpoint_present", False)),
        "models": {
            "sam3": sam3_status,
            "sam3d": sam3d_status,
        },
        "last_load_error": model_manager.load_error or segmenter_client.load_error,
    }


@app.get("/gpu")
async def gpu() -> dict[str, object]:
    sam3_status = await segmenter_client.gpu_status()
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "cuda_available": False,
            "runtimes": {"sam3": sam3_status},
        }

    cuda_available = torch.cuda.is_available()
    result: dict[str, object] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "runtimes": {
            "sam3": sam3_status,
            "sam3d": {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": cuda_available,
            },
        },
    }
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        result.update(
            {
                "memory_free_bytes": free_bytes,
                "memory_total_bytes": total_bytes,
                "sam3d_memory_allocated_bytes": torch.cuda.memory_allocated(0),
                "sam3d_memory_reserved_bytes": torch.cuda.memory_reserved(0),
            }
        )
    return result


@app.post(FC_WARMUP_PATH, include_in_schema=False)
async def warmup_sam3d(request: Request) -> dict[str, object]:
    require_loopback(request)
    sam3_status = await segmenter_client.ready_status()
    sam3_ready = bool(
        sam3_status.get("ready", sam3_status.get("model_loaded", False))
    )
    if not sam3_ready:
        detail = (
            sam3_status.get("last_load_error")
            or segmenter_client.load_error
            or "SAM3 模型尚未预热"
        )
        raise HTTPException(status_code=503, detail=str(detail))

    try:
        await model_manager.initialize()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("SAM3D 模型预热失败")
        raise HTTPException(status_code=500, detail="SAM3D 模型预热失败") from exc

    return {
        "warmed": True,
        "model": "sam3d",
        "config_path": str(settings.config_path),
    }


@app.post("/invoke")
async def invoke() -> dict[str, object]:
    return {
        "message": "请使用同一个 HTTP 触发器调用 POST /segment 或 POST /generate。",
        "model_loaded": model_manager.loaded and segmenter_client.loaded,
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
    if not segmenter_client.loaded:
        raise HTTPException(
            status_code=503,
            detail="SAM3 模型尚未由 FC Initializer 预热；请检查实例初始化日志",
        )
    image_data = await _read_upload(image, "image")
    try:
        async with gpu_inference_lock:
            result = await segmenter_client.segment(
                image=image_data,
                filename=Path(image.filename or "image-upload").name,
                content_type=image.content_type or "application/octet-stream",
                points=points,
            )
    except SegmenterBackendError as exc:
        if exc.status_code >= 500:
            LOGGER.error("SAM3 内部服务失败：%s", exc)
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    headers = {"Cache-Control": "no-store"}
    if result.score is not None:
        headers["X-Segment-Score"] = result.score
    return Response(content=result.content, media_type="image/png", headers=headers)


@app.post("/generate", response_class=FileResponse)
async def generate(
    image: Annotated[UploadFile, File(description="RGB/RGBA 输入图片")],
    mask: Annotated[UploadFile, File(description="非零像素表示目标对象的 Mask")],
    seed: Annotated[int, Form(ge=0, le=2_147_483_647)] = 42,
) -> FileResponse:
    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="SAM3D 模型尚未由 FC Initializer 预热；请检查实例初始化日志",
        )
    image_data = await _read_upload(image, "image")
    mask_data = await _read_upload(mask, "mask")
    if len(image_data) + len(mask_data) > settings.max_request_bytes:
        max_mb = settings.max_request_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"image 与 mask 的总大小不能超过 {max_mb} MB",
        )

    try:
        image_array = await run_in_threadpool(
            decode_image,
            image_data,
            settings.max_image_pixels,
        )
        image_size = (image_array.shape[1], image_array.shape[0])
        mask_array = await run_in_threadpool(
            decode_mask,
            mask_data,
            image_size,
            settings.max_image_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not mask_array.any():
        raise HTTPException(status_code=422, detail="Mask 不能为空")

    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=settings.tmp_dir))
    output_path = request_dir / "result.glb"

    try:
        await model_manager.generate(
            image=image_array,
            mask=mask_array,
            seed=seed,
            output_path=output_path,
        )
    except asyncio.CancelledError:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except ModelNotReadyError as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        LOGGER.exception("SAM3D 推理失败")
        raise HTTPException(status_code=500, detail="SAM3D 推理失败") from exc

    return TemporaryFileResponse(
        path=output_path,
        cleanup_dir=request_dir,
        media_type="model/gltf-binary",
        filename="sam3d-result.glb",
    )


async def _read_upload(upload: UploadFile, field_name: str) -> bytes:
    try:
        data = await upload.read(settings.max_upload_bytes + 1)
    finally:
        await upload.close()

    if not data:
        raise HTTPException(status_code=422, detail=f"{field_name} 文件不能为空")
    if len(data) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"{field_name} 文件不能超过 {max_mb} MB")
    return data
