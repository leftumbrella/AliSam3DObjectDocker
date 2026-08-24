from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.types import Receive, Scope, Send

from app.model import ModelManager, ModelNotReadyError, decode_image, decode_mask
from app.settings import Settings

LOGGER = logging.getLogger(__name__)

settings = Settings.from_env()
model_manager = ModelManager(settings)

app = FastAPI(
    title="SAM 3D Objects on Alibaba Cloud FC",
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
        "service": "sam3d-fc",
        "status": "ok",
        "model_loaded": model_manager.loaded,
        "endpoints": [
            "/healthz",
            "/readyz",
            "/gpu",
            "/initialize",
            "/generate",
            "/invoke",
        ],
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    config_present = settings.config_path.is_file()
    return {
        "ready": model_manager.loaded,
        "model_loaded": model_manager.loaded,
        "config_present": config_present,
        "config_path": str(settings.config_path),
        "last_load_error": model_manager.load_error,
    }


@app.get("/gpu")
async def gpu() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_installed": False, "cuda_available": False}

    cuda_available = torch.cuda.is_available()
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


@app.post(
    "/initialize",
    summary="FC Initializer 生命周期回调",
    description="由函数计算在实例启动后调用；业务请求不应手动触发。",
)
async def initialize() -> dict[str, object]:
    try:
        await model_manager.initialize()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("SAM3D 模型初始化失败")
        raise HTTPException(status_code=500, detail="SAM3D 模型初始化失败") from exc

    return {"initialized": True, "config_path": str(settings.config_path)}


@app.post("/invoke")
async def invoke() -> dict[str, object]:
    return {
        "message": "请使用 HTTP 触发器调用 POST /generate；/invoke 不接受二进制推理输入。",
        "model_loaded": model_manager.loaded,
    }


@app.post("/generate", response_class=FileResponse)
async def generate(
    image: Annotated[UploadFile, File(description="RGB/RGBA 输入图片")],
    mask: Annotated[UploadFile, File(description="非零像素表示目标对象的 Mask")],
    seed: Annotated[int, Form(ge=0, le=2_147_483_647)] = 42,
    output_format: Annotated[Literal["ply", "glb"], Form()] = "ply",
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
    output_path = request_dir / f"result.{output_format}"

    try:
        await model_manager.generate(
            image=image_array,
            mask=mask_array,
            seed=seed,
            output_format=output_format,
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

    media_type = "application/octet-stream" if output_format == "ply" else "model/gltf-binary"
    return TemporaryFileResponse(
        path=output_path,
        cleanup_dir=request_dir,
        media_type=media_type,
        filename=f"sam3d-result.{output_format}",
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
