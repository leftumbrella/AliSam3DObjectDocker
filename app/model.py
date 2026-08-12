from __future__ import annotations

import asyncio
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.settings import Settings

LOGGER = logging.getLogger(__name__)


async def _wait_for_worker(worker: asyncio.Task[None]) -> None:
    """请求取消后继续等待后台线程结束，避免过早释放 GPU 串行锁。"""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break


class ModelNotReadyError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def initialize(self) -> None:
        if self.loaded:
            return

        async with self._load_lock:
            if self.loaded:
                return
            worker = asyncio.create_task(asyncio.to_thread(self._load_model))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_worker(worker)
                if not worker.cancelled():
                    error = worker.exception()
                    if error is not None:
                        self._load_error = f"{type(error).__name__}: {error}"
                raise
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise

    async def generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed: int,
        output_format: str,
        output_path: Path,
    ) -> None:
        await self.initialize()

        async with self._inference_lock:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self._generate,
                    image,
                    mask,
                    seed,
                    output_format,
                    output_path,
                )
            )
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_worker(worker)
                if not worker.cancelled():
                    error = worker.exception()
                    if error is not None:
                        LOGGER.error(
                            "请求取消后，SAM3D 后台推理失败",
                            exc_info=(type(error), error, error.__traceback__),
                        )
                raise

    def _load_model(self) -> None:
        config_path = self._settings.config_path
        if not config_path.is_file():
            raise ModelNotReadyError(
                f"未找到模型配置：{config_path}。请先把 checkpoint 挂载到容器。"
            )

        notebook_path = self._settings.sam3d_root / "notebook"
        if not notebook_path.is_dir():
            raise ModelNotReadyError(f"未找到 SAM3D notebook 目录：{notebook_path}")

        import torch

        if not torch.cuda.is_available():
            raise ModelNotReadyError("未检测到可用的 CUDA GPU")

        notebook_path_text = str(notebook_path)
        if notebook_path_text not in sys.path:
            sys.path.insert(0, notebook_path_text)

        from inference import Inference

        LOGGER.info("开始加载 SAM3D 模型，配置文件：%s", config_path)
        self._model = Inference(
            str(config_path),
            compile=self._settings.compile_model,
        )
        self._load_error = None
        LOGGER.info("SAM3D 模型加载完成")

    def _generate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed: int,
        output_format: str,
        output_path: Path,
    ) -> None:
        if self._model is None:
            raise ModelNotReadyError("SAM3D 模型尚未初始化")

        output = self._model(image, mask, seed=seed)
        if output_format == "ply":
            gaussian_splat = output.get("gs")
            if gaussian_splat is None:
                raise RuntimeError("SAM3D 推理结果中没有 gs，无法导出 PLY")
            gaussian_splat.save_ply(str(output_path))
        elif output_format == "glb":
            mesh = output.get("glb")
            if mesh is None:
                raise RuntimeError("SAM3D 推理结果中没有 glb，无法导出 GLB")
            mesh.export(str(output_path), file_type="glb")
        else:
            raise ValueError(f"不支持的输出格式：{output_format}")

        if not output_path.is_file():
            raise RuntimeError(f"推理完成，但未生成输出文件：{output_path}")


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise ValueError(f"图片像素数不能超过 {max_pixels}")
            source.load()
            return np.array(source.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解析上传的图片") from exc


def decode_mask(data: bytes, expected_size: tuple[int, int], max_pixels: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise ValueError(f"Mask 像素数不能超过 {max_pixels}")
            if source.size != expected_size:
                raise ValueError(
                    f"图片尺寸为 {expected_size[0]}x{expected_size[1]}，"
                    f"Mask 尺寸为 {source.width}x{source.height}"
                )
            source.load()
            mask = np.array(source, copy=True)
            if mask.ndim == 3:
                mask = mask[..., -1]
            if mask.ndim != 2:
                raise ValueError("Mask 必须是单通道、RGB 或 RGBA 图片")
            return mask > 0
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解析上传的 Mask") from exc
