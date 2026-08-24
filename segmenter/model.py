from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from segmenter.settings import Settings
from shared.gpu_lock import InterProcessGpuLock

LOGGER = logging.getLogger(__name__)


async def _wait_for_worker(worker: asyncio.Task[Any]) -> None:
    """Wait for a GPU thread after request cancellation before releasing its lock."""
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break


class SegmenterNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PointPrompt:
    x: float
    y: float
    label: int


PredictorLoader = Callable[[Settings], Any]


class _CudaAutocastPredictor:
    """Keep the official interactive predictor inside CUDA BF16 inference contexts."""

    def __init__(self, predictor: Any, torch_module: Any) -> None:
        self._predictor = predictor
        self._torch = torch_module

    def set_image(self, image: np.ndarray) -> None:
        with self._torch.inference_mode(), self._torch.autocast(
            "cuda", dtype=self._torch.bfloat16
        ):
            self._predictor.set_image(image)

    def predict(self, **kwargs: Any) -> Any:
        with self._torch.inference_mode(), self._torch.autocast(
            "cuda", dtype=self._torch.bfloat16
        ):
            return self._predictor.predict(**kwargs)


class SegmenterManager:
    def __init__(
        self,
        settings: Settings,
        *,
        predictor_loader: PredictorLoader | None = None,
        inference_lock: asyncio.Lock | None = None,
        gpu_lock: InterProcessGpuLock | None = None,
    ) -> None:
        self._settings = settings
        self._predictor_loader = predictor_loader or _load_sam3_predictor
        self._predictor: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = inference_lock or asyncio.Lock()
        self._gpu_lock = gpu_lock or InterProcessGpuLock(settings.gpu_lock_path)
        self._load_error: str | None = None
        self._last_image_digest: str | None = None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def last_image_digest(self) -> str | None:
        return self._last_image_digest

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

    async def segment(
        self,
        image: np.ndarray,
        points: Sequence[PointPrompt],
    ) -> tuple[np.ndarray, float]:
        if not self.loaded:
            raise SegmenterNotReadyError(
                "SAM3 模型尚未由 FC Initializer 预热；请检查实例初始化日志"
            )

        async with self._inference_lock:
            worker = asyncio.create_task(
                asyncio.to_thread(self._segment, image, tuple(points))
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                await _wait_for_worker(worker)
                raise

    def _load_model(self) -> None:
        predictor = self._predictor_loader(self._settings)
        if not callable(getattr(predictor, "set_image", None)):
            raise TypeError("SAM3 predictor 缺少 set_image 方法")
        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("SAM3 predictor 缺少 predict 方法")
        self._predictor = predictor
        self._last_image_digest = None
        self._load_error = None
        LOGGER.info("SAM3 点选分割模型加载完成")

    def _segment(
        self,
        image: np.ndarray,
        points: tuple[PointPrompt, ...],
    ) -> tuple[np.ndarray, float]:
        predictor = self._predictor
        if predictor is None:
            raise SegmenterNotReadyError("SAM3 模型尚未初始化")

        with self._gpu_lock.acquire():
            image_digest = _image_digest(image)
            if image_digest != self._last_image_digest:
                predictor.set_image(image)
                self._last_image_digest = image_digest

            point_coords = np.asarray(
                [[point.x, point.y] for point in points],
                dtype=np.float32,
            )
            point_labels = np.asarray(
                [point.label for point in points],
                dtype=np.int32,
            )
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=len(points) == 1,
                return_logits=False,
                normalize_coords=True,
            )
            return _select_best_mask(masks, scores, image.shape[:2])


def _load_sam3_predictor(settings: Settings) -> Any:
    checkpoint_path = settings.checkpoint_path
    if not checkpoint_path.is_file():
        raise SegmenterNotReadyError(
            f"未找到 SAM3 checkpoint：{checkpoint_path}。请先把模型挂载到容器。"
        )
    if not settings.sam3_root.is_dir():
        raise SegmenterNotReadyError(f"未找到 SAM3 源码目录：{settings.sam3_root}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    root_text = str(settings.sam3_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    import torch

    if not torch.cuda.is_available():
        raise SegmenterNotReadyError("未检测到可用的 CUDA GPU")

    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
    from sam3.model_builder import build_tracker

    LOGGER.info("开始加载 SAM3 点选分割模型：%s", checkpoint_path)
    # The interactive predictor calls ``tracker.forward_image`` itself.  A tracker
    # embedded in the full SAM3 image model borrows detector features and therefore
    # has no backbone, but this standalone service must own the shared visual
    # backbone.  Loading only the tracker and that backbone still avoids constructing
    # the text encoder, detector transformer, and segmentation head.
    tracker = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=True,
    )
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    tracker_state = _tracker_state_dict(checkpoint)
    tracker.load_state_dict(tracker_state, strict=True)
    del checkpoint, tracker_state
    gc.collect()

    tracker.to(device="cuda").eval()
    device_properties = torch.cuda.get_device_properties(0)
    if device_properties.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = SAM3InteractiveImagePredictor(tracker)
    return _CudaAutocastPredictor(predictor, torch)


def _tracker_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Build the strict state dict for a standalone interactive tracker.

    Official checkpoints store the interactive decoder under ``tracker.*`` but
    store the shared visual backbone under ``detector.backbone.*``.  A tracker
    built with ``with_backbone=True`` expects that latter subtree without the
    leading ``detector.`` prefix.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError("SAM3 checkpoint 顶层必须是字典")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("SAM3 checkpoint 的 model 字段必须是字典")

    tracker_prefix = "tracker."
    tracker_state = {
        key[len(tracker_prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(tracker_prefix)
    }
    if not tracker_state:
        raise RuntimeError("SAM3 checkpoint 中没有 tracker 权重")

    detector_backbone_prefix = "detector.backbone.vision_backbone."
    tracker_backbone_prefix = "backbone.vision_backbone."
    backbone_state = {
        tracker_backbone_prefix + key[len(detector_backbone_prefix) :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(detector_backbone_prefix)
    }
    if not backbone_state:
        raise RuntimeError("SAM3 checkpoint 中没有 detector 视觉 backbone 权重")

    collisions = tracker_state.keys() & backbone_state.keys()
    if collisions:
        names = ", ".join(sorted(collisions)[:3])
        raise RuntimeError(f"SAM3 checkpoint 的 tracker/backbone 权重冲突：{names}")
    tracker_state.update(backbone_state)
    return tracker_state


def _image_digest(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _select_best_mask(
    masks: Any,
    scores: Any,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    mask_array = np.asarray(masks)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3 or mask_array.shape[0] == 0:
        raise RuntimeError(f"SAM3 返回了无效的 Mask 形状：{mask_array.shape}")

    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if score_array.size != mask_array.shape[0] or not np.isfinite(score_array).any():
        raise RuntimeError("SAM3 返回的 Mask 分数无效")
    safe_scores = np.where(np.isfinite(score_array), score_array, -np.inf)
    best_index = int(np.argmax(safe_scores))
    best_mask = np.asarray(mask_array[best_index]) > 0
    if best_mask.shape != expected_shape:
        raise RuntimeError(
            "SAM3 返回的 Mask 尺寸与输入图不一致："
            f"期望 {expected_shape[1]}x{expected_shape[0]}，"
            f"实际 {best_mask.shape}"
        )
    return np.array(best_mask, dtype=np.bool_, copy=True), float(score_array[best_index])


def decode_image(data: bytes, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > max_pixels:
                raise ValueError(f"图片像素数不能超过 {max_pixels}")
            source.load()
            transposed = ImageOps.exif_transpose(source)
            return np.array(transposed.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法解析上传的图片") from exc


def encode_binary_mask(mask: np.ndarray) -> bytes:
    if mask.ndim != 2:
        raise ValueError("Mask 必须是二维数组")
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    output = BytesIO()
    image.save(output, format="PNG", compress_level=4)
    return output.getvalue()
