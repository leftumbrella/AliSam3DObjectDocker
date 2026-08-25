"""在镜像构建阶段确认统一运行时的模块、版本与 CUDA ABI。"""

from __future__ import annotations

import importlib
from importlib import metadata
import os
from pathlib import Path
import re
import sys


REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "pytorch3d._C",
    "gsplat",
    "gsplat.csrc",
    "spconv.pytorch",
    "moge.model.v1",
    "utils3d",
    "sam3d_objects.pipeline.inference_pipeline_pointmap",
    "inference",
    "sam3.model_builder",
    "sam3.model.sam1_task_predictor",
    "fastapi",
    "uvicorn",
)

EXPECTED_PYTHON = (3, 12)
EXPECTED_DISTRIBUTIONS = {
    "gsplat": "1.5.3",
    "pytorch3d": "0.7.9",
    "spconv-cu126": "2.3.8",
    "torch": "2.7.1+cu126",
    "torchvision": "0.22.1+cu126",
    "triton": "3.3.1",
}

FORBIDDEN_DISTRIBUTIONS = {
    # 训练、云平台与音频依赖
    "librosa",
    "lightning",
    "mosaicml-streaming",
    "pytorch-lightning",
    "sagemaker",
    "tensorboard",
    "torchmetrics",
    "wandb",
    # Jupyter 与交互环境
    "ipykernel",
    "ipython",
    "ipycanvas",
    "ipyevents",
    "ipywidgets",
    "jupyter",
    "jupyter-client",
    "jupyter-console",
    "jupyter-core",
    "jupyter-server",
    "jupyterlab",
    "nbclient",
    "nbconvert",
    "nbformat",
    "notebook",
    # Notebook/网页演示与当前关闭的可选后处理
    "dash",
    "gradio",
    "igraph",
    "kaolin",
    "matplotlib",
    "open3d",
    "plotly",
    "pymeshfix",
    "pyvista",
    "seaborn",
    "xatlas",
    # 开发、格式化与测试工具
    "autoflake",
    "black",
    "flake8",
    "hypothesis",
    "pdoc3",
    "pytest",
    "usort",
    # Ada FC 固定使用 PyTorch SDPA，不需要额外 attention wheel。
    "flash-attn",
    "xformers",
    # The unified image must not retain the former CUDA 12.1 sparse runtime.
    "spconv-cu121",
}


def _canonicalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    failures: list[str] = []

    if sys.version_info[:2] != EXPECTED_PYTHON:
        failures.append(
            "Python 版本不匹配："
            f"expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )

    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    notebook_path = Path("/opt/sam-3d-objects/notebook")
    if str(notebook_path) not in sys.path:
        sys.path.insert(0, str(notebook_path))

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # 构建期需要汇总全部缺失项。
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    installed = {
        _canonicalize_distribution_name(dist.metadata["Name"])
        for dist in metadata.distributions()
        if dist.metadata.get("Name")
    }
    for distribution, expected_version in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            failures.append(f"缺少发行包：{distribution}=={expected_version}")
            continue
        if actual_version != expected_version:
            failures.append(
                f"发行包版本不匹配：{distribution} "
                f"expected {expected_version}, got {actual_version}"
            )

    torch_module = sys.modules.get("torch")
    torch_cuda_version = getattr(
        getattr(torch_module, "version", None),
        "cuda",
        None,
    )
    if torch_cuda_version != "12.6":
        failures.append(
            f"PyTorch CUDA 版本不匹配：expected 12.6, got {torch_cuda_version!r}"
        )

    unexpected = sorted(FORBIDDEN_DISTRIBUTIONS & installed)
    if unexpected:
        failures.append("镜像包含禁止的发行包：" + ", ".join(unexpected))

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"统一运行时导入校验失败：\n{details}")

    print("统一运行时的关键模块与 CUDA ABI 校验通过。")


if __name__ == "__main__":
    main()
