"""在镜像构建阶段确认裁剪后的关键运行时模块与扩展仍完整。"""

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
    "fastapi",
    "uvicorn",
)

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
}


def _canonicalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    failures: list[str] = []

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
    unexpected = sorted(FORBIDDEN_DISTRIBUTIONS & installed)
    if unexpected:
        failures.append("镜像包含禁止的发行包：" + ", ".join(unexpected))

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"裁剪后的运行时导入校验失败：\n{details}")

    print("裁剪后的关键运行时完整性校验通过。")


if __name__ == "__main__":
    main()
