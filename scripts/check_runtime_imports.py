"""在镜像构建阶段确认裁剪后的关键运行时模块与扩展仍完整。"""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path


REQUIRED_MODULES = (
    "torch",
    "pytorch3d._C",
    "flash_attn_2_cuda",
    "gsplat.csrc",
    "spconv.pytorch",
    "fastapi",
    "uvicorn",
)


def _check_kaolin_extension() -> str | None:
    """只检查扩展文件，避免执行 Kaolin 0.17 的顶层 Warp 导入链。"""
    try:
        distribution = metadata.distribution("kaolin")
    except metadata.PackageNotFoundError:
        return "kaolin: 未找到已安装的发行包"

    candidates = []
    for entry in distribution.files or ():
        if (
            len(entry.parts) >= 2
            and entry.parts[-2] == "kaolin"
            and entry.name.startswith("_C")
            and entry.suffix == ".so"
        ):
            candidates.append(Path(distribution.locate_file(entry)))

    if not any(candidate.is_file() for candidate in candidates):
        return "kaolin: 未找到 kaolin/_C*.so 编译扩展"
    return None


def main() -> None:
    failures: list[str] = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # 构建期需要汇总全部缺失项。
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    kaolin_failure = _check_kaolin_extension()
    if kaolin_failure is not None:
        failures.append(kaolin_failure)

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"裁剪后的运行时导入校验失败：\n{details}")

    print("裁剪后的关键运行时完整性校验通过。")


if __name__ == "__main__":
    main()
