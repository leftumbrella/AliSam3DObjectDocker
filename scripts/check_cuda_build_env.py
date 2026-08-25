"""Fail fast when the unified CUDA toolchain cannot compile extensions."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess


SUPPORTED_MACHINES = {"x86_64", "amd64"}


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"CUDA 构建环境缺少 {name}。")
    return Path(value)


def main() -> None:
    machine = platform.machine().lower()
    if machine not in SUPPORTED_MACHINES:
        raise SystemExit(f"不支持的 CUDA 构建架构：{machine}")

    cuda_home = _required_env_path("CUDA_HOME")
    conda_prefix = _required_env_path("CONDA_PREFIX")
    include_candidates = (
        cuda_home / "include",
        cuda_home / "targets" / "x86_64-linux" / "include",
        conda_prefix / "targets" / "x86_64-linux" / "include",
    )
    include_dir = next(
        (
            candidate
            for candidate in include_candidates
            if (candidate / "cuda_runtime_api.h").is_file()
        ),
        None,
    )
    if include_dir is None:
        searched = ", ".join(str(path) for path in include_candidates)
        raise SystemExit(f"CUDA 运行时头文件不存在，已检查：{searched}")
    runtime_header = include_dir / "cuda_runtime_api.h"

    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise SystemExit(f"CUDA 编译器不存在：{nvcc}")

    cxx_command = shlex.split(os.environ.get("CXX", "c++"))
    if not cxx_command or shutil.which(cxx_command[0]) is None:
        raise SystemExit(f"C++ 编译器不可用：{cxx_command!r}")

    compiler_flags: list[str] = []
    for variable in ("CPPFLAGS", "CXXFLAGS"):
        compiler_flags.extend(shlex.split(os.environ.get(variable, "")))

    probe = subprocess.run(
        [
            *cxx_command,
            *compiler_flags,
            f"-I{include_dir}",
            "-E",
            "-x",
            "c++",
            "-",
        ],
        input="#include <cuda_runtime_api.h>\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "已安装 CUDA 头文件，但当前 C++ 编译环境无法发现它。"
            "请检查 CUDA_HOME、编译器和 include 路径。\n"
            + probe.stderr.strip()
        )

    print(f"CUDA 构建环境校验通过：{runtime_header}")


if __name__ == "__main__":
    main()
