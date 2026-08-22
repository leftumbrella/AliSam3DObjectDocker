"""Fail fast when the activated Conda CUDA toolchain cannot compile headers."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess


TARGETS = {
    "x86_64": "x86_64-linux",
    "amd64": "x86_64-linux",
}


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"CUDA 构建环境缺少 {name}。")
    return Path(value)


def main() -> None:
    machine = platform.machine().lower()
    target = TARGETS.get(machine)
    if target is None:
        raise SystemExit(f"不支持的 CUDA 构建架构：{machine}")

    conda_prefix = _required_env_path("CONDA_PREFIX")
    cuda_home = _required_env_path("CUDA_HOME")
    target_include = conda_prefix / "targets" / target / "include"
    runtime_header = target_include / "cuda_runtime_api.h"
    if not runtime_header.is_file():
        raise SystemExit(f"Conda CUDA 运行时头文件不存在：{runtime_header}")

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
        [*cxx_command, *compiler_flags, "-E", "-x", "c++", "-"],
        input="#include <cuda_runtime_api.h>\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "已安装 CUDA 头文件，但当前 C++ 编译环境无法发现它。"
            "请通过 micromamba run/activate 执行 CUDA 扩展构建。\n"
            + probe.stderr.strip()
        )

    print(f"CUDA 构建环境校验通过：{runtime_header}")


if __name__ == "__main__":
    main()
