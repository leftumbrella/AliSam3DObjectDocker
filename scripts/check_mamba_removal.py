from __future__ import annotations

import json
import sys
from pathlib import Path


PROTECTED_PACKAGES = {
    "cuda-cudart",
    "cuda-cudart_linux-64",
    "cuda-cupti",
    "cuda-libraries",
    "cuda-nvrtc",
    "cuda-nvtx",
    "cuda-version",
    "libcublas",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libgcc",
    "libgcc-ng",
    "libgomp",
    "libnpp",
    "libnvjitlink",
    "libnvjpeg",
    "libstdcxx",
    "libstdcxx-ng",
    "python",
}


def _package_names(records: object) -> set[str]:
    if not isinstance(records, list):
        return set()

    names: set[str] = set()
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("name"), str):
            names.add(record["name"])
    return names


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "用法：check_mamba_removal.py <dry-run-json> <允许删除的包>..."
        )

    plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if plan.get("success") is not True:
        raise SystemExit("micromamba dry-run 未成功，拒绝裁剪环境")

    actions = plan.get("actions")
    if not isinstance(actions, dict):
        raise SystemExit("micromamba dry-run 缺少 actions，拒绝裁剪环境")

    unlink_records = actions.get("UNLINK", actions.get("unlink"))
    if unlink_records is None:
        raise SystemExit("micromamba dry-run 缺少 UNLINK 计划，拒绝裁剪环境")

    removed = _package_names(unlink_records)
    allowed = set(sys.argv[2:])
    unexpected = sorted(removed - allowed)
    if unexpected:
        names = ", ".join(unexpected)
        raise SystemExit(f"裁剪计划包含未明确允许的级联删除：{names}")

    protected_removed = sorted(removed & PROTECTED_PACKAGES)
    if protected_removed:
        names = ", ".join(protected_removed)
        raise SystemExit(f"裁剪计划会删除运行时保护包：{names}")

    print(f"裁剪计划检查通过，预计删除 {len(removed)} 个 Conda 包")


if __name__ == "__main__":
    main()
