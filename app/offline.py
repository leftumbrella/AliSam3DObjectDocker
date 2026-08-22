from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DINOV2_REPOSITORY_ID = "facebookresearch/dinov2"
DINOV2_REPOSITORY_DIRECTORY = "facebookresearch_dinov2_main"
DINOV2_WEIGHT_FILENAME = "dinov2_vitl14_reg4_pretrain.pth"
DINOV2_WEIGHT_SIZE = 1_217_607_321


class OfflineRuntimeError(RuntimeError):
    pass


def configure_offline_environment(torch_home: Path) -> tuple[Path, Path]:
    """Force supported model loaders into offline mode and return DINO paths."""
    repository = torch_home / "hub" / DINOV2_REPOSITORY_DIRECTORY
    weight = torch_home / "hub" / "checkpoints" / DINOV2_WEIGHT_FILENAME
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["SAM3D_DINOV2_REPO"] = str(repository)
    return repository, weight


def install_offline_torch_hub_guard(
    torch_module: Any,
    *,
    repository: Path,
    weight: Path,
) -> None:
    """Redirect the one supported Hub repository locally and deny downloads."""
    repository = repository.expanduser().resolve()
    weight = weight.expanduser().resolve()
    hubconf = repository / "hubconf.py"
    if not hubconf.is_file():
        raise OfflineRuntimeError(f"DINOv2 本地源码不完整，缺少：{hubconf}")
    if not weight.is_file() or weight.is_symlink():
        raise OfflineRuntimeError(f"DINOv2 本地权重不存在：{weight}")
    if weight.stat().st_size != DINOV2_WEIGHT_SIZE:
        raise OfflineRuntimeError(
            "DINOv2 本地权重大小不正确："
            f"应为 {DINOV2_WEIGHT_SIZE} 字节，实际为 {weight.stat().st_size} 字节"
        )

    hub = torch_module.hub
    installed_for = getattr(hub, "_sam3d_offline_paths", None)
    expected_paths = (str(repository), str(weight))
    if installed_for is not None:
        if installed_for != expected_paths:
            raise OfflineRuntimeError("Torch Hub 离线门禁已使用不同的资源路径初始化")
        return

    original_load = hub.load

    def offline_load(
        repo_or_dir: str | os.PathLike[str],
        model: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        source = kwargs.pop("source", "github")
        repo_text = os.fspath(repo_or_dir)
        if repo_text == DINOV2_REPOSITORY_ID:
            repo_text = str(repository)
            source = "local"
        elif source != "local" or Path(repo_text).expanduser().resolve() != repository:
            raise OfflineRuntimeError(
                "离线模式拒绝 Torch Hub 远程或未登记的仓库"
                f"（source={source}）"
            )
        return original_load(
            repo_text,
            model,
            *args,
            source="local",
            **kwargs,
        )

    def deny_download(_url: str, *_args: Any, **_kwargs: Any) -> None:
        # 下载 URL 可能带签名参数，错误信息不能回显潜在凭证。
        raise OfflineRuntimeError("离线模式拒绝 Torch Hub 下载")

    hub.load = offline_load
    hub.download_url_to_file = deny_download
    hub._sam3d_offline_paths = expected_paths
