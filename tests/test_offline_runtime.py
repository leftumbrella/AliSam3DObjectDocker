"""Regression checks for the FC runtime's zero-egress model loading guard."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.offline import (  # noqa: E402
    DINOV2_REPOSITORY_ID,
    DINOV2_WEIGHT_SIZE,
    OfflineRuntimeError,
    configure_offline_environment,
    install_offline_torch_hub_guard,
)


class FakeHub:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(self, *args: object, **kwargs: object) -> str:
        self.calls.append((args, kwargs))
        return "loaded"

    def download_url_to_file(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("original downloader must not be called")


class OfflineRuntimeTests(unittest.TestCase):
    def _assets(self, root: Path) -> tuple[Path, Path]:
        repository, weight = configure_offline_environment(root)
        repository.mkdir(parents=True)
        repository.joinpath("hubconf.py").write_text("# local hub\n", encoding="utf-8")
        weight.parent.mkdir(parents=True)
        with weight.open("wb") as stream:
            stream.truncate(DINOV2_WEIGHT_SIZE)
        return repository, weight

    def test_environment_forces_hugging_face_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, _ = configure_offline_environment(Path(directory))

        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
        self.assertEqual(os.environ["SAM3D_DINOV2_REPO"], str(repository))

    def test_github_dinov2_is_redirected_to_registered_local_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, weight = self._assets(Path(directory))
            hub = FakeHub()
            torch_module = SimpleNamespace(hub=hub)
            install_offline_torch_hub_guard(
                torch_module,
                repository=repository,
                weight=weight,
            )

            result = hub.load(
                DINOV2_REPOSITORY_ID,
                "dinov2_vitl14_reg",
                pretrained=False,
                source="github",
            )

        self.assertEqual(result, "loaded")
        args, kwargs = hub.calls[0]
        self.assertEqual(args[:2], (str(repository.resolve()), "dinov2_vitl14_reg"))
        self.assertEqual(kwargs["source"], "local")
        self.assertFalse(kwargs["pretrained"])

    def test_unregistered_hub_repo_and_download_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository, weight = self._assets(Path(directory))
            hub = FakeHub()
            install_offline_torch_hub_guard(
                SimpleNamespace(hub=hub),
                repository=repository,
                weight=weight,
            )

            with self.assertRaisesRegex(OfflineRuntimeError, "未登记"):
                hub.load("someone/other-repo", "model")
            with self.assertRaisesRegex(OfflineRuntimeError, "拒绝 Torch Hub 下载"):
                hub.download_url_to_file("https://example.com/model.pt", weight)

    def test_missing_or_truncated_weight_fails_before_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, weight = configure_offline_environment(root)
            repository.mkdir(parents=True)
            repository.joinpath("hubconf.py").write_text("# local hub\n", encoding="utf-8")
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"truncated")

            with self.assertRaisesRegex(OfflineRuntimeError, "大小不正确"):
                install_offline_torch_hub_guard(
                    SimpleNamespace(hub=FakeHub()),
                    repository=repository,
                    weight=weight,
                )


if __name__ == "__main__":
    unittest.main()
