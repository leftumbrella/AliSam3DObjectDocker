"""Regression checks for the FC image build contract."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"


def _run_blocks(dockerfile: str) -> list[str]:
    """Return shell-form RUN instructions with continuations joined."""
    blocks: list[str] = []
    lines = iter(dockerfile.splitlines())
    for line in lines:
        if not line.startswith("RUN "):
            continue
        block = [line]
        while block[-1].rstrip().endswith("\\"):
            block.append(next(lines))
        blocks.append("\n".join(block))
    return blocks


class DockerfileCudaBuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.run_blocks = _run_blocks(cls.dockerfile)

    def _extension_block(self, package: str) -> str:
        matches = [block for block in self.run_blocks if f'"{package} @' in block]
        self.assertEqual(len(matches), 1, f"expected one {package} build block")
        return matches[0]

    def test_cuda_extensions_use_activated_micromamba_environment(self) -> None:
        for package in ("pytorch3d", "gsplat"):
            with self.subTest(package=package):
                block = self._extension_block(package)
                self.assertIn(
                    "micromamba run --no-capture-output -n sam3d-objects",
                    block,
                )

    def test_cuda_preflight_precedes_pytorch3d_compilation(self) -> None:
        block = self._extension_block("pytorch3d")
        preflight = block.find("python /tmp/check_cuda_build_env.py")
        install = block.find("python -m pip install")
        self.assertGreaterEqual(preflight, 0)
        self.assertGreater(install, preflight)


if __name__ == "__main__":
    unittest.main()
