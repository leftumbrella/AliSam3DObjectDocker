#!/usr/bin/env python3
"""Apply the pinned zero-egress DINO runtime edit to SAM 3D Objects."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile


RELATIVE_TARGET = Path("sam3d_objects/model/backbone/dit/embedder/dino.py")
EXPECTED_INPUT_SHA256 = (
    "78b7fee87f96fcc11049cbe0c58fe96f6ab991d4d7170b3f4d856535b5581ec1"
)
EXPECTED_OUTPUT_SHA256 = (
    "51d8429a07816c9d534b5cd1d01aa2816a3316374369d3ca73ba1f0ed39a05ee"
)
IMPORT_BEFORE = "import torch\n"
IMPORT_AFTER = "import os\nimport torch\n"
BLOCK_BEFORE = """        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
"""
BLOCK_AFTER = """        if backbone_kwargs is None:
            backbone_kwargs = {}

        offline_repo = os.environ.get("SAM3D_DINOV2_REPO")
        if offline_repo:
            repo_or_dir = offline_repo
            source = "local"
            backbone_kwargs = {**backbone_kwargs, "pretrained": False}

        with warnings.catch_warnings():
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source_root: Path) -> Path:
    target = source_root.resolve() / RELATIVE_TARGET
    original = target.read_bytes()
    original_digest = digest(original)
    if original_digest == EXPECTED_OUTPUT_SHA256:
        return target
    if original_digest != EXPECTED_INPUT_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected {target}: SHA-256 is {original_digest}"
        )

    text = original.decode("utf-8")
    if text.count(IMPORT_BEFORE) != 1 or text.count(BLOCK_BEFORE) != 1:
        raise RuntimeError(f"pinned DINO patch context is not unique: {target}")
    updated = text.replace(IMPORT_BEFORE, IMPORT_AFTER, 1).replace(
        BLOCK_BEFORE,
        BLOCK_AFTER,
        1,
    )
    updated_bytes = updated.encode("utf-8")
    if digest(updated_bytes) != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(f"patched DINO source has an unexpected digest: {target}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=".dino.py.",
            delete=False,
        ) as stream:
            stream.write(updated_bytes)
            temporary = Path(stream.name)
        os.chmod(temporary, target.stat().st_mode & 0o7777)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} SAM3D_SOURCE_ROOT", file=sys.stderr)
        return 2
    try:
        target = patch_source(Path(sys.argv[1]))
    except (OSError, RuntimeError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"patched offline DINO loader: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
