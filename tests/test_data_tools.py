from __future__ import annotations

from pathlib import Path
import io
import tarfile

import numpy as np
from PIL import Image
import pytest

from benchmarking.data import _safe_extract, _write_lock, verify_dataset


def _make_kodak(root: Path) -> None:
    images = root / "kodak/images"
    images.mkdir(parents=True)
    for index in range(1, 25):
        Image.fromarray(np.full((4, 4, 3), index, dtype=np.uint8)).save(images / f"kodim{index:02d}.png")


def test_kodak_lock_detects_corruption(tmp_path: Path) -> None:
    _make_kodak(tmp_path)
    _write_lock("kodak", tmp_path, "test")
    assert verify_dataset("kodak", tmp_path)["count"] == 24
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "kodak/images/kodim01.png")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset("kodak", tmp_path)


def test_wrong_count_is_rejected(tmp_path: Path) -> None:
    _make_kodak(tmp_path)
    (tmp_path / "kodak/images/kodim24.png").unlink()
    with pytest.raises(ValueError, match="expected 24"):
        verify_dataset("kodak", tmp_path, require_lock=False)


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../escape.txt")
        payload = b"unsafe"
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        _safe_extract(archive, tmp_path / "extract")
