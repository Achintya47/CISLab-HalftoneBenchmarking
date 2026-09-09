from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking_v3.cmyk_pipeline import CHANNEL_NAMES, combine_channels, separate_channels  # noqa: E402


def test_separate_then_combine_is_the_identity() -> None:
    rng = np.random.default_rng(0)
    cmyk = rng.random((10, 12, 4))
    planes = separate_channels(cmyk)
    assert set(planes) == set(CHANNEL_NAMES)
    for name, plane in planes.items():
        assert plane.shape == (10, 12)
    recombined = combine_channels(planes)
    np.testing.assert_allclose(cmyk, recombined, atol=1e-12)


def test_separate_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="HxWx4"):
        separate_channels(np.zeros((4, 4, 3)))


def test_combine_rejects_missing_plane() -> None:
    planes = {"C": np.zeros((4, 4)), "M": np.zeros((4, 4)), "Y": np.zeros((4, 4))}
    with pytest.raises(ValueError, match="Missing CMYK plane"):
        combine_channels(planes)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
