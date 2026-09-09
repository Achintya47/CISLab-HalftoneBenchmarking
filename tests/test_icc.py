from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarking_v3.icc import (  # noqa: E402
    DEFAULT_CMYK_PROFILE_PATH,
    ICCPipeline,
    ICCProfileSpec,
    _naive_cmyk_to_rgb,
    _naive_rgb_to_cmyk,
)


def test_default_cmyk_profile_ships_with_the_repo() -> None:
    assert DEFAULT_CMYK_PROFILE_PATH.is_file(), "the CC0 CMYK ICC profile must be committed to the repo, not fetched at run time"


def test_naive_full_gcr_conversion_is_exactly_invertible() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.random((32, 32, 3))
    cmyk = _naive_rgb_to_cmyk(rgb)
    assert cmyk.shape == (32, 32, 4)
    back = _naive_cmyk_to_rgb(cmyk)
    assert np.abs(rgb - back).max() < 1e-9


def test_naive_conversion_black_channel_is_the_shared_minimum() -> None:
    rgb = np.array([[[0.2, 0.5, 0.8]]])
    cmyk = _naive_rgb_to_cmyk(rgb)
    c0, m0, y0 = 1 - 0.2, 1 - 0.5, 1 - 0.8
    expected_k = min(c0, m0, y0)
    assert cmyk[0, 0, 3] == pytest.approx(expected_k, abs=1e-6)


def test_pure_white_and_black_round_trip_naively() -> None:
    rgb = np.array([[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]])
    cmyk = _naive_rgb_to_cmyk(rgb)
    back = _naive_cmyk_to_rgb(cmyk)
    np.testing.assert_allclose(rgb, back, atol=1e-6)


def test_shipped_profile_is_detected_as_one_directional_and_falls_back_consistently() -> None:
    """The bundled CGATS001Compat-v2-micro.icc is a real CMYK ICC profile
    (verified: 4-channel, device class 'scnr') but only supports the
    device->PCS direction. `ICCPipeline` must detect this automatically
    (rather than assume any profile is bidirectional) and use the naive
    fallback for BOTH directions so forward/inverse stay consistent."""
    spec = ICCProfileSpec(cmyk_profile_path=DEFAULT_CMYK_PROFILE_PATH)
    pipeline = ICCPipeline(spec)
    assert pipeline.use_icc is False
    assert pipeline._fallback_reason is not None

    rng = np.random.default_rng(1)
    rgb = rng.random((16, 16, 3))
    cmyk = pipeline.rgb_to_cmyk(rgb)
    assert cmyk.shape == (16, 16, 4)
    back = pipeline.cmyk_to_rgb(cmyk)
    assert np.abs(rgb - back).max() < 1e-6  # naive path used both ways -> exact round trip


def test_missing_profile_path_falls_back_without_crashing(tmp_path: Path) -> None:
    spec = ICCProfileSpec(cmyk_profile_path=tmp_path / "does-not-exist.icc")
    pipeline = ICCPipeline(spec)
    assert pipeline.use_icc is False
    assert "not found" in pipeline._fallback_reason

    rgb = np.random.default_rng(2).random((8, 8, 3))
    cmyk = pipeline.rgb_to_cmyk(rgb)
    assert cmyk.shape == (8, 8, 4)


def test_fingerprint_is_a_complete_provenance_record() -> None:
    spec = ICCProfileSpec(cmyk_profile_path=DEFAULT_CMYK_PROFILE_PATH, rendering_intent="perceptual", black_point_compensation=False)
    pipeline = ICCPipeline(spec)
    fingerprint = pipeline.fingerprint()
    for key in ("source_profile", "cmyk_profile_path", "cmyk_profile_sha256", "rendering_intent", "black_point_compensation", "use_icc", "pillow_version", "littlecms_version"):
        assert key in fingerprint
    assert fingerprint["rendering_intent"] == "perceptual"
    assert fingerprint["black_point_compensation"] is False
    assert len(fingerprint["cmyk_profile_sha256"]) == 64  # sha256 hex digest


def test_unknown_rendering_intent_raises_a_clear_error() -> None:
    spec = ICCProfileSpec(cmyk_profile_path=DEFAULT_CMYK_PROFILE_PATH, rendering_intent="not_a_real_intent")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown rendering_intent"):
        ICCPipeline(spec)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
