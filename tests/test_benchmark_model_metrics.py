from __future__ import annotations

import numpy as np
import pytest

from benchmarking.metrics import color_metrics, luminance_metrics, nasanen_kernel, spectral_metrics
from benchmarking.model import HalftoneResult, validate_result


def test_nasanen_kernel_is_normalized_and_symmetric() -> None:
    kernel = nasanen_kernel()
    assert kernel.shape == (11, 11)
    assert float(kernel.sum()) == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])


def test_identical_images_have_perfect_metrics() -> None:
    gray = np.linspace(0, 1, 64, dtype=np.float64).reshape(8, 8)
    luma = luminance_metrics(gray, gray)
    assert luma["viewed_mse"] == pytest.approx(0.0)
    assert luma["viewed_psnr"] == float("inf")
    assert luma["viewed_ssim"] == pytest.approx(1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    color = color_metrics(rgb, rgb)
    assert color["viewed_rgb_mse"] == pytest.approx(0.0)
    assert color["delta_e_00_mean"] == pytest.approx(0.0)


def test_zero_spectral_field_is_finite() -> None:
    metrics = spectral_metrics(np.full((16, 16), 0.5), 0.5)
    assert metrics == {"density_abs_error": 0.0, "low_frequency_power_ratio": 0.0, "radial_anisotropy": 0.0}


def test_result_validation_rejects_shape_and_range() -> None:
    validate_result(HalftoneResult("ok", np.zeros((4, 4)), 0.1, 0), (4, 4))
    with pytest.raises(ValueError):
        validate_result(HalftoneResult("bad", np.zeros((3, 4)), 0.1, 0), (4, 4))
    with pytest.raises(ValueError):
        validate_result(HalftoneResult("bad", np.full((4, 4), 2.0), 0.1, 0), (4, 4))

