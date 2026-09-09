from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 RGB image, got {array.shape}")
    array = array[..., :3].astype(np.float64)
    if np.issubdtype(image.dtype, np.integer):
        array /= float(np.iinfo(image.dtype).max)
    return np.clip(array, 0.0, 1.0)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(normalize_rgb(rgb), BT601, axes=([-1], [0]))


def nasanen_kernel(size: int = 11, scale_factor: float = 2000.0, luminance: float = 11.0) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Nasanen kernel size must be a positive odd integer")
    if scale_factor <= 0 or luminance <= 0:
        raise ValueError("scale_factor and luminance must be positive")
    coords = np.arange(size, dtype=np.float64) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    c = 0.525
    d = 3.91
    k = (math.pi * scale_factor) / (180.0 * (c * math.log(luminance) + d))
    spatial_radius = (2.0 * math.pi / scale_factor) * radius
    kernel = np.power(k * k + spatial_radius * spatial_radius, -1.5)
    kernel /= kernel.sum()
    return kernel


def viewed(image: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    kernel = nasanen_kernel() if kernel is None else np.asarray(kernel, dtype=np.float64)
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return convolve(array, kernel, mode="reflect")
    if array.ndim == 3 and array.shape[-1] == 3:
        return np.stack([convolve(array[..., c], kernel, mode="reflect") for c in range(3)], axis=-1)
    raise ValueError(f"Expected grayscale or RGB image, got {array.shape}")


def luminance_metrics(reference: np.ndarray, output: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    if reference.shape != output.shape or reference.ndim != 2:
        raise ValueError("Luminance reference and output must be equal-shaped 2-D arrays")
    filtered_reference = viewed(reference)
    filtered_output = viewed(output)
    mse = float(np.mean((filtered_reference - filtered_output) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    min_side = min(reference.shape)
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    ssim = 1.0 if mse == 0 else float(structural_similarity(filtered_reference, filtered_output, data_range=1.0, win_size=max(3, win_size)))
    return {"viewed_mse": mse, "viewed_psnr": psnr, "viewed_ssim": ssim, "density_abs_error": float(abs(output.mean() - reference.mean()))}


def color_metrics(reference_rgb: np.ndarray, output_rgb: np.ndarray) -> dict[str, float]:
    reference = normalize_rgb(reference_rgb)
    output = normalize_rgb(output_rgb)
    if reference.shape != output.shape:
        raise ValueError("RGB reference and output must have equal shapes")
    filtered_reference = np.clip(viewed(reference), 0.0, 1.0)
    filtered_output = np.clip(viewed(output), 0.0, 1.0)
    mse = float(np.mean((filtered_reference - filtered_output) ** 2))
    delta = deltaE_ciede2000(rgb2lab(filtered_reference), rgb2lab(filtered_output))
    return {"viewed_rgb_mse": mse, "viewed_rgb_psnr": float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse)), "delta_e_00_mean": float(np.mean(delta)), "delta_e_00_p95": float(np.percentile(delta, 95))}


def spectral_metrics(output: np.ndarray, target_level: float, low_frequency_radius: int = 4) -> dict[str, float]:
    array = np.asarray(output, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Spectral diagnostics require a 2-D image")
    centered = array - array.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered, norm="ortho"))) ** 2
    height, width = power.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cy, cx = height // 2, width // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = float(power.sum())
    low_ratio = 0.0 if total == 0 else float(power[radius <= low_frequency_radius].sum() / total)
    ring_terms: list[float] = []
    for ring in range(1, int(radius.max()) + 1):
        values = power[(radius >= ring - 0.5) & (radius < ring + 0.5)]
        if values.size > 1:
            ring_terms.append(float(values.var()))
    return {"density_abs_error": float(abs(array.mean() - target_level)), "low_frequency_power_ratio": low_ratio, "radial_anisotropy": float(np.mean(ring_terms)) if ring_terms else 0.0}
