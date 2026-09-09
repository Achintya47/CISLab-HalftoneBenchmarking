"""Spec step 9-10: RGB -> CIELAB (via skimage's standard sRGB->XYZ->Lab
pipeline) and the final digital metrics.

Unlike benchmarking_v2 (which reconstructs and scores in grayscale/luma
space, with the color track as a secondary RGB comparison), v3's primary
evaluation space IS RGB/Lab -- by the time metrics run, CMYK has already
been fully round-tripped back through the ICC pipeline to RGB (spec step
8), so there is exactly one metrics entry point here, not a separate
grayscale/color split.

PSNR/SSIM/LPIPS and Anisotropy Index reuse benchmarking_v2's
implementations directly (imported, not re-derived) since nothing about
their definitions changes here -- only *what* gets fed into them does
(a CMYK-plane halftone and an ICC round-tripped RGB reconstruction, instead
of a luma halftone and a Gaussian-blur reconstruction).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from benchmarking_v2.metrics import anisotropy_index, lpips_distance, lpips_unavailable_reason  # noqa: F401  (re-exported)


def _win_size(shape: tuple[int, int]) -> int:
    min_side = min(shape)
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return max(3, win)


def rgb_reconstruction_metrics(reconstructed_rgb: np.ndarray, reference_rgb: np.ndarray, *, compute_lpips: bool = True) -> Dict[str, float]:
    """PSNR / SSIM / LPIPS / Delta E00, all computed directly on
    (reconstructed_rgb, reference_rgb) -- no additional reconstruction
    happens here, since CMYK->RGB (spec step 8) already produced the final
    continuous-tone image being scored."""
    reconstructed = np.clip(np.asarray(reconstructed_rgb, dtype=np.float64), 0.0, 1.0)
    reference = np.clip(np.asarray(reference_rgb, dtype=np.float64), 0.0, 1.0)
    mse = float(np.mean((reconstructed - reference) ** 2))
    psnr = float("inf") if mse == 0 else float(peak_signal_noise_ratio(reference, reconstructed, data_range=1.0))
    win = _win_size(reference.shape[:2])
    ssim = 1.0 if mse == 0 else float(structural_similarity(reference, reconstructed, data_range=1.0, channel_axis=-1, win_size=win))

    # Spec step 9: RGB -> XYZ -> CIELAB. skimage.color.rgb2lab implements
    # exactly that standard pipeline (sRGB -> linear -> CIE XYZ -> CIELAB);
    # reused rather than hand-rolled, same as benchmarking_v2's color track.
    lab_reconstructed = rgb2lab(reconstructed)
    lab_reference = rgb2lab(reference)
    delta_e00 = deltaE_ciede2000(lab_reference, lab_reconstructed)

    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "delta_e00": float(np.mean(delta_e00)),
    }
    metrics["lpips"] = lpips_distance(reconstructed, reference) if compute_lpips else float("nan")
    return metrics


def multichannel_anisotropy_index(halftone_planes: Dict[str, np.ndarray]) -> float:
    """A single scalar Anisotropy Index for a 4-plane CMYK halftone: the
    mean of each colorant plane's own (raw-halftone, per spec Sec. 5.1)
    ring-variance anisotropy score. Averaging across planes is a documented
    choice (see EXTENDING.md) -- swap in a weighted or per-channel-reported
    variant there if a different aggregation is preferred."""
    return float(np.mean([anisotropy_index(plane) for plane in halftone_planes.values()]))
