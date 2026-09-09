from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .spec import RECONSTRUCTION_SIGMA


def reconstruct(halftone: np.ndarray, sigma: float = RECONSTRUCTION_SIGMA) -> np.ndarray:
    """R(H): float-cast + fixed-sigma Gaussian observer blur (spec Sec. 5.2).

    This is the ONLY reconstruction operator used for PSNR / SSIM / LPIPS /
    Delta E in this benchmark. It intentionally ignores whatever HVS model
    (Gaussian sigma=1.0/1.5, or a Nasanen kernel with S=2000) a given
    algorithm used internally during its own training/optimization -- those
    are optimization-time choices baked into each method, not the
    benchmark's evaluation-time reconstruction contract. Mixing the two was
    the exact inconsistency flagged when this pipeline was generalized.
    """
    array = np.clip(np.asarray(halftone, dtype=np.float64), 0.0, 1.0)
    if array.ndim == 2:
        return np.clip(gaussian_filter(array, sigma=sigma, mode="reflect"), 0.0, 1.0)
    if array.ndim == 3:
        channels = [gaussian_filter(array[..., c], sigma=sigma, mode="reflect") for c in range(array.shape[-1])]
        return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)
    raise ValueError(f"Unsupported array shape for reconstruction: {array.shape}")
