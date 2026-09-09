from __future__ import annotations

import math
from typing import Optional

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .reconstruction import reconstruct
from .spec import RECONSTRUCTION_SIGMA

_LPIPS_MODEL = None
_LPIPS_UNAVAILABLE_REASON: Optional[str] = None


def _get_lpips_model():
    """Lazily load an LPIPS network. Optional dependency: `pip install lpips`.

    If unavailable (not installed, or no torch/network for weights), LPIPS
    is reported as NaN and the reason is recorded once in the run's
    provenance rather than raising, since it is one of several metrics.
    """
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE_REASON
    if _LPIPS_MODEL is not None or _LPIPS_UNAVAILABLE_REASON is not None:
        return _LPIPS_MODEL
    try:
        import lpips  # type: ignore
        import torch

        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        _LPIPS_MODEL.eval()
    except Exception as exc:  # pragma: no cover - environment dependent
        _LPIPS_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        _LPIPS_MODEL = None
    return _LPIPS_MODEL


def lpips_unavailable_reason() -> Optional[str]:
    return _LPIPS_UNAVAILABLE_REASON


def lpips_distance(reconstructed_rgb: np.ndarray, reference_rgb: np.ndarray) -> float:
    model = _get_lpips_model()
    if model is None:
        return float("nan")
    import torch

    def to_tensor(rgb: np.ndarray) -> "torch.Tensor":
        arr = np.clip(rgb, 0.0, 1.0).astype(np.float32) * 2.0 - 1.0  # LPIPS expects [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    with torch.no_grad():
        value = model(to_tensor(reconstructed_rgb), to_tensor(reference_rgb))
    return float(value.item())


def anisotropy_index(halftone: np.ndarray) -> float:
    """Ring-variance of the normalized power spectrum of the RAW halftone.

    Per spec Sec. 5.1, anisotropy is the one metric explicitly excluded from
    the reconstruction operator, so callers must pass the raw binary/near-
    binary halftone (or its luma projection for color methods), not R(H).
    """
    array = np.asarray(halftone, dtype=np.float64)
    if array.ndim == 3:
        array = array.mean(axis=-1)
    centered = array - array.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    total = float(power.sum())
    if total <= 0:
        return 0.0
    power = power / total
    height, width = power.shape
    yy, xx = np.indices((height, width))
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
    scores = []
    for r in range(1, int(radius.max()) + 1):
        mask = radius == r
        if mask.sum() <= 1:
            continue
        ring = power[mask]
        mean = ring.mean()
        if mean <= 1e-12:
            continue
        scores.append(float(np.mean((ring - mean) ** 2)))
    return float(np.mean(scores)) if scores else 0.0


def _win_size(shape: tuple[int, int]) -> int:
    min_side = min(shape)
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return max(3, win)


def grayscale_full_reference_metrics(halftone: np.ndarray, reference_gray: np.ndarray, *, compute_lpips: bool = True) -> dict[str, float]:
    """PSNR/SSIM/LPIPS on R(H) vs. reference; Anisotropy on the raw halftone."""
    recon = reconstruct(halftone)
    reference = np.clip(np.asarray(reference_gray, dtype=np.float64), 0.0, 1.0)
    mse = float(np.mean((recon - reference) ** 2))
    psnr = float("inf") if mse == 0 else float(peak_signal_noise_ratio(reference, recon, data_range=1.0))
    ssim = 1.0 if mse == 0 else float(structural_similarity(reference, recon, data_range=1.0, win_size=_win_size(reference.shape)))
    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "anisotropy_index": anisotropy_index(halftone),
        "reconstruction_sigma": RECONSTRUCTION_SIGMA,
    }
    if compute_lpips:
        recon_rgb = np.repeat(recon[..., None], 3, axis=-1)
        reference_rgb = np.repeat(reference[..., None], 3, axis=-1)
        metrics["lpips"] = lpips_distance(recon_rgb, reference_rgb)
    else:
        metrics["lpips"] = float("nan")
    return metrics


def color_full_reference_metrics(halftone_rgb: np.ndarray, reference_rgb: np.ndarray, *, compute_lpips: bool = True) -> dict[str, float]:
    recon = reconstruct(halftone_rgb)
    reference = np.clip(np.asarray(reference_rgb, dtype=np.float64), 0.0, 1.0)
    mse = float(np.mean((recon - reference) ** 2))
    psnr = float("inf") if mse == 0 else float(peak_signal_noise_ratio(reference, recon, data_range=1.0))
    min_side = min(reference.shape[:2])
    win = _win_size((min_side, min_side))
    ssim = 1.0 if mse == 0 else float(structural_similarity(reference, recon, data_range=1.0, channel_axis=-1, win_size=win))
    delta = deltaE_ciede2000(rgb2lab(reference), rgb2lab(recon))
    metrics = {
        "psnr": psnr,
        "ssim": ssim,
        "anisotropy_index": anisotropy_index(halftone_rgb),
        "delta_e00": float(np.mean(delta)),
        "reconstruction_sigma": RECONSTRUCTION_SIGMA,
    }
    metrics["lpips"] = lpips_distance(recon, reference) if compute_lpips else float("nan")
    return metrics
