"""Ordered (Bayer) dithering.

The vendored repository implements DBS, Error Diffusion, and Deep-RL
halftoning, but the spec's four method families also include Ordered
Dithering, which had no implementation anywhere in the project. This is a
new, from-scratch implementation added specifically to close that gap, kept
in the same `MethodAdapter`-shaped interface as everything else so it drops
into the registry/orchestrator without special-casing.
"""

from __future__ import annotations

import time

import numpy as np

from benchmarking.metrics import normalize_rgb, rgb_to_luma
from benchmarking.model import HalftoneResult


def bayer_matrix(n: int) -> np.ndarray:
    if n & (n - 1) != 0 or n < 2:
        raise ValueError("Bayer matrix size must be a power of two >= 2")
    if n == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float64)
    smaller = bayer_matrix(n // 2)
    return np.block(
        [
            [4 * smaller, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1],
        ]
    )


def _threshold_map(matrix_size: int, height: int, width: int) -> np.ndarray:
    matrix = bayer_matrix(matrix_size)
    threshold = (matrix + 0.5) / (matrix_size * matrix_size)
    reps_h = height // matrix_size + 1
    reps_w = width // matrix_size + 1
    tiled = np.tile(threshold, (reps_h, reps_w))
    return tiled[:height, :width]


def ordered_dither_channel(channel: np.ndarray, matrix_size: int = 8) -> np.ndarray:
    height, width = channel.shape
    threshold = _threshold_map(matrix_size, height, width)
    return (channel >= threshold).astype(np.float32)


def ordered_dither_rgb(rgb: np.ndarray, matrix_size: int = 8) -> np.ndarray:
    return np.stack([ordered_dither_channel(rgb[..., c], matrix_size) for c in range(rgb.shape[-1])], axis=-1)


class OrderedDitheringAdapter:
    name = "ordered_dithering"
    family = "ordered_dithering"
    stochastic = False
    supports_color = True

    def __init__(self, matrix_size: int = 8) -> None:
        if matrix_size & (matrix_size - 1) != 0:
            raise ValueError("matrix_size must be a power of two (2, 4, 8, 16, ...)")
        self.matrix_size = matrix_size

    def run(self, image: np.ndarray, seed: int) -> HalftoneResult:
        rgb = normalize_rgb(image).astype(np.float32)
        start = time.perf_counter()
        halftone_rgb = ordered_dither_rgb(rgb, self.matrix_size)
        elapsed = time.perf_counter() - start
        return HalftoneResult(
            method=self.name,
            luma_output=rgb_to_luma(halftone_rgb),
            runtime_sec=elapsed,
            seed=seed,
            rgb_output=halftone_rgb.astype(np.float64),
            preview_rgb=halftone_rgb.astype(np.float64),
            metadata={"matrix_size": self.matrix_size},
        )
