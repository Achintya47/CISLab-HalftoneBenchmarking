from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class HalftoneResult:
    method: str
    luma_output: np.ndarray
    runtime_sec: float
    seed: int
    rgb_output: np.ndarray | None = None
    preview_rgb: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _validate_image(name: str, value: np.ndarray, channels: int | None) -> np.ndarray:
    array = np.asarray(value)
    expected_ndim = 2 if channels is None else 3
    if array.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got {array.shape}")
    if channels is not None and array.shape[-1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def validate_result(result: HalftoneResult, expected_shape: tuple[int, int]) -> None:
    if not result.method:
        raise ValueError("method must not be empty")
    if not np.isfinite(result.runtime_sec) or result.runtime_sec < 0:
        raise ValueError("runtime_sec must be finite and non-negative")
    luma = _validate_image("luma_output", result.luma_output, None)
    if luma.shape != expected_shape:
        raise ValueError(f"luma_output shape {luma.shape} does not match input {expected_shape}")
    for name, image in (("rgb_output", result.rgb_output), ("preview_rgb", result.preview_rgb)):
        if image is not None:
            rgb = _validate_image(name, image, 3)
            if rgb.shape[:2] != expected_shape:
                raise ValueError(f"{name} shape {rgb.shape[:2]} does not match input {expected_shape}")
