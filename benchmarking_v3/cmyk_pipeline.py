"""CMYK separation / combination (spec steps 3, 4, 7).

Deliberately tiny: once `icc.ICCPipeline.rgb_to_cmyk` has produced an
`HxWx4` array, "separating" it into 4 independent grayscale planes and
"combining" 4 planes back are just array slicing / stacking. There is no
RGB-luminance-style weighting here (spec step 4 explicitly: "This is
channel extraction, not RGB luminance conversion") -- each colorant plane
IS its own grayscale image, unchanged.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

CHANNEL_NAMES = ("C", "M", "Y", "K")


def separate_channels(cmyk: np.ndarray) -> Dict[str, np.ndarray]:
    """`cmyk`: HxWx4 float [0,1] -> {"C": HxW, "M": HxW, "Y": HxW, "K": HxW}."""
    if cmyk.ndim != 3 or cmyk.shape[-1] != 4:
        raise ValueError(f"Expected an HxWx4 CMYK array, got shape {cmyk.shape}")
    return {name: cmyk[..., index] for index, name in enumerate(CHANNEL_NAMES)}


def combine_channels(planes: Dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of `separate_channels`."""
    missing = [name for name in CHANNEL_NAMES if name not in planes]
    if missing:
        raise ValueError(f"Missing CMYK plane(s): {missing}")
    return np.clip(np.stack([planes[name] for name in CHANNEL_NAMES], axis=-1), 0.0, 1.0)
