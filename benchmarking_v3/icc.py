"""Centralized, ICC-standard RGB<->CMYK color management.

Uses `PIL.ImageCms` (a wrapper around Little CMS / lcms2 -- the same engine
behind virtually every real-world color-managed workflow: Photoshop,
GIMP, ImageMagick, browsers, print RIPs) to do genuine ICC-based
conversion: source profile -> PCS -> destination profile, honoring a
configurable rendering intent and black-point-compensation flag, exactly
per the spec's diagram:

    RGB / sRGB -> Source ICC Profile -> LittleCMS/ImageCms
        -> Rendering Intent + BPC -> Destination CMYK ICC Profile -> CMYK

This module is deliberately profile-agnostic: point `cmyk_profile_path` at
*any* ICC/ICM profile file and it is used as-is. No profile is special-
cased by name.

## Why there's a fallback, and why that's the correct design here

A full print-characterization ("prtr" device class) ICC profile ships both
directions -- device->PCS (AtoB, needed for CMYK->RGB/Lab) and PCS->device
(BtoA, needed for RGB->CMYK) -- but real ones (SWOP/FOGRA/GRACoL-style) are
almost never freely redistributable, and this project ships one anyway
(`icc_profiles/CGATS001Compat-v2-micro.icc`, CC0-licensed, from
https://github.com/saucecontrol/Compact-ICC-Profiles) so the pipeline is
runnable out of the box. That profile is an input/"legacy content
assumption" ("scnr" device class) profile: it only has an AtoB table, so it
can genuinely ICC-convert CMYK->RGB, but NOT RGB->CMYK (there's no BtoA
table to build that transform from -- this isn't a bug in this code, it's
inherent to what the profile contains; verified by probing
`ImageCms.buildTransform` directly, see `tests/test_icc.py`).

Rather than silently mixing "real ICC one way, made-up math the other way"
(which would make the reconstructed CMYK -> RGB leg inconsistent with
whatever produced that CMYK in the first place), `ICCPipeline` probes BOTH
directions at construction time. If both build, every conversion is fully
ICC-managed. If either fails, BOTH directions fall back to a documented,
exactly-invertible analytic conversion (naive full-GCR), so the forward and
inverse legs always agree with each other. Point `cmyk_profile_path` at your
own real bidirectional output profile (e.g. a licensed SWOP/FOGRA/GRACoL
.icc) to get a fully ICC-managed round trip -- no code changes needed, it's
auto-detected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

try:
    from PIL import Image, ImageCms

    _HAS_IMAGECMS = True
except ImportError:  # pragma: no cover - Pillow always ships ImageCms in practice
    _HAS_IMAGECMS = False

DEFAULT_CMYK_PROFILE_PATH = Path(__file__).with_name("icc_profiles") / "CGATS001Compat-v2-micro.icc"

RenderingIntent = Literal["perceptual", "relative_colorimetric", "saturation", "absolute_colorimetric"]


@dataclass(frozen=True)
class ICCProfileSpec:
    """Everything that determines the RGB<->CMYK conversion, frozen so it
    can be hashed/logged as a single provenance unit (spec calls this out
    explicitly: "a frozen CMYK profile, rendering intent, black-point
    compensation, and conversion-library version")."""

    cmyk_profile_path: Path
    rendering_intent: RenderingIntent = "relative_colorimetric"
    black_point_compensation: bool = True
    source_profile: str = "sRGB"  # only "sRGB" is supported today; see EXTENDING.md to add others

    @property
    def cmyk_profile_sha256(self) -> str:
        return hashlib.sha256(Path(self.cmyk_profile_path).read_bytes()).hexdigest()


def _naive_rgb_to_cmyk(rgb: np.ndarray) -> np.ndarray:
    """Standard textbook naive/full-GCR conversion: as much black as the
    darkest channel allows, then re-derive C/M/Y from what's left. Exactly
    invertible by `_naive_cmyk_to_rgb` (see module docstring / EXTENDING.md
    for the one-line proof), which is what makes it a safe, deterministic
    fallback rather than a lossy guess."""
    rgb = np.clip(rgb, 0.0, 1.0)
    c0 = 1.0 - rgb[..., 0]
    m0 = 1.0 - rgb[..., 1]
    y0 = 1.0 - rgb[..., 2]
    k = np.minimum(np.minimum(c0, m0), y0)
    denom = np.clip(1.0 - k, 1e-8, None)
    c = np.where(k < 1.0, (c0 - k) / denom, 0.0)
    m = np.where(k < 1.0, (m0 - k) / denom, 0.0)
    y = np.where(k < 1.0, (y0 - k) / denom, 0.0)
    return np.clip(np.stack([c, m, y, k], axis=-1), 0.0, 1.0)


def _naive_cmyk_to_rgb(cmyk: np.ndarray) -> np.ndarray:
    """Exact inverse of `_naive_rgb_to_cmyk` (standard subtractive formula)."""
    cmyk = np.clip(cmyk, 0.0, 1.0)
    c, m, y, k = cmyk[..., 0], cmyk[..., 1], cmyk[..., 2], cmyk[..., 3]
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


class ICCPipeline:
    def __init__(self, spec: ICCProfileSpec):
        self.spec = spec
        self.use_icc = False
        self._fallback_reason: Optional[str] = None
        self._to_cmyk_transform = None
        self._to_rgb_transform = None
        self._build()

    def _intent_value(self):
        mapping = {
            "perceptual": ImageCms.Intent.PERCEPTUAL,
            "relative_colorimetric": ImageCms.Intent.RELATIVE_COLORIMETRIC,
            "saturation": ImageCms.Intent.SATURATION,
            "absolute_colorimetric": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
        }
        if self.spec.rendering_intent not in mapping:
            raise ValueError(f"Unknown rendering_intent {self.spec.rendering_intent!r}; choose from {sorted(mapping)}")
        return mapping[self.spec.rendering_intent]

    def _build(self) -> None:
        intent = self._intent_value()  # raises ValueError immediately for a bad config value -- never silently swallowed into fallback
        if not _HAS_IMAGECMS:
            self._fallback_reason = "Pillow was built without ImageCms/lcms support"
            return
        profile_path = Path(self.spec.cmyk_profile_path)
        if not profile_path.is_file():
            self._fallback_reason = f"CMYK ICC profile not found: {profile_path}"
            return
        try:
            srgb_profile = ImageCms.createProfile(self.spec.source_profile)
            cmyk_profile = ImageCms.ImageCmsProfile(str(profile_path))
            flags = ImageCms.Flags.BLACKPOINTCOMPENSATION if self.spec.black_point_compensation else ImageCms.Flags(0)

            # Probe BOTH directions before committing to either -- see
            # module docstring for why a one-way-only ICC transform is
            # treated as "not usable" rather than mixed with the fallback.
            to_cmyk = ImageCms.buildTransform(srgb_profile, cmyk_profile, "RGB", "CMYK", renderingIntent=intent, flags=flags)
            to_rgb = ImageCms.buildTransform(cmyk_profile, srgb_profile, "CMYK", "RGB", renderingIntent=intent, flags=flags)
        except Exception as exc:
            self._fallback_reason = (
                f"ICC profile does not support a full RGB<->CMYK round trip at this rendering intent "
                f"({type(exc).__name__}: {exc}); falling back to the naive full-GCR conversion for both "
                f"directions so forward/inverse stay consistent. Supply a real bidirectional output-class "
                f"ICC profile (e.g. a licensed SWOP/FOGRA/GRACoL .icc) to get a fully ICC-managed round trip."
            )
            return
        self._to_cmyk_transform = to_cmyk
        self._to_rgb_transform = to_rgb
        self.use_icc = True

    # -- public conversion API -------------------------------------------------
    def rgb_to_cmyk(self, rgb: np.ndarray) -> np.ndarray:
        """`rgb`: HxWx3 float [0,1]. Returns HxWx4 float [0,1] (C, M, Y, K)."""
        if self.use_icc:
            return self._apply(rgb, "RGB", "CMYK", self._to_cmyk_transform, 4)
        return _naive_rgb_to_cmyk(rgb)

    def cmyk_to_rgb(self, cmyk: np.ndarray) -> np.ndarray:
        """`cmyk`: HxWx4 float [0,1]. Returns HxWx3 float [0,1]."""
        if self.use_icc:
            return self._apply(cmyk, "CMYK", "RGB", self._to_rgb_transform, 3)
        return _naive_cmyk_to_rgb(cmyk)

    def _apply(self, array: np.ndarray, mode_in: str, mode_out: str, transform, out_channels: int) -> np.ndarray:
        u8 = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
        image = Image.fromarray(u8, mode=mode_in)
        converted = ImageCms.applyTransform(image, transform)
        assert converted.mode == mode_out
        out = np.asarray(converted).astype(np.float64) / 255.0
        if out.ndim == 2:
            out = out[..., None]
        return out[..., :out_channels]

    def fingerprint(self) -> dict:
        """Frozen provenance record: exactly what the spec asks for --
        profile, rendering intent, BPC, and conversion-library version."""
        try:
            import PIL

            pillow_version = PIL.__version__
        except Exception:
            pillow_version = "unknown"
        littlecms_version = None
        if _HAS_IMAGECMS:
            try:
                littlecms_version = ImageCms.core.littlecms_version
            except Exception:
                littlecms_version = None
        return {
            "source_profile": self.spec.source_profile,
            "cmyk_profile_path": str(self.spec.cmyk_profile_path),
            "cmyk_profile_sha256": self.spec.cmyk_profile_sha256,
            "rendering_intent": self.spec.rendering_intent,
            "black_point_compensation": self.spec.black_point_compensation,
            "use_icc": self.use_icc,
            "fallback_reason": self._fallback_reason,
            "pillow_version": pillow_version,
            "littlecms_version": littlecms_version,
        }
