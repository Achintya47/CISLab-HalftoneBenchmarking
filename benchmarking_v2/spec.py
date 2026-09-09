"""Constants transcribed from `benchmark_specifications.pdf`.

Keeping these in one module means a spec revision (e.g. a different
reconstruction sigma, a different stress-variant parameter) is a one-line
change here instead of a hunt through adapter code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


# --- Section 5.2: Reconstruction operator ----------------------------------
# "apply a Gaussian observer blur with fixed standard deviation sigma = 1.2"
# This is DELIBERATELY different from the sigma=1.0/1.5 or Nasanen kernels
# used internally by some of the vendored algorithm implementations for their
# own training-time perceptual loss. Benchmark-time reconstruction always
# uses this value, regardless of what an algorithm used internally.
RECONSTRUCTION_SIGMA = 1.2

# Anisotropy is the one full-reference-adjacent metric the spec (Sec. 5.1)
# explicitly says is NOT computed on the reconstructed image -- it is
# computed on the raw halftone dot pattern, since blurring erases the very
# structure the metric measures.
ANISOTROPY_ON_RECONSTRUCTION = False

# --- Section 3/4: content families & dataset keys --------------------------
# The full spec proposes 450 base images / 150 stress-selected images across
# 5 content families (30 images/family -> 450 perturbed items). This project
# runs a scaled-down but structurally identical version, matching the run
# request of "50 images from 5 datasets, 3 stress variants each": 10 base
# images per family (50 total) x 3 variants/family = 150 stress items, i.e.
# an exact 1/3 scale-down of the paper protocol (150 base / 450 stress).
# Both numbers are configurable in config/default.toml.
DEFAULT_IMAGES_PER_FAMILY = 10

CONTENT_FAMILIES: Dict[str, Dict[str, object]] = {
    "edge_text": {
        "dataset_keys": ["financial_data", "rvl_cdip"],
        "properties": ["edge_sharpness", "fine_line_preservation", "text_readability"],
    },
    "scenery_gradient": {
        "dataset_keys": ["kodak", "intel"],
        "properties": ["smooth_gradient_rendering", "tone_accuracy_midtones", "highlight_shadow_detail"],
    },
    "texture": {
        "dataset_keys": ["kth_tips", "dtd"],
        "properties": ["texture_preservation", "microtexture", "local_contrast_preservation"],
    },
    "color_skin": {
        "dataset_keys": ["flowers", "fairface"],
        "properties": ["colour_fidelity", "neutral_gray_balance", "saturated_colour", "skin_tone_accuracy"],
    },
    "pattern": {
        "dataset_keys": ["bsds500"],
        "properties": ["periodic_pattern_handling", "directional_isotropy", "structure_preservation"],
    },
}

# --- Section 2: evaluation tracks -------------------------------------------
GRAYSCALE_METRICS = ("psnr", "ssim", "lpips", "anisotropy_index")
COLOR_METRICS = ("psnr", "ssim", "lpips", "anisotropy_index", "delta_e00")

# --- Section 4 / Table 1: per-family stress-variant parameters -------------
# These are the numeric knobs from Table 1; the callables that *use* them
# live in stress_variants.py so this module stays purely declarative.

DIRECTIONAL_BLUR = {"length_px": 9, "angles_deg": (0, 90)}
DOWN_UP_SAMPLE = {"scale": 0.5}
EDGE_TEXT_CHART = {"edge_angles_deg": (15, 30, 45), "stroke_widths_px": (1, 2, 4), "font_sizes_pt": (8, 12, 16)}

GAUSSIAN_BLUR_COLLAPSE = {"sigma": 1.5}
QUANTIZE_JPEG = {"bit_depth": 6, "jpeg_quality": 35}
RAMP_CHART = {"contrast_pct": (1, 2, 4)}

ADDITIVE_NOISE = {"std": 3 / 255}
MILD_BLUR = {"sigma": 0.8}
TEXTURE_PATCH = {"michelson_contrast": (0.05, 0.10, 0.20)}

JPEG_COMPRESS = {"jpeg_quality": 25}
GRAYSCALE_RAMP = {"luminance_pct": (25, 50, 75)}
COLOR_PATCHES = {"saturation_pct": (60, 80, 100)}

ORIENTATION_ROTATION = {"angle_deg": 7.5}
CHECKERBOARD_STRIPES = {"periods_px": (4, 8, 16), "angles_deg": (0, 45, 90)}
FINE_LINE_PATTERN = {"line_widths_px": (1, 2, 4)}


@dataclass(frozen=True)
class VariantSpec:
    key: str
    description: str


FAMILY_VARIANTS: Dict[str, List[VariantSpec]] = {
    "edge_text": [
        VariantSpec("directional_blur", "Motion blur, length 9px, angle 0/90 deg"),
        VariantSpec("downsample_upsample", "Downsample 0.5x, bicubic upsample back"),
        VariantSpec("edge_text_chart", "Slanted-edge + text readability chart"),
    ],
    "scenery_gradient": [
        VariantSpec("gaussian_blur_collapse", "Gaussian blur sigma=1.5"),
        VariantSpec("quantize_jpeg", "6-bit quantization + JPEG Q35"),
        VariantSpec("ramp_chart", "Linear/radial/soft ramps with highlight/shadow inserts"),
    ],
    "texture": [
        VariantSpec("additive_noise", "Additive Gaussian noise, std 3/255"),
        VariantSpec("mild_blur", "Mild Gaussian blur sigma=0.8"),
        VariantSpec("texture_patch", "Low-contrast texture patch at Michelson contrast levels"),
    ],
    "color_skin": [
        VariantSpec("jpeg_compress", "JPEG Q25"),
        VariantSpec("grayscale_ramp", "Neutral gray ramps/patches"),
        VariantSpec("color_patches", "RGBCMY patches at saturation levels"),
    ],
    "pattern": [
        VariantSpec("orientation_rotation", "Rotation by 7.5 deg and recrop"),
        VariantSpec("checkerboard_stripes", "Checkerboards/stripes at multiple periods/angles"),
        VariantSpec("fine_line_pattern", "Fine-line targets at multiple widths/orientations"),
    ],
}
