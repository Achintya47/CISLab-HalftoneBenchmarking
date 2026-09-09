"""Table 1 stress variants, one function per (family, slot).

Each function has the signature `(rgb: np.ndarray, rng: np.random.Generator)
-> np.ndarray` and returns a same-size float RGB image in [0, 1]. Two kinds
of variants exist per the spec:

  * "perturbation" variants transform the given base image (blur, noise,
    compression, rotation, resample).
  * "chart" variants are synthetic test targets (text/edge charts, ramps,
    checkerboards, fine lines, colour patches) -- per the spec these probe
    a specific failure mode directly rather than degrading a photo, so they
    ignore the input pixels but keep its size/seed for determinism.

`FAMILY_VARIANT_FUNCS[family]` is an ordered list of exactly 3 callables,
matching `spec.FAMILY_VARIANTS[family]`.
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, rotate

from . import spec

Variant = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _from_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    kernel[size // 2, :] = 1.0
    pil_kernel = Image.fromarray((kernel * 255).astype(np.uint8))
    pil_kernel = pil_kernel.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    kernel = np.asarray(pil_kernel, dtype=np.float64)
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    image = Image.fromarray(_to_uint8(rgb), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return _from_uint8(np.asarray(decoded.convert("RGB")))


# ---------------------------------------------------------------- edge_text
def directional_blur(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.choice(spec.DIRECTIONAL_BLUR["angles_deg"]))
    kernel = _motion_blur_kernel(spec.DIRECTIONAL_BLUR["length_px"], angle)
    from scipy.signal import fftconvolve

    channels = [np.clip(fftconvolve(rgb[..., c], kernel, mode="same"), 0.0, 1.0) for c in range(3)]
    return np.stack(channels, axis=-1)


def downsample_upsample(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = spec.DOWN_UP_SAMPLE["scale"]
    small = Image.fromarray(_to_uint8(rgb)).resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
    back = small.resize((width, height), Image.BICUBIC)
    return _from_uint8(np.asarray(back))


def edge_text_chart(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    angles = spec.EDGE_TEXT_CHART["edge_angles_deg"]
    widths = spec.EDGE_TEXT_CHART["stroke_widths_px"]
    band_h = height / max(1, len(angles))
    for i, (angle, stroke) in enumerate(zip(angles, widths)):
        y0 = i * band_h
        import math

        dx = math.cos(math.radians(angle)) * width
        dy = math.sin(math.radians(angle)) * band_h
        draw.line([(0, y0), (dx, y0 + dy)], fill=(0, 0, 0), width=max(1, int(stroke)))
    sizes = spec.EDGE_TEXT_CHART["font_sizes_pt"]
    for i, font_size in enumerate(sizes):
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((4, height - (i + 1) * (font_size + 4)), "Halftone 0123 Aa", fill=(0, 0, 0), font=font)
    return _from_uint8(np.asarray(image))


# ----------------------------------------------------------- scenery_gradient
def gaussian_blur_collapse(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = spec.GAUSSIAN_BLUR_COLLAPSE["sigma"]
    return np.clip(np.stack([gaussian_filter(rgb[..., c], sigma=sigma) for c in range(3)], axis=-1), 0.0, 1.0)


def quantize_jpeg(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bits = spec.QUANTIZE_JPEG["bit_depth"]
    levels = 2**bits
    quantized = np.round(rgb * (levels - 1)) / (levels - 1)
    return _jpeg_roundtrip(quantized, spec.QUANTIZE_JPEG["jpeg_quality"])


def ramp_chart(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    linear = xx
    yc, xc = height / 2.0, width / 2.0
    radial = np.clip(np.sqrt((yy * height - yc) ** 2 + (xx * width - xc) ** 2) / np.hypot(xc, yc), 0.0, 1.0)
    soft = 0.5 + 0.5 * np.sin(2 * np.pi * xx)
    band_h = height // 3
    ramp = np.zeros((height, width), dtype=np.float64)
    ramp[0:band_h] = linear[0:band_h]
    ramp[band_h : 2 * band_h] = radial[band_h : 2 * band_h]
    ramp[2 * band_h :] = soft[2 * band_h :]
    for i, pct in enumerate(spec.RAMP_CHART["contrast_pct"]):
        cx = int(width * (i + 1) / (len(spec.RAMP_CHART["contrast_pct"]) + 1))
        patch = 10
        ramp[0:patch, max(0, cx - patch) : cx + patch] = 1.0 - pct / 100.0  # highlight insert
        ramp[-patch:, max(0, cx - patch) : cx + patch] = pct / 100.0  # shadow insert
    return np.repeat(np.clip(ramp, 0.0, 1.0)[..., None], 3, axis=-1)


# ------------------------------------------------------------------- texture
def additive_noise(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, spec.ADDITIVE_NOISE["std"], size=rgb.shape)
    return np.clip(rgb + noise, 0.0, 1.0)


def mild_blur(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = spec.MILD_BLUR["sigma"]
    return np.clip(np.stack([gaussian_filter(rgb[..., c], sigma=sigma) for c in range(3)], axis=-1), 0.0, 1.0)


def texture_patch(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    contrasts = spec.TEXTURE_PATCH["michelson_contrast"]
    base = rng.random((height, width))
    base = gaussian_filter(base, sigma=1.0)
    base = (base - base.min()) / max(1e-8, base.max() - base.min())
    out = np.full((height, width), 0.5)
    band = height // len(contrasts)
    mean = 0.5
    for i, contrast in enumerate(contrasts):
        amplitude = mean * contrast
        out[i * band : (i + 1) * band] = mean + amplitude * (2 * base[i * band : (i + 1) * band] - 1)
    return np.repeat(np.clip(out, 0.0, 1.0)[..., None], 3, axis=-1)


# ---------------------------------------------------------------- color_skin
def jpeg_compress(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _jpeg_roundtrip(rgb, spec.JPEG_COMPRESS["jpeg_quality"])


def grayscale_ramp(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    levels = spec.GRAYSCALE_RAMP["luminance_pct"]
    band = height // len(levels)
    out = np.zeros((height, width))
    for i, pct in enumerate(levels):
        out[i * band : (i + 1) * band] = pct / 100.0
    return np.repeat(out[..., None], 3, axis=-1)


def color_patches(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    hues = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]  # RGBCMY
    saturations = spec.COLOR_PATCHES["saturation_pct"]
    out = np.zeros((height, width, 3))
    n_cols = len(hues)
    col_w = width // n_cols
    row_h = height // len(saturations)
    for row, sat_pct in enumerate(saturations):
        sat = sat_pct / 100.0
        for col, hue in enumerate(hues):
            color = np.array(hue) * sat + (1 - sat) * 0.5
            out[row * row_h : (row + 1) * row_h, col * col_w : (col + 1) * col_w] = color
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------- pattern
def orientation_rotation(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = spec.ORIENTATION_ROTATION["angle_deg"]
    height, width = rgb.shape[:2]
    rotated = np.stack([rotate(rgb[..., c], angle, reshape=False, mode="reflect") for c in range(3)], axis=-1)
    return np.clip(rotated, 0.0, 1.0)


def checkerboard_stripes(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    period = int(rng.choice(spec.CHECKERBOARD_STRIPES["periods_px"]))
    angle = float(rng.choice(spec.CHECKERBOARD_STRIPES["angles_deg"]))
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    import math

    theta = math.radians(angle)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    stripes = (np.floor(proj / period) % 2).astype(np.float64)
    return np.repeat(stripes[..., None], 3, axis=-1)


def fine_line_pattern(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    widths = spec.FINE_LINE_PATTERN["line_widths_px"]
    out = np.ones((height, width))
    band = height // len(widths)
    for i, line_width in enumerate(widths):
        period = max(2, line_width * 4)
        yy, xx = np.mgrid[i * band : (i + 1) * band, 0:width]
        out[i * band : (i + 1) * band] = ((xx % period) < line_width).astype(np.float64)
    return np.repeat((1.0 - out)[..., None], 3, axis=-1)


FAMILY_VARIANT_FUNCS: Dict[str, List[Variant]] = {
    "edge_text": [directional_blur, downsample_upsample, edge_text_chart],
    "scenery_gradient": [gaussian_blur_collapse, quantize_jpeg, ramp_chart],
    "texture": [additive_noise, mild_blur, texture_patch],
    "color_skin": [jpeg_compress, grayscale_ramp, color_patches],
    "pattern": [orientation_rotation, checkerboard_stripes, fine_line_pattern],
}


def apply_family_variants(family: str, rgb: np.ndarray, seed: int) -> List[tuple[str, np.ndarray]]:
    funcs = FAMILY_VARIANT_FUNCS[family]
    names = [v.key for v in spec.FAMILY_VARIANTS[family]]
    outputs = []
    for name, func in zip(names, funcs):
        rng = np.random.default_rng(seed)
        outputs.append((name, np.clip(func(rgb, rng), 0.0, 1.0)))
    return outputs
