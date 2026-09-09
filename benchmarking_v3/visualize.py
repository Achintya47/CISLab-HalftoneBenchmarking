"""Visual comparison panels for the CMYK/ICC pipeline: Original RGB, each
of the four halftoned colorant planes, and the final ICC round-tripped
Reconstructed RGB -- one row per sample, one panel per method.

Sample selection reuses `benchmarking_v2.visualize.select_sample_item_indices`
verbatim (spread across content families, adapts down if fewer items exist)
since nothing about how samples are chosen changes in v3, only what's drawn
for each one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarking_v2.visualize import select_sample_item_indices  # noqa: F401  (re-exported for run_benchmark.py)

from .cmyk_pipeline import CHANNEL_NAMES
from .reconstruction import RECONSTRUCTION_SIGMA

CELL = 150
LABEL_WIDTH = 190
HEADER_HEIGHT = 30
GRID_COLOR = (205, 205, 205)
COLUMN_TITLES = ("Original (RGB)", "C halftone", "M halftone", "Y halftone", "K halftone", f"Recon. RGB (s={RECONSTRUCTION_SIGMA})")


def _load_font():
    try:
        return ImageFont.load_default(size=12)
    except TypeError:
        return ImageFont.load_default()


def _rgb_panel(rgb: np.ndarray, cell: int) -> Image.Image:
    array = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")
    if image.size != (cell, cell):
        image = image.resize((cell, cell), Image.BICUBIC)
    return image


def _plane_panel(plane: np.ndarray, cell: int) -> Image.Image:
    array = np.clip(np.asarray(plane, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L").convert("RGB")
    if image.size != (cell, cell):
        image = image.resize((cell, cell), Image.NEAREST)  # keep individual dots crisp
    return image


def build_method_panel(method_name: str, samples: Sequence[Dict[str, Any]], *, cell: int = CELL) -> Image.Image:
    """`samples`: dicts with keys `family`, `variant`, `image_id`,
    `reference_rgb` (HxWx3 [0,1]), `halftone_planes` (dict C/M/Y/K -> HxW
    [0,1]), `reconstructed_rgb` (HxWx3 [0,1])."""
    rows = max(1, len(samples))
    n_cols = len(COLUMN_TITLES)
    width = LABEL_WIDTH + n_cols * cell
    height = HEADER_HEIGHT + rows * cell
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font()

    draw.text((6, 6), method_name, fill=(0, 0, 0), font=font)
    for col, title in enumerate(COLUMN_TITLES):
        draw.text((LABEL_WIDTH + col * cell + 4, 6), title, fill=(0, 0, 0), font=font)

    if not samples:
        draw.text((6, HEADER_HEIGHT + 6), "(no samples available)", fill=(120, 120, 120), font=font)
        return canvas

    for row, sample in enumerate(samples):
        y = HEADER_HEIGHT + row * cell
        label = f"{sample['family']}\n{sample['variant']}\n{sample['image_id']}"
        draw.multiline_text((6, y + 6), label, fill=(0, 0, 0), font=font, spacing=2)

        canvas.paste(_rgb_panel(sample["reference_rgb"], cell), (LABEL_WIDTH + 0 * cell, y))
        for index, channel in enumerate(CHANNEL_NAMES, start=1):
            canvas.paste(_plane_panel(sample["halftone_planes"][channel], cell), (LABEL_WIDTH + index * cell, y))
        canvas.paste(_rgb_panel(sample["reconstructed_rgb"], cell), (LABEL_WIDTH + (n_cols - 1) * cell, y))

    for col in range(n_cols + 1):
        x = LABEL_WIDTH + col * cell if col > 0 else LABEL_WIDTH
        draw.line([(x, HEADER_HEIGHT), (x, height)], fill=GRID_COLOR)
    for row in range(rows + 1):
        y = HEADER_HEIGHT + row * cell
        draw.line([(0, y), (width, y)], fill=GRID_COLOR)
    return canvas


def save_method_panel(method_name: str, samples: Sequence[Dict[str, Any]], output_dir: Path, *, cell: int = CELL) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_method_panel(method_name, samples, cell=cell)
    safe_name = method_name.replace("/", "_").replace(" ", "_")
    path = output_dir / f"{safe_name}_cmyk_comparison.png"
    panel.save(path)
    return path
