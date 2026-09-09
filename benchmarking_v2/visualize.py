"""Visual comparison panels: original (continuous-tone) vs. binary halftone
vs. this benchmark's R(H) reconstruction (Gaussian, sigma=1.2, spec Sec.
5.2), per method.

Grayscale/luma space is used uniformly across all methods -- including
color-capable ones (DBS, Ordered Dithering) -- since that's the space every
method is actually scored in for the primary track, and it keeps panels
directly comparable across methods that are grayscale-only (Error
Diffusion, Deep Learning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .reconstruction import RECONSTRUCTION_SIGMA

CELL = 170
LABEL_WIDTH = 190
HEADER_HEIGHT = 30
GRID_COLOR = (205, 205, 205)
COLUMN_TITLES = ("Original", "Halftone (binary)", f"Recon. sigma={RECONSTRUCTION_SIGMA}")


def _load_font():
    try:
        return ImageFont.load_default(size=13)
    except TypeError:  # older Pillow without the `size` kwarg
        return ImageFont.load_default()


def _to_rgb_image(gray: np.ndarray, cell: int, *, preserve_dots: bool) -> Image.Image:
    array = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    image = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="L")
    if image.size != (cell, cell):
        # NEAREST for the halftone keeps individual dots crisp instead of
        # blurring them into gray mush when up/down-scaling for the grid.
        resample = Image.NEAREST if preserve_dots else Image.BICUBIC
        image = image.resize((cell, cell), resample)
    return image.convert("RGB")


def select_sample_item_indices(item_families: Sequence[str], item_variants: Sequence[str], count: int) -> List[int]:
    """Pick up to `count` items spread across content families, preferring
    each family's unperturbed "base" item first so the panel shows a clean
    representative sample per family before spending extra slots on stress
    variants. Adapts down automatically if fewer than `count` items exist.
    """
    if count <= 0 or not item_families:
        return []
    by_family: Dict[str, List[int]] = {}
    for index, family in enumerate(item_families):
        by_family.setdefault(family, []).append(index)
    families = list(by_family.keys())

    selected: List[int] = []
    prefer_base = True
    while len(selected) < count and any(by_family[f] for f in families):
        progressed = False
        for family in families:
            if len(selected) >= count:
                break
            pool = by_family[family]
            if not pool:
                continue
            chosen = None
            if prefer_base:
                for position, idx in enumerate(pool):
                    if item_variants[idx] == "base":
                        chosen = pool.pop(position)
                        break
            if chosen is None:
                chosen = pool.pop(0)
            selected.append(chosen)
            progressed = True
        prefer_base = False  # only prefer "base" during the first full pass
        if not progressed:
            break
    return selected[:count]


def build_method_panel(method_name: str, samples: Sequence[Dict[str, Any]], *, cell: int = CELL) -> Image.Image:
    """`samples`: list of dicts with keys `family`, `variant`, `image_id`,
    `reference` (2D [0,1] array), `halftone` (2D [0,1] array), `reconstructed`
    (2D [0,1] array)."""
    rows = max(1, len(samples))
    width = LABEL_WIDTH + 3 * cell
    height = HEADER_HEIGHT + rows * cell
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font()

    draw.text((6, 6), method_name, fill=(0, 0, 0), font=font)
    for col, title in enumerate(COLUMN_TITLES):
        draw.text((LABEL_WIDTH + col * cell + 6, 6), title, fill=(0, 0, 0), font=font)

    if not samples:
        draw.text((6, HEADER_HEIGHT + 6), "(no samples available)", fill=(120, 120, 120), font=font)
        return canvas

    for row, sample in enumerate(samples):
        y = HEADER_HEIGHT + row * cell
        label = f"{sample['family']}\n{sample['variant']}\n{sample['image_id']}"
        draw.multiline_text((6, y + 6), label, fill=(0, 0, 0), font=font, spacing=2)

        canvas.paste(_to_rgb_image(sample["reference"], cell, preserve_dots=False), (LABEL_WIDTH + 0 * cell, y))
        canvas.paste(_to_rgb_image(sample["halftone"], cell, preserve_dots=True), (LABEL_WIDTH + 1 * cell, y))
        canvas.paste(_to_rgb_image(sample["reconstructed"], cell, preserve_dots=False), (LABEL_WIDTH + 2 * cell, y))

    for col in range(4):
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
    path = output_dir / f"{safe_name}_comparison.png"
    panel.save(path)
    return path


def build_cross_method_panel(
    reference: np.ndarray,
    halftones_by_method: Dict[str, np.ndarray],
    reconstructions_by_method: Dict[str, np.ndarray],
    *,
    label: str,
    cell: int = CELL,
) -> Image.Image:
    """One image, one row per method: Original | Halftone | Reconstructed,
    so methods can be compared directly on identical input."""
    methods = list(halftones_by_method)
    rows = len(methods)
    header_height = 2 * HEADER_HEIGHT  # two lines: sample identity, then column titles
    width = LABEL_WIDTH + 3 * cell
    height = header_height + rows * cell
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font()

    draw.text((6, 6), f"Sample: {label}", fill=(0, 0, 0), font=font)
    draw.text((6, HEADER_HEIGHT + 6), "Method", fill=(0, 0, 0), font=font)
    for col, title in enumerate(COLUMN_TITLES):
        draw.text((LABEL_WIDTH + col * cell + 6, HEADER_HEIGHT + 6), title, fill=(0, 0, 0), font=font)
    draw.line([(0, HEADER_HEIGHT), (width, HEADER_HEIGHT)], fill=GRID_COLOR)

    for row, method_name in enumerate(methods):
        y = header_height + row * cell
        draw.text((6, y + 6), method_name, fill=(0, 0, 0), font=font)
        canvas.paste(_to_rgb_image(reference, cell, preserve_dots=False), (LABEL_WIDTH + 0 * cell, y))
        canvas.paste(_to_rgb_image(halftones_by_method[method_name], cell, preserve_dots=True), (LABEL_WIDTH + 1 * cell, y))
        canvas.paste(_to_rgb_image(reconstructions_by_method[method_name], cell, preserve_dots=False), (LABEL_WIDTH + 2 * cell, y))

    for col in range(4):
        x = LABEL_WIDTH + col * cell if col > 0 else LABEL_WIDTH
        draw.line([(x, header_height), (x, height)], fill=GRID_COLOR)
    for row in range(rows + 1):
        y = header_height + row * cell
        draw.line([(0, y), (width, y)], fill=GRID_COLOR)
    return canvas


def save_cross_method_panel(
    reference: np.ndarray,
    halftones_by_method: Dict[str, np.ndarray],
    reconstructions_by_method: Dict[str, np.ndarray],
    *,
    label: str,
    output_dir: Path,
    cell: int = CELL,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_cross_method_panel(reference, halftones_by_method, reconstructions_by_method, label=label, cell=cell)
    path = output_dir / "all_methods_comparison.png"
    panel.save(path)
    return path
