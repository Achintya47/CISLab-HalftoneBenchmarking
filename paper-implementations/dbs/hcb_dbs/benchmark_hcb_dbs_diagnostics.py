from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from hcb_dbs import HCBDBSConfig, run_hcb_dbs


def image_to_rgb_u8(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def rgb_f32_to_u8(array: np.ndarray) -> np.ndarray:
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def luma_u8(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def power_spectrum_image(gray_u8: np.ndarray) -> Image.Image:
    centered = gray_u8.astype(np.float32) / 255.0 - np.mean(gray_u8.astype(np.float32) / 255.0)
    power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    log = np.log1p(power)
    log -= log.min()
    if log.max() > 0:
        log /= log.max()
    return Image.fromarray(np.uint8(np.round(log * 255.0)), mode="L")


def radial_profile(gray_u8: np.ndarray) -> list[float]:
    centered = gray_u8.astype(np.float32) / 255.0 - np.mean(gray_u8.astype(np.float32) / 255.0)
    power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    height, width = power.shape
    yy, xx = np.indices((height, width))
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
    profile: list[float] = []
    for radius in range(int(rr.max()) + 1):
        mask = rr == radius
        profile.append(float(power[mask].mean()) if np.any(mask) else 0.0)
    total = sum(profile)
    if total > 0:
        profile = [value / total for value in profile]
    return profile


def anisotropy_score(gray_u8: np.ndarray) -> float:
    centered = gray_u8.astype(np.float32) / 255.0 - np.mean(gray_u8.astype(np.float32) / 255.0)
    power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    total = power.sum()
    if total <= 0:
        return 0.0
    power /= total
    height, width = power.shape
    yy, xx = np.indices((height, width))
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
    values: list[float] = []
    for radius in range(1, int(rr.max()) + 1):
        mask = rr == radius
        if mask.sum() <= 1:
            continue
        ring = power[mask]
        mean = ring.mean()
        if mean <= 1e-12:
            continue
        values.append(float(np.mean((ring - mean) ** 2)))
    return float(np.mean(values)) if values else 0.0


def make_zoom_box(width: int, height: int, fraction: float = 0.18) -> tuple[int, int, int, int]:

    """
        Silently fails for images smaller than 32 Pixels, ImageDraw.rectangle / 
        Image.crop don't throw errors on out-of-bounds, thus this fails silently.
    """
    crop_size = crop_size = min(max(32, int(min(width, height) * fraction)), min(width, height))
    x = max(0, (width - crop_size) // 2)
    y = max(0, (height - crop_size) // 2)
    return x, y, crop_size, crop_size


def add_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    x, y, w, h = box
    draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(220, 40, 40), width=2)
    return rgb


def crop_and_zoom(image: Image.Image, box: tuple[int, int, int, int], scale: int = 4) -> Image.Image:
    x, y, w, h = box
    crop = image.crop((x, y, x + w, y + h))
    return crop.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)


def stack_rows(rows: Iterable[tuple[str, Image.Image]], label_width: int = 140, background: str = "white") -> Image.Image:
    row_list = list(rows)
    heights = [max(36, image.size[1]) for _label, image in row_list]
    width = label_width + max(image.size[0] for _label, image in row_list)
    canvas = Image.new("RGB", (width, sum(heights)), color=background)
    y = 0
    for (label, image), row_height in zip(row_list, heights):
        label_image = Image.new("RGB", (label_width, row_height), color=background)
        draw = ImageDraw.Draw(label_image)
        draw.text((8, 8), label, fill=(0, 0, 0))
        canvas.paste(label_image, (0, y))
        canvas.paste(image, (label_width, y))
        y += row_height
    return canvas


def save_diagnostics(
    image_name: str,
    source_rgb: np.ndarray,
    preview_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    viewed_rgb: np.ndarray,
    output_dir: Path,
) -> None:
    source_image = Image.fromarray(source_rgb, mode="RGB")
    preview_image = Image.fromarray(preview_rgb, mode="RGB")
    rendered_image = Image.fromarray(rendered_rgb, mode="RGB")
    viewed_image = Image.fromarray(viewed_rgb, mode="RGB")
    box = make_zoom_box(source_image.size[0], source_image.size[1])

    overview = stack_rows(
        [
            ("source", add_box(source_image, box)),
            ("preview_rgb", add_box(preview_image, box)),
            ("rendered_rgb", add_box(rendered_image, box)),
            ("viewed_rgb", add_box(viewed_image, box)),
        ]
    )
    zoom = stack_rows(
        [
            ("source_zoom", crop_and_zoom(source_image, box)),
            ("preview_zoom", crop_and_zoom(preview_image, box)),
            ("rendered_zoom", crop_and_zoom(rendered_image, box)),
            ("viewed_zoom", crop_and_zoom(viewed_image, box)),
        ]
    )
    fft = stack_rows(
        [
            ("source_luma", power_spectrum_image(luma_u8(source_rgb)).convert("RGB")),
            ("preview_luma", power_spectrum_image(luma_u8(preview_rgb)).convert("RGB")),
            ("rendered_luma", power_spectrum_image(luma_u8(rendered_rgb)).convert("RGB")),
            ("viewed_luma", power_spectrum_image(luma_u8(viewed_rgb)).convert("RGB")),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    overview.save(output_dir / f"{image_name}-overview.png")
    zoom.save(output_dir / f"{image_name}-zoom.png")
    fft.save(output_dir / f"{image_name}-fft.png")
    profiles = {
        "source_luma": radial_profile(luma_u8(source_rgb)),
        "preview_luma": radial_profile(luma_u8(preview_rgb)),
        "rendered_luma": radial_profile(luma_u8(rendered_rgb)),
        "viewed_luma": radial_profile(luma_u8(viewed_rgb)),
    }
    (output_dir / f"{image_name}-radial-profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, float]:
    metrics = ["runtime_sec", "viewed_rgb_mse", "viewed_luma_mse", "preview_luma_anisotropy", "viewed_luma_anisotropy", "source_luma_mean", "viewed_luma_mean"]
    summary: dict[str, float] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kodak and constant-gray diagnostics for HCB-DBS.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/kodak/images"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper-implementations/dbs/hcb-dbs/output/kodak-diagnostics-maxside-128"),
    )
    parser.add_argument("--max-side", type=int, default=128, help="Resize the longer side before running HCB-DBS. Use 0 for native size.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on Kodak images. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostic-images", nargs="*", default=["kodim01", "kodim07", "kodim12", "kodim15", "kodim18"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(args.dataset_root.glob("*.png"))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No Kodak PNGs found under {args.dataset_root}")

    output_dir = args.output_dir
    method_output_dir = output_dir / "method_outputs"
    diagnostics_dir = output_dir / "diagnostics"
    constant_gray_dir = output_dir / "constant_gray_diagnostics"
    for subdir in [
        method_output_dir / "preview_rgb",
        method_output_dir / "rendered_rgb",
        method_output_dir / "viewed_rgb",
        diagnostics_dir,
        constant_gray_dir,
    ]:
        subdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    selected = set(args.diagnostic_images)

    for index, image_path in enumerate(image_paths):
        print(f"[hcb-dbs] {index + 1}/{len(image_paths)} {image_path.stem}", flush=True)
        source_rgb_u8 = image_to_rgb_u8(image_path, args.max_side)
        source_rgb = source_rgb_u8.astype(np.float32) / 255.0
        start = time.perf_counter()
        result = run_hcb_dbs(source_rgb, HCBDBSConfig(random_seed=args.seed + index))
        runtime = time.perf_counter() - start

        preview_rgb_u8 = rgb_f32_to_u8(result.preview_rgb)
        rendered_rgb_u8 = rgb_f32_to_u8(result.rendered_rgb)
        viewed_rgb_u8 = rgb_f32_to_u8(result.viewed_rgb)

        Image.fromarray(preview_rgb_u8, mode="RGB").save(method_output_dir / "preview_rgb" / f"{image_path.stem}.png")
        Image.fromarray(rendered_rgb_u8, mode="RGB").save(method_output_dir / "rendered_rgb" / f"{image_path.stem}.png")
        Image.fromarray(viewed_rgb_u8, mode="RGB").save(method_output_dir / "viewed_rgb" / f"{image_path.stem}.png")

        source_luma = luma_u8(source_rgb_u8)
        preview_luma = luma_u8(preview_rgb_u8)
        viewed_luma = luma_u8(viewed_rgb_u8)
        row = {
            "image": image_path.stem,
            "runtime_sec": runtime,
            "source_luma_mean": float(source_luma.mean() / 255.0),
            "viewed_luma_mean": float(viewed_luma.mean() / 255.0),
            "viewed_rgb_mse": float(np.mean((viewed_rgb_u8.astype(np.float32) - source_rgb_u8.astype(np.float32)) ** 2) / (255.0 ** 2)),
            "viewed_luma_mse": float(np.mean((viewed_luma.astype(np.float32) - source_luma.astype(np.float32)) ** 2) / (255.0 ** 2)),
            "preview_luma_anisotropy": anisotropy_score(preview_luma),
            "viewed_luma_anisotropy": anisotropy_score(viewed_luma),
        }
        rows.append(row)

        if image_path.stem in selected:
            save_diagnostics(
                image_name=image_path.stem,
                source_rgb=source_rgb_u8,
                preview_rgb=preview_rgb_u8,
                rendered_rgb=rendered_rgb_u8,
                viewed_rgb=viewed_rgb_u8,
                output_dir=diagnostics_dir,
            )

    summary = summarize_rows(rows)
    with (output_dir / "kodak_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_lines = [
        "# HCB-DBS Kodak Diagnostics",
        "",
        f"Dataset: `{args.dataset_root}` ({len(image_paths)} images)",
        f"Max side: `{args.max_side}`",
        "",
        "## Mean Metrics",
        "",
        f"- runtime_sec_mean: `{summary['runtime_sec_mean']:.4f}`",
        f"- viewed_rgb_mse_mean: `{summary['viewed_rgb_mse_mean']:.6f}`",
        f"- viewed_luma_mse_mean: `{summary['viewed_luma_mse_mean']:.6f}`",
        f"- preview_luma_anisotropy_mean: `{summary['preview_luma_anisotropy_mean']:.6e}`",
        f"- viewed_luma_anisotropy_mean: `{summary['viewed_luma_anisotropy_mean']:.6e}`",
        "",
        "## Notes",
        "",
        "- `preview_rgb` is the additive dot-color visualization returned by `run_hcb_dbs`.",
        "- `rendered_rgb` is the subtractive rendering before viewing blur.",
        "- `viewed_rgb` is the blurred viewing model output used for visual diagnostics.",
        "- FFT/radial diagnostics are computed on luminance images, because HCB-DBS is a color algorithm.",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    levels = [64, 128, 192]
    constant_summary: dict[str, dict[str, float]] = {}
    for level in levels:
        source = np.full((args.max_side if args.max_side > 0 else 192, args.max_side if args.max_side > 0 else 192, 3), level, dtype=np.uint8)
        result = run_hcb_dbs(source.astype(np.float32) / 255.0, HCBDBSConfig(random_seed=args.seed + level))
        preview_rgb_u8 = rgb_f32_to_u8(result.preview_rgb)
        rendered_rgb_u8 = rgb_f32_to_u8(result.rendered_rgb)
        viewed_rgb_u8 = rgb_f32_to_u8(result.viewed_rgb)

        constant_summary[str(level)] = {
            "preview_luma_mean": float(luma_u8(preview_rgb_u8).mean() / 255.0),
            "rendered_luma_mean": float(luma_u8(rendered_rgb_u8).mean() / 255.0),
            "viewed_luma_mean": float(luma_u8(viewed_rgb_u8).mean() / 255.0),
            "preview_luma_anisotropy": anisotropy_score(luma_u8(preview_rgb_u8)),
            "viewed_luma_anisotropy": anisotropy_score(luma_u8(viewed_rgb_u8)),
        }

        rows_to_save = [
            ("source", Image.fromarray(source, mode="RGB")),
            ("preview_rgb", Image.fromarray(preview_rgb_u8, mode="RGB")),
            ("rendered_rgb", Image.fromarray(rendered_rgb_u8, mode="RGB")),
            ("viewed_rgb", Image.fromarray(viewed_rgb_u8, mode="RGB")),
        ]
        panel = stack_rows(rows_to_save)
        fft_panel = stack_rows(
            [
                ("source_luma", power_spectrum_image(luma_u8(source)).convert("RGB")),
                ("preview_luma", power_spectrum_image(luma_u8(preview_rgb_u8)).convert("RGB")),
                ("rendered_luma", power_spectrum_image(luma_u8(rendered_rgb_u8)).convert("RGB")),
                ("viewed_luma", power_spectrum_image(luma_u8(viewed_rgb_u8)).convert("RGB")),
            ]
        )
        canvas = Image.new("RGB", (max(panel.size[0], fft_panel.size[0]), panel.size[1] + fft_panel.size[1]), color="white")
        canvas.paste(panel, (0, 0))
        canvas.paste(fft_panel, (0, panel.size[1]))
        canvas.save(constant_gray_dir / f"gray_{level}.png")

    (constant_gray_dir / "summary.json").write_text(json.dumps(constant_summary, indent=2), encoding="utf-8")
    print(f"Wrote HCB-DBS diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
