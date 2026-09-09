from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from GAED_implementation import GAED


VARIANTS = ("floyd", "gaed_threshold", "gaed_adaptive", "gaed")
PAPER_IMAGE_NAMES = ("boats", "bridge", "airplane", "peppers", "NTUSTlogo", "ESPLlogo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the extracted GAED implementation against paper-style trends.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing evaluation images.")
    parser.add_argument("--manifest", type=Path, help="Optional newline-delimited list of image paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmg", type=float, default=0.1)
    parser.add_argument("--sigma-reconstruct", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images. 0 keeps all.")
    parser.add_argument("--max-side", type=int, default=0, help="Optional resize limit for the longest side. 0 disables resizing.")
    parser.add_argument("--boundary-mode", choices=["renormalize", "strict"], default="renormalize")
    return parser.parse_args()


def resolve_inputs(input_dir: Path | None, manifest: Path | None) -> list[Path]:
    if manifest is not None:
        return [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if input_dir is None:
        raise ValueError("Provide either --input-dir or --manifest.")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in exts)


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def maybe_resize(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(image).resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def save_variant_outputs(output_dir: Path, stem: str, variant: str, halftone: np.ndarray, reconstructed: np.ndarray) -> None:
    Image.fromarray((halftone * 255).astype(np.uint8)).save(output_dir / f"{stem}_{variant}_halftone.png")
    Image.fromarray((np.clip(reconstructed, 0.0, 1.0) * 255).astype(np.uint8)).save(
        output_dir / f"{stem}_{variant}_reconstructed.png"
    )


def paper_name_matches(paths: Iterable[Path]) -> bool:
    lower_names = {path.stem.lower() for path in paths}
    return all(name.lower() in lower_names for name in PAPER_IMAGE_NAMES)


def main() -> int:
    args = parse_args()
    inputs = resolve_inputs(args.input_dir, args.manifest)
    if args.limit > 0:
        inputs = inputs[: args.limit]
    if not inputs:
        raise ValueError("No input images found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = GAED(
        sigma_reconstruct=args.sigma_reconstruct,
        tmg=args.tmg,
        boundary_mode=args.boundary_mode,
    )

    rows: list[dict[str, object]] = []
    for image_path in inputs:
        image = load_image(image_path)
        image = maybe_resize(image, args.max_side)
        stem = image_path.stem
        for variant in VARIANTS:
            halftone = model.halftone_variant(image, variant=variant)
            metrics = model.evaluate(image, halftone)
            save_variant_outputs(args.output_dir, stem, variant, halftone, metrics["reconstructed"])
            rows.append(
                {
                    "image": str(image_path),
                    "stem": stem,
                    "variant": variant,
                    "mse": float(metrics["mse"]),
                    "psnr": float(metrics["psnr"]),
                    "edge_corr": float(metrics["edge_corr"]),
                }
            )

    metrics_csv = args.output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "stem", "variant", "mse", "psnr", "edge_corr"])
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "inputs": [str(path) for path in inputs],
        "paper_image_names_present": paper_name_matches(inputs),
        "boundary_mode": args.boundary_mode,
        "variants": {},
    }
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {
            "psnr_mean": float(np.mean([row["psnr"] for row in variant_rows])),
            "edge_corr_mean": float(np.mean([row["edge_corr"] for row in variant_rows])),
        }

    summary["trend_vs_floyd"] = {
        "gaed_edge_advantage": float(summary["variants"]["gaed"]["edge_corr_mean"]) - float(summary["variants"]["floyd"]["edge_corr_mean"]),
        "gaed_psnr_delta": float(summary["variants"]["gaed"]["psnr_mean"]) - float(summary["variants"]["floyd"]["psnr_mean"]),
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
