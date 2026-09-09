from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cb_dbs import CBDBSConfig, load_rgb_image, run_cb_dbs, save_gray_image, save_rgb_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render full-image CB-DBS halftones.")
    parser.add_argument("--input", nargs="+", required=True, help="Input RGB image paths.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--hvs-kernel-size", type=int, default=11)
    parser.add_argument("--hvs-scale-factor", type=float, default=2000.0)
    parser.add_argument("--hvs-luminance", type=float, default=100.0)
    parser.add_argument("--mono-window-size", type=int, default=3)
    parser.add_argument("--cm-window-size", type=int, default=7)
    """
        cb-dbs-paper-alignment.md states, "Default pass caps were raised from 
        the earlier speed shortcut to `10` with early stopping, matching the paper's 
        statement that convergence typically takes about `10` iterations.", but the implementation
        takes '4' as a  default for the CLI parameters. Likely a missed check.
    """
    parser.add_argument("--mono-passes", type=int, default=10)
    parser.add_argument("--cm-passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def save_panel(path: Path, source: np.ndarray, preview: np.ndarray, rendered: np.ndarray, viewed: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), dpi=150)
    panels = [
        (source, "Source"),
        (preview, "Dot Preview"),
        (rendered, "Subtractive RGB"),
        (viewed, "Viewed Blur"),
    ]
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CBDBSConfig(
        hvs_kernel_size=args.hvs_kernel_size,
        hvs_scale_factor=args.hvs_scale_factor,
        hvs_luminance=args.hvs_luminance,
        mono_window_size=args.mono_window_size,
        cm_window_size=args.cm_window_size,
        mono_passes=args.mono_passes,
        cm_passes=args.cm_passes,
        random_seed=args.seed,
    )

    all_metrics = []
    for input_path in args.input:
        image_path = Path(input_path)
        print(f"[cb-dbs] rendering {image_path}", flush=True)
        source = load_rgb_image(image_path)
        started = time.perf_counter()
        result = run_cb_dbs(source, config)
        runtime = time.perf_counter() - started
        stem_dir = output_dir / image_path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)

        save_rgb_image(stem_dir / "source.png", result.source_rgb)
        save_rgb_image(stem_dir / "preview_rgb.png", result.preview_rgb)
        save_rgb_image(stem_dir / "halftone_rgb.png", result.rendered_rgb)
        save_rgb_image(stem_dir / "viewed_rgb.png", result.viewed_rgb)
        save_gray_image(stem_dir / "c_plane.png", result.c_plane)
        save_gray_image(stem_dir / "m_plane.png", result.m_plane)
        save_gray_image(stem_dir / "y_plane.png", result.y_plane)
        save_gray_image(stem_dir / "cm_density.png", result.cm_density)
        save_gray_image(stem_dir / "cm_total.png", result.cm_total)
        save_gray_image(stem_dir / "blue_mask.png", result.blue_mask)
        save_panel(
            stem_dir / "panel.png",
            result.source_rgb,
            result.preview_rgb,
            result.rendered_rgb,
            result.viewed_rgb,
        )

        metrics = {
            "image": image_path.name,
            "height": int(source.shape[0]),
            "width": int(source.shape[1]),
            "runtime_seconds": runtime,
            **result.metrics,
        }
        with (stem_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        all_metrics.append(metrics)
        print(f"[cb-dbs] done {image_path.name} in {runtime:.2f}s", flush=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)


if __name__ == "__main__":
    main()