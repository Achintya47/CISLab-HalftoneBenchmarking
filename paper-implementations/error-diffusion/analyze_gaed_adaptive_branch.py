from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import GAED_implementation as gaed


DIRECTION_LABELS = {
    -1: "none",
    0: "right",
    1: "down_left",
    2: "down",
    3: "down_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GAED adaptive-filter behavior on a full image and dump a crop-level diagnosis.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--top", type=int, default=-1, help="Optional crop top. Use -1 for auto.")
    parser.add_argument("--left", type=int, default=-1, help="Optional crop left. Use -1 for auto.")
    parser.add_argument("--boundary-mode", choices=["renormalize", "strict"], default="strict")
    return parser.parse_args()


def auto_pick_crop(g: np.ndarray, crop_size: int) -> tuple[int, int]:
    magnitude, _ = gaed._sobel_gradient(g)
    height, width = magnitude.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    best_score = -1.0
    best = (0, 0)
    step = max(8, crop_size // 4)
    for top in range(0, max(1, height - crop_h + 1), step):
        for left in range(0, max(1, width - crop_w + 1), step):
            score = float(magnitude[top : top + crop_h, left : left + crop_w].mean())
            if score > best_score:
                best_score = score
                best = (top, left)
    return best


def run_analysis(g: np.ndarray, boundary_mode: str) -> dict[str, np.ndarray]:
    rows, cols = g.shape
    ge = g.copy()
    halftone = np.zeros((rows, cols), dtype=np.uint8)
    use_adaptive = np.zeros((rows, cols), dtype=np.uint8)
    dfmg_map = np.zeros((rows, cols), dtype=np.float64)
    dg_map = np.zeros((rows, cols), dtype=np.float64)
    dt_map = np.zeros((rows, cols), dtype=np.float64)
    top_r_map = np.zeros((rows, cols), dtype=np.float64)
    second_r_map = np.zeros((rows, cols), dtype=np.float64)
    gap_r_map = np.zeros((rows, cols), dtype=np.float64)
    dominant_idx_map = np.full((rows, cols), -1, dtype=np.int8)
    dominant_weight_map = np.zeros((rows, cols), dtype=np.float64)
    confidence_map = np.zeros((rows, cols), dtype=np.float64)


    for row in range(rows):
        for col in range(cols):
            mg_xy, og_xy = gaed._local_sobel_at(ge, row, col)
            dfmg_xy = gaed._dfmg(mg_xy)
            v_xy = gaed._local_mean_difference(ge, row, col)
            dg_xy = gaed._dg(v_xy)
            dt_xy = gaed._K_EDGE * dfmg_xy * dg_xy

            dfmg_map[row, col] = dfmg_xy
            dg_map[row, col] = dg_xy
            dt_map[row, col] = dt_xy

            threshold = max(0.0, min(1.0, gaed._T_FIXED + dt_xy))
            current = float(ge[row, col])
            h = 1 if current >= threshold else 0
            halftone[row, col] = h
            error = current - float(h)
            if error == 0.0:
                continue

            """
            Deviation from GAED_implementation.py (Line214), is_edge_xy = mg_xy > tmg,
            the implementation here uses a different edge detection mechanism, can cause
            inconsistencies
            """
            is_edge_xy = mg_xy > gaed._TMG
            neighbor_edge = False
            valid = [False, False, False, False]
            for idx in range(4):
                nr = row + int(gaed._FS_DR[idx])
                nc = col + int(gaed._FS_DC[idx])
                if 0 <= nr < rows and 0 <= nc < cols:
                    valid[idx] = True
                    mg_nb, _ = gaed._local_sobel_at(ge, nr, nc)
                    if gaed._dfmg(mg_nb) > 0.0:
                        neighbor_edge = True

            adaptive = is_edge_xy and neighbor_edge
            use_adaptive[row, col] = 1 if adaptive else 0
            weights = np.zeros(4, dtype=np.float64)

            if adaptive:
                ranks = np.full(4, -1.0, dtype=np.float64)
                for idx in range(4):
                    if valid[idx]:
                        nr = row + int(gaed._FS_DR[idx])
                        nc = col + int(gaed._FS_DC[idx])
                        mg_nb, og_nb = gaed._local_sobel_at(ge, nr, nc)
                        ranks[idx] = gaed._dfmg(mg_nb) * gaed._dfog(og_xy, og_nb)

                valid_ranks = [(idx, float(ranks[idx])) for idx in range(4) if valid[idx]]
                valid_ranks.sort(key=lambda item: item[1], reverse=True)
                if valid_ranks:
                    dominant_idx_map[row, col] = valid_ranks[0][0]
                    top_r_map[row, col] = valid_ranks[0][1]
                    second_r_map[row, col] = valid_ranks[1][1] if len(valid_ranks) > 1 else 0.0
                    gap_r_map[row, col] = top_r_map[row, col] - second_r_map[row, col]
                """
                Deviation from GAED_implementation.py logic, which uses adaptive confidence : 
                weights = (1.0 - confidence) * standard_weights + confidence * rank_weights, 
                here rank confidence is being used directly.
                """
                standard_weights = gaed._standard_weights_for_valid(valid)
                rank_weights = gaed._rank_weights_for_valid(valid, ranks)
                confidence = gaed._adaptive_confidence(valid, ranks)
                confidence_map[row, col] = confidence
                weights = (1.0 - confidence) * standard_weights + confidence * rank_weights
            else:
                for idx in range(4):
                    if valid[idx]:
                        weights[idx] = gaed._FS_W_STD[idx]

            if boundary_mode == "renormalize":
                weight_sum = weights.sum()
                if weight_sum > 0.0:
                    weights = weights / weight_sum
            elif boundary_mode != "strict":
                raise ValueError(f"Unsupported boundary mode: {boundary_mode}")

            if adaptive and dominant_idx_map[row, col] >= 0:
                dominant_weight_map[row, col] = weights[dominant_idx_map[row, col]]

            for idx in range(4):
                if valid[idx] and weights[idx] > 0.0:
                    nr = row + int(gaed._FS_DR[idx])
                    nc = col + int(gaed._FS_DC[idx])
                    ge[nr, nc] += weights[idx] * error

    return {
        "halftone": halftone,
        "use_adaptive": use_adaptive,
        "dfmg": dfmg_map,
        "dg": dg_map,
        "dt": dt_map,
        "top_r": top_r_map,
        "second_r": second_r_map,
        "gap_r": gap_r_map,
        "dominant_idx": dominant_idx_map,
        "dominant_weight": dominant_weight_map,
    }


def save_panel(output_path: Path, source: np.ndarray, analysis: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=180)
    axes[0, 0].imshow(source, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Source Crop")
    axes[0, 1].imshow(analysis["halftone"], cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("GAED Halftone")
    axes[0, 2].imshow(analysis["use_adaptive"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Adaptive Branch Used")

    idx_vis = analysis["dominant_idx"].astype(np.float64)
    idx_vis[idx_vis < 0] = np.nan
    axes[1, 0].imshow(idx_vis, cmap="tab10", vmin=0.0, vmax=3.0)
    axes[1, 0].set_title("Dominant Neighbor Index")
    axes[1, 1].imshow(analysis["top_r"], cmap="viridis")
    axes[1, 1].set_title("Top R")
    axes[1, 2].imshow(analysis["gap_r"], cmap="viridis")
    axes[1, 2].set_title("Top-Second R Gap")

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(args.image).convert("L"), dtype=np.uint8)
    g = gaed._to_gray_float(image)

    crop_top, crop_left = (args.top, args.left)
    if crop_top < 0 or crop_left < 0:
        crop_top, crop_left = auto_pick_crop(g, args.crop_size)

    analysis = run_analysis(g, boundary_mode=args.boundary_mode)

    crop_h = min(args.crop_size, g.shape[0] - crop_top)
    crop_w = min(args.crop_size, g.shape[1] - crop_left)
    sl = np.s_[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]
    crop_source = g[sl]
    crop_analysis = {key: value[sl] for key, value in analysis.items()}

    mask = crop_analysis["use_adaptive"] > 0
    dominant_counts: dict[str, int] = {}
    if np.any(mask):
        values, counts = np.unique(crop_analysis["dominant_idx"][mask], return_counts=True)
        dominant_counts = {DIRECTION_LABELS[int(v)]: int(c) for v, c in zip(values, counts)}

    summary = {
        "image": str(args.image),
        "boundary_mode": args.boundary_mode,
        "crop_top": int(crop_top),
        "crop_left": int(crop_left),
        "crop_height": int(crop_h),
        "crop_width": int(crop_w),
        "adaptive_fraction": float(mask.mean()),
        "adaptive_pixels": int(mask.sum()),
        "mean_top_r": float(crop_analysis["top_r"][mask].mean()) if np.any(mask) else 0.0,
        "mean_second_r": float(crop_analysis["second_r"][mask].mean()) if np.any(mask) else 0.0,
        "mean_gap_r": float(crop_analysis["gap_r"][mask].mean()) if np.any(mask) else 0.0,
        "median_gap_r": float(np.median(crop_analysis["gap_r"][mask])) if np.any(mask) else 0.0,
        "small_gap_fraction_lt_0.05": float((crop_analysis["gap_r"][mask] < 0.05).mean()) if np.any(mask) else 0.0,
        "dominant_direction_counts": dominant_counts,
    }

    panel_path = args.output_dir / "adaptive_branch_panel.png"
    save_panel(panel_path, crop_source, crop_analysis)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
