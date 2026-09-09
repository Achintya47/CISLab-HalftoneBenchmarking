from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter

"""
Extracted from the former `GAED_implementation.ipynb`.

This file preserves the notebook's current algorithmic behavior so it can be
reviewed, tested, and refactored as a normal Python module.
"""

""" 
    Paper/implementation provenance requires verification.
"""
_T_FIXED = 0.5
_K_EDGE = 2.5
_TMG = 0.1
_RMG = _TMG / 1.5
_TV = 0.6
_RV = _TV / 1.5
_ROG = math.pi / 3

""" 
    Derived from the assumed maximum Sobel response:
    sqrt(4^2 + 4^2). Verify that the Sobel kernel/normalization used by
    this implementation actually produces component-wise maxima of ±4.
"""
_SOBEL_MAG_MAX = math.sqrt(4.0**2 + 4.0**2)

"""
    Implementation-specific scale; source/justification requires verification.
"""
_ADAPTIVE_GAP_SCALE = 0.2

_FS_DR = np.array([0, 1, 1, 1], dtype=np.int32)
_FS_DC = np.array([1, -1, 0, 1], dtype=np.int32)
_FS_W_STD = np.array([7, 3, 5, 1], dtype=np.float64) / 16.0
_FS_W_RANK = np.array([7, 5, 3, 1], dtype=np.float64) / 16.0

_EC_W = np.array(
    [
        [0.1035, 0.1465, 0.1035],
        [0.1465, 0.0, 0.1465],
        [0.1035, 0.1465, 0.1035],
    ],
    dtype=np.float64,
)

_GX = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
_GY = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def _sample_with_zero_padding(image: np.ndarray, row: int, col: int) -> float:
    if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
        return float(image[row, col])
    return 0.0


def _local_sobel_at(image: np.ndarray, row: int, col: int) -> Tuple[float, float]:
    rx = 0.0
    ry = 0.0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            value = _sample_with_zero_padding(image, row + dr, col + dc)
            rx += value * float(_GX[dr + 1, dc + 1])
            ry += value * float(_GY[dr + 1, dc + 1])
    magnitude = math.sqrt(rx * rx + ry * ry) / _SOBEL_MAG_MAX
    orientation = math.atan2(rx, ry)
    return magnitude, orientation


def _local_mean_difference(image: np.ndarray, row: int, col: int) -> float:
    neighborhood_sum = 0.0
    neighborhood_count = 0
    for nr in range(max(row - 1, 0), min(row + 2, image.shape[0])):
        for nc in range(max(col - 1, 0), min(col + 2, image.shape[1])):
            if nr == row and nc == col:
                continue
            neighborhood_sum += float(image[nr, nc])
            neighborhood_count += 1
    if neighborhood_count == 0:
        return 0.0
    return neighborhood_sum / neighborhood_count - float(image[row, col])


def _wrap_orientation_difference(og_xy: float, og_nb: float) -> float:
    diff = abs(og_xy - og_nb)
    diff = math.fmod(diff, 2.0 * math.pi)

    """
        Dead Branch : diff will always be greater than 0.0 since we've
        applied abs()
    """

    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return min(diff, math.pi - diff)


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    elif img.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D image, got shape {img.shape}")

    img = img.astype(np.float64)
    if np.issubdtype(image.dtype, np.unsignedinteger):
        bits = np.iinfo(image.dtype).bits
        img = img / (2.0**bits - 1.0)
    elif np.issubdtype(image.dtype, np.integer):
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _sobel_gradient(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rx = convolve(g, _GX, mode="constant", cval=0.0)
    ry = convolve(g, _GY, mode="constant", cval=0.0)
    magnitude = np.sqrt(rx**2 + ry**2)
    magnitude_max = magnitude.max()
    if magnitude_max > 0:
        magnitude = magnitude / magnitude_max
    orientation = np.arctan2(rx, ry)
    return magnitude.astype(np.float64), orientation.astype(np.float64)


def _dfmg(mg: float) -> float:
    return _dfmg_with_threshold(mg, _TMG)


def _dfmg_with_threshold(mg: float, tmg: float) -> float:
    rmg = tmg / 1.5
    if mg > tmg:
        return 1.0
    num = math.exp(-(mg - tmg) ** 2 / (rmg**2)) - math.exp(-(tmg**2) / (rmg**2))
    den = 1.0 - math.exp(-(tmg**2) / (rmg**2))
    return num / den if den != 0.0 else 0.0


def _dg(v: float) -> float:
    if v > _TV:
        return 1.0
    if v < -_TV:
        return -1.0
    sign = 1.0 if v >= 0 else -1.0
    abs_v = abs(v)
    num = math.exp(-(abs_v - _TV) ** 2 / (_RV**2)) - math.exp(-(_TV**2) / (_RV**2))
    den = 1.0 - math.exp(-(_TV**2) / (_RV**2))
    val = num / den if den != 0.0 else 0.0
    return sign * val


def _dfog(og_xy: float, og_nb: float) -> float:
    diff = _wrap_orientation_difference(og_xy, og_nb)
    half_pi = math.pi / 2.0
    num = math.exp(-(diff**2) / (_ROG**2)) - math.exp(-(half_pi**2) / (_ROG**2))
    den = 1.0 - math.exp(-(half_pi**2) / (_ROG**2))
    val = num / den if den != 0.0 else 0.0
    return max(0.0, min(1.0, val))


def _standard_weights_for_valid(valid: list[bool]) -> np.ndarray:
    weights = np.zeros(4, dtype=np.float64)
    for idx in range(4):
        if valid[idx]:
            weights[idx] = _FS_W_STD[idx]
    return weights


def _rank_weights_for_valid(valid: list[bool], ranks: np.ndarray) -> np.ndarray:
    weights = np.zeros(4, dtype=np.float64)
    valid_indices = [idx for idx in range(4) if valid[idx]]
    sorted_indices = sorted(valid_indices, key=lambda idx: ranks[idx], reverse=True)
    for rank, idx in enumerate(sorted_indices):
        if rank < len(_FS_W_RANK):
            weights[idx] = _FS_W_RANK[rank]
    return weights


def _adaptive_confidence(valid: list[bool], ranks: np.ndarray) -> float:
    valid_scores = sorted((float(ranks[idx]) for idx in range(4) if valid[idx]), reverse=True)
    if not valid_scores:
        return 0.0
    top = valid_scores[0]
    second = valid_scores[1] if len(valid_scores) > 1 else 0.0
    gap = max(0.0, top - second)
    return max(0.0, min(1.0, gap / _ADAPTIVE_GAP_SCALE))


def _gaed_numpy(g: np.ndarray) -> np.ndarray:
    return _halftone_core(g, use_threshold_modulation=True, use_adaptive_filter=True, boundary_mode="renormalize", tmg=_TMG)


def _halftone_core(
    g: np.ndarray,
    *,
    use_threshold_modulation: bool,
    use_adaptive_filter: bool,
    boundary_mode: Literal["renormalize", "strict"],
    tmg: float,
) -> np.ndarray:
    rows, cols = g.shape
    ge = g.copy()
    halftone = np.zeros((rows, cols), dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            mg_xy, og_xy = _local_sobel_at(ge, row, col)
            dfmg_xy = _dfmg_with_threshold(mg_xy, tmg)
            v_xy = _local_mean_difference(ge, row, col)
            dg_xy = _dg(v_xy)
            dt_xy = _K_EDGE * dfmg_xy * dg_xy if use_threshold_modulation else 0.0
            tm_xy = max(0.0, min(1.0, _T_FIXED + dt_xy))

            current = float(ge[row, col])
            h = 1 if current >= tm_xy else 0
            halftone[row, col] = h
            error = current - float(h)
            if error == 0.0:
                continue

            """
                Wasted 4 x _local_sobel_at() calls even when use_adaptive_filter == False,
                Not critical but worth fixing.
            """
            valid = [False, False, False, False]
            for idx in range(4):
                nr = row + int(_FS_DR[idx])
                nc = col + int(_FS_DC[idx])
                if 0 <= nr < rows and 0 <= nc < cols:
                    valid[idx] = True

            use_adaptive = False
            if use_adaptive_filter:
                is_edge_xy = mg_xy > tmg
                neighbor_edge = False
                for idx in range(4):
                    if valid[idx]:
                        nr = row + int(_FS_DR[idx])
                        nc = col + int(_FS_DC[idx])
                        mg_nb, _ = _local_sobel_at(ge, nr, nc)
                        if mg_nb > tmg:
                            neighbor_edge = True
                use_adaptive = is_edge_xy and neighbor_edge

            weights = np.zeros(4, dtype=np.float64)

            if use_adaptive:
                ranks = np.full(4, -1.0)
                for idx in range(4):
                    if valid[idx]:
                        nr = row + int(_FS_DR[idx])
                        nc = col + int(_FS_DC[idx])
                        mg_nb, og_nb = _local_sobel_at(ge, nr, nc)
                        dfmg_nb = _dfmg_with_threshold(mg_nb, tmg)
                        dfog_nb = _dfog(og_xy, og_nb)
                        ranks[idx] = dfmg_nb * dfog_nb
                standard_weights = _standard_weights_for_valid(valid)
                rank_weights = _rank_weights_for_valid(valid, ranks)
                confidence = _adaptive_confidence(valid, ranks)
                weights = (1.0 - confidence) * standard_weights + confidence * rank_weights
            else:
                weights = _standard_weights_for_valid(valid)

            if boundary_mode == "renormalize":
                weight_sum = weights.sum()
                if weight_sum > 0.0:
                    weights = weights / weight_sum
            elif boundary_mode != "strict":
                raise ValueError(f"Unsupported boundary mode: {boundary_mode}")

            for idx in range(4):
                if valid[idx] and weights[idx] > 0.0:
                    nr = row + int(_FS_DR[idx])
                    nc = col + int(_FS_DC[idx])
                    ge[nr, nc] += weights[idx] * error

    return halftone


def _edge_correlation(g: np.ndarray, z: np.ndarray) -> float:
    rows, cols = g.shape
    ec_sum = 0.0
    offsets = [(k, l) for k in range(-1, 2) for l in range(-1, 2) if not (k == 0 and l == 0)]

    for k, l in offsets:
        w = _EC_W[k + 1, l + 1]
        for row in range(rows):
            src_row = row - k
            if not (0 <= src_row < rows):
                continue
            for col in range(cols):
                src_col = col - l
                if not (0 <= src_col < cols):
                    continue
                dg = float(g[row, col]) - float(g[src_row, src_col])
                dz = float(z[row, col]) - float(z[src_row, src_col])
                ec_sum += w * dg * dz

    return ec_sum / (rows * cols)


class GAED:
    def __init__(
        self,
        sigma_reconstruct: float = 1.0,
        tmg: float = _TMG,
        boundary_mode: Literal["renormalize", "strict"] = "renormalize",
    ) -> None:
        if boundary_mode not in {"renormalize", "strict"}:
            raise ValueError(f"Unsupported boundary mode: {boundary_mode}")
        if tmg <= 0:
            raise ValueError("tmg must be positive")
        self.sigma_reconstruct = float(sigma_reconstruct)
        self.boundary_mode = boundary_mode
        self.tmg = float(tmg)

    def halftone(self, image: np.ndarray) -> np.ndarray:
        return _halftone_core(
            _to_gray_float(image),
            use_threshold_modulation=True,
            use_adaptive_filter=True,
            boundary_mode=self.boundary_mode,
            tmg=self.tmg,
        )

    def halftone_variant(
        self,
        image: np.ndarray,
        variant: Literal["floyd", "gaed_threshold", "gaed_adaptive", "gaed"] = "gaed",
    ) -> np.ndarray:
        g = _to_gray_float(image)
        if variant == "floyd":
            return _halftone_core(
                g,
                use_threshold_modulation=False,
                use_adaptive_filter=False,
                boundary_mode=self.boundary_mode,
                tmg=self.tmg,
            )
        if variant == "gaed_threshold":
            return _halftone_core(
                g,
                use_threshold_modulation=True,
                use_adaptive_filter=False,
                boundary_mode=self.boundary_mode,
                tmg=self.tmg,
            )
        if variant == "gaed_adaptive":
            return _halftone_core(
                g,
                use_threshold_modulation=False,
                use_adaptive_filter=True,
                boundary_mode=self.boundary_mode,
                tmg=self.tmg,
            )
        if variant == "gaed":
            return _halftone_core(
                g,
                use_threshold_modulation=True,
                use_adaptive_filter=True,
                boundary_mode=self.boundary_mode,
                tmg=self.tmg,
            )
        raise ValueError(f"Unsupported GAED variant: {variant}")

    def evaluate(self, image: np.ndarray, halftone: Optional[np.ndarray] = None) -> dict:
        g = _to_gray_float(image)
        if halftone is None:
            halftone = self.halftone(image)
        reconstructed = gaussian_filter(halftone.astype(np.float64), sigma=self.sigma_reconstruct, truncate=3.0)
        mse = float(np.mean((g - reconstructed) ** 2))
        psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
        edge_corr = _edge_correlation(g, reconstructed)
        return {
            "halftone": halftone,
            "reconstructed": reconstructed,
            "mse": mse,
            "psnr": psnr,
            "edge_corr": edge_corr,
        }


def gaed_halftone(
    image: np.ndarray,
    sigma_reconstruct: float = 1.0,
    tmg: float = _TMG,
    evaluate: bool = False,
    variant: Literal["floyd", "gaed_threshold", "gaed_adaptive", "gaed"] = "gaed",
    boundary_mode: Literal["renormalize", "strict"] = "renormalize",
):
    model = GAED(sigma_reconstruct=sigma_reconstruct, tmg=tmg, boundary_mode=boundary_mode)
    halftone = model.halftone_variant(image, variant=variant)
    if evaluate:
        return halftone, model.evaluate(image, halftone)
    return halftone


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current GAED notebook extraction.")
    parser.add_argument("image", type=Path, nargs="?", help="Optional input image path.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--tmg", type=float, default=_TMG)
    parser.add_argument("--sigma-reconstruct", type=float, default=1.0)
    parser.add_argument(
        "--variant",
        choices=["floyd", "gaed_threshold", "gaed_adaptive", "gaed"],
        default="gaed",
    )
    parser.add_argument("--boundary-mode", choices=["renormalize", "strict"], default="renormalize")
    return parser.parse_args()


def _synthetic_test_image() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    xx, yy = np.meshgrid(x, y)
    image = np.sin(8 * np.pi * xx) * np.cos(8 * np.pi * yy) * 0.4 + 0.5
    return np.clip(image, 0, 1).astype(np.float64)


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image is None:
        image = (_synthetic_test_image() * 255).astype(np.uint8)
        input_stem = "synthetic"
    else:
        image = np.asarray(Image.open(args.image).convert("L"), dtype=np.uint8)
        input_stem = args.image.stem

    halftone, metrics = gaed_halftone(
        image,
        sigma_reconstruct=args.sigma_reconstruct,
        tmg=args.tmg,
        evaluate=True,
        variant=args.variant,
        boundary_mode=args.boundary_mode,
    )

    halftone_path = args.output_dir / f"{input_stem}_{args.variant}_halftone.png"
    recon_path = args.output_dir / f"{input_stem}_{args.variant}_reconstructed.png"
    Image.fromarray((halftone * 255).astype(np.uint8)).save(halftone_path)
    Image.fromarray((metrics["reconstructed"] * 255).astype(np.uint8)).save(recon_path)

    print(f"MSE: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"Edge corr: {metrics['edge_corr']:.4f}")
    print(f"Saved halftone -> {halftone_path}")
    print(f"Saved reconstructed -> {recon_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
