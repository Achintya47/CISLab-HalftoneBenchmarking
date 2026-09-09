from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMPLEMENTATION_DIR = Path(__file__).resolve().parent
if str(IMPLEMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_DIR))

from generate_appendix_b import adaptive_error_diffusion, local_refine_dbs  # noqa: E402
from efficient_halftoning_drl import (  # noqa: E402
    DRLHalftoningConfig,
    build_reference_model,
    compute_reward_map,
    infer_halftone,
)


def image_to_u8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def u8_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_u8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze().detach().cpu().clamp(0.0, 1.0).numpy()
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def list_to_u8(image: list[list[float]]) -> np.ndarray:
    return np.clip(np.round(np.asarray(image, dtype=np.float32)), 0, 255).astype(np.uint8)


def floyd_steinberg(image: np.ndarray) -> np.ndarray:
    work = image.astype(np.float32).copy()
    height, width = work.shape
    out = np.zeros_like(work)
    for y in range(height):
        if y % 2 == 0:
            x_range = range(width)
            neighbors = ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16))
        else:
            x_range = range(width - 1, -1, -1)
            neighbors = ((-1, 0, 7 / 16), (1, 1, 3 / 16), (0, 1, 5 / 16), (-1, 1, 1 / 16))
        for x in x_range:
            old = work[y, x]
            new = 255.0 if old >= 128.0 else 0.0
            out[y, x] = new
            error = old - new
            for dx, dy, weight in neighbors:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    work[ny, nx] += error * weight
    return out.astype(np.uint8)


def run_drl(
    image_u8: np.ndarray,
    model: torch.nn.Module,
    config: DRLHalftoningConfig,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    contone = u8_to_tensor(image_u8, device)
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    noise = torch.randn(contone.shape, generator=generator, device=device, dtype=contone.dtype)
    result = infer_halftone(model, contone, config, noise=noise)
    halftone = tensor_to_u8(result["halftone"])
    probabilities = tensor_to_u8(result["probabilities"])
    return halftone, probabilities


def evaluate_halftone(halftone_u8: np.ndarray, contone_u8: np.ndarray, device: torch.device, config: DRLHalftoningConfig) -> Dict[str, float]:
    halftone = u8_to_tensor(halftone_u8, device)
    contone = u8_to_tensor(contone_u8, device)
    reward_map, metrics = compute_reward_map(halftone, contone, config)
    density_error = float(abs(halftone.float().mean().item() - contone.float().mean().item()))
    binary_ratio = float(halftone.float().mean().item())
    return {
        "reward": float(metrics["reward"].item()),
        "tone_error": float(metrics["tone_error"].item()),
        "cssim": float(metrics["cssim"].item()),
        "density_abs_error": density_error,
        "white_ratio": binary_ratio,
        "reward_sum": float(reward_map.sum().item()),
    }


def power_spectrum_image(halftone_u8: np.ndarray) -> Image.Image:
    centered = halftone_u8.astype(np.float32) / 255.0 - np.mean(halftone_u8.astype(np.float32) / 255.0)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    spectrum = np.log1p(spectrum)
    spectrum -= spectrum.min()
    max_value = spectrum.max()
    if max_value > 0:
        spectrum /= max_value
    return Image.fromarray(np.clip(np.round(spectrum * 255.0), 0, 255).astype(np.uint8), mode="L")


def radial_profile(halftone_u8: np.ndarray) -> list[float]:
    centered = halftone_u8.astype(np.float32) / 255.0 - np.mean(halftone_u8.astype(np.float32) / 255.0)
    power = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    height, width = power.shape
    yy, xx = np.indices((height, width))
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    rr = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
    max_r = int(rr.max())
    profile: list[float] = []
    for radius in range(max_r + 1):
        mask = rr == radius
        if not np.any(mask):
            profile.append(0.0)
            continue
        profile.append(float(power[mask].mean()))
    total = sum(profile)
    if total > 0:
        profile = [value / total for value in profile]
    return profile


def anisotropy_score(halftone_u8: np.ndarray) -> float:
    centered = halftone_u8.astype(np.float32) / 255.0 - np.mean(halftone_u8.astype(np.float32) / 255.0)
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
    max_r = int(rr.max())
    scores: list[float] = []
    for radius in range(1, max_r + 1):
        mask = rr == radius
        if mask.sum() <= 1:
            continue
        ring = power[mask]
        mean = ring.mean()
        if mean <= 1e-12:
            continue
        scores.append(float(np.mean((ring - mean) ** 2)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def make_zoom_box(width: int, height: int, fraction: float = 0.18) -> tuple[int, int, int, int]:
    crop_size = max(64, int(min(width, height) * fraction))
    x = max(0, (width - crop_size) // 2)
    y = max(0, (height - crop_size) // 2)
    return x, y, crop_size, crop_size


def crop_and_zoom(image_u8: np.ndarray, box: tuple[int, int, int, int], scale: int = 4) -> Image.Image:
    x, y, w, h = box
    crop = Image.fromarray(image_u8, mode="L").crop((x, y, x + w, y + h))
    return crop.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)


def add_box(image_u8: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    image = Image.fromarray(image_u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = box
    draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(220, 40, 40), width=3)
    return image.convert("L")


def stack_labeled_row(images: Iterable[tuple[str, Image.Image]], label_width: int = 120, bg: int = 255) -> Image.Image:
    rows = list(images)
    widths = [label_width + image.size[0] for _label, image in rows]
    heights = [max(36, image.size[1]) for _label, image in rows]
    canvas = Image.new("L", (max(widths), sum(heights)), color=bg)
    y = 0
    for (label, image), height in zip(rows, heights):
        label_img = Image.new("L", (label_width, height), color=bg)
        draw = ImageDraw.Draw(label_img)
        draw.text((8, 8), label, fill=0)
        canvas.paste(label_img, (0, y))
        canvas.paste(image, (label_width, y))
        y += height
    return canvas


def hstack(images: list[Image.Image], pad: int = 8, bg: int = 255) -> Image.Image:
    width = sum(image.size[0] for image in images) + pad * (len(images) - 1)
    height = max(image.size[1] for image in images)
    canvas = Image.new("L", (width, height), color=bg)
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.size[1]) // 2))
        x += image.size[0] + pad
    return canvas


def save_comparison_artifacts(
    image_name: str,
    source_u8: np.ndarray,
    method_outputs: dict[str, np.ndarray],
    method_profiles: dict[str, list[float]],
    output_dir: Path,
) -> None:
    height, width = source_u8.shape
    box = make_zoom_box(width, height)
    zoom_images: list[tuple[str, Image.Image]] = [("source", add_box(source_u8, box))]
    zoom_crops: list[tuple[str, Image.Image]] = [("source-zoom", crop_and_zoom(source_u8, box))]
    spectra: list[tuple[str, Image.Image]] = []

    for method_name, halftone_u8 in method_outputs.items():
        zoom_images.append((method_name, add_box(halftone_u8, box)))
        zoom_crops.append((f"{method_name}-zoom", crop_and_zoom(halftone_u8, box)))
        spectra.append((method_name, power_spectrum_image(halftone_u8)))

    output_dir.mkdir(parents=True, exist_ok=True)
    stack_labeled_row(zoom_images).save(output_dir / f"{image_name}-overview.png")
    stack_labeled_row(zoom_crops).save(output_dir / f"{image_name}-zoom.png")
    stack_labeled_row(spectra).save(output_dir / f"{image_name}-fft.png")
    (output_dir / f"{image_name}-radial-profiles.json").write_text(
        json.dumps(method_profiles, indent=2),
        encoding="utf-8",
    )


def summarize_rows(rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    metrics = ["runtime_sec", "reward", "tone_error", "cssim", "density_abs_error", "white_ratio", "anisotropy"]
    methods = sorted({str(row["method"]) for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_summary: dict[str, float] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows]
            method_summary[f"{metric}_mean"] = float(np.mean(values))
            method_summary[f"{metric}_std"] = float(np.std(values))
        summary[method] = method_summary
    return summary


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, DRLHalftoningConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DRLHalftoningConfig(**checkpoint["config"])
    model = build_reference_model(config, device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DRL halftoning on Kodak against local baselines.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/output/full-bsd-kodak-safe/checkpoints/latest.pt"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/kodak/images"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmarking/output/efficient-halftoning-via-deep-reinforcement-learning/full-bsd-kodak-safe"
        ),
    )
    parser.add_argument(
        "--diagnostic-images",
        nargs="*",
        default=["kodim01", "kodim07", "kodim12", "kodim15", "kodim18"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model, config = load_model(args.checkpoint, device)

    image_paths = sorted(path for path in args.dataset_root.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Kodak PNGs found under {args.dataset_root}")

    method_order = ["drl", "floyd_steinberg", "adaptive_ed", "dbs_style"]
    rows: list[dict[str, float | str]] = []
    per_image_outputs: dict[str, dict[str, np.ndarray]] = {}
    per_image_profiles: dict[str, dict[str, list[float]]] = {}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    method_output_dir = output_dir / "method_outputs"
    diagnostics_dir = output_dir / "diagnostics"
    method_output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        image_name = image_path.stem
        print(f"[benchmark] {index + 1}/{len(image_paths)} {image_name}", flush=True)
        with Image.open(image_path) as image:
            source_u8 = image_to_u8(image)
        source_list = source_u8.astype(np.float32).tolist()

        drl_start = time.perf_counter()
        drl_halftone, drl_probabilities = run_drl(source_u8, model, config, device, seed=args.seed + index)
        drl_runtime = time.perf_counter() - drl_start

        baseline_start = time.perf_counter()
        floyd_halftone = floyd_steinberg(source_u8)
        floyd_runtime = time.perf_counter() - baseline_start

        baseline_start = time.perf_counter()
        adaptive_halftone = list_to_u8(adaptive_error_diffusion(source_list))
        adaptive_runtime = time.perf_counter() - baseline_start

        baseline_start = time.perf_counter()
        dbs_halftone, _dbs_recon, _snapshots = local_refine_dbs(source_list, adaptive_halftone.astype(np.float32).tolist())
        dbs_runtime = time.perf_counter() - baseline_start
        dbs_halftone_u8 = list_to_u8(dbs_halftone)

        outputs = {
            "drl": drl_halftone,
            "floyd_steinberg": floyd_halftone,
            "adaptive_ed": adaptive_halftone,
            "dbs_style": dbs_halftone_u8,
        }
        profiles = {method_name: radial_profile(halftone_u8) for method_name, halftone_u8 in outputs.items()}

        per_image_outputs[image_name] = outputs
        per_image_profiles[image_name] = profiles

        probabilities_path = method_output_dir / "drl" / f"{image_name}-probability.png"
        probabilities_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(drl_probabilities, mode="L").save(probabilities_path)

        runtimes = {
            "drl": drl_runtime,
            "floyd_steinberg": floyd_runtime,
            "adaptive_ed": adaptive_runtime,
            "dbs_style": dbs_runtime,
        }

        for method_name in method_order:
            halftone_u8 = outputs[method_name]
            metrics = evaluate_halftone(halftone_u8, source_u8, device, config)
            method_dir = method_output_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(halftone_u8, mode="L").save(method_dir / f"{image_name}.png")
            row = {
                "image": image_name,
                "method": method_name,
                "runtime_sec": runtimes[method_name],
                "anisotropy": anisotropy_score(halftone_u8),
                **metrics,
            }
            rows.append(row)

    for image_name in args.diagnostic_images:
        if image_name not in per_image_outputs:
            continue
        source_path = args.dataset_root / f"{image_name}.png"
        with Image.open(source_path) as image:
            source_u8 = image_to_u8(image)
        save_comparison_artifacts(
            image_name=image_name,
            source_u8=source_u8,
            method_outputs=per_image_outputs[image_name],
            method_profiles=per_image_profiles[image_name],
            output_dir=diagnostics_dir,
        )

    summary = summarize_rows(rows)
    csv_path = output_dir / "kodak_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranking_lines = []
    ranking_lines.append("# Kodak Benchmark Summary")
    ranking_lines.append("")
    ranking_lines.append(f"Checkpoint: `{args.checkpoint}`")
    ranking_lines.append(f"Dataset: `{args.dataset_root}` ({len(image_paths)} images)")
    ranking_lines.append("")
    ranking_lines.append("## Mean Metrics")
    ranking_lines.append("")
    ranking_lines.append("| Method | Reward | Tone Error | CSSIM | Density Abs Err | Anisotropy | Runtime (s) |")
    ranking_lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in method_order:
        stats = summary[method]
        ranking_lines.append(
            f"| {method} | {stats['reward_mean']:.6f} | {stats['tone_error_mean']:.6f} | {stats['cssim_mean']:.6f} | {stats['density_abs_error_mean']:.6f} | {stats['anisotropy_mean']:.6e} | {stats['runtime_sec_mean']:.4f} |"
        )
    ranking_lines.append("")
    ranking_lines.append("## Notes")
    ranking_lines.append("")
    ranking_lines.append("- `drl` uses a single noisy forward pass with fixed seed offset per Kodak image, then 0.5 thresholding.")
    ranking_lines.append("- `floyd_steinberg`, `adaptive_ed`, and `dbs_style` are local baselines available in this repository.")
    ranking_lines.append("- `dbs_style` is the compact local refinement from `generate_appendix_b.py`, not a claim of exact canonical DBS reproduction.")
    (output_dir / "README.md").write_text("\n".join(ranking_lines) + "\n", encoding="utf-8")

    print(f"Wrote benchmark artifacts to {output_dir}")


if __name__ == "__main__":
    main()
