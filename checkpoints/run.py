from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .adapters import REPO_ROOT, MethodAdapter, build_adapter
from .metrics import color_metrics, luminance_metrics, normalize_rgb, rgb_to_luma, spectral_metrics
from .model import HalftoneResult, validate_result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if int(config.get("protocol_version", 0)) != 1:
        raise ValueError("Unsupported or missing protocol_version")
    if not isinstance(config.get("methods"), dict) or not config["methods"]:
        raise ValueError("Configuration must define at least one [methods.*] table")
    return config


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(config_path: Path, config: dict[str, Any], device: str) -> dict[str, Any]:
    import PIL
    import scipy
    import skimage
    """
    GAED, CB DBS, HCB DBS etc.. are non-torch methods, does this pipeline crashes
    for an import which is optional for most cases, a later subsequent crash is much suited.
    For a researched benchmarking on non-torch methods, installing torch should also be optional.
    """
    try :
        import torch

        torch_version : str | None = torch.__version__
    except ImportError :
        torch_version = None

    return {
        "protocol_version": config["protocol_version"],
        "config": str(config_path),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "device_request": device,
        "dependencies": {
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "scipy": scipy.__version__,
            "scikit-image": skimage.__version__,
            "torch": torch.__version__,
        },
    }


def _timed_result(adapter: MethodAdapter, image: np.ndarray, seed: int, warmup: int, repeats: int) -> HalftoneResult:
    if repeats <= 0 or warmup < 0:
        raise ValueError("timing repeats must be positive and warmup non-negative")
    for _ in range(warmup):
        adapter.run(image, seed)
    results = [adapter.run(image, seed) for _ in range(repeats)]
    runtime = statistics.median(result.runtime_sec for result in results)
    return replace(results[-1], runtime_sec=runtime)


def _save_result(output_dir: Path, stem: str, result: HalftoneResult) -> None:
    target = output_dir / "artifacts" / result.method / stem / f"seed-{result.seed}"
    target.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(result.luma_output, 0, 1) * 255).astype(np.uint8), mode="L").save(target / "luma.png")
    if result.rgb_output is not None:
        Image.fromarray((np.clip(result.rgb_output, 0, 1) * 255).astype(np.uint8), mode="RGB").save(target / "rgb.png")
    if result.preview_rgb is not None:
        Image.fromarray((np.clip(result.preview_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB").save(target / "preview.png")
    (target / "metadata.json").write_text(json.dumps(dict(result.metadata), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _mean_rows(rows: list[dict[str, Any]], kind: str, track: str | None = None) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["kind"] != kind or (track is not None and row.get("track") != track):
            continue
        method = str(row["method"])
        for key, value in row.items():
            if key in {"kind", "track", "method", "image", "seed", "gray_level"} or not isinstance(value, (int, float)):
                continue
            if np.isfinite(value):
                grouped.setdefault(method, {}).setdefault(key, []).append(float(value))
    return {method: {key: float(np.mean(values)) for key, values in metrics.items()} for method, metrics in grouped.items()}


def _leaderboard(summary: dict[str, Any]) -> str:
    lines = ["# Halftoning Benchmark", "", "## Primary luminance leaderboard", "", "| Method | HVS MSE | HVS PSNR | Viewed SSIM | Density error | Runtime (s) |", "|---|---:|---:|---:|---:|---:|"]
    for method, values in sorted(summary["kodak"].items(), key=lambda item: item[1].get("viewed_mse", float("inf"))):
        lines.append(f"| {method} | {values.get('viewed_mse', float('nan')):.6f} | {values.get('viewed_psnr', float('nan')):.3f} | {values.get('viewed_ssim', float('nan')):.5f} | {values.get('density_abs_error', float('nan')):.6f} | {values.get('runtime_sec', float('nan')):.4f} |")
    if summary.get("kodak_secondary"):
        lines.extend(["", "## Secondary implementation variants", "", "| Method | HVS MSE | HVS PSNR | Viewed SSIM | Density error | Runtime (s) |", "|---|---:|---:|---:|---:|---:|"])
        for method, values in sorted(summary["kodak_secondary"].items(), key=lambda item: item[1].get("viewed_mse", float("inf"))):
            lines.append(f"| {method} | {values.get('viewed_mse', float('nan')):.6f} | {values.get('viewed_psnr', float('nan')):.3f} | {values.get('viewed_ssim', float('nan')):.5f} | {values.get('density_abs_error', float('nan')):.6f} | {values.get('runtime_sec', float('nan')):.4f} |")
    lines.extend(["", "## Secondary color leaderboard", "", "| Method | RGB MSE | RGB PSNR | Delta E00 mean | Delta E00 p95 |", "|---|---:|---:|---:|---:|"])
    color_rows = [(method, values) for method, values in summary["kodak"].items() if "viewed_rgb_mse" in values]
    for method, values in sorted(color_rows, key=lambda item: item[1]["viewed_rgb_mse"]):
        lines.append(f"| {method} | {values['viewed_rgb_mse']:.6f} | {values['viewed_rgb_psnr']:.3f} | {values['delta_e_00_mean']:.4f} | {values['delta_e_00_p95']:.4f} |")
    lines.extend(["", "Runtime values are comparable only for rows produced on the same recorded hardware and device.", ""])
    return "\n".join(lines)


def run_benchmark(config_path: Path, *, method_names: list[str] | None = None, limit: int = 0, device: str = "auto", output_override: Path | None = None) -> Path:
    config = load_config(config_path)
    dataset_root = Path(config["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    image_paths = sorted(dataset_root.glob("kodim*.png"))
    if limit > 0:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No Kodak PNG images found under {dataset_root}")
    output_dir = output_override or Path(config.get("output_dir", "benchmarking/output/paper"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = method_names or [name for name, values in config["methods"].items() if values.get("enabled", True)]
    adapters = {name: build_adapter(name, dict(config["methods"][name]), device=device) for name in selected}
    timing = config.get("timing", {})
    warmup, repeats = int(timing.get("warmup", 1)), int(timing.get("repeats", 3))
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2, 3, 4])]
    rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        rgb = normalize_rgb(np.asarray(Image.open(image_path).convert("RGB")))
        reference_luma = rgb_to_luma(rgb)
        for name, adapter in adapters.items():
            method_seeds = seeds if adapter.stochastic else [seeds[0]]
            for seed in method_seeds:
                result = _timed_result(adapter, rgb, seed, warmup, repeats)
                validate_result(result, reference_luma.shape)
                track = str(config["methods"][name].get("track", "primary"))
                row: dict[str, Any] = {"kind": "kodak", "track": track, "method": name, "image": image_path.stem, "seed": seed, "runtime_sec": result.runtime_sec}
                row.update(luminance_metrics(reference_luma, result.luma_output))
                if result.rgb_output is not None:
                    row.update(color_metrics(rgb, result.rgb_output))
                rows.append(row)
                _save_result(output_dir, image_path.stem, result)

    gray_config = config.get("constant_gray", {})
    size = int(gray_config.get("size", 256))
    levels = [float(level) for level in gray_config.get("levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
    for level in levels:
        rgb = np.full((size, size, 3), level, dtype=np.float64)
        for name, adapter in adapters.items():
            method_seeds = seeds if adapter.stochastic else [seeds[0]]
            for seed in method_seeds:
                result = adapter.run(rgb, seed)
                validate_result(result, (size, size))
                track = str(config["methods"][name].get("track", "primary"))
                row = {"kind": "constant_gray", "track": track, "method": name, "image": "", "seed": seed, "gray_level": level, "runtime_sec": result.runtime_sec}
                row.update(spectral_metrics(result.luma_output, level))
                rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "kodak": _mean_rows(rows, "kodak", "primary"),
        "kodak_secondary": _mean_rows(rows, "kodak", "secondary"),
        "constant_gray": _mean_rows(rows, "constant_gray", "primary"),
        "constant_gray_secondary": _mean_rows(rows, "constant_gray", "secondary"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "leaderboard.md").write_text(_leaderboard(summary), encoding="utf-8")
    run_payload = provenance(config_path, config, device)
    run_payload.update({"images": [path.name for path in image_paths], "methods": selected, "seeds": seeds, "rows": len(rows)})
    (output_dir / "run.json").write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical cross-method halftoning benchmark.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "benchmarking/configs/paper.toml")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_benchmark(args.config, method_names=args.methods, limit=args.limit, device=args.device, output_override=args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
