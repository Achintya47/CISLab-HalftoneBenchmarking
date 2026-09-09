from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from .algorithms import build_algorithm
from .datasets import DatasetImage, build_manifest, load_image_rgb, save_manifest
from .metrics import color_full_reference_metrics, grayscale_full_reference_metrics, lpips_unavailable_reason
from .reconstruction import RECONSTRUCTION_SIGMA, reconstruct
from .spec import CONTENT_FAMILIES, DEFAULT_IMAGES_PER_FAMILY
from .stress_variants import apply_family_variants
from .visualize import build_cross_method_panel, save_cross_method_panel, save_method_panel, select_sample_item_indices

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


@dataclass(frozen=True)
class EvalItem:
    family: str
    dataset_key: str
    image_id: str
    variant: str  # "base" or one of the 3 stress-variant keys
    synthetic: bool
    rgb: np.ndarray


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def apply_overrides(
    config: Dict[str, Any],
    *,
    images_per_family: int | None = None,
    only_methods: Sequence[str] | None = None,
    disable_methods: Sequence[str] | None = None,
    seed: int | None = None,
    no_lpips: bool = False,
) -> Dict[str, Any]:
    """Mutates and returns `config` in place, applying CLI-level overrides
    on top of whatever a TOML config already specifies. Kept as a pure
    dict-transform (rather than argparse-specific) so it's usable both from
    the CLI and from Python/tests directly."""
    if images_per_family is not None:
        families = config.setdefault("families", {})
        for name in CONTENT_FAMILIES:
            families.setdefault(name, {})["count"] = images_per_family

    methods = config.setdefault("methods", {})
    if only_methods:
        only = set(only_methods)
        for name, algo_cfg in methods.items():
            algo_cfg["enabled"] = name in only
    if disable_methods:
        disabled = set(disable_methods)
        for name in disabled:
            if name not in methods:
                raise ValueError(f"--disable references unknown method '{name}'. Known methods: {sorted(methods)}")
            methods[name]["enabled"] = False

    if seed is not None:
        config["seed"] = seed
    if no_lpips:
        config["compute_lpips"] = False
    return config


def build_eval_items(config: Dict[str, Any], local_datasets_root: Path):
    families_cfg = config.get("families", {})
    counts = {name: int(families_cfg.get(name, {}).get("count", DEFAULT_IMAGES_PER_FAMILY)) for name in CONTENT_FAMILIES}
    seed = int(config.get("seed", 0))
    synthetic_size = int(config.get("synthetic_size", 128))
    working_size = int(config.get("working_size", synthetic_size))

    manifest = build_manifest(counts, local_datasets_root=local_datasets_root, seed=seed, synthetic_size=synthetic_size)
    items: List[EvalItem] = []
    for entry in manifest:
        rgb = load_image_rgb(entry, synthetic_size=synthetic_size, seed=seed)
        rgb = _resize(rgb, working_size)
        items.append(EvalItem(entry.family, entry.dataset_key, entry.image_id, "base", entry.synthetic, rgb))
        for variant_name, perturbed in apply_family_variants(entry.family, rgb, seed=seed):
            items.append(EvalItem(entry.family, entry.dataset_key, entry.image_id, variant_name, entry.synthetic, perturbed))
    return items, manifest


def _resize(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    image = Image.fromarray(np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8))
    resized = image.resize((size, size), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def evaluate_item(algorithm: Any, item: EvalItem, seed: int, compute_lpips: bool, capture: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = algorithm.run(item.rgb, seed)
    reference_gray = np.tensordot(item.rgb, np.array([0.299, 0.587, 0.114]), axes=([-1], [0]))
    gray_metrics = grayscale_full_reference_metrics(result.luma_output, reference_gray, compute_lpips=compute_lpips)

    if capture is not None:
        # Grayscale/luma space, uniformly across methods (see visualize.py) --
        # this is exactly what gray_metrics above was just computed from, so
        # the comparison panel always matches the reported numbers.
        capture.update(
            family=item.family,
            variant=item.variant,
            image_id=item.image_id,
            reference=reference_gray,
            halftone=result.luma_output,
            reconstructed=reconstruct(result.luma_output),
        )

    row: Dict[str, Any] = {
        "family": item.family,
        "dataset_key": item.dataset_key,
        "image_id": item.image_id,
        "variant": item.variant,
        "synthetic": item.synthetic,
        "method": algorithm.name,
        "method_family": getattr(algorithm, "family", "unknown"),
        "seed": seed,
        "runtime_sec": result.runtime_sec,
        "track": "grayscale",
        **{f"gray_{k}": v for k, v in gray_metrics.items()},
    }

    if getattr(algorithm, "supports_color", False) and result.rgb_output is not None:
        color_metrics = color_full_reference_metrics(result.rgb_output, item.rgb, compute_lpips=compute_lpips)
        row.update({f"color_{k}": v for k, v in color_metrics.items()})
        row["track"] = "grayscale+color"

    return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    summary: Dict[str, Any] = {}
    for method, method_rows in by_method.items():
        numeric_keys = sorted({k for row in method_rows for k, v in row.items() if isinstance(v, (int, float)) and k not in {"seed"}})
        means: Dict[str, float] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in method_rows if key in row and np.isfinite(row[key])]
            if values:
                means[key] = float(np.mean(values))
        summary[method] = means

        by_variant: Dict[str, Dict[str, float]] = {}
        for variant in sorted({row["variant"] for row in method_rows}):
            variant_rows = [row for row in method_rows if row["variant"] == variant]
            by_variant[variant] = {
                key: float(np.mean([float(row[key]) for row in variant_rows if key in row and np.isfinite(row[key])]))
                for key in ("gray_psnr", "gray_ssim", "gray_anisotropy_index")
                if any(key in row for row in variant_rows)
            }
        summary[method]["_by_variant"] = by_variant
    return summary


def leaderboard_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Halftoning Benchmark v2 — Leaderboard", "", f"Reconstruction operator: Gaussian blur, sigma={RECONSTRUCTION_SIGMA} (spec Sec. 5.2)", ""]
    lines += ["| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Aniso ↓ | ΔE00 ↓ | Runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, values in sorted(summary.items(), key=lambda kv: kv[1].get("gray_psnr", 0.0), reverse=True):
        lpips_value = values.get("gray_lpips", float("nan"))
        delta_e = values.get("color_delta_e00", float("nan"))
        lines.append(
            f"| {method} | {values.get('gray_psnr', float('nan')):.3f} | {values.get('gray_ssim', float('nan')):.4f} | "
            f"{lpips_value:.4f} | {values.get('gray_anisotropy_index', float('nan')):.6e} | {delta_e:.4f} | {values.get('runtime_sec', float('nan')):.4f} |"
        )
    reason = lpips_unavailable_reason()
    if reason:
        lines += ["", f"_LPIPS reported as NaN: optional `lpips` package unavailable ({reason})._"]
    return "\n".join(lines)


def run(
    config_path: Path,
    *,
    output_override: Path | None = None,
    images_per_family: int | None = None,
    only_methods: Sequence[str] | None = None,
    disable_methods: Sequence[str] | None = None,
    seed_override: int | None = None,
    no_lpips: bool = False,
    quiet: bool = False,
    comparisons_per_method: int | None = None,
) -> Path:
    started = time.perf_counter()
    config = load_config(config_path)
    apply_overrides(
        config,
        images_per_family=images_per_family,
        only_methods=only_methods,
        disable_methods=disable_methods,
        seed=seed_override,
        no_lpips=no_lpips,
    )

    output_dir = output_override or (REPO_ROOT / config.get("output_dir", "benchmarking_v2/output/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    local_datasets_root = REPO_ROOT / config.get("datasets_root", "datasets")

    _log(f"[setup] config={config_path}", quiet=quiet)
    _log(f"[setup] output_dir={output_dir}", quiet=quiet)

    items, manifest = build_eval_items(config, local_datasets_root)
    save_manifest(manifest, output_dir / "manifest.json")
    n_synthetic = sum(1 for item in items if item.synthetic)
    _log(
        f"[dataset] {len(items)} evaluation items "
        f"({len(manifest)} base images x (1 + 3 stress variants){' -- ' + str(n_synthetic) + ' synthetic-fallback items' if n_synthetic else ''})",
        quiet=quiet,
    )

    seed = int(config.get("seed", 0))
    device = config.get("device", "auto")
    compute_lpips = bool(config.get("compute_lpips", True))
    save_comparisons = bool(config.get("save_comparisons", True)) and comparisons_per_method != 0
    n_comparisons = comparisons_per_method if comparisons_per_method is not None else int(config.get("comparisons_per_method", 5))

    method_configs = {name: cfg for name, cfg in config.get("methods", {}).items() if cfg.get("enabled", True)}
    if not method_configs:
        raise ValueError("No methods enabled -- check [methods.*].enabled / --only / --disable.")

    algorithms = []
    for name, algo_cfg in method_configs.items():
        build_start = time.perf_counter()
        _log(f"[build] {name} ...", quiet=quiet)
        algorithm = build_algorithm(algo_cfg.get("algorithm", name), algo_cfg, device=device)
        algorithm.name = algo_cfg.get("display_name", name)
        algorithms.append(algorithm)
        _log(f"[build] {name} ready ({time.perf_counter() - build_start:.1f}s)", quiet=quiet)

    total_runs = len(items) * len(algorithms)
    _log(f"[run] {len(algorithms)} method(s) x {len(items)} item(s) = {total_runs} evaluations", quiet=quiet)

    sample_indices: List[int] = []
    if save_comparisons and n_comparisons > 0:
        sample_indices = select_sample_item_indices([item.family for item in items], [item.variant for item in items], n_comparisons)
        _log(
            f"[compare] will save original/halftone/reconstructed panels for {len(sample_indices)} sample item(s) per method "
            f"(families: {sorted({items[i].family for i in sample_indices})})",
            quiet=quiet,
        )
    cross_method_index = sample_indices[0] if sample_indices else None
    cross_method_halftones: Dict[str, np.ndarray] = {}
    cross_method_reconstructions: Dict[str, np.ndarray] = {}
    cross_method_reference: np.ndarray | None = None
    cross_method_label = ""

    rows: List[Dict[str, Any]] = []
    completed = 0
    for algorithm in algorithms:
        method_start = time.perf_counter()
        method_samples: List[Dict[str, Any]] = []
        sample_index_set = set(sample_indices)
        for item_index, item in enumerate(items):
            capture = {} if item_index in sample_index_set else None
            row = evaluate_item(algorithm, item, seed, compute_lpips, capture=capture)
            rows.append(row)
            completed += 1
            _log(
                f"[{completed}/{total_runs}] {algorithm.name:<18} {item.family:<17} {item.variant:<20} "
                f"psnr={row.get('gray_psnr', float('nan')):6.2f}  ({row['runtime_sec']:.3f}s)",
                quiet=quiet,
            )
            if capture:
                method_samples.append(capture)
                if item_index == cross_method_index:
                    cross_method_halftones[algorithm.name] = capture["halftone"]
                    cross_method_reconstructions[algorithm.name] = capture["reconstructed"]
                    cross_method_reference = capture["reference"]
                    cross_method_label = f"{capture['family']} / {capture['variant']} / {capture['image_id']}"
        _log(f"[run] {algorithm.name} done in {time.perf_counter() - method_start:.1f}s", quiet=quiet)

        if save_comparisons and method_samples:
            panel_path = save_method_panel(algorithm.name, method_samples, output_dir / "comparisons")
            _log(f"[compare] wrote {panel_path}", quiet=quiet)

    if save_comparisons and cross_method_reference is not None and len(cross_method_halftones) > 1:
        cross_path = save_cross_method_panel(
            cross_method_reference,
            cross_method_halftones,
            cross_method_reconstructions,
            label=cross_method_label,
            output_dir=output_dir / "comparisons",
        )
        _log(f"[compare] wrote {cross_path}", quiet=quiet)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "leaderboard.md").write_text(leaderboard_markdown(summary), encoding="utf-8")

    provenance = {
        "config": str(config_path),
        "git_commit": _git("rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "reconstruction_sigma": RECONSTRUCTION_SIGMA,
        "n_items": len(items),
        "n_algorithms": len(algorithms),
        "methods": [algorithm.name for algorithm in algorithms],
        "lpips_unavailable_reason": lpips_unavailable_reason(),
        "any_synthetic_images": any(item.synthetic for item in items),
        "elapsed_sec": time.perf_counter() - started,
        "comparisons_per_method_requested": n_comparisons if save_comparisons else 0,
        "comparisons_per_method_actual": len(sample_indices),
        "comparisons_dir": str(output_dir / "comparisons") if save_comparisons and sample_indices else None,
    }
    (output_dir / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _log(f"[done] wrote results to {output_dir} in {provenance['elapsed_sec']:.1f}s", quiet=quiet)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the generalized halftoning benchmark (spec v2).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "benchmarking_v2/config/default.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--images-per-family",
        type=int,
        default=None,
        help="Override every [families.*].count in the config, e.g. --images-per-family 2 "
        "for a ~10-base-image (2 x 5 families) smoke run instead of the default 50.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Run only these method keys from [methods.*] (e.g. --only dbs error_diffusion ordered_dithering).",
    )
    parser.add_argument(
        "--disable",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Disable these method keys, e.g. --disable deep_learning (useful if you have no DRL checkpoint "
        "and don't want to pay even the short bootstrap-training cost).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    parser.add_argument("--no-lpips", action="store_true", help="Force-disable LPIPS even if the config enables it.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging (only the final path is printed).")
    parser.add_argument(
        "--comparisons-per-method",
        type=int,
        default=None,
        metavar="N",
        help="Save an original/halftone/reconstructed comparison panel (PNG) with up to N sample images per "
        "method, spread across content families (default: 5, from config's `comparisons_per_method`). "
        "Use 0 to disable panel generation entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(
        args.config,
        output_override=args.output_dir,
        images_per_family=args.images_per_family,
        only_methods=args.only,
        disable_methods=args.disable,
        seed_override=args.seed,
        no_lpips=args.no_lpips,
        quiet=args.quiet,
        comparisons_per_method=args.comparisons_per_method,
    )
    print(output)


if __name__ == "__main__":
    main()
