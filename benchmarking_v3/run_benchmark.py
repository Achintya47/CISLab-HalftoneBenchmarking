"""Printer-style CMYK/ICC benchmark orchestrator.

Pipeline (numbered per the spec this module implements):

 1. [SPEC] image condition variants (base + this content family's stress
    variants) -- reuses `benchmarking_v2.datasets` + `.stress_variants`
    completely unchanged; the *original RGB item is kept immutable* and
    used as the reference for every downstream metric.
 2. [NEW]  RGB -> CMYK through a fixed, benchmark-wide `icc.ICCPipeline`.
 3. [NEW]  CMYK -> {C, M, Y, K} (`cmyk_pipeline.separate_channels`).
 4. [NEW]  each plane IS its grayscale representation already (no-op,
    documented in `cmyk_pipeline.py`).
 5. [SPEC] halftoning: each plane run independently through the selected
    method's per-plane adapter (`algorithms.py`).
 6. [SPEC] binary -> continuous per plane, Gaussian sigma=1.2
    (`reconstruction.py`, reused verbatim from benchmarking_v2).
 7. [NEW]  {C, M, Y, K} reconstructed planes -> CMYK
    (`cmyk_pipeline.combine_channels`).
 8. [NEW]  CMYK -> RGB through the SAME fixed `ICCPipeline` instance (so the
    inverse leg always matches whatever produced the forward leg).
 9. [SPEC] RGB -> CIELAB (inside `metrics.py`, via skimage).
10. [SPEC] PSNR / SSIM / LPIPS / Anisotropy / Delta E00.
11. [SPEC] final report: metrics.csv / summary.json / leaderboard.md /
    run.json / comparisons/*.png -- same shape as benchmarking_v2's.
"""

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

from benchmarking_v2.datasets import build_manifest, load_image_rgb, save_manifest
from benchmarking_v2.metrics import lpips_unavailable_reason
from benchmarking_v2.spec import CONTENT_FAMILIES, DEFAULT_IMAGES_PER_FAMILY

from .algorithms import build_plane_algorithm
from .cmyk_pipeline import combine_channels, separate_channels
from .icc import DEFAULT_CMYK_PROFILE_PATH, ICCPipeline, ICCProfileSpec
from .metrics import multichannel_anisotropy_index, rgb_reconstruction_metrics
from .reconstruction import RECONSTRUCTION_SIGMA, reconstruct
from .visualize import save_method_panel, select_sample_item_indices
from benchmarking_v2.stress_variants import apply_family_variants

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class EvalItem:
    family: str
    dataset_key: str
    image_id: str
    variant: str
    synthetic: bool
    rgb: np.ndarray


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
    cmyk_profile_path: str | None = None,
    rendering_intent: str | None = None,
    no_bpc: bool = False,
) -> Dict[str, Any]:
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
        for name in disable_methods:
            if name not in methods:
                raise ValueError(f"--disable references unknown method '{name}'. Known methods: {sorted(methods)}")
            methods[name]["enabled"] = False

    if seed is not None:
        config["seed"] = seed
    if no_lpips:
        config["compute_lpips"] = False

    icc_cfg = config.setdefault("icc", {})
    if cmyk_profile_path is not None:
        icc_cfg["cmyk_profile_path"] = cmyk_profile_path
    if rendering_intent is not None:
        icc_cfg["rendering_intent"] = rendering_intent
    if no_bpc:
        icc_cfg["black_point_compensation"] = False
    return config


def build_icc_pipeline(config: Dict[str, Any]) -> ICCPipeline:
    icc_cfg = config.get("icc", {})
    profile_path = Path(icc_cfg.get("cmyk_profile_path", DEFAULT_CMYK_PROFILE_PATH))
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    spec = ICCProfileSpec(
        cmyk_profile_path=profile_path,
        rendering_intent=icc_cfg.get("rendering_intent", "relative_colorimetric"),
        black_point_compensation=bool(icc_cfg.get("black_point_compensation", True)),
        source_profile=icc_cfg.get("source_profile", "sRGB"),
    )
    return ICCPipeline(spec)


def _resize(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    image = Image.fromarray(np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8))
    resized = image.resize((size, size), Image.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


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


def evaluate_item(
    algorithm_name: str,
    algorithm: Any,
    item: EvalItem,
    icc_pipeline: ICCPipeline,
    seed: int,
    compute_lpips: bool,
    capture: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    # Steps 2-3: original RGB (immutable reference) -> ICC CMYK -> 4 planes.
    cmyk = icc_pipeline.rgb_to_cmyk(item.rgb)
    planes = separate_channels(cmyk)  # step 4 is a no-op: planes ARE grayscale already

    # Step 5-6: halftone + reconstruct each plane independently.
    halftone_planes: Dict[str, np.ndarray] = {}
    reconstructed_planes: Dict[str, np.ndarray] = {}
    channel_runtime = 0.0
    for name, plane in planes.items():
        start = time.perf_counter()
        halftone = algorithm.run_plane(plane, seed)
        channel_runtime += time.perf_counter() - start
        halftone_planes[name] = halftone
        reconstructed_planes[name] = reconstruct(halftone)

    # Step 7-8: recombine -> CMYK -> RGB through the SAME ICC pipeline.
    reconstructed_cmyk = combine_channels(reconstructed_planes)
    reconstructed_rgb = icc_pipeline.cmyk_to_rgb(reconstructed_cmyk)

    # Step 9-10: RGB/Lab metrics against the untouched original RGB.
    metrics = rgb_reconstruction_metrics(reconstructed_rgb, item.rgb, compute_lpips=compute_lpips)
    anisotropy = multichannel_anisotropy_index(halftone_planes)

    if capture is not None:
        capture.update(
            family=item.family,
            variant=item.variant,
            image_id=item.image_id,
            reference_rgb=item.rgb,
            reconstructed_rgb=reconstructed_rgb,
            halftone_planes=halftone_planes,
        )

    return {
        "family": item.family,
        "dataset_key": item.dataset_key,
        "image_id": item.image_id,
        "variant": item.variant,
        "synthetic": item.synthetic,
        "method": algorithm_name,
        "seed": seed,
        "runtime_sec": channel_runtime,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "delta_e00": metrics["delta_e00"],
        "anisotropy_index": anisotropy,
        "reconstruction_sigma": RECONSTRUCTION_SIGMA,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary: Dict[str, Any] = {}
    numeric_keys = ("psnr", "ssim", "lpips", "delta_e00", "anisotropy_index", "runtime_sec")
    for method, method_rows in by_method.items():
        means: Dict[str, float] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in method_rows if key in row and np.isfinite(row[key])]
            if values:
                means[key] = float(np.mean(values))
        summary[method] = means
    return summary


def leaderboard_markdown(summary: Dict[str, Any], icc_fingerprint: Dict[str, Any]) -> str:
    lines = [
        "# Halftoning Benchmark v3 (Printer-Style CMYK/ICC) — Leaderboard",
        "",
        f"Reconstruction operator: Gaussian blur, sigma={RECONSTRUCTION_SIGMA} (spec Sec. 5.2), applied per CMYK plane",
        f"ICC pipeline: {'ICC-managed' if icc_fingerprint['use_icc'] else 'naive full-GCR fallback'} "
        f"(profile: `{Path(icc_fingerprint['cmyk_profile_path']).name}`, intent: {icc_fingerprint['rendering_intent']}, "
        f"BPC: {icc_fingerprint['black_point_compensation']})",
        "",
        "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Aniso ↓ | ΔE00 ↓ | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in sorted(summary.items(), key=lambda kv: kv[1].get("psnr", 0.0), reverse=True):
        lines.append(
            f"| {method} | {values.get('psnr', float('nan')):.3f} | {values.get('ssim', float('nan')):.4f} | "
            f"{values.get('lpips', float('nan')):.4f} | {values.get('anisotropy_index', float('nan')):.6e} | "
            f"{values.get('delta_e00', float('nan')):.4f} | {values.get('runtime_sec', float('nan')):.4f} |"
        )
    reason = lpips_unavailable_reason()
    if reason:
        lines += ["", f"_LPIPS reported as NaN: optional `lpips` package unavailable ({reason})._"]
    if not icc_fingerprint["use_icc"]:
        lines += ["", f"_ICC fallback reason: {icc_fingerprint['fallback_reason']}_"]
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
    cmyk_profile_path: str | None = None,
    rendering_intent: str | None = None,
    no_bpc: bool = False,
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
        cmyk_profile_path=cmyk_profile_path,
        rendering_intent=rendering_intent,
        no_bpc=no_bpc,
    )

    output_dir = output_override or (REPO_ROOT / config.get("output_dir", "benchmarking_v3/output/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    local_datasets_root = REPO_ROOT / config.get("datasets_root", "datasets")

    _log(f"[setup] config={config_path}", quiet=quiet)
    _log(f"[setup] output_dir={output_dir}", quiet=quiet)

    icc_pipeline = build_icc_pipeline(config)
    icc_fingerprint = icc_pipeline.fingerprint()
    (output_dir / "icc_fingerprint.json").write_text(json.dumps(icc_fingerprint, indent=2), encoding="utf-8")
    _log(
        f"[icc] {'ICC-managed' if icc_pipeline.use_icc else 'NAIVE FALLBACK'} | "
        f"profile={Path(icc_fingerprint['cmyk_profile_path']).name} | intent={icc_fingerprint['rendering_intent']} | "
        f"bpc={icc_fingerprint['black_point_compensation']}",
        quiet=quiet,
    )
    if not icc_pipeline.use_icc:
        _log(f"[icc] fallback reason: {icc_fingerprint['fallback_reason']}", quiet=quiet)

    items, manifest = build_eval_items(config, local_datasets_root)
    save_manifest(manifest, output_dir / "manifest.json")
    n_synthetic = sum(1 for item in items if item.synthetic)
    _log(
        f"[dataset] {len(items)} evaluation items ({len(manifest)} base images x (1 + 3 stress variants))"
        f"{f' -- {n_synthetic} synthetic-fallback items' if n_synthetic else ''}",
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
        algorithm = build_plane_algorithm(algo_cfg.get("algorithm", name), algo_cfg, device=device)
        display_name = algo_cfg.get("display_name", name)
        algorithms.append((display_name, algorithm))
        _log(f"[build] {name} ready ({time.perf_counter() - build_start:.1f}s)", quiet=quiet)

    total_runs = len(items) * len(algorithms)
    _log(f"[run] {len(algorithms)} method(s) x {len(items)} item(s) = {total_runs} evaluations", quiet=quiet)

    sample_indices: List[int] = []
    if save_comparisons and n_comparisons > 0:
        sample_indices = select_sample_item_indices([item.family for item in items], [item.variant for item in items], n_comparisons)
        _log(f"[compare] will save CMYK comparison panels for {len(sample_indices)} sample item(s) per method", quiet=quiet)
    sample_index_set = set(sample_indices)

    rows: List[Dict[str, Any]] = []
    completed = 0
    for display_name, algorithm in algorithms:
        method_start = time.perf_counter()
        method_samples: List[Dict[str, Any]] = []
        for item_index, item in enumerate(items):
            capture = {} if item_index in sample_index_set else None
            row = evaluate_item(display_name, algorithm, item, icc_pipeline, seed, compute_lpips, capture=capture)
            rows.append(row)
            completed += 1
            _log(
                f"[{completed}/{total_runs}] {display_name:<18} {item.family:<17} {item.variant:<20} "
                f"psnr={row['psnr']:6.2f} dE00={row['delta_e00']:6.2f} ({row['runtime_sec']:.3f}s)",
                quiet=quiet,
            )
            if capture:
                method_samples.append(capture)
        _log(f"[run] {display_name} done in {time.perf_counter() - method_start:.1f}s", quiet=quiet)

        if save_comparisons and method_samples:
            panel_path = save_method_panel(display_name, method_samples, output_dir / "comparisons")
            _log(f"[compare] wrote {panel_path}", quiet=quiet)

    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "leaderboard.md").write_text(leaderboard_markdown(summary, icc_fingerprint), encoding="utf-8")

    provenance = {
        "config": str(config_path),
        "git_commit": _git("rev-parse", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "reconstruction_sigma": RECONSTRUCTION_SIGMA,
        "icc": icc_fingerprint,
        "n_items": len(items),
        "n_algorithms": len(algorithms),
        "methods": [name for name, _ in algorithms],
        "lpips_unavailable_reason": lpips_unavailable_reason(),
        "any_synthetic_images": any(item.synthetic for item in items),
        "elapsed_sec": time.perf_counter() - started,
        "comparisons_per_method_requested": n_comparisons if save_comparisons else 0,
        "comparisons_per_method_actual": len(sample_indices),
    }
    (output_dir / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _log(f"[done] wrote results to {output_dir} in {provenance['elapsed_sec']:.1f}s", quiet=quiet)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the printer-style CMYK/ICC halftoning benchmark (v3).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "benchmarking_v3/config/default.toml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--images-per-family", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None, metavar="METHOD")
    parser.add_argument("--disable", nargs="+", default=None, metavar="METHOD")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--comparisons-per-method", type=int, default=None, metavar="N")
    parser.add_argument("--cmyk-profile", type=str, default=None, dest="cmyk_profile_path", help="Path to any destination CMYK ICC profile.")
    parser.add_argument(
        "--rendering-intent",
        choices=["perceptual", "relative_colorimetric", "saturation", "absolute_colorimetric"],
        default=None,
    )
    parser.add_argument("--no-bpc", action="store_true", help="Disable black point compensation.")
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
        cmyk_profile_path=args.cmyk_profile_path,
        rendering_intent=args.rendering_intent,
        no_bpc=args.no_bpc,
    )
    print(output)


if __name__ == "__main__":
    main()
