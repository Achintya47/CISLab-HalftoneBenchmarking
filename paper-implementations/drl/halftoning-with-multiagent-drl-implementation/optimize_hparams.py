from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from multiagent_drl import (
    DEFAULT_HPARAMS_PATH,
    MultiAgentDRLConfig,
    anisotropy_loss_from_probs,
    build_optimizer_and_scheduler,
    build_reference_model,
    compute_ssim,
    hvs_filter,
    infer_halftone,
    low_frequency_ratio_loss,
    list_images,
    load_grayscale_image,
    reference_config,
    train_step,
)
from train_multiagent_drl import (
    CONSOLE,
    DeterministicRandomCropDataset,
    apply_brightness_jitter,
    format_duration,
    format_progress_bar,
    load_manifest,
    move_optimizer_state,
    pad_if_needed,
    resolve_device,
    resize_for_preview,
    sample_constant_gray_batch,
    set_global_seed,
)


@dataclass(frozen=True)
class TrialSpec:
    trial_id: int
    overrides: Dict[str, object]


@dataclass(frozen=True)
class StageResult:
    step: int
    score: float
    metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Successive-halving hyperparameter optimization for the multi-agent DRL halftoner.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing training images.")
    parser.add_argument("--eval-root", type=Path, required=True, help="Root directory containing evaluation images.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Optional training manifest.")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="Optional evaluation manifest.")
    parser.add_argument("--run-dir", type=Path, default=Path("paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/hpo"), help="Directory for HPO artifacts.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Enable stricter deterministic settings.")
    parser.add_argument("--cache-images", action="store_true", help="Cache decoded grayscale images per worker.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count per trial. Default 0 for robustness during search.")
    parser.add_argument("--max-train-images", type=int, default=0, help="Optional cap on discovered training images. Use 0 for all.")
    parser.add_argument("--max-eval-images", type=int, default=8, help="Number of evaluation images to score each trial on.")
    parser.add_argument("--eval-max-side", type=int, default=256, help="Maximum evaluation image side length.")
    parser.add_argument("--gray-eval-size", type=int, default=128, help="Square size for constant-gray evaluation patches.")
    parser.add_argument("--gray-levels", default="0.25,0.5,0.75", help="Comma-separated gray levels for constant-gray evaluation.")
    parser.add_argument("--trials", type=int, default=12, help="Number of initial hyperparameter trials.")
    parser.add_argument("--stage-steps", default="400,1200,3200", help="Comma-separated cumulative training steps for successive halving stages.")
    parser.add_argument("--survivor-fraction", type=float, default=0.5, help="Fraction of trials to keep after each stage.")
    parser.add_argument(
        "--search-space",
        default="paper_weights",
        choices=["paper_weights", "nonpaper_impl"],
        help="Which hyperparameter family to search. Use nonpaper_impl to keep paper-specified weights fixed.",
    )
    parser.add_argument("--brightness-jitter", type=float, default=0.9, help="Brightness jitter amount used during trial training.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size during search.")
    parser.add_argument("--crop-size", type=int, default=None, help="Override crop size during search.")
    parser.add_argument("--channels", type=int, default=None, help="Override model width during search.")
    parser.add_argument("--num-res-blocks", type=int, default=None, help="Override residual block count during search.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate during search.")
    parser.add_argument("--learning-rate-min", type=float, default=None, help="Override cosine minimum learning rate during search.")
    parser.add_argument("--dispersion-weight", type=float, default=None, help="Override natural-image dispersion regularizer weight during search.")
    parser.add_argument("--threshold-density-weight", type=float, default=None, help="Override soft-threshold density-alignment weight during search.")
    parser.add_argument("--threshold-temperature", type=float, default=None, help="Override soft-threshold surrogate temperature during search.")
    parser.add_argument("--trial-log-interval", type=int, default=0, help="Per-trial logging interval in steps. Use 0 for auto.")
    parser.add_argument("--apply-best", action="store_true", help="Write the winning hyperparameters into the trainer default file.")
    parser.add_argument("--applied-config-path", type=Path, default=DEFAULT_HPARAMS_PATH, help="Path to write winning overrides when --apply-best is set.")
    parser.add_argument("--keep-trial-checkpoints", action="store_true", help="Keep per-trial checkpoints and directories for non-winning trials.")
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    args.gray_levels = [float(value) for value in args.gray_levels.split(",") if value.strip()]
    args.stage_steps = [int(value) for value in args.stage_steps.split(",") if value.strip()]
    if not args.stage_steps:
        raise ValueError("--stage-steps must contain at least one value.")
    if args.stage_steps != sorted(args.stage_steps):
        raise ValueError("--stage-steps must be sorted in ascending order.")
    if not (0.0 < args.survivor_fraction <= 1.0):
        raise ValueError("--survivor-fraction must be in (0, 1].")
    return args


def print_separator() -> None:
    print(CONSOLE.paint("=" * 96, CONSOLE.dim))


def format_override_summary(overrides: Dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in overrides.items())


def print_hpo_header(args: argparse.Namespace, device: torch.device, train_count: int, eval_count: int) -> None:
    print_separator()
    print(CONSOLE.paint("Multi-Agent DRL HPO", CONSOLE.bold, CONSOLE.blue))
    print(
        f"{CONSOLE.paint('dataset', CONSOLE.cyan)}={args.dataset_root} ({train_count} train) | "
        f"{CONSOLE.paint('eval', CONSOLE.cyan)}={args.eval_root} ({eval_count} images)"
    )
    print(
        f"{CONSOLE.paint('run_dir', CONSOLE.cyan)}={args.run_dir} | "
        f"{CONSOLE.paint('device', CONSOLE.cyan)}={device} | "
        f"{CONSOLE.paint('search_space', CONSOLE.cyan)}={args.search_space}"
    )
    print(
        f"{CONSOLE.paint('trials', CONSOLE.cyan)}={args.trials} | "
        f"{CONSOLE.paint('stages', CONSOLE.cyan)}={args.stage_steps} | "
        f"{CONSOLE.paint('survivor_fraction', CONSOLE.cyan)}={args.survivor_fraction:.2f}"
    )
    print_separator()


def print_stage_header(stage_index: int, total_stages: int, stage_step: int, active_trials: int) -> None:
    print(
        f"{CONSOLE.paint('[stage]', CONSOLE.bold, CONSOLE.magenta)} "
        f"{CONSOLE.paint(f'{stage_index}/{total_stages}', CONSOLE.bold)} "
        f"{CONSOLE.paint('target_step', CONSOLE.gray)}={stage_step} | "
        f"{CONSOLE.paint('active_trials', CONSOLE.gray)}={active_trials}"
    )


def print_trial_header(trial_id: int, stage_step: int, start_step: int, overrides: Dict[str, object]) -> None:
    print(
        f"  {CONSOLE.paint('[trial]', CONSOLE.bold, CONSOLE.green)} "
        f"{CONSOLE.paint(f'{trial_id:03d}', CONSOLE.bold)} | "
        f"{CONSOLE.paint('resume', CONSOLE.gray)}={start_step} -> "
        f"{CONSOLE.paint('target', CONSOLE.gray)}={stage_step} | "
        f"{CONSOLE.paint(format_override_summary(overrides), CONSOLE.dim)}"
    )


def print_trial_progress(
    trial_id: int,
    local_step: int,
    local_total: int,
    metrics: Dict[str, float],
    lr: float,
    elapsed_seconds: float,
) -> None:
    progress = 100.0 * local_step / max(1, local_total)
    iterations_per_second = local_step / max(elapsed_seconds, 1e-9)
    eta_seconds = max(0, local_total - local_step) / max(iterations_per_second, 1e-9)
    bar = format_progress_bar(local_step, local_total, width=20)
    print(
        "    "
        f"{CONSOLE.paint('[step]', CONSOLE.bold, CONSOLE.yellow)} "
        f"{CONSOLE.paint(f'trial={trial_id:03d}', CONSOLE.gray)} "
        f"{CONSOLE.paint('[' + bar + ']', CONSOLE.yellow)} "
        f"{CONSOLE.paint(f'{local_step:5d}/{local_total}', CONSOLE.bold)} "
        f"{CONSOLE.paint(f'{progress:6.2f}%', CONSOLE.bold, CONSOLE.yellow)} | "
        f"{CONSOLE.paint(f'{iterations_per_second:5.2f} it/s', CONSOLE.magenta)} | "
        f"{CONSOLE.paint('elapsed', CONSOLE.gray)} {format_duration(elapsed_seconds)} | "
        f"{CONSOLE.paint('eta', CONSOLE.gray)} {format_duration(eta_seconds)} | "
        f"{CONSOLE.paint('lr', CONSOLE.gray)} {lr:.3e}"
    )
    print(
        "        "
        f"{CONSOLE.paint('loss', CONSOLE.cyan)} {metrics['total_loss']:10.6f} | "
        f"{CONSOLE.paint('reward', CONSOLE.cyan)} {metrics['reward']:10.6f} | "
        f"{CONSOLE.paint('tone', CONSOLE.cyan)} {metrics['tone_error']:10.6f} | "
        f"{CONSOLE.paint('ssim', CONSOLE.cyan)} {metrics['ssim']:10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('disp', CONSOLE.cyan)} {metrics['dispersion_loss']:10.6f} | "
        f"{CONSOLE.paint('anis', CONSOLE.cyan)} {metrics['anisotropy_loss']:10.6f} | "
        f"{CONSOLE.paint('thr_dens', CONSOLE.cyan)} {metrics['threshold_density_loss']:10.6f} | "
        f"{CONSOLE.paint('p_mean', CONSOLE.cyan)} {metrics['prob_mean']:10.6f} | "
        f"{CONSOLE.paint('p_std', CONSOLE.cyan)} {metrics['prob_std']:10.6f} | "
        f"{CONSOLE.paint('g_mean', CONSOLE.cyan)} {metrics['gray_prob_mean']:10.6f}"
    )


def print_trial_result(stage_result: StageResult, trial_id: int) -> None:
    metrics = stage_result.metrics
    print(
        "    "
        f"{CONSOLE.paint('[eval]', CONSOLE.bold, CONSOLE.blue)} "
        f"{CONSOLE.paint(f'trial={trial_id:03d}', CONSOLE.gray)} | "
        f"{CONSOLE.paint('score', CONSOLE.cyan)}={stage_result.score:.6f} | "
        f"{CONSOLE.paint('tone', CONSOLE.cyan)}={metrics['tone_error_mean']:.6f} | "
        f"{CONSOLE.paint('ssim', CONSOLE.cyan)}={metrics['ssim_mean']:.6f} | "
        f"{CONSOLE.paint('nat_disp', CONSOLE.cyan)}={metrics['natural_dispersion_mean']:.6f} | "
        f"{CONSOLE.paint('gray_gap', CONSOLE.cyan)}={metrics['gray_halftone_density_gap_mean']:.6f}"
    )


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index > 0:
        result += fraction * (index % base)
        index //= base
        fraction /= base
    return result


def halton_vector(index: int, dimensions: int) -> List[float]:
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    if dimensions > len(primes):
        raise ValueError("Requested more Halton dimensions than configured primes.")
    return [halton(index, primes[i]) for i in range(dimensions)]


def lerp_log(unit: float, minimum: float, maximum: float) -> float:
    unit = min(max(unit, 1e-9), 1.0 - 1e-9)
    return math.exp(math.log(minimum) + unit * (math.log(maximum) - math.log(minimum)))


def lerp_linear(unit: float, minimum: float, maximum: float) -> float:
    unit = min(max(unit, 0.0), 1.0)
    return minimum + unit * (maximum - minimum)


def lerp_int_log(unit: float, minimum: int, maximum: int, quantum: int = 10) -> int:
    value = int(round(lerp_log(unit, float(minimum), float(maximum)) / quantum) * quantum)
    return max(minimum, min(maximum, value))


def sample_discrete(unit: float, values: List[object]) -> object:
    if not values:
        raise ValueError("Discrete search space must not be empty.")
    scaled = min(max(unit, 0.0), 1.0 - 1e-12)
    index = min(int(math.floor(scaled * len(values))), len(values) - 1)
    return values[index]


def sample_trial_overrides(index: int, search_space: str) -> Dict[str, object]:
    if search_space == "paper_weights":
        u = halton_vector(index, 2)
        return {
            "anisotropy_weight": lerp_log(u[0], 5e-4, 5e-3),
            "ssim_weight": lerp_log(u[1], 0.003, 0.012),
        }
    if search_space == "nonpaper_impl":
        u = halton_vector(index, 7)
        return {
            "hvs_kernel_size": int(sample_discrete(u[0], [7, 9, 11, 13, 15])),
            "hvs_luminance": lerp_linear(u[1], 8.0, 16.0),
            "ssim_kernel_size": int(sample_discrete(u[2], [7, 9, 11, 13])),
            "ssim_sigma": lerp_linear(u[3], 1.0, 2.0),
            "dispersion_weight": lerp_log(u[4], 1e-4, 5e-3),
            "threshold_density_weight": lerp_log(u[5], 1e-3, 5e-2),
            "threshold_temperature": lerp_linear(u[6], 0.03, 0.12),
        }
    raise ValueError(f"Unsupported search space: {search_space}")


def build_search_config(base: MultiAgentDRLConfig, args: argparse.Namespace, overrides: Dict[str, object]) -> MultiAgentDRLConfig:
    merged = asdict(base)
    merged.update(overrides)
    if args.crop_size is not None:
        merged["crop_size"] = args.crop_size
    if args.batch_size is not None:
        merged["batch_size"] = args.batch_size
    if args.channels is not None:
        merged["channels"] = args.channels
    if args.num_res_blocks is not None:
        merged["num_res_blocks"] = args.num_res_blocks
    if args.learning_rate is not None:
        merged["learning_rate"] = args.learning_rate
    if args.learning_rate_min is not None:
        merged["learning_rate_min"] = args.learning_rate_min
    if args.dispersion_weight is not None:
        merged["dispersion_weight"] = args.dispersion_weight
    if args.threshold_density_weight is not None:
        merged["threshold_density_weight"] = args.threshold_density_weight
    if args.threshold_temperature is not None:
        merged["threshold_temperature"] = args.threshold_temperature
    return MultiAgentDRLConfig(**merged)


def select_eval_images(image_paths: List[Path], max_images: int, seed: int) -> List[Path]:
    if max_images <= 0 or len(image_paths) <= max_images:
        return list(image_paths)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(image_paths), size=max_images, replace=False)
    return [image_paths[int(index)] for index in sorted(indices)]


def build_trial_dataloader(
    image_paths: List[Path],
    config: MultiAgentDRLConfig,
    args: argparse.Namespace,
    start_step: int,
    target_step: int,
    device: torch.device,
) -> DataLoader[Tensor]:
    dataset = DeterministicRandomCropDataset(
        image_paths=image_paths,
        crop_size=config.crop_size,
        total_samples=max(0, target_step - start_step) * config.batch_size,
        base_seed=args.seed,
        start_index=start_step * config.batch_size,
        cache_images=args.cache_images,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def save_trial_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
) -> None:
    payload: Dict[str, object] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def load_trial_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    move_optimizer_state(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.random.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
    return int(checkpoint["step"])


def evaluate_trial(
    model: torch.nn.Module,
    config: MultiAgentDRLConfig,
    eval_images: List[Path],
    gray_levels: List[float],
    eval_max_side: int,
    gray_eval_size: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    tone_errors: List[float] = []
    ssims: List[float] = []
    prob_means: List[float] = []
    prob_density_gaps: List[float] = []
    halftone_density_gaps: List[float] = []
    natural_dispersion_terms: List[float] = []
    gray_prob_means: List[float] = []
    gray_density_gaps: List[float] = []
    gray_halftone_density_gaps: List[float] = []
    gray_anisotropy_logs: List[float] = []

    model.eval()
    with torch.no_grad():
        for index, image_path in enumerate(eval_images):
            contone = load_grayscale_image(image_path).unsqueeze(0).to(device)
            contone = resize_for_preview(contone, eval_max_side)
            generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
            generator.manual_seed(seed + index)
            noise = torch.randn(contone.shape, generator=generator, device=device, dtype=contone.dtype)
            result = infer_halftone(model, contone, config, noise=noise, binary_mode="threshold")
            filtered_halftone = hvs_filter(result["halftone"], config)
            filtered_contone = hvs_filter(contone, config)
            tone_errors.append(float((filtered_halftone - filtered_contone).square().mean().item()))
            ssims.append(float(compute_ssim(result["halftone"], contone, config).mean().item()))
            prob_mean = float(result["probability"].mean().item())
            halftone_mean = float(result["halftone"].mean().item())
            target_mean = float(contone.mean().item())
            prob_means.append(prob_mean)
            prob_density_gaps.append(abs(prob_mean - target_mean))
            halftone_density_gaps.append(abs(halftone_mean - target_mean))
            natural_error = result["halftone"] - contone
            natural_dispersion_terms.append(
                float(
                    (
                        anisotropy_loss_from_probs(natural_error)
                        + low_frequency_ratio_loss(natural_error, max_radius=config.dispersion_lowfreq_radius, eps=config.eps)
                    ).item()
                )
            )

        for index, gray_level in enumerate(gray_levels):
            gray = torch.full((1, 1, gray_eval_size, gray_eval_size), float(gray_level), device=device)
            generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
            generator.manual_seed(seed + 10_000 + index)
            noise = torch.randn(gray.shape, generator=generator, device=device, dtype=gray.dtype)
            result = infer_halftone(model, gray, config, noise=noise, binary_mode="threshold")
            probs = result["probability"]
            halftone = result["halftone"]
            gray_prob_mean = float(probs.mean().item())
            gray_prob_means.append(gray_prob_mean)
            gray_density_gaps.append(abs(gray_prob_mean - float(gray_level)))
            gray_halftone_density_gaps.append(abs(float(halftone.mean().item()) - float(gray_level)))
            gray_anisotropy_logs.append(math.log1p(float(anisotropy_loss_from_probs(probs).item())))

    metrics = {
        "tone_error_mean": float(np.mean(tone_errors)),
        "ssim_mean": float(np.mean(ssims)),
        "prob_mean_mean": float(np.mean(prob_means)),
        "prob_density_gap_mean": float(np.mean(prob_density_gaps)),
        "halftone_density_gap_mean": float(np.mean(halftone_density_gaps)),
        "natural_dispersion_mean": float(np.mean(natural_dispersion_terms)),
        "gray_prob_mean_mean": float(np.mean(gray_prob_means)),
        "gray_density_gap_mean": float(np.mean(gray_density_gaps)),
        "gray_halftone_density_gap_mean": float(np.mean(gray_halftone_density_gaps)),
        "gray_anisotropy_log_mean": float(np.mean(gray_anisotropy_logs)),
    }
    metrics["score"] = (
        metrics["tone_error_mean"]
        + 0.25 * metrics["halftone_density_gap_mean"]
        + 0.05 * metrics["natural_dispersion_mean"]
        + metrics["gray_halftone_density_gap_mean"]
        + 0.01 * metrics["gray_anisotropy_log_mean"]
        - 0.1 * metrics["ssim_mean"]
    )
    return metrics


def train_trial_to_step(
    trial_dir: Path,
    trial_id: int,
    train_images: List[Path],
    eval_images: List[Path],
    base_config: MultiAgentDRLConfig,
    overrides: Dict[str, object],
    args: argparse.Namespace,
    target_step: int,
    max_stage_step: int,
    device: torch.device,
) -> StageResult:
    config = build_search_config(base_config, args, overrides)
    model = build_reference_model(config, device=device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, max_stage_step, config)
    checkpoint_path = trial_dir / "checkpoint.pt"
    start_step = 0
    if checkpoint_path.exists():
        start_step = load_trial_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
    if start_step > target_step:
        raise RuntimeError(f"Trial {trial_id} checkpoint step {start_step} exceeds requested target step {target_step}.")

    print_trial_header(trial_id, target_step, start_step, overrides)
    if start_step < target_step:
        dataloader = build_trial_dataloader(train_images, config, args, start_step, target_step, device)
        data_iter = iter(dataloader)
        trial_total = target_step - start_step
        auto_interval = max(1, min(500, max(1, trial_total // 8)))
        trial_log_interval = args.trial_log_interval if args.trial_log_interval > 0 else auto_interval
        interval_metrics: Dict[str, float] = {}
        interval_count = 0
        trial_start = time.perf_counter()
        for step in range(start_step + 1, target_step + 1):
            contone = next(data_iter).to(device)
            contone = apply_brightness_jitter(contone, args.brightness_jitter)
            gray = sample_constant_gray_batch(contone, config)
            metrics = train_step(model, optimizer, contone, gray, config, step=step)
            scheduler.step()
            for key, value in metrics.items():
                interval_metrics[key] = interval_metrics.get(key, 0.0) + float(value)
            interval_count += 1

            local_step = step - start_step
            if local_step % trial_log_interval == 0 or step == target_step:
                averaged = {key: value / interval_count for key, value in interval_metrics.items()}
                print_trial_progress(
                    trial_id=trial_id,
                    local_step=local_step,
                    local_total=trial_total,
                    metrics=averaged,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    elapsed_seconds=time.perf_counter() - trial_start,
                )
                interval_metrics = {}
                interval_count = 0
        save_trial_checkpoint(checkpoint_path, model, optimizer, scheduler, target_step)

    eval_seed = args.seed + trial_id * 100_000 + target_step * 37
    metrics = evaluate_trial(
        model=model,
        config=config,
        eval_images=eval_images,
        gray_levels=args.gray_levels,
        eval_max_side=args.eval_max_side,
        gray_eval_size=args.gray_eval_size,
        seed=eval_seed,
        device=device,
    )
    stage_result = StageResult(step=target_step, score=metrics["score"], metrics=metrics)
    save_json(
        trial_dir / f"stage-{target_step:06d}.json",
        {
            "trial_id": trial_id,
            "step": target_step,
            "overrides": overrides,
            "config": asdict(config),
            "metrics": metrics,
        },
    )
    print_trial_result(stage_result, trial_id)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stage_result


def prune_trial_dir(trial_dir: Path) -> None:
    if trial_dir.exists():
        shutil.rmtree(trial_dir)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_global_seed(args.seed, args.deterministic)

    train_images = load_manifest(args.dataset_manifest, args.dataset_root) if args.dataset_manifest is not None else list_images(args.dataset_root)
    if args.max_train_images > 0:
        train_images = train_images[: args.max_train_images]
    if not train_images:
        raise RuntimeError(f"No training images found under {args.dataset_root}")
    eval_source = load_manifest(args.eval_manifest, args.eval_root) if args.eval_manifest is not None else list_images(args.eval_root)
    if not eval_source:
        raise RuntimeError(f"No evaluation images found under {args.eval_root}")
    eval_images = select_eval_images(eval_source, args.max_eval_images, args.seed)

    run_dir = args.run_dir
    trials_dir = run_dir / "trials"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = reference_config()
    save_json(
        run_dir / "search_config.json",
        {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "base_config": asdict(base_config),
            "train_image_count": len(train_images),
            "eval_image_count": len(eval_images),
            "eval_images": [str(path) for path in eval_images],
        },
    )

    trial_specs = [TrialSpec(trial_id=index, overrides=sample_trial_overrides(index + 1, args.search_space)) for index in range(args.trials)]
    active_trials = list(trial_specs)
    stage_history: List[Dict[str, object]] = []
    max_stage_step = args.stage_steps[-1]
    start_time = time.perf_counter()

    print_hpo_header(args, device, len(train_images), len(eval_images))

    for stage_index, stage_step in enumerate(args.stage_steps):
        print_stage_header(stage_index + 1, len(args.stage_steps), stage_step, len(active_trials))
        stage_results: List[Dict[str, object]] = []
        for rank, trial in enumerate(active_trials, start=1):
            trial_dir = trials_dir / f"trial-{trial.trial_id:03d}"
            stage_result = train_trial_to_step(
                trial_dir=trial_dir,
                trial_id=trial.trial_id,
                train_images=train_images,
                eval_images=eval_images,
                base_config=base_config,
                overrides=trial.overrides,
                args=args,
                target_step=stage_step,
                max_stage_step=max_stage_step,
                device=device,
            )
            payload = {
                "trial_id": trial.trial_id,
                "rank_input": rank,
                "step": stage_result.step,
                "score": stage_result.score,
                "overrides": trial.overrides,
                "metrics": stage_result.metrics,
            }
            stage_results.append(payload)

        stage_results.sort(key=lambda item: item["score"])
        keep_count = 1 if stage_index == len(args.stage_steps) - 1 else max(1, math.ceil(len(stage_results) * args.survivor_fraction))
        surviving_ids = {item["trial_id"] for item in stage_results[:keep_count]}
        stage_summary = {
            "stage_index": stage_index,
            "target_step": stage_step,
            "keep_count": keep_count,
            "results": stage_results,
        }
        stage_history.append(stage_summary)
        save_json(run_dir / f"stage-{stage_step:06d}.json", stage_summary)
        kept_trials = ", ".join(f"{trial_id:03d}" for trial_id in sorted(surviving_ids))
        print(
            f"{CONSOLE.paint('[stage-done]', CONSOLE.bold, CONSOLE.green)} "
            f"{CONSOLE.paint('target_step', CONSOLE.gray)}={stage_step} | "
            f"{CONSOLE.paint('keep', CONSOLE.gray)}={keep_count} | "
            f"{CONSOLE.paint('survivors', CONSOLE.gray)}={kept_trials}"
        )

        if stage_index < len(args.stage_steps) - 1:
            next_active_trials: List[TrialSpec] = []
            for trial in active_trials:
                if trial.trial_id in surviving_ids:
                    next_active_trials.append(trial)
                elif not args.keep_trial_checkpoints:
                    prune_trial_dir(trials_dir / f"trial-{trial.trial_id:03d}")
            active_trials = next_active_trials

    best_result = stage_history[-1]["results"][0]
    best_overrides = best_result["overrides"]
    best_config = build_search_config(base_config, args, best_overrides)
    best_payload = {
        "best_trial_id": best_result["trial_id"],
        "best_score": best_result["score"],
        "best_overrides": best_overrides,
        "best_config": asdict(best_config),
        "best_metrics": best_result["metrics"],
        "elapsed_seconds": time.perf_counter() - start_time,
        "stages": stage_history,
    }
    save_json(run_dir / "best_result.json", best_payload)
    save_json(run_dir / "best_hparams.json", best_overrides)

    """
        default_hparams.json is meant to be a cummulative storage file.
        reference_config() reads and merges its contents, but the writing
        part replaces the file's contents with just the current search space's
        overrides. 

        Thus the correct behaviour should be, read the existing file, merge best_overrides
        on top and write the merged dictionary, no overwrites.
    """
    if args.apply_best:
        existing_overrides: Dict[str, object] = {}
        if args.applied_config_path.exists():
            loaded = json.loads(args.applied_config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_overrides = loaded
        merged_overrides = {**existing_overrides, **best_overrides}
        save_json(args.applied_config_path, merged_overrides)
        print(
            f"Applied best overrides to {args.applied_config_path} "
            f"(merged {len(best_overrides)} new key(s) with {len(existing_overrides)} existing key(s))"
        )

    print_separator()
    print(
        f"{CONSOLE.paint('[best]', CONSOLE.bold, CONSOLE.magenta)} "
        f"trial={best_result['trial_id']:03d} | "
        f"score={best_result['score']:.6f} | "
        f"elapsed={format_duration(time.perf_counter() - start_time)}"
    )
    print(CONSOLE.paint(json.dumps(best_overrides, sort_keys=True), CONSOLE.dim))


if __name__ == "__main__":
    main()
