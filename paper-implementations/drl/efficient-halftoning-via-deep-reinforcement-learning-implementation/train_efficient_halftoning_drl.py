from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from efficient_halftoning_drl import (
    DRLHalftoningConfig,
    PolicyNet,
    build_optimizer_and_scheduler,
    build_reference_model,
    infer_halftone,
    list_images,
    load_grayscale_image,
    pad_if_needed,
    paper_reference_config,
    save_triptych,
    stabilized_reference_config,
    train_step,
)


class DeterministicRandomCropDataset(Dataset[Tensor]):
    def __init__(
        self,
        image_paths: List[Path],
        crop_size: int,
        total_samples: int,
        base_seed: int,
        start_index: int = 0,
        cache_images: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.total_samples = total_samples
        self.base_seed = base_seed
        self.start_index = start_index
        self.cache_images = cache_images
        self._cache: Dict[Path, Tensor] = {}

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tensor:
        global_index = self.start_index + index
        rng = np.random.default_rng(self.base_seed + global_index)
        image_path = self.image_paths[int(rng.integers(0, len(self.image_paths)))]
        image = self._load_image(image_path)
        image = pad_if_needed(image, self.crop_size)
        _, height, width = image.shape
        top = int(rng.integers(0, height - self.crop_size + 1))
        left = int(rng.integers(0, width - self.crop_size + 1))
        return image[:, top : top + self.crop_size, left : left + self.crop_size].contiguous()

    def _load_image(self, path: Path) -> Tensor:
        if not self.cache_images:
            return load_grayscale_image(path)
        cached = self._cache.get(path)
        if cached is None:
            cached = load_grayscale_image(path)
            self._cache[path] = cached
        return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact DRL halftoning reference.")
    parser.add_argument("--variant", choices=["stabilized", "paper"], default="stabilized", help="Training variant. `stabilized` preserves the collapse-resistant run; `paper` targets closer paper fidelity.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing training images.")
    parser.add_argument("--eval-root", type=Path, default=None, help="Optional root directory for preview images. Defaults to the training dataset root.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Optional newline-delimited file listing training images relative to the dataset root or repo root.")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="Optional newline-delimited file listing preview images relative to the eval root or repo root.")
    parser.add_argument("--run-dir", type=Path, default=Path("paper-implementations/drl/efficient-halftoning-via-deep-reinforcement-learning-implementation/output/train-run"), help="Directory for checkpoints, previews, and logs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--seed", type=int, default=0, help="Global seed used for initialization, crops, and previews.")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader worker count.")
    parser.add_argument("--cache-images", action="store_true", help="Cache decoded grayscale images per worker for faster repeated access.")
    parser.add_argument("--total-iterations", type=int, default=200000, help="Total optimization iterations.")
    parser.add_argument("--log-interval", type=int, default=100, help="How often to print and append averaged metrics.")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="How often to save checkpoints.")
    parser.add_argument("--preview-interval", type=int, default=5000, help="How often to save preview triptychs.")
    parser.add_argument("--preview-count", type=int, default=4, help="Number of fixed preview images to track across training.")
    parser.add_argument("--preview-max-side", type=int, default=256, help="Maximum side length for preview inference. Use 0 to keep original size.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--allow-resume-mismatch", action="store_true", help="Allow resume even if the saved config or total iteration count differs from the current run.")
    parser.add_argument("--deterministic", action="store_true", help="Enable stricter deterministic settings at the cost of throughput.")
    parser.add_argument("--max-train-images", type=int, default=0, help="Optional cap on discovered training images. Use 0 for all.")
    parser.add_argument("--crop-size", type=int, default=None, help="Training crop size. Defaults to the chosen variant.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size. Defaults to the chosen variant.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial Adam learning rate. Defaults to the chosen variant.")
    parser.add_argument("--learning-rate-min", type=float, default=None, help="Minimum cosine-decay learning rate. Defaults to the chosen variant.")
    parser.add_argument("--cssim-weight", type=float, default=None, help="CSSIM reward weight. Defaults to the chosen variant.")
    parser.add_argument("--anisotropy-weight", type=float, default=None, help="Anisotropy penalty weight. Defaults to the chosen variant.")
    parser.add_argument("--density-weight", type=float, default=None, help="Uniform-gray density preservation weight. Defaults to the chosen variant.")
    args = parser.parse_args()
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be a positive integer.")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def select_preview_images(image_paths: List[Path], count: int, seed: int) -> List[Path]:
    if count <= 0 or not image_paths:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(image_paths), size=min(count, len(image_paths)), replace=False)
    return [image_paths[int(index)] for index in sorted(indices)]


def resize_for_preview(contone: Tensor, max_side: int) -> Tensor:
    if max_side <= 0:
        return contone
    _, _, height, width = contone.shape
    longest = max(height, width)
    if longest <= max_side:
        return contone
    scale = max_side / float(longest)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.interpolate(contone, size=(new_height, new_width), mode="bilinear", align_corners=False)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class ConsoleStyle:
    def __init__(self) -> None:
        force_color = os.environ.get("FORCE_COLOR") == "1"
        self.enabled = force_color or sys.stdout.isatty()
        self.reset = "\033[0m" if self.enabled else ""
        self.bold = "\033[1m" if self.enabled else ""
        self.dim = "\033[2m" if self.enabled else ""
        self.blue = "\033[38;5;39m" if self.enabled else ""
        self.cyan = "\033[38;5;45m" if self.enabled else ""
        self.green = "\033[38;5;42m" if self.enabled else ""
        self.yellow = "\033[38;5;220m" if self.enabled else ""
        self.magenta = "\033[38;5;213m" if self.enabled else ""
        self.red = "\033[38;5;203m" if self.enabled else ""
        self.gray = "\033[38;5;245m" if self.enabled else ""

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled:
            return text
        return "".join(styles) + text + self.reset


CONSOLE = ConsoleStyle()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_progress_bar(step: int, total_steps: int, width: int = 28) -> str:
    total_steps = max(1, total_steps)
    filled = min(width, int(width * step / total_steps))
    return "#" * filled + "-" * (width - filled)


def print_separator() -> None:
    print(CONSOLE.paint("=" * 92, CONSOLE.dim))


def print_run_header(
    args: argparse.Namespace,
    config: DRLHalftoningConfig,
    device: torch.device,
    image_count: int,
    preview_count: int,
) -> None:
    print_separator()
    print(CONSOLE.paint("Efficient Halftoning DRL Training", CONSOLE.bold, CONSOLE.blue))
    print(
        f"{CONSOLE.paint('variant', CONSOLE.cyan)}={args.variant} | "
        f"{CONSOLE.paint('estimator', CONSOLE.cyan)}={config.policy_estimator} | "
        f"{CONSOLE.paint('dataset', CONSOLE.cyan)}={args.dataset_root} ({image_count} images) | "
        f"{CONSOLE.paint('eval', CONSOLE.cyan)}={args.eval_root if args.eval_root is not None else args.dataset_root} ({preview_count} previews)"
    )
    print(
        f"{CONSOLE.paint('run_dir', CONSOLE.cyan)}={args.run_dir} | "
        f"{CONSOLE.paint('device', CONSOLE.cyan)}={device} | "
        f"{CONSOLE.paint('steps', CONSOLE.cyan)}={args.total_iterations} | "
        f"{CONSOLE.paint('batch', CONSOLE.cyan)}={config.batch_size} | "
        f"{CONSOLE.paint('crop', CONSOLE.cyan)}={config.crop_size}"
    )
    print(
        f"{CONSOLE.paint('workers', CONSOLE.cyan)}={args.num_workers} | "
        f"{CONSOLE.paint('cache_images', CONSOLE.cyan)}={args.cache_images} | "
        f"{CONSOLE.paint('log_every', CONSOLE.cyan)}={args.log_interval} | "
        f"{CONSOLE.paint('ckpt_every', CONSOLE.cyan)}={args.checkpoint_interval} | "
        f"{CONSOLE.paint('preview_every', CONSOLE.cyan)}={args.preview_interval}"
    )
    print_separator()


def print_training_log(log_payload: Dict[str, float], total_iterations: int, elapsed_total: float) -> None:
    step = int(log_payload["step"])
    progress = 100.0 * step / max(1, total_iterations)
    iterations_per_second = float(log_payload["iterations_per_second"])
    remaining_steps = max(0, total_iterations - step)
    eta_seconds = remaining_steps / max(iterations_per_second, 1e-9)
    bar = format_progress_bar(step, total_iterations)
    progress_label = CONSOLE.paint("[train]", CONSOLE.bold, CONSOLE.green)
    step_text = f"{step:7d}/{total_iterations}"
    lr_text = f"{float(log_payload['lr']):.3e}"
    print(
        f"{progress_label} "
        f"{CONSOLE.paint('step', CONSOLE.gray)} {CONSOLE.paint(step_text, CONSOLE.bold)} "
        f"{CONSOLE.paint('[' + bar + ']', CONSOLE.yellow)} "
        f"{CONSOLE.paint(f'{progress:6.2f}%', CONSOLE.bold, CONSOLE.yellow)} | "
        f"{CONSOLE.paint(f'{iterations_per_second:5.2f} it/s', CONSOLE.magenta)} | "
        f"{CONSOLE.paint('elapsed', CONSOLE.gray)} {format_duration(elapsed_total)} | "
        f"{CONSOLE.paint('eta', CONSOLE.gray)} {format_duration(eta_seconds)} | "
        f"{CONSOLE.paint('lr', CONSOLE.gray)} {lr_text}"
    )
    print(
        "        "
        f"{CONSOLE.paint('total_loss', CONSOLE.cyan)} {float(log_payload['total_loss']):10.6f} | "
        f"{CONSOLE.paint('policy_loss', CONSOLE.cyan)} {float(log_payload['policy_loss']):10.6f} | "
        f"{CONSOLE.paint('density_loss', CONSOLE.cyan)} {float(log_payload['density_loss']):10.6f} | "
        f"{CONSOLE.paint('reward', CONSOLE.cyan)} {float(log_payload['reward']):10.6f} | "
        f"{CONSOLE.paint('tone_error', CONSOLE.cyan)} {float(log_payload['tone_error']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('cssim', CONSOLE.cyan)} {float(log_payload['cssim']):10.6f} | "
        f"{CONSOLE.paint('anisotropy', CONSOLE.cyan)} {float(log_payload['anisotropy_loss']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('prob_mean', CONSOLE.cyan)} {float(log_payload['prob_mean']):10.6f} | "
        f"{CONSOLE.paint('prob_std', CONSOLE.cyan)} {float(log_payload['prob_std']):10.6f} | "
        f"{CONSOLE.paint('gray_prob_mean', CONSOLE.cyan)} {float(log_payload['gray_prob_mean']):10.6f} | "
        f"{CONSOLE.paint('gray_prob_std', CONSOLE.cyan)} {float(log_payload['gray_prob_std']):10.6f}"
    )


def checkpoint_payload(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    config: DRLHalftoningConfig,
    args: argparse.Namespace,
    step: int,
    preview_images: Iterable[Path],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "step": step,
        "config": asdict(config),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "preview_images": [str(path) for path in preview_images],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    checkpoint_path: Path,
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    move_optimizer_state(optimizer, device)
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_rng_state"])
    if device.type == "cuda" and checkpoint.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
    return checkpoint


def validate_resume_compatibility(
    checkpoint: Dict[str, object],
    config: DRLHalftoningConfig,
    args: argparse.Namespace,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != asdict(config):
        raise RuntimeError("Checkpoint config does not match the current run config.")
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_total_iterations = checkpoint_args.get("total_iterations")
    if checkpoint_total_iterations is not None and int(checkpoint_total_iterations) != args.total_iterations:
        raise RuntimeError(
            "Checkpoint total_iterations does not match the current run. "
            "Use the original total_iterations or pass --allow-resume-mismatch if you really want this."
        )


def save_preview_triptychs(
    model: PolicyNet,
    config: DRLHalftoningConfig,
    preview_images: List[Path],
    output_dir: Path,
    device: torch.device,
    step: int,
    seed: int,
    preview_max_side: int,
) -> None:
    if not preview_images:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(preview_images):
        contone = load_grayscale_image(image_path).unsqueeze(0).to(device)
        contone = resize_for_preview(contone, preview_max_side)
        generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
        generator.manual_seed(seed + step * 1009 + index)
        noise = torch.randn(contone.shape, generator=generator, device=device, dtype=contone.dtype)
        result = infer_halftone(model, contone, config, noise=noise)
        stem = image_path.stem.replace(" ", "-")
        save_triptych(result, output_dir / f"step-{step:07d}-{stem}.png")


def build_dataloader(
    image_paths: List[Path],
    config: DRLHalftoningConfig,
    args: argparse.Namespace,
    start_step: int,
    device: torch.device,
) -> DataLoader[Tensor]:
    remaining_iterations = args.total_iterations - start_step
    dataset = DeterministicRandomCropDataset(
        image_paths=image_paths,
        crop_size=config.crop_size,
        total_samples=remaining_iterations * config.batch_size,
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


def build_config(args: argparse.Namespace) -> DRLHalftoningConfig:
    base = paper_reference_config() if args.variant == "paper" else stabilized_reference_config()
    return DRLHalftoningConfig(
        variant=base.variant,
        policy_estimator=base.policy_estimator,
        crop_size=args.crop_size if args.crop_size is not None else base.crop_size,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        num_res_blocks=base.num_res_blocks,
        channels=base.channels,
        hvs_model=base.hvs_model,
        hvs_kernel_size=base.hvs_kernel_size,
        hvs_sigma=base.hvs_sigma,
        hvs_scale_factor=base.hvs_scale_factor,
        hvs_luminance=base.hvs_luminance,
        contrast_kernel_size=base.contrast_kernel_size,
        contrast_sigma=base.contrast_sigma,
        contrast_normalization=base.contrast_normalization,
        cssim_weight=args.cssim_weight if args.cssim_weight is not None else base.cssim_weight,
        anisotropy_weight=args.anisotropy_weight if args.anisotropy_weight is not None else base.anisotropy_weight,
        density_weight=args.density_weight if args.density_weight is not None else base.density_weight,
        learning_rate=args.learning_rate if args.learning_rate is not None else base.learning_rate,
        learning_rate_min=args.learning_rate_min if args.learning_rate_min is not None else base.learning_rate_min,
        white_threshold=base.white_threshold,
        init_std=base.init_std,
        eps=base.eps,
    )


def load_manifest(manifest_path: Path, root: Path) -> List[Path]:
    resolved: List[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if candidate.is_absolute():
            path = candidate
        else:
            rooted = root / candidate
            path = rooted if rooted.exists() else (Path.cwd() / candidate)
        if not path.exists():
            raise FileNotFoundError(f"Manifest entry does not exist: {line}")
        resolved.append(path)
    return resolved


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device = resolve_device(args.device)
    set_global_seed(args.seed, args.deterministic)

    image_paths = load_manifest(args.dataset_manifest, args.dataset_root) if args.dataset_manifest is not None else list_images(args.dataset_root)
    if args.max_train_images > 0:
        image_paths = image_paths[: args.max_train_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {args.dataset_root}")

    run_dir = args.run_dir
    checkpoints_dir = run_dir / "checkpoints"
    previews_dir = run_dir / "previews"
    logs_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_root = args.eval_root if args.eval_root is not None else args.dataset_root
    preview_source = load_manifest(args.eval_manifest, eval_root) if args.eval_manifest is not None else (list_images(eval_root) if args.eval_root is not None else image_paths)
    preview_images = select_preview_images(preview_source, args.preview_count, args.seed)
    save_json(
        run_dir / "run_config.json",
        {
            "config": asdict(config),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "image_count": len(image_paths),
            "preview_source_count": len(preview_source),
            "preview_images": [str(path) for path in preview_images],
        },
    )
    print_run_header(args, config, device, len(image_paths), len(preview_images))

    model = build_reference_model(config, device=device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args.total_iterations, config)

    start_step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        if not args.allow_resume_mismatch:
            validate_resume_compatibility(checkpoint, config, args)
        start_step = int(checkpoint["step"])
        preview_images = [Path(path) for path in checkpoint.get("preview_images", preview_images)]
        if start_step >= args.total_iterations:
            raise RuntimeError("Checkpoint step already meets or exceeds total_iterations.")

    dataloader = build_dataloader(image_paths, config, args, start_step, device)
    data_iter = iter(dataloader)

    if start_step == 0:
        save_preview_triptychs(model, config, preview_images, previews_dir, device, 0, args.seed, args.preview_max_side)

    metrics_accumulator: Dict[str, float] = {}
    window_count = 0
    window_start = time.perf_counter()
    training_start = time.perf_counter()

    for step in range(start_step + 1, args.total_iterations + 1):
        contone_batch = next(data_iter)
        contone_batch = contone_batch.to(device, non_blocking=device.type == "cuda")
        metrics = train_step(model, optimizer, scheduler, contone_batch, config)
        for key, value in metrics.items():
            metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + value
        window_count += 1

        if step % args.log_interval == 0:
            elapsed = time.perf_counter() - window_start
            averaged = {key: value / window_count for key, value in metrics_accumulator.items()}
            log_payload = {
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "iterations_per_second": window_count / max(elapsed, 1e-9),
                **averaged,
            }
            print_training_log(log_payload, args.total_iterations, time.perf_counter() - training_start)
            append_jsonl(logs_path, log_payload)
            metrics_accumulator.clear()
            window_count = 0
            window_start = time.perf_counter()

        if args.preview_interval > 0 and step % args.preview_interval == 0:
            save_preview_triptychs(
                model,
                config,
                preview_images,
                previews_dir,
                device,
                step,
                args.seed,
                args.preview_max_side,
            )
            if preview_images:
                print(f"{CONSOLE.paint('[preview]', CONSOLE.bold, CONSOLE.blue)} wrote preview images for step {step} to {previews_dir}")

        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            payload = checkpoint_payload(model, optimizer, scheduler, config, args, step, preview_images)
            save_checkpoint(checkpoints_dir / f"step-{step:07d}.pt", payload)
            save_checkpoint(checkpoints_dir / "latest.pt", payload)
            print(f"{CONSOLE.paint('[checkpoint]', CONSOLE.bold, CONSOLE.magenta)} saved step-{step:07d}.pt and latest.pt in {checkpoints_dir}")

    if window_count > 0:
        elapsed = time.perf_counter() - window_start
        averaged = {key: value / window_count for key, value in metrics_accumulator.items()}
        log_payload = {
            "step": args.total_iterations,
            "lr": optimizer.param_groups[0]["lr"],
            "iterations_per_second": window_count / max(elapsed, 1e-9),
            **averaged,
        }
        print_training_log(log_payload, args.total_iterations, time.perf_counter() - training_start)
        append_jsonl(logs_path, log_payload)

    payload = checkpoint_payload(model, optimizer, scheduler, config, args, args.total_iterations, preview_images)
    save_checkpoint(checkpoints_dir / f"step-{args.total_iterations:07d}.pt", payload)
    save_checkpoint(checkpoints_dir / "latest.pt", payload)
    save_preview_triptychs(
        model,
        config,
        preview_images,
        previews_dir,
        device,
        args.total_iterations,
        args.seed,
        args.preview_max_side,
    )
    print(f"{CONSOLE.paint('[checkpoint]', CONSOLE.bold, CONSOLE.magenta)} saved final checkpoints in {checkpoints_dir}")
    if preview_images:
        print(f"{CONSOLE.paint('[preview]', CONSOLE.bold, CONSOLE.blue)} wrote final preview images to {previews_dir}")
    print_separator()
    print(f"{CONSOLE.paint('[done]', CONSOLE.bold, CONSOLE.green)} training completed in {format_duration(time.perf_counter() - training_start)}")


if __name__ == "__main__":
    main()
