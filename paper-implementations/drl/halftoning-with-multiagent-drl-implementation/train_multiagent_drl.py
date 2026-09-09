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
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from multiagent_drl import (
    MultiAgentDRLConfig,
    PolicyNet,
    anisotropy_loss_from_probs,
    build_optimizer_and_scheduler,
    build_reference_model,
    infer_halftone,
    list_images,
    load_grayscale_image,
    pad_if_needed,
    reference_config,
    paper_reference_config,
    realize_halftone,
    save_triptych,
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
    parser = argparse.ArgumentParser(description="Train the multi-agent DRL halftoning reference extracted from the notebook.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing training images.")
    parser.add_argument("--eval-root", type=Path, default=None, help="Optional root directory for fixed preview images.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Optional newline-delimited file listing training images relative to dataset-root or repo root.")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="Optional newline-delimited file listing preview images relative to eval-root or repo root.")
    parser.add_argument("--run-dir", type=Path, default=Path("paper-implementations/drl/halftoning-with-multiagent-drl-implementation/output/train-run"), help="Directory for checkpoints, previews, and logs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--variant", default="paper", choices=["paper", "tuned"], help="Paper objective or repository-tuned auxiliary objective.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="DataLoader worker count.")
    parser.add_argument("--cache-images", action="store_true", help="Cache decoded grayscale images per worker.")
    parser.add_argument("--total-iterations", type=int, default=200000, help="Total optimization iterations.")
    parser.add_argument("--log-interval", type=int, default=100, help="How often to print and append averaged metrics.")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="How often to save checkpoints.")
    parser.add_argument("--preview-interval", type=int, default=5000, help="How often to save preview triptychs.")
    parser.add_argument("--preview-count", type=int, default=4, help="Number of preview images to track.")
    parser.add_argument("--preview-max-side", type=int, default=256, help="Maximum side length for preview inference. Use 0 to keep original size.")
    parser.add_argument("--gray-probe-levels", default="0.25,0.5,0.6,0.75", help="Comma-separated constant-gray probe levels saved with preview checkpoints.")
    parser.add_argument("--gray-probe-seeds", type=int, default=4, help="Number of noise seeds to render per gray probe level. Use 0 to disable gray probes.")
    parser.add_argument("--gray-probe-size", type=int, default=256, help="Spatial size of each constant-gray probe tile. Use 0 to disable gray probes.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--allow-resume-mismatch", action="store_true", help="Allow resume even if config or total iteration count differs.")
    parser.add_argument("--deterministic", action="store_true", help="Enable stricter deterministic settings.")
    parser.add_argument("--max-train-images", type=int, default=0, help="Optional cap on discovered training images. Use 0 for all.")
    parser.add_argument("--crop-size", type=int, default=None, help="Training crop size. Defaults to the reference configuration.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size. Defaults to the reference configuration.")
    parser.add_argument("--channels", type=int, default=None, help="Model width. Defaults to the reference configuration.")
    parser.add_argument("--num-res-blocks", type=int, default=None, help="Residual block count. Defaults to the reference configuration.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Initial Adam learning rate. Defaults to the reference configuration.")
    parser.add_argument("--learning-rate-min", type=float, default=None, help="Minimum cosine-decay learning rate. Defaults to the reference configuration.")
    parser.add_argument("--ssim-weight", type=float, default=None, help="Reward SSIM weight. Defaults to the reference configuration.")
    parser.add_argument("--anisotropy-weight", type=float, default=None, help="Anisotropy penalty weight. Defaults to the reference configuration.")
    parser.add_argument("--dispersion-weight", type=float, default=None, help="Natural-image dispersion regularizer weight on the soft-thresholded error map.")
    parser.add_argument("--threshold-density-weight", type=float, default=None, help="Soft-threshold density-alignment weight for constant-gray regularization.")
    parser.add_argument("--threshold-temperature", type=float, default=None, help="Temperature for the soft threshold surrogate used in constant-gray regularization.")
    parser.add_argument("--brightness-jitter", type=float, default=0.9, help="Brightness jitter amount in the torchvision ColorJitter sense. Matches the paper default when left unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the setup, save step-0 previews, and exit before training.")
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


def configure_torch_multiprocessing() -> None:
    try:
        if torch.multiprocessing.get_sharing_strategy() != "file_system":
            torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError):
        pass


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def apply_brightness_jitter(batch: Tensor, amount: float) -> Tensor:
    if amount <= 0:
        return batch
    minimum = max(0.0, 1.0 - amount)
    maximum = 1.0 + amount
    factors = torch.empty(batch.shape[0], 1, 1, 1, device=batch.device, dtype=batch.dtype).uniform_(minimum, maximum)
    return (batch * factors).clamp(0.0, 1.0)


def sample_constant_gray_batch(
    reference: Tensor,
    config: "MultiAgentDRLConfig",
    minimum: float | None = None,
    maximum: float | None = None,
) -> Tensor:
    if minimum is None or maximum is None:
        """
            The paper's anisotropy loss (Eq. 11) is defined over c_g ~ U(0,1)
            i.e. the full unit interval, not a restricted one.
            The [0.25, 0.75] range below is a specific choice for
            the `tuned` variant only; it must not silently apply to the
            `paper` variant, which is supposed to match the paper exactly.
        """
        if config.variant == "paper":
            minimum, maximum = 0.0, 1.0
        else:
            minimum, maximum = 0.25, 0.75
    gray_levels = torch.empty(reference.shape[0], 1, 1, 1, device=reference.device, dtype=reference.dtype).uniform_(minimum, maximum)
    return gray_levels.expand_as(reference)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_gray_probe_levels(raw: str) -> List[float]:
    levels: List[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Gray probe level must be in [0, 1], got {value}")
        levels.append(value)
    return levels


def _probe_radial_indices(height: int, width: int, device: torch.device) -> Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    cy = height // 2
    cx = width // 2
    return torch.round(torch.sqrt((yy - cy).square() + (xx - cx).square())).to(dtype=torch.long).reshape(-1)


def low_frequency_ring_energy(image: Tensor, max_radius: int = 4) -> float:
    power = torch.abs(torch.fft.fftshift(torch.fft.fft2(image.squeeze(1), norm="ortho"), dim=(-2, -1))) ** 2
    batch, height, width = power.shape
    flat_power = power.reshape(batch, height * width)
    flat_radii = _probe_radial_indices(height, width, image.device)
    total = image.new_tensor(0.0)
    for radius in range(1, min(max_radius, int(flat_radii.max().item())) + 1):
        mask = flat_radii == radius
        if int(mask.sum().item()) == 0:
            continue
        total = total + flat_power[:, mask].mean(dim=1).mean()
    return float(total.detach().item())


def tensor_to_pil_image(array_2d: Tensor) -> Image.Image:
    image = array_2d.detach().cpu().clamp(0.0, 1.0).numpy()
    return Image.fromarray(np.uint8(np.round(image * 255.0)), mode="L").convert("RGB")


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
    print(CONSOLE.paint("=" * 96, CONSOLE.dim))


def print_worker_fallback_warning(requested_workers: int) -> None:
    print(
        f"{CONSOLE.paint('[warn]', CONSOLE.bold, CONSOLE.yellow)} "
        f"DataLoader worker transport failed; falling back to num_workers=0 "
        f"(requested {requested_workers})."
    )


def print_run_header(
    args: argparse.Namespace,
    config: MultiAgentDRLConfig,
    device: torch.device,
    image_count: int,
    preview_count: int,
) -> None:
    print_separator()
    print(CONSOLE.paint("Multi-Agent DRL Halftoning Training", CONSOLE.bold, CONSOLE.blue))
    print(
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
        f"{CONSOLE.paint('channels', CONSOLE.cyan)}={config.channels} | "
        f"{CONSOLE.paint('res_blocks', CONSOLE.cyan)}={config.num_res_blocks} | "
        f"{CONSOLE.paint('workers', CONSOLE.cyan)}={args.num_workers} | "
        f"{CONSOLE.paint('cache_images', CONSOLE.cyan)}={args.cache_images}"
    )
    print(
        f"{CONSOLE.paint('log_every', CONSOLE.cyan)}={args.log_interval} | "
        f"{CONSOLE.paint('ckpt_every', CONSOLE.cyan)}={args.checkpoint_interval} | "
        f"{CONSOLE.paint('preview_every', CONSOLE.cyan)}={args.preview_interval} | "
        f"{CONSOLE.paint('brightness_jitter', CONSOLE.cyan)}={args.brightness_jitter}"
    )
    print(
        f"{CONSOLE.paint('objective', CONSOLE.cyan)} "
        f"ssim={config.ssim_weight:.3e} | "
        f"disp={config.dispersion_weight:.3e} | "
        f"anis={config.anisotropy_weight:.3e} | "
        f"thr_dens={config.threshold_density_weight:.3e} | "
        f"thr_temp={config.threshold_temperature:.3e}"
    )
    if args.gray_probe_size > 0 and args.gray_probe_seeds > 0:
        print(
            f"{CONSOLE.paint('gray_probe', CONSOLE.cyan)} "
            f"levels={args.gray_probe_levels} | "
            f"seeds={args.gray_probe_seeds} | "
            f"size={args.gray_probe_size}"
        )
    print_separator()


def print_training_log(log_payload: Dict[str, float], total_iterations: int, elapsed_total: float) -> None:
    step = int(log_payload["step"])
    progress = 100.0 * step / max(1, total_iterations)
    iterations_per_second = float(log_payload["iterations_per_second"])
    remaining_steps = max(0, total_iterations - step)
    eta_seconds = remaining_steps / max(iterations_per_second, 1e-9)
    bar = format_progress_bar(step, total_iterations)
    print(
        f"{CONSOLE.paint('[train]', CONSOLE.bold, CONSOLE.green)} "
        f"{CONSOLE.paint('step', CONSOLE.gray)} {CONSOLE.paint(f'{step:7d}/{total_iterations}', CONSOLE.bold)} "
        f"{CONSOLE.paint('[' + bar + ']', CONSOLE.yellow)} "
        f"{CONSOLE.paint(f'{progress:6.2f}%', CONSOLE.bold, CONSOLE.yellow)} | "
        f"{CONSOLE.paint(f'{iterations_per_second:5.2f} it/s', CONSOLE.magenta)} | "
        f"{CONSOLE.paint('elapsed', CONSOLE.gray)} {format_duration(elapsed_total)} | "
        f"{CONSOLE.paint('eta', CONSOLE.gray)} {format_duration(eta_seconds)} | "
        f"{CONSOLE.paint('lr', CONSOLE.gray)} {float(log_payload['lr']):.3e}"
    )
    print(
        "        "
        f"{CONSOLE.paint('total_loss', CONSOLE.cyan)} {float(log_payload['total_loss']):10.6f} | "
        f"{CONSOLE.paint('policy_loss', CONSOLE.cyan)} {float(log_payload['policy_loss']):10.6f} | "
        f"{CONSOLE.paint('reg_loss', CONSOLE.cyan)} {float(log_payload['regularization_loss']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('dispersion', CONSOLE.cyan)} {float(log_payload['dispersion_loss']):10.6f} | "
        f"{CONSOLE.paint('disp_term', CONSOLE.cyan)} {float(log_payload['dispersion_term']):10.6f} | "
        f"{CONSOLE.paint('disp_lf', CONSOLE.cyan)} {float(log_payload['dispersion_lowfreq_loss']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('reward', CONSOLE.cyan)} {float(log_payload['reward']):10.6f} | "
        f"{CONSOLE.paint('tone_error', CONSOLE.cyan)} {float(log_payload['tone_error']):10.6f} | "
        f"{CONSOLE.paint('ssim', CONSOLE.cyan)} {float(log_payload['ssim']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('anisotropy', CONSOLE.cyan)} {float(log_payload['anisotropy_loss']):10.6f} | "
        f"{CONSOLE.paint('anis_term', CONSOLE.cyan)} {float(log_payload['anisotropy_term']):10.6f} | "
        f"{CONSOLE.paint('thr_dens', CONSOLE.cyan)} {float(log_payload['threshold_density_loss']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('disp_w', CONSOLE.cyan)} {float(log_payload['dispersion_weight']):10.6f} | "
        f"{CONSOLE.paint('aniso_w', CONSOLE.cyan)} {float(log_payload['anisotropy_weight']):10.6f} | "
        f"{CONSOLE.paint('thr_w', CONSOLE.cyan)} {float(log_payload['threshold_density_weight']):10.6f} | "
        f"{CONSOLE.paint('gray_target', CONSOLE.cyan)} {float(log_payload['gray_target_mean']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('prob_mean', CONSOLE.cyan)} {float(log_payload['prob_mean']):10.6f} | "
        f"{CONSOLE.paint('prob_std', CONSOLE.cyan)} {float(log_payload['prob_std']):10.6f}"
    )
    print(
        "        "
        f"{CONSOLE.paint('gray_prob_mean', CONSOLE.cyan)} {float(log_payload['gray_prob_mean']):10.6f} | "
        f"{CONSOLE.paint('gray_prob_std', CONSOLE.cyan)} {float(log_payload['gray_prob_std']):10.6f} | "
        f"{CONSOLE.paint('gray_soft_mean', CONSOLE.cyan)} {float(log_payload['gray_soft_threshold_mean']):10.6f}"
    )


def checkpoint_payload(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    config: MultiAgentDRLConfig,
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
    config: MultiAgentDRLConfig,
    args: argparse.Namespace,
) -> None:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != asdict(config):
        raise RuntimeError("Checkpoint config does not match the current run config.")
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_total_iterations = checkpoint_args.get("total_iterations")
    if checkpoint_total_iterations is not None and int(checkpoint_total_iterations) != args.total_iterations:
        raise RuntimeError("Checkpoint total_iterations does not match the current run.")


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


def save_preview_triptychs(
    model: PolicyNet,
    config: MultiAgentDRLConfig,
    preview_images: List[Path],
    output_dir: Path,
    step: int,
    device: torch.device,
    preview_max_side: int,
    seed: int,
) -> None:
    if not preview_images:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(preview_images):
        contone = load_grayscale_image(image_path).unsqueeze(0).to(device)
        contone = resize_for_preview(contone, preview_max_side)
        noise_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
        noise_generator.manual_seed(seed + step * 1009 + index)
        noise = torch.randn(contone.shape, generator=noise_generator, device=device, dtype=contone.dtype)
        result = infer_halftone(model, contone, config, noise=noise, binary_mode="threshold")
        save_triptych(result, output_dir / f"step-{step:07d}-{image_path.stem.replace(' ', '-')}.png")


def save_gray_probe_sheet(
    model: PolicyNet,
    config: MultiAgentDRLConfig,
    output_dir: Path,
    metrics_path: Path,
    step: int,
    device: torch.device,
    levels: List[float],
    probe_size: int,
    probe_seeds: int,
    seed: int,
) -> None:
    if probe_size <= 0 or probe_seeds <= 0 or not levels:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_count = 3 + probe_seeds
    canvas = Image.new("RGB", (panel_count * probe_size, len(levels) * probe_size), color="white")
    payload: Dict[str, object] = {"step": step, "levels": []}

    for level_index, level in enumerate(levels):
        contone = torch.full((1, 1, probe_size, probe_size), level, device=device)
        row_panels = [tensor_to_pil_image(contone[0, 0])]
        level_payload: Dict[str, object] = {"level": level, "seed_metrics": []}

        for seed_index in range(probe_seeds):
            generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
            generator.manual_seed(seed + step * 1009 + level_index * 10007 + seed_index)
            noise = torch.randn(contone.shape, generator=generator, device=device, dtype=contone.dtype)
            result = infer_halftone(model, contone, config, noise=noise, binary_mode="threshold")
            probability = result["probability"]
            halftone = result["halftone"]

            sample_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
            sample_generator.manual_seed(seed + step * 2027 + level_index * 20011 + seed_index)
            sampled_halftone = realize_halftone(probability, config.white_threshold, mode="sample", generator=sample_generator)

            if seed_index == 0:
                row_panels.append(tensor_to_pil_image(probability[0, 0]))
                row_panels.append(tensor_to_pil_image(halftone[0, 0]))
            row_panels.append(tensor_to_pil_image(sampled_halftone[0, 0]))

            level_payload["seed_metrics"].append(
                {
                    "seed": seed_index,
                    "prob_mean": float(probability.mean().detach().item()),
                    "prob_std": float(probability.std(unbiased=False).detach().item()),
                    "prob_anisotropy": float(anisotropy_loss_from_probs(probability).detach().item()),
                    "argmax_mean": float(halftone.mean().detach().item()),
                    "argmax_anisotropy": float(anisotropy_loss_from_probs(halftone).detach().item()),
                    "argmax_lowfreq_r4": low_frequency_ring_energy(halftone, max_radius=4),
                    "sample_mean": float(sampled_halftone.mean().detach().item()),
                    "sample_anisotropy": float(anisotropy_loss_from_probs(sampled_halftone).detach().item()),
                    "sample_lowfreq_r4": low_frequency_ring_energy(sampled_halftone, max_radius=4),
                }
            )

        for panel_index, panel in enumerate(row_panels):
            canvas.paste(panel, (panel_index * probe_size, level_index * probe_size))

        seed_metrics = level_payload["seed_metrics"]
        level_payload["mean_prob_mean"] = float(np.mean([entry["prob_mean"] for entry in seed_metrics]))
        level_payload["mean_prob_std"] = float(np.mean([entry["prob_std"] for entry in seed_metrics]))
        level_payload["mean_argmax_mean"] = float(np.mean([entry["argmax_mean"] for entry in seed_metrics]))
        level_payload["mean_prob_anisotropy"] = float(np.mean([entry["prob_anisotropy"] for entry in seed_metrics]))
        level_payload["mean_argmax_anisotropy"] = float(np.mean([entry["argmax_anisotropy"] for entry in seed_metrics]))
        level_payload["mean_argmax_lowfreq_r4"] = float(np.mean([entry["argmax_lowfreq_r4"] for entry in seed_metrics]))
        level_payload["mean_sample_mean"] = float(np.mean([entry["sample_mean"] for entry in seed_metrics]))
        level_payload["mean_sample_anisotropy"] = float(np.mean([entry["sample_anisotropy"] for entry in seed_metrics]))
        level_payload["mean_sample_lowfreq_r4"] = float(np.mean([entry["sample_lowfreq_r4"] for entry in seed_metrics]))
        payload["levels"].append(level_payload)

    canvas.save(output_dir / f"step-{step:07d}.png")
    append_jsonl(metrics_path, payload)


def build_dataloader(
    image_paths: List[Path],
    config: MultiAgentDRLConfig,
    args: argparse.Namespace,
    start_step: int,
    device: torch.device,
    num_workers: int | None = None,
) -> DataLoader[Tensor]:
    remaining_iterations = args.total_iterations - start_step
    worker_count = args.num_workers if num_workers is None else num_workers
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
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
        "persistent_workers": worker_count > 0,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


def is_worker_transport_error(exc: BaseException) -> bool:
    message = str(exc)
    markers = (
        "torch_shm_manager",
        "DataLoader worker process",
        "Operation not permitted",
        "Broken pipe",
        "resource_sharer",
    )
    return any(marker in message for marker in markers)


def build_config(args: argparse.Namespace) -> MultiAgentDRLConfig:
    base = paper_reference_config() if args.variant == "paper" else reference_config()
    return MultiAgentDRLConfig(
        variant=base.variant,
        crop_size=args.crop_size if args.crop_size is not None else base.crop_size,
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        channels=args.channels if args.channels is not None else base.channels,
        num_res_blocks=args.num_res_blocks if args.num_res_blocks is not None else base.num_res_blocks,
        hvs_kernel_size=base.hvs_kernel_size,
        hvs_scale_factor=base.hvs_scale_factor,
        hvs_luminance=base.hvs_luminance,
        ssim_kernel_size=base.ssim_kernel_size,
        ssim_sigma=base.ssim_sigma,
        ssim_weight=args.ssim_weight if args.ssim_weight is not None else base.ssim_weight,
        dispersion_weight=args.dispersion_weight if args.dispersion_weight is not None else base.dispersion_weight,
        anisotropy_weight=args.anisotropy_weight if args.anisotropy_weight is not None else base.anisotropy_weight,
        threshold_density_weight=(
            args.threshold_density_weight if args.threshold_density_weight is not None else base.threshold_density_weight
        ),
        dispersion_lowfreq_radius=base.dispersion_lowfreq_radius,
        threshold_temperature=(
            args.threshold_temperature if args.threshold_temperature is not None else base.threshold_temperature
        ),
        learning_rate=args.learning_rate if args.learning_rate is not None else base.learning_rate,
        learning_rate_min=args.learning_rate_min if args.learning_rate_min is not None else base.learning_rate_min,
        white_threshold=base.white_threshold,
        init_std=base.init_std,
        eps=base.eps,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    device = resolve_device(args.device)
    gray_probe_levels = parse_gray_probe_levels(args.gray_probe_levels)
    configure_torch_multiprocessing()
    set_global_seed(args.seed, args.deterministic)

    image_paths = load_manifest(args.dataset_manifest, args.dataset_root) if args.dataset_manifest is not None else list_images(args.dataset_root)
    if args.max_train_images > 0:
        image_paths = image_paths[: args.max_train_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {args.dataset_root}")

    run_dir = args.run_dir
    checkpoints_dir = run_dir / "checkpoints"
    previews_dir = run_dir / "previews"
    gray_probe_dir = run_dir / "gray_probes"
    gray_probe_metrics_path = run_dir / "gray_probe_metrics.jsonl"
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
            "gray_probe_levels": gray_probe_levels,
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

    active_num_workers = args.num_workers
    dataloader = build_dataloader(image_paths, config, args, start_step, device, num_workers=active_num_workers)
    data_iter = iter(dataloader)

    if start_step == 0 and preview_images:
        save_preview_triptychs(model, config, preview_images, previews_dir, 0, device, args.preview_max_side, args.seed)
    if start_step == 0:
        save_gray_probe_sheet(
            model,
            config,
            gray_probe_dir,
            gray_probe_metrics_path,
            0,
            device,
            gray_probe_levels,
            args.gray_probe_size,
            args.gray_probe_seeds,
            args.seed,
        )

    if args.dry_run:
        try:
            batch = next(data_iter).to(device)
        except RuntimeError as exc:
            if active_num_workers > 0 and is_worker_transport_error(exc):
                print_worker_fallback_warning(active_num_workers)
                active_num_workers = 0
                dataloader = build_dataloader(image_paths, config, args, start_step, device, num_workers=active_num_workers)
                data_iter = iter(dataloader)
                batch = next(data_iter).to(device)
            else:
                raise
        except StopIteration as exc:
            raise RuntimeError("Dry-run failed to retrieve a batch.") from exc
        batch = apply_brightness_jitter(batch, args.brightness_jitter)
        with torch.no_grad():
            probs = model(batch, torch.randn_like(batch)).clamp(config.eps, 1.0 - config.eps)
            gray = sample_constant_gray_batch(batch, config)
            gray_probs = model(gray, torch.randn_like(gray)).clamp(config.eps, 1.0 - config.eps)
        print(
            f"{CONSOLE.paint('[dry-run]', CONSOLE.bold, CONSOLE.green)} "
            f"batch={tuple(batch.shape)} | "
            f"prob_mean={float(probs.mean()):.6f} | prob_std={float(probs.std(unbiased=False)):.6f} | "
            f"gray_target_mean={float(gray.mean()):.6f} | "
            f"gray_prob_mean={float(gray_probs.mean()):.6f} | gray_prob_std={float(gray_probs.std(unbiased=False)):.6f}"
        )
        if preview_images:
            print(f"{CONSOLE.paint('[preview]', CONSOLE.bold, CONSOLE.blue)} wrote step-0 previews to {previews_dir}")
        if args.gray_probe_size > 0 and args.gray_probe_seeds > 0 and gray_probe_levels:
            print(f"{CONSOLE.paint('[gray]', CONSOLE.bold, CONSOLE.yellow)} wrote step-0 gray probes to {gray_probe_dir}")
        print(f"{CONSOLE.paint('[ready]', CONSOLE.bold, CONSOLE.magenta)} setup validated; exiting before training.")
        return

    metrics_accumulator: Dict[str, float] = {}
    window_count = 0
    window_start = time.perf_counter()
    training_start = time.perf_counter()

    for step in range(start_step + 1, args.total_iterations + 1):
        try:
            contone = next(data_iter)
        except RuntimeError as exc:
            if active_num_workers > 0 and is_worker_transport_error(exc):
                print_worker_fallback_warning(active_num_workers)
                active_num_workers = 0
                dataloader = build_dataloader(
                    image_paths,
                    config,
                    args,
                    step - 1,
                    device,
                    num_workers=active_num_workers,
                )
                data_iter = iter(dataloader)
                contone = next(data_iter)
            else:
                raise
        except StopIteration as exc:
            raise RuntimeError("Training DataLoader ended unexpectedly.") from exc
        contone = contone.to(device)
        contone = apply_brightness_jitter(contone, args.brightness_jitter)
        gray = sample_constant_gray_batch(contone, config)
        metrics = train_step(model, optimizer, contone, gray, config, step=step)
        scheduler.step()
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        for key, value in metrics.items():
            metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + float(value)
        window_count += 1

        if step % args.log_interval == 0 or step == args.total_iterations:
            elapsed_window = time.perf_counter() - window_start
            elapsed_total = time.perf_counter() - training_start
            log_payload = {key: value / window_count for key, value in metrics_accumulator.items()}
            log_payload["step"] = step
            log_payload["iterations_per_second"] = window_count / max(elapsed_window, 1e-9)
            append_jsonl(logs_path, log_payload)
            print_training_log(log_payload, args.total_iterations, elapsed_total)
            metrics_accumulator = {}
            window_count = 0
            window_start = time.perf_counter()

        if args.preview_interval > 0 and step % args.preview_interval == 0:
            save_preview_triptychs(model, config, preview_images, previews_dir, step, device, args.preview_max_side, args.seed)
            save_gray_probe_sheet(
                model,
                config,
                gray_probe_dir,
                gray_probe_metrics_path,
                step,
                device,
                gray_probe_levels,
                args.gray_probe_size,
                args.gray_probe_seeds,
                args.seed,
            )
            if preview_images:
                print(f"{CONSOLE.paint('[preview]', CONSOLE.bold, CONSOLE.blue)} wrote preview images for step {step} to {previews_dir}")
            if args.gray_probe_size > 0 and args.gray_probe_seeds > 0 and gray_probe_levels:
                print(f"{CONSOLE.paint('[gray]', CONSOLE.bold, CONSOLE.yellow)} wrote gray probes for step {step} to {gray_probe_dir}")

        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            payload = checkpoint_payload(model, optimizer, scheduler, config, args, step, preview_images)
            save_checkpoint(checkpoints_dir / f"step-{step:07d}.pt", payload)
            save_checkpoint(checkpoints_dir / "latest.pt", payload)
            print(f"{CONSOLE.paint('[checkpoint]', CONSOLE.bold, CONSOLE.magenta)} saved step-{step:07d}.pt and latest.pt in {checkpoints_dir}")

    payload = checkpoint_payload(model, optimizer, scheduler, config, args, args.total_iterations, preview_images)
    save_checkpoint(checkpoints_dir / f"step-{args.total_iterations:07d}.pt", payload)
    save_checkpoint(checkpoints_dir / "latest.pt", payload)
    save_preview_triptychs(model, config, preview_images, previews_dir, args.total_iterations, device, args.preview_max_side, args.seed)
    save_gray_probe_sheet(
        model,
        config,
        gray_probe_dir,
        gray_probe_metrics_path,
        args.total_iterations,
        device,
        gray_probe_levels,
        args.gray_probe_size,
        args.gray_probe_seeds,
        args.seed,
    )
    print(f"{CONSOLE.paint('[checkpoint]', CONSOLE.bold, CONSOLE.magenta)} saved final checkpoints in {checkpoints_dir}")
    if preview_images:
        print(f"{CONSOLE.paint('[preview]', CONSOLE.bold, CONSOLE.blue)} wrote final preview images to {previews_dir}")
    if args.gray_probe_size > 0 and args.gray_probe_seeds > 0 and gray_probe_levels:
        print(f"{CONSOLE.paint('[gray]', CONSOLE.bold, CONSOLE.yellow)} wrote final gray probes to {gray_probe_dir}")
    print(f"{CONSOLE.paint('[done]', CONSOLE.bold, CONSOLE.green)} training complete in {format_duration(time.perf_counter() - training_start)}")


if __name__ == "__main__":
    main()
