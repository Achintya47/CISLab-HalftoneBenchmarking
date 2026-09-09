from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F

DEFAULT_HPARAMS_PATH = Path(__file__).with_name("default_hparams.json")


@dataclass(frozen=True)
class MultiAgentDRLConfig:
    variant: str = "tuned"
    crop_size: int = 64
    batch_size: int = 64
    channels: int = 32
    num_res_blocks: int = 16
    hvs_kernel_size: int = 11
    hvs_scale_factor: float = 2000.0
    hvs_luminance: float = 11.0
    ssim_kernel_size: int = 11
    ssim_sigma: float = 1.5
    ssim_weight: float = 0.006
    anisotropy_weight: float = 2e-3
    learning_rate: float = 3e-4
    learning_rate_min: float = 1e-5
    white_threshold: float = 0.5
    init_std: float = 0.01
    dispersion_weight: float = 1e-3
    dispersion_lowfreq_radius: int = 4
    threshold_density_weight: float = 0.01
    threshold_temperature: float = 0.05
    eps: float = 1e-6


def load_default_hparams(path: Path = DEFAULT_HPARAMS_PATH) -> Dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Default hyperparameter file must contain a JSON object: {path}")
    valid_keys = set(MultiAgentDRLConfig.__dataclass_fields__.keys())
    unknown_keys = sorted(set(payload.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown hyperparameter keys in {path}: {', '.join(unknown_keys)}")
    return {key: value for key, value in payload.items() if value is not None}


def reference_config() -> MultiAgentDRLConfig:
    base = MultiAgentDRLConfig()
    overrides = load_default_hparams()
    if not overrides:
        return base
    merged = asdict(base)
    merged.update(overrides)
    return MultiAgentDRLConfig(**merged)


def paper_reference_config() -> MultiAgentDRLConfig:
    """Configuration restricted to the objective and weights printed in the paper."""
    return MultiAgentDRLConfig(
        variant="paper",
        ssim_weight=0.006,
        anisotropy_weight=0.002,
        dispersion_weight=0.0,
        threshold_density_weight=0.0,
    )


def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts]
    return sorted(paths)


def load_grayscale_image(path: Path) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def pad_if_needed(image: Tensor, crop_size: int) -> Tensor:
    _, height, width = image.shape
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h == 0 and pad_w == 0:
        return image
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(image, (left, right, top, bottom), mode="reflect")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.conv1(x), inplace=False)
        x = self.conv2(x)
        return F.relu(x + residual, inplace=False)


class PolicyNet(nn.Module):
    def __init__(self, config: MultiAgentDRLConfig) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(2, config.channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(config.channels) for _ in range(config.num_res_blocks)])
        self.output_conv = nn.Conv2d(config.channels, 1, kernel_size=1)

    def forward(self, contone: Tensor, noise: Tensor) -> Tensor:
        x = torch.cat([contone, noise], dim=1)
        x = F.relu(self.input_conv(x), inplace=False)
        x = self.res_blocks(x)
        return torch.sigmoid(self.output_conv(x))


def initialize_policy(model: nn.Module, std: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_reference_model(config: MultiAgentDRLConfig, device: torch.device | None = None) -> PolicyNet:
    model = PolicyNet(config)
    initialize_policy(model, std=config.init_std)
    if device is not None:
        model = model.to(device)
    return model


def build_optimizer_and_scheduler(
    model: PolicyNet,
    total_iterations: int,
    config: MultiAgentDRLConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iterations),
        eta_min=config.learning_rate_min,
    )
    return optimizer, scheduler


@lru_cache(maxsize=None)
def _gaussian_kernel(kernel_size: int, sigma: float) -> Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


@lru_cache(maxsize=None)
def _nasanen_kernel(kernel_size: int, scale_factor: float, luminance: float) -> Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    c = 0.525
    d = 3.91
    luminance = max(float(luminance), 1e-6)
    scale_factor = max(float(scale_factor), 1e-6)
    k = (math.pi * scale_factor) / (180.0 * (c * math.log(luminance) + d))
    spatial_radius = (2.0 * math.pi / scale_factor) * radius
    kernel = torch.pow(k * k + spatial_radius.square(), -1.5)
    kernel = kernel / kernel.sum()
    return kernel


def hvs_filter(image: Tensor, config: MultiAgentDRLConfig) -> Tensor:
    kernel = _nasanen_kernel(config.hvs_kernel_size, config.hvs_scale_factor, config.hvs_luminance)
    kernel_size = config.hvs_kernel_size
    kernel = kernel.to(device=image.device, dtype=image.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(image.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])


def _ssim_stats(x: Tensor, y: Tensor, kernel_size: int, sigma: float, eps: float) -> Dict[str, Tensor]:
    kernel = _gaussian_kernel(kernel_size, sigma).to(device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.conv2d(x, kernel, padding=padding)
    mu_y = F.conv2d(y, kernel, padding=padding)
    mean_sq_x = F.conv2d(x * x, kernel, padding=padding)
    sigma_y = F.conv2d(y * y, kernel, padding=padding) - mu_y.square()
    sigma_x = torch.clamp(mean_sq_x - mu_x.square(), min=0.0)
    mean_xy = F.conv2d(x * y, kernel, padding=padding)
    sigma_xy = mean_xy - mu_x * mu_y
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2).clamp_min(eps)
    )
    return {
        "kernel": kernel,
        "padding": torch.tensor(padding, device=x.device),
        "c1": torch.tensor(c1, device=x.device, dtype=x.dtype),
        "c2": torch.tensor(c2, device=x.device, dtype=x.dtype),
        "mu_x": mu_x,
        "mu_y": mu_y,
        "mean_sq_x": mean_sq_x,
        "sigma_y": sigma_y,
        "mean_xy": mean_xy,
        "ssim_map": ssim_map,
    }


def compute_ssim(x: Tensor, y: Tensor, config: MultiAgentDRLConfig) -> Tensor:
    stats = _ssim_stats(x, y, config.ssim_kernel_size, config.ssim_sigma, config.eps)
    return stats["ssim_map"].mean(dim=(1, 2, 3))


def compute_reward(sample: Tensor, contone: Tensor, config: MultiAgentDRLConfig) -> tuple[Tensor, Tensor, Tensor]:
    filtered_sample = hvs_filter(sample, config)
    filtered_contone = hvs_filter(contone, config)
    tone_error = (filtered_sample - filtered_contone).square().mean(dim=(1, 2, 3))
    ssim = compute_ssim(sample, contone, config)
    reward = -(tone_error - config.ssim_weight * ssim)
    return reward, tone_error, ssim


@lru_cache(maxsize=None)
def _radial_indices(height: int, width: int) -> Tensor:
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    # `fftshift` places the DC component at the integer midpoint for even sizes.
    cy = height // 2
    cx = width // 2
    radii = torch.round(torch.sqrt((yy - cy).square() + (xx - cx).square())).to(dtype=torch.long)
    return radii.reshape(-1)


def anisotropy_loss_from_probs(probs: Tensor) -> Tensor:
    power = torch.abs(torch.fft.fftshift(torch.fft.fft2(probs.squeeze(1), norm="ortho"), dim=(-2, -1))) ** 2
    batch, height, width = power.shape
    flat_power = power.reshape(batch, height * width)
    flat_radii = _radial_indices(height, width).to(device=probs.device)
    losses: List[Tensor] = []
    max_radius = int(flat_radii.max().item())
    for radius in range(1, max_radius + 1):
        mask = flat_radii == radius
        count = int(mask.sum().item())
        if count <= 1:
            continue
        ring = flat_power[:, mask]
        ring_mean = ring.mean(dim=1, keepdim=True)
        losses.append((ring - ring_mean).square().mean(dim=1))
    if not losses:
        return probs.new_tensor(0.0)
    return torch.stack(losses, dim=1).mean()


def density_loss_from_probs(probs: Tensor, target: Tensor) -> Tensor:
    target_density = target.mean(dim=(1, 2, 3))
    prob_density = probs.mean(dim=(1, 2, 3))
    return (prob_density - target_density).square().mean()


def low_frequency_ratio_loss(signal: Tensor, max_radius: int, eps: float) -> Tensor:
    power = torch.abs(torch.fft.fftshift(torch.fft.fft2(signal.squeeze(1), norm="ortho"), dim=(-2, -1))) ** 2
    batch, height, width = power.shape
    flat_power = power.reshape(batch, height * width)
    flat_radii = _radial_indices(height, width).to(device=signal.device)
    non_dc_mask = flat_radii > 0
    lowfreq_mask = (flat_radii > 0) & (flat_radii <= max_radius)
    if int(lowfreq_mask.sum().item()) == 0 or int(non_dc_mask.sum().item()) == 0:
        return signal.new_tensor(0.0)
    lowfreq_energy = flat_power[:, lowfreq_mask].sum(dim=1)
    total_energy = flat_power[:, non_dc_mask].sum(dim=1).clamp_min(eps)
    return (lowfreq_energy / total_energy).mean()


def smooth_threshold_map(
    probabilities: Tensor,
    white_threshold: float,
    temperature: float,
    eps: float,
) -> Tensor:
    safe_temperature = max(float(temperature), float(eps))
    logits = (probabilities - white_threshold) / safe_temperature
    return torch.sigmoid(logits)


def binarization_loss_from_probs(probs: Tensor) -> Tensor:
    return (probs * (1.0 - probs)).mean()


def binary_entropy_from_probs(probs: Tensor, eps: float) -> Tensor:
    safe_probs = probs.clamp(eps, 1.0 - eps)
    entropy = -(safe_probs * safe_probs.log() + (1.0 - safe_probs) * (1.0 - safe_probs).log())
    return entropy.mean()


@lru_cache(maxsize=None)
def _best_candidate_blue_noise_mask(size: int = 64, candidates: int = 24, seed: int = 7) -> Tensor:
    rng = np.random.default_rng(seed)
    available = [(x, y) for y in range(size) for x in range(size)]
    chosen: List[tuple[int, int]] = []
    ranks = np.zeros((size, size), dtype=np.float32)

    def toroidal_distance_sq(a: tuple[int, int], b: tuple[int, int]) -> int:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dx = min(dx, size - dx)
        dy = min(dy, size - dy)
        return dx * dx + dy * dy

    for rank in range(size * size):
        if not chosen:
            best_index = int(rng.integers(0, len(available)))
            best = available.pop(best_index)
        else:
            cand_count = min(candidates, len(available))
            cand_indices = rng.choice(len(available), size=cand_count, replace=False)
            best_index = None
            best_score = -1
            for index in cand_indices.tolist():
                candidate = available[index]
                score = min(toroidal_distance_sq(candidate, point) for point in chosen)
                if score > best_score:
                    best_score = score
                    best_index = index
            assert best_index is not None
            best = available.pop(best_index)
        chosen.append(best)
        x, y = best
        ranks[y, x] = (rank + 0.5) / float(size * size)
    return torch.from_numpy(ranks)


def _tile_blue_noise_mask(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    base = _best_candidate_blue_noise_mask().to(device=device, dtype=dtype)
    tile_h, tile_w = base.shape
    reps_h = (height + tile_h - 1) // tile_h
    reps_w = (width + tile_w - 1) // tile_w
    tiled = base.repeat(reps_h, reps_w)
    return tiled[:height, :width].unsqueeze(0).unsqueeze(0)


def _smoothed_probability_target(probabilities: Tensor) -> Tensor:
    kernel = _gaussian_kernel(5, 1.0).to(device=probabilities.device, dtype=probabilities.dtype).view(1, 1, 5, 5)
    blurred = F.conv2d(probabilities, kernel, padding=2)
    return blurred


def _posthoc_blue_noise_realize(
    probabilities: Tensor,
    contone: Tensor,
    config: MultiAgentDRLConfig,
) -> Tensor:
    # The learned probability field is good at indicating where structure matters,
    # but poor as a direct binary tone target. Use the contone for tone and let
    # the MARL residual act only as a weak structural prior.
    blurred = _smoothed_probability_target(probabilities)
    detail_prior = probabilities - blurred
    target = (contone + 0.10 * detail_prior).clamp(config.eps, 1.0 - config.eps)
    height, width = target.shape[-2:]
    mask = _tile_blue_noise_mask(height, width, target.device, target.dtype)
    return (target >= mask).to(dtype=target.dtype)


def realize_halftone(
    probabilities: Tensor,
    white_threshold: float,
    mode: str = "sample",
    generator: torch.Generator | None = None,
    *,
    contone: Tensor | None = None,
    config: MultiAgentDRLConfig | None = None,
) -> Tensor:
    if mode == "sample":
        samples = torch.rand(
            probabilities.shape,
            generator=generator,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        return (samples < probabilities).to(dtype=probabilities.dtype)
    if mode == "threshold":
        return (probabilities >= white_threshold).to(dtype=probabilities.dtype)
    if mode == "posthoc_blue_noise":
        if contone is None or config is None:
            raise ValueError("posthoc_blue_noise realization requires contone and config")
        return _posthoc_blue_noise_realize(probabilities, contone, config)
    raise ValueError(f"Unsupported halftone realization mode: {mode}")


def infer_halftone(
    model: PolicyNet,
    contone: Tensor,
    config: MultiAgentDRLConfig,
    noise: Tensor | None = None,
    *,
    binary_mode: str = "threshold",
    sample_generator: torch.Generator | None = None,
) -> Dict[str, Tensor]:
    model.eval()
    with torch.no_grad():
        if noise is None:
            noise = torch.randn_like(contone)
        probs = model(contone, noise).clamp(config.eps, 1.0 - config.eps)
        halftone = realize_halftone(
            probs,
            config.white_threshold,
            mode=binary_mode,
            generator=sample_generator,
            contone=contone,
            config=config,
        )
    return {"contone": contone, "probability": probs, "halftone": halftone}


def _shift_with_zeros(tensor: Tensor, shift_y: int, shift_x: int) -> Tensor:
    batch, channels, height, width = tensor.shape
    result = torch.zeros_like(tensor)

    src_y0 = max(0, shift_y)
    src_y1 = min(height, height + shift_y)
    dst_y0 = max(0, -shift_y)
    dst_y1 = min(height, height - shift_y)

    src_x0 = max(0, shift_x)
    src_x1 = min(width, width + shift_x)
    dst_x0 = max(0, -shift_x)
    dst_x1 = min(width, width - shift_x)

    if src_y0 < src_y1 and src_x0 < src_x1:
        result[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[:, :, src_y0:src_y1, src_x0:src_x1]
    return result


def _counterfactual_ssim_diff(sample: Tensor, contone: Tensor, config: MultiAgentDRLConfig) -> Tensor:
    stats = _ssim_stats(sample, contone, config.ssim_kernel_size, config.ssim_sigma, config.eps)
    kernel = stats["kernel"]
    center = config.ssim_kernel_size // 2
    ssim_diff = torch.zeros_like(sample)
    valid_pixels = torch.ones_like(sample)
    c1 = float(stats["c1"].item())
    c2 = float(stats["c2"].item())
    total_positions = float(sample.shape[-2] * sample.shape[-1])

    for ky in range(config.ssim_kernel_size):
        for kx in range(config.ssim_kernel_size):
            weight = float(kernel[0, 0, ky, kx].item())
            if weight == 0.0:
                continue
            offset_y = ky - center
            offset_x = kx - center
            sample_pixel = _shift_with_zeros(sample, offset_y, offset_x)
            contone_pixel = _shift_with_zeros(contone, offset_y, offset_x)
            valid_window = _shift_with_zeros(valid_pixels, offset_y, offset_x)
            delta_one = 1.0 - sample_pixel
            delta_zero = -sample_pixel

            mu_h_one = stats["mu_x"] + delta_one * weight
            mu_h_zero = stats["mu_x"] + delta_zero * weight
            mean_sq_h_one = stats["mean_sq_x"] + delta_one * weight
            mean_sq_h_zero = stats["mean_sq_x"] + delta_zero * weight
            sigma_h_one = torch.clamp(mean_sq_h_one - mu_h_one.square(), min=0.0)
            sigma_h_zero = torch.clamp(mean_sq_h_zero - mu_h_zero.square(), min=0.0)
            mean_hc_one = stats["mean_xy"] + delta_one * weight * contone_pixel
            mean_hc_zero = stats["mean_xy"] + delta_zero * weight * contone_pixel
            sigma_hc_one = mean_hc_one - mu_h_one * stats["mu_y"]
            sigma_hc_zero = mean_hc_zero - mu_h_zero * stats["mu_y"]

            ssim_one = ((2.0 * mu_h_one * stats["mu_y"] + c1) * (2.0 * sigma_hc_one + c2)) / (
                (mu_h_one.square() + stats["mu_y"].square() + c1) * (sigma_h_one + stats["sigma_y"] + c2).clamp_min(config.eps)
            )
            ssim_zero = ((2.0 * mu_h_zero * stats["mu_y"] + c1) * (2.0 * sigma_hc_zero + c2)) / (
                (mu_h_zero.square() + stats["mu_y"].square() + c1) * (sigma_h_zero + stats["sigma_y"] + c2).clamp_min(config.eps)
            )
            local_diff = ((ssim_one - ssim_zero) * valid_window) / total_positions
            ssim_diff = ssim_diff + _shift_with_zeros(local_diff, -offset_y, -offset_x)

    return ssim_diff


def _counterfactual_mse_diff(sample: Tensor, contone: Tensor, config: MultiAgentDRLConfig) -> Tensor:
    kernel_2d = _nasanen_kernel(config.hvs_kernel_size, config.hvs_scale_factor, config.hvs_luminance)
    kernel_size = config.hvs_kernel_size
    kernel = kernel_2d.to(device=sample.device, dtype=sample.dtype).view(1, 1, kernel_size, kernel_size)
    filtered_sample = F.conv2d(sample, kernel, padding=kernel_size // 2)
    filtered_contone = F.conv2d(contone, kernel, padding=kernel_size // 2)
    error = filtered_sample - filtered_contone
    cross = F.conv2d(error, kernel, padding=kernel_size // 2)
    kernel_sq = kernel.square()
    support = F.conv2d(torch.ones_like(sample), kernel_sq, padding=kernel_size // 2)
    normalization = float(sample.shape[-2] * sample.shape[-1])
    return (2.0 * cross + (1.0 - 2.0 * sample) * support) / normalization


def local_expectation_policy_loss(
    probabilities: Tensor,
    contone: Tensor,
    config: MultiAgentDRLConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
    sampled_halftone = realize_halftone(probabilities, config.white_threshold, mode="sample")
    reward, tone_error, ssim = compute_reward(sampled_halftone, contone, config)
    reward_diff = -_counterfactual_mse_diff(sampled_halftone, contone, config) + config.ssim_weight * _counterfactual_ssim_diff(sampled_halftone, contone, config)
    policy_loss = -(probabilities * reward_diff.detach()).sum(dim=(1, 2, 3)).mean()
    return policy_loss, {
        "sampled_halftone": sampled_halftone,
        "reward": reward.mean(),
        "tone_error": tone_error.mean(),
        "ssim": ssim.mean(),
        "reward_diff_mean": reward_diff.mean(),
    }


def save_triptych(result: Dict[str, Tensor], path: Path) -> None:
    contone = result["contone"][0, 0].detach().cpu().clamp(0.0, 1.0).numpy()
    probability = result["probability"][0, 0].detach().cpu().clamp(0.0, 1.0).numpy()
    halftone = result["halftone"][0, 0].detach().cpu().clamp(0.0, 1.0).numpy()
    panels = []
    for array in [contone, probability, halftone]:
        image = Image.fromarray(np.uint8(np.round(array * 255.0)), mode="L")
        panels.append(image.convert("RGB"))
    canvas = Image.new("RGB", (panels[0].width * 3, panels[0].height), color="white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel.width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def train_step(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    contone: Tensor,
    gray: Tensor,
    config: MultiAgentDRLConfig,
    step: int,
) -> Dict[str, float]:
    model.train()
    probs = model(contone, torch.randn_like(contone)).clamp(config.eps, 1.0 - config.eps)
    policy_loss, policy_payload = local_expectation_policy_loss(probs, contone, config)
    if config.variant == "paper":
        dispersion_anisotropy_loss = probs.new_tensor(0.0)
        dispersion_lowfreq_loss = probs.new_tensor(0.0)
        dispersion_loss = probs.new_tensor(0.0)
        dispersion_term = probs.new_tensor(0.0)
    else:
        soft_halftone = smooth_threshold_map(
            probs,
            config.white_threshold,
            config.threshold_temperature,
            config.eps,
        )
        natural_error = soft_halftone - contone
        dispersion_anisotropy_loss = anisotropy_loss_from_probs(natural_error)
        dispersion_lowfreq_loss = low_frequency_ratio_loss(
            natural_error,
            max_radius=config.dispersion_lowfreq_radius,
            eps=config.eps,
        )
        dispersion_loss = dispersion_anisotropy_loss + dispersion_lowfreq_loss
        dispersion_term = config.dispersion_weight * dispersion_loss
    gray_probs = model(gray, torch.randn_like(gray)).clamp(config.eps, 1.0 - config.eps)
    gray_soft_threshold = smooth_threshold_map(gray_probs, config.white_threshold, config.threshold_temperature, config.eps)
    anisotropy_input = gray_probs if config.variant == "paper" else gray_soft_threshold
    anisotropy_loss = anisotropy_loss_from_probs(anisotropy_input)
    threshold_density_loss = probs.new_tensor(0.0) if config.variant == "paper" else density_loss_from_probs(gray_soft_threshold, gray)
    anisotropy_term = config.anisotropy_weight * anisotropy_loss
    threshold_density_term = config.threshold_density_weight * threshold_density_loss
    regularization_loss = anisotropy_term + threshold_density_term
    total_loss = policy_loss + dispersion_term + regularization_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "reward": float(policy_payload["reward"].detach().item()),
        "tone_error": float(policy_payload["tone_error"].detach().item()),
        "ssim": float(policy_payload["ssim"].detach().item()),
        "reward_diff_mean": float(policy_payload["reward_diff_mean"].detach().item()),
        "dispersion_anisotropy_loss": float(dispersion_anisotropy_loss.detach().item()),
        "dispersion_lowfreq_loss": float(dispersion_lowfreq_loss.detach().item()),
        "dispersion_loss": float(dispersion_loss.detach().item()),
        "dispersion_term": float(dispersion_term.detach().item()),
        "dispersion_weight": float(config.dispersion_weight),
        "anisotropy_loss": float(anisotropy_loss.detach().item()),
        "anisotropy_term": float(anisotropy_term.detach().item()),
        "threshold_density_loss": float(threshold_density_loss.detach().item()),
        "threshold_density_term": float(threshold_density_term.detach().item()),
        "regularization_loss": float(regularization_loss.detach().item()),
        "anisotropy_weight": float(config.anisotropy_weight),
        "threshold_density_weight": float(config.threshold_density_weight),
        "prob_mean": float(probs.mean().detach().item()),
        "prob_std": float(probs.std(unbiased=False).detach().item()),
        "gray_prob_mean": float(gray_probs.mean().detach().item()),
        "gray_prob_std": float(gray_probs.std(unbiased=False).detach().item()),
        "gray_soft_threshold_mean": float(gray_soft_threshold.mean().detach().item()),
        "gray_target_mean": float(gray.mean().detach().item()),
    }


def serialize_config(config: MultiAgentDRLConfig) -> Dict[str, object]:
    return asdict(config)
