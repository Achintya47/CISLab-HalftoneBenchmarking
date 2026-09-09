from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F


_KERNEL_CACHE: Dict[tuple[object, ...], Tensor] = {}


@dataclass(frozen=True)
class DRLHalftoningConfig:
    variant: str = "stabilized"
    policy_estimator: str = "reinforce"
    crop_size: int = 64
    batch_size: int = 64
    num_res_blocks: int = 16
    channels: int = 32
    hvs_model: str = "gaussian"
    hvs_kernel_size: int = 11
    hvs_sigma: float = 1.5
    hvs_scale_factor: float = 2000.0
    hvs_luminance: float = 11.0
    contrast_kernel_size: int = 11
    contrast_sigma: float = 1.5
    contrast_normalization: float = 1.0
    cssim_weight: float = 1.0
    anisotropy_weight: float = 1e-3
    density_weight: float = 10.0
    learning_rate: float = 3e-4
    learning_rate_min: float = 1e-5
    white_threshold: float = 0.5
    init_std: float = 0.01
    eps: float = 1e-6


def stabilized_reference_config() -> DRLHalftoningConfig:
    return DRLHalftoningConfig(
        variant="stabilized",
        policy_estimator="reinforce",
        hvs_model="gaussian",
        hvs_kernel_size=11,
        hvs_sigma=1.5,
        hvs_scale_factor=2000.0,
        hvs_luminance=11.0,
        contrast_kernel_size=11,
        contrast_sigma=1.5,
        contrast_normalization=1.0,
        cssim_weight=1.0,
        anisotropy_weight=1e-3,
        density_weight=10.0,
    )


def paper_reference_config() -> DRLHalftoningConfig:
    return DRLHalftoningConfig(
        variant="paper",
        policy_estimator="local_expectation",
        hvs_model="nasanen",
        hvs_kernel_size=11,
        hvs_sigma=1.5,
        hvs_scale_factor=2000.0,
        hvs_luminance=11.0,
        contrast_kernel_size=11,
        contrast_sigma=1.5,
        contrast_normalization=2.0,
        cssim_weight=0.06,
        anisotropy_weight=0.002,
        density_weight=0.0,
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        return F.relu(x + residual, inplace=True)


class PolicyNet(nn.Module):
    def __init__(self, channels: int = 32, num_res_blocks: int = 16) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        self.output_conv = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, contone: Tensor, noise: Tensor) -> Tensor:
        x = torch.cat([contone, noise], dim=1)
        x = F.relu(self.input_conv(x), inplace=True)
        x = self.res_blocks(x)
        logits = self.output_conv(x)
        return torch.sigmoid(logits)


def initialize_policy(model: nn.Module, std: float = 0.01) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def build_optimizer_and_scheduler(
    model: nn.Module,
    total_iterations: int,
    config: DRLHalftoningConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iterations),
        eta_min=config.learning_rate_min,
    )
    return optimizer, scheduler


def list_images(root: str | Path) -> List[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def load_grayscale_image(path: str | Path) -> Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def random_crop_batch(
    image_paths: Sequence[str | Path],
    config: DRLHalftoningConfig,
    device: torch.device,
) -> Tensor:
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    samples: List[Tensor] = []
    for _ in range(config.batch_size):
        image = load_grayscale_image(np.random.choice(image_paths))
        image = pad_if_needed(image, config.crop_size)
        _, height, width = image.shape
        top = np.random.randint(0, height - config.crop_size + 1)
        left = np.random.randint(0, width - config.crop_size + 1)
        samples.append(image[:, top : top + config.crop_size, left : left + config.crop_size])
    return torch.stack(samples, dim=0).to(device)


def pad_if_needed(image: Tensor, crop_size: int) -> Tensor:
    _, height, width = image.shape
    pad_height = max(0, crop_size - height)
    pad_width = max(0, crop_size - width)
    if pad_height == 0 and pad_width == 0:
        return image
    return F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")


def sample_actions(probabilities: Tensor) -> tuple[Tensor, Tensor]:
    actions = torch.bernoulli(probabilities)
    log_prob = actions * torch.log(probabilities.clamp_min(1e-6))
    log_prob = log_prob + (1.0 - actions) * torch.log((1.0 - probabilities).clamp_min(1e-6))
    return actions, log_prob


def gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    cache_key = (size, sigma, str(device), str(dtype))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    axis = torch.arange(-(size // 2), size // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def nasanen_kernel(
    size: int,
    scale_factor: float,
    luminance: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    cache_key = ("nasanen", size, scale_factor, luminance, str(device), str(dtype))
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    axis = torch.arange(-(size // 2), size // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    c = 0.525
    d = 3.91
    luminance = max(luminance, 1e-6)
    scale_factor = max(scale_factor, 1e-6)
    k = (math.pi * scale_factor) / (180.0 * (c * math.log(luminance) + d))
    spatial_radius = (2.0 * math.pi / scale_factor) * radius
    kernel = torch.pow(k * k + spatial_radius.square(), -1.5)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, size, size)
    _KERNEL_CACHE[cache_key] = kernel
    return kernel


def hvs_kernel(config: DRLHalftoningConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    if config.hvs_model == "nasanen":
        return nasanen_kernel(
            config.hvs_kernel_size,
            config.hvs_scale_factor,
            config.hvs_luminance,
            device,
            dtype,
        )
    return gaussian_kernel(
        config.hvs_kernel_size,
        config.hvs_sigma,
        device,
        dtype,
    )


def apply_low_pass(image: Tensor, kernel: Tensor) -> Tensor:
    padding = kernel.shape[-1] // 2
    return F.conv2d(image, kernel, padding=padding)


def compute_contrast_map(contone: Tensor, config: DRLHalftoningConfig) -> Tensor:
    kernel = gaussian_kernel(
        config.contrast_kernel_size,
        config.contrast_sigma,
        contone.device,
        contone.dtype,
    )
    mean = apply_low_pass(contone, kernel)
    mean_sq = apply_low_pass(contone.square(), kernel)
    variance = torch.clamp(mean_sq - mean.square(), min=0.0)
    contrast = torch.sqrt(variance + config.eps)
    if config.variant == "paper":
        scale = max(config.contrast_normalization, config.eps)
        return torch.clamp(contrast * scale, 0.0, 1.0)
    denom = contrast.amax(dim=(-2, -1), keepdim=True).clamp_min(config.eps)
    return contrast / denom


def ssim_map(halftone: Tensor, contone: Tensor, config: DRLHalftoningConfig) -> Tensor:
    kernel = gaussian_kernel(
        config.contrast_kernel_size,
        config.contrast_sigma,
        halftone.device,
        halftone.dtype,
    )
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_h = apply_low_pass(halftone, kernel)
    mu_c = apply_low_pass(contone, kernel)
    sigma_h = apply_low_pass(halftone.square(), kernel) - mu_h.square()
    sigma_c = apply_low_pass(contone.square(), kernel) - mu_c.square()
    sigma_hc = apply_low_pass(halftone * contone, kernel) - mu_h * mu_c
    numerator = (2 * mu_h * mu_c + c1) * (2 * sigma_hc + c2)
    denominator = (mu_h.square() + mu_c.square() + c1) * (sigma_h + sigma_c + c2)
    return numerator / denominator.clamp_min(config.eps)


def cssim_map(halftone: Tensor, contone: Tensor, config: DRLHalftoningConfig) -> Tensor:
    cssim = ssim_map(halftone, contone, config)
    contrast = compute_contrast_map(contone, config)
    if config.variant == "paper":
        return contrast * cssim + (1.0 - contrast)
    return cssim * contrast


def compute_reward_map(halftone: Tensor, contone: Tensor, config: DRLHalftoningConfig) -> tuple[Tensor, Dict[str, Tensor]]:
    hvs = hvs_kernel(config, halftone.device, halftone.dtype)
    filtered_halftone = apply_low_pass(halftone, hvs)
    filtered_contone = apply_low_pass(contone, hvs)
    tone_error_map = (filtered_halftone - filtered_contone).square()
    cssim_values = cssim_map(halftone, contone, config)
    reward_map = -tone_error_map + config.cssim_weight * cssim_values
    metrics = {
        "tone_error": tone_error_map.mean(),
        "cssim": cssim_values.mean(),
        "reward": reward_map.mean(),
    }
    return reward_map, metrics


def anisotropy_loss(probabilities: Tensor, config: DRLHalftoningConfig) -> Tensor:
    if config.variant == "paper":
        fft_map = torch.fft.fft2(probabilities.squeeze(1), norm="ortho")
        shifted = torch.fft.fftshift(fft_map.abs().square(), dim=(-2, -1))
    else:
        centered = probabilities.squeeze(1) - probabilities.squeeze(1).mean(dim=(-2, -1), keepdim=True)
        fft_map = torch.fft.fft2(centered, norm="ortho")
        power = fft_map.abs().square()
        shifted = torch.fft.fftshift(power, dim=(-2, -1))
        total_power = shifted.sum(dim=(-2, -1), keepdim=True).clamp_min(config.eps)
        shifted = shifted / total_power
    height, width = shifted.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=shifted.device),
        torch.arange(width, device=shifted.device),
        indexing="ij",
    )
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = torch.sqrt((yy - center_y).square() + (xx - center_x).square())
    radius = radius.round().to(torch.long)
    max_radius = int(radius.max().item())

    anisotropy_terms: List[Tensor] = []
    for radius_index in range(1, max_radius + 1):
        mask = radius == radius_index
        count = int(mask.sum().item())
        if count <= 1:
            continue
        ring_values = shifted[:, mask]
        ring_mean = ring_values.mean(dim=1, keepdim=True)
        anisotropy_terms.append((ring_values - ring_mean).square().mean())
    if not anisotropy_terms:
        return probabilities.new_tensor(0.0)
    return torch.stack(anisotropy_terms).mean()


def density_consistency_loss(probabilities: Tensor, target_contone: Tensor) -> Tensor:
    predicted_density = probabilities.mean(dim=(-2, -1), keepdim=True)
    target_density = target_contone.mean(dim=(-2, -1), keepdim=True)
    return (predicted_density - target_density).square().mean()


def make_uniform_gray_batch(config: DRLHalftoningConfig, device: torch.device, dtype: torch.dtype) -> Tensor:
    gray_values = torch.rand(config.batch_size, 1, 1, 1, device=device, dtype=dtype)
    return gray_values.expand(-1, 1, config.crop_size, config.crop_size).contiguous()


def local_expectation_policy_loss(
    probabilities: Tensor,
    contone_batch: Tensor,
    config: DRLHalftoningConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
    sampled_halftone = torch.bernoulli(probabilities).detach()

    hvs = hvs_kernel(config, contone_batch.device, contone_batch.dtype)
    filtered_sample = apply_low_pass(sampled_halftone, hvs)
    filtered_contone = apply_low_pass(contone_batch, hvs)

    """
            This currently computes the correct HVS-Filtered value for every agent A,
            at position A itself (hvs_center) : hvs_center = hvs[:, :, hvs.shape[-2] // 2, hvs.shape[-1] // 2]

            It is correctly measuring how much pixel A's value contributes to the filtered response at 
            A, but the paper explicitly mentions that, 

            "First, all agents can reuse the same HVS filtered map. Second, an agent only needs to be responsible
            for the local window around it on the reward map. Third, instead of calculating R(·) from scratch, the
            opposite action's reward window equals the current action's reward window plus/minus the HVS filter, from
            each agent's perspective."

            Thus we are not calculating the Reward Window all, a single low_pass_filter call
            is required, no other kernel bookkeeping method needed.

    """
    residual = filtered_sample - filtered_contone
    conv_residual = apply_low_pass(residual, hvs)
    hvs_sq_sum = hvs.square().sum()
    n_pixels = residual.shape[-2] * residual.shape[-1]
    # This is the Σ kernel² term exactly like in the paper
    base_tone_error_mean = residual.square().mean(dim=(-2, -1), keepdim=True)

    cssim_kernel = gaussian_kernel(
        config.contrast_kernel_size,
        config.contrast_sigma,
        contone_batch.device,
        contone_batch.dtype,
    )
    cssim_center = cssim_kernel[:, :, cssim_kernel.shape[-2] // 2, cssim_kernel.shape[-1] // 2]
    mu_h = apply_low_pass(sampled_halftone, cssim_kernel)
    mu_c = apply_low_pass(contone_batch, cssim_kernel)
    mean_sq_h = apply_low_pass(sampled_halftone.square(), cssim_kernel)
    sigma_c = apply_low_pass(contone_batch.square(), cssim_kernel) - mu_c.square()
    mean_hc = apply_low_pass(sampled_halftone * contone_batch, cssim_kernel)
    contrast = compute_contrast_map(contone_batch, config)

    sampled_metrics_reward, sampled_metrics = compute_reward_map(sampled_halftone, contone_batch, config)

    def reward_if(target: float) -> Tensor:
        target_tensor = sampled_halftone.new_full(sampled_halftone.shape, target)
        delta = target_tensor - sampled_halftone
        tone_error = (
            base_tone_error_mean
            + (2.0 / n_pixels) * delta * conv_residual
            + (1.0 / n_pixels) * delta.square() * hvs_sq_sum
        )

        """
            The CSSIM term below also uses the same single Pixel update. This can be seen
            as an approximation but is not a correct Paper Implementation, it credits agent A only with the
            reward at its own Pixel, not the full window-sum.

            The exact windowed correction isn't as simple as a single extra HVS filte call, thus
            once I figure it out, will fix it immediately. (cssim_center = cssim_kernel[:, :, cssim_kernel.shape[-2] // 2, cssim_kernel.shape[-1] // 2])
        """
        mu_h_target = mu_h + delta * cssim_center
        mean_sq_h_target = mean_sq_h + (target_tensor.square() - sampled_halftone.square()) * cssim_center
        sigma_h_target = torch.clamp(mean_sq_h_target - mu_h_target.square(), min=0.0)
        mean_hc_target = mean_hc + (target_tensor * contone_batch - sampled_halftone * contone_batch) * cssim_center
        sigma_hc_target = mean_hc_target - mu_h_target * mu_c

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * mu_h_target * mu_c + c1) * (2 * sigma_hc_target + c2)
        denominator = (mu_h_target.square() + mu_c.square() + c1) * (sigma_h_target + sigma_c + c2)
        ssim_target = numerator / denominator.clamp_min(config.eps)
        cssim_target = contrast * ssim_target + (1.0 - contrast)
        return -tone_error + config.cssim_weight * cssim_target

    reward_zero = reward_if(0.0)
    reward_one = reward_if(1.0)
    local_expectation = probabilities * reward_one + (1.0 - probabilities) * reward_zero
    policy_loss = -local_expectation.mean()
    return policy_loss, {
        "sampled_reward": sampled_metrics_reward.mean(),
        "sampled_cssim": sampled_metrics["cssim"],
        "sampled_tone_error": sampled_metrics["tone_error"],
        "sampled_halftone": sampled_halftone,
    }


def train_step(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    contone_batch: Tensor,
    config: DRLHalftoningConfig,
) -> Dict[str, float]:
    model.train()
    noise = torch.randn_like(contone_batch)
    probabilities = model(contone_batch, noise)
    if config.policy_estimator == "local_expectation":
        policy_loss, estimator_payload = local_expectation_policy_loss(probabilities, contone_batch, config)
        halftone = estimator_payload["sampled_halftone"]
        metrics = {
            "reward": estimator_payload["sampled_reward"],
            "tone_error": estimator_payload["sampled_tone_error"],
            "cssim": estimator_payload["sampled_cssim"],
        }
    else:
        halftone, log_prob = sample_actions(probabilities)
        reward_map, metrics = compute_reward_map(halftone, contone_batch, config)
        baseline = reward_map.mean(dim=(-2, -1), keepdim=True)
        advantage = reward_map - baseline
        policy_loss = -(advantage.detach() * log_prob).mean()

    gray_batch = make_uniform_gray_batch(config, contone_batch.device, contone_batch.dtype)
    gray_noise = torch.randn_like(gray_batch)
    gray_probabilities = model(gray_batch, gray_noise)
    blue_noise_loss = anisotropy_loss(gray_probabilities, config)
    density_loss = density_consistency_loss(gray_probabilities, gray_batch) if config.density_weight > 0 else gray_probabilities.new_tensor(0.0)

    total_loss = (
        policy_loss
        + config.anisotropy_weight * blue_noise_loss
        + config.density_weight * density_loss
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {
        "total_loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "anisotropy_loss": float(blue_noise_loss.item()),
        "density_loss": float(density_loss.item()),
        "reward": float(metrics["reward"].item()),
        "tone_error": float(metrics["tone_error"].item()),
        "cssim": float(metrics["cssim"].item()),
        "prob_mean": float(probabilities.mean().item()),
        "prob_std": float(probabilities.std(unbiased=False).item()),
        "gray_prob_mean": float(gray_probabilities.mean().item()),
        "gray_prob_std": float(gray_probabilities.std(unbiased=False).item()),
    }


@torch.inference_mode()
def infer_halftone(
    model: PolicyNet,
    contone: Tensor,
    config: DRLHalftoningConfig,
    noise: Tensor | None = None,
) -> Dict[str, Tensor]:
    model.eval()
    if contone.ndim == 3:
        contone = contone.unsqueeze(0)
    if noise is None:
        noise = torch.randn_like(contone)
    probabilities = model(contone, noise)
    halftone = (probabilities >= config.white_threshold).to(contone.dtype)
    return {
        "contone": contone,
        "probabilities": probabilities,
        "halftone": halftone,
    }


def save_triptych(result: Dict[str, Tensor], output_path: str | Path) -> None:
    contone = tensor_to_image(result["contone"][0])
    probabilities = tensor_to_image(result["probabilities"][0])
    halftone = tensor_to_image(result["halftone"][0])
    width, height = contone.size
    canvas = Image.new("L", (width * 3, height))
    canvas.paste(contone, (0, 0))
    canvas.paste(probabilities, (width, 0))
    canvas.paste(halftone, (width * 2, 0))
    canvas.save(output_path)


def tensor_to_image(tensor: Tensor) -> Image.Image:
    array = tensor.squeeze().detach().cpu().clamp(0.0, 1.0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="L")


def build_reference_model(
    config: DRLHalftoningConfig,
    device: torch.device | None = None,
) -> PolicyNet:
    model = PolicyNet(channels=config.channels, num_res_blocks=config.num_res_blocks)
    initialize_policy(model, std=config.init_std)
    if device is not None:
        model = model.to(device)
    return model


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal no-training DRL halftoning inference.")
    parser.add_argument("input_image", type=Path, help="Path to the input image.")
    parser.add_argument("output_image", type=Path, help="Path to save the grayscale triptych output.")
    parser.add_argument("--size", type=int, default=256, help="Square resize used for the minimal demo run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible noise.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = default_device()
    config = stabilized_reference_config()
    model = build_reference_model(config, device=device)
    contone = load_grayscale_image(args.input_image).unsqueeze(0).to(device)
    contone = F.interpolate(contone, size=(args.size, args.size), mode="bilinear", align_corners=False)
    result = infer_halftone(model, contone, config)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    save_triptych(result, args.output_image)


if __name__ == "__main__":
    run_cli()
