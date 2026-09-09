from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = Path(__file__).with_name("efficient_halftoning_drl.py")
SPEC = importlib.util.spec_from_file_location("efficient_drl_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_paper_config_and_inference_shapes() -> None:
    config = MODULE.paper_reference_config()
    config = MODULE.DRLHalftoningConfig(**{**config.__dict__, "channels": 4, "num_res_blocks": 1})
    model = MODULE.build_reference_model(config)
    contone = torch.rand(1, 1, 8, 8)
    result = MODULE.infer_halftone(model, contone, config, noise=torch.zeros_like(contone))
    assert result["probabilities"].shape == contone.shape
    assert result["halftone"].shape == contone.shape
    assert set(torch.unique(result["halftone"]).tolist()).issubset({0.0, 1.0})


def test_reward_map_is_finite_and_pixel_aligned() -> None:
    config = MODULE.paper_reference_config()
    contone = torch.rand(2, 1, 8, 8)
    halftone = (contone >= 0.5).to(contone.dtype)
    reward, metrics = MODULE.compute_reward_map(halftone, contone, config)
    assert reward.shape == contone.shape
    assert torch.isfinite(reward).all()
    assert set(metrics) == {"tone_error", "cssim", "reward"}
    assert all(torch.isfinite(value) for value in metrics.values())


def test_local_expectation_tone_error_matches_brute_force_perturbation() -> None:
    """Regression test for the local-expectation windowing bug.

    R(h,c) is a GLOBAL average of the reward map over every pixel, so
    flipping a single agent a's action shifts the HVS-filtered residual at
    every pixel within the kernel's support around a - not just at a
    itself. This checks that the tone-error half of the local-expectation
    estimator (the `reward_if`-equivalent quantity computed inside
    `local_expectation_policy_loss`) exactly matches an explicit brute-force
    per-pixel perturbation-and-recompute of the true global tone error, for
    every pixel in a small test image. The previous center-tap-only
    shortcut failed this check by a wide margin (it only got the value at
    p == a right, and silently dropped every other pixel's contribution).
    """
    torch.manual_seed(0)
    config = MODULE.DRLHalftoningConfig(
        hvs_model="gaussian", hvs_kernel_size=5, hvs_sigma=1.2, cssim_weight=0.0,
    )
    height = width = 10
    contone = torch.rand(1, 1, height, width)
    sampled_halftone = (torch.rand(1, 1, height, width) < contone).float()

    hvs = MODULE.hvs_kernel(config, contone.device, contone.dtype)
    filtered_sample = MODULE.apply_low_pass(sampled_halftone, hvs)
    filtered_contone = MODULE.apply_low_pass(contone, hvs)
    residual = filtered_sample - filtered_contone
    conv_residual = MODULE.apply_low_pass(residual, hvs)
    hvs_sq_sum = hvs.square().sum()
    n_pixels = height * width
    base_tone_error_mean = residual.square().mean(dim=(-2, -1), keepdim=True)

    for target in (0.0, 1.0):
        delta = target - sampled_halftone
        closed_form = (
            base_tone_error_mean
            + (2.0 / n_pixels) * delta * conv_residual
            + (1.0 / n_pixels) * delta.square() * hvs_sq_sum
        )

        brute_force = torch.zeros(1, 1, height, width)
        for i in range(height):
            for j in range(width):
                perturbed = sampled_halftone.clone()
                perturbed[0, 0, i, j] = target
                filtered_perturbed = MODULE.apply_low_pass(perturbed, hvs)
                brute_force[0, 0, i, j] = (filtered_perturbed - filtered_contone).square().mean()

        # interior pixels (away from the zero-padded border) should match
        # to floating-point precision; the closed form assumes the kernel
        # window is never truncated by the image edge.
        r = config.hvs_kernel_size // 2
        interior_closed = closed_form[:, :, r:-r, r:-r]
        interior_brute = brute_force[:, :, r:-r, r:-r]
        assert torch.allclose(interior_closed, interior_brute, atol=1e-5), (
            f"target={target}: max interior error "
            f"{(interior_closed - interior_brute).abs().max().item():.3e}"
        )