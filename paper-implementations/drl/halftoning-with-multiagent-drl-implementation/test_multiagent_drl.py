from __future__ import annotations

import itertools
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).with_name("multiagent_drl.py")
SPEC = importlib.util.spec_from_file_location("multiagent_drl_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CounterfactualTermsTest(unittest.TestCase):
    def _build_config(self, ssim_kernel_size: int) -> MODULE.MultiAgentDRLConfig:
        return MODULE.MultiAgentDRLConfig(
            crop_size=5,
            batch_size=1,
            channels=4,
            num_res_blocks=1,
            hvs_kernel_size=5,
            ssim_kernel_size=ssim_kernel_size,
            ssim_sigma=1.0,
        )

    def _assert_counterfactual_terms_exact(self, config: MODULE.MultiAgentDRLConfig) -> None:
        torch.manual_seed(0)
        contone = torch.rand(1, 1, 5, 5)
        sample = (torch.rand(1, 1, 5, 5) > 0.5).to(dtype=contone.dtype)

        mse_diff = MODULE._counterfactual_mse_diff(sample, contone, config)
        ssim_diff = MODULE._counterfactual_ssim_diff(sample, contone, config)

        max_mse_err = 0.0
        max_ssim_err = 0.0
        for y in range(sample.shape[-2]):
            for x in range(sample.shape[-1]):
                halftone_one = sample.clone()
                halftone_zero = sample.clone()
                halftone_one[0, 0, y, x] = 1.0
                halftone_zero[0, 0, y, x] = 0.0

                filtered_one = MODULE.hvs_filter(halftone_one, config)
                filtered_zero = MODULE.hvs_filter(halftone_zero, config)
                filtered_contone = MODULE.hvs_filter(contone, config)
                brute_mse_diff = (
                    -((filtered_one - filtered_contone).square().mean())
                    + ((filtered_zero - filtered_contone).square().mean())
                ).item()
                brute_ssim_diff = (
                    MODULE.compute_ssim(halftone_one, contone, config)
                    - MODULE.compute_ssim(halftone_zero, contone, config)
                ).item()

                max_mse_err = max(max_mse_err, abs(brute_mse_diff - (-mse_diff[0, 0, y, x].item())))
                max_ssim_err = max(max_ssim_err, abs(brute_ssim_diff - ssim_diff[0, 0, y, x].item()))

        self.assertLess(max_mse_err, 1e-6)
        self.assertLess(max_ssim_err, 1e-6)

    def test_counterfactual_terms_are_exact_for_kernel_size_3(self) -> None:
        self._assert_counterfactual_terms_exact(self._build_config(ssim_kernel_size=3))

    def test_counterfactual_terms_are_exact_for_kernel_size_5(self) -> None:
        self._assert_counterfactual_terms_exact(self._build_config(ssim_kernel_size=5))


class ObjectiveConsistencyTest(unittest.TestCase):
    def test_paper_config_disables_repository_auxiliary_losses(self) -> None:
        config = MODULE.paper_reference_config()
        self.assertEqual(config.variant, "paper")
        self.assertEqual(config.dispersion_weight, 0.0)
        self.assertEqual(config.threshold_density_weight, 0.0)
        self.assertAlmostEqual(config.ssim_weight, 0.006)
        self.assertAlmostEqual(config.anisotropy_weight, 0.002)

    def test_smooth_threshold_map_is_centered_and_monotonic(self) -> None:
        probabilities = torch.tensor([[[[0.4, 0.5, 0.6]]]], dtype=torch.float32)
        softened = MODULE.smooth_threshold_map(probabilities, white_threshold=0.5, temperature=0.05, eps=1e-6)
        self.assertLess(float(softened[0, 0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(softened[0, 0, 0, 1]), 0.5, places=6)
        self.assertGreater(float(softened[0, 0, 0, 2]), 0.5)

    def test_low_frequency_ratio_loss_is_zero_for_zero_signal(self) -> None:
        signal = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        loss = MODULE.low_frequency_ratio_loss(signal, max_radius=4, eps=1e-6)
        self.assertEqual(float(loss), 0.0)

    def test_constant_probability_field_has_zero_anisotropy_for_even_crops(self) -> None:
        for level in (0.1, 0.25, 0.5, 0.75, 0.9):
            probs = torch.full((1, 1, 64, 64), level, dtype=torch.float32)
            loss = MODULE.anisotropy_loss_from_probs(probs)
            self.assertLess(float(loss), 1e-6)

    def test_local_expectation_gradient_matches_exact_expected_reward_gradient(self) -> None:
        config = MODULE.MultiAgentDRLConfig(
            crop_size=2,
            batch_size=1,
            channels=4,
            num_res_blocks=1,
            hvs_kernel_size=3,
            ssim_kernel_size=3,
            ssim_sigma=1.0,
        )
        contone = torch.tensor([[[[0.2, 0.7], [0.4, 0.9]]]], dtype=torch.float32)
        probabilities = torch.tensor([[[[0.3, 0.6], [0.8, 0.25]]]], dtype=torch.float32, requires_grad=True)

        exact_reward = probabilities.new_tensor(0.0)
        exact_grad = torch.zeros_like(probabilities)
        for bits in itertools.product((0.0, 1.0), repeat=4):
            halftone = torch.tensor(bits, dtype=probabilities.dtype).view(1, 1, 2, 2)
            mass = probabilities.new_tensor(1.0)
            for index, bit in enumerate(bits):
                y, x = divmod(index, 2)
                prob = probabilities[0, 0, y, x]
                mass = mass * (prob if bit else (1.0 - prob))
            reward, _, _ = MODULE.compute_reward(halftone, contone, config)
            exact_reward = exact_reward + mass * reward.squeeze(0)

            reward_diff = (
                -MODULE._counterfactual_mse_diff(halftone, contone, config)
                + config.ssim_weight * MODULE._counterfactual_ssim_diff(halftone, contone, config)
            )
            exact_grad = exact_grad + mass.detach() * reward_diff

        exact_reward.backward()
        self.assertTrue(torch.allclose(probabilities.grad, exact_grad, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
