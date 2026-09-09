from __future__ import annotations

import math
import unittest

import numpy as np

import GAED_implementation as gaed


class TestGAEDImplementation(unittest.TestCase):
    def test_dfog_prefers_aligned_orientations(self) -> None:
        aligned = gaed._dfog(0.0, 0.0)
        diagonal = gaed._dfog(0.0, math.pi / 4.0)
        perpendicular = gaed._dfog(0.0, math.pi / 2.0)
        opposite = gaed._dfog(0.0, math.pi)

        self.assertGreater(aligned, diagonal)
        self.assertGreater(diagonal, perpendicular)
        self.assertAlmostEqual(aligned, 1.0, places=6)
        self.assertAlmostEqual(opposite, aligned, places=6)
        self.assertAlmostEqual(perpendicular, 0.0, places=6)

    def test_local_mean_difference_uses_evolving_image_values(self) -> None:
        image = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected = (1.0 / 8.0) - 0.5
        self.assertAlmostEqual(gaed._local_mean_difference(image, 1, 1), expected, places=12)

    def test_local_sobel_magnitude_is_normalized(self) -> None:
        image = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        magnitude, _ = gaed._local_sobel_at(image, 1, 1)
        self.assertGreaterEqual(magnitude, 0.0)
        self.assertLessEqual(magnitude, 1.0)

    def test_edge_correlation_uses_non_wrapping_boundaries(self) -> None:
        g = np.zeros((3, 3), dtype=np.float64)
        z = np.zeros((3, 3), dtype=np.float64)
        g[0, 0] = 1.0
        z[0, 0] = 1.0

        manual = 0.0
        for k in range(-1, 2):
            for l in range(-1, 2):
                if k == 0 and l == 0:
                    continue
                w = gaed._EC_W[k + 1, l + 1]
                for row in range(g.shape[0]):
                    src_row = row - k
                    if not (0 <= src_row < g.shape[0]):
                        continue
                    for col in range(g.shape[1]):
                        src_col = col - l
                        if not (0 <= src_col < g.shape[1]):
                            continue
                        dg = g[row, col] - g[src_row, src_col]
                        dz = z[row, col] - z[src_row, src_col]
                        manual += w * dg * dz
        manual /= g.size

        self.assertAlmostEqual(gaed._edge_correlation(g, z), manual, places=12)

    def test_gaed_halftone_is_binary(self) -> None:
        gradient = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float64), (16, 1))
        halftone = gaed.gaed_halftone((gradient * 255).astype(np.uint8))
        values = np.unique(halftone)
        self.assertTrue(set(values.tolist()).issubset({0, 1}))

    def test_all_variants_return_binary_outputs(self) -> None:
        image = (np.tile(np.linspace(0.0, 1.0, 12, dtype=np.float64), (12, 1)) * 255).astype(np.uint8)
        model = gaed.GAED()
        for variant in ("floyd", "gaed_threshold", "gaed_adaptive", "gaed"):
            halftone = model.halftone_variant(image, variant=variant)
            self.assertTrue(set(np.unique(halftone).tolist()).issubset({0, 1}), msg=variant)

    def test_strict_boundary_mode_runs(self) -> None:
        image = (np.tile(np.linspace(0.0, 1.0, 10, dtype=np.float64), (10, 1)) * 255).astype(np.uint8)
        model = gaed.GAED(boundary_mode="strict")
        halftone = model.halftone_variant(image, variant="floyd")
        self.assertTrue(set(np.unique(halftone).tolist()).issubset({0, 1}))

    def test_adaptive_confidence_tracks_r_gap(self) -> None:
        valid = [True, True, True, True]
        weak = np.array([0.50, 0.49, 0.10, 0.05], dtype=np.float64)
        strong = np.array([0.90, 0.20, 0.10, 0.05], dtype=np.float64)
        self.assertLess(gaed._adaptive_confidence(valid, weak), gaed._adaptive_confidence(valid, strong))

    def test_edge_trigger_should_be_stricter_than_dfmg_nonzero(self) -> None:
        self.assertGreater(gaed._dfmg(0.05), 0.0)
        self.assertLess(0.05, gaed._TMG)

    def test_invalid_variant_and_boundary_are_rejected(self) -> None:
        image = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            gaed.GAED().halftone_variant(image, variant="unknown")
        with self.assertRaises(ValueError):
            gaed.GAED(boundary_mode="unknown").halftone(image)

    def test_instances_do_not_share_tmg_configuration(self) -> None:
        first = gaed.GAED(tmg=0.05)
        second = gaed.GAED(tmg=0.2)
        self.assertAlmostEqual(first.tmg, 0.05)
        self.assertAlmostEqual(second.tmg, 0.2)


if __name__ == "__main__":
    unittest.main()
