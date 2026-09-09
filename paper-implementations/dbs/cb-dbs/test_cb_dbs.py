from __future__ import annotations

import unittest

import numpy as np

from cb_dbs import (
    CBDBSConfig,
    build_objective_state,
    evaluate_local_update,
    nasanen_kernel,
    optimize_cm_swaps,
    preprocess_cmy,
    run_cb_dbs,
)


class TestCBDBS(unittest.TestCase):
    def test_preprocess_matches_piecewise_definition(self) -> None:
        rgb = np.array(
            [
                [[0.2, 0.6, 0.8], [0.1, 0.1, 0.9]],
                [[0.7, 0.4, 0.5], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        terms = preprocess_cmy(rgb)
        c = 1.0 - rgb[:, :, 0]
        m = 1.0 - rgb[:, :, 1]
        combined = c + m
        expected_c_prime = np.where(combined <= 1.0, c, 1.0 - m)
        expected_m_prime = np.where(combined <= 1.0, m, 1.0 - c)

        np.testing.assert_allclose(terms["c_prime"], expected_c_prime, atol=1e-7)
        np.testing.assert_allclose(terms["m_prime"], expected_m_prime, atol=1e-7)
        np.testing.assert_allclose(terms["cm_total"], expected_c_prime + expected_m_prime, atol=1e-7)
        np.testing.assert_allclose(terms["combined_cm"], combined, atol=1e-7)

    def test_local_update_matches_full_energy_recomputation(self) -> None:
        kernel = nasanen_kernel(7, 2000.0, 100.0)
        binary = np.zeros((8, 8), dtype=np.float32)
        binary[2, 3] = 1.0
        binary[5, 4] = 1.0
        target = np.full((8, 8), 0.25, dtype=np.float32)
        objective = build_objective_state(binary, target, kernel)

        update = evaluate_local_update(objective.filtered_error, kernel, [(2, 3, -1.0), (2, 4, 1.0)], binary.shape)

        updated = binary.copy()
        updated[2, 3] = 0.0
        updated[2, 4] = 1.0
        recomputed = build_objective_state(updated, target, kernel)
        expected_delta = recomputed.energy - objective.energy
        self.assertAlmostEqual(update.delta_energy, expected_delta, places=5)

    def test_cm_constraint_is_preserved_by_swap_stage(self) -> None:
        rng = np.random.default_rng(0)
        rgb = np.linspace(0.0, 1.0, num=16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
        terms = preprocess_cmy(rgb)
        g_cm = (rng.random((16, 16)) < terms["cm_total"]).astype(np.float32)
        kernel = nasanen_kernel(7, 2000.0, 100.0)
        c_map, m_map = optimize_cm_swaps(
            g_cm=g_cm,
            c_prime=terms["c_prime"],
            m_prime=terms["m_prime"],
            kernel=kernel,
            config=CBDBSConfig(cm_passes=1, random_seed=0),
            rng=np.random.default_rng(0),
        )
        np.testing.assert_array_equal(c_map + m_map, g_cm)
        self.assertTrue(np.all((c_map == 0.0) | (c_map == 1.0)))
        self.assertTrue(np.all((m_map == 0.0) | (m_map == 1.0)))

    def test_final_planes_preserve_nonblue_constraint(self) -> None:
        rgb = np.array(
            [
                [[0.2, 0.6, 0.8], [0.1, 0.1, 0.9], [0.6, 0.1, 0.4], [0.0, 0.0, 1.0]],
                [[0.7, 0.4, 0.5], [0.0, 0.0, 1.0], [0.3, 0.8, 0.7], [0.1, 0.2, 0.3]],
                [[0.9, 0.2, 0.1], [0.5, 0.5, 0.5], [0.8, 0.1, 0.2], [0.2, 0.2, 0.2]],
                [[0.3, 0.3, 0.3], [0.4, 0.6, 0.2], [0.7, 0.1, 0.6], [0.9, 0.9, 0.9]],
            ],
            dtype=np.float32,
        )
        result = run_cb_dbs(rgb, CBDBSConfig(mono_passes=1, cm_passes=1, random_seed=0))
        non_blue_c = result.c_plane - result.blue_mask
        non_blue_m = result.m_plane - result.blue_mask
        np.testing.assert_array_equal(non_blue_c + non_blue_m, result.cm_total)
        np.testing.assert_array_equal(result.blue_mask, ((result.cm_total == 0.0) & ((1.0 - rgb[:, :, 0]) + (1.0 - rgb[:, :, 1]) >= 1.0)).astype(np.float32))
        self.assertTrue(np.all((result.c_plane == 0.0) | (result.c_plane == 1.0)))
        self.assertTrue(np.all((result.m_plane == 0.0) | (result.m_plane == 1.0)))
        self.assertTrue(np.all((result.y_plane == 0.0) | (result.y_plane == 1.0)))

    def test_blue_overlap_follows_zero_cm_dots_at_or_above_sum_one(self) -> None:
        rgb = np.array([[[0.4, 0.6, 1.0]]], dtype=np.float32)
        result = run_cb_dbs(rgb, CBDBSConfig(mono_passes=1, cm_passes=1, random_seed=0))
        self.assertAlmostEqual(float((1.0 - rgb[:, :, 0] + 1.0 - rgb[:, :, 1])[0, 0]), 1.0, places=6)
        np.testing.assert_array_equal(result.blue_mask, ((result.cm_total == 0.0) & np.array([[True]])).astype(np.float32))


if __name__ == "__main__":
    unittest.main()
