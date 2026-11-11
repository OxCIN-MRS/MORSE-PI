import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from morse_pi import cartesian, core, morse_pi, morse_pi_cartesian, noncartesian


class NoiseCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1 + 1j, 2 - 1j], [3 + 2j, 4 - 2j]])
        self.trajectory = np.zeros((2, 2))

    def test_noise_covariance_validation(self):
        with self.assertRaisesRegex(ValueError, "noise_cov must have shape"):
            morse_pi(self.data, self.trajectory, (2, 2), "unused", noise_cov=np.eye(3))
        with self.assertRaisesRegex(ValueError, "noise_cov is required"):
            morse_pi(self.data, self.trajectory, (2, 2), "unused", noise_whiten=True)

    def test_whitening_precedes_reconstruction(self):
        noise_cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        expected = np.linalg.solve(np.linalg.cholesky(noise_cov), self.data)
        central_kspace = np.array(
            [
                [[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]],
                [[2 + 0j, 1 + 0j], [4 + 0j, 3 + 0j]],
            ]
        )

        with patch.object(
            noncartesian, "cartesian_kspace_centre", return_value=central_kspace
        ) as centre:
            morse_pi(
                self.data,
                self.trajectory,
                (2, 2),
                "unused",
                noise_cov=noise_cov,
                noise_whiten=True,
            )

        np.testing.assert_allclose(centre.call_args.args[2], expected)


class VrcTests(unittest.TestCase):
    def test_seed_uses_masked_rss_not_squared_weights(self):
        rss = np.zeros((7, 5))
        rss[:5, 1] = [1, 0.5, 0.5, 0.5, 4]
        rss[6, 4] = 10  # Brighter, but smaller disconnected component.
        self.assertEqual(core._vrc_seed(rss), (3, 1))

    def test_seed_ignores_background_outside_mask(self):
        rss = np.full((10, 10), 0.1)
        rss[0, 0] = 100
        self.assertEqual(core._vrc_seed(rss), (0, 0))

    def test_seed_3d_connectivity_axis_order_and_rounding(self):
        rss = np.zeros((4, 6, 8))
        rss[1, 2, 3] = rss[2, 3, 4] = 1
        self.assertEqual(core._vrc_seed(rss), (2, 3, 4))

    def test_seed_2d_diagonal_connectivity(self):
        rss = np.zeros((5, 5))
        rss[1, 1] = rss[2, 2] = 1
        self.assertEqual(core._vrc_seed(rss), (2, 2))

    def test_seed_volume_tie_uses_matlab_order(self):
        rss = np.zeros((5, 5))
        rss[0, 4] = rss[4, 0] = 1
        self.assertEqual(core._vrc_seed(rss), (4, 0))

    def test_invalid_calibration_has_no_seed(self):
        for rss in (np.zeros((3, 4)), np.full((3, 4), np.nan)):
            with self.assertRaisesRegex(ValueError, "cannot define a VRC seed"):
                core._vrc_seed(rss)
        with self.assertRaisesRegex(ValueError, "cannot define a VRC seed"):
            core.morse_pi_core(np.zeros((2, 3, 4), dtype=complex))

    def test_core_uses_all_calibration_coils_before_reference_truncation(self):
        image = np.zeros((2, 5, 7), dtype=complex)
        image[0, 1, 2] = 1j
        image[1, 3, 4] = 3
        image[1, 3, 5] = 2
        with (
            patch.object(core, "_vrc_seed", wraps=core._vrc_seed) as seed,
            patch.object(core, "morse_pi_vrc", wraps=core.morse_pi_vrc) as vrc,
        ):
            core.morse_pi_core(core.centred_fft(image), N_ref=1)
        np.testing.assert_allclose(
            seed.call_args.args[0], np.linalg.norm(image, axis=0), atol=1e-12
        )
        self.assertEqual(vrc.call_args.kwargs["centre"], (3, 4))

    def test_only_first_order_is_phase_corrected(self):
        sens = np.array([[[1 + 0j, 1j]], [[2 + 3j, 4 + 5j]]])
        factor = np.array([[1.0, 0.0], [0.5, 1.0]])
        coloured = np.einsum("ij,...j->...i", factor, sens[0])
        expected_vrc = np.sum(coloured * np.exp(-1j * np.angle(coloured[0])), axis=-1)

        corrected, vrc = core.morse_pi_vrc(
            sens, centre=(0,), noise_colour_factor=factor
        )

        np.testing.assert_allclose(vrc, expected_vrc)
        np.testing.assert_array_equal(corrected[1], sens[1])
        np.testing.assert_allclose(
            corrected[0], sens[0] * np.exp(-1j * np.angle(vrc))[..., np.newaxis]
        )
        np.testing.assert_allclose(abs(corrected), abs(sens))
        np.testing.assert_allclose(
            corrected[0, ..., 0] * corrected[0, ..., 1].conj(),
            sens[0, ..., 0] * sens[0, ..., 1].conj(),
        )


class CartesianTests(unittest.TestCase):
    def setUp(self):
        image = np.stack([np.ones((4, 4)), np.arange(16).reshape(4, 4) + 1j])
        self.kspace = core.centred_fft(image)

    def test_cartesian_estimation_and_resize(self):
        sens = morse_pi_cartesian(
            self.kspace, resolution=(6, 2), N_ref=2, smoothing_sd=1
        )
        self.assertEqual(sens.shape, (6, 2, 2, 2))
        self.assertTrue(np.isfinite(sens).all())

    def test_resize_keeps_dc_aligned(self):
        kspace = np.arange(16).reshape(1, 4, 4)
        resized = cartesian._resize_kspace(kspace, (6, 2))
        self.assertEqual(resized.shape, (1, 6, 2))
        self.assertEqual(resized[0, 3, 1], kspace[0, 2, 2])

    def test_cartesian_whitening_precedes_core(self):
        noise_cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        expected = np.linalg.solve(
            np.linalg.cholesky(noise_cov), self.kspace.reshape(2, -1)
        ).reshape(self.kspace.shape)
        with patch.object(
            cartesian, "morse_pi_core", return_value=np.empty(0)
        ) as core_fn:
            morse_pi_cartesian(self.kspace, noise_cov=noise_cov, noise_whiten=True)
        np.testing.assert_allclose(core_fn.call_args.args[0], expected)

    def test_cartesian_import_does_not_load_mrinufft(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from morse_pi import cartesian; import sys; "
                    "assert 'mrinufft' not in sys.modules"
                ),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
