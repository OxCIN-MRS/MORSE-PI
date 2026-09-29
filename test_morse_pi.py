import unittest
from unittest.mock import patch

import numpy as np

import morse_pi


class NoiseCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1 + 1j, 2 - 1j], [3 + 2j, 4 - 2j]])
        self.trajectory = np.zeros((2, 2))

    def test_noise_covariance_validation(self):
        with self.assertRaisesRegex(ValueError, "noise_cov must have shape"):
            morse_pi.morse_pi(
                self.data, self.trajectory, (2, 2), "unused", noise_cov=np.eye(3))
        with self.assertRaisesRegex(ValueError, "noise_cov is required"):
            morse_pi.morse_pi(
                self.data, self.trajectory, (2, 2), "unused", noise_whiten=True)

    def test_whitening_precedes_reconstruction(self):
        noise_cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        expected = np.linalg.solve(np.linalg.cholesky(noise_cov), self.data)
        central_kspace = np.array(
            [[[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]],
            [[2 + 0j, 1 + 0j], [4 + 0j, 3 + 0j]]])

        with patch.object(morse_pi, "cartesian_kspace_centre", return_value=central_kspace) as centre:
            morse_pi.morse_pi(
                self.data, self.trajectory, (2, 2), "unused",
                noise_cov=noise_cov, noise_whiten=True)

        np.testing.assert_allclose(centre.call_args.args[2], expected)


class VrcTests(unittest.TestCase):
    def test_only_first_order_is_phase_corrected(self):
        sens = np.array([[[1 + 0j, 1j]], [[2 + 3j, 4 + 5j]]])
        factor = np.array([[1.0, 0.0], [0.5, 1.0]])
        coloured = np.einsum("ij,...j->...i", factor, sens[0])
        expected_vrc = np.sum(
            coloured * np.exp(-1j * np.angle(coloured[0])), axis=-1)

        corrected, vrc = morse_pi._morse_pi_vrc(
            sens, centre=(0,), noise_colour_factor=factor)

        np.testing.assert_allclose(vrc, expected_vrc)
        np.testing.assert_array_equal(corrected[1], sens[1])
        np.testing.assert_allclose(
            corrected[0], sens[0] * np.exp(-1j * np.angle(vrc))[..., np.newaxis])


if __name__ == "__main__":
    unittest.main()
