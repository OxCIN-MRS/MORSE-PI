"""Shared, Cartesian-k-space MORSE-PI sensitivity estimation."""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import center_of_mass, gaussian_filter, label


def centred_fft(image: NDArray) -> NDArray:
    """Orthonormal FFT with the k-space origin at the centre of each axis."""
    axes = tuple(range(1, image.ndim))
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def centred_ifft(kspace: NDArray) -> NDArray:
    """Orthonormal inverse FFT for centre-origin k-space."""
    axes = tuple(range(1, kspace.ndim))
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def prepare_noise(
    data: NDArray,
    noise_cov: NDArray | None,
    noise_whiten: bool,
    noise_colour_power: int,
) -> tuple[NDArray, NDArray | None]:
    """Validate optional noise handling and return whitened data and VRC factor."""
    if noise_cov is not None:
        noise_cov = np.asarray(noise_cov)
        expected_shape = (data.shape[0], data.shape[0])
        if noise_cov.shape != expected_shape:
            raise ValueError(
                f"noise_cov must have shape {expected_shape}, got {noise_cov.shape}."
            )

    if not isinstance(noise_colour_power, int) or noise_colour_power < 0:
        raise ValueError("noise_colour_power must be a non-negative integer.")
    if (noise_whiten or noise_colour_power) and noise_cov is None:
        raise ValueError(
            "noise_cov is required when noise_whiten or noise_colour_power is enabled."
        )

    if noise_cov is None or not (noise_whiten or noise_colour_power):
        return data, None

    noise_factor = np.linalg.cholesky(noise_cov)
    if noise_whiten:
        data = np.linalg.solve(noise_factor, data.reshape(data.shape[0], -1)).reshape(
            data.shape
        )
    if noise_colour_power:
        return data, np.linalg.matrix_power(noise_factor, noise_colour_power)
    return data, None


def _vrc_seed(rss: NDArray) -> tuple[int, ...]:
    """RSS-weighted centroid of the largest component, matching the authors."""
    if not np.isfinite(rss).all() or not np.any(rss > 0):
        raise ValueError("Calibration magnitude cannot define a VRC seed.")
    mask = rss > 0.8 * np.mean(rss)
    # Reverse axes so component labels follow MATLAB's column-major traversal.
    components, count = label(mask.T, structure=np.ones((3,) * rss.ndim))
    components = components.T
    if not count:
        raise ValueError("Calibration magnitude cannot define a VRC seed.")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    centroid = center_of_mass(rss, components, int(np.argmax(sizes)))
    # MATLAB rounds positive half ties up; np.rint uses ties to even.
    return tuple(int(np.floor(x + 0.5)) for x in centroid)


def morse_pi_vrc(
    sens: NDArray,
    centre: tuple[int, ...],
    noise_colour_factor: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Correct only first-order sensitivities with the VRC phase."""
    vrc_sens = sens[0]
    if noise_colour_factor is not None:
        vrc_sens = np.einsum("ij,...j->...i", noise_colour_factor, vrc_sens)

    phi_coil_centre = np.angle(vrc_sens[centre + (slice(None),)])
    vrc = np.sum(vrc_sens * np.exp(-1j * phi_coil_centre), axis=-1)

    sens_corrected = sens.copy()
    sens_corrected[0] *= np.exp(-1j * np.angle(vrc)[..., np.newaxis])
    return sens_corrected, vrc


def morse_pi_core(
    c_kspace: NDArray,
    N_ref: int = 6,
    smoothing_sd: int = 3,
    noise_colour_factor: NDArray | None = None,
) -> NDArray:
    """Estimate sensitivities from centre-origin Cartesian k-space."""
    ndim = c_kspace.ndim - 1
    U, S, Vh = np.linalg.svd(
        c_kspace.reshape(c_kspace.shape[0], -1).T, full_matrices=False
    )
    smallref = (U @ np.diag(S)).T.reshape(c_kspace.shape)
    smallref_img = centred_ifft(smallref)
    centre = _vrc_seed(np.linalg.norm(smallref_img, axis=0))

    E = np.einsum("i...,j...->ij...", smallref_img.conj()[:N_ref], smallref_img)
    E = gaussian_filter(E, (0, 0) + (smoothing_sd,) * ndim)

    U2, _, _ = np.linalg.svd(E.T, full_matrices=False)
    sens = (Vh.T @ U2).T
    sens_corrected, _ = morse_pi_vrc(
        np.moveaxis(sens, 1, -1), centre=centre, noise_colour_factor=noise_colour_factor
    )
    return np.ascontiguousarray(np.moveaxis(sens_corrected, 0, -1))
