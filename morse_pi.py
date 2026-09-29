"""OxCIN MRS Group implementation of MORSE-PI
https://doi.org/10.48550/arXiv.2606.21336

Code copyright William Clarke, University of Oxford 2026
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from mrinufft.extras.cartesian import fft, ifft
from mrinufft.extras.smaps import _extract_kspace_center


def cartesian_kspace_centre(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 12,
    plot: bool = False
) -> NDArray:
    """
    Trajectory between ±pi, rescaled to ±0.5 on call to _extract_kspace_center

    This function is copied from the mri-nufft espirit pipeline
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator

    k_space, samples, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj / (np.pi * 2),
        threshold=tuple(float(sh) for sh in calib_width / np.asarray(shape)),
        density=density,
        window_fun="rect",
    )
    central_kspace_img = get_operator(backend)(
        samples,
        shape,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    ).pinv_solver(k_space, max_iter=max_iter)

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2,)
        coil = 2
        axes[0, 0].scatter((traj / (np.pi * 2))[:, 0], (traj / (np.pi * 2))[:, 1], s=1, c=np.abs(kspace_data[coil,...]))
        axes[0, 1].scatter(samples[:, 0], samples[:, 1], s=1, c=np.abs(k_space[coil,...]))
        axes[1, 0].imshow(np.abs(central_kspace_img[coil,:,:]))
        axes[1, 1].imshow(np.abs(fft(central_kspace_img)[coil,:,:]))
        plt.show()

    return fft(central_kspace_img, dims=2)


def _morse_pi_vrc(
        sens: NDArray,
        centre: tuple[int, ...] | None = None,
        noise_colour_factor: NDArray | None = None):

    if centre is None:
        centre = tuple(int(x / 2) for x in sens.shape[1:-1])
    vrc_sens = sens[0, ...]
    if noise_colour_factor is not None:
        vrc_sens = np.einsum(
            "ij,...j->...i", noise_colour_factor, vrc_sens)

    centre_coils = centre + (slice(None), )

    phi_coil_centre = np.angle(vrc_sens[centre_coils])

    vrc = np.sum(
        vrc_sens * np.exp(-1j * phi_coil_centre),
        axis=-1)

    sens_corrected = sens.copy()
    sens_corrected[0, ...] *= np.exp(-1j * np.angle(vrc)[..., np.newaxis])

    return sens_corrected, vrc


def morse_pi(
    data: NDArray,
    trajectory: NDArray,
    resolution: tuple[int, ...],
    backend: str,
    N_ref: int = 6,
    smoothing_sd: int = 3,
    noise_cov: NDArray | None = None,
    noise_whiten: bool = False,
    noise_colour_power: int = 0,
) -> NDArray:
    """Calculate sensitivity maps up to order N_ref using the 
    MORSE-PI method.

    See:
    MORSE-PI - Flexible and artefact-free image reconstruction for structural 
    and functional QSM and other phase-critical imaging applications.
    Barbara Dymerska et al.
    Proc. Intl. Soc. Mag. Reson. Med. 32 (2024)4265 DOI: https://doi.org/10.58530/2024/4265

    and

    MORSE: Multiple Orthogonal Reference Sensitivity Encoding
    Oliver Josephs et al
    https://doi.org/10.48550/arXiv.2510.09098

    :param data: Complex k-space data (NCoils x Npoints)
    :type data: NDArray
    :param trajectory: K-space trajectory scaled between ±pi (NPoints x 3/2 spatial dimensions)
    :type trajectory: NDArray
    :param backend: MRI-NUFFT backend string, e.g. 'torchkbnufft-cpu'
    :type backend: str
    :param resolution: Three/two tuple of spatial resolution to reconstruct maps to.
    :type resolution: tuple[int]
    :param N_ref: Number of reference coils to construct and coil sensitivity order, defaults to 6
    :type N_ref: int, optional
    :param noise_cov: Noise covariance matrix with shape (NCoils, NCoils).
    :type noise_cov: NDArray, optional
    :param noise_whiten: Whiten k-space with the Cholesky factor of noise_cov before processing.
    :type noise_whiten: bool, optional
    :param noise_colour_power: Cholesky-factor power used to colour first-order sensitivities for VRC construction.
    :type noise_colour_power: int, optional
    :return: Complex coil sensitivites (x, y, z, Ncoils, NRef)
    :rtype: NDArray
    """

    if len(resolution) != trajectory.shape[-1] \
            and len(resolution) not in (2, 3):
        raise ValueError(
            "The length of resolution must be 2 or three, "
            "and must match the number of spatial dimensions in trajectory.")

    # Handle noise covariance if provided.
    # The matrix's shape must match the number of coils in data.
    if noise_cov is not None:
        noise_cov = np.asarray(noise_cov)
        expected_noise_cov_shape = (data.shape[0], data.shape[0])
        if noise_cov.shape != expected_noise_cov_shape:
            raise ValueError(
                "noise_cov must have shape "
                f"{expected_noise_cov_shape}, got {noise_cov.shape}.")

    if not isinstance(noise_colour_power, int) or noise_colour_power < 0:
        raise ValueError("noise_colour_power must be a non-negative integer.")

    if (noise_whiten or noise_colour_power) and noise_cov is None:
        raise ValueError(
            "noise_cov is required when noise_whiten or noise_colour_power is enabled.")

    noise_factor = None
    if noise_cov is not None and (noise_whiten or noise_colour_power):
        noise_factor = np.linalg.cholesky(noise_cov)
        if noise_whiten:
            data = np.linalg.solve(noise_factor, data)

    c_kspace = cartesian_kspace_centre(
        trajectory,
        resolution,
        data,
        backend=backend
    )

    # Perform first SVD to form reference coils
    [U, S, Vh] = np.linalg.svd(
        c_kspace.reshape(c_kspace.shape[0], -1).T,
        full_matrices=False)

    smallref = U @ np.diag(S)
    smallref = smallref.T.reshape(c_kspace.shape)

    smallref_img = ifft(smallref, dims=2)

    # Form E, voxel-wise NCoil x NRef tensor
    E = np.einsum(
        "i...,j...->ij...",
        smallref_img.conj()[:N_ref, ...],
        smallref_img)

    # Smoothing step
    spatial_smoothing_kernel = (smoothing_sd, ) * len(resolution)
    E = gaussian_filter(
        E,
        (0, 0) + spatial_smoothing_kernel)

    # Second SVD
    [U2, S2, Vh2] = np.linalg.svd(
        E.T,
        full_matrices=False)

    sens = (Vh.T @ U2).T

    # From VRC
    noise_colour_factor = None
    if noise_colour_power and noise_factor is not None:
        noise_colour_factor = np.linalg.matrix_power(
            noise_factor,
            noise_colour_power)

    sens_corrected, _ = _morse_pi_vrc(
        np.moveaxis(sens, 1, -1),
        noise_colour_factor=noise_colour_factor)

    return np.ascontiguousarray(np.moveaxis(sens_corrected, 0, -1))
