"""Non-Cartesian MORSE-PI coil sensitivity estimation."""

import numpy as np
from numpy.typing import NDArray

from morse_core import centred_fft, morse_pi_core, prepare_noise


def cartesian_kspace_centre(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 12,
    plot: bool = False,
) -> NDArray:
    """Extract and reconstruct the Cartesian k-space centre from samples."""
    from mrinufft import get_operator
    from mrinufft.extras.smaps import _extract_kspace_center

    k_space, samples, _ = _extract_kspace_center(
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

        _, axes = plt.subplots(2, 2)
        coil = 2
        axes[0, 0].scatter(
            (traj / (np.pi * 2))[:, 0],
            (traj / (np.pi * 2))[:, 1],
            s=1,
            c=np.abs(kspace_data[coil]),
        )
        axes[0, 1].scatter(samples[:, 0], samples[:, 1], s=1, c=np.abs(k_space[coil]))
        axes[1, 0].imshow(np.abs(central_kspace_img[coil]))
        axes[1, 1].imshow(np.abs(centred_fft(central_kspace_img)[coil]))
        plt.show()

    return centred_fft(central_kspace_img)


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
    """Estimate MORSE-PI sensitivities from non-Cartesian k-space samples.

    ``data`` has shape ``(NCoils, NPoints)`` and ``trajectory`` is scaled to
    ±pi. MRI-NUFFT is required only by this non-Cartesian interface.
    """
    if len(resolution) not in (2, 3) or len(resolution) != trajectory.shape[-1]:
        raise ValueError(
            "resolution must contain two or three entries and match the "
            "trajectory spatial dimensions."
        )

    data, noise_colour_factor = prepare_noise(
        data, noise_cov, noise_whiten, noise_colour_power
    )

    c_kspace = cartesian_kspace_centre(trajectory, resolution, data, backend=backend)

    return morse_pi_core(
        c_kspace,
        N_ref=N_ref,
        smoothing_sd=smoothing_sd,
        noise_colour_factor=noise_colour_factor,
    )
