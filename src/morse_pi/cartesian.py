"""Cartesian MORSE-PI coil sensitivity estimation without MRI-NUFFT."""

import numpy as np
from numpy.typing import NDArray

from .core import morse_pi_core, prepare_noise


def _resize_kspace(kspace: NDArray, resolution: tuple[int, ...]) -> NDArray:
    """Centre-crop or zero-pad centre-origin Cartesian k-space."""
    if len(resolution) != kspace.ndim - 1:
        raise ValueError(
            "resolution must have one entry per spatial dimension "
            f"(got {len(resolution)} for {kspace.ndim - 1} spatial dims)."
        )

    out = kspace
    for axis, (size, target) in enumerate(zip(kspace.shape[1:], resolution), start=1):
        if target == size:
            continue
        if target < size:
            start = size // 2 - target // 2
            out = out.take(range(start, start + target), axis=axis)
        else:
            before = target // 2 - size // 2
            padding = [(0, 0)] * out.ndim
            padding[axis] = (before, target - size - before)
            out = np.pad(out, padding)
    return np.ascontiguousarray(out)


def morse_pi_cartesian(
    calib_kspace: NDArray,
    resolution: tuple[int, ...] | None = None,
    N_ref: int = 6,
    smoothing_sd: int = 3,
    noise_cov: NDArray | None = None,
    noise_whiten: bool = False,
    noise_colour_power: int = 0,
) -> NDArray:
    """Estimate MORSE-PI sensitivities from centre-origin Cartesian k-space.

    ``calib_kspace`` must be fully sampled and shaped ``(NCoils, *spatial)``
    for two or three spatial dimensions. No trajectory or NUFFT backend is used.
    """
    calib_kspace = np.asarray(calib_kspace)
    if calib_kspace.ndim not in (3, 4):
        raise ValueError("calib_kspace must have coil plus two or three spatial axes.")

    calib_kspace, noise_colour_factor = prepare_noise(
        calib_kspace, noise_cov, noise_whiten, noise_colour_power
    )
    if resolution is not None:
        calib_kspace = _resize_kspace(calib_kspace, resolution)
    return morse_pi_core(
        calib_kspace,
        N_ref=N_ref,
        smoothing_sd=smoothing_sd,
        noise_colour_factor=noise_colour_factor,
    )
