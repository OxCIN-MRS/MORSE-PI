# MORSE-PI Coil Sensitivity Estimation

This repository is a third-party Python implementation of the [MORSE-PI](https://doi.org/10.48550/arXiv.2606.21336) coil sensitivity estimation method published by [Barbara Dymerska](https://profiles.ucl.ac.uk/67063-barbara-dymerska) and colleagues at the FIL, UCL.

The code accepts either non-Cartesian trajectories, using [MRI-nufft](https://github.com/mind-inria/mri-nufft), or already-gridded Cartesian calibration k-space.

## Install

```bash
pip install .
```

Or create the development environment with `pixi install`.

## Non-Cartesian k-space
```
from morse_pi import morse_pi

morse_pi_sens = morse_pi(
    k_data.reshape((N_coils, -1)),
    traj.reshape((-1, n_dim)),
    reconstruction_resolution,
    backend='torchkbnufft-cpu',
    noise_cov=noise_cov,
    noise_whiten=True,
    noise_colour_power=2)
```

Where `k_data` is the complex k-space data, `traj` is the k-space trajectory scaled to ±Pi, `reconstruction_resolution` is a tuple specifying the reconstruction resolution.

The output `morse_pi_sens` is the estimated complex coil sensitivites `(x, y, z, N_coils, NRef)`

`noise_cov` is an optional complex noise covariance matrix with shape
`(N_coils, N_coils)`. Set `noise_whiten=True` to pre-whiten k-space before
MORSE processing. `noise_colour_power` applies the lower Cholesky factor to
the first-order sensitivities only while constructing the VRC: `0` disables
colouring, `1` applies it once, and so on. When whitening is enabled, power
`1` restores inherent coil correlation and power `2` adds the paper's
enhanced-correlation step.

## Cartesian k-space

```python
from morse_pi import morse_pi_cartesian

morse_pi_sens = morse_pi_cartesian(
    calib_kspace,  # centre-origin, fully sampled: (N_coils, kx, ky[, kz])
    resolution=(128, 128),  # optional centre crop / zero-pad target
    noise_cov=noise_cov,
    noise_whiten=True,
    noise_colour_power=2,
)
```

The Cartesian interface does not need a trajectory, backend, or MRI-NUFFT.

Both interfaces select the VRC phase-reference voxel using the authors' masked
RSS-weighted centroid, rather than the array midpoint. Calibration RSS uses all
coils after any requested whitening and before VRC colouring. The mask is
`RSS > 0.8 * mean(RSS)`; the largest connected component (8-connectivity in 2D,
26-connectivity in 3D) supplies the RSS-weighted centroid. Coordinates are rounded
to the nearest voxel, with half ties rounded up. Equal-size components are
resolved in MATLAB column-major order. Calibration without positive magnitude
raises `ValueError`. This seed rule does not guarantee removal of every shared
phase singularity.

## Attribution
If you use this code in your own work please cite the original MORSE(-PI) publications.

1.  Dymerska B, Josephs O, James B, Malekian V, Graedel NN, Callaghan MF. MORSE-PI -- Flexible and artefact-free image reconstruction for structural and functional QSM and other phase-critical imaging applications. 2026. doi:10.48550/ARXIV.2606.21336
2.  Josephs O, Dymerska B, Graedel NN, Balbastre Y, Corbin N, Callaghan MF. MORSE: Multiple Orthogonal Reference Sensitivity Encoding. 2025. doi:10.48550/ARXIV.2510.09098

## Dependencies
- [MRI-nufft](https://github.com/mind-inria/mri-nufft)
- Numpy
- Scipy

Tested on Python 3.14

## License
This software is released, open-source under the MIT License.

## Other implementations
The original authors have their own `Matlab+Gadgetron`-based implementation which is available as described in their own publications:

```
The open-source code, as well as scripts necessary to compile a Gadgetron image and create a Docker container encapsulating the MORSE-PI reconstruction, are available on:
https://github.com/fil-physics/gadgetron-matlab.
```
