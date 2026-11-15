# SPDX-License-Identifier: GPL-3.0-or-later
"""PyTorch re-implementation of Gnuastro's NoiseChisel (2D images)."""
from .core import (
    DEFAULT_KERNEL,
    NoiseChisel,
    NoiseChiselParams,
    NoiseChiselResult,
    gaussian_kernel,
    noisechisel,
)
from .io import run_fits, write_fits

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_KERNEL",
    "NoiseChisel",
    "NoiseChiselParams",
    "NoiseChiselResult",
    "gaussian_kernel",
    "noisechisel",
    "run_fits",
    "write_fits",
    "__version__",
]
