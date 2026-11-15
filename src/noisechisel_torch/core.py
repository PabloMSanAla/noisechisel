# SPDX-License-Identifier: GPL-3.0-or-later
"""
noisechisel_torch
=================

A PyTorch re-implementation of Gnuastro's **NoiseChisel** detection program
(2D images), written by following the C source of Gnuastro (``bin/noisechisel``
plus the ``lib/`` routines it calls: tile statistics, neighbour interpolation,
outlier rejection, sigma clipping, binary morphology, hole filling, ...).

It is *derived from Gnuastro* (GPLv3+, (C) Free Software Foundation) and is
therefore distributed under the same license.

Usage
-----
Function interface::

    from noisechisel_torch import noisechisel
    res = noisechisel(image, device="cuda")          # image: numpy array / tensor
    res.detections      # bool  (H, W)
    res.labels          # int32 (H, W), 0 = sky, 1..N = detections
    res.sky, res.std    # float32 (H, W) Sky and Sky STD
    res.info()          # number of detections, S/N threshold, STD range, ...

Class interface (reuse the same configuration on many images)::

    nc = NoiseChisel(device="cuda", tilesize=(64, 64), qthresh=0.3)
    res = nc(image)

Everything that is a NoiseChisel option keeps its Gnuastro name and default
(``tilesize``, ``qthresh``, ``erode``, ``snquant``, ``detgrowquant`` ...).

What runs where
---------------
Convolution, tile statistics (sorting / quantiles / sigma clipping), neighbour
interpolation, thresholding, erosion / dilation / opening, hole filling, S/N
measurement and growth are all PyTorch tensor operations (so they run on GPU
when ``device="cuda"``).  Connected-component labelling has no efficient
primitive in PyTorch (nor in torchvision); two back-ends are provided:
``ccl="cupy"`` (CUDA, ``cupyx.scipy.ndimage.label``, zero-copy through DLPack)
and ``ccl="scipy"`` (CPU; on CUDA the mask is copied to the host and back).
``ccl="auto"`` picks cupy on CUDA when it is installed, scipy otherwise.  Both
give identical labels (raster order).

Fidelity: the arithmetic follows Gnuastro 0.24 closely enough to reproduce
its output bit for bit on real data (float32 products accumulated in double
in the convolution, Gnuastro's neighbour visiting order for interpolation and
outlier rejection, sigma/MAD clipping exactly as ``gal_statistics_clip_*``).
Apple MPS has no float64, so ``device="mps"`` falls back to the CPU.

Known differences from the C program: 2D only, single channel
(``numchannels=1,1``); NaN pixels are treated as never-detected background
(``--blankasforeground`` is not implemented); no check images.
"""
from __future__ import annotations

import heapq
import math
import time
import warnings
from functools import lru_cache
from dataclasses import dataclass, fields, replace
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "NoiseChisel",
    "NoiseChiselParams",
    "NoiseChiselResult",
    "noisechisel",
    "gaussian_kernel",
    "DEFAULT_KERNEL",
]

# --------------------------------------------------------------------------
# Default kernel: the 11x11 sharp Gaussian (FWHM=2 pix) shipped with Gnuastro
# (bin/noisechisel/kernel-2d.h), copied verbatim.
# --------------------------------------------------------------------------
DEFAULT_KERNEL = np.array([
        [0, 0, 0, 0, 0, 2.599797e-08, 0, 0, 0, 0, 0],
        [0, 0, 3.008479e-08, 6.938075e-07, 4.493532e-06, 8.276223e-06, 4.515019e-06, 6.947793e-07, 3.04628e-08, 0, 0],
        [0, 3.009687e-08, 2.556034e-06, 5.936867e-05, 0.0003808578, 0.0007126221, 0.0003827095, 5.902729e-05, 2.553342e-06, 2.978137e-08, 0],
        [0, 7.021852e-07, 5.912285e-05, 0.00137637, 0.008863639, 0.01648383, 0.008855942, 0.001365171, 5.925718e-05, 7.021184e-07, 0],
        [0, 4.490787e-06, 0.0003826718, 0.008857355, 0.05742518, 0.1062628, 0.05727194, 0.008880079, 0.0003826067, 4.478989e-06, 0],
        [2.595735e-08, 8.31301e-06, 0.0007113572, 0.01640853, 0.1061298, 0.1971036, 0.1062611, 0.01647962, 0.000708363, 8.379878e-06, 2.593496e-08],
        [0, 4.516684e-06, 0.0003846966, 0.008860709, 0.05739478, 0.1062216, 0.05725683, 0.00881713, 0.000383981, 4.473017e-06, 0],
        [0, 6.950547e-07, 5.920586e-05, 0.00137483, 0.00887785, 0.0164709, 0.008855232, 0.001372743, 5.939038e-05, 7.016624e-07, 0],
        [0, 3.006322e-08, 2.587011e-06, 5.92911e-05, 0.0003843824, 0.0007118155, 0.000386519, 5.974654e-05, 2.585581e-06, 3.048036e-08, 0],
        [0, 0, 3.041056e-08, 7.05225e-07, 4.497418e-06, 8.388542e-06, 4.478833e-06, 7.018358e-07, 2.995504e-08, 0, 0],
        [0, 0, 0, 0, 0, 2.567377e-08, 0, 0, 0, 0, 0]
], dtype=np.float32)


def gaussian_kernel(fwhm: float, size: Optional[int] = None) -> np.ndarray:
    """Normalised 2D Gaussian kernel (odd size). Handy for ``widekernel``."""
    sigma = fwhm / 2.354820045
    if size is None:
        size = 2 * int(math.ceil(3 * fwhm)) + 1
    if size % 2 == 0:
        size += 1
    r = np.arange(size) - size // 2
    yy, xx = np.meshgrid(r, r, indexing="ij")
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


# ==========================================================================
# Parameters and result container
# ==========================================================================
@dataclass
class NoiseChiselParams:
    """NoiseChisel options. Names and defaults follow Gnuastro 0.2x
    (``astnoisechisel -P``), 2D case."""

    # Tessellation
    tilesize: Tuple[int, int] = (30, 30)       # --tilesize (y, x)
    remainderfrac: float = 0.1                 # --remainderfrac
    largetilesize: Optional[Tuple[int, int]] = (200, 200)  # --largetilesize
    # (None: fill holes / open pseudo-detections on the whole image instead
    #  of per large tile; avoids tile-edge artefacts, deviates from C.)

    # Interpolation / smoothing of tile values
    interpnumngb: int = 15                     # --interpnumngb
    interpmetric: str = "radial"               # --interpmetric (radial|manhattan)
    interponlyblank: bool = False              # --interponlyblank
    smoothwidth: int = 5                       # --smoothwidth

    # Quantile threshold / initial detection
    meanmedqdiff: float = 0.01                 # -Q
    qthresh: float = 0.3                       # -t
    noerodequant: float = 0.99                 # --noerodequant
    outliernumngb: int = 15                    # --outliernumngb
    outliersigma: float = 5.0                  # --outliersigma
    outliersclip: Tuple[float, float] = (3.0, 0.2)   # --outliersclip
    erode: int = 2                             # -e
    erodengb: int = 4                          # --erodengb
    opening: int = 1                           # -p
    openingngb: int = 8                        # --openingngb

    # Sky / STD
    minskyfrac: float = 0.7                    # -B
    skyfracnoblank: bool = False               # --skyfracnoblank
    sigmaclip: Tuple[float, float] = (3.0, 0.2)      # -s

    # Pseudo-detections / false-detection removal
    dthresh: float = 0.0                       # -R
    dopening: int = 1                          # --dopening
    dopeningngb: int = 4                       # --dopeningngb
    holengb: int = 8                           # --holengb
    pseudoconcomp: int = 4                     # --pseudoconcomp
    snminarea: int = 10                        # -m
    minnumfalse: int = 100                     # --minnumfalse
    snquant: float = 0.99                      # -c
    snthresh: Optional[float] = None           # --snthresh (None: measure)

    # Growth
    detgrowquant: float = 0.90                 # -d  (1.0 disables growth)
    detgrowmaxholesize: int = 100              # --detgrowmaxholesize
    cleangrowndet: bool = False                # --cleangrowndet


@dataclass
class NoiseChiselResult:
    """Output of :class:`NoiseChisel`. All tensors live on the run device.

    ``sky`` and ``std`` are full-resolution images (each pixel holds the value
    of its tile, NaN where the input is blank), like Gnuastro's ``SKY`` and
    ``SKY_STD`` outputs.  ``info()`` prints a summary."""

    detections: torch.Tensor          # bool    (H, W)
    labels: torch.Tensor              # int32   (H, W), 0 = not detected
    sky: torch.Tensor                 # float32 (H, W) Sky
    std: torch.Tensor                 # float32 (H, W) Sky STD
    num_detections: int
    num_initial_detections: int
    det_sn_threshold: float           # S/N threshold of true pseudo-detections
    min_std: float
    med_std: float
    max_std: float
    sky_tiles: torch.Tensor           # float32 (ny, nx) one Sky value per tile
    std_tiles: torch.Tensor           # float32 (ny, nx) one Sky STD value per tile
    params: Optional[NoiseChiselParams] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.detections.shape)

    def meta(self) -> dict:
        """Scalar results (what ``info()`` prints) as a dict."""
        sky = self.sky_tiles[torch.isfinite(self.sky_tiles)]
        return {
            "shape": self.shape,
            "device": str(self.detections.device),
            "tile_grid": tuple(self.sky_tiles.shape),
            "num_initial_detections": self.num_initial_detections,
            "num_detections": self.num_detections,
            "detected_fraction": float(self.detections.float().mean()),
            "det_sn_threshold": self.det_sn_threshold,
            "median_sky": float(sky.median()) if sky.numel() else float("nan"),
            "min_std": self.min_std,
            "med_std": self.med_std,
            "max_std": self.max_std,
        }

    def info(self, params: bool = False) -> None:
        """Print the scalar results (and the options used with ``params=True``)."""
        rows = list(self.meta().items())
        if params and self.params is not None:
            rows += [(f.name, getattr(self.params, f.name)) for f in fields(self.params)]
        w = max(len(k) for k, _ in rows)
        print("NoiseChisel result")
        for k, v in rows:
            if isinstance(v, float):
                v = f"{v:.6g}"
            print(f"  {k:<{w}}  {v}")

    def numpy(self) -> dict:
        """Convenience: the images as numpy arrays."""
        return {k: getattr(self, k).cpu().numpy() for k in ("detections", "labels", "sky", "std")}


# ==========================================================================
# Tessellation
# ==========================================================================
def _tile_sizes(n: int, regular: int, remainderfrac: float):
    """Sizes of the tiles along one axis, replicating
    ``gal_tile_full_regular_first`` of Gnuastro."""
    if regular >= n:
        return [n]
    rem = n % regular
    if rem == 0:
        return [regular] * (n // regular)
    if rem > remainderfrac * regular:
        nt = n // regular + 1
        first = (rem + regular) // 2
        last = n - (first + regular * (nt - 2))
        return [first] + [regular] * (nt - 2) + [last]
    nt = n // regular
    first = rem + regular
    return [first] + [regular] * (nt - 1)


class _Grid:
    """Regular tessellation of an (H, W) image (one channel)."""

    def __init__(self, shape, tilesize, remainderfrac, device):
        H, W = shape
        self.shape = (H, W)
        self.device = device
        self.hs = _tile_sizes(H, int(tilesize[0]), remainderfrac)
        self.ws = _tile_sizes(W, int(tilesize[1]), remainderfrac)
        self.ny, self.nx = len(self.hs), len(self.ws)
        hs_t = torch.tensor(self.hs, device=device)
        ws_t = torch.tensor(self.ws, device=device)
        self.row_tile = torch.repeat_interleave(torch.arange(self.ny, device=device), hs_t)
        self.col_tile = torch.repeat_interleave(torch.arange(self.nx, device=device), ws_t)
        y0 = torch.cumsum(hs_t, 0) - hs_t
        x0 = torch.cumsum(ws_t, 0) - ws_t
        self.maxh, self.maxw = max(self.hs), max(self.ws)
        a = torch.arange(self.maxh, device=device)
        b = torch.arange(self.maxw, device=device)
        self.rows_idx = (y0[:, None] + a[None, :]).clamp(max=H - 1)
        self.cols_idx = (x0[:, None] + b[None, :]).clamp(max=W - 1)
        self.rows_valid = a[None, :] < hs_t[:, None]
        self.cols_valid = b[None, :] < ws_t[:, None]
        self.area = hs_t[:, None] * ws_t[None, :]          # (ny, nx)
        # Tile of each row/column as given by gal_tile_full_id_from_coord,
        # used by NoiseChisel's S/N measurement: when the first tile is not
        # of regular size, the coordinate equal to the first tile's size
        # (the first pixel of the 2nd tile) is attributed to the 1st tile.
        self.row_tile_c = self._id_from_coord(H, self.hs, int(tilesize[0]), device)
        self.col_tile_c = self._id_from_coord(W, self.ws, int(tilesize[1]), device)

    @staticmethod
    def _id_from_coord(n, sizes, regular, device):
        c = torch.arange(n, device=device)
        first = sizes[0]
        if first == regular or len(sizes) == 1:
            return (c // regular).clamp(max=len(sizes) - 1)
        return torch.where(c <= first, torch.zeros_like(c), 1 + (c - first) // regular)

    def expand(self, v: torch.Tensor) -> torch.Tensor:
        """(ny, nx) tile values -> (H, W) image (value of the pixel's tile)."""
        return v[self.row_tile[:, None], self.col_tile[None, :]]

    def rows_per_chunk(self, max_elems: int) -> int:
        per_row = self.nx * self.maxh * self.maxw
        return max(1, int(max_elems // per_row))

    def gather(self, img: torch.Tensor, r0: int, r1: int, fill):
        """Tiles of tile-rows [r0, r1) as a (k*nx, maxh*maxw) tensor, padded
        with ``fill`` (NaN for float images, bool for masks)."""
        ri = self.rows_idx[r0:r1]
        t = img[ri[:, None, :, None], self.cols_idx[None, :, None, :]]
        valid = self.rows_valid[r0:r1][:, None, :, None] & self.cols_valid[None, :, None, :]
        if t.dtype == torch.bool:
            t = (t | ~valid) if fill else (t & valid)
        else:
            t = torch.where(valid, t, torch.full((), fill, dtype=t.dtype, device=t.device))
        return t.reshape(-1, self.maxh * self.maxw)


# ==========================================================================
# Per-row (per-tile) statistics on NaN-padded (T, P) tensors
# ==========================================================================
def _sort_rows(x: torch.Tensor):
    """Sort rows ascending with NaN (blank/padding) moved to the end as +inf.
    Returns (sorted, number of non-blank values per row)."""
    isn = torch.isnan(x)
    n = (~isn).sum(1)
    xs = torch.where(isn, torch.full((), float("inf"), dtype=x.dtype, device=x.device), x)
    xs, _ = torch.sort(xs, dim=1)
    return xs, n


def _quantile_index(n: torch.Tensor, q: float) -> torch.Tensor:
    """``gal_statistics_quantile_index``: round-half-down of (n-1)*q."""
    fi = (n - 1).clamp(min=0).double() * q
    fl = torch.floor(fi)
    return (fl + ((fi - fl) > 0.5).double()).long()


def _row_quantile(xs: torch.Tensor, n: torch.Tensor, q: float) -> torch.Tensor:
    idx = _quantile_index(n, q).clamp(0, xs.shape[1] - 1)
    v = xs.gather(1, idx[:, None]).squeeze(1)
    return torch.where(n > 0, v, torch.full_like(v, float("nan")))


def _range_mean_std(xd: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    """Mean and STD of the sorted range [lo, hi) of each float64 row, with
    Gnuastro's arithmetic: sequential double sums of the values and their
    squares (``gal_statistics_mean`` / ``gal_statistics_std_from_sums``).
    A row-wise ``cumsum`` is sequential, so masked values (0) do not change
    the rounding of the sums."""
    pos = torch.arange(xd.shape[1], device=xd.device)[None, :]
    inr = (pos >= lo[:, None]) & (pos < hi[:, None])
    v = torch.where(inr, xd, torch.zeros((), dtype=xd.dtype, device=xd.device))
    s1 = torch.cumsum(v, 1)[:, -1]
    s2 = torch.cumsum(v * v, 1)[:, -1]
    cnt = hi - lo
    cntf = cnt.clamp(min=1).to(xd.dtype)
    mean = s1 / cntf
    ss = s1 * s1 / cntf
    std = torch.where(ss > s2, torch.zeros_like(s2), torch.sqrt(((s2 - ss) / cntf).clamp(min=0)))
    std = torch.where(cnt == 1, torch.zeros_like(std), std)
    return mean, std


def _range_median(xs, lo, cnt):
    """Median of the sorted range [lo, lo+cnt) in the type of ``xs``
    (``statistics_median_in_sorted_no_blank``)."""
    P = xs.shape[1]
    ia = (lo + cnt // 2).clamp(0, P - 1)
    ib = (lo + cnt // 2 - 1).clamp(0, P - 1)
    va = xs.gather(1, ia[:, None]).squeeze(1)
    vb = xs.gather(1, ib[:, None]).squeeze(1)
    return torch.where(cnt % 2 == 1, va, (va + vb) / 2)


def _clip_range(xd, lo, hi, center, spread, mult):
    """``CLIPALL``: keep center - mult*spread < x < center + mult*spread."""
    tlo = (center - mult * spread).contiguous()
    thi = (center + mult * spread).contiguous()
    nlo = torch.searchsorted(xd, tlo[:, None], right=True).squeeze(1)
    nhi = torch.searchsorted(xd, thi[:, None], right=False).squeeze(1)
    return torch.maximum(lo, nlo), torch.minimum(hi, nhi)


def _sigma_clip_rows(xs: torch.Tensor, n: torch.Tensor, mult: float, param: float,
                     max_converge: int = 50):
    """Vectorised ``gal_statistics_clip_sigma`` (Gnuastro 0.24) on sorted rows
    (+inf padded).  Clipping is around the median; the loop stops on zero
    STD, by tolerance (``param < 1``) or after ``param`` rounds.  The output
    statistics are those of the last measured round (mean of that set).
    Returns (median, mean, std) as float64; NaN on failure."""
    T, P = xs.shape
    dev = xs.device
    nan = float("nan")
    mult, param = float(np.float32(mult)), float(np.float32(param))   # C 'float' args
    bytol = param < 1.0
    maxnum = max_converge if bytol else int(param)
    xd = xs.double()
    lo = torch.zeros(T, dtype=torch.long, device=dev)
    hi = n.clone().long()
    nanv = lambda: torch.full((T,), nan, dtype=torch.float64, device=dev)
    out_med, out_mean, out_std = nanv(), nanv(), nanv()
    old_std = nanv()
    alive = n > 0
    num = 0
    while num < maxnum:
        alive = alive & (hi > lo)
        if not bool(alive.any()):
            break
        cnt = hi - lo
        med = _range_median(xs, lo, cnt).double()
        mean, std = _range_mean_std(xd, lo, hi)
        out_med = torch.where(alive, med, out_med)
        out_mean = torch.where(alive, mean, out_mean)
        out_std = torch.where(alive, std, out_std)
        stop = std == 0
        if bytol and num > 0:
            stop = stop | (((old_std - std) / std) < param)
        alive = alive & ~stop
        nlo, nhi = _clip_range(xd, lo, hi, med, std, mult)
        lo = torch.where(alive, nlo, lo)
        hi = torch.where(alive, nhi, hi)
        old_std = std
        num += 1
    # Not converged by tolerance (or emptied): NaN, as Gnuastro.
    fail = (n == 0) | (hi <= lo)
    if bytol:
        fail = fail | alive
    out_med = torch.where(fail, torch.full_like(out_med, nan), out_med)
    out_mean = torch.where(fail, torch.full_like(out_mean, nan), out_mean)
    out_std = torch.where(fail, torch.full_like(out_std, nan), out_std)
    return out_med, out_mean, out_std


def _mad_clip_rows(xs: torch.Tensor, n: torch.Tensor, mult: float, param: float,
                   max_converge: int = 50):
    """Vectorised ``gal_statistics_clip_mad`` with the extra STD column (as
    used by ``gal_statistics_outlier_bydistance``) on sorted float64 rows
    (+inf padded).  Returns (median, std of the final set), NaN on failure."""
    T, P = xs.shape
    dev = xs.device
    nan = float("nan")
    inf = float("inf")
    mult, param = float(np.float32(mult)), float(np.float32(param))   # C 'float' args
    bytol = param < 1.0
    maxnum = max_converge if bytol else int(param)
    xd = xs.double()
    pos = torch.arange(P, device=dev)[None, :]
    lo = torch.zeros(T, dtype=torch.long, device=dev)
    hi = n.clone().long()
    old_size = torch.full((T,), -1, dtype=torch.long, device=dev)
    old_spread = torch.full((T,), nan, dtype=torch.float64, device=dev)
    center = torch.full((T,), nan, dtype=torch.float64, device=dev)
    alive = n > 0
    num = 0
    while num < maxnum:
        alive = alive & (hi > lo)
        if not bool(alive.any()):
            break
        cnt = hi - lo
        med = _range_median(xd, lo, cnt)
        inr = (pos >= lo[:, None]) & (pos < hi[:, None])
        dv = torch.where(inr, (xd - med[:, None]).abs(), torch.full((), inf, dtype=xd.dtype, device=dev))
        dv, _ = torch.sort(dv, dim=1)
        mad = _range_median(dv, torch.zeros_like(lo), cnt)
        # MAD == 0: use the distance of the nearest values around the median.
        madzero = mad == 0
        if bool((madzero & alive).any()):
            ia = torch.searchsorted(xd, med[:, None].contiguous(), right=False).squeeze(1) - 1
            ib = torch.searchsorted(xd, med[:, None].contiguous(), right=True).squeeze(1)
            has_a = ia >= lo
            has_b = ib < hi
            av = xd.gather(1, ia.clamp(0, P - 1)[:, None]).squeeze(1)
            bv = xd.gather(1, ib.clamp(0, P - 1)[:, None]).squeeze(1)
            da, db = med - av, bv - med
            sq = torch.sqrt(med.abs())
            both = torch.where(da / db > 3, db, torch.where(db / da > 3, da, bv - av))
            alt = torch.where(has_a & has_b, both,
                  torch.where(has_b, torch.minimum(db, sq),
                  torch.where(has_a, torch.minimum(da, sq), torch.zeros_like(med))))
            mad = torch.where(madzero, alt, mad)
        center = torch.where(alive, med, center)
        stop = (mad == 0) | (madzero & (cnt == old_size))
        if bytol and num > 0:
            stop = stop | (~madzero & (((old_spread - mad) / mad) < param))
        alive = alive & ~stop
        old_size = cnt
        nlo, nhi = _clip_range(xd, lo, hi, med, mad, mult)
        lo = torch.where(alive, nlo, lo)
        hi = torch.where(alive, nhi, hi)
        old_spread = mad
        num += 1
    _, std = _range_mean_std(xd, lo, hi)
    fail = (n == 0) | (hi <= lo)
    if bytol:
        fail = fail | alive
    center = torch.where(fail, torch.full_like(center, nan), center)
    std = torch.where(fail, torch.full_like(std, nan), std)
    # The output table of Gnuastro is float32.
    return center.float().double(), std.float().double()


def _outlier_threshold(vals_sorted: torch.Tensor, sigma: float, clip_mult: float,
                       clip_param: float, max_elems: int = 8_000_000):
    """``gal_statistics_outlier_bydistance`` (positive side, window = 1/3 of
    the values) on an ascending float32 1D tensor.  Returns the value just
    below the first outlier, or None."""
    a = vals_sorted.float()
    m = a.numel()
    w = m // 3
    if w <= 2:
        return None
    d = (a[1:] - a[:-1]).double()                    # float32 differences
    win = d.unfold(0, w - 1, 1)                      # (m-w+1, w-1) view
    B = max(1, max_elems // (w - 1))
    sigma = float(np.float32(sigma))
    for s0 in range(0, m - w, B):
        s1 = min(m - w, s0 + B)
        rows, _ = torch.sort(win[s0:s1], dim=1)
        nn = torch.full((rows.shape[0],), w - 1, dtype=torch.long, device=a.device)
        med, std = _mad_clip_rows(rows, nn, clip_mult, clip_param)
        cur = d[s0 + w - 1: s1 + w - 1]              # a[i]-a[i-1], i = s + w
        hit = (cur - med) > (torch.tensor(sigma, dtype=torch.float32) * std.float()).double()
        if bool(hit.any()):
            k = int(torch.nonzero(hit)[0])
            i = s0 + k + w
            return float(a[i - 1])
    return None


# ==========================================================================
# Neighbour search over the tile grid (K nearest *valid* tiles)
# ==========================================================================
@lru_cache(maxsize=16)
def _gnuastro_order(R: int, metric: str) -> np.ndarray:
    """Order in which Gnuastro's neighbour search (``gal_interpolate_neighbors``
    and ``gal_tileinternal_no_outlier_local``) visits the tiles around a
    centre, as (dy, dx) offsets within a (2R+1)^2 window.

    Gnuastro pops tiles from a distance-sorted list (FIFO among equal
    distances) and pushes the 4-connected neighbours of each popped tile.
    The centre is not flagged when it is first pushed, so it is pushed (and
    counted) a second time right after the first neighbour is popped.  The
    relative order does not depend on the image edges (the first-popped
    neighbour of any tile is always closer to the centre), so one window
    serves every tile; out-of-image offsets are simply skipped later.
    Only offsets within distance R are returned (complete disc)."""
    W = 2 * R + 1
    checked = np.zeros((W, W), dtype=bool)
    heap = [(0.0, 0, R, R)]
    seq = 1
    order = []
    while heap:
        _, _, y, x = heapq.heappop(heap)
        order.append((y - R, x - R))
        for ny_, nx_ in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny_ < W and 0 <= nx_ < W and not checked[ny_, nx_]:
                dy, dx = ny_ - R, nx_ - R
                d = math.sqrt(dy * dy + dx * dx) if metric == "radial" else abs(dy) + abs(dx)
                heapq.heappush(heap, (float(np.float32(d)), seq, ny_, nx_))
                seq += 1
                checked[ny_, nx_] = True
    order = np.asarray(order, dtype=np.int64)
    # Keep the disc inside the window: beyond R the window is incomplete.
    dy, dx = order[:, 0], order[:, 1]
    d = np.sqrt(dy * dy + dx * dx) if metric == "radial" else np.abs(dy) + np.abs(dx)
    return order[d <= R]


def _knn_values(arrays, valid: torch.Tensor, K: int, metric: str,
                max_elems: int = 20_000_000):
    """For every tile, the values of each array at the first K valid tiles in
    Gnuastro's visiting order (:func:`_gnuastro_order`).  Returns a list of
    (ny, nx, K) tensors."""
    ny, nx = valid.shape
    dev = valid.device
    nvalid = int(valid.sum())
    if nvalid < K:
        raise ValueError(f"only {nvalid} usable tiles, but {K} neighbours are required "
                         "(decrease `interpnumngb`/`tilesize`, or increase "
                         "`meanmedqdiff` or the outlier thresholds)")
    outs = [torch.empty((ny, nx, K), dtype=a.dtype, device=dev) for a in arrays]
    flat_valid = valid.reshape(-1)
    flat_arrays = [a.reshape(-1) for a in arrays]
    density = max(nvalid / (ny * nx), 1e-6)
    R = int(max(3, math.ceil(math.sqrt(2 * K / (math.pi * density)))))
    todo = torch.arange(ny * nx, device=dev)
    Rmax = max(ny, nx)
    while todo.numel():
        R = min(R, Rmax)
        off = torch.as_tensor(_gnuastro_order(R, metric), device=dev)
        L = off.shape[0]
        step = max(1, max_elems // L)
        failed = []
        for s0 in range(0, todo.numel(), step):
            t = todo[s0:s0 + step]
            yy = (t // nx)[:, None] + off[None, :, 0]
            xx = (t % nx)[:, None] + off[None, :, 1]
            ing = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            idx = (yy.clamp(0, ny - 1) * nx + xx.clamp(0, nx - 1))
            ok = ing & flat_valid[idx]
            cs = torch.cumsum(ok, 1)
            enough = cs[:, -1] >= K
            take = ok & (cs <= K)
            # positions of the first K valid entries, in visiting order
            pos = torch.argsort((~take).to(torch.int8), dim=1, stable=True)[:, :K]
            sel = idx.gather(1, pos)
            g = enough.nonzero().squeeze(1)
            ty, tx = t[g] // nx, t[g] % nx
            for o, fa in zip(outs, flat_arrays):
                o[ty, tx] = fa[sel[g]]
            if not bool(enough.all()):
                failed.append(t[~enough])
        todo = torch.cat(failed) if failed else todo[:0]
        if todo.numel() and R >= Rmax:
            raise ValueError("not enough valid tiles for the neighbour search")
        R *= 2
    return outs


def _median_last(v: torch.Tensor) -> torch.Tensor:
    """Median over the last dim (mean of the two middle values if even)."""
    K = v.shape[-1]
    s, _ = torch.sort(v, dim=-1)
    return (s[..., (K - 1) // 2] + s[..., K // 2]) / 2


def _smooth_tiles(v: torch.Tensor, width: int) -> torch.Tensor:
    """Flat (box) smoothing of tile values with edge correction
    (``gal_tile_full_values_smooth``: kernel values are float(1/width^2))."""
    if width <= 1:
        return v
    k = torch.full((width, width), 1.0 / (width * width), dtype=torch.float32, device=v.device)
    return _convolve(v.to(torch.float32), k)


def _interp_smooth(arrays, K, metric, onlyblank, smoothwidth):
    """``threshold_interp_smooth``: replace each tile value by the median of
    its K nearest valid tiles (all tiles, unless ``onlyblank``), then smooth
    with a flat kernel.  Blank structure is taken from ``arrays[0]``."""
    valid = ~torch.isnan(arrays[0])
    arrays = [torch.where(valid, a, torch.full_like(a, float("nan"))) for a in arrays]
    knn = _knn_values(arrays, valid, K, metric)
    out = []
    for a, kv in zip(arrays, knn):
        m = _median_last(kv)
        if onlyblank:
            m = torch.where(valid, a, m)
        out.append(_smooth_tiles(m, smoothwidth))
    return out


# ==========================================================================
# Convolution (NaN aware, with edge correction as in gal_convolve_spatial)
# ==========================================================================
def _acc_dtype(device) -> torch.dtype:
    """float64 accumulator, except on backends without float64 (MPS)."""
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def _tap_sum(x: torch.Tensor, kernel: torch.Tensor, H: int, W: int, acc_dtype):
    """Sum over kernel taps (raster order) of float32 products, accumulated
    in ``acc_dtype``: the arithmetic of ``gal_convolve_spatial``.  ``x`` is
    the input zero-padded by half the kernel size on each side."""
    kh, kw = kernel.shape
    acc = torch.zeros((H, W), dtype=acc_dtype, device=x.device)
    for i, row in enumerate(kernel.tolist()):
        for j, v in enumerate(row):
            if v != 0.0:                 # adding +0.0 never changes the sum
                acc += x[i:i + H, j:j + W] * v
    return acc


def _convolve(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``gal_convolve_spatial`` with edge correction (2D, NaN aware): the
    kernel is applied as a correlation, each float32 product is accumulated
    in double and divided by the sum of the overlapping (non-NaN) kernel
    values.  This reproduces Gnuastro bit by bit (float64 ``conv2d`` would
    differ by an ulp, which matters when Sky >> noise)."""
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sizes must be odd")
    H, W = img.shape
    ph, pw = kh // 2, kw // 2
    acc_dtype = _acc_dtype(img.device)
    kernel = kernel.to(torch.float32)
    ok = ~torch.isnan(img)
    allok = bool(ok.all())
    x = img if allok else torch.where(ok, img, torch.zeros_like(img))
    x = F.pad(x.to(torch.float32)[None, None], (pw, pw, ph, ph))[0, 0]
    num = _tap_sum(x, kernel, H, W, acc_dtype)
    if allok:
        den = _edge_ksum(kernel, H, W, acc_dtype, img.device)
    else:
        m = F.pad(ok.to(torch.float32)[None, None], (pw, pw, ph, ph))[0, 0]
        den = _tap_sum(m, kernel, H, W, acc_dtype)
    out = (num / den).to(torch.float32)
    bad = (~ok) | (den == 0)
    return torch.where(bad, torch.full_like(out, float("nan")), out)


def _edge_ksum(kernel, H, W, acc_dtype, device):
    """Per-pixel sum of the kernel values that fall inside an all-valid
    image (same summation order as :func:`_tap_sum`).  It only depends on
    the distance to the borders, so it is computed on a small canvas."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    hs = H if H <= 4 * ph + 1 else 4 * ph + 1
    ws = W if W <= 4 * pw + 1 else 4 * pw + 1
    ones = F.pad(torch.ones((1, 1, hs, ws), dtype=torch.float32, device=device),
                 (pw, pw, ph, ph))[0, 0]
    small = _tap_sum(ones, kernel, hs, ws, acc_dtype)

    def index_map(n, ns, p):
        r = torch.arange(n, device=device)
        if n == ns:
            return r
        return torch.where(r < p, r, torch.where(r >= n - p, ns - (n - r), torch.full_like(r, 2 * p)))

    return small[index_map(H, hs, ph)[:, None], index_map(W, ws, pw)[None, :]]


# ==========================================================================
# Binary morphology (bool tensors) with Gnuastro's border semantics
# ==========================================================================
def _dilate1(m: torch.Tensor, conn: int) -> torch.Tensor:
    """One dilation; pixels outside the array count as background."""
    if conn == 4:
        out = m.clone()
        out[1:] |= m[:-1]
        out[:-1] |= m[1:]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        return out
    h = m.clone()
    h[:, 1:] |= m[:, :-1]
    h[:, :-1] |= m[:, 1:]
    out = h.clone()
    out[1:] |= h[:-1]
    out[:-1] |= h[1:]
    return out


def _check_conn(conn: int):
    if conn not in (4, 8):
        raise ValueError("connectivity must be 4 or 8 in 2D")
    return conn


# ==========================================================================
# Connected components
# ==========================================================================
def _label_scipy(mask: torch.Tensor, conn: int):
    from scipy import ndimage
    st = ndimage.generate_binary_structure(2, 1 if conn == 4 else 2)
    lab, n = ndimage.label(mask.cpu().numpy(), structure=st)
    return torch.from_numpy(lab.astype(np.int32, copy=False)).to(mask.device), int(n)


def _raster_relabel(lab: torch.Tensor, n: int) -> torch.Tensor:
    """Renumber labels 1..n in raster order of each component's first pixel
    (the Gnuastro / scipy convention)."""
    if n == 0:
        return lab
    flat = lab.reshape(-1).long()
    N = flat.numel()
    first = torch.full((n + 1,), N, dtype=torch.long, device=lab.device)
    first.scatter_reduce_(0, flat, torch.arange(N, device=lab.device), reduce="amin")
    order = torch.argsort(first[1:])
    new = torch.zeros(n + 1, dtype=torch.int32, device=lab.device)
    new[1 + order] = torch.arange(1, n + 1, dtype=torch.int32, device=lab.device)
    return new[flat].view(lab.shape)


def _cupy_available() -> bool:
    try:
        import cupy                        # noqa: F401
        from cupyx.scipy import ndimage    # noqa: F401
    except Exception:
        return False
    return True


def _label_cupy(mask: torch.Tensor, conn: int):
    """``cupyx.scipy.ndimage.label`` on a CUDA tensor (no host copies)."""
    import cupy
    from cupyx.scipy import ndimage
    st = cupy.asarray(ndimage.generate_binary_structure(2, 1 if conn == 4 else 2))
    with cupy.cuda.Device(mask.device.index or 0):
        m = cupy.from_dlpack(mask.contiguous().to(torch.uint8))
        lab, n = ndimage.label(m, structure=st, output=cupy.int32)
        lab = torch.from_dlpack(lab)
    return _raster_relabel(lab, int(n)), int(n)


# ==========================================================================
# Tiled canvas: run per-tile binary ops with ONE global tensor op.
# Tiles are laid out with 1-pixel "gap" lines between them; ops restore the
# gap value after each iteration so tiles never exchange information (this
# reproduces Gnuastro's per-large-tile fill-holes / opening exactly).
# ==========================================================================
class _Canvas:
    def __init__(self, shape, tilesize, remainderfrac, device):
        H, W = shape
        self.shape = (H, W)
        if tilesize is None:
            hs, ws = [H], [W]
        else:
            hs = _tile_sizes(H, int(tilesize[0]), remainderfrac)
            ws = _tile_sizes(W, int(tilesize[1]), remainderfrac)
        ny, nx = len(hs), len(ws)
        self.trivial = (ny == 1 and nx == 1)
        hs_t = torch.tensor(hs, device=device)
        ws_t = torch.tensor(ws, device=device)
        rt = torch.repeat_interleave(torch.arange(ny, device=device), hs_t)
        ct = torch.repeat_interleave(torch.arange(nx, device=device), ws_t)
        self.rmap = torch.arange(H, device=device) + rt
        self.cmap = torch.arange(W, device=device) + ct
        Hc, Wc = H + ny - 1, W + nx - 1
        y0 = torch.cumsum(hs_t, 0) - hs_t
        x0 = torch.cumsum(ws_t, 0) - ws_t
        brow = torch.zeros(Hc, dtype=torch.bool, device=device)
        bcol = torch.zeros(Wc, dtype=torch.bool, device=device)
        brow[self.rmap[y0]] = True
        brow[self.rmap[y0 + hs_t - 1]] = True
        bcol[self.cmap[x0]] = True
        bcol[self.cmap[x0 + ws_t - 1]] = True
        if self.trivial:
            self.gap = None
            self.border = brow[:, None] | bcol[None, :]
        else:
            gap = torch.ones((Hc, Wc), dtype=torch.bool, device=device)
            gap[self.rmap[:, None], self.cmap[None, :]] = False
            self.gap = gap
            self.border = (brow[:, None] | bcol[None, :]) & ~gap

    def to(self, x, fill=False):
        if self.trivial:
            return x.clone()
        Hc, Wc = self.border.shape
        c = torch.full((Hc, Wc), fill, dtype=torch.bool, device=x.device)
        c[self.rmap[:, None], self.cmap[None, :]] = x
        return c

    def back(self, c):
        return c if self.trivial else c[self.rmap[:, None], self.cmap[None, :]]

    # -- operations on canvas arrays (gap pixels hold the "outside" value) --
    def _erode_c(self, c, n, conn, fixed=None):
        for _ in range(n):
            c = c & ~_dilate1(~c, conn)
            if self.gap is not None:
                c |= self.gap
            if fixed is not None:
                c |= fixed
        return c

    def _dilate_c(self, c, n, conn):
        for _ in range(n):
            c = _dilate1(c, conn)
            if self.gap is not None:
                c &= ~self.gap
        return c

    def erode(self, m, n, conn, fixed=None):
        _check_conn(conn)
        c = self.to(m, fill=True)
        f = None if fixed is None else self.to(fixed, fill=False)
        c = self._erode_c(c, n, conn, f)
        if self.gap is not None:
            c &= ~self.gap
        return self.back(c)

    def dilate(self, m, n, conn):
        _check_conn(conn)
        return self.back(self._dilate_c(self.to(m, fill=False), n, conn))

    def open(self, m, n, conn):
        _check_conn(conn)
        c = self.to(m, fill=True)
        c = self._erode_c(c, n, conn)
        if self.gap is not None:
            c &= ~self.gap
        c = self._dilate_c(c, n, conn)
        return self.back(c)

    def fill_holes(self, m, conn, maxsize, label_fn):
        """``gal_binary_holes_fill``: background components that do not touch
        the (tile) border are filled, if smaller than ``maxsize`` pixels."""
        _check_conn(conn)
        c = self.to(m, fill=False)
        inv = ~c
        if self.gap is not None:
            inv &= ~self.gap
        lab, n = label_fn(inv, conn)
        if n == 0:
            return m.clone()
        outside = torch.zeros(n + 1, dtype=torch.bool, device=m.device)
        outside[lab[self.border & inv].long()] = True
        sizes = torch.bincount(lab.reshape(-1).long(), minlength=n + 1)
        is_hole = ~outside
        if maxsize is not None:
            is_hole &= sizes <= maxsize
        is_hole[0] = False
        return self.back(c | is_hole[lab.long()])


# ==========================================================================
# The NoiseChisel pipeline
# ==========================================================================
class NoiseChisel:
    """NoiseChisel detection on a 2D image with PyTorch.

    Parameters
    ----------
    device : str | torch.device | None
        Default: CUDA if available, else CPU.
    kernel : "default" | None | array-like
        Sharp convolution kernel (Gnuastro's default 11x11 Gaussian,
        FWHM=2 px).  ``None`` disables convolution.
    widekernel : None | array-like
        Optional wider kernel (``--widekernel``) used to pick sky-like tiles.
    ccl : "auto" | "cupy" | "scipy"
        Connected-component back-end ("auto": cupy on CUDA if installed,
        scipy otherwise).
    verbose : bool
        Print step timings.
    numthreads : int | None
        CPU threads used by PyTorch during a run (Gnuastro's ``-N``).
        ``None``/0: PyTorch's default (all physical cores).
    chunk_elems : int
        Max elements of the temporary per-tile tensors (memory knob).
    **params
        Any field of :class:`NoiseChiselParams` (e.g. ``tilesize=(64, 64)``).
    """

    def __init__(self, device=None, kernel="default", widekernel=None, ccl="auto",
                 verbose=False, numthreads=None, chunk_elems=16_000_000, **params):
        valid = {f.name for f in fields(NoiseChiselParams)}
        bad = set(params) - valid
        if bad:
            raise TypeError(f"unknown NoiseChisel option(s): {sorted(bad)}")
        self.p = NoiseChiselParams(**params)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "mps":
            # Tile statistics and the convolution need float64 accumulation
            # to reproduce Gnuastro (Sky levels can be >> noise), which MPS
            # does not provide.
            warnings.warn("MPS has no float64 support: running NoiseChisel on the CPU",
                          RuntimeWarning, stacklevel=2)
            self.device = torch.device("cpu")
        if ccl not in ("auto", "cupy", "scipy"):
            raise ValueError("ccl must be 'auto', 'cupy' or 'scipy'")
        if ccl == "auto":
            ccl = "cupy" if self.device.type == "cuda" and _cupy_available() else "scipy"
        if ccl == "cupy" and (self.device.type != "cuda" or not _cupy_available()):
            raise ValueError("ccl='cupy' needs device='cuda' and CuPy "
                             "(pip install 'noisechisel-torch[cuda12]' or '[cuda13]')")
        self.ccl = ccl
        self.verbose = verbose
        if numthreads is not None and int(numthreads) < 0:
            raise ValueError("numthreads must be >= 0")
        self.numthreads = int(numthreads) if numthreads else None
        self.chunk_elems = int(chunk_elems)
        self.kernel = self._as_kernel(kernel)
        self.widekernel = self._as_kernel(widekernel)
        self._validate()

    # ---------------------------------------------------------------- utils
    def _validate(self):
        p = self.p
        if p.interpnumngb <= 3:
            raise ValueError("interpnumngb must be larger than 3")
        if p.interpmetric not in ("radial", "manhattan"):
            raise ValueError("interpmetric must be 'radial' or 'manhattan'")
        if p.smoothwidth > 1 and p.smoothwidth % 2 == 0:
            raise ValueError("smoothwidth must be odd")
        for nm in ("erodengb", "openingngb", "dopeningngb", "holengb", "pseudoconcomp"):
            if getattr(p, nm) not in (4, 8):
                raise ValueError(f"{nm} must be 4 or 8")

    def _as_kernel(self, k):
        if k is None:
            return None
        # Gnuastro applies its built-in kernel as a correlation, while
        # kernels read from files are flipped first (a true convolution).
        flip = True
        if isinstance(k, str):
            if k != "default":
                raise ValueError("kernel must be 'default', None or an array")
            k, flip = DEFAULT_KERNEL, False
        k = torch.as_tensor(np.asarray(k, dtype=np.float32) if not torch.is_tensor(k) else k,
                            dtype=torch.float32, device=self.device)
        if k.ndim != 2:
            raise ValueError("kernel must be 2D")
        if flip:                       # as gal_fits_img_read_kernel: normalise, flip
            k = torch.nan_to_num(k, nan=0.0)
            k = torch.flip(k / k.sum(), (0, 1))
        return k

    def _log(self, msg, t0=None):
        if self.verbose:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = "" if t0 is None else f" ({time.perf_counter() - t0:.2f}s)"
            print(f"  - {msg}{dt}")
        return time.perf_counter()

    def _label(self, mask, conn):
        _check_conn(conn)
        if self.ccl == "cupy":
            return _label_cupy(mask, conn)
        return _label_scipy(mask, conn)

    # --------------------------------------------------- quantile threshold
    def _quantile_thresholds(self, grid, conv, wconv):
        p = self.p
        ny, nx = grid.ny, grid.nx
        dev = self.device
        nanf = lambda: torch.full((ny, nx), float("nan"), dtype=torch.float32, device=dev)
        e_th, ne_th, ex_th = nanf(), nanf(), nanf()
        meanconv = wconv if wconv is not None else conv
        mmq = float(np.float32(p.meanmedqdiff))
        k = grid.rows_per_chunk(self.chunk_elems)
        want_expand = p.detgrowquant != 1.0
        for r0 in range(0, ny, k):
            r1 = min(ny, r0 + k)
            tm = grid.gather(meanconv, r0, r1, float("nan"))
            xs, n = _sort_rows(tm)
            P = xs.shape[1]
            mean = (torch.where(torch.isnan(tm), torch.zeros_like(tm), tm).double().sum(1)
                    / n.clamp(min=1)).float()
            j = torch.searchsorted(xs, mean[:, None], right=True).squeeze(1)
            a_prev = xs.gather(1, (j - 1).clamp(0, P - 1)[:, None]).squeeze(1)
            a_j = xs.gather(1, j.clamp(0, P - 1)[:, None]).squeeze(1)
            parsed = (n >= 2) & (mean >= xs[:, 0]) & (j >= 1) & (j < n)
            idx = torch.where((mean - a_prev) < (a_j - mean), j - 1, j)
            q = idx.double() / (n - 1).clamp(min=1).double()
            good = parsed & ((q - 0.5).abs() < mmq)
            del tm
            if not bool(good.any()):
                continue
            if wconv is not None:
                tc = grid.gather(conv, r0, r1, float("nan"))[good]
                xq, nq = _sort_rows(tc)
            else:
                xq, nq = xs[good], n[good]
            gi = torch.nonzero(good).squeeze(1)
            for out, qq in ((e_th, p.qthresh), (ne_th, p.noerodequant),
                            (ex_th, p.detgrowquant if want_expand else None)):
                if qq is None:
                    continue
                v = _row_quantile(xq, nq, qq)
                flat = out.view(-1)
                flat[r0 * nx + gi] = v
        return e_th, ne_th, ex_th

    def _remove_outliers(self, arrays):
        """``gal_tileinternal_no_outlier_local``: mask tiles whose K-neighbour
        spread is an outlier of the distribution."""
        p = self.p
        first = arrays[0]
        valid = ~torch.isnan(first)
        if int(valid.sum()) < p.outliernumngb:
            return arrays
        kv = _knn_values([first], valid, p.outliernumngb, p.interpmetric)[0]
        s, _ = torch.sort(kv, dim=-1)
        measure = s[..., -1] - s[..., 1]
        measure = torch.where(valid, measure, torch.full_like(measure, float("nan")))
        mv = torch.sort(measure[valid])[0]
        thr = _outlier_threshold(mv, p.outliersigma, p.outliersclip[0], p.outliersclip[1])
        if thr is None:
            return arrays
        kill = measure.double() > thr
        nan = float("nan")
        return [torch.where(kill, torch.full_like(a, nan), a) for a in arrays]

    # ---------------------------------------------------------- sky and std
    def _sky_and_std(self, grid, img, binary, noskytiles):
        p = self.p
        ny, nx = grid.ny, grid.nx
        dev = self.device
        sky = torch.full((ny, nx), float("nan"), dtype=torch.float64, device=dev).view(-1)
        std = sky.clone()
        minfrac = torch.tensor(np.float32(p.minskyfrac), device=dev)
        k = grid.rows_per_chunk(self.chunk_elems)
        nosky = noskytiles.reshape(-1)
        for r0 in range(0, ny, k):
            r1 = min(ny, r0 + k)
            x = grid.gather(img, r0, r1, float("nan"))
            b = grid.gather(binary, r0, r1, True)
            und = (~b) & ~torch.isnan(x)
            numsky = und.sum(1)
            if p.skyfracnoblank:
                area = (~torch.isnan(x)).sum(1)
            else:
                area = grid.area[r0:r1].reshape(-1)
            frac_ok = (numsky.float() / area.clamp(min=1).float()) > minfrac
            sel = frac_ok & ~nosky[r0 * nx: r1 * nx] & (numsky > 0)
            if not bool(sel.any()):
                continue
            xv = torch.where(und[sel], x[sel], torch.full((), float("nan"), device=dev))
            xs, n = _sort_rows(xv)
            med, mean, sd = _sigma_clip_rows(xs, n, p.sigmaclip[0], p.sigmaclip[1])
            # sky.c tests output column 3 (the clipped *median*) against 0.
            zero = med.float() == 0
            mean = torch.where(zero, torch.full_like(mean, float("nan")), mean)
            sd = torch.where(zero, torch.full_like(sd, float("nan")), sd)
            gi = r0 * nx + torch.nonzero(sel).squeeze(1)
            sky[gi] = mean
            std[gi] = sd
        sky = sky.view(ny, nx).float()
        std = std.view(ny, nx).float()
        ok = ~torch.isnan(std)
        if not bool(ok.any()):
            raise RuntimeError("no tile could be used to measure the Sky (too much "
                               "signal?). Try different tile size / minskyfrac.")
        sv = torch.sort(std[ok])[0]
        n = sv.numel()
        med = float(sv[n // 2] if n % 2 else (sv[n // 2] + sv[n // 2 - 1]) / 2)
        minstd, maxstd = float(sv[0]), float(sv[-1])
        cpscorr = 1.0 if minstd > 1 else minstd
        sky, std = _interp_smooth([sky, std], p.interpnumngb, p.interpmetric,
                                  p.interponlyblank, p.smoothwidth)
        return sky, std, (minstd, med, maxstd), cpscorr

    # ------------------------------------------------------------------ S/N
    def _label_sn(self, grid, labels, num, img, sky_t, std_t, cpscorr, exclude=None):
        """S/N of each label, as ``detection_sn``: Sky-subtracted flux
        (float32) summed in double, flux-weighted centre to pick the STD."""
        p = self.p
        H, W = labels.shape
        flat = labels.reshape(-1)
        idx = torch.nonzero(flat > 0).squeeze(1)
        lab = flat[idx].long()
        y, x = idx // W, idx % W
        val = img.reshape(-1)[idx] - sky_t[grid.row_tile_c[y], grid.col_tile_c[x]]
        ok = ~torch.isnan(val)
        idx, lab, val, y, x = idx[ok], lab[ok], val[ok].double(), y[ok], x[ok]
        L = num + 1
        area = torch.bincount(lab, minlength=L).double()
        bright = torch.bincount(lab, weights=val, minlength=L)
        pos = val.clamp(min=0)
        ps = torch.bincount(lab, weights=pos, minlength=L)
        py = torch.bincount(lab, weights=pos * y.double(), minlength=L)
        px = torch.bincount(lab, weights=pos * x.double(), minlength=L)
        if exclude is not None:
            area = torch.where(exclude, torch.zeros_like(area), area)
        ave = bright / area.clamp(min=1)
        cond = (area >= p.snminarea) & (ave > 0) & (ps > 0)
        cond[0] = False
        psafe = ps.clamp(min=1e-300)

        def rnd(c):                                   # GAL_DIMENSION_FLT_TO_INT
            fl = torch.floor(c)
            return (fl + ((c - fl) > 0.5).double()).long()

        cy = rnd(py / psafe).clamp(0, H - 1)
        cx = rnd(px / psafe).clamp(0, W - 1)
        err = std_t[grid.row_tile_c[cy], grid.col_tile_c[cx]].double()
        # sqrt((float)area / cpscorr): float division, double sqrt
        a_c = (area.float() / torch.tensor(cpscorr, dtype=torch.float32)).double()
        sn = torch.sqrt(a_c) * ave / torch.sqrt((ave + err * err).clamp(min=1e-300))
        return torch.where(cond, sn, torch.full_like(sn, float("nan"))).float()

    # ----------------------------------------------------------------- main
    @torch.no_grad()
    def __call__(self, image) -> NoiseChiselResult:
        if self.numthreads is None:
            return self._run(image)
        prev = torch.get_num_threads()
        torch.set_num_threads(self.numthreads)
        try:
            return self._run(image)
        finally:
            torch.set_num_threads(prev)

    def _run(self, image) -> NoiseChiselResult:
        p = self.p
        dev = self.device
        t_all = time.perf_counter()
        if not torch.is_tensor(image):
            # e.g. astropy.io.fits.getdata(): big-endian and/or integer arrays
            image = np.asarray(image, dtype=np.float32)
        img = torch.as_tensor(image).to(dev)
        if img.ndim != 2:
            raise ValueError("only 2D images are supported")
        img = img.to(torch.float32).contiguous()
        H, W = img.shape
        img = torch.where(torch.isinf(img), torch.full_like(img, float("nan")), img)
        grid = _Grid((H, W), p.tilesize, p.remainderfrac, dev)
        whole = _Canvas((H, W), None, p.remainderfrac, dev)
        lt = _Canvas((H, W), p.largetilesize, p.remainderfrac, dev)
        if self.verbose:
            print(f"NoiseChisel-torch on {tuple(img.shape)} [{dev}, {torch.get_num_threads()} threads], "
                  f"tiles {grid.ny}x{grid.nx}")

        # 1. convolution --------------------------------------------------
        t = self._log("start")
        conv = _convolve(img, self.kernel) if self.kernel is not None else img
        wconv = _convolve(img, self.widekernel) if self.widekernel is not None else None
        t = self._log("convolved", t)

        # 2. quantile thresholds on tiles --------------------------------
        e_th, ne_th, ex_th = self._quantile_thresholds(grid, conv, wconv)
        e_th, ne_th, ex_th = self._remove_outliers([e_th, ne_th, ex_th])
        noskytiles = torch.isnan(e_th)
        nval = int((~noskytiles).sum())
        if nval < p.interpnumngb:
            raise ValueError(
                f"{nval} tiles usable after removing outliers, fewer than interpnumngb="
                f"{p.interpnumngb}. Use a larger input, smaller `tilesize`, larger "
                "`meanmedqdiff`, or smaller `interpnumngb`.")
        arrs = [e_th, ne_th] + ([ex_th] if p.detgrowquant != 1.0 else [])
        arrs = _interp_smooth(arrs, p.interpnumngb, p.interpmetric, p.interponlyblank, p.smoothwidth)
        e_th, ne_th = arrs[0], arrs[1]
        ex_th = arrs[2] if len(arrs) > 2 else None
        t = self._log("quantile thresholds found", t)

        # 3. initial detections -------------------------------------------
        above = conv > grid.expand(e_th)
        noerode = above & (conv > grid.expand(ne_th))
        mask = whole.erode(above, p.erode, p.erodengb, fixed=noerode)
        del above, noerode
        mask = whole.open(mask, p.opening, p.openingngb)
        olabel, ninit = self._label(mask, 8)
        t = self._log(f"{ninit} initial detections", t)

        # 4. initial sky / std, pseudo-detections -------------------------
        sky_t, std_t, _, cpscorr = self._sky_and_std(grid, img, olabel > 0, noskytiles)
        sub = img - grid.expand(sky_t)
        dbin = sub > p.dthresh * grid.expand(std_t)
        opos = olabel > 0
        t = self._log("initial sky found", t)

        def pseudo(det_mode):
            w = (opos & dbin) if det_mode else (opos | dbin)
            w = lt.fill_holes(w, p.holengb, None, self._label)
            w = lt.open(w, p.dopening, p.dopeningngb)
            return self._label(w, p.pseudoconcomp)

        if p.snthresh is None:
            plab, npse = pseudo(False)
            touches = torch.bincount(plab[opos & (plab > 0)].long(), minlength=npse + 1) > 0
            sn = self._label_sn(grid, plab, npse, img, sky_t, std_t, cpscorr, exclude=touches)
            sn = sn[~torch.isnan(sn)]
            if sn.numel() < p.minnumfalse:
                raise RuntimeError(
                    f"only {sn.numel()} pseudo-detections over the Sky; fewer than "
                    f"minnumfalse={p.minnumfalse}. Adjust dthresh / snminarea / tile settings.")
            sv = torch.sort(sn)[0]
            snthr = float(sv[int(_quantile_index(torch.tensor([sv.numel()]), p.snquant)[0])])
            del plab
        else:
            snthr = float(p.snthresh)
        plab, npse = pseudo(True)
        sn = self._label_sn(grid, plab, npse, img, sky_t, std_t, cpscorr)
        keep = sn > snthr
        keep[0] = False
        true_pseudo = keep[plab]
        del plab
        t = self._log(f"pseudo-detection S/N threshold {snthr:.3f}", t)

        # 5. keep initial detections that contain a true pseudo-detection --
        hit = torch.bincount(olabel[true_pseudo & opos].long(), minlength=ninit + 1) > 0
        hit[0] = False
        binary = hit[olabel]
        newlab = (torch.cumsum(hit, 0) * hit).to(torch.int32)
        nfinal = int(hit.sum())
        del true_pseudo

        # 6. growth ---------------------------------------------------------
        if ex_th is not None:
            cand = conv > grid.expand(ex_th)
            numexpand = int((cand & ~binary).sum())
            if numexpand == 0:
                labels = newlab[olabel]
            else:
                M = binary | cand
                comp, nc = self._label(M, 8)
                has = torch.bincount(comp[binary].long(), minlength=nc + 1) > 0
                has[0] = False
                grown = has[comp]
                grown = whole.dilate(grown, 1, 4)
                grown = whole.fill_holes(grown, 4, p.detgrowmaxholesize, self._label)
                labels, nfinal = self._label(grown, 8)
                binary = labels > 0
            del cand
        else:
            labels = newlab[olabel]
        if ex_th is not None and numexpand == 0:
            binary = labels > 0
        if p.cleangrowndet:
            sn = self._label_sn(grid, labels, nfinal, img, sky_t, std_t, cpscorr)
            k2 = sn > snthr
            k2[0] = False
            nl = (torch.cumsum(k2, 0) * k2).to(torch.int32)
            labels = nl[labels]
            nfinal = int(k2.sum())
            binary = labels > 0
        t = self._log(f"{nfinal} final detections", t)

        # 7. final sky and std --------------------------------------------
        sky_f, std_f, (mn, md, mx), _ = self._sky_and_std(grid, img, binary, noskytiles)
        # Tile values -> full-resolution images; blank input pixels stay blank.
        blank = torch.isnan(img)
        rows, cols = grid.row_tile[:, None], grid.col_tile[None, :]
        nan = torch.tensor(float("nan"), device=dev)
        sky_img = torch.where(blank, nan, sky_f[rows, cols])
        std_img = torch.where(blank, nan, std_f[rows, cols])
        t = self._log("final sky found", t)
        self._log("done", t_all)
        return NoiseChiselResult(
            detections=binary, labels=labels.to(torch.int32), sky=sky_img, std=std_img,
            num_detections=nfinal, num_initial_detections=ninit, det_sn_threshold=snthr,
            min_std=mn, med_std=md, max_std=mx, sky_tiles=sky_f, std_tiles=std_f, params=p)


def noisechisel(image, device=None, **kwargs) -> NoiseChiselResult:
    """Functional interface: ``noisechisel(image, device='cuda', qthresh=0.3, ...)``.

    ``kernel``, ``widekernel``, ``ccl``, ``verbose``, ``numthreads`` and
    ``chunk_elems`` go to
    the :class:`NoiseChisel` constructor; every other keyword is a NoiseChisel
    option (see :class:`NoiseChiselParams`).
    """
    return NoiseChisel(device=device, **kwargs)(image)
