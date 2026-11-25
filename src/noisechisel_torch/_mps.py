# SPDX-License-Identifier: GPL-3.0-or-later
"""Apple MPS: the NoiseChisel convolution as a Metal shader.

``gal_convolve_spatial`` accumulates float32 products in double, and Metal has
no float64.  The shader therefore does the float64 additions and the final
division (IEEE binary64, round to nearest even) with integer arithmetic, so its
output is bit-identical to the CPU path.

Apple GPUs flush float32 subnormals to zero.  :func:`convolve` returns ``None``
whenever it cannot guarantee the CPU result, and the caller then convolves on
the CPU.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// binary64 values held in a ulong.  Only zeros and normal numbers occur: every
// operand is a float32 product (or a sum of them), far inside the double range.
constant ulong SIGN = 1UL << 63;
constant ulong FRAC = 0x000fffffffffffffUL;
constant ulong HIDDEN = 1UL << 52;

static ulong f2d(float f) {                      // zero or normal float32
    uint u = as_type<uint>(f);
    ulong s = (ulong)(u >> 31) << 63;
    uint e = (u >> 23) & 255;
    if (e == 0) return s;
    return s | ((ulong)(e + 896) << 52) | ((ulong)(u & 0x7fffff) << 29);
}

// m: leading 1 at bit 55, i.e. 53 significant bits + guard, round, sticky.
static ulong pack(ulong s, int e, ulong m) {
    ulong r = m & 7;
    m >>= 3;
    if (r > 4 || (r == 4 && (m & 1))) m++;
    if (m >> 53) { m >>= 1; e++; }
    return s | ((ulong)e << 52) | (m & FRAC);
}

static ulong dadd(ulong a, ulong b) {
    if ((a << 1) < (b << 1)) { ulong t = a; a = b; b = t; }        // |a| >= |b|
    if ((b << 1) == 0) return ((a << 1) == 0) ? (a & b) : a;
    int ea = (int)((a >> 52) & 0x7ff), eb = (int)((b >> 52) & 0x7ff);
    ulong ma = ((a & FRAC) | HIDDEN) << 3, mb = ((b & FRAC) | HIDDEN) << 3;
    int d = ea - eb;
    if (d > 60) mb = 1;
    else if (d > 0) mb = (mb >> d) | (ulong)((mb & ((1UL << d) - 1)) != 0);
    ulong s = a & SIGN;
    if ((a ^ b) >> 63) {
        ma -= mb;
        if (ma == 0) return 0;                                      // +0.0
        int sh = (int)clz(ma) - 8;
        ma <<= sh;
        ea -= sh;
    } else {
        ma += mb;
        if (ma >> 56) { ma = (ma >> 1) | (ma & 1); ea++; }
    }
    return pack(s, ea, ma);
}

// (float)(a / b): the quotient rounded to double, then to float32, as in C.
// b must be non-zero.  ``bad`` is set if the result is not zero or a normal
// float32.
static float ddiv_f32(ulong a, ulong b, thread bool &bad) {
    uint sign = (uint)((a ^ b) >> 63);
    if ((a << 1) == 0) return as_type<float>(sign << 31);
    int e = (int)((a >> 52) & 0x7ff) - (int)((b >> 52) & 0x7ff) + 1023;
    ulong ma = (a & FRAC) | HIDDEN, mb = (b & FRAC) | HIDDEN;
    if (ma < mb) { ma <<= 1; e--; }
    ulong q = 0;
    for (int i = 0; i < 56; i++) {
        q <<= 1;
        if (ma >= mb) { ma -= mb; q |= 1; }
        ma <<= 1;
    }
    q |= (ulong)(ma != 0);
    ulong d = pack(0, e, q);
    int ef = (int)(d >> 52) - 1023 + 127;
    ulong m = d & FRAC;
    uint mf = (uint)(m >> 29);
    ulong r = m & 0x1fffffffUL;
    if (r > 0x10000000UL || (r == 0x10000000UL && (mf & 1))) mf++;
    if (ef < 1 || ef > 253) { bad = true; return 0.0f; }
    return as_type<float>((sign << 31) | (((uint)ef << 23) + mf));
}

// x, m: the zero-padded image and validity mask.  taps / off: the non-zero
// kernel values in raster order and their offsets in the padded image.
kernel void nc_convolve(
    device const float* x [[buffer(0)]], device const float* m [[buffer(1)]],
    device const float* taps [[buffer(2)]], device const int* off [[buffer(3)]],
    constant int& ntaps [[buffer(4)]], constant int& W [[buffer(5)]],
    constant int& PW [[buffer(6)]], constant int& row0 [[buffer(7)]],
    device float* out [[buffer(8)]], device atomic_uint* flag [[buffer(9)]],
    uint2 g [[thread_position_in_grid]])
{
    int row = row0 + (int)g.y;
    int base = row * PW + (int)g.x;
    ulong num = 0, den = 0;
    for (int t = 0; t < ntaps; t++) {
        int i = base + off[t];
        num = dadd(num, f2d(x[i] * taps[t]));
        den = dadd(den, f2d(m[i] * taps[t]));
    }
    bool bad = false;
    float v = as_type<float>(0x7fc00000u);                          // NaN
    if ((den << 1) != 0) v = ddiv_f32(num, den, bad);
    if (bad) atomic_store_explicit(flag, 1u, memory_order_relaxed);
    out[row * W + (int)g.x] = v;
}

// The two primitives alone, for the unit tests.
kernel void nc_test_dadd(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                         device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    out[i] = dadd(a[i], b[i]);
}

kernel void nc_test_ddiv(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                         device float* out [[buffer(2)]], device int* badout [[buffer(3)]],
                         uint i [[thread_position_in_grid]])
{
    bool bad = false;
    out[i] = ddiv_f32(a[i], b[i], bad);
    badout[i] = bad;
}
"""

# Kernel taps evaluated per command buffer (keeps each well below the GPU
# watchdog time limit).
_TAPS_PER_DISPATCH = 250_000_000

# Magnitude range of float32 values for which no intermediate of the pipeline
# can become subnormal (flushed to zero on Apple GPUs) or overflow.
SAFE_MIN, SAFE_MAX = 2.0 ** -100, 2.0 ** 100

_FLT_MIN, _FLT_MAX = 2.0 ** -126, 3.4028234663852886e38

_library = None
_library_error: Optional[str] = None


def is_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


@lru_cache(maxsize=None)
def _morphology_works() -> bool:
    """Compare the binary morphology on the GPU with the CPU (once)."""
    from .core import _dilate1

    mask = torch.rand((67, 71), generator=torch.Generator().manual_seed(1)) > 0.7
    gpu = mask.to("mps")
    return all(torch.equal(_dilate1(gpu, c).cpu(), _dilate1(mask, c)) for c in (4, 8))


def unsupported_reason() -> Optional[str]:
    """Why NoiseChisel cannot run on this MPS set-up, or None if it can."""
    if not is_available():
        return "MPS is not available"
    # Older releases give wrong results without any error: torch 2.6 corrupts
    # in-place ``|=`` on column slices of bool tensors.
    if not hasattr(torch.mps, "compile_shader"):
        return f"MPS needs torch 2.7 or later (this is torch {torch.__version__})"
    if not _morphology_works():
        return "MPS binary morphology does not match the CPU with this torch / macOS"
    return None


def safe_range(x: torch.Tensor) -> bool:
    """True if every finite non-zero value of the CPU float32 tensor ``x`` has
    a magnitude in [SAFE_MIN, SAFE_MAX]."""
    a = x.abs()
    bad = ((a > 0) & (a < SAFE_MIN)) | ((a > SAFE_MAX) & torch.isfinite(a))
    return not bool(bad.any())


def library():
    """The compiled shader library, or None (warning once) if it cannot be
    built or fails the self-test."""
    global _library, _library_error
    if _library is None and _library_error is None:
        try:
            lib = torch.mps.compile_shader(_SOURCE)
            _self_test(lib)
            _library = lib
        except Exception as e:
            _library_error = (str(e).splitlines() or [type(e).__name__])[0]
            warnings.warn(f"Metal convolution unavailable ({_library_error}); convolving on "
                          "the CPU", RuntimeWarning, stacklevel=2)
    return _library


def _self_test(lib):
    """Compare the shader with the CPU on a small image with blank pixels."""
    from .core import DEFAULT_KERNEL, _convolve

    g = torch.Generator().manual_seed(0)
    img = torch.randn((37, 41), generator=g) * 3 + 1000
    img[5:8, 9:12] = float("nan")
    kernel = torch.from_numpy(DEFAULT_KERNEL.copy())
    got = _run(lib, img.to("mps"), kernel)
    ref = _convolve(img, kernel)
    if got is None or not torch.equal(torch.nan_to_num(got.cpu()), torch.nan_to_num(ref)):
        raise RuntimeError("shader self-test does not match the CPU")


def _exact_on_gpu(img: torch.Tensor, ok: torch.Tensor, kernel: torch.Tensor) -> bool:
    """Whether every float32 product x * k (and k itself) is zero or a normal
    number, so flush-to-zero cannot change it, and cannot overflow."""
    if not bool(torch.isfinite(kernel).all()):
        return False
    k = kernel.abs()
    k = k[k > 0]
    if not k.numel():
        return False
    kmin, kmax = float(k.min()), float(k.max())
    if kmin < _FLT_MIN:
        return False
    # Judged from the bits: float comparisons on the GPU may treat subnormals
    # as zero.
    mag = torch.where(ok, img, torch.zeros_like(img)).view(torch.int32) & 0x7fffffff
    top = int(mag.max())
    if top == 0:
        return True
    big = torch.full_like(mag, 0x7fffffff)
    bottom = int(torch.where(mag == 0, big, mag).min())

    def value(bits):
        return float(torch.tensor(bits, dtype=torch.int32).view(torch.float32))

    vmin, vmax = value(bottom), value(top)
    return (vmin >= _FLT_MIN and vmin * kmin >= 2 * _FLT_MIN
            and vmax * max(kmax, 1.0) * k.numel() <= _FLT_MAX / 2)


def _run(lib, img: torch.Tensor, kernel: torch.Tensor) -> Optional[torch.Tensor]:
    """Convolve the MPS float32 image with the CPU float32 kernel."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W = img.shape
    PW = W + 2 * pw
    ok = ~torch.isnan(img)
    if not _exact_on_gpu(img, ok, kernel):
        return None
    x = torch.where(ok, img, torch.zeros_like(img))
    x = F.pad(x[None, None], (pw, pw, ph, ph))[0, 0].contiguous()
    m = F.pad(ok.to(torch.float32)[None, None], (pw, pw, ph, ph))[0, 0].contiguous()
    nz = torch.nonzero(kernel)                       # raster order, as on the CPU
    taps = kernel[nz[:, 0], nz[:, 1]].contiguous().to(img.device)
    off = (nz[:, 0] * PW + nz[:, 1]).to(torch.int32).to(img.device)
    ntaps = int(taps.numel())
    out = torch.empty((H, W), dtype=torch.float32, device=img.device)
    flag = torch.zeros(1, dtype=torch.int32, device=img.device)
    rows = max(1, _TAPS_PER_DISPATCH // (W * ntaps))
    for row0 in range(0, H, rows):
        lib.nc_convolve(x, m, taps, off, ntaps, W, PW, row0, out, flag,
                        threads=(W, min(rows, H - row0)))
        torch.mps.synchronize()
    if int(flag.item()):
        return None
    return torch.where(ok, out, torch.full_like(out, float("nan")))


def convolve(img: torch.Tensor, kernel: torch.Tensor) -> Optional[torch.Tensor]:
    """``core._convolve`` of an MPS image on the GPU, or None if the shader is
    unavailable or cannot reproduce the CPU for these values."""
    lib = library()
    if lib is None:
        return None
    return _run(lib, img.to(torch.float32).contiguous(), kernel.cpu().to(torch.float32))
