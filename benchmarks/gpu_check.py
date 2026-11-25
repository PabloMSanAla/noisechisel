#!/usr/bin/env python
"""Check the GPU path (CUDA or Apple MPS) against the CPU path (which is
bit-identical to Gnuastro) and time it.

    python benchmarks/gpu_check.py                 # synthetic Roman-like 4088^2 image
    python benchmarks/gpu_check.py img.fits ...    # your own images (HDU 0)
    python benchmarks/gpu_check.py --device mps    # default: cuda, else mps
"""
import argparse
import time

import numpy as np
import torch

import noisechisel_torch as nct
from noisechisel_torch import _mps
from noisechisel_torch.core import _cupy_available


def synthetic(n=4088, sky=0.2846, sigma=7.08e-6, nsrc=4000, seed=0):
    """Large Sky / noise ratio like Roman count-rate images (stresses the
    float32 / float64 arithmetic)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(sky, sigma, (n, n)).astype(np.float64)
    for _ in range(nsrc):
        y, x = rng.integers(32, n - 32, 2)
        s = rng.uniform(1.0, 5.0)
        a = rng.lognormal(np.log(10 * sigma), 1.2)
        r = int(5 * s) + 1
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        img[y - r:y + r + 1, x - r:x + r + 1] += a * np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s))
    return img.astype(np.float32)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run(nc, img, repeat):
    nc(img)                                            # warm-up (kernels, caches)
    ts = []
    for _ in range(repeat):
        sync(nc.device)
        t0 = time.perf_counter()
        res = nc(img)
        sync(nc.device)
        ts.append(time.perf_counter() - t0)
    return res, min(ts)


def same(a, b):
    out = {}
    for k in ("detections", "labels", "sky", "std"):
        x, y = getattr(a, k).cpu(), getattr(b, k).cpu()
        diff = (x != y) & ~((x != x) & (y != y))        # NaN == NaN (blank pixels)
        out[k] = f"DIFF {int(diff.sum())}" if diff.any() else "identical"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", choices=["cuda", "mps"], default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "mps")
    if device == "cuda":
        assert torch.cuda.is_available(), "no CUDA device"
        name = torch.cuda.get_device_name()
    else:
        assert _mps.unsupported_reason() is None, _mps.unsupported_reason()
        name = "Apple MPS"
    cupy_on = device == "cuda" and _cupy_available()
    print(f"noisechisel-torch {nct.__version__}, torch {torch.__version__}, "
          f"GPU {name}, CPU threads {torch.get_num_threads()}")
    if cupy_on:
        import cupy
        print(f"cupy {cupy.__version__}")
    elif device == "cuda":
        print("cupy NOT installed: ccl='cupy' skipped")

    if a.images:
        from astropy.io import fits
        images = [(p, fits.getdata(p)) for p in a.images]
    else:
        images = [("synthetic 4088^2", synthetic())]
    ccl = "cupy" if cupy_on else "scipy"
    cfgs = [("cpu", "scipy"), (device, ccl)]
    for name, img in images:
        print(f"\n== {name} {img.shape}")
        ref = None
        for dev, ccl in cfgs:
            res, t = run(nct.NoiseChisel(device=dev, ccl=ccl), img, a.repeat)
            line = f"{dev:4s} ccl={ccl:5s} {t:7.3f}s  {res.num_detections} detections, DETSN {res.det_sn_threshold:.4f}"
            if ref is None:
                ref, tref = res, t
            else:
                line += f"  x{tref / t:.1f} vs CPU  {same(ref, res)}"
            print(line, flush=True)
        print(f"stage timings ({device}, {ccl}):")
        nct.NoiseChisel(device=device, ccl=ccl, verbose=True)(img)


if __name__ == "__main__":
    main()
