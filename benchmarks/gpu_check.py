#!/usr/bin/env python
"""Check the CUDA path against the CPU path (which is bit-identical to
Gnuastro) and time it.

    python benchmarks/gpu_check.py                 # synthetic Roman-like 4088^2 image
    python benchmarks/gpu_check.py img.fits ...    # your own images (HDU 0)
    python benchmarks/gpu_check.py --torch-ccl     # also time ccl="torch" on the GPU
"""
import argparse
import time

import numpy as np
import torch

import noisechisel_torch as nct
from noisechisel_torch.core import _cupy_available


def synthetic(n=4088, sky=0.2846, sigma=7.08e-6, nsrc=4000, seed=0):
    """Large Sky / noise ratio like Roman count-rate images (stresses the
    float32 / float64 arithmetic)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(sky, sigma, (n, n)).astype(np.float64)
    for _ in range(nsrc):
        y, x = rng.integers(16, n - 16, 2)
        s = rng.uniform(1.0, 5.0)
        a = rng.lognormal(np.log(10 * sigma), 1.2)
        r = int(5 * s) + 1
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        img[y - r:y + r + 1, x - r:x + r + 1] += a * np.exp(-(yy ** 2 + xx ** 2) / (2 * s * s))
    return img.astype(np.float32)


def run(nc, img, repeat):
    nc(img)                                            # warm-up (kernels, caches)
    ts = []
    for _ in range(repeat):
        if nc.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = nc(img)
        if nc.device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return res, min(ts)


def same(a, b):
    out = {}
    for k in ("detections", "labels", "sky", "std"):
        x, y = getattr(a, k).cpu(), getattr(b, k).cpu()
        out[k] = "identical" if torch.equal(x, y) else f"DIFF {int((x != y).sum())}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--torch-ccl", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "no CUDA device"
    print(f"noisechisel-torch {nct.__version__}, torch {torch.__version__}, "
          f"GPU {torch.cuda.get_device_name()}, CPU threads {torch.get_num_threads()}")
    if _cupy_available():
        import cupy
        print(f"cupy {cupy.__version__}")
    else:
        print("cupy NOT installed: ccl='cupy' skipped")

    images = [(p, nct.read_image(p)[0]) for p in a.images] or [("synthetic 4088^2", synthetic())]
    cfgs = [("cpu", "scipy")]
    if _cupy_available():
        cfgs.append(("cuda", "cupy"))
    if a.torch_ccl or not _cupy_available():
        cfgs.append(("cuda", "torch"))
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
        if _cupy_available():
            print("stage timings (cuda, cupy):")
            nct.NoiseChisel(device="cuda", ccl="cupy", verbose=True)(img)


if __name__ == "__main__":
    main()
