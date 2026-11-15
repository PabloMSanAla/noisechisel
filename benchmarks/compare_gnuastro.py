#!/usr/bin/env python
"""Run Gnuastro's ``astnoisechisel`` and ``noisechisel_torch`` on the same FITS
images and compare detection masks, Sky and Sky STD, plus run times.

    python benchmarks/compare_gnuastro.py IMG1.fits [IMG2.fits ...] \
        --workdir out/ --device cpu --csv results.csv

Gnuastro outputs are cached in ``--workdir`` (re-used when present unless
``--rerun-gnuastro``).  Extra NoiseChisel options can be given as
``--opt name=value`` and are passed to both programs.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from astropy.io import fits

from noisechisel_torch import NoiseChisel


def run_gnuastro(path, hdu, out, opts, threads):
    exe = shutil.which("astnoisechisel")
    if exe is None:
        raise SystemExit("astnoisechisel not found in PATH")
    cmd = [exe, path, f"--hdu={hdu}", f"--output={out}", "--quiet"]
    if threads:
        cmd.append(f"--numthreads={threads}")
    cmd += [f"--{k}={v}" for k, v in opts.items()]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def load_gnuastro(out):
    with fits.open(out) as hl:
        det = hl["DETECTIONS"].data.astype(bool)
        sky = hl["SKY"].data.astype(np.float32)
        std = hl["SKY_STD"].data.astype(np.float32)
        h = hl["SKY_STD"].header
        detsn = hl["DETECTIONS"].header.get("DETSN", np.nan)
    return det, sky, std, h.get("MEDSTD", np.nan), detsn


def opt_to_kw(opts):
    kw = {}
    for k, v in opts.items():
        if "," in v:
            parts = v.split(",")
            kw[k] = tuple(int(p) if p.lstrip("-").isdigit() else float(p) for p in parts)
        else:
            try:
                kw[k] = int(v)
            except ValueError:
                kw[k] = float(v)
    return kw


def compare(det_g, sky_g, std_g, det_t, sky_t, std_t):
    good = np.isfinite(sky_g) & np.isfinite(sky_t)
    inter = np.count_nonzero(det_g & det_t)
    union = np.count_nonzero(det_g | det_t)
    ng, nt = np.count_nonzero(det_g), np.count_nonzero(det_t)
    dsky = (sky_t - sky_g)[good] / std_g[good]          # Sky difference in units of noise
    dstd = (std_t - std_g)[good] / std_g[good]          # relative STD difference
    return {
        "frac_det_gnu": ng / det_g.size,
        "frac_det_torch": nt / det_t.size,
        "pix_agree": float(np.mean(det_g == det_t)),
        "n_mask_diff": int(np.count_nonzero(det_g != det_t)),
        "n_sky_diff": int(np.count_nonzero(sky_g != sky_t)),
        "n_std_diff": int(np.count_nonzero(std_g != std_t)),
        "iou": inter / union if union else 1.0,
        "gnu_recovered": inter / ng if ng else 1.0,
        "torch_in_gnu": inter / nt if nt else 1.0,
        "dsky_med_sig": float(np.median(dsky)),
        "dsky_absmax_sig": float(np.max(np.abs(dsky))),
        "dsky_p99_sig": float(np.percentile(np.abs(dsky), 99)),
        "dstd_med_rel": float(np.median(dstd)),
        "dstd_absmax_rel": float(np.max(np.abs(dstd))),
        "dstd_p99_rel": float(np.percentile(np.abs(dstd), 99)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--hdu", default="0")
    ap.add_argument("--workdir", default="nc_compare")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ccl", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="Gnuastro --numthreads (0: all)")
    ap.add_argument("--repeat", type=int, default=1, help="torch timing repetitions (min is kept)")
    ap.add_argument("--rerun-gnuastro", action="store_true")
    ap.add_argument("--opt", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    opts = dict(o.split("=", 1) for o in a.opt)
    os.makedirs(a.workdir, exist_ok=True)
    nc = NoiseChisel(device=a.device, ccl=a.ccl, **opt_to_kw(opts))
    print(f"torch device: {nc.device}, ccl: {nc.ccl}, threads: {torch.get_num_threads()}")
    rows = []
    for path in a.images:
        name = os.path.splitext(os.path.basename(path))[0]
        gout = os.path.join(a.workdir, f"{name}_gnuastro.fits")
        t_gnu = np.nan
        if a.rerun_gnuastro or not os.path.exists(gout):
            t_gnu = run_gnuastro(path, a.hdu, gout, opts, a.threads)
        det_g, sky_g, std_g, medstd_g, detsn_g = load_gnuastro(gout)

        hdu = int(a.hdu) if a.hdu.isdigit() else a.hdu
        data = fits.getdata(path, ext=hdu)
        times = []
        for _ in range(max(1, a.repeat)):
            t0 = time.perf_counter()
            res = nc(data)
            if nc.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        det_t = res.detections.cpu().numpy()
        sky_t = res.sky.cpu().numpy()
        std_t = res.std.cpu().numpy()

        row = {"image": name, "t_gnuastro_s": t_gnu, "t_torch_s": min(times),
               "detsn_gnu": detsn_g, "detsn_torch": res.det_sn_threshold,
               "medstd_gnu": medstd_g, "medstd_torch": res.med_std}
        row.update(compare(det_g, sky_g, std_g, det_t, sky_t, std_t))
        rows.append(row)
        print(f"{name}: gnu {t_gnu:6.2f}s  torch {min(times):6.2f}s | IoU {row['iou']:.4f} "
              f"mask-diff {row['n_mask_diff']} px | sky-diff {row['n_sky_diff']} std-diff {row['n_std_diff']} px | dSky med {row['dsky_med_sig']:+.2e} "
              f"max {row['dsky_absmax_sig']:.2e} sigma | dSTD med {row['dstd_med_rel']:+.2e} "
              f"max {row['dstd_absmax_rel']:.2e} | DETSN {detsn_g:.3f}/{res.det_sn_threshold:.3f}",
              flush=True)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
