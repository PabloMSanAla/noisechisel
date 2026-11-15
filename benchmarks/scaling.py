#!/usr/bin/env python
"""Run time vs number of CPU threads for Gnuastro's ``astnoisechisel`` and
noisechisel-torch on one image.  Both timings include reading the input and
writing the output FITS file (default options, same HDUs).

    python benchmarks/scaling.py IMAGE.fits --threads 1,2,4,8,14 \
        --workdir /tmp/nc --csv benchmarks/results/scaling.csv --plot docs/images/scaling.png
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import time

import numpy as np
import torch

from noisechisel_torch import run_fits


def time_gnuastro(path, hdu, out, n):
    cmd = [shutil.which("astnoisechisel"), path, f"--hdu={hdu}", f"--output={out}",
           f"--numthreads={n}", "--quiet"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def time_torch(path, hdu, out, n):
    t0 = time.perf_counter()
    run_fits(path, out, hdu=hdu, device="cpu", numthreads=n)
    return time.perf_counter() - t0


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"Gnuastro 0.24": "#a3a29c", "noisechisel-torch": "#2a78d6"}
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.edgecolor": grid})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110)
    progs = sorted({r["program"] for r in rows}, key=lambda p: p != "Gnuastro 0.24")
    nmax = max(r["threads"] for r in rows)
    smax = 1.0
    for prog in progs:
        rr = sorted((r for r in rows if r["program"] == prog), key=lambda r: r["threads"])
        n = np.array([r["threads"] for r in rr])
        t = np.array([r["time_s"] for r in rr])
        c = colors.get(prog, "#2a78d6")
        kw = dict(color=c, lw=2, marker="o", ms=6, mec="white", mew=1.5)
        dy = 7 if prog == progs[0] else -7      # keep the two end labels apart
        a1.plot(n, t, **kw)
        a1.annotate(f"{prog}  {t[-1]:.2f} s", (n[-1], t[-1]), xytext=(6, dy),
                    textcoords="offset points", va="center", fontsize=9, color=ink)
        a2.plot(n, t[0] / t, **kw)
        a2.annotate(f"{prog}  {t[0] / t[-1]:.1f}×", (n[-1], t[0] / t[-1]), xytext=(6, dy),
                    textcoords="offset points", va="center", fontsize=9, color=ink)
        smax = max(smax, (t[0] / t).max())
    ytop = np.ceil(smax + 1.5)
    a2.plot([1, ytop], [1, ytop], color=ink2, lw=1, ls=(0, (3, 3)))
    a2.text(ytop - 0.2, ytop - 0.2, "ideal", color=ink2, fontsize=8.5, ha="right", va="top")
    a2.set_ylim(0, ytop)
    a1.set_ylim(bottom=0)
    a1.set_ylabel("seconds per frame (lower is better)")
    a2.set_ylabel("speed-up vs 1 thread")
    a1.set_title("Run time", loc="left", fontsize=11)
    a2.set_title("Scaling", loc="left", fontsize=11)
    for a in (a1, a2):
        a.set_xlabel("CPU threads")
        a.set_xlim(0.5, nmax * 1.55)
        a.set_xticks(sorted({r["threads"] for r in rows}))
        a.grid(True, color=grid, lw=0.8)
        a.set_axisbelow(True)
        for k in ("top", "right"):
            a.spines[k].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.01, f"{rows[0].get('machine', '')}. Median of repeated runs, "
             "including FITS read and write.", fontsize=8.5, color=ink2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.2, wspace=0.25)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--hdu", default="0")
    ap.add_argument("--threads", default=None, help="comma list (default: 1,2,4,... up to all cores)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workdir", default="nc_scaling", help="where the output FITS files go")
    ap.add_argument("--csv", default="scaling.csv")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--from-csv", action="store_true", help="only re-plot an existing --csv")
    a = ap.parse_args()

    if a.from_csv:
        with open(a.csv) as f:
            rows = [dict(r, threads=int(r["threads"]), time_s=float(r["time_s"]))
                    for r in csv.DictReader(f)]
    else:
        ncpu = os.cpu_count()
        threads = ([int(x) for x in a.threads.split(",")] if a.threads else
                   sorted({1, 2, 4, 8, ncpu} | set(range(2, ncpu + 1, 2))))
        os.makedirs(a.workdir, exist_ok=True)
        hdu = int(a.hdu) if a.hdu.isdigit() else a.hdu
        machine = platform.processor() or platform.machine()
        if platform.system() == "Darwin":
            machine = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip() or machine
        time_torch(a.image, hdu, os.path.join(a.workdir, "warm.fits"), None)   # warm-up
        rows = []
        for n in threads:
            for prog, fn in (("Gnuastro 0.24", time_gnuastro), ("noisechisel-torch", time_torch)):
                out = os.path.join(a.workdir, "out.fits")
                ts = [fn(a.image, hdu, out, n) for _ in range(a.repeat)]
                rows.append({"program": prog, "threads": n, "time_s": float(np.median(ts)),
                             "runs": " ".join(f"{t:.3f}" for t in ts), "machine": machine,
                             "torch": torch.__version__})
                print(f"{prog:18s} N={n:2d}  {np.median(ts):6.2f} s  ({rows[-1]['runs']})", flush=True)
        shutil.rmtree(a.workdir, ignore_errors=True)
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.plot:
        plot(rows, a.plot, f"Thread scaling on a 4088×4088 Roman WFI frame — {os.path.basename(a.image)}")


if __name__ == "__main__":
    main()
