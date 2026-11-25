#!/usr/bin/env python
"""Run time vs number of CPU threads for Gnuastro's ``astnoisechisel`` and
noisechisel-torch on one image.  Both timings include reading the input and
writing the output FITS file (default options, same HDUs).

    python benchmarks/scaling.py IMAGE.fits --workdir /tmp/nc \
        --csv benchmarks/results/scaling.csv --plot docs/images/scaling.png \
        --comparison-plot docs/images/performance.png --compare-threads 10 \
        --mps-time 1.57 --gpu-time 0.664
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


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Gnuastro 0.24": "#a3a29c", "noisechisel-torch": "#2a78d6"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    return plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor="white")


def _times(rows, prog):
    rr = sorted((r for r in rows if r["program"] == prog), key=lambda r: r["threads"])
    return np.array([r["threads"] for r in rr]), np.array([r["time_s"] for r in rr])


def plot_scaling(rows, path, title):
    """Run time vs number of threads, one line per program."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    progs = sorted({r["program"] for r in rows}, key=lambda p: p != "Gnuastro 0.24")
    for k, prog in enumerate(progs):
        n, t = _times(rows, prog)
        ax.plot(n, t, color=COLORS.get(prog, "#2a78d6"), lw=2, marker="o", ms=6,
                mec="white", mew=1.5)
        ax.annotate(f"{prog}  {t[-1]:.2f} s", (n[-1], t[-1]), xytext=(8, 7 if k == 0 else -7),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    nmax = max(r["threads"] for r in rows)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, nmax * 1.4)
    ax.set_xticks(sorted({r["threads"] for r in rows}))
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("seconds per frame (lower is better)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
    fig.text(0.01, 0.01, f"{rows[0].get('machine', '')}. Median of {_nruns(rows)} runs, "
             "including FITS read and write.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.2)
    _save(fig, path)


def plot_comparison(rows, path, threads, gpu_time=None, gpu_label="NVIDIA T4 GPU",
                    mps_time=None):
    """Bars: both programs at ``threads`` CPU threads, plus optional GPU times
    (Apple MPS and ``gpu_label``)."""
    plt = _pyplot()
    pick = {r["program"]: r["time_s"] for r in rows if r["threads"] == threads}
    tg = pick["Gnuastro 0.24"]
    bars = [(f"Gnuastro 0.24 · CPU, {threads} threads", tg, COLORS["Gnuastro 0.24"]),
            (f"noisechisel-torch · CPU, {threads} threads", pick["noisechisel-torch"],
             COLORS["noisechisel-torch"])]
    if mps_time:
        bars.append(("noisechisel-torch · M3 Max GPU (MPS)", mps_time, COLORS["noisechisel-torch"]))
    if gpu_time:
        bars.append((f"noisechisel-torch · {gpu_label}", gpu_time, COLORS["noisechisel-torch"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.65 * len(bars)), dpi=110)
    y = np.arange(len(bars))[::-1]
    vals = [b[1] for b in bars]
    ax.barh(y, vals, color=[b[2] for b in bars], height=0.6, edgecolor="white", linewidth=2)
    for yi, v in zip(y, vals):
        txt = f"{v:.2f} s" + ("" if v == tg else f"  ({tg / v:.1f}× faster)")
        ax.text(v + max(vals) * 0.015, yi, txt, va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=9.5)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel("seconds per 4088×4088 frame (lower is better)", fontsize=9)
    fig.suptitle(f"CPU ({threads} threads) vs GPU, simulated Roman WFI frame (4088×4088)",
                 x=0.01, ha="left", fontsize=11)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis="y", length=0)
    note = (f"CPU: {rows[0].get('machine', '')}, median of {_nruns(rows)} runs incl. FITS I/O "
            "(benchmarks/scaling.py).")
    if mps_time:
        note += "\nMPS: Apple M3 Max, computation only, median of 4 runs."
    if gpu_time:
        note += f"\n{gpu_label}: Google Colab with CuPy, computation only."
    if mps_time or gpu_time:
        note += "\nSpeed-ups are relative to Gnuastro, whose time includes FITS I/O."
    fig.text(0.01, 0.02, note, fontsize=8.5, color=INK2, va="bottom", linespacing=1.4)
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.33, right=0.98, top=1 - 0.45 / height,
                        bottom=(0.65 + 0.17 * (note.count("\n") + 1)) / height)
    _save(fig, path)


def _nruns(rows):
    return len(str(rows[0].get("runs", "")).split()) or "repeated"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--hdu", default="0")
    ap.add_argument("--threads", default=None, help="comma list (default: 1..all cores)")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--workdir", default="nc_scaling", help="where the output FITS files go")
    ap.add_argument("--csv", default="scaling.csv")
    ap.add_argument("--plot", default=None, help="time vs threads figure")
    ap.add_argument("--comparison-plot", default=None,
                    help="bar chart of both programs at --compare-threads (+ --gpu-time)")
    ap.add_argument("--compare-threads", type=int, default=10)
    ap.add_argument("--gpu-time", type=float, default=None,
                    help="GPU seconds per frame to add to the bar chart (e.g. from the Colab notebook)")
    ap.add_argument("--gpu-label", default="NVIDIA T4 GPU")
    ap.add_argument("--mps-time", type=float, default=None,
                    help="Apple GPU (device='mps') seconds per frame, computation only")
    ap.add_argument("--from-csv", action="store_true", help="only re-plot an existing --csv")
    a = ap.parse_args()

    if a.from_csv:
        with open(a.csv) as f:
            rows = [dict(r, threads=int(r["threads"]), time_s=float(r["time_s"]))
                    for r in csv.DictReader(f)]
    else:
        ncpu = os.cpu_count()
        threads = ([int(x) for x in a.threads.split(",")] if a.threads else
                   list(range(1, ncpu + 1)))
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
                             "mean_s": float(np.mean(ts)),
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
        plot_scaling(rows, a.plot, "Run time vs CPU threads, simulated Roman WFI frame (4088×4088)")
    if a.comparison_plot:
        plot_comparison(rows, a.comparison_plot, a.compare_threads, a.gpu_time, a.gpu_label,
                        a.mps_time)


if __name__ == "__main__":
    main()
