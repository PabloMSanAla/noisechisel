# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line interface, modelled on ``astnoisechisel``::

    noisechisel-torch image.fits -h 1 -o out.fits --tilesize 30,30 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import fields
from typing import get_args, get_origin, get_type_hints

from .core import NoiseChiselParams

# Sizes given in FITS (x, y) order on the command line, as in Gnuastro; the
# Python API uses numpy (rows, cols) order.
FITS_ORDER = ("tilesize", "largetilesize")

# Gnuastro's short option names.
SHORT = {"meanmedqdiff": "-Q", "qthresh": "-t", "erode": "-e", "opening": "-p",
         "minskyfrac": "-B", "sigmaclip": "-s", "dthresh": "-R", "snminarea": "-m",
         "snquant": "-c", "detgrowquant": "-d"}


def _parse_value(text, typ):
    origin = get_origin(typ)
    args = [a for a in get_args(typ) if a is not type(None)]
    if origin is not None and type(None) in get_args(typ):       # Optional[...]
        if text.lower() == "none":
            return None
        return _parse_value(text, args[0])
    if origin is tuple:
        parts = text.split(",")
        return tuple(a(p) for a, p in zip(args, parts))
    if typ is bool:
        return text.lower() in ("1", "true", "yes", "y")
    return typ(text)


def _hdu(text):
    return int(text) if text.lstrip("-").isdigit() else text


def _read_kernel(path, hdu):
    from astropy.io import fits
    return fits.getdata(path, ext=_hdu(hdu))


def build_parser():
    ap = argparse.ArgumentParser(
        prog="noisechisel-torch", add_help=False,
        description="PyTorch NoiseChisel: detect signal and measure Sky/STD of a FITS image.")
    ap.add_argument("--help", action="help")
    ap.add_argument("input", help="input FITS image")
    ap.add_argument("-h", "--hdu", default="0", help="input HDU (number or name)")
    ap.add_argument("-o", "--output", help="output FITS (default: <input>_detected.fits)")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: auto)")
    ap.add_argument("-N", "--numthreads", type=int, default=None, metavar="N",
                    help="number of CPU threads (default: all cores)")
    ap.add_argument("--ccl", default="auto", choices=["auto", "cupy", "scipy"])
    ap.add_argument("-k", "--kernel", default=None,
                    help="kernel FITS file, or 'none' (default: Gnuastro's built-in kernel)")
    ap.add_argument("--khdu", default="1", help="HDU of --kernel")
    ap.add_argument("--widekernel", default=None, help="wide kernel FITS file")
    ap.add_argument("--whdu", default="1", help="HDU of --widekernel")
    ap.add_argument("--label", action="store_true", help="write labels instead of a 0/1 mask")
    ap.add_argument("--rawoutput", action="store_true", help="skip the INPUT-NO-SKY HDU")
    ap.add_argument("--oneelempertile", action="store_true",
                    help="write SKY/SKY_STD with one value per tile (smaller output)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hints = get_type_hints(NoiseChiselParams)
    for f in fields(NoiseChiselParams):
        names = ([SHORT[f.name]] if f.name in SHORT else []) + [f"--{f.name}"]
        if hints[f.name] is bool:       # flags, as in Gnuastro
            ap.add_argument(*names, dest=f.name, action="store_const", const=True,
                            default=None, help="(default: off)")
        else:
            ap.add_argument(*names, dest=f.name, default=None, metavar="VAL",
                            help=f"(default: {f.default})")
    ap.set_defaults(_hints=hints)
    return ap


def main(argv=None):
    from .io import run_fits

    ap = build_parser()
    a = ap.parse_args(argv)
    hints = a._hints
    params = {}
    for f in fields(NoiseChiselParams):
        v = getattr(a, f.name)
        if v is True:
            params[f.name] = True
        elif v is not None:
            params[f.name] = _parse_value(v, hints[f.name])
            if f.name in FITS_ORDER and isinstance(params[f.name], tuple):
                params[f.name] = params[f.name][::-1]
    hdu = _hdu(a.hdu)
    if a.kernel is not None:
        params["kernel"] = None if a.kernel.lower() == "none" else _read_kernel(a.kernel, a.khdu)
    if a.widekernel is not None:
        params["widekernel"] = _read_kernel(a.widekernel, a.whdu)
    out = a.output or os.path.splitext(os.path.basename(a.input))[0] + "_detected.fits"
    t0 = time.perf_counter()
    try:
        res = run_fits(a.input, out, hdu=hdu, device=a.device, ccl=a.ccl,
                       numthreads=a.numthreads, label=a.label,
                       rawoutput=a.rawoutput, oneelempertile=a.oneelempertile,
                       verbose=not a.quiet, **params)
    except OSError as e:
        print(f"noisechisel-torch: error: {e}", file=sys.stderr)
        return 1
    if not a.quiet:
        print(f"{res.num_detections} detections, S/N thresh {res.det_sn_threshold:.3f}, "
              f"median std {res.med_std:.6g}  -> {out}  [{time.perf_counter() - t0:.2f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
