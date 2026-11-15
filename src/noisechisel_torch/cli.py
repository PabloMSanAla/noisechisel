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


def build_parser():
    ap = argparse.ArgumentParser(
        prog="noisechisel-torch", add_help=False,
        description="PyTorch NoiseChisel: detect signal and measure Sky/STD of a FITS image.")
    ap.add_argument("--help", action="help")
    ap.add_argument("input", help="input FITS image")
    ap.add_argument("-h", "--hdu", default="0", help="input HDU (number or name)")
    ap.add_argument("-o", "--output", help="output FITS (default: <input>_detected.fits)")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:N (default: auto)")
    ap.add_argument("--ccl", default="auto", choices=["auto", "cupy", "scipy"])
    ap.add_argument("--label", action="store_true", help="write labels instead of a 0/1 mask")
    ap.add_argument("--rawoutput", action="store_true", help="skip the INPUT-NO-SKY HDU")
    ap.add_argument("-q", "--quiet", action="store_true")
    hints = get_type_hints(NoiseChiselParams)
    for f in fields(NoiseChiselParams):
        ap.add_argument(f"--{f.name}", dest=f.name, default=None, metavar="VAL",
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
        if v is not None:
            params[f.name] = _parse_value(v, hints[f.name])
    hdu = int(a.hdu) if a.hdu.lstrip("-").isdigit() else a.hdu
    out = a.output or os.path.splitext(os.path.basename(a.input))[0] + "_detected.fits"
    t0 = time.perf_counter()
    res = run_fits(a.input, out, hdu=hdu, device=a.device, ccl=a.ccl, label=a.label,
                   rawoutput=a.rawoutput, verbose=not a.quiet, **params)
    if not a.quiet:
        print(f"{res.num_detections} detections, S/N thresh {res.det_sn_threshold:.3f}, "
              f"median std {res.med_std:.6g}  -> {out}  [{time.perf_counter() - t0:.2f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
