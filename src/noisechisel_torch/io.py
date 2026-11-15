# SPDX-License-Identifier: GPL-3.0-or-later
"""FITS input/output with the same HDU layout as ``astnoisechisel``.

Requires ``astropy`` (``pip install noisechisel-torch[fits]``).
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional, Union

import numpy as np

from .core import NoiseChisel, NoiseChiselResult


def _fits():
    try:
        from astropy.io import fits
    except ImportError as e:  # pragma: no cover
        raise ImportError("FITS I/O needs astropy: pip install 'noisechisel-torch[fits]'") from e
    return fits


def _wcs_header(header):
    if header is None:
        return None
    try:
        from astropy.wcs import WCS
        return WCS(header).to_header(relax=True)
    except Exception:
        return None


def write_fits(res: NoiseChiselResult, path: str, header=None, config: Optional[dict] = None,
               overwrite: bool = True, rawoutput: bool = False, label: bool = False,
               oneelempertile: bool = False, image=None):
    """Write the result like ``astnoisechisel``: HDUs ``INPUT-NO-SKY``,
    ``DETECTIONS`` (uint8, or int32 labels with ``label=True``), ``SKY`` and
    ``SKY_STD``.  With ``rawoutput=True`` only DETECTIONS/SKY/SKY_STD are
    written; ``INPUT-NO-SKY`` needs the input ``image``.  With ``oneelempertile=True`` SKY/SKY_STD hold one value per
    tile instead of being expanded to full resolution (much smaller file)."""
    fits = _fits()
    wh = _wcs_header(header)

    def img(data, name, extra=None):
        h = fits.ImageHDU(data=data, header=wh.copy() if wh is not None else None, name=name)
        for k, v in (extra or {}).items():
            h.header[k] = v
        return h

    phdu = fits.PrimaryHDU()
    phdu.header["EXTNAME"] = "NOISECHISEL-CONFIG"
    phdu.header["PROGRAM"] = "noisechisel-torch"
    for k, v in (config or {}).items():
        if k in ("tilesize", "largetilesize") and isinstance(v, (tuple, list)):
            v = tuple(v)[::-1]          # FITS (x, y) order, as Gnuastro writes it
        if isinstance(v, (tuple, list)):
            v = ",".join(str(x) for x in v)
        if v is None:
            v = "none"
        phdu.header["HIERARCH " + k.upper()] = v
    hdus = [phdu]
    sky_full = res.sky.cpu().numpy()
    blank = np.isnan(np.asarray(image, dtype=np.float32)) if image is not None else np.isnan(sky_full)
    if oneelempertile:
        sky = res.sky_tiles.cpu().numpy().astype(np.float32)
        std = res.std_tiles.cpu().numpy().astype(np.float32)
    else:
        sky = sky_full.astype(np.float32)
        std = res.std.cpu().numpy().astype(np.float32)
    if not rawoutput and image is not None:
        hdus.append(img(np.asarray(image, dtype=np.float32) - sky_full, "INPUT-NO-SKY"))
    det = (res.labels.cpu().numpy().astype(np.int32) if label
           else res.detections.cpu().numpy().astype(np.uint8))
    extra = {"DETSN": res.det_sn_threshold, "NUMLABS": res.num_detections}
    if blank.any():
        # Blank input pixels are blank in all outputs (as astnoisechisel);
        # res.sky / res.std already are.
        if label:
            det[blank] = -2147483648
            extra["BLANK"] = -2147483648
        else:
            det[blank] = 255
            extra["BLANK"] = 255
    hdus.append(img(det, "DETECTIONS", extra))
    if oneelempertile:
        wh = None   # the input WCS does not apply to the tile grid
    hdus.append(img(sky, "SKY"))
    hdus.append(img(std, "SKY_STD", {"MEDSTD": res.med_std, "MINSTD": res.min_std,
                                     "MAXSTD": res.max_std}))
    try:
        fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    except OSError as e:
        # Don't leave a truncated file behind (e.g. when the disk is full).
        if os.path.exists(path):
            os.remove(path)
        need = sum(h.data.nbytes for h in hdus if h.data is not None) / 2**20
        raise OSError(f"could not write {path} ({need:.0f} MB needed): {e}") from e


def run_fits(input_path: str, output_path: Optional[str] = None, hdu: Union[int, str] = 0,
             device=None, label: bool = False, rawoutput: bool = False,
             oneelempertile: bool = False, **kwargs):
    """Run NoiseChisel on a FITS image; optionally write the output file.
    Extra keywords go to :class:`~noisechisel_torch.NoiseChisel`."""
    data, header = _fits().getdata(input_path, ext=hdu, header=True)
    nc = NoiseChisel(device=device, **kwargs)
    res = nc(data)
    if output_path:
        write_fits(res, output_path, header=header, config=asdict(nc.p), image=data,
                   rawoutput=rawoutput, label=label, oneelempertile=oneelempertile)
    return res
