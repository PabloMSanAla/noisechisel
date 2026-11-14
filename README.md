# noisechisel-torch

[![tests](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml/badge.svg)](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml)

A PyTorch re-implementation of [Gnuastro](https://www.gnu.org/software/gnuastro/)'s
**NoiseChisel** (2D detection of signal buried in noise, plus Sky and Sky-STD
measurement), following the C sources of Gnuastro 0.24 (`bin/noisechisel` and
the `lib/` routines it calls).

On the Roman WFI simulated images it was validated on, the output (detection
mask, Sky and Sky STD) is **bit-identical** to `astnoisechisel` 0.24 with
default options.  It runs on CPU or CUDA.

## Install

```bash
pip install "noisechisel-torch[fits] @ git+https://github.com/PabloMSanAla/noisechisel.git"
# or from a clone:
pip install .                 # core: numpy, scipy, torch
pip install ".[fits]"         # + astropy for FITS I/O and the CLI
pip install ".[fits,cuda]"    # + CuPy (CUDA 12) for fast GPU labelling
pip install -e ".[test]"      # development (pytest, astropy)
```

Development environment with Gnuastro (for comparisons):

```bash
mamba create -n noisechisel-dev -c conda-forge python=3.12 gnuastro=0.24 pytorch numpy scipy astropy pytest pip
mamba activate noisechisel-dev
pip install -e ".[test]"
pytest
```

## Usage

```python
from noisechisel_torch import noisechisel, NoiseChisel, read_image

img, hdr = read_image("image.fits", hdu=0)
res = noisechisel(img, device="cuda")      # or "cpu"
res.detections        # bool  (H, W)
res.labels            # int32 (H, W)  0 = sky, 1..N detections
res.sky_image()       # per-pixel Sky (tile values expanded)
res.std_image()       # per-pixel Sky STD
res.sky, res.std      # one value per tile

nc = NoiseChisel(device="cuda", tilesize=(64, 64), qthresh=0.3)   # reuse on many images
res = nc(img)
```

Option names and defaults are Gnuastro's (`tilesize`, `largetilesize`,
`qthresh`, `erode`, `snquant`, `detgrowquant`, `outliernumngb`, ...); see
`NoiseChiselParams`.

Command line (writes the same HDUs as `astnoisechisel`: `INPUT-NO-SKY`,
`DETECTIONS`, `SKY`, `SKY_STD`):

```bash
noisechisel-torch image.fits -h 0 -o image_detected.fits --device cuda --tilesize 30,30
```

## Validation against Gnuastro

```bash
python benchmarks/compare_gnuastro.py /path/to/*.fits --workdir out --device cpu --csv results.csv
```

runs both programs and reports pixel differences of the mask / Sky / STD and
the run times.  `tests/test_noisechisel.py::test_identical_to_gnuastro` does
the same on a synthetic image when `astnoisechisel` is on the `PATH`.

Results on the 18 Roman WFI detectors of PID1083 (4088×4088, Apple M3 Max,
default options):

| | Gnuastro 0.24 (14 threads) | noisechisel-torch (CPU) |
|---|---|---|
| mean time / detector | 3.80 s | 2.25 s (1.7× faster) |
| mask / Sky / STD pixels differing | — | 0 / 0 / 0 on all 18 |

The S/N thresholds match on every detector (6.11–6.33). An image with NaN
regions also gives identical output.

What it takes to be bit-exact (relevant when the Sky level is much larger than
the noise, e.g. Roman count-rate images where Sky/σ ≈ 4·10⁴, so one float32 ulp
is ≈ 0.004σ and tile ranks are very sensitive):

* the built-in kernel is applied as a **correlation**; kernels you pass are
  normalised and flipped (true convolution), as Gnuastro does for kernel files;
* convolution/smoothing: float32 products accumulated in double, in kernel
  raster order, divided by the overlapping kernel sum (`gal_convolve_spatial`);
* neighbour interpolation and outlier rejection visit tiles in Gnuastro's queue
  order (distance, then FIFO), which also counts the central tile twice;
* sigma clipping and MAD clipping follow `gal_statistics_clip_{sigma,mad}` of
  0.24 (statistics of the last measured round, sequential double sums);
* outliers of the quantile thresholds via `gal_statistics_outlier_bydistance`;
* S/N of pseudo-detections uses Gnuastro's `gal_tile_full_id_from_coord`
  pixel→tile mapping (which puts the first pixel of the 2nd tile in the 1st
  tile when the first tile is not of regular size).

## Notes / limitations

* 2D, single channel only; NaN pixels are never detected (no
  `--blankasforeground`); no check images.
* Apple MPS has no float64, so `device="mps"` falls back to the CPU.
* Connected components: `ccl="cupy"` (CUDA, `cupyx.scipy.ndimage.label`,
  install with `pip install "noisechisel-torch[cuda]"`), `ccl="torch"`
  (iterative label propagation, any device, slow) or `ccl="scipy"` (CPU).
  `"auto"` uses cupy on CUDA when installed, else torch; scipy on CPU.  All
  produce identical labels.
* CUDA was not available for testing.  On CUDA, `cumsum` is a parallel scan,
  so Sky STD values may differ from Gnuastro in the last float32 bit.

## License

GPL-3.0-or-later (derived from Gnuastro, © Free Software Foundation).
