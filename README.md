# noisechisel-torch

[![tests](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml/badge.svg)](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/noisechisel-torch)](https://pypi.org/project/noisechisel-torch/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PabloMSanAla/noisechisel/blob/main/notebooks/colab_gpu.ipynb)

A PyTorch port of [Gnuastro](https://www.gnu.org/software/gnuastro/)'s
**NoiseChisel**, which detects faint signal in noise and measures the Sky and
Sky STD. It runs on CPU, CUDA or Apple GPUs (MPS), and its detection mask, Sky
and Sky STD are **bit-identical** to `astnoisechisel` 0.24 with default
options (MPS output is tested against the CPU output).

![Gnuastro vs noisechisel-torch outputs](https://raw.githubusercontent.com/PabloMSanAla/noisechisel/7e12eafcb5def5b281cd5c516f8d2b5dd89f612f/docs/images/comparison.png)

![CPU vs GPU run time per 4088×4088 frame](docs/images/performance.png)

### Scaling with CPU threads

![Thread scaling](https://raw.githubusercontent.com/PabloMSanAla/noisechisel/7e12eafcb5def5b281cd5c516f8d2b5dd89f612f/docs/images/scaling.png)

Both programs gain little beyond about 6–10 threads (at most 2.4–2.5× over
one thread), and noisechisel-torch is faster at every thread count. The CPU
times in the bar chart above are taken from this test at 10 threads.

All benchmarks use simulated Roman WFI frames (4088×4088). CPU: Apple M3 Max,
median of 4 runs including FITS read and write. GPU times are computation only:
Colab T4 with CuPy, and the M3 Max GPU through MPS (torch 2.15). On the same
frame, MPS took 1.71 s including FITS read and write.
Data: `benchmarks/results/scaling_m3max.csv`, made with `benchmarks/scaling.py`.

## Install

```bash
pip install "noisechisel-torch[fits]"            # CPU, with astropy for FITS I/O and the CLI
pip install "noisechisel-torch[fits,cuda12]"     # + CuPy for GPU labelling (cuda13 for CUDA 13 drivers)
```

Install only one CuPy package, matched to your driver's CUDA version.

## Usage

```python
from astropy.io import fits
from noisechisel_torch import noisechisel, NoiseChisel

img = fits.getdata("image.fits", ext=0)
res = noisechisel(img, device="cuda")   # or device="cpu", numthreads=8
res.detections                          # bool mask (H, W)
res.labels                              # int32 (H, W), 0 = sky, 1..N detections
res.sky, res.std                        # float32 (H, W) Sky and Sky STD (NaN where input is blank)
res.info()                              # detections, S/N threshold, Sky/STD summary
res.meta()                              # the same values as a dict

nc = NoiseChisel(device="cuda", tilesize=(64, 64), qthresh=0.3)   # reuse on many images
res = nc(img)
```

```bash
noisechisel-torch image.fits -h 0 -o image_detected.fits --device cuda
noisechisel-torch image.fits -h 0 -o image_detected.fits --device cpu -N 8   # 8 CPU threads
```

The output file has the same HDUs as `astnoisechisel`, including the
sky-subtracted `INPUT-NO-SKY`, which exists only in the file (in Python,
subtract `res.sky` yourself). Add `--oneelempertile` to write Sky/STD with one
value per tile, which cuts a 4088² output from ~200 MB to ~85 MB.

## Apple GPUs (MPS)

```python
res = noisechisel(img, device="mps")     # CLI: --device mps
```

MPS needs torch 2.7 or later and is never selected by default. It returns the
same bits as the CPU, with the result tensors on the GPU. Metal has no float64,
so the image convolution runs in a shader that does float64 arithmetic with
integers, the other float64 steps run on the CPU, and labelling uses scipy.

| Simulated frame | CPU (10 threads) | MPS |
|---|---|---|
| 512² | 0.067 s | 0.057 s |
| 1024² | 0.18 s | 0.13 s |
| 2048² | 0.52 s | 0.36 s |
| 4088² | 2.0 s | 1.2 s |
| 8176² | 7.9 s | 5.5 s |

Apple M3 Max, torch 2.15, computation only (`benchmarks/gpu_check.py` times the
4088² frame). 8176²: median of 5 runs; other sizes: best of 5 runs.

With a `RuntimeWarning`, the whole run moves to the CPU (and so do the result
tensors) if torch is too old, MPS is unavailable or an operator is missing, or
the image has non-zero values smaller than 2⁻¹⁰⁰ or larger than 2¹⁰⁰ in
magnitude (Apple GPUs flush tiny float32 values to zero). If only the shader
cannot be used, the convolution alone runs on the CPU.

## Gnuastro options

Every `astnoisechisel` option is a keyword argument with the same name and
default. Values that Gnuastro writes as `a,b` are tuples in Python. Booleans
are `True`/`False`. The CLI takes the Gnuastro syntax unchanged, including the
short forms.

```bash
astnoisechisel    img.fits -h0 --tilesize=50,40 -t0.35 -d0.8 --interponlyblank -k kernel.fits
noisechisel-torch img.fits -h0 --tilesize=50,40 -t0.35 -d0.8 --interponlyblank -k kernel.fits
```

```python
from astropy.io import fits
from noisechisel_torch import noisechisel

res = noisechisel(img, tilesize=(40, 50), qthresh=0.35, detgrowquant=0.8,
                  interponlyblank=True, kernel=fits.getdata("kernel.fits", ext=1))
```

**Tile sizes:** Gnuastro gives `--tilesize` and `--largetilesize` in FITS
order (x, y). Python uses numpy order (rows, cols), so `--tilesize=50,40` is
`tilesize=(40, 50)`. The CLI converts for you.

| Gnuastro | Python keyword | Default |
|---|---|---|
| `-k`, `--kernel FILE` (`--khdu`) | `kernel=fits.getdata(FILE)`; `None` = no convolution | built-in FWHM=2 Gaussian |
| `--widekernel FILE` (`--whdu`) | `widekernel=fits.getdata(FILE)` | `None` |
| `--tilesize` | `tilesize` | `(30, 30)` |
| `--remainderfrac` | `remainderfrac` | `0.1` |
| `--largetilesize` | `largetilesize` | `(200, 200)` |
| `--interpnumngb` | `interpnumngb` | `15` |
| `--interpmetric` | `interpmetric` (`"radial"` or `"manhattan"`) | `"radial"` |
| `--interponlyblank` | `interponlyblank` | `False` |
| `--smoothwidth` | `smoothwidth` | `5` |
| `-Q`, `--meanmedqdiff` | `meanmedqdiff` | `0.01` |
| `-t`, `--qthresh` | `qthresh` | `0.3` |
| `--noerodequant` | `noerodequant` | `0.99` |
| `--outliernumngb` | `outliernumngb` | `15` |
| `--outliersigma` | `outliersigma` | `5.0` |
| `--outliersclip` | `outliersclip` | `(3.0, 0.2)` |
| `-e`, `--erode` | `erode` | `2` |
| `--erodengb` | `erodengb` | `4` |
| `-p`, `--opening` | `opening` | `1` |
| `--openingngb` | `openingngb` | `8` |
| `-B`, `--minskyfrac` | `minskyfrac` | `0.7` |
| `--skyfracnoblank` | `skyfracnoblank` | `False` |
| `-s`, `--sigmaclip` | `sigmaclip` | `(3.0, 0.2)` |
| `-R`, `--dthresh` | `dthresh` | `0.0` |
| `--dopening` | `dopening` | `1` |
| `--dopeningngb` | `dopeningngb` | `4` |
| `--holengb` | `holengb` | `8` |
| `--pseudoconcomp` | `pseudoconcomp` | `4` |
| `-m`, `--snminarea` | `snminarea` | `10` |
| `--minnumfalse` | `minnumfalse` | `100` |
| `-c`, `--snquant` | `snquant` | `0.99` |
| `--snthresh` | `snthresh` (`None` = measure it) | `None` |
| `-d`, `--detgrowquant` | `detgrowquant` | `0.9` |
| `--detgrowmaxholesize` | `detgrowmaxholesize` | `100` |
| `--cleangrowndet` | `cleangrowndet` | `False` |
| `-N`, `--numthreads` | `numthreads` | all cores |

Output options are on the CLI and on `run_fits` / `write_fits`: `-h`/`--hdu`,
`-o`/`--output`, `--label` (int32 labels instead of a 0/1 mask), `--rawoutput`
(no `INPUT-NO-SKY`) and `--oneelempertile`. Options only in this version:
`device` and `ccl` (labelling backend). `NoiseChiselParams` lists the options
in Python, and `noisechisel-torch --help` on the command line.

Not available: check images (`--check*`), `--blankasforeground`,
`--numchannels` other than 1, and `--convolved`.

## Validation

* Identical masks, Sky and STD on 18 simulated Roman WFI detectors, on the frame shown
  above and on images with NaN regions.
* `benchmarks/compare_gnuastro.py` runs both programs on your images and
  reports pixel differences and timings. `benchmarks/readme_figures.py` makes
  the output comparison figure and `benchmarks/scaling.py` the timing figures.
* CI runs the tests on Linux, macOS and Windows (Python 3.11/3.12) and
  compares against Gnuastro 0.24 on Linux and macOS.

## Development

```bash
mamba create -n noisechisel-dev -c conda-forge python=3.12 gnuastro=0.24 pytorch numpy scipy astropy pytest pip
mamba activate noisechisel-dev
pip install -e ".[test]"
pytest
```

To release, bump `version` in `pyproject.toml`, merge to `main`, then tag that
commit and push the tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The `publish` workflow checks the tag is on `main` and matches the version,
runs the tests, and uploads to PyPI.

## Limitations

* 2D single-channel images only. NaN pixels are never detected, and check
  images are not written.
* `device="mps"` is tested on an M3 Max with macOS 26 and torch 2.7.0, 2.7.1
  and 2.15.
* Without CuPy, labelling on CUDA goes through host memory (scipy), which is
  slower.

## Citation

NoiseChisel is part of [Gnuastro](https://www.gnu.org/software/gnuastro/)
([NoiseChisel manual](https://www.gnu.org/software/gnuastro/manual/html_node/NoiseChisel.html)).
If you use this package, please cite the NoiseChisel papers:

* Akhlaghi & Ichikawa (2015), *Noise-based Detection and Segmentation of
  Nebulous Objects*, ApJS 220, 1.
  [doi:10.1088/0067-0049/220/1/1](https://doi.org/10.1088/0067-0049/220/1/1),
  [arXiv:1505.01664](https://arxiv.org/abs/1505.01664)
* Akhlaghi (2019), *Carving out the low surface brightness universe with
  NoiseChisel*. [arXiv:1909.11230](https://arxiv.org/abs/1909.11230)

`astnoisechisel --cite` prints the BibTeX for the Gnuastro version you have.
To cite this port, use `CITATION.cff` ("Cite this repository" on GitHub) or:

If this package is useful to you, please give the repository a ⭐ on
[GitHub](https://github.com/PabloMSanAla/noisechisel).

## License

GPL-3.0-or-later. This is a derivative of Gnuastro, © Free Software Foundation.
