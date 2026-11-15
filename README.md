# noisechisel-torch

[![tests](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml/badge.svg)](https://github.com/PabloMSanAla/noisechisel/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/noisechisel-torch)](https://pypi.org/project/noisechisel-torch/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PabloMSanAla/noisechisel/blob/main/notebooks/colab_gpu_benchmark.ipynb)

A PyTorch port of [Gnuastro](https://www.gnu.org/software/gnuastro/)'s
**NoiseChisel**, which detects faint signal in noise and measures the Sky and
Sky STD. It runs on CPU or CUDA, and its detection mask, Sky and Sky STD are
**bit-identical** to `astnoisechisel` 0.24 with default options.

![Gnuastro vs noisechisel-torch outputs](https://raw.githubusercontent.com/PabloMSanAla/noisechisel/e443b80d24baf68303d4649e5ebe3ccb41da189d/docs/images/comparison.png)

![Run time per 4088×4088 frame](https://raw.githubusercontent.com/PabloMSanAla/noisechisel/e443b80d24baf68303d4649e5ebe3ccb41da189d/docs/images/performance.png)

### Scaling with CPU threads

![Thread scaling](https://raw.githubusercontent.com/PabloMSanAla/noisechisel/b9a37c442fd206be9007d6060e61a725dd3fb5be/docs/images/scaling.png)

Both programs stop gaining beyond about 6–10 threads (2.3–2.5× over one
thread), so most of the run is serial or memory-bound. noisechisel-torch is
faster at every thread count. Data: `benchmarks/results/scaling_m3max.csv`,
made with `benchmarks/scaling.py`.

## Install

```bash
pip install "noisechisel-torch[fits]"            # CPU, with astropy for FITS I/O and the CLI
pip install "noisechisel-torch[fits,cuda12]"     # + CuPy for GPU labelling (cuda13 for CUDA 13 drivers)
```

Install only one CuPy package, matched to your driver's CUDA version.

## Usage

```python
from noisechisel_torch import noisechisel, NoiseChisel, read_image

img, hdr = read_image("image.fits", hdu=0)
res = noisechisel(img, device="cuda")   # or device="cpu", numthreads=8
res.detections                          # bool mask (H, W)
res.sky_image(), res.std_image()        # per-pixel Sky and Sky STD

nc = NoiseChisel(device="cuda", tilesize=(64, 64), qthresh=0.3)   # reuse on many images
res = nc(img)
```

```bash
noisechisel-torch image.fits -h 0 -o image_detected.fits --device cuda
noisechisel-torch image.fits -h 0 -o image_detected.fits --device cpu -N 8   # 8 CPU threads
```

Options keep Gnuastro's names and defaults (`tilesize`, `qthresh`, `erode`,
`snquant`, ...; see `NoiseChiselParams`). The output file has the same HDUs as
`astnoisechisel`. Add `--oneelempertile` to write Sky/STD with one value per
tile, which cuts a 4088² output from ~200 MB to ~85 MB.

## Validation

* Identical masks, Sky and STD on 18 Roman WFI detectors, on the frames shown
  above and on images with NaN regions.
* `benchmarks/compare_gnuastro.py` runs both programs on your images and
  reports pixel differences and timings. `benchmarks/readme_figures.py` makes
  the figures above, and `benchmarks/scaling.py` the thread-scaling test.
* CI runs the tests on Linux, macOS and Windows (Python 3.11/3.12) and
  compares against Gnuastro 0.24 on Linux and macOS.

## Development

```bash
mamba create -n noisechisel-dev -c conda-forge python=3.12 gnuastro=0.24 pytorch numpy scipy astropy pytest pip
mamba activate noisechisel-dev
pip install -e ".[test]"
pytest
```

To release, bump `version` in `pyproject.toml`. When the tests pass on `main`,
the `publish` workflow uploads the new version to PyPI.

## Limitations

* 2D single-channel images only. NaN pixels are never detected, and check
  images are not written.
* Apple MPS has no float64, so `device="mps"` runs on the CPU.
* Without CuPy, labelling on CUDA goes through host memory (scipy), which is
  slower.

## License

GPL-3.0-or-later. This is a derivative of Gnuastro, © Free Software Foundation.
