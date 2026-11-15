import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

import noisechisel_torch.core as C
from noisechisel_torch import DEFAULT_KERNEL, NoiseChisel, noisechisel


def synthetic_image(n=800, sky=100.0, sigma=5.0, nsrc=150, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.normal(sky, sigma, (n, n))
    yy, xx = np.mgrid[:n, :n]
    truth = np.zeros((n, n), bool)
    for _ in range(nsrc):
        y, x = rng.uniform(20, n - 20, 2)
        s = rng.uniform(1.5, 6)
        a = rng.uniform(3, 40) * sigma
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        img += a * np.exp(-r2 / (2 * s * s))
        truth |= r2 < (2 * s) ** 2
    return img.astype(np.float32), truth


@pytest.fixture(scope="module")
def image():
    return synthetic_image()


@pytest.fixture(scope="module")
def result(image):
    return noisechisel(image[0], device="cpu")


def test_outputs(result, image):
    img, truth = image
    assert result.detections.shape == img.shape
    assert result.labels.max() == result.num_detections > 0
    assert result.sky.shape == result.std.shape == img.shape
    # every bright source centre is detected
    assert result.detections.numpy()[truth].mean() > 0.95


def test_sky_and_std(result):
    sky = result.sky.numpy()
    std = result.std.numpy()
    assert abs(np.median(sky) - 100.0) < 0.3
    assert abs(np.median(std) - 5.0) / 5.0 < 0.05


def test_convolution_is_correlation_with_edge_correction():
    rng = np.random.default_rng(3)
    img = torch.as_tensor(rng.normal(0, 1, (60, 70)).astype(np.float32))
    k = torch.as_tensor(DEFAULT_KERNEL)
    out = C._convolve(img, k).numpy()
    x = img.numpy().astype(np.float64)
    kd = DEFAULT_KERNEL.astype(np.float64)
    # interior: plain correlation divided by the kernel sum
    y, xx = 30, 40
    ref = np.sum(x[y - 5:y + 6, xx - 5:xx + 6] * kd) / kd.sum()
    assert out[y, xx] == pytest.approx(ref, rel=1e-6)
    # corner: only the overlapping part of the kernel is used and normalised
    ref = np.sum(x[:6, :6] * kd[5:, 5:]) / kd[5:, 5:].sum()
    assert out[0, 0] == pytest.approx(ref, rel=1e-6)


def _clip_sigma_reference(values, mult, param, maxnum=50):
    """Literal Python port of Gnuastro 0.24 ``statistics_clip`` (sigma)."""
    a = np.sort(np.asarray(values, np.float32))
    param = float(np.float32(param))
    bytol = param < 1
    maxnum = maxnum if bytol else int(param)
    start, size, num, oldspread = 0, a.size, 0, math.nan
    while num < maxnum and size:
        cur = a[start:start + size]
        n = cur.size
        center = float(cur[n // 2] if n % 2 else np.float32((cur[n // 2] + cur[n // 2 - 1]) / np.float32(2)))
        s = s2 = 0.0
        for v in cur.astype(np.float64):
            s += v
            s2 += v * v
        ss = s * s / n
        spread = 0.0 if n == 1 or ss > s2 else math.sqrt((s2 - ss) / n)
        mean = s / n
        if spread == 0 or (bytol and num > 0 and (oldspread - spread) / spread < param):
            break
        lo = np.nonzero(cur > center - mult * spread)[0][0]
        hi = np.nonzero(cur < center + mult * spread)[0][-1]
        start, size = start + lo, hi - lo + 1
        oldspread = spread
        num += 1
    if size == 0 or (bytol and num == maxnum):
        return math.nan, math.nan, math.nan
    return center, mean, spread


def test_sigma_clip_matches_reference():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(20):
        v = rng.normal(10, 2, 300 + 10 * i)
        v[:5] += 50                                   # outliers
        rows.append(v.astype(np.float32))
    P = max(r.size for r in rows)
    x = torch.full((len(rows), P), float("nan"))
    for i, r in enumerate(rows):
        x[i, :r.size] = torch.as_tensor(r)
    xs, n = C._sort_rows(x)
    med, mean, std = C._sigma_clip_rows(xs, n, 3.0, 0.2)
    for i, r in enumerate(rows):
        rm, rmean, rstd = _clip_sigma_reference(r, 3.0, 0.2)
        assert float(med[i]) == pytest.approx(rm, rel=1e-7)
        assert float(mean[i]) == pytest.approx(rmean, rel=1e-12)
        assert float(std[i]) == pytest.approx(rstd, rel=1e-12)


def _gnuastro_queue_order(ny, nx, y0, x0, n):
    """Literal emulation of Gnuastro's sorted linked-list neighbour queue."""
    checked = np.zeros((ny, nx), bool)
    Q, out = [(0.0, y0, x0)], []
    while Q and len(out) < n:
        _, y, x = Q.pop()
        out.append((y - y0, x - x0))
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < ny and 0 <= xx < nx and not checked[yy, xx]:
                d = np.float32(math.sqrt((yy - y0) ** 2 + (xx - x0) ** 2))
                i = 0
                while i < len(Q) and not d >= Q[i][0]:
                    i += 1
                Q.insert(i, (d, yy, xx))
                checked[yy, xx] = True
    return out


@pytest.mark.parametrize("pos", [(20, 20), (0, 0), (2, 39), (39, 5)])
def test_neighbour_order(pos):
    ny = nx = 40
    ref = _gnuastro_queue_order(ny, nx, *pos, 150)
    order = C._gnuastro_order(12, "radial")
    mine = [(int(a), int(b)) for a, b in order
            if 0 <= pos[0] + a < ny and 0 <= pos[1] + b < nx][:150]
    # compare only within the complete disc of radius 12
    k = sum(1 for a, b in ref if math.hypot(a, b) <= 12)
    # Gnuastro re-queues the centre after the first neighbour it pops (which
    # depends on the edges): only the first three positions may be permuted.
    assert sorted(mine[:3]) == sorted(ref[:3])
    assert mine[3:k] == ref[3:k]


def test_params_validation():
    with pytest.raises(TypeError):
        NoiseChisel(device="cpu", not_an_option=1)
    with pytest.raises(ValueError):
        NoiseChisel(device="cpu", erodengb=6)


@pytest.mark.skipif(shutil.which("astnoisechisel") is None, reason="Gnuastro not installed")
def test_identical_to_gnuastro(tmp_path, image):
    fits = pytest.importorskip("astropy.io.fits")
    img = image[0]
    inp = tmp_path / "in.fits"
    out = tmp_path / "out.fits"
    fits.writeto(inp, img)
    subprocess.run(["astnoisechisel", str(inp), "-h0", "-o", str(out), "--quiet"], check=True)
    with fits.open(out) as hl:
        det = hl["DETECTIONS"].data.astype(bool)
        sky = hl["SKY"].data
        std = hl["SKY_STD"].data
    res = noisechisel(img, device="cpu")
    assert np.array_equal(res.detections.numpy(), det)
    assert np.array_equal(res.sky.numpy(), sky, equal_nan=True)
    assert np.array_equal(res.std.numpy(), std, equal_nan=True)


def test_fits_roundtrip(tmp_path, image):
    fits = pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import run_fits
    inp = tmp_path / "in.fits"
    out = tmp_path / "out.fits"
    fits.writeto(inp, image[0])
    res = run_fits(str(inp), str(out), device="cpu")
    with fits.open(out) as hl:
        assert [h.name for h in hl] == ["NOISECHISEL-CONFIG", "INPUT-NO-SKY", "DETECTIONS",
                                        "SKY", "SKY_STD"]
        assert hl["DETECTIONS"].data.sum() == int(res.detections.sum())


def test_fits_oneelempertile(tmp_path, image):
    fits = pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import run_fits
    inp = tmp_path / "in.fits"
    out = tmp_path / "out.fits"
    fits.writeto(inp, image[0])
    res = run_fits(str(inp), str(out), device="cpu", oneelempertile=True)
    with fits.open(out) as hl:
        assert hl["SKY"].data.shape == tuple(res.sky_tiles.shape)
        assert np.array_equal(hl["SKY_STD"].data, res.std_tiles.numpy(), equal_nan=True)
        assert hl["DETECTIONS"].data.shape == image[0].shape


def test_write_failure_removes_partial_file(tmp_path, image, monkeypatch):
    fits = pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import noisechisel, write_fits
    res = noisechisel(image[0], device="cpu")
    out = tmp_path / "out.fits"

    def full_disk(self, path, overwrite=False):
        open(path, "wb").write(b"\0" * 2880)
        raise OSError("16711744 requested and 0 written")
    monkeypatch.setattr(fits.HDUList, "writeto", full_disk)
    with pytest.raises(OSError, match="MB needed"):
        write_fits(res, str(out))
    assert not out.exists()


def test_raster_relabel():
    rng = np.random.default_rng(0)
    m = torch.as_tensor(rng.random((200, 150)) > 0.5)
    ref, n = C._label_scipy(m, 8)
    perm = torch.cat([torch.zeros(1, dtype=torch.int32), torch.randperm(n).int() + 1])
    assert torch.equal(C._raster_relabel(perm[ref.long()], n), ref)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@cuda
@pytest.mark.skipif(not C._cupy_available(), reason="CuPy not installed")
def test_ccl_cupy_matches_scipy():
    rng = np.random.default_rng(0)
    m = torch.as_tensor(rng.random((500, 400)) > 0.55)
    for conn in (4, 8):
        a, na = C._label_cupy(m.cuda(), conn)
        b, nb = C._label_scipy(m, conn)
        assert na == nb
        assert torch.equal(a.cpu(), b)


@cuda
def test_cuda_matches_cpu(image, result):
    for ccl in ["scipy"] + (["cupy"] if C._cupy_available() else []):
        res = noisechisel(image[0], device="cuda", ccl=ccl)
        assert torch.equal(res.detections.cpu(), result.detections)
        assert torch.equal(res.labels.cpu(), result.labels)
        assert torch.equal(res.sky.cpu(), result.sky)
        assert torch.equal(res.std.cpu(), result.std)


def test_numthreads(image):
    prev = torch.get_num_threads()
    ref = noisechisel(image[0], device="cpu")
    one = noisechisel(image[0], device="cpu", numthreads=1)
    assert torch.get_num_threads() == prev
    assert torch.equal(ref.detections, one.detections)
    assert torch.equal(ref.sky, one.sky) and torch.equal(ref.std, one.std)
    with pytest.raises(ValueError):
        NoiseChisel(numthreads=-1)


def test_fits_getdata_input(tmp_path, image):
    """Arrays from astropy.io.fits.getdata are big-endian; results must not change."""
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "in.fits"
    fits.writeto(path, image[0])
    data = fits.getdata(path)
    assert data.dtype.byteorder == ">"
    a = noisechisel(data, device="cpu")
    b = noisechisel(image[0], device="cpu")
    assert torch.equal(a.detections, b.detections) and torch.equal(a.sky, b.sky)


def test_cli_tilesize_fits_order(tmp_path, image, monkeypatch):
    """--tilesize=X,Y is in FITS order, as in astnoisechisel."""
    pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import cli
    seen = {}
    monkeypatch.setattr("noisechisel_torch.io.run_fits", lambda *a, **k: seen.update(k) or
                        noisechisel(image[0], device="cpu"))
    cli.main([str(tmp_path / "x.fits"), "-q", "--tilesize", "30,45", "--largetilesize", "200,100"])
    assert seen["tilesize"] == (45, 30) and seen["largetilesize"] == (100, 200)


@pytest.mark.skipif(shutil.which("astnoisechisel") is None, reason="Gnuastro not installed")
def test_cli_options_identical_to_gnuastro(tmp_path, image):
    """Same command line (custom asymmetric kernel, non-square tiles, short
    options, a flag) gives the same output as astnoisechisel."""
    fits = pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import cli, gaussian_kernel
    inp, kern = tmp_path / "in.fits", tmp_path / "k.fits"
    fits.writeto(inp, image[0])
    k = np.asarray(gaussian_kernel(2.5, 5), dtype=np.float32).copy()
    k[:, :2] *= 1.5
    fits.writeto(kern, k)
    opts = ["-k", str(kern), "--khdu=0", "--tilesize=30,40", "-t0.35", "--interponlyblank"]
    subprocess.run(["astnoisechisel", str(inp), "-h0", "-o", str(tmp_path / "g.fits"),
                    "--quiet"] + opts, check=True)
    assert cli.main([str(inp), "-h0", "-o", str(tmp_path / "t.fits"), "-q"] + opts) == 0
    for e in ("DETECTIONS", "SKY", "SKY_STD"):
        a = fits.getdata(tmp_path / "g.fits", e)
        b = fits.getdata(tmp_path / "t.fits", e)
        assert np.array_equal(a, b, equal_nan=True), e
