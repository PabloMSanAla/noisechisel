"""Apple GPU: ``device="mps"`` must give bit-identical results to the CPU."""
import warnings

import numpy as np
import pytest
import torch

import noisechisel_torch.core as C
from noisechisel_torch import DEFAULT_KERNEL, NoiseChisel, gaussian_kernel, noisechisel
from noisechisel_torch import _mps
from test_noisechisel import synthetic_image


def _usable_mps():
    # Virtual machines (CI runners) may report MPS but be unable to allocate.
    if not (_mps.is_available() and hasattr(torch.mps, "compile_shader")):
        return False
    try:
        torch.ones(4, device="mps").sum().item()
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _usable_mps(),
                                reason="no usable MPS device, or torch older than 2.7")

TENSORS = ("detections", "labels", "sky", "std", "sky_tiles", "std_tiles")
SCALARS = ("num_detections", "num_initial_detections", "det_sn_threshold",
           "min_std", "med_std", "max_std")


def assert_same(a, b, name=""):
    """Equal shapes, types and values; NaNs in the same places."""
    a, b = a.cpu(), b.cpu()
    assert a.shape == b.shape and a.dtype == b.dtype, name
    same = (a == b) | (torch.isnan(a) & torch.isnan(b)) if a.is_floating_point() else a == b
    assert bool(same.all()), f"{name}: {int((~same).sum())} of {a.numel()} differ"


def compare(img, **kwargs):
    """Run on both devices, with any CPU fallback on MPS turned into an error."""
    ref = NoiseChisel(device="cpu", **kwargs)(img)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = NoiseChisel(device="mps", **kwargs)(img)
    for k in TENSORS:
        assert getattr(res, k).device.type == "mps", k
        assert_same(getattr(res, k), getattr(ref, k), k)
    for k in SCALARS:
        assert getattr(res, k) == getattr(ref, k), k
    a, b = res.meta(), ref.meta()
    assert a.pop("device").startswith("mps") and b.pop("device") == "cpu"
    assert a == b
    return res


def test_metal_library_builds():
    """The self-tests succeed (else MPS silently loses speed or is not used)."""
    assert _mps.unsupported_reason() is None
    assert _mps.library() is not None, _mps._library_error


# ------------------------------------------------------------- whole pipeline
def with_blanks(img):
    img = img.copy()
    img[100:180, 300:420] = np.nan
    img[np.random.default_rng(2).random(img.shape) < 0.002] = np.nan
    img[5, 7], img[6, 8] = np.inf, -np.inf
    return img


CASES = {
    "default": (lambda im: im, {}),
    "blanks and inf": (with_blanks, {}),
    "sky >> noise": (lambda im: ((im - 100) * 1.4e-6 + 0.2846).astype(np.float32), {}),
    "zero sky": (lambda im: im - 100, {}),
    "non-square image and tiles": (lambda im: im[:700, :530], {"tilesize": (37, 24)}),
    "whole-image holes and opening": (lambda im: im, {"largetilesize": None}),
    "no growth": (lambda im: im, {"detgrowquant": 1.0}),
    "clean grown detections": (lambda im: im, {"cleangrowndet": True}),
    "fixed S/N threshold": (lambda im: im, {"snthresh": 6.0}),
    "manhattan, blank-only interpolation": (
        lambda im: im, {"interpmetric": "manhattan", "interponlyblank": True}),
    "no convolution": (lambda im: im, {"kernel": None}),
    "wide kernel": (lambda im: im, {"widekernel": gaussian_kernel(5.0)}),
    "asymmetric kernel": (
        lambda im: im, {"kernel": gaussian_kernel(2.5, 5) * np.linspace(1, 1.5, 5, dtype=np.float32)}),
}


@pytest.mark.parametrize("name", CASES)
def test_pipeline_matches_cpu(name):
    make, kwargs = CASES[name]
    compare(make(synthetic_image()[0]), **kwargs)


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_pipeline_matches_cpu_other_images(seed):
    compare(synthetic_image(n=600, nsrc=100, sky=-3.0, sigma=0.01, seed=seed)[0])


def test_input_types():
    img = synthetic_image()[0]
    ref = noisechisel(img, device="cpu")
    inputs = [img.astype(">f4"), img.astype(np.float64), np.asfortranarray(img),
              torch.from_numpy(img), torch.from_numpy(img).double(),
              torch.from_numpy(img).to("mps")]
    for x in inputs:
        assert_same(noisechisel(x, device="mps").sky, ref.sky)


def test_input_is_not_modified():
    img = with_blanks(synthetic_image()[0])
    before = img.copy()
    noisechisel(img, device="mps")
    assert np.array_equal(img, before, equal_nan=True)


# --------------------------------------------------------------------- stages
def conv_both(img, kernel):
    img = torch.as_tensor(img, dtype=torch.float32)
    kernel = torch.as_tensor(np.array(kernel, dtype=np.float32))
    return C._convolve(img.to("mps"), kernel.to("mps")), C._convolve(img, kernel)


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (11, 11), (12, 200), (513, 257)])
@pytest.mark.parametrize("blank", [0.0, 0.3])
def test_convolution_shapes(shape, blank):
    rng = np.random.default_rng(7)
    img = rng.normal(0, 1, shape).astype(np.float32)
    img[rng.random(shape) < blank] = np.nan
    assert_same(*conv_both(img, DEFAULT_KERNEL))


def on_gpu(img, kernel):
    """The shader result, requiring that it did not defer to the CPU."""
    out = _mps.convolve(torch.as_tensor(img).to("mps"), torch.as_tensor(kernel))
    if _mps.library() is None:
        pytest.skip("Metal shader unavailable")
    assert out is not None
    return out


@pytest.mark.parametrize("kind", ["noise", "sky >> noise", "wide range", "integers", "zeros",
                                  "signed zeros", "cancellation"])
def test_shader_convolution_values(kind):
    rng = np.random.default_rng(11)
    n = (600, 700)
    img = {"noise": lambda: rng.normal(0, 1, n),
           "sky >> noise": lambda: rng.normal(0.2846, 7e-6, n),
           "wide range": lambda: rng.normal(0, 1, n) * 10.0 ** rng.uniform(-12, 12, n),
           "integers": lambda: rng.integers(-3, 4, n),                    # many ties
           "zeros": lambda: np.zeros(n),
           "signed zeros": lambda: np.where(rng.random(n) < 0.5, -0.0, 0.0),
           "cancellation": lambda: np.indices(n).sum(0) % 2 * 2e6 - 1e6}[kind]().astype(np.float32)
    kernel = DEFAULT_KERNEL.copy()
    assert_same(on_gpu(img, kernel), C._convolve(torch.from_numpy(img), torch.from_numpy(kernel)))


@pytest.mark.parametrize("kernel", [
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],                 # negative taps
    [[1, 0, -1]],                                          # zero kernel sum
    [[0.25, 0.5, 0.25]], [[1], [3], [1]],
])
def test_shader_convolution_kernels(kernel):
    rng = np.random.default_rng(13)
    img = rng.integers(-5, 6, (90, 80)).astype(np.float32)
    img[rng.random(img.shape) < 0.1] = np.nan
    kernel = np.array(kernel, dtype=np.float32)
    assert_same(on_gpu(img, kernel), C._convolve(torch.from_numpy(img), torch.from_numpy(kernel)))


def test_convolution_is_chunked(monkeypatch):
    monkeypatch.setattr(_mps, "_TAPS_PER_DISPATCH", 100_000)
    img = np.random.default_rng(1).normal(0, 1, (301, 211)).astype(np.float32)
    assert_same(on_gpu(img, DEFAULT_KERNEL.copy()),
                C._convolve(torch.from_numpy(img), torch.from_numpy(DEFAULT_KERNEL.copy())))


@pytest.mark.parametrize("scale", [1e-45, 1e-40, 1e-36, 1e-31, 1e37])
def test_convolution_unsafe_values_use_cpu(scale):
    """Values whose products would be subnormal (or overflow) are refused by
    the shader and convolved on the CPU."""
    img = (np.random.default_rng(1).uniform(1, 2, (64, 64)) * scale).astype(np.float32)
    k = torch.from_numpy(DEFAULT_KERNEL.copy())
    assert _mps.convolve(torch.from_numpy(img).to("mps"), k) is None
    assert_same(*conv_both(img, DEFAULT_KERNEL))


def test_convolution_subnormal_quotient_uses_cpu():
    """Normal inputs that cancel to a subnormal result."""
    tiny = np.float32(2.0 ** -100)
    img = np.array([[tiny, np.nextafter(tiny, np.float32(1)), 0]] * 4, dtype=np.float32)
    k = np.array([[2.0 ** -3, -(2.0 ** -3), 2.0 ** 20]], dtype=np.float32)
    if _mps.library() is not None:
        assert _mps.convolve(torch.from_numpy(img).to("mps"), torch.from_numpy(k)) is None
    assert_same(*conv_both(img, k))


def test_row_statistics_helpers():
    rng = np.random.default_rng(4)
    x = torch.from_numpy(rng.normal(0, 1, (50, 400)).astype(np.float32))
    x[torch.from_numpy(rng.random((50, 400)) < 0.2)] = float("nan")
    (xs, n), (gs, gn) = C._sort_rows(x), C._sort_rows(x.to("mps"))
    assert_same(gs, xs)
    assert_same(gn, n)
    for q in (0.0, 0.3, 0.5, 0.99, 1.0):
        assert_same(C._row_quantile(gs, gn, q), C._row_quantile(xs, n, q), q)


@pytest.mark.parametrize("conn", [4, 8])
def test_morphology_and_holes(conn):
    mask = torch.from_numpy(np.random.default_rng(9).random((450, 380)) > 0.45)
    a = C._Canvas(mask.shape, (200, 200), 0.1, torch.device("cpu"))
    b = C._Canvas(mask.shape, (200, 200), 0.1, torch.device("mps"))
    for op in ("erode", "dilate", "open"):
        assert_same(getattr(b, op)(mask.to("mps"), 2, conn), getattr(a, op)(mask, 2, conn), op)
    for maxsize in (None, 3):
        assert_same(b.fill_holes(mask.to("mps"), conn, maxsize, C._label_scipy),
                    a.fill_holes(mask, conn, maxsize, C._label_scipy), maxsize)


# ------------------------------------------------------------ soft float64
def random_doubles(rng, n):
    """float32 products and short sums of them: what the shader adds."""
    def f32():
        e = rng.integers(-60, 60, n)
        return (rng.uniform(-2, 2, n) * 2.0 ** e).astype(np.float32).astype(np.float64)
    a, b = f32(), f32()
    wide = a + b * 2.0 ** rng.integers(-70, 71, n)           # all 53 bits in use
    ties = np.ldexp(rng.integers(-2 ** 12, 2 ** 12, n).astype(np.float64),
                    rng.integers(-60, 2, n)) + 2.0 ** 53 * rng.integers(-1, 2, n)
    special = np.array([0.0, -0.0, 1.0, -1.0, 2.0 ** -53, 1 + 2.0 ** -52, 2.0 ** 53])
    return np.concatenate([a, wide, ties, rng.choice(special, n)])


def bits(x):
    return torch.from_numpy(x.view(np.int64).copy()).to("mps")


def test_soft_double_add():
    lib = _mps.library()
    if lib is None:
        pytest.skip("Metal shader unavailable")
    rng = np.random.default_rng(21)
    a = random_doubles(rng, 100_000)
    b = rng.permutation(a)
    near = -a * (1 + rng.integers(-3, 4, a.size) * 2.0 ** -52)    # cancellation
    for y in (b, -a, near, a):
        out = torch.empty(a.size, dtype=torch.int64, device="mps")
        lib.nc_test_dadd(bits(a), bits(y), out, threads=a.size)
        assert np.array_equal(out.cpu().numpy(), (a + y).view(np.int64))


def test_soft_double_divide():
    lib = _mps.library()
    if lib is None:
        pytest.skip("Metal shader unavailable")
    rng = np.random.default_rng(22)
    a = random_doubles(rng, 100_000)
    b = rng.permutation(a)
    # quotients at the midpoint of two float32 values, where the two roundings
    # depend on the last bit of the double quotient
    mid = 1 + (2 * rng.integers(0, 2 ** 23, 100_000) + 1) * 2.0 ** -24
    den = rng.uniform(1, 2, mid.size)
    near = mid * den
    near = near + rng.integers(-1, 2, mid.size) * np.spacing(near)
    a, b = np.concatenate([a, near]), np.concatenate([b, den])
    keep = b != 0
    a, b = a[keep], b[keep]
    out = torch.empty(a.size, dtype=torch.float32, device="mps")
    bad = torch.empty(a.size, dtype=torch.int32, device="mps")
    lib.nc_test_ddiv(bits(a), bits(b), out, bad, threads=a.size)
    with np.errstate(over="ignore", under="ignore"):
        ref = (a / b).astype(np.float32)
    good = ~bad.cpu().numpy().astype(bool)
    assert np.array_equal(out.cpu().numpy()[good].view(np.int32), ref[good].view(np.int32))
    # only results outside the normal float32 range are refused
    flagged = np.abs(ref[~good])
    assert np.all((flagged <= 2.0 ** -126) | (flagged >= 2.0 ** 127))
    assert good.mean() > 0.5


# ------------------------------------------------------------------ fallbacks
def test_unsafe_image_runs_on_cpu():
    img = synthetic_image()[0]
    img[0, 0] = 1e-40
    ref = noisechisel(img, device="cpu")
    with pytest.warns(RuntimeWarning, match="on the CPU"):
        res = noisechisel(img, device="mps")
    assert res.detections.device.type == "cpu"
    assert_same(res.sky, ref.sky)


def test_missing_operator_runs_on_cpu(monkeypatch):
    img = synthetic_image()[0]
    ref = noisechisel(img, device="cpu", widekernel=gaussian_kernel(5.0))
    nc = NoiseChisel(device="mps", widekernel=gaussian_kernel(5.0))
    run = NoiseChisel._run

    def fake(self, image):
        if self.device.type == "mps":
            raise NotImplementedError("The operator 'aten::x' is not implemented for MPS")
        return run(self, image)
    monkeypatch.setattr(NoiseChisel, "_run", fake)
    with pytest.warns(RuntimeWarning, match="aten::x"):
        res = nc(img)
    assert res.detections.device.type == "cpu" and nc.device.type == "mps"
    assert nc.kernel.device.type == "mps"
    for k in TENSORS:
        assert_same(getattr(res, k), getattr(ref, k), k)


def test_pipeline_errors_are_not_swallowed():
    with pytest.raises(ValueError):
        noisechisel(np.random.default_rng(0).normal(0, 1, (60, 60)), device="mps")


def test_without_metal_shader(monkeypatch):
    """The shader fails to build: convolution on the CPU, the rest on the GPU."""
    monkeypatch.setattr(_mps, "_library", None)
    monkeypatch.setattr(_mps, "_library_error", None)

    def broken(source):
        raise RuntimeError("compilation failed")
    _mps.unsupported_reason()
    monkeypatch.setattr(torch.mps, "compile_shader", broken)
    img = synthetic_image()[0]
    with pytest.warns(RuntimeWarning, match="Metal convolution unavailable"):
        res = noisechisel(img, device="mps")
    assert res.detections.device.type == "mps"
    assert_same(res.sky, noisechisel(img, device="cpu").sky)


@pytest.mark.parametrize("patch, message", [
    (lambda m: m.setattr(_mps, "is_available", lambda: False), "not available"),
    (lambda m: m.delattr(torch.mps, "compile_shader"), "needs torch 2.7"),
    (lambda m: m.setattr(_mps, "_morphology_works", lambda: False), "does not match"),
])
def test_unsupported_setup_runs_on_cpu(monkeypatch, patch, message):
    patch(monkeypatch)
    with pytest.warns(RuntimeWarning, match=message):
        assert NoiseChisel(device="mps").device.type == "cpu"


def test_fits_and_cli(tmp_path, capsys):
    fits = pytest.importorskip("astropy.io.fits")
    from noisechisel_torch import cli
    img = with_blanks(synthetic_image()[0])
    inp = tmp_path / "in.fits"
    fits.writeto(inp, img)
    for dev in ("cpu", "mps"):
        assert cli.main([str(inp), "-h0", "-o", str(tmp_path / f"{dev}.fits"), "--device", dev]) == 0
    for ext in ("INPUT-NO-SKY", "DETECTIONS", "SKY", "SKY_STD"):
        assert np.array_equal(fits.getdata(tmp_path / "mps.fits", ext),
                              fits.getdata(tmp_path / "cpu.fits", ext), equal_nan=True), ext
