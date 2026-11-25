# MPS backend plan

**Status:** implemented on `mps-backend`. See [Results](#results) at the end,
including where the implementation departs from this plan.

## Goal

`device="mps"` runs on the Apple GPU and returns **bit-identical** results to
`device="cpu"` (and therefore to Gnuastro). No new dependencies, and no change
to CPU or CUDA results.

## Measurements

Apple M3 Max, macOS 26.7.3, torch 2.15.0, synthetic 4088² frame
(`benchmarks/gpu_check.py`). Other macOS and torch versions are untested.

**What MPS can do** (small tensors, compared with CPU):

| Operation | Result |
|---|---|
| `int64` add, mul, shift, cumsum, gather, scatter, bincount | matches |
| float32 mul, divide, sqrt, sort, searchsorted | matches |
| TwoSum error-free addition | matches |
| float32 `cumsum` | differs (different summation order) |
| subnormal float32 arithmetic | differs (flushed to zero) |
| `torch.mps.compile_shader` | available |

**CPU time by stage** (2.37 s total, stages partly nested): convolution 0.85 s,
scipy labelling 0.50 s, sky and STD 0.22 s, label S/N 0.16 s, neighbour search
0.15 s, quantile thresholds 0.10 s.

**Convolution:**

* Plain float32 accumulation (what `_acc_dtype` selects for MPS today) differs
  from float64 in 83–84% of pixels.
* A Metal prototype that does float64 add and divide with integer arithmetic
  had **0 mismatches** on four 4088² images (sky ≫ noise, zero-mean noise,
  values spread over 16 decades, 2% NaN). It took 0.09–0.11 s against
  0.85–0.90 s on the CPU (1.6 s with NaN). The times include transfers but not
  shader compilation.

## Approach

Metal has no float64 type, so each float64 step gets one of two treatments:

1. **Emulate it in a Metal shader** if it is a large, hot loop. At first this
   means only the convolution.
2. **Run it on the CPU** with the existing code otherwise. The results are then
   the same by construction.

Everything that is only float32, bool or integer runs on MPS: tile gathering,
sorting, thresholding, morphology and mask operations.

### Alternatives not chosen

| Option | Reason |
|---|---|
| Plain float32 | Not identical (measured above) |
| Double-float (two float32) | About 48 bits instead of 53, so identical rounding is not guaranteed (not benchmarked) |
| `int64` fixed-point sums | Exact sums differ from float64 wherever float64 itself rounds. The float64 result depended on tap order in 3.5% of zero-mean noise pixels, so this happens |
| MLX or another framework | New dependency, and it does not remove the Metal float64 limit |

## Where float64 is used

| Code (`core.py`) | Placement |
|---|---|
| `_tap_sum`, `_edge_ksum`, `_convolve` on the image | Metal shader |
| `_convolve` on the tile grid (`_smooth_tiles`) | CPU |
| `_quantile_index` | CPU (one value per tile) |
| `_quantile_thresholds`: tile mean and `q` | Gather and sort on MPS, float64 part on CPU |
| `_sigma_clip_rows`, `_range_mean_std` | CPU, with rows gathered and sorted on MPS |
| `_outlier_threshold`, `_mad_clip_rows`, `_knn_values`, `_interp_smooth` | CPU (small tile-grid arrays) |
| `_sky_and_std`: tile arrays | CPU, then the final float32 images go back to MPS |
| `_label_sn` | Select labelled pixels on MPS, sum on CPU in the same order |
| `_as_kernel` | Already NumPy |

Moving a tensor to float64 must always be written as `.cpu().double()`, in that
order.

## Phases

Each phase ends with MPS output equal to CPU output.

### 0. Parity tests
* Add `mps` tests that mirror `test_cuda_matches_cpu`, skipped when MPS is
  unavailable. Compare detections, labels, sky, STD, tile arrays and the S/N
  threshold.
* Add per-stage comparisons for convolution, thresholds, sky and label S/N.
* Cover NaN and inf input, images smaller than the kernel, custom and
  asymmetric kernels, `widekernel`, non-square tiles, `cleangrowndet` and
  `detgrowquant=1`.
* Let `benchmarks/gpu_check.py` select `cuda` or `mps`.

### 1. Hybrid pipeline (no custom shaders)
* Add one internal "float64 device": CPU when running on MPS, otherwise the run
  device. CPU and CUDA then follow the same path as today.
* Route the operations in the table through it. The convolution stays on the CPU
  in this phase.
* Remove `_acc_dtype`'s float32 branch so no path can silently lose precision.
* Remove the forced CPU fallback in `NoiseChisel.__init__`.
* Label with scipy (copy to the host and back).
* Synchronise MPS in the verbose timings.

This phase establishes correctness. Transfers may make it no faster than the CPU.

### 2. Metal convolution
* Move the prototype to `noisechisel_torch/_mps.py`. Compile on first use and
  cache the result.
* Before production use, add handling and tests for what the prototype omits:
  * **Subnormals:** if `|x| × smallest nonzero kernel tap` can fall below the
    normal float32 range, convolve on the CPU. Test the input bits as int32 or
    on the CPU, since MPS float comparisons may flush small values to zero.
  * **Overflow:** the shader does not handle inf or double subnormals. Use the
    CPU if products or the result could leave the normal range.
  * **Zero denominator** (possible with kernels that have negative taps), signed
    zeros, and exact cancellation.
  * **Rounding:** unit-test add and divide against CPU float64 on random and
    tie cases.
  * **GPU watchdog:** test large images and kernels, and split the dispatch into
    chunks if needed.
* Check that the shader's float32 `x * k` matches the CPU under Metal's default
  compile settings. The prototype's zero mismatches support this on this machine
  only.

### 3. Profile, then decide
* Re-run the stage timings. Reduce host/device round trips first.
* Add another shader (sigma-clip moments or label sums) only if profiling shows
  it matters. These need float64 multiply and sqrt as well.
* Labelling on the GPU is a separate, optional task. It is probably the largest
  remaining cost, and it is unrelated to float64.

### 4. Release
* Merge only if MPS is clearly faster than the multi-threaded CPU on 4088²
  frames. Also check small images and record where the CPU wins.
* Update the module docstring, README, CLI `--device` help and benchmarks.
* Keep the default device unchanged (CUDA, else CPU), so MPS is opt-in.

## Compatibility

| Case | Behaviour |
|---|---|
| torch with `compile_shader` (believed 2.7 and later; verify) | Hybrid plus Metal convolution |
| Older torch (`>=2.0` stays supported) or shader compilation fails | Hybrid with CPU convolution, warning once |
| An MPS operator is missing on an older macOS or torch | Fall back to the whole CPU path, with a warning |
| `device="mps"` but MPS is unavailable | Same behaviour as today (CPU with a warning) |
| Linux, Windows, CUDA | No change |

The fallbacks must not require `PYTORCH_ENABLE_MPS_FALLBACK`.

One visible change: with `device="mps"`, result tensors will be on MPS instead
of the CPU. `.numpy()` and the FITS writer already call `.cpu()`.

## Risks

* **Testing in CI:** GitHub's hosted macOS runners may not provide a usable GPU.
  If they don't, MPS tests have to be run locally before each release.
* **Compiler changes:** a macOS update could change shader behaviour. A short
  self-test on first use (a small convolution compared with the CPU) can trigger
  the CPU fallback. It catches gross breakage, not every edge case.
* **Speed:** only the convolution speedup has been measured. No end-to-end MPS
  time exists yet.

## Results

Phases 0–3 are done, and the phase 4 documentation is updated. Merging is still
to be decided.

* **Identical output:** MPS matches the CPU in all pipeline test cases
  (`tests/test_mps.py`). CPU output matches `main` in 8 cases. CUDA was not
  re-tested, and neither was a direct comparison with Gnuastro (not installed).
* **Speed** (M3 Max, 10 CPU threads): 1.2 s against 2.0 s at 4088², and 5.5 s
  against 7.9 s at 8176². The table in the README has the smaller sizes.
* **Slow `bincount`:** the first hybrid version took about 100 s per 4088²
  frame. `torch.bincount` on MPS took about 60 s per call in `fill_holes`,
  where one label covers most pixels. Those counts now run on the CPU.
* **Departure from the compatibility table:** torch older than 2.7 now runs
  entirely on the CPU, with no hybrid mode. On torch 2.6.0, in-place `|=` on
  column slices of bool tensors gave wrong values without any error, which
  broke the morphology. A one-time morphology check also guards later versions.
* **No more shaders for now:** at 4088² the convolution takes about 0.09 s on
  MPS. By the CPU stage timings, scipy labelling (about 0.5 s) is probably the
  largest remaining cost. It was not timed separately on MPS.
* **CI:** not yet run. The MPS tests skip if a small MPS allocation fails, but
  how GitHub's macOS runners behave is unknown.
