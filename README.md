# sim-to-recon

Multi-view 3D reconstruction benchmark with stress-tested honest evaluation
under viewpoint perturbation, applied to COLMAP-based dense MVS on DTU.
This repo applies the evaluation discipline of
[sim-to-data](https://github.com/tyy0811/sim-to-data) — controlled stress,
honest failure reporting, explicit scope statements — to 3D reconstruction.

## Why This Matters

Multi-view 3D reconstruction pipelines are routinely benchmarked on canonical
datasets at full view density, then deployed in conditions where view coverage,
camera calibration, and image quality are all degraded. Standard benchmarks
report a single number per scene and call it a day. They do not tell you
*where* a pipeline fails when input conditions degrade, or *how fast* quality
collapses as the input gets sparser.

The goal here is not novelty in reconstruction methodology. The goal is to know
what the pipeline does when the inputs are not what the benchmark assumed.

### Summary of Findings

**COLMAP's default dense MVS produces a view-count degradation curve whose
within-N variance dominates the between-N trend below n=15. Roughly one-third
of runs at n=15 and n=8 produce degenerate reconstructions (<1,000 fused
points). Single-seed benchmarks — the standard practice — hide this completely.**

Across 3 independent reconstructions per view count on DTU scan9 at 800x600:

- **n=49:** all 3 seeds produce usable reconstructions (119k–258k fused points, chamfer 7.3–19.0mm)
- **n=30:** all 3 seeds non-degenerate but highly variable (34k–246k points, chamfer 17.4–51.4mm)
- **n=15:** 2 of 3 seeds produce near-total failures (579 and 6,698 points); only seed 7 gives a usable 89,832-point result
- **n=8:** 1 of 3 seeds produces a total failure (11 points); the other 2 give 41k–52k points

**Reporting the variance is the contribution.**

The debugging path — from an early monotonic-looking baseline through three
rejected alternatives (shared-SfM scaffold, fixed-pose re-triangulation,
single-seed with `set_random_seed`) to the current multi-seed report — is
documented in [DECISIONS.md](DECISIONS.md) entries 14–16. That path is the
judgment this repo is trying to show.

## Problem

Dense multi-view stereo (MVS) reconstructs 3D geometry from multiple
overlapping images. The quality of the reconstruction depends critically on
the number and distribution of input views — but standard benchmarks
evaluate only at full view density, hiding the degradation curve.

This benchmark stress-tests COLMAP's dense PatchMatch MVS on DTU scan9 by
systematically reducing the number of input views and measuring how
reconstruction quality collapses. Per-point failure analysis shows *where*
the pipeline loses geometry, not just *how much*.

## Approach

1. **Reconstruct:** COLMAP SfM + dense PatchMatch MVS
2. **Calibrate:** Standalone C++17/OpenCV-C++ binary for sensor onboarding
3. **Evaluate:** Chamfer distance, accuracy, completeness, F-score against DTU GT
4. **Stress:** View-count reduction with per-point failure analysis

## Results

### Reconstruction quality vs view count

All runs use COLMAP SfM + dense PatchMatch MVS on DTU scan9 at 800x600 with
default PatchMatch parameters. We run 3 independent reconstructions per view
count with seeds {42, 123, 7}. Metrics are computed after Open3D ICP alignment
to DTU GT (scale + rotation + translation).

**Example reconstruction** — best n=49 run (seed 123, 258k points, chamfer 7.35mm)
vs DTU laser-scan ground truth, rendered in 3D from the same viewpoint:

![Reconstruction 3D](docs/figures/reconstruction_3d.png)

*Point sizes are balanced to equalize visual density: GT is ~13x denser in
points per unit volume than the prediction, so rendering both at the same point
size would oversaturate GT and hide the shape comparison.*

The reconstructed scene (a house facade) is clearly recognizable but noticeably
sparser than the laser scan, with gaps around edges and occluded regions — the
characteristic signature of default PatchMatch at 800x600 resolution.

![Variance scatter](docs/figures/variance_scatter.png)

*12 reconstructions — 3 seeds × 4 view counts. The within-N spread at n=15 and
n=8 is larger than the between-N trend — the finding of this benchmark. Log
scale on points so the 11-point degenerate run at n=8 seed 123 remains visible.*

**What the variance looks like geometrically.** Three reconstructions of the
same scene. Left is the best n=49 run. Middle is n=8 seed 42 — the pipeline
with only 8 views, but the SfM initialization happened to succeed. Right is
n=8 seed 123 — **same view count, same pipeline, different seed**, but the
SfM initialization produced a degenerate 11-point reconstruction. This is the
failure mode the variance scatter above is quantifying.

![Reconstruction contrast](docs/figures/reconstruction_contrast.png)

*Uniform point size across all three panels — the visible density differences
reflect real reconstruction density, not a rendering artifact. n=49 is genuinely
~6x denser than the working n=8 case, and the failed case really does contain
only 11 points.*

| N | Registered | Median points | Range (points) | Median chamfer (mm) | Median F@5mm | Median F@10mm |
|---|-----------|---------------|----------------|---------------------|--------------|---------------|
| 49 | 37–38 | 156,711 | 118,793 – 257,684 | 18.31 | 0.459 | 0.639 |
| 30 | 23 | 166,169 | 34,035 – 245,557 | 28.65 | 0.351 | 0.490 |
| 15 | 10–11 | 6,698 | **579 – 89,832** | 37.65 | 0.061 | 0.154 |
| 8 | 3–7 | 41,656 | **11 – 51,815** | 19.53 | 0.397 | 0.585 |

**Degenerate runs** (<1,000 fused points, bolded ranges above):
- n=15 seed=42 → 579 points
- n=8 seed=123 → 11 points

> **Note on the n=8 medians.** The table shows n=8 median chamfer (19.53mm)
> as apparently *better* than n=15 (37.65mm) and comparable to n=49. This is
> an artifact: the 11-point degenerate run at n=8 seed 123 is so small that
> ICP alignment collapses onto itself and produces a spurious small chamfer.
> The same caveat applies to n=8 median points: 41,656 is the middle of three
> very different outcomes (11 / 41,656 / 51,815), not a typical expectation.
> F@5mm is the most informative column at low N (n=15: 0.061, n=8: 0.397) and
> captures the bimodal effect in reverse. In this regime, read the scatter,
> not the medians.

**The failure originates in SfM, not PatchMatch.** At n=8, seed 7 registered 7
of 8 images (yielding 52k points); seeds 42 and 123 registered only 3 of 8. The
3-image cases produced one usable reconstruction (seed 42 got lucky with dense
triangulation on a narrow region) and one total failure (seed 123 with 11
points). SfM initialization — which image pair seeds incremental mapping — is
where the bimodal outcome is decided.

**Even at n=49, only 37–38 of 49 images register.** COLMAP fails to include
~22% of the input views in the sparse reconstruction even with the full view
set. This is expected behavior for DTU scan9 at 800x600 with default
parameters (some viewpoints have too little feature overlap with neighbors at
this resolution to pass SfM's registration thresholds) and is a stronger
version of the main finding: the pipeline's unreliability isn't confined to
low view counts — it's just that at low N, the consequences are catastrophic
instead of marginal.

![Failure regions](docs/figures/failure_gallery.png)

*Per-point error heatmaps for the best seed at each view count, in three
orthographic projections. Red regions are where the reconstruction deviates
most from GT; colorbar scales vary by row because absolute errors span an
order of magnitude between n=49 and n=15.*

### Why we report variance, not a clean curve

COLMAP's dense MVS pipeline on GPU is non-deterministic. Two sources of
randomness stack:

1. **Incremental SfM** uses RANSAC internally. `pycolmap.set_random_seed` controls
   the pseudorandom state but does not eliminate run-to-run variance from
   floating-point bundle adjustment.
2. **GPU PatchMatch** is non-deterministic at the hardware level due to CUDA
   thread scheduling — verified: two runs with identical seeds produced 24,322
   and 178 fused points at n=30.

At n=49 and n=30 the seed-to-seed variance is large but results are uniformly
non-degenerate. At n=15 and n=8, seeds produce a **bimodal outcome**:
successful reconstruction (tens of thousands of fused points) or near-total
failure (<1,000 points). The median understates this because the distribution
is not Gaussian — the full range is the finding.

A standard benchmark would pick the best seed per N and report a clean
degradation curve. We report all seeds and the within-N variance explicitly.
Practitioners running COLMAP below 15 views on similar scenes should expect
roughly one-third of runs to fail and should plan for re-runs or fallback
pipelines.

### The SOTA gap

Published COLMAP on DTU reports 0.5–0.9mm chamfer at native 1600x1200 with
dataset-tuned PatchMatch parameters and the official mask-aware Matlab
evaluator. Our 7–19mm range at 800x600 with defaults and ICP evaluation is
the expected order-of-magnitude offset from those three choices — not the
contribution under investigation. Background contamination was tested and
ruled out (Decision 14): only 1.1% of reconstructed points lie outside the
object region, moving chamfer by 1.2mm.

## C++ Calibration Module

Standalone C++17 camera calibration binary using OpenCV:

```bash
./build/cpp/calib/calib \
    --images path/to/chessboard/images \
    --pattern 9x6 \
    --square 25.0 \
    --output calib.json
```

8 GoogleTest unit tests cover: object point geometry, corner detection,
calibration accuracy on synthetic data, edge cases, JSON serialization,
and intrinsics validity.

## Honest Scope

- Single dataset, single scene (DTU scan9 — one house-model scan)
- Single reconstruction method (COLMAP SfM + dense PatchMatch) with default parameters, no dataset-specific tuning
- 800x600 resolution on Modal A10G (not DTU's native 1600x1200)
- 3 seeds per view count, median ± range reporting; GPU PatchMatch non-determinism quantified rather than hidden
- Open3D ICP alignment for evaluation, not DTU's official mask-aware Matlab evaluator — accounts for most of the absolute-number gap vs published SOTA

## Methodological Lineage

This benchmark is the geometric sibling of
[sim-to-data](https://github.com/tyy0811/sim-to-data), which applies the
same evaluation philosophy to defect detection under sensor shift.

## Engineering

- **Tests**: 30 pytest (metrics, alignment, DTU loader, COLMAP runner, sweep schemas, failure regions) + 8 GoogleTest (calibration accuracy, corner detection, JSON serialization, edge cases)
- **CI**: ruff lint, pytest, CMake build + ctest on each push
- **Modal infrastructure**: deployed as `simtorecon-mvs` with persistent volumes (`simtorecon-dtu-data` caches scan9; `simtorecon-workspace` holds per-run artifacts). Feature extraction + SfM + PatchMatch + fusion run on A10G
- **Sweep is resumable**: each (view_count, seed) run writes its own JSON cache under `results/stress_view_count/views_NNN/seed_SS/result.json`. If a run fails or is interrupted, re-running the sweep script skips completed entries. Critical for multi-seed experiments where individual GPU runs can fail non-deterministically
- **Per-run artifacts**: fused PLY + per-point error `.npz` + ICP transform saved for every reconstruction. V2 can recompute metrics or failure visualizations without re-running PatchMatch
- **Reproducibility**: SfM seeded via `pycolmap.set_random_seed`; GPU PatchMatch non-determinism documented and quantified rather than hidden

## Quick Start

```bash
git clone https://github.com/tyy0811/sim-to-recon
cd sim-to-recon
conda env create -f environment.yml
conda activate simtorecon
pip install -e ".[dev]"

make deploy     # one-time: deploys sfm/dense/download functions to Modal
make data       # one-time: downloads DTU scan9 to Modal volume (~250MB)
make build      # builds the C++ calibration binary
make baseline   # runs a single n=49 reconstruction on Modal GPU (~10 min)
make stress     # runs the 3-seed view-count sweep on Modal GPU (~90 min, 12 runs)
make figures    # generates variance scatter + failure gallery + contrast figure
make test       # runs pytest (30 tests) + ctest (8 GoogleTest cases)
```

**Requires** a Modal account (`pip install modal && modal setup`) and ~$1 of
GPU credit for the full sweep.

## Decisions

See [DECISIONS.md](DECISIONS.md) for documented architectural decisions. Entries
14–16 trace the debugging path from an early single-seed baseline to the
current multi-seed variance report — including three rejected alternatives and
why each was wrong.

## What I'd do next (V1.5 roadmap)

- **Multiple DTU scenes.** The unreliability floor between n=15 and n=8 may be
  scan9-specific (house model, medium-frequency architectural texture). Running
  the same sweep on DTU scans with different texture and geometry characteristics
  — smooth objects, high-frequency organic shapes, reflective surfaces — would
  tell us whether the floor is a property of the pipeline or of this scene.
- **PatchMatch parameter sweep.** Does tuning `min_num_pixels`, `window_radius`,
  or `filter_min_ncc` move the floor? A cheap way to find out whether the bimodal
  failure is fundamental to default PatchMatch or an artifact of the defaults.
- **3D Gaussian Splatting comparison.** 3DGS has different failure modes
  (scale/density vs geometric accuracy). Same stress sweep on the same scene
  with 3DGS would give a cross-method variance profile — does 3DGS also show
  an unreliability floor, or does it fail differently?
- **Conformal calibration of reconstruction quality.** Given the variance, a
  calibrated "at 95% confidence, n=X views will produce ≥Y fused points" bound
  would be more useful to practitioners than a point estimate. This is the
  direct analogue of sim-to-data's conformal thresholds, applied to geometry.
