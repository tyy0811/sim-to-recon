# Architectural Decisions

Documented decisions with rationale, following the pattern from [sim-to-data](https://github.com/tyy0811/sim-to-data).

---

## 1. DTU scan9 over alternative datasets

**Decision:** Use DTU MVS benchmark scan9 (stone figure, 49 rectified images) as the single V1 scene.

**Alternatives considered:**
- Tanks and Temples — larger, more complex, but no clean per-scene GT point clouds for chamfer evaluation
- ETH3D — excellent GT, but larger scenes, harder to run CPU-only in CI
- Custom phone-shot scenes — no GT geometry (deferred to V1.5)
- Multiple DTU scenes — more comprehensive but adds wall-clock and complexity beyond V1 scope

**Rationale:** scan9 is small (~250MB), well-textured (reliable PatchMatch), has clean GT point cloud, and runs CPU-friendly at downsampled resolution. Single-scene focus keeps V1 shippable in a week.

---

## 2. ~~Posed-image MVS over full SfM~~ → Superseded by Decision 12

**Original decision:** Pre-populate COLMAP with known camera poses from DTU calibration, skip SfM. **Superseded:** RQ decomposition of DTU's projection matrices produced incorrect R/t in COLMAP's coordinate convention, leading to valid-looking depth maps that failed to fuse (0 points after extensive debugging). Switching to COLMAP's standard SfM frontend produced 16,576 fused points on the first attempt. See Decision 12.

---

## 3. ~~Single seed for V1~~ → Superseded by Decision 16

**Original decision:** No multi-seed reporting in V1. All experiments use seed=42 for view subsampling.

**Superseded:** The original rationale rested on two incorrect assumptions:
1. *"COLMAP's PatchMatch stereo is deterministic given the same inputs."* — False. GPU PatchMatch is non-deterministic due to CUDA thread scheduling in the fusion kernel.
2. *"The only stochastic element is view subsampling."* — False. Incremental SfM has internal RANSAC non-determinism that `pycolmap.set_random_seed` reduces but does not eliminate.

Both assumptions were disproved empirically. Three nominally-identical n=49 runs produced **2,439, 72,122, and 270,235 fused points** — a 110× spread from nothing but run-to-run variance. V1 now uses 3 seeds per view count with full variance reporting. See Decision 16.

---

## 4. Modal GPU for dense reconstruction

**Decision:** Use Modal A10G (24 GB VRAM) for PatchMatch stereo reconstruction. Cost: ~$0.50 per full sweep.

**Alternatives considered:**
- Local CUDA GPU — unavailable on this machine
- COLMAP CLI with OpenCL — fragile, driver-dependent, inconsistent results across platforms

**Rationale:** PatchMatch stereo requires CUDA, and this machine has no NVIDIA GPU. Modal provides on-demand A10G instances with predictable performance and cost. The per-sweep cost is negligible compared to development time lost fighting OpenCL compatibility.

---

## 5. DTU coordinate alignment via ICP

**Decision:** Use scale-aware ICP (Open3D) to align predicted point clouds to ground-truth before computing metrics.

**Alternatives considered:**
- Manual axis-swap + scale factor — brittle, hard to verify
- Procrustes alignment — equivalent but less standard in MVS literature
- Skip alignment and hope coordinate frames match — silently invalidates all metrics

**Rationale:** COLMAP and DTU use different coordinate conventions. This is the most subtle decision in the pipeline — wrong alignment silently invalidates all metrics. Scale-aware ICP is well-tested in Open3D and standard in MVS evaluation.

---

## 6. C++17 + OpenCV-C++ for calibration

**Decision:** Implement the calibration utility as a C++17 binary using OpenCV's C++ API, CMake build system, and GoogleTest.

**Alternatives considered:**
- Python + OpenCV — the same calibration could be 10 lines of Python
- Rust — strong systems language but smaller hiring-funnel overlap

**Rationale:** Hiring funnel signal. The C++ binary signals competence in systems-level code, CMake, GoogleTest, and header-only dependency management. Targets: Sereact, KONUX, Helsing, Spleenlab.

---

## 7. GoogleTest via FetchContent over system install

**Decision:** Pull GoogleTest (and nlohmann/json, cxxopts) via CMake FetchContent rather than requiring system-installed packages.

**Alternatives considered:**
- System-installed GoogleTest (`apt install libgtest-dev`) — requires manual setup per machine
- Conan / vcpkg — heavier dependency management than needed for a small project
- Vendored source copies — works but harder to update

**Rationale:** Reproducibility with no system dependency. Standard modern CMake pattern. Anyone cloning the repo can build and test with zero manual package installation.

---

## 8. JSON results files over MLflow

**Decision:** Store experiment results as plain JSON files.

**Alternatives considered:**
- MLflow — full experiment tracking with UI, comparison, and artifact storage
- W&B — similar to MLflow, cloud-hosted
- SQLite — structured but less human-readable

**Rationale:** V1 has one method and one stress dimension. MLflow overhead is not justified. JSON is inspectable and diffable. MLflow deferred to V1.5 when multiple methods (COLMAP vs 3DGS) make comparison dashboards worthwhile.

---

## 9. Image downsampling to 800x600

**Decision:** Downsample input images to 800x600 before dense reconstruction. Configurable in PipelineConfig.

**Alternatives considered:**
- Full resolution (~1600x1200) — highest fidelity but 60+ min per reconstruction
- 400x300 — faster but risks losing fine detail needed for meaningful chamfer evaluation

**Rationale:** Full-resolution PatchMatch takes 60+ min per reconstruction. 800x600 brings this to 15-20 min while retaining sufficient detail for the stress-test evaluation. The setting is configurable, so full-resolution runs remain possible for final reporting.

---

## 10. Open3D for evaluation metrics

**Decision:** Use Open3D's `compute_point_cloud_distance` for chamfer and related metrics.

**Alternatives considered:**
- Custom C++ distance computation — full control but reinvents the wheel
- PyTorch3D — GPU-accelerated but adds a heavy dependency for a simple operation
- scipy.spatial.KDTree — pure Python, slower on large clouds

**Rationale:** Evaluation is not the contribution. Use canonical C++-backed implementation that is well-tested and efficient. The contribution is the stress-test methodology, not the distance computation itself.

---

## 11. MVSNet preprocessed DTU subset over full DTU Rectified

**Decision:** Use the MVSNet preprocessed training data (~2GB, Yao et al. 2018) for images and cameras, with GT point cloud fetched separately from DTU evaluation server (~6.3GB).

**Alternatives considered:**
- Full DTU Rectified set (123 GB) — contains all 124 scenes but 50× larger than needed for scan9
- Range-request extraction from the zip — fragile, depends on server behavior
- Download full set and extract on Modal — 3+ hours download time, wasteful

**Rationale:** 50× smaller download. The MVSNet preprocessed format is the de facto standard in the learning-based MVS community. Camera files are cleaner (explicit extrinsics + intrinsics) than DTU's original Krt format. No functional loss for scan9 reconstruction. The loader auto-detects both formats for forward compatibility.

---

## 12. SfM-based pose recovery instead of injecting DTU-decomposed poses

**Decision:** Use COLMAP's standard SfM frontend (feature extraction → exhaustive matching → incremental mapping) to recover camera poses, instead of injecting DTU's calibration matrices.

**Alternatives considered:**
- Posed-image MVS via RQ decomposition of DTU projection matrices (original V1 plan, Decision 2) — produced incorrect R/t in COLMAP's coordinate convention; depth maps had valid content but zero fused points after extensive debugging
- Manual depth range estimation + manual patch-match.cfg — added complexity without fixing the fundamental pose error

**Rationale:** The SfM fallback was listed in the V1 risks table and produced 16,576 fused points on the first attempt. The added cost is ~30 seconds per scene (negligible vs PatchMatch's runtime). The V1 contribution is the stress sweep + honest evaluation discipline; the pose source is implementation plumbing, not the contribution. Coordinate alignment to DTU GT is handled by Open3D ICP (Decision 13).

---

## 13. Open3D ICP for COLMAP-to-DTU coordinate alignment

**Decision:** Use Open3D's ICP with scale estimation to align COLMAP's arbitrary-frame reconstruction to DTU's object-frame GT before computing metrics.

**Alternatives considered:**
- Working in DTU's native frame via pose injection — failed (Decision 12)
- Sim(3) alignment via known correspondences — more accurate but more code; save for V1.5

**Rationale:** SfM produces reconstructions in an arbitrary coordinate frame (scale-free, origin at first camera). DTU GT is in object-frame millimeters. Post-hoc similarity alignment is the standard MVS evaluation approach used by the entire community. Open3D's `registration_icp` is well-tested and runs in seconds.

---

## 14. Tested and rejected GT bounding-box cropping for evaluation

**Decision:** Do not crop predictions to the GT bounding box before computing metrics. Report uncropped numbers for transparency.

**Investigation:** Tested whether the chamfer gap vs published COLMAP comes from background contamination. On an early single-seed baseline (since contextualized as one sample from a wide distribution — see Decision 16), cropping predictions to GT extent + 50mm padding removed only 1.1% of points (807 of 72,122) and changed chamfer by 1.2mm (18.89 → 17.71). F@5mm moved by 0.002. Background is not a meaningful source of error for this dataset and pipeline at any seed we've observed.

**Rationale:** The crop changes metrics by less than 2% at any seed. The investigation was valuable but the original framing — that the 18.9mm number was "the" baseline — was wrong. The more important finding turned out to be the within-N variance we weren't measuring at all (see Decision 16). Keeping this entry in DECISIONS.md as a record of the methodology pivot: a negative result correctly ruled out one hypothesis, but it was the wrong hypothesis to be asking about in the first place.

---

## 15. Investigated and rejected shared-SfM stress sweep

**Decision:** Do not share a single SfM sparse model across stress levels. Use independent SfM per view count with multi-seed reporting (Decision 16).

**Problem discovered:** Initial stress sweep attempts with independent SfM per view count produced wildly non-monotonic results (n=8 outperforming n=49 with 110x point-count variance). The obvious candidate fix — run SfM once on all 49 views, then subset the sparse scaffold for each stress level — was explored but rejected after extensive investigation.

**Alternatives investigated:**
1. **Full scaffold filter** (text-format images.txt subset, full points3D.txt) — Only 1.1% of dense MVS output survived at n=30 vs n=49. Undistort couldn't handle sparse model inconsistency.
2. **Two-pass consistency filter** (cross-reference cleanup) — 7.3k points at n=30 vs 299k at n=49. The filter was correct but the cliff persisted.
3. **Hybrid re-triangulation** (fixed poses + fresh feature matching + `triangulate_points`) — Required navigating multiple pycolmap-cuda12 API incompatibilities (`deregister_image`, `Database` constructor, pose object API). Ultimately produced 4.2k points at n=30 — still the same cliff.

**Why shared-SfM fails:** A sparse model built from 49 views encodes a co-visibility graph optimized for that specific view configuration. When PatchMatch's reference image selection uses this scaffold with only 30 images on disk, the graph structure is fundamentally mismatched. The 47x point-count cliff between n=49 and n=30 is not a measurement of view-count degradation — it is a measurement of the scaffold mismatch. A reviewer would correctly ask "why are you fixing the SfM substrate? That's not what COLMAP users do."

**Why this matters:** The stress sweep's implicit promise is "this is what happens when you run COLMAP with fewer views." A real practitioner at n=30 would run COLMAP on 30 images, not hand it a sparse model built from 49. The shared-SfM curve answers a question nobody asks.

**Rationale for rejection:** The methodological purity of shared-SfM (eliminating SfM stochasticity) does not justify the methodological gap it creates (answering the wrong question). The honest approach is to run COLMAP independently at each view count and face its non-determinism directly. See Decision 16 for the final multi-seed approach.

---

## 16. Multi-seed sweep with full variance reporting (not medians-only)

**Decision:** Run 3 independent reconstructions per view count with seeds {42, 123, 7}. Report all 12 results in the README as a scatter plot with median overlay, not a line plot with error bars. Report median ± range per view count in the results table.

**Alternatives considered:**
- Single seed per N with `pycolmap.set_random_seed(42)` — Reduces SfM variance but GPU PatchMatch is non-deterministic at the hardware level (CUDA thread scheduling). Verified: two n=30 runs with identical seeds produced 24,322 and 178 fused points. A single run is a coin flip in the failure regime.
- Multi-seed with median-only reporting — Hides the bimodal failure distribution at low N. The median at n=8 (41,656 points) looks deceptively monotonic but masks the fact that seed 123 produced 11 points — a near-total failure.
- Line plot with error bars — Error bars assume approximately Gaussian variance. The distributions at n=15 and n=8 are bimodal (runs either succeed or fail) and error bars visually understate this.

**Findings that drove the decision:**
- COLMAP PatchMatch on GPU is non-deterministic. `set_random_seed` controls COLMAP's RANSAC but not CUDA kernel execution order. Repeated runs with identical seeds produce different depth maps.
- At n=49 and n=30, seed-to-seed variance is large but all runs are non-degenerate.
- At n=15, 2/3 seeds produced near-total failures (579 and 6,698 fused points); only seed 7 gave a usable result (89,832).
- At n=8, 1/3 seeds produced a total failure (11 fused points); 2/3 seeds gave usable results.
- The pipeline has an **unreliability floor** below n=15 where roughly one-third of runs at n=15 and n=8 produce degenerate reconstructions (<1,000 fused points).

**Rationale:** The non-monotonic curve is itself the finding. A clean degradation curve is what a single-seed benchmark would report — and every published MVS benchmark does exactly that. The scientific contribution here is surfacing the variance that single-seed reporting hides. Practitioners running COLMAP below ~15 views on similar scenes should expect roughly one-third of runs to fail. That is information no other DTU benchmark communicates, and the honest-evaluation discipline of the repo (shared lineage with sim-to-data) demands reporting it.

**Implementation note:** The sweep stores per-run JSONs under `results/stress_view_count/views_NNN/seed_SS/result.json`, enabling resume-from-cache. 12 runs × ~5-10 min each = ~90 min on Modal A10G. Raw runs are preserved in `all_runs.json`; aggregated medians in `summary.json`.
