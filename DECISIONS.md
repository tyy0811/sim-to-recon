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

---

## 17. gsplat (nerfstudio) over original 3DGS for V1.5 neural extension

**Decision:** Use `gsplat >= 1.3` (nerfstudio-project/gsplat) as the 3D Gaussian Splatting implementation for V1.5's neural comparison, initialized from V1's best COLMAP baseline (run_id `scan9_v49_s123_3d428b`, n=49, 258k fused points). Training runs on Modal A10G using a new `gsplat_image` built on `nvidia/cuda:12.4.1-devel` with torch 2.4 (cu124) and pycolmap-cuda12.

**Alternatives considered:**
- Original 3DGS from graphdeco-inria — reference implementation but has a fragile custom CUDA build and no pip wheels; adds a submodule dependency.
- Nerfstudio full framework — heavyweight, opinionated, and would require re-parsing our COLMAP output through Nerfstudio's dataset abstractions. Too much scaffolding for a single scene.
- Writing our own Gaussian rasterizer — V1.5 contribution is the cross-method comparison, not the rasterizer; reinventing is counter to the scope discipline.

**Rationale:** gsplat has prebuilt CUDA wheels, a clean standalone Python API (`gsplat.rendering.rasterization`, `gsplat.strategy.DefaultStrategy`), and is the same codebase most active 3DGS research now builds on. It accepts pycolmap-format sparse models natively, which lets us initialize from the exact sparse reconstruction V1 shipped without a conversion step. The spec's "default gsplat recipe" is a direct match to `DefaultStrategy`.

**Scope notes for V1.5:**
- Day 8 uses L1-only loss (no SSIM loss term) as a smoke-run simplification. If Day 8 PSNR lands below ~22 dB, the L1+SSIM mix from gsplat's `simple_trainer.py` becomes Day 9's first thing to add. The spec's target is mid-20s to low-30s dB; L1-only on DTU should get there.
- View-independent colors via SH degree-3 with zero-initialized higher-order terms. DTU scenes are nearly Lambertian so the SH ramp-up matters less than for glossy scenes.
- Day 8 is single-seed. Day 9 adds 3-seed multi-run (seeds {42, 123, 7}) with median+range reporting to match V1's variance discipline (Decision 16). The gsplat side of multi-seed measures training-loop stochasticity only — the upstream COLMAP sparse model stays fixed at scan9_v49_s123_3d428b, so the 3DGS variance column is NOT directly comparable to V1's COLMAP variance column. This distinction is called out in the V1.5 README section and must be maintained.

---

## 18. Held-out view split for novel-view evaluation: every 10th image

**Decision:** Hold out every 10th image (sorted by filename) from gsplat training, yielding 5 test views / 44 train views from scan9's 49 rectified images. Applied deterministically via `test_every=10` in `train_gsplat`.

**Alternatives considered:**
- Random fixed seed split — equivalent in expectation but harder to inspect and reproduce across environments.
- First N or last N as test — biased toward one end of the scanning trajectory.
- Every 8th image (standard LLFF convention) — yields 6 test / 43 train; almost identical but 5 test views lines up cleanly with V1.5's spec wording ("5 held-out novel views").

**Rationale:** Every-k-th is the canonical 3DGS novel-view evaluation protocol and directly reproducible: anyone cloning the repo and running the same Modal function gets the same split. Document the split explicitly so V1.5's cross-method table can reference it by name ("PSNR on 5 held-out DTU views, every-10th split").

**Cross-method implications:** the spec's Day 12 cross-method comparison will need to render COLMAP's fused point cloud at these same 5 held-out cameras and compute PSNR against the same ground-truth frames. That's the shared axis that makes the comparison honest — without it, the table is two unrelated scorecards side by side. COLMAP novel-view rendering from a sparse point cloud will be lower quality than 3DGS by construction, and that *is* the finding: classical dense MVS is a worse novel-view synthesizer than neural splatting even when it produces better geometry.

---

## 19. gsplat DefaultStrategy packed=True flag — V1.5 Day 9 densification fix

**Decision:** Pass `packed=True` to `strategy.step_post_backward(...)` in `train_gsplat` (modal_app.py) to match gsplat's rasterization which returns info dicts in packed format (`info["means2d"].shape == (nnz, 2)`, not `(C, N, 2)`).

**Symptom before the fix:** Day 8's 7000-iter run and every Day 9 smoke through mid-afternoon had N stuck at 9044 (the initial sparse point count) for the entire training run. Loss plateaued around 0.03 as the fixed-capacity model ran out of expressivity. Day 8 PSNR 22.62 dB was below the spec's "mid-20s to low-30s" target band. The hypothesis going into Day 9 was "absgrad flag not passed to DefaultStrategy"; that fix plus adding the L1+0.2·SSIM loss did not restore densification, which is what eventually pointed at the real bug.

**Root cause:** `gsplat.strategy.DefaultStrategy._update_state` has two code paths — a packed branch that reads `info["gaussian_ids"]` directly, and an unpacked branch that does `sel = (info["radii"] > 0.0).all(dim=-1)` followed by `gs_ids = torch.where(sel)[1]`. The unpacked branch expects `info["radii"]` shape `[C, N, 2]`. gsplat's rasterization actually returns packed info where `radii` is `(nnz, 2)` — 2D. `.all(dim=-1)` on a 2D tensor gives a 1D boolean, `torch.where` returns a 1-tuple, `[1]` raises `IndexError: tuple index out of range`. The exception was silently swallowed by a fragile `try/except` in `train_gsplat` (see methodology note below), so `strategy_ok` went `False` at step 0 and densification never fired for any subsequent step in any run.

**Verification:** After flipping `packed=False` → `packed=True` and rerunning the 700-iter smoke, `_grow_gs` fires at step 600 as designed. A step-600 diagnostic print shows 7446/9044 Gaussians above the grow threshold; gsplat's verbose output reports `342 GSs duplicated, 7105 GSs split. Now having 16491 GSs` followed by `267 GSs pruned. Now having 16224 GSs`. The arithmetic closes (342 + 7105 grown ≈ 7446 above-threshold ± 1 boundary case; 9044 + 7447 − 267 = 16224). PSNR at 700 iters moves from 16.38 dB (broken strategy) to 18.66 dB (+2.28 dB from densification alone), SSIM from 0.740 → 0.765, LPIPS from 0.308 → 0.251. The `rect_028` outlier moves from 15.86 → 16.98 dB, suggesting the added capacity resolves difficulty at the hardest held-out view.

**Alternatives considered:**
- Pass `packed=True` to `rasterization(...)` explicitly and keep the strategy flag aligned — redundant because gsplat's rasterization already returns packed info without being asked. The explicit argument would muddy the call site without changing behavior.
- Rewrite `_update_state` to auto-detect packed vs unpacked — out of scope. gsplat's own API is the source of truth; upstream may canonicalize in a future release.
- Lower `grow_grad2d` to work around the dead strategy — would have been a dangerous non-fix: densification never fires either way, and the threshold is not the bug. Rejected once the IndexError surfaced.

**Rationale:** The correct fix is one line. Detection of packed vs unpacked is empirical: print `info["means2d"].shape` at step 0 and check whether the first dimension is `nnz` (visible-Gaussians-across-cameras, less than total N and not equal to C) or `C` (unpacked). If packed, set the strategy flag accordingly.

**Methodology note — the except-guard anti-pattern that hid this for weeks:** the original `train_gsplat` wrapped `strategy.step_pre_backward` and `strategy.step_post_backward` in `try/except` with narrow print guards (`if step == 0:` and `if step == densify_start_iter:`). When the strategy crashed at step 0 with `step == 500` being False, the print was suppressed, `strategy_ok` went `False`, and at step 500 the outer `if strategy_ok:` skipped the try block entirely — meaning the print-condition step was never re-reached. The exception never surfaced in any log, on any run. Day 8 shipped and Day 9's first several hours of investigation proceeded as if the strategy were silently doing nothing, when in fact it had never run past step 0. Fixed by making both except blocks print unconditionally on first failure with step number and exception type name. General lesson (which should apply to any training loop in this codebase and any future codebase I touch): error-path visibility should never be gated on step conditions. Use the disable-flag pattern to prevent repeat prints — never a step filter. Cost of this anti-pattern on V1.5: approximately 4–6 hours of Day 9 investigating hypotheses downstream of a crash I could have seen in one log line.

---

## 20. Random-stream seeding posture for V1.5 gsplat training

**Decision:** Before entering the training loop, `train_gsplat` explicitly seeds torch (CPU + CUDA), numpy (legacy global + default_rng), and sets cudnn to deterministic mode. The intent is declared here: V1.5's multi-seed variance across seeds {42, 123, 7} measures gsplat's own sampling randomness (for split/duplicate child positions, drawn from torch's seeded RNG) plus residual CUDA rasterizer scheduling non-determinism from atomic adds and scatter kernels. This variance source is methodologically comparable to — but not identical to — V1's COLMAP CUDA-scheduling variance (Decision 16), and must be labeled accordingly in any cross-method table that places V1 and V1.5 columns side by side.

**Seeded components (explicit code in train_gsplat):**
- `torch.manual_seed(seed)` — PyTorch CPU RNG and, since torch 1.4, the default CUDA RNG for all devices.
- `torch.cuda.manual_seed_all(seed)` — explicit (redundant) seeding of all CUDA devices. Kept for clarity and as defense-in-depth against a future torch version drift.
- `np.random.seed(seed)` — legacy NumPy global RNG.
- `rng = np.random.default_rng(seed)` (later in the loop) — generator for the per-step image-index sampler. Separate from the legacy global state.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` — forces cudnn-backed ops (including torchmetrics SSIM's conv2d with a Gaussian window) to deterministic kernels.

**NOT seeded / left non-deterministic on purpose:**
- `torch.use_deterministic_algorithms(True)` — NOT set. gsplat's CUDA rasterizer backward uses atomic adds and scatter operations that do not have deterministic alternatives in PyTorch. Forcing determinism via this flag would raise `RuntimeError` at the first backward pass. Leaving it off means the backward scatter path has residual non-determinism — which is exactly the variance source the multi-seed experiment is meant to surface.
- `CUBLAS_WORKSPACE_CONFIG` env var — NOT set. Irrelevant for gsplat's custom CUDA kernels; would only affect cuBLAS matmuls, which cudnn.deterministic already handles indirectly for our call path.

**What multi-seed variance actually measures (pre-committed interpretation):** three seeds produce three different reconstructions because (a) the seed-dependent split/duplicate sampling paths diverge immediately at the first densification event (step 600), causing downstream capacity-growth trajectories to diverge, and (b) atomic-add ordering in the rasterizer backward pass is non-deterministic even at fixed seed, introducing a second, seed-independent variance component. The reported "V1.5 gsplat variance" column will be a weighted sum of both — primarily training-loop stochasticity from (a), with a floor from (b).

**Why this matters to commit before launching the single-seed run:** V1 Decision 3 was superseded by Decision 16 precisely because the variance posture was not declared up front — three runs were observed to produce 110× different fused point counts, and only after the fact was it rationalized as "SfM + PatchMatch non-determinism under identical seeds." Writing the posture into DECISIONS before the single-seed run means the multi-seed numbers that come back are interpretable in pre-committed terms, not retrofit to whatever the observed spread turns out to be.

**Cross-method comparison to V1's variance column:** V1's variance (Decision 16) came from COLMAP's CUDA PatchMatch fusion non-determinism plus SfM RANSAC non-determinism, measured as the spread across 3 seeds × multiple view counts. V1.5's variance comes from a different combination (gsplat sampling for split positions + rasterizer atomic-add ordering, measured as the spread across 3 seeds at a single fixed view count with the COLMAP sparse init held constant). Both are "run-to-run variance under nominally-identical inputs" but the causal mechanisms differ, and the magnitudes will almost certainly differ by an order of magnitude or more. V1.5's README cross-method table must label each column's variance source explicitly and warn against interpreting them as equivalent noise floors. See also the persistent memory entry `feedback_variance_labeling.md` for the general rule.

---

## 21. gsplat densification net-hurts PSNR on sparse-init / low-view-count scenes — Day 9 four-row ablation

**Decision:** V1.5 ships the four-row densification ablation as the Day 9 finding. Working densification under gsplat's standard recipe is net-negative for PSNR on DTU scan9 across four tested recipe variants. The frozen baseline (Day 8's silently-broken strategy, which left N at the initialization count of 9,044) outperforms every densified configuration on PSNR.

| Config | Densification | Loss | Final N | PSNR median | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| Frozen baseline (Day 8) | OFF (silent strategy bug) | L1 only | 9,044 | **22.62** | 0.816 | 0.187 |
| Over-densified | ON, grow_grad2d=2e-4 | L1 + 0.2·(1−SSIM) | 1,067,117 | 19.17 | **0.885** | **0.088** |
| Under-densified | ON, grow_grad2d=5e-3 | L1 + 0.2·(1−SSIM) | 15,626 | 17.56 | 0.843 | 0.169 |
| Under-densified L1-only | ON, grow_grad2d=5e-3 | L1 only | 9,990 | 16.23 | 0.729 | 0.270 |

All four runs use gsplat 1.5.3, the V1 best COLMAP sparse init (run_id `scan9_v49_s123_3d428b`, 9,044 points), 33 train / 4 held-out views (test_every=10), 7,000 training iterations, refine_start_iter=500, refine_stop_iter=5000, refine_every=100, reset_opacity_iter=3000. PSNR ranges across the 4 held-out views are recorded in `results/gsplat/baseline_scan9_s42.json` for each row; they capture view-to-view variance on a single training run, NOT seed-to-seed variance.

**Position relative to recent literature:** Six recent sparse-view 3DGS papers (FSGS, CoR-GS, SE-GS, AD-GS, InstantSplat, Opacity-Gradient Densification Control) attribute sparse-view 3DGS failure to "uncontrolled densification" and propose method-level corrections. All compare their corrected method against vanilla 3DGS. None publish the direct ablation — what happens if you simply disable densification on the same dataset their method targets. The four-row table is that direct ablation in the ~33-view regime: not "sparse" by the published convention (most sparse-view papers test at 3, 6, or 12 views) but well below the ~150-view density gsplat's threshold defaults assume. The published literature's "vanilla vs improved method" comparison structure is not designed to surface this kind of result.

**Mechanism (best current explanation):** gsplat's `grow_grad2d=2e-4` default is calibrated against sparse inits with ~100k–200k points (typical for Mip-NeRF360-scale SfM). Scan9's init at 9,044 points is ~15× sparser. Per-Gaussian 2D gradients are proportionally larger because each Gaussian is responsible for more of the image area. At step 600 (the first refinement event), the default threshold fires on **82% of Gaussians per event** (verified empirically: 7,446 / 9,044 above threshold), versus the calibration regime's target of ~5–15%. Compounded across 44 refinement events from step 600 to 4900, N grows to 1,067,117 — 2.2 Gaussians per pixel for a 480k-pixel image. The over-parameterized model produces sub-pixel noise that doesn't hurt feature-space metrics (SSIM 0.885, LPIPS 0.088) and drives PSNR below the frozen baseline by 3.45 dB. We verified this is high-frequency noise rather than a brightness shift: per-channel mean and std of rendered RGB match GT to within 5 / 256 across all 4 held-out views, but per-pixel RMSE is 18–32 / 256.

Raising `grow_grad2d` to 5e-3 (25× the default) brings first-event growth to 8.82% — in the 5–15% calibration band — but the growth then collapses to near-zero by step 3000 because gsplat's `prune_scale3d=0.1` cleanup activates after `reset_every=3000`. Net N at step 5000 is 15,626, only 73% above the 9,044 init. Neither extreme — over-densified at 1.07M nor under-densified at 15k — works at this scene + view-count combination.

**Alternatives considered and rejected:**
- Lower `grow_grad2d` further (1e-4, 5e-5) — would produce even more aggressive over-densification with worse PSNR, not better.
- Tune `grow_grad2d` to a third intermediate value (3e-3, 4e-3) — three runs already span the 2e-4 → 5e-3 range without finding a sweet spot; an intermediate is unlikely to invert the trend without addressing the underlying init-density mismatch.
- Add `scale_reg`, `opacity_reg` from gsplat simple_trainer.py — multi-variable change, defers attribution. May help but cannot be combined with threshold tuning in a single run without losing single-variable discipline.
- Rebuild the sparse init from COLMAP's *dense* fused point cloud (~250k points) instead of the SfM cloud — would put the init density inside gsplat's calibration regime. Highest remaining upside but also highest plumbing cost; deferred to V1.5+ as "richer init" experiment.
- Switch to MCMCStrategy (gsplat's alternative densification class targeting fixed total N) — different strategy entirely; out of scope for the densification-ablation question Day 9 was designed to answer.
- Train longer (15k–30k iterations) — the Day 8 lesson that "longer training on a misconfigured recipe doesn't help" applied. Not pursued.

**Contribution framing:** The contribution is the four-row table itself. A single-line plumbing fix to gsplat's strategy call (DECISIONS 19, the `packed=True` flag) exposed a recipe–data interaction that the published sparse-view 3DGS literature does not directly test in the regime sim-to-recon targets. The table is not "we found a new bug" — it is "we tested a claim the literature makes without evidence in the ~33-view regime, and the result is more nuanced than the literature framing suggests."

**Honest scope of the claim:** This holds for DTU scan9 at 800x599, 33 train / 4 test views, COLMAP SfM sparse init at 9,044 points, gsplat 1.5.3 default `DefaultStrategy` with the parameters above, single seed (42), no scale/opacity regularization, no `random_bkgd` (see DECISIONS 22 for the random_bkgd ablation). Generalizing beyond this configuration is out of scope. A reviewer who wants to know "does this hold on Blender, LLFF, Tanks+Temples?" gets the honest answer: "we don't know; the experiment is scoped to scan9 because V1.5's contribution is the cross-method comparison vs V1's COLMAP, which is also scan9-only." Multi-scene extension is V2 territory.

**Cross-method implication for Day 12:** The 4 held-out cameras the gsplat runs are scored against will be the same 4 cameras COLMAP's fused point cloud is rendered against in Day 12. With gsplat's PSNR underperforming the spec target across all four configurations, the Day 12 framing should NOT be "neural splatting beats classical MVS on novel-view synthesis." It should be "at this scene + view-count combination, neither method is in its happy regime, and the failure modes are different — classical dense MVS loses geometry; neural splatting either over-parameterizes (hurting PSNR while preserving perceptual metrics) or under-fits (hurting all metrics) depending on the densification threshold." That's a more interesting comparison than the expected headline.

---

## 22. random_bkgd costs ~3 dB PSNR on DTU scan9 — independent replication of nerfstudio PR #1441

**Decision:** Document the random_bkgd experiment as Day 9 supporting evidence. When the over-densified gsplat configuration (DECISIONS 21, row 2: grow_grad2d=2e-4, L1 + 0.2·SSIM) was rerun with `random_bkgd=True` (gsplat simple_trainer.py's default), the held-out PSNR at step 700 dropped from **18.66 dB → 15.73 dB** (−2.93 dB) with no change in densification growth rate (83.2% vs 82.3% at step 600 — within noise). The smoke run was sufficient to falsify "random_bkgd will make densification work better on this scene"; the experiment was not extended to a full 7,000-iter run.

**Mechanism:** the `random_bkgd` training trick composites rendered colors with a per-step uniform random background: `colors = α·foreground + (1−α)·random_bkgd`. The trick assumes GT backgrounds ≈ 0 (synthetic scenes with alpha-composited objects against pure black — NeRF Synthetic, Blender RGBA). Under that assumption, pixels in background regions carry no consistent training signal, the model is pushed to learn α=0 there, and capacity is concentrated on the foreground object. DTU scan9's GT backgrounds are gray (~30/255 in dark regions, varying per view), violating the assumption. The L1 residual against GT in background pixels becomes `|α·foreground + (1−α)·random_bkgd − gray_constant|`, which has a noise component of ≈0.5·(1−α) from the random bg term, drowning out the foreground signal during the early phase of training where α has not yet converged.

**Independent replication of nerfstudio PR #1441.** That PR documented up to **8 dB PSNR cost** on Blender's non-RGBA scenes when random_bkgd is enabled, attributed to the same root cause (Blender non-RGBA configurations have backgrounds that aren't pure black). Our 3 dB cost on DTU scan9 lies in the lower half of PR #1441's reported range. The magnitude difference is consistent with the background-color delta: Blender non-RGBA backgrounds are often saturated colors far from black, while DTU's gray ~12% backgrounds are closer to the assumed-zero floor. Larger background-magnitude predicts larger random_bkgd cost, which is what both datasets show.

**Why this is a positive finding rather than just a negative result:** the experiment is two things at once. First, it is a direct datapoint extending PR #1441's claim to a new dataset regime — exactly the kind of independent replication of published claims that strengthens literature without requiring novel methodology. Second, it is a useful warning for practitioners using gsplat's `simple_trainer.py` defaults on real-photography datasets — the default `random_bkgd=True` setting was tuned for synthetic / pure-bg-zero scenes, and applying it unmodified to natural images (DTU, Tanks+Temples, Mip-NeRF360 with non-zero environments) systematically degrades PSNR for the same reason.

**Audit detail — ruling out random_bkgd as a confound on the Day 9 four-row table.** A research review on Day 9 raised the possibility that `random_bkgd=True` had been silently active across all four rows of DECISIONS 21, in which case the table would have been confounded with the random_bkgd cost rather than measuring densification cleanly. We checked via `git show HEAD:modal_app.py | grep random_bkgd` and got "no match in HEAD" — confirming that `random_bkgd` was added to the codebase only in the most recent commit (the experiment that produced the 15.73 dB result), and was not present during the four full runs that produced the four-row table. The four-row table is therefore unconfounded by random_bkgd. The check took 30 seconds; a rerun-to-control would have cost ~$0.40 of Modal and ~2 hours of attention. The cheap version of audit-before-act caught the inference error before it cost real money. This is the generalized lesson recorded in the persistent memory entry `feedback_temporal_blind_spot.md`: any audit that reads current-tree code to reason about past experiments has a temporal blind spot, because the code may have been edited after the experiment ran. Using `git show HEAD:` to reconstruct historical state is the cheap fix.

**Honest scope of the claim:** the 3 dB cost is for DTU scan9 specifically, at 33 training views, with gsplat 1.5.3's DefaultStrategy at grow_grad2d=2e-4 (the over-densified config), measured at training step 700 from a single smoke run. We did not run random_bkgd in the under-densified or L1-only configurations because the smoke at the 18.66-baseline configuration was sufficient to falsify the "random_bkgd helps densification" hypothesis, and continuing to ablate it across all DECISIONS 21 rows would consume budget without changing the answer to the Day 9 ablation question. Extending the random_bkgd ablation to other gsplat configurations or to additional datasets is out of scope for V1.5; the four-row densification table (DECISIONS 21) is the headline contribution and the random_bkgd replication is supporting evidence.
