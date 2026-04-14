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

---

## 23. Day 10 multi-seed scope — frozen + over-dens × {42, 123, 7}

**Decision:** Day 10's multi-seed sweep targets two recipes from DECISIONS 21's four-row table — "frozen" (row 1) and "over-densified" (row 2) — each replicated across the three seeds matching V1's stress sweep convention: {42, 123, 7}. Total: 6 Modal runs. The two recipes selected bound the extrema of the densification axis (N=9,044 vs N≈1.07M), and characterizing those extrema with seed-to-seed variance is sufficient for Day 12's cross-method table to report a bidirectional failure finding. The two interior recipes (under-dens at grow_grad2d=5e-3 with L1+SSIM, and under-dens L1-only) remain single-seeded in DECISIONS 21 as ablation context.

**What this is NOT a decision about:**
- The "3DGS baseline" framing for Day 12's cross-method table. That framing gets resolved in a Day 12 decision after Day 10's actual numbers land. Day 10 produces the median+range data; Day 12 decides how it's narrated.
- Training 3DGS from scratch, novel recipe development, or any methodological work beyond reproducing DECISIONS 21 recipes with seed-to-seed variance. The V1.5 spec's "use the standard gsplat recipe" scope discipline still holds — Day 10 tests how stable the standard recipe's behavior is across seeds, not whether it's the right recipe.
- Per-scene or per-view-count generalizability of the four-row finding. Day 10 is scan9 at 33 views, same as DECISIONS 21.

**Rationale — load-bearing (cross-method honesty):** Day 12's cross-method comparison against V1's COLMAP needs to report 3DGS with seed-to-seed variance matching V1's variance discipline (DECISIONS 16), and it needs to report more than one 3DGS operating point to be an honest method comparison. A single frozen-only row would make the table read as "3DGS as novel-view synthesis on V1's geometry" rather than as a method alternative, because the frozen recipe renders V1's sparse points through a trained radiance field without adding any Gaussians. Two rows — frozen (PSNR winner) and over-dens (SSIM/LPIPS winner) — expose the bidirectional trade-off the four-row ablation surfaced and force the Day 12 write-up to engage with the densification axis rather than paper over it with a single headline number.

**Rationale — incidental (V2 ensemble variance):** If V2 gets built (trigger condition unchanged from the V1.5/V2 revised plan), its per-point confidence channel will draw on ensemble variance from 3DGS multi-seed. Multi-seeding the over-dens recipe gives V2 a real signal — ~1M Gaussians diverging across seeds under different sampling paths starting at step 600. Multi-seeding the frozen recipe gives V2 a near-zero signal — N=9,044 is fixed, only the radiance field parameters evolve. This is a free benefit of the two-recipe choice, not a reason for it. Selecting recipes to pre-enable V2 features would be V1.5 scope creep; the explicit framing here is that V2 utility is incidental and the cross-method honesty is load-bearing. If V2 never triggers, no Day 10 decision was distorted.

**Selection-effect rebuttal — characterize the extrema, not exclude the losers:** A careful reviewer could read "we multi-seeded the PSNR winner and the perceptual winner and single-seeded the two under-dens rows" as cherry-picking. The honest answer is that the two extrema (frozen at N=9,044 and over-dens at N≈1.07M) bound the densification axis the ablation explores, while the two under-dens rows (N=15,626 and N=9,990) sit on the same side as over-dens — they are interior points of the "working densification at different thresholds" regime. Multi-seeding interior points adds noise estimates without changing the qualitative finding (densification hurts PSNR on sparse inits at ~9k points with gsplat's default threshold). The four-row ablation in DECISIONS 21 already documents the interior single-seed numbers; Day 10's multi-seed extends the endpoints. The framing is "characterize the extrema" rather than "exclude the underperformers," and DECISIONS 21's presence in the README preserves the honest single-seed data for the interior.

**Pre-committed success bands (no goalpost moving after observation):**

Frozen recipe, PSNR median across 3 seeds:
- Success band: 22.3 – 23.0 dB (±0.4 dB of the DECISIONS 21 row 1 anchor 22.62)
- Defensible floor: 21.8 dB (wider seed noise than expected but within one decibel)
- Stop-and-diagnose: < 21.8 dB → frozen-mechanism substitute does NOT reproduce the bug's behavior; reopen DECISIONS 24 investigation.

Over-dens recipe, PSNR median across 3 seeds:
- Success band: 18.8 – 19.5 dB (±0.4 dB of DECISIONS 21 row 2 anchor 19.17)
- Defensible floor: 18.3 dB
- Stop-and-diagnose: < 18.3 dB, OR N_final outside [850k, 1.25M] → mechanism divergence from the ablation.

Seed-to-seed range (max − min) on either recipe:
- Expected: ≲ 1.0 dB (3DGS on a fixed COLMAP init is stochastic but not chaotic)
- Notable if > 2.0 dB → becomes a finding in its own right, reported like DECISIONS 16.

**Why seeds {42, 123, 7}:** V1 DECISIONS 16 committed to these three seeds for the stress sweep. Using the same set in V1.5 preserves V1 → V1.5 narrative continuity in the README's methodology section and avoids the confounding footnote "V1.5 chose different seeds because its variance posture differs." Seed 42 is the lead (matches Day 8/9 single-seed runs); 123 matches V1's best-performing baseline (the scan9_v49_s123_3d428b init this sweep uses); 7 is an independent third seed with no correlation to V1 artifacts.

**Alternatives considered and rejected:**
- Multi-seed all four rows of DECISIONS 21 (option (D) from Day 9 dialogue): rejected because the two under-dens rows are dominated by over-dens on every metric; adding seed noise on them would cost ~$1.50 without changing the qualitative finding. See rebuttal above.
- Multi-seed frozen only (option (A)): rejected because it weakens the cross-method table narrative. See load-bearing rationale above.
- Multi-seed over-dens only (option (B)): rejected because the PSNR winner from the ablation is frozen, and omitting frozen from the cross-method table would hide the headline result.

**Cost:** 6 runs × ~$0.25 each ≈ $1.50. Running total after sweep: ~$0.65 (through Day 9) + ~$1.77 (Day 10 smoke + verification + sweep) ≈ $2.42 of $50 cap.

---

## 24. Frozen-baseline mechanism substitution — Day 10 multi-seed sweep

**Decision:** The "frozen" recipe in Day 10's multi-seed sweep (DECISIONS 23) is produced by setting `densify_start_iter = densify_stop_iter = reset_opacity_iter = 999999` when calling `train_gsplat`. This makes gsplat's `DefaultStrategy` functionally inert for all 7000 training steps: no densification events fire, no splits or duplicates occur, no opacity resets happen. Final state N = 9,044 is identical to DECISIONS 21 row 1. This substitutes a clean mechanism for Day 8's silent `packed=False` `IndexError` bug (DECISIONS 19), which produced the same final state via `step_post_backward` crashing at step 0 with all subsequent strategy calls silently caught by a narrow `try/except` guard.

**What differs between the two mechanisms:**

*Bug mechanism (Day 8, documented in DECISIONS 19):* `DefaultStrategy.step_post_backward` is called every step. `_update_state`'s packed=False branch accesses `info["radii"]` expecting shape `[C, N, 2]` but receives packed shape `(nnz, 2)` and raises `IndexError: tuple index out of range`. The exception is caught by `train_gsplat`'s wrapper, `strategy_ok` flips to `False`, and all subsequent `step_pre_backward` and `step_post_backward` calls skip the strategy body entirely. Neither refinement nor opacity reset ever executes. The strategy's internal state (`grad2d`, `count` accumulators) is never populated.

*Clean substitute mechanism (Day 10, this decision):* `DefaultStrategy` constructs cleanly with three refinement triggers set past `n_iterations`. `step_pre_backward` and `step_post_backward` are called every step; inside each, the `step >= refine_start_iter` and `step < refine_stop_iter` guards return early without invoking `_grow_gs` or `_prune_gs`. `reset_opacity_iter=999999` prevents the opacity-reset branch from firing. `_update_state` *does* execute (the strategy is "alive"), accumulating per-Gaussian 2D gradient statistics into `strategy_state`, but the accumulated state is never consumed by any refinement op because those ops are gated behind triggers that never fire.

**Observable difference at training time:** the clean substitute runs per-step `_update_state` accumulation into `strategy_state["grad2d"]` and `strategy_state["count"]` via a cluster of small CUDA kernels. The bug mechanism did neither — `_update_state` raised `IndexError` before any kernel launched. The wall-clock difference is ~350 ms over 7000 steps, immaterial at a 5–15 minute training budget.

**What is NOT known a priori:** whether the kernels `_update_state` launches in the clean substitute advance the `torch.cuda` RNG state in ways that shift downstream sampling. `_update_state` itself does not call `torch.rand*` in the code paths audited during Day 9, but a thorough audit of gsplat's source to prove no kernel internally touches the generator state would take longer than just running the verification run and measuring the outcome. The concrete question is: does seed=42 under the clean substitute produce the same *trajectory* as seed=42 under the bug, or merely the same *distribution* of trajectories? If torch's CUDA RNG is advanced by different amounts between the two mechanisms, the answer is "same distribution but different trajectory" — and seed=42 specifically could land anywhere in that distribution regardless of the shared seed.

**Why not re-introduce the bug?** Two reasons. First, the codebase post-dc49f8f has the `packed=True` fix landed (DECISIONS 19) and no longer raises `IndexError`; reintroducing the bug would require reverting the fix, which would simultaneously break the over-dens recipe. Second, the bug's error-catching path relied on an `except`-guard anti-pattern that has been explicitly rewritten to print on first failure unconditionally (DECISIONS 19 methodology note; `feedback_except_guard_antipattern.md`). Going back to that pattern for "backwards-compat reproducibility" would re-open a silent-failure surface for no methodological benefit. The clean substitute is the only defensible path, and its equivalence to the bug's final state is the empirical claim the verification run tests.

**Pre-launch verification run (step 7 in the Day 10 execution sequence):** one run, seed=42, frozen recipe, full 7000 iterations, full PSNR/SSIM/LPIPS evaluation on the 4 held-out views. Expected outcome: PSNR median within ±0.5 dB of DECISIONS 21 row 1's 22.62 dB (seed-to-seed noise bound consistent with 3DGS on a fixed COLMAP init).

Actual outcome (verification run, 2026-04-14, clean pass):
- seed=42 frozen PSNR median: **22.59 dB** (range 15.90 – 23.69 across 4 held-out views; per-view: rect_001 21.54, rect_014 23.65, rect_028 15.90, rect_041 23.69)
- seed=42 frozen SSIM median: **0.816** (range 0.767 – 0.831)
- seed=42 frozen LPIPS median: **0.187** (range 0.174 – 0.230)
- seed=42 frozen N_final: **9044** (hard constraint satisfied)
- |ΔPSNR vs DECISIONS 21 row 1|: **0.03 dB** (22.59 vs 22.62; well inside the ±0.5 dB pass band)
- SSIM and LPIPS match DECISIONS 21 row 1 to 3 decimal places (0.816 → 0.816; 0.187 → 0.187)
- Wall-clock: 158 s on a warm container (verification reused the container left warm by the over_dens smoke a few minutes earlier)

Pass criterion: |ΔPSNR| ≤ 0.5 dB AND N_final == 9044 AND no exceptions in the Modal log.

**Verification verdict: PASS on all criteria.** The clean substitute mechanism produces a final state that matches DECISIONS 21 row 1's bug-induced result to within the floating-point-reordering floor at seed=42. The RNG-consumption divergence concern from the "What is NOT known a priori" paragraph did not fire empirically: not only is the final distribution consistent, the seed=42 trajectory itself is effectively identical modulo a 0.03 dB residual consistent with single-kernel reordering. Step-0 gradient magnitudes from the preceding smoke run (bit-exact matching Day 9 Smoke A's grad 3.301e-05 / absgrad 4.905e-05) confirmed bit-for-bit equivalence at the first backward pass; the verification run demonstrates that equivalence propagates through 7000 iterations to a final PSNR median 0.03 dB off the anchor. DECISIONS 24's mechanism-substitution claim is validated empirically and the 6-run sweep gate is cleared for the frozen recipe.

The `rect_028` view at 15.90 dB is the same "hardest view" outlier that Day 9's Smoke D comparison flagged (it jumped from 15.86 → 16.98 when densification capacity was added at step 600–700 in the over-dens regime). Its presence at 15.90 here — unmoved from the Day 8 frozen-baseline value despite the full 7000 iterations of training — is consistent with the DECISIONS 21 row 1 conclusion that the frozen recipe cannot fit this particular view even with full training, because the fit is capacity-limited at N=9044 rather than optimizer-limited. This is another weak-but-consistent signal that the clean substitute faithfully reproduces the bug path's behavior on hard views, not just on the median.

**Why the verification run is load-bearing, not insurance:** given the RNG-consumption uncertainty above, the verification run is the empirical test that distinguishes "different mechanism, same distribution" (acceptable) from "different mechanism, genuinely different dynamics that happen to share the N=9044 failure mode" (NOT acceptable — would invalidate Day 12's cross-method frozen row). A PSNR median within ±0.5 dB of 22.62 dB is consistent with "same distribution, seed-42-landed-similarly." A PSNR median outside that band is evidence the mechanisms produce distinct distributions, and DECISIONS 24's final-state-equivalence claim fails. The 6-run sweep does NOT proceed until the verification run answers this question.

**If outside band — diagnostic path:** (1) re-read gsplat's `DefaultStrategy.step_post_backward` and `_update_state` source with a targeted grep for `torch.rand`, `.random_`, `multinomial`, `.normal_`, or any kernel that could touch generator state; (2) run a diagnostic pair of smokes (current clean-substitute at seed=42 vs an explicit first-class `freeze` flag that short-circuits all strategy calls before `_update_state` runs — the latter is a definitively-matching-the-bug alternative mechanism) with `torch.cuda.get_rng_state()` dumps at steps 0, 1, 10, 100, 1000, 6999 to localize where the trajectories diverge. Diagnostic cost: ~$0.10. The diagnostic result informs whether to (a) revise DECISIONS 24 to switch to the first-class `freeze` flag and re-run the verification, or (b) accept the divergence as "different distribution, report both numbers, defer DECISIONS 21 row 1's applicability to a Day 12 footnote." **Option (a) is preferred:** it preserves the four-row ablation's headline framing by switching to a mechanism that matches the bug's RNG-skip behavior. Option (b) is a fallback only if the diagnostic reveals the divergence is large enough that no clean substitute is achievable, and demoting the headline is the only honest path.

**Alternatives considered and rejected:**
- *Add a first-class `freeze: bool = False` parameter to `train_gsplat` that short-circuits all strategy calls at the call site.* Cleaner semantically (a reader sees `freeze=True` and understands intent immediately) but adds a code surface. The `densify_start_iter=999999` approach uses existing infrastructure, is three explicit parameter overrides rather than a schema change, and the in-script comment at `RECIPES["frozen"]` in `experiments/run_gsplat_multiseed.py` documents intent clearly. Net: less code change, equivalent clarity — at the cost of the RNG-consumption uncertainty documented in the "What is NOT known a priori" paragraph above. If the verification run fails and the diagnostic path localizes the divergence to `_update_state`'s kernel stream, the first-class `freeze` flag becomes the first-class recovery path (diagnostic-path option (a), preferred over demoting DECISIONS 21 row 1's headline framing).
- *Re-introduce the bug via a `_day8_compat: bool = False` flag.* Rejected on sight. The bug's except-guard anti-pattern is explicitly fixed (DECISIONS 19) and reintroducing it would reopen a silent-failure surface for no benefit.
- *Set only `densify_start_iter=999999`, leave `reset_opacity_iter=3000` at default.* Rejected because opacity reset at step 3000 would fire in the clean substitute but did NOT fire in the Day 8 bug (the whole `step_post_backward` body was caught). Leaving `reset_opacity_iter` at default would introduce a mechanism divergence (one opacity reset event at step 3000) that would likely shift the PSNR result away from 22.62 dB by 0.5-2 dB. Setting all three refinement triggers past `n_iterations` makes the strategy truly inert and preserves final-state equivalence.
- *Set `densify_stop_iter < densify_start_iter` as a guard pattern.* Rejected because gsplat's `DefaultStrategy.__init__` may sanity-check this and raise, blocking the substitute before it can be tested. The 999999-on-all-three approach is the least clever and least likely to trip an unknown validator.

**Honest scope of the claim:** the mechanism substitution is validated by the verification run's PSNR matching DECISIONS 21 row 1 within seed noise. It is NOT validated against the *bit-level* intermediate gradient trajectory of Day 8's bug-induced run. Final state (N=9044, final PSNR ≈ 22.62 dB) is what is claimed to be preserved; intermediate state during training (grad accumulators in `strategy_state`, CUDA RNG consumption count) differs. A reviewer who wants to know "does the clean substitute produce identical intermediate optimizer states" gets the honest answer: "no, and it doesn't need to — what we report is final novel-view metrics, not intermediate training trajectories, and the verification run gates on the final-metric equivalence."

---

## 25. Day 10 multi-seed sweep reframes frozen and over-dens as 3-seed distributions — DECISIONS 23 anchoring falsified, DECISIONS 20 variance-source list incomplete

**Decision:** Supersede DECISIONS 21 row 1 and row 2 as the canonical recipe characterizations for downstream work. The frozen and over-dens recipes are now characterized by the 3-seed distributions from Day 10's multi-seed sweep (seeds {42, 123, 7}, commit `acf2e1b`), not by DECISIONS 21's single-seed-42 point estimates. DECISIONS 21 and DECISIONS 23 are preserved as pre-commit records without retroactive edit; this entry documents what the observed data forced us to update, and names two distinct pre-commit gaps that the sweep and the P2 diagnostic surfaced.

---

**Gap 1 — DECISIONS 23's anchoring failure (quantitative).**

DECISIONS 23 pre-committed the frozen-recipe PSNR success band as "22.3 – 23.0 dB (±0.4 dB of DECISIONS 21 row 1 anchor 22.62)" under the implicit assumption that seed-to-seed variance would be ≲1 dB. Both parts of that assumption were falsified by observation:

- The anchor was a single-seed-42 datapoint from Day 9 (DECISIONS 21 row 1), not a multi-seed median. Anchoring a multi-seed success band to a single-seed point estimate was a methodological error made before any seed-variance data existed for this recipe at this scene+view-count combination.
- The actual 3-seed range is **1.53 dB** (22.56 at seed=42 down to 21.03 at seed=123), and the median-of-medians is **21.32 dB** — 0.48 dB below DECISIONS 23's "defensible floor" of 21.8 dB and fully outside the 22.3–23.0 success band.

The over-dens success band (18.8–19.5 dB) was also falsified: median-of-medians 18.50 dB (below the band, above the 18.3 dB floor) and seed-to-seed range 3.19 dB (above the 2.0 dB "notable" threshold → a finding on its own per DECISIONS 23's seed-range trigger). N_final band (850k–1.25M) held — all three seeds landed between 1.09M and 1.13M.

This is a methodology lesson about pre-commit *quality*, not about either recipe's mechanism. A single-seed point estimate is the wrong kind of anchor for a multi-seed success band, and the ≲1 dB noise assumption was not grounded in any prior data for this regime. DECISIONS 25's corrected characterization uses the distribution itself as the record, with no external anchor.

**Gap 2 — DECISIONS 20's structural failure (qualitative).**

DECISIONS 20 pre-committed the posture for what V1.5's multi-seed variance would measure. It named two sources: (a) gsplat's split/duplicate sampling paths diverging from step 600 onward under different torch RNG seeds, and (b) rasterizer atomic-add ordering non-determinism. It did NOT name a third source that turns out to dominate on the frozen recipe: (c) per-step training-image sampler permutation via `np.random.default_rng(seed)` at the line that draws the next image each iteration.

For the frozen recipe, source (a) is structurally unreachable (no refinement events fire), source (b) is at the noise floor (<0.03 dB as demonstrated by the P2 diagnostic's torch-seed isolation below), and source (c) is ~1.5 dB — the entire observed seed range. The "what multi-seed variance will measure" paragraph in DECISIONS 20 is incomplete not because its named sources were wrong but because it missed the third source entirely. This is a different kind of pre-commit failure than Gap 1 — structural (which sources were worth naming) rather than quantitative (how wide the band should be).

Both gaps are documented here because variance-as-contribution is the load-bearing V1 → V1.5 narrative, and folding either failure silently into the DECISIONS 25 mechanism citation would undo exactly the pre-commit discipline the variance posture was supposed to enforce. Future multi-seed posture declarations in this repo should name all per-step sampling streams as first-class variance sources, not just torch-seed-driven sources. Generalized memory entry will be added on DECISIONS 25 commit.

---

**Corrected characterization — 3-seed distributions**

**Frozen recipe** (L1-only, `densify_start_iter = densify_stop_iter = reset_opacity_iter = 999999`, see DECISIONS 24), seeds {42, 123, 7}, 7000 iterations, 4 held-out views:

| Seed | PSNR median | SSIM | LPIPS | N_final |
|---|---|---|---|---|
| 42 | 22.56 dB | 0.816 | 0.187 | 9,044 |
| 123 | 21.03 dB | 0.808 | 0.200 | 9,044 |
| 7 | 21.32 dB | 0.813 | 0.194 | 9,044 |

- Median-of-medians: **21.32 dB**
- Seed-to-seed range: **1.53 dB** (22.56 − 21.03)
- N_final: 9,044 across all three seeds (hard constraint, mechanism preserves initial point count)

**Over-dens recipe** (L1 + 0.2·SSIM, `grow_grad2d = 2e-4`, gsplat `DefaultStrategy` defaults), seeds {42, 123, 7}, 7000 iterations, 4 held-out views:

| Seed | PSNR median | SSIM | LPIPS | N_final |
|---|---|---|---|---|
| 42 | 18.90 dB | 0.884 | 0.087 | 1,093,077 |
| 123 | 15.71 dB | 0.864 | 0.120 | 1,129,574 |
| 7 | 18.50 dB | 0.894 | 0.084 | 1,107,423 |

- Median-of-medians: **18.50 dB**
- Seed-to-seed range: **3.19 dB** (18.90 − 15.71)
- N_final distribution: [1.09M, 1.13M, 1.11M], ±3% across seeds — growth dynamics are stable, variance is not in how many Gaussians we end up with but in where they go.

DECISIONS 21 row 1's 22.62 dB and row 2's 19.17 dB are single samples from these distributions, measured at seed=42 before any variance structure was characterized. Seed=42 is (coincidentally, given it was chosen in Day 8 before any seed-variance data existed) the *high* outlier of the 3-seed distribution on both recipes: +1.24 dB above the median-of-medians on frozen, +0.40 dB above on over-dens. **Seed=42's high-outlier status was not known at Day 9 and does not constitute post-hoc selection bias: the seed was chosen as V1's stress-sweep lead (DECISIONS 16) before any V1.5 seed-variance data existed.** This is a historical footnote, not a cherry-picking finding, and Day 9's single-seed ablation framing stands unchanged.

---

**Frozen-recipe mechanism — P2 diagnostic, Band α with quantitative surplus**

The P2 diagnostic (run immediately after the sweep, same day, pre-committed against three bands in `preflight/2026-04-14_day10_p2_diagnostic.txt`) was designed to test whether the per-step training-image sampler drives the frozen recipe's observed seed range. Protocol: two Modal runs on the frozen recipe with `torch.manual_seed(42)` held constant and only `image_order_seed` (a new optional parameter added to `train_gsplat`, see DECISIONS 25 commit diff on `modal_app.py`) varied between 42 and 123.

| | Run A | Run B |
|---|---|---|
| torch seed (manual + cuda + np.random legacy) | 42 | 42 |
| image_order_seed (drives `np.random.default_rng`) | 42 | 123 |
| PSNR median | **22.5623 dB** | **21.0570 dB** |
| SSIM median | 0.8158 | 0.8082 |
| LPIPS median | 0.1872 | 0.1996 |
| N_final | 9,044 | 9,044 |
| elapsed | 156.2 s (cold) | 69.8 s (warm) |

**|ΔPSNR(B − A)| = 1.5053 dB.** Pre-committed bands: α > 0.5 dB (image-order dominant), γ 0.2–0.5 dB (partial), β < 0.2 dB (not image-order). Observed ΔPSNR lands cleanly in **Band α**. The 1.5053 dB delta is also effectively equal to the 3-seed sweep range of 1.53 dB across {42, 123, 7} — image-order permutation alone reproduces the entire observed spread.

**Two-sided quantitative confirmation (result exceeds the pre-committed Band α claim):**

1. **Run A ≡ sweep-frozen_s42.** Run A's 22.5623 dB compared against sweep-frozen_s42's 22.5625 dB is a |Δ| of **0.0002 dB**. This validates the code change (the `image_order_seed=None → seed` fallback is bit-identical to the pre-change path) AND confirms that at torch seed=42 and image_order seed=42, the torch-seed component and code-path component of variance are both at the floating-point-rounding floor.
2. **Run B ≡ sweep-frozen_s123.** Run B's 21.0570 dB compared against sweep-frozen_s123's 21.0297 dB (torch seed=123 via legacy single-seed path, image_order=123) is a |Δ| of **0.027 dB**. Only 0.027 dB separates a run where torch seed is 42 from a run where torch seed is 123 — when image_order is held at 123 in both cases.

Reading both sides together: **of the 1.53 dB observed frozen seed range, the torch-seed component is <0.03 dB and the image-order component is ~1.5 dB.** Image order isn't just the dominant driver the diagnostic was pre-committed to test — it's effectively the *sole* driver at the 4-view PSNR resolution being measured. This result exceeds the pre-committed Band α claim and is flagged explicitly here rather than folded silently into the mechanism citation: the diagnostic was set up to test "dominant driver" (|Δ| > 0.5 threshold) and returned |Δ| ≈ 1.5 dB — a result that accounts for the entire 1.53 dB 3-seed sweep range within the invocation-noise floor.

**Mechanism interpretation:** at 7000 iterations over 33 training images with ~212 visits per image on average, different per-step image-order permutations drive the 9044-Gaussian optimizer to different local minima in the radiance-field parameter space. The fixed-capacity model cannot average out the permutation effect the way a higher-capacity model might. Each permutation biases the optimizer's step-by-step visit pattern, and different biases produce different fits on the same fixed Gaussian cloud. This is consistent with the published literature on optimizer-path dependence in stochastic-gradient training but has not, to the best of this repo's lit search, been directly reported as a dominant seed-variance source on frozen-init 3DGS at sparse-init scale.

---

**Over-dens-recipe mechanism — partial, multi-source, qualitative frame finding**

The P2 diagnostic was run on the frozen recipe only. Over-dens variance has multiple live stochasticity sources: (a) gsplat split/duplicate sampling (live — refinement fires as designed, N grows to ~1.1M), (b) rasterizer atomic-add ordering (live, always is), (c) per-step training-image sampler (live, same code path as frozen). Their relative contributions for the over-dens recipe are NOT isolated in this phase. An analogous P2-style diagnostic (vary `image_order_seed`, hold torch seed) on over-dens would cost ~$0.10 and would answer the image-order question for this recipe; it is deferred to Day 13 as a contingency rather than pursued here, because the V1.5 timeline is tight and the qualitative frame inspection below is sufficient to characterize the variance at a defensible level for the Day 12 cross-method writeup.

**Frame-inspection result (P1 sub-step, parallel to DECISIONS 25 drafting, frames already downloaded by the sweep):**

Compared over_dens_s42 and over_dens_s123 rendered PNGs at the held-out views. Per-view PSNR breakdown across the three over-dens seeds:

| View | s42 | s123 | s7 | s123 vs s42 |
|---|---|---|---|---|
| rect_001 | 21.76 | 15.77 | 23.25 | −5.99 dB |
| rect_014 | 16.53 | **18.67** | 16.58 | **+2.14 dB (s123 wins)** |
| rect_028 | 19.51 | 15.64 | 17.88 | −3.87 dB |
| rect_041 | 18.29 | 14.34 | 19.12 | −3.95 dB |

The per-view PSNR crossing on `rect_014` (s123 beats both s42 and s7 by ~2 dB on this view, while losing the other three views by 4–6 dB) **rules out mode collapse as the failure mechanism**. A collapsed reconstruction would uniformly lose every view, not flip the ordering on one specific view. Visual inspection of the PNGs at rect_001 and rect_014 confirms: both s42 and s123 render the DTU scan9 miniature building model with comparable foreground geometry, building edges, and color fidelity. The material difference is in the **off-surface floater distribution** — dark speckle noise scattered across the white background regions. s123 has denser floater speckle than s42 in the `rect_001` background (hurting that view) but less floater speckle in the `rect_014` background below the buildings (helping that view).

**Mechanism interpretation for over-dens variance:** at 1.07M total Gaussians, the model has enough capacity to cover the foreground scene and still have excess Gaussians that settle into off-surface floater positions. Different image-order permutations (plus possibly different split/duplicate sampling paths, which are not isolated in this phase) drive the ~1M-scale Gaussian cloud to different local minima where the *floater distribution in 3D space differs across seeds*. Each held-out view's PSNR depends on whether its line of sight passes through a sparse or dense floater region. This manifests as the per-view PSNR crossings observed above, and it extends DECISIONS 21's Day 9 mechanism claim ("the over-parameterized model produces sub-pixel noise that doesn't hurt SSIM/LPIPS and drives PSNR below the frozen baseline") with a secondary finding: *the sub-pixel noise is not uniformly distributed across views; its per-view intensity is seed-dependent and can swing PSNR by ~6 dB on a single view across otherwise-comparable reconstructions.*

Caveat on SSIM/LPIPS: over_dens_s123 (0.864 / 0.120) is slightly worse than s42 (0.884 / 0.087) and s7 (0.894 / 0.084) across the median — not only on PSNR. So the s123 result is not purely "same reconstruction, floaters moved around without affecting perceptual metrics." It is a *slightly* worse reconstruction in feature-space metrics AND a floater-redistribution PSNR pattern. The floater story is the dominant per-view narrative but not the entire story. A full decomposition (image-order vs refinement-sampling, plus per-seed perceptual-metric drift) would require the deferred over-dens P2 diagnostic.

---

**Preservation of DECISIONS 20, 21, 23 as pre-commit records**

Neither DECISIONS 20 nor 21 nor 23 is edited retroactively. All three stand as pre-commit records of the methodology that produced the Day 10 findings:

- **DECISIONS 20** reports the pre-committed RNG posture and variance-source list. Source (c) is missing. DECISIONS 25 identifies the gap; DECISIONS 20 is preserved unchanged.
- **DECISIONS 21** reports the four-row densification ablation at seed=42 single-seed. Its numbers are correct at that seed; they're single samples from distributions Day 10 now characterizes. DECISIONS 21's Day 9 contribution framing stands.
- **DECISIONS 23** reports the pre-committed multi-seed sweep scope, seeds, and success bands. The bands were falsified by observation; that falsification is itself the finding DECISIONS 25 documents, and rewriting DECISIONS 23 post-hoc would undo exactly the pre-commit discipline it was designed to enforce.
- **DECISIONS 24** is unchanged by DECISIONS 25 — in fact, the Day 10 sweep *validates* the DECISIONS 24 mechanism-substitution claim a second time beyond the original verification run. Against DECISIONS 21 row 1's 22.62 dB anchor, both seed=42 reproductions pass DECISIONS 24's pre-committed ±0.5 dB band: the verification run landed at 22.59 dB (|Δ|=0.03 dB), and the sweep-frozen_s42 run landed at 22.56 dB (|Δ|=0.06 dB). The P2 Run A at 22.5623 dB provides a third independent seed=42 datapoint matching sweep-frozen_s42 to 0.0002 dB. **No revision to DECISIONS 24 is needed.** DECISIONS 25 adds context about the 3-seed distribution but does not weaken DECISIONS 24's seed=42 mechanism claim: at that seed the mechanism substitute and the bug's final state are indistinguishable at the floating-point-rounding floor; seed=42 itself is just the high outlier of a ~1.5 dB distribution we now know about.

Editing pre-commits after they're falsified is the failure mode the entire pre-commit discipline is designed to prevent. Keep them; document the falsification here.

Footnote pointers from DECISIONS 20/21/23 to DECISIONS 25 will be added at README edit time (Day 13), not now — the footnotes are a presentation-layer concern, not a DECISIONS.md edit.

---

**Implications for downstream work**

1. **V1.5 README, when written (Day 13 per plan):** lead with the 3-seed distributions, not DECISIONS 21's single-point rows. The four-row ablation table remains in the README as single-seed ablation context (its Day 9 methodological role), but the frozen and over-dens recipes' *headline numbers* for the cross-method comparison are median-of-medians and 3-seed range, sourced from DECISIONS 25. The variance source is labeled explicitly per `feedback_variance_labeling.md`: view-to-view within a run vs seed-to-seed across runs are not the same quantity.

2. **Day 12 cross-method comparison table:** uses 3-seed distributions for both 3DGS rows. DECISIONS 23's "two recipes = two rows in Day 12's table" framing holds; each row reports median+range from DECISIONS 25 rather than DECISIONS 21's single points. Day 12 also needs to decide whether to footnote the over-dens per-view PSNR crossings or fold them into the table narrative — a presentation call to make when drafting, not now.

3. **Seed=42 as high outlier on both recipes:** supplementary observation only. Day 9's choice of seed=42 predates any seed-variance data and is not re-cast as cherry-picking. The methodological mitigation is that subsequent phases report distributions rather than point estimates — DECISIONS 16 already established this for V1 and DECISIONS 25 extends it to V1.5.

4. **Over-dens variance decomposition:** Day 13 contingency. If the Day 12 / Day 13 narrative needs the over-dens mechanism fully characterized (image-order vs refinement-sampling split, plus per-seed perceptual-metric drift), run an analogous P2-style diagnostic at ~$0.10. If the README is satisfied with the frame-inspection qualitative finding and the "multiple live sources, dominant mechanism unidentified" honest framing, skip it. This is a decision to make when drafting the README, not now.

5. **DECISIONS 20 generalized lesson:** future multi-seed posture declarations in this repo (or any future repo touched from this codebase) should name all per-step sampling streams as first-class variance sources, not just torch-seed-driven sources. The `np.random.default_rng(seed)` per-step training-image sampler was a third variance source that DECISIONS 20 missed; any future pre-commit that lists "what multi-seed variance will measure" should explicitly enumerate every RNG stream the training loop consumes. A memory entry `feedback_rng_stream_enumeration.md` will be added on the DECISIONS 25 commit.

---

**Cost accounting**

| Phase | Cost |
|---|---|
| V1 total | ~$5.00 |
| V1.5 Day 8 | ~$0.30 |
| V1.5 Day 9 | ~$0.35 |
| V1.5 Day 10 smoke + verification | ~$0.22 |
| V1.5 Day 10 sweep (6 runs) | ~$1.09 |
| V1.5 Day 10 P2 diagnostic (2 runs) | ~$0.10 |
| **V1.5 running total** | **~$2.06 of $50 cap** |

Day 10 total ≈ $1.41, well under the spec's $50 cap and well under the original V1.5 Day-8–14 phase budget estimate.

**Planned vs actual:** the post-Day-9 plan doc estimated Day 10 multi-seed at ~$0.30. Actual spend was ~$1.41 — a **4.7× overrun**. The overrun reflects the cost of verification + reproducibility checks + the P2 diagnostic triggered by the DECISIONS 23 band violation, NOT a multi-seed-per-se cost blowout. The 6-run sweep alone was $1.09, in line with naive multi-seed scaling from Day 8's ~$0.30-per-run estimate. The extra $0.32 is the discipline tax: verification gate ($0.17), smoke ($0.05), P2 diagnostic ($0.10). Future days' budgets should expect similar multipliers on top of naive run-count × per-run estimates if verification and diagnostic discipline are followed as written. The rate-limiter on Day 11+ is user review time and writeup discipline, not Modal compute.

---

**Honest scope of the claims in DECISIONS 25:**

- The frozen-recipe image-order-dominant mechanism claim is empirically validated at torch seed=42 by the P2 diagnostic's two-sided confirmation (Run A ≡ sweep-frozen_s42 at 0.0002 dB; Run B ≡ sweep-frozen_s123 at 0.027 dB). It has NOT been independently verified at torch seeds 123 or 7 — the diagnostic only ran two image_order values at one torch seed. Extending the isolation across all three torch seeds would be a ~$0.30 additional diagnostic not pursued in this phase. The claim is strong but not exhaustive; a reviewer who asks "does the image-order-dominant mechanism hold at non-42 torch seeds?" gets the honest answer "we didn't test, but the two-sided confirmation at seed=42 is as strong as a two-seed diagnostic can produce."
- The over-dens floater-distribution mechanism claim is based on frame inspection at seeds 42 and 123 only, at two of the four held-out views (`rect_001` and `rect_014`). Seed 7's frames and the other two views were not visually inspected. The per-view PSNR crossing is verified from the JSON data across all seeds and views (table above). The claim is qualitative and informed by the crossing pattern plus the two inspected frame pairs; a full quantitative mechanism decomposition for over-dens variance is deferred to Day 13 contingency.
- The DECISIONS 20 structural-gap claim (source (c) missing) is validated for the frozen recipe but not for over-dens. On over-dens, multiple sources are live and DECISIONS 25 does not claim (c) dominates — only that it contributes and is worth naming in any future pre-commit of this kind.

All three claims are consistent with the V1 / V1.5 variance-as-contribution framing: name the mechanism to the strength the data supports, name the uncertainty where it doesn't, and let the reader judge.

---

## 26. Day 11 dense-init experiment — adaptive escalation, per-recipe thresholds, pre-committed writeup templates

**Decision (pre-commit form, written before any Day 11 compute):** Day 11's COLMAP-dense-init experiment runs seed=42 first on both recipes (frozen at dense init, default-densification at dense init), and escalates to seeds {123, 7} *per recipe independently* if the observed |Δ| against the matched-seed-42 sparse-init anchor falls inside that recipe's pre-committed escalation band. The anchors are Day 10 sweep results at seed=42 — frozen 22.56 dB, over-dens 18.90 dB — not DECISIONS 21's single-seed-from-Day-9 numbers and not DECISIONS 25's median-of-medians. The comparison is matched-seed sparse-init vs dense-init; the regime delta is the quantity of interest.

**Anchor rationale.** The matched-seed-42 choice holds image-order variance constant between the sparse-init anchor and the dense-init experimental run, isolating the regime effect from image-order permutation (DECISIONS 25's dominant variance source for frozen). Median-of-medians would confound the regime effect with cross-seed averaging of image-order permutations; DECISIONS 21's single-seed numbers would additionally confound with the pre-`image_order_seed` code path (before DECISIONS 25's `modal_app.py` addition of explicit image-order RNG control). Seed=42 matched-seed is the unique choice that leaves only the init-density variable free.

**What this is NOT a decision about:**
- Whether dense-init is the right Day 11 experiment. Locked by the post-Day-9 plan; this entry is about *how* to run it, not *whether*.
- The specific COLMAP dense-MVS threshold settings used to produce the dense init. Per the plan, "reasonable defaults, not a tuned choice"; the dense init point count is whatever PatchMatch produces at `modal_app.py`'s current settings on `scan9_v49_s123_3d428b`'s sparse model (V1-era cached value: ~257k points; may vary run-to-run per source (d) non-determinism). The post-Day-9 plan's "30–50k" target was an approximate scope marker rather than a binding range; the actual count is logged in the preflight file as observed metadata.
- The specific COLMAP dense-MVS seed or random-state choice used to produce dense init A. The seed is logged in the preflight but not pre-committed; if source (d) verification passes, the specific seed's identity is irrelevant to the escalation rule and becomes reproducibility metadata, not a decision variable.
- Whether V2 gets built. Trigger condition unchanged from the post-Day-9 plan §V2.1.
- Day 12 cross-method seed strategy. Day 12 reuses Day 10 multi-seed artifacts for the 3DGS rows; the COLMAP MVS row's seed treatment is a separate decision at Day 12 launch.

**Variance sources enumerated before launch** (per `feedback_rng_stream_enumeration.md`, applied as a first-class operational rule, not a reasoning hint):

| Source | Frozen dense-init | Over-dens dense-init |
|---|---|---|
| (a) gsplat split/duplicate sampling | structurally inactive (frozen) | live |
| (b) rasterizer atomic-add ordering | live, ~0.03 dB floor (Day 10 P2 Run A vs sweep_s42) | live, magnitude not isolated |
| (c) per-step image sampler `np.random.default_rng(image_order_seed)` | live, ~1.5 dB on sparse init (DECISIONS 25 P2) | live, magnitude not isolated |
| (d) COLMAP dense-MVS internal stochasticity (newly active for Day 11; not present in Day 10's sparse-init experiments) | known stochasticity at the CUDA-thread-scheduling level per DECISIONS 16; magnitude at n=33 in the dense-init regime unknown; pre-launch verification quantifies (hard stop if magnitude exceeds 0.5 dB floor) | same source, same verification (run on frozen only); failure invalidates over-dens escalation rule too |

**Source (d) pre-launch verification.** Source (d) is the only new variance source the dense-init experiment introduces, and it sits *upstream* of the entire escalation rule — if COLMAP dense-MVS produces inits whose downstream PSNR varies by an amount comparable to the 2.0/4.0 dB escalation thresholds, the rule cannot isolate a regime effect from COLMAP noise. Operational threshold: **if repeated COLMAP dense-MVS runs at fixed seed produce inits whose downstream 3DGS frozen-recipe PSNR at seed=42 differs by more than 0.5 dB, source (d) invalidates the Day 11 escalation rule and the rule needs revision before launch.** The 0.5 dB operational floor is chosen to leave the frozen escalation threshold (2.0 dB) at least 4× headroom above source (d) noise — the same "noise floor must be substantially below threshold" discipline DECISIONS 24's verification gate applied. The frozen recipe is the binding constraint; over-dens's 4.0 dB threshold has 8× headroom at the same floor.

Six-step verification sequence:

1. COLMAP dense-MVS run #1 at fixed seed → dense init A
2. COLMAP dense-MVS run #2 at the same fixed seed → dense init B
3. Frozen 3DGS at seed=42 on dense init A → PSNR_A
4. Frozen 3DGS at seed=42 on dense init B → PSNR_B
5. |PSNR_A − PSNR_B| < 0.5 dB → source (d) is below operational floor; proceed with escalation rule; set dense init A as the experimental init; the specific COLMAP seed becomes reproducibility metadata (per scope item above)
6. |PSNR_A − PSNR_B| ≥ 0.5 dB → STOP. Source (d) is live and meaningfully large. Day 11 design needs revision: either fix to a multi-COLMAP-seed × multi-3DGS-seed factorial (substantially more expensive), or characterize source (d) as a third variance dimension and report dense-init results with explicit (d)-magnitude framing. This is a hard gate — no fallback to "fold it into the writeup later," because (d) being uncharacterized would propagate through both recipes' escalation decisions and produce a falsified DECISIONS 26 of exactly the kind DECISIONS 25 just documented.

Verification is run on frozen only, not over-dens. Frozen isolates source (d) cleanly — no densification dynamics to confound the PSNR comparison. Running verification on over-dens would mix source (d) with source (a) split/duplicate sampling variance, defeating the purpose of measuring (d) in isolation. The frozen verification result is treated as bounding source (d) across both recipes.

Verification cost: ~$0.15 (one extra COLMAP dense-MVS run + one extra 3DGS frozen run at seed=42), ~0.5 hour wall-clock. Folds into the Day 11 baseline cost summary below.

V1 documented in DECISIONS 16 that GPU PatchMatch is non-deterministic at the CUDA-thread-scheduling level, with two n=30 runs at identical seeds producing fused point counts differing by ~100× (24,322 vs 178) in the failure regime. Day 11 runs PatchMatch at n=33, three views above V1's measured n=30 stress point. The verification is therefore not insurance against a low-probability failure — it is a load-bearing measurement of a known-non-zero stochasticity source at a view count V1 did not directly characterize. Prior on the n=33 magnitude: plausibly ranges from "much smaller than at n=30 because we are above the stress cliff" to "still severe because n=33 is close to n=30." The 0.5 dB gate discriminates between these regimes — if the gate does not fire, the dense-init experiment proceeds with a single fixed init; if it fires, Day 11's design needs revision per the hard-stop branch above before any further compute lands.

**Per-recipe escalation rule.**

**Decisive-band semantics.** Decisive means the single-seed observation is the conclusion. No discretionary escalation — if Δ lands in the decisive band, the sweep does not run additional seeds even if the result is surprising. The Δ > +2.0 dB frozen branch is the case where the temptation is strongest (the result is unexpected and "let me confirm with more seeds" sounds like good practice); the pre-commit explicitly forbids it. Surprise goes to the writeup, not to additional runs.

**Frozen recipe at dense init.**
- Anchor: Day 10 sweep, seed=42, sparse-init frozen = 22.56 dB
- Decisive band: |Δ_frozen| > 2.0 dB
  - Δ > +2.0 dB: dense init *helps* frozen. Unexpected — the frozen recipe is capacity-fixed at N=9044 by mechanism design, so a positive Δ would mean dense init is teaching the optimizer something the sparse init couldn't, and DECISIONS 21's mechanism story needs revision. Single-seed sufficient; write up the surprise.
  - Δ < −2.0 dB: dense init *hurts* frozen. Consistent with "the bigger init handed the optimizer more wrong points to render at fixed capacity." Single-seed sufficient.
  - −2.0 dB ≤ Δ ≤ +2.0 dB: escalate to seeds {123, 7}.
- Threshold rationale: 2.0 dB is above Day 10's frozen sparse-init seed range (1.53 dB) by 0.47 dB. Sufficient margin that a single-seed delta exceeding it cannot be confused for image-order permutation noise at the resolution Day 10 characterized.

**Default-densification recipe at dense init.**
- Anchor: Day 10 sweep, seed=42, sparse-init over-dens = 18.90 dB
- Decisive band: |Δ_overdens| > 4.0 dB
  - Δ > +4.0 dB: dense init *recovers* default densification. Headline narrows to "default densification is sparse-init-broken specifically; dense init makes it work." Strong narrative shift. Single-seed sufficient.
  - Δ < −4.0 dB: dense init makes default densification *even worse*. Strengthens headline ("densification miscalibration extends across init densities"). Single-seed sufficient.
  - −4.0 dB ≤ Δ ≤ +4.0 dB: escalate to seeds {123, 7}.
- Threshold rationale: 4.0 dB is above Day 10's over-dens sparse-init seed range (3.19 dB) by 0.81 dB. Tighter relative margin than frozen's because over-dens's variance floor itself is wider; tightening the threshold further would risk escalating in cases where the actual regime effect is real but small.

**No joint sign-consistency clause.** Each recipe is evaluated against its own threshold independently. A frozen-positive / over-dens-negative outcome (or vice versa) is a *finding* (asymmetric init-density dependence; see outcome 2 below), not a reason to escalate.

**Threshold honesty.** The 2.0 / 4.0 thresholds are pre-committed against Day 10's sparse-init seed ranges. They do NOT account for dense-init seed ranges, which are a known unknown. If a recipe escalates and its dense-init range turns out to be wider than its sparse-init range, the escalation produces an interpretable multi-seed distribution to report — the threshold's job is to detect when single-seed is insufficient, not to land in band. This is the structural difference from DECISIONS 23: 23 pre-committed a *target band* that observation falsified; 26 pre-commits a *trigger* that observation either fires or doesn't, with the writeup well-defined in both cases.

**Pre-committed writeup templates** (skeletons; numerical blanks `{value}` filled in separately at observation time, narrative blanks composed at observation time; no on-the-fly arithmetic — derived fields are recorded as their own blanks):

**Outcome 1 — Both decisive, same-sign Δs** (both recipes' |Δ|s exceed their thresholds; both Δs share a sign).

> "Dense init at ~{N_dense}k points {helps/hurts} both recipes. Dense-init frozen PSNR: {value} dB (Δ_frozen = {value} dB). Dense-init over-dens PSNR: {value} dB (Δ_overdens = {value} dB). Frozen-over-dens advantage at dense init: {value} dB. Compare sparse-init advantage: 3.66 dB. {Headline-strengthening framing if frozen advantage preserved or grown — extend DECISIONS 21's grow_grad2d=2e-4 calibration claim from sparse-init artifact to a broader regime statement. Headline-narrowing framing if frozen advantage shrunk but did not reverse — partial regime-dependence with the boundary above ~{N_dense}k.}"

The frozen-over-dens advantage is a derived field — recorded as its own blank, not computed inside the template. The person filling the template enters all four PSNR/Δ values from the run JSONs, then enters the advantage as a fifth separate value computed in a Python REPL or by hand and double-checked. This separates observation (the four primary blanks) from arithmetic (the fifth derived blank).

**Outcome 2 — Both decisive, opposite-sign Δs (asymmetric init-density dependence).**

> "Dense init bounds the headline asymmetrically. Frozen Δ = {value} dB ({sign}); over-dens Δ = {value} dB ({opposite sign}). {Whichever recipe benefits} responds to dense init while {the other} does not. Headline narrows to: 'frozen wins on PSNR at sparse init; the densification-vs-frozen ordering is init-density-dependent,' with the regime boundary identified at the observed crossover. Stronger finding than outcome 1 because it identifies a falsifiable bound where the headline ordering ceases to apply, rather than a robustness claim."

**Outcome 3 — One recipe decisive, one escalated.**

> "{Decisive recipe} dense-init Δ = {value} dB ({sign}), outside the {2.0/4.0} dB single-seed escalation band; reportable from one seed. {Escalated recipe} seed=42 Δ = {value} dB inside its band; multi-seed extension to {123, 7} added; observed median Δ = {value} dB across 3 seeds, seed-to-seed range {value} dB. Combined finding: {composed at observation time once both recipes have defensible characterizations.}"

Classification into the post-escalation analog of outcome 1 or outcome 2 is mechanical, not a judgment call: combine the decisive recipe's single-seed Δ with the escalated recipe's median Δ and apply the same same-sign vs opposite-sign criterion that distinguishes outcome 1 from outcome 2 in the both-decisive case. The classification is a derived field, recorded as such alongside the observed values.

**Outcome 4 — Both escalated.**

> "Single-seed seed=42 was insufficient on either recipe (frozen Δ = {value}, over-dens Δ = {value}, both inside their escalation bands). Both recipes extended to seeds {42, 123, 7}; observed frozen median Δ = {value} dB with seed range {value} dB; over-dens median Δ = {value} dB with seed range {value} dB. {Bounded headline language composed at observation time.}"

Classification into the post-escalation analog of outcome 1 or outcome 2 is mechanical: compare frozen-median Δ and over-dens-median Δ against the same-sign vs opposite-sign criterion. Derived field, recorded as such.

**Cost summary.**

| Outcome | Approximate cost | Wall-clock |
|---|---|---|
| Source (d) verification fires the gate (hard stop, design revision needed) | ~$0.15 (verification only); redesign is a separate decision triggered by the verification result; plausible magnitude is multi-seed × multi-init factorial (~$2–4) or dense-init abandoned entirely ($0) | stop and reconsider |
| Verification passes + both decisive (no escalation) | ~$0.20 | ~0.5 day |
| Verification passes + one recipe escalates | ~$0.35 | ~0.75 day |
| Verification passes + both recipes escalate | ~$0.50 | ~1 day |

Ceiling ~$0.50 *conditional on the source (d) verification gate not firing* (~3.3× the post-Day-9 plan's single-seed $0.15 estimate). If the gate fires, the dense-init experiment design needs revision and the additional cost is not pre-committed in this entry — that is a separate decision triggered by the verification result, not a row of this table. The bottom three rows of the table are reachable only after the verification passes; a reader who reads them as the bounded cost surface should understand they are conditional on the first row not firing. Expected value across the verification-passes branches lands somewhere between $0.20 and $0.50, weighted by which outcome the data produces. Each escalation absorbs the per-recipe discipline tax from Day 10's pattern (verification + smoke before launching the multi-seed extension). The 3.3× multiplier on Day 11 is *lower* than Day 10's ~5× discipline tax (Day 10 = verification + reproducibility check + P2 diagnostic on top of the naive sweep), reflecting that verification is smaller relative to Day 11's main experiment and that escalation is conditional rather than unconditional. Methodology machinery maturing per-decision, not paying a fixed tax.

**Amendment 1 (post-Step-9, 2026-04-14): Ceiling recalibration pre-escalation.**

Triggered by Step 9 empirical cost landing at ~$0.12 (395.2s wall-clock) against an implicit projection of ~$0.06 in the original cost table. Over-dens × seed=42 × init A at 7000 iters grew Gaussians from 257,687 to 1,545,441 across densify iters 500–5000, and the densify-stop tail (iters 5000–7000) runs at substantially higher per-iter cost than the frozen-recipe probe measured. The original cost table extrapolated both recipes from a single calibration probe at frozen N=257k (~13 ms/iter, weakly N-dependent), and that Y estimate is a lower bound on over-dens per-iter, not a point estimate. The original table under-projected over-dens by ~2×.

Recalibrated cost table (Amendment 1):

| Outcome | Original | Amended | Reason |
|---|---|---|---|
| Source (d) gate fires (hard stop) | ~$0.15 | ~$0.15 | Unchanged — gate fires before over-dens runs, so recalibration does not apply |
| Verification passes + both decisive (no escalation) | ~$0.20 | ~$0.35 | Verification legs + Step 9 at correct over-dens cost (~$0.12 vs projected ~$0.06) |
| Verification passes + one recipe escalates | ~$0.35 | ~$0.55 | Above + 2 additional recipe-specific seeds at correct cost (frozen escalation ~$0.12, over-dens escalation ~$0.24) |
| Verification passes + both recipes escalate | ~$0.50 | ~$0.70 | Above + the other recipe's 2 seeds |

Ceiling: ~$0.70 in the worst case (down from originally-projected $0.50 being a factor-of-√2 off from reality, not off by an order of magnitude — the structural shape of the cost table is correct, the per-recipe scaling was wrong).

**Budget pressure is methodological, not financial.** V1.5 total at the ~$0.70 Day 11 ceiling would land at ~$2.76 of the $50 V1.5 cap — still ~$47 of headroom. The amendment exists to preserve the discipline of "amend pre-commits before exceeding them, not after," which is the DECISIONS 25 pattern whose violation produced the DECISIONS 23 anchoring failure. If this amendment were deferred until after escalation compute landed, the Day 11 entry would read "ceiling $0.50, actual $0.70, 40% overrun" — structurally identical to DECISIONS 23's pre-commit-vs-observation mismatch. The amendment preserves pre-commit → observation comparison honesty.

**Calibration miss lesson (for future dense-init cost projections):** probe-based per-iter estimates are recipe-specific. A frozen-recipe probe measures held-capacity per-iter at fixed N and cannot be applied directly to densifying recipes where N grows during training. Future dense-init cost projections should calibrate both recipes separately (one probe at the frozen target N, one probe at the over-dens target with a short densify window) before committing a ceiling to a pre-commit. The fix is structural — use recipe-matched calibration probes — not "pad the ceiling by 2× to cover uncertainty." Padding without calibration would reintroduce the DECISIONS 20 structural-gap pattern at the cost-estimation level.

This amendment does NOT change: escalation thresholds, decisive bands, outcome templates, source (d) gate, or any methodology contract. It changes only the projected cost ceiling and records the lesson about recipe-specific calibration.

**Alternatives considered and rejected:**

- *Unconditional single-seed (brainstorming option A):* knowingly repeats the DECISIONS 23 failure pattern at the same statistical resolution Day 10 just falsified.
- *Unconditional multi-seed at {42, 123, 7} (brainstorming option B):* overpays when the dense-init effect is decisive at single-seed. Day 10's 4.7× cost overrun is the relevant counterfactual; pre-paying for multi-seed + verification on a likely-decisive experiment compounds the discipline tax.
- *Joint sign-consistency escalation rule (the original (C) from brainstorming):* collapses asymmetric findings (outcome 2) into "ambiguous, escalate" and overpays in the most informative case.
- *Bayesian posterior-threshold escalation:* over-engineered, no prior runs to fit a posterior on, harder to pre-commit and verify than a simple per-recipe threshold rule.

**Connection to the methodology meta-pattern (DECISIONS 20 → 23 → 24 → 25 → 26):**

DECISIONS 20 named the wrong variance sources (structural gap). DECISIONS 23 anchored on a single point estimate (quantitative gap). DECISIONS 24 pre-committed mechanism + verification gate (success). DECISIONS 25 documented both gaps and added the RNG-enumeration generalization. DECISIONS 26 attempts to get all three right at once: variance enumeration before launch (the variance sources table + source (d) verification above), conditional escalation thresholds rather than fixed bands (the per-recipe escalation rule), and pre-committed writeup templates so observation-time writeup is mechanical not creative-under-pressure (outcomes 1–4 above). Whether DECISIONS 26 succeeds is decided post-observation when the actual outcome lands and the templates either fit cleanly or get modified — which itself becomes the next entry in the sequence.

**Self-catch during consolidation (taxonomy addendum).** The initial consolidated draft of DECISIONS 26 framed source (d) with a determinism prior. Drafting the preflight surfaced contradictory evidence already in the repo: DECISIONS 16 documents GPU PatchMatch as non-deterministic at the CUDA-thread-scheduling level, with two n=30 runs at identical seed producing 24,322 vs 178 fused points. The entry was amended before commit; edits scoped to the source (d) framing (table cell + verification-paragraph rewrite) and not the structural pieces of the rule. This adds a *third* pre-commit failure mode to the DECISIONS 25 taxonomy: alongside the *structural* gap (Gap 2, missing source (c)) and the *quantitative* gap (Gap 1, anchored success bands), there is now a *retrieval* gap — reasoning from memory about prior repo findings without grepping the file. The catch mechanism is forcing memory into concrete commands; the preflight is itself a pre-commit verification step, not just operational scaffolding. See `feedback_rng_stream_enumeration.md` for the analogous RNG-enumeration self-catch.

**Honest scope of what this decision pre-commits:**

- Escalation thresholds (2.0 / 4.0 dB) are pre-committed against Day 10's observed sparse-init seed ranges. They do NOT account for dense-init seed ranges (the unknown the experiment is characterizing). The threshold's job is detection of insufficient single-seed evidence, not landing in a target band.
- Writeup templates fix the *structure* (which outcome, which clauses, which numbers) before observation. The *language* inside narrative blanks is composed at observation time. The pre-commit covers the mechanical part only; observation-time judgment is required for the narrative, and derived fields (frozen-over-dens advantage, outcome-3/4 classification) are recorded as their own blanks rather than computed in-template.
- Variance source enumeration is best-effort. Sources (a)–(c) come directly from Day 10; source (d) is a known stochasticity source from V1 (DECISIONS 16) whose magnitude at n=33 is gated on the verification above. If a fifth source surfaces during verification or the seed=42 smoke, DECISIONS 26 needs revisiting *before* escalation, not after.
- Source (d) verification covers COLMAP dense-MVS *PSNR-impact magnitude at n=33* only. It does not cover whether the specific COLMAP threshold settings produce any particular point count — n_points is observed, not pre-committed (see scope correction in the "What this is NOT" section above).
- This decision does not pre-commit what happens if `rerun_dense_mvs` cannot reliably produce a usable dense init (PatchMatch crashes mid-run, fusion produces fewer than ~1000 points, or both verification runs degenerate to near-empty outputs). The fallback per the post-Day-9 plan's spirit: document the attempt in the preflight + writeup, ship Day 9 headline with explicit "dense-init bounding not achieved, framework preserved for future work" caveat. That's a Day 11 execution contingency, not a DECISIONS 26 contingency.

## 27. Day 11 Step 9 N_final band override — documented, not retroactively rewritten

**Event (2026-04-14).** Step 9 (over-dens × seed=42 × init A, 7000 iters, dense init at N=257,687) ran successfully to completion and produced an over-dens PSNR median of 15.354 dB. All pre-committed Phase 2 acceptance bands cleared EXCEPT N_final: the observed value 1,545,441 fell outside the pre-committed band [800,000, 1,300,000] by +19% above the upper bound. Preflight Section 5 Phase 2 language was ambiguous about whether band trip constitutes "failure" halting escalation, so the execution halted at Step 9 and reported to the user for direction.

**Decision.** Override the N_final band trip and proceed to escalation (over-dens only; frozen resolved decisive at Δ ≈ −4.05 dB per DECISIONS 26). The override is documented as a first-class methodology event in this DECISIONS entry rather than laundered as a retroactive band widening in the preflight. The distinction matters: retroactive widening corrupts the pre-commit record by making the band match observations post-hoc, obscuring the fact that a band trip occurred. Documented override preserves the pre-commit exactly as written and records the override event + rationale + lessons as first-class methodology.

**Override rationale (all four must hold for the override to be valid):**

1. **Band scope was known-limited at authoring time.** The preflight's Phase 2 language was explicit: *"N final in [800k, 1300k] (over_dens densification produces ~1M Gaussians on Day 10; on a denser init the count may differ but should land in this order of magnitude)."* The band was calibrated against Day 10 sparse-init over-dens's 1,093,077 Gaussians with no empirical anchor for dense-init over-dens. The authors (me) flagged the uncertainty explicitly but committed to the tight numeric band anyway. 1,545,441 is inside "the same order of magnitude as ~1M" by any reasonable reading; the specific tight numeric band was a pre-commit coverage gap, not a well-founded methodology contract.

2. **All other Phase 2 bands cleared.** PSNR median 15.35 ∈ [10, 30] ✓. SSIM median 0.862 ∈ [0.5, 0.95] ✓. LPIPS median 0.142 ∈ [0.05, 0.5] ✓. Wall-clock 395.2s < 900s ✓. success=True ✓. The single band trip was the tight N_final guess, not a methodology violation across the full acceptance surface.

3. **Growth-self-regulation is a substantive mechanism finding that explains the trip non-pathologically.** Day 10 sparse over-dens grew 9,044 → 1,093,077 (121× growth). Step 9 dense over-dens grew 257,687 → 1,545,441 (6.0× growth). Final N ratio: 1.55M / 1.09M = 1.42× despite 28.5× higher starting N. Dense init substantially self-regulates default densification at scan9 scale — the densification machinery doesn't multiply proportionally from a larger base; it grows to approximately the same total magnitude regardless of starting point. This is a real mechanism observation about gsplat default-densification dynamics, not a sign of pathological behavior. The 1.55M end-N is inside "growth-self-regulated dense-init over-dens" regime, which is a new regime the preflight's band-at-authoring-time did not anticipate.

4. **The PSNR value is plausible and informative.** Δ_overdens = 15.354 − 18.90 = −3.55 dB. This is inside DECISIONS 26's over-dens escalation band [-4.0, +4.0] dB, triggering escalation per the pre-commit rule. The value is not a degenerate output; it is a substantive measurement that feeds outcome 3 classification (frozen decisive, over-dens escalated).

**What the override does NOT establish.**

- It does NOT set a precedent that "N_final bands are always overridable." Each future band trip is evaluated on its own merits against the four criteria above. The criteria are load-bearing: known-limited scope, other bands clear, substantive mechanism explanation, plausible observation. If any of the four fails, the default is stop-and-redesign, not override.
- It does NOT rewrite the preflight's Phase 2 band to [800k, 1,600k] or similar. The band stays as written.
- It does NOT change DECISIONS 26's methodology contract. Thresholds, outcome templates, source (d) gate, and escalation rule are all unchanged.

**What the override DOES establish.**

- A worked example of the override pattern for future reference. When a pre-commit band is tripped AND the four criteria hold, the override mechanism is: (a) write a DECISIONS entry with the four-criterion analysis, (b) record the lesson about the gap, (c) proceed with the decision the override authorizes. The four criteria are the template.
- A methodology data point for Section 6.5's scenario-coverage gap catalogue (below).
- Authorization to proceed with Step 11 escalation using the Step 9 over-dens PSNR as the seed=42 anchor for the over-dens recipe.

**Scenario-coverage gap — fourth pre-commit failure mode for the DECISIONS 25 taxonomy.**

The DECISIONS 25 taxonomy previously catalogued three pre-commit failure modes:
  1. **Structural gap** (Gap 2 in DECISIONS 25): missing variance source in enumeration. Example: DECISIONS 20 missed source (c) image-order RNG.
  2. **Quantitative gap** (Gap 1 in DECISIONS 25): anchoring on a single-point-estimate target band that observation falsifies. Example: DECISIONS 23's "0.04 dB target" anchoring.
  3. **Retrieval gap** (DECISIONS 26 self-catch addendum): reasoning from memory about prior repo findings without grepping the file. Example: DECISIONS 26's original source (d) determinism prior (corrected pre-commit).

Day 11 Step 9 surfaced a fourth distinct failure mode:

  4. **Scenario-coverage gap**: pre-committed enumeration of outcomes had interpretive gaps even though threshold logic was complete. Example: preflight Section 6.5's O2 scenario framing ("over-dens inside escalation band") assumed sub-cases (a) "close to anchor" or (b) "slightly positive," but the actual observation landed in sub-case (c) "distinctly negative but not decisive" — a sub-case the enumeration missed despite the |Δ| < 4.0 dB threshold logic being complete and correct.

**Distinguishing scenario-coverage gaps from the other three:**

- **Not a structural gap.** Structural gaps miss an entire causal factor (a variance source, a confound, a branch of the decision tree). Scenario-coverage gaps miss interpretive sub-cases WITHIN an already-enumerated branch — the decision tree is complete, the interpretive enumeration inside a branch is incomplete.
- **Not a quantitative gap.** Quantitative gaps anchor on a target numeric that observation falsifies. Scenario-coverage gaps do not have target numerics at all; they have narrative framings for decision-tree branches, and the gap is in the narrative coverage, not the numeric calibration.
- **Not a retrieval gap.** Retrieval gaps miss concrete evidence that was in the repo at authoring time. Scenario-coverage gaps miss a sub-case in the space of plausible observations — there may have been no concrete repo evidence at authoring time to catch it; the miss is in the author's enumeration of plausible-given-priors.

**Where scenario-coverage gaps come from.** The author enumerates outcome framings based on their directional prior about where observations will land. The prior is often implicit and skews the enumeration toward "likely" sub-cases, leaving "unlikely but still-inside-threshold" sub-cases uncovered. In Section 6.5's O2 case, my implicit prior was that over-dens at dense init would either be close to zero (rescue) or slightly positive (partial rescue) — I was not seriously considering "distinctly negative but not decisive" because I was anchored on the "dense init is roughly neutral for over-dens" hypothesis. The threshold logic was complete; the narrative enumeration was not.

**Catch mechanism for scenario-coverage gaps.** For each outcome branch in the decision tree, list the maximum and minimum plausible observations compatible with the branch. If the narrative framings cover only part of that range, enumerate the uncovered sub-ranges explicitly even if they feel unlikely. For O2 (|Δ_overdens| ∈ [−4, +4] dB), the range is a 8 dB window. My enumeration covered ~[−2, +2] dB and left the tails uncovered. Explicit enumeration would have caught this before observation.

**Connection to methodology meta-pattern.** The four-failure-mode taxonomy is:

  | Failure mode | What it misses | Example |
  |---|---|---|
  | Structural gap | a causal factor or decision branch | DECISIONS 20 → source (c) |
  | Quantitative gap | calibration of a target numeric | DECISIONS 23 → 0.04 dB anchor |
  | Retrieval gap | concrete repo evidence at authoring time | DECISIONS 26 → source (d) determinism |
  | Scenario-coverage gap | interpretive sub-cases within an enumerated branch | DECISIONS 27 → O2 sub-case (c) |

Each failure mode has a distinct catch mechanism:
  - Structural: enumerate all causal factors mechanically (e.g., grep every RNG stream)
  - Quantitative: avoid target numerics in pre-commits; use trigger thresholds instead
  - Retrieval: force memory into concrete commands (grep, read, measure)
  - Scenario-coverage: enumerate the max and min plausible observations for each decision-tree branch, cover the full range not just the prior-weighted center

These are complementary, not substitutes. A pre-commit that passes all four is DECISIONS-25-robust by the current-as-of-Day-11 taxonomy. A fifth failure mode may surface in future work; the taxonomy is additive.

**Honest scope of this decision.**

- This entry authorizes proceeding to Step 11 escalation with the Step 9 over-dens PSNR as the anchor. It does not authorize anything beyond that.
- The four-criteria override mechanism is a template for future band trips, not a general-purpose escape hatch for pre-commit friction. Future overrides must show all four criteria hold and must be documented as separate DECISIONS entries.
- The scenario-coverage gap taxonomy expansion applies retroactively to the DECISIONS 25 catalogue as a fourth failure mode. It does not invalidate or supersede any prior entry; it adds a classification dimension.
- This entry does not pre-commit how future dense-init experiments should calibrate N_final bands. Recipe-specific calibration probes (queued as a Day 12 prerequisite for timing; extendable to N_final calibration) are the operational mitigation, but the specific probe design is not pre-committed here.

**Precedent value.** If future work runs into pre-commit friction from a tight acceptance band on an uncalibrated metric, the pattern is: document the band trip, evaluate the four criteria, write the DECISIONS entry, and proceed (or stop, if any criterion fails). The override mechanism exists because pre-commit discipline cannot be *so* strict that any band trip halts progress regardless of context — that would create a perverse incentive to write intentionally loose bands. Strict bands + documented override is a better equilibrium than loose bands without review, because the override forces explicit analysis of the trip rather than ignoring it.
