# Roadmap: next steps and V2 motivation

What V1 and V1.5 did not cover, and what a V2 should look like if it
gets built. V2 is explicitly conditional — see the trigger condition
at the end of this file. Until that trigger fires, the repo stays at
V1.5.

The top-level [README](../README.md) has a one-paragraph roadmap
pointer. The [V1.5 writeup](V1.5.md) has a "What V1.5 did not
characterize" section that complements this roadmap at the
experimental level.

## Near-term extensions (scan9-scoped)

- **Multiple DTU scenes.** The unreliability floor between n=15 and
  n=8 may be scan9-specific (house model, medium-frequency
  architectural texture). Running the same sweep on DTU scans with
  different texture and geometry characteristics — smooth objects,
  high-frequency organic shapes, reflective surfaces — would tell
  us whether the floor is a property of the pipeline or of this
  scene.
- **PatchMatch parameter sweep.** Does tuning `min_num_pixels`,
  `window_radius`, or `filter_min_ncc` move the floor? A cheap way
  to find out whether the bimodal failure is fundamental to default
  PatchMatch or an artifact of the defaults.
- **Conformal calibration of reconstruction quality.** Given the
  variance, a calibrated "at 95% confidence, n=X views will produce
  ≥Y fused points" bound would be more useful to practitioners than
  a point estimate. This is the direct analogue of sim-to-data's
  conformal thresholds, applied to geometry.

## V1.5 is complete

3DGS comparison was the main V1.5 extension. See the
[V1.5 writeup](V1.5.md) for the full experimental chain:

- **Day 9 four-row ablation** showed working densification is
  net-negative for PSNR on scan9's sparse init + 33-view regime,
  with the literature's "uncontrolled densification" framing not
  directly tested by published methods.
- **Day 10 multi-seed sweep** extended to 3-seed reporting and
  confirmed the frozen advantage holds (median 21.32 dB vs 18.50
  dB, seed-to-seed ranges 1.53 dB frozen / 3.20 dB over-dens).
- **Day 11 dense-init bounding experiment** (~257k points from
  COLMAP PatchMatch) showed the frozen-over-dens PSNR advantage
  narrows from 3.66 dB to 2.34 dB but preserves sign, so the
  headline holds at both tested init densities.
- **Day 12 cross-method three-regime comparison** reports COLMAP
  MVS on geometric metrics and 3DGS on novel-view synthesis
  metrics with dashes in non-applicable cells per option (c) in
  DECISIONS 29 Amendment 2 (no forced apples-to-apples).

## V2 — a calibrated gate for silent failure modes

V1 + V1.5 together identified three distinct *silent* failure modes
— each of which produces plausible-looking aggregate numbers while
being fundamentally broken, and each of which V1/V1.5 caught by
multi-seed discipline rather than aggregate-metric inspection:

- **V1 bimodality at n ≤ 15.** SfM initialization either succeeds or
  collapses to a near-degenerate reconstruction; median point counts
  hide the distribution's bimodal character. Caught by 3-seed
  reporting at each view count.
- **Gsplat densification miscalibration at 9k init** (DECISIONS 21).
  Default `grow_grad2d = 2e-4` is calibrated for ~100k–200k SfM
  points; at 9k it fires on 82% of Gaussians per refinement event,
  producing over-parameterized reconstructions with sub-pixel noise
  that hurts PSNR while preserving perceptual metrics. Caught by
  the four-row ablation with a frozen baseline.
- **View-level seed instability under dense-init over-dens** (Day
  11, DECISIONS 28). View `rect_001` at over-dens × dense init
  shows a 7.60 dB range across 3 seeds — an order of magnitude
  above Day 10's sparse-init seed ranges. Caught by Day 11's
  escalation to 3 seeds after the seed=42 single-seed framing was
  falsified.

V2's goal is a calibrated gate that turns each of these silent
failures into a visible, actionable signal at reconstruction time —
without requiring a full 3-seed multi-run to surface it. The gate's
design is not pre-committed here; the motivation is that three
silent modes is enough evidence that visible-failure-at-runtime is a
load-bearing property for a practitioner downstream of this
pipeline.

## V2 trigger condition

V2 is explicitly *not* being built speculatively. The trigger for
starting V2 work is an interview conversation that specifically asks
for calibrated uncertainty quantification on reconstructions — not a
generic "more rigor" request. Until that trigger fires, the
methodology discipline V1.5 demonstrates (pre-commit templates,
escalation rules, the [five-gap failure-mode
taxonomy](methodology.md), the commit-chain-as-methodology thesis)
is the artifact; V2 stays unbuilt.

The default state is don't build, send applications. V2 is a
conditional extension, not a planned next step.
