# Pre-commit authoring failure modes — V1.5 taxonomy

Over the V1 → V1.5 progression, the repo catalogued five distinct
pre-commit authoring failure modes. Each entry has a DECISIONS
reference and a git-traceable motivating example; each emerged from
a specific failure-caught-in-progress event, not from a priori
design. The taxonomy is additive — new modes get appended as new
evidence accumulates.

This file is the methodology artifact of V1.5. The top-level
[README](../README.md) has the two-paragraph headline; the
[V1.5 experimental writeup](V1.5.md) has the Day 9 → Day 12 detail
that surfaced each of these modes in sequence. [DECISIONS.md](../DECISIONS.md)
entries 20 through 30 are the per-mode canonical records.

## The five modes

| # | Gap | Catalogued in | Motivating example | Catch mechanism |
|---|---|---|---|---|
| 1 | **Quantitative** | DECISIONS 23 → 25 | DECISIONS 23 pre-committed the frozen-recipe PSNR success band as 22.3 – 23.0 dB (±0.4 dB of a single-seed anchor at 22.62). The 3-seed Day 10 sweep produced median 21.32 dB with range 1.53 dB — outside the band, because the ≲1 dB noise assumption was not grounded in any prior data for this regime. | Use trigger thresholds, not target numerics |
| 2 | **Structural** | DECISIONS 20 → 25 | DECISIONS 20's variance-source enumeration named sources (a) densification sampling and (b) rasterizer atomic-add ordering. It missed source (c) per-step image-order RNG entirely. Day 10's P2 diagnostic showed image-order accounts for ~1.5 dB of the 1.53 dB 3-seed frozen range — the entire observed variance. | Grep every RNG stream mechanically before pre-committing variance sources |
| 3 | **Retrieval** | DECISIONS 26 addendum → DECISIONS 30 (promoted) | Four instances in the V1.5 chain with consistent mechanism; the fourth is the first post-commit instance (Day 12 Finding 1/2 rediscovery of V1's seed=123 best cell, caught during Day 13 re-read). See retrieval-gap section below and [DECISIONS.md](../DECISIONS.md) entry 30 for the full inventory and the promotion rationale. | Force memory into concrete commands (grep/ls/read/measure); re-read passes are load-bearing post-commit |
| 4 | **Scenario-coverage** | DECISIONS 27 | Day 11 preflight Section 6.5's O2 scenario framing ("over-dens inside the [−4, +4] dB escalation band") enumerated sub-cases (a) "close to anchor" and (b) "slightly positive" but missed sub-case (c) "distinctly negative but not decisive." Step 9 observation at Δ = −3.55 dB landed in (c). Threshold logic was complete; interpretive enumeration was not. | Enumerate max and min plausible observations per branch, not just the prior-weighted center |
| 5 | **Single-seed generalization** | DECISIONS 28 | Day 11 pre-committed "uniform degradation across all 4 views" for over-dens × dense init based on seed=42 alone (per-view [12.85, 16.46, 14.48, 16.23]). Step 11 escalation at seeds {123, 7} revealed view 001's 3-seed range was 7.60 dB (12.85 / 18.39 / 20.45) — a single-seed low-tail draw, not a recipe-level effect. Content framing committed before escalation data landed. | Distinguish structural framings (safe to pre-commit before escalation) from content framings (unsafe); replace content with "[pending escalation data]" placeholders |

## Retrieval gap — the most-recurring mode

Retrieval gap has four cataloged instances across the V1.5 chain with
consistent mechanism: an author reasons from memory about a concrete
repo fact when a `grep`/`ls`/`read`/`measure` would resolve it at
lower cost and higher reliability.

1. **Source (d) determinism prior** (DECISIONS 26 consolidation,
   pre-commit). The initial DECISIONS 26 draft framed GPU PatchMatch
   as deterministic at fixed seed; DECISIONS 16 records 24,322 vs 178
   fused points at n=30 under identical seed — explicit
   non-determinism. Caught during drafting, before commit.
2. **Workspace-contract scaffolding for `rerun_dense_mvs`** (Day 11
   Smoke A iterations, pre-commit). The initial mirror-cached-tree
   strategy enumerated specific files from memory and missed layers
   that pycolmap's workspace contract requires (`patch-match.cfg`,
   `fusion.cfg`, helper scripts). Caught during smoke iterations,
   fixed by switching to a walk-and-copy traversal of the actual
   cached tree.
3. **MVS view count label n=33 vs n=37** (Day 12 preflight,
   pre-commit, commit `812929e`). The initial Day 12 pre-commit
   labeled MVS as running at n=33 (the 3DGS train view count); the
   actual PatchMatch view count was n=37 (`modal volume ls` → 37 .png
   files in the undistorted images directory, `patch-match.cfg` → 37
   entries). Caught by running the ls during the preflight
   verification before Day 12 compute landed.
4. **Day 12 Finding 1/2 rediscovery of V1's seed=123 best cell**
   (commit `b4e2c7d`, **post-commit**). The Day 12 writeup at commit
   `e8c6846` framed Day 11's 7.35 mm chamfer as "2.6× tighter than
   V1's baseline" (Finding 1) and V1's SfM-seed chamfer variance as
   a new methodology finding (Finding 2). Both were rediscoveries.
   `results/stress_view_count/summary.json` already reported the
   full 3-seed n=49 chamfer distribution (median 18.31, range 11.69,
   per-seed {42: 18.31, 123: 7.35, 7: 19.04}), and V1's writeup
   explicitly names the seed=123 7.35 mm best cell. Day 11's rerun
   reproduces V1's seed=123 best cell, which V1 had already measured
   and surfaced. Caught during Day 13's polish re-read pass — the
   first post-commit instance in the chain.

**Why the post-commit character matters.** The first three instances
share a structure: the author reached into memory for a concrete
fact, the memory was stale or wrong, and the catch mechanism fired
before the incorrect claim landed in git. The fourth instance is
different: the incorrect claim *did* land in git and stayed there
until Day 13's re-read caught it. This demonstrates two things the
first three instances did not:

1. **Post-commit damage is reachable under the existing discipline.**
   The retrieval-gap catch mechanism works when the author thinks to
   use it or when a reviewer notices the mismatch. It does not
   automatically fire on every committed artifact. If neither the
   author nor the reviewer has independently prompted the check, a
   retrieval-gap rediscovery can land in git and stay there until a
   later pass catches it.
2. **Re-read is a necessary catch mechanism post-commit, not a
   nice-to-have.** Day 13's polish re-read was planned as an
   editorial pass (surface the 5-gap taxonomy, audit honest-scope,
   budget line). It was not planned as a factual re-verification
   pass. The fourth retrieval-gap instance was caught anyway because
   the author re-read enough of the repo to find the mismatch.
   Absent the re-read, the writeup would have shipped with the
   rediscovery framings intact.

Neither observation introduced a new operational rule. The right
posture is the one implicit in the workflow: catch it where you
catch it, log the recurrence, and update the taxonomy as evidence
accumulates. The fourth instance was the promotion trigger for
[DECISIONS 30](../DECISIONS.md) — retrieval gap moved from an
addendum of DECISIONS 26 to a standalone entry on the strength of
the four instances including the post-commit one, with re-read
passes added to the informal catch-mechanism inventory.

## Candidate sixth gap — stated-uncertainty flattening (deferred)

Two Day 12 instances share a consistent mechanism: a known
quantity's uncertainty is correctly recorded at measurement time,
then flattened to a point estimate in downstream reasoning. Neither
has produced downstream damage.

1. **Y-refinement miss** (Day 11 → Day 12). Day 11's calibration
   probe recorded Y ∈ [3, 23] ms/iter with 75% relative uncertainty,
   but the point estimate (13.4 ms/iter) was then used in subsequent
   cost projections as if it were the distribution's center. Day
   12's n=1000 probe measured Y = 24.1 ± 1.4 ms/iter, at the upper
   edge of Day 11's envelope. No downstream damage: ceilings held,
   budget held, narrative unchanged.
2. **Open3D wall-clock miss** (Day 12 task #3 preflight). Estimated
   ~30s local CPU for the MVS evaluation; actual was 577s. Estimated
   from a small-PLY prior, without propagating the uncertainty
   through the actual target (Open3D chamfer against a 3.3M-point GT
   where KD-tree NN-search dominates). Off by ~20×. No downstream
   damage.

Both instances share: known quantity, stated-or-inferable
uncertainty, flattening to a point estimate in downstream reasoning,
zero consequence. Formal cataloguing as a sixth taxonomy slot is
deferred under the repo's taxonomy-inflation guard — a candidate
gap with no motivating example bearing downstream damage is weaker
than the existing five, and promotion would overpay for the
visibility. Trigger conditions for promotion: (a) a third instance
with consistent mechanism, or (b) a first instance with downstream
damage. Neither has materialized. This note is the holding record;
a future session either adds a third instance and promotes, or lets
the candidate age out.

## What the taxonomy is and isn't

The five gaps (plus the candidate sixth) are failure modes of
*pre-commit authoring discipline* — mistakes an author can make when
writing a plan, a pre-commit, or a writeup under methodology
pressure. They are not failure modes of experimental code, the eval
pipeline, or the domain methods being compared.

A reviewer skeptical of the methodology thesis can verify the
taxonomy by following the commit refs and DECISIONS entries backward
to the motivating events. Each catch is git-legible: the commit that
caught the failure is visible in `git log`, the prior commit showing
the incorrect claim is visible in the diff, and the DECISIONS entry
that catalogued the failure mode is visible in
[DECISIONS.md](../DECISIONS.md). The commit chain is the
methodology artifact; this file is the index.

## Recommended reading order

For an interviewer with 10–30 minutes:

1. This file (taxonomy inventory + retrieval-gap detail) — 5 min
2. [DECISIONS.md](../DECISIONS.md) entries 20, 23, 25, 26 addendum,
   27, 28, 30 (one per mode) — 15 min
3. [V1.5 writeup](V1.5.md), specifically the Day 11 per-view
   correction and the Day 12 transparent rediscovery correction —
   10 min

For a hiring reviewer with 30–90 seconds: stay on the top-level
README; this file is for deeper engagement.
