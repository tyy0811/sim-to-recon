"""V1.5 Day 10 P2 diagnostic — image-order-variance isolation.

Single-purpose one-off script for the DECISIONS 23 band-violation
investigation. Runs frozen × seed=42 twice with varying image_order_seed
(42 and 123) to test whether the per-step training-image sampler drives
the 1.53 dB frozen seed-to-seed PSNR range observed in the Day 10 sweep.

See preflight/2026-04-14_day10_p2_diagnostic.txt for the pre-committed
three-band |ΔPSNR| criterion and the four stop conditions (a)–(d).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# Reuse the sweep script's RECIPES dict and default colmap run_id so
# any future recipe edit stays in sync without a separate source of
# truth. sys.path.insert keeps this script standalone-runnable.
sys.path.insert(0, str(Path(__file__).parent))
from run_gsplat_multiseed import DEFAULT_COLMAP_RUN_ID, RECIPES  # noqa: E402

# Pre-committed constants (preflight/2026-04-14_day10_p2_diagnostic.txt).
# SWEEP_FROZEN_S42_PSNR is from results/gsplat/multiseed/frozen_s42.json
# as written by the Day 10 sweep at commit acf2e1b.
SWEEP_FROZEN_S42_PSNR = 22.562543105867384
STOP_CONDITION_B_TOLERANCE = 0.15  # dB, Run A vs sweep-frozen_s42
BAND_ALPHA_THRESHOLD = 0.5  # |ΔPSNR| > 0.5 → image-order dominant
BAND_BETA_THRESHOLD = 0.2   # |ΔPSNR| < 0.2 → image-order not driver


def launch_one(image_order_seed: int, label: str, output_dir: Path) -> dict:
    """Launch a single frozen × seed=42 × given-image_order_seed run."""
    train_gsplat = modal.Function.from_name("simtorecon-mvs", "train_gsplat")
    kwargs = dict(RECIPES["frozen"])
    kwargs.update(
        colmap_run_id=DEFAULT_COLMAP_RUN_ID,
        seed=42,
        n_iterations=7000,
        test_every=10,
        image_order_seed=image_order_seed,
    )
    print(f"[p2] launching {label}  image_order_seed={image_order_seed}")
    for k, v in kwargs.items():
        print(f"[p2]   {k} = {v}")
    result = train_gsplat.remote(**kwargs)
    out_path = output_dir / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    if not result.get("success"):
        print(f"[p2] {label} FAILED: {result.get('error')}")
        return result
    print(
        f"[p2] {label} done  "
        f"PSNR={result['psnr_median']:.4f} dB  "
        f"SSIM={result['ssim_median']:.4f}  "
        f"LPIPS={result['lpips_median']:.4f}  "
        f"N={result['n_gaussians_final']}  "
        f"elapsed={result['elapsed_seconds']:.1f}s"
    )
    return result


def interpret_band(delta: float) -> str:
    """Pre-committed three-band mapping. No ad-hoc re-interpretation."""
    if delta > BAND_ALPHA_THRESHOLD:
        return "alpha (image-order DOMINANT driver)"
    if delta < BAND_BETA_THRESHOLD:
        return "beta (image-order NOT the driver)"
    return "gamma (image-order CONTRIBUTES, partial mechanism)"


def main() -> None:
    output_dir = Path("results/gsplat/multiseed/diag_p2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V1.5 Day 10 P2 diagnostic — image-order-variance isolation")
    print("=" * 60)
    print(f"Sweep-frozen_s42 reference: {SWEEP_FROZEN_S42_PSNR:.4f} dB")
    print(f"Stop condition (b): |Run A - sweep| > {STOP_CONDITION_B_TOLERANCE} dB")
    print(f"Band alpha: |ΔPSNR| > {BAND_ALPHA_THRESHOLD} dB (image-order dominant)")
    print(f"Band beta:  |ΔPSNR| < {BAND_BETA_THRESHOLD} dB (image-order not driver)")
    print(
        f"Band gamma: {BAND_BETA_THRESHOLD} <= |ΔPSNR| <= "
        f"{BAND_ALPHA_THRESHOLD} dB (partial)"
    )
    print()

    # --- Run A: image_order_seed=42 ---
    print("Run A: frozen × seed=42 × image_order_seed=42")
    result_a = launch_one(42, "run_A", output_dir)
    if not result_a.get("success"):
        print("[p2] STOP — Run A failed, do not retry, do not launch Run B")
        print("[p2] Per stop-condition (a): report failure, await direction.")
        raise SystemExit(1)

    psnr_a = float(result_a["psnr_median"])
    drift_from_sweep = abs(psnr_a - SWEEP_FROZEN_S42_PSNR)
    print(
        f"[p2] Run A PSNR median = {psnr_a:.4f} dB  "
        f"|Δ vs sweep-frozen_s42| = {drift_from_sweep:.4f} dB"
    )

    # --- Stop condition (b): Run A must match sweep-frozen_s42 ±0.15 dB ---
    if drift_from_sweep > STOP_CONDITION_B_TOLERANCE:
        print(
            f"[p2] STOP — stop-condition (b) triggered: "
            f"Run A drifted {drift_from_sweep:.4f} dB from sweep-frozen_s42 "
            f"(tolerance {STOP_CONDITION_B_TOLERANCE} dB)"
        )
        print(
            "[p2] The image_order_seed=None → seed fallback may not be "
            "bit-identical to pre-change code."
        )
        print(
            "[p2] Do NOT launch Run B. Do NOT retry. "
            "Report drift, investigate code change, await direction."
        )
        raise SystemExit(2)

    print("[p2] Run A passes stop-condition (b). Launching Run B.")
    print()

    # --- Run B: image_order_seed=123 ---
    print("Run B: frozen × seed=42 × image_order_seed=123")
    result_b = launch_one(123, "run_B", output_dir)
    if not result_b.get("success"):
        print("[p2] STOP — Run B failed after Run A passed.")
        print(
            "[p2] Per stop-condition (a) clarification: mid-diagnostic Modal "
            "failure is itself data."
        )
        print("[p2] Do NOT retry, do NOT launch a replacement Run B. Report.")
        raise SystemExit(3)

    psnr_b = float(result_b["psnr_median"])
    delta = abs(psnr_b - psnr_a)
    print(
        f"[p2] Run B PSNR median = {psnr_b:.4f} dB  "
        f"|ΔPSNR vs Run A| = {delta:.4f} dB"
    )
    print()

    # --- Boundary check (stop condition c) ---
    on_alpha_boundary = abs(delta - BAND_ALPHA_THRESHOLD) < 1e-6
    on_beta_boundary = abs(delta - BAND_BETA_THRESHOLD) < 1e-6
    if on_alpha_boundary or on_beta_boundary:
        boundary_value = (
            BAND_ALPHA_THRESHOLD if on_alpha_boundary else BAND_BETA_THRESHOLD
        )
        print("=" * 60)
        print(
            f"[p2] BOUNDARY CASE (stop condition c): |ΔPSNR| = {delta:.4f} dB "
            f"lands exactly on band boundary {boundary_value}."
        )
        print("[p2] Report both adjacent bands, defer to user.")
        print("=" * 60)
    else:
        band = interpret_band(delta)
        print("=" * 60)
        print(f"[p2] Band interpretation: {band}")
        print(f"[p2] Run A (image_order=42):  {psnr_a:.4f} dB")
        print(f"[p2] Run B (image_order=123): {psnr_b:.4f} dB")
        print(f"[p2] |ΔPSNR|                = {delta:.4f} dB")
        print("=" * 60)

    # --- Summary JSON (always written, even for boundary cases) ---
    summary = {
        "diagnostic": "p2_image_order_variance",
        "preflight_ref": "preflight/2026-04-14_day10_p2_diagnostic.txt",
        "sweep_frozen_s42_reference_psnr": SWEEP_FROZEN_S42_PSNR,
        "stop_condition_b_tolerance_db": STOP_CONDITION_B_TOLERANCE,
        "band_alpha_threshold_db": BAND_ALPHA_THRESHOLD,
        "band_beta_threshold_db": BAND_BETA_THRESHOLD,
        "run_A": {
            "image_order_seed": 42,
            "psnr_median": psnr_a,
            "ssim_median": float(result_a["ssim_median"]),
            "lpips_median": float(result_a["lpips_median"]),
            "n_gaussians_final": int(result_a["n_gaussians_final"]),
            "elapsed_seconds": float(result_a["elapsed_seconds"]),
            "drift_from_sweep_frozen_s42_db": drift_from_sweep,
        },
        "run_B": {
            "image_order_seed": 123,
            "psnr_median": psnr_b,
            "ssim_median": float(result_b["ssim_median"]),
            "lpips_median": float(result_b["lpips_median"]),
            "n_gaussians_final": int(result_b["n_gaussians_final"]),
            "elapsed_seconds": float(result_b["elapsed_seconds"]),
        },
        "delta_psnr_db": delta,
        "band_boundary_case": on_alpha_boundary or on_beta_boundary,
        "band_interpretation": (
            interpret_band(delta)
            if not (on_alpha_boundary or on_beta_boundary)
            else "BOUNDARY — defer to user"
        ),
    }
    summary_path = output_dir / "diag_p2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[p2] summary written to {summary_path}")


if __name__ == "__main__":
    main()
