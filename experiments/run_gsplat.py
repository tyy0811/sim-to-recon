"""Run gsplat training on a V1 COLMAP reconstruction via Modal GPU.

V1.5 Day 8: single-seed smoke run against V1's best baseline
(n=49, seed=123, run_id=scan9_v49_s123_3d428b, 257k fused points).

Day 9 will extend this to 3 seeds with median+range reporting to match
the variance discipline of V1's shipped view-count sweep.

Default seed is 42, NOT 123, to avoid confounding the gsplat training
randomness with the COLMAP seed that produced the sparse model. If the
COLMAP init came from seed=123 and gsplat also uses seed=123, any
downstream correlation analysis can't separate "initialization bias"
from "training-loop noise".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal  # noqa: I001

DEFAULT_COLMAP_RUN_ID = "scan9_v49_s123_3d428b"  # V1 best baseline (README: 257k pts)


def _resolve_colmap_run_id(
    override: str | None,
    views_seed_path: Path,
) -> str:
    """Pick the COLMAP run_id to initialize gsplat from.

    Order of preference:
    1. --colmap-run-id CLI override
    2. views_049/seed_123/result.json["run_id"] (V1's best shipped baseline)
    3. Hardcoded DEFAULT_COLMAP_RUN_ID fallback
    """
    if override:
        return override

    if views_seed_path.exists():
        with open(views_seed_path) as f:
            data = json.load(f)
        run_id = data.get("run_id")
        if run_id:
            return run_id

    return DEFAULT_COLMAP_RUN_ID


def _download_renders(
    workspace_volume: modal.Volume,
    remote_dir: str,
    test_view_names: list[str],
    local_dir: Path,
) -> None:
    """Download rendered held-out PNGs from the Modal workspace volume."""
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in test_view_names:
        remote_path = f"{remote_dir}/{name}"
        local_path = local_dir / name
        try:
            with open(local_path, "wb") as f:
                for chunk in workspace_volume.read_file(remote_path):
                    f.write(chunk)
        except Exception as e:
            print(f"  WARN: failed to download {remote_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gsplat training on Modal")
    parser.add_argument(
        "--colmap-run-id",
        default=None,
        help=(
            "COLMAP run_id on the simtorecon-workspace volume. "
            "Defaults to V1's best shipped baseline (n=49 seed=123)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="gsplat training seed")
    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
        help="Number of gsplat training iterations",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="Hold out every k-th image for novel-view evaluation",
    )
    parser.add_argument(
        "--output",
        default="results/gsplat",
        help="Local output directory",
    )
    parser.add_argument(
        "--v1-sweep-dir",
        default="results/stress_view_count/views_049/seed_123/result.json",
        help="Path to V1 seed=123 n=49 result.json (used to auto-detect run_id)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading rendered PNGs (saves bandwidth)",
    )
    args = parser.parse_args()

    colmap_run_id = _resolve_colmap_run_id(args.colmap_run_id, Path(args.v1_sweep_dir))
    print(f"[run_gsplat] colmap_run_id = {colmap_run_id}")
    print(f"[run_gsplat] seed = {args.seed}")
    print(f"[run_gsplat] iterations = {args.iterations}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Invoke the deployed Modal function
    train_gsplat = modal.Function.from_name("simtorecon-mvs", "train_gsplat")
    print("[run_gsplat] dispatching to Modal...")

    result = train_gsplat.remote(
        colmap_run_id=colmap_run_id,
        seed=args.seed,
        n_iterations=args.iterations,
        test_every=args.test_every,
    )

    if not result.get("success"):
        print(f"[run_gsplat] FAILED: {result.get('error')}")
        fail_path = output_dir / f"baseline_scan9_s{args.seed}_FAILED.json"
        with open(fail_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[run_gsplat] failure details written to {fail_path}")
        raise SystemExit(1)

    print(
        f"[run_gsplat] done in {result['elapsed_seconds']:.1f}s  "
        f"final N={result['n_gaussians_final']}"
    )
    print(
        f"[run_gsplat] test views: {result['n_test_views']}  "
        f"train views: {result['n_train_views']}"
    )
    if result.get("psnr_median") is not None:
        print(
            f"[run_gsplat] PSNR  median={result['psnr_median']:.2f} dB  "
            f"range={result['psnr_range']}"
        )
    if result.get("ssim_median") is not None:
        print(
            f"[run_gsplat] SSIM  median={result['ssim_median']:.3f}  "
            f"range={result['ssim_range']}"
        )
    if result.get("lpips_median") is not None:
        print(
            f"[run_gsplat] LPIPS median={result['lpips_median']:.3f}  "
            f"range={result['lpips_range']}"
        )

    # --- Save local JSON result ---
    result_path = output_dir / f"baseline_scan9_s{args.seed}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[run_gsplat] result written to {result_path}")

    # --- Download rendered held-out PNGs ---
    if args.skip_download:
        return

    workspace_volume = modal.Volume.from_name("simtorecon-workspace")
    local_renders = output_dir / result["run_id"] / "renders"
    print(f"[run_gsplat] downloading {result['n_test_views']} renders to {local_renders}...")
    _download_renders(
        workspace_volume,
        result["renders_dir"],
        result["test_view_names"],
        local_renders,
    )
    print(f"[run_gsplat] renders saved under {local_renders}")


if __name__ == "__main__":
    main()
