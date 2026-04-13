"""V1.5 Day 10 multi-seed gsplat sweep.

Two recipes × three seeds = six runs launched sequentially on Modal.
Recipes reproduce DECISIONS 21 rows 1 and 2:

- "frozen": L1-only loss, no densification events. Reproduces Day 8's
  silent-bug frozen state via a clean mechanism — see DECISIONS 24 for
  why densify_start_iter = densify_stop_iter = reset_opacity_iter = 999999
  substitutes for the packed=False IndexError path.
- "over_dens": L1 + 0.2·(1 - SSIM). gsplat DefaultStrategy defaults.
  Reproduces the ~1M-Gaussian over-densified regime of DECISIONS 21 row 2.

Both recipes use random_bkgd=False (matches the implicit posture of the
four-row table — random_bkgd was not in the codebase at ablation time,
see DECISIONS 22 audit note).

Seeds {42, 123, 7} match V1's stress sweep (DECISIONS 16).

Results:
- Per-run JSON at results/gsplat/multiseed/{recipe}_s{seed}.json
- Per-run renders at results/gsplat/multiseed/{recipe}_s{seed}/renders/
- Sweep aggregate at results/gsplat/multiseed/sweep_summary.json

Modes:
- default: all recipes × all seeds at full n_iterations
- --smoke: 3 iters × all recipes × single seed (42), for launch-surface check
- --verify-frozen: full n_iters × frozen × seed=42 only, for DECISIONS 24 gate
- --dry-run: print launch plan, exit without calling Modal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import modal

DEFAULT_COLMAP_RUN_ID = "scan9_v49_s123_3d428b"
DEFAULT_SEEDS = (42, 123, 7)

# Pre-committed recipe kwargs per DECISIONS 23 and 24.
# Passed literally into train_gsplat.remote so the launch is self-documenting.
# Changing a parameter here is a recipe change and should land as a new
# DECISIONS entry, not a silent edit.
RECIPES: dict[str, dict] = {
    "frozen": dict(
        # L1-only loss, matches Day 8 and DECISIONS 21 row 1.
        ssim_lambda=0.0,
        # Push all refinement-event triggers past n_iterations so the
        # strategy is functionally inert for the whole run. See DECISIONS 24
        # for why this substitutes cleanly for Day 8's silent packed=False
        # IndexError: final state (N=9044) identical, mechanism differs.
        densify_start_iter=999999,
        densify_stop_iter=999999,
        reset_opacity_iter=999999,
        # grow_grad2d is unused (refinement never fires) but kept explicit
        # so a reader of the launch config sees the full recipe surface.
        densify_grad_threshold=2e-4,
        # Matches the implicit posture of the four-row table (random_bkgd
        # wasn't in the codebase when the ablation ran; see DECISIONS 22).
        random_bkgd=False,
    ),
    "over_dens": dict(
        # L1 + 0.2*(1 - SSIM), matches DECISIONS 21 row 2.
        ssim_lambda=0.2,
        # gsplat DefaultStrategy defaults: refinement events fire every
        # 100 steps between step 500 and 5000, opacity reset at step 3000.
        densify_start_iter=500,
        densify_stop_iter=5000,
        reset_opacity_iter=3000,
        densify_grad_threshold=2e-4,
        random_bkgd=False,
    ),
}


@dataclass(frozen=True)
class RunSpec:
    recipe: str
    seed: int
    n_iterations: int
    colmap_run_id: str

    @property
    def slug(self) -> str:
        return f"{self.recipe}_s{self.seed}"


def _download_renders(
    workspace_volume: modal.Volume,
    remote_dir: str,
    test_view_names: list[str],
    local_dir: Path,
) -> None:
    """Download rendered held-out PNGs from the Modal workspace volume.

    Copied from experiments/run_gsplat.py to keep this script standalone.
    Each (recipe, seed) run writes to the same Modal output path because
    train_gsplat's out_run_id depends only on (colmap_run_id, seed). We
    pull renders locally after each run so the next run's overwrite
    doesn't erase prior artifacts.
    """
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


def _launch_one(
    run_spec: RunSpec,
    output_dir: Path,
    download_renders: bool,
    workspace_volume: modal.Volume | None,
) -> dict:
    """Launch a single (recipe, seed) Modal run and save its result JSON."""
    train_gsplat = modal.Function.from_name("simtorecon-mvs", "train_gsplat")
    kwargs = dict(RECIPES[run_spec.recipe])
    kwargs.update(
        colmap_run_id=run_spec.colmap_run_id,
        seed=run_spec.seed,
        n_iterations=run_spec.n_iterations,
        test_every=10,
    )
    print(f"[multiseed] launching {run_spec.slug}  n_iter={run_spec.n_iterations}")
    # Pretty-print the recipe kwargs — this is the self-documenting launch surface.
    for k, v in kwargs.items():
        print(f"[multiseed]   {k} = {v}")

    result = train_gsplat.remote(**kwargs)

    slug_dir = output_dir / run_spec.slug
    slug_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_spec.slug}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if not result.get("success"):
        print(f"[multiseed] {run_spec.slug} FAILED: {result.get('error')}")
        return result

    psnr = result.get("psnr_median")
    ssim = result.get("ssim_median")
    lpips = result.get("lpips_median")
    n_gs = result.get("n_gaussians_final")
    elapsed = result.get("elapsed_seconds")
    psnr_str = f"{psnr:.2f} dB" if psnr is not None else "n/a"
    ssim_str = f"{ssim:.3f}" if ssim is not None else "n/a"
    lpips_str = f"{lpips:.3f}" if lpips is not None else "n/a"
    print(
        f"[multiseed] {run_spec.slug} done  "
        f"PSNR={psnr_str}  SSIM={ssim_str}  LPIPS={lpips_str}  "
        f"N={n_gs}  elapsed={elapsed:.1f}s"
    )

    if download_renders and workspace_volume is not None:
        local_renders = slug_dir / "renders"
        print(
            f"[multiseed] downloading {result['n_test_views']} renders "
            f"→ {local_renders}"
        )
        _download_renders(
            workspace_volume,
            result["renders_dir"],
            result["test_view_names"],
            local_renders,
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1.5 Day 10 multi-seed gsplat sweep"
    )
    parser.add_argument(
        "--recipes",
        default=",".join(RECIPES.keys()),
        help=f"Comma-separated recipe names. Known: {','.join(RECIPES.keys())}",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (V1 convention: 42,123,7)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=7000,
        help="Number of gsplat training iterations per run",
    )
    parser.add_argument(
        "--colmap-run-id",
        default=DEFAULT_COLMAP_RUN_ID,
        help=f"Sparse init run_id on simtorecon-workspace (default: {DEFAULT_COLMAP_RUN_ID})",
    )
    parser.add_argument(
        "--output",
        default="results/gsplat/multiseed",
        help="Local output directory for result JSONs and renders",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 3 iterations × first seed × all recipes",
    )
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help="Verification mode: full iterations × frozen × seed=42 only (DECISIONS 24 gate)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip per-run render download (saves bandwidth; metrics still saved)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan and exit without calling Modal",
    )
    args = parser.parse_args()

    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            raise SystemExit(
                f"unknown recipe: {r}  (known: {list(RECIPES.keys())})"
            )

    seeds = [int(s) for s in args.seeds.split(",")]

    iterations = args.iterations
    if args.smoke:
        # 3 iters, first seed only, both recipes — catches launch-surface bugs
        # before committing to a full sweep. Per feedback_sequence_fix_then_sweep.md.
        seeds = seeds[:1]
        iterations = 3

    if args.verify_frozen:
        # DECISIONS 24 gate: full iterations on frozen × seed=42 only.
        # PSNR must land within ±0.5 dB of DECISIONS 21 row 1's 22.62 before
        # the full 6-run sweep is launched.
        recipes = ["frozen"]
        seeds = [42]

    specs: list[RunSpec] = [
        RunSpec(
            recipe=r,
            seed=s,
            n_iterations=iterations,
            colmap_run_id=args.colmap_run_id,
        )
        for r in recipes
        for s in seeds
    ]

    output_dir = Path(args.output)
    print(f"[multiseed] {len(specs)} run(s) planned (iterations={iterations}):")
    for spec in specs:
        print(f"  {spec.slug}  init={spec.colmap_run_id}")

    if args.dry_run:
        print("[multiseed] --dry-run set, exiting without launch")
        return

    workspace_volume: modal.Volume | None = None
    if not args.skip_download:
        workspace_volume = modal.Volume.from_name("simtorecon-workspace")

    results: list[dict] = []
    for spec in specs:
        results.append(
            _launch_one(
                spec,
                output_dir,
                download_renders=not args.skip_download,
                workspace_volume=workspace_volume,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "recipes": recipes,
                "seeds": seeds,
                "n_iterations": iterations,
                "colmap_run_id": args.colmap_run_id,
                "results": [
                    {
                        "recipe": spec.recipe,
                        "seed": spec.seed,
                        "slug": spec.slug,
                        "success": r.get("success"),
                        "psnr_median": r.get("psnr_median"),
                        "psnr_range": r.get("psnr_range"),
                        "ssim_median": r.get("ssim_median"),
                        "ssim_range": r.get("ssim_range"),
                        "lpips_median": r.get("lpips_median"),
                        "lpips_range": r.get("lpips_range"),
                        "n_gaussians_final": r.get("n_gaussians_final"),
                        "elapsed_seconds": r.get("elapsed_seconds"),
                        "error": r.get("error"),
                    }
                    for spec, r in zip(specs, results)
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[multiseed] sweep summary written to {summary_path}")


if __name__ == "__main__":
    main()
