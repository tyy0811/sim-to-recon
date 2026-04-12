"""Run view count stress sweep on DTU scan9 via Modal GPU.

Multi-seed approach (Decision 15): COLMAP PatchMatch on GPU is non-deterministic
due to CUDA thread scheduling. We run 3 independent reconstructions per view count
with different seeds, report median metrics and range. SfM initialization is
controlled via pycolmap.set_random_seed per run.
"""

import argparse
import json
from pathlib import Path

import modal
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="View count stress sweep")
    parser.add_argument("--stress-config", default="configs/stress/view_count.yaml")
    parser.add_argument("--output", default="results/stress_view_count")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--seeds", default="42,123,7",
                        help="Comma-separated seeds for multi-seed runs")
    args = parser.parse_args()

    with open(args.stress_config) as f:
        stress_cfg = yaml.safe_load(f)

    view_counts = stress_cfg["stress"]["view_counts"]
    seeds = [int(s) for s in args.seeds.split(",")]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download GT once for evaluation
    gt_ply = output_dir / "gt.ply"
    gt = None
    try:
        import open3d as o3d

        gt_volume = modal.Volume.from_name("simtorecon-dtu-data")
        if not gt_ply.exists():
            print("Downloading ground truth point cloud...")
            with open(gt_ply, "wb") as f:
                for chunk in gt_volume.read_file("scan9/gt/stl009_total.ply"):
                    f.write(chunk)
        gt = o3d.io.read_point_cloud(str(gt_ply))
        print(f"GT loaded: {len(gt.points)} points")
    except Exception as e:
        print(f"GT not available, skipping evaluation: {e}")

    reconstruct = modal.Function.from_name("simtorecon-mvs", "reconstruct_dtu_scan9")
    workspace_volume = modal.Volume.from_name("simtorecon-workspace")

    # Run all (view_count, seed) combinations
    all_runs = []

    for n_views in sorted(view_counts, reverse=True):
        for seed in seeds:
            run_dir = output_dir / f"views_{n_views:03d}" / f"seed_{seed}"
            cache_file = run_dir / "result.json"

            # Resume from cache
            if cache_file.exists():
                with open(cache_file) as f:
                    cached = json.load(f)
                all_runs.append(cached)
                print(f"[cached] n={n_views} seed={seed}: "
                      f"{cached['n_points']} pts")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[running] n={n_views} seed={seed}...")
            try:
                result = reconstruct.remote(
                    n_views=n_views,
                    seed=seed,
                    target_width=args.width,
                    target_height=args.height,
                )

                if not result["success"]:
                    print(f"[FAILED] n={n_views} seed={seed}: {result['error']}")
                    result_data = {
                        "scene_name": "dtu_scan9",
                        "n_views": result.get("n_views", n_views),
                        "seed": seed,
                        "n_registered": result.get("n_registered", 0),
                        "n_sparse_points": result.get("n_sparse_points", 0),
                        "n_points": 0,
                        "elapsed_seconds": result.get("elapsed_seconds", 0),
                        "f_score_1mm": None,
                        "f_score_5mm": None,
                        "f_score_10mm": None,
                        "chamfer": None,
                        "accuracy": None,
                        "completeness": None,
                        "icp_fitness": None,
                        "error": result.get("error", "unknown"),
                    }
                    all_runs.append(result_data)
                    with open(cache_file, "w") as f:
                        json.dump(result_data, f, indent=2)
                    continue

                # Download PLY
                output_ply = run_dir / "dense.ply"
                with open(output_ply, "wb") as f:
                    for chunk in workspace_volume.read_file(result["fused_ply_path"]):
                        f.write(chunk)

                result_data = {
                    "scene_name": "dtu_scan9",
                    "n_views": result["n_views"],
                    "seed": seed,
                    "n_registered": result.get("n_registered", None),
                    "n_sparse_points": result.get("n_sparse_points", None),
                    "n_points": result["n_points"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "output_ply": str(output_ply),
                    "run_id": result["run_id"],
                }

                # Evaluate against GT
                if gt is not None:
                    import open3d as o3d

                    from simtorecon.evaluation.alignment import align_to_gt
                    from simtorecon.evaluation.failure import per_point_error
                    from simtorecon.evaluation.metrics import (
                        accuracy,
                        chamfer_distance,
                        completeness,
                        f_score,
                    )

                    pred = o3d.io.read_point_cloud(str(output_ply))
                    aligned, icp_transform, icp_fitness = align_to_gt(
                        pred, gt, threshold=20.0, max_iterations=200,
                    )

                    result_data["chamfer"] = chamfer_distance(aligned, gt)
                    result_data["accuracy"] = accuracy(aligned, gt)
                    result_data["completeness"] = completeness(aligned, gt)
                    result_data["f_score_1mm"] = f_score(aligned, gt, threshold=1.0)
                    result_data["f_score_5mm"] = f_score(aligned, gt, threshold=5.0)
                    result_data["f_score_10mm"] = f_score(aligned, gt, threshold=10.0)
                    result_data["icp_fitness"] = icp_fitness
                    result_data["icp_transform"] = icp_transform.tolist()

                    # Save per-point errors for later analysis
                    errors = per_point_error(aligned, gt)
                    np.savez(run_dir / "per_point_errors.npz", errors=errors)

                all_runs.append(result_data)

                # Cache
                with open(cache_file, "w") as f:
                    json.dump(result_data, f, indent=2, default=str)

                metrics_str = ""
                if result_data.get("f_score_5mm") is not None:
                    metrics_str = (f" F@5mm={result_data['f_score_5mm']:.3f}"
                                   f" ch={result_data['chamfer']:.2f}mm"
                                   f" ICP={result_data['icp_fitness']:.3f}")
                print(f"[done] n={n_views} seed={seed}: "
                      f"{result['n_points']} pts, "
                      f"reg={result.get('n_registered', '?')}/{n_views}, "
                      f"{result['elapsed_seconds']:.0f}s{metrics_str}")

            except Exception as e:
                print(f"[FAILED] n={n_views} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                result_data = {
                    "scene_name": "dtu_scan9",
                    "n_views": n_views,
                    "seed": seed,
                    "n_points": 0,
                    "elapsed_seconds": 0,
                    "error": str(e),
                }
                all_runs.append(result_data)

    # Compute per-view-count summary (median ± range across seeds)
    # Group runs by the view count requested (from config), using seed to match
    summary = []
    for n_views in sorted(view_counts, reverse=True):
        seed_runs = [r for r in all_runs if r.get("seed") in seeds
                     and r.get("n_views") is not None
                     and r["n_views"] == n_views]

        # n_views in the result might differ from config if SfM used fewer
        # Fall back to matching by seed + approximate view count
        if not seed_runs:
            seed_runs = [r for r in all_runs if r.get("seed") in seeds
                         and r.get("n_views") is not None
                         and abs(r["n_views"] - n_views) <= 1]

        def med(key):
            vals = [r[key] for r in seed_runs if r.get(key) is not None]
            return float(np.median(vals)) if vals else None

        def rng(key):
            vals = [r[key] for r in seed_runs if r.get(key) is not None]
            return float(max(vals) - min(vals)) if len(vals) >= 2 else None

        entry = {
            "n_views": n_views,
            "n_seeds": len(seed_runs),
            "median_points": med("n_points"),
            "range_points": rng("n_points"),
            "median_chamfer": med("chamfer"),
            "range_chamfer": rng("chamfer"),
            "median_f_score_5mm": med("f_score_5mm"),
            "range_f_score_5mm": rng("f_score_5mm"),
            "median_f_score_10mm": med("f_score_10mm"),
            "range_f_score_10mm": rng("f_score_10mm"),
            "median_icp_fitness": med("icp_fitness"),
            "per_seed": [{
                "seed": r.get("seed"),
                "n_points": r.get("n_points"),
                "n_registered": r.get("n_registered"),
                "chamfer": r.get("chamfer"),
                "f_score_5mm": r.get("f_score_5mm"),
                "icp_fitness": r.get("icp_fitness"),
            } for r in seed_runs],
        }
        summary.append(entry)

    # Save all runs + summary
    with open(output_dir / "all_runs.json", "w") as f:
        json.dump(all_runs, f, indent=2, default=str)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print("View count stress sweep — median across 3 seeds")
    print(f"{'='*70}")
    print(f"{'N':>4s}  {'Points':>10s}  {'Chamfer':>12s}  {'F@5mm':>12s}  "
          f"{'F@10mm':>12s}  {'ICP':>6s}")
    print("-" * 70)
    for e in summary:
        def fmt(med_key, rng_key):
            m = e.get(med_key)
            r = e.get(rng_key)
            if m is None:
                return "N/A"
            if r is not None:
                return f"{m:.2f}±{r:.2f}"
            return f"{m:.2f}"

        pts = e.get("median_points")
        pts_r = e.get("range_points")
        pts_str = f"{pts:.0f}" if pts else "N/A"
        if pts_r:
            pts_str += f"±{pts_r:.0f}"

        print(f"{e['n_views']:4d}  {pts_str:>10s}  "
              f"{fmt('median_chamfer', 'range_chamfer'):>12s}  "
              f"{fmt('median_f_score_5mm', 'range_f_score_5mm'):>12s}  "
              f"{fmt('median_f_score_10mm', 'range_f_score_10mm'):>12s}  "
              f"{fmt('median_icp_fitness', None):>6s}")

    print(f"\nAll runs: {output_dir / 'all_runs.json'}")
    print(f"Summary:  {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
