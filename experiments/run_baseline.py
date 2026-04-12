"""Run baseline dense MVS reconstruction on DTU scan9 via Modal GPU."""

import argparse
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description="Run baseline reconstruction")
    parser.add_argument("--n-views", type=int, default=49, help="Number of views")
    parser.add_argument("--output", default="results/baseline_scan9",
                        help="Output directory")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run reconstruction on Modal (calls the deployed function)
    reconstruct = modal.Function.from_name("simtorecon-mvs", "reconstruct_dtu_scan9")

    print(f"Running reconstruction on Modal (n_views={args.n_views})...")
    result = reconstruct.remote(
        n_views=args.n_views,
        target_width=args.width,
        target_height=args.height,
    )

    if not result["success"]:
        print(f"FAILED: {result['error']}")
        return

    print(f"Reconstruction complete: {result['n_points']} points, "
          f"{result['elapsed_seconds']:.1f}s")

    # Download the fused PLY from Modal volume
    output_ply = output_dir / "dense.ply"
    volume = modal.Volume.from_name("simtorecon-workspace")
    print(f"Downloading result to {output_ply}...")

    with open(output_ply, "wb") as f:
        for chunk in volume.read_file(result["fused_ply_path"]):
            f.write(chunk)

    # Evaluate against ground truth
    print("Evaluating against ground truth...")
    import open3d as o3d

    from simtorecon.evaluation.alignment import align_to_gt
    from simtorecon.evaluation.metrics import accuracy, chamfer_distance, completeness, f_score

    pred = o3d.io.read_point_cloud(str(output_ply))

    # Download GT from Modal volume too
    gt_ply = output_dir / "gt.ply"
    gt_volume = modal.Volume.from_name("simtorecon-dtu-data")
    try:
        with open(gt_ply, "wb") as f:
            for chunk in gt_volume.read_file("scan9/gt/stl009_total.ply"):
                f.write(chunk)
        gt = o3d.io.read_point_cloud(str(gt_ply))
        print(f"GT loaded: {len(gt.points)} points")

        aligned, transform, icp_fitness = align_to_gt(pred, gt, threshold=20.0, max_iterations=200)

        metrics = {
            "chamfer": chamfer_distance(aligned, gt),
            "accuracy": accuracy(aligned, gt),
            "completeness": completeness(aligned, gt),
            "f_score_1mm": f_score(aligned, gt, threshold=1.0),
            "f_score_2mm": f_score(aligned, gt, threshold=2.0),
            "f_score_5mm": f_score(aligned, gt, threshold=5.0),
        }
    except Exception as e:
        print(f"GT evaluation skipped: {e}")
        metrics = {}

    # Save result
    result_data = {
        "scene_name": "dtu_scan9",
        "n_views": result["n_views"],
        "n_points": result["n_points"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output_ply": str(output_ply),
        "run_id": result["run_id"],
        **metrics,
    }

    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)

    print("\nBaseline complete:")
    print(f"  Points:       {result['n_points']}")
    print(f"  Time:         {result['elapsed_seconds']:.1f}s")
    if metrics:
        print(f"  Chamfer:      {metrics['chamfer']:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Completeness: {metrics['completeness']:.4f}")
        print(f"  F-score@1mm:  {metrics['f_score_1mm']:.4f}")
        print(f"  F-score@2mm:  {metrics['f_score_2mm']:.4f}")
        print(f"  F-score@5mm:  {metrics['f_score_5mm']:.4f}")
    print(f"  Output: {output_ply}")


if __name__ == "__main__":
    main()
