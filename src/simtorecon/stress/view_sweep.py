"""View count stress sweep: measure reconstruction quality vs number of input views."""

from __future__ import annotations

import json
from pathlib import Path

from simtorecon.data.dtu import DTUScene
from simtorecon.pipeline.colmap_runner import ColmapRunner
from simtorecon.pipeline.schemas import PipelineConfig, ReconstructionResult


def view_count_sweep(
    scene: DTUScene,
    view_counts: list[int],
    config: PipelineConfig,
    output_dir: Path,
    seed: int = 42,
) -> list[ReconstructionResult]:
    """Run reconstruction at each view count, caching results to disk.

    Args:
        scene: Full DTU scene (all views).
        view_counts: List of view counts to test (e.g., [49, 30, 15, 8]).
        config: Pipeline configuration.
        output_dir: Directory for per-run outputs.
        seed: Random seed for view subsampling.

    Returns:
        List of ReconstructionResult, one per view count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for n_views in sorted(view_counts, reverse=True):
        run_dir = output_dir / f"views_{n_views:03d}"
        cache_file = run_dir / "result.json"

        # Resume from cache if available
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            result = ReconstructionResult(**data)
            results.append(result)
            print(f"[cached] n_views={n_views}: {result.n_points} points, "
                  f"{result.elapsed_seconds:.1f}s")
            continue

        # Subsample views
        sub_scene = scene.subsample(n_views, seed=seed)
        print(f"[running] n_views={n_views} (subsampled from {scene.n_images})...")

        try:
            runner = ColmapRunner(sub_scene, config)
            result = runner.run(run_dir)

            # Evaluate against GT if available
            if scene.has_ground_truth() and result.n_points > 0:
                import open3d as o3d

                from simtorecon.evaluation.alignment import align_to_gt
                from simtorecon.evaluation.metrics import (
                    accuracy,
                    chamfer_distance,
                    completeness,
                    f_score,
                )

                pred = o3d.io.read_point_cloud(str(result.output_ply))
                gt = scene.get_ground_truth()
                aligned, _ = align_to_gt(pred, gt)

                result.chamfer = chamfer_distance(aligned, gt)
                result.accuracy = accuracy(aligned, gt)
                result.completeness = completeness(aligned, gt)
                result.f_score = f_score(aligned, gt, threshold=1.0)

            # Cache result
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

            metrics_str = ""
            if result.f_score is not None:
                metrics_str = f", F={result.f_score:.3f}, chamfer={result.chamfer:.3f}"
            print(f"[done] n_views={n_views}: {result.n_points} points, "
                  f"{result.elapsed_seconds:.1f}s{metrics_str}")

        except Exception as e:
            print(f"[FAILED] n_views={n_views}: {e}")
            result = ReconstructionResult(
                scene_name=scene.name,
                n_views=n_views,
                n_points=0,
                output_ply=run_dir / "dense.ply",
                elapsed_seconds=0.0,
                config_hash="",
            )

        results.append(result)

    return results
