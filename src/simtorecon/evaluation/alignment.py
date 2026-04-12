"""Point cloud alignment via scale estimation, centroid alignment, and ICP refinement."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d


def align_to_gt(
    pred: o3d.geometry.PointCloud,
    gt: o3d.geometry.PointCloud,
    max_iterations: int = 50,
    threshold: float = 5.0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    """Align a predicted point cloud to a ground-truth point cloud.

    The alignment pipeline:
      1. Scale estimation from bounding-box diagonal ratio.
      2. Centroid alignment (translation).
      3. Coarse ICP (point-to-point, large threshold).
      4. Fine ICP (point-to-point, caller-specified threshold).

    Args:
        pred: Predicted point cloud.
        gt: Ground-truth point cloud.
        max_iterations: Maximum ICP iterations for each stage.
        threshold: Distance threshold for the fine ICP pass.

    Returns:
        A tuple of (aligned_cloud, transform_4x4, icp_fitness) where
        transform_4x4 is the full 4x4 transformation matrix that maps pred
        to gt (including scale), and icp_fitness is the fine ICP fitness
        (fraction of inlier correspondences).
    """
    if len(pred.points) == 0 or len(gt.points) == 0:
        return copy.deepcopy(pred), np.eye(4), 0.0

    # --- Step 1: Scale estimation via bounding-box diagonal ratio ---
    pred_bb = pred.get_axis_aligned_bounding_box()
    gt_bb = gt.get_axis_aligned_bounding_box()

    pred_diag = np.linalg.norm(pred_bb.get_max_bound() - pred_bb.get_min_bound())
    gt_diag = np.linalg.norm(gt_bb.get_max_bound() - gt_bb.get_min_bound())

    if pred_diag < 1e-12:
        scale = 1.0
    else:
        scale = float(gt_diag / pred_diag)

    # Apply scale
    aligned = copy.deepcopy(pred)
    points_scaled = np.asarray(aligned.points) * scale
    aligned.points = o3d.utility.Vector3dVector(points_scaled)

    # Build scale matrix
    scale_mat = np.eye(4)
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale
    scale_mat[2, 2] = scale

    # --- Step 2: Centroid alignment ---
    pred_centroid = np.mean(np.asarray(aligned.points), axis=0)
    gt_centroid = np.mean(np.asarray(gt.points), axis=0)
    translation = gt_centroid - pred_centroid

    points_translated = np.asarray(aligned.points) + translation
    aligned.points = o3d.utility.Vector3dVector(points_translated)

    translate_mat = np.eye(4)
    translate_mat[:3, 3] = translation

    # --- Step 3: Coarse ICP ---
    coarse_threshold = threshold * 5.0
    coarse_result = o3d.pipelines.registration.registration_icp(
        aligned,
        gt,
        coarse_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    aligned.transform(coarse_result.transformation)

    # --- Step 4: Fine ICP ---
    fine_result = o3d.pipelines.registration.registration_icp(
        aligned,
        gt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    aligned.transform(fine_result.transformation)

    # --- Compose full transform: fine @ coarse @ translate @ scale ---
    transform_4x4 = (
        fine_result.transformation
        @ coarse_result.transformation
        @ translate_mat
        @ scale_mat
    )

    return aligned, transform_4x4, fine_result.fitness
