"""Point cloud evaluation metrics: chamfer distance, accuracy, completeness, F-score.

All metrics follow the DTU MVS benchmark conventions.
"""

import numpy as np
import open3d as o3d


def _nearest_neighbor_distances(
    source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud
) -> np.ndarray:
    """Compute nearest-neighbor distances from source to target.

    Uses Open3D's C++ implementation for performance (handles 800k+ points).
    """
    if len(source.points) == 0:
        return np.array([])
    return np.asarray(source.compute_point_cloud_distance(target))


def chamfer_distance(
    pred: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud
) -> float:
    """Symmetric chamfer distance: mean of pred->gt and gt->pred NN distances."""
    if len(pred.points) == 0 or len(gt.points) == 0:
        return float("inf")
    d_pred_to_gt = _nearest_neighbor_distances(pred, gt)
    d_gt_to_pred = _nearest_neighbor_distances(gt, pred)
    return float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))


def accuracy(
    pred: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud
) -> float:
    """DTU accuracy: mean distance from predicted points to nearest GT point.

    Lower is better. Reports the mean distance (in mm for DTU).
    """
    if len(pred.points) == 0:
        return float("inf")
    distances = _nearest_neighbor_distances(pred, gt)
    return float(distances.mean())


def completeness(
    pred: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud
) -> float:
    """DTU completeness: mean distance from GT points to nearest predicted point.

    Lower is better. Measures how much of the GT surface is covered.
    """
    if len(pred.points) == 0:
        return float("inf")
    distances = _nearest_neighbor_distances(gt, pred)
    return float(distances.mean())


def f_score(
    pred: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud, threshold: float = 1.0
) -> float:
    """F-score: harmonic mean of precision and recall at a distance threshold.

    Precision: fraction of predicted points within threshold of GT.
    Recall: fraction of GT points within threshold of predicted.
    """
    if len(pred.points) == 0 or len(gt.points) == 0:
        return 0.0

    d_pred_to_gt = _nearest_neighbor_distances(pred, gt)
    d_gt_to_pred = _nearest_neighbor_distances(gt, pred)

    precision = float(np.mean(d_pred_to_gt < threshold))
    recall = float(np.mean(d_gt_to_pred < threshold))

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
