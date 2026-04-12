"""Per-point failure analysis: identify and visualize reconstruction error regions."""

import numpy as np
import open3d as o3d

from simtorecon.evaluation.metrics import _nearest_neighbor_distances


def per_point_error(
    pred: o3d.geometry.PointCloud, gt: o3d.geometry.PointCloud
) -> np.ndarray:
    """Compute per-point distance from each predicted point to nearest GT point."""
    return _nearest_neighbor_distances(pred, gt)


def failure_regions(
    pred: o3d.geometry.PointCloud,
    gt: o3d.geometry.PointCloud,
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """Identify failure regions: points with error above the given percentile.

    Returns:
        mask: boolean array, True for failure points
        threshold: the error threshold at the given percentile
    """
    errors = per_point_error(pred, gt)
    if len(errors) == 0:
        return np.array([], dtype=bool), 0.0
    threshold = float(np.percentile(errors, percentile))
    mask = errors > threshold
    return mask, threshold


def visualize_failures(
    pred: o3d.geometry.PointCloud,
    errors: np.ndarray,
    output_path: str | None = None,
) -> o3d.geometry.PointCloud:
    """Color point cloud by error magnitude (blue=low, red=high).

    Returns the colored point cloud. Optionally saves a rendered image.
    """
    colored = o3d.geometry.PointCloud(pred)

    if len(errors) == 0:
        return colored

    # Normalize errors to [0, 1] for colormap
    e_min, e_max = errors.min(), errors.max()
    if e_max > e_min:
        normalized = (errors - e_min) / (e_max - e_min)
    else:
        normalized = np.zeros_like(errors)

    # Blue (low error) to Red (high error)
    colors = np.zeros((len(errors), 3))
    colors[:, 0] = normalized       # R increases with error
    colors[:, 2] = 1.0 - normalized  # B decreases with error
    colored.colors = o3d.utility.Vector3dVector(colors)

    if output_path is not None:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1280, height=960)
            vis.add_geometry(colored)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(output_path)
            vis.destroy_window()
        except Exception:
            # Offscreen rendering may fail without display (CI)
            pass

    return colored
