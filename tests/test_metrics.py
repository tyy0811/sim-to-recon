"""Tests for evaluation metrics: chamfer, accuracy, completeness, F-score, alignment."""

import numpy as np
import open3d as o3d
import pytest

from simtorecon.evaluation.alignment import align_to_gt
from simtorecon.evaluation.metrics import (
    accuracy,
    chamfer_distance,
    completeness,
    f_score,
)


class TestChamferDistance:
    def test_identical_clouds_have_zero_chamfer(self, identical_clouds):
        a, b = identical_clouds
        cd = chamfer_distance(a, b)
        assert cd == pytest.approx(0.0, abs=1e-10)

    def test_shifted_cloud_has_known_chamfer(self):
        import numpy as np
        import open3d as o3d

        # Single point: NN distance is exactly the shift magnitude
        a = o3d.geometry.PointCloud()
        a.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        b = o3d.geometry.PointCloud()
        b.points = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))
        cd = chamfer_distance(a, b)
        assert cd == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self, synthetic_cloud_pair):
        a, b = synthetic_cloud_pair
        assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a), rel=1e-6)


class TestAccuracyCompleteness:
    def test_accuracy_completeness_are_symmetric(self, synthetic_cloud_pair):
        a, b = synthetic_cloud_pair
        # accuracy(a, b) = mean dist from a to b
        # completeness(a, b) = mean dist from b to a
        # For a uniform shift, these should be equal
        acc = accuracy(a, b)
        comp = completeness(a, b)
        assert acc == pytest.approx(comp, rel=0.1)


class TestFScore:
    def test_f_score_is_between_zero_and_one(self, synthetic_cloud_pair):
        a, b = synthetic_cloud_pair
        fs = f_score(a, b, threshold=2.0)
        assert 0.0 <= fs <= 1.0

    def test_identical_clouds_have_perfect_f_score(self, identical_clouds):
        a, b = identical_clouds
        fs = f_score(a, b, threshold=1.0)
        assert fs == pytest.approx(1.0, abs=1e-6)

    def test_empty_pred_returns_zero(self, empty_cloud, identical_clouds):
        _, gt = identical_clouds
        fs = f_score(empty_cloud, gt, threshold=1.0)
        assert fs == 0.0


class TestPerPointError:
    def test_per_point_error_shape(self, synthetic_cloud_pair):
        from simtorecon.evaluation.failure import per_point_error

        a, b = synthetic_cloud_pair
        errors = per_point_error(a, b)
        assert errors.shape == (len(a.points),)


class TestFailureRegions:
    def test_failure_regions_returns_correct_shape(self, synthetic_cloud_pair):
        from simtorecon.evaluation.failure import failure_regions

        a, b = synthetic_cloud_pair
        mask, threshold = failure_regions(a, b, percentile=95.0)
        assert mask.shape == (len(a.points),)
        assert mask.dtype == bool
        # ~5% of points should be flagged at 95th percentile
        assert 0.0 < mask.mean() < 0.15

    def test_failure_regions_threshold_at_percentile(self):
        import numpy as np
        import open3d as o3d

        from simtorecon.evaluation.failure import failure_regions

        # Deterministic cloud: 100 points on a line, GT is origin
        points = np.zeros((100, 3))
        points[:, 0] = np.arange(100, dtype=float)
        pred = o3d.geometry.PointCloud()
        pred.points = o3d.utility.Vector3dVector(points)
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))

        mask, threshold = failure_regions(pred, gt, percentile=90.0)
        assert threshold == pytest.approx(np.percentile(np.arange(100.0), 90.0), rel=0.01)


class TestAlignment:
    def test_aligned_clouds_have_low_chamfer(self):
        rng = np.random.RandomState(123)
        gt_points = rng.randn(500, 3).astype(np.float64)

        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(gt_points)

        # Transform: scale 2x + shift [10, 0, 0]
        pred_points = gt_points * 2.0 + np.array([10.0, 0.0, 0.0])
        pred = o3d.geometry.PointCloud()
        pred.points = o3d.utility.Vector3dVector(pred_points)

        aligned, transform, fitness = align_to_gt(pred, gt)
        cd = chamfer_distance(aligned, gt)
        assert cd < 1.0, f"Chamfer distance after alignment too large: {cd}"
        assert transform.shape == (4, 4)
        assert 0.0 <= fitness <= 1.0
