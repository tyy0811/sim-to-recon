"""Shared test fixtures for sim-to-recon."""

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def synthetic_cloud_pair():
    """Create a pair of point clouds for metric testing.

    Returns (cloud_a, cloud_b) where cloud_b is cloud_a shifted by [1, 0, 0].
    """
    rng = np.random.RandomState(42)
    points_a = rng.randn(1000, 3).astype(np.float64)

    cloud_a = o3d.geometry.PointCloud()
    cloud_a.points = o3d.utility.Vector3dVector(points_a)

    points_b = points_a + np.array([1.0, 0.0, 0.0])
    cloud_b = o3d.geometry.PointCloud()
    cloud_b.points = o3d.utility.Vector3dVector(points_b)

    return cloud_a, cloud_b


@pytest.fixture
def identical_clouds():
    """Create two identical point clouds."""
    rng = np.random.RandomState(42)
    points = rng.randn(500, 3).astype(np.float64)

    cloud_a = o3d.geometry.PointCloud()
    cloud_a.points = o3d.utility.Vector3dVector(points)

    cloud_b = o3d.geometry.PointCloud()
    cloud_b.points = o3d.utility.Vector3dVector(points.copy())

    return cloud_a, cloud_b


@pytest.fixture
def empty_cloud():
    """Create an empty point cloud."""
    return o3d.geometry.PointCloud()
