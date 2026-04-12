"""Unit tests for ColmapRunner quaternion conversion and workspace preparation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from simtorecon.pipeline.colmap_runner import ColmapRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_quaternion_close(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-12) -> None:
    """Assert two quaternions represent the same rotation.

    Quaternions q and -q encode the same rotation, so we compare against
    both signs and pass if either matches.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    close_pos = np.allclose(actual, expected, atol=atol)
    close_neg = np.allclose(actual, -expected, atol=atol)
    assert close_pos or close_neg, (
        f"Quaternion mismatch:\n  actual:   {actual}\n  expected: +/-{expected}"
    )


# ---------------------------------------------------------------------------
# TestRotationToQuaternion
# ---------------------------------------------------------------------------

class TestRotationToQuaternion:
    """Tests for ColmapRunner._rotation_to_quaternion."""

    def test_identity(self) -> None:
        """Identity rotation -> [1, 0, 0, 0]."""
        R = np.eye(3)
        q = ColmapRunner._rotation_to_quaternion(R)
        _assert_quaternion_close(q, [1.0, 0.0, 0.0, 0.0])

    def test_90_deg_about_z(self) -> None:
        """90-degree rotation about Z -> [cos(45deg), 0, 0, sin(45deg)]."""
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        q = ColmapRunner._rotation_to_quaternion(R)
        _assert_quaternion_close(q, expected, atol=1e-10)

    def test_90_deg_about_x(self) -> None:
        """90-degree rotation about X -> [cos(45deg), sin(45deg), 0, 0]."""
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c,  -s],
            [0.0, s,   c],
        ])
        expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        q = ColmapRunner._rotation_to_quaternion(R)
        _assert_quaternion_close(q, expected, atol=1e-10)

    def test_90_deg_about_y(self) -> None:
        """90-degree rotation about Y -> [cos(45deg), 0, sin(45deg), 0]."""
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R = np.array([
            [ c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ])
        expected = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
        q = ColmapRunner._rotation_to_quaternion(R)
        _assert_quaternion_close(q, expected, atol=1e-10)

    def test_180_deg_about_x(self) -> None:
        """180-degree rotation about X (trace = -1, enters non-trace branch).

        Expected quaternion: [0, 1, 0, 0] (or sign flip).
        """
        R = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])
        q = ColmapRunner._rotation_to_quaternion(R)
        _assert_quaternion_close(q, [0.0, 1.0, 0.0, 0.0], atol=1e-10)

    def test_random_rotation_unit_norm(self) -> None:
        """A random rotation matrix should produce a unit-norm quaternion."""
        rng = np.random.RandomState(0)
        # Generate a random rotation via QR decomposition
        A = rng.randn(3, 3)
        Q, _ = np.linalg.qr(A)
        # Ensure proper rotation (det = +1)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        q = ColmapRunner._rotation_to_quaternion(Q)
        norm = np.linalg.norm(q)
        assert abs(norm - 1.0) < 1e-10, f"Quaternion norm {norm} is not 1"

    def test_random_rotation_roundtrip(self) -> None:
        """Converting R -> q -> R' should recover the original rotation."""
        rng = np.random.RandomState(7)
        A = rng.randn(3, 3)
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        q = ColmapRunner._rotation_to_quaternion(Q)
        w, x, y, z = q

        # Reconstruct rotation matrix from quaternion
        R_reconstructed = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])
        np.testing.assert_allclose(R_reconstructed, Q, atol=1e-10)


# ---------------------------------------------------------------------------
# TestPrepareWorkspace
# ---------------------------------------------------------------------------

class TestPrepareWorkspace:
    """Tests for ColmapRunner._prepare_workspace."""

    @staticmethod
    def _make_mock_scene(tmp_path: Path) -> MagicMock:
        """Build a mock DatasetAdapter backed by 3 dummy images and calibration files.

        Each pose is derived from a projection matrix P = K @ [R|t] written
        to calibration/pos_NNN.txt files, matching the DTU convention.
        """
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create 3 dummy PNG images (small 4x4 RGB)
        import cv2

        names = ["rect_001_3_r5000.png", "rect_002_3_r5000.png", "rect_003_3_r5000.png"]
        for name in names:
            dummy = np.zeros((4, 4, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / name), dummy)

        # Calibration: 3 cameras with known K, R, t
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()

        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])

        poses = []  # list of 4x4 world-to-camera matrices
        for i in range(3):
            # Rotate slightly around Y for each camera
            angle = i * 0.3
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([
                [ c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ])
            t = np.array([i * 0.5, 0.0, 5.0])
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses.append(pose)

            # Write projection matrix P = K @ [R | t] flattened (3x4 -> 12 values)
            Rt = np.hstack([R, t.reshape(3, 1)])
            P = K @ Rt
            flat = P.flatten()
            with open(cal_dir / f"pos_{i:03d}.txt", "w") as f:
                f.write(" ".join(f"{v:.10f}" for v in flat))

        # Build mock DatasetAdapter
        scene = MagicMock()
        scene.name = "mock_scan"
        scene.n_images = 3
        scene.get_image_size.return_value = (4, 4)
        scene.get_image.side_effect = lambda idx: np.zeros((4, 4, 3), dtype=np.uint8)
        scene.get_intrinsics.side_effect = lambda idx: K.copy()
        scene.get_pose.side_effect = lambda idx: poses[idx]

        return scene

    def test_prepare_workspace_writes_valid_colmap_files(self, tmp_path: Path) -> None:
        """_prepare_workspace should create cameras.txt, images.txt, and
        points3D.txt in sparse/0/ with correct COLMAP text format content
        for a 3-image mock scene.
        """
        from simtorecon.pipeline.schemas import PipelineConfig

        scene = self._make_mock_scene(tmp_path)
        config = PipelineConfig(
            target_width=4,
            target_height=4,
        )
        runner = ColmapRunner(scene, config)
        runner._prepare_workspace(tmp_path / "workspace")

        sparse_dir = tmp_path / "workspace" / "sparse" / "0"

        # ---- cameras.txt ----
        cameras_path = sparse_dir / "cameras.txt"
        assert cameras_path.exists(), "cameras.txt was not created"
        cameras_text = cameras_path.read_text()

        # Should contain header comments
        assert "CAMERA_ID" in cameras_text
        # Should have 3 camera entries (non-comment, non-empty lines)
        cam_lines = [
            ln for ln in cameras_text.splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert len(cam_lines) == 3, f"Expected 3 camera lines, got {len(cam_lines)}"
        # Each line must be: ID PINHOLE W H fx fy cx cy
        for line in cam_lines:
            parts = line.split()
            assert len(parts) == 8, f"Camera line has {len(parts)} fields: {line}"
            assert parts[1] == "PINHOLE"
            assert parts[2] == "4"  # target_width
            assert parts[3] == "4"  # target_height
            # Intrinsic values should be parseable floats
            for val in parts[4:]:
                float(val)

        # ---- images.txt ----
        images_path = sparse_dir / "images.txt"
        assert images_path.exists(), "images.txt was not created"
        images_text = images_path.read_text()

        assert "IMAGE_ID" in images_text
        # Non-comment lines: 3 image lines + 3 empty lines for 2D points
        img_lines = [
            ln for ln in images_text.splitlines()
            if not ln.startswith("#")
        ]
        # Filter to data lines (non-empty) — should be exactly 3
        data_lines = [ln for ln in img_lines if ln.strip()]
        assert len(data_lines) == 3, f"Expected 3 image data lines, got {len(data_lines)}"
        # Each data line: IMG_ID QW QX QY QZ TX TY TZ CAM_ID NAME
        for line in data_lines:
            parts = line.split()
            assert len(parts) == 10, f"Image line has {len(parts)} fields: {line}"
            # Quaternion values should be finite floats
            qw, qx, qy, qz = [float(parts[i]) for i in range(1, 5)]
            q_norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            assert abs(q_norm - 1.0) < 1e-6, f"Quaternion norm {q_norm} != 1"
            # Translation values parseable
            for idx in range(5, 8):
                float(parts[idx])
            # Image name should end with .png
            assert parts[9].endswith(".png")

        # ---- points3D.txt ----
        points3d_path = sparse_dir / "points3D.txt"
        assert points3d_path.exists(), "points3D.txt was not created"
        points3d_text = points3d_path.read_text()
        # Should be mostly empty (just a comment)
        non_comment = [
            ln for ln in points3d_text.splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert len(non_comment) == 0, "points3D.txt should have no data lines"

        # ---- images/ directory should have 3 images ----
        workspace_images = list((tmp_path / "workspace" / "images").glob("*.png"))
        assert len(workspace_images) == 3
