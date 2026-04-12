"""Tests for DTU data loader."""


import numpy as np

from simtorecon.data.dtu import DTUScene


class TestDTUScene:
    def test_subsample_reduces_count(self, tmp_path):
        """Subsampling should produce exactly n_views images."""
        # Create a minimal mock scene directory
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        calib_dir = tmp_path / "calibration"
        calib_dir.mkdir()

        # Create 10 dummy images
        for i in range(10):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_dir / f"rect_{i+1:03d}_3_r5000.png"), img)

        scene = DTUScene(root=tmp_path, scan_id=9, light_idx=3)
        assert scene.n_images == 10

        sub = scene.subsample(5, seed=42)
        assert sub.n_images == 5

    def test_subsample_is_deterministic(self, tmp_path):
        """Same seed should produce same subset."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(10):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_dir / f"rect_{i+1:03d}_3_r5000.png"), img)

        scene = DTUScene(root=tmp_path, scan_id=9, light_idx=3)
        sub1 = scene.subsample(5, seed=42)
        sub2 = scene.subsample(5, seed=42)

        paths1 = [p.name for p in sub1._image_paths]
        paths2 = [p.name for p in sub2._image_paths]
        assert paths1 == paths2

    def test_empty_scene_has_zero_images(self, tmp_path):
        """Scene with no images directory returns 0 images."""
        scene = DTUScene(root=tmp_path)
        assert scene.n_images == 0
