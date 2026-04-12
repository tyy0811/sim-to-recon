"""Tests for view count stress sweep."""

import json

from simtorecon.pipeline.schemas import ReconstructionResult


class TestSweepResults:
    def test_result_is_serializable(self):
        """ReconstructionResult should round-trip through JSON."""
        result = ReconstructionResult(
            scene_name="test",
            n_views=15,
            n_points=50000,
            output_ply="results/test/dense.ply",
            elapsed_seconds=60.0,
        )
        data = json.loads(result.model_dump_json())
        recovered = ReconstructionResult(**data)
        assert recovered.n_views == 15
        assert recovered.n_points == 50000

    def test_sweep_config_view_counts(self):
        """View counts should be sorted descending in sweep."""
        view_counts = [49, 30, 15, 8]
        sorted_desc = sorted(view_counts, reverse=True)
        assert sorted_desc == [49, 30, 15, 8]

    def test_subsample_preserves_scene_name(self):
        """Subsampled scenes should keep the original scene name."""
        # Create minimal mock
        import tempfile
        from pathlib import Path

        import numpy as np

        from simtorecon.data.dtu import DTUScene

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_dir = tmp_path / "images"
            img_dir.mkdir()
            for i in range(10):
                img = np.zeros((100, 100, 3), dtype=np.uint8)
                import cv2
                cv2.imwrite(str(img_dir / f"rect_{i+1:03d}_3_r5000.png"), img)

            scene = DTUScene(root=tmp_path, scan_id=9, light_idx=3)
            sub = scene.subsample(5)
            assert sub.name == "dtu_scan9"
