"""Smoke test for the COLMAP pipeline.

This test is marked as slow and requires DTU data to be present.
It runs a minimal reconstruction (5 images, 400x300) to verify
the pipeline works end-to-end.
"""

from pathlib import Path

import pytest

from simtorecon.data.dtu import DTUScene
from simtorecon.pipeline.schemas import PipelineConfig


@pytest.mark.slow
@pytest.mark.smoke
def test_pipeline_smoke(tmp_path):
    """Smoke test: run reconstruction on 5-image subset at low resolution.

    Skipped if DTU data is not available or pycolmap is not installed.
    """
    data_root = Path("data/dtu/scan9")
    if not data_root.exists():
        pytest.skip("DTU scan9 data not available (run 'make data' first)")

    try:
        import importlib.util

        if importlib.util.find_spec("pycolmap") is None:
            raise ImportError
    except ImportError:
        pytest.skip("pycolmap not installed")

    scene = DTUScene(root=data_root, scan_id=9, light_idx=3)
    if scene.n_images == 0:
        pytest.skip("No images found in DTU scan9")

    sub = scene.subsample(5, seed=42)

    config = PipelineConfig(
        target_width=400,
        target_height=300,
        max_image_size=400,
    )

    from simtorecon.pipeline.colmap_runner import ColmapRunner

    runner = ColmapRunner(sub, config)
    result = runner.run(tmp_path / "smoke_output")

    assert result.n_points > 0
    assert result.output_ply.exists()
    assert result.elapsed_seconds > 0
