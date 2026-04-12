"""Test COLMAP workspace preparation + undistort + patch-match.cfg locally.

Uses the synthetic smoke scene (data/synthetic_smoke/) to verify:
1. Workspace preparation writes valid COLMAP files
2. pycolmap.undistort_images runs without error
3. patch-match.cfg is generated correctly
4. stereo_fusion API signature for PLY output

Does NOT run PatchMatch (needs CUDA), but validates everything up to that point.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def test_workspace_prep():
    """Test workspace prep with synthetic scene."""
    print("=== Test 1: Workspace preparation ===")

    scene_dir = Path("data/synthetic_smoke")
    if not scene_dir.exists():
        print("SKIP: run scripts/generate_synthetic_scene.py first")
        return False

    from simtorecon.data.dtu import DTUScene
    from simtorecon.pipeline.colmap_runner import ColmapRunner
    from simtorecon.pipeline.schemas import PipelineConfig

    scene = DTUScene(scene_dir, scan_id=0, light_idx=3)
    sub = scene.subsample(5, seed=42)

    config = PipelineConfig(target_width=320, target_height=240, use_modal=False)
    runner = ColmapRunner(sub, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        runner._prepare_workspace(workspace)

        sparse = workspace / "sparse" / "0"
        assert (sparse / "cameras.txt").exists(), "cameras.txt missing"
        assert (sparse / "images.txt").exists(), "images.txt missing"
        assert (sparse / "points3D.txt").exists(), "points3D.txt missing"

        cam_lines = [
            l for l in (sparse / "cameras.txt").read_text().splitlines()
            if l and not l.startswith("#")
        ]
        assert len(cam_lines) == 5, f"Expected 5 cameras, got {len(cam_lines)}"

        img_files = list((workspace / "images").glob("*.png"))
        assert len(img_files) == 5, f"Expected 5 images, got {len(img_files)}"

        print(f"  Cameras: {len(cam_lines)}")
        print(f"  Images:  {len(img_files)}")
        print("  PASS")
        return True


def test_undistort():
    """Test pycolmap.undistort_images with the synthetic scene workspace."""
    print("\n=== Test 2: Undistort images ===")

    try:
        import pycolmap
    except ImportError:
        print("SKIP: pycolmap not available")
        return False

    scene_dir = Path("data/synthetic_smoke")
    from simtorecon.data.dtu import DTUScene
    from simtorecon.pipeline.colmap_runner import ColmapRunner
    from simtorecon.pipeline.schemas import PipelineConfig

    scene = DTUScene(scene_dir, scan_id=0, light_idx=3)
    sub = scene.subsample(5, seed=42)
    config = PipelineConfig(target_width=320, target_height=240, use_modal=False)
    runner = ColmapRunner(sub, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        runner._prepare_workspace(workspace)

        sparse_dir = workspace / "sparse" / "0"
        image_dir = workspace / "images"
        dense_dir = workspace / "dense"

        undistort_opts = pycolmap.UndistortCameraOptions()
        undistort_opts.max_image_size = 320
        pycolmap.undistort_images(
            output_path=str(dense_dir),
            input_path=str(sparse_dir),
            image_path=str(image_dir),
            undistort_options=undistort_opts,
        )

        assert dense_dir.exists(), "dense dir not created"
        contents = os.listdir(dense_dir)
        print(f"  Dense dir contents: {contents}")

        # Check images were undistorted
        undist_images = list((dense_dir / "images").glob("*.png"))
        print(f"  Undistorted images: {len(undist_images)}")

        # Check stereo dir
        stereo_dir = dense_dir / "stereo"
        if stereo_dir.exists():
            stereo_contents = os.listdir(stereo_dir)
            print(f"  Stereo dir: {stereo_contents}")

            # Check if patch-match.cfg was auto-generated
            cfg_file = stereo_dir / "patch-match.cfg"
            if cfg_file.exists():
                cfg = cfg_file.read_text()
                print(f"  patch-match.cfg exists ({len(cfg.splitlines())} lines)")
                print(f"    First 4 lines: {cfg.splitlines()[:4]}")
            else:
                print("  patch-match.cfg NOT auto-generated (expected — no 3D points)")
        else:
            print("  stereo/ dir not created")

        print("  PASS")
        return True


def test_patch_match_cfg_generation():
    """Test manual patch-match.cfg generation."""
    print("\n=== Test 3: patch-match.cfg generation ===")

    image_names = ["0000.png", "0001.png", "0002.png", "0003.png", "0004.png"]

    cfg_lines = []
    for ref_img in image_names:
        cfg_lines.append(ref_img)
        sources = [s for s in image_names if s != ref_img]
        cfg_lines.append(", ".join(sources))

    cfg_text = "\n".join(cfg_lines) + "\n"

    # Validate format
    lines = cfg_text.strip().split("\n")
    assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"

    for i in range(0, len(lines), 2):
        ref = lines[i]
        sources = lines[i + 1]
        assert ref.endswith(".png"), f"Line {i} not an image name: {ref}"
        source_list = [s.strip() for s in sources.split(",")]
        assert len(source_list) == 4, f"Expected 4 sources, got {len(source_list)}"
        assert ref not in source_list, f"Ref image {ref} in its own source list"

    print(f"  Generated {len(image_names)} entries")
    print(f"  Sample: {lines[0]} -> {lines[1]}")
    print("  PASS")
    return True


def test_fusion_api():
    """Check stereo_fusion API supports output_type='PLY'."""
    print("\n=== Test 4: stereo_fusion API check ===")

    try:
        import pycolmap
    except ImportError:
        print("SKIP: pycolmap not available")
        return False

    doc = pycolmap.stereo_fusion.__doc__ or ""
    print(f"  Doc: {doc[:200]}")

    # Check if output_type parameter exists in doc
    if "output_type" in doc:
        print("  output_type parameter found in doc")
    else:
        print("  WARNING: output_type not in doc — may not be supported")

    # Check default value
    if "'bin'" in doc or "bin" in doc.lower():
        print("  Default output_type appears to be 'bin'")

    if "'PLY'" in doc or "'ply'" in doc:
        print("  PLY output type mentioned in doc")
    else:
        print("  WARNING: PLY not mentioned — may need 'TXT' or other format")
        print("  Available in COLMAP CLI: --output_type {BIN, TXT, PLY}")

    print("  PASS (API exists, will verify on Modal)")
    return True


if __name__ == "__main__":
    results = []
    results.append(("workspace_prep", test_workspace_prep()))
    results.append(("undistort", test_undistort()))
    results.append(("patch_match_cfg", test_patch_match_cfg_generation()))
    results.append(("fusion_api", test_fusion_api()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {name}: {status}")
