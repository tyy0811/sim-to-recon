# sim-to-recon V1 Remaining Tasks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the V1 benchmark — from "skeleton with 21 passing tests" to "shippable repo with real reconstruction results, stress sweep, figures, and polished README."

**Architecture:** ColmapRunner orchestrates pycolmap's posed-image dense MVS (undistort → PatchMatch → fusion) on DTU scan9. Evaluation computes chamfer/accuracy/completeness/F-score against GT. View count sweep at N∈{49,30,15,8} produces the stress test. C++ calibration module is already complete — just needs local build verification and synthetic test fixtures.

**Tech Stack:** Python 3.10+ / pycolmap / Open3D / Pydantic / C++17 / OpenCV / CMake / GoogleTest

**Critical constraint discovered during review:** `pycolmap.has_cuda = False` on this machine. COLMAP PatchMatch stereo requires CUDA GPU. The plan must resolve this before any reconstruction can run.

---

## Current State

- 21 Python tests passing, lint clean
- All source modules written (data, pipeline, evaluation, stress, schemas)
- C++ calibration module complete with 8 GoogleTest tests (not yet built locally)
- 3 critical bugs fixed (RQ decomposition, calibration index mapping, seed usage)
- DTU data NOT yet downloaded
- No reconstructions run
- No figures generated
- DECISIONS.md has 3 of 10 entries
- README has placeholder numbers
- CI not tested on GitHub

---

### Task 1: Resolve GPU constraint for PatchMatch

**Files:**
- Modify: `src/simtorecon/pipeline/colmap_runner.py`
- Modify: `src/simtorecon/pipeline/schemas.py`
- Modify: `DECISIONS.md`

**Context:** `pycolmap.has_cuda` is `False` on this machine (macOS, Intel, no NVIDIA GPU). COLMAP's PatchMatch stereo is a CUDA-only algorithm. The current `ColmapRunner.run()` will crash at `pycolmap.patch_match_stereo()`. There are two viable paths:

- **Option A (CLI fallback):** Shell out to the `colmap` CLI binary which can use OpenCL on macOS. Requires `brew install colmap`.
- **Option B (Cloud GPU):** Run PatchMatch on Modal with A10G. More complex but guaranteed to work.
- **Option C (Accept constraint):** Require a CUDA machine. Document this in Honest Scope.

**Step 1: Decide on approach**

Check if COLMAP CLI is installable via Homebrew with Metal/OpenCL support:
```bash
brew info colmap
```
If available, go with Option A. If not, go with Option C and document the CUDA requirement.

**Step 2: If Option A — add CLI fallback to ColmapRunner**

Add a `use_cli: bool = False` field to `PipelineConfig` in `src/simtorecon/pipeline/schemas.py:15`:

```python
use_cli: bool = Field(default=False, description="Use COLMAP CLI instead of pycolmap for GPU-less machines")
colmap_binary: str = Field(default="colmap", description="Path to COLMAP CLI binary")
```

Add a `_run_dense_cli()` method to `ColmapRunner` in `src/simtorecon/pipeline/colmap_runner.py` that shells out:

```python
def _run_dense_cli(self, undistorted_dir: Path) -> None:
    """Run PatchMatch + fusion via COLMAP CLI (OpenCL/Metal support)."""
    import subprocess

    subprocess.run([
        self.config.colmap_binary, "patch_match_stereo",
        "--workspace_path", str(undistorted_dir),
        "--PatchMatchStereo.max_image_size", str(self.config.max_image_size),
        "--PatchMatchStereo.geom_consistency", "true" if self.config.geom_consistency else "false",
    ], check=True)

    fused_ply = undistorted_dir / "fused.ply"
    subprocess.run([
        self.config.colmap_binary, "stereo_fusion",
        "--workspace_path", str(undistorted_dir),
        "--output_path", str(fused_ply),
        "--StereoFusion.min_num_pixels", str(self.config.min_num_pixels),
        "--StereoFusion.max_reproj_error", str(self.config.max_reproj_error),
        "--StereoFusion.max_depth_error", str(self.config.max_depth_error),
    ], check=True)
```

Modify `run()` to dispatch to CLI or pycolmap:

```python
# In run(), replace Steps 3-4 with:
if self.config.use_cli:
    self._run_dense_cli(undistorted_dir)
    fused_ply = undistorted_dir / "fused.ply"
else:
    # existing pycolmap code for Steps 3-4
```

**Step 3: If Option C — document CUDA requirement**

Add to `DECISIONS.md`:
```markdown
## 4. CUDA GPU required for dense reconstruction

**Decision:** Accept that PatchMatch stereo requires CUDA. Document in Honest Scope.

**Rationale:** COLMAP's PatchMatch is fundamentally a GPU algorithm. Adding CPU fallbacks
(OpenMVS, PMVS) would change the pipeline under test. The benchmark tests COLMAP's MVS,
not an alternative. CI runs only unit tests; reconstruction requires a GPU machine.
```

**Step 4: Update `run()` to handle the pycolmap API correctly**

The current `run()` passes keyword args in a dict style (`PatchMatchStereo={...}`) which is not the actual pycolmap 3.12 API. Fix to use `PatchMatchOptions`:

In `src/simtorecon/pipeline/colmap_runner.py`, replace the PatchMatch call:
```python
# Step 3: PatchMatch stereo
options = pycolmap.PatchMatchOptions()
options.max_image_size = self.config.max_image_size
options.geom_consistency = self.config.geom_consistency
options.num_iterations = self.config.num_iterations

pycolmap.patch_match_stereo(
    workspace_path=undistorted_dir,
    options=options,
)
```

And the fusion call:
```python
# Step 4: Stereo fusion
fused_ply = undistorted_dir / "fused.ply"
fusion_options = pycolmap.StereoFusionOptions()
fusion_options.min_num_pixels = self.config.min_num_pixels
fusion_options.max_reproj_error = self.config.max_reproj_error
fusion_options.max_depth_error = self.config.max_depth_error

pycolmap.stereo_fusion(
    workspace_path=undistorted_dir,
    output_path=fused_ply,
    options=fusion_options,
)
```

**Step 5: Run lint**

```bash
ruff check src/simtorecon/pipeline/
```
Expected: All checks passed.

**Step 6: Commit**

```bash
git add src/simtorecon/pipeline/ DECISIONS.md
git commit -m "feat: resolve GPU constraint for PatchMatch, fix pycolmap API"
```

---

### Task 2: Add quaternion rotation unit test

**Files:**
- Modify: `tests/test_metrics.py` (or create `tests/test_colmap_runner.py`)

**Context:** `ColmapRunner._rotation_to_quaternion()` has 4 code paths and zero test coverage. This function is critical — wrong quaternions = wrong COLMAP workspace = wrong reconstruction.

**Step 1: Write the test**

Create `tests/test_colmap_runner.py`:

```python
"""Tests for ColmapRunner utilities."""

import numpy as np
import pytest

from simtorecon.pipeline.colmap_runner import ColmapRunner


class TestRotationToQuaternion:
    """Test all 4 branches of _rotation_to_quaternion."""

    def test_identity_rotation(self):
        R = np.eye(3)
        q = ColmapRunner._rotation_to_quaternion(R)
        # Identity rotation → quaternion [1, 0, 0, 0]
        assert q == pytest.approx([1, 0, 0, 0], abs=1e-10)

    def test_90deg_rotation_about_z(self):
        # R_z(90°) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = ColmapRunner._rotation_to_quaternion(R)
        # Expected: [cos(45°), 0, 0, sin(45°)]
        expected = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        assert q == pytest.approx(expected, abs=1e-10)

    def test_90deg_rotation_about_x(self):
        # R_x(90°) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        q = ColmapRunner._rotation_to_quaternion(R)
        expected = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        assert q == pytest.approx(expected, abs=1e-10)

    def test_90deg_rotation_about_y(self):
        # R_y(90°) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        q = ColmapRunner._rotation_to_quaternion(R)
        expected = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
        assert q == pytest.approx(expected, abs=1e-10)

    def test_180deg_rotation_about_x(self):
        # R_x(180°) — trace = -1, exercises different branch
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        q = ColmapRunner._rotation_to_quaternion(R)
        # [0, 1, 0, 0] or [0, -1, 0, 0] (sign ambiguity)
        assert abs(q[0]) < 1e-10
        assert abs(abs(q[1]) - 1.0) < 1e-10

    def test_quaternion_is_unit_norm(self):
        # Random valid rotation matrix via QR
        rng = np.random.RandomState(42)
        A = rng.randn(3, 3)
        R, _ = np.linalg.qr(A)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        q = ColmapRunner._rotation_to_quaternion(R)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-10)
```

**Step 2: Run the tests**

```bash
pytest tests/test_colmap_runner.py -v
```
Expected: 6 passed.

**Step 3: Commit**

```bash
git add tests/test_colmap_runner.py
git commit -m "test: add quaternion rotation unit tests for ColmapRunner"
```

---

### Task 3: DTU data acquisition + loader verification

**Files:**
- Modify: `tests/test_dtu_loader.py` (add real-data tests)
- Modify: `Makefile`

**Context:** DTU scan9 data must be manually downloaded. The loader has been tested with synthetic mock data but never with real DTU calibration files.

**Step 1: Download DTU data**

Run `make data` to get download instructions, then manually download:
1. Rectified images for scan9 (~250MB) → `data/dtu/scan9/images/`
2. Ground truth point cloud → `data/dtu/scan9/gt/stl009_total.ply`
3. Calibration files → `data/dtu/scan9/calibration/pos_000.txt` through `pos_048.txt`

The calibration files come from DTU's `SampleSet.zip` → `Calibration/cal18/`.

**Step 2: Verify the loader loads real data**

```bash
python -c "
from pathlib import Path
from simtorecon.data.dtu import DTUScene
scene = DTUScene(Path('data/dtu/scan9'))
print(f'Images: {scene.n_images}')
print(f'Image size: {scene.get_image_size()}')
print(f'Has GT: {scene.has_ground_truth()}')
K = scene.get_intrinsics(0)
print(f'K[0]:\n{K}')
pose = scene.get_pose(0)
print(f'Pose[0]:\n{pose}')
"
```

Expected: 49 images, 1600×1200, GT available, K has realistic focal lengths (~2800 for DTU), pose is a valid rigid transform.

**Step 3: Sanity check — verify K has positive focal length and R is orthogonal**

```bash
python -c "
from pathlib import Path
import numpy as np
from simtorecon.data.dtu import DTUScene
scene = DTUScene(Path('data/dtu/scan9'))
for i in range(scene.n_images):
    K = scene.get_intrinsics(i)
    assert K[0,0] > 0, f'fx negative at {i}'
    assert K[1,1] > 0, f'fy negative at {i}'
    assert K[2,2] == 1.0, f'K not normalized at {i}'
    pose = scene.get_pose(i)
    R = pose[:3,:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f'R not orthogonal at {i}'
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), f'det(R) != 1 at {i}'
print('All 49 calibrations valid.')
"
```

Expected: No assertions fired.

**Step 4: Verify subsampling preserves correct calibration pairing**

```bash
python -c "
from pathlib import Path
import numpy as np
from simtorecon.data.dtu import DTUScene
scene = DTUScene(Path('data/dtu/scan9'))
sub = scene.subsample(10, seed=42)
print(f'Subsampled to {sub.n_images} views')
print(f'Index map: {sub._index_map}')
# Verify subsampled intrinsics match original at mapped index
for i in range(sub.n_images):
    K_sub = sub.get_intrinsics(i)
    K_orig = scene.get_intrinsics(sub._index_map[i])
    assert np.allclose(K_sub, K_orig), f'K mismatch at sub idx {i}'
print('Calibration pairing verified.')
"
```

Expected: Calibration pairing verified.

**Step 5: Verify GT point cloud loads**

```bash
python -c "
from pathlib import Path
from simtorecon.data.dtu import DTUScene
scene = DTUScene(Path('data/dtu/scan9'))
gt = scene.get_ground_truth()
print(f'GT points: {len(gt.points)}')
import numpy as np
pts = np.asarray(gt.points)
print(f'GT bounds: min={pts.min(axis=0)}, max={pts.max(axis=0)}')
"
```

Expected: GT has ~300k-800k points, coordinates in DTU object space.

**Step 6: Commit**

```bash
git add tests/test_dtu_loader.py
git commit -m "chore: verify DTU loader on real scan9 data"
```

---

### Task 4: COLMAP end-to-end reconstruction

**Files:**
- Modify: `src/simtorecon/pipeline/colmap_runner.py`
- Run: `experiments/run_baseline.py`

**Context:** This is the Day 2 critical milestone. The posed-image MVS pipeline must produce a `.ply` point cloud. Requires GPU (see Task 1).

**Step 1: Verify workspace preparation produces valid COLMAP files**

Write a test that checks the COLMAP text files are valid without running the actual reconstruction:

Add to `tests/test_colmap_runner.py`:

```python
class TestPrepareWorkspace:
    def test_workspace_produces_valid_colmap_files(self, tmp_path):
        """Verify cameras.txt, images.txt, points3D.txt are written correctly."""
        # Create a minimal mock scene with 3 images
        import cv2
        img_dir = tmp_path / "scene" / "images"
        img_dir.mkdir(parents=True)
        calib_dir = tmp_path / "scene" / "calibration"
        calib_dir.mkdir()

        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"rect_{i+1:03d}_3_r5000.png"), img)
            # Write a synthetic 3x4 projection matrix
            K = np.array([[500, 0, 50], [0, 500, 50], [0, 0, 1]], dtype=float)
            R = np.eye(3)
            t = np.array([0, 0, i * 0.5])
            P = K @ np.hstack([R, t.reshape(3, 1)])
            np.savetxt(calib_dir / f"pos_{i:03d}.txt", P.flatten())

        from simtorecon.data.dtu import DTUScene
        from simtorecon.pipeline.schemas import PipelineConfig

        scene = DTUScene(tmp_path / "scene")
        config = PipelineConfig(target_width=100, target_height=100)
        runner = ColmapRunner(scene, config)

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Call _prepare_workspace directly
        import pycolmap
        runner._prepare_workspace(workspace, pycolmap)

        cameras_txt = workspace / "sparse" / "0" / "cameras.txt"
        images_txt = workspace / "sparse" / "0" / "images.txt"
        points3d_txt = workspace / "sparse" / "0" / "points3D.txt"

        assert cameras_txt.exists()
        assert images_txt.exists()
        assert points3d_txt.exists()

        # Verify camera count
        camera_lines = [l for l in cameras_txt.read_text().splitlines()
                        if l and not l.startswith('#')]
        assert len(camera_lines) == 3

        # Verify image count (each image = 2 lines: data + empty)
        image_lines = [l for l in images_txt.read_text().splitlines()
                       if l and not l.startswith('#')]
        assert len(image_lines) == 3  # 3 data lines (empty lines filtered)
```

**Step 2: Run the test**

```bash
pytest tests/test_colmap_runner.py::TestPrepareWorkspace -v
```
Expected: PASS.

**Step 3: Run baseline reconstruction on real DTU data**

This requires a CUDA GPU. Run on a GPU machine or via the CLI fallback:

```bash
# If using CLI fallback:
python experiments/run_baseline.py --scene scan9 --config configs/scenes/dtu_scan9.yaml

# If on a GPU machine:
python experiments/run_baseline.py --scene scan9
```

Expected: `results/baseline_scan9/dense.ply` produced, ~100k-500k points, 15-25 min runtime at 800×600.

**Step 4: Visual sanity check**

```bash
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('results/baseline_scan9/dense.ply')
print(f'Points: {len(pcd.points)}')
o3d.visualization.draw_geometries([pcd])
"
```

Expected: Recognizable stone figure (DTU scan9).

**Step 5: Commit**

```bash
git add tests/test_colmap_runner.py src/simtorecon/pipeline/
git commit -m "feat: COLMAP pipeline end-to-end verified on DTU scan9"
```

---

### Task 5: DTU coordinate alignment + evaluation on baseline

**Files:**
- Modify: `src/simtorecon/evaluation/metrics.py` (if alignment wrapper needed)
- Create: `src/simtorecon/evaluation/alignment.py`
- Modify: `experiments/run_baseline.py` (add evaluation step)
- Modify: `DECISIONS.md`

**Context:** DTU GT point clouds are in object coordinates. COLMAP output is in COLMAP's coordinate system. The spec calls this out as "the single most likely thing to silently break." An alignment step is needed before metrics make sense.

**Step 1: Write the alignment module**

Create `src/simtorecon/evaluation/alignment.py`:

```python
"""Point cloud alignment for DTU evaluation.

DTU provides per-scene transformation matrices that align COLMAP's
coordinate system to the GT object frame. Without this alignment,
chamfer distances will be meaninglessly large.
"""

import numpy as np
import open3d as o3d


def align_to_gt(
    pred: o3d.geometry.PointCloud,
    gt: o3d.geometry.PointCloud,
    max_iterations: int = 50,
    threshold: float = 5.0,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Align predicted point cloud to GT using ICP.

    Uses coarse-to-fine ICP: first rough alignment at large threshold,
    then refined alignment at the specified threshold.

    Returns:
        aligned: The aligned predicted point cloud.
        transform: 4x4 transformation matrix applied.
    """
    # Estimate scale from bounding box ratio
    pred_bbox = pred.get_axis_aligned_bounding_box()
    gt_bbox = gt.get_axis_aligned_bounding_box()
    pred_extent = pred_bbox.get_max_bound() - pred_bbox.get_min_bound()
    gt_extent = gt_bbox.get_max_bound() - gt_bbox.get_min_bound()
    scale = np.median(gt_extent / (pred_extent + 1e-10))

    # Scale pred to match GT
    pred_scaled = o3d.geometry.PointCloud(pred)
    pred_scaled.scale(scale, center=pred_scaled.get_center())

    # Translate to match centroids
    pred_center = np.asarray(pred_scaled.get_center())
    gt_center = np.asarray(gt.get_center())
    pred_scaled.translate(gt_center - pred_center)

    # Coarse ICP
    result_coarse = o3d.pipelines.registration.registration_icp(
        pred_scaled, gt,
        max_correspondence_distance=threshold * 5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Fine ICP
    result_fine = o3d.pipelines.registration.registration_icp(
        pred_scaled, gt,
        max_correspondence_distance=threshold,
        init=result_coarse.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    aligned = o3d.geometry.PointCloud(pred_scaled)
    aligned.transform(result_fine.transformation)

    return aligned, result_fine.transformation
```

**Step 2: Write alignment test**

Add to `tests/test_metrics.py`:

```python
class TestAlignment:
    def test_aligned_clouds_have_low_chamfer(self):
        """Alignment of a scaled+shifted cloud should recover low chamfer."""
        import numpy as np
        import open3d as o3d
        from simtorecon.evaluation.alignment import align_to_gt

        rng = np.random.RandomState(42)
        gt_pts = rng.randn(500, 3)
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(gt_pts)

        # Transform: scale by 2x, shift by [10, 0, 0]
        pred_pts = gt_pts * 2.0 + np.array([10, 0, 0])
        pred = o3d.geometry.PointCloud()
        pred.points = o3d.utility.Vector3dVector(pred_pts)

        aligned, _ = align_to_gt(pred, gt, threshold=2.0)
        cd = chamfer_distance(aligned, gt)
        assert cd < 1.0  # Should be very close after alignment
```

**Step 3: Run test**

```bash
pytest tests/test_metrics.py::TestAlignment -v
```
Expected: PASS.

**Step 4: Add evaluation to baseline experiment**

Modify `experiments/run_baseline.py` to load GT, align, and compute metrics after the reconstruction completes. Add after the `runner.run()` call:

```python
# Evaluate against ground truth
if scene.has_ground_truth():
    import open3d as o3d
    from simtorecon.evaluation.alignment import align_to_gt
    from simtorecon.evaluation.metrics import chamfer_distance, accuracy, completeness, f_score

    pred = o3d.io.read_point_cloud(str(result.output_ply))
    gt = scene.get_ground_truth()

    aligned, transform = align_to_gt(pred, gt)
    result.chamfer = chamfer_distance(aligned, gt)
    result.accuracy = accuracy(aligned, gt)
    result.completeness = completeness(aligned, gt)
    result.f_score = f_score(aligned, gt, threshold=1.0)

    print(f"  Chamfer:      {result.chamfer:.4f}")
    print(f"  Accuracy:     {result.accuracy:.4f}")
    print(f"  Completeness: {result.completeness:.4f}")
    print(f"  F-score:      {result.f_score:.4f}")
```

**Step 5: Run on real data and sanity check**

```bash
python experiments/run_baseline.py --scene scan9
```

**Sanity check:** Chamfer should be in single-digit mm for scan9. If > 50mm, alignment is broken — debug alignment before proceeding.

**Step 6: Add DECISIONS.md entry**

```markdown
## 5. DTU coordinate alignment via ICP

**Decision:** Use scale-aware ICP (Open3D) to align COLMAP output to DTU GT coordinates.

**Alternatives considered:**
- DTU official `pos_*.txt` transformation matrices — undocumented format for COLMAP's coordinate system
- Procrustes alignment — equivalent but ICP handles noisy/incomplete data better

**Rationale:** COLMAP and DTU use different coordinate conventions. ICP with initial scale
estimation provides robust alignment without depending on undocumented format details.
Most subtle decision in the pipeline — visualize pred+GT overlay to verify.
```

**Step 7: Commit**

```bash
git add src/simtorecon/evaluation/alignment.py tests/test_metrics.py experiments/run_baseline.py DECISIONS.md
git commit -m "feat: DTU coordinate alignment + evaluation on baseline"
```

---

### Task 6: Build C++ calibration module locally

**Files:**
- Verify: `cpp/calib/` builds and tests pass
- Create: `cpp/calib/tests/fixtures/gen_fixtures.py` (synthetic chessboard generator)

**Step 1: Build the C++ module**

```bash
cmake -B build cpp/calib
cmake --build build
```

Expected: Builds without errors. If OpenCV is not found, set `OpenCV_DIR`:
```bash
cmake -B build cpp/calib -DOpenCV_DIR=$(brew --prefix opencv)/lib/cmake/opencv4
```

**Step 2: Run GoogleTest**

```bash
cd build && ctest --output-on-failure
```

Expected: 8 tests pass.

**Step 3: Generate synthetic chessboard fixtures**

Create `cpp/calib/tests/fixtures/gen_fixtures.py`:

```python
"""Generate synthetic 9x6 chessboard images for GoogleTest fixtures."""

import cv2
import numpy as np
from pathlib import Path

def generate_chessboard(
    pattern_size=(9, 6),
    square_size_px=40,
    image_size=(640, 480),
    output_dir="cpp/calib/tests/fixtures",
    n_views=5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = pattern_size
    board_w = (cols + 1) * square_size_px
    board_h = (rows + 1) * square_size_px

    for view_idx in range(n_views):
        img = np.ones(image_size[::-1] + (3,), dtype=np.uint8) * 220

        # Offset the chessboard slightly per view
        rng = np.random.RandomState(view_idx)
        ox = (image_size[0] - board_w) // 2 + rng.randint(-20, 20)
        oy = (image_size[1] - board_h) // 2 + rng.randint(-20, 20)

        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    x1, y1 = ox + c * square_size_px, oy + r * square_size_px
                    x2, y2 = x1 + square_size_px, y1 + square_size_px
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        cv2.imwrite(str(output_dir / f"chessboard_{view_idx:02d}.png"), img)
        print(f"Generated: chessboard_{view_idx:02d}.png")

if __name__ == "__main__":
    generate_chessboard()
```

Run it:
```bash
python cpp/calib/tests/fixtures/gen_fixtures.py
```

**Step 4: Test the CLI on fixtures**

```bash
./build/calib --images cpp/calib/tests/fixtures/ --pattern 9x6 --square 25.0 --output /tmp/calib_test.json
cat /tmp/calib_test.json
```

Expected: Valid JSON with fx, fy, cx, cy, distortion, reprojection_error.

**Step 5: Commit fixtures**

```bash
git add cpp/calib/tests/fixtures/
git commit -m "test: add synthetic chessboard fixtures for C++ calibration tests"
```

---

### Task 7: View count stress sweep

**Files:**
- Run: `experiments/run_stress_view_count.py`
- Modify: `src/simtorecon/stress/view_sweep.py` (add evaluation integration)

**Context:** Requires baseline reconstruction from Task 4 to have succeeded. Runs 4 reconstructions at N∈{49, 30, 15, 8}.

**Step 1: Integrate evaluation into the sweep**

Modify `src/simtorecon/stress/view_sweep.py` to compute metrics for each view count. After the `runner.run()` call in the sweep loop, add:

```python
# Evaluate if GT available
if scene.has_ground_truth():
    import open3d as o3d
    from simtorecon.evaluation.alignment import align_to_gt
    from simtorecon.evaluation.metrics import (
        chamfer_distance, accuracy, completeness, f_score,
    )

    pred = o3d.io.read_point_cloud(str(result.output_ply))
    gt = scene.get_ground_truth()
    aligned, _ = align_to_gt(pred, gt)

    result.chamfer = chamfer_distance(aligned, gt)
    result.accuracy = accuracy(aligned, gt)
    result.completeness = completeness(aligned, gt)
    result.f_score = f_score(aligned, gt, threshold=1.0)
```

**Step 2: Run the sweep**

```bash
python experiments/run_stress_view_count.py
```

Expected: ~60 min total. Produces `results/stress_view_count/summary.json` with metrics for all 4 view counts.

**Step 3: Check results**

```bash
python -c "
import json
with open('results/stress_view_count/summary.json') as f:
    results = json.load(f)
for r in results:
    print(f'n={r[\"n_views\"]:3d}: F={r.get(\"f_score\", \"N/A\"):>6s}  chamfer={r.get(\"chamfer\", \"N/A\"):>8s}  pts={r[\"n_points\"]:>8d}  time={r[\"elapsed_seconds\"]:>6.1f}s')
"
```

Expected: F-score decreasing as views decrease. If n=8 fails, that's an honest result — report it.

**Step 4: Commit**

```bash
git add src/simtorecon/stress/view_sweep.py results/stress_view_count/summary.json
git commit -m "feat: view count stress sweep with evaluation"
```

---

### Task 8: Generate figures

**Files:**
- Modify: `experiments/generate_figures.py` (add failure region gallery)
- Output: `docs/figures/robustness_curve.png`, `docs/figures/failure_regions_n8.png`

**Step 1: Update figure generator for failure regions**

Add to `experiments/generate_figures.py`:

```python
def plot_failure_regions(result_dir: Path, output_path: Path) -> None:
    """Render failure regions for lowest view count reconstruction."""
    import open3d as o3d
    from simtorecon.evaluation.failure import per_point_error, visualize_failures
    from simtorecon.evaluation.alignment import align_to_gt
    from simtorecon.data.dtu import DTUScene

    # Find the lowest view count result
    scene = DTUScene(Path("data/dtu/scan9"))
    gt = scene.get_ground_truth()

    # Load the n=8 reconstruction (or lowest available)
    for n in [8, 15, 30]:
        ply = result_dir / f"views_{n:03d}" / "dense.ply"
        if ply.exists():
            pred = o3d.io.read_point_cloud(str(ply))
            aligned, _ = align_to_gt(pred, gt)
            errors = per_point_error(aligned, gt)
            colored = visualize_failures(aligned, errors, str(output_path))
            print(f"Saved failure regions (n={n}): {output_path}")
            return

    print("No reconstruction results found for failure visualization.")
```

**Step 2: Run figure generation**

```bash
python experiments/generate_figures.py \
    --results results/stress_view_count/summary.json \
    --output-dir docs/figures
```

Expected: `docs/figures/robustness_curve.png` generated.

**Step 3: Commit figures**

```bash
git add experiments/generate_figures.py docs/figures/
git commit -m "feat: generate robustness curve and failure region figures"
```

---

### Task 9: Complete DECISIONS.md

**Files:**
- Modify: `DECISIONS.md`

Add entries 4-10 (entry 5 was added in Task 5):

```markdown
## 4. CUDA GPU required for dense reconstruction
(from Task 1)

## 6. C++17 + OpenCV-C++ for calibration

**Decision:** Write a standalone C++ calibration binary rather than using Python's OpenCV.

**Rationale:** C++ code is the differentiator in the target hiring funnel (Sereact, KONUX,
Helsing, Spleenlab). The same calibration could be done in 10 lines of Python, but the
C++ implementation signals competence in systems-level code, CMake, GoogleTest, and
header-only dependency management.

## 7. GoogleTest via FetchContent over system install

**Decision:** Fetch GoogleTest and other C++ dependencies (nlohmann/json, cxxopts) via
CMake FetchContent rather than requiring system installs.

**Rationale:** Reproducibility. Anyone who can run CMake can build and test the project
without manual dependency installation. This is the standard modern CMake pattern.

## 8. JSON results files over MLflow

**Decision:** Store experiment results as plain JSON files in `results/`.

**Rationale:** V1 has one method and one stress dimension. MLflow's overhead
(server, UI, Python dependency) is not justified. JSON files are inspectable,
diffable, and version-controllable. MLflow comes in V1.5 when multiple methods
and stress dimensions justify the tracking infrastructure.

## 9. Image downsampling to 800x600

**Decision:** Downsample DTU images from 1600x1200 to 800x600 before reconstruction.

**Rationale:** Full-resolution PatchMatch on CPU-class hardware would take 60+ minutes
per reconstruction. 800x600 brings this to 15-20 minutes — fast enough for iterative
development and the 4-point stress sweep. Resolution is configurable in `PipelineConfig`.

## 10. Open3D for evaluation metrics

**Decision:** Use Open3D's `compute_point_cloud_distance` for nearest-neighbor computation
rather than implementing KD-tree search from scratch.

**Rationale:** Evaluation metrics are not the contribution. Using the canonical, C++-backed
implementation avoids bugs and runs in seconds on 800k-point clouds. The contribution is
the stress-test methodology, not the distance computation.
```

**Step 1: Write the entries**

Append entries 6-10 to `DECISIONS.md`.

**Step 2: Commit**

```bash
git add DECISIONS.md
git commit -m "docs: complete DECISIONS.md with entries 4-10"
```

---

### Task 10: Fix CI workflow

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml` (optional deps)

**Context:** The current CI will fail because:
1. `pycolmap` requires COLMAP C++ libs — not available via pip alone on ubuntu-latest
2. The reconstruction tests need CUDA — must be excluded from CI

**Step 1: Make pycolmap and open3d optional for unit tests**

Modify `pyproject.toml` to split dependencies:

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
pipeline = [
    "pycolmap>=0.6.1",
    "open3d>=0.17.0",
    "opencv-python>=4.8",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4.0",
    "open3d>=0.17.0",
    "opencv-python>=4.8",
]
all = [
    "simtorecon[pipeline,dev]",
]
```

**Step 2: Guard imports in test files**

For tests that require open3d (test_metrics.py, conftest.py), add:

```python
pytest.importorskip("open3d")
```

at the module level. This lets CI skip those tests if open3d is unavailable, while they still run locally.

**Step 3: Update CI workflow**

```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install ruff
      - run: ruff check src/ tests/ experiments/

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/ -v -m "not slow and not smoke"

  cpp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake libopencv-dev
      - name: Build
        run: |
          cmake -B build cpp/calib
          cmake --build build
      - name: Run tests
        run: cd build && ctest --output-on-failure
```

**Step 4: Run lint and tests locally to verify**

```bash
ruff check src/ tests/ experiments/
pytest tests/ -v -m "not slow and not smoke"
```

Expected: All pass.

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml tests/
git commit -m "fix: CI workflow — optional pycolmap, guard open3d imports"
```

---

### Task 11: Complete README with real numbers

**Files:**
- Modify: `README.md`

**Step 1: Populate the results table**

Read `results/stress_view_count/summary.json` and fill in the table:

```markdown
| ID | Setup | Views | Chamfer ↓ | Accuracy ↓ | Completeness ↓ | F-score ↑ | Time (min) |
|----|-------|-------|-----------|------------|----------------|-----------|------------|
| B1 | COLMAP MVS, full | 49 | X.XX | X.XX | X.XX | 0.XX | XX |
| B2 | COLMAP MVS, 30 views | 30 | X.XX | X.XX | X.XX | 0.XX | XX |
| B3 | COLMAP MVS, 15 views | 15 | X.XX | X.XX | X.XX | 0.XX | XX |
| B4 | COLMAP MVS, 8 views | 8 | X.XX | X.XX | X.XX | 0.XX | XX |
```

Replace X.XX with actual numbers from the sweep.

**Step 2: Write Summary of Findings**

Based on actual results, write 4-5 bullet points:

```markdown
### Summary of Findings

- COLMAP dense MVS on DTU scan9 achieves F-score = 0.XX at 49 views (full)
- View count reduction to 8 views drops F-score to 0.XX (ΔF = 0.XX)
- Failure regions concentrate on [actual finding from failure visualization]
- Reconstruction time scales linearly with view count: XX min (49) → XX min (8)
- [Any honest negative finding, e.g., "n=8 failed to converge"]
```

**Step 3: Embed figures**

```markdown
### Reconstruction quality vs view count

![Robustness curve](docs/figures/robustness_curve.png)

### Failure region analysis

At n=8, COLMAP loses [X]% of points relative to full-view baseline.
Failure regions concentrate on [finding].

![Failure regions](docs/figures/failure_regions_n8.png)
```

**Step 4: Verify README renders**

```bash
# Open in browser to check rendering
open README.md  # or use a markdown previewer
```

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README with real results, figures, and honest findings"
```

---

### Task 12: Final validation against V1 Definition of Done

**Files:** All

**Step 1: Run full quality check**

```bash
ruff check src/ tests/ experiments/
pytest tests/ -v -m "not slow and not smoke"
```

Expected: All pass.

**Step 2: Build C++ and run tests**

```bash
cmake -B build cpp/calib && cmake --build build
cd build && ctest --output-on-failure
```

Expected: 8 tests pass.

**Step 3: Walk the V1 Definition of Done checklist**

```
- [ ] CI is green on the main branch
- [ ] README is complete with real numbers and at least one failure figure
- [ ] DECISIONS.md has at least 8 entries
- [ ] C++ calibration binary builds, runs, produces valid JSON
- [ ] `python experiments/run_baseline.py --scene scan9` completes
- [ ] `python experiments/run_stress_view_count.py` completes the 4-point sweep
- [ ] `python experiments/generate_figures.py` produces both figures
- [ ] Honest Scope section is written and accurate
- [ ] All ~20 tests pass locally and on CI
- [ ] Repo is pinned to GitHub profile
```

**Step 4: The 90-second test**

Open the README in a private browser window. Scroll for 90 seconds. Ask:

> Would a hiring manager at ImFusion / Sereact / Helsing immediately see
> (a) you can ship a working reconstruction pipeline,
> (b) you write C++ when the role calls for it,
> (c) you evaluate honestly under stress, and
> (d) you document scope explicitly?

If yes → ship. If no → fix the gaps.

**Step 5: Push to GitHub**

```bash
git remote add origin git@github.com:tyy0811/sim-to-recon.git
git push -u origin main
```

**Step 6: Pin to GitHub profile**

Go to GitHub profile → Customize pinned repositories → Add sim-to-recon.

---

## Task Dependency Graph

```
Task 1 (GPU constraint)
  ↓
Task 3 (DTU data) ← independent
  ↓
Task 4 (COLMAP end-to-end) ← depends on Task 1 + Task 3
  ↓
Task 5 (Alignment + eval) ← depends on Task 4
  ↓
Task 7 (Stress sweep) ← depends on Task 5
  ↓
Task 8 (Figures) ← depends on Task 7
  ↓
Task 11 (README) ← depends on Task 8
  ↓
Task 12 (Final validation) ← depends on all

Independent tracks (can run in parallel):
- Task 2 (quaternion tests) — anytime
- Task 6 (C++ build) — anytime
- Task 9 (DECISIONS.md) — anytime
- Task 10 (CI fix) — anytime
```
