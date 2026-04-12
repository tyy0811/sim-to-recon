"""COLMAP dense MVS pipeline orchestration.

The key trick for posed-image MVS: pre-populate COLMAP's database with
known cameras, images, and poses from DTU calibration, then skip SfM
entirely and go straight to undistort -> PatchMatch -> fusion.

PatchMatch stereo requires CUDA. Two execution modes:
- Modal (default): upload workspace to Modal, run on A10G GPU, download result.
- Local: requires a machine with CUDA GPU and pycolmap built with CUDA.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from simtorecon.data.adapters import DatasetAdapter
from simtorecon.pipeline.schemas import PipelineConfig, ReconstructionResult


class ColmapRunner:
    """Orchestrates COLMAP dense MVS on a posed-image dataset."""

    def __init__(self, scene: DatasetAdapter, config: PipelineConfig) -> None:
        self.scene = scene
        self.config = config

    def run(self, output_dir: Path) -> ReconstructionResult:
        """Run the full dense MVS pipeline.

        Flow:
        1. Prepare workspace (write images + COLMAP sparse model)
        2. Undistort images (local, CPU)
        3. PatchMatch stereo + fusion (Modal GPU or local CUDA)
        4. Load result

        Returns a ReconstructionResult with the output .ply path and timing.
        """
        import pycolmap

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ply = output_dir / "dense.ply"

        start = time.time()

        with tempfile.TemporaryDirectory(prefix="simtorecon_") as tmpdir:
            workspace = Path(tmpdir)

            # Step 1: Prepare workspace — write images and COLMAP sparse model
            self._prepare_workspace(workspace)

            # Step 2: Undistort images (CPU, works without CUDA)
            sparse_dir = workspace / "sparse" / "0"
            image_dir = workspace / "images"
            undistorted_dir = workspace / "dense"

            undistort_opts = pycolmap.UndistortCameraOptions()
            undistort_opts.max_image_size = self.config.max_image_size
            pycolmap.undistort_images(
                output_path=str(undistorted_dir),
                input_path=str(sparse_dir),
                image_path=str(image_dir),
                undistort_options=undistort_opts,
            )

            # Step 3+4: PatchMatch + Fusion
            if self.config.use_modal:
                fused_ply = self._run_on_modal(undistorted_dir)
            else:
                fused_ply = self._run_local(undistorted_dir, pycolmap)

            # Copy result to output directory
            shutil.copy2(fused_ply, output_ply)

        elapsed = time.time() - start

        # Count points in output
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(output_ply))
        n_points = len(pcd.points)

        return ReconstructionResult(
            scene_name=self.scene.name,
            n_views=self.scene.n_images,
            n_points=n_points,
            output_ply=output_ply,
            elapsed_seconds=elapsed,
            config_hash=self._config_hash(),
        )

    def _run_on_modal(self, undistorted_dir: Path) -> Path:
        """Upload workspace to Modal, run PatchMatch + fusion on GPU, download result."""
        import modal

        volume = modal.Volume.from_name("simtorecon-workspace", create_if_missing=True)

        # Unique subdirectory for this run
        run_id = uuid.uuid4().hex[:8]
        remote_subdir = f"run_{run_id}"

        # Upload the undistorted workspace to the volume
        print(f"Uploading workspace to Modal volume ({remote_subdir})...")
        with volume.batch_upload() as batch:
            batch.put_directory(str(undistorted_dir), remote_subdir)

        # Call the Modal function
        from modal_app import run_patch_match_and_fusion

        print("Running PatchMatch + fusion on Modal A10G...")
        result = run_patch_match_and_fusion.remote(
            workspace_subdir=remote_subdir,
            max_image_size=self.config.max_image_size,
            geom_consistency=self.config.geom_consistency,
            num_iterations=self.config.num_iterations,
            min_num_pixels=self.config.min_num_pixels,
            max_reproj_error=self.config.max_reproj_error,
            max_depth_error=self.config.max_depth_error,
        )

        if not result["success"]:
            raise RuntimeError(f"Modal PatchMatch failed: {result['error']}")

        print(f"Modal complete: {result['n_points']} points fused.")

        # Download the fused PLY from the volume
        fused_ply = undistorted_dir / "fused.ply"
        remote_ply_path = result["fused_ply_path"]

        for entry in volume.read_file(remote_ply_path):
            with open(fused_ply, "wb") as f:
                f.write(entry)

        return fused_ply

    def _run_local(self, undistorted_dir: Path, pycolmap) -> Path:
        """Run PatchMatch + fusion locally (requires CUDA GPU)."""
        # PatchMatch stereo
        options = pycolmap.PatchMatchOptions()
        options.max_image_size = self.config.max_image_size
        options.geom_consistency = self.config.geom_consistency
        options.num_iterations = self.config.num_iterations

        pycolmap.patch_match_stereo(
            workspace_path=str(undistorted_dir),
            options=options,
        )

        # Stereo fusion
        fused_ply = undistorted_dir / "fused.ply"
        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.min_num_pixels = self.config.min_num_pixels
        fusion_options.max_reproj_error = self.config.max_reproj_error
        fusion_options.max_depth_error = self.config.max_depth_error

        pycolmap.stereo_fusion(
            output_path=str(fused_ply),
            workspace_path=str(undistorted_dir),
            options=fusion_options,
        )

        return fused_ply

    def _prepare_workspace(self, workspace: Path) -> None:
        """Create COLMAP workspace with known cameras, images, and poses.

        This is the core of posed-image MVS: we bypass SfM by writing
        the sparse reconstruction directly from DTU calibration.
        """
        image_dir = workspace / "images"
        image_dir.mkdir(parents=True)
        sparse_dir = workspace / "sparse" / "0"
        sparse_dir.mkdir(parents=True)

        # Write downsampled images
        h, w = self.scene.get_image_size()
        th, tw = self.config.target_height, self.config.target_width
        scale_x = tw / w
        scale_y = th / h

        cameras_txt_lines = []
        images_txt_lines = []

        for idx in range(self.scene.n_images):
            img = self.scene.get_image(idx)
            if (h, w) != (th, tw):
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

            img_name = f"{idx:04d}.png"
            cv2.imwrite(str(image_dir / img_name), img)

            # Scale intrinsics to match downsampled resolution
            K = self.scene.get_intrinsics(idx).copy()
            K[0, 0] *= scale_x  # fx
            K[1, 1] *= scale_y  # fy
            K[0, 2] *= scale_x  # cx
            K[1, 2] *= scale_y  # cy

            # COLMAP camera (PINHOLE model: fx, fy, cx, cy)
            cam_id = idx + 1
            cameras_txt_lines.append(
                f"{cam_id} PINHOLE {tw} {th} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}"
            )

            # Convert pose to COLMAP format (quaternion + translation)
            pose = self.scene.get_pose(idx)
            R = pose[:3, :3]
            t = pose[:3, 3]
            quat = self._rotation_to_quaternion(R)

            img_id = idx + 1
            images_txt_lines.append(
                f"{img_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                f"{t[0]} {t[1]} {t[2]} {cam_id} {img_name}"
            )
            # Empty line for 2D points (none for posed images)
            images_txt_lines.append("")

        # Write COLMAP text format files
        with open(sparse_dir / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {self.scene.n_images}\n")
            for line in cameras_txt_lines:
                f.write(line + "\n")

        with open(sparse_dir / "images.txt", "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {self.scene.n_images}\n")
            for line in images_txt_lines:
                f.write(line + "\n")

        with open(sparse_dir / "points3D.txt", "w") as f:
            f.write("# 3D point list (empty for posed-image MVS)\n")

    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    def _config_hash(self) -> str:
        """Deterministic hash of the pipeline configuration."""
        config_str = self.config.model_dump_json(exclude={"workspace"})
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
