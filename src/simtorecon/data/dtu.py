"""DTU MVS dataset loader.

Supports two calibration formats:
1. MVSNet preprocessed (default): cam_XXXXXXXX.txt with 4x4 extrinsics,
   3x3 intrinsics, and depth range. This is the standard format from
   Yao et al. 2018, used by the learning-based MVS community.
2. DTU original: pos_XXX.txt with flattened 3x4 projection matrix P = K[R|t].

Reference:
  MVSNet: https://github.com/YoYo000/MVSNet
  DTU: https://roboimagedata.compute.dtu.dk/?page_id=36
"""

from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from simtorecon.data.adapters import DatasetAdapter


def download_scan9(dest: str | Path) -> None:
    """Check for and print instructions to obtain DTU scan9 data.

    DTU data requires manual download from the official website.
    This function checks what's present and prints instructions for missing files.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    images_dir = dest / "images"
    gt_dir = dest / "gt"

    if images_dir.exists() and len(list(images_dir.glob("*.png"))) >= 49:
        print(f"DTU scan9 images already present at {images_dir}")
    else:
        print("DTU scan9 images must be downloaded manually.")
        print("1. Visit: https://roboimagedata.compute.dtu.dk/?page_id=36")
        print("2. Download 'Rectified (Train)' dataset")
        print(f"3. Extract scan9 images to: {images_dir}/")
        print("   Images should be named rect_001_*.png through rect_049_*.png")

    if gt_dir.exists() and len(list(gt_dir.glob("*.ply"))) >= 1:
        print(f"DTU scan9 GT already present at {gt_dir}")
    else:
        gt_dir.mkdir(parents=True, exist_ok=True)
        print("DTU scan9 ground truth must be downloaded manually.")
        print("1. Visit: https://roboimagedata.compute.dtu.dk/?page_id=36")
        print("2. Download 'Points' dataset")
        print(f"3. Place stl009_total.ply in: {gt_dir}/")

    calib_dir = dest / "calibration"
    if not calib_dir.exists():
        calib_dir.mkdir(parents=True, exist_ok=True)
        print("DTU calibration files must be downloaded manually.")
        print("1. Visit: https://roboimagedata.compute.dtu.dk/?page_id=36")
        print("2. Download 'SampleSet' or 'Calibration' data")
        print(f"3. Place cal18/pos_*.txt files in: {calib_dir}/")


class DTUScene(DatasetAdapter):
    """Loader for a single DTU MVS scene (rectified format).

    Expected directory structure:
        root/
        ├── images/          # rect_001_*.png ... rect_049_*.png
        ├── calibration/     # pos_000.txt ... pos_048.txt
        └── gt/              # stl009_total.ply (or similar)
    """

    def __init__(self, root: Path, scan_id: int = 9, light_idx: int = 3) -> None:
        self._root = Path(root)
        self._scan_id = scan_id
        self._light_idx = light_idx

        # Auto-detect directory layout:
        # MVSNet preprocessed: root/Rectified/scanN_train/, root/Cameras/train/
        # Custom/local:        root/images/, root/calibration/
        self._images_dir, self._calib_dir, self._gt_dir = self._detect_layout()

        self._image_paths = self._discover_images()
        self._calibrations: dict[int, dict] = {}
        self._gt_cloud: o3d.geometry.PointCloud | None = None
        self._index_map: list[int] | None = None

    def _detect_layout(self) -> tuple[Path, Path, Path]:
        """Auto-detect whether this is MVSNet or custom directory layout."""
        # MVSNet layout: root/Rectified/scan{id}_train/
        mvsnet_images = self._root / "Rectified" / f"scan{self._scan_id}_train"
        mvsnet_cams = self._root / "Cameras" / "train"

        if mvsnet_images.exists():
            gt_dir = self._root / "gt"
            return mvsnet_images, mvsnet_cams, gt_dir

        # Custom/local layout: root/images/, root/calibration/
        return self._root / "images", self._root / "calibration", self._root / "gt"

    @property
    def name(self) -> str:
        return f"dtu_scan{self._scan_id}"

    @property
    def n_images(self) -> int:
        return len(self._image_paths)

    @property
    def root(self) -> Path:
        return self._root

    def _discover_images(self) -> list[Path]:
        """Find all rectified images for this scene, sorted by index."""
        if not self._images_dir.exists():
            return []
        # DTU rectified naming: rect_XXX_Y_r5000.png
        # XXX = image index (1-based), Y = lighting condition
        pattern = f"rect_*_{self._light_idx}_r5000.png"
        paths = sorted(self._images_dir.glob(pattern))
        if not paths:
            # Fall back to any available images
            paths = sorted(self._images_dir.glob("rect_*.png"))
        return paths

    def get_image(self, idx: int) -> np.ndarray:
        """Load image at index idx as H x W x 3 uint8 BGR array."""
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        img = cv2.imread(str(self._image_paths[idx]))
        if img is None:
            raise OSError(f"Failed to load image: {self._image_paths[idx]}")
        return img

    def get_image_path(self, idx: int) -> Path:
        """Return path to image at index idx."""
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        return self._image_paths[idx]

    def get_intrinsics(self, idx: int) -> np.ndarray:
        """Return 3x3 intrinsics matrix K for image idx."""
        calib = self._load_calibration(idx)
        return calib["K"]

    def get_pose(self, idx: int) -> np.ndarray:
        """Return 4x4 world-to-camera extrinsics [R|t] for image idx."""
        calib = self._load_calibration(idx)
        return calib["extrinsics"]

    def get_image_size(self) -> tuple[int, int]:
        """Return (height, width) of images."""
        if self.n_images == 0:
            raise RuntimeError("No images found")
        img = self.get_image(0)
        return img.shape[0], img.shape[1]

    def has_ground_truth(self) -> bool:
        return self._gt_dir.exists() and len(list(self._gt_dir.glob("*.ply"))) > 0

    def get_ground_truth(self) -> o3d.geometry.PointCloud:
        """Load ground truth point cloud."""
        if self._gt_cloud is not None:
            return self._gt_cloud

        ply_files = sorted(self._gt_dir.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No GT point cloud in {self._gt_dir}")

        self._gt_cloud = o3d.io.read_point_cloud(str(ply_files[0]))
        return self._gt_cloud

    def _load_calibration(self, idx: int) -> dict:
        """Load calibration for image idx.

        Supports two formats (auto-detected):
        1. MVSNet: cam_XXXXXXXX.txt with extrinsic (4x4), intrinsic (3x3), depth range
        2. DTU original: pos_XXX.txt with flattened 3x4 projection matrix
        """
        if idx in self._calibrations:
            return self._calibrations[idx]

        calib_idx = self._index_map[idx] if self._index_map is not None else idx

        # Try MVSNet format first (cam_00000000.txt)
        mvsnet_file = self._calib_dir / f"{calib_idx:08d}_cam.txt"
        if mvsnet_file.exists():
            result = self._parse_mvsnet_cam(mvsnet_file)
        else:
            # Fall back to DTU original format (pos_000.txt)
            dtu_file = self._calib_dir / f"pos_{calib_idx:03d}.txt"
            if dtu_file.exists():
                result = self._parse_dtu_pos(dtu_file)
            else:
                raise FileNotFoundError(
                    f"No calibration file found for index {calib_idx}. "
                    f"Tried: {mvsnet_file}, {dtu_file}"
                )

        self._calibrations[idx] = result
        return result

    @staticmethod
    def _parse_mvsnet_cam(path: Path) -> dict:
        """Parse MVSNet camera file format.

        Format:
            extrinsic
            E00 E01 E02 E03
            E10 E11 E12 E13
            E20 E21 E22 E23
            E30 E31 E32 E33

            intrinsic
            K00 K01 K02
            K10 K11 K12
            K20 K21 K22

            DEPTH_MIN DEPTH_INTERVAL
        """
        with open(path) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Find extrinsic block (4x4 matrix after "extrinsic" header)
        ext_start = None
        int_start = None
        for i, line in enumerate(lines):
            if line.lower() == "extrinsic":
                ext_start = i + 1
            elif line.lower() == "intrinsic":
                int_start = i + 1

        if ext_start is None or int_start is None:
            raise ValueError(f"Invalid MVSNet camera file: {path}")

        extrinsics = np.array(
            [[float(x) for x in lines[ext_start + r].split()] for r in range(4)]
        )

        K = np.array(
            [[float(x) for x in lines[int_start + r].split()] for r in range(3)]
        )

        # Depth range (optional, after intrinsic block)
        depth_min, depth_interval = 0.0, 1.0
        depth_line_idx = int_start + 3
        if depth_line_idx < len(lines):
            depth_parts = lines[depth_line_idx].split()
            if len(depth_parts) >= 2:
                depth_min = float(depth_parts[0])
                depth_interval = float(depth_parts[1])

        return {
            "K": K,
            "extrinsics": extrinsics,
            "depth_min": depth_min,
            "depth_interval": depth_interval,
        }

    @staticmethod
    def _parse_dtu_pos(path: Path) -> dict:
        """Parse DTU original calibration format (flattened 3x4 projection matrix)."""
        from scipy.linalg import rq

        P = np.loadtxt(path).reshape(3, 4)

        K, R = rq(P[:, :3])
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = t

        return {"K": K, "extrinsics": extrinsics, "P": P}

    def subsample(self, n_views: int, seed: int = 42) -> DTUScene:
        """Create a new DTUScene with a deterministic subset of views.

        Views are evenly spaced across the original sequence with a
        seeded random offset to vary which views are selected.
        """
        if n_views >= self.n_images:
            return self
        if n_views <= 0:
            raise ValueError(f"n_views must be positive, got {n_views}")

        rng = np.random.RandomState(seed)

        # Evenly spaced base indices with a random offset
        step = self.n_images / n_views
        offset = rng.uniform(0, step)
        indices = np.array([int(offset + i * step) for i in range(n_views)])
        indices = np.clip(indices, 0, self.n_images - 1)

        # Build the original-index map for calibration lookup
        # If self already has an index map, compose the mappings
        if self._index_map is not None:
            original_indices = [self._index_map[i] for i in indices]
        else:
            original_indices = indices.tolist()

        # Shallow copy — safe against future attribute additions
        sub = copy.copy(self)
        sub._image_paths = [self._image_paths[i] for i in indices]
        sub._calibrations = {}
        sub._gt_cloud = None
        sub._index_map = original_indices

        return sub

    def get_all_intrinsics(self) -> list[np.ndarray]:
        """Return list of 3x3 intrinsics matrices for all images."""
        return [self.get_intrinsics(i) for i in range(self.n_images)]

    def get_all_poses(self) -> list[np.ndarray]:
        """Return list of 4x4 extrinsics matrices for all images."""
        return [self.get_pose(i) for i in range(self.n_images)]
