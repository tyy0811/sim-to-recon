"""Generate a small synthetic scene for local smoke testing.

Creates 10 images of a synthetic textured cube with known camera poses
and a ground truth point cloud. ~5MB total, exercises the full pipeline
without needing DTU data.

Usage:
    python scripts/generate_synthetic_scene.py
"""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def make_camera_matrix(fx: float = 500.0, fy: float = 500.0,
                       cx: float = 320.0, cy: float = 240.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Compute a 4x4 world-to-camera matrix looking from eye at target."""
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=0)
    t = -R @ eye

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def render_points(points_3d: np.ndarray, colors: np.ndarray,
                  K: np.ndarray, pose: np.ndarray,
                  width: int, height: int) -> np.ndarray:
    """Render 3D points onto a 2D image by projection (simple z-buffer)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray bg

    R = pose[:3, :3]
    t = pose[:3, 3]

    # Transform points to camera frame
    pts_cam = (R @ points_3d.T).T + t

    # Project
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    cols = colors[valid]

    px = (K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]).astype(int)
    py = (K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]).astype(int)

    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = px[in_bounds], py[in_bounds]
    cols = cols[in_bounds]
    depths = pts_cam[in_bounds, 2]

    # Sort by depth (far to near) for simple z-buffer
    order = np.argsort(-depths)
    for i in order:
        cv2.circle(img, (px[i], py[i]), 2, cols[i].tolist(), -1)

    return img


def generate_synthetic_scene(output_dir: str = "data/synthetic_smoke"):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    calib_dir = output_dir / "calibration"
    gt_dir = output_dir / "gt"

    for d in [images_dir, calib_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate a textured cube as a dense point cloud
    rng = np.random.RandomState(42)
    n_pts_per_face = 5000
    points = []
    colors = []

    # 6 faces of a unit cube centered at origin
    face_colors = [
        [255, 50, 50], [50, 255, 50], [50, 50, 255],
        [255, 255, 50], [255, 50, 255], [50, 255, 255],
    ]
    for axis in range(3):
        for sign in [-0.5, 0.5]:
            pts = rng.uniform(-0.5, 0.5, (n_pts_per_face, 3))
            pts[:, axis] = sign
            face_idx = axis * 2 + (0 if sign < 0 else 1)
            col = np.array(face_colors[face_idx])
            # Add texture noise
            col = np.clip(col + rng.randint(-30, 30, (n_pts_per_face, 3)), 0, 255)
            points.append(pts)
            colors.append(col)

    points_3d = np.vstack(points)
    colors_3d = np.vstack(colors).astype(int)

    # Save GT point cloud
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(points_3d)
    gt_pcd.colors = o3d.utility.Vector3dVector(colors_3d / 255.0)
    o3d.io.write_point_cloud(str(gt_dir / "gt.ply"), gt_pcd)
    print(f"GT point cloud: {len(gt_pcd.points)} points")

    # Generate camera views around the cube
    K = make_camera_matrix()
    n_views = 10
    width, height = 640, 480

    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        radius = 2.0
        eye = np.array([radius * np.cos(angle), 0.3, radius * np.sin(angle)])
        target = np.array([0.0, 0.0, 0.0])
        pose = look_at(eye, target)

        # Render image
        img = render_points(points_3d, colors_3d, K, pose, width, height)
        img_name = f"rect_{i + 1:03d}_3_r5000.png"
        cv2.imwrite(str(images_dir / img_name), img)

        # Save calibration as 3x4 projection matrix (flattened)
        P = K @ pose[:3, :]
        np.savetxt(calib_dir / f"pos_{i:03d}.txt", P.flatten())

        print(f"View {i}: {img_name}")

    print(f"\nSynthetic scene written to {output_dir}")
    print(f"  {n_views} images ({width}x{height})")
    print(f"  {n_views} calibration files")
    print(f"  GT: {len(gt_pcd.points)} points")


if __name__ == "__main__":
    generate_synthetic_scene()
