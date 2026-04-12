"""Minimal reconstruction test on Modal — 3 images, verifies the full pipeline."""

import modal

app = modal.App("test-recon-minimal")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libsm6", "libice6")
    .pip_install(
        "pycolmap-cuda12", "numpy>=1.24",
        "opencv-python-headless>=4.8", "scipy>=1.10",
    )
)

dtu_vol = modal.Volume.from_name("simtorecon-dtu-data")


@app.function(image=image, gpu="A10G", timeout=600, volumes={"/dtu_data": dtu_vol})
def test_minimal_recon():
    import os
    import re

    import cv2
    import numpy as np
    import pycolmap
    from scipy.linalg import rq

    print(f"pycolmap {pycolmap.COLMAP_version}, has_cuda={pycolmap.has_cuda}")

    # Use 3 images from DTU scan9
    images_dir = "/dtu_data/scan9/images"
    calib_dir = "/dtu_data/scan9/calibration"

    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and "_3_" in f  # light_idx=3
    ])
    print(f"Available images: {len(all_images)}")
    print(f"First 3: {all_images[:3]}")

    # Use spread-out images for better stereo baseline
    n = len(all_images)
    selected = [all_images[i] for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]]
    print(f"Selected {len(selected)} spread-out images: {selected}")

    # Build workspace
    workspace = "/tmp/test_workspace"
    ws_images = f"{workspace}/images"
    sparse_dir = f"{workspace}/sparse/0"
    os.makedirs(ws_images, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    target_w, target_h = 400, 300
    cameras_lines = []
    images_lines = []

    for new_idx, img_file in enumerate(selected):
        img = cv2.imread(f"{images_dir}/{img_file}")
        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        out_name = f"{new_idx:04d}.png"
        cv2.imwrite(f"{ws_images}/{out_name}", img)

        # Get image ID from filename
        id_match = re.search(r"rect_(\d+)", img_file)
        img_1based = int(id_match.group(1))

        # Load calibration
        calib_file = f"{calib_dir}/pos_{img_1based:03d}.txt"
        if not os.path.exists(calib_file):
            print(f"  WARNING: {calib_file} not found, trying 0-indexed")
            calib_file = f"{calib_dir}/pos_{img_1based - 1:03d}.txt"

        P = np.loadtxt(calib_file).reshape(3, 4)
        K, R = rq(P[:, :3])
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])

        # Scale intrinsics
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        K[0, 0] *= scale_x
        K[1, 1] *= scale_y
        K[0, 2] *= scale_x
        K[1, 2] *= scale_y

        cam_id = new_idx + 1
        cameras_lines.append(
            f"{cam_id} PINHOLE {target_w} {target_h} "
            f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}"
        )

        # Quaternion
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            qw, qx, qy, qz = 1, 0, 0, 0  # fallback

        images_lines.append(
            f"{cam_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {out_name}"
        )
        images_lines.append("")

        cam_center = -R.T @ t
        depth_to_origin = np.linalg.norm(cam_center)
        print(f"  [{new_idx}] {img_file} -> K diag=({K[0,0]:.0f},{K[1,1]:.0f})"
              f" cam_center={cam_center} dist_to_origin={depth_to_origin:.1f}")

    # Write COLMAP files
    with open(f"{sparse_dir}/cameras.txt", "w") as f:
        f.write("\n".join(cameras_lines) + "\n")
    with open(f"{sparse_dir}/images.txt", "w") as f:
        f.write("\n".join(images_lines) + "\n")
    with open(f"{sparse_dir}/points3D.txt", "w") as f:
        f.write("# empty\n")

    # Undistort
    print("\n--- Undistort ---")
    dense_dir = f"{workspace}/dense"
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = target_w
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=sparse_dir,
        image_path=ws_images,
        undistort_options=undistort_opts,
    )
    print(f"Dense dir: {os.listdir(dense_dir)}")

    # Write manual patch-match.cfg
    stereo_dir = f"{dense_dir}/stereo"
    dense_images = sorted(os.listdir(f"{dense_dir}/images"))
    cfg_lines = []
    for ref in dense_images:
        cfg_lines.append(ref)
        sources = [s for s in dense_images if s != ref]
        cfg_lines.append(", ".join(sources))
    with open(f"{stereo_dir}/patch-match.cfg", "w") as f:
        f.write("\n".join(cfg_lines) + "\n")
    print(f"patch-match.cfg: {len(dense_images)} entries")

    # PatchMatch
    print("\n--- PatchMatch ---")
    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = target_w
    pm_opts.num_iterations = 3
    pm_opts.depth_min = 200.0
    pm_opts.depth_max = 1200.0

    # Pass 1: photometric (no geom consistency)
    pm_opts.geom_consistency = False
    print("PatchMatch pass 1 (photometric)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # Pass 2: geometric consistency
    pm_opts.geom_consistency = True
    print("PatchMatch pass 2 (geometric)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # Check depth maps
    depth_dir = f"{stereo_dir}/depth_maps"
    if os.path.exists(depth_dir):
        depth_files = os.listdir(depth_dir)
        print(f"Depth maps: {len(depth_files)} files")
    else:
        print("WARNING: no depth maps generated")

    # Fusion
    print("\n--- Fusion ---")
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = 2

    fused_dir = f"{dense_dir}/fused"
    os.makedirs(fused_dir, exist_ok=True)

    # Try geometric first (default), fall back to photometric
    recon = pycolmap.stereo_fusion(
        output_path=fused_dir,
        workspace_path=dense_dir,
        options=fusion_opts,
        input_type="geometric",
    )
    n_geo = len(recon.points3D)
    print(f"Geometric fusion: {n_geo} points")

    if n_geo == 0:
        print("Trying photometric fusion...")
        recon = pycolmap.stereo_fusion(
            output_path=fused_dir,
            workspace_path=dense_dir,
            options=fusion_opts,
            input_type="photometric",
        )
        print(f"Photometric fusion: {len(recon.points3D)} points")

    n_points = len(recon.points3D)
    print(f"Fused points: {n_points}")
    print(f"Fused dir: {os.listdir(fused_dir)}")

    # Export as PLY using Reconstruction.write
    recon_attrs = [a for a in dir(recon) if "write" in a.lower() or "export" in a.lower() or "ply" in a.lower()]
    print(f"Write/export methods: {recon_attrs}")

    # Try write method
    if hasattr(recon, "write"):
        ply_dir = f"{dense_dir}/ply_export"
        os.makedirs(ply_dir, exist_ok=True)
        recon.write(ply_dir)
        print(f"PLY export dir: {os.listdir(ply_dir)}")

    # Also try extracting points directly
    if n_points > 0:
        import numpy as np
        pts = np.array([p.xyz for p in recon.points3D.values()])
        print(f"Points shape: {pts.shape}")
        print(f"Points range: {pts.min(axis=0)} to {pts.max(axis=0)}")

    return "done"


@app.local_entrypoint()
def main():
    result = test_minimal_recon.remote()
    print(result)
