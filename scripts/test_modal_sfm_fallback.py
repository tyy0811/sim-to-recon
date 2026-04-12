"""Test: let COLMAP do its own SfM instead of providing poses.

If SfM + dense MVS produces points, our pose decomposition is the problem.
"""

import modal

app = modal.App("test-sfm")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libsm6", "libice6")
    .pip_install(
        "pycolmap-cuda12", "numpy>=1.24",
        "opencv-python-headless>=4.8",
    )
)

dtu_vol = modal.Volume.from_name("simtorecon-dtu-data")
ws_vol = modal.Volume.from_name("simtorecon-workspace", create_if_missing=True)


@app.function(
    image=image, gpu="A10G", timeout=1800,
    volumes={"/dtu_data": dtu_vol, "/workspace": ws_vol},
)
def test_sfm():
    import os
    import re

    import cv2
    import numpy as np
    import pycolmap

    print(f"pycolmap {pycolmap.COLMAP_version}, has_cuda={pycolmap.has_cuda}")

    # Prepare 10 images at low res
    images_dir = "/dtu_data/scan9/images"
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and "_3_" in f
    ])
    n = len(all_images)
    selected = [all_images[i] for i in range(0, n, n // 10)][:10]
    print(f"Using {len(selected)} images for SfM: {selected}")

    workspace = "/tmp/sfm_test"
    ws_images = f"{workspace}/images"
    os.makedirs(ws_images, exist_ok=True)

    target_w, target_h = 400, 300
    for img_file in selected:
        img = cv2.imread(f"{images_dir}/{img_file}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{ws_images}/{img_file}", img)

    # Run COLMAP SfM
    db_path = f"{workspace}/database.db"
    sparse_dir = f"{workspace}/sparse"
    os.makedirs(sparse_dir, exist_ok=True)

    print("\n--- Feature extraction ---")
    pycolmap.extract_features(db_path, ws_images)

    print("--- Exhaustive matching ---")
    pycolmap.match_exhaustive(db_path)

    print("--- Incremental mapping ---")
    maps = pycolmap.incremental_mapping(
        db_path, ws_images, sparse_dir,
        options=pycolmap.IncrementalPipelineOptions(),
    )
    print(f"Reconstructions: {len(maps)}")

    if not maps:
        print("SfM FAILED — no reconstruction")
        return "sfm_failed"

    # Use the largest reconstruction
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    print(f"Best: {best.num_reg_images()} images, {len(best.points3D)} points")

    # Write sparse model
    best_dir = f"{sparse_dir}/0"
    os.makedirs(best_dir, exist_ok=True)
    best.write(best_dir)

    # Undistort
    dense_dir = f"{workspace}/dense"
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = target_w
    print("\n--- Undistort ---")
    pycolmap.undistort_images(
        output_path=dense_dir, input_path=best_dir,
        image_path=ws_images, undistort_options=undistort_opts,
    )

    # PatchMatch (no manual depth range needed — SfM provides 3D points)
    print("--- PatchMatch (photometric) ---")
    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = target_w
    pm_opts.num_iterations = 5
    pm_opts.geom_consistency = False
    pm_opts.filter = True
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    print("--- PatchMatch (geometric) ---")
    pm_opts.geom_consistency = True
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # Fusion
    print("--- Fusion ---")
    fused_dir = f"{dense_dir}/fused"
    os.makedirs(fused_dir, exist_ok=True)
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = 3

    recon = pycolmap.stereo_fusion(
        output_path=fused_dir, workspace_path=dense_dir, options=fusion_opts,
    )
    n_points = len(recon.points3D)
    print(f"\nFused points: {n_points}")

    if n_points > 0:
        pts = np.array([p.xyz for p in recon.points3D.values()])
        print(f"Range: {pts.min(axis=0)} to {pts.max(axis=0)}")

        ply_path = "/workspace/sfm_test_fused.ply"
        recon.export_PLY(ply_path)
        ws_vol.commit()
        print(f"Exported PLY: {os.path.getsize(ply_path)} bytes")

    return f"sfm={best.num_reg_images()} imgs, {len(best.points3D)} sparse pts, fused={n_points}"


@app.local_entrypoint()
def main():
    print(test_sfm.remote())
