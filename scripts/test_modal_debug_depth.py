"""Debug depth range: read MVSNet camera files + inspect depth maps."""

import modal

app = modal.App("debug-depth")

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
def debug_depth():
    import os
    import re
    import struct

    import cv2
    import numpy as np
    import pycolmap
    from scipy.linalg import rq

    # --- Step 1: Read MVSNet camera file to get depth range ---
    calib_dir = "/dtu_data/scan9/calibration"
    calib_files = sorted(os.listdir(calib_dir))
    print(f"Calibration files: {calib_files[:5]}...")
    print(f"Total: {len(calib_files)}")

    # Read first calibration file raw
    first_calib = f"{calib_dir}/{calib_files[0]}"
    print(f"\n=== Raw content of {calib_files[0]} ===")
    with open(first_calib) as f:
        content = f.read()
    print(content[:500])

    # Parse all calibration files to get camera distances
    print("\n=== Camera positions ===")
    images_dir = "/dtu_data/scan9/images"
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and "_3_" in f
    ])

    distances = []
    for img_file in all_images[:5]:
        id_match = re.search(r"rect_(\d+)", img_file)
        img_id = int(id_match.group(1))
        calib_file = f"{calib_dir}/pos_{img_id:03d}.txt"
        if not os.path.exists(calib_file):
            print(f"  {img_file}: calib not found at {calib_file}")
            continue

        P = np.loadtxt(calib_file).reshape(3, 4)
        K, R = rq(P[:, :3])
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])
        cam_center = -R.T @ t
        dist = np.linalg.norm(cam_center)
        distances.append(dist)
        print(f"  {img_file}: dist={dist:.1f}, t={t}")

    if distances:
        print(f"\n  Distance range: {min(distances):.1f} - {max(distances):.1f}")
        print(f"  Suggested depth_min: {min(distances) * 0.3:.1f}")
        print(f"  Suggested depth_max: {max(distances) * 2.0:.1f}")

    # --- Step 2: Run PatchMatch with correct depth range and inspect depth maps ---
    print("\n=== Building workspace with 5 views ===")
    n = len(all_images)
    selected = [all_images[i] for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]]

    workspace = "/tmp/debug_workspace"
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

        id_match = re.search(r"rect_(\d+)", img_file)
        img_1based = int(id_match.group(1))
        calib_file = f"{calib_dir}/pos_{img_1based:03d}.txt"

        P = np.loadtxt(calib_file).reshape(3, 4)
        K, R = rq(P[:, :3])
        T_sign = np.diag(np.sign(np.diag(K)))
        K = K @ T_sign
        R = T_sign @ R
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])

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
            qw, qx, qy, qz = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw, qx, qy, qz = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw, qx, qy, qz = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw, qx, qy, qz = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s

        images_lines.append(
            f"{cam_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {out_name}"
        )
        images_lines.append("")

    with open(f"{sparse_dir}/cameras.txt", "w") as f:
        f.write("\n".join(cameras_lines) + "\n")
    with open(f"{sparse_dir}/images.txt", "w") as f:
        f.write("\n".join(images_lines) + "\n")
    with open(f"{sparse_dir}/points3D.txt", "w") as f:
        f.write("# empty\n")

    # Undistort
    dense_dir = f"{workspace}/dense"
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = target_w
    pycolmap.undistort_images(
        output_path=dense_dir,
        input_path=sparse_dir,
        image_path=ws_images,
        undistort_options=undistort_opts,
    )

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

    # PatchMatch with depth from camera distances
    d_min = min(distances) * 0.3 if distances else 100.0
    d_max = max(distances) * 2.0 if distances else 2000.0
    print(f"\n=== PatchMatch with depth_min={d_min:.1f}, depth_max={d_max:.1f} ===")

    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = target_w
    pm_opts.num_iterations = 5
    pm_opts.filter = True
    pm_opts.depth_min = d_min
    pm_opts.depth_max = d_max

    # Pass 1: photometric
    pm_opts.geom_consistency = False
    print("Pass 1 (photometric)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # Pass 2: geometric
    pm_opts.geom_consistency = True
    print("Pass 2 (geometric)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # --- Step 3: Inspect depth maps ---
    print("\n=== Depth map inspection ===")
    depth_dir = f"{stereo_dir}/depth_maps"
    if os.path.exists(depth_dir):
        depth_files = sorted(os.listdir(depth_dir))
        print(f"Depth map files: {depth_files}")

        for df in depth_files:
            depth_path = f"{depth_dir}/{df}"
            fsize = os.path.getsize(depth_path)
            print(f"  {df}: {fsize} bytes")

            if df.endswith(".geometric.bin") and fsize > 0:
                with open(depth_path, "rb") as f:
                    first_40 = f.read(40)
                hex_str = first_40.hex()
                print(f"    First 40 bytes hex: {hex_str}")
                # Try interpreting as different formats
                ints = struct.unpack("iii", first_40[:12])
                print(f"    As 3 ints: {ints}")
                floats = struct.unpack("10f", first_40)
                print(f"    As 10 floats: {[f'{x:.4f}' for x in floats]}")
                # Expected image size 400x300 = 120000 pixels * 4 bytes = 480000
                expected_no_header = 400 * 300 * 4
                print(f"    File size {fsize} vs 400*300*4={expected_no_header}")
                if abs(fsize - expected_no_header) < 100:
                    print("    -> Likely NO header, raw float32 data")
                    with open(depth_path, "rb") as f:
                        data = np.frombuffer(f.read(), dtype=np.float32)
                    depth_map = data.reshape(300, 400)
                    valid = depth_map[depth_map > 0]
                    print(f"    Valid: {len(valid)}/{depth_map.size} "
                          f"({100*len(valid)/depth_map.size:.1f}%)")
                    if len(valid) > 0:
                        print(f"    Depth range: {valid.min():.1f} - {valid.max():.1f}")
                break
    else:
        print("No depth_maps directory!")

    # --- Step 4: Fusion ---
    print("\n=== Fusion ===")
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = 2
    fusion_opts.max_reproj_error = 2.0
    fusion_opts.max_depth_error = 0.05  # relaxed from 0.01

    fused_dir = f"{dense_dir}/fused"
    os.makedirs(fused_dir, exist_ok=True)
    recon = pycolmap.stereo_fusion(
        output_path=fused_dir,
        workspace_path=dense_dir,
        options=fusion_opts,
    )

    n_points = len(recon.points3D)
    print(f"Fused points: {n_points}")

    if n_points > 0:
        pts = np.array([p.xyz for p in recon.points3D.values()])
        print(f"Point cloud range: {pts.min(axis=0)} to {pts.max(axis=0)}")

        # Export PLY
        recon.export_PLY(f"{dense_dir}/fused.ply")
        ply_size = os.path.getsize(f"{dense_dir}/fused.ply")
        print(f"Exported PLY: {ply_size} bytes")

    return f"depth_range=({d_min:.1f},{d_max:.1f}), fused={n_points}"


@app.local_entrypoint()
def main():
    result = debug_depth.remote()
    print(f"\nResult: {result}")
