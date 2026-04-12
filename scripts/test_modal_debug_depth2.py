"""Targeted depth map diagnosis — no guessing."""

import modal

app = modal.App("debug-depth2")

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
def diagnose():
    import os
    import re

    import cv2
    import numpy as np
    import pycolmap
    from scipy.linalg import rq

    def read_colmap_depth_map(path):
        with open(path, "rb") as f:
            header = b""
            ampersands = 0
            while ampersands < 3:
                c = f.read(1)
                if c == b"&":
                    ampersands += 1
                header += c
            w, h, ch = map(int, header[:-1].split(b"&"))
            data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape(h, w, ch).squeeze()

    # --- Step 1: Read MVSNet camera file for depth range ---
    calib_dir = "/dtu_data/scan9/calibration"
    first_file = sorted(os.listdir(calib_dir))[0]
    print(f"=== Calibration file: {first_file} ===")
    with open(f"{calib_dir}/{first_file}") as f:
        print(f.read())

    # --- Step 2: Build workspace with 5 views ---
    images_dir = "/dtu_data/scan9/images"
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and "_3_" in f
    ])
    n = len(all_images)
    selected = [all_images[i] for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]]
    print(f"\nUsing {len(selected)} images: {selected}")

    workspace = "/tmp/ws"
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

        img_id = int(re.search(r"rect_(\d+)", img_file).group(1))
        P = np.loadtxt(f"{calib_dir}/pos_{img_id:03d}.txt").reshape(3, 4)
        K, R = rq(P[:, :3])
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        K = K / K[2, 2]
        t = np.linalg.solve(K, P[:, 3])

        K[0, 0] *= target_w / orig_w
        K[1, 1] *= target_h / orig_h
        K[0, 2] *= target_w / orig_w
        K[1, 2] *= target_h / orig_h

        cam_id = new_idx + 1
        cameras_lines.append(
            f"{cam_id} PINHOLE {target_w} {target_h} "
            f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}"
        )

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
        output_path=dense_dir, input_path=sparse_dir,
        image_path=ws_images, undistort_options=undistort_opts,
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

    # --- Step 3: PatchMatch ---
    # Use conservative depth range from camera distances
    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = target_w
    pm_opts.num_iterations = 5
    pm_opts.depth_min = 50.0
    pm_opts.depth_max = 1500.0
    pm_opts.filter = True
    pm_opts.filter_min_ncc = 0.1

    # Pass 1: photometric
    pm_opts.geom_consistency = False
    print("\nPatchMatch pass 1 (photometric, filter=True)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # Pass 2: geometric
    pm_opts.geom_consistency = True
    print("PatchMatch pass 2 (geometric, filter=True)...")
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    # --- Step 4: List depth map files ---
    depth_dir = f"{stereo_dir}/depth_maps"
    print(f"\n=== Depth maps in {depth_dir} ===")
    depth_files = sorted(os.listdir(depth_dir))
    for df in depth_files:
        size = os.path.getsize(f"{depth_dir}/{df}")
        print(f"  {df}: {size} bytes")

    # --- Step 5: Read depth maps ---
    for suffix in [".geometric.bin", ".photometric.bin"]:
        target_file = None
        for df in depth_files:
            if df.endswith(suffix):
                target_file = df
                break
        if target_file is None:
            print(f"\nNo {suffix} files found!")
            continue

        print(f"\n=== Reading {target_file} ===")
        d = read_colmap_depth_map(f"{depth_dir}/{target_file}")
        print(f"shape={d.shape} dtype={d.dtype}")
        print(f"min={d.min():.2f} max={d.max():.2f} mean={d.mean():.2f}")
        print(f"nonzero fraction={(d > 0).sum() / d.size:.3f}")
        print(f"unique count={len(np.unique(d))}")

    # --- Step 6: Fusion attempts ---
    for input_type in ["geometric", "photometric"]:
        for max_de in [0.01, 0.1, 1.0, 10.0]:
            fusion_opts = pycolmap.StereoFusionOptions()
            fusion_opts.min_num_pixels = 2
            fusion_opts.max_depth_error = max_de
            fusion_opts.max_reproj_error = 4.0

            fused_dir = f"{dense_dir}/fused_{input_type}_{max_de}"
            os.makedirs(fused_dir, exist_ok=True)
            recon = pycolmap.stereo_fusion(
                output_path=fused_dir, workspace_path=dense_dir,
                options=fusion_opts, input_type=input_type,
            )
            n = len(recon.points3D)
            print(f"  {input_type} max_depth_error={max_de}: {n} points")
            if n > 0:
                pts = np.array([p.xyz for p in recon.points3D.values()])
                print(f"    Range: {pts.min(axis=0)} to {pts.max(axis=0)}")
                recon.export_PLY(f"{dense_dir}/fused.ply")
                print(f"    Exported: {os.path.getsize(f'{dense_dir}/fused.ply')} bytes")
                return f"fused={n}, {input_type}, max_de={max_de}"

    return "fused=0 (all attempts failed)"


@app.local_entrypoint()
def main():
    print(diagnose.remote())
