"""Modal GPU stub for COLMAP PatchMatch stereo + fusion.

Runs dense MVS on Modal A10G because PatchMatch requires CUDA.
Local pipeline prepares the workspace (images + sparse model),
uploads it, runs PatchMatch + fusion on GPU, downloads the result.

Usage:
    # Deploy (one-time):
    modal deploy modal_app.py

    # Download DTU data into Modal volume:
    modal run modal_app.py::download_dtu_scan9

    # Or run directly:
    modal run modal_app.py
"""

import modal

app = modal.App("simtorecon-mvs")

# Image with pycolmap-cuda12 for GPU PatchMatch stereo
# Needs CUDA runtime libs → use nvidia/cuda base image
colmap_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libice6",
    )
    .pip_install(
        "pycolmap-cuda12",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "scipy>=1.10",
    )
)

# Shared volume for workspace transfer
workspace_volume = modal.Volume.from_name("simtorecon-workspace", create_if_missing=True)

VOLUME_MOUNT = "/workspace"


@app.function(
    image=colmap_image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={VOLUME_MOUNT: workspace_volume},
)
def run_patch_match_and_fusion(
    workspace_subdir: str,
    max_image_size: int = 800,
    geom_consistency: bool = True,
    num_iterations: int = 5,
    min_num_pixels: int = 5,
    max_reproj_error: float = 2.0,
    max_depth_error: float = 0.01,
) -> dict:
    """Run PatchMatch stereo + fusion on GPU.

    Args:
        workspace_subdir: Subdirectory within the volume containing
            the undistorted workspace (images/ + sparse/ from pycolmap.undistort_images).
        max_image_size: Maximum image dimension for PatchMatch.
        geom_consistency: Use geometric consistency in PatchMatch.
        num_iterations: Number of PatchMatch iterations.
        min_num_pixels: Minimum pixels for fusion.
        max_reproj_error: Maximum reprojection error for fusion.
        max_depth_error: Maximum depth error for fusion.

    Returns:
        dict with 'success', 'fused_ply_path', 'n_points', 'error'.
    """
    import pycolmap

    workspace_path = f"{VOLUME_MOUNT}/{workspace_subdir}"

    print(f"CUDA available: {pycolmap.has_cuda}")
    print(f"Workspace: {workspace_path}")

    try:
        # Step 1: PatchMatch stereo
        print("Running PatchMatch stereo...")
        options = pycolmap.PatchMatchOptions()
        options.max_image_size = max_image_size
        options.geom_consistency = geom_consistency
        options.num_iterations = num_iterations

        pycolmap.patch_match_stereo(
            workspace_path=workspace_path,
            options=options,
        )
        print("PatchMatch stereo complete.")

        # Step 2: Stereo fusion
        fused_ply = f"{workspace_path}/fused.ply"
        print("Running stereo fusion...")

        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.min_num_pixels = min_num_pixels
        fusion_options.max_reproj_error = max_reproj_error
        fusion_options.max_depth_error = max_depth_error

        pycolmap.stereo_fusion(
            output_path=fused_ply,
            workspace_path=workspace_path,
            options=fusion_options,
        )
        print("Stereo fusion complete.")

        # Count points

        # Read PLY header to get point count (lightweight)
        n_points = 0
        with open(fused_ply, "rb") as f:
            for line in f:
                line = line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    n_points = int(line.split()[-1])
                    break

        workspace_volume.commit()

        return {
            "success": True,
            "fused_ply_path": f"{workspace_subdir}/fused.ply",
            "n_points": n_points,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "fused_ply_path": None,
            "n_points": 0,
            "error": str(e),
        }


# ---- Full end-to-end reconstruction from Modal-hosted DTU data ----

dtu_data_volume = modal.Volume.from_name("simtorecon-dtu-data", create_if_missing=True)
DTU_DATA_MOUNT = "/dtu_data"


@app.function(
    image=colmap_image,
    gpu="A10G",
    timeout=7200,  # 2 hours
    volumes={
        VOLUME_MOUNT: workspace_volume,
        DTU_DATA_MOUNT: dtu_data_volume,
    },
)
def reconstruct_dtu_scan9(
    n_views: int = 49,
    seed: int = 42,
    target_width: int = 800,
    target_height: int = 600,
    max_image_size: int = 800,
    num_iterations: int = 5,
    light_idx: int = 3,
) -> dict:
    """Full SfM + dense MVS reconstruction of DTU scan9.

    Uses COLMAP's standard pipeline: feature extraction → exhaustive matching
    → incremental SfM → undistort → PatchMatch stereo → fusion.

    SfM recovers poses automatically (no external calibration injection).
    Coordinate alignment to DTU GT is handled downstream via ICP.
    """
    import os
    import time
    import uuid

    import cv2
    import numpy as np
    import pycolmap

    # Fix COLMAP's internal RANSAC seed for deterministic SfM
    pycolmap.set_random_seed(seed)

    start = time.time()
    run_id = f"scan9_v{n_views}_s{seed}_{uuid.uuid4().hex[:6]}"
    run_dir = f"{VOLUME_MOUNT}/{run_id}"

    try:
        # --- Load images from volume ---
        images_dir = f"{DTU_DATA_MOUNT}/scan9/images"
        all_images = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(".png") and f"_{light_idx}_" in f
        ])
        if not all_images:
            all_images = sorted([
                f for f in os.listdir(images_dir) if f.endswith(".png")
            ])

        print(f"Found {len(all_images)} images")

        # Subsample views
        total = len(all_images)
        if n_views >= total:
            selected = all_images
        else:
            rng = np.random.RandomState(seed)
            step = total / n_views
            offset = rng.uniform(0, step)
            indices = [min(int(offset + i * step), total - 1) for i in range(n_views)]
            selected = [all_images[i] for i in indices]

        print(f"Using {len(selected)} views")

        # Prepare downsampled images
        ws_images = f"{run_dir}/images"
        os.makedirs(ws_images, exist_ok=True)

        for img_file in selected:
            img = cv2.imread(f"{images_dir}/{img_file}")
            img = cv2.resize(
                img, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(f"{ws_images}/{img_file}", img)

        # --- SfM: feature extraction + matching + mapping ---
        db_path = f"{run_dir}/database.db"
        sparse_dir = f"{run_dir}/sparse"
        os.makedirs(sparse_dir, exist_ok=True)

        print("Feature extraction...")
        pycolmap.extract_features(db_path, ws_images)

        print("Exhaustive matching...")
        pycolmap.match_exhaustive(db_path)

        print("Incremental mapping...")
        maps = pycolmap.incremental_mapping(db_path, ws_images, sparse_dir)

        if not maps:
            raise RuntimeError(
                f"SfM failed: no reconstruction from {len(selected)} images"
            )

        best = max(maps.values(), key=lambda r: r.num_reg_images())
        n_reg = best.num_reg_images()
        n_sparse = len(best.points3D)
        print(f"SfM: {n_reg}/{len(selected)} registered, {n_sparse} sparse points")

        best_dir = f"{sparse_dir}/0"
        os.makedirs(best_dir, exist_ok=True)
        best.write(best_dir)

        # --- Dense MVS: undistort + PatchMatch + fusion ---
        dense_dir = f"{run_dir}/dense"
        undistort_opts = pycolmap.UndistortCameraOptions()
        undistort_opts.max_image_size = max_image_size

        print("Undistorting...")
        pycolmap.undistort_images(
            output_path=dense_dir, input_path=best_dir,
            image_path=ws_images, undistort_options=undistort_opts,
        )

        pm_opts = pycolmap.PatchMatchOptions()
        pm_opts.max_image_size = max_image_size
        pm_opts.num_iterations = num_iterations
        pm_opts.filter = True

        print("PatchMatch (photometric)...")
        pm_opts.geom_consistency = False
        pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

        print("PatchMatch (geometric)...")
        pm_opts.geom_consistency = True
        pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

        print("Stereo fusion...")
        fused_dir = f"{dense_dir}/fused"
        os.makedirs(fused_dir, exist_ok=True)
        fusion_opts = pycolmap.StereoFusionOptions()
        fusion_opts.min_num_pixels = 3

        recon = pycolmap.stereo_fusion(
            output_path=fused_dir, workspace_path=dense_dir, options=fusion_opts,
        )

        n_points = len(recon.points3D)
        print(f"Fused: {n_points} points")

        # Export PLY
        fused_ply = f"{run_dir}/fused.ply"
        recon.export_PLY(fused_ply)

        elapsed = time.time() - start
        workspace_volume.commit()

        print(f"\nReconstruction complete: {n_points} points in {elapsed:.1f}s")

        return {
            "success": True,
            "run_id": run_id,
            "n_views": len(selected),
            "n_registered": n_reg,
            "n_sparse_points": n_sparse,
            "n_points": n_points,
            "elapsed_seconds": elapsed,
            "fused_ply_path": f"{run_id}/fused.ply",
            "error": None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "run_id": run_id,
            "n_views": n_views,
            "n_registered": 0,
            "n_sparse_points": 0,
            "n_points": 0,
            "elapsed_seconds": time.time() - start,
            "fused_ply_path": None,
            "error": str(e),
        }


# ---- Shared-SfM approach: run SfM once, dense MVS per view subset ----


@app.function(
    image=colmap_image,
    gpu="A10G",
    timeout=7200,
    volumes={
        VOLUME_MOUNT: workspace_volume,
        DTU_DATA_MOUNT: dtu_data_volume,
    },
)
def sfm_dtu_scan9(
    target_width: int = 800,
    target_height: int = 600,
    light_idx: int = 3,
) -> dict:
    """Run SfM on all DTU scan9 views. Save sparse model to volume.

    This is the first half of the shared-SfM pipeline. Run once, then use
    dense_mvs_subset() to run PatchMatch + fusion on view-count subsets
    against the fixed sparse model.
    """
    import os
    import time
    import uuid

    import cv2
    import pycolmap

    start = time.time()
    run_id = f"scan9_sfm_{uuid.uuid4().hex[:6]}"
    run_dir = f"{VOLUME_MOUNT}/{run_id}"

    # Load all images for the selected lighting condition
    images_dir = f"{DTU_DATA_MOUNT}/scan9/images"
    all_images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".png") and f"_{light_idx}_" in f
    ])
    if not all_images:
        all_images = sorted([
            f for f in os.listdir(images_dir) if f.endswith(".png")
        ])

    print(f"Found {len(all_images)} images for light condition {light_idx}")

    # Prepare downsampled images
    ws_images = f"{run_dir}/images"
    os.makedirs(ws_images, exist_ok=True)

    for img_file in all_images:
        img = cv2.imread(f"{images_dir}/{img_file}")
        img = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(f"{ws_images}/{img_file}", img)

    # SfM: feature extraction + matching + incremental mapping
    db_path = f"{run_dir}/database.db"
    sparse_dir = f"{run_dir}/sparse"
    os.makedirs(sparse_dir, exist_ok=True)

    print("Feature extraction...")
    pycolmap.extract_features(db_path, ws_images)

    print("Exhaustive matching...")
    pycolmap.match_exhaustive(db_path)

    print("Incremental mapping...")
    maps = pycolmap.incremental_mapping(db_path, ws_images, sparse_dir)

    if not maps:
        return {
            "success": False,
            "run_id": run_id,
            "error": f"SfM failed: no reconstruction from {len(all_images)} images",
            "elapsed_seconds": time.time() - start,
        }

    best = max(maps.values(), key=lambda r: r.num_reg_images())
    n_reg = best.num_reg_images()
    n_sparse = len(best.points3D)

    best_dir = f"{sparse_dir}/0"
    os.makedirs(best_dir, exist_ok=True)
    best.write(best_dir)

    # Get registered image names (sorted for deterministic subsetting)
    registered_names = sorted([img.name for img in best.images.values()])

    elapsed = time.time() - start
    workspace_volume.commit()

    print(f"\nSfM complete: {n_reg}/{len(all_images)} registered, "
          f"{n_sparse} sparse points, {elapsed:.1f}s")

    return {
        "success": True,
        "run_id": run_id,
        "n_total_images": len(all_images),
        "n_registered": n_reg,
        "n_sparse_points": n_sparse,
        "registered_image_names": registered_names,
        "elapsed_seconds": elapsed,
    }


@app.function(
    image=colmap_image,
    gpu="A10G",
    timeout=7200,
    volumes={VOLUME_MOUNT: workspace_volume},
)
def dense_mvs_subset(
    sfm_run_id: str,
    selected_image_names: list,
    max_image_size: int = 800,
    num_iterations: int = 5,
) -> dict:
    """Run dense MVS on a subset of images with fixed poses from full SfM.

    Hybrid approach (Decision 15): camera poses are fixed from the full n=49
    SfM run, but feature extraction, matching, and triangulation are re-run
    on the subset. This gives each view count an adapted sparse scaffold
    while eliminating SfM initialization stochasticity.

    Pipeline:
      1. Copy subset images to fresh workspace
      2. Extract features + exhaustive matching on subset
      3. Load full reconstruction, keep only subset image poses
      4. triangulate_points() with fixed poses → adapted sparse model
      5. Undistort + PatchMatch + fusion
    """
    import os
    import shutil
    import time
    import uuid

    import pycolmap

    start = time.time()
    n_views = len(selected_image_names)
    run_id = f"dense_v{n_views}_{uuid.uuid4().hex[:6]}"
    run_dir = f"{VOLUME_MOUNT}/{run_id}"

    sfm_sparse_dir = f"{VOLUME_MOUNT}/{sfm_run_id}/sparse/0"
    sfm_images_dir = f"{VOLUME_MOUNT}/{sfm_run_id}/images"

    print(f"Loading SfM model from {sfm_run_id}...")
    print(f"Dense MVS with {n_views} views: {selected_image_names[:5]}...")

    # Copy selected images to new workspace
    ws_images = f"{run_dir}/images"
    os.makedirs(ws_images, exist_ok=True)

    selected_set = set(selected_image_names)
    for name in selected_image_names:
        src = f"{sfm_images_dir}/{name}"
        dst = f"{ws_images}/{name}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: image {name} not found in SfM workspace")

    # --- Step 1: Reuse the SfM database (features + matches already computed) ---
    # Copy the original SfM database — it has features and matches for all 49
    # images. triangulate_points only processes matches between registered
    # images, so having extra images in the DB is harmless.
    sfm_db_path = f"{VOLUME_MOUNT}/{sfm_run_id}/database.db"
    db_path = f"{run_dir}/database.db"
    shutil.copy2(sfm_db_path, db_path)
    print(f"Copied SfM database (reusing features + matches)")

    # --- Step 2: Build pose-only reconstruction for subset (original IDs) ---
    full_recon = pycolmap.Reconstruction(sfm_sparse_dir)
    n_full = full_recon.num_reg_images()

    # Write full recon to text, then filter to subset with original IDs
    full_text_dir = f"{run_dir}/full_text"
    os.makedirs(full_text_dir, exist_ok=True)
    full_recon.write_text(full_text_dir)

    subset_sparse_dir = f"{run_dir}/sparse/0"
    os.makedirs(subset_sparse_dir, exist_ok=True)
    shutil.copy2(f"{full_text_dir}/cameras.txt", f"{subset_sparse_dir}/cameras.txt")

    # Filter images.txt: keep only subset images with their ORIGINAL IDs + poses
    n_kept = 0
    with open(f"{full_text_dir}/images.txt") as fin, \
         open(f"{subset_sparse_dir}/images.txt", "w") as fout:
        fout.write("# Image list with two lines of data per image:\n")
        fout.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fout.write("#   POINTS2D[] as (X, Y, POINT3D_ID) ...\n")
        lines = fin.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or line == "":
                i += 1
                continue
            parts = line.split()
            img_name = parts[-1]
            if img_name in selected_set:
                # Keep original ID and pose, clear POINTS2D
                fout.write(line + "\n")
                fout.write("\n")  # empty — triangulate_points fills this
                n_kept += 1
            i += 2

    # Empty points3D.txt
    with open(f"{subset_sparse_dir}/points3D.txt", "w") as fout:
        fout.write("# 3D point list — empty, triangulate_points will populate\n")

    # Read back the pose-only model
    pose_recon = pycolmap.Reconstruction(subset_sparse_dir)
    print(f"Pose-only reconstruction: {pose_recon.num_reg_images()}/{n_full} images "
          f"(database IDs, fixed poses from full SfM)")

    # --- Step 3: Triangulate new 3D points from subset's feature matches ---
    print("Triangulating points with fixed poses...")
    tri_output = f"{run_dir}/sparse_triangulated"
    os.makedirs(tri_output, exist_ok=True)

    tri_recon = pycolmap.triangulate_points(
        pose_recon, db_path, ws_images, tri_output,
        clear_points=True, refine_intrinsics=False,
    )

    n_tri_pts = tri_recon.num_points3D()
    n_tri_imgs = tri_recon.num_reg_images()
    print(f"Triangulated: {n_tri_pts} points from {n_tri_imgs} images")

    # --- Step 4: Undistort + PatchMatch + fusion ---
    dense_dir = f"{run_dir}/dense"
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = max_image_size

    print("Undistorting...")
    pycolmap.undistort_images(
        output_path=dense_dir, input_path=tri_output,
        image_path=ws_images, undistort_options=undistort_opts,
    )

    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.max_image_size = max_image_size
    pm_opts.num_iterations = num_iterations
    pm_opts.filter = True

    print("PatchMatch (photometric)...")
    pm_opts.geom_consistency = False
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    print("PatchMatch (geometric)...")
    pm_opts.geom_consistency = True
    pycolmap.patch_match_stereo(workspace_path=dense_dir, options=pm_opts)

    print("Stereo fusion...")
    fused_dir = f"{dense_dir}/fused"
    os.makedirs(fused_dir, exist_ok=True)
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = 3

    recon = pycolmap.stereo_fusion(
        output_path=fused_dir, workspace_path=dense_dir, options=fusion_opts,
    )

    n_points = len(recon.points3D)
    print(f"Fused: {n_points} points")

    # Export PLY
    fused_ply = f"{run_dir}/fused.ply"
    recon.export_PLY(fused_ply)

    elapsed = time.time() - start
    workspace_volume.commit()

    print(f"\nDense MVS complete: {n_points} points in {elapsed:.1f}s")

    return {
        "success": True,
        "run_id": run_id,
        "n_views": n_views,
        "n_points": n_points,
        "elapsed_seconds": elapsed,
        "fused_ply_path": f"{run_id}/fused.ply",
        "error": None,
    }


# ---- DTU data download (runs on Modal, fast datacenter internet) ----

download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "unzip")
    .pip_install("gdown>=5.0")
)


@app.function(
    image=download_image,
    timeout=21600,  # 6 hours — Rectified is 123GB at ~8MB/s
    volumes={DTU_DATA_MOUNT: dtu_data_volume},
)
def download_dtu_scan9() -> dict:
    """Download DTU scan9 data into Modal volume.

    Resumable: keeps zip files on volume so interrupted downloads continue.
    Uses Python zipfile for extraction (portable, no unzip CLI quirks).

    Run: modal run modal_app.py::download_dtu_scan9
    """
    import os
    import subprocess
    import zipfile

    base = DTU_DATA_MOUNT
    scan9_dir = f"{base}/scan9"
    os.makedirs(scan9_dir, exist_ok=True)

    results = {}

    def download(url: str, dest: str) -> None:
        """Download with curl -C - for resume support."""
        subprocess.run(
            ["curl", "-C", "-", "-L", "-o", dest, url, "--connect-timeout", "30"],
            check=True,
        )

    def is_valid_zip(path: str) -> bool:
        """Check if a zip file is complete and valid."""
        try:
            with zipfile.ZipFile(path) as zf:
                return zf.testzip() is None
        except (zipfile.BadZipFile, Exception):
            return False

    def extract_matching(
        zip_path: str, match_fn, dest_dir: str, flat: bool = True
    ) -> list[str]:
        """Extract files from zip where match_fn(name) is True.

        If flat=True, strips directory paths (like unzip -j).
        Returns list of extracted filenames.
        """
        extracted = []
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if match_fn(name):
                    data = zf.read(name)
                    basename = os.path.basename(name) if flat else name
                    out_path = f"{dest_dir}/{basename}"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    extracted.append(basename)
        return extracted

    def ensure_download(url: str, dest: str) -> None:
        """Download file, resuming if partial, re-downloading if corrupt."""
        if os.path.exists(dest):
            size_gb = os.path.getsize(dest) / 1e9
            print(f"  {os.path.basename(dest)} on volume ({size_gb:.1f} GB)")
        download(url, dest)
        dtu_data_volume.commit()

        if not is_valid_zip(dest):
            print(f"  {os.path.basename(dest)} corrupt, re-downloading...")
            os.remove(dest)
            download(url, dest)
            dtu_data_volume.commit()

    # --- 1. Calibration from SampleSet ---
    calib_dir = f"{scan9_dir}/calibration"
    os.makedirs(calib_dir, exist_ok=True)
    calib_files = [f for f in os.listdir(calib_dir) if f.startswith("pos_")]

    if len(calib_files) >= 49:
        results["calibration"] = f"already present ({len(calib_files)})"
        print(f"Calibration already present ({len(calib_files)} files).")
    else:
        sampleset_zip = f"{base}/SampleSet.zip"
        print("Step 1: SampleSet (calibration)...")
        ensure_download(
            "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip",
            sampleset_zip,
        )
        print("  Extracting calibration files...")
        extracted = extract_matching(
            sampleset_zip,
            lambda n: "pos_" in n and n.endswith(".txt"),
            calib_dir,
        )
        results["calibration"] = f"{len(extracted)} files"
        print(f"  Calibration: {len(extracted)} files extracted.")

        if len(extracted) >= 49:
            os.remove(sampleset_zip)
            print("  Deleted SampleSet.zip.")

    dtu_data_volume.commit()

    # --- 2. GT point cloud from Points ---
    gt_dir = f"{scan9_dir}/gt"
    os.makedirs(gt_dir, exist_ok=True)
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".ply")]

    if gt_files:
        results["ground_truth"] = f"already present: {gt_files}"
        print(f"Ground truth already present: {gt_files}")
    else:
        points_zip = f"{base}/Points.zip"
        print("Step 2: Points (ground truth)...")
        ensure_download(
            "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip",
            points_zip,
        )
        print("  Extracting scan9 ground truth...")
        extracted = extract_matching(
            points_zip,
            lambda n: "stl009" in n and n.endswith(".ply"),
            gt_dir,
        )
        results["ground_truth"] = f"{len(extracted)} files: {extracted}"
        print(f"  Ground truth: {extracted}")

        if extracted:
            os.remove(points_zip)
            print("  Deleted Points.zip.")

    dtu_data_volume.commit()

    # --- 3. Rectified images for scan9 only ---
    images_dir = f"{scan9_dir}/images"
    os.makedirs(images_dir, exist_ok=True)
    existing_images = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    if len(existing_images) >= 49:
        results["images"] = f"already present ({len(existing_images)})"
        print(f"Images already present ({len(existing_images)}).")
    else:
        rectified_zip = f"{base}/Rectified.zip"
        print("Step 3: Rectified (scan9 images, ~123 GB download)...")
        ensure_download(
            "http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip",
            rectified_zip,
        )
        print("  Extracting scan9 images...")
        extracted = extract_matching(
            rectified_zip,
            lambda n: "scan9" in n.lower() and n.endswith(".png"),
            images_dir,
        )
        results["images"] = f"{len(extracted)} images"
        print(f"  Images: {len(extracted)} extracted.")

        if len(extracted) >= 49:
            os.remove(rectified_zip)
            print("  Deleted Rectified.zip.")

    dtu_data_volume.commit()

    print("\n=== DTU scan9 data summary ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results
