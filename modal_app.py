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


@app.function(
    image=colmap_image,
    gpu="A10G",
    timeout=1800,  # 30 min — dense MVS at moderate view counts (~30 images at 800×600) typically completes in 5-10 min on A10G; cap is generous for larger scenes
    volumes={VOLUME_MOUNT: workspace_volume},
)
def rerun_dense_mvs(
    colmap_run_id: str,
    seed: int = 42,
) -> dict:
    """Rerun PatchMatch + fusion on an existing undistorted sparse model.

    Loads the existing undistorted sparse model from /workspace/{colmap_run_id}/dense/
    and runs only the dense MVS stage (PatchMatch photometric → PatchMatch geometric
    → stereo fusion). Writes the fused PLY to a unique subdirectory so multiple
    invocations on the same colmap_run_id do not overwrite each other or the
    cached V1-era fused PLY.

    Initially used by Day 11's source (d) verification protocol per DECISIONS 26
    (two invocations at the same seed, downstream PSNR comparison via the 0.5 dB
    gate). The function itself is scene-agnostic — it operates on whatever
    colmap_run_id is passed and is reusable for future verification experiments
    on other scenes or view counts. GPU PatchMatch is non-deterministic at the
    CUDA-thread-scheduling level per DECISIONS 16 — repeated calls at the same
    seed produce different outputs, which is what the verification protocol
    measures.

    Settings match modal_app.py's reconstruct_dtu_scan9 dense-MVS stage exactly
    (lines 270-291) and dense_mvs_subset's dense-MVS stage (lines 577-598):
        max_image_size = 800
        num_iterations = 5
        filter = True (both photometric and geometric passes)
        fusion min_num_pixels = 3
    Settings are hardcoded here rather than parameterized to prevent accidental
    drift between verification runs and any other dense-MVS invocation in the
    repo. The point of source (d) verification is to measure variance under
    fixed configuration; parameterizing the settings would defeat that.

    Returns:
        dict with success, colmap_run_id, seed, rerun_id (unique subdirectory
        name), n_points, elapsed_seconds, fused_ply_path (workspace-relative),
        and error. The fused_ply_path can be passed directly to train_gsplat
        as dense_init_ply_path.
    """
    import os
    import time
    import uuid

    import pycolmap

    pycolmap.set_random_seed(seed)

    start = time.time()
    rerun_id = f"dense_rerun_s{seed}_{uuid.uuid4().hex[:8]}"
    base = f"{VOLUME_MOUNT}/{colmap_run_id}"
    rerun_dir = f"{base}/{rerun_id}"

    try:
        # Locate the existing undistorted sparse model + images. Same fallback
        # candidates as train_gsplat to handle the dense/sparse vs dense/sparse/0
        # path ambiguity left by pycolmap.undistort_images convention drift.
        candidates = [
            (f"{base}/dense/sparse", f"{base}/dense/images"),
            (f"{base}/dense/sparse/0", f"{base}/dense/images"),
        ]
        sparse_path = None
        images_path = None
        for sp, ip in candidates:
            has_cams = (
                os.path.exists(f"{sp}/cameras.bin")
                or os.path.exists(f"{sp}/cameras.txt")
            )
            if has_cams and os.path.isdir(ip):
                sparse_path = sp
                images_path = ip
                break

        if sparse_path is None:
            return {
                "success": False,
                "colmap_run_id": colmap_run_id,
                "seed": seed,
                "rerun_id": rerun_id,
                "n_points": 0,
                "elapsed_seconds": time.time() - start,
                "fused_ply_path": None,
                "error": (
                    f"no undistorted sparse model found under {base}/dense/sparse "
                    f"or fallbacks; reconstruct_dtu_scan9 must have been run for "
                    f"this colmap_run_id first"
                ),
            }

        print(f"[rerun_dense_mvs] sparse: {sparse_path}")
        print(f"[rerun_dense_mvs] images: {images_path}")
        print(f"[rerun_dense_mvs] writing to: {rerun_dir}")

        # Build a fresh workspace directory with symlinks to the existing sparse
        # model + images, plus a fresh stereo/ directory pre-populated with the
        # cached patch-match.cfg and fusion.cfg. PatchMatch reads from
        # {workspace}/sparse + /images + /stereo/patch-match.cfg, and writes its
        # depth/normal/consistency maps to {workspace}/stereo/. By symlinking
        # sparse + images into a fresh rerun_dir, PatchMatch reads from the
        # cached V1 locations (via the symlinks) and writes its stereo/ outputs
        # to the rerun_dir without touching V1's cached state.
        #
        # The .cfg files are required because pycolmap.patch_match_stereo and
        # pycolmap.stereo_fusion both read existing config files from the
        # workspace's stereo/ subdirectory. These files are normally created by
        # pycolmap.undistort_images during the dense pipeline setup. Since we
        # are NOT running undistort_images here (the cached dense/ already has
        # undistorted sparse + images), we copy the cfg files over from the
        # cached dense/stereo/. We do NOT copy the depth/normal/consistency map
        # subdirectories — PatchMatch creates those fresh, which is the whole
        # point of the rerun. patch-match.cfg references images by basename
        # relative to {workspace}/images/, so the copy works as-is — PatchMatch
        # resolves basenames against the rerun_dir's symlinked images/, which
        # transitively reaches the cached undistorted images. No path rewriting
        # needed.
        import shutil
        os.makedirs(rerun_dir, exist_ok=True)
        rerun_sparse = f"{rerun_dir}/sparse"
        rerun_images = f"{rerun_dir}/images"
        rerun_stereo = f"{rerun_dir}/stereo"
        if not os.path.exists(rerun_sparse):
            os.symlink(sparse_path, rerun_sparse)
        if not os.path.exists(rerun_images):
            os.symlink(images_path, rerun_images)
        os.makedirs(rerun_stereo, exist_ok=True)
        cached_dense_dir = f"{base}/dense"
        for cfg_name in ("patch-match.cfg", "fusion.cfg"):
            cached_cfg = f"{cached_dense_dir}/stereo/{cfg_name}"
            rerun_cfg = f"{rerun_stereo}/{cfg_name}"
            if not os.path.exists(cached_cfg):
                return {
                    "success": False,
                    "colmap_run_id": colmap_run_id,
                    "seed": seed,
                    "rerun_id": rerun_id,
                    "n_points": 0,
                    "elapsed_seconds": time.time() - start,
                    "fused_ply_path": None,
                    "error": (
                        f"required dense-MVS config file not found at {cached_cfg}; "
                        f"the cached dense/ workspace is missing files normally created "
                        f"by pycolmap.undistort_images. rerun_dense_mvs cannot proceed "
                        f"without copying these files into the fresh workspace."
                    ),
                }
            if not os.path.exists(rerun_cfg):
                shutil.copy(cached_cfg, rerun_cfg)

        # PatchMatch + fusion at hardcoded settings matching reconstruct_dtu_scan9
        # and dense_mvs_subset (see docstring above for the parity rationale)
        pm_opts = pycolmap.PatchMatchOptions()
        pm_opts.max_image_size = 800
        pm_opts.num_iterations = 5
        pm_opts.filter = True

        print("[rerun_dense_mvs] PatchMatch (photometric)...")
        pm_opts.geom_consistency = False
        pycolmap.patch_match_stereo(workspace_path=rerun_dir, options=pm_opts)

        print("[rerun_dense_mvs] PatchMatch (geometric)...")
        pm_opts.geom_consistency = True
        pycolmap.patch_match_stereo(workspace_path=rerun_dir, options=pm_opts)

        print("[rerun_dense_mvs] Stereo fusion...")
        fused_dir = f"{rerun_dir}/fused"
        os.makedirs(fused_dir, exist_ok=True)
        fusion_opts = pycolmap.StereoFusionOptions()
        fusion_opts.min_num_pixels = 3

        recon = pycolmap.stereo_fusion(
            output_path=fused_dir, workspace_path=rerun_dir, options=fusion_opts,
        )

        n_points = len(recon.points3D)
        print(f"[rerun_dense_mvs] Fused: {n_points} points")

        fused_ply = f"{rerun_dir}/fused.ply"
        recon.export_PLY(fused_ply)

        elapsed = time.time() - start
        workspace_volume.commit()

        print(f"[rerun_dense_mvs] complete: {n_points} points in {elapsed:.1f}s")

        return {
            "success": True,
            "colmap_run_id": colmap_run_id,
            "seed": seed,
            "rerun_id": rerun_id,
            "n_points": n_points,
            "elapsed_seconds": elapsed,
            "fused_ply_path": f"{colmap_run_id}/{rerun_id}/fused.ply",
            "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "colmap_run_id": colmap_run_id,
            "seed": seed,
            "rerun_id": rerun_id,
            "n_points": 0,
            "elapsed_seconds": time.time() - start,
            "fused_ply_path": None,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


@app.function(
    image=colmap_image,
    timeout=60,
    volumes={VOLUME_MOUNT: workspace_volume},
)
def inspect_workspace_tree(colmap_run_id: str) -> dict:
    """Diagnostic: structured tree of /workspace/{colmap_run_id}/dense/ for
    understanding pycolmap's workspace layout post-undistort.

    Read-only — does not modify any workspace state. Used to discover what
    files and directories pycolmap.undistort_images produces, so that
    rerun_dense_mvs can mirror the structure into a fresh workspace without
    enumerating requirements from memory (which is the retrieval-gap pattern
    DECISIONS 26 names; cf. Day 11's Smoke A iteration history).

    Returns three levels: dense/ top-level, dense/<subdir>/ for each top-level
    subdirectory, and dense/<subdir>/<subsubdir>/ for each second-level
    subdirectory. Files at each level are listed with byte sizes; the first 10
    files per directory are returned verbatim, with `n_files` reporting the
    full count and `total_size_bytes` reporting the cumulative byte count for
    that directory's files (useful for cost-estimating a wholesale copy).

    Returns:
        dict with success, levels (a dict mapping path → {dirs, n_files,
        files_first_10, total_size_bytes}), elapsed_seconds, and error.
    """
    import os
    import time

    start = time.time()
    base = f"{VOLUME_MOUNT}/{colmap_run_id}"
    dense = f"{base}/dense"

    if not os.path.isdir(dense):
        return {
            "success": False,
            "levels": {},
            "elapsed_seconds": time.time() - start,
            "error": f"no dense directory at {dense}",
        }

    def listdir_summary(path: str, max_files: int = 10) -> dict:
        try:
            entries = sorted(os.listdir(path))
        except OSError as e:
            return {"error": f"listdir failed: {e}"}
        dirs = []
        files = []
        for e in entries:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                dirs.append(e)
            elif os.path.isfile(full):
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = -1
                files.append((e, size))
        return {
            "dirs": dirs,
            "n_files": len(files),
            "files_first_10": files[:max_files],
            "total_size_bytes": sum(s for _, s in files if s > 0),
        }

    levels: dict = {}
    levels["dense/"] = listdir_summary(dense)
    for top_subdir in levels["dense/"].get("dirs", []):
        sub_path = f"{dense}/{top_subdir}"
        sub_key = f"dense/{top_subdir}/"
        levels[sub_key] = listdir_summary(sub_path)
        for sub_sub in levels[sub_key].get("dirs", []):
            sub_sub_path = f"{sub_path}/{sub_sub}"
            levels[f"dense/{top_subdir}/{sub_sub}/"] = listdir_summary(sub_sub_path)

    return {
        "success": True,
        "levels": levels,
        "elapsed_seconds": time.time() - start,
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


# ---- V1.5: 3D Gaussian Splatting training (gsplat) ----
#
# Runs gsplat on Modal A10G against the undistorted sparse model produced by
# reconstruct_dtu_scan9 (V1). Given a colmap_run_id like "scan9_v49_s123_3d428b",
# reads /workspace/{run_id}/dense/sparse and /workspace/{run_id}/dense/images,
# trains a 3D Gaussian model, renders held-out test views, and writes outputs
# under /workspace/gsplat_{run_id}_s{seed}/ on the same workspace volume.
#
# Day 8 is a single-seed smoke run. Day 9 calls this function with 3 seeds.
#
# The image uses nvidia/cuda:12.4.1-devel (not runtime) so nvcc is available
# in case gsplat falls back to a source build on a torch/cuda combo without
# a prebuilt wheel.

gsplat_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libice6",
        "git",
        "build-essential",
    )
    # Single pip layer. The first deploy used two layers (torch first, then
    # gsplat) and pip's second-layer resolver picked triton 2.0 from Modal's
    # mirror, which force-downgraded torch to 2.0.1 — breaking numpy ABI.
    # Collapsing into one layer with extra_index_url lets pip see cu124 torch
    # and pypi gsplat simultaneously, and the explicit triton pin prevents
    # the downgrade.
    .pip_install(
        "numpy==1.26.4",
        "torch==2.4.1",
        "torchvision==0.19.1",
        "triton==3.0.0",
        "gsplat",
        "pycolmap-cuda12",
        "scipy>=1.10",
        "opencv-python-headless>=4.8",
        "scikit-image>=0.22",
        "lpips>=0.1.4",
        "torchmetrics>=1.0",
        "pillow>=10.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)


def _read_ply_xyz_rgb(path: str):
    """Minimal binary little-endian PLY reader for x/y/z + r/g/b.

    Designed for pycolmap's exported fused.ply format (the output of
    pycolmap.stereo_fusion → recon.export_PLY). Does NOT support ASCII PLYs,
    big-endian PLYs, or PLY files without the expected x/y/z/red/green/blue
    properties. Fails loud on any unexpected structure rather than guessing.

    Returns:
        (xyz, rgb) where xyz is float32 shape (N, 3) and rgb is float32 shape
        (N, 3) with values in [0, 255] — matching the sparse-init code path's
        convention so the downstream gsplat init logic does not need to branch
        on color dtype.

    Used by train_gsplat's dense_init_ply_path branch (Day 11 experiment per
    DECISIONS 26). The helper is module-level rather than nested so it is
    reusable by any other Modal function that needs to load a fused PLY without
    pulling in plyfile / open3d as new dependencies (the gsplat_image deliberately
    keeps its dependency list minimal).
    """
    import numpy as np

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY {path}: unexpected EOF in header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        header_str = b"".join(header_lines).decode("ascii", errors="replace")
        if not header_str.startswith("ply\n"):
            raise ValueError(f"PLY {path}: missing 'ply' magic")
        if "format binary_little_endian" not in header_str:
            raise ValueError(
                f"PLY {path}: only binary_little_endian format supported; "
                f"header begins: {header_str[:200]!r}"
            )

        # Parse vertex element + properties from the header
        n_vertices = None
        in_vertex_element = False
        properties = []
        for hline in header_str.splitlines():
            hline = hline.strip()
            if hline.startswith("element vertex"):
                n_vertices = int(hline.split()[2])
                in_vertex_element = True
            elif hline.startswith("element ") and in_vertex_element:
                in_vertex_element = False
            elif hline.startswith("property ") and in_vertex_element:
                parts = hline.split()
                # `property <type> <name>` — list properties (`property list ...`)
                # are not expected in pycolmap's fused.ply and are not handled
                # here. If this assumption breaks, the dtype construction below
                # will fail loudly.
                if len(parts) != 3:
                    raise ValueError(
                        f"PLY {path}: unexpected property declaration: {hline!r}"
                    )
                properties.append((parts[1], parts[2]))

        if n_vertices is None:
            raise ValueError(f"PLY {path}: no vertex element found")
        if not properties:
            raise ValueError(f"PLY {path}: vertex element has no properties")

        # Build numpy structured dtype matching the property declaration order
        type_map = {
            "float": "<f4", "float32": "<f4",
            "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1",
            "char": "i1", "int8": "i1",
            "ushort": "<u2", "uint16": "<u2",
            "short": "<i2", "int16": "<i2",
            "uint": "<u4", "uint32": "<u4",
            "int": "<i4", "int32": "<i4",
        }
        dtype_fields = []
        for ptype, pname in properties:
            if ptype not in type_map:
                raise ValueError(
                    f"PLY {path}: unsupported property type {ptype!r} for {pname!r}"
                )
            dtype_fields.append((pname, type_map[ptype]))
        dtype = np.dtype(dtype_fields)

        # Read binary body
        data = np.fromfile(f, dtype=dtype, count=n_vertices)
        if data.shape[0] != n_vertices:
            raise ValueError(
                f"PLY {path}: read {data.shape[0]} vertices, expected {n_vertices}"
            )

    field_names = data.dtype.names or ()
    for required in ("x", "y", "z", "red", "green", "blue"):
        if required not in field_names:
            raise ValueError(
                f"PLY {path}: missing required field {required!r}; "
                f"available fields: {field_names}"
            )

    xyz = np.stack(
        [data["x"], data["y"], data["z"]], axis=1
    ).astype(np.float32)
    rgb = np.stack(
        [data["red"], data["green"], data["blue"]], axis=1
    ).astype(np.float32)
    return xyz, rgb


@app.function(
    image=gsplat_image,
    gpu="A10G",
    timeout=3600,  # 1 hour — 7000 iters typically runs in 5-15 min
    volumes={VOLUME_MOUNT: workspace_volume},
)
def train_gsplat(
    colmap_run_id: str,
    n_iterations: int = 7000,
    seed: int = 42,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_sh0: float = 2.5e-3,
    lr_shN: float = 1.25e-4,
    ssim_lambda: float = 0.2,  # 0.8*L1 + 0.2*(1 - SSIM) — matches gsplat simple_trainer.py
    densify_start_iter: int = 500,
    densify_stop_iter: int = 5000,
    densify_grad_threshold: float = 2e-4,  # gsplat default; works when random_bkgd is on
    reset_opacity_iter: int = 3000,
    test_every: int = 10,
    sh_degree: int = 3,
    refine_every: int = 100,
    random_bkgd: bool = False,  # costs ~3 dB PSNR on DTU-style gray backgrounds; see DECISIONS 22. Opt-in for synthetic pure-black-bg datasets only.
    image_order_seed: int | None = None,  # When set, drives np.random.default_rng (per-step training-image sampler) independently of `seed`. When None, falls back to `seed` — backward-compatible with all Day 8/9/10 runs. Added for DECISIONS 23 P2 image-order-variance diagnostic.
    dense_init_ply_path: str | None = None,  # When set, replaces sparse-SfM init with xyz+rgb from this PLY file. Path is relative to /workspace mount or absolute. Camera poses + intrinsics still come from colmap_run_id. None preserves Day 8/9/10 sparse-init behavior. Added for Day 11 dense-init experiment per DECISIONS 26.
) -> dict:
    """Train 3D Gaussian Splatting on a V1 COLMAP reconstruction.

    Loads the undistorted sparse model from /workspace/{colmap_run_id}/dense/
    and trains Gaussians initialized from the sparse points. Holds out every
    test_every-th image (sorted by name) as a novel-view test set. Computes
    PSNR / SSIM / LPIPS on held-out views and saves rendered PNGs + a
    checkpoint to /workspace/gsplat_{colmap_run_id}_s{seed}/.

    Returns a dict matching simtorecon.neural.gsplat_trainer.GsplatResult.
    """
    import json
    import math
    import os
    import time

    import cv2
    import numpy as np
    import pycolmap
    import torch
    from torch import nn
    from torch.nn.functional import l1_loss, normalize
    from torchmetrics.functional.image import (
        structural_similarity_index_measure as tm_ssim,
    )

    def math_ok(x: float) -> bool:
        return not (math.isnan(x) or math.isinf(x))

    start = time.time()

    # RNG posture is pinned explicitly to make V1.5's multi-seed variance interpretable.
    # See DECISIONS.md entry 20 for what is and isn't deterministic and why.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda"

    out_run_id = f"gsplat_{colmap_run_id}_s{seed}"
    out_dir = f"{VOLUME_MOUNT}/{out_run_id}"
    renders_dir = f"{out_dir}/renders"
    os.makedirs(renders_dir, exist_ok=True)

    def _fail(msg: str, **extra) -> dict:
        print(f"[train_gsplat] FAILED: {msg}")
        return {
            "success": False,
            "run_id": out_run_id,
            "colmap_run_id": colmap_run_id,
            "seed": seed,
            "error": msg,
            "elapsed_seconds": time.time() - start,
            **extra,
        }

    # --- Locate the undistorted sparse model + images ---
    # V1 writes: /workspace/{run_id}/dense/sparse  +  /workspace/{run_id}/dense/images
    # (pycolmap.undistort_images writes flat 'sparse/' without the /0/ subdir).
    base = f"{VOLUME_MOUNT}/{colmap_run_id}"
    candidates = [
        (f"{base}/dense/sparse", f"{base}/dense/images"),
        (f"{base}/dense/sparse/0", f"{base}/dense/images"),
        (f"{base}/sparse/0", f"{base}/images"),  # pre-undistort fallback
    ]
    sparse_path = None
    images_path = None
    for sp, ip in candidates:
        has_cams = (
            os.path.exists(f"{sp}/cameras.bin")
            or os.path.exists(f"{sp}/cameras.txt")
        )
        if has_cams and os.path.isdir(ip):
            sparse_path = sp
            images_path = ip
            break

    if sparse_path is None:
        return _fail(
            f"no sparse model found under {base}/dense/sparse or fallbacks"
        )

    print(f"[train_gsplat] sparse: {sparse_path}")
    print(f"[train_gsplat] images: {images_path}")

    # --- Load COLMAP reconstruction ---
    try:
        recon = pycolmap.Reconstruction(sparse_path)
    except Exception as e:
        return _fail(f"pycolmap.Reconstruction failed: {e}")

    n_recon_imgs = recon.num_reg_images()
    n_sparse_pts = len(recon.points3D)
    print(f"[train_gsplat] loaded {n_recon_imgs} images, {n_sparse_pts} sparse points")

    if n_sparse_pts < 100:
        return _fail(f"too few sparse points ({n_sparse_pts}) to initialize gsplat")

    # --- Build per-image data (viewmat, K, RGB tensor) ---
    images_info: list[dict] = []
    for img in recon.images.values():
        if not img.has_pose:
            continue
        cam = recon.cameras[img.camera_id]

        # cam_from_world: world -> camera (gsplat viewmat convention).
        # pycolmap-cuda12 exposes cam_from_world as a method, upstream pycolmap
        # as an attribute — dispatch defensively.
        cfw = img.cam_from_world
        rigid = cfw() if callable(cfw) else cfw
        # rigid.matrix() returns the (3, 4) world-to-cam transform.
        _mat = rigid.matrix
        W = np.asarray(_mat() if callable(_mat) else _mat, dtype=np.float32)
        if W.shape == (3, 4):
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :4] = W
        elif W.shape == (4, 4):
            viewmat = W.astype(np.float32)
        else:
            return _fail(
                f"unexpected cam_from_world matrix shape: {W.shape}"
            )

        # Pinhole intrinsics (dense/sparse uses PINHOLE model after undistort)
        params = cam.params  # depends on model; for PINHOLE: [fx, fy, cx, cy]
        if cam.model_name == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif cam.model_name == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            # Fall back to focal_length_x/y for pre-undistort models
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
        K = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
        )

        img_path = f"{images_path}/{img.name}"
        if not os.path.exists(img_path):
            print(f"[train_gsplat] WARN: image not found: {img_path}")
            continue
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[train_gsplat] WARN: cv2 failed to read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # CLAUDE.md Check 1 (baked in): print real per-image stats once.
        if not images_info:
            print(
                f"=== train_gsplat dataset sanity ===\n"
                f"  first image   : {img.name}\n"
                f"  rgb.shape     : {rgb.shape}\n"
                f"  rgb.dtype     : {rgb.dtype}\n"
                f"  rgb.min/max   : {rgb.min():.4f} / {rgb.max():.4f}\n"
                f"  rgb.mean/std  : {rgb.mean():.4f} / {rgb.std():.4f}\n"
                f"  viewmat.shape : {viewmat.shape}\n"
                f"  K diag (fx,fy): {K[0, 0]:.2f}, {K[1, 1]:.2f}\n"
                f"  K pp (cx, cy) : {K[0, 2]:.2f}, {K[1, 2]:.2f}\n"
                f"  cam model     : {cam.model_name}\n"
                f"  camera wxh    : {cam.width} x {cam.height}\n"
                f"  K pp in bounds: "
                f"{0 < K[0, 2] < cam.width and 0 < K[1, 2] < cam.height}\n"
                f"==================================="
            )
            if rgb.min() < 0.0 or rgb.max() > 1.0:
                return _fail(
                    f"rgb out of [0,1] range: min={rgb.min()}, max={rgb.max()}"
                )
            if rgb.shape[2] != 3:
                return _fail(f"rgb not HxWx3: got {rgb.shape}")
            if np.isnan(rgb).any() or np.isinf(rgb).any():
                return _fail("rgb contains nan/inf")

        images_info.append(
            {
                "name": img.name,
                "viewmat": viewmat,
                "K": K,
                "rgb": rgb,  # (H, W, 3) float32
                "height": rgb.shape[0],
                "width": rgb.shape[1],
            }
        )

    if len(images_info) < 5:
        return _fail(f"too few loaded images ({len(images_info)}) for training")

    # Sort by name for deterministic train/test split
    images_info.sort(key=lambda x: x["name"])
    test_imgs = [img for i, img in enumerate(images_info) if i % test_every == 0]
    train_imgs = [img for i, img in enumerate(images_info) if i % test_every != 0]
    print(
        f"[train_gsplat] split: {len(train_imgs)} train / {len(test_imgs)} test "
        f"(test_every={test_every})"
    )
    if len(train_imgs) < 5 or len(test_imgs) < 1:
        return _fail("degenerate train/test split")

    # --- Initialize Gaussians (from sparse SfM points OR from a fused PLY) ---
    # Day 8/9/10: dense_init_ply_path is None → sparse SfM init (recon.points3D).
    # Day 11+: dense_init_ply_path is set → load xyz+rgb from a fused PLY.
    # Camera poses, intrinsics, and image data still come from `recon` regardless
    # of the init source — the PLY only replaces the point-cloud initialization.
    if dense_init_ply_path is not None:
        full_ply_path = (
            dense_init_ply_path
            if dense_init_ply_path.startswith("/")
            else f"{VOLUME_MOUNT}/{dense_init_ply_path}"
        )
        print(f"[train_gsplat] dense init from PLY: {full_ply_path}")
        try:
            points, colors_u8 = _read_ply_xyz_rgb(full_ply_path)
        except Exception as e:
            return _fail(f"PLY read failed for dense_init_ply_path={dense_init_ply_path}: {e}")
        print(
            f"[train_gsplat] PLY init: {len(points)} points "
            f"(replaces {n_sparse_pts} sparse SfM points; cameras + intrinsics still from colmap_run_id)"
        )
        # Degeneracy gate, not a methodology constraint: 100 is arbitrary but
        # defensible — a usable dense init has thousands to hundreds of thousands
        # of points; degenerate outputs are typically empty or single-digit. The
        # threshold's job is to fail loud on a degenerate PLY, not to enforce a
        # specific point-count target.
        if len(points) < 100:
            return _fail(
                f"too few points in dense init PLY ({len(points)}) — expected at least 100"
            )
    else:
        points = np.asarray(
            [pt.xyz for pt in recon.points3D.values()], dtype=np.float32
        )  # (N, 3)
        colors_u8 = np.asarray(
            [pt.color for pt in recon.points3D.values()], dtype=np.float32
        )  # (N, 3) in [0, 255]
    rgb_init = colors_u8 / 255.0

    # Scales initialized from mean distance to 3 nearest neighbors (log space).
    # Operates on `points` regardless of init source. At dense-init density
    # (~257k points for Day 11's scan9 fused PLY), nearest-neighbor distances
    # are substantially tighter than at sparse-init density (~9k points), so
    # per-point initial scales come out smaller — structurally correct, not a
    # bug. A future reader debugging "why are dense-init Gaussians small?" gets
    # the convention here.
    from scipy.spatial import KDTree

    tree = KDTree(points)
    dists, _ = tree.query(points, k=min(4, points.shape[0]))
    mean_dist = dists[:, 1:].mean(axis=1).clip(min=1e-6)
    log_scale_init = np.log(mean_dist).astype(np.float32)

    N = points.shape[0]
    means = torch.from_numpy(points).to(device)
    scales = torch.from_numpy(
        np.broadcast_to(log_scale_init[:, None], (N, 3)).copy()
    ).to(device)
    quats = torch.zeros((N, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # (w, x, y, z) identity
    opacities = torch.full(
        (N,), float(np.log(0.1 / 0.9)), dtype=torch.float32, device=device
    )  # logit(0.1)

    # SH coefficients: sh0 holds view-independent base color, shN is zero-init
    C0 = 0.28209479177387814  # SH basis evaluated at degree 0
    sh0_init = ((rgb_init - 0.5) / C0).astype(np.float32)
    sh0 = torch.from_numpy(sh0_init).unsqueeze(1).to(device)  # (N, 1, 3)
    n_shN = (sh_degree + 1) ** 2 - 1
    shN = torch.zeros((N, n_shN, 3), dtype=torch.float32, device=device)

    params = {
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
        "shN": nn.Parameter(shN),
    }
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=lr_means),
        "scales": torch.optim.Adam([params["scales"]], lr=lr_scales),
        "quats": torch.optim.Adam([params["quats"]], lr=lr_quats),
        "opacities": torch.optim.Adam([params["opacities"]], lr=lr_opacities),
        "sh0": torch.optim.Adam([params["sh0"]], lr=lr_sh0),
        "shN": torch.optim.Adam([params["shN"]], lr=lr_shN),
    }

    print(f"[train_gsplat] initialized {N} Gaussians")

    # --- gsplat imports (inside function to keep image builds lazy) ---
    from gsplat.rendering import rasterization

    try:
        from gsplat.strategy import DefaultStrategy

        strategy = DefaultStrategy(
            # verbose=True makes gsplat print n_dupli/n_split/n_prune at every refinement event —
            # essential observability for debugging N-stuck-at-init-count symptoms.
            verbose=True,
            # Must match rasterization's absgrad=True — DefaultStrategy reads info["means2d"].absgrad when True.
            absgrad=True,
            refine_start_iter=densify_start_iter,
            refine_stop_iter=densify_stop_iter,
            reset_every=reset_opacity_iter,
            refine_every=refine_every,
            grow_grad2d=densify_grad_threshold,
        )
        try:
            strategy.check_sanity(params, optimizers)
        except Exception as e:
            print(f"[train_gsplat] strategy.check_sanity warning: {e}")
        strategy_state = strategy.initialize_state()
        strategy_ok = True
    except Exception as e:
        print(f"[train_gsplat] DefaultStrategy unavailable, training without densify: {e}")
        strategy = None
        strategy_state = None
        strategy_ok = False

    # --- Intermediate PSNR diagnostic (Day 9 experiment E) ---
    def _eval_psnr_now(step_num: int) -> None:
        """Quick PSNR-only eval of test views for intermediate diagnostics.

        Fires at step 600 (before first refinement) and step 3000 (mid-densification)
        to surface early whether PSNR is tracking the expected recovery. ~250ms total.
        """
        psnrs_here: list[float] = []
        with torch.no_grad():
            for img_ in test_imgs:
                vm = torch.from_numpy(img_["viewmat"]).unsqueeze(0).to(device)
                K_ = torch.from_numpy(img_["K"]).unsqueeze(0).to(device)
                H_, W_ = img_["height"], img_["width"]
                colors_all_ = torch.cat([params["sh0"], params["shN"]], dim=1)
                renders_, _, _ = rasterization(
                    means=params["means"],
                    quats=normalize(params["quats"], dim=-1),
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=colors_all_,
                    viewmats=vm,
                    Ks=K_,
                    width=W_,
                    height=H_,
                    sh_degree=sh_degree,
                    render_mode="RGB",
                )
                pred_ = renders_[0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
                gt_ = img_["rgb"]
                mse_ = float(np.mean((pred_ - gt_) ** 2))
                psnrs_here.append(
                    -10.0 * np.log10(mse_) if mse_ > 0 else 100.0
                )
        med = float(np.median(psnrs_here))
        n_g = int(params["means"].shape[0])
        print(
            f"[train_gsplat] step {step_num:5d}/{n_iterations}  INTERMEDIATE PSNR  "
            f"median={med:.2f} dB  range=({min(psnrs_here):.2f}, "
            f"{max(psnrs_here):.2f})  N={n_g}"
        )

    # --- Training loop ---
    print(f"[train_gsplat] starting {n_iterations} iterations")
    rng = np.random.default_rng(seed if image_order_seed is None else image_order_seed)

    losses_log: list[float] = []
    for step in range(n_iterations):
        img = train_imgs[int(rng.integers(len(train_imgs)))]
        viewmat = torch.from_numpy(img["viewmat"]).unsqueeze(0).to(device)  # (1, 4, 4)
        K = torch.from_numpy(img["K"]).unsqueeze(0).to(device)  # (1, 3, 3)
        gt_rgb = torch.from_numpy(img["rgb"]).to(device)  # (H, W, 3)
        H, W = img["height"], img["width"]

        # Active SH degree ramps up linearly to full over first 3000 iters
        active_sh = min(sh_degree, step // max(1, (n_iterations // (sh_degree + 1))))

        colors_all = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K, 3)

        try:
            renders, alphas, info = rasterization(
                means=params["means"],
                quats=normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors_all,
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=active_sh,
                render_mode="RGB",
                absgrad=True,  # DefaultStrategy reads means2d.absgrad for densify
            )
        except Exception as e:
            return _fail(f"rasterization failed at step {step}: {e}")

        # Random-background augmentation per gsplat simple_trainer.py: composite the
        # foreground renders against a per-step random background so that pixels with
        # near-zero alpha carry no consistent training signal. Forces the model to
        # learn alpha masks for foreground-only scenes like DTU instead of baking the
        # background into Gaussian colors. Only during training; eval uses deterministic
        # black background.
        if random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            renders = renders + bkgd * (1.0 - alphas)

        pred_rgb = renders[0]  # (H, W, 3)
        loss_l1 = l1_loss(pred_rgb, gt_rgb)
        # SSIM in (1, 3, H, W) per gsplat simple_trainer.py convention.
        pred_bchw = pred_rgb.permute(2, 0, 1).unsqueeze(0)
        gt_bchw = gt_rgb.permute(2, 0, 1).unsqueeze(0)
        loss_ssim = 1.0 - tm_ssim(pred_bchw, gt_bchw, data_range=1.0)
        loss = (1.0 - ssim_lambda) * loss_l1 + ssim_lambda * loss_ssim

        if strategy_ok:
            try:
                strategy.step_pre_backward(
                    params=params,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                )
            except Exception as e:
                # Print on FIRST failure regardless of step — the old `if step == 0` guard
                # silently swallowed any later exception, which made debugging impossible.
                print(
                    f"[train_gsplat] strategy.step_pre_backward disabled at step {step}: "
                    f"{type(e).__name__}: {e}"
                )
                strategy_ok = False

        loss.backward()

        if step == 0:
            m2d = info.get("means2d")
            _grad = getattr(m2d, "grad", None)
            _absgrad = getattr(m2d, "absgrad", None)
            m2d_shape = tuple(m2d.shape) if m2d is not None else None
            if _grad is not None:
                grad_desc = f"populated |g|.max={_grad.abs().max().item():.3e}"
            else:
                grad_desc = "None"
            if _absgrad is not None:
                absgrad_desc = f"populated ag.max={_absgrad.max().item():.3e}"
            else:
                absgrad_desc = "absent"
            print(
                "=== DefaultStrategy sanity (step 0) ===\n"
                f"  info keys         : {list(info.keys())}\n"
                f"  means2d type      : {type(m2d).__name__}\n"
                f"  means2d.shape     : {m2d_shape}\n"
                f"  means2d.grad      : {grad_desc}\n"
                f"  means2d.absgrad   : {absgrad_desc}\n"
                f"  strategy.absgrad  : {getattr(strategy, 'absgrad', 'n/a')}\n"
                f"  grow_grad2d thr   : {getattr(strategy, 'grow_grad2d', 'n/a')}\n"
                "======================================="
            )

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step == 600 or step == 3000:
            _eval_psnr_now(step)

        if step == 600:
            # Day 9 diagnostic: observe what _grow_gs will see at the first refinement event.
            # Fires once in full runs, never in 5-iter smokes. Prints: grad2d stats, count stats,
            # scene_scale, scale distribution, and the is_small/is_grad_high counts that gate
            # duplication and splitting.
            n_before = int(params["means"].shape[0])
            print(f"\n=== pre-refine diagnostic (step 600) ===")
            print(f"  strategy_ok = {strategy_ok}")
            print(f"  N before    = {n_before}")
            if strategy_ok and isinstance(strategy_state, dict):
                g2d = strategy_state.get("grad2d")
                cnt = strategy_state.get("count")
                ssc = strategy_state.get("scene_scale")
                print(f"  scene_scale = {ssc}")
                if g2d is not None and cnt is not None:
                    mean_grad = g2d / cnt.clamp_min(1)
                    above = int((mean_grad > strategy.grow_grad2d).sum().item())
                    print(
                        f"  grad2d    : min={g2d.min().item():.3e} "
                        f"max={g2d.max().item():.3e} mean={g2d.mean().item():.3e}"
                    )
                    print(
                        f"  count     : min={int(cnt.min().item())} "
                        f"max={int(cnt.max().item())} mean={cnt.mean().item():.2f}"
                    )
                    print(
                        f"  mean_grad : min={mean_grad.min().item():.3e} "
                        f"max={mean_grad.max().item():.3e} "
                        f"mean={mean_grad.mean().item():.3e}"
                    )
                    print(
                        f"  above {strategy.grow_grad2d:.1e} thr : {above} / {int(len(g2d))}"
                    )
                    # Multi-threshold sweep — lets a single smoke expose the full
                    # above-threshold curve so tuning is data-driven, not guess-and-check.
                    for _thr in (1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2, 5e-2):
                        _n = int((mean_grad > _thr).sum().item())
                        _pct = 100.0 * _n / int(len(g2d))
                        print(f"    above {_thr:.0e} : {_n:>5d} ({_pct:5.2f}%)")
                else:
                    print(f"  grad2d/count uninitialized: g2d={g2d} cnt={cnt}")
                scales_exp = torch.exp(params["scales"]).max(dim=-1).values
                thr_small = strategy.grow_scale3d * (ssc if isinstance(ssc, (int, float)) else 1.0)
                is_small_n = int((scales_exp <= thr_small).sum().item())
                print(
                    f"  scales exp: min={scales_exp.min().item():.3e} "
                    f"max={scales_exp.max().item():.3e} mean={scales_exp.mean().item():.3e}"
                )
                print(
                    f"  is_small (scale<={thr_small:.3e}): {is_small_n} / "
                    f"{int(scales_exp.numel())}"
                )
            print("==========================================\n")

        if strategy_ok:
            try:
                # packed=True matches gsplat's packed info dict (means2d shape (nnz, 2),
                # not (C, N, 2)). The Day 8 / Day 9 smokes used packed=False and crashed
                # silently in _update_state at step 0 — masked by the broken except-guard.
                strategy.step_post_backward(
                    params=params,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=step,
                    info=info,
                    packed=True,
                )
            except Exception as e:
                # Print on FIRST failure, unconditionally — the old `if step == densify_start_iter`
                # guard could never fire because strategy_ok would already be False by the time
                # step reached 500 if any earlier step had failed.
                print(
                    f"[train_gsplat] strategy.step_post_backward disabled at step {step}: "
                    f"{type(e).__name__}: {e}"
                )
                strategy_ok = False

        if step % 500 == 0:
            n_g = int(params["means"].shape[0])
            print(
                f"[train_gsplat] step {step:5d}/{n_iterations}  "
                f"loss={loss.item():.4f}  N={n_g}"
            )
            losses_log.append(float(loss.item()))

    n_final = int(params["means"].shape[0])
    train_elapsed = time.time() - start
    print(f"[train_gsplat] training done in {train_elapsed:.1f}s  final N={n_final}")

    # --- Evaluation on held-out test views ---
    from skimage.metrics import structural_similarity as _ssim

    lpips_available = True
    try:
        import lpips as lpips_pkg

        lpips_model = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:
        print(f"[train_gsplat] LPIPS disabled: {e}")
        lpips_available = False
        lpips_model = None

    def _psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = float(np.mean((a - b) ** 2))
        if mse == 0.0:
            return 100.0
        return float(-10.0 * np.log10(mse))

    per_view: list[dict] = []
    with torch.no_grad():
        for img in test_imgs:
            viewmat = torch.from_numpy(img["viewmat"]).unsqueeze(0).to(device)
            K = torch.from_numpy(img["K"]).unsqueeze(0).to(device)
            gt_rgb_np = img["rgb"]  # (H, W, 3) float32
            H, W = img["height"], img["width"]

            colors_all = torch.cat([params["sh0"], params["shN"]], dim=1)
            renders, _, _ = rasterization(
                means=params["means"],
                quats=normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=colors_all,
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=sh_degree,
                render_mode="RGB",
            )
            pred_rgb = renders[0].clamp(0.0, 1.0)
            pred_np = pred_rgb.cpu().numpy().astype(np.float32)

            # Save rendered PNG (BGR for cv2.imwrite)
            pred_u8 = (pred_np * 255.0).round().clip(0, 255).astype(np.uint8)
            bgr_out = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{renders_dir}/{img['name']}", bgr_out)

            psnr_val = _psnr(pred_np, gt_rgb_np)
            ssim_val = float(
                _ssim(gt_rgb_np, pred_np, data_range=1.0, channel_axis=-1)
            )

            if lpips_available:
                def _to_lp(x: np.ndarray) -> "torch.Tensor":
                    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
                    return t * 2.0 - 1.0

                lpips_val = float(
                    lpips_model(_to_lp(pred_np), _to_lp(gt_rgb_np)).item()
                )
            else:
                lpips_val = None

            per_view.append(
                {
                    "name": img["name"],
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                }
            )
            print(
                f"[train_gsplat]   test  {img['name']}  "
                f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.3f}  "
                f"LPIPS={lpips_val if lpips_val is not None else 'n/a'}"
            )

    def _median(xs: list[float]) -> float:
        return float(np.median(xs))

    def _range(xs: list[float]) -> tuple[float, float]:
        return (float(min(xs)), float(max(xs)))

    psnr_vals = [v["psnr"] for v in per_view]
    ssim_vals = [v["ssim"] for v in per_view]
    lpips_vals = [v["lpips"] for v in per_view if v["lpips"] is not None]

    psnr_median = _median(psnr_vals) if psnr_vals else None
    psnr_range = _range(psnr_vals) if psnr_vals else None
    ssim_median = _median(ssim_vals) if ssim_vals else None
    ssim_range = _range(ssim_vals) if ssim_vals else None
    lpips_median = _median(lpips_vals) if lpips_vals else None
    lpips_range = _range(lpips_vals) if lpips_vals else None

    # CLAUDE.md Check 5 (baked in): hard assertion on held-out PSNR range.
    # For gsplat on DTU at 800x600, any value below 5 dB or above 60 dB
    # (or nan/inf) means the eval is broken — a numerical clamp hiding
    # identity, a data-range mismatch, or a rasterization failure. Better
    # to fail loudly here than ship wrong numbers downstream.
    # Skipped for n_iterations=0 sanity runs (expected initial PSNR depends
    # on the COLMAP sparse init and may be outside this range).
    if n_iterations >= 5 and psnr_median is not None:
        if not math_ok(psnr_median):
            return _fail(
                f"median PSNR is nan/inf: {psnr_median}. "
                f"per-view: {per_view}"
            )
        if not (5.0 < psnr_median < 60.0):
            return _fail(
                f"median PSNR {psnr_median:.2f} dB outside plausible range (5, 60). "
                f"Either the rasterization is broken, the data range is wrong, "
                f"or the evaluation is computing PSNR in the wrong space."
            )

    # --- Save checkpoint ---
    ckpt_path = f"{out_dir}/gaussians.pt"
    try:
        torch.save(
            {k: v.detach().cpu() for k, v in params.items()},
            ckpt_path,
        )
    except Exception as e:
        print(f"[train_gsplat] checkpoint save failed: {e}")

    # --- Save summary JSON on-volume for posterity ---
    summary = {
        "run_id": out_run_id,
        "colmap_run_id": colmap_run_id,
        "seed": seed,
        "n_iterations": n_iterations,
        "n_gaussians_final": n_final,
        "n_train_views": len(train_imgs),
        "n_test_views": len(test_imgs),
        "test_view_names": [v["name"] for v in per_view],
        "psnr_median": psnr_median,
        "ssim_median": ssim_median,
        "lpips_median": lpips_median,
        "per_view": per_view,
        "losses_log": losses_log,
    }
    try:
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[train_gsplat] summary save failed: {e}")

    workspace_volume.commit()

    total_elapsed = time.time() - start
    print(f"[train_gsplat] total elapsed {total_elapsed:.1f}s")

    return {
        "success": True,
        "run_id": out_run_id,
        "colmap_run_id": colmap_run_id,
        "seed": seed,
        "n_iterations": n_iterations,
        "n_gaussians_final": n_final,
        "elapsed_seconds": total_elapsed,
        "n_train_views": len(train_imgs),
        "n_test_views": len(test_imgs),
        "test_view_names": [v["name"] for v in per_view],
        "psnr_median": psnr_median,
        "psnr_range": psnr_range,
        "ssim_median": ssim_median,
        "ssim_range": ssim_range,
        "lpips_median": lpips_median,
        "lpips_range": lpips_range,
        "per_view": per_view,
        "checkpoint_path": f"{out_run_id}/gaussians.pt",
        "renders_dir": f"{out_run_id}/renders",
        "error": None,
    }
