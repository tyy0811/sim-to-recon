"""Debug reconstruction steps on Modal."""

import modal

app = modal.App("debug-recon")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libsm6", "libice6")
    .pip_install("pycolmap-cuda12", "numpy>=1.24", "opencv-python-headless>=4.8", "scipy>=1.10")
)

dtu_vol = modal.Volume.from_name("simtorecon-dtu-data")


@app.function(image=image, gpu="A10G", timeout=600, volumes={"/dtu_data": dtu_vol})
def debug_recon():
    import os
    import inspect
    import pycolmap

    print(f"pycolmap version: {pycolmap.COLMAP_version}")
    print(f"has_cuda: {pycolmap.has_cuda}")

    # Check API docs
    print("\n=== stereo_fusion doc ===")
    print(pycolmap.stereo_fusion.__doc__[:500] if pycolmap.stereo_fusion.__doc__ else "no doc")

    print("\n=== patch_match_stereo doc ===")
    print(pycolmap.patch_match_stereo.__doc__[:500] if pycolmap.patch_match_stereo.__doc__ else "no doc")

    # Check data on volume
    scan9 = "/dtu_data/scan9"
    print(f"\n=== Data check ===")
    for subdir in ["images", "calibration", "gt"]:
        path = f"{scan9}/{subdir}"
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"  {subdir}: {len(files)} files")
            if files:
                print(f"    first: {sorted(files)[0]}")
                print(f"    last:  {sorted(files)[-1]}")
        else:
            print(f"  {subdir}: NOT FOUND")

    return "done"


@app.local_entrypoint()
def main():
    result = debug_recon.remote()
    print(result)
