"""Debug pycolmap-cuda12 import on Modal."""

import modal

app = modal.App("debug-pycolmap")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libsm6", "libice6")
    .pip_install("pycolmap-cuda12", "numpy>=1.24")
)


@app.function(image=image, gpu="A10G", timeout=300)
def test_pycolmap():
    import subprocess
    import os

    # Check CUDA
    print("=== CUDA check ===")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")

    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(r.stdout[:500] if r.returncode == 0 else f"nvidia-smi failed: {r.stderr}")

    # Check what pycolmap installed
    print("\n=== pycolmap files ===")
    r = subprocess.run(
        ["find", "/usr/local/lib/python3.10", "-path", "*/pycolmap/*", "-name", "*.so"],
        capture_output=True, text=True,
    )
    print(r.stdout)

    # Check missing shared libs
    print("=== ldd on _core ===")
    so_files = r.stdout.strip().split("\n")
    for so in so_files:
        if so:
            r2 = subprocess.run(["ldd", so], capture_output=True, text=True)
            missing = [l for l in r2.stdout.splitlines() if "not found" in l]
            if missing:
                print(f"{so}:")
                for m in missing:
                    print(f"  {m}")
            else:
                print(f"{so}: all deps satisfied")

    # Try import
    print("\n=== import test ===")
    try:
        import pycolmap
        print(f"pycolmap imported OK, has_cuda={pycolmap.has_cuda}")
    except Exception as e:
        print(f"Import failed: {e}")

    return "done"


@app.local_entrypoint()
def main():
    result = test_pycolmap.remote()
    print(result)
