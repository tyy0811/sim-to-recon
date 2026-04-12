"""Test zip extraction logic locally with a small synthetic zip.

Simulates the exact subprocess calls the Modal download function makes,
so we can debug path matching and unzip behavior without downloading 6GB.
"""

import os
import subprocess
import tempfile
import zipfile
from pathlib import Path


def create_test_zip(zip_path: str, structure: dict[str, str]) -> None:
    """Create a zip with the given path→content mapping."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path, content in structure.items():
            zf.writestr(path, content)


def test_sampleset_extraction():
    """Simulate SampleSet.zip extraction for calibration files."""
    print("=== Testing SampleSet extraction ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a zip mimicking DTU SampleSet structure
        # Based on Modal logs: path is "Data/Calibration/cal18/pos_XXX.txt"
        zip_path = f"{tmpdir}/SampleSet.zip"
        structure = {}
        for i in range(64):
            structure[f"Data/Calibration/cal18/pos_{i:03d}.txt"] = f"calib data {i}"
        structure["Data/readme.txt"] = "readme"
        structure["Data/Evaluation/some_file.m"] = "eval code"

        create_test_zip(zip_path, structure)
        print(f"Created test zip with {len(structure)} files")

        # Step 1: List zip and find calibration paths (same logic as modal_app.py)
        ls_result = subprocess.run(
            ["unzip", "-l", zip_path],
            capture_output=True, text=True,
        )
        cal_paths = [
            ln.split()[-1] for ln in ls_result.stdout.splitlines()
            if "pos_" in ln and ln.strip().endswith(".txt")
        ]
        print(f"Found {len(cal_paths)} calibration files")
        if cal_paths:
            print(f"  First: {cal_paths[0]}")
            print(f"  Last:  {cal_paths[-1]}")
            cal_prefix = "/".join(cal_paths[0].split("/")[:-1])
            print(f"  Prefix: {cal_prefix}")

        # Step 2: Test glob extraction (what was failing)
        calib_dir = f"{tmpdir}/calibration"
        os.makedirs(calib_dir)

        print("\nTest A: unzip with glob pattern (f'{cal_prefix}/*')")
        result_a = subprocess.run(
            ["unzip", "-o", "-j", zip_path, f"{cal_prefix}/*", "-d", calib_dir],
            capture_output=True, text=True,
        )
        count_a = len(os.listdir(calib_dir))
        print(f"  Exit code: {result_a.returncode}")
        print(f"  Extracted: {count_a} files")
        if result_a.returncode != 0:
            print(f"  Stderr: {result_a.stderr[:200]}")

        # Clean for next test
        for f in os.listdir(calib_dir):
            os.remove(f"{calib_dir}/{f}")

        print("\nTest B: unzip each file individually")
        for cal_path in cal_paths[:3]:  # just test 3
            r = subprocess.run(
                ["unzip", "-o", "-j", zip_path, cal_path, "-d", calib_dir],
                capture_output=True, text=True,
            )
        count_b = len(os.listdir(calib_dir))
        print(f"  Extracted: {count_b} files (from 3 attempted)")

        # Clean for next test
        for f in os.listdir(calib_dir):
            os.remove(f"{calib_dir}/{f}")

        print("\nTest C: unzip with shell=True for glob expansion")
        cmd = f'unzip -o -j {zip_path} "{cal_prefix}/*" -d {calib_dir}'
        result_c = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        count_c = len(os.listdir(calib_dir))
        print(f"  Exit code: {result_c.returncode}")
        print(f"  Extracted: {count_c} files")

        # Clean for next test
        for f in os.listdir(calib_dir):
            os.remove(f"{calib_dir}/{f}")

        print("\nTest D: python zipfile extraction")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if "pos_" in name and name.endswith(".txt"):
                    # Extract flat (junk paths)
                    data = zf.read(name)
                    basename = os.path.basename(name)
                    with open(f"{calib_dir}/{basename}", "wb") as f:
                        f.write(data)
        count_d = len(os.listdir(calib_dir))
        print(f"  Extracted: {count_d} files")

        print(f"\n=== Results ===")
        print(f"  Glob pattern:     {count_a} files (exit {result_a.returncode})")
        print(f"  Individual files: {count_b}/3 files")
        print(f"  Shell glob:       {count_c} files (exit {result_c.returncode})")
        print(f"  Python zipfile:   {count_d} files")


def test_rectified_extraction():
    """Simulate Rectified.zip extraction for scan9 images."""
    print("\n=== Testing Rectified extraction ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = f"{tmpdir}/Rectified.zip"
        structure = {}
        # DTU Rectified naming: rect_XXX_Y_r5000.png
        # 7 lighting conditions (0-6), 49 images per scan
        for scan in [1, 9, 15]:
            for img in range(1, 50):
                for light in range(7):
                    name = f"Rectified/scan{scan}_train/rect_{img:03d}_{light}_r5000.png"
                    structure[name] = f"image data scan{scan} img{img} light{light}"

        create_test_zip(zip_path, structure)
        print(f"Created test zip with {len(structure)} entries")

        # Find scan9 images
        ls_result = subprocess.run(
            ["unzip", "-l", zip_path],
            capture_output=True, text=True,
        )
        scan9_paths = [
            ln.split()[-1] for ln in ls_result.stdout.splitlines()
            if "scan9" in ln.lower() and ln.strip().endswith(".png")
        ]
        print(f"Found {len(scan9_paths)} scan9 images")
        if scan9_paths:
            print(f"  First: {scan9_paths[0]}")
            scan9_prefix = "/".join(scan9_paths[0].split("/")[:-1])
            print(f"  Prefix: {scan9_prefix}")

        images_dir = f"{tmpdir}/images"
        os.makedirs(images_dir)

        print("\nTest A: glob pattern")
        result_a = subprocess.run(
            ["unzip", "-o", "-j", zip_path, f"{scan9_prefix}/*", "-d", images_dir],
            capture_output=True, text=True,
        )
        count_a = len(os.listdir(images_dir))
        print(f"  Exit code: {result_a.returncode}, extracted: {count_a}")

        for f in os.listdir(images_dir):
            os.remove(f"{images_dir}/{f}")

        print("\nTest B: python zipfile")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if "scan9" in name and name.endswith(".png"):
                    data = zf.read(name)
                    basename = os.path.basename(name)
                    with open(f"{images_dir}/{basename}", "wb") as f:
                        f.write(data)
        count_b = len(os.listdir(images_dir))
        print(f"  Extracted: {count_b} files")

        print(f"\n=== Results ===")
        print(f"  Glob:   {count_a} (exit {result_a.returncode})")
        print(f"  Python: {count_b}")


if __name__ == "__main__":
    test_sampleset_extraction()
    test_rectified_extraction()
