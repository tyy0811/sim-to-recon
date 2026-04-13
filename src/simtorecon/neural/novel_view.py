"""Helpers for loading gsplat novel-view renders and their ground truth.

gsplat training runs on Modal GPU and writes rendered held-out views to
a directory on the simtorecon-workspace volume. Locally we download those
PNGs plus the matching DTU ground-truth images and hand them to
evaluation/perceptual.py for PSNR / SSIM / LPIPS scoring.

The matching ground-truth for each rendered view is the DTU image with
the same filename (gsplat preserves the COLMAP image name).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image_as_float(path: Path) -> np.ndarray:
    """Load an image from disk as float32 RGB in [0, 1]. Shape (H, W, 3)."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def load_rendered_views(
    renders_dir: Path,
    gt_dir: Path,
    names: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load (name, rendered, ground_truth) triples for a list of held-out views.

    Args:
        renders_dir: Local directory of rendered PNGs (downloaded from Modal).
        gt_dir: Local directory of DTU source images (the originals that were
            held out of training).
        names: List of image names to load.

    Returns:
        List of (name, rendered_rgb, gt_rgb) tuples, each rgb in [0, 1]
        with shape (H, W, 3) as float32. Rendered and GT are resized to the
        smaller of the two if they disagree.
    """
    triples: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in names:
        rendered = load_image_as_float(renders_dir / name)
        gt = load_image_as_float(gt_dir / name)

        if rendered.shape != gt.shape:
            import cv2

            h, w = min(rendered.shape[0], gt.shape[0]), min(
                rendered.shape[1], gt.shape[1]
            )
            rendered = cv2.resize(rendered, (w, h), interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)

        triples.append((name, rendered, gt))

    return triples
