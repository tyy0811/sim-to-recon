"""Perceptual similarity metrics for novel-view evaluation: PSNR, SSIM, LPIPS.

These metrics consume two images in [0, 1] float32 RGB format (shape
H x W x 3, same size) and return a single float. They exist alongside
evaluation/metrics.py (geometric) because V1.5 compares COLMAP vs 3DGS
on novel-view photometric quality — the only axis on which both methods
can be scored fairly.

PSNR and SSIM are implemented via scikit-image and run on CPU.
LPIPS requires torch; the import is deferred so CPU-only environments
can still use PSNR/SSIM for tests and sanity checks.
"""

from __future__ import annotations

import math

import numpy as np

_PSNR_MAX_FINITE = 100.0  # cap for identical inputs (PSNR is mathematically inf)


def _check_pair(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}"
        )
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {pred.shape}")
    if pred.dtype != np.float32 and pred.dtype != np.float64:
        raise ValueError(f"expected float dtype, got {pred.dtype}")


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio. Identical inputs return a finite cap (100 dB).

    Higher is better. Typical 3DGS novel-view PSNR on DTU scan9 at 800x600
    lands in the mid-20s to low-30s dB range.
    """
    _check_pair(pred, gt)
    mse = float(np.mean((pred - gt) ** 2))
    if mse == 0.0:
        return _PSNR_MAX_FINITE
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity. Returns 1.0 for identical inputs, 0 or negative otherwise.

    Uses scikit-image's reference implementation (channel_axis=-1 for RGB).
    """
    _check_pair(pred, gt)
    from skimage.metrics import structural_similarity as _ssim

    return float(_ssim(gt, pred, data_range=data_range, channel_axis=-1))


def lpips(
    pred: np.ndarray,
    gt: np.ndarray,
    net: str = "alex",
    device: str | None = None,
) -> float:
    """Learned perceptual image patch similarity.

    Lower is better (identical inputs return ~0). Requires torch + lpips.
    For CPU-only smoke tests, call this with device='cpu' — the AlexNet
    backbone is small enough to run on CPU in a second or two per image.
    """
    _check_pair(pred, gt)
    import lpips as _lpips_pkg
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()

    # lpips expects tensors in [-1, 1] with shape (1, 3, H, W)
    def _to_tensor(img: np.ndarray):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        return t * 2.0 - 1.0

    with torch.no_grad():
        val = model(_to_tensor(pred.astype(np.float32)), _to_tensor(gt.astype(np.float32)))
    return float(val.item())


def psnr_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]], data_range: float = 1.0
) -> list[float]:
    """PSNR across a batch of (pred, gt) pairs."""
    return [psnr(p, g, data_range=data_range) for p, g in pairs]


def ssim_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]], data_range: float = 1.0
) -> list[float]:
    """SSIM across a batch of (pred, gt) pairs."""
    return [ssim(p, g, data_range=data_range) for p, g in pairs]


def lpips_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    net: str = "alex",
    device: str | None = None,
) -> list[float]:
    """LPIPS across a batch of (pred, gt) pairs.

    Loads the model once and reuses it — much faster than calling lpips()
    in a loop.
    """
    if not pairs:
        return []

    import lpips as _lpips_pkg
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()

    def _to_tensor(img: np.ndarray):
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
        return t * 2.0 - 1.0

    out: list[float] = []
    with torch.no_grad():
        for p, g in pairs:
            _check_pair(p, g)
            val = model(_to_tensor(p), _to_tensor(g))
            out.append(float(val.item()))
    return out
