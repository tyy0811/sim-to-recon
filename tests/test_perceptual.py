"""Tests for perceptual novel-view metrics.

PSNR and SSIM are tested against known sanity invariants (identity,
additive noise bands, channel symmetry). LPIPS is tested only if torch
and lpips are importable — CI environments without torch should still
pass the PSNR/SSIM portions.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from simtorecon.evaluation.perceptual import psnr, ssim


def _make_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)).astype(np.float32)


# ---------- PSNR ----------


def test_psnr_identical_returns_finite_cap():
    img = _make_image()
    assert psnr(img, img) == 100.0  # capped for identical inputs, not inf


def test_psnr_known_noise_is_in_plausible_range():
    """Additive Gaussian noise with known sigma gives a predictable PSNR.

    PSNR = 20 log10(1) - 10 log10(MSE) = -10 log10(sigma^2) for small noise
    on images in [0, 1]. For sigma=0.01, expected PSNR ≈ 40 dB.
    """
    img = _make_image()
    rng = np.random.default_rng(42)
    noisy = np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)

    val = psnr(img, noisy)
    assert 35.0 < val < 45.0, f"PSNR {val} outside plausible range for sigma=0.01"


def test_psnr_symmetric():
    a = _make_image(seed=1)
    b = _make_image(seed=2)
    assert psnr(a, b) == pytest.approx(psnr(b, a), abs=1e-6)


def test_psnr_raises_on_shape_mismatch():
    a = _make_image(64, 64)
    b = _make_image(32, 32)
    with pytest.raises(ValueError):
        psnr(a, b)


# ---------- SSIM ----------


def test_ssim_identical_returns_one():
    img = _make_image()
    val = ssim(img, img)
    assert val == pytest.approx(1.0, abs=1e-5)


def test_ssim_bounded_in_plausible_range():
    a = _make_image(seed=1)
    b = _make_image(seed=2)
    val = ssim(a, b)
    # Two independent random images should give low SSIM (not bounded to [0,1]
    # for arbitrary inputs, but in practice lands near 0 for random noise).
    assert -0.2 < val < 0.3, f"SSIM {val} implausible for independent random images"


def test_ssim_raises_on_shape_mismatch():
    a = _make_image(64, 64)
    b = _make_image(32, 32)
    with pytest.raises(ValueError):
        ssim(a, b)


# ---------- LPIPS (optional, only if torch + lpips are available) ----------


_has_lpips = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("lpips") is not None
)


@pytest.mark.skipif(not _has_lpips, reason="torch + lpips not installed")
def test_lpips_identical_is_near_zero():
    from simtorecon.evaluation.perceptual import lpips as lpips_metric

    img = _make_image(h=128, w=128)
    val = lpips_metric(img, img, device="cpu")
    assert val < 0.05, f"LPIPS on identical images should be ~0, got {val}"


@pytest.mark.skipif(not _has_lpips, reason="torch + lpips not installed")
def test_lpips_different_is_larger_than_identical():
    from simtorecon.evaluation.perceptual import lpips as lpips_metric

    a = _make_image(h=128, w=128, seed=1)
    b = _make_image(h=128, w=128, seed=2)
    val_diff = lpips_metric(a, b, device="cpu")
    val_same = lpips_metric(a, a, device="cpu")
    assert val_diff > val_same, (
        f"LPIPS(different) should exceed LPIPS(identical), "
        f"got {val_diff} vs {val_same}"
    )
