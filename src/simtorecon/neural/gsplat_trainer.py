"""gsplat training configuration and result types.

The actual training loop lives inside modal_app.py::train_gsplat because
gsplat requires CUDA and runs on Modal A10G. This module defines the
typed interface the local side uses to invoke and consume that function.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class GsplatConfig(BaseModel):
    """Configuration for a gsplat training run.

    Defaults follow gsplat's simple_trainer.py recipe with the iteration
    count reduced from 30000 to 7000 per the V1.5 spec (Day 8 goal is a
    smoke run, not a paper-quality baseline).
    """

    # Initialization — points to a COLMAP sparse reconstruction on the
    # simtorecon-workspace Modal volume (e.g. V1's scan9_v49_s123_3d428b).
    colmap_run_id: str = Field(
        ..., description="COLMAP run_id on simtorecon-workspace volume"
    )

    # Training
    n_iterations: int = Field(default=7000, gt=0)
    seed: int = Field(default=42)
    lr_means: float = Field(default=1.6e-4, gt=0)
    lr_scales: float = Field(default=5e-3, gt=0)
    lr_quats: float = Field(default=1e-3, gt=0)
    lr_opacities: float = Field(default=5e-2, gt=0)
    lr_sh0: float = Field(default=2.5e-3, gt=0)
    lr_shN: float = Field(default=2.5e-3 / 20.0, gt=0)
    ssim_lambda: float = Field(default=0.2, ge=0, le=1)
    random_bkgd: bool = Field(default=True)

    # Densification (DefaultStrategy)
    densify_start_iter: int = Field(default=500, ge=0)
    densify_stop_iter: int = Field(default=5000, ge=0)
    densify_grad_threshold: float = Field(default=2e-4, gt=0)
    reset_opacity_iter: int = Field(default=3000, ge=0)

    # Evaluation split (every k-th image becomes a test view)
    # With 49 DTU images, test_every=10 gives 5 test / 44 train.
    test_every: int = Field(default=10, ge=2)

    # SH degree for view-dependent color
    sh_degree: int = Field(default=3, ge=0, le=3)


class GsplatResult(BaseModel):
    """Result of a gsplat training run returned from Modal."""

    model_config = {"arbitrary_types_allowed": True}

    success: bool
    run_id: str  # gsplat output run_id: gsplat_{colmap_run_id}_s{seed}
    colmap_run_id: str
    seed: int

    # Training metadata
    n_iterations: int = 0
    n_gaussians_final: int = 0
    elapsed_seconds: float = 0.0

    # Split info
    n_train_views: int = 0
    n_test_views: int = 0
    test_view_names: list[str] = []

    # Held-out metrics (median across test views)
    psnr_median: float | None = None
    psnr_range: tuple[float, float] | None = None
    ssim_median: float | None = None
    ssim_range: tuple[float, float] | None = None
    lpips_median: float | None = None
    lpips_range: tuple[float, float] | None = None

    # Per-view metrics (one entry per held-out test view)
    per_view: list[dict] = []

    # Volume paths (all paths are relative to simtorecon-workspace mount)
    checkpoint_path: str | None = None  # .pt file with Gaussian parameters
    renders_dir: str | None = None  # directory of {test_view_name}.png files

    error: str | None = None


def format_run_id(colmap_run_id: str, seed: int) -> str:
    """gsplat output directory naming convention on the workspace volume."""
    return f"gsplat_{colmap_run_id}_s{seed}"


def resolve_local_output_dir(root: Path, result: GsplatResult) -> Path:
    """Where run_gsplat.py writes the downloaded artifacts."""
    return root / result.run_id
