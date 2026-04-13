"""Typed wrapper for invoking the Modal-hosted gsplat training function.

Mirrors the pattern in experiments/run_baseline.py::
    reconstruct = modal.Function.from_name("simtorecon-mvs", "reconstruct_dtu_scan9")
    result = reconstruct.remote(...)

The raw Modal dict gets validated into a GsplatResult so downstream code
can rely on a typed interface.
"""

from __future__ import annotations

from simtorecon.neural.gsplat_trainer import GsplatConfig, GsplatResult


def run_gsplat_on_modal(config: GsplatConfig) -> GsplatResult:
    """Invoke the deployed train_gsplat Modal function.

    Requires `modal deploy modal_app.py` to have been run at least once.
    Raises on Modal-side exceptions; a structured failure (e.g. a gsplat
    training divergence) returns a GsplatResult with success=False.
    """
    import modal

    train_gsplat = modal.Function.from_name("simtorecon-mvs", "train_gsplat")

    raw = train_gsplat.remote(
        colmap_run_id=config.colmap_run_id,
        n_iterations=config.n_iterations,
        seed=config.seed,
        lr_means=config.lr_means,
        lr_scales=config.lr_scales,
        lr_quats=config.lr_quats,
        lr_opacities=config.lr_opacities,
        lr_sh0=config.lr_sh0,
        lr_shN=config.lr_shN,
        ssim_lambda=config.ssim_lambda,
        densify_start_iter=config.densify_start_iter,
        densify_stop_iter=config.densify_stop_iter,
        densify_grad_threshold=config.densify_grad_threshold,
        reset_opacity_iter=config.reset_opacity_iter,
        test_every=config.test_every,
        sh_degree=config.sh_degree,
    )

    return GsplatResult(**raw)
