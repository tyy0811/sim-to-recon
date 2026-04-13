"""3D Gaussian Splatting (gsplat) integration for V1.5.

This subpackage is the local-side orchestration for 3DGS training.
All heavy compute happens on Modal (gsplat requires CUDA). See
modal_app.py::train_gsplat for the GPU-side training function.
"""

from simtorecon.neural.gsplat_trainer import GsplatConfig, GsplatResult
from simtorecon.neural.modal_runner import run_gsplat_on_modal
from simtorecon.neural.novel_view import load_rendered_views

__all__ = [
    "GsplatConfig",
    "GsplatResult",
    "run_gsplat_on_modal",
    "load_rendered_views",
]
