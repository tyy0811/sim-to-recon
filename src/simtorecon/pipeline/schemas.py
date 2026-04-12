"""Pydantic models for pipeline configuration and results."""

from pathlib import Path

from pydantic import BaseModel, Field


class SceneConfig(BaseModel):
    """Configuration for a reconstruction scene."""

    name: str
    root: Path
    scan_id: int = 9
    light_idx: int = 3
    n_views: int | None = None  # None = use all available
    seed: int = 42


class PipelineConfig(BaseModel):
    """Configuration for the COLMAP dense MVS pipeline."""

    # Image resolution
    target_width: int = Field(default=800, gt=0)
    target_height: int = Field(default=600, gt=0)

    # PatchMatch parameters
    max_image_size: int = Field(default=800, gt=0)
    patch_size: int = Field(default=11, gt=0)
    num_iterations: int = Field(default=5, gt=0)
    geom_consistency: bool = True

    # Fusion parameters
    min_num_pixels: int = Field(default=5, gt=0)
    max_reproj_error: float = Field(default=2.0, gt=0)
    max_depth_error: float = Field(default=0.01, gt=0)

    # Execution mode
    use_modal: bool = Field(default=True, description="Run PatchMatch on Modal GPU")

    # Workspace
    workspace: Path = Path("results/workspace")


class ReconstructionResult(BaseModel):
    """Result of a dense MVS reconstruction."""

    model_config = {"arbitrary_types_allowed": True}

    scene_name: str
    n_views: int
    n_points: int
    output_ply: Path
    elapsed_seconds: float
    config_hash: str = ""

    # Metrics (populated after evaluation)
    chamfer: float | None = None
    accuracy: float | None = None
    completeness: float | None = None
    f_score: float | None = None
