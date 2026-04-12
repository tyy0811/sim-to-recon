"""Dataset adapter base class — lifted from sim-to-data / demandops-lite pattern."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class DatasetAdapter(ABC):
    """Base class for scene datasets.

    Provides a uniform interface for accessing images, camera parameters,
    and ground truth across different data sources (DTU, phone scenes, etc.).
    """

    @abstractmethod
    def __init__(self, root: Path) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scene identifier."""
        ...

    @property
    @abstractmethod
    def n_images(self) -> int:
        """Number of images in the scene."""
        ...

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        """Load image at index idx as H×W×3 uint8 array."""
        ...

    @abstractmethod
    def get_intrinsics(self, idx: int) -> np.ndarray:
        """Return 3×3 intrinsics matrix for image idx."""
        ...

    @abstractmethod
    def get_pose(self, idx: int) -> np.ndarray:
        """Return 4×4 world-to-camera pose for image idx."""
        ...

    @abstractmethod
    def get_image_size(self) -> tuple[int, int]:
        """Return (height, width) of images."""
        ...

    @abstractmethod
    def has_ground_truth(self) -> bool:
        """Whether ground truth geometry is available."""
        ...
