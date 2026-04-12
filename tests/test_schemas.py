"""Tests for Pydantic schema validation and round-trips."""

import json
from pathlib import Path

import pytest

from simtorecon.pipeline.schemas import PipelineConfig, ReconstructionResult, SceneConfig


class TestSceneConfig:
    def test_defaults(self):
        cfg = SceneConfig(name="test", root=Path("/tmp"))
        assert cfg.scan_id == 9
        assert cfg.seed == 42

    def test_round_trip(self):
        cfg = SceneConfig(name="test", root=Path("/tmp/data"), n_views=30)
        data = json.loads(cfg.model_dump_json())
        recovered = SceneConfig(**data)
        assert recovered.name == cfg.name
        assert recovered.n_views == 30


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.target_width == 800
        assert cfg.target_height == 600
        assert cfg.geom_consistency is True

    def test_validation_rejects_negative(self):
        with pytest.raises(Exception):
            PipelineConfig(target_width=-1)


class TestReconstructionResult:
    def test_round_trip(self):
        result = ReconstructionResult(
            scene_name="test",
            n_views=49,
            n_points=100000,
            output_ply=Path("/tmp/dense.ply"),
            elapsed_seconds=120.5,
            config_hash="abc123",
            chamfer=1.5,
            f_score=0.85,
        )
        data = json.loads(result.model_dump_json())
        recovered = ReconstructionResult(**data)
        assert recovered.scene_name == "test"
        assert recovered.f_score == pytest.approx(0.85)
