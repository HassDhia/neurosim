"""Comprehensive tests for SignalPipeline.

Tests the composition pipeline that chains drift, noise, and co-adaptation
models sequentially for use in NeuroSim environments.
"""

import numpy as np
import pytest

from neurosim.models.pipeline import SignalPipeline
from neurosim.models.drift import ElectrodeDriftModel


SEED = 42
SHAPE_2D = (4, 100)
SHAPE_1D = (24,)


@pytest.fixture
def clean_signal_2d():
    return np.random.default_rng(0).standard_normal(SHAPE_2D)


@pytest.fixture
def clean_signal_1d():
    return np.random.default_rng(0).standard_normal(SHAPE_1D)


class TestPipelineBasic:
    """Tests for basic pipeline composition."""

    def test_pipeline_empty(self, clean_signal_2d):
        """Pipeline with no models returns a copy of input unchanged."""
        pipe = SignalPipeline(seed=SEED)
        result = pipe.apply(clean_signal_2d, timestep=10)
        np.testing.assert_array_equal(result, clean_signal_2d)
        # Should be a copy, not the same object
        assert result is not clean_signal_2d

    def test_pipeline_applies_drift(self, clean_signal_2d):
        """Pipeline with one drift model should change the signal."""
        drift = ElectrodeDriftModel(alpha_z=0.01, sigma_z=0.01, seed=SEED)
        pipe = SignalPipeline(drift_models=[drift], seed=SEED)
        result = pipe.apply(clean_signal_2d, timestep=100)
        assert not np.allclose(result, clean_signal_2d)


class TestPipelinePresets:
    """Tests for from_preset factory method."""

    def test_pipeline_from_preset_none(self, clean_signal_2d):
        """'none' preset returns signal unchanged."""
        pipe = SignalPipeline.from_preset("none", seed=SEED)
        result = pipe.apply(clean_signal_2d, timestep=50)
        np.testing.assert_array_equal(result, clean_signal_2d)

    def test_pipeline_from_preset_full(self, clean_signal_2d):
        """'full' preset applies all model types and changes signal."""
        pipe = SignalPipeline.from_preset("full", seed=SEED)
        result = pipe.apply(clean_signal_2d, timestep=50)
        assert not np.allclose(result, clean_signal_2d)
        # Full preset should have drift_models, noise_injector, and coadaptation
        assert len(pipe.drift_models) >= 3
        assert pipe.noise_injector is not None
        assert pipe.coadaptation is not None

    def test_pipeline_from_preset_invalid_raises(self):
        """Invalid preset name raises ValueError."""
        with pytest.raises(ValueError):
            SignalPipeline.from_preset("nonexistent", seed=SEED)


class TestPipelineReset:
    """Tests for reset reproducibility."""

    def test_pipeline_reset(self, clean_signal_2d):
        """After reset, pipeline produces same results with same seed."""
        drift = ElectrodeDriftModel(alpha_z=0.01, sigma_z=0.01, seed=SEED)
        pipe = SignalPipeline(drift_models=[drift], seed=SEED)

        first = pipe.apply(clean_signal_2d, timestep=50)
        pipe.reset()
        second = pipe.apply(clean_signal_2d, timestep=50)
        np.testing.assert_array_equal(first, second)


class TestPipelineShapes:
    """Tests for different input dimensionalities."""

    def test_pipeline_1d_input(self, clean_signal_1d):
        """Pipeline works on 1D feature vectors (n_features,)."""
        pipe = SignalPipeline.from_preset("none", seed=SEED)
        result = pipe.apply(clean_signal_1d, timestep=0)
        assert result.shape == SHAPE_1D

    def test_pipeline_2d_input(self, clean_signal_2d):
        """Pipeline works on 2D signals (n_channels, n_timepoints)."""
        pipe = SignalPipeline.from_preset("none", seed=SEED)
        result = pipe.apply(clean_signal_2d, timestep=0)
        assert result.shape == SHAPE_2D
