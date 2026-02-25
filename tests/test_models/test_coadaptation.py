"""Comprehensive tests for co-adaptation model."""

import numpy as np
import pytest

from neurosim.models.coadaptation import CoAdaptationModel


SEED = 42
SHAPE_2D = (4, 100)


@pytest.fixture
def model():
    return CoAdaptationModel(
        adaptation_rate_range=(0.1, 0.5), tau_adapt=50.0, seed=SEED
    )


@pytest.fixture
def signal():
    return np.random.default_rng(0).standard_normal(SHAPE_2D)


@pytest.fixture
def expectation():
    return np.random.default_rng(99).standard_normal(SHAPE_2D)


class TestCoAdaptationBehavior:
    """Tests for adaptation dynamics."""

    def test_coadaptation_increases_over_time(self, model, signal, expectation):
        """Alpha at t=100 should be greater than alpha at t=1."""
        model.adapt(signal, expectation, timestep=1)
        alpha_early = model._history[-1]

        model.adapt(signal, expectation, timestep=100)
        alpha_late = model._history[-1]

        assert alpha_late > alpha_early, (
            f"Adaptation should increase: alpha(t=1)={alpha_early:.4f}, "
            f"alpha(t=100)={alpha_late:.4f}"
        )

    def test_coadaptation_preserves_shape(self, model, signal, expectation):
        """Output shape must match input shape."""
        adapted = model.adapt(signal, expectation, timestep=10)
        assert adapted.shape == signal.shape

    def test_coadaptation_blends_toward_expectation(self, model, signal, expectation):
        """At high timestep, output should be closer to expectation than to signal."""
        # Use a fresh model and high timestep for strong adaptation
        m = CoAdaptationModel(
            adaptation_rate_range=(0.8, 0.9), tau_adapt=10.0, seed=SEED
        )
        adapted = m.adapt(signal, expectation, timestep=500)

        dist_to_signal = np.mean(np.abs(adapted - signal))
        dist_to_expectation = np.mean(np.abs(adapted - expectation))

        assert dist_to_expectation < dist_to_signal, (
            f"Adapted signal should be closer to expectation "
            f"(dist={dist_to_expectation:.4f}) than to original signal "
            f"(dist={dist_to_signal:.4f})"
        )

    def test_coadaptation_zero_at_start(self, signal, expectation):
        """At t=0, output should approximately equal original signal."""
        m = CoAdaptationModel(
            adaptation_rate_range=(0.1, 0.5), tau_adapt=50.0, seed=SEED
        )
        adapted = m.adapt(signal, expectation, timestep=0)
        # alpha(0) = alpha_base * (1 - exp(0)) = 0, plus small noise
        np.testing.assert_allclose(adapted, signal, atol=0.15)


class TestCoAdaptationReset:
    """Tests for reset behavior."""

    def test_coadaptation_reset(self, signal, expectation):
        """After reset, adaptation restarts from zero with same RNG."""
        m = CoAdaptationModel(seed=SEED)
        first = m.adapt(signal, expectation, timestep=10)
        m.reset()
        second = m.adapt(signal, expectation, timestep=10)
        np.testing.assert_array_equal(first, second)


class TestAdaptationCurve:
    """Tests for adaptation history tracking."""

    def test_adaptation_curve_length(self, model, signal, expectation):
        """Curve length should match number of adapt() calls."""
        n_calls = 15
        for t in range(n_calls):
            model.adapt(signal, expectation, timestep=t)
        curve = model.get_adaptation_curve()
        assert len(curve) == n_calls


class TestCoAdaptationValidation:
    """Tests for input validation."""

    def test_shape_mismatch_raises(self, model):
        """Different shapes for signal and expectation should raise ValueError."""
        signal = np.ones((4, 100))
        bad_expectation = np.ones((3, 50))
        with pytest.raises(ValueError, match="Shape mismatch"):
            model.adapt(signal, bad_expectation, timestep=1)
