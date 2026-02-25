"""Comprehensive tests for signal drift models."""

import numpy as np
import pytest

from neurosim.models.drift import ElectrodeDriftModel, FatigueDriftModel, FeatureShiftModel


SEED = 42
SHAPE_2D = (4, 100)


class TestElectrodeDriftModel:
    """Tests for impedance-based electrode drift."""

    def test_electrode_drift_attenuates_signal(self):
        """At high timestep, impedance increases and signal amplitude drops."""
        model = ElectrodeDriftModel(alpha_z=0.01, sigma_z=0.001, seed=SEED)
        signal = np.ones(SHAPE_2D)
        drifted = model.apply(signal, timestep=200)
        # Z(t) >> Z0 at t=200, so attenuation should be significant
        assert np.mean(np.abs(drifted)) < np.mean(np.abs(signal)) * 0.5

    def test_electrode_drift_no_effect_at_t0(self):
        """At timestep=0, signal is roughly unchanged (only noise epsilon)."""
        model = ElectrodeDriftModel(alpha_z=0.001, sigma_z=0.001, seed=SEED)
        signal = np.ones(SHAPE_2D) * 5.0
        drifted = model.apply(signal, timestep=0)
        # At t=0, impedance_ratio = 1 + eps_z, so output ~ signal within noise
        np.testing.assert_allclose(drifted, signal, atol=0.1)

    def test_electrode_drift_reproducible_with_seed(self):
        """Same seed produces identical drift output."""
        signal = np.random.default_rng(0).standard_normal(SHAPE_2D)
        m1 = ElectrodeDriftModel(seed=SEED)
        m2 = ElectrodeDriftModel(seed=SEED)
        out1 = m1.apply(signal, timestep=50)
        out2 = m2.apply(signal, timestep=50)
        np.testing.assert_array_equal(out1, out2)

    def test_electrode_drift_per_channel(self):
        """Different channels drift independently due to per-channel noise."""
        model = ElectrodeDriftModel(alpha_z=0.001, sigma_z=0.05, seed=SEED)
        signal = np.ones(SHAPE_2D)
        drifted = model.apply(signal, timestep=10)
        # Per-channel attenuation should differ (different eps_z per channel)
        channel_means = np.mean(drifted, axis=1)
        assert not np.allclose(
            channel_means, channel_means[0] * np.ones(SHAPE_2D[0])
        ), "All channels have identical drift -- they should be independent"


class TestFatigueDriftModel:
    """Tests for neural fatigue attenuation."""

    def test_fatigue_drift_attenuates_over_time(self):
        """Signal at t=500 should be weaker than at t=0."""
        model = FatigueDriftModel(beta_f=0.3, tau_f=100.0, noise_std=0.0, seed=SEED)
        signal = np.ones(SHAPE_2D) * 10.0
        out_early = model.apply(signal, timestep=0)
        model.reset()
        out_late = model.apply(signal, timestep=500)
        assert np.mean(np.abs(out_late)) < np.mean(np.abs(out_early))

    def test_fatigue_drift_asymptotic(self):
        """Fatigue factor never goes below (1 - beta_f)."""
        beta_f = 0.3
        model = FatigueDriftModel(beta_f=beta_f, tau_f=100.0, noise_std=0.0, seed=SEED)
        signal = np.ones(SHAPE_2D) * 10.0
        # At very large t, fatigue(t) -> 1 - beta_f = 0.7
        out = model.apply(signal, timestep=100_000)
        expected_floor = signal * (1.0 - beta_f)
        np.testing.assert_allclose(out, expected_floor, atol=0.01)

    def test_fatigue_drift_reset(self):
        """After reset, drift restarts from initial RNG state."""
        model = FatigueDriftModel(seed=SEED)
        signal = np.ones(SHAPE_2D)
        first = model.apply(signal, timestep=50)
        model.reset()
        second = model.apply(signal, timestep=50)
        np.testing.assert_array_equal(first, second)


class TestFeatureShiftModel:
    """Tests for feature distribution drift."""

    def test_feature_shift_adds_offset(self):
        """Signal mean shifts over time due to drift_rate."""
        model = FeatureShiftModel(
            drift_rate=0.1, step_probability=0.0, seed=SEED
        )
        signal = np.zeros(SHAPE_2D)
        shifted = model.apply(signal, timestep=100)
        # continuous_shift = 0.1 * 100 = 10.0; step_shifts = 0
        np.testing.assert_allclose(shifted, 10.0, atol=1e-10)

    def test_feature_shift_step_shifts(self):
        """With high step_probability, accumulated shift grows over calls."""
        model = FeatureShiftModel(
            drift_rate=0.0, step_probability=1.0, step_magnitude=1.0, seed=SEED
        )
        signal = np.zeros(SHAPE_2D)
        # Call multiple times to accumulate steps
        for t in range(20):
            model.apply(signal, timestep=0)
        # After 20 calls with p=1.0, accumulated_step_shift should be nonzero
        assert model._accumulated_step_shift != 0.0

    def test_feature_shift_reset_clears_accumulated(self):
        """Reset clears step shift state and re-seeds RNG."""
        model = FeatureShiftModel(step_probability=1.0, seed=SEED)
        signal = np.zeros(SHAPE_2D)
        for t in range(10):
            model.apply(signal, timestep=t)
        assert model._accumulated_step_shift != 0.0
        model.reset()
        assert model._accumulated_step_shift == 0.0
        # And the next call should match a fresh model
        fresh = FeatureShiftModel(step_probability=1.0, seed=SEED)
        result_reset = model.apply(signal, timestep=5)
        result_fresh = fresh.apply(signal, timestep=5)
        np.testing.assert_array_equal(result_reset, result_fresh)
