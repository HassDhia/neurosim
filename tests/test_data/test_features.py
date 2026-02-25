"""Tests for feature extraction module."""
import numpy as np
import pytest

from neurosim.data.features import (
    DEFAULT_BANDS,
    extract_band_power,
    extract_features,
    extract_log_variance,
)
from neurosim.data.formats import NeuralEpoch


def make_epoch(
    n_channels: int = 22,
    n_timepoints: int = 500,
    sfreq: float = 250.0,
    label: int = 0,
) -> NeuralEpoch:
    rng = np.random.default_rng(42)
    signals = rng.standard_normal((n_channels, n_timepoints))
    channels = [f"Ch{i:03d}" for i in range(n_channels)]
    return NeuralEpoch(
        signals=signals,
        label=label,
        sfreq=sfreq,
        channels=channels,
        subject_id=1,
    )


class TestExtractBandPower:
    def test_band_power_shape(self):
        """22 channels, 5 default bands -> output shape (110,)."""
        epoch = make_epoch(n_channels=22)
        result = extract_band_power(epoch)
        assert result.shape == (22 * 5,)

    def test_band_power_nonnegative(self):
        """Power spectral density values are non-negative."""
        epoch = make_epoch()
        result = extract_band_power(epoch)
        assert np.all(result >= 0)

    def test_band_power_custom_bands(self):
        """Custom bands dict produces correct output size."""
        custom_bands = {
            "low": (1.0, 10.0),
            "mid": (10.0, 20.0),
            "high": (20.0, 40.0),
        }
        epoch = make_epoch(n_channels=8)
        result = extract_band_power(epoch, bands=custom_bands)
        assert result.shape == (8 * 3,)


class TestExtractLogVariance:
    def test_log_variance_shape(self):
        """22 channels -> output shape (22,)."""
        epoch = make_epoch(n_channels=22)
        result = extract_log_variance(epoch)
        assert result.shape == (22,)

    def test_log_variance_constant_signal(self):
        """Constant signal -> log of clamped small value, very negative."""
        signals = np.ones((4, 200)) * 5.0  # constant -> variance = 0
        epoch = NeuralEpoch(
            signals=signals,
            label=0,
            sfreq=250.0,
            channels=[f"Ch{i:03d}" for i in range(4)],
            subject_id=1,
        )
        result = extract_log_variance(epoch)
        # log(1e-12) ~ -27.6
        assert np.all(result < -20)


class TestExtractFeatures:
    def test_extract_features_dispatch(self):
        """band_power and log_variance produce different shaped outputs."""
        epoch = make_epoch(n_channels=22)
        bp = extract_features(epoch, method="band_power")
        lv = extract_features(epoch, method="log_variance")
        assert bp.shape != lv.shape
        assert bp.shape == (110,)
        assert lv.shape == (22,)

    def test_extract_features_invalid_method(self):
        """Unknown method raises ValueError."""
        epoch = make_epoch()
        with pytest.raises(ValueError, match="Unknown feature method"):
            extract_features(epoch, method="nonexistent")


class TestOutputDtype:
    def test_output_dtype_float32(self):
        """All extractors return float32 arrays."""
        epoch = make_epoch()
        bp = extract_band_power(epoch)
        lv = extract_log_variance(epoch)
        feat = extract_features(epoch, method="combined")

        assert bp.dtype == np.float32
        assert lv.dtype == np.float32
        assert feat.dtype == np.float32
