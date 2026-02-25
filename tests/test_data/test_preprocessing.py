"""Tests for preprocessing pipeline."""
import numpy as np
import pytest

from neurosim.data.formats import NeuralEpoch
from neurosim.data.preprocessing import (
    bandpass_filter,
    common_average_reference,
    preprocess_pipeline,
    reject_artifacts,
    segment_epochs,
    zscore_normalize,
)


def _make_epoch(
    signals: np.ndarray,
    sfreq: float = 250.0,
    label: int = 0,
) -> NeuralEpoch:
    n_ch = signals.shape[0]
    return NeuralEpoch(
        signals=signals,
        label=label,
        sfreq=sfreq,
        channels=[f"Ch{i:03d}" for i in range(n_ch)],
        subject_id=1,
        session_id=0,
    )


class TestBandpassFilter:
    def test_bandpass_removes_high_freq(self):
        """5 Hz component preserved, 60 Hz component attenuated by 0.5-40 Hz filter."""
        sfreq = 250.0
        duration = 2.0
        t = np.arange(0, duration, 1 / sfreq)
        signal_5hz = np.sin(2 * np.pi * 5 * t)
        signal_60hz = np.sin(2 * np.pi * 60 * t)
        combined = signal_5hz + signal_60hz
        signals = combined.reshape(1, -1)
        epoch = _make_epoch(signals, sfreq=sfreq)

        filtered = bandpass_filter([epoch], low=0.5, high=40.0)
        result = filtered[0].signals[0]

        # Compute power at 60 Hz via FFT
        n = len(result)
        freqs = np.fft.rfftfreq(n, d=1 / sfreq)
        spectrum = np.abs(np.fft.rfft(result))

        idx_60 = np.argmin(np.abs(freqs - 60.0))
        idx_5 = np.argmin(np.abs(freqs - 5.0))

        # 60 Hz should be nearly zero; 5 Hz should be preserved
        assert spectrum[idx_60] < spectrum[idx_5] * 0.01

    def test_bandpass_invalid_range(self):
        """low >= high raises ValueError."""
        signals = np.random.randn(2, 100)
        epoch = _make_epoch(signals)
        with pytest.raises(ValueError):
            bandpass_filter([epoch], low=40.0, high=0.5)
        with pytest.raises(ValueError):
            bandpass_filter([epoch], low=10.0, high=10.0)


class TestCommonAverageReference:
    def test_car_subtracts_mean(self):
        """2 channels [1,1] and [3,3] -> CAR gives [-1,-1] and [1,1]."""
        signals = np.array([[1.0, 1.0], [3.0, 3.0]])
        epoch = _make_epoch(signals)
        result = common_average_reference([epoch])
        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(result[0].signals, expected, atol=1e-12)


class TestZscoreNormalize:
    def test_zscore_zero_mean_unit_var(self):
        """After z-score per channel, mean ~ 0 and std ~ 1."""
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((4, 500)) * 50 + 100
        epoch = _make_epoch(signals)
        result = zscore_normalize([epoch], per_channel=True)
        for ch in range(4):
            ch_data = result[0].signals[ch]
            assert abs(np.mean(ch_data)) < 1e-10
            assert abs(np.std(ch_data) - 1.0) < 1e-10


class TestRejectArtifacts:
    def test_reject_artifacts_removes_large(self):
        """Epoch with ptp > threshold is removed."""
        clean_signals = np.ones((2, 100)) * 0.5e-6
        noisy_signals = np.zeros((2, 100))
        noisy_signals[0, 0] = 200e-6  # ptp = 200 uV on ch0

        clean_epoch = _make_epoch(clean_signals)
        noisy_epoch = _make_epoch(noisy_signals, label=1)

        result = reject_artifacts([clean_epoch, noisy_epoch], threshold=100e-6)
        assert len(result) == 1
        assert result[0].label == 0

    def test_reject_artifacts_keeps_clean(self):
        """Clean epoch is kept."""
        signals = np.ones((2, 100)) * 1e-6
        epoch = _make_epoch(signals)
        result = reject_artifacts([epoch], threshold=100e-6)
        assert len(result) == 1


class TestSegmentEpochs:
    def test_segment_epochs_count(self):
        """2s epoch at 100 Hz, 1s window, no overlap -> 2 segments."""
        sfreq = 100.0
        signals = np.random.randn(2, 200)  # 2 seconds
        epoch = _make_epoch(signals, sfreq=sfreq)
        result = segment_epochs([epoch], window_sec=1.0, overlap_sec=0.0)
        assert len(result) == 2
        assert result[0].n_timepoints == 100
        assert result[1].n_timepoints == 100

    def test_segment_epochs_overlap(self):
        """2s epoch at 100 Hz, 1s window, 0.5s overlap -> 3 segments."""
        sfreq = 100.0
        signals = np.random.randn(2, 200)  # 2 seconds
        epoch = _make_epoch(signals, sfreq=sfreq)
        result = segment_epochs([epoch], window_sec=1.0, overlap_sec=0.5)
        assert len(result) == 3


class TestPreprocessPipeline:
    def test_pipeline_preserves_epochs(self):
        """Pipeline with no rejection keeps epoch count (small amplitudes)."""
        rng = np.random.default_rng(42)
        signals = rng.standard_normal((4, 500)) * 1e-7  # tiny amplitudes
        epoch = _make_epoch(signals, sfreq=250.0)
        result = preprocess_pipeline(
            [epoch],
            low=0.5,
            high=40.0,
            apply_car=True,
            normalize=True,
            reject=True,
            artifact_threshold=100e-6,
        )
        assert len(result) == 1

    def test_pipeline_ordering(self):
        """Rejection runs before normalization (clean data passes through)."""
        rng = np.random.default_rng(99)
        # Very small signal -- well under 100uV threshold
        signals = rng.standard_normal((4, 500)) * 1e-8
        epoch = _make_epoch(signals, sfreq=250.0)

        result = preprocess_pipeline(
            [epoch],
            low=0.5,
            high=40.0,
            reject=True,
            normalize=True,
            artifact_threshold=100e-6,
        )
        # Epoch should survive rejection and then be normalized
        assert len(result) == 1
        # After normalization, mean should be near zero per channel
        for ch in range(4):
            assert abs(np.mean(result[0].signals[ch])) < 1e-6
