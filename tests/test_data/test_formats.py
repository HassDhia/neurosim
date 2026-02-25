"""Tests for NeuralEpoch data format."""
import numpy as np
import pytest

from neurosim.data.formats import NeuralEpoch


class TestNeuralEpoch:
    def test_creation(self):
        epoch = NeuralEpoch(
            signals=np.random.randn(22, 256),
            label=0,
            sfreq=256.0,
            channels=[f"C{i}" for i in range(22)],
            subject_id=1,
        )
        assert epoch.n_channels == 22
        assert epoch.n_timepoints == 256
        assert epoch.duration == 1.0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="signals must be 2D"):
            NeuralEpoch(
                signals=np.random.randn(22),
                label=0,
                sfreq=256.0,
                channels=[f"C{i}" for i in range(22)],
                subject_id=1,
            )

    def test_channel_mismatch_raises(self):
        with pytest.raises(ValueError, match="Channel count mismatch"):
            NeuralEpoch(
                signals=np.random.randn(22, 256),
                label=0,
                sfreq=256.0,
                channels=[f"C{i}" for i in range(10)],
                subject_id=1,
            )

    def test_to_numpy_returns_copy(self):
        signals = np.random.randn(8, 128)
        epoch = NeuralEpoch(
            signals=signals,
            label=1,
            sfreq=128.0,
            channels=[f"C{i}" for i in range(8)],
            subject_id=1,
        )
        copy = epoch.to_numpy()
        assert np.array_equal(copy, signals)
        copy[0, 0] = 999.0
        assert epoch.signals[0, 0] != 999.0

    def test_with_signals_preserves_metadata(self):
        epoch = NeuralEpoch(
            signals=np.random.randn(8, 128),
            label=1,
            sfreq=128.0,
            channels=[f"C{i}" for i in range(8)],
            subject_id=3,
            session_id=2,
        )
        new_signals = np.zeros((8, 128))
        new_epoch = epoch.with_signals(new_signals)
        assert new_epoch.label == 1
        assert new_epoch.subject_id == 3
        assert new_epoch.session_id == 2
        assert np.allclose(new_epoch.signals, 0.0)
