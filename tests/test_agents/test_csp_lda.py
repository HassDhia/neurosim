"""Tests for CSP+LDA baseline classifier.

Verifies that the classical BCI pipeline (Common Spatial Patterns + Linear
Discriminant Analysis) correctly fits, predicts, and evaluates on synthetic
EEG data without requiring any external datasets or GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurosim.agents.csp_lda import CSPLDABaseline
from neurosim.data.formats import NeuralEpoch


def _make_synthetic_epochs(
    n_per_class: int = 20,
    n_channels: int = 8,
    n_timepoints: int = 100,
    snr: float = 2.0,
    seed: int = 42,
) -> list[NeuralEpoch]:
    """Generate synthetic 2-class EEG epochs with known spatial structure.

    Class 0 has stronger activation in the first half of channels,
    class 1 in the second half. SNR controls discriminability.
    """
    rng = np.random.default_rng(seed)
    epochs: list[NeuralEpoch] = []
    for label in [0, 1]:
        for _ in range(n_per_class):
            signals = rng.standard_normal((n_channels, n_timepoints)).astype(np.float32)
            if label == 0:
                signals[:n_channels // 2, :] += snr
            else:
                signals[n_channels // 2:, :] += snr
            epochs.append(NeuralEpoch(
                signals=signals,
                label=label,
                subject_id=0,
                sfreq=250.0,
                channels=[f"C{i}" for i in range(n_channels)],
            ))
    return epochs


class TestCSPLDACreation:
    """Tests for CSPLDABaseline construction and parameter validation."""

    def test_csp_lda_creation(self) -> None:
        """CSPLDABaseline with even n_components succeeds."""
        clf = CSPLDABaseline(n_components=4)
        assert clf.n_components == 4
        assert clf._is_fitted is False

    def test_csp_lda_odd_components_raises(self) -> None:
        """CSPLDABaseline with odd n_components raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be even"):
            CSPLDABaseline(n_components=5)


class TestCSPLDAFitPredict:
    """Tests for fitting and predicting with CSPLDABaseline."""

    def test_csp_lda_fit_on_synthetic(self) -> None:
        """Fitting on synthetic 2-class data sets _is_fitted to True."""
        epochs = _make_synthetic_epochs(n_per_class=20)
        clf = CSPLDABaseline(n_components=4)
        clf.fit(epochs)
        assert clf._is_fitted is True

    def test_csp_lda_predict_returns_valid_label(self) -> None:
        """After fitting, predict returns a valid class label (0 or 1)."""
        epochs = _make_synthetic_epochs(n_per_class=20)
        clf = CSPLDABaseline(n_components=4)
        clf.fit(epochs)

        test_epoch = epochs[0]
        prediction = clf.predict(test_epoch)
        assert prediction in (0, 1)

    def test_csp_lda_predict_before_fit_raises(self) -> None:
        """Predicting before fitting raises RuntimeError."""
        clf = CSPLDABaseline(n_components=4)
        epoch = _make_synthetic_epochs(n_per_class=1)[0]
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(epoch)

    def test_csp_lda_three_classes_raises(self) -> None:
        """Fitting with 3-class data raises ValueError."""
        rng = np.random.default_rng(99)
        epochs: list[NeuralEpoch] = []
        for label in [0, 1, 2]:
            for _ in range(10):
                signals = rng.standard_normal((8, 100)).astype(np.float32)
                epochs.append(NeuralEpoch(
                    signals=signals,
                    label=label,
                    subject_id=0,
                    sfreq=250.0,
                    channels=[f"C{i}" for i in range(8)],
                ))
        clf = CSPLDABaseline(n_components=4)
        with pytest.raises(ValueError, match="exactly 2 classes"):
            clf.fit(epochs)


class TestCSPLDAEvaluation:
    """Tests for CSPLDABaseline evaluation metrics."""

    def test_csp_lda_evaluate_returns_accuracy(self) -> None:
        """evaluate() returns dict with 'accuracy' and 'confusion_matrix'."""
        epochs = _make_synthetic_epochs(n_per_class=20)
        clf = CSPLDABaseline(n_components=4)
        clf.fit(epochs)
        result = clf.evaluate(epochs)

        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["confusion_matrix"].shape == (2, 2)

    def test_csp_lda_above_chance(self) -> None:
        """CSP+LDA achieves above-chance accuracy on high-SNR synthetic data."""
        train = _make_synthetic_epochs(n_per_class=30, snr=3.0, seed=42)
        test = _make_synthetic_epochs(n_per_class=20, snr=3.0, seed=99)

        clf = CSPLDABaseline(n_components=4)
        clf.fit(train)
        result = clf.evaluate(test)

        assert result["accuracy"] > 0.6, (
            f"Expected above-chance accuracy, got {result['accuracy']:.2f}"
        )
