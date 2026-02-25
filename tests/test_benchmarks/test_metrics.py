"""Tests for benchmark metric functions."""
import numpy as np
import pytest

from neurosim.benchmarks.metrics import (
    adaptation_efficiency,
    classification_accuracy,
    information_transfer_rate,
    path_efficiency,
    reach_time,
)


class TestClassificationAccuracy:
    def test_perfect_accuracy(self):
        preds = [0, 1, 2, 3, 0, 1]
        labels = [0, 1, 2, 3, 0, 1]
        assert classification_accuracy(preds, labels) == 1.0

    def test_partial_accuracy(self):
        preds = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 0, 3])
        result = classification_accuracy(preds, labels)
        assert result == pytest.approx(0.75)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            classification_accuracy([], [])

    def test_shape_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            classification_accuracy([0, 1, 2], [0, 1])


class TestInformationTransferRate:
    def test_perfect_accuracy_positive_itr(self):
        itr = information_transfer_rate(n_classes=4, accuracy=1.0, trial_time=2.0)
        # Perfect accuracy on 4 classes: B = log2(4) = 2 bits, ITR = 2 * 30 = 60
        assert itr > 0.0
        assert itr == pytest.approx(60.0, rel=0.01)

    def test_invalid_n_classes_raises(self):
        with pytest.raises(ValueError, match="n_classes"):
            information_transfer_rate(n_classes=1, accuracy=0.8, trial_time=1.0)

    def test_invalid_accuracy_zero_raises(self):
        with pytest.raises(ValueError, match="accuracy"):
            information_transfer_rate(n_classes=4, accuracy=0.0, trial_time=1.0)

    def test_invalid_accuracy_negative_raises(self):
        with pytest.raises(ValueError, match="accuracy"):
            information_transfer_rate(n_classes=4, accuracy=-0.5, trial_time=1.0)

    def test_invalid_trial_time_raises(self):
        with pytest.raises(ValueError, match="trial_time"):
            information_transfer_rate(n_classes=4, accuracy=0.8, trial_time=0.0)

    def test_positive_result_for_above_chance(self):
        itr = information_transfer_rate(n_classes=2, accuracy=0.9, trial_time=1.0)
        assert itr > 0.0


class TestReachTime:
    def test_immediate_reach(self):
        positions = np.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
        target = np.array([0.5, 0.5])
        result = reach_time(positions, target, threshold=0.05)
        assert result == 0.0

    def test_never_reached(self):
        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        target = np.array([0.9, 0.9])
        result = reach_time(positions, target, threshold=0.05)
        assert result == float("inf")


class TestPathEfficiency:
    def test_straight_line_perfect_efficiency(self):
        path = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        efficiency = path_efficiency(path, optimal_distance=1.0)
        assert efficiency == pytest.approx(1.0)

    def test_invalid_optimal_distance_raises(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="optimal_distance"):
            path_efficiency(path, optimal_distance=0.0)

    def test_single_point_returns_zero(self):
        path = np.array([[0.5, 0.5]])
        assert path_efficiency(path, optimal_distance=1.0) == 0.0


class TestAdaptationEfficiency:
    def test_full_recovery(self):
        result = adaptation_efficiency(pre_drift_accuracy=0.9, post_drift_accuracy=0.9)
        assert result == pytest.approx(1.0)

    def test_partial_recovery(self):
        result = adaptation_efficiency(pre_drift_accuracy=0.8, post_drift_accuracy=0.4)
        assert result == pytest.approx(0.5)

    def test_zero_pre_drift_raises(self):
        with pytest.raises(ValueError, match="pre_drift_accuracy"):
            adaptation_efficiency(pre_drift_accuracy=0.0, post_drift_accuracy=0.5)
