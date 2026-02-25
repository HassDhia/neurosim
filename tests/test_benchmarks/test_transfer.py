"""Tests for cross-subject transfer evaluation protocol.

Verifies that the LOSO evaluation pipeline generates correct per-subject
data, runs the full evaluation loop, and produces well-structured results.
All tests use synthetic data only -- no network calls, no GPU, no MOABB.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurosim.benchmarks.transfer import (
    SubjectConfig,
    TransferResult,
    generate_subject_epochs,
    leave_one_subject_out,
)


class TestSubjectConfig:
    """Tests for SubjectConfig cohort generation."""

    def test_subject_config_generate_cohort(self) -> None:
        """generate_cohort returns requested number of configs with unique IDs."""
        configs = SubjectConfig.generate_cohort(n_subjects=3, seed=42)

        assert len(configs) == 3
        subject_ids = [c.subject_id for c in configs]
        assert len(set(subject_ids)) == 3
        # IDs are 1-indexed
        assert subject_ids == [1, 2, 3]

    def test_subject_config_parameter_ranges(self) -> None:
        """Generated configs have parameters within expected ranges."""
        configs = SubjectConfig.generate_cohort(n_subjects=9, seed=42)
        for c in configs:
            assert 0.5 <= c.noise_std <= 2.0
            assert 0.005 <= c.drift_rate <= 0.05
            assert 0.8 <= c.p300_amplitude <= 2.5
            assert 0.1 <= c.adaptation_rate <= 0.5


class TestGenerateSubjectEpochs:
    """Tests for per-subject epoch generation."""

    def test_generate_subject_epochs_shape(self) -> None:
        """Generates correct number of epochs with expected signal shape."""
        config = SubjectConfig(subject_id=1, noise_std=1.0)
        epochs = generate_subject_epochs(
            config, n_epochs=20, n_channels=8, n_timepoints=100
        )

        assert len(epochs) == 20
        for ep in epochs:
            assert ep.signals.shape == (8, 100)
            assert ep.subject_id == 1
            assert len(ep.channels) == 8

    def test_generate_subject_epochs_class_balance(self) -> None:
        """Equal number of class 0 and class 1 epochs."""
        config = SubjectConfig(subject_id=1)
        epochs = generate_subject_epochs(config, n_epochs=40, n_classes=2)

        labels = [e.label for e in epochs]
        assert labels.count(0) == 20
        assert labels.count(1) == 20


class TestLeaveOneSubjectOut:
    """Tests for LOSO cross-validation."""

    def test_leave_one_subject_out_runs(self) -> None:
        """LOSO with 3 subjects completes without error."""
        configs = SubjectConfig.generate_cohort(n_subjects=3, seed=42)
        result = leave_one_subject_out(
            configs,
            epochs_per_subject=30,
            n_channels=8,
            n_timepoints=100,
            seed=42,
        )
        assert isinstance(result, TransferResult)
        assert result.method == "loso"

    def test_leave_one_subject_out_returns_per_subject(self) -> None:
        """Result contains accuracy for each subject_id."""
        configs = SubjectConfig.generate_cohort(n_subjects=3, seed=42)
        result = leave_one_subject_out(
            configs,
            epochs_per_subject=30,
            n_channels=8,
            n_timepoints=100,
            seed=42,
        )
        assert result.n_subjects == 3
        for config in configs:
            assert config.subject_id in result.per_subject_accuracy
            acc = result.per_subject_accuracy[config.subject_id]
            assert 0.0 <= acc <= 1.0


class TestTransferResult:
    """Tests for TransferResult serialization."""

    def test_transfer_result_to_dict(self) -> None:
        """to_dict returns a JSON-serializable dictionary."""
        result = TransferResult(
            method="loso",
            n_subjects=3,
            per_subject_accuracy={1: 0.7, 2: 0.8, 3: 0.6},
            mean_accuracy=0.7,
            std_accuracy=0.08,
        )
        d = result.to_dict()

        assert d["method"] == "loso"
        assert d["n_subjects"] == 3
        assert d["per_subject_accuracy"] == {1: 0.7, 2: 0.8, 3: 0.6}
        assert isinstance(d["mean_accuracy"], float)
        assert isinstance(d["std_accuracy"], float)
