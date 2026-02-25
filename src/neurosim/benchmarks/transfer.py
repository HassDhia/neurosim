"""Cross-subject transfer evaluation protocol.

Implements leave-one-subject-out (LOSO) evaluation for measuring how
well BCI decoders generalize across subjects. Uses per-subject
signal pipeline configurations to simulate inter-subject variability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurosim.data.formats import NeuralEpoch


@dataclass
class SubjectConfig:
    """Per-subject signal configuration for inter-subject variability.

    Each subject has unique noise characteristics, electrode drift rates,
    and co-adaptation dynamics.
    """
    subject_id: int
    noise_std: float = 1.0
    drift_rate: float = 0.01
    p300_amplitude: float = 1.5
    adaptation_rate: float = 0.3

    @classmethod
    def generate_cohort(
        cls, n_subjects: int = 9, seed: int = 42
    ) -> list[SubjectConfig]:
        """Generate a cohort of subjects with realistic inter-subject variability.

        Parameters are sampled from distributions that reflect real BCI populations.
        """
        rng = np.random.default_rng(seed)
        configs = []
        for sid in range(1, n_subjects + 1):
            configs.append(cls(
                subject_id=sid,
                noise_std=float(rng.uniform(0.5, 2.0)),
                drift_rate=float(rng.uniform(0.005, 0.05)),
                p300_amplitude=float(rng.uniform(0.8, 2.5)),
                adaptation_rate=float(rng.uniform(0.1, 0.5)),
            ))
        return configs


def generate_subject_epochs(
    config: SubjectConfig,
    n_epochs: int = 50,
    n_channels: int = 8,
    n_timepoints: int = 100,
    n_classes: int = 2,
    seed: int | None = None,
) -> list[NeuralEpoch]:
    """Generate synthetic EEG epochs for a specific subject.

    Each subject has unique signal characteristics based on their config.
    Class-discriminative information is embedded as amplitude differences
    in specific channels, modulated by subject-specific parameters.
    """
    rng = np.random.default_rng(seed if seed is not None else config.subject_id)
    epochs = []

    for i in range(n_epochs):
        label = i % n_classes
        # Base noise
        signals = rng.standard_normal((n_channels, n_timepoints)).astype(np.float32) * config.noise_std

        # Class-discriminative component (different spatial patterns per class)
        if label == 0:
            # Class 0: stronger signal in first half of channels
            signals[:n_channels // 2, :] += config.p300_amplitude * 0.8
        else:
            # Class 1: stronger signal in second half of channels
            signals[n_channels // 2:, :] += config.p300_amplitude * 0.8

        # Subject-specific drift
        drift = np.linspace(0, config.drift_rate * n_timepoints, n_timepoints)
        signals += drift[np.newaxis, :]

        epochs.append(NeuralEpoch(
            signals=signals,
            label=label,
            subject_id=config.subject_id,
            sfreq=250.0,
            channels=[f"C{c}" for c in range(n_channels)],
        ))

    return epochs


@dataclass
class TransferResult:
    """Results from a cross-subject transfer evaluation."""

    method: str  # "loso" or "train_test_split"
    n_subjects: int
    per_subject_accuracy: dict[int, float]  # subject_id -> accuracy
    mean_accuracy: float
    std_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_subjects": self.n_subjects,
            "per_subject_accuracy": self.per_subject_accuracy,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
        }


def leave_one_subject_out(
    subject_configs: list[SubjectConfig],
    epochs_per_subject: int = 50,
    n_channels: int = 8,
    n_timepoints: int = 100,
    n_classes: int = 2,
    classifier_factory: Any = None,
    seed: int = 42,
) -> TransferResult:
    """Run leave-one-subject-out cross-validation.

    For each subject, train on all other subjects' data and evaluate
    on the held-out subject.

    Args:
        subject_configs: List of SubjectConfig for each subject.
        epochs_per_subject: Number of epochs to generate per subject.
        n_channels: EEG channels.
        n_timepoints: Timepoints per epoch.
        n_classes: Number of classes (must be 2 for CSP-LDA).
        classifier_factory: Callable that returns a fresh classifier with fit()/predict().
            If None, uses CSPLDABaseline.
        seed: Random seed.

    Returns:
        TransferResult with per-subject and aggregate accuracy.
    """
    if classifier_factory is None:
        from neurosim.agents.csp_lda import CSPLDABaseline
        classifier_factory = lambda: CSPLDABaseline(n_components=min(6, n_channels))

    # Generate all subject data
    all_data: dict[int, list[NeuralEpoch]] = {}
    for config in subject_configs:
        all_data[config.subject_id] = generate_subject_epochs(
            config,
            n_epochs=epochs_per_subject,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            seed=seed + config.subject_id,
        )

    per_subject_acc: dict[int, float] = {}

    for held_out in subject_configs:
        # Train on all subjects except held-out
        train_epochs: list[NeuralEpoch] = []
        for sid, epochs in all_data.items():
            if sid != held_out.subject_id:
                train_epochs.extend(epochs)

        test_epochs = all_data[held_out.subject_id]

        # Fit classifier
        clf = classifier_factory()
        clf.fit(train_epochs)

        # Evaluate
        result = clf.evaluate(test_epochs)
        per_subject_acc[held_out.subject_id] = result["accuracy"]

    accuracies = list(per_subject_acc.values())

    return TransferResult(
        method="loso",
        n_subjects=len(subject_configs),
        per_subject_accuracy=per_subject_acc,
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies)),
    )
