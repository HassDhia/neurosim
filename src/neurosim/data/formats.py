"""Internal data formats for NeuroSim.

Standardized epoch representations that decouple MOABB/MNE internals
from the rest of the simulation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuralEpoch:
    """Standardized neural epoch representation.

    A single trial/epoch of neural data with all metadata needed for
    simulation, drift modeling, and RL environment interaction.

    Attributes:
        signals: Neural signals array with shape (n_channels, n_timepoints).
        label: Ground-truth class label (0-indexed integer).
        sfreq: Sampling frequency in Hz.
        channels: Ordered list of channel names (e.g., ["C3", "Cz", "C4"]).
        subject_id: Subject identifier (1-indexed, matching MOABB convention).
        session_id: Session identifier for temporal drift modeling.
        metadata: Paradigm-specific metadata (e.g., event codes, run info).
    """

    signals: np.ndarray
    label: int
    sfreq: float
    channels: list[str]
    subject_id: int
    session_id: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate epoch dimensions on creation."""
        if self.signals.ndim != 2:
            raise ValueError(
                f"signals must be 2D (n_channels, n_timepoints), "
                f"got shape {self.signals.shape}"
            )
        if self.signals.shape[0] != len(self.channels):
            raise ValueError(
                f"Channel count mismatch: signals has {self.signals.shape[0]} "
                f"channels but {len(self.channels)} channel names provided"
            )

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.signals.shape[0]

    @property
    def n_timepoints(self) -> int:
        """Number of time samples in this epoch."""
        return self.signals.shape[1]

    @property
    def duration(self) -> float:
        """Epoch duration in seconds."""
        return self.n_timepoints / self.sfreq

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the signals array.

        Returns:
            Copy of signals with shape (n_channels, n_timepoints).
        """
        return self.signals.copy()

    def with_signals(self, new_signals: np.ndarray) -> NeuralEpoch:
        """Return a new epoch with replaced signals, preserving metadata.

        Args:
            new_signals: Replacement signal array (n_channels, n_timepoints).

        Returns:
            New NeuralEpoch with updated signals.
        """
        return NeuralEpoch(
            signals=new_signals,
            label=self.label,
            sfreq=self.sfreq,
            channels=list(self.channels),
            subject_id=self.subject_id,
            session_id=self.session_id,
            metadata=dict(self.metadata),
        )
