"""Preprocessing pipeline for NeuroSim neural epochs.

Pure-numpy implementations of standard BCI preprocessing steps.
Each function takes a list of NeuralEpoch and returns a new list
(no mutation of originals). The pipeline function chains them all.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from neurosim.data.formats import NeuralEpoch

logger = logging.getLogger(__name__)


def bandpass_filter(
    epochs: Sequence[NeuralEpoch],
    low: float = 0.5,
    high: float = 40.0,
) -> list[NeuralEpoch]:
    """Apply a bandpass filter to each epoch's signals.

    Uses a simple FFT-based brick-wall filter. For production use,
    consider replacing with scipy.signal.butter + filtfilt.

    Args:
        epochs: Input epochs to filter.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.

    Returns:
        New list of NeuralEpoch with bandpass-filtered signals.

    Raises:
        ValueError: If low >= high or frequencies are negative.
    """
    if low < 0 or high < 0:
        raise ValueError(f"Cutoff frequencies must be non-negative, got low={low}, high={high}")
    if low >= high:
        raise ValueError(f"Low cutoff ({low}) must be less than high cutoff ({high})")

    filtered: list[NeuralEpoch] = []
    for epoch in epochs:
        signals = epoch.signals.copy()
        n_timepoints = signals.shape[1]
        freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / epoch.sfreq)

        # Build frequency mask
        mask = (freqs >= low) & (freqs <= high)

        # Apply per channel
        for ch_idx in range(signals.shape[0]):
            spectrum = np.fft.rfft(signals[ch_idx])
            spectrum[~mask] = 0.0
            signals[ch_idx] = np.fft.irfft(spectrum, n=n_timepoints)

        filtered.append(epoch.with_signals(signals))

    logger.debug("Bandpass filtered %d epochs (%.1f-%.1f Hz)", len(filtered), low, high)
    return filtered


def common_average_reference(
    epochs: Sequence[NeuralEpoch],
) -> list[NeuralEpoch]:
    """Apply Common Average Reference (CAR) to each epoch.

    Subtracts the mean across all channels at each timepoint,
    which is the standard spatial filter for EEG.

    Args:
        epochs: Input epochs.

    Returns:
        New list of NeuralEpoch with CAR applied.
    """
    rereferenced: list[NeuralEpoch] = []
    for epoch in epochs:
        signals = epoch.signals.copy()
        avg = signals.mean(axis=0, keepdims=True)
        signals -= avg
        rereferenced.append(epoch.with_signals(signals))

    logger.debug("Applied CAR to %d epochs", len(rereferenced))
    return rereferenced


def zscore_normalize(
    epochs: Sequence[NeuralEpoch],
    per_channel: bool = True,
) -> list[NeuralEpoch]:
    """Z-score normalize each epoch's signals.

    Transforms signals to zero mean, unit variance. This is applied
    independently per epoch (not across the dataset) to preserve
    temporal dynamics.

    Args:
        epochs: Input epochs.
        per_channel: If True, normalize each channel independently.
            If False, normalize across all channels jointly.

    Returns:
        New list of NeuralEpoch with normalized signals.
    """
    normalized: list[NeuralEpoch] = []
    for epoch in epochs:
        signals = epoch.signals.copy().astype(np.float64)

        if per_channel:
            # Per-channel normalization: axis=1 (time dimension)
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
        else:
            # Global normalization across all channels and timepoints
            mean = signals.mean()
            std = signals.std()

        # Avoid division by zero for flat channels
        std = np.where(std < 1e-12, 1.0, std)
        signals = (signals - mean) / std

        normalized.append(epoch.with_signals(signals))

    logger.debug("Z-score normalized %d epochs (per_channel=%s)", len(normalized), per_channel)
    return normalized


def reject_artifacts(
    epochs: Sequence[NeuralEpoch],
    threshold: float = 100e-6,
) -> list[NeuralEpoch]:
    """Reject epochs containing artifact-level amplitudes.

    Removes any epoch where the peak-to-peak amplitude on any channel
    exceeds the threshold. This is a simple but effective first-pass
    artifact rejection.

    Args:
        epochs: Input epochs.
        threshold: Maximum allowed peak-to-peak amplitude in volts.
            Default is 100 uV (100e-6), standard for EEG.

    Returns:
        New list of NeuralEpoch with artifact epochs removed.
    """
    clean: list[NeuralEpoch] = []
    rejected_count = 0

    for epoch in epochs:
        # Peak-to-peak per channel
        ptp: NDArray[np.floating] = np.ptp(epoch.signals, axis=1)
        if np.all(ptp <= threshold):
            clean.append(epoch)
        else:
            rejected_count += 1

    logger.info(
        "Artifact rejection: kept %d / %d epochs (rejected %d, threshold=%.1f uV)",
        len(clean),
        len(list(epochs)),
        rejected_count,
        threshold * 1e6,
    )
    return clean


def segment_epochs(
    epochs: Sequence[NeuralEpoch],
    window_sec: float = 1.0,
    overlap_sec: float = 0.0,
) -> list[NeuralEpoch]:
    """Segment long epochs into shorter fixed-length windows.

    Useful for creating uniform-length inputs for neural decoders.

    Args:
        epochs: Input epochs (can have variable lengths).
        window_sec: Window duration in seconds.
        overlap_sec: Overlap between consecutive windows in seconds.

    Returns:
        New list of NeuralEpoch, each with duration == window_sec.

    Raises:
        ValueError: If window is longer than any epoch or overlap >= window.
    """
    if overlap_sec >= window_sec:
        raise ValueError(
            f"Overlap ({overlap_sec}s) must be less than window ({window_sec}s)"
        )

    segmented: list[NeuralEpoch] = []
    for epoch in epochs:
        window_samples = int(window_sec * epoch.sfreq)
        step_samples = int((window_sec - overlap_sec) * epoch.sfreq)

        if window_samples > epoch.n_timepoints:
            # Skip epochs shorter than the window
            continue

        start = 0
        seg_idx = 0
        while start + window_samples <= epoch.n_timepoints:
            segment_signals = epoch.signals[:, start : start + window_samples].copy()
            seg_epoch = NeuralEpoch(
                signals=segment_signals,
                label=epoch.label,
                sfreq=epoch.sfreq,
                channels=list(epoch.channels),
                subject_id=epoch.subject_id,
                session_id=epoch.session_id,
                metadata={
                    **epoch.metadata,
                    "segment_index": seg_idx,
                    "segment_start_sample": start,
                },
            )
            segmented.append(seg_epoch)
            start += step_samples
            seg_idx += 1

    logger.debug(
        "Segmented %d epochs into %d windows (%.1fs, overlap=%.1fs)",
        len(list(epochs)),
        len(segmented),
        window_sec,
        overlap_sec,
    )
    return segmented


def preprocess_pipeline(
    epochs: Sequence[NeuralEpoch],
    low: float = 0.5,
    high: float = 40.0,
    apply_car: bool = True,
    normalize: bool = True,
    per_channel_norm: bool = True,
    reject: bool = True,
    artifact_threshold: float = 100e-6,
) -> list[NeuralEpoch]:
    """Run the full preprocessing pipeline on a list of epochs.

    Pipeline order:
        1. Bandpass filter (low-high Hz)
        2. Common Average Reference (optional)
        3. Artifact rejection (optional)
        4. Z-score normalization (optional)

    Note: Artifact rejection runs BEFORE normalization so that
    thresholding operates on physical voltage units.

    Args:
        epochs: Raw input epochs.
        low: Bandpass low cutoff in Hz.
        high: Bandpass high cutoff in Hz.
        apply_car: Whether to apply Common Average Reference.
        normalize: Whether to apply z-score normalization.
        per_channel_norm: If normalizing, whether to do it per-channel.
        reject: Whether to reject artifact epochs.
        artifact_threshold: Peak-to-peak rejection threshold in volts.

    Returns:
        Fully preprocessed list of NeuralEpoch.
    """
    logger.info(
        "Starting preprocessing pipeline on %d epochs "
        "(bandpass=%.1f-%.1f Hz, CAR=%s, reject=%s, normalize=%s)",
        len(list(epochs)),
        low,
        high,
        apply_car,
        reject,
        normalize,
    )

    processed = list(epochs)

    # Step 1: Bandpass filter
    processed = bandpass_filter(processed, low=low, high=high)

    # Step 2: Common Average Reference
    if apply_car:
        processed = common_average_reference(processed)

    # Step 3: Artifact rejection (before normalization)
    if reject:
        processed = reject_artifacts(processed, threshold=artifact_threshold)

    # Step 4: Z-score normalization
    if normalize:
        processed = zscore_normalize(processed, per_channel=per_channel_norm)

    logger.info("Preprocessing complete: %d epochs remaining", len(processed))
    return processed
