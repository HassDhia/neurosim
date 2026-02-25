"""Feature extraction bridge for NeuroSim.

Extracts standard BCI features from NeuralEpoch objects, bridging raw MOABB
signals to the feature vectors that RL environments consume.

All functions return numpy float32 arrays suitable for direct use as
observation vectors in Gymnasium environments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import welch

from neurosim.data.formats import NeuralEpoch

# Standard EEG frequency bands (Hz)
DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def extract_band_power(
    epoch: NeuralEpoch,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> np.ndarray:
    """Compute power spectral density in standard EEG frequency bands.

    Uses Welch's method to estimate PSD, then integrates power within each
    frequency band for each channel. Returns a flat feature vector ordered
    as [ch0_band0, ch0_band1, ..., ch1_band0, ...].

    Args:
        epoch: A NeuralEpoch with shape (n_channels, n_timepoints).
        bands: Dict mapping band name to (low_hz, high_hz). Defaults to
            standard EEG bands (delta, theta, alpha, beta, gamma).

    Returns:
        1D float32 array of shape (n_channels * n_bands,) containing
        average power per band per channel.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    band_list = list(bands.values())
    n_bands = len(band_list)

    # Welch PSD: nperseg capped at signal length for short epochs
    nperseg = min(epoch.n_timepoints, int(epoch.sfreq * 2))
    if nperseg < 4:
        nperseg = epoch.n_timepoints

    freqs, psd = welch(
        epoch.signals,
        fs=epoch.sfreq,
        nperseg=nperseg,
        axis=-1,
    )
    # psd shape: (n_channels, n_freqs)

    features = np.zeros((epoch.n_channels, n_bands), dtype=np.float64)

    for i, (low, high) in enumerate(band_list):
        mask = (freqs >= low) & (freqs <= high)
        if mask.any():
            # Average power in band (mean, not sum, for scale invariance)
            features[:, i] = np.mean(psd[:, mask], axis=-1)
        else:
            features[:, i] = 0.0

    return features.ravel().astype(np.float32)


def extract_log_variance(epoch: NeuralEpoch) -> np.ndarray:
    """Compute log-variance of each channel's signal.

    A simple but effective feature inspired by CSP (Common Spatial Patterns).
    Log-variance captures energy differences between classes and is one of
    the most robust single-feature approaches for motor imagery BCI.

    Args:
        epoch: A NeuralEpoch with shape (n_channels, n_timepoints).

    Returns:
        1D float32 array of shape (n_channels,) with log-variance per channel.
    """
    variance = np.var(epoch.signals, axis=-1)
    # Clamp to avoid log(0)
    variance = np.maximum(variance, 1e-12)
    return np.log(variance).astype(np.float32)


def extract_features(
    epoch: NeuralEpoch,
    method: str = "band_power",
) -> np.ndarray:
    """Extract a 1D feature vector from a NeuralEpoch.

    Convenience dispatcher that selects the appropriate feature extraction
    method by name.

    Supported methods:
        - "band_power": Power in standard EEG frequency bands (default).
        - "log_variance": Log-variance per channel.
        - "combined": Concatenation of band_power and log_variance.

    Args:
        epoch: A NeuralEpoch with shape (n_channels, n_timepoints).
        method: Feature extraction method name.

    Returns:
        1D float32 feature vector. Length depends on method and n_channels.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "band_power":
        return extract_band_power(epoch)
    elif method == "log_variance":
        return extract_log_variance(epoch)
    elif method == "combined":
        bp = extract_band_power(epoch)
        lv = extract_log_variance(epoch)
        return np.concatenate([bp, lv]).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown feature method '{method}'. "
            f"Supported: 'band_power', 'log_variance', 'combined'."
        )
