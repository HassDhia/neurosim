"""Noise injection models for realistic BCI signal corruption.

Implements multiple noise types observed in real EEG recordings:
- Pink (1/f) noise: Dominant background noise in neural signals
- EMG artifacts: Muscle tension contamination
- Eye blink artifacts: Frontal electrode contamination
- Line noise: 50/60 Hz power line interference

All noise is generated with controlled RNG for reproducibility.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


# Noise level presets: maps level name to per-type amplitude multipliers
_NOISE_PRESETS: dict[str, dict[str, float]] = {
    "low": {
        "pink_amplitude": 0.5e-6,
        "emg_probability": 0.02,
        "blink_probability": 0.01,
        "line_amplitude": 0.2e-6,
    },
    "medium": {
        "pink_amplitude": 2.0e-6,
        "emg_probability": 0.05,
        "blink_probability": 0.03,
        "line_amplitude": 1.0e-6,
    },
    "high": {
        "pink_amplitude": 5.0e-6,
        "emg_probability": 0.10,
        "blink_probability": 0.06,
        "line_amplitude": 3.0e-6,
    },
}


class NoiseInjector:
    """Injects realistic physiological and environmental noise into EEG signals.

    Supports individual noise types or a combined injection with
    configurable severity presets (low/medium/high).

    Args:
        sfreq: Sampling frequency of the signals in Hz.
        seed: Random seed for reproducibility.

    Example::

        injector = NoiseInjector(sfreq=250.0, seed=0)
        noisy = injector.inject(clean_signal, noise_level="medium")
    """

    def __init__(self, sfreq: float = 250.0, seed: int = 42) -> None:
        self.sfreq = sfreq
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def add_pink_noise(
        self,
        signal: np.ndarray,
        amplitude: float = 2.0e-6,
    ) -> np.ndarray:
        """Add 1/f (pink) noise to signal.

        Pink noise has equal power per octave, which matches the spectral
        profile of background EEG activity.

        Args:
            signal: Input signal (n_channels, n_timepoints).
            amplitude: RMS amplitude of the pink noise in volts.

        Returns:
            Signal with pink noise added.
        """
        n_channels, n_timepoints = signal.shape
        result = signal.copy()

        for ch in range(n_channels):
            # Generate white noise in frequency domain
            white = self._rng.standard_normal(n_timepoints)
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / self.sfreq)

            # Apply 1/f scaling (avoid division by zero at DC)
            freqs[0] = 1.0
            pink_spectrum = spectrum / np.sqrt(freqs)

            # Convert back and scale
            pink = np.fft.irfft(pink_spectrum, n=n_timepoints)
            pink = pink / (np.std(pink) + 1e-12) * amplitude

            result[ch] += pink

        return result

    def add_emg_artifact(
        self,
        signal: np.ndarray,
        p: float = 0.05,
    ) -> np.ndarray:
        """Add EMG (muscle) artifact bursts.

        EMG artifacts are high-frequency broadband bursts that
        contaminate EEG, especially at temporal and frontal electrodes.
        Modeled as windowed broadband noise bursts.

        Args:
            signal: Input signal (n_channels, n_timepoints).
            p: Probability of EMG burst at each timepoint.

        Returns:
            Signal with EMG artifacts added.
        """
        n_channels, n_timepoints = signal.shape
        result = signal.copy()

        # Generate EMG burst mask
        burst_mask = self._rng.random(n_timepoints) < p

        if not np.any(burst_mask):
            return result

        # EMG is high-frequency: generate broadband noise where burst occurs
        for ch in range(n_channels):
            emg_noise = self._rng.standard_normal(n_timepoints)
            # Scale by signal RMS for realistic relative amplitude
            signal_rms = np.sqrt(np.mean(signal[ch] ** 2)) + 1e-12
            emg_noise = emg_noise * signal_rms * 2.0  # 2x signal amplitude
            result[ch] += emg_noise * burst_mask

        return result

    def add_eye_blink(
        self,
        signal: np.ndarray,
        p: float = 0.03,
    ) -> np.ndarray:
        """Add eye blink artifacts.

        Eye blinks produce large-amplitude slow deflections primarily
        in frontal channels. Modeled as Gaussian-enveloped slow waves.

        Args:
            signal: Input signal (n_channels, n_timepoints).
            p: Probability of a blink event per second.

        Returns:
            Signal with blink artifacts added.
        """
        n_channels, n_timepoints = signal.shape
        result = signal.copy()
        duration_sec = n_timepoints / self.sfreq

        # Determine number of blinks (Poisson process)
        expected_blinks = p * duration_sec * self.sfreq
        n_blinks = self._rng.poisson(max(expected_blinks, 0.1))

        if n_blinks == 0:
            return result

        signal_rms = np.sqrt(np.mean(signal ** 2)) + 1e-12

        for _ in range(n_blinks):
            # Random blink center
            center = self._rng.integers(0, n_timepoints)

            # Blink duration ~150-400ms
            blink_width_samples = int(
                self._rng.uniform(0.15, 0.4) * self.sfreq
            )

            # Gaussian envelope
            t = np.arange(n_timepoints) - center
            envelope = np.exp(-0.5 * (t / (blink_width_samples / 4)) ** 2)

            # Blink amplitude: large relative to signal
            blink_amplitude = signal_rms * self._rng.uniform(3.0, 8.0)

            # Frontal channels affected more (simple linear gradient)
            for ch in range(n_channels):
                # First ~25% of channels get full blink, fading toward posterior
                attenuation = max(0.0, 1.0 - (ch / n_channels) * 2.0)
                result[ch] += envelope * blink_amplitude * attenuation

        return result

    def add_line_noise(
        self,
        signal: np.ndarray,
        freq: float = 60.0,
        amplitude: float = 1.0e-6,
    ) -> np.ndarray:
        """Add power line noise (sinusoidal interference).

        Args:
            signal: Input signal (n_channels, n_timepoints).
            freq: Line noise frequency in Hz (50 or 60).
            amplitude: Peak amplitude of the sinusoidal noise.

        Returns:
            Signal with line noise added.
        """
        n_channels, n_timepoints = signal.shape
        result = signal.copy()

        t = np.arange(n_timepoints) / self.sfreq
        # Slightly randomize phase per channel (realistic)
        for ch in range(n_channels):
            phase = self._rng.uniform(0, 2 * np.pi)
            line = amplitude * np.sin(2 * np.pi * freq * t + phase)
            result[ch] += line

        return result

    def inject(
        self,
        signal: np.ndarray,
        noise_level: Literal["low", "medium", "high"] = "medium",
    ) -> np.ndarray:
        """Apply all noise types at a preset severity level.

        Args:
            signal: Input signal (n_channels, n_timepoints).
            noise_level: Preset severity ("low", "medium", "high").

        Returns:
            Signal with all noise types applied.

        Raises:
            ValueError: If noise_level is not a valid preset.
        """
        if noise_level not in _NOISE_PRESETS:
            raise ValueError(
                f"Unknown noise_level '{noise_level}'. "
                f"Choose from: {list(_NOISE_PRESETS.keys())}"
            )

        preset = _NOISE_PRESETS[noise_level]

        result = self.add_pink_noise(signal, amplitude=preset["pink_amplitude"])
        result = self.add_emg_artifact(result, p=preset["emg_probability"])
        result = self.add_eye_blink(result, p=preset["blink_probability"])
        result = self.add_line_noise(result, amplitude=preset["line_amplitude"])

        return result

    def reset(self) -> None:
        """Reset RNG to initial seed for reproducible noise sequences."""
        self._rng = np.random.default_rng(self._seed)
