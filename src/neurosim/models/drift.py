"""Signal drift models for non-stationarity simulation.

Implements the three drift mechanisms from the NeuroSim architecture spec:

1. ElectrodeDriftModel  - Impedance changes: Z(t) = Z0 * (1 + alpha_z * t + eps_z(t))
2. FatigueDriftModel    - Neural fatigue:    fatigue(t) = 1 - beta_f * (1 - exp(-t / tau_f))
3. FeatureShiftModel    - Distribution drift: mu_shifted(t) = mu_0 + drift_rate * t + step_shifts

Each model independently transforms neural signals to simulate real-world
non-stationarity that BCI decoders must handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDriftModel(ABC):
    """Abstract base for all drift models.

    Provides common RNG management and the apply/reset interface.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize drift model with reproducible RNG.

        Args:
            seed: Random seed for reproducibility.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def apply(self, signal: np.ndarray, timestep: int) -> np.ndarray:
        """Apply drift to a signal at the given timestep.

        Args:
            signal: Input signal array (n_channels, n_timepoints).
            timestep: Current discrete timestep (e.g., epoch index).

        Returns:
            Modified signal with drift applied. Same shape as input.
        """

    def reset(self) -> None:
        """Reset drift model to initial state (re-seed RNG)."""
        self._rng = np.random.default_rng(self._seed)


class ElectrodeDriftModel(BaseDriftModel):
    """Simulates electrode impedance drift over time.

    Models gradual impedance changes due to gel drying, electrode
    displacement, and skin conductance shifts. The impedance increase
    attenuates signal amplitude multiplicatively.

    Formula:
        Z(t) = Z0 * (1 + alpha_z * t + eps_z(t))
        signal_out = signal_in / Z(t) * Z0

    Where eps_z(t) ~ N(0, sigma_z^2) is random impedance noise.

    Args:
        alpha_z: Linear impedance drift rate per timestep.
        sigma_z: Standard deviation of impedance noise.
        seed: Random seed.
    """

    def __init__(
        self,
        alpha_z: float = 0.001,
        sigma_z: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.alpha_z = alpha_z
        self.sigma_z = sigma_z

    def apply(self, signal: np.ndarray, timestep: int) -> np.ndarray:
        """Apply impedance drift attenuation to signal.

        Args:
            signal: Neural signal (n_channels, n_timepoints).
            timestep: Current epoch/trial index.

        Returns:
            Attenuated signal reflecting impedance increase.
        """
        # Per-channel impedance noise (different electrodes drift independently)
        n_channels = signal.shape[0]
        eps_z = self._rng.normal(0, self.sigma_z, size=n_channels)

        # Z(t) / Z0 = 1 + alpha_z * t + eps_z(t)
        impedance_ratio = 1.0 + self.alpha_z * timestep + eps_z

        # Attenuation: higher impedance -> lower signal amplitude
        # signal_out = signal_in / impedance_ratio
        attenuation = 1.0 / np.maximum(impedance_ratio, 0.01)  # prevent div-by-zero
        return signal * attenuation[:, np.newaxis]


class FatigueDriftModel(BaseDriftModel):
    """Simulates neural fatigue effects on signal amplitude.

    Models the decrease in neural signal strength as subjects tire
    during extended BCI sessions. Follows an exponential saturation
    curve with optional noise.

    Formula:
        fatigue(t) = 1 - beta_f * (1 - exp(-t / tau_f))
        signal_out = signal_in * fatigue(t)

    As t -> inf, fatigue(t) -> 1 - beta_f (asymptotic attenuation).

    Args:
        beta_f: Maximum fatigue effect (0 = no fatigue, 1 = full attenuation).
        tau_f: Time constant controlling fatigue onset speed.
        noise_std: Random noise on the fatigue factor.
        seed: Random seed.
    """

    def __init__(
        self,
        beta_f: float = 0.3,
        tau_f: float = 100.0,
        noise_std: float = 0.02,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.beta_f = beta_f
        self.tau_f = tau_f
        self.noise_std = noise_std

    def apply(self, signal: np.ndarray, timestep: int) -> np.ndarray:
        """Apply fatigue attenuation to signal.

        Args:
            signal: Neural signal (n_channels, n_timepoints).
            timestep: Current epoch/trial index.

        Returns:
            Fatigue-attenuated signal.
        """
        # Core fatigue curve
        fatigue_factor = 1.0 - self.beta_f * (1.0 - np.exp(-timestep / self.tau_f))

        # Add per-application noise
        noise = self._rng.normal(0, self.noise_std)
        fatigue_factor = np.clip(fatigue_factor + noise, 0.01, 1.0)

        return signal * fatigue_factor


class FeatureShiftModel(BaseDriftModel):
    """Simulates gradual and sudden shifts in feature distributions.

    Models the non-stationarity where the statistical properties of
    neural features (e.g., band power, CSP features) drift over time.
    Includes both continuous drift and discrete step shifts.

    Formula:
        mu_shifted(t) = mu_0 + drift_rate * t + sum(step_shifts)

    Step shifts represent sudden distribution changes (e.g., subject
    adjusts seating position, mental strategy change).

    Args:
        drift_rate: Continuous drift rate per timestep (additive to signal).
        step_probability: Probability of a step shift at each timestep.
        step_magnitude: Standard deviation of step shift size.
        seed: Random seed.
    """

    def __init__(
        self,
        drift_rate: float = 0.0001,
        step_probability: float = 0.01,
        step_magnitude: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.drift_rate = drift_rate
        self.step_probability = step_probability
        self.step_magnitude = step_magnitude
        self._accumulated_step_shift: float = 0.0

    def apply(self, signal: np.ndarray, timestep: int) -> np.ndarray:
        """Apply feature distribution shift to signal.

        Args:
            signal: Neural signal (n_channels, n_timepoints).
            timestep: Current epoch/trial index.

        Returns:
            Signal with shifted baseline.
        """
        # Continuous linear drift
        continuous_shift = self.drift_rate * timestep

        # Stochastic step shift
        if self._rng.random() < self.step_probability:
            step = self._rng.normal(0, self.step_magnitude)
            self._accumulated_step_shift += step

        # Total shift is additive to the signal
        total_shift = continuous_shift + self._accumulated_step_shift

        return signal + total_shift

    def reset(self) -> None:
        """Reset drift model including accumulated step shifts."""
        super().reset()
        self._accumulated_step_shift = 0.0
