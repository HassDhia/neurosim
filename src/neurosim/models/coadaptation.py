"""Co-adaptation model for BCI user-decoder interaction.

Simulates the bidirectional adaptation process where:
1. The decoder adapts to the user's neural patterns
2. The user adapts their neural strategies to the decoder's expectations

This mutual adaptation is a critical real-world BCI phenomenon that
significantly impacts long-term decoding performance.
"""

from __future__ import annotations

import numpy as np


class CoAdaptationModel:
    """Simulates user co-adaptation to a BCI decoder.

    Models how a user's neural signals gradually shift toward what
    the decoder expects, creating a feedback loop between human
    and machine learning.

    The adaptation modifies signal features by interpolating between
    the user's natural signal and the decoder's expected signal pattern.

    Formula:
        adapted(t) = (1 - alpha(t)) * signal + alpha(t) * decoder_expectation
        alpha(t) = alpha_base * (1 - exp(-t / tau_adapt))

    Where alpha(t) increases over time as the user learns the decoder's
    preference, saturating at alpha_base.

    Args:
        adaptation_rate_range: (min, max) bounds for the adaptation rate.
            The actual rate is sampled uniformly for inter-subject variability.
        tau_adapt: Time constant for adaptation onset.
        seed: Random seed.

    Example::

        coadapt = CoAdaptationModel(adaptation_rate_range=(0.1, 0.5))
        adapted_signal = coadapt.adapt(raw_signal, decoder_expectation, timestep=50)
    """

    def __init__(
        self,
        adaptation_rate_range: tuple[float, float] = (0.1, 0.5),
        tau_adapt: float = 50.0,
        seed: int = 42,
    ) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.adaptation_rate_range = adaptation_rate_range
        self.tau_adapt = tau_adapt

        # Sample subject-specific adaptation rate
        self.alpha_base: float = float(
            self._rng.uniform(*adaptation_rate_range)
        )

        # Track adaptation state
        self._timestep: int = 0
        self._history: list[float] = []

    @property
    def current_alpha(self) -> float:
        """Current adaptation strength (0 = no adaptation, alpha_base = max)."""
        return self.alpha_base * (1.0 - np.exp(-self._timestep / self.tau_adapt))

    def adapt(
        self,
        signal: np.ndarray,
        decoder_expectation: np.ndarray,
        timestep: int | None = None,
    ) -> np.ndarray:
        """Apply co-adaptation to a neural signal.

        Blends the user's actual signal with what the decoder expects,
        simulating how users learn to produce signals the decoder
        can classify more easily.

        Args:
            signal: User's neural signal (n_channels, n_timepoints).
            decoder_expectation: Decoder's expected signal pattern for
                the intended class. Same shape as signal.
            timestep: Optional explicit timestep. If None, uses internal counter.

        Returns:
            Co-adapted signal (same shape as input).

        Raises:
            ValueError: If signal and decoder_expectation shapes mismatch.
        """
        if signal.shape != decoder_expectation.shape:
            raise ValueError(
                f"Shape mismatch: signal {signal.shape} vs "
                f"decoder_expectation {decoder_expectation.shape}"
            )

        if timestep is not None:
            self._timestep = timestep
        else:
            self._timestep += 1

        alpha = self.current_alpha

        # Add noise to adaptation (imperfect human learning)
        noise = self._rng.normal(0, 0.02)
        alpha = float(np.clip(alpha + noise, 0.0, self.alpha_base))

        # Interpolate between natural signal and decoder expectation
        adapted = (1.0 - alpha) * signal + alpha * decoder_expectation

        self._history.append(alpha)
        return adapted

    def get_adaptation_curve(self) -> np.ndarray:
        """Return the history of adaptation strengths.

        Returns:
            Array of alpha values over time.
        """
        return np.array(self._history)

    def reset(self) -> None:
        """Reset adaptation state to initial conditions."""
        self._rng = np.random.default_rng(self._seed)
        self.alpha_base = float(
            self._rng.uniform(*self.adaptation_rate_range)
        )
        self._timestep = 0
        self._history = []
