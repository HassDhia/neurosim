"""Signal processing pipeline compositing drift, noise, and co-adaptation.

Provides a convenient way to chain multiple signal models together
for use in NeuroSim environments.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from neurosim.models.drift import BaseDriftModel, ElectrodeDriftModel, FatigueDriftModel, FeatureShiftModel
from neurosim.models.noise import NoiseInjector
from neurosim.models.coadaptation import CoAdaptationModel


class SignalPipeline:
    """Chains drift models, noise injection, and co-adaptation sequentially.

    Models are applied in order: drift models first, then noise, then
    co-adaptation.  This matches the real signal corruption chain where
    neural signals are first degraded by electrode/fatigue drift, then
    corrupted by environmental/physiological noise, and finally modified
    by user learning.

    Args:
        drift_models: Sequence of drift models to apply in order.
        noise_injector: Optional noise injector.
        coadaptation: Optional co-adaptation model.
        seed: Random seed for any internal RNG needs.

    Example::

        pipeline = SignalPipeline.from_preset("moderate", seed=42)
        corrupted = pipeline.apply(clean_signal, timestep=50)
    """

    def __init__(
        self,
        drift_models: Sequence[BaseDriftModel] | None = None,
        noise_injector: NoiseInjector | None = None,
        coadaptation: CoAdaptationModel | None = None,
        seed: int = 42,
    ) -> None:
        self.drift_models = list(drift_models) if drift_models else []
        self.noise_injector = noise_injector
        self.coadaptation = coadaptation
        self._seed = seed

    def apply(
        self,
        signal: np.ndarray,
        timestep: int,
        decoder_expectation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the full signal pipeline.

        Args:
            signal: Input signal array (n_channels, n_timepoints) or (n_features,).
            timestep: Current discrete timestep.
            decoder_expectation: Expected signal for co-adaptation (optional).

        Returns:
            Processed signal with all models applied.
        """
        result = signal.copy()

        # Apply drift models in order
        for drift_model in self.drift_models:
            if result.ndim == 1:
                result = drift_model.apply(result[np.newaxis, :], timestep)[0]
            else:
                result = drift_model.apply(result, timestep)

        # Apply noise
        if self.noise_injector is not None:
            if result.ndim == 1:
                result = self.noise_injector.inject(result[np.newaxis, :], noise_level="medium")[0]
            else:
                result = self.noise_injector.inject(result)

        # Apply co-adaptation
        if self.coadaptation is not None and decoder_expectation is not None:
            if result.ndim == 1:
                result = self.coadaptation.adapt(
                    result[np.newaxis, :], decoder_expectation[np.newaxis, :], timestep
                )[0]
            else:
                result = self.coadaptation.adapt(result, decoder_expectation, timestep)

        return result

    def reset(self) -> None:
        """Reset all models to initial state."""
        for drift_model in self.drift_models:
            drift_model.reset()
        if self.noise_injector is not None:
            self.noise_injector.reset()
        if self.coadaptation is not None:
            self.coadaptation.reset()

    @classmethod
    def from_preset(
        cls,
        level: str = "moderate",
        seed: int = 42,
    ) -> SignalPipeline:
        """Create a pipeline from a difficulty preset.

        Args:
            level: Difficulty level ("none", "mild", "moderate", "full").
            seed: Random seed.

        Returns:
            Configured SignalPipeline.

        Raises:
            ValueError: If level is not a valid preset.
        """
        if level == "none":
            return cls(seed=seed)
        elif level == "mild":
            return cls(
                drift_models=[ElectrodeDriftModel(alpha_z=0.0005, sigma_z=0.005, seed=seed)],
                seed=seed,
            )
        elif level == "moderate":
            return cls(
                drift_models=[
                    ElectrodeDriftModel(alpha_z=0.001, sigma_z=0.01, seed=seed),
                    FatigueDriftModel(beta_f=0.15, tau_f=200.0, seed=seed + 1),
                ],
                noise_injector=NoiseInjector(sfreq=250.0, seed=seed + 2),
                seed=seed,
            )
        elif level == "full":
            return cls(
                drift_models=[
                    ElectrodeDriftModel(alpha_z=0.005, sigma_z=0.02, seed=seed),
                    FatigueDriftModel(beta_f=0.3, tau_f=100.0, seed=seed + 1),
                    FeatureShiftModel(drift_rate=0.001, step_probability=0.01, step_magnitude=0.5, seed=seed + 2),
                ],
                noise_injector=NoiseInjector(sfreq=250.0, seed=seed + 3),
                coadaptation=CoAdaptationModel(adaptation_rate_range=(0.1, 0.5), seed=seed + 4),
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown preset '{level}'. Choose from: none, mild, moderate, full")
