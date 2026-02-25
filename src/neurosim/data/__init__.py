"""Data pipeline for loading and preprocessing BCI datasets via MOABB."""

from neurosim.data.formats import NeuralEpoch
from neurosim.data.loader import MOABBLoader
from neurosim.data.preprocessing import (
    bandpass_filter,
    common_average_reference,
    zscore_normalize,
    reject_artifacts,
    segment_epochs,
    preprocess_pipeline,
)
from neurosim.data.features import extract_band_power, extract_log_variance, extract_features

__all__ = [
    "MOABBLoader",
    "NeuralEpoch",
    "bandpass_filter",
    "common_average_reference",
    "zscore_normalize",
    "reject_artifacts",
    "segment_epochs",
    "preprocess_pipeline",
    "extract_band_power",
    "extract_log_variance",
    "extract_features",
]
