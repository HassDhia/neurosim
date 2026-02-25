"""Benchmark tier definitions for NeuroSim evaluation.

Defines five progressive difficulty tiers (T1-T5) that test BCI
decoders under increasingly challenging non-stationarity conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TierLevel(Enum):
    """Progressive difficulty levels for BCI evaluation."""

    T1_STATIONARY = "T1"
    T2_MILD_DRIFT = "T2"
    T3_FULL_DRIFT = "T3"
    T4_CROSS_SUBJECT = "T4"
    T5_ADVERSARIAL = "T5"


@dataclass(frozen=True)
class BenchmarkTier:
    """A single benchmark tier specification.

    Attributes:
        level: TierLevel enum value.
        name: Human-readable tier name.
        subjects: List of subject IDs to evaluate on.
        non_stationarity: Whether non-stationarity is enabled.
        drift_sources: Which drift models are active.
        noise_level: Noise severity ("low", "medium", "high").
        environments: List of Gymnasium environment IDs to benchmark.
        success_metrics: Minimum metric thresholds for tier completion.
    """

    level: TierLevel
    name: str
    subjects: list[int]
    non_stationarity: bool
    drift_sources: list[str]
    noise_level: str
    environments: list[str]
    success_metrics: dict[str, float]


BENCHMARK_TIERS: dict[TierLevel, BenchmarkTier] = {
    TierLevel.T1_STATIONARY: BenchmarkTier(
        level=TierLevel.T1_STATIONARY,
        name="Stationary",
        subjects=[1],
        non_stationarity=False,
        drift_sources=[],
        noise_level="low",
        environments=["neurosim/DecoderAdapt-v0"],
        success_metrics={"accuracy": 0.85},
    ),
    TierLevel.T2_MILD_DRIFT: BenchmarkTier(
        level=TierLevel.T2_MILD_DRIFT,
        name="Mild Drift",
        subjects=[1],
        non_stationarity=True,
        drift_sources=["electrode"],
        noise_level="medium",
        environments=["neurosim/DecoderAdapt-v0"],
        success_metrics={"accuracy": 0.75, "max_recalibrations": 5},
    ),
    TierLevel.T3_FULL_DRIFT: BenchmarkTier(
        level=TierLevel.T3_FULL_DRIFT,
        name="Full Drift",
        subjects=[1, 2, 3],
        non_stationarity=True,
        drift_sources=["electrode", "fatigue", "feature_shift", "coadaptation"],
        noise_level="medium",
        environments=[
            "neurosim/DecoderAdapt-v0",
            "neurosim/CursorControl-v0",
        ],
        success_metrics={"accuracy": 0.65, "reach_time": 2.5},
    ),
    TierLevel.T4_CROSS_SUBJECT: BenchmarkTier(
        level=TierLevel.T4_CROSS_SUBJECT,
        name="Cross-Subject",
        subjects=list(range(1, 10)),
        non_stationarity=True,
        drift_sources=["electrode", "fatigue", "feature_shift", "coadaptation"],
        noise_level="high",
        environments=[
            "neurosim/DecoderAdapt-v0",
            "neurosim/CursorControl-v0",
            "neurosim/SpellerNav-v0",
        ],
        success_metrics={"accuracy": 0.55, "itr": 20.0},
    ),
    TierLevel.T5_ADVERSARIAL: BenchmarkTier(
        level=TierLevel.T5_ADVERSARIAL,
        name="Adversarial",
        subjects=list(range(1, 10)),
        non_stationarity=True,
        drift_sources=[
            "electrode",
            "fatigue",
            "feature_shift",
            "coadaptation",
            "sudden_shift",
        ],
        noise_level="high",
        environments=[
            "neurosim/DecoderAdapt-v0",
            "neurosim/CursorControl-v0",
            "neurosim/SpellerNav-v0",
        ],
        success_metrics={"accuracy": 0.45},
    ),
}
