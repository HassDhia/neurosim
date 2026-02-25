"""Tests for benchmark tier definitions."""
import pytest

from neurosim.benchmarks.tiers import BENCHMARK_TIERS, BenchmarkTier, TierLevel


class TestBenchmarkTiers:
    def test_five_tiers_defined(self):
        assert len(BENCHMARK_TIERS) == 5

    def test_tier_names(self):
        assert BENCHMARK_TIERS[TierLevel.T1_STATIONARY].name == "Stationary"
        assert BENCHMARK_TIERS[TierLevel.T5_ADVERSARIAL].name == "Adversarial"

    def test_all_tiers_have_environments(self):
        for tier in BENCHMARK_TIERS.values():
            assert len(tier.environments) > 0

    def test_tier_levels_match_keys(self):
        for level, tier in BENCHMARK_TIERS.items():
            assert tier.level == level

    def test_success_metrics_are_non_empty(self):
        for tier in BENCHMARK_TIERS.values():
            assert len(tier.success_metrics) > 0

    def test_progressive_difficulty(self):
        """Higher tiers should have more drift sources."""
        t1 = BENCHMARK_TIERS[TierLevel.T1_STATIONARY]
        t5 = BENCHMARK_TIERS[TierLevel.T5_ADVERSARIAL]
        assert len(t5.drift_sources) > len(t1.drift_sources)

    def test_cross_subject_has_multiple_subjects(self):
        t4 = BENCHMARK_TIERS[TierLevel.T4_CROSS_SUBJECT]
        assert len(t4.subjects) > 1

    def test_frozen_dataclass(self):
        """BenchmarkTier should be immutable."""
        tier = BENCHMARK_TIERS[TierLevel.T1_STATIONARY]
        with pytest.raises(AttributeError):
            tier.name = "Modified"  # type: ignore[misc]
