"""Benchmark runner for evaluating agents across NeuroSim tiers.

Orchestrates running an agent through a benchmark tier's environments
and collecting standardized metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from neurosim.benchmarks.metrics import classification_accuracy
from neurosim.benchmarks.tiers import BENCHMARK_TIERS, BenchmarkTier, TierLevel


class AgentProtocol(Protocol):
    """Minimal interface that benchmark agents must satisfy."""

    def predict(self, obs: dict | np.ndarray) -> tuple: ...


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        tier: The benchmark tier that was evaluated.
        agent_name: Name/identifier of the agent.
        metrics: Aggregated metrics across all episodes.
        episodes: Number of episodes evaluated.
        seed: Random seed used.
        elapsed_seconds: Wall-clock time for the benchmark.
        per_environment: Per-environment breakdown of metrics.
    """

    tier: str
    agent_name: str
    metrics: dict[str, float]
    episodes: int
    seed: int
    elapsed_seconds: float = 0.0
    per_environment: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "tier": self.tier,
            "agent_name": self.agent_name,
            "metrics": self.metrics,
            "episodes": self.episodes,
            "seed": self.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "per_environment": self.per_environment,
        }


class BenchmarkRunner:
    """Runs agents through NeuroSim benchmark tiers.

    Orchestrates environment creation, episode execution, and
    metric aggregation for standardized evaluation.
    """

    def run(
        self,
        tier: BenchmarkTier,
        agent_factory: type | Any,
        n_episodes: int = 50,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a benchmark tier evaluation.

        Args:
            tier: The BenchmarkTier specification.
            agent_factory: Either an AgentProtocol instance, or a callable
                that accepts (env) and returns an agent with predict().
            n_episodes: Number of episodes per environment.
            seed: Random seed for reproducibility.

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        start_time = time.monotonic()
        all_rewards: list[float] = []
        all_accuracies: list[float] = []
        per_env_metrics: dict[str, dict[str, float]] = {}

        for env_id in tier.environments:
            env = gym.make(env_id)

            # Create agent (if factory is callable) or use directly
            if callable(agent_factory) and not hasattr(agent_factory, "predict"):
                agent = agent_factory(env)
            else:
                agent = agent_factory

            env_rewards: list[float] = []
            env_correct = 0
            env_total = 0

            for ep in range(n_episodes):
                ep_seed = seed + ep
                obs, info = env.reset(seed=ep_seed)
                total_reward = 0.0
                done = False

                while not done:
                    action, _ = agent.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

                    # Track accuracy if info provides it
                    if "correct" in info:
                        env_total += 1
                        if info["correct"]:
                            env_correct += 1

                env_rewards.append(total_reward)

            # Per-environment metrics
            env_metrics: dict[str, float] = {
                "mean_reward": float(np.mean(env_rewards)),
                "std_reward": float(np.std(env_rewards)),
            }
            if env_total > 0:
                env_metrics["accuracy"] = env_correct / env_total

            per_env_metrics[env_id] = env_metrics
            all_rewards.extend(env_rewards)
            env.close()

        # Aggregate metrics
        aggregated: dict[str, float] = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "total_episodes": float(len(all_rewards)),
        }

        elapsed = time.monotonic() - start_time

        return BenchmarkResult(
            tier=tier.name,
            agent_name=getattr(agent_factory, "__name__", str(type(agent_factory).__name__)),
            metrics=aggregated,
            episodes=len(all_rewards),
            seed=seed,
            elapsed_seconds=elapsed,
            per_environment=per_env_metrics,
        )


def main() -> None:
    """CLI entry point for running NeuroSim benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run NeuroSim benchmark tiers."
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="T1",
        choices=["T1", "T2", "T3", "T4", "T5"],
        help="Benchmark tier to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes per environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON.",
    )

    args = parser.parse_args()

    # Map tier string to TierLevel
    tier_map = {
        "T1": TierLevel.T1_STATIONARY,
        "T2": TierLevel.T2_MILD_DRIFT,
        "T3": TierLevel.T3_FULL_DRIFT,
        "T4": TierLevel.T4_CROSS_SUBJECT,
        "T5": TierLevel.T5_ADVERSARIAL,
    }
    tier_level = tier_map[args.tier]
    tier = BENCHMARK_TIERS[tier_level]

    print(f"Running benchmark: {tier.name} ({tier.level.value})")
    print(f"  Environments: {tier.environments}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Seed: {args.seed}")

    # Import here to avoid circular import at module level
    from neurosim.agents.random_agent import RandomAgent

    runner = BenchmarkRunner()
    result = runner.run(
        tier=tier,
        agent_factory=RandomAgent,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    print(f"\nResults ({result.elapsed_seconds:.1f}s):")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
