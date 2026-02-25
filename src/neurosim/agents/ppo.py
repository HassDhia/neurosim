"""PPO agent wrapper around Stable-Baselines3.

Provides a high-level interface for training PPO on NeuroSim environments
with pre-configured hyperparameters from the architecture spec.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class PPOConfig:
    """PPO hyperparameter configuration."""

    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    total_timesteps: int = 500_000
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MultiInputPolicy"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# Architecture-spec configs per environment
DEFAULT_CONFIGS: dict[str, PPOConfig] = {
    "neurosim/DecoderAdapt-v0": PPOConfig(
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        total_timesteps=500_000,
        n_epochs=10,
        gamma=0.99,
        policy="MultiInputPolicy",
    ),
    "neurosim/CursorControl-v0": PPOConfig(
        learning_rate=1e-4,
        batch_size=128,
        n_steps=4096,
        total_timesteps=1_000_000,
        n_epochs=10,
        gamma=0.995,
        policy="MultiInputPolicy",
    ),
    "neurosim/SpellerNav-v0": PPOConfig(
        learning_rate=2e-4,
        batch_size=256,
        n_steps=2048,
        total_timesteps=2_000_000,
        n_epochs=10,
        gamma=0.99,
        policy="MultiInputPolicy",
    ),
}


def _import_sb3():
    """Lazy import of stable_baselines3 with helpful error."""
    try:
        from stable_baselines3 import PPO as SB3_PPO

        return SB3_PPO
    except ImportError:
        print(
            "ERROR: stable-baselines3 is required for PPOAgent.\n"
            "Install it with:\n"
            "  pip install 'neurosim[rl]'\n"
            "  # or: pip install stable-baselines3",
            file=sys.stderr,
        )
        raise SystemExit(1)


class PPOAgent:
    """PPO agent using Stable-Baselines3 backend.

    Wraps SB3's PPO with NeuroSim-specific defaults and evaluation.

    Args:
        env: A Gymnasium environment (or env ID string).
        config: PPO hyperparameter configuration. If None, uses
                DEFAULT_CONFIGS for the environment or generic defaults.
    """

    def __init__(
        self,
        env: gym.Env | str,
        config: PPOConfig | None = None,
    ) -> None:
        if isinstance(env, str):
            self.env_id = env
            self.env = gym.make(env)
        else:
            self.env_id = getattr(env, "spec", None)
            self.env_id = self.env_id.id if self.env_id else "unknown"
            self.env = env

        if config is None:
            config = DEFAULT_CONFIGS.get(self.env_id, PPOConfig())
        self.config = config

        SB3_PPO = _import_sb3()
        self.model = SB3_PPO(
            policy=config.policy,
            env=self.env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=1,
            **config.extra_kwargs,
        )

    def train(self) -> None:
        """Train the PPO agent for config.total_timesteps steps."""
        self.model.learn(total_timesteps=self.config.total_timesteps)

    def predict(self, obs: dict | np.ndarray) -> tuple:
        """Get action from trained policy.

        Args:
            obs: Current observation.

        Returns:
            Tuple of (action, info dict).
        """
        action, _states = self.model.predict(obs, deterministic=True)
        return action, {}

    def evaluate(self, n_episodes: int = 100) -> dict[str, float]:
        """Evaluate the trained agent over n_episodes.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Dict with mean_reward, std_reward, min_reward, max_reward.
        """
        rewards: list[float] = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }

    def save(self, path: str | Path) -> None:
        """Save trained model to disk.

        Args:
            path: File path (without extension; SB3 adds .zip).
        """
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to saved model.
        """
        SB3_PPO = _import_sb3()
        self.model = SB3_PPO.load(str(path), env=self.env)


def main() -> None:
    """CLI entry point for neurosim-train script."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a NeuroSim environment."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="neurosim/DecoderAdapt-v0",
        help="Gymnasium environment ID.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides default config).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes after training.",
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIGS.get(args.env, PPOConfig())
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps

    print(f"Training PPO on {args.env} for {config.total_timesteps} timesteps...")
    agent = PPOAgent(env=args.env, config=config)
    agent.train()

    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    metrics = agent.evaluate(n_episodes=args.eval_episodes)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.save_path:
        agent.save(args.save_path)
        print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()
