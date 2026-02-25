"""Random baseline agent."""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class RandomAgent:
    """Uniformly random action selection baseline."""

    def __init__(self, env: gym.Env, seed: int = 42) -> None:
        self.env = env
        self.action_space = env.action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: dict | np.ndarray) -> tuple:
        """Select a random action from the action space.

        Args:
            obs: Current observation (unused by random agent).

        Returns:
            Tuple of (action, empty info dict).
        """
        return self.action_space.sample(), {}

    def evaluate(self, n_episodes: int = 100) -> dict[str, float]:
        """Run evaluation episodes and return summary statistics.

        Args:
            n_episodes: Number of episodes to evaluate over.

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
